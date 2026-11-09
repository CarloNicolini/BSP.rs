mod execution;
mod mutation;
mod options;
mod report;
mod search;
mod sp;
#[cfg(test)]
mod tests;

use crate::bitset::BitSet;
use crate::decision_heap::DecisionHeap;
use crate::{Error, Formula, Result, StatusResult};
use execution::Execution;
use mutation::TrueCounts;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Arc;
use std::time::Instant;

pub use execution::{solve_many, Backend, ExecutionOptions, ExecutionReport, Precision, SolveJob};
pub use options::Options;
pub use report::{Action, CancellationToken, SolveResult, SpReport, Status, Step};

pub type StepHook = Arc<dyn Fn(&Step) + Send + Sync>;

/// All mutable state belongs to one solver. Formula topology is shared through Arc.
pub struct Solver {
    pub(crate) execution: Execution,
    pub(crate) formula: Formula,
    pub(crate) options: Options,
    pub(crate) rng: SmallRng,
    pub(crate) assignment: Vec<i8>,
    pub(crate) active: BitSet,
    free: BitSet,
    forced: Vec<bool>,
    masks: crate::formula::ClauseMasks,
    true_count: TrueCounts,
    degree: Vec<u32>,
    decisions: DecisionHeap,
    eta: Vec<f64>,
    // Frozen incidences retain eta; active/free slots can hold observable norms.
    saved_eta: Vec<f64>,
    q: Vec<f64>,
    plus: Vec<f64>,
    minus: Vec<f64>,
    surveys: Vec<[f64; 3]>,
    cert: Vec<f64>,
    candidates: Vec<u32>,
    unit_queue: Vec<u32>,
    incident: Vec<u64>,
    clause_edges: Vec<usize>,
    prefix: Vec<f64>,
    started: Instant,
    cancel: CancellationToken,
    sp_sweeps: u64,
    flips: u64,
    decimations: u64,
    releases: u64,
    unit_assignments: u64,
    pending_units: usize,
    last_batch: usize,
    dec_moves: u64,
    back_moves: u64,
    steps: Vec<Step>,
    terminal: Option<Status>,
    solution: Option<Vec<i8>>,
    walksat_hook: Option<Arc<dyn Fn() + Send + Sync>>,
    step_hook: Option<StepHook>,
}

impl Solver {
    pub fn new(formula: Formula, options: Options) -> Result<Self> {
        let started = Instant::now();
        options.validate()?;
        let f = formula.data();
        let n = f.n;
        let m = formula.m();
        let edges = f.edges();
        if edges >= u32::MAX as usize || m >= u32::MAX as usize {
            return Err(Error("solver exceeds u32 edge/clause capacity; Feistel topology supports larger descriptors only".into()));
        }
        let mut rng = SmallRng::seed_from_u64(options.seed);
        let eta: Vec<_> = (0..edges).map(|_| rng.random::<f64>()).collect();
        let mut width = f.width;
        let mut unit_queue = Vec::new();
        if width == 1 {
            unit_queue.extend(0..m as u32);
        } else if width == 0 {
            for c in 0..m {
                let count = f.offset(c + 1) - f.offset(c);
                width = width.max(count);
                if count <= 1 {
                    unit_queue.push(c as u32);
                }
            }
        }
        let degree = if let Some(topology) = &f.feistel {
            topology.initial_degrees()
        } else {
            (0..n)
                .map(|v| f.var_offsets[2 * v + 2] - f.var_offsets[2 * v])
                .collect()
        };
        Ok(Self {
            execution: Execution::new(ExecutionOptions::default())?,
            masks: f.initial_masks(width),
            saved_eta: eta.clone(),
            formula,
            options,
            rng,
            assignment: vec![-1; n],
            active: BitSet::full(m),
            free: BitSet::full(n),
            forced: vec![false; n],
            true_count: TrueCounts::new(m, width),
            degree,
            decisions: DecisionHeap::new(n),
            eta,
            q: vec![0.0; edges],
            plus: vec![1.0; n],
            minus: vec![1.0; n],
            surveys: vec![[0.0, 0.0, 1.0]; n],
            cert: vec![0.0; n],
            candidates: Vec::with_capacity(n),
            unit_queue,
            incident: Vec::new(),
            clause_edges: Vec::with_capacity(width),
            prefix: Vec::with_capacity(width),
            started,
            cancel: CancellationToken::default(),
            sp_sweeps: 0,
            flips: 0,
            decimations: 0,
            releases: 0,
            unit_assignments: 0,
            pending_units: 0,
            last_batch: 1,
            dec_moves: 1,
            back_moves: 1,
            steps: Vec::new(),
            terminal: None,
            solution: None,
            walksat_hook: None,
            step_hook: None,
        })
    }
    /// Register a CPU callback for the start of WalkSAT.
    /// Device sweeps do not call it.
    pub fn on_walksat(&mut self, hook: Arc<dyn Fn() + Send + Sync>) {
        self.walksat_hook = Some(hook);
    }
    /// Register a callback that receives every recorded step.
    pub fn on_step(&mut self, hook: StepHook) {
        self.step_hook = Some(hook);
    }
    pub(crate) fn mark_walksat(&self) {
        if let Some(hook) = &self.walksat_hook {
            hook();
        }
    }
    pub fn formula(&self) -> &Formula {
        &self.formula
    }
    pub fn options(&self) -> &Options {
        &self.options
    }
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel.clone()
    }
    pub fn assignment(&self) -> &[i8] {
        &self.assignment
    }
    pub fn messages(&self) -> &[f64] {
        &self.eta
    }
    /// Surveys are refreshed by `converge`; fixed variables have one-hot surveys.
    pub fn surveys(&self) -> &[[f64; 3]] {
        &self.surveys
    }
    pub fn active_clauses(&self) -> usize {
        self.active.len()
    }
    pub fn free_variables(&self) -> usize {
        self.free.len()
    }
    pub fn history(&self) -> &[Step] {
        &self.steps
    }
    pub fn set_messages(&mut self, values: &[f64]) -> Result<()> {
        if values.len() != self.eta.len() || values.iter().any(|v| !(0.0..=1.0).contains(v)) {
            return Err(Error(
                "messages must have one finite value in [0, 1] per normalized edge".into(),
            ));
        }
        for (e, &value) in values.iter().enumerate() {
            self.saved_eta[e] = value;
            let f = self.formula.data();
            self.eta[e] = if self.active.contains(f.edge_clause(e))
                && self.assignment[(f.literal(e) >> 1) as usize] < 0
            {
                value
            } else {
                0.0
            };
        }
        self.terminal = None;
        self.solution = None;
        self.upload_messages();
        self.sync_messages();
        if let Some(error) = &self.execution.report.device_error {
            return Err(Error(error.clone()));
        }
        Ok(())
    }
    pub(crate) fn check_stop(&self) -> StatusResult<()> {
        if self.execution.report.device_error.is_some() {
            return Err(Status::NumericalFailure);
        }
        if self.cancel.is_cancelled() {
            return Err(Status::Cancelled);
        }
        if self
            .options
            .timeout_seconds
            .is_some_and(|limit| self.started.elapsed().as_secs_f64() >= limit)
        {
            return Err(Status::Timeout);
        }
        Ok(())
    }
    /// Fix a zero-based variable. Unit propagation is an explicit separate operation.
    pub fn fix(&mut self, variable: usize, value: bool) -> Result<()> {
        if variable >= self.formula.n() || self.assignment[variable] >= 0 {
            return Err(Error("fix requires a free, zero-based variable".into()));
        }
        self.set_fixed(variable, i8::from(value), false);
        self.terminal = None;
        self.solution = None;
        self.sync_messages();
        if let Some(error) = &self.execution.report.device_error {
            return Err(Error(error.clone()));
        }
        Ok(())
    }
    pub fn release(&mut self, variable: usize) -> Result<()> {
        if variable >= self.formula.n() || self.assignment[variable] < 0 || self.forced[variable] {
            return Err(Error(
                "release requires a fixed decision variable; unit assignments stay fixed".into(),
            ));
        }
        self.unfix(variable);
        self.terminal = None;
        self.solution = None;
        self.sync_messages();
        if let Some(error) = &self.execution.report.device_error {
            return Err(Error(error.clone()));
        }
        Ok(())
    }
}
