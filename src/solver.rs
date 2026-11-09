use crate::bitset::{BitSet, Ones};
use crate::decision_heap::DecisionHeap;
use crate::{Error, Formula, Result, StatusResult};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;
#[path = "execution.rs"]
mod execution;
use execution::Execution;
pub use execution::{solve_many, Backend, ExecutionOptions, ExecutionReport, SolveJob};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub seed: u64,
    pub backtrack_ratio: f64,
    pub batch_fraction: f64,
    pub epsilon: f64,
    /// Weight of the previous message. Convergence uses the undamped residual.
    pub damping: f64,
    pub max_sweeps: usize,
    pub max_steps: usize,
    /// Total WalkSAT flips; zero selects a budget scaled to the residual size.
    pub max_flips: u64,
    pub walksat_restarts: usize,
    pub walksat_noise: f64,
    pub timeout_seconds: Option<f64>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 1,
            backtrack_ratio: 0.9,
            batch_fraction: 0.00125,
            epsilon: 0.01,
            damping: 0.0,
            max_sweeps: 1024,
            max_steps: 100_000,
            max_flips: 0,
            walksat_restarts: 4,
            walksat_noise: 0.5,
            timeout_seconds: None,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        let checks = [
            (
                "backtrack_ratio",
                (0.0..1.0).contains(&self.backtrack_ratio),
                "in [0, 1)",
            ),
            (
                "batch_fraction",
                self.batch_fraction > 0.0 && self.batch_fraction <= 1.0,
                "in (0, 1]",
            ),
            ("epsilon", (0.0..1.0).contains(&self.epsilon), "in [0, 1)"),
            ("damping", (0.0..1.0).contains(&self.damping), "in [0, 1)"),
            (
                "walksat_noise",
                (0.0..=1.0).contains(&self.walksat_noise),
                "in [0, 1]",
            ),
            ("max_sweeps", self.max_sweeps > 0, "greater than 0"),
            ("max_steps", self.max_steps > 0, "greater than 0"),
            (
                "walksat_restarts",
                self.walksat_restarts > 0,
                "greater than 0",
            ),
            (
                "timeout_seconds",
                self.timeout_seconds
                    .is_none_or(|t| t.is_finite() && t > 0.0),
                "positive and finite",
            ),
        ];
        match checks.iter().find(|(_, valid, _)| !valid) {
            Some((name, _, expected)) => Err(Error(format!(
                "invalid option '{name}': must be {expected}"
            ))),
            None => Ok(()),
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Sat,
    Contradiction,
    NonConverged,
    NegativeComplexity,
    NumericalFailure,
    StepLimit,
    FlipLimit,
    Cancelled,
    Timeout,
}
impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Sat => "sat",
            Self::Contradiction => "contradiction",
            Self::NonConverged => "non_converged",
            Self::NegativeComplexity => "negative_complexity",
            Self::NumericalFailure => "numerical_failure",
            Self::StepLimit => "step_limit",
            Self::FlipLimit => "flip_limit",
            Self::Cancelled => "cancelled",
            Self::Timeout => "timeout",
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SpReport {
    pub sweeps: usize,
    pub residual: f64,
    pub complexity: f64,
    pub paramagnetic: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct Step {
    pub index: usize,
    pub action: String,
    pub free_variables: usize,
    pub active_clauses: usize,
    pub sp_sweeps: u64,
    pub residual: f64,
    pub complexity: f64,
    pub changed: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct SolveResult {
    pub execution: ExecutionReport,
    pub status: Status,
    pub assignment: Option<Vec<i8>>,
    pub partial_assignment: Vec<i8>,
    pub seed: u64,
    pub wall_seconds: f64,
    pub sp_sweeps: u64,
    pub flips: u64,
    pub decimations: u64,
    pub releases: u64,
    pub unit_assignments: u64,
    pub steps: Vec<Step>,
}

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
}

enum TrueCounts {
    Byte(Vec<u8>),
    Wide(Vec<u32>),
}

impl TrueCounts {
    fn new(clauses: usize, max_width: usize) -> Self {
        if max_width <= u8::MAX as usize {
            Self::Byte(vec![0; clauses])
        } else {
            Self::Wide(vec![0; clauses])
        }
    }

    #[inline]
    fn get(&self, clause: usize) -> u32 {
        match self {
            Self::Byte(counts) => u32::from(counts[clause]),
            Self::Wide(counts) => counts[clause],
        }
    }

    #[inline]
    fn change(&mut self, clause: usize, fix: bool) -> u32 {
        let count = if fix {
            self.get(clause) + 1
        } else {
            self.get(clause) - 1
        };
        match self {
            Self::Byte(counts) => counts[clause] = count as u8,
            Self::Wide(counts) => counts[clause] = count,
        }
        count
    }
}

// A full width-three clause has three independent pair products. Keep
// damping lane-local, and reduce the residual after computing each lane.
#[inline]
fn update_full_triple(q: &[f64], eta: &mut [f64], damping: f64) -> f64 {
    let raw = [q[1] * q[2], q[0] * q[2], q[0] * q[1]];
    let [a, b, c] = raw.map(|x| if x < 1e-16 { 0.0 } else { x });
    let old = [eta[0], eta[1], eta[2]];
    let delta = [old[0] - a, old[1] - b, old[2] - c];
    eta[..3].copy_from_slice(&[
        a + delta[0] * damping,
        b + delta[1] * damping,
        c + delta[2] * damping,
    ]);
    delta[0].abs().max(delta[1].abs()).max(delta[2].abs())
}

struct ScalarCavity<'a> {
    q: &'a mut [f64],
    eta: &'a [f64],
    saved_eta: &'a mut [f64],
    plus: &'a mut [f64],
    minus: &'a mut [f64],
    active: &'a BitSet,
}

impl ScalarCavity<'_> {
    #[inline]
    fn update<I: DoubleEndedIterator<Item = usize>>(
        &mut self,
        v: usize,
        store_norms: bool,
        edges: impl Fn(usize) -> I,
        clause_of: impl Fn(usize) -> usize,
    ) -> StatusResult<()> {
        let mut products = [1.0; 2];
        for (sign, product) in products.iter_mut().enumerate() {
            for e in edges(sign) {
                self.q[e] = *product;
                *product *= 1.0 - self.eta[e];
            }
        }
        self.plus[v] = products[0];
        self.minus[v] = products[1];
        for sign in 0..2 {
            let opposite = products[1 - sign];
            let mut suffix = 1.0;
            for e in edges(sign).rev() {
                let same = self.q[e] * suffix;
                suffix *= 1.0 - self.eta[e];
                let norm = opposite + (1.0 - opposite) * same;
                if norm <= 0.0 || !norm.is_finite() {
                    return Err(Status::NumericalFailure);
                }
                self.q[e] = ((1.0 - opposite) * same / norm).clamp(0.0, 1.0);
                if store_norms && self.active.contains(clause_of(e)) {
                    // Active/free incidences do not need a frozen value:
                    // freeze saves their current eta before any restoration.
                    self.saved_eta[e] = norm;
                }
            }
        }
        Ok(())
    }
}

impl Solver {
    pub fn new(formula: Formula, options: Options) -> Result<Self> {
        let started = Instant::now();
        options.validate()?;
        let f = &formula.0;
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
        })
    }
    /// Register a CPU callback for the start of WalkSAT.
    /// Device sweeps do not call it.
    pub fn on_walksat(&mut self, hook: Arc<dyn Fn() + Send + Sync>) {
        self.walksat_hook = Some(hook);
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
            let f = &self.formula.0;
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
    fn set_fixed(&mut self, v: usize, value: i8, forced: bool) {
        self.timed("graph_mutations", |s| s.set_fixed_inner(v, value, forced));
    }
    fn set_fixed_inner(&mut self, v: usize, value: i8, forced: bool) {
        if !forced {
            self.decisions.insert(v as u32, &self.cert);
        }
        self.degree[v] = 0;
        self.assignment[v] = value;
        self.execution.set_assignment(v, value);
        self.free.set(v, false);
        self.forced[v] = forced;
        self.surveys[v] = if value == 1 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let f = &self.formula.0;
        let mut incident = Vec::new();
        for e in f.variable_edges(v, &mut incident) {
            let c = f.edge_clause(e);
            let start = f.offset(c);
            let local = e - start;
            self.masks
                .set_bit(f.mask_offset(c) + local / 64, local % 64, false);
            if self.active.contains(c) {
                self.execution.freeze_edge(e);
                self.saved_eta[e] = self.eta[e];
                self.eta[e] = 0.0;
            }
            if value != i8::from(f.negative(e)) {
                if self.true_count.change(c, true) == 1 {
                    self.active.set(c, false);
                    self.execution.set_clause_active(c, false);
                    for w in f.mask_offset(c)..f.mask_offset(c + 1) {
                        for other in (Ones {
                            word: self.masks.get(w),
                            base: start + 64 * (w - f.mask_offset(c)),
                        }) {
                            self.degree[(f.literal(other) >> 1) as usize] -= 1;
                            self.execution.freeze_edge(other);
                            self.saved_eta[other] = self.eta[other];
                            self.eta[other] = 0.0;
                        }
                    }
                }
            } else if self.active.contains(c) && self.free_literals_up_to_two(c).0 <= 1 {
                self.unit_queue.push(c as u32);
            }
        }
        if forced {
            self.unit_assignments += 1;
            self.pending_units += 1;
        }
    }
    fn unfix(&mut self, v: usize) {
        self.timed("graph_mutations", |s| s.unfix_inner(v));
    }
    fn unfix_inner(&mut self, v: usize) {
        self.decisions.remove(v as u32, &self.cert);
        let value = self.assignment[v];
        self.assignment[v] = -1;
        self.execution.set_assignment(v, -1);
        self.free.set(v, true);
        let f = &self.formula.0;
        let mut incident = Vec::new();
        for e in f.variable_edges(v, &mut incident) {
            let c = f.edge_clause(e);
            let start = f.offset(c);
            let local = e - start;
            self.masks
                .set_bit(f.mask_offset(c) + local / 64, local % 64, true);
            let true_count = if value != i8::from(f.negative(e)) {
                self.true_count.change(c, false)
            } else {
                self.true_count.get(c)
            };
            if true_count == 0 {
                if self.active.contains(c) {
                    self.degree[v] += 1;
                    self.execution.restore_edge(e);
                    self.eta[e] = self.saved_eta[e];
                } else {
                    self.active.set(c, true);
                    self.execution.set_clause_active(c, true);
                    for w in f.mask_offset(c)..f.mask_offset(c + 1) {
                        for other in (Ones {
                            word: self.masks.get(w),
                            base: start + 64 * (w - f.mask_offset(c)),
                        }) {
                            self.degree[(f.literal(other) >> 1) as usize] += 1;
                            self.execution.restore_edge(other);
                            self.eta[other] = self.saved_eta[other];
                        }
                    }
                }
                if self.free_literals_up_to_two(c).0 <= 1 {
                    self.unit_queue.push(c as u32);
                }
            }
        }
    }
    // Unit propagation only distinguishes zero, one and multiple free bits.
    // Stop as soon as two bits are found, even for very wide clauses.
    fn free_literals_up_to_two(&self, c: usize) -> (u8, usize) {
        let f = &self.formula.0;
        let mut unit_edge = None;
        let first = f.mask_offset(c);
        for w in first..f.mask_offset(c + 1) {
            let mask = self.masks.get(w);
            if mask == 0 {
                continue;
            }
            if unit_edge.is_some() || mask & (mask - 1) != 0 {
                return (2, 0);
            }
            unit_edge = Some(f.offset(c) + 64 * (w - first) + mask.trailing_zeros() as usize);
        }
        unit_edge.map_or((0, 0), |edge| (1, edge))
    }

    pub fn propagate(&mut self) -> StatusResult<usize> {
        let result = self.timed("propagation", Self::propagate_inner);
        self.sync_messages();
        if self.execution.report.device_error.is_some() {
            return Err(Status::NumericalFailure);
        }
        result
    }
    fn propagate_inner(&mut self) -> StatusResult<usize> {
        let before = self.free.len();
        while let Some(c) = self.unit_queue.pop() {
            self.check_stop()?;
            let c = c as usize;
            if !self.active.contains(c) {
                continue;
            }
            let (count, e) = self.free_literals_up_to_two(c);
            if count == 0 {
                self.unit_queue.push(c as u32);
                return Err(Status::Contradiction);
            }
            if count != 1 {
                continue;
            }
            let f = &self.formula.0;
            let lit = f.literal(e);
            self.cert[(lit >> 1) as usize] = 1.0;
            self.set_fixed((lit >> 1) as usize, (1 - (lit & 1)) as i8, true);
        }
        Ok(before - self.free.len())
    }
    /// Prefix/suffix products exclude an edge without division by (1-eta).
    /// This covers exact zero factors and makes every cavity O(1) after O(E) work.
    fn cavities(&mut self, store_norms: bool) -> StatusResult<()> {
        self.timed("cavities", |s| s.scalar_cavities(store_norms))
    }
    fn scalar_cavities(&mut self, store_norms: bool) -> StatusResult<()> {
        let formula = self.formula.clone();
        let f = &formula.0;
        let mut state = ScalarCavity {
            q: &mut self.q,
            eta: &self.eta,
            saved_eta: &mut self.saved_eta,
            plus: &mut self.plus,
            minus: &mut self.minus,
            active: &self.active,
        };
        if let Some(topology) = &f.feistel {
            let mut scratch = Vec::new();
            let width = topology.k() as usize;
            for v in self.free.ones() {
                topology.ordered_edges(v as u64, &mut scratch);
                let split = scratch.partition_point(|&edge| edge < (1 << 32));
                let (positive, negative) = scratch.split_at(split);
                let groups = [positive, negative];
                state.update(
                    v,
                    store_norms,
                    |sign| groups[sign].iter().map(|&e| e as u32 as usize),
                    |e| e / width,
                )?;
            }
        } else {
            for v in self.free.ones() {
                state.update(
                    v,
                    store_norms,
                    |sign| {
                        let group = 2 * v + sign;
                        f.var_edges
                            [f.var_offsets[group] as usize..f.var_offsets[group + 1] as usize]
                            .iter()
                            .map(|&e| e as usize)
                    },
                    |e| f.edge_clause(e),
                )?;
            }
        }
        Ok(())
    }

    /// One synchronous SP sweep, returning the largest undamped message change.
    /// Clause exclusion uses prefix/suffix products, including multiple zero q's.
    pub fn sweep(&mut self) -> StatusResult<f64> {
        let result = self.sweep_inner();
        self.sync_messages();
        if self.execution.report.device_error.is_some() {
            return Err(Status::NumericalFailure);
        }
        result
    }
    /// Run raw synchronous sweeps, synchronizing snapshots once at the end.
    pub fn sweep_many(&mut self, count: usize) -> StatusResult<f64> {
        self.check_stop()?;
        if count == 0 {
            return Ok(0.0);
        }
        let result = if let Some(result) = self.gpu_compute(true, count) {
            result.map(|r| r.residual)
        } else {
            (|| {
                let mut residual = 0.0;
                for _ in 0..count {
                    residual = self.sweep_inner()?;
                }
                Ok(residual)
            })()
        };
        self.sync_messages();
        if self.execution.report.device_error.is_some() {
            return Err(Status::NumericalFailure);
        }
        result
    }
    fn sweep_inner(&mut self) -> StatusResult<f64> {
        self.check_stop()?;
        if let Some(result) = self.gpu_converge(true) {
            return result.map(|r| r.residual);
        }
        if let Some(result) = self.parallel_sweep() {
            return result;
        }
        self.cavities(false)?;
        let result = self.timed("clauses", Self::scalar_clauses);
        if result.is_ok() {
            self.sp_sweeps += 1;
        }
        result
    }
    fn scalar_clauses(&mut self) -> StatusResult<f64> {
        let f = &self.formula.0;
        let damping = self.options.damping;
        let mut residual: f64 = 0.0;
        for c in self.active.ones() {
            self.clause_edges.clear();
            self.prefix.clear();
            let start = f.offset(c);
            let first_word = f.mask_offset(c);
            if f.offset(c + 1) - start == 3 && self.masks.get(first_word) == 7 {
                residual = residual.max(update_full_triple(
                    &self.q[start..start + 3],
                    &mut self.eta[start..start + 3],
                    damping,
                ));
                continue;
            }
            let mut product = 1.0;
            for w in first_word..f.mask_offset(c + 1) {
                for e in (Ones {
                    word: self.masks.get(w),
                    base: start + 64 * (w - first_word),
                }) {
                    self.clause_edges.push(e);
                    self.prefix.push(product);
                    product *= self.q[e];
                }
            }
            if self.clause_edges.is_empty() {
                return Err(Status::Contradiction);
            }
            let mut suffix = 1.0;
            for (&e, &prefix) in self.clause_edges.iter().zip(&self.prefix).rev() {
                let raw = prefix * suffix;
                suffix *= self.q[e];
                let raw = if raw < 1e-16 { 0.0 } else { raw };
                residual = residual.max((raw - self.eta[e]).abs());
                self.eta[e] = raw + (self.eta[e] - raw) * damping;
            }
        }
        Ok(residual)
    }
    fn observables(&mut self) -> StatusResult<(f64, bool)> {
        self.timed("observables", |s| {
            if let Some(result) = s.parallel_observables() {
                result
            } else {
                s.scalar_observables()
            }
        })
    }
    fn scalar_observables(&mut self) -> StatusResult<(f64, bool)> {
        self.cavities(true)?;
        self.scalar_observables_from_cavities()
    }
    fn scalar_observables_from_cavities(&mut self) -> StatusResult<(f64, bool)> {
        let f = &self.formula.0;
        let mut variable_term = 0.0;
        let mut paramagnetic = true;
        for v in self.free.ones() {
            let plus = self.plus[v];
            let minus = self.minus[v];
            let norm = plus + (1.0 - plus) * minus;
            if norm <= 0.0 {
                return Err(Status::NumericalFailure);
            }
            let st = (1.0 - plus) * minus / norm;
            let sf = (1.0 - minus) * plus / norm;
            self.surveys[v] = [st, sf, (plus * minus / norm).clamp(0.0, 1.0)];
            self.cert[v] = 1.0 - st.min(sf);
            let degree = self.degree[v];
            variable_term += (degree as f64 - 1.0) * norm.ln();
            if degree >= 2 && norm != 1.0 {
                paramagnetic = false;
            }
        }
        let mut clause_term = 0.0;
        for c in self.active.ones() {
            let mut product = 1.0;
            let start = f.offset(c);
            let first_word = f.mask_offset(c);
            for w in first_word..f.mask_offset(c + 1) {
                for e in (Ones {
                    word: self.masks.get(w),
                    base: start + 64 * (w - first_word),
                }) {
                    product *= self.q[e];
                    clause_term += self.saved_eta[e].ln();
                }
            }
            if product >= 1.0 {
                return Err(Status::NumericalFailure);
            }
            clause_term += (-product).ln_1p();
        }
        let complexity = if paramagnetic {
            0.0
        } else {
            clause_term - variable_term
        };
        if !complexity.is_finite() {
            return Err(Status::NumericalFailure);
        }
        Ok((complexity, paramagnetic))
    }
    pub fn converge(&mut self) -> StatusResult<SpReport> {
        let result = self.converge_inner();
        self.sync_messages();
        if self.execution.report.device_error.is_some() {
            return Err(Status::NumericalFailure);
        }
        result
    }
    fn converge_inner(&mut self) -> StatusResult<SpReport> {
        self.timed("propagation", Self::propagate_inner)?;
        if let Some(result) = self.gpu_converge(false) {
            return result;
        }
        for sweeps in 1..=self.options.max_sweeps {
            let residual = self.sweep_inner()?;
            if residual <= self.options.epsilon {
                let (complexity, paramagnetic) = self.observables()?;
                return Ok(SpReport {
                    sweeps,
                    residual,
                    complexity,
                    paramagnetic,
                });
            }
        }
        Err(Status::NonConverged)
    }
    /// Select only the top batch, using deterministic variable-ID tie breaking.
    pub fn decimate(&mut self) -> usize {
        let result = self.timed("decimation", Self::decimate_inner);
        self.sync_messages();
        result
    }
    fn decimate_inner(&mut self) -> usize {
        let ranking_started = self
            .execution
            .options
            .profile
            .as_ref()
            .map(|_| Instant::now());
        self.terminal = None;
        self.solution = None;
        self.candidates.clear();
        self.candidates.extend(self.free.ones().map(|v| v as u32));
        let k = ((self.options.batch_fraction * self.free.len() as f64) as usize)
            .max(1)
            .min(self.candidates.len());
        if k == 0 {
            return 0;
        }
        let cert = &self.cert;
        let order = |&a: &u32, &b: &u32| {
            cert[b as usize]
                .total_cmp(&cert[a as usize])
                .then_with(|| a.cmp(&b))
        };
        if k < self.candidates.len() {
            self.candidates.select_nth_unstable_by(k, order);
        }
        self.candidates[..k].sort_unstable_by(order);
        if let Some(started) = ranking_started {
            self.execution
                .record("ranking", started.elapsed().as_secs_f64());
        }
        for i in 0..k {
            let v = self.candidates[i] as usize;
            self.set_fixed(v, i8::from(self.surveys[v][0] > self.surveys[v][1]), false);
        }
        self.decimations += k as u64;
        self.last_batch = k;
        self.pending_units = 0;
        k
    }
    pub fn backtrack(&mut self) -> usize {
        let result = self.timed("backtracking", Self::backtrack_inner);
        self.sync_messages();
        result
    }
    fn backtrack_inner(&mut self) -> usize {
        self.terminal = None;
        self.solution = None;
        let k = (self.last_batch + self.pending_units).min(self.decisions.len());
        self.pending_units = 0;
        for _ in 0..k {
            let v = self
                .decisions
                .peek()
                .expect("release batch fits fixed decision count");
            self.unfix(v as usize);
        }
        self.releases += k as u64;
        k
    }
    /// Advance through one fixed point and one BSP action. Terminal states are sticky.
    pub fn advance(&mut self) -> StatusResult<Step> {
        let result = self.advance_internal();
        self.sync_messages();
        if self.execution.report.device_error.is_some() {
            return Err(Status::NumericalFailure);
        }
        result
    }
    fn advance_internal(&mut self) -> StatusResult<Step> {
        if let Some(status) = self.terminal {
            return Err(status);
        }
        let result = self.advance_inner();
        if let Err(status) = result {
            self.terminal = Some(status);
        }
        result
    }
    fn advance_inner(&mut self) -> StatusResult<Step> {
        self.check_stop()?;
        if self.steps.len() >= self.options.max_steps {
            return Err(Status::StepLimit);
        }
        let report = self.converge_inner()?;
        let mut step = Step {
            index: self.steps.len(),
            action: String::new(),
            free_variables: self.free.len(),
            active_clauses: self.active.len(),
            sp_sweeps: self.sp_sweeps,
            residual: report.residual,
            complexity: report.complexity,
            changed: 0,
        };
        if report.paramagnetic || self.active.is_empty() {
            step.action = "walksat".into();
            self.steps.push(step.clone());
            let assignment = self.timed("walksat", Self::walksat)?;
            self.solution = Some(assignment);
            self.terminal = Some(Status::Sat);
            return Ok(step);
        }
        if self.back_moves as f64 / (self.dec_moves as f64) < self.options.backtrack_ratio {
            self.back_moves += 1;
            step.action = "backtrack".into();
            step.changed = self.timed("backtracking", Self::backtrack_inner);
        } else {
            self.dec_moves += 1;
            if report.complexity < -2.0 {
                return Err(Status::NegativeComplexity);
            }
            step.action = "decimate".into();
            step.changed = self.timed("decimation", Self::decimate_inner);
        }
        self.steps.push(step.clone());
        Ok(step)
    }
    pub(crate) fn residual_variable_count(&self) -> usize {
        self.free.ones().filter(|&v| self.degree[v] > 0).count()
    }
    pub fn solve(&mut self) -> SolveResult {
        while self.terminal.is_none() {
            let _ = self.advance_internal();
        }
        self.sync_messages();
        if let Err(error) = self.write_profile() {
            self.execution.report.profile_error = Some(error.to_string());
        }
        SolveResult {
            execution: self.execution.report.clone(),
            status: self.terminal.unwrap(),
            assignment: self.solution.clone(),
            partial_assignment: self.assignment.clone(),
            seed: self.options.seed,
            wall_seconds: self.started.elapsed().as_secs_f64(),
            sp_sweeps: self.sp_sweeps,
            flips: self.flips,
            decimations: self.decimations,
            releases: self.releases,
            unit_assignments: self.unit_assignments,
            steps: self.steps.clone(),
        }
    }
    pub fn residual_formula(&self) -> Formula {
        let f = &self.formula.0;
        let clauses = self
            .active
            .ones()
            .map(|c| {
                (f.offset(c)..f.offset(c + 1))
                    .filter_map(|e| {
                        let l = f.literal(e);
                        let v = (l >> 1) as usize;
                        if self.assignment[v] >= 0 {
                            None
                        } else {
                            Some(if l & 1 == 0 {
                                (v + 1) as i32
                            } else {
                                -((v + 1) as i32)
                            })
                        }
                    })
                    .collect()
            })
            .collect();
        Formula::new(f.n, clauses).expect("residual preserves validated topology bounds")
    }
    pub(crate) fn record_flip(&mut self) {
        self.flips += 1;
    }
}

#[cfg(test)]
mod index_tests {
    use super::*;
    fn check_indexes(s: &Solver) {
        let f = &s.formula.0;
        let mut degree = vec![0u32; f.n];
        for c in 0..s.formula.m() {
            let mut free = 0usize;
            let mut true_count = 0;
            for e in f.offset(c)..f.offset(c + 1) {
                let lit = f.literal(e);
                let v = (lit >> 1) as usize;
                let is_free = s.assignment[v] < 0;
                let local = e - f.offset(c);
                let mask_free =
                    s.masks.get(f.mask_offset(c) + local / 64) & (1u64 << (local % 64)) != 0;
                assert_eq!(is_free, mask_free);
                if is_free {
                    free += 1;
                } else if s.assignment[v] == (1 - (lit & 1)) as i8 {
                    true_count += 1;
                }
            }
            assert_eq!(s.free_literals_up_to_two(c).0, free.min(2) as u8);
            assert_eq!(s.true_count.get(c), true_count);
            assert_eq!(s.active.contains(c), true_count == 0);
            if true_count == 0 {
                for e in f.offset(c)..f.offset(c + 1) {
                    let v = (f.literal(e) >> 1) as usize;
                    if s.assignment[v] < 0 {
                        degree[v] += 1;
                    }
                }
            }
        }
        assert_eq!(s.degree, degree);
        let eligible: Vec<_> = (0..f.n)
            .filter(|&v| s.assignment[v] >= 0 && !s.forced[v])
            .collect();
        let best = eligible
            .iter()
            .copied()
            .min_by(|&a, &b| s.cert[a].total_cmp(&s.cert[b]).then(a.cmp(&b)));
        assert_eq!(s.decisions.len(), eligible.len());
        assert_eq!(s.decisions.peek().map(|v| v as usize), best);
    }
    #[test]
    fn generated_width_indexes_match_generic_indexes() {
        for (k, alpha) in [(1, 1.0), (3, 4.0), (64, 1.0), (65, 1.0), (3, 0.0)] {
            let generated = Formula::random(k, alpha, 130, 17).unwrap();
            let generic = Formula::new(generated.n(), generated.clauses()).unwrap();
            let f = &generated.0;
            assert!(f.offsets.is_empty() && f.edge_clause.is_empty() && f.mask_offsets.is_empty());
            assert_eq!(f.var_offsets, generic.0.var_offsets);
            assert_eq!(f.var_edges, generic.0.var_edges);
            assert_eq!(generated.to_dimacs(), generic.to_dimacs());
            for e in 0..f.edges() {
                assert_eq!(f.edge_clause(e), generic.0.edge_clause(e));
            }
            for c in 0..=f.m() {
                assert_eq!(f.mask_offset(c), generic.0.mask_offset(c));
            }

            let mut a = Solver::new(generated, Options::default()).unwrap();
            let mut b = Solver::new(generic, Options::default()).unwrap();
            assert_eq!(a.unit_queue, b.unit_queue);
            for v in [0, 63, 64, 129] {
                a.fix(v, true).unwrap();
                b.fix(v, true).unwrap();
                check_indexes(&a);
                assert_eq!(a.degree, b.degree);
            }
        }
    }
    #[test]
    fn cached_indexes_match_full_recounts_through_mixed_mutations() {
        let mut rng = SmallRng::seed_from_u64(91);
        for seed in 0..5 {
            let formula = Formula::random(5, 3.0, 130, seed).unwrap();
            let mut s = Solver::new(formula, Options::default()).unwrap();
            check_indexes(&s);
            for step in 0usize..1200 {
                let v = rng.random_range(0..130);
                if s.assignment[v] < 0 {
                    s.fix(v, rng.random()).unwrap();
                } else if !s.forced[v] {
                    s.release(v).unwrap();
                }
                if step.is_multiple_of(41) {
                    let _ = s.propagate();
                }
                check_indexes(&s);
            }
        }
    }
    #[test]
    fn heap_backtracking_matches_sorted_frozen_scores_for_batches() {
        let mut s =
            Solver::new(Formula::random(4, 1.0, 130, 7).unwrap(), Options::default()).unwrap();
        for v in 0..100 {
            s.cert[v] = (v % 7) as f64 / 7.0;
            s.fix(v, v.is_multiple_of(2)).unwrap();
        }
        for v in [0, 12, 63, 64, 91] {
            s.release(v).unwrap();
            s.fix(v, false).unwrap();
        }
        s.last_batch = 9;
        s.pending_units = 3;
        let mut expected: Vec<_> = (0..100).collect();
        expected.sort_unstable_by(|&a, &b| s.cert[a].total_cmp(&s.cert[b]).then(a.cmp(&b)));
        assert_eq!(s.backtrack(), 12);
        for v in 0..100 {
            assert_eq!(s.assignment[v] < 0, expected[..12].contains(&v));
        }
        check_indexes(&s);
    }
}
