use crate::bitset::{BitSet, Ones};
use crate::{Error, Formula, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

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
            max_flips: 10_000_000,
            walksat_restarts: 4,
            walksat_noise: 0.5,
            timeout_seconds: None,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        if !self.backtrack_ratio.is_finite()
            || !(0.0..1.0).contains(&self.backtrack_ratio)
            || !self.batch_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.batch_fraction)
            || self.batch_fraction == 0.0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || self.epsilon >= 1.0
            || !self.damping.is_finite()
            || !(0.0..1.0).contains(&self.damping)
            || !self.walksat_noise.is_finite()
            || !(0.0..=1.0).contains(&self.walksat_noise)
            || self.max_sweeps == 0
            || self.max_steps == 0
            || self.walksat_restarts == 0
            || self
                .timeout_seconds
                .is_some_and(|t| !t.is_finite() || t <= 0.0)
        {
            return Err(Error(
                "invalid BSP options: check probability ranges, positive limits, and timeout"
                    .into(),
            ));
        }
        Ok(())
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
    pub(crate) formula: Formula,
    pub(crate) options: Options,
    pub(crate) rng: SmallRng,
    pub(crate) assignment: Vec<i8>,
    pub(crate) active: BitSet,
    free: BitSet,
    forced: Vec<bool>,
    masks: Vec<u64>,
    free_count: Vec<u32>,
    true_count: Vec<u32>,
    eta: Vec<f64>,
    saved_eta: Vec<f64>,
    q: Vec<f64>,
    cavity_norm: Vec<f64>,
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
}

impl Solver {
    pub fn new(formula: Formula, options: Options) -> Result<Self> {
        let started = Instant::now();
        options.validate()?;
        let f = &formula.0;
        let n = f.n;
        let m = formula.m();
        let edges = f.literals.len();
        let mut rng = SmallRng::seed_from_u64(options.seed);
        let eta: Vec<_> = (0..edges).map(|_| rng.random::<f64>()).collect();
        let free_count: Vec<_> = f.offsets.windows(2).map(|w| w[1] - w[0]).collect();
        let unit_queue = free_count
            .iter()
            .enumerate()
            .filter(|(_, &k)| k <= 1)
            .map(|(c, _)| c as u32)
            .collect();
        let width = free_count.iter().copied().max().unwrap_or(0) as usize;
        Ok(Self {
            masks: f.masks.clone(),
            saved_eta: eta.clone(),
            formula,
            options,
            rng,
            assignment: vec![-1; n],
            active: BitSet::full(m),
            free: BitSet::full(n),
            forced: vec![false; n],
            free_count,
            true_count: vec![0; m],
            eta,
            q: vec![0.0; edges],
            cavity_norm: vec![1.0; edges],
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
        })
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
        if values.len() != self.eta.len()
            || values
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(Error(
                "messages must have one finite value in [0, 1] per normalized edge".into(),
            ));
        }
        for (e, &value) in values.iter().enumerate() {
            self.saved_eta[e] = value;
            let f = &self.formula.0;
            self.eta[e] = if self.active.contains(f.edge_clause[e] as usize)
                && self.assignment[(f.literals[e] >> 1) as usize] < 0
            {
                value
            } else {
                0.0
            };
        }
        self.terminal = None;
        self.solution = None;
        Ok(())
    }
    pub(crate) fn check_stop(&self) -> std::result::Result<(), Status> {
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
        Ok(())
    }
    fn set_fixed(&mut self, v: usize, value: i8, forced: bool) {
        self.assignment[v] = value;
        self.free.set(v, false);
        self.forced[v] = forced;
        self.surveys[v] = if value == 1 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let f = &self.formula.0;
        for p in f.var_offsets[2 * v] as usize..f.var_offsets[2 * v + 2] as usize {
            let e = f.var_edges[p] as usize;
            let c = f.edge_clause[e] as usize;
            let start = f.offsets[c] as usize;
            let local = e - start;
            self.masks[f.mask_offsets[c] as usize + local / 64] &= !(1u64 << (local % 64));
            self.free_count[c] -= 1;
            if self.active.contains(c) {
                self.saved_eta[e] = self.eta[e];
                self.eta[e] = 0.0;
            }
            if value == (1 - (f.literals[e] & 1)) as i8 {
                self.true_count[c] += 1;
                if self.true_count[c] == 1 {
                    self.active.set(c, false);
                    for w in f.mask_offsets[c] as usize..f.mask_offsets[c + 1] as usize {
                        for other in (Ones {
                            word: self.masks[w],
                            base: start + 64 * (w - f.mask_offsets[c] as usize),
                        }) {
                            self.saved_eta[other] = self.eta[other];
                            self.eta[other] = 0.0;
                        }
                    }
                }
            } else if self.active.contains(c) && self.free_count[c] <= 1 {
                self.unit_queue.push(c as u32);
            }
        }
        if forced {
            self.unit_assignments += 1;
            self.pending_units += 1;
        }
    }
    fn unfix(&mut self, v: usize) {
        let value = self.assignment[v];
        self.assignment[v] = -1;
        self.free.set(v, true);
        let f = &self.formula.0;
        for p in f.var_offsets[2 * v] as usize..f.var_offsets[2 * v + 2] as usize {
            let e = f.var_edges[p] as usize;
            let c = f.edge_clause[e] as usize;
            let start = f.offsets[c] as usize;
            let local = e - start;
            self.masks[f.mask_offsets[c] as usize + local / 64] |= 1u64 << (local % 64);
            self.free_count[c] += 1;
            if value == (1 - (f.literals[e] & 1)) as i8 {
                self.true_count[c] -= 1;
            }
            if self.true_count[c] == 0 {
                if self.active.contains(c) {
                    self.eta[e] = self.saved_eta[e];
                } else {
                    self.active.set(c, true);
                    for w in f.mask_offsets[c] as usize..f.mask_offsets[c + 1] as usize {
                        for other in (Ones {
                            word: self.masks[w],
                            base: start + 64 * (w - f.mask_offsets[c] as usize),
                        }) {
                            self.eta[other] = self.saved_eta[other];
                        }
                    }
                }
                if self.free_count[c] <= 1 {
                    self.unit_queue.push(c as u32);
                }
            }
        }
    }
    pub fn propagate(&mut self) -> std::result::Result<usize, Status> {
        let before = self.free.len();
        while let Some(c) = self.unit_queue.pop() {
            self.check_stop()?;
            let c = c as usize;
            if !self.active.contains(c) {
                continue;
            }
            if self.free_count[c] == 0 {
                self.unit_queue.push(c as u32);
                return Err(Status::Contradiction);
            }
            if self.free_count[c] != 1 {
                continue;
            }
            let f = &self.formula.0;
            let w = (f.mask_offsets[c] as usize..f.mask_offsets[c + 1] as usize)
                .find(|&w| self.masks[w] != 0)
                .unwrap();
            let e = f.offsets[c] as usize
                + 64 * (w - f.mask_offsets[c] as usize)
                + self.masks[w].trailing_zeros() as usize;
            let lit = f.literals[e];
            self.cert[(lit >> 1) as usize] = 1.0;
            self.set_fixed((lit >> 1) as usize, (1 - (lit & 1)) as i8, true);
        }
        Ok(before - self.free.len())
    }
    /// Prefix/suffix products exclude an edge without division by (1-eta).
    /// This covers exact zero factors and makes every cavity O(1) after O(E) work.
    fn cavities(&mut self) -> std::result::Result<(), Status> {
        let f = &self.formula.0;
        for v in self.free.ones() {
            let mut products = [1.0; 2];
            for (sign, product) in products.iter_mut().enumerate() {
                let g = 2 * v + sign;
                for &edge in &f.var_edges[f.var_offsets[g] as usize..f.var_offsets[g + 1] as usize]
                {
                    let e = edge as usize;
                    self.q[e] = *product;
                    *product *= 1.0 - self.eta[e];
                }
            }
            self.plus[v] = products[0];
            self.minus[v] = products[1];
            for sign in 0..2 {
                let g = 2 * v + sign;
                let opposite = products[1 - sign];
                let mut suffix = 1.0;
                for &edge in f.var_edges[f.var_offsets[g] as usize..f.var_offsets[g + 1] as usize]
                    .iter()
                    .rev()
                {
                    let e = edge as usize;
                    let same = self.q[e] * suffix;
                    suffix *= 1.0 - self.eta[e];
                    let norm = opposite + (1.0 - opposite) * same;
                    if norm <= 0.0 || !norm.is_finite() {
                        return Err(Status::NumericalFailure);
                    }
                    self.q[e] = ((1.0 - opposite) * same / norm).clamp(0.0, 1.0);
                    self.cavity_norm[e] = norm;
                }
            }
        }
        Ok(())
    }
    /// One synchronous SP sweep, returning the largest undamped message change.
    /// Clause exclusion uses prefix/suffix products, including multiple zero q's.
    pub fn sweep(&mut self) -> std::result::Result<f64, Status> {
        self.check_stop()?;
        self.cavities()?;
        let f = &self.formula.0;
        let damping = self.options.damping;
        let mut residual: f64 = 0.0;
        for c in self.active.ones() {
            if self.free_count[c] == 0 {
                return Err(Status::Contradiction);
            }
            self.clause_edges.clear();
            self.prefix.clear();
            let start = f.offsets[c] as usize;
            let first_word = f.mask_offsets[c] as usize;
            let mut product = 1.0;
            for w in first_word..f.mask_offsets[c + 1] as usize {
                for e in (Ones {
                    word: self.masks[w],
                    base: start + 64 * (w - first_word),
                }) {
                    self.clause_edges.push(e);
                    self.prefix.push(product);
                    product *= self.q[e];
                }
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
        self.sp_sweeps += 1;
        Ok(residual)
    }
    fn observables(&mut self) -> std::result::Result<(f64, bool), Status> {
        self.cavities()?;
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
            let degree = f.var_edges
                [f.var_offsets[2 * v] as usize..f.var_offsets[2 * v + 2] as usize]
                .iter()
                .filter(|&&e| self.active.contains(f.edge_clause[e as usize] as usize))
                .count();
            variable_term += (degree as f64 - 1.0) * norm.ln();
            if degree >= 2 && norm != 1.0 {
                paramagnetic = false;
            }
        }
        let mut clause_term = 0.0;
        for c in self.active.ones() {
            let mut product = 1.0;
            let start = f.offsets[c] as usize;
            let first_word = f.mask_offsets[c] as usize;
            for w in first_word..f.mask_offsets[c + 1] as usize {
                for e in (Ones {
                    word: self.masks[w],
                    base: start + 64 * (w - first_word),
                }) {
                    product *= self.q[e];
                    clause_term += self.cavity_norm[e].ln();
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
    pub fn converge(&mut self) -> std::result::Result<SpReport, Status> {
        self.propagate()?;
        for sweeps in 1..=self.options.max_sweeps {
            let residual = self.sweep()?;
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
        self.terminal = None;
        self.solution = None;
        self.candidates.clear();
        self.candidates.extend(
            (0..self.formula.n())
                .filter(|&v| self.assignment[v] >= 0 && !self.forced[v])
                .map(|v| v as u32),
        );
        let k = (self.last_batch + self.pending_units).min(self.candidates.len());
        self.pending_units = 0;
        if k == 0 {
            return 0;
        }
        let cert = &self.cert;
        let order = |&a: &u32, &b: &u32| {
            cert[a as usize]
                .total_cmp(&cert[b as usize])
                .then_with(|| a.cmp(&b))
        };
        if k < self.candidates.len() {
            self.candidates.select_nth_unstable_by(k, order);
        }
        self.candidates[..k].sort_unstable_by(order);
        for i in 0..k {
            self.unfix(self.candidates[i] as usize);
        }
        self.releases += k as u64;
        k
    }
    /// Advance through one fixed point and one BSP action. Terminal states are sticky.
    pub fn advance(&mut self) -> std::result::Result<Step, Status> {
        if let Some(status) = self.terminal {
            return Err(status);
        }
        let result = self.advance_inner();
        if let Err(status) = result {
            self.terminal = Some(status);
        }
        result
    }
    fn advance_inner(&mut self) -> std::result::Result<Step, Status> {
        self.check_stop()?;
        if self.steps.len() >= self.options.max_steps {
            return Err(Status::StepLimit);
        }
        let report = self.converge()?;
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
        if report.paramagnetic || self.active.len() == 0 {
            step.action = "walksat".into();
            self.steps.push(step.clone());
            let assignment = self.walksat()?;
            self.solution = Some(assignment);
            self.terminal = Some(Status::Sat);
            return Ok(step);
        }
        if self.back_moves as f64 / (self.dec_moves as f64) < self.options.backtrack_ratio {
            self.back_moves += 1;
            step.action = "backtrack".into();
            step.changed = self.backtrack();
        } else {
            self.dec_moves += 1;
            if report.complexity < -2.0 {
                return Err(Status::NegativeComplexity);
            }
            step.action = "decimate".into();
            step.changed = self.decimate();
        }
        self.steps.push(step.clone());
        Ok(step)
    }
    pub fn solve(&mut self) -> SolveResult {
        while self.terminal.is_none() {
            let _ = self.advance();
        }
        SolveResult {
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
                (f.offsets[c] as usize..f.offsets[c + 1] as usize)
                    .filter_map(|e| {
                        let l = f.literals[e];
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
