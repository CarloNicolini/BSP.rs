use super::*;
use crate::formula::Data;
use std::collections::VecDeque;

/// Work for the residual schedule. Graph mutations enqueue clauses whose free
/// literals changed and mark free variables whose incoming messages changed.
/// While `current` holds, q matches eta for every free variable that is not stale.
pub(crate) struct ResidualQueue {
    clauses: VecDeque<u32>,
    queued: BitSet,
    variables: Vec<u32>,
    stale: BitSet,
    pub(crate) current: bool,
    /// Visited free incidences of active clauses; one sweep visits each twice.
    pub(crate) incidences: u64,
}

impl ResidualQueue {
    #[cfg(test)]
    pub(super) fn new(n: usize, m: usize) -> Self {
        Self {
            clauses: VecDeque::new(),
            queued: BitSet::empty(m),
            variables: Vec::new(),
            stale: BitSet::empty(n),
            current: false,
            incidences: 0,
        }
    }
    pub(super) fn clause(&mut self, c: usize) {
        if !self.queued.contains(c) {
            self.queued.set(c, true);
            self.clauses.push_back(c as u32);
        }
    }
    pub(super) fn variable(&mut self, v: usize) {
        if !self.stale.contains(v) {
            self.stale.set(v, true);
            self.variables.push(v as u32);
        }
    }
}

impl Solver {
    /// Residual scheduling exists only for scalar `f64` CPU execution.
    #[cfg(test)]
    pub(super) fn enable_residual_schedule(&mut self) {
        assert_eq!(self.execution.report.backend, Backend::Cpu);
        self.residual = Some(ResidualQueue::new(self.formula.n(), self.formula.m()));
    }
    pub(super) fn residual_converge(&mut self) -> StatusResult<SpReport> {
        let mut queue = self.residual.take().expect("residual schedule is enabled");
        let report = self
            .timed("residual", |s| s.drain(&mut queue))
            .and_then(|residual| {
                let (complexity, paramagnetic) = self.observables()?;
                self.enqueue_unsettled(&mut queue)?;
                Ok(SpReport {
                    sweeps: 0,
                    residual,
                    complexity,
                    paramagnetic,
                })
            });
        // Observables recompute every cavity from the converged eta.
        queue.current = report.is_ok();
        self.residual = Some(queue);
        report
    }
    /// Settle the queue, then write one full undamped pass within epsilon.
    fn drain(&mut self, queue: &mut ResidualQueue) -> StatusResult<f64> {
        let formula = self.formula.clone();
        let f = formula.data();
        let limit = queue.incidences + 2 * self.options.max_sweeps as u64 * f.edges() as u64;
        loop {
            self.settle(queue, limit)?;
            // Like the last synchronous sweep, write every message. Paramagnetic
            // detection needs exact zeros, which only repeated updates reach.
            let mut largest: f64 = 0.0;
            for c in 0..f.m() {
                if self.active.contains(c) {
                    largest = largest.max(self.clause_residual(f, c)?);
                    queue.incidences += self.clause_edges.len() as u64;
                    for (&e, &raw) in self.clause_edges.iter().zip(&self.prefix) {
                        self.eta[e] = raw + (self.eta[e] - raw) * self.options.damping;
                    }
                }
            }
            queue.current = false;
            if largest <= self.options.epsilon {
                return Ok(largest);
            }
        }
    }
    /// Re-evaluate queued clauses until every active clause is within epsilon
    /// of its update. Afterwards q matches eta for every free variable.
    pub(super) fn settle(&mut self, queue: &mut ResidualQueue, limit: u64) -> StatusResult<()> {
        let formula = self.formula.clone();
        let f = formula.data();
        let (epsilon, damping) = (self.options.epsilon, self.options.damping);
        let variables = std::mem::take(&mut queue.variables);
        for &v in &variables {
            queue.stale.set(v as usize, false);
        }
        if queue.current {
            for &v in &variables {
                if self.free.contains(v as usize) {
                    self.refresh(f, queue, v as usize, usize::MAX)?;
                }
            }
        } else {
            self.cavities(false)?;
            queue.incidences += self
                .free
                .ones()
                .map(|v| u64::from(self.degree[v]))
                .sum::<u64>();
            for c in self.active.ones() {
                queue.clause(c);
            }
        }
        queue.variables = variables;
        queue.variables.clear();
        queue.current = true;
        let mut pops = 0u64;
        while let Some(c) = queue.clauses.pop_front() {
            let c = c as usize;
            queue.queued.set(c, false);
            if !self.active.contains(c) {
                continue;
            }
            pops += 1;
            if pops.is_multiple_of(1024) {
                self.check_stop()?;
            }
            if queue.incidences > limit {
                return Err(Status::NonConverged);
            }
            let residual = self.clause_residual(f, c)?;
            queue.incidences += self.clause_edges.len() as u64;
            if residual <= epsilon {
                continue;
            }
            for (&e, &raw) in self.clause_edges.iter().zip(&self.prefix) {
                self.eta[e] = raw + (self.eta[e] - raw) * damping;
            }
            for i in 0..self.clause_edges.len() {
                let v = (f.literal(self.clause_edges[i]) >> 1) as usize;
                self.refresh(f, queue, v, c)?;
            }
            if damping * residual > epsilon {
                queue.clause(c);
            }
        }
        Ok(())
    }
    /// The final pass moved every message, so any clause may exceed epsilon
    /// once observables refresh q.
    fn enqueue_unsettled(&mut self, queue: &mut ResidualQueue) -> StatusResult<()> {
        let formula = self.formula.clone();
        let f = formula.data();
        for c in 0..f.m() {
            if self.active.contains(c) {
                if self.clause_residual(f, c)? > self.options.epsilon {
                    queue.clause(c);
                }
                queue.incidences += self.clause_edges.len() as u64;
            }
        }
        Ok(())
    }
    /// Rebuild the queue after sweeps converged and observables refreshed q.
    #[cfg(test)]
    pub(super) fn resume_residual_queue(&mut self) -> StatusResult<()> {
        let mut queue = self.residual.take().expect("residual schedule is enabled");
        while let Some(c) = queue.clauses.pop_front() {
            queue.queued.set(c as usize, false);
        }
        queue.current = true;
        let result = self.enqueue_unsettled(&mut queue);
        self.residual = Some(queue);
        result
    }
    /// Recompute the cavities of `v` and enqueue its active clauses except `except`.
    fn refresh(
        &mut self,
        f: &Data,
        queue: &mut ResidualQueue,
        v: usize,
        except: usize,
    ) -> StatusResult<()> {
        self.variable_cavity(v)?;
        queue.incidences += 2 * u64::from(self.degree[v]);
        for e in f.variable_edges(v, &mut self.incident) {
            let c = f.edge_clause(e);
            if c != except && self.active.contains(c) {
                queue.clause(c);
            }
        }
        Ok(())
    }
    /// Leave the undamped update of clause `c` in `prefix`, aligned with
    /// `clause_edges`, and return its largest change without writing eta.
    fn clause_residual(&mut self, f: &Data, c: usize) -> StatusResult<f64> {
        self.clause_edges.clear();
        self.prefix.clear();
        let mut product = 1.0;
        for e in self.masks.free_edges(f, c) {
            self.clause_edges.push(e);
            self.prefix.push(product);
            product *= self.q[e];
        }
        if self.clause_edges.is_empty() {
            return Err(Status::Contradiction);
        }
        let mut suffix = 1.0;
        let mut residual: f64 = 0.0;
        for (&e, raw) in self.clause_edges.iter().zip(&mut self.prefix).rev() {
            *raw *= suffix;
            suffix *= self.q[e];
            if *raw < 1e-16 {
                *raw = 0.0;
            }
            residual = residual.max((*raw - self.eta[e]).abs());
        }
        Ok(residual)
    }
}
