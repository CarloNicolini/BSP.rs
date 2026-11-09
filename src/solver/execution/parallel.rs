use super::*;

// Implicit output is bounded independently of the graph. Temporary incidence
// decoding uses at most one variable's edges per Rayon task.
pub(super) const IMPLICIT_MIN: usize = 1024;
pub(super) fn explicit_min(threads: usize) -> usize {
    // Direct CSR traversal is cheap. Leave enough work per worker to
    // amortize the cavity, scatter, and clause dispatches.
    (threads * 4096).max(8192)
}

impl Solver {
    fn parallel_work_available(&self) -> bool {
        if self
            .execution
            .batch_active
            .as_ref()
            .is_some_and(|n| n.load(Ordering::Relaxed) * 2 >= self.execution.report.threads)
        {
            return false;
        }
        let min = if self.formula.feistel_topology().is_some() {
            IMPLICIT_MIN
        } else {
            explicit_min(self.execution.report.threads)
        };
        self.free.len().saturating_mul(self.formula.edges()) / self.formula.n().max(1) >= min
    }

    pub(crate) fn parallel_sweep(&mut self) -> Option<StatusResult<f64>> {
        if !self.parallel_work_available() {
            return None;
        }
        let pool = self.execution.pool.as_ref()?.clone();
        let result = if let Some(mut scratch) = self.execution.implicit_parallel.take() {
            let result = pool.install(|| {
                self.timed("cavities", |s| scratch.cavities(s, false))
                    .and_then(|()| self.timed("clauses", parallel_clauses))
            });
            self.execution.implicit_parallel = Some(scratch);
            result
        } else {
            let mut scratch = self.execution.parallel.take()?;
            let result = pool.install(|| {
                self.timed("cavities", |s| scratch.cavities(s, false))
                    .and_then(|()| self.timed("clauses", parallel_clauses))
            });
            self.execution.parallel = Some(scratch);
            result
        };
        if result.is_ok() {
            self.sp_sweeps += 1;
            self.execution.report.parallel_sweeps += 1;
        }
        Some(result)
    }

    pub(crate) fn parallel_observables(&mut self) -> Option<StatusResult<(f64, bool)>> {
        if !self.parallel_work_available() {
            return None;
        }
        let pool = self.execution.pool.as_ref()?.clone();
        if let Some(mut scratch) = self.execution.implicit_parallel.take() {
            let result = pool.install(|| self.timed("cavities", |s| scratch.cavities(s, true)));
            self.execution.implicit_parallel = Some(scratch);
            return Some(result.and_then(|()| self.scalar_observables_from_cavities()));
        }
        let mut scratch = self.execution.parallel.take()?;
        let result = pool.install(|| {
            self.timed("cavities", |s| scratch.cavities(s, true))
                .and_then(|()| scratch.observables(self))
        });
        self.execution.parallel = Some(scratch);
        Some(result)
    }
}

fn parallel_clauses(s: &mut Solver) -> StatusResult<f64> {
    let f = &s.formula.0;
    let chunk = s
        .formula
        .m()
        .div_ceil(s.execution.report.threads * 4)
        .max(1);
    let mut eta = s.eta.as_mut_slice();
    let mut chunks = Vec::new();
    for start in (0..s.formula.m()).step_by(chunk) {
        let end = (start + chunk).min(s.formula.m());
        let first = f.offset(start);
        let len = f.offset(end) - first;
        let (a, b) = eta.split_at_mut(len);
        eta = b;
        chunks.push((start, end, first, a));
    }
    let active = &s.active;
    let masks = &s.masks;
    let q = &s.q;
    let damping = s.options.damping;
    chunks
        .into_par_iter()
        .map_init(
            Vec::<(usize, f64)>::new,
            |prefix, (start, end, base, eta)| {
                let mut residual: f64 = 0.0;
                for c in start..end {
                    if !active.contains(c) {
                        continue;
                    }
                    prefix.clear();
                    let mut product = 1.0;
                    let start = f.offset(c);
                    let first_word = f.mask_offset(c);
                    if f.offset(c + 1) - start == 3 && masks.get(first_word) == 7 {
                        residual = residual.max(crate::solver::sp::update_full_triple(
                            &q[start..start + 3],
                            &mut eta[start - base..start - base + 3],
                            damping,
                        ));
                        continue;
                    }
                    for w in first_word..f.mask_offset(c + 1) {
                        for e in (crate::bitset::Ones {
                            word: masks.get(w),
                            base: start + 64 * (w - first_word),
                        }) {
                            prefix.push((e, product));
                            product *= q[e];
                        }
                    }
                    if prefix.is_empty() {
                        return Err(Status::Contradiction);
                    }
                    let mut suffix = 1.0;
                    for &(e, p) in prefix.iter().rev() {
                        let raw = p * suffix;
                        suffix *= q[e];
                        let raw = if raw < 1e-16 { 0.0 } else { raw };
                        residual = residual.max((raw - eta[e - base]).abs());
                        eta[e - base] = raw + (eta[e - base] - raw) * damping;
                    }
                }
                Ok(residual)
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok(a.max(b)))
}
