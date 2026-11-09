use super::*;

// A full width-three clause has three independent pair products. Keep
// damping lane-local, and reduce the residual after computing each lane.
#[inline]
pub(super) fn update_full_triple(q: &[f64], eta: &mut [f64], damping: f64) -> f64 {
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

impl Solver {
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
    pub(super) fn sweep_inner(&mut self) -> StatusResult<f64> {
        self.check_stop()?;
        if let Some(result) = self.gpu_converge(true) {
            return result.map(|r| r.residual);
        }
        if let Some(result) = self.parallel_sweep() {
            return result;
        }
        if let Some(queue) = &mut self.residual {
            queue.current = false;
        }
        self.cavities(false)?;
        let result = self.timed("clauses", Self::scalar_clauses);
        if result.is_ok() {
            self.sp_sweeps += 1;
        }
        result
    }
    pub(super) fn scalar_clauses(&mut self) -> StatusResult<f64> {
        let f = self.formula.data();
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
                for e in (crate::bitset::Ones {
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
    pub(super) fn observables(&mut self) -> StatusResult<(f64, bool)> {
        self.timed("observables", |s| {
            if let Some(result) = s.parallel_observables() {
                result
            } else {
                s.scalar_observables()
            }
        })
    }
    pub(super) fn scalar_observables(&mut self) -> StatusResult<(f64, bool)> {
        self.cavities(true)?;
        self.scalar_observables_from_cavities()
    }
    pub(super) fn scalar_observables_from_cavities(&mut self) -> StatusResult<(f64, bool)> {
        let f = self.formula.data();
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
            for e in self.masks.free_edges(f, c) {
                product *= self.q[e];
                clause_term += self.saved_eta[e].ln();
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
    pub(super) fn converge_inner(&mut self) -> StatusResult<SpReport> {
        self.timed("propagation", Self::propagate_inner)?;
        if let Some(result) = self.gpu_converge(false) {
            return result;
        }
        if self.residual.is_some() {
            return self.residual_converge();
        }
        self.sweep_converge()
    }
    pub(super) fn sweep_converge(&mut self) -> StatusResult<SpReport> {
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
}
