use super::*;

pub(super) struct Parallel {
    q: Vec<f64>,
    norm: Vec<f64>,
    products: Vec<[f64; 2]>,
    slot: Vec<u32>,
    clause_logs: Vec<f64>,
    variable_terms: Vec<f64>,
    ranges: Vec<(usize, usize)>,
}
impl Parallel {
    pub(super) fn allocated_bytes(&self) -> usize {
        (self.q.len() + self.norm.len() + self.clause_logs.len() + self.variable_terms.len()) * 8
            + self.slot.len() * std::mem::size_of::<u32>()
            + self.products.len() * 16
            + self.ranges.len() * 16
    }
    pub(super) fn new(s: &Solver) -> Self {
        let f = &s.formula.0;
        let mut slot = vec![0; f.literals.len()];
        for (p, &e) in f.var_edges.iter().enumerate() {
            slot[e as usize] = p as u32;
        }
        let chunk = f.n.div_ceil(s.execution.report.threads * 4).max(1);
        Self {
            q: vec![0.0; slot.len()],
            norm: vec![1.0; slot.len()],
            products: vec![[1.0; 2]; f.n],
            clause_logs: vec![0.0; s.formula.m()],
            variable_terms: vec![0.0; f.n],
            slot,
            ranges: (0..f.n)
                .step_by(chunk)
                .map(|v| (v, (v + chunk).min(f.n)))
                .collect(),
        }
    }
    pub(super) fn cavities(&mut self, s: &mut Solver, store_norms: bool) -> StatusResult<()> {
        let f = &s.formula.0;
        let mut q = self.q.as_mut_slice();
        let mut norm = self.norm.as_mut_slice();
        let mut products = self.products.as_mut_slice();
        let mut chunks = Vec::with_capacity(self.ranges.len());
        for &(start, end) in &self.ranges {
            let first = f.var_offsets[2 * start] as usize;
            let len = (f.var_offsets[2 * end] as usize) - first;
            let (a, b) = q.split_at_mut(len);
            q = b;
            let (c, d) = norm.split_at_mut(len);
            norm = d;
            let (e, g) = products.split_at_mut(end - start);
            products = g;
            chunks.push((start, first, a, c, e));
        }
        chunks
            .into_par_iter()
            .try_for_each(|(start, base, q, norm, products)| {
                for (local, products) in products.iter_mut().enumerate() {
                    let v = start + local;
                    if s.assignment[v] >= 0 {
                        continue;
                    }
                    *products = [1.0; 2];
                    for (sign, product) in products.iter_mut().enumerate() {
                        for p in f.var_offsets[2 * v + sign] as usize
                            ..f.var_offsets[2 * v + sign + 1] as usize
                        {
                            q[p - base] = *product;
                            *product *= 1.0 - s.eta[f.var_edges[p] as usize];
                        }
                    }
                    for sign in 0..2 {
                        let opposite = products[1 - sign];
                        let mut suffix = 1.0;
                        for p in (f.var_offsets[2 * v + sign] as usize
                            ..f.var_offsets[2 * v + sign + 1] as usize)
                            .rev()
                        {
                            let same = q[p - base] * suffix;
                            suffix *= 1.0 - s.eta[f.var_edges[p] as usize];
                            let z = opposite + (1.0 - opposite) * same;
                            if z <= 0.0 || !z.is_finite() {
                                return Err(Status::NumericalFailure);
                            }
                            q[p - base] = ((1.0 - opposite) * same / z).clamp(0.0, 1.0);
                            if store_norms {
                                norm[p - base] = z;
                            }
                        }
                    }
                }
                Ok(())
            })?;
        if store_norms {
            let active = &s.active;
            let assignment = &s.assignment;
            s.q.par_iter_mut()
                .zip(&mut s.saved_eta)
                .enumerate()
                .for_each(|(e, (q, norm))| {
                    *q = self.q[self.slot[e] as usize];
                    if active.contains(f.edge_clause(e))
                        && assignment[(f.literals[e] >> 1) as usize] < 0
                    {
                        *norm = self.norm[self.slot[e] as usize];
                    }
                });
        } else {
            s.q.par_iter_mut()
                .enumerate()
                .for_each(|(e, q)| *q = self.q[self.slot[e] as usize]);
        }
        s.plus
            .par_iter_mut()
            .zip(&mut s.minus)
            .zip(&self.products)
            .for_each(|((p, m), data)| {
                *p = data[0];
                *m = data[1];
            });
        Ok(())
    }
    pub(super) fn observables(&mut self, s: &mut Solver) -> StatusResult<(f64, bool)> {
        let f = &s.formula.0;
        self.norm
            .par_iter_mut()
            .zip(&s.saved_eta)
            .for_each(|(log, &norm)| *log = norm.ln());
        s.surveys
            .par_iter_mut()
            .zip(&mut s.cert)
            .zip(&mut self.variable_terms)
            .enumerate()
            .try_for_each(|(v, ((survey, cert), term))| {
                if s.assignment[v] >= 0 {
                    *term = 0.0;
                    return Ok(());
                }
                let plus = s.plus[v];
                let minus = s.minus[v];
                let norm = plus + (1.0 - plus) * minus;
                if norm <= 0.0 {
                    return Err(Status::NumericalFailure);
                }
                let st = (1.0 - plus) * minus / norm;
                let sf = (1.0 - minus) * plus / norm;
                *survey = [st, sf, (plus * minus / norm).clamp(0.0, 1.0)];
                *cert = 1.0 - st.min(sf);
                *term = (s.degree[v] as f64 - 1.0) * norm.ln();
                Ok(())
            })?;
        self.clause_logs
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(c, log)| {
                if !s.active.contains(c) {
                    return Ok(());
                }
                let mut product = 1.0;
                for e in f.clause_range(c) {
                    if s.assignment[(f.literals[e] >> 1) as usize] < 0 {
                        product *= s.q[e];
                    }
                }
                if product >= 1.0 {
                    return Err(Status::NumericalFailure);
                }
                *log = (-product).ln_1p();
                Ok(())
            })?;
        // Ordered accumulation preserves scalar rounding and decision trajectories.
        let mut variable_term = 0.0;
        let mut paramagnetic = true;
        for v in s.free.ones() {
            variable_term += self.variable_terms[v];
            let norm = s.plus[v] + (1.0 - s.plus[v]) * s.minus[v];
            if s.degree[v] >= 2 && norm != 1.0 {
                paramagnetic = false;
            }
        }
        let mut clause_term = 0.0;
        for c in s.active.ones() {
            for e in f.clause_range(c) {
                if s.assignment[(f.literals[e] >> 1) as usize] < 0 {
                    clause_term += self.norm[e];
                }
            }
            clause_term += self.clause_logs[c];
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
}
