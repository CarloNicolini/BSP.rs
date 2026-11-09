use super::*;

const OUTPUT_WINDOW: usize = 2048;

#[derive(Clone, Copy, Default)]
pub(super) struct CavityOutput {
    edge: u32,
    q: f64,
    norm: f64,
}

pub(super) struct ImplicitParallel {
    pub(super) outputs: Vec<CavityOutput>,
}

impl ImplicitParallel {
    pub(super) fn new(edges: usize) -> Self {
        Self {
            outputs: vec![CavityOutput::default(); edges.min(OUTPUT_WINDOW)],
        }
    }

    pub(super) fn cavities(&mut self, s: &mut Solver, store_norms: bool) -> StatusResult<()> {
        let topology = s.formula.feistel_topology().unwrap();
        let k = topology.k() as usize;
        if s.degree.iter().any(|&d| d as usize > self.outputs.len()) {
            return s.scalar_cavities(store_norms);
        }

        let mut start = 0;
        while start < s.formula.n() {
            let mut end = start;
            let mut len = 0;
            while end < s.formula.n() && len + s.degree[end] as usize <= self.outputs.len() {
                len += s.degree[end] as usize;
                end += 1;
            }
            let chunk = (end - start)
                .div_ceil(s.execution.report.threads * 2)
                .max(1);
            let mut jobs = Vec::new();
            let mut output = &mut self.outputs[..len];
            let mut plus = &mut s.plus[start..end];
            let mut minus = &mut s.minus[start..end];
            for first in (start..end).step_by(chunk) {
                let last = (first + chunk).min(end);
                let edges: usize = s.degree[first..last].iter().map(|&d| d as usize).sum();
                let (head, tail) = output.split_at_mut(edges);
                let (p, p_tail) = plus.split_at_mut(last - first);
                let (m, m_tail) = minus.split_at_mut(last - first);
                jobs.push((first, head, p, m));
                (output, plus, minus) = (tail, p_tail, m_tail);
            }
            let active = &s.active;
            let eta = &s.eta;
            let assignment = &s.assignment;
            let degree = &s.degree;
            jobs.into_par_iter().try_for_each_init(
                Vec::new,
                |edges, (first, mut output, plus, minus)| {
                    for (i, (plus, minus)) in plus.iter_mut().zip(minus).enumerate() {
                        let v = first + i;
                        if assignment[v] >= 0 {
                            continue;
                        }
                        topology.ordered_edges(v as u64, edges);
                        // Inactive messages are zero. Their factors are exactly one,
                        // and their cavity values are never consumed by active clauses.
                        edges.retain(|&e| active.contains(e as u32 as usize / k));
                        debug_assert_eq!(edges.len(), degree[v] as usize);
                        let (variable, tail) = output.split_at_mut(degree[v] as usize);
                        output = tail;
                        let split = edges.partition_point(|&e| e < (1 << 32));
                        let mut products = [1.0; 2];
                        for (range, product) in [0..split, split..edges.len()]
                            .into_iter()
                            .zip(&mut products)
                        {
                            for i in range {
                                let edge = edges[i] as u32;
                                variable[i] = CavityOutput {
                                    edge,
                                    q: *product,
                                    norm: 1.0,
                                };
                                *product *= 1.0 - eta[edge as usize];
                            }
                        }
                        [*plus, *minus] = products;
                        for (range, opposite) in [0..split, split..edges.len()]
                            .into_iter()
                            .zip([products[1], products[0]])
                        {
                            let mut suffix = 1.0;
                            for i in range.rev() {
                                let slot = &mut variable[i];
                                let same = slot.q * suffix;
                                suffix *= 1.0 - eta[slot.edge as usize];
                                let norm = opposite + (1.0 - opposite) * same;
                                if norm <= 0.0 || !norm.is_finite() {
                                    return Err(Status::NumericalFailure);
                                }
                                slot.q = ((1.0 - opposite) * same / norm).clamp(0.0, 1.0);
                                slot.norm = norm;
                            }
                        }
                    }
                    Ok(())
                },
            )?;
            for &entry in &self.outputs[..len] {
                let edge = entry.edge as usize;
                s.q[edge] = entry.q;
                if store_norms {
                    s.saved_eta[edge] = entry.norm;
                }
            }
            start = end;
        }
        Ok(())
    }
}
