use super::*;
use crate::formula::Data;

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

    #[inline]
    fn variable(
        &mut self,
        f: &Data,
        v: usize,
        store_norms: bool,
        scratch: &mut Vec<u64>,
    ) -> StatusResult<()> {
        if let Some(topology) = &f.feistel {
            let width = topology.k() as usize;
            topology.ordered_edges(v as u64, scratch);
            let split = scratch.partition_point(|&edge| edge < (1 << 32));
            let (positive, negative) = scratch.split_at(split);
            let groups = [positive, negative];
            self.update(
                v,
                store_norms,
                |sign| groups[sign].iter().map(|&e| e as u32 as usize),
                |e| e / width,
            )
        } else {
            self.update(
                v,
                store_norms,
                |sign| {
                    let group = 2 * v + sign;
                    f.var_edges[f.var_offsets[group] as usize..f.var_offsets[group + 1] as usize]
                        .iter()
                        .map(|&e| e as usize)
                },
                |e| f.edge_clause(e),
            )
        }
    }
}

impl Solver {
    /// Prefix/suffix products exclude an edge without division by (1-eta).
    /// This covers exact zero factors and makes every cavity O(1) after O(E) work.
    pub(super) fn cavities(&mut self, store_norms: bool) -> StatusResult<()> {
        self.timed("cavities", |s| s.scalar_cavities(store_norms))
    }
    pub(super) fn scalar_cavities(&mut self, store_norms: bool) -> StatusResult<()> {
        let f = self.formula.data();
        let mut state = ScalarCavity {
            q: &mut self.q,
            eta: &self.eta,
            saved_eta: &mut self.saved_eta,
            plus: &mut self.plus,
            minus: &mut self.minus,
            active: &self.active,
        };
        let mut scratch = Vec::new();
        for v in self.free.ones() {
            state.variable(f, v, store_norms, &mut scratch)?;
        }
        Ok(())
    }
    /// Refresh the outgoing cavities of one free variable from the current eta.
    pub(super) fn variable_cavity(&mut self, v: usize) -> StatusResult<()> {
        let f = self.formula.data();
        let mut state = ScalarCavity {
            q: &mut self.q,
            eta: &self.eta,
            saved_eta: &mut self.saved_eta,
            plus: &mut self.plus,
            minus: &mut self.minus,
            active: &self.active,
        };
        state.variable(f, v, false, &mut self.incident)
    }
}
