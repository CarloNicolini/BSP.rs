use super::*;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum FreeLiterals {
    None,
    One(usize),
    Many,
}

#[cfg_attr(test, derive(Clone))]
pub(super) enum TrueCounts {
    Byte(Vec<u8>),
    Wide(Vec<u32>),
}

impl TrueCounts {
    pub(super) fn new(clauses: usize, max_width: usize) -> Self {
        if max_width <= u8::MAX as usize {
            Self::Byte(vec![0; clauses])
        } else {
            Self::Wide(vec![0; clauses])
        }
    }

    #[inline]
    pub(super) fn get(&self, clause: usize) -> u32 {
        match self {
            Self::Byte(counts) => u32::from(counts[clause]),
            Self::Wide(counts) => counts[clause],
        }
    }

    #[inline]
    pub(super) fn change(&mut self, clause: usize, fix: bool) -> u32 {
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

impl Solver {
    pub(super) fn set_fixed(&mut self, v: usize, value: i8, forced: bool) {
        self.timed("graph_mutations", |s| s.set_fixed_inner(v, value, forced));
    }
    pub(super) fn set_fixed_inner(&mut self, v: usize, value: i8, forced: bool) {
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
        let f = self.formula.data();
        let mut incident = std::mem::take(&mut self.incident);
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
                if let Some(queue) = &mut self.residual {
                    queue.clause(c);
                }
            }
            if value != i8::from(f.negative(e)) {
                if self.true_count.change(c, true) == 1 {
                    self.active.set(c, false);
                    self.execution.set_clause_active(c, false);
                    for other in self.masks.free_edges(f, c) {
                        let w = (f.literal(other) >> 1) as usize;
                        self.degree[w] -= 1;
                        self.execution.freeze_edge(other);
                        self.saved_eta[other] = self.eta[other];
                        self.eta[other] = 0.0;
                        if let Some(queue) = &mut self.residual {
                            queue.variable(w);
                        }
                    }
                }
            } else if self.active.contains(c)
                && !matches!(self.free_literals(c), FreeLiterals::Many)
            {
                self.unit_queue.push(c as u32);
            }
        }
        self.incident = incident;
        if forced {
            self.unit_assignments += 1;
            self.pending_units += 1;
        }
    }
    pub(super) fn unfix(&mut self, v: usize) {
        self.timed("graph_mutations", |s| s.unfix_inner(v));
    }
    pub(super) fn unfix_inner(&mut self, v: usize) {
        self.decisions.remove(v as u32, &self.cert);
        let value = self.assignment[v];
        self.assignment[v] = -1;
        self.execution.set_assignment(v, -1);
        self.free.set(v, true);
        if let Some(queue) = &mut self.residual {
            queue.variable(v);
        }
        let f = self.formula.data();
        let mut incident = std::mem::take(&mut self.incident);
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
                    for other in self.masks.free_edges(f, c) {
                        let w = (f.literal(other) >> 1) as usize;
                        self.degree[w] += 1;
                        self.execution.restore_edge(other);
                        self.eta[other] = self.saved_eta[other];
                        if let Some(queue) = &mut self.residual {
                            queue.variable(w);
                        }
                    }
                }
                if let Some(queue) = &mut self.residual {
                    queue.clause(c);
                }
                if !matches!(self.free_literals(c), FreeLiterals::Many) {
                    self.unit_queue.push(c as u32);
                }
            }
        }
        self.incident = incident;
    }
    // Unit propagation only distinguishes zero, one and multiple free bits.
    // Stop as soon as two bits are found, even for very wide clauses.
    pub(super) fn free_literals(&self, c: usize) -> FreeLiterals {
        let f = self.formula.data();
        let mut unit_edge = None;
        let first = f.mask_offset(c);
        for w in first..f.mask_offset(c + 1) {
            let mask = self.masks.get(w);
            if mask == 0 {
                continue;
            }
            if unit_edge.is_some() || mask & (mask - 1) != 0 {
                return FreeLiterals::Many;
            }
            unit_edge = Some(f.offset(c) + 64 * (w - first) + mask.trailing_zeros() as usize);
        }
        unit_edge.map_or(FreeLiterals::None, FreeLiterals::One)
    }

    pub fn propagate(&mut self) -> StatusResult<usize> {
        let result = self.timed("propagation", Self::propagate_inner);
        self.sync_messages();
        if self.execution.report.device_error.is_some() {
            return Err(Status::NumericalFailure);
        }
        result
    }
    pub(super) fn propagate_inner(&mut self) -> StatusResult<usize> {
        let before = self.free.len();
        while let Some(c) = self.unit_queue.pop() {
            self.check_stop()?;
            let c = c as usize;
            if !self.active.contains(c) {
                continue;
            }
            let e = match self.free_literals(c) {
                FreeLiterals::None => {
                    self.unit_queue.push(c as u32);
                    return Err(Status::Contradiction);
                }
                FreeLiterals::One(edge) => edge,
                FreeLiterals::Many => continue,
            };
            let f = self.formula.data();
            let lit = f.literal(e);
            self.cert[(lit >> 1) as usize] = 1.0;
            self.set_fixed((lit >> 1) as usize, (1 - (lit & 1)) as i8, true);
        }
        Ok(before - self.free.len())
    }
}
