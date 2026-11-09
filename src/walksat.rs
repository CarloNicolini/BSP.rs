use crate::{bitset::BitSet, Solver, Status, StatusResult};
use rand::Rng;

/// Random selection and removal both cost O(1). A word mask would make random
/// unsatisfied-clause selection depend on rank/select or a second index.
struct Unsatisfied {
    clauses: Vec<u32>,
    positions: Vec<u32>,
}
impl Unsatisfied {
    fn new(m: usize) -> Self {
        Self {
            clauses: Vec::new(),
            positions: vec![u32::MAX; m],
        }
    }
    fn insert(&mut self, clause: usize, slot: usize) {
        debug_assert_eq!(self.positions[slot], u32::MAX);
        self.positions[slot] = self.clauses.len() as u32;
        self.clauses.push(clause as u32);
    }
    fn remove(&mut self, slot: usize, slot_of: impl FnOnce(usize) -> usize) {
        let p = self.positions[slot] as usize;
        self.clauses.swap_remove(p);
        if p < self.clauses.len() {
            self.positions[slot_of(self.clauses[p] as usize)] = p as u32;
        }
        self.positions[slot] = u32::MAX;
    }
}

impl Solver {
    pub(crate) fn walksat(&mut self) -> StatusResult<Vec<i8>> {
        self.mark_walksat();
        // Arc clone lets the flip loop mutate the solver's counters and RNG.
        let formula = self.formula.clone();
        let f = formula.data();
        let mut values = self.assignment.clone();
        // Clause activity is immutable during WalkSAT. Store dense state for
        // the residual when it is sparse, using rank to map original IDs.
        let ranks = (self.active.len() <= formula.m() / 2).then(|| self.active.rank_index());
        let slots = if ranks.is_some() {
            self.active.len()
        } else {
            formula.m()
        };
        let slot_of = |active: &BitSet, c: usize| {
            ranks
                .as_ref()
                .map_or(c, |index| active.rank_before(index, c))
        };
        let mut count = vec![0u32; slots];
        let mut true_xor = vec![0u32; slots];
        let mut breaks = vec![0u32; f.n];
        let mut unsat = Unsatisfied::new(slots);
        let width = self
            .active
            .ones()
            .map(|c| f.offset(c + 1) - f.offset(c))
            .max()
            .unwrap_or(0);
        let mut choices = Vec::with_capacity(width);
        let mut incident = Vec::new();
        let mut best = Vec::with_capacity(width);
        // Only free variables occurring in active clauses can be flipped.
        let residual_variables = self.residual_variable_count();
        let max_flips = flip_budget(
            self.options.max_flips,
            self.options.walksat_restarts,
            residual_variables,
        );
        let per_restart = max_flips.div_ceil(self.options.walksat_restarts as u64);
        let mut total_flips = 0u64;
        for _ in 0..self.options.walksat_restarts {
            let initialization_started = self
                .execution
                .options
                .profile
                .as_ref()
                .map(|_| std::time::Instant::now());
            self.check_stop()?;
            for (v, value) in values.iter_mut().enumerate() {
                if self.assignment[v] < 0 {
                    *value = i8::from(self.rng.random::<bool>());
                }
            }
            count.fill(0);
            true_xor.fill(0);
            breaks.fill(0);
            unsat.clauses.clear();
            unsat.positions.fill(u32::MAX);
            for (i, c) in self.active.ones().enumerate() {
                let slot = if ranks.is_some() { i } else { c };
                for e in f.offset(c)..f.offset(c + 1) {
                    let lit = f.literal(e);
                    let v = (lit >> 1) as usize;
                    if values[v] == (1 - (lit & 1)) as i8 {
                        count[slot] += 1;
                        true_xor[slot] ^= v as u32;
                    }
                }
                if count[slot] == 0 {
                    unsat.insert(c, slot);
                } else if count[slot] == 1 {
                    breaks[true_xor[slot] as usize] += 1;
                }
            }
            if let Some(started) = initialization_started {
                self.execution
                    .record("walksat_initialization", started.elapsed().as_secs_f64());
            }
            for local_flips in 0..=per_restart {
                if unsat.clauses.is_empty() {
                    if !self.timed("verification", |_| formula.is_satisfied(&values)) {
                        return Err(Status::NumericalFailure);
                    }
                    return Ok(values);
                }
                if total_flips >= max_flips {
                    return Err(Status::FlipLimit);
                }
                if total_flips & 255 == 0 {
                    self.check_stop()?;
                }
                // The inclusive loop checks satisfaction after the final flip.
                // A restart starts before another flip exceeds its quota.
                if local_flips == per_restart {
                    break;
                }
                let c = unsat.clauses[self.rng.random_range(0..unsat.clauses.len())] as usize;
                choices.clear();
                best.clear();
                let mut min_break = u32::MAX;
                for e in f.offset(c)..f.offset(c + 1) {
                    let v = (f.literal(e) >> 1) as usize;
                    if self.assignment[v] >= 0 {
                        continue;
                    }
                    choices.push(v);
                    if breaks[v] < min_break {
                        min_break = breaks[v];
                        best.clear();
                    }
                    if breaks[v] == min_break {
                        best.push(v);
                    }
                }
                if choices.is_empty() {
                    return Err(Status::Contradiction);
                }
                let pick = if min_break > 0 && self.rng.random::<f64>() < self.options.walksat_noise
                {
                    &choices
                } else {
                    &best
                };
                let v = pick[self.rng.random_range(0..pick.len())];
                let old = values[v];
                values[v] ^= 1;
                for e in f.variable_edges(v, &mut incident) {
                    let c = f.edge_clause(e);
                    if !self.active.contains(c) {
                        continue;
                    }
                    let slot = slot_of(&self.active, c);
                    if old != i8::from(f.negative(e)) {
                        if count[slot] == 1 {
                            breaks[v] -= 1;
                            unsat.insert(c, slot);
                        }
                        count[slot] -= 1;
                        true_xor[slot] ^= v as u32;
                        if count[slot] == 1 {
                            breaks[true_xor[slot] as usize] += 1;
                        }
                    } else {
                        if count[slot] == 1 {
                            breaks[true_xor[slot] as usize] -= 1;
                        }
                        if count[slot] == 0 {
                            unsat.remove(slot, |other| slot_of(&self.active, other));
                            breaks[v] += 1;
                        }
                        count[slot] += 1;
                        true_xor[slot] ^= v as u32;
                    }
                }
                total_flips += 1;
                self.record_flip();
            }
        }
        Err(Status::FlipLimit)
    }
}

fn flip_budget(explicit: u64, restarts: usize, variables: usize) -> u64 {
    if explicit > 0 {
        return explicit;
    }
    (variables as u64)
        .saturating_mul(100)
        .max(10_000)
        .saturating_mul(restarts as u64)
}

#[cfg(test)]
mod budget_tests {
    use super::flip_budget;
    #[test]
    fn sparse_residual_keeps_walksat_order() {
        use crate::{Formula, Options, Solver};
        use rand::{rngs::SmallRng, SeedableRng};

        let residual: Vec<Vec<i32>> = Formula::random(3, 2.0, 64, 7)
            .unwrap()
            .clauses()
            .into_iter()
            .map(|clause| clause.into_iter().map(|lit| lit + lit.signum()).collect())
            .collect();
        let interleaved = residual
            .iter()
            .flat_map(|c| [vec![1, 2], c.clone(), vec![1, -2]])
            .collect();
        let mut compact =
            Solver::new(Formula::new(65, interleaved).unwrap(), Options::default()).unwrap();
        let mut dense =
            Solver::new(Formula::new(65, residual).unwrap(), Options::default()).unwrap();
        for s in [&mut compact, &mut dense] {
            s.fix(0, true).unwrap();
            s.rng = SmallRng::seed_from_u64(11);
        }
        assert_eq!(compact.active.len(), dense.active.len());
        let expected = dense.walksat().unwrap();
        assert_eq!(compact.walksat().unwrap(), expected);
    }

    #[test]
    fn scales_with_residual_and_restarts_and_preserves_overrides() {
        assert_eq!(flip_budget(0, 4, 470_226), 188_090_400);
        assert_eq!(flip_budget(0, 2, 1), 20_000);
        assert_eq!(flip_budget(17, 4, 470_226), 17);
        assert_eq!(flip_budget(0, usize::MAX, usize::MAX), u64::MAX);
    }
}
