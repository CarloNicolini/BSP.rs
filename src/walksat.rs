use crate::{Solver, Status};
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
            clauses: Vec::with_capacity(m),
            positions: vec![u32::MAX; m],
        }
    }
    fn insert(&mut self, c: usize) {
        debug_assert_eq!(self.positions[c], u32::MAX);
        self.positions[c] = self.clauses.len() as u32;
        self.clauses.push(c as u32);
    }
    fn remove(&mut self, c: usize) {
        let p = self.positions[c] as usize;
        self.clauses.swap_remove(p);
        if p < self.clauses.len() {
            self.positions[self.clauses[p] as usize] = p as u32;
        }
        self.positions[c] = u32::MAX;
    }
}

impl Solver {
    pub(crate) fn walksat(&mut self) -> std::result::Result<Vec<i8>, Status> {
        // Arc clone lets the flip loop mutate the solver's counters and RNG.
        let formula = self.formula.clone();
        let f = &formula.0;
        let mut values = self.assignment.clone();
        let mut count = vec![0u32; formula.m()];
        let mut true_xor = vec![0u32; formula.m()];
        let mut breaks = vec![0u32; f.n];
        let mut unsat = Unsatisfied::new(formula.m());
        let width = f
            .offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        let mut choices = Vec::with_capacity(width);
        let mut best = Vec::with_capacity(width);
        let per_restart = self
            .options
            .max_flips
            .div_ceil(self.options.walksat_restarts as u64);
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
            for c in self.active.ones() {
                for e in f.offsets[c] as usize..f.offsets[c + 1] as usize {
                    let lit = f.literals[e];
                    let v = (lit >> 1) as usize;
                    if values[v] == (1 - (lit & 1)) as i8 {
                        count[c] += 1;
                        true_xor[c] ^= v as u32;
                    }
                }
                if count[c] == 0 {
                    unsat.insert(c);
                } else if count[c] == 1 {
                    breaks[true_xor[c] as usize] += 1;
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
                if total_flips >= self.options.max_flips {
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
                for e in f.offsets[c] as usize..f.offsets[c + 1] as usize {
                    let v = (f.literals[e] >> 1) as usize;
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
                for p in f.var_offsets[2 * v] as usize..f.var_offsets[2 * v + 2] as usize {
                    let e = f.var_edges[p] as usize;
                    let c = f.edge_clause[e] as usize;
                    if !self.active.contains(c) {
                        continue;
                    }
                    if old == (1 - (f.literals[e] & 1)) as i8 {
                        if count[c] == 1 {
                            breaks[v] -= 1;
                            unsat.insert(c);
                        }
                        count[c] -= 1;
                        true_xor[c] ^= v as u32;
                        if count[c] == 1 {
                            breaks[true_xor[c] as usize] += 1;
                        }
                    } else {
                        if count[c] == 1 {
                            breaks[true_xor[c] as usize] -= 1;
                        }
                        if count[c] == 0 {
                            unsat.remove(c);
                            breaks[v] += 1;
                        }
                        count[c] += 1;
                        true_xor[c] ^= v as u32;
                    }
                }
                total_flips += 1;
                self.record_flip();
            }
        }
        Err(Status::FlipLimit)
    }
}
