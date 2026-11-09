use super::*;

impl Solver {
    /// Select only the top batch, using deterministic variable-ID tie breaking.
    pub fn decimate(&mut self) -> usize {
        let result = self.timed("decimation", Self::decimate_inner);
        self.sync_messages();
        result
    }
    pub(super) fn decimate_inner(&mut self) -> usize {
        let ranking_started = self
            .execution
            .options
            .profile
            .as_ref()
            .map(|_| Instant::now());
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
        if let Some(started) = ranking_started {
            self.execution
                .record("ranking", started.elapsed().as_secs_f64());
        }
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
        let result = self.timed("backtracking", Self::backtrack_inner);
        self.sync_messages();
        result
    }
    pub(super) fn backtrack_inner(&mut self) -> usize {
        self.terminal = None;
        self.solution = None;
        let k = (self.last_batch + self.pending_units).min(self.decisions.len());
        self.pending_units = 0;
        for _ in 0..k {
            let v = self
                .decisions
                .peek()
                .expect("release batch fits fixed decision count");
            self.unfix(v as usize);
        }
        self.releases += k as u64;
        k
    }
    /// Advance through one fixed point and one BSP action. Terminal states are sticky.
    pub fn advance(&mut self) -> StatusResult<Step> {
        let result = self.advance_internal();
        self.sync_messages();
        if self.execution.report.device_error.is_some() {
            return Err(Status::NumericalFailure);
        }
        result
    }
    pub(super) fn advance_internal(&mut self) -> StatusResult<Step> {
        if let Some(status) = self.terminal {
            return Err(status);
        }
        let result = self.advance_inner();
        if let Err(status) = result {
            self.terminal = Some(status);
        }
        result
    }
    pub(super) fn advance_inner(&mut self) -> StatusResult<Step> {
        self.check_stop()?;
        if self.steps.len() >= self.options.max_steps {
            return Err(Status::StepLimit);
        }
        let report = self.converge_inner()?;
        // Counters are recorded before the action changes them.
        let (index, free_variables, active_clauses, sp_sweeps) = (
            self.steps.len(),
            self.free.len(),
            self.active.len(),
            self.sp_sweeps,
        );
        let step = |action, changed| Step {
            index,
            action,
            free_variables,
            active_clauses,
            sp_sweeps,
            residual: report.residual,
            complexity: report.complexity,
            changed,
        };
        if report.paramagnetic || self.active.is_empty() {
            let step = step(Action::Walksat, 0);
            self.steps.push(step);
            let assignment = self.timed("walksat", Self::walksat)?;
            self.solution = Some(assignment);
            self.terminal = Some(Status::Sat);
            return Ok(step);
        }
        let step =
            if self.back_moves as f64 / (self.dec_moves as f64) < self.options.backtrack_ratio {
                self.back_moves += 1;
                step(
                    Action::Backtrack,
                    self.timed("backtracking", Self::backtrack_inner),
                )
            } else {
                self.dec_moves += 1;
                if report.complexity < -2.0 {
                    return Err(Status::NegativeComplexity);
                }
                step(
                    Action::Decimate,
                    self.timed("decimation", Self::decimate_inner),
                )
            };
        self.steps.push(step);
        Ok(step)
    }
    pub(crate) fn residual_variable_count(&self) -> usize {
        self.free.ones().filter(|&v| self.degree[v] > 0).count()
    }
    pub fn solve(&mut self) -> SolveResult {
        while self.terminal.is_none() {
            let _ = self.advance_internal();
        }
        self.sync_messages();
        if let Err(error) = self.write_profile() {
            self.execution.report.profile_error = Some(error.to_string());
        }
        SolveResult {
            execution: self.execution.report.clone(),
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
                (f.offset(c)..f.offset(c + 1))
                    .filter_map(|e| {
                        let l = f.literal(e);
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
