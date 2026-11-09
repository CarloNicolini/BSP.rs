use super::*;

impl Solver {
    pub(super) fn cuda_compute(
        &mut self,
        single: bool,
        limit: usize,
    ) -> Option<StatusResult<SpReport>> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            let mut gpu = self.execution.cuda.take()?;
            let mut count = 0;
            let mut batch = 1;
            let mut fallback = None;
            let result = (|| {
                loop {
                    self.check_stop()?;
                    let report = gpu
                        .sweeps(
                            batch.min(limit - count),
                            if single { -1.0 } else { self.options.epsilon },
                            self.options.damping,
                            count == 0,
                        )
                        .map_err(|e| self.device_failure(e))?;
                    self.execution.record("gpu", report.gpu_seconds);
                    self.execution
                        .record("gpu_submission_and_wait", report.wall_seconds);
                    for (name, seconds) in &report.phases {
                        self.execution.record(name, *seconds);
                    }
                    count += report.sweeps;
                    self.sp_sweeps += report.sweeps as u64;
                    self.execution.report.cuda_sweeps += report.sweeps as u64;
                    if report.numerical_error {
                        fallback = Some("CUDA cavity arithmetic underflowed or became non-finite");
                        return Err(Status::NumericalFailure);
                    }
                    if report.contradiction {
                        return Err(Status::Contradiction);
                    }
                    if single && count >= limit {
                        return Ok(SpReport {
                            sweeps: count,
                            residual: report.residual,
                            complexity: 0.0,
                            paramagnetic: false,
                        });
                    }
                    if !single && report.residual <= self.options.epsilon {
                        let obs = gpu.observables().map_err(|e| self.device_failure(e))?;
                        self.execution.record("gpu", obs.gpu_seconds);
                        for (name, seconds) in &obs.phases {
                            self.execution.record(name, *seconds);
                        }
                        if obs.numerical_error {
                            fallback = Some("CUDA observables became numerically unsafe");
                            return Err(Status::NumericalFailure);
                        }
                        gpu.surveys(&mut self.surveys, &mut self.cert)
                            .map_err(|e| self.device_failure(e))?;
                        // Re-evaluate decision-sensitive observables in f64 on
                        // the current rounded messages, without changing them.
                        let guard =
                            32.0 * if self.execution.report.precision == Precision::F32 {
                                f32::EPSILON as f64
                            } else {
                                f64::EPSILON
                            } * self.formula.edges() as f64;
                        let (complexity, paramagnetic) =
                            if obs.paramagnetic || obs.complexity <= guard {
                                gpu.snapshot(&mut self.eta, &mut self.saved_eta)
                                    .map_err(|e| self.device_failure(e))?;
                                self.execution.report.observables_validations += 1;
                                self.timed("observables_f64_validation", Self::scalar_observables)?
                            } else {
                                (obs.complexity, obs.paramagnetic)
                            };
                        return Ok(SpReport {
                            sweeps: count,
                            residual: report.residual,
                            complexity,
                            paramagnetic,
                        });
                    }
                    if count >= limit {
                        return Err(Status::NonConverged);
                    }
                    // Target <=2ms of GPU work per cancellation checkpoint, cap at 16 sweeps.
                    batch = ((0.002 / (report.gpu_seconds / report.sweeps.max(1) as f64)).floor()
                        as usize)
                        .clamp(1, 16);
                }
            })();
            if let Some(reason) = fallback {
                if let Err(error) = gpu
                    .restore_checkpoint()
                    .and_then(|()| gpu.snapshot(&mut self.eta, &mut self.saved_eta))
                {
                    self.execution.cuda = Some(gpu);
                    return Some(Err(self.device_failure(error)));
                }
                self.cpu_fallback(reason);
                return if single {
                    Some((|| {
                        let mut residual = 0.0;
                        for _ in 0..limit {
                            residual = self.sweep_inner()?;
                        }
                        Ok(SpReport {
                            sweeps: limit,
                            residual,
                            complexity: 0.0,
                            paramagnetic: false,
                        })
                    })())
                } else {
                    Some(self.converge_inner())
                };
            }
            self.execution.report.allocated_bytes = gpu.allocated_bytes;
            self.execution.cuda = Some(gpu);
            Some(result)
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        {
            let _ = (single, limit);
            None
        }
    }
}
