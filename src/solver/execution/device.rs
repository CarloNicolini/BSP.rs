use super::*;

impl Solver {
    pub(crate) fn sync_messages(&mut self) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &self.execution.gpu {
            let started = self
                .execution
                .options
                .profile
                .as_ref()
                .map(|_| Instant::now());
            gpu.snapshot(&mut self.eta, &mut self.saved_eta);
            if let Some(started) = started {
                self.execution
                    .record("snapshots", started.elapsed().as_secs_f64());
            }
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(cuda) = &mut self.execution.cuda {
            let started = self
                .execution
                .options
                .profile
                .as_ref()
                .map(|_| Instant::now());
            if let Err(error) = cuda.snapshot(&mut self.eta, &mut self.saved_eta) {
                self.device_failure(error);
            }
            if let Some(started) = started {
                self.execution
                    .record("snapshots", started.elapsed().as_secs_f64());
            }
        }
    }
    #[cfg(any(
        all(feature = "metal", target_os = "macos"),
        all(feature = "cuda", target_os = "linux")
    ))]
    pub(super) fn cpu_fallback(&mut self, reason: &str) {
        self.execution.report.fallbacks.push(reason.into());
        self.execution.report.backend = Backend::Cpu;
        self.execution.report.precision = Precision::F64;
        self.execution.report.threads = 1;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            self.execution.gpu = None;
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        {
            self.execution.cuda = None;
        }
    }
    pub(crate) fn upload_messages(&mut self) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &mut self.execution.gpu {
            if let Err(reason) = gpu.upload(&self.eta, &self.saved_eta) {
                self.cpu_fallback(&reason);
            }
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.execution.cuda.is_some() {
            if self.execution.report.precision == Precision::F32
                && (!bsp_cuda::safe_conversion(&self.eta)
                    || !bsp_cuda::safe_conversion(&self.saved_eta))
            {
                self.cpu_fallback("unsafe CUDA f32 message conversion");
            } else if let Some(cuda) = &mut self.execution.cuda {
                if let Err(error) = cuda.upload(&self.eta, &self.saved_eta) {
                    self.device_failure(error);
                }
            }
        }
    }
    pub(crate) fn gpu_converge(&mut self, single: bool) -> Option<StatusResult<SpReport>> {
        self.gpu_compute(single, if single { 1 } else { self.options.max_sweeps })
    }
    pub(crate) fn gpu_compute(
        &mut self,
        single: bool,
        limit: usize,
    ) -> Option<StatusResult<SpReport>> {
        if let Some(result) = self.metal_compute(single, limit) {
            Some(result)
        } else {
            self.cuda_compute(single, limit)
        }
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(super) fn device_failure(&mut self, error: String) -> Status {
        self.execution.report.device_error = Some(error);
        self.terminal = Some(Status::NumericalFailure);
        self.solution = None;
        Status::NumericalFailure
    }
}
