use super::implicit::{CavityOutput, ImplicitParallel};
use super::parallel::{explicit_min, IMPLICIT_MIN};
use super::*;

impl Solver {
    pub fn with_execution(
        formula: Formula,
        options: Options,
        execution: ExecutionOptions,
    ) -> Result<Self> {
        if formula.feistel_topology().is_some() && execution.backend.is_cuda() {
            return Err(Error(
                "Feistel formulas require --backend cpu, cpu-parallel, or metal".into(),
            ));
        }
        let construction_started = Instant::now();
        options.validate()?;
        execution.backend.ensure_available()?;
        let setup = execution.profile.as_ref().map(|_| Instant::now());
        let configured = Execution::new(execution)?;
        // Check device availability before allocating the host solver arrays.
        // Unsupported f32 settings intentionally fall back to CPU without a device.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if configured.options.backend == Backend::Metal
            && options.damping as f32 != 1.0
            && options.epsilon as f32 != 0.0
        {
            bsp_metal::device_budget()?;
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if configured.options.backend.is_cuda()
            && (configured.options.backend == Backend::Cuda
                || options.damping as f32 != 1.0 && options.epsilon as f32 != 0.0)
        {
            bsp_cuda::device_budget()?;
        }
        let mut s = Self::new(formula, options)?;
        s.started = construction_started;
        s.execution = configured;
        match s.execution.options.backend {
            Backend::Cpu => (),
            Backend::CpuParallel => {
                if s.execution.report.threads > 1 {
                    if s.formula.feistel_topology().is_some() && s.formula.edges() >= IMPLICIT_MIN {
                        let scratch = ImplicitParallel::new(s.formula.edges());
                        s.execution.report.allocated_bytes =
                            scratch.outputs.len() * std::mem::size_of::<CavityOutput>();
                        s.execution.implicit_parallel = Some(scratch);
                    } else if s.formula.feistel_topology().is_none()
                        && s.formula.edges() >= explicit_min(s.execution.report.threads)
                    {
                        let scratch = Parallel::new(&s);
                        s.execution.report.allocated_bytes = scratch.allocated_bytes();
                        s.execution.parallel = Some(scratch);
                    }
                }
            }
            Backend::Metal => s.init_metal()?,
            Backend::Cuda | Backend::CudaF32 => s.init_cuda()?,
        }
        s.sync_messages();
        if let Some(error) = &s.execution.report.device_error {
            return Err(Error(error.clone()));
        }
        if let Some(started) = setup {
            s.execution
                .record("backend_setup", started.elapsed().as_secs_f64());
        }
        Ok(s)
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn init_metal(&mut self) -> Result<()> {
        if self.options.damping as f32 == 1.0 || self.options.epsilon as f32 == 0.0 {
            self.cpu_fallback("epsilon or damping cannot be represented safely in f32");
            return Ok(());
        }
        self.round_messages_to_f32();
        let f = self.formula.data();
        let profile = self.execution.options.profile.is_some();
        let gpu = if let Some(topology) = &f.feistel {
            topology.metal_solver(&self.eta, profile)
        } else {
            let (offsets, edge_clause) = f.device_clause_indexes();
            bsp_metal::MetalSolver::new(
                bsp_metal::Topology {
                    n: f.n,
                    offsets: &offsets,
                    literals: &f.literals,
                    edge_clause: &edge_clause,
                    var_offsets: &f.var_offsets,
                    var_edges: &f.var_edges,
                },
                &self.eta,
                profile,
            )
        }?;
        self.execution.report.device = Some(gpu.device.clone());
        self.execution.report.allocated_bytes = gpu.allocated_bytes;
        self.execution.gpu = Some(gpu);
        Ok(())
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn init_metal(&mut self) -> Result<()> {
        Backend::Metal.ensure_available()
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    fn init_cuda(&mut self) -> Result<()> {
        let fp32 = self.execution.options.backend == Backend::CudaF32;
        if fp32 && (self.options.damping as f32 == 1.0 || self.options.epsilon as f32 == 0.0) {
            self.cpu_fallback("epsilon or damping cannot be represented safely in f32");
            return Ok(());
        }
        if fp32 {
            self.round_messages_to_f32();
        }
        let f = self.formula.data();
        let (offsets, edge_clause) = f.device_clause_indexes();
        let cuda = bsp_cuda::CudaSolver::new(
            bsp_cuda::Topology {
                n: f.n,
                offsets: &offsets,
                literals: &f.literals,
                edge_clause: &edge_clause,
                var_offsets: &f.var_offsets,
                var_edges: &f.var_edges,
            },
            &self.eta,
            fp32,
            self.execution.options.profile.is_some(),
        )?;
        self.execution.report.device = Some(cuda.device.clone());
        self.execution.report.allocated_bytes = cuda.allocated_bytes;
        self.execution.cuda = Some(cuda);
        Ok(())
    }

    #[cfg(not(all(feature = "cuda", target_os = "linux")))]
    fn init_cuda(&mut self) -> Result<()> {
        Backend::Cuda.ensure_available()
    }

    /// Random initialization has no externally supplied precision to preserve.
    /// Keep its f32 distribution inside [0,1).
    #[cfg(any(
        all(feature = "metal", target_os = "macos"),
        all(feature = "cuda", target_os = "linux")
    ))]
    fn round_messages_to_f32(&mut self) {
        for eta in &mut self.eta {
            *eta = (*eta as f32).min(f32::from_bits(0x3f7fffff)) as f64;
        }
        self.saved_eta.clone_from(&self.eta);
    }

    pub fn execution_report(&self) -> &ExecutionReport {
        &self.execution.report
    }
    pub fn write_profile(&self) -> Result<()> {
        if let Some(path) = &self.execution.options.profile {
            let report = serde_json::json!({"execution":self.execution.report,"seed":self.options.seed,"n":self.formula.n(),"m":self.formula.m(),
                "edges":self.formula.edges(),"sp_sweeps":self.sp_sweeps,"flips":self.flips,"arch":std::env::consts::ARCH,"os":std::env::consts::OS,
                "package_version":env!("CARGO_PKG_VERSION"),"compiler":env!("BSP_RUSTC"),"build_profile":env!("BSP_BUILD_PROFILE"),
                "rustflags":env!("BSP_RUSTFLAGS"),"wall_seconds":self.started.elapsed().as_secs_f64()});
            std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        }
        Ok(())
    }
    pub(crate) fn timed<T>(&mut self, name: &str, action: impl FnOnce(&mut Self) -> T) -> T {
        let started = self
            .execution
            .options
            .profile
            .as_ref()
            .map(|_| Instant::now());
        let result = action(self);
        if let Some(t) = started {
            self.execution.record(name, t.elapsed().as_secs_f64());
        }
        result
    }
}
