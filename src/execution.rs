use super::{Options, SolveResult, Solver, SpReport, Status};
use crate::{Error, Formula, Result};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    sync::{Arc, Mutex, OnceLock, Weak},
    time::Instant,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    #[default]
    Cpu,
    CpuParallel,
    Metal,
    Cuda,
    CudaF32,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExecutionOptions {
    pub backend: Backend,
    /// Zero uses the available logical CPUs. Shared pools prevent nested oversubscription.
    pub threads: usize,
    /// Profiling intentionally adds synchronization to resolve GPU phase durations.
    pub profile: Option<PathBuf>,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct Phase {
    pub calls: u64,
    pub seconds: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct ExecutionReport {
    pub requested_backend: Backend,
    pub backend: Backend,
    pub precision: String,
    pub threads: usize,
    pub device: Option<String>,
    pub allocated_bytes: usize,
    pub parallel_sweeps: u64,
    pub metal_sweeps: u64,
    pub cuda_sweeps: u64,
    pub device_error: Option<String>,
    pub observables_validations: u64,
    pub fallbacks: Vec<String>,
    pub phases: BTreeMap<String, Phase>,
    pub profile_error: Option<String>,
}
pub(crate) struct Execution {
    pub options: ExecutionOptions,
    pub report: ExecutionReport,
    pub pool: Option<Arc<ThreadPool>>,
    pub parallel: Option<Parallel>,
    implicit_parallel: Option<ImplicitParallel>,
    pub batch_active: Option<Arc<AtomicUsize>>,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub gpu: Option<bsp_metal::MetalSolver>,
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub cuda: Option<bsp_cuda::CudaSolver>,
}
impl Execution {
    pub fn new(options: ExecutionOptions) -> Result<Self> {
        if options.threads > 1024 {
            return Err(Error("threads must be at most 1024".into()));
        }
        let threads = if options.backend == Backend::Cpu {
            1
        } else if options.threads == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            options.threads
        };
        if threads > 1024 {
            return Err(Error("threads must be at most 1024".into()));
        }
        let pool = if options.backend == Backend::CpuParallel {
            Some(pool(threads)?)
        } else {
            None
        };
        Ok(Self {
            report: ExecutionReport {
                requested_backend: options.backend,
                backend: options.backend,
                precision: if matches!(options.backend, Backend::Metal | Backend::CudaF32) {
                    "f32"
                } else {
                    "f64"
                }
                .into(),
                threads: if matches!(
                    options.backend,
                    Backend::Metal | Backend::Cuda | Backend::CudaF32
                ) {
                    1
                } else {
                    threads
                },
                device: None,
                allocated_bytes: 0,
                parallel_sweeps: 0,
                metal_sweeps: 0,
                cuda_sweeps: 0,
                device_error: None,
                observables_validations: 0,
                fallbacks: Vec::new(),
                phases: BTreeMap::new(),
                profile_error: None,
            },
            options,
            pool,
            parallel: None,
            implicit_parallel: None,
            batch_active: None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu: None,
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            cuda: None,
        })
    }
    pub fn record(&mut self, name: &str, seconds: f64) {
        if self.options.profile.is_some() {
            let p = self.report.phases.entry(name.into()).or_default();
            p.calls += 1;
            p.seconds += seconds;
        }
    }
}
fn pool(threads: usize) -> Result<Arc<ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Weak<ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|e| Error(e.to_string()))?;
    if let Some(p) = pools.get(&threads).and_then(Weak::upgrade) {
        return Ok(p);
    }
    let p = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| Error(e.to_string()))?,
    );
    pools.insert(threads, Arc::downgrade(&p));
    Ok(p)
}
pub struct SolveJob {
    pub formula: Formula,
    pub options: Options,
}
/// Independent jobs, ordered results. Queued jobs allocate state only when a worker is free.
pub fn solve_many(jobs: Vec<SolveJob>, execution: ExecutionOptions) -> Result<Vec<SolveResult>> {
    let threads = if execution.threads == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        execution.threads
    };
    if threads == 0 || threads > 1024 {
        return Err(Error(
            "threads must be in 1..=1024, or zero for automatic".into(),
        ));
    }
    if execution.profile.is_some() {
        return Err(Error("solve_many profiling requires per-solver profile paths; profile individual jobs instead".into()));
    }
    for job in &jobs {
        job.options.validate()?;
    }
    #[allow(unused_mut)]
    let mut workers = threads.min(jobs.len().max(1));
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if execution.backend == Backend::Metal {
        let budget = bsp_metal::device_budget().map_err(Error)? as usize / 2;
        let largest = jobs
            .iter()
            .map(|j| {
                j.formula.edges().saturating_mul(112)
                    + j.formula.n().saturating_mul(128)
                    + j.formula.m().saturating_mul(48)
            })
            .max()
            .unwrap_or(1)
            .max(1);
        workers = workers.min((budget / largest).max(1));
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    if matches!(execution.backend, Backend::Cuda | Backend::CudaF32) {
        let budget = bsp_cuda::device_budget().map_err(Error)? as usize / 2;
        let largest = jobs
            .iter()
            .map(|j| {
                j.formula.edges().saturating_mul(112)
                    + j.formula.n().saturating_mul(128)
                    + j.formula.m().saturating_mul(48)
            })
            .max()
            .unwrap_or(1)
            .max(1);
        workers = workers.min((budget / largest).max(1));
    }
    #[cfg(not(all(feature = "cuda", target_os = "linux")))]
    if matches!(execution.backend, Backend::Cuda | Backend::CudaF32) {
        return Err(Error(
            "CUDA requires Linux and a build with --features cuda".into(),
        ));
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    if execution.backend == Backend::Metal {
        return Err(Error(
            "Metal requires macOS and a build with --features metal".into(),
        ));
    }
    let p = pool(threads)?;
    // Fixed worker lanes cap GPU/host memory while sharing the CPU parallel pool.
    let jobs: Mutex<_> = Mutex::new(jobs.into_iter().enumerate());
    let active = Arc::new(AtomicUsize::new(0));
    let lanes: Vec<Result<Vec<(usize, SolveResult)>>> = p.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|_| {
                let mut results = Vec::new();
                loop {
                    let next = jobs.lock().map_err(|e| Error(e.to_string()))?.next();
                    let Some((i, job)) = next else { break };
                    let mut s =
                        Solver::with_execution(job.formula, job.options, execution.clone())?;
                    if execution.backend == Backend::CpuParallel {
                        s.execution.batch_active = Some(active.clone());
                    }
                    active.fetch_add(1, Ordering::Relaxed);
                    results.push((i, s.solve()));
                    active.fetch_sub(1, Ordering::Relaxed);
                }
                Ok(results)
            })
            .collect()
    });
    let mut ordered = Vec::new();
    for lane in lanes {
        ordered.extend(lane?);
    }
    ordered.sort_unstable_by_key(|(i, _)| *i);
    Ok(ordered.into_iter().map(|(_, r)| r).collect())
}

impl Solver {
    pub fn with_execution(
        formula: Formula,
        options: Options,
        execution: ExecutionOptions,
    ) -> Result<Self> {
        if formula.feistel_topology().is_some()
            && !matches!(execution.backend, Backend::Cpu | Backend::CpuParallel)
        {
            return Err(Error(
                "Feistel formulas require --backend cpu or cpu-parallel".into(),
            ));
        }
        let construction_started = Instant::now();
        options.validate()?;
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        if execution.backend == Backend::Metal {
            return Err(Error(
                "Metal requires macOS and a build with --features metal".into(),
            ));
        }
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        if matches!(execution.backend, Backend::Cuda | Backend::CudaF32) {
            return Err(Error(
                "CUDA requires Linux and a build with --features cuda".into(),
            ));
        }
        let setup = execution.profile.as_ref().map(|_| Instant::now());
        let configured = Execution::new(execution)?;
        // Check device availability before allocating the host solver arrays.
        // Unsupported f32 settings intentionally fall back to CPU without a device.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if configured.options.backend == Backend::Metal
            && options.damping as f32 != 1.0
            && options.epsilon as f32 != 0.0
        {
            bsp_metal::device_budget().map_err(Error)?;
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if matches!(configured.options.backend, Backend::Cuda | Backend::CudaF32)
            && (configured.options.backend == Backend::Cuda
                || options.damping as f32 != 1.0 && options.epsilon as f32 != 0.0)
        {
            bsp_cuda::device_budget().map_err(Error)?;
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
            Backend::Metal => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    if s.options.damping as f32 == 1.0 || s.options.epsilon as f32 == 0.0 {
                        s.cpu_fallback("epsilon or damping cannot be represented safely in f32");
                    } else {
                        // Random initialization has no externally supplied precision
                        // to preserve. Keep its f32 distribution inside [0,1).
                        for eta in &mut s.eta {
                            *eta = (*eta as f32).min(f32::from_bits(0x3f7fffff)) as f64;
                        }
                        s.saved_eta.clone_from(&s.eta);
                        let f = &s.formula.0;
                        let (offsets, edge_clause) = f.device_clause_indexes();
                        let gpu = bsp_metal::MetalSolver::new(
                            bsp_metal::Topology {
                                n: f.n,
                                offsets: &offsets,
                                literals: &f.literals,
                                edge_clause: &edge_clause,
                                var_offsets: &f.var_offsets,
                                var_edges: &f.var_edges,
                            },
                            &s.eta,
                            s.execution.options.profile.is_some(),
                        )
                        .map_err(Error)?;
                        s.execution.report.device = Some(gpu.device.clone());
                        s.execution.report.allocated_bytes = gpu.allocated_bytes;
                        s.execution.gpu = Some(gpu);
                    }
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                return Err(Error(
                    "Metal requires macOS and a build with --features metal".into(),
                ));
            }
            Backend::Cuda | Backend::CudaF32 => {
                #[cfg(all(feature = "cuda", target_os = "linux"))]
                {
                    let fp32 = s.execution.options.backend == Backend::CudaF32;
                    if fp32 && (s.options.damping as f32 == 1.0 || s.options.epsilon as f32 == 0.0)
                    {
                        s.cpu_fallback("epsilon or damping cannot be represented safely in f32");
                    } else {
                        if fp32 {
                            for eta in &mut s.eta {
                                *eta = (*eta as f32).min(f32::from_bits(0x3f7fffff)) as f64;
                            }
                            s.saved_eta.clone_from(&s.eta);
                        }
                        let f = &s.formula.0;
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
                            &s.eta,
                            fp32,
                            s.execution.options.profile.is_some(),
                        )
                        .map_err(Error)?;
                        s.execution.report.device = Some(cuda.device.clone());
                        s.execution.report.allocated_bytes = cuda.allocated_bytes;
                        s.execution.cuda = Some(cuda);
                    }
                }
                #[cfg(not(all(feature = "cuda", target_os = "linux")))]
                return Err(Error(
                    "CUDA requires Linux and a build with --features cuda".into(),
                ));
            }
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
    pub fn execution_report(&self) -> &ExecutionReport {
        &self.execution.report
    }
    pub fn write_profile(&self) -> Result<()> {
        if let Some(path) = &self.execution.options.profile {
            let report = serde_json::json!({"execution":self.execution.report,"seed":self.options.seed,"n":self.formula.n(),"m":self.formula.m(),
                "edges":self.formula.edges(),"sp_sweeps":self.sp_sweeps,"flips":self.flips,"arch":std::env::consts::ARCH,"os":std::env::consts::OS,
                "package_version":env!("CARGO_PKG_VERSION"),"compiler":env!("BSP_RUSTC"),"build_profile":env!("BSP_BUILD_PROFILE"),
                "rustflags":env!("BSP_RUSTFLAGS"),"wall_seconds":self.started.elapsed().as_secs_f64()});
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&report).map_err(|e| Error(e.to_string()))?,
            )
            .map_err(|e| Error(e.to_string()))?;
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
    pub(super) fn sync_messages(&mut self) {
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
    fn cpu_fallback(&mut self, reason: &str) {
        self.execution.report.fallbacks.push(reason.into());
        self.execution.report.backend = Backend::Cpu;
        self.execution.report.precision = "f64".into();
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
    pub(super) fn upload_messages(&mut self) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &mut self.execution.gpu {
            if let Err(reason) = gpu.upload(&self.eta, &self.saved_eta) {
                self.cpu_fallback(&reason);
            }
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.execution.cuda.is_some() {
            if self.execution.report.precision == "f32"
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
    pub(super) fn gpu_assignment(&mut self, v: usize, value: i8) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &mut self.execution.gpu {
            gpu.set_assignment(v, value);
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(cuda) = &mut self.execution.cuda {
            cuda.set_assignment(v, value);
        }
        let _ = (v, value);
    }
    fn parallel_work_available(&self) -> bool {
        if self
            .execution
            .batch_active
            .as_ref()
            .is_some_and(|n| n.load(Ordering::Relaxed) * 2 >= self.execution.report.threads)
        {
            return false;
        }
        let min = if self.formula.feistel_topology().is_some() {
            IMPLICIT_MIN
        } else {
            explicit_min(self.execution.report.threads)
        };
        self.free.len().saturating_mul(self.formula.edges()) / self.formula.n().max(1) >= min
    }

    pub(super) fn parallel_sweep(&mut self) -> Option<std::result::Result<f64, Status>> {
        if !self.parallel_work_available() {
            return None;
        }
        let pool = self.execution.pool.as_ref()?.clone();
        let result = if let Some(mut scratch) = self.execution.implicit_parallel.take() {
            let result = pool.install(|| {
                self.timed("cavities", |s| scratch.cavities(s, false))
                    .and_then(|()| self.timed("clauses", parallel_clauses))
            });
            self.execution.implicit_parallel = Some(scratch);
            result
        } else {
            let mut scratch = self.execution.parallel.take()?;
            let result = pool.install(|| {
                self.timed("cavities", |s| scratch.cavities(s, false))
                    .and_then(|()| self.timed("clauses", parallel_clauses))
            });
            self.execution.parallel = Some(scratch);
            result
        };
        if result.is_ok() {
            self.sp_sweeps += 1;
            self.execution.report.parallel_sweeps += 1;
        }
        Some(result)
    }

    pub(super) fn parallel_observables(
        &mut self,
    ) -> Option<std::result::Result<(f64, bool), Status>> {
        if !self.parallel_work_available() {
            return None;
        }
        let pool = self.execution.pool.as_ref()?.clone();
        if let Some(mut scratch) = self.execution.implicit_parallel.take() {
            let result = pool.install(|| self.timed("cavities", |s| scratch.cavities(s, true)));
            self.execution.implicit_parallel = Some(scratch);
            return Some(result.and_then(|()| self.scalar_observables_from_cavities()));
        }
        let mut scratch = self.execution.parallel.take()?;
        let result = pool.install(|| {
            self.timed("cavities", |s| scratch.cavities(s, true))
                .and_then(|()| scratch.observables(self))
        });
        self.execution.parallel = Some(scratch);
        Some(result)
    }
    pub(super) fn gpu_converge(
        &mut self,
        single: bool,
    ) -> Option<std::result::Result<SpReport, Status>> {
        self.gpu_compute(single, if single { 1 } else { self.options.max_sweeps })
    }
    pub(super) fn gpu_compute(
        &mut self,
        single: bool,
        limit: usize,
    ) -> Option<std::result::Result<SpReport, Status>> {
        if let Some(result) = self.metal_compute(single, limit) {
            Some(result)
        } else {
            self.cuda_compute(single, limit)
        }
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    fn device_failure(&mut self, error: String) -> Status {
        self.execution.report.device_error = Some(error);
        self.terminal = Some(Status::NumericalFailure);
        self.solution = None;
        Status::NumericalFailure
    }
    pub(super) fn metal_compute(
        &mut self,
        single: bool,
        limit: usize,
    ) -> Option<std::result::Result<SpReport, Status>> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let mut gpu = self.execution.gpu.take()?;
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
                        .map_err(|e| {
                            self.execution.report.fallbacks.push(e);
                            Status::NumericalFailure
                        })?;
                    self.execution.record("gpu", report.gpu_seconds);
                    self.execution
                        .record("gpu_submission_and_wait", report.wall_seconds);
                    for (name, seconds) in &report.phases {
                        self.execution.record(name, *seconds);
                    }
                    count += report.sweeps;
                    self.sp_sweeps += report.sweeps as u64;
                    self.execution.report.metal_sweeps += report.sweeps as u64;
                    if report.numerical_error {
                        fallback = Some("Metal cavity arithmetic underflowed or became non-finite");
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
                        let obs = gpu.observables().map_err(|e| {
                            self.execution.report.fallbacks.push(e);
                            Status::NumericalFailure
                        })?;
                        self.execution.record("gpu", obs.gpu_seconds);
                        for (name, seconds) in &obs.phases {
                            self.execution.record(name, *seconds);
                        }
                        if obs.numerical_error {
                            fallback = Some("Metal observables became numerically unsafe");
                            return Err(Status::NumericalFailure);
                        }
                        gpu.surveys(&mut self.surveys, &mut self.cert);
                        // Re-evaluate decision-sensitive observables in f64 on
                        // the current rounded messages, without changing them.
                        let guard = 32.0 * f32::EPSILON as f64 * self.formula.edges() as f64;
                        let (complexity, paramagnetic) =
                            if obs.paramagnetic || obs.complexity <= guard {
                                gpu.snapshot(&mut self.eta, &mut self.saved_eta);
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
                gpu.restore_checkpoint();
                gpu.snapshot(&mut self.eta, &mut self.saved_eta);
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
            self.execution.gpu = Some(gpu);
            Some(result)
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            let _ = (single, limit);
            None
        }
    }
    pub(super) fn cuda_compute(
        &mut self,
        single: bool,
        limit: usize,
    ) -> Option<std::result::Result<SpReport, Status>> {
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
                            32.0 * if self.execution.report.precision == "f32" {
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

pub(crate) struct Parallel {
    q: Vec<f64>,
    norm: Vec<f64>,
    products: Vec<[f64; 2]>,
    slot: Vec<u32>,
    clause_logs: Vec<f64>,
    variable_terms: Vec<f64>,
    ranges: Vec<(usize, usize)>,
}
impl Parallel {
    fn allocated_bytes(&self) -> usize {
        (self.q.len() + self.norm.len() + self.clause_logs.len() + self.variable_terms.len()) * 8
            + self.slot.len() * std::mem::size_of::<u32>()
            + self.products.len() * 16
            + self.ranges.len() * 16
    }
    fn new(s: &Solver) -> Self {
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
    fn cavities(&mut self, s: &mut Solver, store_norms: bool) -> std::result::Result<(), Status> {
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
    fn observables(&mut self, s: &mut Solver) -> std::result::Result<(f64, bool), Status> {
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

fn parallel_clauses(s: &mut Solver) -> std::result::Result<f64, Status> {
    let f = &s.formula.0;
    let chunk = s
        .formula
        .m()
        .div_ceil(s.execution.report.threads * 4)
        .max(1);
    let mut eta = s.eta.as_mut_slice();
    let mut chunks = Vec::new();
    for start in (0..s.formula.m()).step_by(chunk) {
        let end = (start + chunk).min(s.formula.m());
        let first = f.offset(start);
        let len = f.offset(end) - first;
        let (a, b) = eta.split_at_mut(len);
        eta = b;
        chunks.push((start, end, first, a));
    }
    let active = &s.active;
    let masks = &s.masks;
    let q = &s.q;
    let damping = s.options.damping;
    chunks
        .into_par_iter()
        .map_init(
            Vec::<(usize, f64)>::new,
            |prefix, (start, end, base, eta)| {
                let mut residual: f64 = 0.0;
                for c in start..end {
                    if !active.contains(c) {
                        continue;
                    }
                    prefix.clear();
                    let mut product = 1.0;
                    let start = f.offset(c);
                    let first_word = f.mask_offset(c);
                    if f.offset(c + 1) - start == 3 && masks.get(first_word) == 7 {
                        residual = residual.max(super::update_full_triple(
                            &q[start..start + 3],
                            &mut eta[start - base..start - base + 3],
                            damping,
                        ));
                        continue;
                    }
                    for w in first_word..f.mask_offset(c + 1) {
                        for e in (crate::bitset::Ones {
                            word: masks.get(w),
                            base: start + 64 * (w - first_word),
                        }) {
                            prefix.push((e, product));
                            product *= q[e];
                        }
                    }
                    if prefix.is_empty() {
                        return Err(Status::Contradiction);
                    }
                    let mut suffix = 1.0;
                    for &(e, p) in prefix.iter().rev() {
                        let raw = p * suffix;
                        suffix *= q[e];
                        let raw = if raw < 1e-16 { 0.0 } else { raw };
                        residual = residual.max((raw - eta[e - base]).abs());
                        eta[e - base] = raw + (eta[e - base] - raw) * damping;
                    }
                }
                Ok(residual)
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok(a.max(b)))
}

// Implicit output is bounded independently of the graph. Temporary incidence
// decoding uses at most one variable's edges per Rayon task.
const IMPLICIT_MIN: usize = 1024;
fn explicit_min(threads: usize) -> usize {
    // Direct CSR traversal is cheap. Leave enough work per worker to
    // amortize the cavity, scatter, and clause dispatches.
    (threads * 4096).max(8192)
}
const OUTPUT_WINDOW: usize = 2048;

#[derive(Clone, Copy, Default)]
struct CavityOutput {
    edge: u32,
    q: f64,
    norm: f64,
}

struct ImplicitParallel {
    outputs: Vec<CavityOutput>,
}

impl ImplicitParallel {
    fn new(edges: usize) -> Self {
        Self {
            outputs: vec![CavityOutput::default(); edges.min(OUTPUT_WINDOW)],
        }
    }

    fn cavities(&mut self, s: &mut Solver, store_norms: bool) -> std::result::Result<(), Status> {
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
