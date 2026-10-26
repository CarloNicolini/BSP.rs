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
    pub batch_active: Option<Arc<AtomicUsize>>,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub gpu: Option<bsp_metal::MetalSolver>,
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
                precision: if options.backend == Backend::Metal {
                    "f32"
                } else {
                    "f64"
                }
                .into(),
                threads: if options.backend == Backend::Metal {
                    1
                } else {
                    threads
                },
                device: None,
                allocated_bytes: 0,
                parallel_sweeps: 0,
                metal_sweeps: 0,
                observables_validations: 0,
                fallbacks: Vec::new(),
                phases: BTreeMap::new(),
                profile_error: None,
            },
            options,
            pool,
            parallel: None,
            batch_active: None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu: None,
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
        let setup = execution.profile.as_ref().map(|_| Instant::now());
        let mut s = Self::new(formula, options)?;
        s.execution = Execution::new(execution)?;
        match s.execution.options.backend {
            Backend::Cpu => (),
            Backend::CpuParallel => {
                if s.formula.edges() >= 32768 && s.execution.report.threads > 1 {
                    let scratch = Parallel::new(&s);
                    s.execution.report.allocated_bytes = scratch.allocated_bytes();
                    s.execution.parallel = Some(scratch);
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
                        let gpu = bsp_metal::MetalSolver::new(
                            bsp_metal::Topology {
                                n: f.n,
                                offsets: &f.offsets,
                                literals: &f.literals,
                                edge_clause: &f.edge_clause,
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
        }
        s.sync_messages();
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
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn cpu_fallback(&mut self, reason: &str) {
        self.execution.report.fallbacks.push(reason.into());
        self.execution.report.backend = Backend::Cpu;
        self.execution.report.precision = "f64".into();
        self.execution.report.threads = 1;
        self.execution.gpu = None;
    }
    pub(super) fn upload_messages(&mut self) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &mut self.execution.gpu {
            if let Err(reason) = gpu.upload(&self.eta, &self.saved_eta) {
                self.cpu_fallback(&reason);
            }
        }
    }
    pub(super) fn gpu_assignment(&mut self, v: usize, value: i8) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &mut self.execution.gpu {
            gpu.set_assignment(v, value);
        }
        let _ = (v, value);
    }
    pub(super) fn parallel_sweep(&mut self) -> Option<std::result::Result<f64, Status>> {
        if self
            .execution
            .batch_active
            .as_ref()
            .is_some_and(|n| n.load(Ordering::Relaxed) * 2 >= self.execution.report.threads)
        {
            return None;
        }
        if self.free.len().saturating_mul(self.formula.edges()) / self.formula.n().max(1) < 32768 {
            return None;
        }
        let mut scratch = self.execution.parallel.take()?;
        let p = self.execution.pool.as_ref().unwrap().clone();
        let result = p.install(|| {
            self.timed("cavities", |s| scratch.cavities(s, false))
                .and_then(|()| self.timed("clauses", |s| scratch.clauses(s)))
        });
        if result.is_ok() {
            self.sp_sweeps += 1;
            self.execution.report.parallel_sweeps += 1;
        }
        self.execution.parallel = Some(scratch);
        Some(result)
    }
    pub(super) fn parallel_observables(
        &mut self,
    ) -> Option<std::result::Result<(f64, bool), Status>> {
        if self
            .execution
            .batch_active
            .as_ref()
            .is_some_and(|n| n.load(Ordering::Relaxed) * 2 >= self.execution.report.threads)
        {
            return None;
        }
        if self.free.len().saturating_mul(self.formula.edges()) / self.formula.n().max(1) < 32768 {
            return None;
        }
        let mut scratch = self.execution.parallel.take()?;
        let p = self.execution.pool.as_ref().unwrap().clone();
        let result = p.install(|| {
            self.timed("cavities", |s| scratch.cavities(s, true))
                .and_then(|()| scratch.observables(self))
        });
        self.execution.parallel = Some(scratch);
        Some(result)
    }
    pub(super) fn metal_converge(
        &mut self,
        single: bool,
    ) -> Option<std::result::Result<SpReport, Status>> {
        self.metal_compute(single, if single { 1 } else { self.options.max_sweeps })
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
}

pub(crate) struct Parallel {
    q: Vec<f64>,
    norm: Vec<f64>,
    products: Vec<[f64; 2]>,
    slot: Vec<usize>,
    logs: Vec<f64>,
    clause_logs: Vec<f64>,
    variable_terms: Vec<f64>,
    ranges: Vec<(usize, usize)>,
}
impl Parallel {
    fn allocated_bytes(&self) -> usize {
        (self.q.len()
            + self.norm.len()
            + self.logs.len()
            + self.clause_logs.len()
            + self.variable_terms.len())
            * 8
            + self.slot.len() * std::mem::size_of::<usize>()
            + self.products.len() * 16
            + self.ranges.len() * 16
    }
    fn new(s: &Solver) -> Self {
        let f = &s.formula.0;
        let mut slot = vec![0; f.literals.len()];
        for (p, &e) in f.var_edges.iter().enumerate() {
            slot[e as usize] = p;
        }
        let chunk = f.n.div_ceil(s.execution.report.threads * 4).max(1);
        Self {
            q: vec![0.0; slot.len()],
            norm: vec![1.0; slot.len()],
            products: vec![[1.0; 2]; f.n],
            logs: vec![0.0; slot.len()],
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
            s.q.par_iter_mut()
                .zip(&mut s.cavity_norm)
                .enumerate()
                .for_each(|(e, (q, norm))| {
                    *q = self.q[self.slot[e]];
                    *norm = self.norm[self.slot[e]];
                });
        } else {
            s.q.par_iter_mut()
                .enumerate()
                .for_each(|(e, q)| *q = self.q[self.slot[e]]);
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
    fn clauses(&mut self, s: &mut Solver) -> std::result::Result<f64, Status> {
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
            let first = f.offsets[start] as usize;
            let len = f.offsets[end] as usize - first;
            let (a, b) = eta.split_at_mut(len);
            eta = b;
            chunks.push((start, end, first, a));
        }
        let active = &s.active;
        let assignment = &s.assignment;
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
                        for e in f.offsets[c] as usize..f.offsets[c + 1] as usize {
                            if assignment[(f.literals[e] >> 1) as usize] < 0 {
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
    fn observables(&mut self, s: &mut Solver) -> std::result::Result<(f64, bool), Status> {
        let f = &s.formula.0;
        self.logs
            .par_iter_mut()
            .zip(&s.cavity_norm)
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
                for e in f.offsets[c] as usize..f.offsets[c + 1] as usize {
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
            for e in f.offsets[c] as usize..f.offsets[c + 1] as usize {
                if s.assignment[(f.literals[e] >> 1) as usize] < 0 {
                    clause_term += self.logs[e];
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
