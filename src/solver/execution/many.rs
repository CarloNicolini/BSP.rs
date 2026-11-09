use super::*;

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
    execution.backend.ensure_available()?;
    let device_budget: Option<u64> = match execution.backend {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Backend::Metal => Some(bsp_metal::device_budget()?),
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        Backend::Cuda | Backend::CudaF32 => Some(bsp_cuda::device_budget()?),
        _ => None,
    };
    let workers = threads.min(jobs.len().max(1));
    let workers = match device_budget {
        Some(budget) => {
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
            workers.min((budget as usize / 2 / largest).max(1))
        }
        None => workers,
    };
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
                    let next = jobs.lock()?.next();
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
