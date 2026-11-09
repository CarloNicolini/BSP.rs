mod cuda;
mod device;
mod explicit;
mod implicit;
mod many;
mod metal;
mod parallel;
mod setup;

pub use many::{solve_many, SolveJob};

use super::{Options, SolveResult, Solver, SpReport, Status};
use crate::{Error, Formula, Result, StatusResult};
use explicit::Parallel;
use implicit::ImplicitParallel;
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
impl Backend {
    pub(crate) fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda | Self::CudaF32)
    }

    /// Reject a device backend that this build or host OS cannot run.
    pub(crate) fn ensure_available(self) -> Result<()> {
        let message = match self {
            Self::Metal if cfg!(not(all(feature = "metal", target_os = "macos"))) => {
                "Metal requires macOS and a build with --features metal"
            }
            Self::Cuda | Self::CudaF32 if cfg!(not(all(feature = "cuda", target_os = "linux"))) => {
                "CUDA requires Linux and a build with --features cuda"
            }
            _ => return Ok(()),
        };
        Err(Error(message.into()))
    }
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    F64,
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
    pub precision: Precision,
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
    parallel: Option<Parallel>,
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
                    Precision::F32
                } else {
                    Precision::F64
                },
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

    /// Mirror graph mutations on the device. Without a device these compile to nothing.
    #[inline]
    pub fn freeze_edge(&mut self, edge: usize) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &mut self.gpu {
            gpu.freeze(edge);
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(cuda) = &mut self.cuda {
            cuda.freeze(edge);
        }
        let _ = edge;
    }

    #[inline]
    pub fn restore_edge(&mut self, edge: usize) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &mut self.gpu {
            gpu.restore(edge);
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(cuda) = &mut self.cuda {
            cuda.restore(edge);
        }
        let _ = edge;
    }

    #[inline]
    pub fn set_clause_active(&mut self, clause: usize, active: bool) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &mut self.gpu {
            gpu.set_active(clause, active);
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(cuda) = &mut self.cuda {
            cuda.set_active(clause, active);
        }
        let _ = (clause, active);
    }

    #[inline]
    pub fn set_assignment(&mut self, variable: usize, value: i8) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(gpu) = &mut self.gpu {
            gpu.set_assignment(variable, value);
        }
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(cuda) = &mut self.cuda {
            cuda.set_assignment(variable, value);
        }
        let _ = (variable, value);
    }
}
fn pool(threads: usize) -> Result<Arc<ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Weak<ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS.get_or_init(Mutex::default).lock()?;
    if let Some(p) = pools.get(&threads).and_then(Weak::upgrade) {
        return Ok(p);
    }
    let p = Arc::new(ThreadPoolBuilder::new().num_threads(threads).build()?);
    pools.insert(threads, Arc::downgrade(&p));
    Ok(p)
}
