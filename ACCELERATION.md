# CPU and GPU execution

The default solver remains single-threaded `f64`. Select `cpu-parallel` for large
graphs while retaining the scalar arithmetic and decision order. Select `metal`
explicitly to use `f32` messages on an Apple GPU. GPU trajectories can differ.
Every SAT assignment is checked on the CPU against the original formula.
On Linux, the optional `cuda` feature provides NVIDIA `cuda` (`f64`) and
`cuda-f32` backends. See [CUDA execution](CUDA.md) for installation, device
architecture selection, validation, and measured results.

## Build and run

CPU support is portable. Metal is an optional macOS feature:

```sh
cargo build --release --locked --features metal
target/release/bsp-rust --backend metal --generate 3 4.2 100000 --timeout 30
target/release/bsp-rust --backend cpu-parallel --threads 6 instance.cnf
target/release/bsp-rust --backend cpu --profile /tmp/bsp-profile.json instance.cnf
```

Metal source compiles at runtime and pipelines are cached per process. Building
the bridge requires Xcode Command Line Tools. Full Xcode is needed for Instruments
and Metal System Trace, but is unnecessary for running the backend. An explicitly
requested unavailable Metal device or build produces an error.

Build the Python package with both features:

```sh
python -m venv .venv
.venv/bin/pip install maturin numpy pytest
.venv/bin/maturin develop --release --features extension-module,metal
```

```python
from pybsp_rust import Formula, Options, ExecutionOptions, Solver, solve_many

formula = Formula.random(3, 4.2, 100000, seed=7)
solver = Solver(formula, execution=ExecutionOptions(backend="metal"))
result = solver.solve()
print(result.execution)

# Independent seeds share immutable topology. Results preserve input order.
results = solve_many(
    [formula] * 8,
    [Options(seed=seed, timeout_seconds=30) for seed in range(8)],
    execution=ExecutionOptions(backend="cpu-parallel", threads=8),
)
```

Rust exposes `Backend`, `ExecutionOptions`, `Solver::with_execution`, `SolveJob`,
and `solve_many`. Existing constructors and option JSON continue to work.
`Solver::sweep_many(count)` performs raw synchronous sweeps and refreshes snapshots
once at return. Repeated `sweep()` calls also refresh snapshots after each call;
use `sweep_many` to benchmark GPU throughput without repeated snapshot costs.

## Scheduling and numerical behavior

The multicore backend uses shared Rayon pools with a user-specified worker limit.
Small residual workloads use the scalar implementation when the estimated number
of free incidences falls below 32,768. This avoids thread scheduling overhead.
Per-variable product order and final complexity accumulation retain scalar order.
Batch jobs also use scalar sweeps while at least half the worker budget is already
occupied by independent formulas, then enable inner parallelism as jobs finish.

Metal uses SIMD-group scans for degrees up to 32, individual threads for medium
degrees, and cooperative groups for degrees at least 128. Short-clause kernels
specialize widths 3, 4, 6, and 12; general clauses use persistent scratch buffers.
Prefix/suffix exclusion handles zero factors without dividing by a factor.

Messages, saved messages, cavities, and topology stay in shared GPU allocations.
Messages retain normalized clause order, which performed better across the
measured workloads than the tested variable-order layouts. Degree buckets avoid
dispatching unused cavity variants. CPU and GPU sweeps omit normalization storage;
observables recompute it when needed. GPU sweeps also omit active-degree gathering.
CPU graph edits update only affected entries after the previous submission has
completed. Internal solves download surveys and control data at fixed points;
message snapshots refresh at public API boundaries. Independent batch jobs share
one device runtime and command queue. They submit their own graph dispatches;
graphs are not padded or fused into one kernel grid.

Convergence submissions adapt to approximately 2 ms of GPU work, with at most 16
sweeps per submission. Device flags stop at the first converged sweep. A single
large sweep can exceed this target; cancellation and timeout remain cooperative.
Batch workers limit simultaneous solver allocations using the device working-set
recommendation. The budget does not include formulas already owned by the caller.

Fast math is disabled. External message conversions that would round an interior
probability to zero or one trigger a recorded CPU `f64` fallback. Cavity underflow
or non-finite arithmetic retries from the pre-convergence message checkpoint.
Paramagnetic or decision-sensitive complexity values receive an additional CPU
`f64` evaluation of the current rounded messages. This does not reproduce the
original `f64` search trajectory. Failed Metal command buffers report a numerical
failure; hardware errors are not treated as successful fallback computations.
Apple's [Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
excludes `double` and permits flushing subnormal values to zero.

WalkSAT flips, unit propagation, and BSP decisions remain on the CPU. Independent
searches run concurrently. GPU WalkSAT chains remain future work.

## Reproduce measurements

```sh
.venv/bin/python bench/accelerators.py --max-n 1000000 --repeats 5 \
  --full-seeds 0 --output /tmp/accelerator-kernels.json
.venv/bin/python bench/accelerators.py --full-only --full-large --full-seeds 5 \
  --output /tmp/accelerator-solves.json
cargo test --release --locked
cargo test --release --features metal --test acceleration -- --ignored --test-threads=1
```

The sweep driver alternates backend order, discards warmup, records raw samples,
checks CPU-parallel messages exactly, and records Metal errors relative to `f64`.
It includes mixed-width and planted, partially assigned residual formulas. Full
campaigns record every status, seed, work counter, fallback, and checked solution.
Solve probabilities require a substantially larger campaign than these examples.

Profiling is separate from timing. `--profile` records phase times, GPU timestamps,
submission/wait duration, construction, snapshots, compiler, build flags, and
backend allocation sizes. CPU phases can be nested. Metal profiling intentionally
splits phases into synchronized submissions; those wall times are unsuitable for
speedup claims. Python users can inspect `solver.execution_report`; profile files
are written by `solve()`. A library profile-write failure appears in
`execution.profile_error`; the CLI also returns an error.

For GPU memory and API validation, run the ignored Metal tests with
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`. CI compiles Metal on macOS and keeps
GPU tests explicitly ignored unless they are selected on a host with an Apple GPU.
