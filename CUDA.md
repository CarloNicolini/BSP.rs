# NVIDIA CUDA execution

CUDA accelerates variable cavities, clause updates, convergence reductions, and
observables. Select `cuda` for double precision, or `cuda-f32` for single precision.
The scalar CPU implementation remains the default. Every SAT assignment is
verified against the original formula on the CPU.

## Build and compare

On Linux, install Rust 1.88 or newer and a CUDA toolkit providing `nvcc`, the CUDA
runtime headers, and `libcudart`. The NVIDIA driver must support the toolkit.
On the tested Ubuntu host, `sudo apt install nvidia-cuda-toolkit` supplied CUDA
12.0 and was sufficient. Build for the GPU's compute capability, written as digits:

```sh
# GTX 1060: compute capability 6.1 (the current build default).
BSP_CUDA_ARCH=61 cargo build --release --locked --features cuda

target/release/bsp-rust --backend cpu --generate 3 4.2 100000 \
  --seed 7 --damping 0.2 --sweeps 300
target/release/bsp-rust --backend cuda --generate 3 4.2 100000 \
  --seed 7 --damping 0.2 --sweeps 300
target/release/bsp-rust --backend cuda-f32 --generate 3 4.2 100000 \
  --seed 7 --damping 0.2 --sweeps 300
```

Compare the JSON `seconds` values. These commands measure raw sweeps including
public snapshot refresh, excluding graph construction and CUDA initialization.
Remove `--sweeps 300` and add a timeout to compare complete solves. Different
single-precision trajectories can perform different amounts of search work.

Set `BSP_CUDA_ARCH` for other GPUs, for example `86` for compute capability 8.6.
Set `NVCC` to choose another compiler, and `CUDA_HOME` for a nonstandard toolkit
library directory. Both native cubin and forward-compatible PTX are embedded.
Pascal GPUs such as the GTX 1060 require a toolkit through CUDA 12.9; CUDA 13
removed offline compilation for architectures before compute capability 7.5.
See [NVIDIA's compatibility guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/).
The version displayed by `nvidia-smi` describes driver capability and can differ
from the installed `nvcc` version.

CUDA support is optional and currently Linux-only. Selecting a CUDA backend in an
unsupported build returns an explicit error. Portable CPU builds and Apple Metal
builds do not invoke `nvcc` or link the CUDA runtime.

For Python, build the native extension in an activated virtual environment:

```sh
python -m pip install maturin numpy pytest patchelf
maturin develop --release --features extension-module,cuda
```

```python
from pybsp_rust import ExecutionOptions, Formula, Options, Solver, solve_many

formula = Formula.random(3, 4.2, 100000, seed=7)
solver = Solver(formula, execution=ExecutionOptions(backend="cuda"))
print(solver.sweep_many(300), solver.execution_report)
results = solve_many(
    [formula] * 4,
    [Options(seed=s, timeout_seconds=30) for s in range(4)],
    execution=ExecutionOptions(backend="cuda-f32", threads=4),
)
```

## Precision and scheduling

The isolated `bsp-cuda` companion crate owns CUDA allocations and unsafe FFI.
The solver core retains `forbid(unsafe_code)`. NVCC builds with precise division,
gradual underflow, and fused multiply-add contraction disabled. Fast math is off.

Double-precision cavities preserve the scalar prefix/suffix arithmetic order,
including exclusion products with exact zero factors. Observable reductions use
hierarchical sums, so arbitrary `f64` search trajectories are not guaranteed
identical. Critical complexity values and paramagnetic decisions receive a CPU
`f64` evaluation. The tested complete-solve regression matches the CPU assignment
and sweep count; paired raw benchmarks record every message difference.

Single precision has a separate numerical contract. Unsafe message conversions,
underflow, and non-finite arithmetic trigger a recorded CPU `f64` retry from a
coherent pre-convergence checkpoint. Results report the actual backend, precision,
device, allocated device bytes, completed CUDA sweeps, and fallback reasons.
A CUDA runtime error reports a numerical failure with `device_error`; it does
not silently become a successful CPU computation.

Topology, messages, saved messages, cavities, and scratch storage remain in device
memory. Ordered edit queues transfer only affected assignments, clause masks, and
freeze/restore operations between fixed points. Public methods refresh borrowed
host snapshots before returning. Internal solves download surveys and control
state, with message readback for numerical validation when needed.

Clause kernels specialize widths 3, 4, 6, and 12 and provide persistent scratch
for arbitrary widths. The convergence chain remains synchronous: cavities,
clauses, residual. Submission batches adapt toward 2 ms, capped at 16 sweeps, with
flags stopping updates at the first converged or failed sweep. A single large
sweep may exceed this target. Cancellation and timeouts remain cooperative.

Independent solvers use separate allocations and streams on device 0 (respecting
`CUDA_VISIBLE_DEVICES`). `solve_many` preserves input order and limits workers by
the CPU thread budget and a conservative estimate of available device memory.
A solver rejects an allocation estimate above 75% of currently free VRAM.
Concurrent external GPU use can still cause an explicit allocation failure.
WalkSAT chains and graph control remain on the CPU.

## Validation and measurements

On the tested GTX 1060, 30 sweeps of a one-million-variable 3-SAT formula took
13.147 s on scalar CPU, 2.386 s on CUDA `f64`, and 1.041 s on CUDA `f32` (medians
of five trials). Double-precision messages matched exactly in these workloads;
the maximum single-precision error was 7.178e-7. Small complete solves remained
faster on CPU. See [the full measurements and limits](PERFORMANCE.md#nvidia-cuda-on-ottino-lan-2026-10-26)
and [raw samples](bench/cuda-results.json).

```sh
cargo test --release --locked --features cuda
cargo test --release --locked --features cuda --test cuda \
  -- --include-ignored --test-threads=1
cargo clippy --all-targets --locked --features cuda -- -D warnings

python bench/accelerators.py --backends cpu cpu-parallel cuda cuda-f32 \
  --threads 8 --max-n 1000000 --repeats 5 --full-seeds 0 \
  --output /tmp/cuda-kernels.json
python bench/accelerators.py --backends cpu cpu-parallel cuda cuda-f32 \
  --threads 8 --full-only --full-large --full-seeds 5 \
  --output /tmp/cuda-solves.json
```

The driver alternates backend order, discards warmup, saves raw samples and build
metadata, checks `f64` CUDA messages against the scalar reference, and records
single-precision errors. Hardware tests cover zero/unit factors, subnormal values,
wide clauses, high-degree variables, fix/release restoration, full solves,
independent concurrent jobs, numerical recovery, cancellation, and timeouts.
Single-precision kernels also compare against an independent CPU `f32` definition.

Profiling runs must be separate from timings:

```sh
target/release/bsp-rust --backend cuda-f32 --generate 3 4.2 100000 \
  --sweeps 30 --profile /tmp/cuda-profile.json
nvprof --log-file /tmp/cuda-nvprof.txt target/release/bsp-rust \
  --backend cuda-f32 --generate 3 4.2 100000 --sweeps 30
compute-sanitizer --tool memcheck --error-exitcode 99 \
  target/release/deps/cuda-TEST_BINARY_HASH --include-ignored --test-threads=1
```

Use the test executable printed by Cargo for the sanitizer command. Profiling
splits kernel phases into synchronized submissions and changes timing behavior.
NVIDIA profiler support depends on the GPU and toolkit; `nvprof` was usable on
the tested Pascal device. The Ubuntu-packaged CUDA 12.0 sanitizer failed before
instrumentation on this host. The NVIDIA CUDA 12.9 sanitizer, extracted into
`/tmp` without replacing system packages, completed memory checking successfully.
Saved evidence includes [memory checking](bench/cuda-memcheck.txt),
[race checking](bench/cuda-racecheck.txt), [synchronization checking](bench/cuda-synccheck.txt),
the [NVIDIA activity profile](bench/cuda-nvprof.txt), and
[native phase timings](bench/cuda-phase-profile.json).
