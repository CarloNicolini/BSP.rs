# Measured SP throughput and remaining limits

The native Rust SP kernel is 1.23-2.18 times faster than the extracted C++ kernel
on the four measured inputs below. These measurements cover one ARM64 Mac.
They establish a kernel improvement on these inputs. Complete-solve speed and
success rates require a larger paired campaign.

## Matched kernel measurement

Both programs use the same normalized CNF, identical per-edge starting messages,
damping 0.2, and 30 synchronous sweeps. Each measurement excludes formula parsing,
graph construction, message loading, and result serialization. The driver drops
one warmup round, alternates program order, and records five samples per program.
The table reports their medians. Both kernels run on one thread.

The C++ harness uses the existing `Graph`, `Clause`, and `Vertex` layouts,
`Vertex::make_products`, and the nested zero-temperature message loop extracted
from `Graph::convergence_messages`. It omits convergence exits, diagnostics,
complexity evaluation, and unit propagation. It also omits the original
`_go_forward` checks because every benchmark edge is free. This gives C++ a
conservative baseline for the SP computation. Rust likewise runs raw sweeps.

| k | N | alpha | Edges | C++ milliseconds | Rust milliseconds | C++ / Rust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 10,000 | 4.2 | 126,000 | 69.72 | 45.50 | 1.53 |
| 4 | 10,000 | 9.5 | 380,000 | 204.12 | 129.85 | 1.57 |
| 6 | 2,000 | 40 | 480,000 | 210.91 | 171.60 | 1.23 |
| 12 | 1,000 | 20 | 240,000 | 159.90 | 73.25 | 2.18 |

The maximum difference between corresponding final messages is 2.69e-13.
The driver checks every message for agreement and finiteness. These are
computational workloads; the k=12 case does not represent a measured SAT phase
or solver threshold. The raw samples, compiler versions, build flags, and source
and binary hashes are in [kernel-results.json](bench/kernel-results.json).
The visible variation in the k=4 samples limits precision. No hardware-counter
or statistical confidence-interval claim accompanies these wall times.

Reference host: ARM64, macOS 26.5.2, 8 logical CPUs. Rust is 1.88.0. C++ is Apple
Clang 21.0.0. Rust uses `-C target-cpu=native`, optimization level 3, ThinLTO, and
one code-generation unit. C++ uses the repository's Release/native flags,
fast-math, loop unrolling, LTO, and no OpenMP.

Reproduce from the BSP.rs root after cloning the separate C++ reference:

```sh
git clone https://github.com/CarloNicolini/pybsp.git ../pybsp
export BSP_CPP_SOURCE="$(cd ../pybsp && pwd)"
```

The C++ checkout must contain the baseline interfaces used by the harness. The
saved measurements predate this repository extraction; their source hashes and
paths describe the original workspace and are retained as historical evidence.
No C++ checkout is needed to build or test the Rust/Python solver.

Then run:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --locked

cmake -S "$BSP_CPP_SOURCE" -B /private/tmp/bsp-rust-cpp-release \
  -DCMAKE_BUILD_TYPE=Release -DBSP_BUILD_CLI=OFF -DBSP_BUILD_TESTS=OFF \
  -DBSP_ENABLE_OPENMP=OFF -DBSP_IDE_LINK=OFF -DBSP_NATIVE_ARCH=ON
cmake --build /private/tmp/bsp-rust-cpp-release -j 4

c++ -std=c++14 -O3 -ffast-math -fno-math-errno -funroll-loops \
  -fomit-frame-pointer -march=native -flto -I"$BSP_CPP_SOURCE/include" \
  -c bench/cpp_kernel.cpp -o /private/tmp/bsp-cpp-kernel.o
c++ -O3 -march=native -flto /private/tmp/bsp-cpp-kernel.o \
  /private/tmp/bsp-rust-cpp-release/src/libbsp_core.a -pthread \
  -o /private/tmp/bsp-cpp-kernel

python bench/compare.py --cpp /private/tmp/bsp-cpp-kernel \
  --repeats 5 --sweeps 30 --output /private/tmp/bsp-kernel-results.json
```

The two-stage compile/link commands also avoid an Apple linker failure observed
when the LTO input object was a transient compiler temporary. On other systems,
use a writable temporary directory in place of `/private/tmp`.

## Complete solves

The Rust tests include a successful k=3, alpha=4.15, N=400 run with both
decimation and backtracking. Its final assignment passes clause evaluation.
Other dense inputs reach nonconvergence; the solver reports this outcome.
The implementation does not establish better solve probabilities than C++.

The separate driver records full runs on shared CNFs:

```sh
python bench/solve_compare.py --timeout 5 \
  --output /private/tmp/bsp-solve-results.json
```

This requires both Python packages. C++ collects ten solutions and whitens them;
Rust returns one checked assignment. Their RNGs and trajectories also differ.
The recorded wall times therefore describe different work and must not be used
as a direct speedup estimate. The driver records failures and checks every
returned solution. Installed Python packages use portable builds, separately
from the native-CPU kernel binaries above.

The recorded sample is in [solve-results.json](bench/solve-results.json). Each
row is one run, so these values have no uncertainty estimate:

| k | alpha | N | Seed | C++ seconds | Rust seconds | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 2.5 | 400 | 3 | 0.0248 | 0.000687 | Both return verified solutions |
| 3 | 4.15 | 400 | 3 | 0.4669 | 0.3915 | Both return verified solutions |
| 3 | 4.15 | 400 | 2 | 0.0725 | 0.0509 | Both report SP nonconvergence |
| 4 | 9.5 | 400 | 3 | 2.0810 | 1.8490 | Both return verified solutions |

For the k=3, alpha=4.15 successful run, Rust records 6,247 SP sweeps, 912
decimations, 822 releases, and 1,734 fixed points. The C++ run also records
6,247 SP sweeps and 1,734 fixed points. The k=4 trajectories differ: Rust records
14,593 sweeps and C++ records 15,080 sweeps. C++ returns ten checked solutions in
each successful row; Rust returns one.

## Assembly and locality

Emit native assembly and inspect only the SP functions:

```sh
RUSTFLAGS='-C target-cpu=native' cargo rustc -p bsp-rust \
  --release --locked --lib -- --emit=asm
python bench/inspect_asm.py target/release/deps/bsp_rust.s
```

Cargo can add a hash to the assembly filename. Use the emitted file in
`target/release/deps/` if `bsp_rust.s` does not exist. The inspection script accepts
the legacy symbol mangling emitted by the measured Rust compiler.

Inspection on ARM64 shows `rbit`/`clz` for set-bit iteration and scalar `fdiv` for
cavity normalization. Indirect CSR indexing retains bounds-check branches and
cold panic paths. The dependency on gathers and floating-point reductions limits
automatic vectorization. The source does not assert vectorized probability
updates or hand-written SIMD. Native compilation permits LLVM to select the
instructions available on the build CPU; it does not force every instruction.

The saved [assembly inspection](bench/assembly-results.json) counts eight bounds
panic paths in `sweep` and eleven in `cavities`, including cold error blocks.
It is an instruction inventory, not a count of executed instructions or CPU
cycles. It shows the remaining indirect-index checks explicitly.

Contiguous clause messages and compact topology reduce the pointer traffic of
the C++ object graph. Variable-to-edge access still gathers scattered messages.
These workloads exceed small CPU caches. Zero cache misses cannot be guaranteed.
The current evidence measures time and numerical agreement, not cache misses,
branch misses, or memory bandwidth. Linux `perf stat` or platform hardware
counters are needed for those quantities.

The final kernel uses separate cavity arrays. A paired local comparison measured
about 20% less time than a smaller variable-product layout on k=3 and k=4. Extra
memory is an explicit speed trade. The retained prefix/suffix recurrence also
passes the subnormal-product regression test.

The original measurements above did not cover parallel execution. The optional
CPU and Metal backends now include degree/width specializations; their separate
measurements follow below. x86-64/AVX, other ARM CPUs, hardware-counter profiles,
and profile-guided optimization still require measurements.

## Search-index audit

The [complexity audit](COMPLEXITY.md) identifies repeated scans, their replacements,
and scans constrained by changing inputs or output size. The current changes
retain an indexed heap for frozen backtracking scores, cached active degrees,
and XOR lookup of a unit clause's sole free edge. A hierarchical bitset was
tested and rejected for the production solver after complete-run measurements.

[Paired full-solver samples](bench/indexing-results.json) compare the previous
Rust binary with the indexed version on the same five workloads. Every result
field except elapsed time must agree, including all assignments and step records.
Both binaries use generic Release builds with the same compiler and LTO settings.
These are separate from the older Rust/C++ kernel comparisons above. See
[selection/bitset microbenchmarks](bench/index-primitives.json) and the
[bitset diagnostic](bench/indexing-ablation.json) for component-level evidence.

## Apple silicon acceleration: profiling basis

Initial macOS `sample` profiles on 2026-10-26 identified the following distribution
on an M1 Pro with six performance CPU cores, two efficiency CPU cores, 14 GPU
cores, and 16 GB unified memory. The complete solves were 3-SAT, alpha 4.2,
N=1,000, seed 1, and 4-SAT, alpha 9.5, N=400, seed 3. Their main-thread profiles
contained 3,093 and 1,279 samples respectively, at a requested 1 ms interval.

| Work | Share of sampled complete-solve time | Retained implementation |
| --- | ---: | --- |
| Variable cavities, including the observable refresh | 50–58% | CPU partitions with variable-order scratch; Metal SIMD-group, medium-degree, and cooperative high-degree kernels |
| Clause message updates | 27–28% | Disjoint CPU clause partitions; specialized Metal widths 3, 4, 6, 12 and a general path |
| Remaining observables and logarithms | 13–22% | Parallel element calculations and reductions; ordered CPU accumulation |
| Ranking, graph edits, and other control | Under 1% combined | Keep CPU control and existing indexed mutations |

A separate raw profile with N=100,000 and 1,260,000 edges attributed about 79%
to cavities. A controlled WalkSAT-only diagnostic attributed almost all sampled
execution to its sequential flip chain. Independent searches are the main
parallel opportunity there. Parsing, normalization, initial state construction,
and snapshots are measured separately rather than treated as iterative hot paths.

Full Xcode was unavailable, so Instruments and hardware-counter claims are not
part of this evidence. Runtime Metal compilation and GPU timestamps were tested.
A synthetic dispatch-and-wait probe took about 190 microseconds for negligible
work, establishing the need to amortize submissions. Optional phase profiling
splits Metal work into separate submissions and must not be enabled for timings.

The default remains scalar `f64`; `cpu-parallel` preserves the scalar arithmetic
order and uses scalar work below its measured scheduling threshold. Metal is an
explicit `f32` mode. [Execution documentation](ACCELERATION.md) describes numeric
fallbacks, cancellation, API boundaries, and reproduction commands.

## Measured optional backends (2026-10-26)

Five measured trials follow one discarded warmup. Backend order alternates. All
cases use 30 synchronous sweeps, damping 0.2, and shared normalized formulas and
starting messages `0.05 + 0.25*((edge*37)%997)/997`. Times below are medians in
milliseconds. They include the final native host-snapshot refresh, and exclude
formula construction, backend setup, and NumPy snapshot export.

The Python extension uses Rust 1.99.1, generic Release optimization level 3,
ThinLTO, and one code-generation unit. The only recorded Rust flags are the
standard Python extension dynamic-linker arguments. Metal disables fast math.
The CPU-parallel worker budget is eight. These comparisons combine precision,
memory-traffic, and parallelism differences; they do not isolate GPU arithmetic
alone. Raw samples, compiler settings, source hashes, and the extension binary
hash are in [accelerator-results.json](bench/accelerator-results.json).

| k | N | Kind | CPU f64 ms | Parallel f64 ms | Metal f32 ms | CPU / Metal |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 400 | dense | 1.177 | 1.208 | 2.590 | 0.45 |
| 3 | 1,000 | dense | 3.287 | 3.242 | 3.123 | 1.05 |
| 3 | 10,000 | dense | 37.126 | 17.386 | 11.352 | 3.27 |
| 3 | 100,000 | dense | 611.799 | 158.261 | 43.134 | 14.18 |
| 3 | 1,000,000 | dense | 9986.066 | 2486.768 | 1853.503 | 5.39 |
| 4 | 10,000 | dense | 99.204 | 37.388 | 25.876 | 3.83 |
| 6 | 2,000 | dense | 118.502 | 48.304 | 38.711 | 3.06 |
| 12 | 1,000 | dense | 52.961 | 26.742 | 28.090 | 1.89 |
| 3 | 1,000 | residual | 1.250 | 1.257 | 1.745 | 0.72 |
| 3 | 10,000 | mixed | 36.419 | 20.005 | 15.393 | 2.37 |

CPU-parallel messages match scalar messages exactly. The largest measured Metal
message difference after 30 sweeps is 6.214e-7. No numerical fallback occurred in
these sweep workloads. Small problems remain sensitive to submission and worker
overheads; the scalar implementation remains the default.

The earlier development diagnostic in
[accelerator-before-layout.json](bench/accelerator-before-layout.json) preserves
measurements before the sweep-only memory-traffic optimizations. It is historical
evidence, with its own source hashes, rather than an additional final-build trial.

### Complete-solve campaign

Five seeds (0–4) were tested per workload, with a 100,000-flip budget and a
10-second per-job timeout. Formula generation is outside the campaign timer;
solver construction, execution, and result export are inside. Jobs execute
concurrently within the eight-worker budget. Each returned SAT assignment passes
independent CPU clause evaluation. Raw outcomes and counters are in
[accelerator-solves.json](bench/accelerator-solves.json).

| k | N | Backend | Campaign seconds | SAT | Nonconverged | Timeout |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 400 | cpu | 0.607 | 2 | 3 | 0 |
| 3 | 400 | cpu-parallel | 0.608 | 2 | 3 | 0 |
| 3 | 400 | metal | 3.958 | 2 | 3 | 0 |
| 4 | 400 | cpu | 1.577 | 1 | 4 | 0 |
| 4 | 400 | cpu-parallel | 1.585 | 1 | 4 | 0 |
| 4 | 400 | metal | 9.790 | 1 | 4 | 0 |
| 3 | 10,000 | cpu | 10.020 | 0 | 0 | 5 |
| 3 | 10,000 | cpu-parallel | 10.020 | 0 | 0 | 5 |
| 3 | 10,000 | metal | 10.030 | 0 | 0 | 5 |

CPU is substantially faster on these small complete solves. The larger workload
does not establish a time-to-solution benefit because every job timed out. SP
sweep counts across Metal and CPU runs are diagnostic: their message precision
and subsequent graph trajectories differ. These limited samples do not establish
equal solve probabilities or a general complete-solver speedup.

### Validation

Local checks passed for the portable Rust debug and Release suites, the optional
Metal Release build, Python tests, Clippy, and formatting. Three explicitly
selected GPU tests also passed with both Metal API and shader validation enabled.
They cover independent f32 message definitions, direct f64 complexity evaluation,
zero factors, arbitrary widths, high-degree scans, message restoration, real
underflow checkpoint recovery, cancellation, timeouts, and first-sweep convergence.
CPU tests cover exact trajectories and subnormal exclusion products on the
parallel path. A complete indexed-solver trajectory also matches the preserved
pre-acceleration binary. The Python source distribution was unpacked and its
optional Metal feature built successfully without relying on repository files.

## NVIDIA CUDA on ottino-lan (2026-10-26)

The remote host has an AMD Ryzen 7 1700 (8 cores, 16 logical CPUs), 16 GB RAM,
and a GeForce GTX 1060 with 3 GB VRAM, compute capability 6.1. The driver is
580.178.04; NVCC is 12.0.140. Rust 1.99.1 builds the generic Release extension
with ThinLTO, one code-generation unit, and no custom Rust flags. NVCC uses
`-O3 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false` and embeds
`sm_61` code plus `compute_61` PTX. The CPU governor was `schedutil`.

The protocol matches the Apple sweep measurements above: 30 sweeps, the same
starting messages and damping, one warmup, five alternating trials, and final
native host snapshots inside the timer. Construction, CUDA initialization,
allocation, and NumPy exports remain outside this timer. The parallel CPU budget
is eight workers. [Raw samples and hashes](bench/cuda-results.json) record the
exact build. These results compare backends on the remote host; they do not
isolate device arithmetic or establish a cross-host GPU ranking.

| k | N | Kind | CPU f64 ms | Parallel f64 ms | CUDA f64 ms | CUDA f32 ms | CPU / CUDA f64 | CPU / CUDA f32 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 400 | dense | 1.764 | 1.779 | 2.369 | 0.559 | 0.74 | 3.15 |
| 3 | 1,000 | dense | 4.711 | 4.649 | 2.548 | 0.702 | 1.85 | 6.71 |
| 3 | 10,000 | dense | 49.462 | 34.127 | 11.801 | 3.793 | 4.19 | 13.04 |
| 3 | 100,000 | dense | 1014.066 | 271.514 | 203.182 | 77.228 | 4.99 | 13.13 |
| 3 | 1,000,000 | dense | 13147.255 | 3810.818 | 2385.908 | 1041.329 | 5.51 | 12.63 |
| 4 | 10,000 | dense | 162.503 | 75.192 | 49.050 | 34.536 | 3.31 | 4.71 |
| 6 | 2,000 | dense | 222.070 | 85.179 | 54.103 | 44.528 | 4.10 | 4.99 |
| 12 | 1,000 | dense | 86.974 | 51.552 | 35.202 | 17.212 | 2.47 | 5.05 |
| 3 | 1,000 | residual | 1.824 | 1.830 | 2.191 | 0.520 | 0.83 | 3.51 |
| 3 | 10,000 | mixed | 50.644 | 35.202 | 14.991 | 6.248 | 3.38 | 8.11 |

All CUDA `f64` final messages matched the scalar reference exactly in these
workloads. The maximum CUDA `f32` difference was 7.178e-7. No fallback or device
error occurred in these sweep trials. This is measured numerical agreement,
not a guarantee that all GPU search trajectories match scalar execution.
One million variables required 951,137,472 device bytes for `f64` and
582,874,988 for `f32`. The process peak RSS was approximately 3.39 GiB, including
the simultaneously resident CPU/GPU solver instances used by the driver.

The [cavity-layout pilots](bench/cuda-cavity-layouts.json) compare a warp against
one thread per variable for low degrees, with the other degree bins unchanged.
At N=100,000 the warp takes 77.3 ms versus 179.2 ms for the scalar-thread `f32`
layout. The retained single-precision path uses a warp through degree 32, a
thread for degrees 33–127, and a cooperative block at degree 128 or above.
Double precision retains scalar per-variable arithmetic order.

A separate [SMT diagnostic](bench/cuda-cpu16.json) uses 16 CPU workers and three
measured trials. SMT helps the medium-sized parallel workloads but shows little
change at N=100,000. These samples are separate from the eight-worker paired
comparison above. Both worker counts remain explicitly selectable.

### CUDA complete-solve campaign

Five seeds (0–4) use the same 100,000-flip limit and 10-second timeout as the
Apple campaign. Jobs execute concurrently within eight worker lanes. Each batch
is measured once, so these campaign times have no repeated-trial uncertainty
estimate. Construction and result export are included, as is initial CUDA
context creation in the first CUDA batch. Formula generation is excluded.
[Every outcome and counter](bench/cuda-solves.json) is retained.

| k | N | Backend | Campaign seconds | SAT | Nonconverged | Timeout |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 400 | cpu | 0.942 | 2 | 3 | 0 |
| 3 | 400 | cpu-parallel | 0.945 | 2 | 3 | 0 |
| 3 | 400 | cuda | 2.853 | 2 | 3 | 0 |
| 3 | 400 | cuda-f32 | 1.774 | 2 | 3 | 0 |
| 4 | 400 | cpu | 2.338 | 1 | 4 | 0 |
| 4 | 400 | cpu-parallel | 2.378 | 1 | 4 | 0 |
| 4 | 400 | cuda | 4.522 | 1 | 4 | 0 |
| 4 | 400 | cuda-f32 | 4.401 | 1 | 4 | 0 |
| 3 | 10,000 | cpu | 10.022 | 0 | 0 | 5 |
| 3 | 10,000 | cpu-parallel | 10.051 | 0 | 0 | 5 |
| 3 | 10,000 | cuda | 10.025 | 0 | 0 | 5 |
| 3 | 10,000 | cuda-f32 | 10.077 | 0 | 0 | 5 |

CPU is faster on these small complete solves despite the GPU sweep speedups.
All larger jobs time out, so this campaign establishes no complete-solver
time-to-solution improvement. All returned SAT assignments pass independent
CPU clause evaluation. No CUDA fallback or device error occurred. Five seeds
are insufficient to establish equal solve probabilities across precisions.

### CUDA profiling

The separate [NVIDIA activity profile](bench/cuda-nvprof.txt) attributes 75.4%
of GPU activity time to warp cavities and 15.0% to 3-SAT clauses at N=100,000.
These percentages include transfer and memset activity in the denominator.
[Native phase timing](bench/cuda-phase-profile.json) records 159.6 ms for cold
backend setup, 73.2 ms of GPU event time, and 7.64 ms for the combined initial
and final snapshots. These instrumented samples characterize costs and are
excluded from the speedup table. [Hardware details](bench/cuda-hardware.txt)
include CPU topology and host compiler information.

### CUDA validation

The CUDA hardware suite passes on the GTX 1060 in both precisions, including an
independent CPU `f32` definition, mixed widths, cooperative high degrees, exact
zeros, double-precision subnormals, graph edits and restoration, verified solves,
concurrent jobs, cancellation, timeouts, and stopping at the first failed sweep.
CUDA 12.9 Compute Sanitizer reports zero errors across the six hardware tests;
the failure-stop regression also passes race checking, and the numerical-limit
test passes synchronization checking. An earlier full-suite race check reports
zero hazards. Profiling and sanitizer runs are separate from timing trials.

A constructed single-zero cavity exposed an exclusion-product underflow hidden
by the legitimate zero of the full product. The previous CUDA `f32` result
differed from `f64` by 1.526e-5. The added guard restores the pre-convergence
checkpoint and retries on the CPU; the regression now matches every reference
message exactly. The equivalent guard and regression pass Metal API and shader
validation on the M1 Pro.

Portable Rust Release tests, optional backend builds, eight Python tests,
Clippy, formatting, and unpacked Python source-distribution builds also pass.
See [CUDA execution](CUDA.md) for build and reproduction commands.
