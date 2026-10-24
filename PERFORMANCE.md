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

Further work needs measurements: x86-64/AVX and other ARM CPUs, hardware-counter
profiles, degree/width-specific kernels, profile-guided optimization, and any
parallel sweep implementation. Those changes need numerical comparisons and
complete-solver validation before they replace this kernel.
