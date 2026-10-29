# Time to one verified solution: corrected C++ comparison

2026-10-29. Apple M1 Pro, 16 GiB RAM, macOS 26.5.2. Random 3-SAT, alpha = 4, three identical shared inputs per size across implementations.

This replaces the C++ comparison in the initial report. The earlier C++ column timed a stock API call that returns ten assignments and whitens them; Rust returns one assignment. The earlier table alone could not rank time to one solution. The original measurements remain available in [the stock API report](STOCK_API_REPORT.md).

**What ten solutions means**

C++ runs the full BSP reduction once, producing a residual CNF. WalkSAT searches that residual until it has ten satisfying assignments. The code then lifts each assignment to the original variables, checks the original clauses, and whitens each assignment. These are ten assignments for the same input formula. They share the entire BSP reduction. In this dataset, every original ten-output batch contains ten distinct assignments; the code does not explicitly deduplicate them.

Write the batch time as `T10 = B + C1 + ... + C10`, where B is the shared setup and BSP work, and each C is completion and postprocessing. Then `T10/10 = B/10 + average(C)`. This is a valid amortized cost per assignment in that batch. Finding the first verified assignment still requires the full B. Dividing by ten cannot measure that latency.

For a workload requesting ten assignments from one CNF, batch throughput is relevant. A fair implementation comparison for that workload would give both implementations an opportunity to reuse their BSP reduction. It would be a separate experiment from solving each input formula once.

**Direct one-solution measurements**

I built the C++ extension from an isolated source copy with the package's Release configuration. The temporary controls request one WalkSAT solution and disable post-solution whitening. Phase timers are added at algorithm boundaries. Search parameters and the unchanged public Python wrapper are retained. Both original source checkouts and the installed C++ extension remain unchanged.

All three columns below return exactly one verified assignment. Values are seconds per public Python solve call, averaged equally over three formula seeds after taking each formula's median across repeats. Rust CPU and Metal measurements are reused from the initial campaign. C++ is a later, instrumented Release rebuild, so differences between the earlier and later C++ totals include build and timing variation as well as output policy.

| Variables | Rust CPU, one solution (s) | Rust Metal, one solution (s) | C++, one solution, no whitening (s) |
|---:|---:|---:|---:|
| 500 | 0.298 | 1.470 | 0.403 |
| 1,000 | 1.122 | 2.876 | 1.387 |
| 2,000 | 3.095 | 3.753 | 4.103 |
| 5,000 | 10.348 | 6.391 | 25.055 |
| 10,000 | 21.087 | 8.733 | 73.474 |
| 20,000 | 42.499 | 14.331 | 181.735 |

Rust CPU has the shortest one-solution time through 2,000 variables in this sample. Metal has the shortest time at 5,000–20,000 variables. The tested C++ build remains slower than Rust CPU for one verified solution at all six sizes. This conclusion comes from measured one-output calls, without dividing a batch time by ten.

At 500 variables there are six repeats per formula; at 1,000 and 2,000 variables there are five. Larger cases have one call for each of the three formula seeds. These are the same repetition counts, CNFs, solver seeds, timing boundaries, and shared BSP settings as the original campaign. Calls run sequentially. All native CPU execution is serial. Rust CPU is f64, Metal is f32.

**Where the C++ time goes**

In the instrumented ten-output control at N=500, formula seed 1, the complete API call took 0.989 seconds. The shared BSP reduction took 0.914 seconds, all ten WalkSAT completions took 0.025 seconds, and whitening all ten assignments took 0.036 seconds. The remaining time includes graph setup, other checks, API conversion, and teardown. The returned ten assignments and BSP sweep count match the original stock call.

For the one-output N=20,000, seed-1 case, the API call took 190.545 seconds: BSP used 190.051 seconds, the first WalkSAT completion used 0.142 seconds, and verification/postprocessing after WalkSAT used 0.0036 seconds. The shared computation dominates even when only one output is requested.

| Variables | Median fraction of one-output C++ API time in shared BSP work |
|---:|---:|
| 500 | 95.58% |
| 1,000 | 98.49% |
| 2,000 | 98.87% |
| 5,000 | 99.44% |
| 10,000 | 99.66% |
| 20,000 | 99.71% |

BSP here means the full reduction, including its repeated convergence, decimation, and backtracking work; it is not a single message-passing sweep. Graph loading before that phase is additional shared work. These phase measurements establish why the whole call cannot be scaled down by the number of outputs.

For reference, the original N=20,000 C++ batch averaged 194.609 seconds for ten outputs, or 19.461 seconds per returned assignment when amortized over that batch. The directly measured one-output C++ call averaged 181.735 seconds. The two values answer different latency/throughput questions. The difference between old and new total call times should not be attributed entirely to the output count, because the one-output version was rebuilt and measured in a later campaign.

**Correctness and reproducibility**

All 57 new C++ calls return exactly one satisfying assignment. Every one is bit-for-bit identical to the first assignment returned by its original ten-output run, and its cumulative BSP sweep count is unchanged. The saved witnesses were independently reloaded and checked against every original CNF clause and formula hash. The combined one-output table contains 171 successful calls and 171 verified assignments: 57 Rust CPU, 57 Rust Metal, and 57 C++.

No timeouts, invalid assignments, or Metal CPU fallbacks occur. The shared options remain r=0.9, batch fraction 0.00125, epsilon 0.01, damping 0, and solver seeds 1001–1003. The C++ first-output controls change the number of requested completions and omit whitening after correctness verification. They do not change the BSP loop, WalkSAT selection rule, or flip cutoff.

The isolated C++ build uses AppleClang 21, Release with O3/fast-math as configured by the project, LTO, native architecture tuning OFF, and OpenMP OFF. The complete patch, binary SHA256, CMake configuration, and compiler commands are preserved. The timing instrumentation sits outside the hot BSP sweep loop. Source-preservation hashes match both original checkouts. Rust binary hashes are unchanged from the initial campaign.

- [Corrected comparison CSV](comparison.csv)
- [Combined one-output audit and timing statistics](paired_first_solution/summary.json)
- [Every new C++ call and phase timing](first_solution/runs.jsonl)
- [New C++ metadata](first_solution/metadata.json)
- [First-assignment and sweep-count audit](first_solution/trajectory_audit.json)
- [Isolated C++ patch](first_solution_build/benchmark.patch)
- [Build configuration and binary identity](first_solution_build/build_metadata.json)
- [Compiler commands](first_solution_build/compile_commands.json)
- [Original source hashes still match](first_solution_build/original_sources_unchanged.json)
- [Original stock API report](STOCK_API_REPORT.md)
- [Original stock API CSV](stock_api_comparison.csv)

The C++ baseline is the package in `/Users/carlo/workspace/The-Backtracking-Survey-Propagation-Algorithm-code`; the requested `~/workspace/pybsp` directory is absent.

From the BSP.rs checkout, rebuild and rerun the C++ first-output experiment using new temporary paths:

```sh
/Users/carlo/workspace/The-Backtracking-Survey-Propagation-Algorithm-code/.venv/bin/python bench/prepare_cpp_first_solution.py --source /Users/carlo/workspace/The-Backtracking-Survey-Propagation-Algorithm-code --build-root /private/tmp/bsp-first-output-rebuild
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/carlo/workspace/The-Backtracking-Survey-Propagation-Algorithm-code/.venv/bin/python bench/first_solution_compare.py --extension /private/tmp/bsp-first-output-rebuild/build/_native.cpython-312-darwin.so --original bench/results/easy_alpha4_20261029/measured --output /private/tmp/bsp-first-output-rerun --numsol 1 --whiten 0
```
