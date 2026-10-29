> Archived stock API comparison: the C++ column returns ten assignments. See [the corrected one-solution report](REPORT.md) for matched output counts.

# Python solve times for random 3-SAT at alpha = 4

2026-10-29. Apple M1 Pro, 16 GiB RAM, 8 logical CPU cores, macOS 26.5.2, Python 3.12.12, NumPy 2.5.3.

On these formulas, the Rust CPU backend is faster at 500–2,000 variables. Apple Metal becomes faster at 5,000 variables and reaches a 2.97× speedup over Rust CPU at 20,000 variables. All 171 measured calls succeed, and all 684 returned assignments satisfy every original clause.

The Apple backend is named `metal` and uses custom Metal compute kernels directly. The scalar CPU backend uses f64; Metal uses f32. The C++ `pybsp` API returns ten solutions and whitens them, while BSP.rs returns one solution. Its column measures that complete stock API call. C++/Rust per-call ratios therefore include different amounts of solution and whitening work; dividing the C++ time by ten would not estimate first-solution latency.

Times below are seconds per public Python solve call. Each entry is the arithmetic mean of the per-formula median times across the same three formula seeds. A speedup above 1 favors Metal.

| Variables | Clauses | Rust CPU, 1 thread (s) | Rust Metal (s) | C++ pybsp, 10 solutions (s) | CPU time / Metal time | Verified calls per backend |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 2,000 | 0.298 | 1.470 | 0.424 | 0.20× | 18/18 |
| 1,000 | 4,000 | 1.122 | 2.876 | 1.522 | 0.39× | 15/15 |
| 2,000 | 8,000 | 3.095 | 3.753 | 4.499 | 0.82× | 15/15 |
| 5,000 | 20,000 | 10.348 | 6.391 | 29.330 | 1.62× | 3/3 |
| 10,000 | 40,000 | 21.087 | 8.733 | 87.491 | 2.41× | 3/3 |
| 20,000 | 80,000 | 42.499 | 14.331 | 194.609 | 2.97× | 3/3 |

**Inputs and timing**

Each size uses three independent, unplanted random 3-SAT formulas generated with BSP.rs formula seeds 1, 2, and 3. Each has exactly 4N clauses, three distinct variables per clause, and unbiased literal signs. The identical normalized clauses are loaded into both Python APIs; hashes and the full DIMACS inputs are saved. Solver seeds are 1001, 1002, and 1003, paired by formula. Matching numeric seeds does not imply identical RNG streams or search trajectories across implementations.

The shared BSP settings are backtracking ratio 0.9, batch fraction 0.00125, convergence epsilon 0.01, and damping 0. The C++ scorer is `cert`; optional research diagnostics and external-oracle features are disabled. Other algorithm settings retain package defaults. Rust allows 10,000,000 total WalkSAT flips over up to four restarts; C++ requests ten solutions with its native WalkSAT policy. A common cooperative 300-second wall limit applies to every measured call; no measured call reaches it.

Every timed call constructs a fresh solver. The timer surrounds the public `solve()` API, including native graph/solver setup, GPU allocations, solving, synchronization, and Python result conversion. Imports, input Formula construction, options-object construction, independent validation, and witness compression are outside the timer. C++ still performs its inherent Formula-to-native conversion inside `solve()`. One unrelated warmup per backend is discarded. GPU pipelines can remain cached in the process; cold process startup and first-ever Metal compilation are excluded. Profiling is disabled for measured calls.

Calls run sequentially, with backend order deterministically shuffled for each formula and repetition. Rust uses the scalar `cpu` backend and one host thread. The installed C++ extension links no OpenMP runtime; OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and VECLIB_MAXIMUM_THREADS are all set to 1. Both APIs use the same Python interpreter and NumPy installation.

At 500 variables, six repetitions per formula yield 18 calls per backend. At 1,000 and 2,000 variables, five repetitions yield 15 calls per backend. At larger sizes, one call per formula yields three calls per backend. The harness adds repetitions until every backend accumulates at least five seconds per size. The 500-variable case needs the extra sixth repetition to meet that threshold.

For the small sizes, timing dispersion within a fixed formula is shown below as median absolute deviation divided by median time. The ranges cover the three formulas. At larger sizes, one timing per formula cannot separate timing noise from input difficulty; those results describe this three-formula sample.

| Variables | Repeats per formula | CPU relative MAD | Metal relative MAD | C++ relative MAD |
|---:|---:|---:|---:|---:|
| 500 | 6 | 0.56–0.99% | 1.24–2.72% | 0.58–1.52% |
| 1,000 | 5 | 0.15–1.74% | 3.71–16.25% | 1.22–5.10% |
| 2,000 | 5 | 3.10–6.65% | 1.69–3.88% | 2.55–6.90% |

**Per-formula timings**

Each small-case time is the median across its repetitions. Larger-case times are single calls. These rows expose the variation in formula difficulty.

| Variables | Formula seed | Rust CPU (s) | Rust Metal (s) | C++ stock API (s) |
|---:|---:|---:|---:|---:|
| 500 | 1 | 0.496568 | 2.497584 | 0.730461 |
| 500 | 2 | 0.127178 | 0.552375 | 0.179383 |
| 500 | 3 | 0.270157 | 1.359791 | 0.363563 |
| 1,000 | 1 | 1.742266 | 4.535667 | 2.337600 |
| 1,000 | 2 | 1.538952 | 3.921091 | 2.085009 |
| 1,000 | 3 | 0.085500 | 0.171738 | 0.143183 |
| 2,000 | 1 | 0.613669 | 0.654366 | 0.863001 |
| 2,000 | 2 | 4.145706 | 4.765526 | 5.556057 |
| 2,000 | 3 | 4.526292 | 5.838190 | 7.079314 |
| 5,000 | 1 | 7.694782 | 4.733025 | 19.879324 |
| 5,000 | 2 | 9.222809 | 6.625528 | 29.310578 |
| 5,000 | 3 | 14.125955 | 7.814279 | 38.798932 |
| 10,000 | 1 | 26.106729 | 11.583133 | 118.859053 |
| 10,000 | 2 | 20.285648 | 8.351307 | 84.548037 |
| 10,000 | 3 | 16.870007 | 6.264599 | 59.065111 |
| 20,000 | 1 | 42.349880 | 14.192148 | 201.257737 |
| 20,000 | 2 | 37.505817 | 12.368175 | 169.546342 |
| 20,000 | 3 | 47.641620 | 16.432692 | 213.023032 |

**Correctness and execution audit**

All 18 distinct formulas are solved by every backend. All 171 measured calls return success: 57 Rust CPU, 57 Rust Metal, and 57 C++ calls. The audit checks 114 Rust assignments and 570 C++ assignments, for 684 in total. Every assignment has the required length and encoding, and every one satisfies all 4N clauses. C++ signed literals must have absolute values 1 through N in variable order; Rust values must be 0 or 1. An independent NumPy clause evaluator checks the original shared inputs, independently of both solvers’ own satisfaction checks.

The saved boolean witnesses are then reloaded and checked again against the saved CNFs by `bench/summarize_easy_regime.py`. Formula SHA256 hashes are verified. Every Metal call reports backend `metal`, device Apple M1 Pro, positive Metal sweep counts, zero fallbacks, and no device error. The benchmark-script hash and both extension-binary hashes still match the metadata after the campaign.

The preliminary pilot is excluded from every table above. Its 20,000-variable C++ call was cancelled at 120 seconds, which motivated the main run’s 300-second cap. The audit excludes this cancelled call from solved-time comparisons. The main run has zero failures, timeouts, invalid assignments, or CPU fallbacks.

**Builds and files**

BSP.rs checkout: `14a176dd8e24312e872dc64ac44c6d49304ffd13`. The Rust extension was rebuilt with `maturin develop --release --locked --features extension-module,metal`; its embedded build metadata confirms release, aarch64, and rustc 1.99.1. C++ checkout reference: `17cea80664e54c92a7e814e0f4c35e47a778abfb`. The existing installed C++ extension was benchmarked unchanged. Its binary SHA256 identifies the tested build independently of the current source checkout; the recorded checkout revision is not a fresh-build guarantee.

- [CSV comparison](stock_api_comparison.csv)
- [Summary and dispersion statistics](stock_api_summary.json)
- [Every measured call](measured/runs.jsonl)
- [Environment, options, and binary hashes](measured/metadata.json)
- [Observed source hashes](source_provenance.json)
- [Embedded Rust build metadata](rust_build_profile.json)
- [Shared formulas](measured/formulas/)
- [All solution witnesses](measured/witnesses/)

From the BSP.rs checkout, reproduce the measured campaign with GPU access:

```sh
env UV_CACHE_DIR=/private/tmp/bsp-benchmark-uv VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --release --locked --features extension-module,metal
env PYTHONPATH="$PWD/python" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/carlo/workspace/The-Backtracking-Survey-Propagation-Algorithm-code/.venv/bin/python bench/easy_regime_compare.py --output /private/tmp/bsp-alpha4-rerun --formula-seeds 1 2 3 --repeats 1 --small-repeats 5 --min-seconds 5 --timeout 300
.venv/bin/python bench/summarize_easy_regime.py /private/tmp/bsp-alpha4-rerun --output /private/tmp/bsp-alpha4-rerun-summary
```

The requested `~/workspace/pybsp` directory was absent. The installed C++ `pybsp` package is in `~/workspace/The-Backtracking-Survey-Propagation-Algorithm-code`; that is the baseline measured here. Solver sources in both repositories were preserved.
