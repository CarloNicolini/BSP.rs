# CUDA layout setup and T4 profiling, 2026-11-04

The production change replaces both stable sorts in `layout::windows` with
bucket counters and reuses the inverse-permutation allocation for `q_slot`.
It preserves every transport index, including stable ordering within windows.
The device buffer sizes and SP arithmetic are unchanged. The retained observable
optimization prefetches eight records while preserving serial Kahan order.

## Test gate

The regression tests were committed as `dc6483d` before changing construction.
They run through `tests/cuda_layout.rs` on CPU-only CI as well as in the CUDA
crate. Tests compare against an independent stable-sort specification, exercise
empty and partial windows, cross both window boundaries with three random
permutations per size, check bijections and inverse mappings, decode warp
records around degree boundaries (including zero-degree variables and hubs),
and check sparse/generic clause headers.

The T4's six CUDA tests cover the independent f32 recurrence, f64/CPU parity,
widths 3/4/6/12/65, fix/release mutations, errors, cancellation, and verified
SAT solutions. Candidate timings additionally require identical complete-state
hashes and observable scalars across the compared binaries.

## Measurement protocol

The public source baseline is `ad6a0d3c496adc5fcf7e01c0ea44cffaba2e02eb`.
`results/metadata.json` records the actual compiler, CUDA toolkit, Nsight Compute,
CPU, GPU, architecture and flags. Device measurements use a Colab Tesla T4,
sm_75, f32, safe math, random 3-SAT with alpha=4.2 and
N=50,000 / 100,000 / 1,000,000. Seeds are 3, 7 and 11.

Host construction uses `setup.rs`, compiled with `rustc -O`, and nine fresh
processes per variant/size, paired by seed and with randomized variant order.
Its generator constructs valid random 3-SAT with distinct variables per clause;
it is deterministic but differs from `Formula::random`. Input construction,
checksum computation and serialization are outside the layout timer. Both
variants must produce the same checksum. A tracking allocator measures peak
live requested bytes during `windows`; Linux `VmHWM` measures whole-process
peak RSS, including input construction and allocator behavior. These are
separate quantities. The reference is preserved in `windows_sort.rs`.

GPU timings use `examples/cuda_layout_bench.rs` and the actual `CudaSolver`.
Each process creates one formula/state, then performs two warmup batches and
three timed batches, each containing 50 raw sweeps. Initial messages are
restored outside each timed batch. Epsilon=-1 disables early stopping in this
low-level benchmark. Formula generation, topology/layout creation, uploads,
snapshots, hashing, and JSON writing are outside CUDA-event timing. The harness
also reports host wall time for the submitted batch. Final messages, saved
messages, surveys and certainties are hashed; complexity and paramagnetic flags
must agree exactly. This is a sweep benchmark, not a full-solve comparison.

For observable timings, pass `--mode obs --output observables.json` to
`run_native.py` with baseline=obsbase-bench, prefetch4 and prefetch8 binaries.
Each batch first restores messages and runs two untimed sweeps, then measures
50 complete observable passes. Two warmup batches precede three timed batches.

For each size, `run_native.py` randomizes binary order within each of three
rounds and three seeds: nine process pairs, with three samples per process.
Summaries use process medians, avoiding treating repeated batches as independent
processes. GPU timings run sequentially, after tests, compilation and Nsight.
An exploratory run overlapped device tests and was discarded. A bookkeeping
error in the initial comparison driver was fixed before the retained trials;
the numerical hashes themselves agreed.

Nsight Compute reports use SpeedOfLight, MemoryWorkloadAnalysis, Occupancy,
LaunchStats, SchedulerStats, and (for sweeps) WarpStateStats. Nsight's replay
measurements are separate from the unprofiled timing samples. The first eight
sweep launches are captured at each size. Observable kernels are profiled
separately. Text and CSV exports are retained alongside raw timing samples.

## Reproduce

On a Linux T4 machine with CUDA, Rust and Python:

```sh
export BSP_CUDA_ARCH=75
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cargo test --locked --test cuda_layout
cargo test --release --locked --features cuda --test cuda -- --include-ignored
mkdir -p experiments/cuda_layout_20261104/results
python3 experiments/cuda_layout_20261104/run_setup.py
python3 experiments/cuda_layout_20261104/build_candidates.py
python3 experiments/cuda_layout_20261104/build_pack4.py
python3 experiments/cuda_layout_20261104/build_observable_candidates.py
python3 experiments/cuda_layout_20261104/build_warp_prefetch.py
```

For example, run the sweep and observable campaigns separately:

```sh
experiment_root="$(pwd)/experiments/cuda_layout_20261104"
python3 "$experiment_root/run_native.py" \
  "baseline=$experiment_root/baseline-bench" \
  "block128=$experiment_root/block128-bench" \
  "q14=$experiment_root/q14-bench" "q18=$experiment_root/q18-bench"
python3 "$experiment_root/run_native.py" --output pack4.json \
  "baseline=$experiment_root/baseline-bench" "pack4=$experiment_root/pack4-bench"
python3 "$experiment_root/run_native.py" --mode obs --output observables.json \
  "baseline=$experiment_root/obsbase-bench" \
  "prefetch4=$experiment_root/prefetch4-bench" \
  "prefetch8=$experiment_root/prefetch8-bench"
```

Run `run_native.py` with `label=/absolute/path/to/binary` arguments for baseline,
block128, q14 and q18; use `--output pack4.json` for the separate baseline/pack4
comparison. Use `profile.py` after building the production executable for sweep
profiles and `profile_observables.py` for observable profiles. Run each phase
sequentially. `summarize.py` produces `results/summary.json` from the raw samples.
Candidate builders read the original kernel from the recorded Git baseline and
restore current production sources in `finally`; their binaries are
experimental artifacts and are ignored by Git.

## Results

The tables below are derived from the archived samples.

### Host construction

Colab exposes two logical CPUs on an Intel Xeon at 2.00 GHz. Nine fresh-process
pairs per size give these medians (decimal MB for allocator measurements):

| N | Stable sorts | Linear counters | Speed ratio | Peak layout bytes, sort → linear | Process peak RSS, sort → linear |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 | 28.28 ms | 13.27 ms | 2.13× | 15.12 → 10.08 MB | 18.96 → 14.21 MiB |
| 100,000 | 51.91 ms | 24.59 ms | 2.11× | 30.24 → 20.16 MB | 35.80 → 26.23 MiB |
| 1,000,000 | 5.145 s | 0.659 s | 7.80× | 302.40 → 201.60 MB | 362.38 → 242.52 MiB |

All 27 paired checksums match. The algorithm removes about one-third of peak
requested layout allocation. At N=1,000,000 the measured difference is 100.8 MB;
whole-process RSS falls by about 119.9 MiB. Timing ratios are specific to this
VM/compiler; allocator peak counts are requested bytes rather than RSS or total
solver memory. This improves host window setup and does not change device
kernel work or device memory.

### Nsight sweep diagnosis

Representative first-sweep measurements at N=1,000,000:

| Kernel | Replay duration | DRAM throughput | Achieved occupancy |
| --- | ---: | ---: | ---: |
| `pack_eta<float>` | 0.776 ms | 75.2% | 88.6% |
| `cavities_simd<float>` | 1.51 ms | 75.9% | 93.1% |
| `clauses<float, 3>` | 0.917 ms | 89.8% | 97.7% |
| `residual_final<float>` | 0.00253 ms | 0.54% | 7.4% |

Cavities use 47 registers per thread and about 0.88 eligible warps per scheduler.
The profile suggests memory traffic/latency is a more useful target than
increasing occupancy. The low occupancy of the tiny final residual kernel is
not a reason to prioritize it over the larger phases. Replay durations do not
sum to the repeated unprofiled batch time: clocks, cache state, launch gaps,
and Nsight replay differ.

### Paired sweep candidates

Each cell is the median of nine paired baseline/candidate process ratios.
Values above 1 favor the candidate. Each process contributes the median of
three CUDA-event batches.

| Candidate | N=50,000 | N=100,000 | N=1,000,000 | Decision |
| --- | ---: | ---: | ---: | --- |
| block128 | 1.009× | 1.010× | 0.982× | Keep 256 threads |
| q14 | 1.006× | 1.036× | 0.943× | Reject: slower at 1M |
| q18 | 0.982× | 1.008× | 1.014× | Keep current window |
| pack4 | 1.026× | 0.994× | 1.002× | Keep scalar pack |

All comparisons match final message/saved-message/survey/certainty hashes,
complexity and paramagnetic flags exactly. The first comparison uses baseline
process medians of 0.192 / 0.359 / 4.536 ms per sweep. The separate pack4
comparison uses baselines of 0.199 / 0.392 / 5.231 ms; do not compare their raw
medians across sessions. Telemetry captures changing clocks and temperatures;
randomized pairing reduces, but does not eliminate, that variability. None of
these four sweep candidates justifies a production change.

### Observable profile and ordered-prefetch experiment

`observable_final<float>` takes 0.0758 / 0.149 / 1.47 ms at the three sizes,
using one active thread for serial Kahan summation. Its achieved occupancy is
about 3.1%, compute throughput about 0.1%, and DRAM throughput about 0.12% at 1M.
`clause_observables<float>` takes about 1.01 ms at 1M and uses 86% of DRAM
throughput. These profiles motivate testing grouped record loads before
consuming records in their original Kahan order. `build_observable_candidates.py`
compares prefetch groups of four and eight records, with an instruction barrier
between loads and consumption. They change neither the record order nor the
summation recurrence. The complete paired comparison preserves exact complexity and state hashes.

### Retained observable optimization

| N | Baseline observable pass | Eight-record prefetch | Median paired speed ratio |
| ---: | ---: | ---: | ---: |
| 50,000 | 0.335 ms | 0.326 ms | 1.029× |
| 100,000 | 0.668 ms | 0.655 ms | 1.020× |
| 1,000,000 | 8.494 ms | 8.328 ms | 1.026× |

These are complete observable passes, including packed cavities, variable and
clause terms, and both reduction stages. Eight-record prefetch wins 9/9,
8/9 and 9/9 process pairs at the three sizes. The effect is modest: about 2–3%
by median paired speed ratio, not a full-solve speedup.

At N=1,000,000, isolated Nsight replay measurements at approximately 1.60 GHz
SM clock are 1.47 ms (original), 1.63 ms (prefetch4), and 1.07 ms (prefetch8).
Eight-record prefetch reduces this kernel's duration by about 27%; register
usage rises from 22 to 39 per thread. A further coalesced-load experiment uses
one warp to load records and lane zero to consume them in their original order.
Its isolated duration is 1.45 ms, so it was screened out before a full paired
campaign. The compiler/source transformation for that rejected candidate is
retained in `build_warp_prefetch.py`.

The original and retained reductions pass the dedicated exact-order regression
for both f32 and f64: cancellation-sensitive fixtures with 1/2/3/4/5/7/8/9/
15/16/17/255/257 records. Partial groups never read past the valid records. The
retained kernel changes only record-load scheduling; the serial Kahan
recurrences, record order, paramagnetic flags and final precision remain the
same. Complete paired state hashes and observable scalars match for all seeds.

## Validation and scope

Local default debug/release suites, formatting, Clippy, synchronized release
metadata and release-policy tests pass. The T4 runs the CUDA-enabled release
suite, including explicitly enabled device tests, plus the CUDA crate's
exact-order reduction test and layout tests. Final commands and outcomes are
recorded in `results/validation.log`. No NVIDIA architecture other than the
recorded T4 was executed. The f64 cancellation regression checks the new load
schedule, but the performance campaign uses f32.

The archived JSON/CSV/text samples are small enough to review in this PR.
`results/manifest.json` records their SHA-256 hashes, the retained source hashes,
and the experiment binary hashes. `results/summary.json` is generated locally
by `summarize.py` from those samples. Raw Nsight reports can be regenerated with
the retained profile drivers; their exported metrics are versioned here.

### Interpretation and next experiments

Kernel changes must earn their extra contracts. Window sizes and block sizes
are hardware-dependent choices; a small T4-only gain with a small-input
regression is insufficient evidence to change defaults for every CUDA device.
The host counter construction is useful independently of those choices.

The next memory experiment worth testing is lossless compression of `q_slot`:
with the current 2^16 q window, its bucket follows the known variable position,
so the stored slot can potentially use a 16-bit offset plus a reconstructed
bucket base. This could reduce index bytes without quantizing SP messages.
It would require an explicit host/device window contract, boundary tests, and
new paired measurements; it is not implemented here. Optimizing reductions by
changing floating-point summation order would need a separate numerical
compatibility decision.
