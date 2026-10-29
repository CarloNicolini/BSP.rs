# Isolated BSP memory and speed experiments

**Integration status.** The three retained changes below were reviewed and
merged into main through PR #8 after fresh Rust, Python, and Metal device tests.
The measurements and preservation statements below remain the historical record
of the isolated campaign at its recorded revisions; they were not rerun after
the SDK/documentation changes. Rejected alternatives are not production changes.

2026-10-29. Reference production commit:
`aec3bed32d49bb1450132594c1df3f16ebc8449a`. Development branch:
`codex/memory-ablation-dev`. Main was kept separate throughout; nothing was
pushed or merged. All sizes below use decimal MB/GB.

## Findings and retained changes

Three changes provide repeatable allocation savings while preserving the
observed numerical trajectory within each backend:

1. Reuse active/free slots of the host frozen-message array for cavity norms,
   removing one FP64 array of length E.
2. Apply the same lifetime reuse on Metal, removing one FP32 array of length E.
3. Allocate Metal's generic prefix buffer only when a generic-width clause
   kernel needs it. Pure 3-SAT and 4-SAT already use specialized local prefixes.

Here E=K*M=K*alpha*N counts literal incidences. No precision, initialization,
residual tolerance, BSP policy or WalkSAT parameters were changed. Frozen
messages remain stored and are restored with their original values.

These are memory improvements with a modest timing cost in some workloads.
The million-variable BSP prefix was about 5–7% slower. Complete 100k Metal
solves were essentially unchanged; the eight-thread CPU solve was about 3%
slower. There is no established general speedup. The code remains isolated on
the development branch so the working main remains available.

## Million-variable measurements

3-SAT, alpha=4.2, N=1,000,000, M=4,200,000, E=12,600,000.
The following is **32 BSP actions**, including observable evaluation and
fix/release operations, from the same initialized instance and solver seed.
All variants performed 142 SP sweeps and ended with identical message-bit
hashes and complete action records within each backend.

| Backend | Baseline peak footprint | Combined peak footprint | Reduction | Baseline time | Combined time | Baseline / combined |
|---|---:|---:|---:|---:|---:|---:|
| CPU, 1 thread, FP64 | 818.2 MB | 717.3 MB | 12.3% | 57.890 s | 60.708 s | 0.954x |
| CPU, 8 threads, FP64 | 1,280.1 MB | 1,179.3 MB | 7.9% | 15.846 s | 16.634 s | 0.953x |
| Apple Metal, FP32 | 1,372.2 MB | 1,169.3 MB | 14.8% | 13.456 s | 14.364 s | 0.937x |

Metal requested buffer payload fell from **582.863 to 482.063 MB**, a **17.3%**
reduction. Unified GPU memory is included in process footprint; these two
metrics must not be added. This is an actual GPU execution, without CPU fallback.

The initial complete-solve sizing attempts at N=1,000,000 exhausted a 60 s
cooperative budget on both scalar CPU and Metal. They did not return a SAT
witness. Following the requested fallback, complete-solution comparisons use
N=100,000. The million-variable measurements establish operation and savings
on a prefix; they do not establish completion at that size.

Raw SP sweeps give a second view without repeated host snapshots. Over three
paired trials of 20 sweeps at N=1,000,000:

| Backend | Baseline / combined footprint, median | Median paired speed ratio |
|---|---:|---:|
| CPU, 1 thread | 753.4 / 652.5 MB | 1.004x |
| CPU, 8 threads | 1,069.1 / 968.5 MB | 0.952x |
| Metal | 1,369.2 / 1,167.5 MB | 1.022x |

Public `advance()` synchronizes messages after each action. Native `solve()`
avoids some of those snapshots. Their timings describe different API workloads.

## Complete solutions at N=100,000

All completed runs below used the same saved 3-SAT instance at alpha=4.2,
formula seed 1, solver seed 1001, epsilon=0.01, damping=0, r=0.9 and default
decimation fraction 0.00125. CPU uses eight threads here. Each SAT witness was
checked by the native verifier and a separate streaming Python DIMACS evaluator.

| Paired comparison | Baseline time | Candidate time | Speed ratio | Baseline / candidate peak footprint |
|---|---:|---:|---:|---:|
| Combined, CPU 8 threads | 409.483 s | 422.985 s | 0.968x | 155.5 / 139.6 MB |
| Combined, Metal, pair 1 | 90.076 s | 88.613 s | 1.017x | 160.1 / 140.0 MB |
| Combined, Metal, reverse order pair 2 | 87.829 s | 87.471 s | 1.004x | 160.5 / 139.9 MB |
| Metal batching, separate experiment | 87.977 s | 83.378 s | 1.055x | 160.0 / 165.4 MB |
| Skip unused CPU logs, separate experiment | 409.483 s | 405.116 s | 1.011x | 155.5 / 153.2 MB |

Every successful Metal variant returned the **same witness and byte-identical
action trace**: 64,452 SP sweeps and 10,144 BSP actions. The CPU variants also
agree exactly with their CPU reference: 64,362 sweeps and 10,106 actions.
CPU FP64 and Metal FP32 follow different trajectories, so their approximately
4.7x end-to-end runtime ratio is an observed backend comparison on this instance,
not an isolated kernel speedup or a statement about general success rates.

Early Metal controls took 167.925 and 86.434 s despite identical work and
witnesses. Early individual candidates took 169.672 s (host norms), 175.163 s
(Metal norms), 128.422 s (lazy scratch), and 87.939 s (conditional prefix).
Those raw results remain in the data. They cannot support attributing a large
speedup to a change. Later comparisons name their nearby control explicitly.

System-wide compression and swap activity occurred during later full solves,
including the CPU control/candidate. This shared laptop was not a stationary
timing environment. Per-process footprint and system VM counter deltas are
recorded separately. Small timing differences need quieter repeated trials
before they can be considered an improvement or regression estimate.

## Individual ablations and rejected ideas

Each of the first four changes was built independently from the same baseline
in its own worktree and target directory. The table reports three paired
million-variable, 20-sweep trials. Memory savings are stronger evidence here
than timing ratios close to one.

| Isolated change | Scalar footprint | CPU 8-thread footprint | Metal footprint | Metal speed ratio | Decision |
|---|---:|---:|---:|---:|---|
| Baseline | 755.2 MB | 1,070.3 MB | 1,372.0 MB | 1.000x | Control |
| Host norm reuse | 654.3 MB | 967.4 MB | 1,268.2 MB | 1.008x | Retain; removes 8E bytes |
| Metal norm reuse | 755.2 MB | 1,070.3 MB | 1,320.7 MB | 0.988x | Retain; removes 4E−4 device bytes |
| Conditional Metal prefix | 755.2 MB | 1,070.3 MB | 1,321.6 MB | 1.006x | Retain; removes 4E−4 bytes for specialized widths |
| Lazy host q/norm scratch | 654.3 MB | 965.8 MB | 1,269.4 MB | 1.001x | Reject in current combination |

Lazy scratch looked attractive during raw GPU sweeps, but q/norm storage
regrows when host observable validation or CPU fallback needs it. Its complete
Metal solve peaked at 157.2 MB, versus 150.0 MB for permanent host norm reuse.
Applying lazy q allocation after norm reuse produced essentially the same
million-variable footprint as the retained combination and slightly slower
short BSP workloads. It offers no useful additive saving in these trials.

Raising Metal's submission target from 2 to 8 ms retained its 16-sweep cap and
passed cancellation/fallback checks. Raw sweeps and short actions stayed near
1.00x. Its single full solve was 1.055x faster, within the observed timing
variation, with a higher footprint and a changed submission/cancellation
latency target. It was not retained.

Skipping logarithms for inactive clauses/fixed variables was also isolated.
Its short CPU workload was 0.974x and complete solve 1.011x, with no allocation
saving. A separate interaction test applies it after norm reuse, where unused
slots contain frozen messages rather than norms. The final result is recorded
in the accompanying summary: 437.721 s versus 422.985 s for the combination
without this change (0.966x), with the same witness and trace. It was rejected.

## Added paired 4-SAT checks

N=100,000, formula seed 1 and solver seed 1001, with the same options and
baseline/combined sources as above. Each pair requested 32 BSP actions.

| Alpha | Backend | Baseline / combined footprint | Baseline / combined time | Speed ratio |
|---|---|---:|---:|---:|
| 4.2 | CPU, 1 thread | 129.2 / 117.6 MB | 0.295 / 0.292 s | 1.009x |
| 4.2 | CPU, 8 threads | 190.7 / 181.8 MB | 0.108 / 0.114 s | 0.944x |
| 4.2 | Metal | 211.6 / 185.8 MB | 0.124 / 0.134 s | 0.929x |
| 9.8 | CPU, 1 thread | 279.6 / 249.5 MB | 9.531 / 11.200 s | 0.851x |
| 9.8 | CPU, 8 threads | 414.0 / 382.4 MB | 3.321 / 3.709 s | 0.895x |
| 9.8 | Metal | 434.2 / 371.5 MB | 2.081 / 2.365 s | 0.880x |

Alpha=4.2 reached SAT after one BSP action including WalkSAT. All six witnesses
were independently verified and matched their respective backend controls.
Alpha=9.8 completed all 32 actions without returning SAT; message hashes and
step records matched exactly. This confirms the memory changes also operate
on K=4, with a visible speed cost in this denser prefix. It does not establish
full-solve success at K=4/9.8. These short trials have one pair each.

At alpha=9.8, Metal buffer payload fell from 165.625 to 134.265 MB (18.9%).
At alpha=4.2 it fell from 73.407 to 59.967 MB (18.3%). The slightly larger
degree-bucket overhead in the dense formula is included in these measurements.

No candidate produced a witness or trajectory mismatch in the completed paired
checks. The rejected changes failed the benefit criterion, rather than being
classified as incorrect. Finite tests do not prove equivalence for every input.

## Final validation

The summary checks **132 paired prefix/sweep controls** with zero mismatches.
It also checks every successful full-solve group for identical witness/trace
hashes within its backend, and six explicitly referenced full-solve timing
pairs. Sixteen full 3-SAT runs and six early-completing 4-SAT runs returned
independently verified SAT witnesses. Full-solve confirmations use one 3-SAT
instance; three independent 3-SAT formulas were used for short action trials.

Final checks on the retained production sources passed:

- Rust debug and release suites: 30 tests each; Metal-enabled release: 31 tests.
- All five explicitly enabled Metal hardware tests, including generic-width
  prefix allocation, freeze/restore, numerical fallback and cancellation.
- Formatting for the root and Metal crate, Clippy with warnings denied, and
  the documentation-test command (no Rust doctests are defined).
- Eight Python tests and the verified solve example, using a freshly built
  Metal-enabled wheel in an isolated temporary Python environment. Main's
  installed Python package was not changed.
- Independent verifier smoke check accepts a known satisfying assignment and
  rejects a known violating assignment.

The first Python collection command used `/dev/null` as its config root and
traversed an unrelated unavailable directory. Explicitly restricting pytest's
root/collection paths fixed collection; the complete suite then passed. Both
attempts are retained in the logs.

Local and remote main remain at the original reference commit. Main's solver,
formula and Metal production sources were compared byte-for-byte with that
commit. Additional README/CUDA/documentation and lock-file edits appeared in
the main checkout during this campaign; this work did not write or revert them.
The nine temporary experiment worktrees were archived with recoverable snapshots;
the development worktree remains attached. Branch revisions and source patches
are included with the measurement archive.

## K=3 and K=4 capacity estimates

The exact savings per million variables are:

| Workload | E/N | Host norm array removed | Each Metal array removed | Total Metal buffer reduction |
|---|---:|---:|---:|---:|
| K=3, alpha=4.2 | 12.6 | 100.8 MB | 50.4 MB | 100.8 MB |
| K=4, alpha=4.2 | 16.8 | 134.4 MB | 67.2 MB | 134.4 MB |
| K=4, alpha=9.8 | 39.2 | 313.6 MB | 156.8 MB | 313.6 MB |

The denser K=4 case is an explicit workload assumption, not an assertion that
alpha=4.2 has comparable difficulty for both K. Memory follows K*alpha, rather
than K alone. K=4 at alpha=9.8 has 3.11 times as many incidences as the K=3 case.

For fixed K=3/4, the retained scalar payload/capacity ledger is approximately
`36E + 36.125M + 74.125N` bytes. Parallel execution adds approximately
`32E + 24N + 8M`. Retained Metal buffers are approximately
`28E + 45N + 20M + 16*ceil(max(N,M)/256)` bytes, plus small constants and
degree-bucket lists. Metal device memory is additional to host state. CUDA's
device layout is unchanged and still uses the larger baseline ledger.

| N | Workload | Scalar host payload | Host with CPU parallel scratch | Metal buffers alone |
|---|---|---:|---:|---:|
| 10^7 | K=3, alpha=4.2 | 6.79 GB | 11.40 GB | 4.82 GB |
| 10^7 | K=4, alpha=4.2 | 8.31 GB | 14.26 GB | 6.00 GB |
| 10^7 | K=4, alpha=9.8 | 18.39 GB | 31.96 GB | 13.39 GB |
| 10^8 | K=3, alpha=4.2 | 67.95 GB | 114.03 GB | 48.21 GB |
| 10^8 | K=4, alpha=4.2 | 83.07 GB | 142.59 GB | 59.97 GB |
| 10^8 | K=4, alpha=9.8 | 183.94 GB | 319.62 GB | 133.92 GB |
| 10^9 | K=3, alpha=4.2 | 679.45 GB | 1,140.25 GB | 482.06 GB |
| 10^9 | K=4, alpha=9.8 | 1,839.35 GB | 3,196.15 GB | 1,339.21 GB |

These are analytical payloads/reserved capacities, not peak footprint or
successful allocation guarantees. Parsing, construction, queues, results,
WalkSAT, driver/compiler memory and allocator overhead add phase-specific
peaks. The billion-variable extrapolations also exceed current u32 index
limits. Incidence addressing permits fewer than roughly 341 million variables
for K=3/4.2 and 110 million for K=4/9.8. Current GPU dispatch independently caps
N at 134,217,727. Widening/tiled addressing is required beyond these limits.

The three changes are a useful first reduction, but **do not make N=10^8 fit
on a 24/40 GB GPU**. The next architecture must remove host mirrors and more
edge arrays. At N=10^7, K=4/9.8 already exceeds this laptop's 16 GiB capacity
in the scalar ledger alone. K=3 may be feasible locally, but full Metal host
plus device state and OS headroom remain tight.

The earlier [design audit](../billion_variables_20261029/REPORT.md) develops
implicit Feistel topology, a single frozen/current message payload, recomputed
cavities, compact clause/variable state and separate phase lifetimes. Its
unimplemented FP32 implicit profiles estimate 8.18 GB at N=10^8 for K=3/4.2 and
19.79 GB for K=4/9.8; corresponding 16-bit profiles estimate 5.66 and 11.95 GB.
Those are conditional design budgets with allowances, not measurements. They
exclude a retained edge q array; adding FP32 q costs another 5.04 or 15.68 GB.
Keep FP32 as the first compact reference before pursuing 16-bit storage.

## Probability arithmetic and K dependence

Storage width and multiplication precision are separate decisions. No finite
floating-point format represents every repeated product in [0,1] exactly.
Use FP32/FP64 arithmetic on decoded values; changing eta storage is an additional
approximation. Native half/BF16 matrix throughput does not predict throughput
for irregular graph gathers and products.

| Format | Bytes/value | Fraction bits | Max absolute rounding error on [0,1] | Smallest normal | Smallest subnormal |
|---|---:|---:|---:|---:|---:|
| FP64 | 8 | 52 | 5.55e-17 | 2.23e-308 | 4.94e-324 |
| FP32 | 4 | 23 | 2.98e-8 | 1.18e-38 | 1.40e-45 |
| BF16 | 2 | 7 | 1.95e-3 | 1.18e-38 | 9.18e-41 |
| FP16 | 2 | 10 | 2.44e-4 | 6.10e-5 | 5.96e-8 |
| UNORM16 | 2 | integer code | 7.63e-6 | uniform spacing 1/65535 | same spacing |

The binary format ranges follow [NVIDIA's floating-point properties](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).
The rounding errors are derived from half the representable spacing below one,
assuming round-to-nearest with subnormals preserved. Hardware flush-to-zero can
lose smaller values earlier.

BF16 rounds interior probabilities within about 1/512 of one to exactly one;
FP16 does so within about 1/4096. That can create exact zero factors in
`1-eta`, and invalid normalizers. At N=10^9 with uniform random initial eta,
expected newly rounded ones are 24.6 million (BF16) and 3.08 million (FP16) for
K=3/4.2; K=4/9.8 raises them to 76.6 and 9.57 million. A wider BF16 exponent
does not protect this endpoint information.

In a clause update, K=3 multiplies two cavity probabilities and K=4 multiplies
three. With equal q=0.001, these give 1e-6 and 1e-9 respectively: the latter
cannot be stored as a positive IEEE FP16 value. Even 1e-6 is subnormal in FP16
and vulnerable to flush-to-zero. Variable updates multiply over signed degree,
not K−1; average signed degree is approximately 6.3 for K=3/4.2 and 19.6 for
K=4/9.8. Degree tails and small factors make stable prefix/suffix handling and
explicit zero checks necessary for both.

Products of two FP16 inputs can be exact in FP32 (22 significand bits and
adequate exponent range). Three may need 33 significand bits; FP64 can hold
that product exactly. Two or three UNORM16 integer codes multiply exactly in
u32 or u64 respectively, followed by one normalization. These observations
only address multiplication of already quantized inputs; input error and
normalization/division error remain.

The saved storage-only probe uses real scalar SP at N=10,000, 80 sweeps, three
seeds for each K, FP64 arithmetic, and no BSP mutations. Plain BF16 and FP16
failed numerically in all 3/3 cold-start trials for both K. UNORM16 completed
3/3 for K=3 and failed 3/3 for K=4. Endpoint escapes prevented those immediate
failures but did not guarantee trajectory agreement. After 200 FP64 warm-up
sweeps, the maximum observed trajectory error over the next 80 sweeps was:

| Stored format | K=3, alpha=4.2 | K=4, alpha=9.8 |
|---|---:|---:|
| FP32 | 2.45e-7 | 9.22e-8 |
| BF16 | 2.15e-2 | 5.85e-3 |
| FP16 | 1.99e-3 | 9.40e-4 |
| UNORM16 | 8.96e-5 | 6.79e-5 |

Those are finite, storage-only diagnostics on a few warmed graphs. They do
not certify full BSP correctness, performance, or success probability. No
16-bit production path was retained. A future candidate needs startup and
reactivation tests, endpoint/tail policies, unrounded residual checks,
FP64 complexity checks, ranking comparisons, verified solves and the full
memory cost of exceptions. The cold-start FP32 probe also had substantial
trajectory divergence in K=4 despite finishing; a warm-start result alone
would hide that sensitivity.

## Reproduction and artifacts

[METHODS.md](METHODS.md) describes timing boundaries, parameters and correctness
checks. `summary.json` contains each paired ratio and raw full-solve result.
`scaling_limits.json` contains both K capacity/format calculations, reproducible
with `python3 bench/scaling_limits.py`. The original numerical probe, source
hashes and CSV data are preserved under
[billion_variables_20261029](../billion_variables_20261029/REPORT.md).

The measurement archive contains valid plans, per-process memory time series,
VM counters, witnesses, action records, compiler/test logs and executable/source
hashes. Large generated CNFs and executables are excluded. Regenerate CNFs with
the reference CLI, for example `--generate 3 4.2 100000 --seed 1 --write-formula
FILE`, and replace absolute paths in a saved plan before rerunning. Build each
revision with its own Cargo target directory; the native harness is common to
all variants. Initial shared-target comparisons were discarded after executable
hashes exposed stale artifact reuse. No table uses those invalid comparisons.

Next useful work is to reduce the parallel mirrored q/norm/log arrays and
specialize fixed-K topology/state, followed by the compact FP32 aggregate
backend. Feistel generation and low precision remain separate, validated
steps. More aggressive scheduling and quantization require solution-quality
evidence before they can replace the current solver.
