# BSP with one billion variables: memory and feasibility audit

2026-10-29. Source audited: `aec3bed32d49bb1450132594c1df3f16ebc8449a`.
This is a design analysis plus a small numerical probe, not an implemented
billion-variable backend. Production solver code has not been changed.

**Agreed staging:** first measure and validate N=10^7 on the current M1 Pro
with 16 GiB unified memory; target N around 10^8 on a **single 24 or 40 GB GPU
in a later session**; consider N=10^9 after correctness and capacity are
established. Multi-GPU implementation is outside the current plan. The
distributed discussion below describes a possible later limit, not proposed
near-term work. k and alpha were not specified; all capacity examples therefore
remain explicitly conditional on the two workloads shown.

## 1. What I would build

For generated random k-SAT, a billion variables is a credible **capacity target**
after changing the representation substantially. Successful completion near the
SAT threshold, and acceptable runtime, remain experimental questions.

My preferred design combines:

1. A bidirectional implicit factor graph: variable degree boundaries plus an
   invertible pseudorandom permutation between variable stubs and clause slots.
2. A single persistent message payload per incidence, retaining frozen messages
   in place when the incidence becomes inactive.
3. A two-phase synchronous sweep using per-variable aggregates to reconstruct
   cavity messages inside each clause, avoiding an edge-sized cavity array.
4. Streamed complexity reductions and on-demand variable surveys.
5. One-byte clause records for fixed k=3 or k=4, packed variable state, bounded
   work queues, and scan/radix selection instead of full candidate/heap arrays.
6. Initially FP32 message storage. Then evaluate 16-bit encodings with endpoint
   handling and selective higher precision. Plain BF16/FP16 conversion already
   fails the small storage-only experiment below.
7. Separate SP, WalkSAT, verification, and checkpoint memory lifetimes.

For the first local milestone, retain **FP64** and the existing prefix/suffix
equations while removing redundant arrays and tightening state lifetimes.
Dropping `saved_eta` and `cavity_norm` alone represents 2.016 GB of payload
savings at N=10^7 in the 3-SAT example, without quantizing messages. Numerical
reduction order still needs validation. The FP32/aggregate/implicit-graph
design is the subsequent GPU target; not every change needs to land together.

### The N=10^8 single-GPU target

| Conditional workload at N=10^8 | 3-SAT alpha=4.2 | 4-SAT alpha=9.8 |
|---|---:|---:|
| Current GPU FP32 main buffers, **excluding** host mirror | About 58.3 GB | About 165.3 GB |
| Proposed implicit, single eta, FP64 aggregates, FP32 eta | 8.18 GB | 19.79 GB |
| Same proposal with a retained FP32 q buffer | 13.22 GB | 35.47 GB |
| Proposed implicit, 16-bit eta, FP64 aggregates | 5.66 GB | 11.95 GB |

The current GPU figures follow the buffer tables: approximately
36E+45N+20M+16*ceil(max(N,M)/256), before degree-bucket lists and tiny constants.
The complete existing host solver is extra. Current integer bounds do allow
these N=10^8 example graphs; memory layout is the immediate barrier there.

For 3-SAT, the 24/40 GB GPU target provides room to prioritize a simpler
FP32 implementation with a retained q buffer. On a 40 GB GPU, an explicit
fixed-k topology with compact state and FP32 eta/q is also a plausible
intermediate capacity target: roughly 23.9 GB under the same provisional
allowances, before headroom. This can validate the compact solver independently
of Feistel generation. For 4-SAT, keeping q in FP32 is much tighter, so the
aggregate route matters sooner. No 16-bit correctness assumption is required
to begin the 3-SAT N=10^8 GPU work.

Data: [memory_n100000000_3sat.json](memory_n100000000_3sat.json),
[memory_n100000000_4sat.json](memory_n100000000_4sat.json).

An explicit provisional ledger gives **81.8 GB with FP32** or **56.6 GB with
16-bit messages** for N=10^9, k=3, alpha=4.2. For k=4, alpha=9.8, the corresponding
figures are **197.9 GB** and **119.5 GB**. These are decimal GB of proposed payload,
including stated workspace allowances; they are not measured RSS, hardware
requirements, or evidence that the 16-bit scheme is accurate enough. Precision
exceptions, rank/select engineering, allocator behavior, and construction must
be measured. Leave substantial physical-memory headroom.

The biggest uncertainty after capacity is **random-access bandwidth**. The
permutation removes stored edges but does not make their messages local.

## 2. Scale and current hard limits

Let M=floor(alpha*N) and E=kM denote literal incidences, not two directed
message arrays. Every extra byte per incidence costs k*alpha GB at N=10^9.

| N=10^9 | k=3, alpha=4.2 | k=4, alpha=9.8 |
|---|---:|---:|
| Clauses M | 4.2 billion | 9.8 billion |
| Incidences E | 12.6 billion | 39.2 billion |
| One 64-bit incidence array | 100.8 GB | 313.6 GB |
| One 32-bit incidence array | 50.4 GB | 156.8 GB |
| One 16-bit incidence array | 25.2 GB | 78.4 GB |
| One bit per incidence | 1.575 GB | 4.900 GB |
| One byte per variable | 1 GB | 1 GB |
| One byte per clause | 4.2 GB | 9.8 GB |

The current implementation cannot construct these instances:

- `src/formula.rs:9`: clause/variable CSR offsets, incidence IDs, and clause IDs
  are `u32`. `Formula::random` explicitly rejects E >= u32::MAX.
  For 3-SAT at alpha=4.2 the incidence limit is around 341 million variables.
- `crates/bsp-metal/src/lib.rs:115` and
  `crates/bsp-cuda/src/lib.rs:169`: the current dispatch/indexing validation
  rejects N > floor(u32::MAX/32) = 134,217,727, as well as oversized edge arrays.
- Packed `u32` literals themselves still work at N=10^9: 30 variable bits plus
  one sign bit suffice. Widening every integer is unnecessary.
- At alpha=4.2 clause IDs still fit `u32`, narrowly; at alpha=9.8 they do not.
  Global incidence IDs require 34 and 36 bits respectively. Offset products,
  prefix sums, allocation lengths, launch arithmetic, counters and file formats
  must all be audited together. Removing a validation check is insufficient.

Use `u64` global arithmetic, with `u32` tile-local addresses where useful.
Fixed-width clauses give `edge=k*clause+slot`; widening an edge-to-clause table
is avoidable because that table need not exist.

## 3. Where current memory goes

For fixed k <= 64, the current scalar payload/reserved-capacity model is:

    Formula: 12E + 16M + 8N bytes
    Solver:  32E + 20.125M + 66.125N bytes
    Total:   44E + 36.125M + 74.125N bytes

Small vector headers, rounding of bitsets, maximum-clause scratch, unit queues,
solution/result copies, allocator slack and construction peaks are omitted.
The heap and candidate capacities are counted even when not all entries have
been touched; reserved address space and resident physical memory can differ.
Rust `Vec<bool>` uses a byte per bool here; it is not a packed bit vector.

| Current scalar arrays, hypothetical u32 layout | 3-SAT GB | 4-SAT GB |
|---|---:|---:|
| Formula topology | 226.400 | 635.200 |
| `eta`, `saved_eta`, `q`, `cavity_norm`: four f64 arrays | 403.200 | 1,254.400 |
| Solver variable arrays and reserved ID storage | 66.125 | 66.125 |
| Clause masks, counts, XORs and active bits | 84.525 | 197.225 |
| **Total** | **780.250** | **2,152.950** |

These extrapolations intentionally exceed the current integer limits. A direct
3-SAT conversion of incidence IDs, CSR offsets, and `free_xor` to u64 adds
about **92 GB**, bringing this scalar design to **872 GB** before overhead.
For 4-SAT further clause-ID and mask-offset widening is also necessary.

`src/execution.rs:786` adds independent multicore arrays for q, norms, logs,
the incidence-to-CSR-slot map, products and reduction terms:

    Parallel scratch = 32E + 24N + 8M bytes, on a 64-bit host.

That is another **460.8 GB** in the 3-SAT example or **1,356.8 GB** in 4-SAT,
before widening those indices. This backend would exceed 1.2 TB for 3-SAT
even in the impossible original u32 layout. Parallelism itself does not require
these duplicates; the current ownership/layout strategy creates them.

GPU selection first constructs the complete host `Solver`, then allocates
device topology, eta/saved eta, cavity/norm/prefix arrays, checkpoints and
observables (`src/execution.rs:250`, GPU `lengths` tables). FP32 GPU storage
therefore does not remove the host f64 arrays. Metal shared allocations also
consume physical unified memory alongside the separate host vectors.
`ExecutionReport.allocated_bytes` reports backend allocations/scratch, not total
process memory. It must not be used as a billion-variable capacity estimate.

**Local N=10^7 baseline actually measured.** Generation and one scalar FP64 SP
sweep completed twice on the current M1 Pro/16 GiB machine, for 42 million
clauses and 126 million incidences. The instrumented repeat observed a peak
physical footprint of **7.36 GB (6.85 GiB)** via Apple's `proc_pid_rusage`
RUSAGE_INFO_V4 lifetime-peak counter. The counter was sampled every 50 ms;
the reported number is the largest lifetime peak observed before child exit.
The sweep itself took **7.73 s** in that repeat; the earlier run took 9.16 s.
Generation and solver initialization are outside those sweep timings.

This is **one raw sweep**, with residual about 1, not convergence or a verified
SAT solve. System compression/paging was active, so these are diagnostic
timings, not a clean performance benchmark. `wait4` RSS reported only 2.52 GB
in the first run and 3.37 GB in the repeat, while physical footprint was 7.36 GB
in the instrumented repeat. RSS alone would therefore give a misleading
memory-saving claim. Apple describes physical-footprint accounting and its
API in [WWDC22: Profile and optimize your game's memory](https://developer.apple.com/videos/play/wwdc2022/10106/).

Data: [peak_footprint_n10000000.json](peak_footprint_n10000000.json), including
before/after system VM counters, and the earlier RSS-only
[peak_rss_n10000000.json](peak_rss_n10000000.json). Future A/B comparisons should
record physical footprint, exact array capacities, compression and swap deltas
alongside time, and exercise observables, releases and completion separately.

A small allocation sanity check used `wait4` peak RSS around a complete child
process generating a formula and executing one scalar/parallel SP sweep:

| Workload | Measured maximum RSS | Extra bytes reported by backend |
|---|---:|---:|
| N=100,000, CPU scalar | 89.9 MB | 0 |
| N=1,000,000, CPU scalar | 753.0 MB | 0 |
| N=1,000,000, CPU parallel, 4 threads | 1,181.7 MB | 460.8 MB |

Data: [peak_rss.json](peak_rss.json). These are decimal MB. Untouched capacity,
zero pages, allocator behavior and phase lifetimes explain why peak resident
bytes differ from the full payload/capacity ledger. This checks the scale of
the audit on the present host; it is not a billion-variable measurement and
does not include convergence observables, graph mutations or WalkSAT.

Construction and output are additional traps:

- `Formula::random` builds `Vec<Vec<i32>>` before CSR: 24M bytes in outer Vec
  elements, 4E in literal payload, billions of small allocations, and a 4N-byte
  Fisher-Yates pool. The outer Vec alone is 100.8 GB for 3-SAT.
- `Formula::new` also allocates a temporary signed-variable CSR cursor. Growing
  vectors can have capacity well above their final lengths. The input and final
  topology do not all peak simultaneously; a correct peak model must follow
  their lifetimes instead of adding every allocation indiscriminately.
- DIMACS input currently reads a whole String; export builds clauses and a
  whole output String. Neither path should be used for this scale.
- `solve()` clones solution and partial assignment vectors. The CLI serializes
  an entire result String. Python snapshot/list conversions can add further
  large copies. Stream binary state and emit summary JSON separately.

## 4. Feistel configuration model: the full design

### 4.1 Both adjacency directions without an incidence list

Give variable v degree d_v and cumulative stub boundaries:

    D_0=0, D_{v+1}=D_v+d_v, D_N=E.

Let pi be an invertible keyed permutation of exactly [0,E). Number clause
slots by e=k*a+j. Define:

    variable stub s -> clause slot e=pi(s)
    clause slot e -> stub s=pi_inverse(e)
    variable(e) = predecessor owner of s in the D boundaries
    sign(e) = independent keyed counter bit

Variable v enumerates s in [D_v,D_{v+1}), applying pi to each. Clause a enumerates
its k slots, applying pi_inverse and the owner query. Message storage is keyed
by clause slot, or by stub if benchmarks favor that orientation.

This eliminates `literals`, `var_edges`, `edge_clause`, clause offsets, and
mask offsets. A hash mapping a clause slot directly to a variable cannot
generally enumerate all clauses incident to a given variable efficiently;
the invertible stub matching is the essential extra property.

Feistel/FPE provides reversible permutations of a finite domain; cycle walking
can restrict a larger permutation to the desired domain. This is the relevant
construction principle, not a recommendation to deploy a particular encryption
standard. See [Bellare et al., Format-Preserving Encryption](https://www.cs.toronto.edu/~ristenpart/papers/fpe.html).

### 4.2 Domain, inverses and randomness

At E=12.6 billion, a 34-bit domain is sufficient. Applying a 64-bit permutation
and cycle-walking into [0,E) would have an expected expansion ratio around
1.46 billion, making it unusable. Use a near-sized domain: 2^34/E is about
1.36 here. Balanced Feistel may require an even bit width; in general choose
an appropriate balanced/unbalanced construction and account for its padding.

Never reduce `permute(s) % E`: modulo destroys bijectivity. Forward and inverse
cycle walking must use matching rules. The average iteration count of a
well-randomized near-sized domain is reasonable; individual walks have a tail,
so benchmark divergence and tail latency. A retry cap with a different fallback
mapping can silently break the inverse.

Do not choose a tiny number of rounds solely for speed. A small keyed family
is not a uniformly sampled element of all E! permutations. Test degree mixing,
short cycles, pair correlations, overlap distributions and solver outcomes
across independently keyed permutations. Statistical tests and sensitivity to
round count are required; bijectivity alone does not establish ensemble quality.

Use independent key domains for matching, signs, initial messages, tie-breaking
and restarts. Counter-based generators support deterministic random access
without an RNG state per node/edge; see [Salmon et al., Random123](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
Their output is useful for signs and random values, but a truncated/modulo
counter output is not automatically a permutation of [0,E).

### 4.3 Degree storage and stub ownership

The permutation is O(1) metadata, but random variable degrees are not free.
Both D_v lookup and the inverse ownership query must be fast.

| Degree-boundary representation, 3-SAT example | Approximate payload | Tradeoff |
|---|---:|---|
| u64 boundaries | 8 GB | Simple; binary-search ownership is too costly per incidence |
| Byte degrees + sampled u64 prefixes | About 1 GB plus ownership index | Sequential traversal is cheap; random ownership needs indexing |
| Unary boundaries with rank/select | (E+N)/8 = 1.7 GB before support | Direct representation of zero degrees and stub owners |
| Elias-Fano boundaries | Roughly 0.7 GB before extra query support | Compact; predecessor/select engineering matters |
| Regular or almost-regular degrees | O(1) metadata | Changes the random-SAT ensemble |
| Regenerated hierarchical degree counts | Small retained hierarchy | Potentially many random operations per ownership query |

The Elias-Fano figure follows its monotone-sequence space bound; it is not a
benchmark or an all-inclusive index estimate. At this scale a slower compact
predecessor can cost more than the saved memory is worth.
[Vigna, Quasi-Succinct Indices](https://vigna.di.unimi.it/ftp/papers/QuasiSuccinctIndices.pdf)
describes the representation and query tradeoffs.

### 4.4 Preserve the intended random ensemble

The existing generator uses distinct variables in each clause. A configuration
model with fixed degrees preserves that degree sequence and can contain the
same variable twice in a clause. Regular degrees, independent Poisson degrees,
and the current clause-wise sampling are different finite-instance ensembles.
[Chodrow, Configuration Models of Random Hypergraphs](https://arxiv.org/abs/1902.09302)
is relevant to these distinctions.

One principled starting ensemble is to sample multinomial occupancy of E stubs
over N variables, then use a uniform matching. Independent Poisson degrees
conditioned on total E give that occupancy law; drawing independent Poisson
degrees and repairing the total arbitrarily does not. Hierarchical binomial
splitting can generate conditioned counts without retaining E sampled labels.
An ideal uniform permutation then gives the with-replacement slot ensemble.
The practical keyed permutation approximates that matching and needs validation.

With-replacement slots have approximately alpha*choose(k,2) repeated-variable
pairs: **12.6** for the 3-SAT example and **58.8** for 4-SAT. Their fraction
vanishes with N, but they do not disappear from an actual instance. Opposite
signs make a tautology; equal signs require literal deduplication. Retain sparse
exception records and canonical slot IDs, or use a properly specified sampler
of simple clauses. Naively resampling the entire graph until no repeats occur
is expensive even though the expected defect count is only O(1).

A sparse endpoint-swap overlay can preserve invertibility while repairing
defects; it does not, without further argument, produce an exactly uniform
sample of simple random k-SAT. State the ensemble actually used. Do not use a
regular-degree shortcut and compare its threshold directly to ordinary random
k-SAT. Verify the final witness against the same seeded instance and its exact
exception policy.

### 4.5 Scope of applicability

This approach replaces topology for **new generated instances**. It does not
encode an arbitrary DIMACS formula in a seed. For existing formulas use fixed-k
specialization, one retained orientation plus a compressed inverse, delta-coded
sorted adjacency, succinct offsets and local IDs. Random neighbor lists have
much less compressible locality than structured industrial formulas.

Without implicit generation, fixed-k specialization can already remove
`edge_clause`, clause offsets, mask offsets and immutable initial masks. That
saves 117.6 GB in the hypothetical 3-SAT topology above, before addressing
index widths. It is useful, but much larger than an implicit representation.

## 5. Remove edge arrays before reducing precision

### 5.1 Keep frozen messages in the same payload

Define an effective read:

    effective_eta(e) = payload[e] if clause(e) is active and variable(e) is free
                       0 otherwise.

Update only active/free incidences. Deactivation or fixing changes the state
gate and leaves the payload untouched; reactivation exposes its retained value.
This reproduces the purpose of `saved_eta` without a second dense array.
Every cavity product, update, initialization, snapshot and observable must use
the effective-state rule consistently. Merely deleting `saved_eta` while still
zeroing eta on deactivation loses backtracking information.

For the 3-SAT example this saves 100.8 GB at f64, 50.4 GB at FP32, or 25.2 GB at
16-bit storage. This is a representation change that can preserve the existing
freeze/restore semantics. APIs exposing zero-filled inactive messages should
materialize only requested slices.

### 5.2 Stream normalization terms instead of storing `cavity_norm`

The current observable is a sum of log cavity norms plus clause log terms,
minus variable terms. Accumulate each log norm during cavity evaluation and
reduce it in blocks; only the sum is needed by the solver. Remove 8E bytes.
No need to retain `ln(norm)` either. FP64 block reductions with compensated or
pairwise summation reduce cancellation error. Different summation order may
change floating-point decisions; validate against the current implementation.

### 5.3 Reconstruct q inside each clause: no q array, no new-eta array

Write g_e=1-effective_eta(e). For each free variable, compute the full product
of g_e over each sign group. For incidence e, let A be its same-sign product
excluding e, and B its opposite-sign product. The code's cavity equation is:

    Z = B + (1-B)*A
    q_e = (1-B)*A / Z
    eta_new(a->v) = product of q on the other free slots of clause a.

Use two phases:

1. Read the complete old message state and compute **per-variable aggregates**.
   Do not modify message payloads in this phase.
2. For each active clause, load all its old messages, reconstruct its local q
   values using the frozen aggregates, compute all outgoing values, compare
   them to their old values, and overwrite that clause's payloads.

This can remain synchronous and work in place: a clause reconstructs q using
only the frozen variable aggregates and its **own** old incidence messages.
Another clause never needs those overwritten individual old messages. Load
the entire local clause input before overwriting any slot. Damping and the
undamped residual are still available from that local old input. Disjoint
clause ownership also permits parallel writes without duplicate q arrays.

The numerical issue is leave-one-out products. Naive division by g_e fails at
eta=1, and a full product may underflow before division. Retain, per sign:

- the count of exact zero factors;
- a product of nonzero factors, or a sum of `log1p(-eta)` over nonzero factors.

With zero count z and nonzero product P, the exclusion is 0 for two or more
zeros; P for the unique excluded zero; 0 when a different zero remains; and
P/g_e when z=0. Log aggregates avoid underflow of P; exclude via subtracting
the edge's log factor, and evaluate normalization ratios with stable log-sum
formulas when needed. Use `-expm1(log_B)` for 1-B near B=1. Large cancellation
in log subtraction and near-zero normalizers still require local higher
precision/recomputation checks. Finite-precision equivalence is not automatic.

The original prefix/suffix method is robust around exact zeros and should be
the reference and a local fallback for problematic variables. A blocked
prefix/suffix variant can trade a bounded scratch buffer for recomputation.
Simply rescanning a variable's entire neighborhood for every cavity costs
roughly an extra factor of average degree and many more permutation calls.

The aggregate design replaces O(E) cavity state with O(N) state. Its arithmetic,
transcendental and gather costs must be compared to retaining a 16/32-bit q
array. The smallest layout may lose badly on throughput.

### 5.4 Surveys and ranking

Surveys are refreshed at a fixed point, not used as persistent edge state.
The current three-f64 survey array costs 24 GB. Reconstruct a survey from the
two variable aggregates when needed. For fixing, retain only the score and
preferred assignment; after fixing retain the frozen release score. If an API
needs a full survey snapshot, stream or materialize a selected slice.

Preserve the actual score `1-min(S_true,S_false)`, the True/False tie rule, and
variable-ID tie breaks. Replacing it with absolute polarization changes the
policy. In this implementation release scores stay frozen while fixed; unit
assignments are not eligible for release (`docs/theory/backtracking.rst`).

The indexed decision heap uses up to 8N bytes for IDs and positions, and the
candidate Vec reserves 4N. A full scan with multi-pass radix selection can find
the exact top/bottom batch without an N-element ID array. Store the selected
batch only, resolve threshold ties by variable ID, and use per-tile counts to
parallelize. It costs extra O(N) passes, which may be reasonable next to O(E)
SP sweeps. Bucketed quantized scores are cheaper but change ranking unless
ambiguous buckets are refined using adequate precision.

## 6. Quantization: storage and arithmetic are separate decisions

No finite floating-point format gives generally exact repeated products of
probabilities. The relevant requirements are error in the cavity ratios,
preservation of important near-zero and near-one values, and stability of the
resulting fixed points and BSP decisions.

Store compressed messages, decode to FP32 or FP64 for arithmetic, and quantize
only when writing the next persistent state. Do not multiply long products in
BF16 or FP16 just because the stored inputs use that format. Tensor-core matrix
throughput is not a throughput prediction for this irregular product/gather
kernel. Official [CUDA floating-point type properties](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
give the underlying exponent/mantissa ranges; the error and endpoint estimates
below follow from their representable spacings and round-to-nearest.

If exact multiplication of **quantized** inputs is useful, two UNORM16 integer
codes multiply exactly in u32, and three multiply exactly in u64. This covers
the two or three factors in a fixed 3-SAT/4-SAT clause update when q itself is
quantized. Normalize once by 65535^2 or 65535^3, instead of rounding after
every multiplication. Products of two FP16 significands also fit FP32's
significand, within FP32's exponent range. Neither observation recovers input
quantization error, makes division exact, or solves the much longer variable
products. FP32/FP64 arithmetic plus stable normalization remains necessary.

| Stored format | Bytes/edge | Approximate worst absolute rounding error on [0,1] | Main issue |
|---|---:|---:|---|
| FP64 | 8 | 5.55e-17 | Capacity |
| FP32 | 4 | 2.98e-8 | Products/normalizers can still be ill-conditioned |
| BF16 | 2 | 1.95e-3 | Only 7 explicit fraction bits; coarse near one |
| IEEE FP16 | 2 | 2.44e-4 | Small positive values underflow; endpoint rounding |
| UNORM16, round(65535*x)/65535 | 2 | 7.63e-6 | Small tails round to zero; values near one round to one |
| Packed UNORM12 | 1.5 | 1.22e-4 | Tail loss and packing overhead |
| UNORM8 | 1 | 1.96e-3 | Too coarse as an initial deployment candidate |
| Custom log/logit 16-bit | 2 | Encoding dependent | Transcendental/decode cost; needs measured distributions |

BF16 has an FP32-like exponent range, but that does not make it precise for
probabilities. Its predecessor of one is 1-2^-8, so an interior value above
1-2^-9 can round to exactly one. FP16's predecessor is 1-2^-11 and its rounding
threshold is 1-2^-12. A newly created one makes `1-eta` exactly zero. Multiple
such factors can make a cavity normalization zero even if the unrounded
problem had a positive denominator. Subtracting after decoding cannot recover
the information lost in storage.

FP16's minimum positive subnormal is 2^-24; values below roughly 2^-25 round
to zero. Hardware flush-to-zero modes can lose still more. The scalar solver
currently cuts raw messages only below 1e-16, so ordinary FP16 moves the lower
tail cutoff upward by many orders of magnitude. UNORM16 is much more accurate
in absolute error across most of [0,1], but deletes all positive values below
about 7.6e-6 without special handling. This is a hypothesis to test, not an
equivalence to the current cutoff.

At N=10^9, events that look rare on a small graph occur many times. For uniform
initial eta in [0,1), the expected newly rounded ones are about E/512 for BF16,
E/4096 for FP16 and E/131070 for UNORM16: approximately **24.6 million**,
**3.1 million**, and **96 thousand** respectively in the 3-SAT example.
These counts alone do not determine the number of invalid normalizers, but
explain why endpoint handling cannot be postponed.

Candidate policies, in order of engineering interest:

1. **FP32 state, FP64 aggregates/diagnostics**, with local checks for sensitive
   normalizers. This establishes a compact reference before 16-bit work.
2. **UNORM16 with selective escapes**, preserving sensitive tails/endpoints in
   FP32 or FP64. Reserve codes or use sparse metadata; measure the actual escape
   fraction throughout startup, decimation, releases and final SP relaxation.
3. **FP16 storage with FP32/FP64 arithmetic** and endpoint/tail handling. It
   preserves small magnitudes better than UNORM16, but has less central absolute
   precision. BF16 should be compared empirically rather than assumed superior.
4. **Custom unsigned floating encoding**: reclaim the unused sign and exponent
   range above one for tail range/fraction precision. A 16-bit unsigned format
   with six exponent bits and ten fraction bits is one possible design; define
   exact endpoint/zero codes and a lower cutoff explicitly. Native arithmetic
   support is not required for storage, but decode cost must be measured.
5. **Log-complement or logit companding**, if message histograms show both tails
   matter. A log encoding for eta alone still represents its complement poorly
   near one. A symmetric logit encoding allocates precision to both tails but
   has larger central absolute error than UNORM16. It also needs dedicated
   endpoint codes and stable evaluation.
6. **12-bit or 8-bit adaptive blocks** only after numerical sensitivity is known.
   Block floating-point is unattractive when a block mixes zeros, values near
   one, and tiny values. Dictionary/entropy coding may help cold state, while
   variable-length random access can destroy the SP kernel's throughput.

Do not clamp every interior value away from endpoints and call it equivalent:
this inserts a lower floor and upper ceiling into the equations. Stochastic
rounding can reduce systematic rounding bias in expectation but adds algorithmic
noise and still needs a policy for exact ones. Neither is a correctness proof.

With a sparse precision side table, account for indices and metadata. A simple
u64-ID/f32-value exception list costs at least 12*p*E bytes, before alignment,
lookups or synchronization. Even a dense one-bit exception mask costs 1.575 GB
in the 3-SAT case. A per-edge byte tag would cost 12.6 GB. Promoting whole blocks
is simpler but can magnify the exceptional fraction.

Keep convergence checks honest: evaluate the **undamped raw map residual against
the decoded current state**, before quantizing its output. An unchanged stored
code can coexist with a residual above epsilon. Report quantization stalling
instead of declaring convergence. Reevaluate critical complexity signs in
FP64 on the actual rounded state; this checks diagnostic evaluation, not the
distance to an unquantized fixed point.

In the current GPU code the FP32 observable guard is
`32*f32::EPSILON*E`, about **48,065** at E=12.6 billion. Near zero total
complexity this can trigger full host FP64 evaluation, requiring the retained
host arrays. A compact backend needs streamed high-precision validation and
local numerical retries; an emergency full f64 solver copy defeats its budget.

## 7. Numerical probe performed for this audit

`bench/prototypes/message_storage.rs` uses the actual public scalar `Solver`
on N=10,000, k=3/alpha=4.2 and k=4/alpha=9.8, formula seeds 1,2,3 and solver
seeds 1001,1002,1003. It quantizes eta before starting and after every sweep,
then calls `set_messages` before the next sweep. All cavity and observable
arithmetic remains FP64; q is not quantized. Each trial requests 80 synchronous
sweeps, with damping zero and no variable fixing or release. This isolates
message-storage error; it is deliberately not an implementation of a compressed
solver and does not measure its memory, speed or full-solve success probability.

The cold-start trials use the current uniform initialization. Observations:

| Plain stored format | 3-SAT trials finishing all 80 sweeps | 4-SAT trials finishing all 80 sweeps |
|---|---:|---:|
| FP64 reference | 3/3 | 3/3 |
| FP32 | 3/3 | 3/3 |
| BF16 | 0/3 | 0/3 |
| FP16 | 0/3 | 0/3 |
| UNORM16 | 3/3 | 0/3 |
| UNORM12 | 0/3 | 0/3 |
| UNORM8 | 0/3 | 0/3 |

Failures are `NumericalFailure`, within the first four attempted sweeps.
BF16 introduces 747 new ones across the three initial 3-SAT graphs and 2,287
across 4-SAT; FP16 introduces 93 and 307 respectively. For comparison the
unmodified FP64 runs complete every requested sweep.

Two additional policies were probed for BF16, FP16 and UNORM16:

- `_open`: map strictly interior inputs to representable strictly interior
  outputs. Exact input zeros and ones remain exact. This deliberately changes
  the floor/ceiling behavior.
- `_escape`: whenever ordinary quantization would create a new zero or one,
  retain that value **exactly in FP64** for this diagnostic. This is an optimistic
  numerical experiment, not a packed exception implementation or evidence that
  FP32 exceptions suffice.

All six guarded policies complete all six cold-start cases. Nevertheless:

- Some transient trajectory differences approach one, even for plain FP32
  in 4-SAT. A small local storage error need not imply a small transient error
  under this initialization.
- For 3-SAT, the largest final edge error across seeds is about 1.79e-4 for
  FP32, 7.84e-3 for plain UNORM16 and 1.64e-3 for UNORM16 with FP64 escapes.
- UNORM16 escapes peak at **10.5%** of incidences in 3-SAT and **38.3%** in
  4-SAT. FP16 escape peaks are about **3.0%** and **20.8%**; BF16 peaks are
  about **0.95%** and **1.22%**. A small sparse escape table cannot be assumed.
- Completing 80 sweeps is not a claim of convergence. The largest 3-SAT FP64
  residual at sweep 80 is about 0.0149, above the default epsilon=0.01.

Cold-start data: [message_storage.csv](message_storage.csv).
Warm-start trials and exact reproduction metadata are recorded alongside it.
These experiments motivate starting in higher precision and assessing a later
switch, while preserving a restart/release policy for newly sensitive messages.
They do not certify any quantization for billion-variable BSP.

The warm-start experiment performs 200 FP64 sweeps first, then runs the same
80-sweep quantization trial. All 13 formats/policies complete all six cases.
For plain encodings, the worst absolute message error **over the entire
80-sweep trajectory and all three seeds** is:

| Stored format after FP64 warm-up | 3-SAT | 4-SAT |
|---|---:|---:|
| FP32 | 2.45e-7 | 9.22e-8 |
| BF16 | 2.15e-2 | 5.85e-3 |
| FP16 | 1.99e-3 | 9.40e-4 |
| UNORM16 | 8.96e-5 | 6.79e-5 |
| UNORM12 | 1.80e-3 | 1.11e-3 |
| UNORM8 | 3.39e-2 | 1.90e-2 |

UNORM16's guarded escape fraction falls below 0.028% in these warm 3-SAT
trials and is zero in the warm 4-SAT trials. This strengthens the case for
phase-dependent encoding; it does not show that every BSP release remains in
the same well-conditioned regime. The warm-up here is FP64, so this is not
evidence that an FP32 warm-up gives identical results. Billion-scale FP32
initialization should also sample directly below one or explicitly protect
rounding into one; the existing GPU initializer already caps rounded random
initial messages below one.

Data: [message_storage_warm.csv](message_storage_warm.csv). Across cold and
warm conditions there are 156 trials, comprising six graphs, 13 storage policies
and two initialization conditions; these are not 156 independent formulas.

## 8. Compact clause and variable state

For fixed small k, replace redundant masks/counts/XORs with a small record:

    free_mask: k bits
    true_count: ceil(log2(k+1)) bits

For k=3 this needs five bits; for k=4 seven bits. Both fit in one byte.
`free_count=popcount(free_mask)`, the last free slot is found with ctz, and its
global edge is k*clause+slot. `active` is equivalent to true_count==0; an
additional active bitmap can accelerate scans. Thus the current mutable u64
mask, three u32 arrays, and active bits can become **1.125M bytes**, instead
of **20.125M**, without losing clause state. No immutable full initial mask
or per-clause mask offset is needed. Wider/mixed clauses need separate layouts;
do not impose this packing on general CNF.

For variables, free/False/True needs two bits and a unit-forced flag one more.
Keep a separate free bitmap if iteration benefits justify it. This is roughly
0.5 GB for a billion variables with simple packed layouts. A complete witness
needs only one bit/variable: **125 MB**. Partial assignment and release eligibility
are more than a single assignment bit; do not discard the forced flag.

Active degree can be accumulated during the variable pass, so a persistent
u32 degree cache is optional. Small-degree encodings and byte zero counts need
checked overflow plus a wide exceptional path, not saturating arithmetic.
Random degree means are small in the example regimes, but a general formula
API must not silently assume a maximum degree of 255.

Use a pending-clause bitset to deduplicate unit work and process bounded queues
in tiles. The current unbounded Vec can contain repeated queued clauses. A
ring-buffer cap alone is invalid if it drops implications: spill IDs, retain a
pending bitmap, or rescan affected tiles until all work is processed.

## 9. A concrete proposed memory ledger

The reproducible estimator is `bench/memory_budget.py`; JSON payloads are
[memory_3sat.json](memory_3sat.json) and [memory_4sat.json](memory_4sat.json).

For 3-SAT, alpha=4.2, with **one** 16-bit incidence payload:

| Proposed component | GB |
|---|---:|
| Message payload, including retained frozen messages | 25.200 |
| Two FP64 aggregate values per variable | 16.000 |
| Two byte zero counts per variable | 2.000 |
| FP32 frozen decision scores | 4.000 |
| Packed variable state plus scan bitset | 0.500 |
| One-byte clause records plus active bitset | 4.725 |
| Unary degree boundaries, before rank/select support | 1.700 |
| Rank/select and small topology-exception allowance | 0.500 |
| Bounded queues, selection and tile-workspace allowance | 2.000 |
| **Provisional total** | **56.625** |

The 0.5N and 2N allowances are explicit **design budgets**, not measured data
structure sizes or guaranteed bounds for arbitrary workloads. Precision escapes
are additional, potentially substantial. No heap or N-element candidate list
is included. Global edge/clause IDs are computed in u64, not stored densely.
The byte zero counters require an exceptional wide-degree path.

| Proposed single-payload profile | 3-SAT GB | 4-SAT GB |
|---|---:|---:|
| FP32 messages, FP64 aggregates | 81.825 | 197.850 |
| 16-bit messages, FP64 aggregates | 56.625 | 119.450 |
| 8-bit messages, FP64 aggregates | 44.025 | 80.250 |

The 8-bit row is a capacity calculation only; plain UNORM8 failed the probe.
Replacing aggregate FP64 with FP32 would save another 8 GB per billion
variables, but is not included in the recommended initial ledger. Succinct
degree boundaries could reduce the unary cost, particularly at higher k*alpha.
Every additional q buffer costs 25.2/78.4 GB at 16 bits, or 50.4/156.8 GB at FP32.
That buffer may still be worth buying in throughput if it avoids expensive
logarithms or gathers.

For the eventual **N=10^9** implementation I would budget at least a 128 GiB
RAM class for the 3-SAT FP32 design and 256 GiB for 4-SAT, then measure actual
peaks; these are not requirements for the agreed local N=10^7 milestone.
The 16-bit route may allow smaller memory, but is not yet a sound
basis for a capacity commitment. Decimal GB differ from GiB; 56.625 GB is
about 52.7 GiB, before the omitted exceptional state and headroom.

A 16 GiB machine cannot hold even the single dense 16-bit message payload for
the 3-SAT example. Feistel topology alone does not change that fact. Achieving
that budget would require much more aggressive unvalidated compression,
external memory, or changing the computational problem.

## 10. Sparse state, compression and reversible simplification

**Exact zero compression.** If many payloads are exactly zero, a block bitmap
plus packed nonzero values can beat dense storage. A simple 16-bit payload costs
approximately 0.125 + 2*f bytes/incidence, where f is the nonzero fraction,
before block directories and allocation slack. The nominal break-even point is
f<0.9375. Random access, inserting new nonzeros and GPU execution may dominate
the savings. Count **retained frozen** values as well as active messages;
zero-valued effective reads of inactive edges do not mean their payloads can
be discarded. Small values may be dropped only under an explicit tested
quantization policy.

**Hot/cold storage.** Active payloads can remain in RAM/HBM and frozen payloads
move to compressed RAM or SSD. Release then restores affected blocks. This
can reduce late-run working memory, but startup still has the full active
graph, and r=0.9 may repeatedly reactivate cold blocks. Measure reload volume
per release and tail latency. A bounded compressed-block cache needs a correct
backing store, not eviction that loses saved state.

**Residual compaction.** Rebuilding active adjacency can shrink the working set,
but a satisfied clause can become active again after releasing a decision.
Deleting it irreversibly, renumbering messages without an inverse map, or
dropping frozen values changes BSP. Prefer a stable implicit original topology
and optional active views; account for temporary overlap when building a new
view. Compacting the final residual after SP ends is much simpler.

**Per-thread scratch.** Allocate scratch per worker or bounded tile, not per
variable/edge. Fixed k=3/4 clause products belong in registers or small local
arrays. GPU checkpoint arrays, generic prefix buffers and width buckets should
be optional or specialized. A whole-message-array retry checkpoint costs
another 25.2 GB even with 16-bit 3-SAT storage. Retry individual tiles with
transactional writes or retain checkpoints on storage, with a documented
recovery protocol.

**Topology compression for arbitrary input.** Packed 34-bit incidence IDs use
about 53.55 GB for the 3-SAT inverse list instead of 100.8 GB in u64. Five-byte
IDs use 63 GB and may be easier to access. A packed literal needs 31 bits
instead of 32, so squeezing that final bit alone saves only about 1.575 GB.
Compress large redundant arrays before paying unaligned-access costs to save
one bit. Delta-coded sorted clause IDs and succinct CSR offsets are possible
when topology must be retained; measure decoding and inverse lookup costs.

**Process and API discipline.** One solver per large instance; share immutable
topology between trials and run restarts sequentially. Export no full snapshot
implicitly. Use a binary packed witness, summary metrics, checksummed streaming
checkpoints and indexed slice access. Memory mapping reserves virtual address
space and enables persistence; it does not remove the resident working set.
OS swap and transparent compression are not substitutes for an access-aware
layout. NUMA placement, first-touch allocation, large pages and alignment can
reduce remote accesses and TLB cost, but do not reduce the payload byte count.

## 11. WalkSAT completion and witness verification

Solving means obtaining and checking a full assignment, not merely iterating
SP on a huge graph. `src/walksat.rs:33` currently retains every SP allocation
while allocating:

    values: N bytes
    true counts: 4M
    true-variable XORs: 4M
    break counts: 4N
    unsatisfied clause IDs and positions: up to 8M

That is **5N+16M** additional bytes: **72.2 GB** in 3-SAT and **161.8 GB** in
4-SAT, before widening clause IDs and capacity overhead. The final residual
may be small, but these allocations are indexed by the original M and N.

At handoff:

1. Retain the seed/topology descriptor, exact fixed assignments, forced flags
   where needed, and an ID map for any compact residual.
2. Drop SP payloads, aggregates, q buffers, survey arrays, GPU state and scratch
   before allocating a full local-search state.
3. Either build only the residual on the available memory budget, or adapt
   WalkSAT to regenerate incidence neighborhoods. Counts up to k can use a
   byte; XOR can store a local true-slot index for fixed k, or recompute the
   sole true literal by scanning k slots. Keep break counts and unsatisfied
   membership/index structures only for the actual residual where practical.
4. Keep all BSP-fixed variables immutable during completion, as today.
5. Verify the final packed assignment by streaming **all original clauses**
   from the seed and exception policy, including clauses removed during BSP.

Implicit topology helps WalkSAT too, but random neighbor regeneration per flip
has a cost. Retaining a compact explicit final residual may be faster and fit
comfortably once SP state is freed. Default max_flips=10 million is only 0.01
flips per original variable at N=10^9; choose a workload-dependent residual
budget rather than assuming the existing absolute default scales.

## 12. Throughput can defeat a memory-successful design

### Single machine

One minimum logical 16-bit eta read and write is 50.4 GB per full 3-SAT sweep.
The aggregate design reads eta in the variable phase, then reads and writes it
in the clause phase: **at least 75.6 GB of logical eta traffic**, plus aggregates,
state, indices, rank/select and cache effects. This is not its physical traffic.

If clause-major eta is gathered in pseudorandom order from a much larger array
than cache, a 2-byte load can fetch a 64-byte CPU cache line with little reuse.
One such pass over E incidences can approach **806 GB of line traffic**.
The clause phase then gathers variable aggregates, potentially incurring a
similarly large amplification. Actual locality and GPU transaction sizes must
be measured. Advertising the 25.2 GB eta payload as sweep bandwidth is wrong.

Benchmark both edge orientations, sorting/bucketing work by memory tile,
partitioned caches, and a retained q buffer. Explicitly route values between
variable-major and clause-major phases if extra sequential passes beat random
gathers. A cached permutation/index can improve speed at a known memory price.
Restricting the random matching to local blocks without sampling the correct
cross-block counts changes the ensemble; it is not a free locality optimization.

Measure effective edges/s, permutation/inverse calls/s, rank/select latency,
memory transactions, bytes per sweep, and the number of complete sweeps per
BSP action. A billion-variable capacity test with one sweep is insufficient.

### Multi-GPU or distributed memory

Shard by stable global ID, retain local u32 offsets, and partition payloads and
node state rather than replicating a complete solver per device. Global
residual maxima and score histograms are small; edge-message exchange is large.
For a basic variable-owner/clause-owner implementation with b-byte messages in
both directions, a random partition into P ranks communicates approximately:

    2*b*E*(1-1/P) bytes per sweep, summed over ranks,

before metadata, buffering and protocol overhead. With b=2 this approaches
**50.4 GB** per 3-SAT sweep or **156.8 GB** per 4-SAT sweep. Pack routes and
stable IDs once or regenerate them; do not attach a wide global ID to every
2-byte message if the schedule can imply the identity. Route construction and
communication buffers belong in the peak budget.

Random factor graphs have little geometric locality, so graph partitioning
cannot generally produce the tiny boundaries seen in spatial meshes. Feistel
can make **generation** communication-free; it does not make SP propagation
communication-free. The original [BSP paper](https://www.nature.com/articles/ncomms12996)
also explicitly notes inter-core bandwidth concerns for expander-like factor
graphs. Its reported finite-size performance is not a guarantee for the
current solver or a new quantized/distributed implementation.

If a single shared-memory machine fits the validated layout, evaluate it
before committing to a distributed design. NUMA is still a real communication
cost. GPUs need tiled buffers, wide address arithmetic, an adequate interconnect
when sharded, and removal of host mirrors/full snapshots. Oversubscribed GPU
unified memory with random gathers can thrash; benchmark it as an external-memory
algorithm instead of assuming it extends HBM capacity cheaply.

### External memory

An SSD-backed solver needs explicit tiled streaming or permutation routing.
Memory-mapping messages and issuing billions of random small reads is likely
to be dominated by page fetches and IOPS. A sequential read/write lower bound
does not describe those accesses. Sort or bucket communication records with
bounded RAM, keep temporary files and double-buffering in the disk budget, and
measure complete iterations. This is an option when RAM is fixed, with a much
larger implementation and runtime risk.

### Total work and stopping rules

Use the actual number of SP sweeps S:

    time approximately sum over sweeps of
      max(memory service, permutation/decoding, arithmetic, communication)
      + propagation/ranking/WalkSAT/checkpoint time.

At the default batch_fraction=0.00125, the initial forward batch is 1.25 million
variables. The code's r=0.9 controls the ratio of backward to forward **actions**;
it is not an independent probability, and release batch sizes include pending
unit assignments. Fixed points can therefore be solved many times before the
residual shrinks. Reducing r, increasing batch size or loosening epsilon can
reduce runtime but also change solve probability. Study these changes separately
from byte-preserving layout changes. Full-scan work may continue to depend on
the original graph even when only a small active residual remains.

## 13. A more aggressive research option: node-field fixed-point equations

The following is an algebraic observation derived here from the scalar SP
equations, **not a tested update algorithm or a claim of novelty**. It concerns
strictly interior, uncut SP fixed points. It may be useful for an SP-only study
or a solver allowed to change relaxation and restart semantics.

For a free variable i let P_i^+,P_i^- be the full sign products. For an incidence
of sign s, define B=P_i^opposite, C=(1-B)*P_i^same and
Z_i=P_i^+ + P_i^- - P_i^+*P_i^-. Let Q_a be the product of all q on clause a.
Using P_i^same=(1-eta_ai)*A_i->a, the cavity equation becomes:

    q_i->a = C / (Z_i - B*eta_ai).

At a fixed point eta_ai*q_i->a=Q_a. Eliminating eta gives:

    q_i->a   = (C + Q_a*B) / Z_i
    eta_ai   = Q_a*Z_i / (C + Q_a*B).

Thus an interior fixed point can be parameterized by **2N+M node scalars**
subject to consistency of P with products of reconstructed 1-eta and Q with
products of reconstructed q. This is smaller than E edge scalars in the
regimes considered. For 3-SAT, one FP32 Q per clause plus two FP64 P values per
variable is 32.8 GB; a 16-bit Q plus FP64 P values is 24.4 GB, before all other
state, stability measures and work buffers.

There are substantial unresolved issues:

- The identities hold at a consistent fixed point. A simultaneous/damped node
  update has a different iteration Jacobian and may converge to a different
  branch or fail. It does not reproduce the current synchronous sweeps.
- C=0, Q=0, eta=1 and zero normalizers create degenerate cases. Unit clauses
  and the solver's explicit 1e-16 cutoff need separate analysis.
- Numerically tiny Q and products require stable encodings and reductions.
- After decimation, retaining node fields does not retain the last frozen
  value of **each** inactive edge. Exact current freeze/restore semantics can
  reintroduce edge-sized history; reinitializing restored messages changes BSP.

I would investigate this only after the single-payload aggregate backend works.
It is a plausible route to further savings, particularly if the goal is SP
fixed points, but it is excluded from all capacity recommendations above.

Other algorithm-changing options include asynchronous or residual-driven
updates, bounded release windows, irreversible commitments and reinforcement.
They can reduce temporary state or work, but require full outcome comparisons.
Population dynamics estimates distributions in an ensemble; it does not solve
and verify a particular one-billion-variable instance.

## 14. Implementation order and evidence needed

| Priority | Work | Required evidence |
|---|---|---|
| 1 | Accurate allocation/peak reporting and overflow audit | Constructor, SP, snapshots, WalkSAT and result peaks separated |
| 2 | Fixed-k clause records, streamed norms/surveys, one frozen/current payload | Paired mutation/release/propagation and full-solve checks |
| 3 | Parallel ownership without mirrored q/norm/log buffers | Same selected precision and policy; measured memory and throughput |
| 4 | FP32 aggregate sweep with stable local fallbacks | Per-sweep comparison, zero/near-one stress, convergence and solve outcomes |
| 5 | Explicit and implicit graph implementations of the same seeded topology | Exhaustive small-domain pi/inverse checks, matching adjacency and witnesses |
| 6 | Degree/ensemble validation and succinct ownership index | Histograms, correlations, short loops, exceptional-clause policy |
| 7 | Warm-start/adaptive 16-bit storage | Error tails, escapes, false stagnation, rankings, complexity, full BSP success |
| 8 | Residual-only WalkSAT and streamed verification/checkpointing | Correct final witness and restart consistency within peak budget |
| 9 | Scale 10^6 -> 10^7 -> 10^8 -> 10^9 | Measured bytes/N, active fractions, sweeps, bandwidth and total-time projection |
| 10 | Distributed/external-memory or node-field variants if necessary | End-to-end evidence beyond a memory-fit demonstration |

For precision evaluation, pair identical materialized small instances and
initial states first. Use both easy and near-threshold densities, several
independent seeds, unit propagation, high degrees, empty/short residuals,
reactivations, repeated freeze/restore, and near-zero complexity. Compare the
unrounded residual on the rounded state, message-error distribution, score
ordering near the batch cutoff, complexity per variable and its total sign,
SP convergence, contradictions, verified solves, runtime and peak memory.
Success rates need uncertainty intervals and enough instances for the effect
being claimed; the three-seed probe is a diagnostic, not that validation.

Build the implicit option behind an explicit workload declaration such as
`generated configuration ensemble`; preserve the current explicit Formula
path for ordinary CNF. Save generator/version/key, degree specification,
exception records, precision policy, assignments, retained message state,
frozen scores, queues/schedule counters and RNG/counter positions needed for
an exact restart. A seed plus assignments alone cannot resume BSP's numerical
state. Stream checkpoints to avoid a second resident copy.

The immediate implementation milestone is the
**local FP64 compact-layout comparison at N=10^7**, followed by a separately
validated FP32 backend. The single-GPU N=10^8 experiment belongs to a later
session. A full satisfying assignment and original-formula verification are
stronger evidence than the one-sweep memory baseline measured in this audit.
The 16-bit target is an additional experiment. Schedule a billion-variable run
only after measured sweep counts, memory peaks and bandwidth support it.

## 15. Reproduction

From the repository root:

```sh
python3 bench/memory_budget.py > bench/results/billion_variables_20261029/memory_3sat.json
python3 bench/memory_budget.py --k 4 --alpha 9.8 > bench/results/billion_variables_20261029/memory_4sat.json
cargo build --release --locked --lib
rustc --edition=2021 -O bench/prototypes/message_storage.rs --extern bsp_rust=target/release/libbsp_rust.rlib -L dependency=target/release/deps -o /private/tmp/bsp-message-storage-probe
/private/tmp/bsp-message-storage-probe 10000 80 0 > bench/results/billion_variables_20261029/message_storage.csv
/private/tmp/bsp-message-storage-probe 10000 80 200 > bench/results/billion_variables_20261029/message_storage_warm.csv
cargo build --release --locked --bin bsp-rust
python3 bench/prototypes/measure_memory.py target/release/bsp-rust bench/results/billion_variables_20261029/peak_rss.json
python3 bench/prototypes/measure_memory.py target/release/bsp-rust bench/results/billion_variables_20261029/peak_footprint_n10000000.json 10000000
```

The probe implements IEEE-style round-to-nearest-even storage values in software
for BF16/FP16, including subnormals. It does not emulate hardware flush-to-zero,
FP32 arithmetic, GPU execution or a precision side table. The FP64 format is an
identity control. No production algorithm, defaults or public API were modified.
