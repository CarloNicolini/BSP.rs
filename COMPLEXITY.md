# Complexity audit of BSP.rs

This audit covers the Rust core, CLI, Python boundary, and formula construction.
The retained changes target repeated searches. Full scans that read new scores,
write every message, or return an entire snapshot have different lower bounds.

## Changes retained

Let N be the variable count, F the number of fixed decision variables, b the
release batch size, d_i a variable's original degree, k_a a clause width, U a
bitset's index space, and A its number of set bits.

| Location | Previous work | New work | Maintenance and memory cost |
| --- | --- | --- | --- |
| `Solver::backtrack` | O(N + F + b log b): filter all variables, partition, sort batch | O(b log F) ranking/removal, with O(1) minimum lookup | O(log F) on each decision fix or arbitrary release; up to 8N extra bytes for IDs and positions |
| `Solver::observables` active degree | O(d_i) recount per free variable per fixed point | O(1) read per variable | 4N bytes; integer updates share the existing clause activation/deactivation loops |
| `Solver::propagate` sole free literal | O(ceil(k_a/64)) search through clause masks | O(1) lookup using XOR of the free edge IDs | 4M bytes; one XOR for each existing fix/release incidence update |

The indexed heap holds one entry per fixed decision. There are no stale duplicate
keys. Unit-propagated assignments stay outside the heap. Scores and variable-ID
tie breaking are identical to the previous sorted-prefix policy. A fixed score
cannot change until the variable leaves the heap.

The additional production index storage is up to 12N + 4M bytes: heap IDs and
positions, active degrees, and free-edge XORs. Arrays remain directly indexed.

These bounds cover ranking and lookup. Releasing a variable must still restore
its incident clauses and affected messages. That graph work remains proportional
to the affected data; it does not become logarithmic because ranking does.

## Hierarchical bitsets: measured and rejected for production

A tested prototype in `bench/prototypes/sparse_bitset.rs` indexes nonempty words.
It supports O(log_64 U) worst-case successor work, with summaries using below
1/63 of the leaf-word storage asymptotically. Dense iteration stays flat; sparse
iteration retains ascending order without heap allocation. The sparse path is
selected only below one eighth of the leaf-word population.

This improved isolated sparse enumeration, but its maintenance and iterator
layout slowed complete solves. A paired diagnostic with five measured trials
compared baseline, indexed solver with flat bitsets, and indexed solver with
hierarchical bitsets. For k=3, N=400, the median seconds were 0.37794, 0.37366,
and 0.38499. For k=4, N=400, they were 1.82710, 1.77470, and 1.87677. These
medians are saved in `bench/indexing-ablation.json`; individual samples were
not saved for this diagnostic. Production therefore keeps the original flat
bitset. The prototype remains test/benchmark-only.

## Linear work retained

| Location | Why it stays linear, or why a tree is not a drop-in improvement |
| --- | --- |
| Free-variable ranking in `decimate` | Every free certitude can change after an SP fixed point. An exact ranking must inspect the new keys. Rebuilding a heap is O(N), while repairing every changed key individually can cost O(N log N). Current partial selection is O(N + b log b). |
| Prefix/suffix variable cavities | All relevant outgoing cavity probabilities are produced. A product tree helps a single update/query, but applying it to every updated message adds a log-degree factor to a complete synchronous sweep. |
| Clause message sweep | Producing every active outgoing message requires Omega(E_active) writes. Current clause products are linear in width, avoiding pairwise neighbor recomputation. |
| WalkSAT minimum-break choice within a clause | It reads a small k-SAT clause. Maintaining a heap per clause would require updating every affected occurrence whenever a variable's break count changes. The update fanout can exceed the saved clause scan. |
| WalkSAT incident-clause updates | Each affected clause count, XOR, and break transition must be updated. Direct CSR access already avoids a global search. |
| WalkSAT unsatisfied-clause set | Uniform selection, insertion, and swap-removal already cost O(1). A balanced tree would add a logarithmic factor. |
| Unit queue | Push/pop are amortized O(1). A priority queue adds ordering the algorithm does not need. |
| Formula parsing, generation, validation, and export | Reading, writing, or checking an entire formula has linear input/output cost. Literal normalization sorts each clause in O(k log k); this is a one-time constructor cost. |
| NumPy snapshots and result/history export | The API returns owned arrays or serialized records. Producing E messages, N assignments, or T history rows takes at least that much output work. Immutable views would be a separate API and ownership trade. |
| Assignment changes | Residual masks and message freezing/restoration touch the variable's incident graph. A global ordered map would add overhead to direct-indexed arrays. |

## Remaining candidates requiring separate measurement

`cavities` still scans original incidence lists of free variables, including zero
messages from inactive clauses. A stable active-incidence index could reduce
late-stage work when most clauses are satisfied. It must preserve sign order,
zero-factor exclusion, and message restoration. It adds per-edge position/mask
state and mutation work; it is not included in this change.

Incremental SP using a dependency work queue could avoid recomputing unchanged
subgraphs. Its benefit depends on the changed-message frontier. It changes
scheduling and convergence checks, so a logarithmic product tree alone does not
establish an equivalent or faster solver.

Per-clause WalkSAT rank structures may be useful for very wide CNFs. The current
benchmarks primarily use short k-SAT clauses. A valid comparison must include
all score-update fanout, not only extraction of the minimum.

A hash table can make duplicate detection expected-linear for very wide input
clauses, while integer radix techniques can improve normalization. Neither
makes full parsing logarithmic, and neither is on the iterative solver path.

## Validation and timing

Unit tests compare the heap with an independent scan over 20,000 arbitrary
removals and reinsertions. Additional tests independently recount clause masks,
free-edge XORs, active degrees, and eligible heap entries after mixed fix,
release, and propagation operations. Prototype bitset tests cover dense/sparse transitions,
empty sets, and word/summary boundaries. Existing scalar SP and solver tests stay
in place.

`bench/indexes.rs` compares a one-pass linear minimum with indexed heap removal,
score change, and reinsertion. The one-pass minimum is a conservative baseline
relative to the old gather-and-partition implementation. Initial heap construction
is outside that timer; its cost is paid by actual insertions in the full-solver
comparison. The same benchmark compares the production flat iterator with the rejected
hierarchical prototype. It does not enable that prototype in the solver.

`bench/compare_indexing.py` alternates before/after complete solves on identical
CNFs. It discards warmup, checks every result field except elapsed time, and
records all samples. Assignment, partial assignment, step history, SP sweeps,
flips, decimations, releases, and statuses must agree exactly. The reported timer
includes native solver-state construction and final verification, but excludes
formula parsing and process startup.

Run from the repository root:

```sh
# Before applying an optimization, preserve a Release baseline:
cargo build --release --locked
cp target/release/bsp-rust /tmp/bsp-before-indexing
# After applying it:
cargo build --release --locked
cargo bench --bench indexes > /tmp/index-primitives.json
python bench/compare_indexing.py --before /tmp/bsp-before-indexing \
  --repeats 5 --output /tmp/indexing-results.json
```

All measurements are workload-specific. A microbenchmark speedup for selecting
one key is not a complete-solver speedup.

## Measured retained changes

Five paired measured trials follow one warmup per workload. Both binaries use
the same Rust 1.88 generic Release settings. Every complete trajectory agrees.

| k | alpha | N | Seed | Outcome | Before seconds | After seconds | Before / after |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 3 | 2.5 | 400 | 3 | sat | 0.000419 | 0.000421 | 0.997 |
| 3 | 4.15 | 400 | 3 | sat | 0.377796 | 0.373465 | 1.012 |
| 3 | 4.15 | 400 | 1 | non_converged | 0.656774 | 0.647623 | 1.014 |
| 4 | 9.5 | 400 | 3 | sat | 1.821428 | 1.768601 | 1.030 |
| 3 | 4.2 | 1000 | 1 | sat | 4.433500 | 4.367394 | 1.015 |

The workloads with backtracking improve by about 1.2-3.0% in these medians.
The easy case with no releases is about 0.3% slower, at sub-millisecond scale.
No statistical significance or general speedup is claimed from this small set.
Samples and hashes are in [indexing-results.json](bench/indexing-results.json).

The isolated heap benchmark measures about 61x, 376x, and 1,339x less time than
a full minimum scan at 1,000, 10,000, and 100,000 fixed keys. Those are ranking
operations with frozen keys, not SP updates or complete solves. Initial heap
setup is excluded from the microbenchmark and included in complete solves.
See [index-primitives.json](bench/index-primitives.json).
