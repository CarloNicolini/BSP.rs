# Memory ablation methods

The reference is main at `aec3bed32d49bb1450132594c1df3f16ebc8449a`.
All changes are developed in separate managed worktrees and local branches.
No experiment changes the reference checkout, main branch, or remote main.

The native harness `examples/memory_ablation.rs` loads a saved DIMACS instance
before timing solver construction and solving. It uses one solution, the
default r=0.9, fraction=0.00125, epsilon=0.01 and damping=0, and a stated timeout.
There is no parameter tuning per variant. CPU scalar uses one thread, CPU
parallel uses eight, and Metal uses the existing FP32 kernels. Each variant is
compared to the same backend and precision in the reference, not to a different
search trajectory on another backend.

The external runner executes one child at a time. It samples macOS
`proc_pid_rusage(RUSAGE_INFO_V4)` every 50 ms, recording resident bytes,
physical footprint and the kernel lifetime-peak footprint. It also saves
system `vm_stat` counters before and after. The recorded lifetime maximum is
the greatest peak observed before the child exits. RSS alone is not treated
as allocated memory. Process footprint includes parsing and setup, while the
native timing boundary excludes formula loading and result serialization.

Metal uses unified memory on the M1 Pro. `ExecutionReport.allocated_bytes`
counts Metal buffer payloads requested by this solver; process footprint also
includes those buffers and driver/compiler state. The two values must not be
added. CPU parallel's corresponding report counts its additional CPU scratch,
not GPU memory. A startup/compiler high-water mark can obscure smaller changes
in long-lived arrays, especially at N=100,000.

The fixed-work comparisons use 128 public BSP actions on three independent
3-SAT formulas at alpha=4.2, with paired formula/solver seeds 1/1001, 2/1002 and
3/1003. Public `advance()` synchronizes state after each action; this measures
that API workload. It is separate from native `solve()` timing, which can avoid
some snapshots. Fixed-sweep comparisons run 20 raw SP sweeps at N=1,000,000,
using formula seed 1, solver seed 1001 and three repetitions. Allocation on
first use is inside compute time for lazy scratch; setup and total native time
are saved separately so this boundary shift is visible.

Every fixed-work comparison checks the ending message-bit hash. BSP-action
comparisons also require identical complete step records. Full-solve results
are checked by the native solver and again by an independent Python DIMACS
evaluator, outside the measured solve time. Paired successful solves are
compared by witness and step-record hashes. Timeouts are reported as censored
runs and never converted into time-to-solution speedups.

Correctness checks include the existing independent scalar/f32 recurrences,
zero and subnormal cases, mixed widths, arbitrary freeze/restore orders,
numerical fallback and cancellation. The three hardware-required Metal tests
are explicitly run with `--ignored`; the normal test command skips them.
The added norm-storage test freezes messages, computes observables and
releases the variable, checking that frozen payloads survive scratch reuse.

Four isolated changes are evaluated:

1. **Host norm reuse.** Remove the FP64 cavity-normalization vector. During
   observable evaluation, active/free entries of `saved_eta` hold norms.
   Inactive/fixed entries retain frozen messages. The normalization formula,
   logarithms and accumulation order are unchanged. Parallel copies obey the
   same eligibility gate.
2. **Metal norm reuse.** Use slot 9 for active/free normalization scratch and
   preserve frozen entries with an active-clause gate. Slot 11 becomes a
   four-byte unused binding. The arithmetic is unchanged.
3. **Lazy host scratch.** Allocate q and norms on first CPU use, including
   high-precision GPU validation/fallback. This is tested independently because
   deferred allocation may reduce steady-state memory without reducing the
   complete solve's peak, and can move work into the timed compute phase.
4. **Specialized Metal prefix storage.** Allocate the edge-sized generic prefix
   buffer only if generic-width clauses exist. Widths 3/4/6/12 already keep
   prefixes locally. Mixed/general-width formulas retain their original storage.

At 3-SAT alpha=4.2, an FP64 edge vector costs 100.8 MB per million variables;
an FP32 vector costs 50.4 MB. These are payload savings, not predictions of
the observed process high-water mark.

An initial build attempt shared a Cargo target directory across worktrees.
The binary hashes and GPU byte counters exposed stale artifact reuse. Those
comparisons were quarantined and excluded. The valid campaign uses a separate
target directory in every worktree, distinct executable hashes, dependency-file
checks of source paths, and smoke checks that the two Metal reductions remove
exactly one edge-sized FP32 buffer. No performance conclusion uses the shared-
build attempt.
