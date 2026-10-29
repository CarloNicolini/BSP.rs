Research experiments and independent jobs
=========================================

Separate instance randomness from search randomness
---------------------------------------------------

For random k-SAT, report k, n, :math:`\alpha=M/N`, the formula seed, and the
solver seed. Save the normalized DIMACS when an experiment needs to survive
future RNG or generator changes. Reproducibility also depends on the package
revision, compiler, backend, precision, stopping criteria, and hardware.

Do not interpret the average of returned SAT assignments as an unbiased sample
of solutions. BSP plus WalkSAT is a search heuristic; its probability of finding
an assignment depends on the algorithm and basin of attraction.

Bounded batches
---------------

.. doctest::

   >>> formula = Formula([[1], [-2]], n=2)
   >>> jobs = [formula] * 3
   >>> configs = [Options(seed=seed) for seed in range(3)]
   >>> results = solve_many(formulas=jobs, options=configs, backend="cpu", threads=2)
   >>> [r.seed for r in results]
   [0, 1, 2]
   >>> [r.status for r in results]
   ['sat', 'sat', 'sat']

Results preserve input order, not completion order. Repeated Formula objects
share immutable CPU topology but not mutable state. Each job has independent
options, seed, counters, time origin, and local-search history. A single Options
object can also be broadcast to every input, but then every repeated formula
receives the same solver seed. An empty input returns an empty list.

The shared execution setting chooses a backend and worker budget. Inner CPU
parallelism is suppressed when enough independent work already occupies the
pool. GPU workers are additionally bounded by device memory. This is a bounded
job scheduler, not a fused batch tensor implementation of the SP equations.

Jobs allocate mutable solver state only when a worker is available. Formula
objects already materialized by the caller still consume memory. The batch
wrapper materializes its input iterable before computing. For a large formula
collection, submit bounded batches yourself instead of an unbounded iterator.

.. code-block:: python

   results = solve_many(
       formulas=[Formula.random(3, 4.15, 400, seed=s) for s in range(5)],
       options=[Options(seed=s, timeout_seconds=10) for s in range(5)],
       backend="cuda-f32",
       threads=4,
   )
   for result in results:
       print(result.seed, result.status, result.wall_seconds, result.execution)

The GPU example requires an optional CUDA build. Use ``cpu`` for portable runs
or ``metal`` after an Apple GPU build. Native setup errors raise for the batch;
ordinary search failures remain individual result records. A shared profile
destination is rejected: construct and profile individual Solver objects instead.

Report failed searches and censored runs
----------------------------------------

For success probability, specify the instance distribution, number of instances,
solver seeds, budget, and treatment of timeouts. Record all attempts, including
``non_converged``, ``negative_complexity``, numerical failures, and timeouts.
Report binomial uncertainty on success fractions when appropriate. Timeouts
are censored observations, not measured times to solution. Averaging only
successful runs selects easier instances and can reverse apparent rankings.

For hardware comparisons, distinguish raw sweeps, fixed-point latency, full
search latency, and campaign throughput. Single precision can perform different
search work. Equal seeds alone do not make wall times a paired kernel comparison;
use matched starting messages and inspect work counters and numerical agreement.

Scope of the implementation
---------------------------

This package implements a zero-energy SP/BSP search with local-search completion.
It does not implement BP as an alternate solver, generic finite-temperature
1RSB population dynamics, arbitrary Parisi reweighting, whitening measurements,
exhaustive SAT enumeration, or an UNSAT proof system. The theory chapters use
BP and state-level measures to explain the implemented SP approximation.
