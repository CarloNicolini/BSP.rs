Parameters and stopping rules
=============================

Pass any parameter below directly to ``solve``, ``Solver``, or ``solve_many``:
``solve(formula=formula, seed=7, damping=0.2, backend="cpu-parallel", threads=2)``.
All settings after the formula are keywords in this preferred calling style.
Batch algorithm keywords apply to every job, including the seed; use a sequence
of Options for distinct per-job settings.

For reusable configurations, algorithm settings also live in
:class:`pybsp_rust.Options` and hardware settings in
:class:`pybsp_rust.ExecutionOptions`. Both are immutable dataclasses validated
by Rust, with the same defaults as direct keywords. Existing object-based calls
remain supported. Do not combine ``options`` with direct algorithm keywords,
or ``execution`` with direct hardware keywords: these ambiguous calls raise
TypeError, even when the values agree. One algorithm object plus direct hardware
keywords (or the reverse) is supported. Unknown keywords also raise TypeError.

.. doctest::

   >>> from dataclasses import replace
   >>> base = Options(seed=7, damping=0.2)
   >>> variant = replace(base, seed=8, epsilon=1e-4)
   >>> base.seed, variant.seed, variant.epsilon
   (7, 8, 0.0001)

BSP control
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Parameter
     - Default / domain
     - Exact role
   * - ``seed``
     - 1; uint64
     - Initializes the solver RNG for messages and WalkSAT. Formula generation
       has a separate seed. Reproducibility requires recording both.
   * - ``backtrack_ratio``
     - 0.9; :math:`0\le r<1`
     - Schedules backward actions when the current backward/forward action ratio
       falls below r. Counters start at one each. Zero disables scheduled
       backtracking. It is neither a random move probability nor a guarantee on
       the ratio of individual variable events when batch sizes differ.
   * - ``batch_fraction``
     - 0.00125; :math:`0<f\le1`
     - With :math:`N_f` free variables, fixes
       :math:`\min(N_f,\max(1,\lfloor fN_f\rfloor))` per decimation.
       Larger batches reduce fixed-point calls but commit more decisions at once.
   * - ``max_steps``
     - 100,000; positive
     - Limits the number of recorded actions, rather than sweeps or variable
       assignments. A handoff to WalkSAT is itself a recorded action.

For a controlled comparison of SID-like and backtracking behavior, change only
``backtrack_ratio`` and hold formulas, solver seeds, limits, backend, and precision
fixed. A ratio close to one can require substantially more repeated work; the
published BSP paper's scaling argument is not a bound on this implementation's
time to convergence.

Fixed-point computation
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Parameter
     - Default / domain
     - Exact role
   * - ``epsilon``
     - 0.01; :math:`0<\epsilon<1`
     - Absolute tolerance on the largest undamped clause-message update over
       active free incidences. It is not a relative tolerance, a bound on true
       marginals, or a SAT-accuracy setting.
   * - ``damping``
     - 0.0; :math:`0\le d<1`
     - Weight of the previous message. A sweep stores
       :math:`\eta^{t+1}=\widehat\eta+d(\eta^t-\widehat\eta)` but measures
       :math:`\max|\widehat\eta-\eta^t|`. Heavy damping cannot fake convergence
       through an artificially small applied increment.
   * - ``max_sweeps``
     - 1,024; positive
     - Per-fixed-point iteration limit. Reaching it returns ``non_converged``
       from a solve or raises RuntimeError from ``converge``. Numerical recovery
       can retry an attempt from its checkpoint using CPU f64.

Raw clause products below :math:`10^{-16}` are set to zero before residual
measurement and damping. Prefix/suffix exclusion handles exact zeros without
division by :math:`1-\eta`. Observables are recomputed from final messages after
convergence. GPU f32 conversion and underflow recovery are described in
:doc:`hardware`; precision and tolerance should be considered together.

If an experiment does not converge, record the failure rather than silently
loosening epsilon. Compare damped runs at a fixed undamped residual criterion.
Changing ``max_sweeps`` changes computational effort, not the approximation's
theoretical validity.

WalkSAT completion
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Parameter
     - Default / domain
     - Exact role
   * - ``max_flips``
     - 10,000,000; uint64
     - Total flip cap across restarts. Zero allows checking an initial completion
       but no flips. The last permitted flip is followed by a satisfaction check.
   * - ``walksat_restarts``
     - 4; positive
     - Maximum fresh randomized completions. Per-restart quota is
       :math:`\lceil\text{max\_flips}/\text{walksat\_restarts}\rceil`; a separate
       cumulative cap prevents overspending the total budget.
   * - ``walksat_noise``
     - 0.5; :math:`0\le p\le1`
     - Selects a random candidate from a sampled unsatisfied clause with
       probability p only when the smallest break count is positive. Otherwise
       select among minimum-break candidates. Ties are randomized.

Only variables free in the BSP partial assignment may flip. A break count is
the number of currently satisfied active clauses that would become unsatisfied
if the candidate flipped. The completion is verified against the original CNF,
not just the residual. BSP.rs returns one assignment; it is not a uniform sampler
or an enumerator of solutions or clusters.

Wall time and resources
-----------------------

``timeout_seconds=None`` disables the wall-time limit. Otherwise it must be
positive and finite. The clock starts at solver construction, including hardware
setup, and does not reset between manual calls. Python formula construction
before the solver exists is outside this clock. Queued batch jobs start their
own clocks when their solver is constructed, not when the batch was submitted.

``ExecutionOptions.backend`` accepts ``cpu``, ``cpu-parallel``, ``metal``,
``cuda``, or ``cuda-f32``. ``threads=0`` selects available logical CPUs;
explicit counts may be 1–1024. Scalar CPU stays single-threaded per solver.
The same budget controls independent jobs in ``solve_many``. GPU jobs can be
further limited by device memory. ``profile`` enables instrumentation and writes
a JSON file from ``solve()``; batch profiling is rejected to avoid path races.

Seed and integer fields reject negative or oversized native values. All floating
parameters must be finite. Probability and tolerance ranges differ as shown
above; in particular, epsilon cannot be zero, and damping cannot be one.
