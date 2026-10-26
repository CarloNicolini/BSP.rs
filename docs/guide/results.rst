Results, failures, and diagnostics
==================================

Successful verification
-----------------------

``result.status == "sat"`` means that a complete 0/1 assignment passed CPU
evaluation of every original normalized clause. ``result.assignment`` then has
shape ``(formula.n,)`` and dtype ``int8``. ``partial_assignment`` retains BSP's
partial state, so it can still contain -1 even on success: WalkSAT completes
free variables without fixing the BSP representation.

All other statuses have ``assignment is None``. ``partial_assignment`` is
diagnostic state and must not be mistaken for a complete satisfying assignment.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Status
     - Meaning
   * - ``sat``
     - Complete assignment verified against the input CNF.
   * - ``contradiction``
     - A residual active clause has no satisfying extension under current fixed
       decisions. Decisions may be heuristic; this is not a global UNSAT proof.
   * - ``non_converged``
     - The SP fixed point did not meet epsilon within its sweep limit.
   * - ``negative_complexity``
     - The implementation rejected an estimated total complexity below -2 on a
       forward action. This numerical heuristic is not an exact cluster count.
   * - ``numerical_failure``
     - An invalid normalization, non-finite observable, device error, or failed
       final verification prevented a valid result. Inspect execution metadata.
   * - ``step_limit``
     - The configured number of BSP action records was reached.
   * - ``flip_limit``
     - WalkSAT exhausted its allowed completion work without a verified solution.
   * - ``cancelled``
     - A cooperative cancellation request was observed.
   * - ``timeout``
     - The cooperative wall-time limit was observed.

``solve()`` returns these outcomes as records. Lower-level methods such as
``converge()`` or ``advance()`` raise ``RuntimeError`` for operational failures.
Invalid configuration or input normally raises ``ValueError``; wrong wrapper
types raise ``TypeError``. Native integer conversion can raise ``OverflowError``.

Counters and history
--------------------

``sp_sweeps`` and ``flips`` are cumulative work counters. ``decimations``,
``releases``, and ``unit_assignments`` count variable events and can exceed n
when variables are revisited. ``steps`` contains dictionaries with:

* ``index``: zero-based position in recorded history;
* ``action``: ``decimate``, ``backtrack``, or ``walksat``;
* ``free_variables``, ``active_clauses``: counts before the action;
* ``sp_sweeps``: cumulative count at that fixed point;
* ``residual``, ``complexity``: final fixed-point diagnostics;
* ``changed``: number of variables fixed or released by that action.

An early failure can leave no action records. A WalkSAT record is appended before
completion, so its ``changed`` remains zero even if many local flips occur.
Use the result's ``flips`` for local-search work.

Execution report
----------------

``result.execution`` and ``solver.execution_report`` expose the requested and
actual ``backend``, internal ``precision``, ``threads`` per solver, ``device``,
and estimated ``allocated_bytes`` of backend storage. For GPUs this is device
state, not the total process memory or all driver allocations. Work counters
include ``parallel_sweeps``, ``metal_sweeps``, and ``cuda_sweeps``.

``fallbacks`` lists numerical recovery reasons. ``observables_validations``
counts CPU evaluations of decision-sensitive GPU observables. ``device_error``
reports CUDA runtime failures; a broken device execution is not relabeled as a
successful fallback. ``phases`` contains accumulated counts and seconds only
when profiling is enabled. ``profile_error`` records profile-write errors.

``wall_seconds`` is elapsed time since solver construction. It includes setup
and manual pauses, and differs from an external batch wall timer. Profile phases
can overlap or nest; do not sum them to claim total time. Instrumented GPU runs
use additional synchronization and must be kept separate from speed measurements.
