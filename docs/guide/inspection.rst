Inspecting and controlling a search
===================================

Converge before reading surveys
-------------------------------

``Solver`` separates raw message updates from fixed-point computation and
discrete actions. Start with ``converge()`` to propagate units, iterate SP, and
compute observables from the final messages. The survey columns are ordered
``[True, False, free]``; see :doc:`../theory/survey-propagation` for their meaning.

.. code-block:: python

   import numpy as np
   from pybsp_rust import Formula, Solver

   formula = Formula.random(3, 4.15, 400, seed=3)
   solver = Solver(formula=formula, seed=3, timeout_seconds=10)
   report = solver.converge()       # may raise RuntimeError: non_converged, etc.
   free = solver.assignment == -1
   assert np.allclose(solver.surveys[free].sum(axis=1), 1.0)
   print(report["residual"], report["complexity"], report["paramagnetic"])

``residual`` is the maximum absolute raw message change in the last sweep.
``complexity`` is an extensive cluster-count estimate, not a log count of
satisfying assignments. ``paramagnetic`` is the solver's algebraic handoff
diagnostic. Neither the residual nor the complexity certifies a SAT solution.

Explicit fix, release, and propagation
--------------------------------------

.. doctest::

   >>> f = Formula([[1, 2], [-2, 3]], n=3)
   >>> s = Solver(f)
   >>> s.fix(0, False)
   >>> s.propagate()
   2
   >>> s.assignment.tolist()
   [0, 1, 1]
   >>> s.release(0)
   >>> s.assignment.tolist()
   [-1, 1, 1]

Only the decision variable was released. Unit-forced variables remain fixed;
``release`` is not a reversible trail rollback and does not unwind the full
implication closure. To explore an independent branch with fresh implications,
construct another solver from the original formula. Fixing an already fixed
variable, releasing a free variable, or releasing a unit-forced variable raises
``ValueError``.

Messages on removed incidences are saved and restored when eligible graph edges
are reactivated. ``set_messages`` replaces these saved values too, so use it
deliberately when studying restoration. Surveys are stale after graph mutations;
call ``converge()`` again before using them to rank free variables.

One BSP action at a time
------------------------

.. code-block:: python

   solver = Solver(formula=formula, seed=3, max_steps=1000, timeout_seconds=10)
   while True:
       try:
           step = solver.advance()
       except RuntimeError:
           break
       print(step["index"], step["action"], step["changed"])
       if step["action"] == "walksat":
           break
   result = solver.solve()         # retrieves or continues the existing state
   print(result.status)

The step's free-variable and active-clause counts are measured **before** its
action; the changed count describes the action. ``sp_sweeps`` is cumulative.
``advance`` raises at a terminal failure and at calls after termination.
``solve`` returns a structured terminal result instead. It continues the current
state, rather than resetting the RNG, clock, cancellation token, or counters.

``decimate()`` and ``backtrack()`` expose the actions directly. They bypass the
outer scheduler. In particular, ``decimate()`` uses cached surveys without first
finding a fixed point; calling it after arbitrary raw sweeps is not the same as
a BSP step.

Raw sweeps and message experiments
----------------------------------

.. doctest::

   >>> import numpy as np
   >>> f = Formula([[1, 2], [-1, 3], [-2, -3]], n=3)
   >>> s = Solver(formula=f, damping=0.2)
   >>> s.set_messages(np.full(f.edges, 0.1))
   >>> residual = s.sweep_many(5)
   >>> bool(np.isfinite(residual) and np.isfinite(s.messages).all())
   True
   >>> snapshot = s.messages
   >>> snapshot[:] = 0
   >>> bool(np.any(s.messages != snapshot))
   True

``sweep`` and ``sweep_many`` do not run unit propagation or refresh surveys.
Use raw updates on a deliberately chosen residual state. ``sweep_many`` avoids
one host snapshot per sweep on a GPU; it still checks cancellation, timeouts,
and numerical failures. Zero raw sweeps return 0.0 without a convergence attempt.

Cancellation from another thread
--------------------------------

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor

   solver = Solver(formula=formula, timeout_seconds=30)
   token = solver.cancellation_token
   with ThreadPoolExecutor(max_workers=1) as pool:
       future = pool.submit(solver.solve)
       token.cancel()              # safe even while native compute releases the GIL
       result = future.result()
   print(result.status)           # cancelled, or a result finished before cancellation

Cancellation is one-way. A token cannot be reset. Do not concurrently mutate
the same solver; use the token for interruption and independent solver objects
for parallel searches. Timeouts are also cooperative, and include time spent
between calls since construction. A large GPU sweep can delay the next check.
