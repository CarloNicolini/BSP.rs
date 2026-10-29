First solve
===========

Build a formula from signed literals
------------------------------------

A clause is a list of literals joined by OR; the outer list joins clauses by
AND. A positive integer denotes a variable and a negative integer its negation.
For example, ``[[1, -2], [2, 3]]`` means
:math:`(x_1\lor\neg x_2)\land(x_2\lor x_3)`.

.. doctest::

   >>> from pybsp_rust import Formula, solve
   >>> formula = Formula([[1], [-2]], n=2)
   >>> result = solve(formula=formula, seed=3, timeout_seconds=10)
   >>> result.status
   'sat'
   >>> result.assignment.tolist()
   [1, 0]
   >>> formula.is_satisfied(result.assignment)
   True

Always inspect ``status`` before using ``assignment``. Only ``sat`` carries a
verified complete assignment. See :doc:`results` for all other outcomes.

Load DIMACS
-----------

.. code-block:: python

   from pybsp_rust import from_file, solve

   formula = from_file("examples/3sat.cnf")
   result = solve(formula)
   if result.status == "sat":
       assert formula.is_satisfied(result.assignment)

``Formula.from_dimacs(text)`` parses a string. Variable labels remain those in
the file even if some variables do not occur after normalization.

Generate a controlled instance
------------------------------

.. doctest::

   >>> formula = Formula.random(k=3, alpha=2.5, n=128, seed=7)
   >>> formula.n, formula.m, formula.edges
   (128, 320, 960)
   >>> result = solve(formula=formula, seed=3, timeout_seconds=10)
   >>> result.status == 'sat' and formula.is_satisfied(result.assignment)
   True

The **formula seed** chooses the problem. The **solver seed** chooses initial
messages and local-search randomness. Changing the solver seed does not create
a different formula. Record both when comparing searches.

Select hardware explicitly
--------------------------

.. code-block:: python

   from pybsp_rust import Solver

   solver = Solver(
       formula=formula,
       seed=3,
       timeout_seconds=10,
       backend="cpu-parallel",
       threads=8,
   )
   result = solver.solve()
   print(result.status, result.execution)

Use ``metal`` on an Apple GPU, ``cuda`` for NVIDIA double precision, or
``cuda-f32`` for NVIDIA single precision, after the corresponding optional build.
Reports describe the actual backend, which can change after numerical recovery.
GPU sweep speed does not by itself imply faster complete solves.
