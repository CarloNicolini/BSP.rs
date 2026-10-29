Compatibility configuration records
===================================

New code can pass every setting directly to ``Solver``, ``solve``, or
``solve_many``. Use dictionaries for reusable settings and ``seeds`` for batch
restart seeds. These optional immutable records remain supported for existing
callers; the preferred keyword path does not construct them.

.. autoclass:: pybsp_rust.Options
   :members:

.. autoclass:: pybsp_rust.ExecutionOptions
   :members:

See :doc:`../guide/parameters` for the effect of every default, its exact domain,
and its interaction with the numerical and computational budgets.
