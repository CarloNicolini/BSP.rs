Solver and cancellation
=======================

.. autoclass:: pybsp_rust.Solver
   :members:

.. autoclass:: pybsp_rust.CancellationToken
   :members:

The cancellation handle is obtained through ``solver.cancellation_token``;
it has no public constructor or reset operation. See
:doc:`../guide/inspection` for state transitions, snapshot freshness, and
concurrent cancellation examples.
