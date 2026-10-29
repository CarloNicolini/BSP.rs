Python API reference
====================

.. currentmodule:: pybsp_rust

The public wrapper is intentionally small. Formula and cancellation objects are
native PyO3 classes. Solver entry points expose explicit keyword settings;
configuration dataclasses are optional compatibility records, created lazily
when requested. Snapshots and structured results wrap the same Rust solver.

.. autosummary::
   :nosignatures:

   Formula
   Options
   ExecutionOptions
   Solver
   SolveResult
   CancellationToken
   solve
   solve_many
   from_file

.. toctree::
   :maxdepth: 1

   formula
   options
   solver
   results
   functions
