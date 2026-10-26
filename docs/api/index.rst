Python API reference
====================

.. currentmodule:: pybsp_rust

The public wrapper is intentionally small. Formula and cancellation objects are
native PyO3 classes; the other documented classes provide Python configuration,
snapshot conversion, and structured results around the same Rust solver.

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
