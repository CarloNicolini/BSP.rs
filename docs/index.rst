BSP.rs: survey propagation from Python
======================================

BSP.rs searches for satisfying assignments of Boolean formulas with a native
Rust implementation of **Backtracking Survey Propagation**, followed by WalkSAT
completion. The Python package, ``pybsp_rust``, exposes the complete search and
the intermediate messages, surveys, partial assignments, and decisions.

Start with an ordinary CPU installation. The same API can select multicore CPU,
Apple Metal, or NVIDIA CUDA execution. Every returned SAT assignment is checked
against the input formula on the CPU. An unsuccessful search is not an UNSAT proof.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Run and inspect a solver
      :link: guide/quickstart
      :link-type: doc

      Install the bindings, solve CNF formulas, and inspect reproducible experiments.

   .. grid-item-card:: Understand the physics
      :link: theory/problem
      :link-type: doc

      Follow factor graphs through belief propagation, surveys, and backtracking.

   .. grid-item-card:: Choose parameters and hardware
      :link: guide/parameters
      :link-type: doc

      Find exact defaults, stopping rules, numerical contracts, and batch settings.

   .. grid-item-card:: Python API reference
      :link: api/index
      :link-type: doc

      Constructors, methods, snapshots, exceptions, and result records.

.. toctree::
   :maxdepth: 2

   User guide <guide/index>
   Theory <theory/index>
   Python API <api/index>
   Development <building>

The theory chapters distinguish the cavity approximation from implementation
choices. The API pages document this repository's behavior, including numerical
recovery and explicit failure states. The cited research papers motivate the
algorithm; their reported thresholds and running times are not guarantees for
this implementation or for arbitrary SAT instances.
