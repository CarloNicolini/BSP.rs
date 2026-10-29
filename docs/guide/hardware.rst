Hardware, precision, and profiling
==================================

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Backend
     - Internal precision
     - Execution contract
   * - ``cpu``
     - f64
     - Portable scalar reference; default. One host thread per solver.
   * - ``cpu-parallel``
     - f64
     - Shared Rayon pools; preserves scalar per-variable arithmetic and ordered
       complexity accumulation. Small residuals use scalar sweeps.
   * - ``metal``
     - f32
     - Custom Apple GPU kernels; optional macOS feature. Requires a supported
       unified-memory device. Runtime compilation, cached pipelines.
   * - ``cuda``
     - f64
     - Optional Linux/NVCC build. Keeps scalar cavity product order; hierarchical
       reductions can still change observable roundoff.
   * - ``cuda-f32``
     - f32
     - NVIDIA single precision, including cooperative scans for high degrees.

All backends implement synchronous cavity updates followed by clause updates and
an undamped residual reduction. CPU graph control, unit propagation, and single
WalkSAT flip chains remain sequential. Independent jobs supply another level of
parallelism; see :doc:`experiments`.

Precision is part of an experiment
----------------------------------

GPU single precision is an explicit numerical contract. Surveys and stopping
decisions can differ, so the same RNG seed does not promise the same assignment,
step history, or solve probability across precisions. Python snapshots always
use ``float64``, even when device messages are ``float32`` internally.

Fast math is disabled. Unsafe f64-to-f32 input conversions, detected underflow,
and non-finite cavity arithmetic cause a recorded retry on CPU f64 from a
coherent message checkpoint. Decision-sensitive complexity and paramagnetic
diagnostics receive additional CPU evaluation on the current rounded messages.
This does not reconstruct the original scalar f64 trajectory.

CUDA runtime failures produce ``numerical_failure`` with ``device_error``;
failed execution is not presented as successful recovery. Always inspect both
the requested and actual backends and the fallback list. Every successful
assignment is verified against the formula on CPU regardless of precision.

Memory and synchronization
--------------------------

GPU topology, live/saved messages, cavities, and scratch buffers stay resident.
CPU and Metal cavity normalizers reuse the saved-message slots of active/free
incidences. Frozen slots retain their original messages for backtracking. This
removes one host f64 array of length E and one Metal f32 array of length E,
where E is the number of normalized literal incidences. Metal also omits its
generic prefix scratch when every clause uses a specialized width-3/4 kernel.
Generic-width inputs still allocate the required scratch. CUDA device buffers
retain their existing layout.

These changes reduce allocation; they do not guarantee a speedup. The recorded
million-variable 3-SAT action prefixes used about 12.3% less scalar CPU footprint
and 14.8% less Metal process footprint, with about 5–7% longer runtimes. See the
`paired memory experiments <https://github.com/CarloNicolini/BSP.rs/blob/main/bench/results/memory_ablation_20261029/REPORT.md>`_
for the workloads, controls, timing limits, and denser 4-SAT results. The process
footprint already includes unified GPU allocations; do not add those metrics.

Graph edits are applied consistently with message freeze/restore order. Public
methods refresh host snapshots before returning; internal solving exchanges
control information and surveys and performs additional readback for numerical
validation when needed. ``sweep_many`` amortizes the public snapshot cost.

Submissions adapt toward about 2 ms of GPU work and contain at most 16 sweeps.
Device flags stop updates at first convergence or numerical failure. One large
sweep can exceed the target. Cancellation and timeout checks remain cooperative.

CUDA solvers use separate streams and allocations on visible device 0; use
``CUDA_VISIBLE_DEVICES`` before starting Python to select another physical GPU.
The campaign scheduler uses a conservative free-memory estimate to bound GPU
workers. Concurrent external GPU users can still cause allocation failures.
Metal uses unified allocations and its device working-set recommendation.

Measure throughput and latency separately
-----------------------------------------

.. code-block:: python

   from time import perf_counter
   from pybsp_rust import Formula, Solver

   formula = Formula.random(k=3, alpha=4.2, n=100000, seed=7)
   solver = Solver(formula=formula, damping=0.2, backend="cpu-parallel", threads=8)
   start = perf_counter()
   solver.sweep_many(30)
   print("sweep seconds:", perf_counter() - start)
   print(solver.execution_report)

This timer includes final host synchronization but excludes formula and solver
construction. For a complete-solve measurement, start the timer before Solver
construction and include ``solve()``. Compare identical formulas and initial
messages, discard warmups, alternate backend order, and report repeated samples.
Run profiling separately. GPU kernel speedups alone do not establish improved
time to a verified solution, particularly on small formulas.

.. code-block:: python

   solver = Solver(formula=formula, timeout_seconds=10,
                   backend="cpu", profile="profile.json")
   result = solver.solve()
   print(result.execution["phases"])

Profiling records phase times, backend setup, snapshots, compiler/build settings,
GPU timestamps where available, and backend memory. GPU phase profiling splits
work into synchronized submissions and perturbs runtime. Profile individual
jobs, since ``solve_many`` rejects a shared profile path. Profile-write failures
appear in ``profile_error``; the returned SAT solution remains independently valid.
