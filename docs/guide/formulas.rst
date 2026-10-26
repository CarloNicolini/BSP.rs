Formulas, indices, and arrays
=============================

Two index conventions
---------------------

CNF literals are signed, **one-based** variable labels. Control methods such as
``solver.fix(variable, value)`` use **zero-based** array indices. Thus ``-3``
means :math:`\neg x_3`, while ``fix(2, False)`` fixes that same variable to False.
The value argument is Boolean; assignment arrays encode False as 0 and True as 1.

.. doctest::

   >>> f = Formula([[1, -3]], n=3)
   >>> s = Solver(f)
   >>> s.fix(2, False)
   >>> s.assignment.tolist()
   [-1, -1, 0]
   >>> s.active_clauses
   0
   >>> s.release(2)
   >>> s.assignment.tolist()
   [-1, -1, -1]

Normalization changes incidence order
-------------------------------------

``Formula`` sorts each clause by variable and sign, removes repeated literals,
and omits tautological clauses containing both signs of a variable. It preserves
input clause order otherwise, repeated whole clauses, empty clauses, and the
declared variable count. ``n=None`` infers the greatest absolute literal.

.. doctest::

   >>> f = Formula([[3, 1, 1], [2, -2], [], [-3, 2]], n=4)
   >>> f.clauses
   [[1, 3], [], [2, -3]]
   >>> f.n, f.m, f.edges
   (4, 3, 4)

``solver.messages`` follows **normalized clauses flattened in order**. Here
its four entries belong to ``(clause 0, x1)``, ``(clause 0, x3)``,
``(clause 2, x2)``, and ``(clause 2, not x3)``. Empty clauses have no incidences.
Use ``formula.clauses`` after construction to map entries; never assume the
original Python input order survives normalization.

Snapshot ownership and precision
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Property
     - Shape / dtype
     - Meaning
   * - ``assignment``
     - ``(n,) int8``
     - Partial state: -1 free, 0 False, 1 True.
   * - ``messages``
     - ``(edges,) float64``
     - Clause-to-variable warning surveys; masked incidences are zero.
   * - ``surveys``
     - ``(n,3) float64``
     - Columns forced True, forced False, not forced within a state.

These are owned arrays, not writable views into Rust or device memory. They
remain valid after later operations or destruction of the solver. Altering a
snapshot does not alter solver state. Call ``set_messages(array)`` to load an
experimental initialization. GPU ``f32`` values are widened to NumPy ``float64``
on export; this does not recover lost precision.

Surveys for free variables are refreshed by successful ``converge()``. Raw
sweeps and graph mutations can leave them stale. Fixed rows are one-hot.
Do not interpret a newly constructed solver's survey array as a converged result.

DIMACS and empty inputs
-----------------------

DIMACS needs one ``p cnf N M`` header and zero-terminated clauses. Clauses may
span lines, and several clauses may share a line. ``M`` counts clauses before
normalization. Blank lines and comment lines beginning with ``c`` are ignored.
Zero is a terminator in DIMACS and is invalid inside ``Formula([...])``.

An empty conjunction is satisfiable; an empty clause is unsatisfiable.
``Formula.is_satisfied`` checks a complete 0/1 assignment of exactly length n;
it returns False for incomplete or wrongly sized assignments.

The native representation uses packed 32-bit incidence indices and signed
32-bit literal labels. Invalid ranges, malformed DIMACS, and storage-capacity
overflow produce validation errors rather than a different interpretation.
