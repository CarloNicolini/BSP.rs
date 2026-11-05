Implicit Feistel formulas
=========================

The experimental ``--feistel`` source reconstructs graph connections on demand.
The default ``fluctuating`` mode targets the ordinary random K-SAT ensemble
used by ``--generate``. Both Feistel modes support scalar CPU BSP and WalkSAT
and verify every returned SAT assignment::

    target/release/bsp-rust --feistel 3 4 100 --seed 7 --backend cpu
    target/release/bsp-rust --feistel 3 4 1000 --seed 7 --sweeps 8
    target/release/bsp-rust --feistel 3 4 100 --feistel-ensemble partitioned

A TOML configuration can select the same source.

.. code-block:: toml

    seed = 7
    backend = "cpu"
    [feistel]
    k = 3
    alpha = 4.0
    n = 100
    ensemble = "fluctuating"

An explicit ``--feistel-ensemble`` overrides the configuration. A DIMACS path,
``generate``, and ``feistel`` conflict across configuration and CLI layers.
``--write-formula PATH`` streams DIMACS. Message import and export use
clause-major edge IDs. Reloading DIMACS normalizes literal order, so positions
may differ from the original fluctuating descriptor.

Rust callers use ``Formula::feistel_fluctuating(k, alpha, n, seed)`` for the
default CLI ensemble or ``Formula::feistel(k, alpha, n, seed)`` for the existing
partitioned constructor. ``Formula::feistel_topology()`` exposes the descriptor.
Calling ``clauses()``, ``to_dimacs()``, or ``Solver::residual_formula()``
materializes the requested output.

Fluctuating degrees
-------------------

There are ``M=floor(alpha*N)`` clauses of width ``K`` and ``E=K*M`` positions.
Inputs require finite alpha at least zero, ``1<=K<=min(N,64)``, and
``N<=i32::MAX`` for the formula's literal representation. Empty formulas are
supported. The binomial sampler uses floating point and the topology limits
``N`` and ``E`` to ``2^53``.

Consider E independent uniform draws from all N variables. The occupancy
vector is multinomial. Conditional on the counts, every sequence with those
counts is equally likely. The generator reconstructs this two-stage model
with a seeded binomial tree and a single Feistel permutation over E.

* The root owns E stubs. A node with T stubs and g variables gives its left
  half a draw from ``Binomial(T, floor(g/2)/g)``. The right half gets the
  remainder. Forward and inverse queries regenerate the same path and leaf
  interval without storing per-variable counts or offsets.
* The permutation maps a clause position to a variable stub. Its inverse
  maps a variable stub back to the clause position.
* One scan finds clauses containing repeated variables and replaces each
  whole clause with an independent ordered sample of K distinct variables.
  The forward lookup uses the replacement. The reverse iterator skips
  original positions from replaced clauses and includes the replacement
  positions from a sparse index. Signs use a separate seed domain.

With ideal independent draws and a uniform permutation this gives the same
law as ``--generate``. An original clause conditioned on being valid is
uniform over distinct tuples. A replacement is drawn from that same law.
At fixed K and alpha the expected number of replacements approaches
``alpha*K*(K-1)/2``. This is about 12 at K=3 and alpha=4. For dense clauses
relative to N, the table can grow with E. The concrete implementation uses
pseudorandom draws, and the small-instance audit measures its agreement with
the target law. A given seed typically produces different formulas in
``--generate`` and ``--feistel``.

Partitioned degrees
-------------------

Set ``--feistel-ensemble partitioned`` or ``ensemble = "partitioned"`` under
``[feistel]``. This mode requires positive finite alpha and at least one
clause. It divides the variables into K contiguous, balanced groups and
places one variable from each group in every clause. A group with g
variables gets M stubs. Each variable gets ``floor(M/g)`` stubs, with one
extra for the first ``M mod g`` variables.

This skips the degree tree and repair scan. The cycle-walk count can be lower
or higher depending on where M and K*M fall relative to powers of four.
The CLI warns on standard error because the fixed groups and balanced
degrees define a different ensemble, including when exporting DIMACS.
Both modes allow duplicate whole clauses.

Generator identities
--------------------

The matching and key derivation use unsigned 64-bit words, logical shifts,
and wrapping multiplication. Define ``mix(x)``::

    x = (x XOR (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x XOR (x >> 27)) * 0x94d049bb133111eb
    return x XOR (x >> 31)

For a nonempty matching of length L, choose
``b=max(1, ceil(bit_length(L-1)/2))`` and ``mask=2^b-1``.
In ``partitioned-feistel-v1``, L=M and group q uses
``key=mix(seed XOR 0x243f6a8885a308d3 XOR mix(q))``.
In ``fluctuating-feistel-v1``, L=K*M and the matching key is
``mix(seed XOR 0x082efa98ec4e6c89)``. Round j=0..7 has
``rk=mix(key XOR (0x9e3779b97f4a7c15*(j+1)))`` and transforms::

    (L,R) -> (R, L XOR (mix(R XOR rk) AND mask))

Apply the eight-round permutation repeatedly until its result is less than
the matching length. Inverse queries reverse the rounds and cycle-walk
backward. Each walk terminates because its cycle contains its valid starting
index. See Black and Rogaway's `Ciphers with Arbitrary Finite Domains
<https://web.cs.ucdavis.edu/~rogaway/papers/subset.pdf>`_.

For edge e, negation is the low bit of ``mix(e XOR mix(seed XOR domain))``.
The sign domain is ``0x13198a2e03707344`` for partitioned and
``0xbe5466cf34e90c6c`` for fluctuating.

The fluctuating tree uses binary heap node IDs rooted at 1. A node's stream
key is ``mix(mix(seed XOR 0xa4093822299f31d0) XOR node)``. For a repaired
clause a, the stream key is ``mix(mix(seed XOR 0x452821e638d01377) XOR a)``.
Each stream emits ``mix(key XOR counter)`` starting at counter 0. Tree counts
use the pinned ``rand_distr 0.5.1`` binomial sampler. Clause resampling uses
Floyd sampling followed by a Fisher–Yates shuffle, with rejection for bounded
integer draws. The mixer makes no cryptographic claim.

Limits and measurements
-----------------------

Topology retains a descriptor, empty explicit-array headers, and in
fluctuating mode a sparse repair index. Mutable messages, clause masks and
counters, assignments, queues, and WalkSAT state still scale with the
instance. Connectivity reconstruction costs more than cached explicit lookup.

Tests query the partitioned descriptor and sample the raw fluctuating
matching at N=10^9 and E=12*10^9. The latter test skips the full repair scan.
The scalar CPU solver still requires ``E<u32::MAX`` and ``M<u32::MAX``.
Oversized CLI solves fail before the scan and solver allocation.

Feistel reports include generator identity, degree histogram, resampled
clause count, construction time, and ``topology_bytes``. That field counts
owned topology payload and vector capacity, excluding Arc headers and
allocator overhead.
