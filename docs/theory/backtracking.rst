From surveys to Backtracking Survey Propagation
===============================================

Decimation and revisiting decisions
-----------------------------------

Survey-inspired decimation alternates SP fixed points with assignments guided
by the surveys. Assigning a variable removes satisfied clauses and shortens
the remaining ones, changing subsequent cavity information. An early choice
may therefore become inappropriate later. Backtracking SP adds selected
unassignments, allowing a search to revisit decisions [Parisi2003]_.

The later BSP study evaluates this idea on random 3-SAT and 4-SAT, relating
algorithmic performance to the structure of solution clusters [MPR2016]_. Its
empirical thresholds, asymptotic claims, and whitening measurements concern
that study and its implementation. They are not guarantees for this package,
arbitrary CNFs, or GPU precision variants.

The policy implemented in BSP.rs
--------------------------------

For each free variable the current score is

.. math::

   c_i=1-\min(S_i^T,S_i^F).

The forward action selects the highest-score batch, with ascending variable ID
for ties. It assigns True if :math:`S_i^T>S_i^F` and False otherwise. In
particular a tie selects False. This score is not the absolute polarization
:math:`|S_i^T-S_i^F|`; replacing one with the other changes trajectories.

Decision scores remain **frozen while fixed**. An indexed min-heap releases
eligible decisions with the lowest stored score, using variable ID for ties.
It does not recalculate a full conditional survey for every fixed variable.
Variables fixed by unit propagation are kept separate and are not released
by this action. This is an implementation policy, not an exact posterior
ranking or a complete logical backtracking procedure.

Let :math:`D,B` be forward/backward action counters initialized to one. If
:math:`B/D<r`, perform a backward action and increment B; otherwise perform
a forward action and increment D. The parameter r is therefore a scheduling
ratio rather than a probability. Batch sizes can vary, so action ratios and
ratios of individual variable events need not coincide.

With :math:`N_f` free variables and batch fraction f, a forward batch contains
:math:`\min(N_f,\max(1,\lfloor fN_f\rfloor))` variables. The backward budget
is the last forward batch plus pending unit implications, capped by the number
of eligible decisions. Message freeze/restore operations preserve the saved
values on temporarily inactive incidences.

An action cycle
---------------

The implemented control flow is:

1. Check cancellation, timeout, and the action limit.
2. Propagate queued unit clauses; contradictory residuals terminate the attempt.
3. Iterate synchronous SP sweeps to the requested residual tolerance.
4. Recompute variable surveys and complexity from the final messages.
5. If paramagnetic or no active clauses remain, run WalkSAT completion.
6. Otherwise choose the backward/forward action by the ratio above.
7. Before a forward action, reject total estimated complexity below -2.
8. Record the action and repeat, stopping explicitly at failure or budget limits.

The negative-complexity guard is a numerical search heuristic, not a proof that
the formula has no clusters. SP nonconvergence also terminates a solve rather
than triggering an undocumented automatic change of tolerance or a full DPLL
search. A GPU numerical retry changes precision and is reported separately.

Unit implications and manual control
------------------------------------

Fixing a decision can force other variables through unit clauses. The native
solver records these as unit assignments. Releasing a decision does not unwind
the whole implication closure or implement a complete decision-level trail.
In particular, the public ``release`` method refuses unit-forced variables.
Use a fresh solver from the original formula for an independent logical branch.
This matters when interpreting the low-level API as an experimental instrument.

WalkSAT handoff
---------------

WalkSAT fills the remaining variables randomly and repeatedly selects an
unsatisfied active clause. It chooses a free variable using break counts,
with noise when every possible flip has positive break count. Fixed BSP and
unit assignments cannot flip. New restarts randomize only the free variables.
The flip budget is cumulative across restarts [SAO2005]_.

The final assignment is checked against the original normalized formula.
Consequently an incorrect approximate marginal cannot produce an unchecked SAT
claim. It can still cause failure, a poorer search trajectory, or larger runtime.
Correct verification is distinct from complete search and from efficient search.

What the approximation does not establish
-----------------------------------------

A fixed-point residual is a consistency diagnostic for an iteration. It does
not establish exact cluster factorization on a loopy graph. Estimated complexity
does not count configurations exactly. A found solution does not imply uniform
sampling. A finite campaign near a random-SAT transition does not prove an
asymptotic algorithmic threshold. Keep these distinctions when using the Python
bindings for statistical-physics research.
