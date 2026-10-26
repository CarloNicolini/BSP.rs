Research references and reading path
====================================

The links below were checked against arXiv records. Read the inference review
for the BP/Bethe perspective, the original SP papers for state-level warnings,
and the two backtracking papers for the historical algorithmic development.
Implementation-specific equations, defaults, and stopping rules in this site
are additionally checked against ``src/solver.rs`` and ``src/walksat.rs``.

.. [ZK2016] Lenka Zdeborová and Florent Krzakala.
   *Statistical physics of inference: Thresholds and algorithms*.
   Advances in Physics 65 (2016).
   `arXiv:1511.02476 <https://arxiv.org/abs/1511.02476>`_.
   A broad pedagogical treatment of message passing, cavity ideas, and inference
   thresholds; it is not a specification of this SAT solver.

.. [MZ2002] Marc Mézard and Riccardo Zecchina.
   *The random K-satisfiability problem: from an analytic solution to an efficient algorithm*.
   Physical Review E 66, 056126 (2002).
   `arXiv:cond-mat/0207194 <https://arxiv.org/abs/cond-mat/0207194>`_.
   The zero-temperature cavity connection between random satisfiability,
   state structure, and algorithm design.

.. [BMZ2005] Alfredo Braunstein, Marc Mézard, and Riccardo Zecchina.
   *Survey propagation: an algorithm for satisfiability*.
   Random Structures & Algorithms 27, 201–226 (2005); preprint first submitted 2002.
   `arXiv:cs/0212002 <https://arxiv.org/abs/cs/0212002>`_.
   Survey messages and survey-inspired decimation for random satisfiability.

.. [BMWZ2003] Alfredo Braunstein, Marc Mézard, Martin Weigt, and Riccardo Zecchina.
   *Constraint Satisfaction by Survey Propagation*.
   Preprint first submitted 2002, revised 2003.
   `arXiv:cond-mat/0212451 <https://arxiv.org/abs/cond-mat/0212451>`_.
   A generic constraint-satisfaction formulation extending the survey viewpoint.

.. [Parisi2003] Giorgio Parisi.
   *A backtracking survey propagation algorithm for K-satisfiability* (2003).
   `arXiv:cond-mat/0308510 <https://arxiv.org/abs/cond-mat/0308510>`_.
   An early proposal to revisit assignments during survey-guided search.

.. [MPR2016] Raffaele Marino, Giorgio Parisi, and Federico Ricci-Tersenghi.
   *The backtracking survey propagation algorithm for solving random K-SAT problems*.
   Nature Communications 7, 12996 (2016); preprint first submitted 2015.
   `arXiv:1508.05117 <https://arxiv.org/abs/1508.05117>`_.
   The BSP study motivating this implementation. Consult its Methods for the
   warning equations and survey-guided/backtracking policy, and distinguish
   its measured thresholds from guarantees about the present software.

.. [SAO2005] Sakari Seitz, Mikko Alava, and Pekka Orponen.
   *Focused Local Search for Random 3-Satisfiability* (2005).
   Journal of Statistical Mechanics: Theory and Experiment, P06006.
   `arXiv:cond-mat/0501707 <https://arxiv.org/abs/cond-mat/0501707>`_.
   Analysis and experiments for focused local search, including WalkSAT.
   This is supporting local-search literature, not the original WalkSAT paper.

Notation and provenance
-----------------------

This site uses Boolean 0/1 values, :math:`\eta` for clause warning surveys,
:math:`q` for the probability of a forced violation in a cavity, and
:math:`S^T,S^F,S^0` for full variable surveys. Research papers use other sign
and message conventions. Translate the satisfying/violating literal before
comparing formulas; matching a symbol name is insufficient.

The TikZ figures are original explanatory diagrams for these docs. They are
schematics with explicit truth and sign conventions, not reproductions of
paper figures, measured clusters, or benchmark results.
