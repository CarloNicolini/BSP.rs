# Entropy-regularized BSP experiment

GitHub issue #69. The experiment uses the scalar f64 solver and one decision per
forward move. Release comparisons use checkpoints where native BSP would release
exactly one decision. Native multi-release steps are preserved in trajectories.

## Questions and controls

1. Can a different choice on the same checkpoint reduce decimation loss or
   increase release gain? Measure signed post-minus-pre total SP complexity,
   with the same convergence tolerance and warm start for all choices.
2. Does a second-step continuation estimate add predictive value beyond the
   immediate post-move estimate? Compare realized net complexity through a
   common native continuation and track its failure rate. A lookahead policy
   can accept a worse immediate change to improve the later outcome.
3. Does temperature help independently of lookahead? Compare hard and soft
   choices at identical continuation depth. The exact native rank must be
   reproduced at zero temperature with the complexity terms disabled.

The root shortlist is the first four native-ranked variables. The reference
log score is `log(b)` for fix and `-log(b_at_fix)` for release. Preserve the SP
sign on fixes. The second step uses the next native phase and the first two
native-ranked candidates. Scores are `prior + weight * delta_sigma`, plus the
second-step soft value when enabled. Temperature is in units of that score.
Failed predictions fall back to the native action if no scored candidate is
available. Failure counts and coverage are reported alongside paired deltas.

Pilot formula seeds 1–4. Held-out formula seeds 101–120. Two random 4-SAT
clause densities, 9.65 and 9.85, at N=160. Record both the first and second
eligible checkpoint of each phase after at least four decisions have been made
net (see amendment for the final checkpoint depths). Formula seeds are the resampling unit within each density. A checkpoint is
eligible when native BSP has a nonparamagnetic converged point and is about to
make the relevant single move. Record skipped or unavailable checkpoints.

Use epsilon 1e-6, damping 0.2, up to 512 sweeps, backtrack ratio 0.7. The run
records heuristic outcomes by status. MiniSat independently classifies the
checkpoint and immediate child residuals. A converged SP report and a SAT
classification describe different properties of a residual. The experiment
keeps positive and negative changes and leaves missing deltas missing.

For closed-loop validation, replay native, bias-soft, immediate-hard,
immediate-soft, lookahead-hard and lookahead-soft policies on the same formulas
with a common 60,000-SP-sweep limit, at most 500 moves and 64,000 WalkSAT flips.
Charge speculative sweeps to that limit and report wall time separately.
The pilot is exploratory. Freeze the configuration before held-out runs and
report any deviations from this protocol.

Method references: Marino et al., doi:10.1038/ncomms12996, eq. 8. Geist et al.,
PMLR 97 (2019), 2160–2169. Ng et al., ICML 1999, on telescoping potentials.

## Pilot amendment before held-out data

The first engineering run exhausted the 30,000-sweep cap even for one native
solve. The smoke test was rerun uncapped against stock `Solver::solve`, and the
paired runs use 60,000 sweeps. The first checkpoint rule also sampled near
duplicates at the same decision depth. The analysis uses the first eligible
checkpoint at or above 4 and 12 net decisions in each phase. Add sparse hard
lookahead every 16 eligible moves in each phase to isolate overhead. The
reference prior, complexity coefficient 1, temperature 0.01, shortlist and
continuation width stay fixed for both stages. MiniSat has a 1 CPU-second cap.

In closed loop all policies have the same sweep cap. Multi-release keeps the
native move, and the single-move counterfactuals therefore make no inference
about choosing a release batch. A hard Bellman score can differ from the
native choice at zero temperature when the complexity terms remain enabled.
At positive temperature, exact ties receive equal weight under the uniform
reference. The implementation deliberately applies native tie-breaking at
exactly zero.

The two-step outcome in the analysis uses a common *native* follow-up for
every first action. This measures transfer beyond the hypothetical optimized
branch. Include signed immediate deltas and their coverage separately.
Terminal children use the first-step outcome, because the native continuation
would stop there. Bootstrap intervals are descriptive and computed within
each density by resampling formula seeds. Aggregate trajectory deltas are
conditional on reaching the next SP point and can reflect different depths
or early stopping. Use the matched-checkpoint estimates for move quality.

The cheap continuation calibration predicts the *next* signed complexity change
under native follow-up. Compare zero, a density/phase mean fitted on the pilot,
the exact reversal heuristic `-delta_sigma`, and pilot-fitted linear regression.
Measure held-out absolute error averaged by formula. This calibration conditions
on the presence of two converged SP reports and does not model failures.

After examining the two-step estimates, run the diagnostic replay on the
held-out checkpoint set. Record an immediate undo only when the second
move is the inverse on the same variable and restores the full partial
assignment, including unit implications. Continue with two more native
actions and measure the four-step transfer of each original first-move
policy. This replay is descriptive. It does not refit the original policies.
