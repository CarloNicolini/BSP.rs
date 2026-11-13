# Continuation at fixed net decimation depth

Follow-up to issue #69 and PR #70. CPU scalar f64. Random 4-SAT at N=640
and 1280, compared at alpha 9.65. Calibration used seeds 201–202 at four
densities 9.25, 9.45, 9.65 and 9.8. The largest size had native success
2/2, 2/2, 1/2 and 0/2 respectively. The paired campaign uses development
seeds 211–214 at both sizes, then independent seeds 301–312 at N=640 and
N=1280. Outcomes from all seeds are retained. The seed-level paired
analysis is stratified by N.

Use SP epsilon 1e-5, damping 0.2, 16,384 sweeps per fixed point (32 times the
first pilot), 1.5M total SP sweeps (25 times), 16,000 actions (32 times),
4M WalkSAT flips (62.5 times), ratio 0.7, and single decision fixes.
Formula seed and SP/WalkSAT seed are independent (SP seed 23).
The native complexity guard remains part of the algorithm, and is
reported separately from sweep/action caps and SP nonconvergence.
No wall-clock censoring. A partial campaign can be resumed in the same
directory. Each arm is written atomically before starting the next. Report
end-to-end verified SAT, pairwise discordance,
sweeps and wall time. All speculative work is charged. Include exact statuses.

At a common converged checkpoint, score the first three native-ranked
single moves. Evaluate the post-action SP point, then follow native BSP
until reaching the first converged state with three more decision variables
fixed than at the root, or until a terminal condition or 64 continuation
moves. Units do not count as decisions. Record first-action delta, net-depth
delta, next-move undo, horizon coverage and competing terminal outcomes.
A terminal or capped branch has no net-depth complexity label.
Sample native checkpoints at the first eligible single move of each phase
at or above 2.5%, 7.5% and 15% decision depth.

Closed-loop arms: native BSP, sparse immediate hard, sparse net-depth hard,
sparse net-depth soft (tau=0.01). Intervention occurs every 128 eligible
moves per phase, beginning at 16 decisions, with identical eligibility
rules across arms. Match the original prior log score. Immediate scoring
uses prior + Sigma(child) - Sigma(root). Net-depth scoring uses prior +
Sigma(endpoint) - Sigma(root). If any branch has no net-depth label,
rank all candidates by the immediate score instead. Fix uses the native
SP sign and multi-release actions stay native. Use the chosen first
point as the next committed state and account for all other rollouts.
At tau=0 choose the native-ranked first maximizer. Disabling the complexity
terms too recovers native selection.

The matched-checkpoint primary mechanism outcome is net-depth complexity,
with immediate change as a secondary outcome. SAT at the common
computational cap is the closed-loop outcome. Do not interpret zero
differences on censored branches as observed zeros. Measure bootstrap
intervals by formula and report precision separately for each size.

As a secondary predictive check, fit ridge regression separately by size and
phase on complete development checkpoints. Predict endpoint complexity
relative to the native first action using the candidate's prior log score
and immediate complexity relative to the native first action. Compare to
zero and carrying the immediate effect forward. Measure held-out formula-level
MAE, selection regret and net-depth complexity using only checkpoints with
all three converged labels. This is an offline calibration, with no claim
about its closed-loop SAT rate.

Sensitivity replays are reported separately. For every held-out arm reaching
the 4M WalkSAT limit, replay with 40M flips and verify that the number of
SP sweeps, speculative sweeps, actions and deepest decision depth agree up
to handoff. Changing the flip budget changes the restart allocation.
For native held-out runs that stop at SP nonconvergence, recheck with a
65,536-sweep fixed-point cap and the same epsilon, damping, seeds, global
cap and all other options. Record whether the original stopping point is
passed. A failed replay remains a heuristic outcome.
