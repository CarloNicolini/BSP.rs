# Staged soft BSP at equal cost

Issue #69 and PR #70. Run the complete campaign with
`python3 experiments/soft_bsp/staged.py OUTPUT_DIRECTORY`. Repeating the
command resumes completed atomic records. Use one campaign per host during
timed work. The test-only Rust driver executes jobs serially. The campaign
requires Cargo and Python 3.10 or later on macOS or Linux, and uses no
Python packages. Check progress with
`python3 experiments/soft_bsp/staged_report.py OUTPUT_DIRECTORY`. Pass
`--plot PATH.png` for the verified-SAT curves after a stage has completed
(requires gnuplot).

## Regime and data splits

For each size 640 and 1280, start with 16 random 4-SAT formulas per density
9.25, 9.35, 9.45 and 9.55. Choose the density nearest 50% native success
among those with 30–70% success, preferring greater density on ties.
If necessary, add one point at the midpoint of the closest consecutive
straddling pair, preferring greater density on ties, or extend by 0.10
in the appropriate direction. Omit a size still without an eligible point.

The frozen seed blocks start at 10000 for calibration, 20000 for the optional
extension, 30000 for training, 40000 for exploration and 50000 for confirmation.
Different sizes or calibration density cells use separate offsets of 1000.
All arms share formula and solver seeds and an independently derived policy
seed. Evaluate the predictor on the exploration formulas, with the training
model frozen. The evaluation replays checkpoint collection independently.

All primary jobs use scalar CPU f64, epsilon 1e-5, damping 0.2, BSP ratio
0.7, single-decision fixes, 65,536 sweeps per fixed point, 1.5M total sweeps,
16,000 moves, 40M WalkSAT flips and 90/180 seconds at N=640/1280. Timing starts
after formula construction and includes solver setup and probes. The
speculative solver copies share the root deadline. Diagnostic checkpoint
replays use their own deadlines. The commit, file hashes, executable hash,
compiler, options, formulas, model and manifests are recorded. Time-limited
results depend on hardware and load.

## Matched states and policy arms

Train on 24 formulas per qualified size. Record the first eligible native
checkpoint in each phase at 5% and 20% fixed decision depth. Probe the
three native-ranked single actions with immediate SP complexity and native
continuation at 1, 3 and 8 additional net decisions, up to 64 follow-up
moves. Track undo and missing endpoints. Label each first move with the
fraction of two verified SAT completions under native continuation with
half the size-specific wall budget, using paired WalkSAT seeds.

Fit phase-specific logistic models, one with log score and depth and one
with four additional first-child features. Standardize on the training
data. Use L2 penalty 0.01 with a free intercept. Include converged
feature vectors in fitting, and retain failed label rollouts as zeros.
The exploration checkpoint evaluation reports Brier score, best-move
selection on non-tied targets, coverage and the net-depth complexity spread.

Explore on 24 independent formulas per qualified size with four arms.
Native uses stock ranking. Bias soft uses temperature 0.01 and the
native log score. Net soft adds the net-depth-3 complexity change,
falling back to immediate evaluation if an endpoint is missing. Predictive
adds 0.01 times estimated SAT probability, falling back to bias-only
sampling if a child cannot be evaluated. All intervention arms use the
same shortlist and temperature, and intervene every 128 eligible single
moves per phase starting at 16 decisions. Reuse the selected child's SP
point when available. Charge all speculative sweeps and time. Compare
SAT witnesses at common time and report cost and stopping cause separately.

## Conditional confirmation

Advance bias soft on at least two net successes against native. Advance
the continuation arm that adds at least two successes against both native
and bias soft. Ties favor lower cumulative run time, then predictive.
If any contrast qualifies, run 160 new formulas per qualified size with
native, bias soft and the qualifying continuation arm, if any. Analyze
paired differences, stratified bootstrap intervals (97.5%) and exact
McNemar tests at 0.025 per planned contrast. A confirmed practical gain
has an estimate of at least ten percentage points and excludes zero at
the stated level. At 320 pairs and 30% expected discordance, the exact
two-sided calculation gives roughly 84% power to detect +10 points.
One qualified size gives 160 pairs and lower precision. Use no
confirmation results to tune the policies.
