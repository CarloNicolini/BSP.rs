//! Ignored research driver. It uses private solver state to fork exact CPU
//! checkpoints, including the saved messages needed to evaluate a release.
use super::*;
use serde::Serialize;
use std::io::Write;
use std::path::Path;
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Fix,
    Release,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Move {
    variable: usize,
    value: bool,
    prior: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Policy {
    name: &'static str,
    temperature: f64,
    immediate: bool,
    continuation: bool,
}

const WEIGHT: f64 = 1.0;
const TEMPERATURE: f64 = 0.01;
const POLICIES: [Policy; 7] = [
    Policy {
        name: "native",
        temperature: 0.0,
        immediate: false,
        continuation: false,
    },
    Policy {
        name: "bias_soft",
        temperature: TEMPERATURE,
        immediate: false,
        continuation: false,
    },
    Policy {
        name: "immediate_hard",
        temperature: 0.0,
        immediate: true,
        continuation: false,
    },
    Policy {
        name: "immediate_soft",
        temperature: TEMPERATURE,
        immediate: true,
        continuation: false,
    },
    Policy {
        name: "lookahead_hard",
        temperature: 0.0,
        immediate: true,
        continuation: true,
    },
    Policy {
        name: "lookahead_soft",
        temperature: TEMPERATURE,
        immediate: true,
        continuation: true,
    },
    Policy {
        name: "sparse_lookahead_hard",
        temperature: 0.0,
        immediate: true,
        continuation: true,
    },
];

fn fork(solver: &Solver) -> Solver {
    let mut copy = Solver::new(solver.formula.clone(), solver.options.clone()).unwrap();
    copy.started = solver.started;
    copy.assignment.clone_from(&solver.assignment);
    copy.active.clone_from(&solver.active);
    copy.free.clone_from(&solver.free);
    copy.forced.clone_from(&solver.forced);
    copy.masks.clone_from(&solver.masks);
    copy.true_count.clone_from(&solver.true_count);
    copy.degree.clone_from(&solver.degree);
    copy.decisions.clone_from(&solver.decisions);
    copy.eta.clone_from(&solver.eta);
    copy.saved_eta.clone_from(&solver.saved_eta);
    copy.surveys.clone_from(&solver.surveys);
    copy.cert.clone_from(&solver.cert);
    copy.unit_queue.clone_from(&solver.unit_queue);
    copy.last_batch = solver.last_batch;
    copy.pending_units = solver.pending_units;
    copy.dec_moves = solver.dec_moves;
    copy.back_moves = solver.back_moves;
    copy.rng = solver.rng.clone();
    copy
}

fn next_phase(solver: &Solver) -> Phase {
    if solver.back_moves as f64 / (solver.dec_moves as f64) < solver.options.backtrack_ratio {
        Phase::Release
    } else {
        Phase::Fix
    }
}

fn release_count(solver: &Solver) -> usize {
    (solver.last_batch + solver.pending_units).min(solver.decisions.len())
}

fn candidates(solver: &Solver, phase: Phase, count: usize) -> Vec<Move> {
    let mut moves: Vec<_> = (0..solver.formula.n())
        .filter(|&v| match phase {
            Phase::Fix => solver.assignment[v] < 0,
            Phase::Release => solver.assignment[v] >= 0 && !solver.forced[v],
        })
        .map(|v| Move {
            variable: v,
            value: solver.surveys[v][0] > solver.surveys[v][1],
            prior: match phase {
                Phase::Fix => solver.cert[v].ln(),
                Phase::Release => -solver.cert[v].ln(),
            },
        })
        .collect();
    moves.sort_unstable_by(|a, b| {
        // Compare stored biases directly so rounding in log cannot break ties.
        let order = solver.cert[a.variable].total_cmp(&solver.cert[b.variable]);
        let order = if phase == Phase::Fix {
            order.reverse()
        } else {
            order
        };
        order.then(a.variable.cmp(&b.variable))
    });
    moves.truncate(count);
    moves
}

fn apply(solver: &mut Solver, phase: Phase, action: Option<Move>) {
    match phase {
        Phase::Fix => {
            solver.dec_moves += 1;
            if let Some(action) = action {
                solver.fix(action.variable, action.value).unwrap();
                solver.last_batch = 1;
                solver.pending_units = 0;
                solver.decimations += 1;
            } else {
                solver.decimate();
            }
        }
        Phase::Release => {
            solver.back_moves += 1;
            if let Some(action) = action {
                solver.release(action.variable).unwrap();
                solver.pending_units = 0;
                solver.releases += 1;
            } else {
                solver.backtrack();
            }
        }
    }
}

fn terminal(solver: &Solver, report: &SpReport, phase: Phase) -> bool {
    report.paramagnetic
        || solver.active.is_empty()
        || (phase == Phase::Fix && report.complexity < -2.0)
}

fn converge(solver: &mut Solver, spent: &mut u64, limit: u64) -> StatusResult<SpReport> {
    let remaining = limit.saturating_sub(*spent);
    if remaining == 0 {
        return Err(Status::StepLimit);
    }
    let max_sweeps = solver.options.max_sweeps;
    solver.options.max_sweeps = max_sweeps.min(remaining as usize);
    let before = solver.sp_sweeps;
    let result = solver.converge();
    let used = solver.sp_sweeps - before;
    *spent += used;
    solver.options.max_sweeps = max_sweeps;
    if matches!(result, Err(Status::NonConverged)) && remaining < max_sweeps as u64 {
        return Err(Status::StepLimit);
    }
    result
}

// Stable weighted log-sum-exp with a uniform prior. Keep the original
// denominator when a branch fails, equivalent to zero weight for that branch.
fn soft_value(scores: &[Option<f64>], temperature: f64) -> Option<f64> {
    let best = scores.iter().flatten().copied().max_by(f64::total_cmp)?;
    if temperature == 0.0 {
        return Some(best);
    }
    let weights: f64 = scores
        .iter()
        .flatten()
        .map(|q| ((q - best) / temperature).exp())
        .sum();
    Some(best + temperature * (weights / scores.len() as f64).ln())
}

fn probabilities(scores: &[Option<f64>], temperature: f64) -> Vec<f64> {
    let Some(best) = scores.iter().flatten().copied().max_by(f64::total_cmp) else {
        return (0..scores.len()).map(|i| f64::from(i == 0)).collect();
    };
    if temperature == 0.0 {
        let first = scores
            .iter()
            .position(|score| *score == Some(best))
            .unwrap();
        return (0..scores.len()).map(|i| f64::from(i == first)).collect();
    }
    let weights: Vec<_> = scores
        .iter()
        .map(|score| score.map_or(0.0, |q| ((q - best) / temperature).exp()))
        .collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|weight| weight / total).collect()
}

fn sample(probabilities: &[f64], rng: &mut SmallRng) -> usize {
    let mut u = rng.random::<f64>();
    for (index, &p) in probabilities.iter().enumerate() {
        if u < p {
            return index;
        }
        u -= p;
    }
    probabilities.iter().rposition(|&p| p > 0.0).unwrap()
}

struct Branch {
    action: Move,
    solver: Solver,
    report: StatusResult<SpReport>,
    second_scores: Vec<Option<f64>>,
    native_second: Option<StatusResult<SpReport>>,
    immediate_undo: bool,
    native_fourth: Option<StatusResult<SpReport>>,
}

fn evaluate(
    solver: &Solver,
    phase: Phase,
    moves: &[Move],
    depth: bool,
    diagnostic: bool,
    spent: &mut u64,
    limit: u64,
) -> Vec<Branch> {
    moves
        .iter()
        .map(|&action| {
            let mut child = fork(solver);
            let root_variable = action.variable;
            let root_phase = phase;
            apply(&mut child, phase, Some(action));
            let report = converge(&mut child, spent, limit);
            let mut second_scores = Vec::new();
            let mut native_second = None;
            let mut immediate_undo = false;
            let mut native_fourth = None;
            if let Ok(first) = &report {
                let phase = next_phase(&child);
                if depth && !terminal(&child, first, phase) {
                    let next_moves: Vec<_> =
                        if phase == Phase::Release && release_count(&child) != 1 {
                            vec![None]
                        } else {
                            candidates(&child, phase, 2).into_iter().map(Some).collect()
                        };
                    for action in next_moves {
                        let mut grandchild = fork(&child);
                        apply(&mut grandchild, phase, action);
                        let second = converge(&mut grandchild, spent, limit);
                        second_scores.push(second.as_ref().ok().map(|report| {
                            action.map_or(0.0, |a| a.prior)
                                + WEIGHT * (report.complexity - first.complexity)
                        }));
                        if native_second.is_none() {
                            if diagnostic {
                                let v = next_moves_variable(&child, phase, action);
                                immediate_undo = v == Some(root_variable)
                                    && phase != root_phase
                                    && grandchild.assignment == solver.assignment;
                                native_fourth = Some(native_followup(
                                    &mut grandchild,
                                    second.clone(),
                                    spent,
                                    limit,
                                ));
                            }
                            native_second = Some(second);
                        }
                    }
                }
            }
            Branch {
                action,
                solver: child,
                report,
                second_scores,
                native_second,
                immediate_undo,
                native_fourth,
            }
        })
        .collect()
}

fn next_moves_variable(solver: &Solver, phase: Phase, action: Option<Move>) -> Option<usize> {
    action.map(|m| m.variable).or_else(|| {
        if phase == Phase::Release && release_count(solver) == 1 {
            solver.decisions.peek().map(|v| v as usize)
        } else {
            None
        }
    })
}

fn native_followup(
    solver: &mut Solver,
    mut result: StatusResult<SpReport>,
    spent: &mut u64,
    limit: u64,
) -> StatusResult<SpReport> {
    for _ in 0..2 {
        let report = result.as_ref().map_err(|&status| status)?;
        let phase = next_phase(solver);
        if terminal(solver, report, phase) {
            break;
        }
        apply(solver, phase, None);
        result = converge(solver, spent, limit);
    }
    result
}

fn weights(moves: &[Move], branches: &[Branch], before: &SpReport, policy: Policy) -> Vec<f64> {
    let scores: Vec<_> = if policy.immediate {
        branches
            .iter()
            .map(|branch| {
                let child = branch.report.as_ref().ok()?;
                let continuation = if policy.continuation && !branch.second_scores.is_empty() {
                    soft_value(&branch.second_scores, policy.temperature)?
                } else {
                    0.0
                };
                Some(
                    branch.action.prior
                        + WEIGHT * (child.complexity - before.complexity)
                        + continuation,
                )
            })
            .collect()
    } else {
        moves.iter().map(|action| Some(action.prior)).collect()
    };
    probabilities(&scores, policy.temperature)
}

fn sat_oracle(solver: &Solver, input: &Path) -> Option<&'static str> {
    let executable = std::env::var_os("SOFT_BSP_MINISAT")?;
    solver
        .residual_formula()
        .write_dimacs(std::fs::File::create(input).unwrap())
        .unwrap();
    let output = Command::new(executable)
        .args(["-verb=0", "-cpu-lim=1"])
        .arg(input)
        .output()
        .unwrap();
    Some(match output.status.code() {
        Some(10) => "sat",
        Some(20) => "unsat",
        Some(0) => "unknown",
        _ => panic!("MiniSat failed: {:?}", output),
    })
}

fn checkpoint(
    solver: &Solver,
    report: &SpReport,
    phase: Phase,
    input: &Path,
    diagnostic: bool,
) -> serde_json::Value {
    let moves = candidates(solver, phase, 4);
    let mut spent = 0;
    let mut native = fork(solver);
    let mut zero = fork(solver);
    apply(&mut native, phase, None);
    let index = weights(&moves, &[], report, POLICIES[0])
        .iter()
        .position(|&p| p == 1.0)
        .unwrap();
    apply(&mut zero, phase, Some(moves[index]));
    assert_eq!(native.assignment, zero.assignment);
    assert_eq!(native.messages(), zero.messages());
    assert_eq!(native.saved_eta, zero.saved_eta);
    assert_eq!(native.pending_units, zero.pending_units);
    assert_eq!(native.last_batch, zero.last_batch);
    let branches = evaluate(
        solver,
        phase,
        &moves,
        true,
        diagnostic,
        &mut spent,
        u64::MAX,
    );
    let outcomes: Vec<_> = branches
        .iter()
        .map(|branch| {
            serde_json::json!({
                "action": branch.action,
                "sp": branch.report,
                "native_second": branch.native_second,
                "immediate_undo": diagnostic.then_some(branch.immediate_undo),
                "native_fourth": branch.native_fourth,
                "second_scores": branch.second_scores,
                "oracle": if matches!(branch.report, Err(Status::Contradiction)) { Some("unsat") }
                          else { sat_oracle(&branch.solver, input) },
            })
        })
        .collect();
    let policies: Vec<_> = POLICIES
        .iter()
        .map(|&policy| {
            serde_json::json!({
                "name": policy.name,
                "probabilities": weights(&moves, &branches, report, policy),
            })
        })
        .collect();
    serde_json::json!({
        "phase": phase, "before": report, "oracle": sat_oracle(solver, input),
        "outcomes": outcomes, "policies": policies, "probe_sweeps": spent,
        "decisions": solver.decisions.len(), "zero_limit_matches": true,
    })
}

#[derive(Default, Serialize)]
struct Deltas {
    sum: f64,
    count: usize,
}

impl Deltas {
    fn push(&mut self, delta: f64) {
        self.sum += delta;
        self.count += 1;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProbeKind {
    Comparison,
    ReversalDiagnostic,
}

fn trajectory(
    formula: &Formula,
    options: &Options,
    policy: Policy,
    seed: u64,
    directory: &Path,
    probe_kind: Option<ProbeKind>,
    limit: u64,
) -> serde_json::Value {
    let mut solver = Solver::new(formula.clone(), options.clone()).unwrap();
    let mut rng = SmallRng::seed_from_u64(seed);
    let started = Instant::now();
    let mut oracle_seconds = 0.0;
    let mut spent = 0;
    let mut checkpoints = Vec::new();
    let mut checkpoint_count = [0; 2];
    let mut eligible = [0; 2];
    let mut deltas = [Deltas::default(), Deltas::default()];
    let mut pending = None;
    let mut cached = None;
    let mut steps = Vec::new();
    let mut witness = None;
    let status = loop {
        let result = cached
            .take()
            .unwrap_or_else(|| converge(&mut solver, &mut spent, limit));
        let report = match result {
            Ok(report) => report,
            Err(status) => break status,
        };
        if let Some((phase, complexity)) = pending.take() {
            deltas[phase as usize].push(report.complexity - complexity);
        }
        let phase = next_phase(&solver);
        if steps.len() >= options.max_steps {
            break Status::StepLimit;
        }
        if report.paramagnetic || solver.active.is_empty() {
            break match solver.walksat() {
                Ok(values) => {
                    assert!(formula.is_satisfied(&values));
                    witness = Some(values);
                    Status::Sat
                }
                Err(status) => status,
            };
        }
        if phase == Phase::Fix && report.complexity < -2.0 {
            break Status::NegativeComplexity;
        }
        let single = phase == Phase::Fix || release_count(&solver) == 1;
        let phase_index = phase as usize;
        let checkpoint_depth = [4, 12].get(checkpoint_count[phase_index]);
        if probe_kind.is_some()
            && single
            && checkpoint_depth.is_some_and(|&depth| solver.decisions.len() >= depth)
        {
            let start_probe = Instant::now();
            checkpoints.push(checkpoint(
                &solver,
                &report,
                phase,
                directory,
                probe_kind == Some(ProbeKind::ReversalDiagnostic),
            ));
            oracle_seconds += start_probe.elapsed().as_secs_f64();
            checkpoint_count[phase as usize] += 1;
        }
        if single {
            eligible[phase_index] += 1;
        }
        let probe = policy.name != "sparse_lookahead_hard"
            || (single && (eligible[phase_index] - 1) % 16 == 0);
        let action = if single && policy.name != "native" && probe {
            let moves = candidates(&solver, phase, 4);
            let mut branches = if policy.immediate {
                evaluate(
                    &solver,
                    phase,
                    &moves,
                    policy.continuation,
                    false,
                    &mut spent,
                    limit,
                )
            } else {
                Vec::new()
            };
            if spent >= limit {
                break Status::StepLimit;
            }
            let index = sample(&weights(&moves, &branches, &report, policy), &mut rng);
            if policy.immediate {
                let branch = branches.swap_remove(index);
                // Preserve committed counters and RNG. Branches only execute one
                // transition and then converge, leaving the WalkSAT RNG intact.
                solver = branch.solver;
                cached = Some(branch.report);
            } else {
                apply(&mut solver, phase, Some(moves[index]));
            }
            Some(moves[index].variable)
        } else {
            let action = if !single {
                None
            } else if phase == Phase::Release {
                solver.decisions.peek().map(|v| v as usize)
            } else {
                None
            };
            apply(&mut solver, phase, None);
            action
        };
        steps.push((phase, action, report.complexity, spent));
        pending = Some((phase, report.complexity));
    };
    serde_json::json!({
        "policy": policy, "status": status, "steps": steps, "sweeps": spent,
        "wall_seconds": started.elapsed().as_secs_f64() - oracle_seconds,
        "deltas": deltas, "checkpoints": checkpoints,
        "partial_assignment": solver.assignment(), "witness": witness,
        "flips": solver.flips, "eligible": eligible,
    })
}

#[test]
fn soft_policy_zero_and_uniform_limits() {
    let scores = [Some(-0.1), Some(-0.1), Some(-0.3)];
    assert_eq!(probabilities(&scores, 0.0), [1.0, 0.0, 0.0]);
    assert_eq!(probabilities(&scores, 1e-9), [0.5, 0.5, 0.0]);
    assert_eq!(soft_value(&scores, 0.0), Some(-0.1));
    assert!((soft_value(&scores, 1e-9).unwrap() + 0.1).abs() < 1e-9);
    assert_eq!(soft_value(&[Some(-1.0); 4], 0.01), Some(-1.0));
    assert_eq!(probabilities(&[None, None], 0.01), [1.0, 0.0]);
    let temperature = 0.1;
    let biases: [f64; 3] = [0.99, 0.95, 0.8];
    let actual = probabilities(&biases.map(|b| Some(b.ln())), temperature);
    let expected = biases.map(|b| b.powf(1.0 / temperature));
    let total: f64 = expected.iter().sum();
    for (p, mass) in actual.iter().zip(expected) {
        assert!((p - mass / total).abs() < 1e-12);
    }
}

#[test]
#[ignore = "opt-in CPU research campaign"]
fn soft_bsp_campaign() {
    let directory = std::env::var("SOFT_BSP_OUT").expect("set SOFT_BSP_OUT");
    let directory = Path::new(&directory);
    std::fs::create_dir_all(directory).unwrap();
    let mode = std::env::var("SOFT_BSP_MODE").expect("set SOFT_BSP_MODE");
    let seeds = match mode.as_str() {
        "smoke" => 1..=1,
        "pilot" => 1..=4,
        "heldout" | "diagnostic" => 101..=120,
        _ => panic!("use smoke, pilot, heldout or diagnostic"),
    };
    let n = 160;
    let options = Options {
        seed: 23,
        epsilon: 1e-6,
        damping: 0.2,
        max_sweeps: 512,
        batch_fraction: 1.0 / (n + 1) as f64,
        backtrack_ratio: 0.7,
        max_steps: 500,
        max_flips: 64_000,
        ..Options::default()
    };
    let mut output = std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(directory.join("cases.jsonl"))
        .unwrap();
    use rayon::prelude::*;
    let jobs: Vec<_> = [9.65, 9.85]
        .into_iter()
        .flat_map(|alpha| seeds.clone().map(move |seed| (alpha, seed)))
        .collect();
    let workers: usize = std::env::var("SOFT_BSP_WORKERS").map_or(1, |n| n.parse().unwrap());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    let limit = if mode == "smoke" { u64::MAX } else { 60_000 };
    let output = std::sync::Mutex::new(&mut output);
    pool.install(|| {
        jobs.par_iter().for_each(|&(alpha, seed)| {
            let formula = Formula::random(4, alpha, n, seed).unwrap();
            let input = directory.join(format!("oracle-{alpha}-{seed}.cnf"));
            let original = directory.join(format!("formula-{alpha}-{seed}.cnf"));
            formula
                .write_dimacs(std::fs::File::create(original).unwrap())
                .unwrap();
            let mut runs = Vec::new();
            for &policy in &POLICIES[..if mode == "smoke" || mode == "diagnostic" {
                1
            } else {
                POLICIES.len()
            }] {
                let run = trajectory(
                    &formula,
                    &options,
                    policy,
                    seed + 5000,
                    &input,
                    match (policy.name, mode.as_str()) {
                        ("native", "diagnostic") => Some(ProbeKind::ReversalDiagnostic),
                        ("native", "pilot" | "heldout") => Some(ProbeKind::Comparison),
                        _ => None,
                    },
                    limit,
                );
                runs.push(run);
            }
            if mode == "smoke" {
                let native = Solver::new(formula.clone(), options.clone())
                    .unwrap()
                    .solve();
                assert_eq!(serde_json::json!(native.status), runs[0]["status"]);
                assert_eq!(
                    serde_json::json!(native.partial_assignment),
                    runs[0]["partial_assignment"]
                );
                let steps = runs[0]["steps"].as_array().unwrap();
                let native_steps = native
                    .steps
                    .iter()
                    .filter(|step| step.action != Action::Walksat);
                assert_eq!(native_steps.clone().count(), steps.len());
                for (expected, observed) in native_steps.zip(steps) {
                    assert_eq!(
                        expected.complexity.to_bits(),
                        observed[2].as_f64().unwrap().to_bits()
                    );
                    assert_eq!(expected.sp_sweeps, observed[3].as_u64().unwrap());
                    assert_eq!(expected.action == Action::Decimate, observed[0] == "fix");
                }
            }
            let mut output = output.lock().unwrap();
            writeln!(
                output,
                "{}",
                serde_json::json!({
                    "mode": mode, "n": n, "k": 4, "alpha": alpha, "seed": seed,
                    "options": options, "policy_seed": seed + 5000, "sweep_limit": limit,
                    "policies": runs,
                })
            )
            .unwrap();
            output.sync_data().unwrap();
            eprintln!("{mode}: completed alpha={alpha}, seed={seed}");
        })
    });
}

#[path = "net_driver.rs"]
mod net;

#[path = "staged_driver.rs"]
mod staged;
