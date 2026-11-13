//! Serial, deadline-aware staged campaign. The native solver is the transition model.
use super::*;
use serde::Deserialize;
use std::io::BufRead;
use std::path::PathBuf;

const SWEEPS: u64 = 1_500_000;
const MOVES: usize = 16_000;
const TEMPERATURE: f64 = 0.01;
const INTERVAL: usize = 128;
const OFFSETS: [usize; 3] = [1, 3, 8];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Arm {
    Native,
    BiasSoft,
    NetSoft,
    Predictive,
}

#[derive(Deserialize)]
struct Linear {
    mean: Vec<f64>,
    scale: Vec<f64>,
    intercept: f64,
    weights: Vec<f64>,
}

impl Linear {
    fn predict(&self, features: &[f64]) -> f64 {
        let score = self.intercept
            + self
                .weights
                .iter()
                .enumerate()
                .map(|(i, w)| w * (features[i] - self.mean[i]) / self.scale[i])
                .sum::<f64>();
        1.0 / (1.0 + (-score).exp())
    }
}

#[derive(Deserialize)]
struct Model {
    fix: Linear,
    release: Linear,
}

impl Model {
    fn phase(&self, phase: Phase) -> &Linear {
        match phase {
            Phase::Fix => &self.fix,
            Phase::Release => &self.release,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Job {
    stage: String,
    n: usize,
    alpha: f64,
    formula_seed: u64,
    solver_seed: u64,
    policy_seed: u64,
    arm: Arm,
    output: PathBuf,
    formula: PathBuf,
}

fn seconds(n: usize) -> f64 {
    if n == 640 {
        90.0
    } else {
        180.0
    }
}

fn options(job: &Job) -> Options {
    Options {
        seed: job.solver_seed,
        epsilon: 1e-5,
        damping: 0.2,
        batch_fraction: 1.0 / (job.n + 1) as f64,
        backtrack_ratio: 0.7,
        max_sweeps: 65_536,
        max_steps: MOVES,
        max_flips: 40_000_000,
        timeout_seconds: Some(seconds(job.n)),
        ..Options::default()
    }
}

fn features(
    root: &Solver,
    child: &Solver,
    action: Move,
    before: &SpReport,
    after: &SpReport,
) -> [f64; 6] {
    let n = root.formula.n() as f64;
    [
        action.prior,
        root.decisions.len() as f64 / n,
        after.complexity - before.complexity,
        after.sweeps as f64,
        child.active.len() as f64 / root.formula.m() as f64,
        child.forced.iter().filter(|&&forced| forced).count() as f64 / n,
    ]
}

#[derive(Serialize)]
struct Endpoint {
    status: Status,
    stop: Option<&'static str>,
    moves: usize,
    sweeps: u64,
    flips: u64,
    seconds: f64,
    witness: Option<Vec<i8>>,
}

fn budget(solver: &Solver, spent: u64, moves: usize) -> Option<(Status, &'static str)> {
    if let Err(status) = solver.check_stop() {
        Some((status, "time"))
    } else if spent >= SWEEPS {
        Some((Status::StepLimit, "sweeps"))
    } else if moves >= MOVES {
        Some((Status::StepLimit, "moves"))
    } else {
        None
    }
}

fn stop_kind(status: Status) -> Option<&'static str> {
    match status {
        Status::Timeout => Some("time"),
        Status::StepLimit => Some("sweeps"),
        Status::FlipLimit => Some("flips"),
        _ => None,
    }
}

fn finish(solver: &mut Solver) -> (Status, Option<Vec<i8>>) {
    match solver.walksat() {
        Ok(witness) => {
            assert!(solver.formula.is_satisfied(&witness));
            match solver.check_stop() {
                Ok(()) => (Status::Sat, Some(witness)),
                Err(status) => (status, None),
            }
        }
        Err(status) => (status, None),
    }
}

fn deadline_fork(solver: &Solver, started: Instant, timeout: f64) -> Solver {
    let mut copy = fork(solver);
    copy.started = started;
    copy.options.timeout_seconds = Some(timeout);
    copy
}

#[derive(Serialize)]
struct DepthTrace {
    complexity: [Option<f64>; 3],
    work: [Option<u64>; 3],
    stop: &'static str,
    undo: bool,
    moves: usize,
}

fn depth_trace(
    root: &Solver,
    child: &Solver,
    first: &SpReport,
    phase: Phase,
    spent: &mut u64,
    max_offset: usize,
) -> DepthTrace {
    let mut copy = fork(child);
    let mut point = first.clone();
    let target = root.decisions.len();
    let start = *spent;
    let mut result = DepthTrace {
        complexity: [None; 3],
        work: [None; 3],
        stop: "reached",
        undo: false,
        moves: 0,
    };
    loop {
        if let Some((_, reason)) = budget(&copy, *spent, result.moves) {
            result.stop = reason;
            break;
        }
        let net = copy.decisions.len().saturating_sub(target);
        for (i, &offset) in OFFSETS
            .iter()
            .enumerate()
            .filter(|(_, offset)| **offset <= max_offset)
        {
            if net >= offset && result.complexity[i].is_none() {
                result.complexity[i] = Some(point.complexity);
                result.work[i] = Some(*spent - start);
            }
        }
        if net >= max_offset {
            break;
        }
        let next = next_phase(&copy);
        if terminal(&copy, &point, next) {
            result.stop = if point.paramagnetic || copy.active.is_empty() {
                "handoff"
            } else {
                "negative_complexity"
            };
            break;
        }
        if result.moves == 64 {
            result.stop = "horizon";
            break;
        }
        apply(&mut copy, next, None);
        if result.moves == 0 && next != phase {
            result.undo = copy.assignment == root.assignment;
        }
        result.moves += 1;
        match converge(&mut copy, spent, SWEEPS) {
            Ok(report) => point = report,
            Err(status) => {
                result.stop = match status {
                    Status::NonConverged => "non_converged",
                    Status::Contradiction => "contradiction",
                    Status::Timeout => "time",
                    Status::StepLimit => "sweeps",
                    _ => "solver_error",
                };
                break;
            }
        }
    }
    result
}

fn label(root: &Solver, phase: Phase, action: Move, seed: u64, timeout: f64) -> Endpoint {
    let started = Instant::now();
    let mut solver = deadline_fork(root, started, timeout);
    solver.rng = SmallRng::seed_from_u64(seed);
    apply(&mut solver, phase, Some(action));
    let mut moves = 1;
    let mut sweeps = 0;
    let (status, stop, witness) = loop {
        if let Some((status, stop)) = budget(&solver, sweeps, moves) {
            break (status, Some(stop), None);
        }
        let point = match converge(&mut solver, &mut sweeps, SWEEPS) {
            Ok(point) => point,
            Err(status) => break (status, stop_kind(status), None),
        };
        if point.paramagnetic || solver.active.is_empty() {
            let (status, witness) = finish(&mut solver);
            break (status, stop_kind(status), witness);
        }
        let phase = next_phase(&solver);
        if terminal(&solver, &point, phase) {
            break (Status::NegativeComplexity, None, None);
        }
        apply(&mut solver, phase, None);
        moves += 1;
    };
    Endpoint {
        status,
        stop,
        moves,
        sweeps,
        flips: solver.flips,
        seconds: started.elapsed().as_secs_f64(),
        witness,
    }
}

fn observe(root: &Solver, before: &SpReport, phase: Phase, seed: u64) -> serde_json::Value {
    let timeout = root.options.timeout_seconds.unwrap() / 2.0;
    let branches: Vec<_> = candidates(root, phase, 3).into_iter().map(|action| {
        let mut child = deadline_fork(root, Instant::now(), timeout);
        apply(&mut child, phase, Some(action));
        let mut spent = 0;
        let first = converge(&mut child, &mut spent, SWEEPS);
        let features = first.as_ref().ok().map(|after| features(root, &child, action, before, after));
        let trace = first.as_ref().ok().map(|after| depth_trace(root, &child, after, phase, &mut spent, 8));
        let labels = [seed, seed + 1].map(|seed| label(root, phase, action, seed, timeout));
        serde_json::json!({
            "action": action, "first": first, "features": features, "trace": trace, "labels": labels,
        })
    }).collect();
    serde_json::json!({"phase": phase, "before": before, "depth": root.decisions.len(), "branches": branches})
}

struct Evaluated {
    action: Move,
    child: Solver,
    point: StatusResult<SpReport>,
    first_sweeps: u64,
    net: Option<f64>,
}

struct Search {
    solver: Solver,
    spent: u64,
    speculative: u64,
    cached: Option<StatusResult<SpReport>>,
}

impl Search {
    fn choose(
        &mut self,
        before: &SpReport,
        phase: Phase,
        arm: Arm,
        model: Option<&Model>,
        rng: &mut SmallRng,
    ) {
        let actions = candidates(&self.solver, phase, 3);
        let start = self.spent;
        let mut evaluated = Vec::new();
        if arm != Arm::BiasSoft {
            for &action in &actions {
                let mut child = fork(&self.solver);
                apply(&mut child, phase, Some(action));
                let before_sweeps = self.spent;
                let point = converge(&mut child, &mut self.spent, SWEEPS);
                let first_sweeps = self.spent - before_sweeps;
                let net = if arm == Arm::NetSoft {
                    point.as_ref().ok().and_then(|point| {
                        depth_trace(&self.solver, &child, point, phase, &mut self.spent, 3)
                            .complexity[1]
                    })
                } else {
                    None
                };
                evaluated.push(Evaluated {
                    action,
                    child,
                    point,
                    first_sweeps,
                    net,
                });
            }
        }
        let net = arm == Arm::NetSoft && evaluated.iter().all(|branch| branch.net.is_some());
        let predictive =
            arm == Arm::Predictive && evaluated.iter().all(|branch| branch.point.is_ok());
        let scores: Vec<_> = match arm {
            Arm::NetSoft => evaluated
                .iter()
                .map(|branch| {
                    let sigma = if net {
                        branch.net?
                    } else {
                        branch.point.as_ref().ok()?.complexity
                    };
                    Some(branch.action.prior + sigma - before.complexity)
                })
                .collect(),
            Arm::Predictive if predictive => evaluated
                .iter()
                .map(|branch| {
                    let after = branch.point.as_ref().ok()?;
                    let features =
                        features(&self.solver, &branch.child, branch.action, before, after);
                    Some(
                        branch.action.prior
                            + TEMPERATURE * model.unwrap().phase(phase).predict(&features),
                    )
                })
                .collect(),
            _ => actions.iter().map(|a| Some(a.prior)).collect(),
        };
        let index = sample(&probabilities(&scores, TEMPERATURE), rng);
        if evaluated.is_empty() {
            apply(&mut self.solver, phase, Some(actions[index]));
        } else {
            let chosen = evaluated.swap_remove(index);
            self.speculative += self.spent - start - chosen.first_sweeps;
            self.solver = chosen.child;
            self.cached = Some(chosen.point);
        }
    }
}

fn run(formula: &Formula, job: &Job, model: Option<&Model>) -> serde_json::Value {
    let started = Instant::now();
    let opts = options(job);
    let mut search = Search {
        solver: Solver::new(formula.clone(), opts.clone()).unwrap(),
        spent: 0,
        speculative: 0,
        cached: None,
    };
    search.solver.started = started;
    let mut rng = SmallRng::seed_from_u64(job.policy_seed);
    let mut moves = 0;
    let mut eligible = [0; 2];
    let mut interventions = [0; 2];
    let mut depth_index = [0; 2];
    let depths = [job.n.div_ceil(20), job.n.div_ceil(5)];
    let mut checkpoints = Vec::new();
    let mut diagnostic = std::time::Duration::ZERO;
    let (status, stop, witness) = loop {
        if let Some((status, stop)) = budget(&search.solver, search.spent, moves) {
            break (status, Some(stop), None);
        }
        let result = search
            .cached
            .take()
            .unwrap_or_else(|| converge(&mut search.solver, &mut search.spent, SWEEPS));
        let point = match result {
            Ok(point) => point,
            Err(status) => break (status, stop_kind(status), None),
        };
        let phase = next_phase(&search.solver);
        if point.paramagnetic || search.solver.active.is_empty() {
            let (status, witness) = finish(&mut search.solver);
            break (status, stop_kind(status), witness);
        }
        if terminal(&search.solver, &point, phase) {
            break (Status::NegativeComplexity, None, None);
        }
        let single = phase == Phase::Fix || release_count(&search.solver) == 1;
        let i = phase as usize;
        let depth = search.solver.decisions.len();
        if matches!(job.stage.as_str(), "train" | "evaluate")
            && single
            && depths
                .get(depth_index[i])
                .is_some_and(|&target| depth >= target)
        {
            let begin = Instant::now();
            let label_seed = job.policy_seed ^ ((moves as u64) << 32);
            checkpoints.push(observe(&search.solver, &point, phase, label_seed));
            let elapsed = begin.elapsed();
            diagnostic += elapsed;
            search.solver.started += elapsed;
            depth_index[i] += 1;
            if depth_index == [2, 2] {
                break (Status::StepLimit, Some("checkpoints_complete"), None);
            }
        }
        if single && depth >= 16 {
            eligible[i] += 1;
        }
        if job.arm != Arm::Native && single && depth >= 16 && (eligible[i] - 1) % INTERVAL == 0 {
            search.choose(&point, phase, job.arm, model, &mut rng);
            interventions[i] += 1;
        } else {
            apply(&mut search.solver, phase, None);
        }
        moves += 1;
    };
    let result = Endpoint {
        status,
        stop,
        witness,
        moves,
        sweeps: search.spent,
        flips: search.solver.flips,
        seconds: started.elapsed().saturating_sub(diagnostic).as_secs_f64(),
    };
    serde_json::json!({
        "job": job, "options": opts, "result": result, "speculative_sweeps": search.speculative,
        "interventions": interventions, "checkpoints": checkpoints, "diagnostic_seconds": diagnostic.as_secs_f64(),
    })
}

fn write_atomic(path: &Path, data: impl FnOnce(&mut std::fs::File)) {
    let pending = path.with_extension("tmp");
    let mut file = std::fs::File::create(&pending).unwrap();
    data(&mut file);
    file.sync_all().unwrap();
    std::fs::rename(pending, path).unwrap();
}

#[test]
#[ignore = "opt-in staged soft BSP campaign"]
fn staged_bsp_campaign() {
    let input = std::env::var_os("SOFT_BSP_JOBS").expect("set SOFT_BSP_JOBS");
    let model: Option<Model> = std::env::var_os("SOFT_BSP_MODEL")
        .map(|path| serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap());
    for line in std::io::BufReader::new(std::fs::File::open(input).unwrap()).lines() {
        let job: Job = serde_json::from_str(&line.unwrap()).unwrap();
        if job.output.exists() {
            continue;
        }
        assert!(job.arm != Arm::Predictive || model.is_some());
        let formula = Formula::random(4, job.alpha, job.n, job.formula_seed).unwrap();
        if job.formula.exists() {
            assert_eq!(
                std::fs::read_to_string(&job.formula).unwrap(),
                formula.to_dimacs()
            );
        } else {
            write_atomic(&job.formula, |file| formula.write_dimacs(file).unwrap());
        }
        let result = run(&formula, &job, model.as_ref());
        write_atomic(&job.output, |file| {
            serde_json::to_writer(file, &result).unwrap()
        });
        eprintln!(
            "{} N={} alpha={} seed={} {:?} {} {:.2}s",
            job.stage,
            job.n,
            job.alpha,
            job.formula_seed,
            job.arm,
            result["result"]["status"],
            result["result"]["seconds"].as_f64().unwrap()
        );
    }
}

#[test]
fn staged_soft_limits_and_native_trajectory() {
    let score = [Some(-0.01), Some(-0.02), Some(-0.03)];
    let expected = [1.0, (-1.0_f64).exp(), (-2.0_f64).exp()];
    let norm = expected.iter().sum::<f64>();
    for (p, expected) in probabilities(&score, TEMPERATURE).iter().zip(expected) {
        assert!((p - expected / norm).abs() < 1e-14);
    }
    let shifted = score.map(|value| value.map(|value| value + 0.125));
    for (p, q) in probabilities(&score, TEMPERATURE)
        .iter()
        .zip(probabilities(&shifted, TEMPERATURE))
    {
        assert!((p - q).abs() < 1e-14);
    }
    assert_eq!(
        probabilities(&[Some(0.0), Some(0.0), Some(-1.0)], 0.0),
        [1.0, 0.0, 0.0]
    );
    let formula = Formula::random(4, 9.65, 160, 1).unwrap();
    let job = Job {
        stage: "test".into(),
        n: 160,
        alpha: 9.65,
        formula_seed: 1,
        solver_seed: 23,
        policy_seed: 4001,
        arm: Arm::Native,
        output: PathBuf::new(),
        formula: PathBuf::new(),
    };
    let mut native = Solver::new(formula.clone(), options(&job)).unwrap();
    let mut greedy = fork(&native);
    for _ in 0..32 {
        let native_point = native.converge().unwrap();
        let greedy_point = greedy.converge().unwrap();
        assert_eq!(native_point.complexity, greedy_point.complexity);
        let phase = next_phase(&native);
        if terminal(&native, &native_point, phase) {
            break;
        }
        if phase == Phase::Fix || release_count(&native) == 1 {
            let actions = candidates(&greedy, phase, 3);
            let scores: Vec<_> = actions.iter().map(|a| Some(a.prior)).collect();
            let index = probabilities(&scores, 0.0)
                .iter()
                .position(|&p| p == 1.0)
                .unwrap();
            apply(&mut greedy, phase, Some(actions[index]));
        } else {
            apply(&mut greedy, phase, None);
        }
        apply(&mut native, phase, None);
        assert_eq!(native.assignment, greedy.assignment);
        assert_eq!(native.messages(), greedy.messages());
        assert_eq!(native.saved_eta, greedy.saved_eta);
    }
    let stock = Solver::new(formula.clone(), options(&job)).unwrap().solve();
    let actual = run(&formula, &job, None);
    assert_eq!(actual["result"]["status"], serde_json::json!(stock.status));
    assert_eq!(
        actual["result"]["witness"],
        serde_json::json!(stock.assignment)
    );
    assert_eq!(actual["result"]["sweeps"], stock.sp_sweeps);
    assert_eq!(actual["result"]["flips"], stock.flips);
    assert_eq!(
        actual["result"]["moves"],
        stock
            .steps
            .iter()
            .filter(|s| s.action != Action::Walksat)
            .count()
    );
}
