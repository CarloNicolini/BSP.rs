//! Follow a native continuation to equal net decision depth across candidates.
use super::*;

const SWEEP_LIMIT: u64 = 1_500_000;
const DEPTH: usize = 3;
const HORIZON: usize = 64;
const INTERVAL: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Arm {
    Native,
    Immediate,
    NetHard,
    NetSoft,
}

impl Arm {
    const ALL: [Self; 4] = [Self::Native, Self::Immediate, Self::NetHard, Self::NetSoft];

    fn uses_depth(self) -> bool {
        matches!(self, Self::NetHard | Self::NetSoft)
    }

    fn temperature(self) -> f64 {
        if self == Self::NetSoft {
            0.01
        } else {
            0.0
        }
    }
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum End {
    Reached,
    Handoff,
    Horizon,
    Status(Status),
}

#[derive(Serialize)]
struct Followup {
    end: End,
    complexity: Option<f64>,
    moves: usize,
    undo: bool,
    decisions: usize,
    sweeps: u64,
}

fn follow(
    root: &Solver,
    child: &Solver,
    phase: Phase,
    action: Move,
    first: &SpReport,
    spent: &mut u64,
    limit: u64,
) -> Followup {
    let target = root.decisions.len() + DEPTH;
    let mut probe = fork(child);
    let mut report = first.clone();
    let mut moves = 0;
    let mut undo = false;
    let before = *spent;
    let end = loop {
        if probe.decisions.len() >= target {
            break End::Reached;
        }
        if report.paramagnetic || probe.active.is_empty() {
            break End::Handoff;
        }
        let next = next_phase(&probe);
        if next == Phase::Fix && report.complexity < -2.0 {
            break End::Status(Status::NegativeComplexity);
        }
        if moves == HORIZON {
            break End::Horizon;
        }
        let reverse = moves == 0 && next != phase;
        apply(&mut probe, next, None);
        moves += 1;
        if reverse {
            undo = probe.assignment[action.variable] == root.assignment[action.variable]
                && probe.assignment == root.assignment;
        }
        match converge(&mut probe, spent, limit) {
            Ok(next) => report = next,
            Err(status) => break End::Status(status),
        }
    };
    Followup {
        complexity: (end == End::Reached).then_some(report.complexity),
        end,
        moves,
        undo,
        decisions: probe.decisions.len(),
        sweeps: *spent - before,
    }
}

struct Candidate {
    action: Move,
    child: Solver,
    first: StatusResult<SpReport>,
    continuation: Option<Followup>,
    first_sweeps: u64,
}

fn probe(root: &Solver, phase: Phase, depth: bool, spent: &mut u64, limit: u64) -> Vec<Candidate> {
    candidates(root, phase, 3)
        .into_iter()
        .map(|action| {
            let mut child = fork(root);
            apply(&mut child, phase, Some(action));
            let before = *spent;
            let first = converge(&mut child, spent, limit);
            let first_sweeps = *spent - before;
            let continuation = if depth {
                first
                    .as_ref()
                    .ok()
                    .map(|first| follow(root, &child, phase, action, first, spent, limit))
            } else {
                None
            };
            Candidate {
                action,
                child,
                first,
                continuation,
                first_sweeps,
            }
        })
        .collect()
}

fn policy_weights(branches: &[Candidate], before: f64, arm: Arm) -> Vec<f64> {
    let use_net = arm.uses_depth()
        && branches.iter().all(|branch| {
            branch
                .continuation
                .as_ref()
                .is_some_and(|follow| follow.end == End::Reached)
        });
    let scores = branches
        .iter()
        .map(|branch| {
            if use_net {
                branch.continuation.as_ref()?.complexity
            } else {
                branch.first.as_ref().ok().map(|point| point.complexity)
            }
            .map(|sigma| branch.action.prior + sigma - before)
        })
        .collect::<Vec<_>>();
    probabilities(&scores, arm.temperature())
}

fn observe(root: &Solver, phase: Phase, before: &SpReport) -> serde_json::Value {
    let mut spent = 0;
    let branches = probe(root, phase, true, &mut spent, u64::MAX);
    let policies = Arm::ALL.map(|arm| {
        let weights = if arm == Arm::Native {
            (0..branches.len()).map(|i| f64::from(i == 0)).collect()
        } else {
            policy_weights(&branches, before.complexity, arm)
        };
        serde_json::json!({"arm": arm, "probabilities": weights})
    });
    let branches: Vec<_> = branches
        .iter()
        .map(|branch| {
            serde_json::json!({
                "action": branch.action,
                "first": branch.first,
                "first_sweeps": branch.first_sweeps,
                "continuation": branch.continuation,
            })
        })
        .collect();
    serde_json::json!({
        "phase": phase, "before": before, "decisions": root.decisions.len(),
        "branches": branches, "policies": policies, "sweeps": spent,
    })
}

struct Search {
    solver: Solver,
    spent: u64,
    cached: Option<StatusResult<SpReport>>,
    work: [u64; 2],
}

impl Search {
    fn point(&mut self) -> StatusResult<SpReport> {
        self.cached
            .take()
            .unwrap_or_else(|| converge(&mut self.solver, &mut self.spent, SWEEP_LIMIT))
    }

    fn choose(&mut self, phase: Phase, before: &SpReport, arm: Arm, rng: &mut SmallRng) {
        let start = self.spent;
        let mut branches = probe(
            &self.solver,
            phase,
            arm.uses_depth(),
            &mut self.spent,
            SWEEP_LIMIT,
        );
        let selected = sample(&policy_weights(&branches, before.complexity, arm), rng);
        let child = branches.swap_remove(selected);
        self.work[0] += self.spent - start - child.first_sweeps;
        self.work[1] += 1;
        self.solver = child.child;
        self.cached = Some(child.first);
    }
}

fn run(
    formula: &Formula,
    options: &Options,
    arm: Arm,
    seed: u64,
    collect_checkpoints: bool,
) -> serde_json::Value {
    let mut search = Search {
        solver: Solver::new(formula.clone(), options.clone()).unwrap(),
        spent: 0,
        cached: None,
        work: [0; 2],
    };
    let mut rng = SmallRng::seed_from_u64(seed);
    let depths = [0.025, 0.075, 0.15].map(|f| (f * formula.n() as f64).ceil() as usize);
    let mut checkpoint_index = [0; 2];
    let mut intervals = [0; 2];
    let mut checkpoints = Vec::new();
    let started = Instant::now();
    let mut diagnostic_time = 0.0;
    let mut witness = None;
    let mut moves = 0;
    let mut stop_budget = None;
    let mut high_water = 0;
    let status = loop {
        if search.spent >= SWEEP_LIMIT || moves >= options.max_steps {
            stop_budget = Some(if search.spent >= SWEEP_LIMIT {
                "sweeps"
            } else {
                "actions"
            });
            break Status::StepLimit;
        }
        let report = match search.point() {
            Ok(report) => report,
            Err(status) => {
                if status == Status::StepLimit {
                    stop_budget = Some("sweeps");
                }
                break status;
            }
        };
        let phase = next_phase(&search.solver);
        let decisions = search.solver.decisions.len();
        high_water = high_water.max(decisions);
        if report.paramagnetic || search.solver.active.is_empty() {
            break match search.solver.walksat() {
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
        let single = phase == Phase::Fix || release_count(&search.solver) == 1;
        let i = phase as usize;
        if collect_checkpoints
            && single
            && arm == Arm::Native
            && depths
                .get(checkpoint_index[i])
                .is_some_and(|&d| decisions >= d)
        {
            let started = Instant::now();
            checkpoints.push(observe(&search.solver, phase, &report));
            diagnostic_time += started.elapsed().as_secs_f64();
            checkpoint_index[i] += 1;
        }
        if single && decisions >= 16 {
            intervals[i] += 1;
        }
        if arm != Arm::Native && single && decisions >= 16 && (intervals[i] - 1) % INTERVAL == 0 {
            search.choose(phase, &report, arm, &mut rng);
        } else {
            apply(&mut search.solver, phase, None);
        }
        moves += 1;
    };
    serde_json::json!({
        "arm": arm, "status": status, "stop_budget": stop_budget,
        "sweeps": search.spent, "speculative_sweeps": search.work[0], "interventions": search.work[1],
        "moves": moves, "high_water_decisions": high_water, "flips": search.solver.flips,
        "wall_seconds": started.elapsed().as_secs_f64() - diagnostic_time,
        "diagnostic_seconds": diagnostic_time, "witness": witness, "checkpoints": checkpoints,
    })
}

fn campaign_options(n: usize) -> Options {
    Options {
        seed: 23,
        epsilon: 1e-5,
        damping: 0.2,
        max_sweeps: 16_384,
        batch_fraction: 1.0 / (n + 1) as f64,
        backtrack_ratio: 0.7,
        max_steps: 16_000,
        max_flips: 4_000_000,
        ..Options::default()
    }
}

#[test]
fn net_bsp_native_matches_stock() {
    let formula = Formula::random(4, 9.65, 160, 1).unwrap();
    let options = campaign_options(formula.n());
    let native = Solver::new(formula.clone(), options.clone())
        .unwrap()
        .solve();
    let experiment = run(&formula, &options, Arm::Native, 0, false);
    assert_eq!(experiment["status"], serde_json::json!(native.status));
    assert_eq!(experiment["sweeps"], native.sp_sweeps);
    assert_eq!(experiment["flips"], native.flips);
    assert_eq!(experiment["witness"], serde_json::json!(native.assignment));
}

#[test]
#[ignore = "opt-in net-depth BSP campaign"]
fn net_bsp_campaign() {
    use rayon::prelude::*;
    let directory = std::env::var("SOFT_BSP_OUT").expect("set SOFT_BSP_OUT");
    let directory = Path::new(&directory);
    std::fs::create_dir_all(directory).unwrap();
    let mode = std::env::var("SOFT_BSP_MODE").expect("set SOFT_BSP_MODE");
    let seeds = match mode.as_str() {
        "development" => 211..=214,
        "heldout" => 301..=312,
        _ => panic!("use development or heldout"),
    };
    let jobs: Vec<_> = [640, 1280]
        .into_iter()
        .flat_map(|n| seeds.clone().map(move |seed| (n, seed)))
        .collect();
    let workers: usize = std::env::var("SOFT_BSP_WORKERS").map_or(1, |n| n.parse().unwrap());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    pool.install(|| {
        jobs.par_iter().for_each(|&(n, seed)| {
            let alpha = 9.65;
            let formula = Formula::random(4, alpha, n, seed).unwrap();
            let path = directory.join(format!("formula-{n}-{seed}.cnf"));
            formula
                .write_dimacs(std::fs::File::create(path).unwrap())
                .unwrap();
            let options = campaign_options(n);
            for arm in Arm::ALL {
                let arm_name = serde_json::json!(arm);
                let path = directory.join(format!(
                    "case-{n}-{seed}-{}.json",
                    arm_name.as_str().unwrap()
                ));
                if path.exists() {
                    continue;
                }
                let result = run(&formula, &options, arm, 5_000 + seed, true);
                let line = serde_json::json!({
                    "mode": mode, "n": n, "alpha": alpha, "k": 4, "seed": seed, "options": options,
                    "sweep_limit": SWEEP_LIMIT, "net_depth": DEPTH, "followup_moves": HORIZON,
                    "interval": INTERVAL, "policy_seed": 5_000 + seed, "result": result,
                });
                let pending = path.with_extension("tmp");
                let mut output = std::fs::File::create(&pending).unwrap();
                writeln!(output, "{line}").unwrap();
                output.sync_data().unwrap();
                std::fs::rename(pending, path).unwrap();
                eprintln!(
                    "{mode} N={n} seed={seed} {} {} sweeps={}",
                    result["arm"], result["status"], result["sweeps"]
                );
            }
        })
    });
}

#[test]
#[ignore = "opt-in sensitivity replays for resource-limited solves"]
fn net_bsp_resource_replay() {
    use rayon::prelude::*;
    let directory = std::env::var("SOFT_BSP_OUT").expect("set SOFT_BSP_OUT");
    let directory = Path::new(&directory);
    let mode = std::env::var("SOFT_BSP_REPLAY").expect("set SOFT_BSP_REPLAY");
    let (prefix, status) = match mode.as_str() {
        "flips" => ("replay-flips", "flip_limit"),
        "sp" => ("replay-sp", "non_converged"),
        _ => panic!("use flips or sp"),
    };
    let jobs: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .filter_map(|path| {
            let path = path.unwrap().path();
            let name = path.file_name()?.to_str()?;
            if !name.starts_with("case-") || path.extension()? != "json" {
                return None;
            }
            let destination = path.with_file_name(format!("{prefix}-{name}"));
            if destination.exists() {
                return None;
            }
            let case: serde_json::Value =
                serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
            let eligible = case["result"]["status"] == status
                && (mode == "flips" || case["result"]["arm"] == "native");
            eligible.then_some((destination, case))
        })
        .collect();
    let workers: usize = std::env::var("SOFT_BSP_WORKERS").map_or(1, |n| n.parse().unwrap());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    pool.install(|| {
        jobs.into_par_iter().for_each(|(path, case)| {
            let n = case["n"].as_u64().unwrap() as usize;
            let seed = case["seed"].as_u64().unwrap();
            let formula = Formula::random(4, case["alpha"].as_f64().unwrap(), n, seed).unwrap();
            let arm = match case["result"]["arm"].as_str().unwrap() {
                "native" => Arm::Native,
                "immediate" => Arm::Immediate,
                "net_hard" => Arm::NetHard,
                "net_soft" => Arm::NetSoft,
                _ => unreachable!(),
            };
            let mut options = campaign_options(n);
            if mode == "flips" {
                options.max_flips = 40_000_000;
            } else {
                options.max_sweeps = 65_536;
            }
            let result = run(
                &formula,
                &options,
                arm,
                case["policy_seed"].as_u64().unwrap(),
                false,
            );
            if mode == "flips" {
                for field in [
                    "sweeps",
                    "moves",
                    "high_water_decisions",
                    "speculative_sweeps",
                ] {
                    assert_eq!(
                        case["result"][field], result[field],
                        "replay diverged before WalkSAT"
                    );
                }
            } else {
                assert!(
                    result["moves"].as_u64().unwrap() >= case["result"]["moves"].as_u64().unwrap()
                );
            }
            let pending = path.with_extension("tmp");
            let mut output = std::fs::File::create(&pending).unwrap();
            writeln!(
                output,
                "{}",
                serde_json::json!({
                    "n": n, "seed": seed, "options": options, "result": result,
                })
            )
            .unwrap();
            output.sync_data().unwrap();
            std::fs::rename(pending, path).unwrap();
            eprintln!(
                "resource replay {mode} N={n} seed={seed} {} {}",
                result["arm"], result["status"]
            );
        })
    });
}
