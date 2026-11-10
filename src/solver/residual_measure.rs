//! Sweeps against the residual queue on random 3-SAT, alpha = 4.2.
//! Run with `cargo test --release residual_measure -- --ignored --nocapture`.
use super::*;

const N: usize = 50_000;
const ALPHA: f64 = 4.2;
const SEEDS: [u64; 3] = [1, 7, 19];

/// One BSP round, converged by both schedules from the same state.
/// Index 0 is the sweep schedule, index 1 the residual queue.
struct Round {
    fixed: f64,
    sweeps: usize,
    work: [u64; 2],
    seconds: [f64; 2],
    complexity_gap: f64,
    overlap: f64,
}

/// The next decimation batch under `cert`, with the ordering of `decimate`.
fn batch(s: &Solver, cert: &[f64]) -> Vec<u32> {
    let mut free: Vec<u32> = s.free.ones().map(|v| v as u32).collect();
    let k = ((s.options.batch_fraction * free.len() as f64) as usize).clamp(1, free.len());
    let order = |&a: &u32, &b: &u32| {
        cert[b as usize]
            .total_cmp(&cert[a as usize])
            .then_with(|| a.cmp(&b))
    };
    if k < free.len() {
        free.select_nth_unstable_by(k, order);
    }
    free.truncate(k);
    free.sort_unstable();
    free
}

/// Follow the sweep trajectory. Before each sweep convergence, converge the
/// same state with the queue, record it, and restore the messages.
fn measure(seed: u64) -> Vec<Round> {
    let formula = Formula::random(3, ALPHA, N, seed).unwrap();
    let options = Options {
        seed,
        ..Options::default()
    };
    let mut s = Solver::new(formula, options).unwrap();
    s.enable_residual_schedule();
    let mut rounds = Vec::new();
    while rounds.len() < s.options.max_steps && s.propagate_inner().is_ok() {
        let snapshot = [&s.eta, &s.saved_eta, &s.q, &s.plus, &s.minus].map(|v| v.clone());
        let before = s.residual.as_ref().unwrap().incidences;
        let started = Instant::now();
        let queued = s.residual_converge();
        let queue_seconds = started.elapsed().as_secs_f64();
        let queue_work = s.residual.as_ref().unwrap().incidences - before;
        let queued_cert = s.cert.clone();
        [s.eta, s.saved_eta, s.q, s.plus, s.minus] = snapshot;

        let started = Instant::now();
        let Ok(report) = s.sweep_converge() else {
            break;
        };
        let sweep_seconds = started.elapsed().as_secs_f64();
        if s.resume_residual_queue().is_err() {
            break;
        }
        let per_sweep = 2 * s.free.ones().map(|v| u64::from(s.degree[v])).sum::<u64>();
        let (complexity_gap, overlap) = match queued {
            Ok(q) => {
                let swept = batch(&s, &s.cert);
                let shared = batch(&s, &queued_cert)
                    .iter()
                    .filter(|v| swept.binary_search(v).is_ok())
                    .count();
                (
                    (q.complexity - report.complexity).abs(),
                    shared as f64 / swept.len() as f64,
                )
            }
            Err(_) => (f64::NAN, f64::NAN),
        };
        rounds.push(Round {
            fixed: 1.0 - s.free.len() as f64 / N as f64,
            sweeps: report.sweeps,
            work: [report.sweeps as u64 * per_sweep, queue_work],
            seconds: [sweep_seconds, queue_seconds],
            complexity_gap,
            overlap,
        });

        if report.paramagnetic || s.active.is_empty() {
            break;
        }
        if (s.back_moves as f64) / (s.dec_moves as f64) < s.options.backtrack_ratio {
            s.back_moves += 1;
            s.backtrack_inner();
        } else {
            s.dec_moves += 1;
            if report.complexity < -2.0 {
                break;
            }
            s.decimate_inner();
        }
    }
    rounds
}

fn summarize(seed: u64, rounds: &[Round]) -> String {
    let mut text = format!(
        "seed {seed}: {} rounds, final fixed fraction {:.3}\n\
         fixed   rounds  sweeps/round  sweep-work/round  queue-work/round  work-ratio  time-ratio  max|dSigma|  min-overlap  queue-failures\n",
        rounds.len(),
        rounds.last().map_or(0.0, |r| r.fixed),
    );
    for decile in 0..10 {
        let group: Vec<&Round> = rounds
            .iter()
            .filter(|r| ((r.fixed * 10.0) as usize).min(9) == decile)
            .collect();
        if group.is_empty() {
            continue;
        }
        let count = group.len() as f64;
        let sum = |value: &dyn Fn(&Round) -> f64| group.iter().map(|r| value(r)).sum::<f64>();
        let ok: Vec<&&Round> = group.iter().filter(|r| !r.overlap.is_nan()).collect();
        text += &format!(
            "{:.1}-{:.1}  {:>6}  {:>12.2}  {:>16.0}  {:>16.0}  {:>10.4}  {:>10.4}  {:>11.2e}  {:>11.3}  {:>14}\n",
            decile as f64 / 10.0,
            (decile + 1) as f64 / 10.0,
            group.len(),
            sum(&|r| r.sweeps as f64) / count,
            sum(&|r| r.work[0] as f64) / count,
            sum(&|r| r.work[1] as f64) / count,
            sum(&|r| r.work[1] as f64) / sum(&|r| r.work[0] as f64),
            sum(&|r| r.seconds[1]) / sum(&|r| r.seconds[0]),
            ok.iter().map(|r| r.complexity_gap).fold(0.0, f64::max),
            ok.iter().map(|r| r.overlap).fold(1.0, f64::min),
            group.len() - ok.len(),
        );
    }
    text
}

#[test]
#[ignore = "measurement; minutes in release mode"]
fn residual_measure_rounds() {
    std::thread::scope(|scope| {
        let handles = SEEDS.map(|seed| scope.spawn(move || summarize(seed, &measure(seed))));
        for handle in handles {
            println!("{}", handle.join().unwrap());
        }
    });
}

#[test]
#[ignore = "measurement; minutes in release mode"]
fn residual_measure_solves() {
    std::thread::scope(|scope| {
        let handles = SEEDS.map(|seed| {
            scope.spawn(move || {
                let formula = Formula::random(3, ALPHA, N, seed).unwrap();
                ["sweep", "residual"].map(|schedule| {
                    let options = Options {
                        seed,
                        ..Options::default()
                    };
                    let mut s = Solver::new(formula.clone(), options).unwrap();
                    if schedule == "residual" {
                        s.enable_residual_schedule();
                    }
                    let result = s.solve();
                    let verified = result
                        .assignment
                        .as_ref()
                        .is_some_and(|a| formula.is_satisfied(a));
                    format!(
                        "seed {seed} {schedule:>8}: {:?}, verified {verified}, {:.2} s, {} steps, {} sweeps",
                        result.status,
                        result.wall_seconds,
                        result.steps.len(),
                        result.sp_sweeps,
                    )
                })
            })
        });
        for handle in handles {
            for line in handle.join().unwrap() {
                println!("{line}");
            }
        }
    });
}
