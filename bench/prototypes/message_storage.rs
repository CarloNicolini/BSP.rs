//! Storage-only quantization probe: real scalar SP, f64 arithmetic, no decimation.
//! See bench/results/billion_variables_20261029/REPORT.md for scope and commands.
use bsp_rust::{Formula, Options, Solver};

fn binary_round(x: f64, fraction: i32, minimum_exponent: i32) -> f64 {
    if x == 0.0 || x == 1.0 {
        return x;
    }
    let exponent = (x.log2().floor() as i32).max(minimum_exponent);
    let step = 2.0_f64.powi(exponent - fraction);
    ((x / step).round_ties_even() * step).clamp(0.0, 1.0)
}

fn quantize(x: f64, format: &str) -> f64 {
    if let Some(base) = format.strip_suffix("_escape") {
        let y = quantize(x, base);
        return if x > 0.0 && y == 0.0 || x < 1.0 && y == 1.0 {
            x
        } else {
            y
        };
    }
    if let Some(base) = format.strip_suffix("_open") {
        let y = quantize(x, base);
        return if x > 0.0 && x < 1.0 {
            let (lo, hi) = match base {
                "bf16" => (2.0_f64.powi(-133), 1.0 - 2.0_f64.powi(-8)),
                "fp16" => (2.0_f64.powi(-24), 1.0 - 2.0_f64.powi(-11)),
                "unorm16" => (1.0 / 65_535.0, 65_534.0 / 65_535.0),
                _ => unreachable!(),
            };
            y.clamp(lo, hi)
        } else {
            y
        };
    }
    match format {
        "f64" => x,
        "f32" => x as f32 as f64,
        "bf16" => binary_round(x, 7, -126),
        "fp16" => binary_round(x, 10, -14),
        "unorm16" | "unorm12" | "unorm8" => {
            let levels = match format {
                "unorm16" => 65_535.0,
                "unorm12" => 4_095.0,
                _ => 255.0,
            };
            (x * levels).round_ties_even() / levels
        }
        _ => unreachable!(),
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(10_000);
    let sweeps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(80);
    let warmup: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(0);
    assert!(n >= 4 && sweeps > 0);
    assert_eq!(binary_round(1.0 - 1.0 / 1024.0, 7, -126), 1.0);
    assert_eq!(binary_round(1.0 - 1.0 / 8192.0, 10, -14), 1.0);
    assert_eq!(binary_round(2.0_f64.powi(-25), 10, -14), 0.0);
    println!("k,alpha,n,seed,warmup,format,sweeps_completed,failure,initial_new_zeros,initial_new_ones,max_trajectory_abs_error,final_abs_error,final_rmse,final_fraction_error_gt_001,raw_residual,stored_motion,max_escape_fraction");
    for (k, alpha) in [(3, 4.2), (4, 9.8)] {
        for seed in 1..=3 {
            let formula = Formula::random(k, alpha, n, seed).unwrap();
            let options = Options {
                seed: seed + 1000,
                ..Options::default()
            };
            let mut starter = Solver::new(formula.clone(), options).unwrap();
            if warmup > 0 {
                starter.sweep_many(warmup).unwrap();
            }
            let initial = starter.messages().to_vec();
            drop(starter);
            for format in [
                "f64",
                "f32",
                "bf16",
                "fp16",
                "unorm16",
                "unorm12",
                "unorm8",
                "bf16_open",
                "fp16_open",
                "unorm16_open",
                "bf16_escape",
                "fp16_escape",
                "unorm16_escape",
            ] {
                let options = Options {
                    seed: seed + 1000,
                    ..Options::default()
                };
                let mut reference = Solver::new(formula.clone(), options.clone()).unwrap();
                let mut rounded = Solver::new(formula.clone(), options).unwrap();
                reference.set_messages(&initial).unwrap();
                let mut stored: Vec<_> = reference
                    .messages()
                    .iter()
                    .map(|&x| quantize(x, format))
                    .collect();
                let initial_zeros = reference
                    .messages()
                    .iter()
                    .zip(&stored)
                    .filter(|(x, y)| **x > 0.0 && **y == 0.0)
                    .count();
                let initial_ones = reference
                    .messages()
                    .iter()
                    .zip(&stored)
                    .filter(|(x, y)| **x < 1.0 && **y == 1.0)
                    .count();
                rounded.set_messages(&stored).unwrap();
                let mut trajectory: f64 = 0.0;
                let mut completed = 0;
                let mut failure = String::new();
                let mut raw_residual = 0.0;
                let mut motion: f64 = 0.0;
                let escape_fraction = |values: &[f64]| {
                    format.strip_suffix("_escape").map_or(0.0, |base| {
                        values
                            .iter()
                            .filter(|&&x| {
                                let y = quantize(x, base);
                                x > 0.0 && y == 0.0 || x < 1.0 && y == 1.0
                            })
                            .count() as f64
                            / values.len() as f64
                    })
                };
                let mut max_escape = escape_fraction(reference.messages());
                for _ in 0..sweeps {
                    if let Err(status) = reference.sweep() {
                        failure = format!("reference_{status}");
                        break;
                    }
                    match rounded.sweep() {
                        Ok(residual) => raw_residual = residual,
                        Err(status) => {
                            failure = format!("rounded_{status}");
                            break;
                        }
                    }
                    motion = 0.0;
                    max_escape = max_escape.max(escape_fraction(rounded.messages()));
                    for ((dst, &x), &old) in stored
                        .iter_mut()
                        .zip(rounded.messages())
                        .zip(reference.messages())
                    {
                        let next = quantize(x, format);
                        motion = motion.max((next - *dst).abs());
                        *dst = next;
                        trajectory = trajectory.max((next - old).abs());
                    }
                    rounded.set_messages(&stored).unwrap();
                    completed += 1;
                }
                let mut final_error: f64 = 0.0;
                let mut squares = 0.0;
                let mut large = 0;
                for (&x, &y) in reference.messages().iter().zip(&stored) {
                    let error = (x - y).abs();
                    final_error = final_error.max(error);
                    squares += error * error;
                    large += usize::from(error > 0.01);
                }
                let edges = formula.edges() as f64;
                // On failure the reference has advanced one extra sweep; omit
                // final comparison fields rather than compare different times.
                let final_fields = if failure.is_empty() {
                    format!(
                        "{final_error:.9e},{:.9e},{:.9e}",
                        (squares / edges).sqrt(),
                        large as f64 / edges
                    )
                } else {
                    ",,".into()
                };
                println!("{k},{alpha},{n},{seed},{warmup},{format},{completed},{failure},{initial_zeros},{initial_ones},{trajectory:.9e},{final_fields},{raw_residual:.9e},{motion:.9e},{max_escape:.9e}");
            }
        }
    }
}
