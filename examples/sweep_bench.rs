//! Fixed-sweep throughput on a random k-SAT instance; setup is outside the timers.
//! usage: sweep_bench BACKEND K ALPHA N SEED SWEEPS REPEATS [feistel]
use bsp_rust::{Backend, ExecutionOptions, Formula, Options, Solver};
use serde_json::json;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 8 {
        return Err("usage: sweep_bench BACKEND K ALPHA N SEED SWEEPS REPEATS [feistel]".into());
    }
    let backend: Backend = serde_json::from_value(json!(args[1]))?;
    let (k, alpha, n, seed): (usize, f64, usize, u64) = (
        args[2].parse()?,
        args[3].parse()?,
        args[4].parse()?,
        args[5].parse()?,
    );
    let (sweeps, repeats): (usize, usize) = (args[6].parse()?, args[7].parse()?);
    let feistel = args.get(8).is_some_and(|a| a == "feistel");
    let formula = if feistel {
        Formula::feistel(k, alpha, n, seed)?
    } else {
        Formula::random(k, alpha, n, seed)?
    };
    let execution = ExecutionOptions {
        backend,
        profile: std::env::var_os("PROFILE").map(Into::into),
        ..ExecutionOptions::default()
    };
    let mut solver = Solver::with_execution(formula.clone(), Options::default(), execution)?;
    solver.sweep_many(2).map_err(|s| s.to_string())?;
    let seconds: Vec<f64> = (0..repeats)
        .map(|_| {
            let started = Instant::now();
            solver
                .sweep_many(sweeps)
                .map(|_| started.elapsed().as_secs_f64())
        })
        .collect::<Result<_, _>>()
        .map_err(|s| s.to_string())?;
    let hash = solver
        .messages()
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        });
    solver.write_profile()?;
    if let Some(path) = std::env::var_os("DUMP") {
        let bytes: Vec<u8> = solver
            .messages()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(path, bytes)?;
    }
    let mut sorted = seconds.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        json!({"backend": args[1], "k": k, "alpha": alpha, "n": n, "seed": seed,
            "feistel": feistel, "edges": formula.edges(), "sweeps": sweeps,
            "median_ms_per_sweep": 1e3 * sorted[repeats / 2] / sweeps as f64,
            "min_ms_per_sweep": 1e3 * sorted[0] / sweeps as f64,
            "message_hash": format!("{hash:016x}"),
            "report_backend": solver.execution_report().backend})
    );
    Ok(())
}
