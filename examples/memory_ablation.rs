//! Matched-instance benchmark; formula parsing and output are outside solve time.
use bsp_rust::{Backend, ExecutionOptions, Formula, Options, Solver, Status};
use serde_json::json;
use std::{fs, io::Write, path::Path, time::Instant};

fn message_hash(solver: &Solver) -> String {
    let hash = solver
        .messages()
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .fold(0xcbf29ce484222325u64, |h, byte| {
            (h ^ byte as u64).wrapping_mul(0x100000001b3)
        });
    format!("{hash:016x}")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 8 {
        return Err(
            "usage: memory_ablation CNF BACKEND MODE COUNT TIMEOUT SEED OUTPUT_PREFIX".into(),
        );
    }
    let backend = match args[2].as_str() {
        "cpu" => Backend::Cpu,
        "cpu-parallel" => Backend::CpuParallel,
        "metal" => Backend::Metal,
        _ => return Err("backend must be cpu, cpu-parallel or metal".into()),
    };
    let parsed = Instant::now();
    let formula = Formula::from_dimacs(&fs::read_to_string(&args[1])?)?;
    let parse_seconds = parsed.elapsed().as_secs_f64();
    let options = Options {
        seed: args[6].parse()?,
        timeout_seconds: Some(args[5].parse()?),
        ..Options::default()
    };
    let started = Instant::now();
    let mut solver = Solver::with_execution(
        formula.clone(),
        options.clone(),
        ExecutionOptions {
            backend,
            threads: if backend == Backend::CpuParallel {
                8
            } else {
                1
            },
            profile: None,
        },
    )?;
    let setup_seconds = started.elapsed().as_secs_f64();
    let output = Path::new(&args[7]);
    let report = if args[3] == "solve" {
        let result = solver.solve();
        let solve_seconds = started.elapsed().as_secs_f64();
        let verification = Instant::now();
        let verified = result
            .assignment
            .as_ref()
            .is_some_and(|a| formula.is_satisfied(a));
        let verify_seconds = verification.elapsed().as_secs_f64();
        if result.status == Status::Sat && !verified {
            return Err("returned SAT assignment failed verification".into());
        }
        if let Some(values) = &result.assignment {
            let mut file = fs::File::create(output.with_extension("assignment.bin"))?;
            let mut block = [0u8; 8192];
            for values in values.chunks(block.len()) {
                for (dst, &value) in block.iter_mut().zip(values) {
                    *dst = value as u8;
                }
                file.write_all(&block[..values.len()])?;
            }
        }
        fs::write(
            output.with_extension("steps.json"),
            serde_json::to_vec(&result.steps)?,
        )?;
        json!({"mode":"solve", "status":result.status, "verified":verified,
            "solve_seconds":solve_seconds, "verify_seconds":verify_seconds,
            "sp_sweeps":result.sp_sweeps, "flips":result.flips,
            "decimations":result.decimations, "releases":result.releases,
            "unit_assignments":result.unit_assignments, "steps":result.steps.len(),
            "execution":result.execution})
    } else if args[3] == "actions" {
        let count: usize = args[4].parse()?;
        let compute = Instant::now();
        let mut steps = Vec::new();
        let mut terminal = None;
        for _ in 0..count {
            match solver.advance() {
                Ok(step) => steps.push(step),
                Err(status) => {
                    terminal = Some(status);
                    break;
                }
            }
        }
        let seconds = compute.elapsed().as_secs_f64();
        fs::write(
            output.with_extension("steps.json"),
            serde_json::to_vec(&steps)?,
        )?;
        json!({"mode":"actions", "status":terminal.map(|s| s.to_string()).unwrap_or("ok".into()),
            "actions":steps.len(), "compute_seconds":seconds,
            "message_hash":message_hash(&solver),
            "sp_sweeps":steps.last().map(|s| s.sp_sweeps), "execution":solver.execution_report()})
    } else if args[3] == "sweeps" {
        let count: usize = args[4].parse()?;
        let compute = Instant::now();
        let residual = solver.sweep_many(count);
        let seconds = compute.elapsed().as_secs_f64();
        json!({"mode":"sweeps", "status":residual.as_ref().map(|_| "ok").unwrap_or("failed"),
            "error":residual.as_ref().err().map(ToString::to_string), "residual":residual.ok(),
            "sweeps":count, "compute_seconds":seconds,
            "message_hash":message_hash(&solver),
            "checksum":solver.messages().iter().sum::<f64>(), "execution":solver.execution_report()})
    } else {
        return Err("mode must be solve or sweeps".into());
    };
    println!(
        "{}",
        json!({"n":formula.n(), "m":formula.m(), "edges":formula.edges(),
        "parse_seconds":parse_seconds, "setup_seconds":setup_seconds,
        "options":options, "compiler":env!("BSP_RUSTC"), "rustflags":env!("BSP_RUSTFLAGS"),
        "report":report})
    );
    Ok(())
}
