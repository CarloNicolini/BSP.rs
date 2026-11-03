mod cli;

use bsp_rust::{ExecutionOptions, Formula, Options, RunClock, RunPhase, Solver, Status};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::{env, fs, process::ExitCode};

fn run() -> Result<ExitCode, Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut options = Options::default();
    let mut execution = ExecutionOptions::default();
    let mut source = None;
    let mut generate = None;
    let mut sweeps = None;
    let mut messages = None;
    let mut dump_messages = false;
    let mut export = None;
    let mut output = None;
    let mut progress = None;
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        if flag == "--help" || flag == "-h" {
            print!("{}", cli::help::text(std::io::stdout().is_terminal()));
            return Ok(ExitCode::SUCCESS);
        }
        if flag == "--version" || flag == "-V" {
            println!("bsp-rust {}", env!("CARGO_PKG_VERSION"));
            return Ok(ExitCode::SUCCESS);
        }
        if flag == "--dump-messages" {
            dump_messages = true;
            i += 1;
            continue;
        }
        if flag == "--progress" || flag == "--no-progress" {
            let enabled = flag == "--progress";
            if progress.is_some_and(|current| current != enabled) {
                return Err("use only one of --progress and --no-progress".into());
            }
            progress = Some(enabled);
            i += 1;
            continue;
        }
        if !flag.starts_with('-') {
            if source.replace(flag.clone()).is_some() {
                return Err("one formula path is allowed".into());
            }
            i += 1;
            continue;
        }
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--backend" => {
                execution.backend = match value.as_str() {
                    "cpu" => bsp_rust::Backend::Cpu,
                    "cpu-parallel" => bsp_rust::Backend::CpuParallel,
                    "metal" => bsp_rust::Backend::Metal,
                    "cuda" => bsp_rust::Backend::Cuda,
                    "cuda-f32" => bsp_rust::Backend::CudaF32,
                    _ => {
                        return Err(
                            "backend must be cpu, cpu-parallel, metal, cuda, or cuda-f32".into(),
                        )
                    }
                }
            }
            "--threads" => execution.threads = value.parse()?,
            "--profile" => execution.profile = Some(value.into()),
            "--seed" => options.seed = value.parse()?,
            "--r" => options.backtrack_ratio = value.parse()?,
            "--frac" => options.batch_fraction = value.parse()?,
            "--eps" => options.epsilon = value.parse()?,
            "--damping" => options.damping = value.parse()?,
            "--max-sweeps" => options.max_sweeps = value.parse()?,
            "--max-steps" => options.max_steps = value.parse()?,
            "--max-flips" => options.max_flips = value.parse()?,
            "--restarts" => options.walksat_restarts = value.parse()?,
            "--timeout" => options.timeout_seconds = Some(value.parse()?),
            "--options" => options = serde_json::from_str(&fs::read_to_string(value)?)?,
            "--messages" => messages = Some(value.clone()),
            "--sweeps" => sweeps = Some(value.parse::<usize>()?),
            "--write-formula" => export = Some(value.clone()),
            "--output" => output = Some(value.clone()),
            "--generate" => {
                let alpha = args
                    .get(i + 2)
                    .ok_or("--generate needs K ALPHA N")?
                    .parse::<f64>()?;
                let n = args
                    .get(i + 3)
                    .ok_or("--generate needs K ALPHA N")?
                    .parse::<usize>()?;
                generate = Some((value.parse::<usize>()?, alpha, n));
                i += 2;
            }
            _ => return Err(format!("unknown option {flag}").into()),
        }
        i += 2;
    }
    options.validate()?;
    let show_progress = progress.unwrap_or_else(|| std::io::stdout().is_terminal());
    let clock = show_progress.then(RunClock::new);
    let mut live = clock.clone().map(cli::progress::LiveLine::start);
    let (formula_name, formula) = match (source, generate) {
        (Some(path), None) => (
            cli::output::formula_token(&path),
            Formula::from_dimacs(&fs::read_to_string(&path)?)?,
        ),
        (None, Some((k, alpha, n))) => (
            cli::output::generated_token(k, alpha, n),
            Formula::random(k, alpha, n, options.seed)?,
        ),
        _ => return Err("provide a DIMACS path or --generate K ALPHA N".into()),
    };
    if let Some(path) = export {
        if let Some(line) = live.as_mut() {
            line.stop();
        }
        fs::write(path, formula.to_dimacs())?;
        return Ok(ExitCode::SUCCESS);
    }
    let mut solver = Solver::with_execution(formula.clone(), options.clone(), execution.clone())?;
    if let Some(clock) = &clock {
        solver.attach_run_clock(clock.clone());
        clock.enter(RunPhase::Bsp);
    }
    if let Some(path) = messages {
        solver.set_messages(&serde_json::from_str::<Vec<f64>>(&fs::read_to_string(
            path,
        )?)?)?;
    }
    let report_path = output.map(PathBuf::from).unwrap_or_else(|| {
        cli::output::default_path(
            &formula_name,
            options.seed,
            options.backtrack_ratio,
            execution.backend,
        )
    });
    if let Some(sweeps) = sweeps {
        if sweeps == 0 {
            return Err("--sweeps must be positive".into());
        }
        let started = std::time::Instant::now();
        let residual = solver
            .sweep_many(sweeps)
            .map_err(|status| status.to_string())?;
        let seconds = started.elapsed().as_secs_f64();
        solver.write_profile()?;
        let mut report = serde_json::json!({
            "n": formula.n(),
            "m": formula.m(),
            "edges": formula.edges(),
            "sweeps": sweeps,
            "seconds": seconds,
            "ns_per_edge_sweep": seconds * 1e9 / (sweeps as f64 * formula.edges().max(1) as f64),
            "residual": residual,
            "checksum": solver.messages().iter().sum::<f64>(),
            "execution": solver.execution_report()
        });
        if dump_messages {
            report["messages"] = serde_json::json!(solver.messages());
        }
        finish(live, clock, &report_path, &report)?;
        return Ok(ExitCode::SUCCESS);
    }
    let result = solver.solve();
    solver.write_profile()?;
    let code = if result.status == Status::Sat {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    };
    finish(live, clock, &report_path, &serde_json::to_value(&result)?)?;
    Ok(code)
}

fn finish(
    mut live: Option<cli::progress::LiveLine>,
    clock: Option<RunClock>,
    path: &Path,
    report: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(line) = live.as_mut() {
        line.stop();
    }
    let snap = clock.as_ref().map(RunClock::snapshot);
    cli::output::write_json(path, report)?;
    if let Some(snap) = snap {
        println!("{}", cli::progress::final_line(&snap, path));
    } else {
        println!("{}", path.display());
    }
    std::io::stdout().flush()?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("bsp-rust: {error}");
            ExitCode::from(2)
        }
    }
}
