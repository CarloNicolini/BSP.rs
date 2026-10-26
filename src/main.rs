use bsp_rust::{Backend, ExecutionOptions, Formula, Options, Solver, Status};
use serde_json::json;
use std::{env, fs, process::ExitCode, time::Instant};

const HELP: &str = "bsp-rust [OPTIONS] formula.cnf
bsp-rust [OPTIONS] --generate K ALPHA N

  --seed N             RNG seed (default 1)
  --r R                Backtracking ratio in [0,1)
  --frac F             Decimation batch fraction in (0,1]
  --eps E              SP undamped residual tolerance
  --damping D          Weight of the old message in [0,1)
  --max-sweeps N       SP sweep limit per fixed point
  --max-steps N        BSP action limit
  --max-flips N        Total WalkSAT flip budget across restarts
  --restarts N         WalkSAT restarts
  --timeout SECONDS    Cooperative time limit, including solver construction
  --backend NAME       cpu (default), cpu-parallel, or metal (f32, optional build)
  --threads N          CPU worker limit; 0 uses available CPUs
  --profile PATH       Write phase timings and execution metadata as JSON
  --options PATH       Read Options JSON before subsequent overrides
  --sweeps N           Benchmark exactly N sweeps, excluding graph construction
  --messages PATH      Load a flat JSON array of starting messages
  --dump-messages      Include ending messages in benchmark JSON
  --write-formula PATH Export generated/normalized DIMACS without solving

Outputs JSON. SAT exits 0; heuristic limits/failures exit 1; input errors exit 2.
Assignments are 0/1, and partial assignments use -1 for free variables.
";

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
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        if flag == "--help" || flag == "-h" {
            print!("{HELP}");
            return Ok(ExitCode::SUCCESS);
        }
        if flag == "--dump-messages" {
            dump_messages = true;
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
                    "cpu" => Backend::Cpu,
                    "cpu-parallel" => Backend::CpuParallel,
                    "metal" => Backend::Metal,
                    _ => return Err("backend must be cpu, cpu-parallel, or metal".into()),
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
    let formula = match (source, generate) {
        (Some(path), None) => Formula::from_dimacs(&fs::read_to_string(path)?)?,
        (None, Some((k, alpha, n))) => Formula::random(k, alpha, n, options.seed)?,
        _ => return Err("provide a DIMACS path or --generate K ALPHA N".into()),
    };
    if let Some(path) = export {
        fs::write(path, formula.to_dimacs())?;
        return Ok(ExitCode::SUCCESS);
    }
    let mut solver = Solver::with_execution(formula.clone(), options, execution)?;
    if let Some(path) = messages {
        solver.set_messages(&serde_json::from_str::<Vec<f64>>(&fs::read_to_string(
            path,
        )?)?)?;
    }
    if let Some(sweeps) = sweeps {
        if sweeps == 0 {
            return Err("--sweeps must be positive".into());
        }
        let started = Instant::now();
        let residual = solver.sweep_many(sweeps).map_err(|s| s.to_string())?;
        let seconds = started.elapsed().as_secs_f64();
        solver.write_profile()?;
        let mut report = json!({"n":formula.n(),"m":formula.m(),"edges":formula.edges(),"sweeps":sweeps,
            "seconds":seconds,"ns_per_edge_sweep":seconds*1e9/(sweeps as f64*formula.edges().max(1) as f64),
            "residual":residual,"checksum":solver.messages().iter().sum::<f64>(),"execution":solver.execution_report()});
        if dump_messages {
            report["messages"] = json!(solver.messages());
        }
        println!("{report}");
        return Ok(ExitCode::SUCCESS);
    }
    let result = solver.solve();
    solver.write_profile()?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(if result.status == Status::Sat {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
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
