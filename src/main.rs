mod cli;

use crate::cli::args::{Cli, FeistelEnsemble};
use crate::cli::config::{FileConfig, Resolved};
use crate::cli::progress::{PhaseClock, BSP, WALKSAT};
use bsp_rust::{Formula, Solver, Status};
use clap::Parser;
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::sync::Arc;
use std::{fs, process::ExitCode};

fn parse_generate(values: &[String]) -> Result<(usize, f64, usize), Box<dyn std::error::Error>> {
    if values.len() != 3 {
        return Err("formula generation needs K ALPHA N".into());
    }
    Ok((values[0].parse()?, values[1].parse()?, values[2].parse()?))
}

fn resolve(cli: Cli) -> Result<Resolved, Box<dyn std::error::Error>> {
    let mut resolved = Resolved::default();
    if let Some(path) = &cli.config {
        FileConfig::load(path)?.apply(&mut resolved);
    }
    if let Some(path) = &cli.options {
        resolved.options = serde_json::from_str(&fs::read_to_string(path)?)?;
    }
    if let Some(path) = cli.formula {
        resolved.formula = Some(path);
    }
    if cli.preserve_literal_order {
        resolved.preserve_literal_order = true;
    }
    if let Some(values) = &cli.generate {
        resolved.generate = Some(parse_generate(values)?);
    }
    if let Some(values) = &cli.feistel {
        resolved.feistel = Some(parse_generate(values)?);
    }
    if let Some(ensemble) = cli.feistel_ensemble {
        resolved.feistel_ensemble = Some(ensemble);
    }
    if let Some(value) = cli.seed {
        resolved.options.seed = value;
    }
    if let Some(value) = cli.backtrack_ratio {
        resolved.options.backtrack_ratio = value;
    }
    if let Some(value) = cli.batch_fraction {
        resolved.options.batch_fraction = value;
    }
    if let Some(value) = cli.epsilon {
        resolved.options.epsilon = value;
    }
    if let Some(value) = cli.damping {
        resolved.options.damping = value;
    }
    if let Some(value) = cli.max_sweeps {
        resolved.options.max_sweeps = value;
    }
    if let Some(value) = cli.max_steps {
        resolved.options.max_steps = value;
    }
    if let Some(value) = cli.max_flips {
        resolved.options.max_flips = value;
    }
    if let Some(value) = cli.restarts {
        resolved.options.walksat_restarts = value;
    }
    if let Some(value) = cli.timeout {
        resolved.options.timeout_seconds = Some(value);
    }
    if let Some(value) = cli.backend {
        resolved.execution.backend = value.into();
    }
    if let Some(value) = cli.threads {
        resolved.execution.threads = value;
    }
    if let Some(path) = cli.profile {
        resolved.execution.profile = Some(path);
    }
    if cli.progress {
        resolved.progress = Some(true);
    } else if cli.no_progress {
        resolved.progress = Some(false);
    }
    if let Some(path) = cli.output {
        resolved.output = Some(path);
    }
    if let Some(value) = cli.sweeps {
        resolved.sweeps = Some(value);
    }
    if let Some(path) = cli.messages {
        resolved.messages = Some(path);
    }
    if cli.dump_messages {
        resolved.dump_messages = true;
    }
    if let Some(path) = cli.write_formula {
        resolved.write_formula = Some(path);
    }
    Ok(resolved)
}

fn run(resolved: Resolved) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let Resolved {
        formula: source,
        preserve_literal_order,
        generate,
        feistel,
        feistel_ensemble,
        options,
        execution,
        sweeps,
        messages,
        dump_messages,
        write_formula: export,
        output,
        progress,
    } = resolved;
    options.validate()?;
    if preserve_literal_order && source.is_none() {
        return Err("--preserve-literal-order requires a DIMACS input".into());
    }
    if feistel.is_none() && feistel_ensemble.is_some() {
        return Err("--feistel-ensemble requires a Feistel formula source".into());
    }
    let partitioned = feistel_ensemble.unwrap_or_default() == FeistelEnsemble::Partitioned;
    let terminal = std::io::stdout().is_terminal();
    let show_progress = progress.unwrap_or(terminal) && terminal;
    let clock = PhaseClock::new();
    let start_live = || show_progress.then(|| cli::progress::LiveLine::start(clock.clone()));
    let mut live = if partitioned { None } else { start_live() };
    if feistel.is_some()
        && !matches!(
            execution.backend,
            bsp_rust::Backend::Cpu | bsp_rust::Backend::CpuParallel
        )
    {
        return Err("Feistel formulas require --backend cpu or cpu-parallel".into());
    }
    let construction_started = std::time::Instant::now();
    let (formula_name, formula) = match (source, generate, feistel) {
        (Some(path), None, None) => (cli::output::formula_token(&path.to_string_lossy()), {
            let text = fs::read_to_string(&path)?;
            if preserve_literal_order {
                Formula::from_dimacs_preserving_order(&text)?
            } else {
                Formula::from_dimacs(&text)?
            }
        }),
        (None, Some((k, alpha, n)), None) => (
            cli::output::generated_token(k, alpha, n),
            Formula::random(k, alpha, n, options.seed)?,
        ),
        (None, None, Some((k, alpha, n))) => {
            // Reject impossible solves before the fluctuating generator's scan.
            if export.is_none() && (alpha * n as f64).floor() * k as f64 >= u32::MAX as f64 {
                return Err("solver exceeds u32 edge/clause capacity".into());
            }
            let token = cli::output::generated_token(k, alpha, n);
            if partitioned {
                (
                    format!("feistel_partitioned_{token}"),
                    Formula::feistel(k, alpha, n, options.seed)?,
                )
            } else {
                (
                    format!("feistel_{token}"),
                    Formula::feistel_fluctuating(k, alpha, n, options.seed)?,
                )
            }
        }
        _ => {
            return Err(
                "provide exactly one DIMACS path, --generate K ALPHA N, or --feistel K ALPHA N"
                    .into(),
            )
        }
    };
    let construction_seconds = construction_started.elapsed().as_secs_f64();
    if partitioned {
        eprintln!(
            "bsp-rust: warning: partitioned --feistel uses a different random k-SAT ensemble from --generate. \
             Each clause draws one variable from each fixed group, and variable degrees are balanced. \
             This skips the repair scan and can reduce cycle walking, depending on instance size."
        );
        live = start_live();
    }
    if let Some(path) = export {
        if let Some(line) = live.as_mut() {
            line.stop();
        }
        let mut writer = std::io::BufWriter::new(fs::File::create(path)?);
        formula.write_dimacs(&mut writer)?;
        writer.flush()?;
        return Ok(ExitCode::SUCCESS);
    }
    let topology_report = formula.feistel_topology().map(|f| {
        serde_json::json!({
            "generator": f.version(),
            "ensemble": if partitioned { "partitioned-balanced-degrees" } else { "random-k-sat-target" },
            "repaired_clauses": f.repair_count(),
            "n": f.n(), "m": f.m(), "k": f.k(), "edges": f.edges(), "seed": f.seed(),
            "degree_counts": f.degree_counts(),
            "construction_seconds": construction_seconds,
            "topology_bytes": formula.topology_bytes()
        })
    });
    let mut solver = Solver::with_execution(formula.clone(), options.clone(), execution.clone())?;
    let watched = clock.clone();
    solver.on_walksat(Arc::new(move || watched.enter(WALKSAT)));
    clock.enter(BSP);
    if let Some(path) = messages {
        solver.set_messages(&serde_json::from_str::<Vec<f64>>(&fs::read_to_string(
            path,
        )?)?)?;
    }
    let report_path = output.unwrap_or_else(|| {
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
        if let Some(topology) = topology_report {
            report["topology"] = topology;
        }
        #[derive(serde::Serialize)]
        struct SweepReport<'a> {
            #[serde(flatten)]
            fields: &'a serde_json::Value,
            #[serde(skip_serializing_if = "Option::is_none")]
            messages: Option<&'a [f64]>,
        }
        let output = SweepReport {
            fields: &report,
            messages: dump_messages.then(|| solver.messages()),
        };
        finish(live, &clock, &report_path, &output, &report)?;
        return Ok(ExitCode::SUCCESS);
    }
    let result = solver.solve();
    solver.write_profile()?;
    let code = if result.status == Status::Sat {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    };
    drop(solver);
    drop(formula);
    let summary_fields = serde_json::json!({
        "status": result.status,
        "seed": result.seed,
        "wall_seconds": result.wall_seconds,
        "sp_sweeps": result.sp_sweeps,
        "flips": result.flips,
        "decimations": result.decimations,
        "releases": result.releases,
        "unit_assignments": result.unit_assignments,
    });
    #[derive(serde::Serialize)]
    struct SolveReport {
        #[serde(flatten)]
        result: bsp_rust::SolveResult,
        #[serde(skip_serializing_if = "Option::is_none")]
        topology: Option<serde_json::Value>,
    }
    let report = SolveReport {
        result,
        topology: topology_report,
    };
    finish(live, &clock, &report_path, &report, &summary_fields)?;
    Ok(code)
}

fn finish(
    mut live: Option<cli::progress::LiveLine>,
    clock: &PhaseClock,
    path: &Path,
    report: &impl serde::Serialize,
    summary_fields: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let times = clock.seconds();
    if let Some(line) = live.as_mut() {
        line.stop();
        println!();
    }
    cli::output::write_json(path, report)?;
    println!("{}", cli::output::summary(path, summary_fields, times));
    std::io::stdout().flush()?;
    Ok(())
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let code = if error.use_stderr() {
                ExitCode::from(2)
            } else {
                ExitCode::SUCCESS
            };
            let _ = error.print();
            return code;
        }
    };
    match resolve(cli).and_then(run) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("bsp-rust: {error}");
            ExitCode::from(2)
        }
    }
}
