use super::args::FeistelEnsemble;
use super::batch::BatchContext;
use super::config::Resolved;
use super::output;
use super::progress::{self, PhaseClock, BSP, WALKSAT};
use bsp_rust::{Formula, Solver, Status};
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::{fs, process::ExitCode};

pub fn run_single(
    resolved: Resolved,
    batch: Option<BatchContext<'_>>,
) -> Result<ExitCode, Box<dyn std::error::Error>> {
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
        update_frequency,
        stream_table,
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
    let row_interval = Duration::from_secs_f64(update_frequency.unwrap_or(3.0).max(0.1));
    let start_live = || -> Result<Option<progress::LiveLine>, Box<dyn std::error::Error>> {
        if !show_progress && stream_table.is_none() {
            return Ok(None);
        }
        Ok(Some(progress::LiveLine::start(
            clock.clone(),
            row_interval,
            show_progress,
            stream_table.clone(),
        )?))
    };
    let mut live = if partitioned { None } else { start_live()? };
    if feistel.is_some()
        && matches!(
            execution.backend,
            bsp_rust::Backend::Cuda | bsp_rust::Backend::CudaF32
        )
    {
        return Err("Feistel formulas require --backend cpu, cpu-parallel, or metal".into());
    }
    let construction_started = std::time::Instant::now();
    let (formula_name, formula) = match (source, generate, feistel) {
        (Some(path), None, None) => (output::formula_token(&path.to_string_lossy()), {
            let reader = std::io::BufReader::new(fs::File::open(&path)?);
            if preserve_literal_order {
                Formula::from_dimacs_reader_preserving_order(reader)?
            } else {
                Formula::from_dimacs_reader(reader)?
            }
        }),
        (None, Some((k, alpha, n)), None) => (
            output::generated_token(k, alpha, n),
            Formula::random(k, alpha, n, options.seed)?,
        ),
        (None, None, Some((k, alpha, n))) => {
            // Reject impossible solves before the fluctuating generator's scan.
            if export.is_none() && (alpha * n as f64).floor() * k as f64 >= u32::MAX as f64 {
                return Err("solver exceeds u32 edge/clause capacity".into());
            }
            let token = output::generated_token(k, alpha, n);
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
        live = start_live()?;
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
    if let Some(line) = &live {
        solver.on_step(line.step_hook());
    }
    clock.enter(BSP);
    if let Some(path) = messages {
        solver.set_messages(&serde_json::from_str::<Vec<f64>>(&fs::read_to_string(
            path,
        )?)?)?;
    }
    let unique_output = output.is_none();
    let report_path = output.unwrap_or_else(|| {
        output::default_path(
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
        finish(
            live,
            &clock,
            &report_path,
            unique_output,
            &output,
            &report,
            batch,
        )?;
        return Ok(ExitCode::SUCCESS);
    }
    let result = solver.solve();
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
    finish(
        live,
        &clock,
        &report_path,
        unique_output,
        &report,
        &summary_fields,
        batch,
    )?;
    Ok(code)
}

fn finish(
    mut live: Option<progress::LiveLine>,
    clock: &PhaseClock,
    path: &Path,
    unique: bool,
    report: &impl serde::Serialize,
    summary_fields: &serde_json::Value,
    batch: Option<BatchContext<'_>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let times = clock.seconds();
    if let Some(line) = live.as_mut() {
        line.stop();
        println!();
    }
    let written = output::write_json(path, report, unique)?;
    let mut summary = output::summary(&written, summary_fields, times);
    if let Some(context) = batch {
        summary["input"] = serde_json::Value::String(context.input.to_string_lossy().into_owned());
    }
    if let Some(bar) = batch.and_then(|context| context.bar) {
        bar.suspend(|| println!("{summary}"));
    } else {
        println!("{summary}");
    }
    std::io::stdout().flush()?;
    Ok(())
}
