use super::output::backend_name;
use bsp_rust::{ExecutionOptions, Options};

pub fn text(color: bool) -> String {
    let options = Options::default();
    let execution = ExecutionOptions::default();
    let rows = [
        row("--version, -V", "Print the package version.", None),
        row("--help, -h", "Print this help.", None),
        row(
            "--seed N",
            "Seed for the messages, WalkSAT, and a generated formula.",
            Some(options.seed.to_string()),
        ),
        row(
            "--r R",
            "Backtracking ratio in [0, 1). The solver backtracks while backtracking actions divided by decimation actions stay below this value. Both counters start at 1.",
            Some(options.backtrack_ratio.to_string()),
        ),
        row(
            "--frac F",
            "Fraction of the free variables fixed by one decimation, in (0, 1].",
            Some(options.batch_fraction.to_string()),
        ),
        row(
            "--eps E",
            "Stop a fixed point when the largest undamped message change is at most this value.",
            Some(options.epsilon.to_string()),
        ),
        row(
            "--damping D",
            "Weight of the previous message, in [0, 1). The convergence test uses the change before damping.",
            Some(options.damping.to_string()),
        ),
        row(
            "--max-sweeps N",
            "Sweep limit for one fixed point.",
            Some(options.max_sweeps.to_string()),
        ),
        row(
            "--max-steps N",
            "Limit on recorded BSP actions. The WalkSAT handoff counts as one action.",
            Some(options.max_steps.to_string()),
        ),
        row(
            "--max-flips N",
            "WalkSAT flip budget across all restarts.",
            Some(options.max_flips.to_string()),
        ),
        row(
            "--restarts N",
            "Number of WalkSAT restarts.",
            Some(options.walksat_restarts.to_string()),
        ),
        row(
            "--timeout SECONDS",
            "Stop after this many seconds. The limit includes solver setup.",
            Some("none".to_string()),
        ),
        row(
            "--backend NAME",
            "One of cpu, cpu-parallel, metal, cuda, or cuda-f32.",
            Some(backend_name(execution.backend).to_string()),
        ),
        row(
            "--threads N",
            "Worker limit for cpu-parallel. Zero uses the available CPUs.",
            Some(execution.threads.to_string()),
        ),
        row(
            "--progress",
            "Show formula setup, BSP, and WalkSAT timers on one stdout line.",
            Some("on when stdout is a terminal".to_string()),
        ),
        row(
            "--no-progress",
            "Hide the timer line. Stdout then contains only the JSON file path.",
            None,
        ),
        row(
            "--output PATH",
            "Write the JSON result to this path.",
            Some("output_<run>_<UTC time>.json".to_string()),
        ),
        row(
            "--profile PATH",
            "Write phase timings and execution metadata as JSON.",
            Some("none".to_string()),
        ),
        row(
            "--options PATH",
            "Read an Options JSON object. Flags after this one replace those values.",
            None,
        ),
        row(
            "--sweeps N",
            "Run this many survey-propagation sweeps and skip the BSP search.",
            None,
        ),
        row(
            "--messages PATH",
            "Load a flat JSON array of starting messages.",
            None,
        ),
        row(
            "--dump-messages",
            "Add the ending messages to a sweep report.",
            None,
        ),
        row(
            "--write-formula PATH",
            "Write the generated or normalized formula as DIMACS and do not solve.",
            None,
        ),
        row(
            "--generate K ALPHA N",
            "Build a random formula with floor(alpha*N) clauses of width K.",
            None,
        ),
    ];
    let width = rows.iter().map(|(flag, _)| flag.len()).max().unwrap_or(0);
    let mut out = String::from(
        "\
bsp-rust [OPTIONS] formula.cnf
bsp-rust [OPTIONS] --generate K ALPHA N

",
    );
    for (flag, detail) in rows {
        let painted = paint(color, "36", &flag);
        let pad = " ".repeat(width - flag.len() + 2);
        out.push_str(&format!("  {painted}{pad}{detail}\n"));
    }
    out.push_str(
        "
The formula timer covers reading or generating the formula and building the solver.
The BSP timer covers survey propagation, decimation, and backtracking.
The WalkSAT timer starts when local search starts.
The program writes the JSON result to a file and prints that path on stdout.
The JSON field residual is the largest undamped message change in the last sweep.
Exit code 0 means a verified SAT assignment.
Exit code 1 means a heuristic failure or a limit.
Exit code 2 means invalid input.
Assignments use 0 and 1. A partial assignment uses -1 for a free variable.
",
    );
    out
}

fn row(flag: &str, detail: &str, default: Option<String>) -> (String, String) {
    let detail = match default {
        Some(value) => format!("{detail} Default: {value}."),
        None => detail.to_string(),
    };
    (flag.to_string(), detail)
}

fn paint(color: bool, code: &str, text: &str) -> String {
    if color {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}
