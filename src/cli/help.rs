use super::output::backend_name;
use bsp_rust::{ExecutionOptions, Options};

pub fn text(color: bool) -> String {
    let options = Options::default();
    let execution = ExecutionOptions::default();
    let rows = [
        line("--version, -V", "Print the package version.", ""),
        line("--help, -h", "Print this help.", ""),
        line(
            "--seed N",
            "Seed for messages, WalkSAT, and formula generation.",
            &options.seed.to_string(),
        ),
        line(
            "--r R",
            "Backtracking ratio in [0, 1). Both action counters start at 1.",
            &options.backtrack_ratio.to_string(),
        ),
        line(
            "--frac F",
            "Fraction of free variables fixed by one decimation, in (0, 1].",
            &options.batch_fraction.to_string(),
        ),
        line(
            "--eps E",
            "Largest undamped message change allowed at a fixed point.",
            &options.epsilon.to_string(),
        ),
        line(
            "--damping D",
            "Weight of the previous message, in [0, 1). Convergence uses the undamped change.",
            &options.damping.to_string(),
        ),
        line(
            "--max-sweeps N",
            "Sweep limit for one fixed point.",
            &options.max_sweeps.to_string(),
        ),
        line(
            "--max-steps N",
            "Limit on recorded BSP actions.",
            &options.max_steps.to_string(),
        ),
        line(
            "--max-flips N",
            "WalkSAT flip budget across all restarts.",
            &options.max_flips.to_string(),
        ),
        line(
            "--restarts N",
            "Number of WalkSAT restarts.",
            &options.walksat_restarts.to_string(),
        ),
        line(
            "--timeout SECONDS",
            "Stop after this many seconds, including solver setup.",
            "none",
        ),
        line(
            "--backend NAME",
            "cpu, cpu-parallel, metal, cuda, or cuda-f32.",
            backend_name(execution.backend),
        ),
        line(
            "--threads N",
            "Worker limit for cpu-parallel. Zero uses the available CPUs.",
            &execution.threads.to_string(),
        ),
        line(
            "--progress",
            "Show formula, BSP, and WalkSAT timers on one stdout line.",
            "on when stdout is a terminal",
        ),
        line(
            "--no-progress",
            "Hide the timer line. Stdout is then one JSON summary.",
            "",
        ),
        line(
            "--output PATH",
            "Write the full JSON result to this path.",
            "output_<run>_<UTC time>.json",
        ),
        line(
            "--profile PATH",
            "Write phase timings and execution metadata as JSON.",
            "none",
        ),
        line(
            "--options PATH",
            "Read Options JSON. Later flags replace those values.",
            "",
        ),
        line(
            "--sweeps N",
            "Run this many survey-propagation sweeps and skip BSP search.",
            "",
        ),
        line(
            "--messages PATH",
            "Load a flat JSON array of starting messages.",
            "",
        ),
        line(
            "--dump-messages",
            "Add the ending messages to a sweep report.",
            "",
        ),
        line(
            "--write-formula PATH",
            "Write the formula as DIMACS and do not solve.",
            "",
        ),
        line(
            "--generate K ALPHA N",
            "Build a random formula with floor(alpha*N) clauses.",
            "",
        ),
    ];
    let width = rows.iter().map(|(flag, _)| flag.len()).max().unwrap_or(0);
    let mut out =
        String::from("bsp-rust [OPTIONS] formula.cnf\nbsp-rust [OPTIONS] --generate K ALPHA N\n\n");
    for (flag, detail) in rows {
        let pad = " ".repeat(width - flag.len() + 2);
        let label = if color {
            format!("\x1b[36m{flag}\x1b[0m")
        } else {
            flag
        };
        out.push_str(&format!("  {label}{pad}{detail}\n"));
    }
    out.push_str("\nThe formula timer covers reading or generating the formula and building the solver.\nThe BSP timer covers survey propagation, decimation, and backtracking.\nThe WalkSAT timer starts when local search starts. A device sweep does not call it.\nThe full JSON result goes to a file.\nStdout then prints one JSON object with the status and the main counters.\nThe JSON field residual is the largest undamped message change in the last sweep.\nExit code 0 means a verified SAT assignment.\nExit code 1 means a heuristic failure or a limit.\nExit code 2 means invalid input.\nAssignments use 0 and 1. A partial assignment uses -1 for a free variable.\n");
    out
}

fn line(flag: &str, detail: &str, default: &str) -> (String, String) {
    let detail = if default.is_empty() {
        detail.to_string()
    } else {
        format!("{detail} Default: {default}.")
    };
    (flag.to_string(), detail)
}
