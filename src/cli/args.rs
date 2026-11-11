use bsp_rust::Backend;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum BackendArg {
    Cpu,
    CpuParallel,
    Metal,
    Cuda,
    CudaF32,
}

impl From<BackendArg> for Backend {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Cpu => Self::Cpu,
            BackendArg::CpuParallel => Self::CpuParallel,
            BackendArg::Metal => Self::Metal,
            BackendArg::Cuda => Self::Cuda,
            BackendArg::CudaF32 => Self::CudaF32,
        }
    }
}

const AFTER_HELP: &str = "\
The formula timer covers reading or generating the formula and building the solver.
The BSP timer covers survey propagation, decimation, and backtracking.
The WalkSAT timer starts when local search starts. A device sweep does not call it.
Load shared parameters with --config PATH (.toml, .yaml, .yml, or .json).
Merge order is defaults, then --config, then --options, then explicit flags.
The full JSON result goes to a file.
--stream-table PATH writes progress rows as TSV for gnuplot or tail -f.
Stdout then prints one JSON object with the status and the main counters.
The JSON field residual is the largest undamped message change in the last sweep.
Exit code 0 means a verified SAT assignment.
Exit code 1 means a heuristic failure or a limit.
Exit code 2 means invalid input.
For batches, each input produces a JSON summary followed by one aggregate summary.
Exit code 2 takes precedence over 1 if any input fails to load or execute.
Assignments use 0 and 1. A partial assignment uses -1 for a free variable.";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FeistelEnsemble {
    #[default]
    Fluctuating,
    Partitioned,
}

#[derive(Debug, Parser)]
#[command(
    name = "bsp-rust",
    version,
    about = "Native Rust Backtracking Survey Propagation for CNF SAT",
    after_help = AFTER_HELP
)]
pub struct Cli {
    /// DIMACS paths or directories (directories contain .cnf files; processed sequentially)
    pub formula: Vec<PathBuf>,

    /// Retain DIMACS literal order for same-seed replay of an exported Feistel formula
    #[arg(long)]
    pub preserve_literal_order: bool,

    /// Build a random formula with floor(alpha*N) clauses
    #[arg(long, num_args = 3, value_names = ["K", "ALPHA", "N"])]
    pub generate: Option<Vec<String>>,

    /// Implicit random formula with fluctuating variable degrees (CPU or Metal)
    #[arg(long, num_args = 3, value_names = ["K", "ALPHA", "N"])]
    pub feistel: Option<Vec<String>>,

    /// Feistel ensemble. Partitioned has balanced degrees and skips the repair scan [default: fluctuating]
    #[arg(long, value_enum)]
    pub feistel_ensemble: Option<FeistelEnsemble>,

    /// Load run parameters from TOML, YAML, or JSON
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,

    /// Read Options JSON. Later flags replace those values
    #[arg(long, value_name = "PATH")]
    pub options: Option<PathBuf>,

    /// Seed for messages, WalkSAT, and formula generation [default: 1]
    #[arg(long, value_name = "N")]
    pub seed: Option<u64>,

    /// Backtracking ratio in [0, 1) [default: 0.9]
    #[arg(long = "r", value_name = "R")]
    pub backtrack_ratio: Option<f64>,

    /// Fraction of free variables fixed by one decimation [default: 0.00125]
    #[arg(long = "frac", value_name = "F")]
    pub batch_fraction: Option<f64>,

    /// Largest undamped message change allowed at a fixed point [default: 0.01]
    #[arg(long = "eps", value_name = "E")]
    pub epsilon: Option<f64>,

    /// Weight of the previous message, in [0, 1) [default: 0.0]
    #[arg(long, value_name = "D")]
    pub damping: Option<f64>,

    /// Sweep limit for one fixed point [default: 1024]
    #[arg(long, value_name = "N")]
    pub max_sweeps: Option<usize>,

    /// Limit on recorded BSP actions [default: 100000]
    #[arg(long, value_name = "N")]
    pub max_steps: Option<usize>,

    /// WalkSAT total flip budget; 0 = restarts * max(10000, 100 * residual variables) [default: 0]
    #[arg(long, value_name = "N")]
    pub max_flips: Option<u64>,

    /// Number of WalkSAT restarts [default: 4]
    #[arg(long, value_name = "N")]
    pub restarts: Option<usize>,

    /// Stop after this many seconds, including solver setup
    #[arg(long, value_name = "SECONDS")]
    pub timeout: Option<f64>,

    /// Execution backend
    #[arg(long, value_name = "NAME", value_enum)]
    pub backend: Option<BackendArg>,

    /// Worker limit for cpu-parallel. Zero uses the available CPUs [default: 0]
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

    /// Show formula, BSP, and WalkSAT timers on one stdout line
    #[arg(long, overrides_with = "no_progress")]
    pub progress: bool,

    /// Seconds between progress table rows [default: 3]
    #[arg(long, value_name = "SECONDS")]
    pub update_frequency: Option<f64>,

    /// Write progress table rows as TSV for gnuplot or tail -f
    #[arg(long, value_name = "PATH")]
    pub stream_table: Option<PathBuf>,

    /// Write structured JSON logs of every BSP step to stderr [default: off]
    #[arg(long, value_name = "LEVEL", value_enum)]
    pub enable_tracing: Option<TraceLevel>,

    /// Hide the timer line. Stdout is then one JSON summary
    #[arg(long, overrides_with = "progress")]
    pub no_progress: bool,

    /// Write the full JSON result to this path
    #[arg(long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Write phase timings and execution metadata as JSON
    #[arg(long, value_name = "PATH")]
    pub profile: Option<PathBuf>,

    /// Run this many survey-propagation sweeps and skip BSP search
    #[arg(long, value_name = "N")]
    pub sweeps: Option<usize>,

    /// Load a flat JSON array of starting messages
    #[arg(long, value_name = "PATH")]
    pub messages: Option<PathBuf>,

    /// Add the ending messages to a sweep report
    #[arg(long)]
    pub dump_messages: bool,

    /// Write the formula as DIMACS and do not solve
    #[arg(long, value_name = "PATH")]
    pub write_formula: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TraceLevel {
    Warn,
    Info,
}

impl From<TraceLevel> for tracing::Level {
    fn from(level: TraceLevel) -> Self {
        match level {
            TraceLevel::Warn => Self::WARN,
            TraceLevel::Info => Self::INFO,
        }
    }
}
