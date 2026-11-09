use super::execution::ExecutionReport;
use serde::Serialize;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone, Default, Debug)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Sat,
    Contradiction,
    NonConverged,
    NegativeComplexity,
    NumericalFailure,
    StepLimit,
    FlipLimit,
    Cancelled,
    Timeout,
}
impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Sat => "sat",
            Self::Contradiction => "contradiction",
            Self::NonConverged => "non_converged",
            Self::NegativeComplexity => "negative_complexity",
            Self::NumericalFailure => "numerical_failure",
            Self::StepLimit => "step_limit",
            Self::FlipLimit => "flip_limit",
            Self::Cancelled => "cancelled",
            Self::Timeout => "timeout",
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SpReport {
    pub sweeps: usize,
    pub residual: f64,
    pub complexity: f64,
    pub paramagnetic: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Decimate,
    Backtrack,
    Walksat,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Step {
    pub index: usize,
    pub action: Action,
    pub free_variables: usize,
    pub active_clauses: usize,
    pub sp_sweeps: u64,
    pub residual: f64,
    pub complexity: f64,
    pub changed: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct SolveResult {
    pub execution: ExecutionReport,
    pub status: Status,
    pub assignment: Option<Vec<i8>>,
    pub partial_assignment: Vec<i8>,
    pub seed: u64,
    pub wall_seconds: f64,
    pub sp_sweeps: u64,
    pub flips: u64,
    pub decimations: u64,
    pub releases: u64,
    pub unit_assignments: u64,
    pub steps: Vec<Step>,
}
