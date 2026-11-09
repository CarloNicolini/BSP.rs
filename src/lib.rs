//! Native, reentrant BSP for ordinary CNF SAT. See `README.md`.
//! The C++ research extensions remain separate from this baseline solver.
#![forbid(unsafe_code)]

mod bitset;
mod decision_heap;
mod feistel;
mod formula;
pub use feistel::FeistelTopology;
#[cfg(feature = "python")]
mod python;
mod solver;
mod walksat;

pub use formula::Formula;
pub use solver::{
    solve_many, Action, Backend, CancellationToken, ExecutionOptions, ExecutionReport, Options,
    Precision, SolveJob, SolveResult, Solver, SpReport, Status, Step,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<String> for Error {
    fn from(message: String) -> Self {
        Self(message)
    }
}
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}
impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}
impl From<rayon::ThreadPoolBuildError> for Error {
    fn from(error: rayon::ThreadPoolBuildError) -> Self {
        Self(error.to_string())
    }
}
impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(error: std::sync::PoisonError<T>) -> Self {
        Self(error.to_string())
    }
}
pub type Result<T> = std::result::Result<T, Error>;
/// Solver outcome that stops a run without a result. Not an `Error`.
pub(crate) type StatusResult<T> = std::result::Result<T, Status>;
