//! Native, reentrant BSP for ordinary CNF SAT. See `README.md`.
//! The C++ research extensions remain separate from this baseline solver.
#![forbid(unsafe_code)]

mod bitset;
mod formula;
#[cfg(feature = "python")]
mod python;
mod solver;
mod walksat;

pub use formula::Formula;
pub use solver::{CancellationToken, Options, SolveResult, Solver, SpReport, Status, Step};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
