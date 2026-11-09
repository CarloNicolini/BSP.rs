use crate::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub seed: u64,
    pub backtrack_ratio: f64,
    pub batch_fraction: f64,
    pub epsilon: f64,
    /// Weight of the previous message. Convergence uses the undamped residual.
    pub damping: f64,
    pub max_sweeps: usize,
    pub max_steps: usize,
    /// Total WalkSAT flips; zero selects a budget scaled to the residual size.
    pub max_flips: u64,
    pub walksat_restarts: usize,
    pub walksat_noise: f64,
    pub timeout_seconds: Option<f64>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 1,
            backtrack_ratio: 0.9,
            batch_fraction: 0.00125,
            epsilon: 0.01,
            damping: 0.0,
            max_sweeps: 1024,
            max_steps: 100_000,
            max_flips: 0,
            walksat_restarts: 4,
            walksat_noise: 0.5,
            timeout_seconds: None,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        let checks = [
            (
                "backtrack_ratio",
                (0.0..1.0).contains(&self.backtrack_ratio),
                "in [0, 1)",
            ),
            (
                "batch_fraction",
                self.batch_fraction > 0.0 && self.batch_fraction <= 1.0,
                "in (0, 1]",
            ),
            ("epsilon", (0.0..1.0).contains(&self.epsilon), "in [0, 1)"),
            ("damping", (0.0..1.0).contains(&self.damping), "in [0, 1)"),
            (
                "walksat_noise",
                (0.0..=1.0).contains(&self.walksat_noise),
                "in [0, 1]",
            ),
            ("max_sweeps", self.max_sweeps > 0, "greater than 0"),
            ("max_steps", self.max_steps > 0, "greater than 0"),
            (
                "walksat_restarts",
                self.walksat_restarts > 0,
                "greater than 0",
            ),
            (
                "timeout_seconds",
                self.timeout_seconds
                    .is_none_or(|t| t.is_finite() && t > 0.0),
                "positive and finite",
            ),
        ];
        match checks.iter().find(|(_, valid, _)| !valid) {
            Some((name, _, expected)) => Err(Error(format!(
                "invalid option '{name}': must be {expected}"
            ))),
            None => Ok(()),
        }
    }
}
