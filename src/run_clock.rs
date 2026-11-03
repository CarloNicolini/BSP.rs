use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RunPhase {
    Formula = 0,
    Bsp = 1,
    Walksat = 2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunSnapshot {
    pub phase: RunPhase,
    pub formula_seconds: f64,
    pub bsp_seconds: f64,
    pub walksat_seconds: f64,
    pub sweeps: u64,
    pub flips: u64,
    pub steps: u64,
    pub free_variables: u64,
}

struct Inner {
    origin: Instant,
    phase: AtomicU8,
    mark_ns: AtomicU64,
    frozen: [AtomicU64; 3],
    sweeps: AtomicU64,
    flips: AtomicU64,
    steps: AtomicU64,
    free_variables: AtomicU64,
}

#[derive(Clone)]
pub struct RunClock {
    inner: Arc<Inner>,
}

impl Default for RunClock {
    fn default() -> Self {
        Self::new()
    }
}

impl RunClock {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                origin: Instant::now(),
                phase: AtomicU8::new(RunPhase::Formula as u8),
                mark_ns: AtomicU64::new(0),
                frozen: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
                sweeps: AtomicU64::new(0),
                flips: AtomicU64::new(0),
                steps: AtomicU64::new(0),
                free_variables: AtomicU64::new(0),
            }),
        }
    }

    pub fn enter(&self, phase: RunPhase) {
        let now = self.elapsed_ns();
        let previous = self.inner.phase.load(Ordering::Relaxed);
        if previous == phase as u8 {
            return;
        }
        let mark = self.inner.mark_ns.swap(now, Ordering::Relaxed);
        self.inner.frozen[previous as usize].fetch_add(now.saturating_sub(mark), Ordering::Relaxed);
        self.inner.phase.store(phase as u8, Ordering::Relaxed);
    }

    pub fn set_counts(&self, sweeps: u64, flips: u64, steps: u64, free_variables: u64) {
        self.inner.sweeps.store(sweeps, Ordering::Relaxed);
        self.inner.flips.store(flips, Ordering::Relaxed);
        self.inner.steps.store(steps, Ordering::Relaxed);
        self.inner
            .free_variables
            .store(free_variables, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> RunSnapshot {
        let phase = match self.inner.phase.load(Ordering::Relaxed) {
            1 => RunPhase::Bsp,
            2 => RunPhase::Walksat,
            _ => RunPhase::Formula,
        };
        let now = self.elapsed_ns();
        let mark = self.inner.mark_ns.load(Ordering::Relaxed);
        let mut seconds = [0.0; 3];
        for (index, stored) in self.inner.frozen.iter().enumerate() {
            let mut nanos = stored.load(Ordering::Relaxed);
            if index == phase as usize {
                nanos = nanos.saturating_add(now.saturating_sub(mark));
            }
            seconds[index] = nanos as f64 / 1e9;
        }
        RunSnapshot {
            phase,
            formula_seconds: seconds[0],
            bsp_seconds: seconds[1],
            walksat_seconds: seconds[2],
            sweeps: self.inner.sweeps.load(Ordering::Relaxed),
            flips: self.inner.flips.load(Ordering::Relaxed),
            steps: self.inner.steps.load(Ordering::Relaxed),
            free_variables: self.inner.free_variables.load(Ordering::Relaxed),
        }
    }

    fn elapsed_ns(&self) -> u64 {
        self.inner.origin.elapsed().as_nanos() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enter_switches_phase_and_counts_round_trip() {
        let clock = RunClock::new();
        assert_eq!(clock.snapshot().phase, RunPhase::Formula);
        clock.set_counts(2, 3, 4, 5);
        clock.enter(RunPhase::Walksat);
        let snap = clock.snapshot();
        assert_eq!(snap.phase, RunPhase::Walksat);
        assert_eq!(
            (snap.sweeps, snap.flips, snap.steps, snap.free_variables),
            (2, 3, 4, 5)
        );
    }
}
