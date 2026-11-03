use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub const BSP: u8 = 1;
pub const WALKSAT: u8 = 2;

struct Inner {
    origin: Instant,
    phase: AtomicU8,
    mark_ns: AtomicU64,
    frozen: [AtomicU64; 3],
}

#[derive(Clone)]
pub struct PhaseClock {
    inner: Arc<Inner>,
}

impl PhaseClock {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                origin: Instant::now(),
                phase: AtomicU8::new(0),
                mark_ns: AtomicU64::new(0),
                frozen: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            }),
        }
    }

    pub fn enter(&self, phase: u8) {
        let now = self.elapsed_ns();
        let previous = self.inner.phase.load(Ordering::Relaxed);
        if previous == phase {
            return;
        }
        let mark = self.inner.mark_ns.swap(now, Ordering::Relaxed);
        self.inner.frozen[previous as usize].fetch_add(now.saturating_sub(mark), Ordering::Relaxed);
        self.inner.phase.store(phase, Ordering::Relaxed);
    }

    pub fn seconds(&self) -> [f64; 3] {
        let phase = self.inner.phase.load(Ordering::Relaxed) as usize;
        let now = self.elapsed_ns();
        let mark = self.inner.mark_ns.load(Ordering::Relaxed);
        let mut seconds = [0.0; 3];
        for (index, stored) in self.inner.frozen.iter().enumerate() {
            let mut nanos = stored.load(Ordering::Relaxed);
            if index == phase {
                nanos = nanos.saturating_add(now.saturating_sub(mark));
            }
            seconds[index] = nanos as f64 / 1e9;
        }
        seconds
    }

    fn elapsed_ns(&self) -> u64 {
        self.inner.origin.elapsed().as_nanos() as u64
    }
}

impl Default for PhaseClock {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LiveLine {
    bar: ProgressBar,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl LiveLine {
    pub fn start(clock: PhaseClock) -> Self {
        let bar = ProgressBar::with_draw_target(None, ProgressDrawTarget::stdout());
        bar.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").expect("template"));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker_bar = bar.clone();
        let worker = thread::spawn(move || {
            while !worker_stop.load(Ordering::Relaxed) {
                let [formula, bsp, walksat] = clock.seconds();
                worker_bar.set_message(format!(
                    "formula {formula:.3}s | bsp {bsp:.3}s | walksat {walksat:.3}s"
                ));
                thread::park_timeout(Duration::from_millis(100));
            }
        });
        bar.enable_steady_tick(Duration::from_millis(120));
        Self {
            bar,
            stop,
            worker: Some(worker),
        }
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            let _ = worker.join();
        }
        self.bar.finish_and_clear();
    }
}

impl Drop for LiveLine {
    fn drop(&mut self) {
        if self.worker.is_some() {
            self.stop();
        }
    }
}
