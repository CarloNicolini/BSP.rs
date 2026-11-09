use bsp_rust::Step;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tabled::settings::{object::Columns, Alignment, Style, Width};
use tabled::Table;

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

const ROW_SECONDS: u64 = 3;
const COLUMNS: [(&str, usize); 9] = [
    ("time", 8),
    ("step", 6),
    ("action", 9),
    ("free", 10),
    ("clauses", 10),
    ("sweeps", 10),
    ("residual", 10),
    ("complexity", 10),
    ("changed", 8),
];

/// One borderless table row, padded to the column widths so rows stack like one table.
fn table_row(cells: [String; 9]) -> String {
    let mut table = Table::from_iter([cells]);
    table.with(Style::blank()).with(Alignment::right());
    for (index, (_, width)) in COLUMNS.iter().enumerate() {
        table.modify(Columns::one(index), Width::increase(*width));
    }
    table.to_string().trim_end().to_string()
}

fn step_row(seconds: f64, step: &Step) -> String {
    table_row([
        format!("{seconds:.1}s"),
        step.index.to_string(),
        format!("{:?}", step.action).to_lowercase(),
        step.free_variables.to_string(),
        step.active_clauses.to_string(),
        step.sp_sweeps.to_string(),
        format!("{:.2e}", step.residual),
        format!("{:.4}", step.complexity),
        step.changed.to_string(),
    ])
}

/// Free-variable bar and status line, plus a table row on stdout every few seconds.
pub struct LiveLine {
    multi: MultiProgress,
    latest: Arc<Mutex<Option<Step>>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl LiveLine {
    pub fn start(clock: PhaseClock) -> Self {
        let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stdout());
        let decided = multi.add(ProgressBar::new(0));
        decided.set_style(
            ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} variables decided")
                .expect("template"),
        );
        let status = multi.add(ProgressBar::new_spinner());
        status.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").expect("template"));
        status.enable_steady_tick(Duration::from_millis(120));
        let latest = Arc::new(Mutex::new(None::<Step>));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = thread::spawn({
            let (multi, latest, stop) = (multi.clone(), Arc::clone(&latest), Arc::clone(&stop));
            let origin = Instant::now();
            move || {
                let mut last_row = origin;
                let mut printed_header = false;
                loop {
                    let finished = stop.load(Ordering::Relaxed);
                    let [formula, bsp, walksat] = clock.seconds();
                    let step = *latest.lock().unwrap();
                    let mut message =
                        format!("formula {formula:.1}s | bsp {bsp:.1}s | walksat {walksat:.1}s");
                    if let Some(step) = step {
                        decided.set_length(
                            decided
                                .length()
                                .unwrap_or(0)
                                .max(step.free_variables as u64),
                        );
                        decided.set_position(
                            decided.length().unwrap_or(0) - step.free_variables as u64,
                        );
                        message += &format!(
                            " | step {} sweeps {} complexity {:.4}",
                            step.index, step.sp_sweeps, step.complexity
                        );
                        if finished || last_row.elapsed().as_secs() >= ROW_SECONDS {
                            if !printed_header {
                                let _ =
                                    multi.println(table_row(COLUMNS.map(|(name, _)| name.into())));
                                printed_header = true;
                            }
                            let _ = multi.println(step_row(origin.elapsed().as_secs_f64(), &step));
                            last_row = Instant::now();
                        }
                    }
                    status.set_message(message);
                    if finished {
                        break;
                    }
                    thread::park_timeout(Duration::from_millis(100));
                }
                decided.finish_and_clear();
                status.finish_and_clear();
            }
        });
        Self {
            multi,
            latest,
            stop,
            worker: Some(worker),
        }
    }

    /// Callback for `Solver::on_step` that feeds the display.
    pub fn step_hook(&self) -> Arc<dyn Fn(&Step) + Send + Sync> {
        let latest = Arc::clone(&self.latest);
        Arc::new(move |step| *latest.lock().unwrap() = Some(*step))
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            let _ = worker.join();
        }
        let _ = self.multi.clear();
    }
}

impl Drop for LiveLine {
    fn drop(&mut self) {
        if self.worker.is_some() {
            self.stop();
        }
    }
}
