use bsp_rust::{RunClock, RunPhase, RunSnapshot};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle, TermLike};
use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub struct LiveLine {
    bar: ProgressBar,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    stopped: bool,
}

impl LiveLine {
    pub fn start(clock: RunClock) -> Self {
        let bar = ProgressBar::with_draw_target(
            None,
            ProgressDrawTarget::term_like_with_hz(Box::new(StdoutTerm), 10),
        );
        bar.set_style(
            ProgressStyle::with_template("{spinner:.green} {msg}").expect("progress template"),
        );
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker_bar = bar.clone();
        let worker = thread::spawn(move || {
            while !worker_stop.load(Ordering::Relaxed) {
                worker_bar.set_message(live_message(&clock));
                thread::sleep(Duration::from_millis(100));
            }
        });
        bar.enable_steady_tick(Duration::from_millis(100));
        Self {
            bar,
            stop,
            worker: Some(worker),
            stopped: false,
        }
    }

    pub fn stop(&mut self) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.bar.finish_and_clear();
    }
}

impl Drop for LiveLine {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn final_line(snap: &RunSnapshot, path: &Path) -> String {
    format!(
        "formula {:.3}s | bsp {:.3}s | walksat {:.3}s | {}",
        snap.formula_seconds,
        snap.bsp_seconds,
        snap.walksat_seconds,
        path.display()
    )
}

fn live_message(clock: &RunClock) -> String {
    let snap = clock.snapshot();
    format!(
        "{} {:.3}s | {} {:.3}s | {} {:.3}s | sweeps {} steps {} free {} flips {}",
        label(snap.phase == RunPhase::Formula, "formula"),
        snap.formula_seconds,
        label(snap.phase == RunPhase::Bsp, "bsp"),
        snap.bsp_seconds,
        label(snap.phase == RunPhase::Walksat, "walksat"),
        snap.walksat_seconds,
        snap.sweeps,
        snap.steps,
        snap.free_variables,
        snap.flips
    )
}

fn label(active: bool, name: &str) -> String {
    if active {
        format!("\x1b[1;32m{name}\x1b[0m")
    } else {
        format!("\x1b[2m{name}\x1b[0m")
    }
}

#[derive(Debug)]
struct StdoutTerm;

impl TermLike for StdoutTerm {
    fn width(&self) -> u16 {
        let columns = console::Term::stdout().size().1;
        if io::stdout().is_terminal() {
            columns.max(40)
        } else {
            120
        }
    }

    fn move_cursor_up(&self, n: usize) -> io::Result<()> {
        self.cursor('A', n)
    }

    fn move_cursor_down(&self, n: usize) -> io::Result<()> {
        self.cursor('B', n)
    }

    fn move_cursor_right(&self, n: usize) -> io::Result<()> {
        self.cursor('C', n)
    }

    fn move_cursor_left(&self, n: usize) -> io::Result<()> {
        self.cursor('D', n)
    }

    fn write_line(&self, text: &str) -> io::Result<()> {
        let mut out = io::stdout().lock();
        writeln!(out, "{text}")?;
        out.flush()
    }

    fn write_str(&self, text: &str) -> io::Result<()> {
        let mut out = io::stdout().lock();
        out.write_all(text.as_bytes())?;
        out.flush()
    }

    fn clear_line(&self) -> io::Result<()> {
        self.write_str("\r\x1b[2K")
    }

    fn flush(&self) -> io::Result<()> {
        io::stdout().flush()
    }
}

impl StdoutTerm {
    fn cursor(&self, direction: char, n: usize) -> io::Result<()> {
        if n > 0 {
            self.write_str(&format!("\x1b[{n}{direction}"))?;
        }
        Ok(())
    }
}
