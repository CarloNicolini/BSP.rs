//! Standalone host setup benchmark: rustc -O --edition=2021 setup.rs -o setup
#[allow(dead_code)]
#[path = "../../crates/bsp-cuda/src/layout.rs"]
mod layout;
mod windows_sort;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
struct Allocator;
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size(), Relaxed) + layout.size();
            PEAK.fetch_max(live, Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        System.dealloc(p, layout);
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let out = System.realloc(p, layout, size);
        if !out.is_null() {
            let live = LIVE.fetch_add(size, Relaxed) + size - layout.size();
            LIVE.fetch_sub(layout.size(), Relaxed);
            PEAK.fetch_max(live, Relaxed);
        }
        out
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args[2].parse().unwrap();
    let mut rng: u64 = args[3].parse().unwrap();
    let edges = (4.2 * n as f64).floor() as usize * 3;
    // Random 3-SAT (distinct variables per clause), stable sign/variable CSR. Same deterministic
    // topology for both variants; generation and checksums are outside timing.
    let mut literals: Vec<usize> = Vec::with_capacity(edges);
    for e in 0..edges {
        loop {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let literal = rng as usize % (2 * n);
            if literals[e - e % 3..]
                .iter()
                .all(|&other| other / 2 != literal / 2)
            {
                literals.push(literal);
                break;
            }
        }
    }
    let mut counts = vec![0usize; 2 * n + 1];
    for &lit in &literals {
        counts[lit + 1] += 1;
    }
    for i in 1..counts.len() {
        counts[i] += counts[i - 1];
    }
    let mut var_edges = vec![0u32; edges];
    for (e, &lit) in literals.iter().enumerate() {
        var_edges[counts[lit]] = e as u32;
        counts[lit] += 1;
    }
    drop(literals);
    drop(counts);
    let retained = LIVE.load(Relaxed);
    PEAK.store(retained, Relaxed);
    let started = std::time::Instant::now();
    let result = if args[1] == "sort" {
        windows_sort::windows(&var_edges)
    } else {
        layout::windows(&var_edges)
    };
    let seconds = started.elapsed().as_secs_f64();
    let peak = PEAK.load(Relaxed);
    let mut hash = 14695981039346656037u64;
    for values in [&result.0, &result.1, &result.2, &result.3] {
        for &value in values {
            hash = (hash ^ value as u64).wrapping_mul(1099511628211);
        }
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let rss_kib = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    println!("{{\"variant\":\"{}\",\"n\":{n},\"seed\":{},\"edges\":{edges},\"seconds\":{seconds},\"peak_live_bytes\":{peak},\"max_rss_kib\":{rss_kib},\"setup_peak_bytes\":{},\"hash\":\"{hash:016x}\"}}", args[1], args[3], peak-retained);
}
