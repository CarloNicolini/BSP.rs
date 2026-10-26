//! Run with cargo bench --bench indexes. Uses only the package's dependencies.
#[allow(dead_code, unused_imports)]
#[path = "../src/bitset.rs"]
mod bitset;
#[allow(dead_code, unused_imports)]
#[path = "../src/decision_heap.rs"]
mod decision_heap;
#[allow(dead_code, unused_imports)]
#[path = "prototypes/sparse_bitset.rs"]
mod sparse_bitset;
use bitset::Ones;
use decision_heap::DecisionHeap;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde_json::json;
use sparse_bitset::BitSet;
use std::{hint::black_box, time::Instant};
fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[values.len() / 2]
}
fn main() {
    let mut selection = Vec::new();
    for n in [1_000, 10_000, 100_000] {
        let ops = 512;
        let mut rng = SmallRng::seed_from_u64(42);
        let scores: Vec<f64> = (0..n).map(|_| rng.random()).collect();
        let updates: Vec<f64> = (0..ops).map(|_| rng.random()).collect();
        let mut linear_times = Vec::new();
        let mut heap_times = Vec::new();
        for trial in 0..8 {
            let mut linear = scores.clone();
            let mut heap_scores = scores.clone();
            let mut heap = DecisionHeap::new(n);
            for v in 0..n {
                heap.insert(v as u32, &heap_scores);
            }
            let mut linear_order = Vec::with_capacity(ops);
            let mut heap_order = Vec::with_capacity(ops);
            // A one-pass minimum is a faster baseline than the old gather +
            // partition code. Include remove, score change, and reinsertion.
            let started = Instant::now();
            for &update in &updates {
                let view = black_box(&linear);
                let v = (0..n)
                    .min_by(|&a, &b| view[a].total_cmp(&view[b]).then(a.cmp(&b)))
                    .unwrap();
                linear_order.push(v as u32);
                linear[v] = update;
            }
            let linear_time = started.elapsed().as_secs_f64();
            let started = Instant::now();
            for &update in &updates {
                let v = black_box(heap.peek().unwrap());
                heap.remove(v, &heap_scores);
                heap_scores[v as usize] = update;
                heap.insert(v, &heap_scores);
                heap_order.push(v);
            }
            let heap_time = started.elapsed().as_secs_f64();
            assert_eq!(linear_order, heap_order);
            if trial > 0 {
                linear_times.push(linear_time);
                heap_times.push(heap_time);
            }
        }
        let linear_seconds = median(&mut linear_times);
        let heap_seconds = median(&mut heap_times);
        selection.push(json!({"n":n,"operations":ops,"linear_seconds":linear_seconds,"heap_seconds":heap_seconds,
            "speedup":linear_seconds/heap_seconds,"linear_samples":linear_times,"heap_samples":heap_times}));
    }
    let mut bitsets = Vec::new();
    for n in [4_096usize, 262_144, 1_048_576] {
        for active in [0, 1, 32, n / 2, n] {
            let mut set = BitSet::full(n);
            let mut flat = vec![0u64; n.div_ceil(64)];
            for i in 0..n {
                set.set(i, false);
            }
            for i in 0..active {
                let bit = i * (n / active);
                set.set(bit, true);
                flat[bit / 64] |= 1u64 << (bit % 64);
            }
            let iterations = if active < 100 {
                2048
            } else {
                (32_000_000 / n).max(16)
            };
            let mut flat_times = Vec::new();
            let mut index_times = Vec::new();
            for trial in 0..8 {
                let mut flat_sum = 0usize;
                let mut index_sum = 0usize;
                let started = Instant::now();
                for _ in 0..iterations {
                    for bit in black_box(&flat)
                        .iter()
                        .enumerate()
                        .flat_map(|(i, &word)| Ones { word, base: 64 * i })
                    {
                        flat_sum = flat_sum.wrapping_add(bit);
                    }
                }
                let flat_time = started.elapsed().as_secs_f64();
                black_box(flat_sum);
                let started = Instant::now();
                for _ in 0..iterations {
                    for bit in black_box(&set).ones() {
                        index_sum = index_sum.wrapping_add(bit);
                    }
                }
                let index_time = started.elapsed().as_secs_f64();
                black_box(index_sum);
                assert_eq!(flat_sum, index_sum);
                if trial > 0 {
                    flat_times.push(flat_time);
                    index_times.push(index_time);
                }
            }
            let flat_seconds = median(&mut flat_times);
            let indexed_seconds = median(&mut index_times);
            bitsets.push(json!({"universe":n,"active":active,"iterations":iterations,"flat_seconds":flat_seconds,
                "indexed_seconds":indexed_seconds,"speedup":flat_seconds/indexed_seconds,"flat_samples":flat_times,"indexed_samples":index_times}));
        }
    }
    println!("{}", json!({"selection":selection,"bitsets":bitsets}));
}
