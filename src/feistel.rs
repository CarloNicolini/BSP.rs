//! Versioned Feistel topologies with reversible incidence queries.
mod fluctuating;
use crate::{Error, Result};
use fluctuating::Fluctuating;
use std::collections::BTreeMap;

/// Reproducible implicit formula descriptor with 64-bit topology IDs.
///
/// The partitioned constructor uses fixed groups. The fluctuating constructor
/// targets the ordinary random K-SAT ensemble. Queries panic on invalid indices.
#[derive(Clone, Debug)]
pub struct FeistelTopology {
    n: u64,
    m: u64,
    k: u64,
    seed: u64,
    matching: Permutation,
    fluctuating: Option<Box<Fluctuating>>,
}

// SplitMix64 finalizer; a deterministic mixer, not a cryptographic PRF.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
const MATCH: u64 = 0x243f6a8885a308d3;
const SIGN: u64 = 0x13198a2e03707344;
const ROUND: u64 = 0x9e3779b97f4a7c15;

#[derive(Clone, Debug)]
struct Permutation {
    limit: u64,
    bits: u32,
}

impl Permutation {
    fn new(limit: u64) -> Self {
        let bits = (64 - limit.saturating_sub(1).leading_zeros())
            .div_ceil(2)
            .max(1);
        Self { limit, bits }
    }

    fn raw(&self, x: u64, key: u64, inverse: bool) -> u64 {
        let mask = (1u64 << self.bits) - 1;
        let (mut left, mut right) = (x >> self.bits, x & mask);
        for j in 0..8u64 {
            let round = if inverse { 7 - j } else { j };
            let rk = mix(key ^ ROUND.wrapping_mul(round + 1));
            if inverse {
                (left, right) = (right ^ (mix(left ^ rk) & mask), left);
            } else {
                (left, right) = (right, left ^ (mix(right ^ rk) & mask));
            }
        }
        (left << self.bits) | right
    }
    fn walk(&self, mut x: u64, key: u64, inverse: bool) -> (u64, u64) {
        let mut steps = 0;
        loop {
            x = self.raw(x, key, inverse);
            steps += 1;
            if x < self.limit {
                return (x, steps);
            }
        }
    }
}

impl FeistelTopology {
    pub const VERSION: &'static str = "partitioned-feistel-v1";

    /// Construct directly from integer dimensions, without traversing the graph.
    pub fn new(k: u64, m: u64, n: u64, seed: u64) -> Result<Self> {
        if k == 0 || k > n || k > 64 || m == 0 || m.checked_mul(k).is_none() {
            return Err(Error(
                "Feistel requires 1 <= K <= min(N,64), M >= 1, and K*M <= u64::MAX".into(),
            ));
        }
        Ok(Self {
            n,
            m,
            k,
            seed,
            matching: Permutation::new(m),
            fluctuating: None,
        })
    }
    pub const FLUCTUATING_VERSION: &'static str = "fluctuating-feistel-v1";

    /// Construct an unrestricted matching and scan it for clauses to resample.
    /// The multinomial degree sampler uses floating point. Both N and K*M
    /// are limited to 2^53 so counts can be represented exactly as f64.
    pub fn new_fluctuating(k: u64, m: u64, n: u64, seed: u64) -> Result<Self> {
        const MAX_EXACT: u64 = 1 << 53;
        if k == 0
            || k > n
            || k > 64
            || n > MAX_EXACT
            || m.checked_mul(k).is_none_or(|edges| edges > MAX_EXACT)
        {
            return Err(Error(
                "Fluctuating Feistel requires 1 <= K <= min(N,64) and N, K*M <= 2^53".into(),
            ));
        }
        Ok(Self {
            n,
            m,
            k,
            seed,
            matching: Permutation::new(m),
            fluctuating: Some(Box::new(Fluctuating::new(n, k, m, seed))),
        })
    }

    /// Stable identifier for the chosen generation algorithm.
    pub fn version(&self) -> &'static str {
        if self.fluctuating.is_some() {
            Self::FLUCTUATING_VERSION
        } else {
            Self::VERSION
        }
    }

    /// Number of clauses replaced after the initial matching.
    pub fn repair_count(&self) -> usize {
        self.fluctuating.as_ref().map_or(0, |f| f.repair_count())
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.fluctuating.as_ref().map_or(0, |f| f.heap_bytes())
    }

    pub fn n(&self) -> u64 {
        self.n
    }
    pub fn m(&self) -> u64 {
        self.m
    }
    pub fn k(&self) -> u64 {
        self.k
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn edges(&self) -> u64 {
        self.m * self.k
    }

    /// Start and size of partition q for the partitioned ensemble.
    /// Panics for a fluctuating descriptor.
    pub fn partition(&self, q: u64) -> (u64, u64) {
        assert!(self.fluctuating.is_none() && q < self.k);
        let base = self.n / self.k;
        let extra = self.n % self.k;
        (q * base + q.min(extra), base + u64::from(q < extra))
    }
    fn variable_partition(&self, v: u64) -> (u64, u64, u64) {
        assert!(v < self.n);
        let base = self.n / self.k;
        let extra = self.n % self.k;
        let split = if extra == 0 { 0 } else { extra * (base + 1) };
        let q = if v < split {
            v / (base + 1)
        } else {
            extra + (v - split) / base
        };
        let (start, size) = self.partition(q);
        (q, v - start, size)
    }
    fn key(&self, q: u64) -> u64 {
        mix(self.seed ^ MATCH ^ mix(q))
    }
    fn walk(&self, x: u64, q: u64, inverse: bool) -> (u64, u64) {
        self.matching.walk(x, self.key(q), inverse)
    }
    /// Raw permutation applications for a group-local index in partitioned mode.
    pub fn walk_steps(&self, index: u64, q: u64, inverse: bool) -> u64 {
        assert!(self.fluctuating.is_none() && index < self.m && q < self.k);
        self.walk(index, q, inverse).1
    }
    /// Raw permutation applications for an index in the original matching.
    /// Partitioned variable stubs are interleaved by group for this diagnostic.
    pub fn edge_walk_steps(&self, index: u64, inverse: bool) -> u64 {
        assert!(index < self.edges());
        self.fluctuating.as_ref().map_or_else(
            || self.walk_steps(index / self.k, index % self.k, inverse),
            |f| f.walk_steps(index, inverse),
        )
    }
    /// Zero-based variable and negation bit for a clause-major edge ID.
    pub fn literal(&self, edge: u64) -> (u64, bool) {
        assert!(edge < self.edges());
        if let Some(f) = &self.fluctuating {
            return (f.variable(edge, self.k), self.negative(edge));
        }
        let q = edge % self.k;
        let t = self.walk(edge / self.k, q, false).0;
        let (start, size) = self.partition(q);
        let d = self.m / size;
        let extra = self.m % size;
        let split = if extra == 0 { 0 } else { extra * (d + 1) };
        // If d == 0 every valid stub is in the first branch.
        let local = if t < split {
            t / (d + 1)
        } else {
            extra + (t - split) / d
        };
        (start + local, self.negative(edge))
    }
    pub(crate) fn negative(&self, edge: u64) -> bool {
        let domain = if self.fluctuating.is_some() {
            0xbe5466cf34e90c6c
        } else {
            SIGN
        };
        mix(edge ^ mix(self.seed ^ domain)) & 1 != 0
    }
    fn partition_stubs(&self, v: u64) -> (u64, std::ops::Range<u64>) {
        let (q, local, size) = self.variable_partition(v);
        let d = self.m / size;
        let extra = self.m % size;
        let start = local * d + local.min(extra);
        let degree = d + u64::from(local < extra);
        (q, start..start + degree)
    }

    /// Clause-major edge IDs incident to a variable, in stub order followed by repairs.
    pub fn variable_edges(&self, v: u64) -> impl DoubleEndedIterator<Item = u64> + '_ {
        let (q, stubs, inserted) = if let Some(f) = &self.fluctuating {
            (0, f.original_stubs(v), f.inserted_edges(v))
        } else {
            let (q, stubs) = self.partition_stubs(v);
            (q, stubs, &[][..])
        };
        stubs
            .filter_map(move |t| {
                if let Some(f) = &self.fluctuating {
                    f.original_edge(t, self.k)
                } else {
                    Some(self.walk(t, q, true).0 * self.k + q)
                }
            })
            .chain(inserted.iter().map(|&(_, e)| e))
    }
    pub fn degree(&self, v: u64) -> u64 {
        if let Some(f) = &self.fluctuating {
            f.degree(v)
        } else {
            let (_, range) = self.partition_stubs(v);
            range.end - range.start
        }
    }
    /// Exact degree histogram. Fluctuating mode traverses the implicit degree tree.
    pub fn degree_counts(&self) -> BTreeMap<u64, u64> {
        if let Some(f) = &self.fluctuating {
            return f.degree_counts();
        }
        let mut counts = BTreeMap::new();
        for q in 0..self.k {
            let (_, size) = self.partition(q);
            let (d, extra) = (self.m / size, self.m % size);
            *counts.entry(d).or_default() += size - extra;
            if extra > 0 {
                *counts.entry(d + 1).or_default() += extra;
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exhaustive_permutations() {
        for m in 1..=260 {
            for seed in [0, 1, 42, u64::MAX] {
                let f = FeistelTopology::new(3, m, 100, seed).unwrap();
                for q in 0..3 {
                    let mut seen = vec![false; m as usize];
                    let mut steps = 0;
                    for a in 0..m {
                        let (t, count) = f.walk(a, q, false);
                        assert!(!seen[t as usize]);
                        seen[t as usize] = true;
                        assert_eq!(f.walk(t, q, true).0, a);
                        assert_eq!(f.walk(f.walk(a, q, true).0, q, false).0, a);
                        steps += count;
                    }
                    assert!(steps <= 1u64 << (2 * f.matching.bits));
                }
            }
        }
    }
    #[test]
    fn invalid_dimensions() {
        for (k, m, n) in [
            (0, 1, 1),
            (2, 1, 1),
            (65, 1, 65),
            (1, 0, 1),
            (3, u64::MAX, 100),
        ] {
            assert!(FeistelTopology::new(k, m, n, 0).is_err());
        }
        let f = FeistelTopology::new(1, u64::MAX, 1, 0).unwrap();
        for a in [0, 1, u64::MAX - 1] {
            assert_eq!(f.walk(f.walk(a, 0, false).0, 0, true).0, a);
        }
    }
}
