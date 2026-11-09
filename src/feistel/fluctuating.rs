use super::{mix, Permutation};
use rand::{rand_core::impls::fill_bytes_via_next, RngCore};
use rand_distr::{Binomial, Distribution};
use std::{collections::BTreeMap, ops::Range};

const SPLIT_CACHE_SLOTS: u64 = 65_536;
const DEGREES: u64 = 0xa4093822299f31d0;
const MATCH: u64 = 0x082efa98ec4e6c89;
const REPAIR: u64 = 0x452821e638d01377;

struct Stream {
    key: u64,
    counter: u64,
}

impl Stream {
    fn new(key: u64) -> Self {
        Self { key, counter: 0 }
    }

    // Rejection avoids modulo bias given uniform source words.
    fn below(&mut self, end: u64) -> u64 {
        let cutoff = end.wrapping_neg() % end;
        loop {
            let value = self.next_u64();
            if value >= cutoff {
                return value % end;
            }
        }
    }
}

impl RngCore for Stream {
    fn next_u64(&mut self) -> u64 {
        let value = mix(self.key ^ self.counter);
        self.counter = self.counter.wrapping_add(1);
        value
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        fill_bytes_via_next(self, dest);
    }
}

/// A multinomial occupancy vector represented by a seed. Both directions
/// reproduce the same split counts along a root-to-leaf path.
#[derive(Clone, Debug)]
struct DegreeTree {
    n: u64,
    edges: u64,
    key: u64,
    upper_splits: Box<[u64]>,
}

impl DegreeTree {
    fn new(n: u64, edges: u64, key: u64) -> Self {
        // At most 512 KiB, independent of N and E. Cache upper binomial
        // counts for forward queries and reverse interval lookup.
        fn fill(tree: &DegreeTree, cache: &mut [u64], node: usize, size: u64, count: u64) {
            if node >= cache.len() || size <= 1 || count == 0 {
                return;
            }
            let left = tree.split(node as u64, size, count);
            cache[node] = left;
            fill(tree, cache, 2 * node, size / 2, left);
            fill(tree, cache, 2 * node + 1, size - size / 2, count - left);
        }
        let mut tree = Self {
            n,
            edges,
            key,
            upper_splits: Box::default(),
        };
        if n > 1 && edges > 0 {
            let mut cache = vec![0; n.next_power_of_two().min(SPLIT_CACHE_SLOTS) as usize];
            fill(&tree, &mut cache, 1, n, edges);
            tree.upper_splits = cache.into_boxed_slice();
        }
        tree
    }

    fn split(&self, node: u64, size: u64, count: u64) -> u64 {
        if count == 0 {
            return 0;
        }
        if node < self.upper_splits.len() as u64 {
            return self.upper_splits[node as usize];
        }
        let mut rng = Stream::new(mix(self.key ^ node));
        Binomial::new(count, (size / 2) as f64 / size as f64)
            .expect("valid split probability")
            .sample(&mut rng)
    }

    fn owner(&self, mut stub: u64) -> u64 {
        let (mut node, mut first, mut size, mut count) = (1, 0, self.n, self.edges);
        debug_assert!(stub < count);
        while size > 1 {
            let half = size / 2;
            let left = self.split(node, size, count);
            node *= 2;
            if stub < left {
                size = half;
                count = left;
            } else {
                first += half;
                size -= half;
                stub -= left;
                count -= left;
                node += 1;
            }
        }
        first
    }

    fn stubs(&self, v: u64) -> Range<u64> {
        assert!(v < self.n);
        let (mut node, mut first, mut size, mut count, mut start) = (1, 0, self.n, self.edges, 0);
        while size > 1 && count > 0 {
            let half = size / 2;
            let left = self.split(node, size, count);
            node *= 2;
            if v < first + half {
                size = half;
                count = left;
            } else {
                first += half;
                size -= half;
                count -= left;
                start += left;
                node += 1;
            }
        }
        start..start + count
    }

    fn initial_degrees(&self) -> Vec<u32> {
        fn fill(tree: &DegreeTree, node: u64, count: u64, degrees: &mut [u32]) {
            if count == 0 {
                return;
            }
            if degrees.len() == 1 {
                degrees[0] = count as u32;
                return;
            }
            let size = degrees.len();
            let left = tree.split(node, size as u64, count);
            let (first, second) = degrees.split_at_mut(size / 2);
            fill(tree, node * 2, left, first);
            fill(tree, node * 2 + 1, count - left, second);
        }
        debug_assert!(self.edges < u32::MAX as u64);
        let mut degrees = vec![0; self.n as usize];
        fill(self, 1, self.edges, &mut degrees);
        degrees
    }

    fn histogram(&self) -> BTreeMap<u64, u64> {
        fn visit(
            tree: &DegreeTree,
            node: u64,
            size: u64,
            count: u64,
            out: &mut BTreeMap<u64, u64>,
        ) {
            if count == 0 {
                *out.entry(0).or_default() += size;
            } else if size == 1 {
                *out.entry(count).or_default() += 1;
            } else {
                let left = tree.split(node, size, count);
                let half = size / 2;
                visit(tree, node * 2, half, left, out);
                visit(tree, node * 2 + 1, size - half, count - left, out);
            }
        }
        let mut out = BTreeMap::new();
        visit(self, 1, self.n, self.edges, &mut out);
        out
    }
}

#[derive(Clone, Debug)]
pub(super) struct Fluctuating {
    tree: DegreeTree,
    matching: Permutation,
    match_keys: [u64; 8],
    round_table: Box<[u8]>,
    // Sorted clause IDs and their corresponding K replacement variables.
    repairs: Vec<u64>,
    replacements: Vec<u64>,
    // Sorted (variable, edge) entries for reverse lookup.
    insertions: Vec<(u64, u64)>,
    degree_delta: Vec<(u64, i64)>,
}

impl Fluctuating {
    pub(super) fn new(n: u64, k: u64, m: u64, seed: u64) -> Self {
        let edges = m * k;
        let matching = Permutation::new(edges);
        let match_keys = Permutation::round_keys(mix(seed ^ MATCH));
        let round_table = matching.round_table(&match_keys);
        let mut topology = Self {
            tree: DegreeTree::new(n, edges, mix(seed ^ DEGREES)),
            matching,
            match_keys,
            round_table,
            repairs: Vec::new(),
            replacements: Vec::new(),
            insertions: Vec::new(),
            degree_delta: Vec::new(),
        };
        if k == 1 {
            return topology;
        }

        let repair_key = mix(seed ^ REPAIR);
        let mut old = [0; 64];
        let old = &mut old[..k as usize];
        let mut changes = BTreeMap::<u64, i64>::new();
        for clause in 0..m {
            let first = clause * k;
            for (slot, v) in old.iter_mut().enumerate() {
                *v = topology.raw_variable(first + slot as u64);
            }
            if !old.iter().enumerate().any(|(i, v)| old[..i].contains(v)) {
                continue;
            }
            for v in old.iter() {
                *changes.entry(*v).or_default() -= 1;
            }
            topology.repairs.push(clause);
            let mut rng = Stream::new(mix(repair_key ^ clause));
            // Floyd sampling gives a uniform subset using K bounded draws.
            // Shuffle it so repaired and original clauses share the slot law.
            let mut replacement = [0; 64];
            let replacement = &mut replacement[..k as usize];
            for (i, j) in (n - k..n).enumerate() {
                let candidate = rng.below(j + 1);
                replacement[i] = if replacement[..i].contains(&candidate) {
                    j
                } else {
                    candidate
                };
            }
            for i in (1..replacement.len()).rev() {
                replacement.swap(i, rng.below(i as u64 + 1) as usize);
            }
            for (slot, &v) in replacement.iter().enumerate() {
                topology.insertions.push((v, first + slot as u64));
                *changes.entry(v).or_default() += 1;
            }
            topology.replacements.extend_from_slice(replacement);
        }
        topology.insertions.sort_unstable();
        topology.degree_delta = changes
            .into_iter()
            .filter(|(_, delta)| *delta != 0)
            .collect();
        topology
    }

    fn walk(&self, index: u64, inverse: bool) -> (u64, u64) {
        if self.round_table.is_empty() {
            self.matching
                .walk_with_keys(index, &self.match_keys, inverse)
        } else {
            self.matching
                .walk_with_table(index, &self.round_table, inverse)
        }
    }

    fn raw_variable(&self, e: u64) -> u64 {
        self.tree.owner(self.walk(e, false).0)
    }

    pub(super) fn variable(&self, e: u64, k: u64) -> u64 {
        if let Ok(i) = self.repairs.binary_search(&(e / k)) {
            self.replacements[i * k as usize + (e % k) as usize]
        } else {
            self.raw_variable(e)
        }
    }

    pub(super) fn original_stubs(&self, v: u64) -> Range<u64> {
        self.tree.stubs(v)
    }

    pub(super) fn original_edge(&self, stub: u64, k: u64) -> Option<u64> {
        let e = self.walk(stub, true).0;
        self.repairs.binary_search(&(e / k)).is_err().then_some(e)
    }

    pub(super) fn inserted_edges(&self, v: u64) -> &[(u64, u64)] {
        let first = self.insertions.partition_point(|&(var, _)| var < v);
        let last = first + self.insertions[first..].partition_point(|&(var, _)| var == v);
        &self.insertions[first..last]
    }

    pub(super) fn degree(&self, v: u64) -> u64 {
        let range = self.tree.stubs(v);
        let delta = self
            .degree_delta
            .binary_search_by_key(&v, |&(var, _)| var)
            .map_or(0, |i| self.degree_delta[i].1);
        (range.end - range.start)
            .checked_add_signed(delta)
            .expect("nonnegative degree")
    }

    pub(super) fn initial_degrees(&self) -> Vec<u32> {
        let mut degrees = self.tree.initial_degrees();
        for &(v, delta) in &self.degree_delta {
            let degree = &mut degrees[v as usize];
            *degree = (i64::from(*degree) + delta) as u32;
        }
        degrees
    }

    pub(super) fn degree_counts(&self) -> BTreeMap<u64, u64> {
        let mut counts = self.tree.histogram();
        for &(v, delta) in &self.degree_delta {
            let stubs = self.tree.stubs(v);
            let old = stubs.end - stubs.start;
            let new = old.checked_add_signed(delta).expect("nonnegative degree");
            *counts.get_mut(&old).expect("old degree in histogram") -= 1;
            *counts.entry(new).or_default() += 1;
        }
        counts.retain(|_, variables| *variables != 0);
        counts
    }

    pub(super) fn repair_count(&self) -> usize {
        self.repairs.len()
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn metal_metadata(&self) -> (Vec<u32>, Vec<u32>, Vec<[u32; 2]>) {
        // Sample the original tree on the CPU so the GPU uses exactly the same
        // integer stub intervals, including stubs of subsequently repaired clauses.
        let mut offsets = self.tree.initial_degrees();
        offsets.push(0);
        let mut start = 0;
        for degree in &mut offsets {
            let end = start + *degree;
            *degree = start;
            start = end;
        }
        (
            offsets,
            self.repairs.iter().map(|&c| c as u32).collect(),
            self.insertions
                .iter()
                .map(|&(v, e)| [v as u32, e as u32])
                .collect(),
        )
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn match_keys(&self) -> [u64; 8] {
        self.match_keys
    }

    pub(super) fn walk_steps(&self, index: u64, inverse: bool) -> u64 {
        self.walk(index, inverse).1
    }

    pub(super) fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.round_table.len()
            + 8 * (self.tree.upper_splits.len()
                + self.repairs.capacity()
                + self.replacements.capacity())
            + 16 * (self.insertions.capacity() + self.degree_delta.capacity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_split_cache_matches_regeneration() {
        for n in [33, 500, 1003] {
            for seed in [0, 7, u64::MAX] {
                let cached = DegreeTree::new(n, 12 * n, mix(seed ^ DEGREES));
                assert!(cached.upper_splits.len() <= SPLIT_CACHE_SLOTS as usize);
                let mut uncached = cached.clone();
                uncached.upper_splits = Box::default();
                for v in 0..n {
                    assert_eq!(cached.stubs(v), uncached.stubs(v));
                }
            }
        }
    }

    #[test]
    fn sampled_large_matching_inverts_without_a_repair_scan() {
        let n = 1_000_000_000;
        let edges = 12 * n;
        let tree = DegreeTree::new(n, edges, mix(11 ^ DEGREES));
        let matching = Permutation::new(edges);
        let key = mix(11 ^ MATCH);
        for edge in [0, edges - 1, 1u64 << 32]
            .into_iter()
            .chain((0..1024).map(|i| mix(i) % edges))
        {
            let stub = matching.walk(edge, key, false).0;
            let v = tree.owner(stub);
            assert!(tree.stubs(v).contains(&stub));
            assert_eq!(matching.walk(stub, key, true).0, edge);
        }
    }
}
