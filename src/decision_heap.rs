/// An indexed min-heap for fixed decisions. Scores stay frozen while a variable
/// is in the heap; arbitrary release removes it before its score can change.
/// Each variable has at most one entry, so repeated fixing cannot grow stale keys.
pub(crate) struct DecisionHeap {
    heap: Vec<u32>,
    positions: Vec<u32>,
}

impl DecisionHeap {
    pub fn new(n: usize) -> Self {
        Self {
            heap: Vec::with_capacity(n),
            positions: vec![u32::MAX; n],
        }
    }
    pub fn len(&self) -> usize {
        self.heap.len()
    }
    pub fn peek(&self) -> Option<u32> {
        self.heap.first().copied()
    }
    #[inline]
    fn less(a: u32, b: u32, scores: &[f64]) -> bool {
        scores[a as usize]
            .total_cmp(&scores[b as usize])
            .then_with(|| a.cmp(&b))
            .is_lt()
    }
    fn swap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.positions[self.heap[a] as usize] = a as u32;
        self.positions[self.heap[b] as usize] = b as u32;
    }
    fn sift_up(&mut self, mut pos: usize, scores: &[f64]) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if !Self::less(self.heap[pos], self.heap[parent], scores) {
                break;
            }
            self.swap(pos, parent);
            pos = parent;
        }
    }
    fn sift_down(&mut self, mut pos: usize, scores: &[f64]) {
        loop {
            let left = 2 * pos + 1;
            if left >= self.heap.len() {
                break;
            }
            let right = left + 1;
            let child = if right < self.heap.len()
                && Self::less(self.heap[right], self.heap[left], scores)
            {
                right
            } else {
                left
            };
            if !Self::less(self.heap[child], self.heap[pos], scores) {
                break;
            }
            self.swap(pos, child);
            pos = child;
        }
    }
    pub fn insert(&mut self, variable: u32, scores: &[f64]) {
        debug_assert_eq!(self.positions[variable as usize], u32::MAX);
        let pos = self.heap.len();
        self.positions[variable as usize] = pos as u32;
        self.heap.push(variable);
        self.sift_up(pos, scores);
    }
    pub fn remove(&mut self, variable: u32, scores: &[f64]) {
        let pos = self.positions[variable as usize] as usize;
        debug_assert_ne!(pos, u32::MAX as usize);
        self.heap.swap_remove(pos);
        self.positions[variable as usize] = u32::MAX;
        if pos == self.heap.len() {
            return;
        }
        self.positions[self.heap[pos] as usize] = pos as u32;
        if pos > 0 && Self::less(self.heap[pos], self.heap[(pos - 1) / 2], scores) {
            self.sift_up(pos, scores);
        } else {
            self.sift_down(pos, scores);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    #[test]
    fn indexed_heap_matches_scan_after_arbitrary_removal_and_reinsertion() {
        let n = 257;
        let mut heap = DecisionHeap::new(n);
        let mut scores = vec![0.0; n];
        let mut present = vec![false; n];
        let mut rng = SmallRng::seed_from_u64(12);
        for _ in 0..20_000 {
            let v = rng.random_range(0..n);
            if present[v] {
                heap.remove(v as u32, &scores);
            } else {
                scores[v] = rng.random_range(0..8) as f64 / 8.0;
                heap.insert(v as u32, &scores);
            }
            present[v] = !present[v];
            let expected = (0..n)
                .filter(|&i| present[i])
                .min_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(a.cmp(&b)))
                .map(|v| v as u32);
            assert_eq!(heap.peek(), expected);
            assert_eq!(heap.len(), present.iter().filter(|&&b| b).count());
            for (pos, &v) in heap.heap.iter().enumerate() {
                assert_eq!(heap.positions[v as usize], pos as u32);
                if pos > 0 {
                    assert!(!DecisionHeap::less(v, heap.heap[(pos - 1) / 2], &scores));
                }
            }
        }
    }
}
