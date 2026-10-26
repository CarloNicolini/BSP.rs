/// Word masks with a hierarchy of nonempty-word summaries. Dense iteration uses
/// the flat leaf array; sparse iteration visits only nonempty summary branches.
#[derive(Clone, Debug)]
pub(crate) struct BitSet {
    levels: Vec<Vec<u64>>,
    count: usize,
}

// Enough base-64 levels to address the target's complete usize index space.
const MAX_LEVELS: usize = (usize::BITS as usize).div_ceil(6);

impl BitSet {
    pub fn full(len: usize) -> Self {
        let mut levels = Vec::new();
        let mut bits = len;
        loop {
            let mut words = vec![u64::MAX; bits.div_ceil(64)];
            if !bits.is_multiple_of(64) {
                *words.last_mut().unwrap() = (1u64 << (bits % 64)) - 1;
            }
            bits = words.len();
            levels.push(words);
            if bits <= 1 {
                break;
            }
        }
        Self { levels, count: len }
    }
    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        self.levels[0][i / 64] & (1u64 << (i % 64)) != 0
    }
    pub fn set(&mut self, i: usize, value: bool) {
        let mut index = i / 64;
        let mask = 1u64 << (i % 64);
        let mut before = self.levels[0][index];
        let mut after = if value { before | mask } else { before & !mask };
        if before == after {
            return;
        }
        self.levels[0][index] = after;
        if value {
            self.count += 1;
        } else {
            self.count -= 1;
        }
        for level in 1..self.levels.len() {
            if (before == 0) == (after == 0) {
                break;
            }
            let mask = 1u64 << (index % 64);
            index /= 64;
            let nonempty = after != 0;
            before = self.levels[level][index];
            after = if nonempty {
                before | mask
            } else {
                before & !mask
            };
            self.levels[level][index] = after;
        }
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn ones(&self) -> BitIter<'_> {
        let level = self.levels.len() - 1;
        let mut pending = [0; MAX_LEVELS];
        pending[level] = self.levels[level].first().copied().unwrap_or(0);
        BitIter {
            set: self,
            word: 0,
            base: 0,
            leaf: 0,
            sparse: level > 0 && self.count < self.levels[0].len().div_ceil(8),
            level,
            pending,
            bases: [0; MAX_LEVELS],
        }
    }
}

pub(crate) struct BitIter<'a> {
    set: &'a BitSet,
    word: u64,
    base: usize,
    leaf: usize,
    sparse: bool,
    level: usize,
    pending: [u64; MAX_LEVELS],
    bases: [usize; MAX_LEVELS],
}
impl BitIter<'_> {
    fn next_word(&mut self) -> Option<(usize, u64)> {
        if !self.sparse {
            while self.leaf < self.set.levels[0].len() {
                let leaf = self.leaf;
                self.leaf += 1;
                let word = self.set.levels[0][leaf];
                if word != 0 {
                    return Some((leaf, word));
                }
            }
            return None;
        }
        while self.level < self.set.levels.len() {
            let word = &mut self.pending[self.level];
            if *word == 0 {
                self.level += 1;
                continue;
            }
            let bit = word.trailing_zeros() as usize;
            *word &= *word - 1;
            let index = self.bases[self.level] + bit;
            if self.level == 1 {
                return Some((index, self.set.levels[0][index]));
            }
            self.level -= 1;
            self.pending[self.level] = self.set.levels[self.level][index];
            self.bases[self.level] = index * 64;
        }
        None
    }
}
impl Iterator for BitIter<'_> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.word == 0 {
            let (leaf, word) = self.next_word()?;
            self.base = leaf * 64;
            self.word = word;
        }
        let bit = self.word.trailing_zeros() as usize;
        self.word &= self.word - 1;
        Some(self.base + bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    fn check(set: &BitSet, expected: &[bool]) {
        let bits: Vec<_> = expected
            .iter()
            .enumerate()
            .filter_map(|(i, &yes)| yes.then_some(i))
            .collect();
        assert_eq!(set.ones().collect::<Vec<_>>(), bits);
        assert_eq!(set.len(), bits.len());
    }
    #[test]
    fn summaries_follow_dense_sparse_and_empty_transitions() {
        for n in [0, 1, 63, 64, 65, 4095, 4096, 4097, 262145] {
            let mut set = BitSet::full(n);
            let mut expected = vec![true; n];
            check(&set, &expected);
            for i in 0..n {
                set.set(i, false);
            }
            expected.fill(false);
            check(&set, &expected);
            if n == 0 {
                continue;
            }
            for i in [0, 63, 64, 4095, 4096, n - 1] {
                if i < n {
                    set.set(i, true);
                    expected[i] = true;
                }
            }
            check(&set, &expected);
            let mut rng = SmallRng::seed_from_u64(n as u64);
            for step in 0usize..2048 {
                let i = rng.random_range(0..n);
                let value = rng.random::<bool>();
                set.set(i, value);
                expected[i] = value;
                assert_eq!(set.contains(i), value);
                if step.is_multiple_of(127) {
                    check(&set, &expected);
                }
            }
            check(&set, &expected);
        }
    }
}
