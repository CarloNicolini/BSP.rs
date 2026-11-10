/// Fixed-capacity word mask. Iteration clears the lowest set bit in a register.
#[derive(Clone, Debug)]
pub(crate) struct BitSet {
    words: Vec<u64>,
    count: usize,
}

impl BitSet {
    pub fn full(len: usize) -> Self {
        let mut words = vec![u64::MAX; len.div_ceil(64)];
        if !len.is_multiple_of(64) {
            *words.last_mut().unwrap() = (1u64 << (len % 64)) - 1;
        }
        Self { words, count: len }
    }
    #[cfg(test)]
    pub fn empty(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            count: 0,
        }
    }
    #[inline]
    pub fn contains(&self, i: usize) -> bool {
        self.words[i / 64] & (1u64 << (i % 64)) != 0
    }
    pub fn set(&mut self, i: usize, value: bool) {
        let word = &mut self.words[i / 64];
        let mask = 1u64 << (i % 64);
        let old = *word & mask != 0;
        if old != value {
            *word ^= mask;
            if value {
                self.count += 1;
            } else {
                self.count -= 1;
            }
        }
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    /// Prefix counts for a bitset whose membership will remain fixed.
    pub fn rank_index(&self) -> Vec<u32> {
        let mut total = 0;
        self.words
            .iter()
            .map(|word| {
                let before = total;
                total += word.count_ones();
                before
            })
            .collect()
    }

    #[inline]
    pub fn rank_before(&self, index: &[u32], bit: usize) -> usize {
        let word = bit / 64;
        let lower = (1u64 << (bit % 64)) - 1;
        (index[word] + (self.words[word] & lower).count_ones()) as usize
    }

    pub fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(base, &word)| Ones {
                word,
                base: 64 * base,
            })
    }
}

pub(crate) struct Ones {
    pub word: u64,
    pub base: usize,
}
impl Iterator for Ones {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.word == 0 {
            return None;
        }
        let bit = self.word.trailing_zeros() as usize;
        self.word &= self.word - 1;
        Some(self.base + bit)
    }
}
