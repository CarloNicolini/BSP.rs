use super::{mix, ROUND};

#[derive(Clone, Debug)]
pub(super) struct Permutation {
    pub(super) limit: u64,
    pub(super) bits: u32,
}

impl Permutation {
    pub(super) fn new(limit: u64) -> Self {
        let bits = (64 - limit.saturating_sub(1).leading_zeros())
            .div_ceil(2)
            .max(1);
        Self { limit, bits }
    }

    pub(super) fn round_keys(key: u64) -> [u64; 8] {
        std::array::from_fn(|round| mix(key ^ ROUND.wrapping_mul(round as u64 + 1)))
    }

    pub(super) fn raw(&self, x: u64, keys: &[u64; 8], inverse: bool) -> u64 {
        let mask = (1u64 << self.bits) - 1;
        let (mut left, mut right) = (x >> self.bits, x & mask);
        if inverse {
            for &key in keys.iter().rev() {
                (left, right) = (right ^ (mix(left ^ key) & mask), left);
            }
        } else {
            for &key in keys {
                (left, right) = (right, left ^ (mix(right ^ key) & mask));
            }
        }
        (left << self.bits) | right
    }
    // Each round reads only a half-word. For small domains an exact lookup
    // table costs at most 8 * 256 bytes, independent of graph connectivity.
    pub(super) fn round_table(&self, keys: &[u64; 8]) -> Box<[u8]> {
        if self.bits > 8 || self.limit == 0 {
            return Box::default();
        }
        let width = 1u64 << self.bits;
        keys.iter()
            .flat_map(|&key| (0..width).map(move |x| (mix(x ^ key) & (width - 1)) as u8))
            .collect()
    }

    pub(super) fn walk_with_table(&self, mut x: u64, table: &[u8], inverse: bool) -> (u64, u64) {
        let width = 1usize << self.bits;
        let mask = width - 1;
        let mut steps = 0;
        loop {
            let (mut left, mut right) = (x as usize >> self.bits, x as usize & mask);
            if inverse {
                for row in table.chunks_exact(width).rev() {
                    (left, right) = (right ^ usize::from(row[left]), left);
                }
            } else {
                for row in table.chunks_exact(width) {
                    (left, right) = (right, left ^ usize::from(row[right]));
                }
            }
            x = ((left << self.bits) | right) as u64;
            steps += 1;
            if x < self.limit {
                return (x, steps);
            }
        }
    }

    pub(super) fn walk(&self, x: u64, key: u64, inverse: bool) -> (u64, u64) {
        self.walk_with_keys(x, &Self::round_keys(key), inverse)
    }

    pub(super) fn walk_with_keys(&self, mut x: u64, keys: &[u64; 8], inverse: bool) -> (u64, u64) {
        let mut steps = 0;
        loop {
            x = self.raw(x, keys, inverse);
            steps += 1;
            if x < self.limit {
                return (x, steps);
            }
        }
    }
}
