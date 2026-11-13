use crate::bitset::Ones;
use crate::FeistelTopology;

#[derive(Debug)]
pub(crate) struct Data {
    pub n: usize,
    pub feistel: Option<FeistelTopology>,
    // Zero means the generic offset tables are used.
    pub width: usize,
    // Clause-major CSR; literal = (zero-based variable << 1) | negative.
    pub offsets: Vec<u32>,
    pub literals: Vec<u32>,
    pub edge_clause: Vec<u32>,
    // Sign-separated variable CSR: group 2*v is positive, 2*v+1 negative.
    pub var_offsets: Vec<u32>,
    pub var_edges: Vec<u32>,
    pub mask_offsets: Vec<u32>,
}

// A byte covers every mask word when all clauses have at most eight literals.
#[cfg_attr(test, derive(Clone))]
pub(crate) enum ClauseMasks {
    Byte(Vec<u8>),
    Word(Vec<u64>),
}

impl ClauseMasks {
    #[inline]
    pub fn get(&self, index: usize) -> u64 {
        match self {
            Self::Byte(masks) => u64::from(masks[index]),
            Self::Word(masks) => masks[index],
        }
    }

    /// Edges of clause `c` whose variable is still free, in increasing order.
    /// The SP sweeps walk the words directly: this adaptor measured slower there.
    #[inline]
    pub fn free_edges(&self, f: &Data, c: usize) -> impl Iterator<Item = usize> + '_ {
        let start = f.offset(c);
        let first = f.mask_offset(c);
        (first..f.mask_offset(c + 1)).flat_map(move |w| Ones {
            word: self.get(w),
            base: start + 64 * (w - first),
        })
    }

    #[inline]
    pub fn set_bit(&mut self, word: usize, bit: usize, free: bool) {
        let mask = 1u64 << bit;
        match self {
            Self::Byte(masks) => {
                debug_assert!(bit < 8);
                masks[word] = (masks[word] & !(mask as u8)) | if free { mask as u8 } else { 0 };
            }
            Self::Word(masks) => {
                masks[word] = (masks[word] & !mask) | if free { mask } else { 0 };
            }
        }
    }
}

impl Data {
    pub fn m(&self) -> usize {
        self.edges()
            .checked_div(self.width)
            .unwrap_or_else(|| self.offsets.len() - 1)
    }
    pub fn edges(&self) -> usize {
        self.feistel
            .as_ref()
            .map_or_else(|| self.literals.len(), |f| f.edges() as usize)
    }
    #[inline]
    pub fn offset(&self, c: usize) -> usize {
        if self.width == 0 {
            self.offsets[c] as usize
        } else {
            c * self.width
        }
    }
    pub fn clause_range(&self, c: usize) -> std::ops::Range<usize> {
        self.offset(c)..self.offset(c + 1)
    }
    #[inline]
    pub fn literal(&self, e: usize) -> u32 {
        self.feistel.as_ref().map_or_else(
            || self.literals[e],
            |f| {
                let (v, negative) = f.literal(e as u64);
                ((v as u32) << 1) | u32::from(negative)
            },
        )
    }
    #[inline]
    pub fn edge_clause(&self, e: usize) -> usize {
        match self.width {
            0 => self.edge_clause[e] as usize,
            3 => e / 3,
            k => e / k,
        }
    }
    #[inline]
    pub fn mask_offset(&self, c: usize) -> usize {
        if self.width == 0 {
            self.mask_offsets[c] as usize
        } else {
            c * self.width.div_ceil(64)
        }
    }

    #[cfg(any(
        all(feature = "metal", target_os = "macos"),
        all(feature = "cuda", target_os = "linux")
    ))]
    pub fn device_clause_indexes(
        &self,
    ) -> (std::borrow::Cow<'_, [u32]>, std::borrow::Cow<'_, [u32]>) {
        use std::borrow::Cow;
        if self.width == 0 {
            (
                Cow::Borrowed(&self.offsets),
                Cow::Borrowed(&self.edge_clause),
            )
        } else {
            (
                Cow::Owned((0..=self.m()).map(|c| self.offset(c) as u32).collect()),
                Cow::Owned(
                    (0..self.m())
                        .flat_map(|c| std::iter::repeat_n(c as u32, self.width))
                        .collect(),
                ),
            )
        }
    }

    pub fn initial_masks(&self, max_width: usize) -> ClauseMasks {
        if max_width <= 8 {
            let mut masks = Vec::with_capacity(self.mask_offset(self.m()));
            if self.width != 0 {
                masks.resize(self.m(), ((1u16 << self.width) - 1) as u8);
            } else {
                for offsets in self.offsets.windows(2) {
                    let width = offsets[1] - offsets[0];
                    if width != 0 {
                        masks.push(((1u16 << width) - 1) as u8);
                    }
                }
            }
            return ClauseMasks::Byte(masks);
        }
        let mut masks = Vec::with_capacity(self.mask_offset(self.m()));
        for c in 0..self.m() {
            let width = self.offset(c + 1) - self.offset(c);
            masks.extend(std::iter::repeat_n(u64::MAX, width / 64));
            if !width.is_multiple_of(64) {
                masks.push(u64::MAX >> (64 - width % 64));
            }
        }
        ClauseMasks::Word(masks)
    }
    #[inline]
    pub fn negative(&self, e: usize) -> bool {
        self.feistel
            .as_ref()
            .map_or_else(|| self.literals[e] & 1 != 0, |f| f.negative(e as u64))
    }

    pub fn variable_edges<'a>(
        &'a self,
        v: usize,
        scratch: &'a mut Vec<u64>,
    ) -> impl Iterator<Item = usize> + 'a {
        let explicit = if let Some(f) = &self.feistel {
            f.ordered_edges(v as u64, scratch);
            &[][..]
        } else {
            scratch.clear();
            &self.var_edges[self.var_offsets[2 * v] as usize..self.var_offsets[2 * v + 2] as usize]
        };
        explicit
            .iter()
            .map(|&e| e as usize)
            .chain(scratch.iter().map(|&e| e as u32 as usize))
    }
}
