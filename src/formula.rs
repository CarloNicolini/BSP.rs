use crate::{Error, FeistelTopology, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Formula(pub(crate) Arc<Data>);

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

impl Formula {
    /// Normalize repeated literals and omit tautological clauses. Empty clauses stay.
    pub fn new(n: usize, clauses: Vec<Vec<i32>>) -> Result<Self> {
        Self::from_clauses(n, clauses, false)
    }

    fn from_clauses(n: usize, clauses: Vec<Vec<i32>>, preserve_order: bool) -> Result<Self> {
        if n > i32::MAX as usize {
            return Err(Error("too many variables for packed literals".into()));
        }
        let mut clauses = clauses.into_iter();
        Self::build(n, preserve_order, (0, 0), |packed| {
            let Some(clause) = clauses.next() else {
                return Ok(false);
            };
            packed.reserve(clause.len());
            for lit in clause {
                let var = lit.unsigned_abs() as usize;
                if var == 0 || var > n {
                    return Err(Error(format!("literal {lit} outside 1..={n}")));
                }
                packed.push((((var - 1) as u32) << 1) | u32::from(lit < 0));
            }
            Ok(true)
        })
    }

    fn build(
        n: usize,
        preserve_order: bool,
        (clauses, width): (usize, usize),
        mut fill: impl FnMut(&mut Vec<u32>) -> Result<bool>,
    ) -> Result<Self> {
        let mut offsets = Vec::new();
        let mut mask_offsets = Vec::new();
        let mut edge_clause = Vec::new();
        if width == 0 {
            offsets.push(0);
            mask_offsets.push(0u32);
        }
        let mut literals = Vec::with_capacity(clauses * width);
        let mut packed = Vec::new();
        while fill(&mut packed)? {
            if preserve_order {
                let mut seen = std::collections::HashSet::with_capacity(packed.len());
                let mut tautology = false;
                packed.retain(|&lit| {
                    tautology |= seen.contains(&(lit ^ 1));
                    seen.insert(lit)
                });
                if tautology {
                    packed.clear();
                    continue;
                }
            } else {
                packed.sort_unstable();
                packed.dedup();
                if packed.windows(2).any(|w| w[0] ^ w[1] == 1) {
                    packed.clear();
                    continue;
                }
            }
            if literals
                .len()
                .checked_add(packed.len())
                .is_none_or(|e| e >= u32::MAX as usize)
                || offsets.len() >= u32::MAX as usize
            {
                return Err(Error("formula exceeds u32 edge/clause capacity".into()));
            }
            let len = packed.len();
            literals.extend_from_slice(&packed);
            packed.clear();
            if width == 0 {
                let c = (offsets.len() - 1) as u32;
                edge_clause.resize(literals.len(), c);
                offsets.push(literals.len() as u32);
                mask_offsets.push(mask_offsets.last().unwrap() + len.div_ceil(64) as u32);
            } else {
                debug_assert_eq!(len, width);
            }
        }
        let mut var_offsets = vec![0u32; 2 * n + 1];
        for &lit in &literals {
            var_offsets[lit as usize + 1] += 1;
        }
        for i in 1..var_offsets.len() {
            var_offsets[i] += var_offsets[i - 1];
        }
        let mut var_edges = vec![0; literals.len()];
        for (e, &lit) in literals.iter().enumerate() {
            let slot = &mut var_offsets[lit as usize];
            var_edges[*slot as usize] = e as u32;
            *slot += 1;
        }
        // The fill advanced each group start to its end. Shift those ends
        // right to recover the original CSR offsets without a second array.
        var_offsets.copy_within(..2 * n, 1);
        var_offsets[0] = 0;
        Ok(Self(Arc::new(Data {
            n,
            feistel: None,
            width,
            offsets,
            literals,
            edge_clause,
            var_offsets,
            var_edges,
            mask_offsets,
        })))
    }
    /// Construct an implicit partitioned formula. Only scalar CPU solving is supported.
    pub fn feistel(k: usize, alpha: f64, n: usize, seed: u64) -> Result<Self> {
        if n > i32::MAX as usize || !alpha.is_finite() || alpha <= 0.0 {
            return Err(Error(
                "Feistel requires N <= i32::MAX and finite alpha > 0".into(),
            ));
        }
        let m = (alpha * n as f64).floor();
        if m < 1.0 || m >= u64::MAX as f64 {
            return Err(Error("Feistel clause count outside 1..u64::MAX".into()));
        }
        let topology = FeistelTopology::new(k as u64, m as u64, n as u64, seed)?;
        Self::from_feistel(n, topology)
    }

    /// Construct an implicit formula targeting the ordinary random K-SAT ensemble.
    /// This scans the original matching once to replace repeated-variable clauses.
    pub fn feistel_fluctuating(k: usize, alpha: f64, n: usize, seed: u64) -> Result<Self> {
        if n > i32::MAX as usize || !alpha.is_finite() || alpha < 0.0 {
            return Err(Error(
                "Feistel requires N <= i32::MAX and finite alpha >= 0".into(),
            ));
        }
        let m = (alpha * n as f64).floor();
        if m > (1u64 << 53) as f64 {
            return Err(Error(
                "Fluctuating Feistel clause count exceeds 2^53".into(),
            ));
        }
        let edges = (m as u64).checked_mul(k as u64);
        if edges.is_none_or(|e| usize::try_from(e).is_err()) {
            return Err(Error("Feistel exceeds address width".into()));
        }
        Self::from_feistel(
            n,
            FeistelTopology::new_fluctuating(k as u64, m as u64, n as u64, seed)?,
        )
    }

    fn from_feistel(n: usize, topology: FeistelTopology) -> Result<Self> {
        usize::try_from(topology.edges())
            .map_err(|_| Error("Feistel exceeds address width".into()))?;
        Ok(Self(Arc::new(Data {
            n,
            width: topology.k() as usize,
            feistel: Some(topology),
            offsets: Vec::new(),
            literals: Vec::new(),
            edge_clause: Vec::new(),
            var_offsets: Vec::new(),
            var_edges: Vec::new(),
            mask_offsets: Vec::new(),
        })))
    }
    pub fn feistel_topology(&self) -> Option<&FeistelTopology> {
        self.0.feistel.as_ref()
    }
    /// Owned topology payload including vector capacity; excludes Arc headers and allocator overhead.
    pub fn topology_bytes(&self) -> usize {
        let f = &self.0;
        std::mem::size_of::<Data>()
            + 4 * (f.offsets.capacity()
                + f.literals.capacity()
                + f.edge_clause.capacity()
                + f.var_offsets.capacity()
                + f.var_edges.capacity()
                + f.mask_offsets.capacity())
            + f.feistel.as_ref().map_or(0, FeistelTopology::heap_bytes)
    }
    pub fn n(&self) -> usize {
        self.0.n
    }
    pub fn m(&self) -> usize {
        self.0.m()
    }
    pub fn edges(&self) -> usize {
        self.0.edges()
    }
    /// Explicitly materialize clauses. Solving an implicit formula never calls this method.
    pub fn clauses(&self) -> Vec<Vec<i32>> {
        (0..self.m())
            .map(|c| {
                self.0
                    .clause_range(c)
                    .map(|e| {
                        let l = self.0.literal(e);
                        let v = ((l >> 1) + 1) as i32;
                        if l & 1 == 0 {
                            v
                        } else {
                            -v
                        }
                    })
                    .collect()
            })
            .collect()
    }
    pub fn is_satisfied(&self, assignment: &[i8]) -> bool {
        assignment.len() == self.n()
            && assignment.iter().all(|&v| v == 0 || v == 1)
            && (0..self.m()).all(|c| {
                self.0.clause_range(c).any(|e| {
                    let l = self.0.literal(e);
                    assignment[(l >> 1) as usize] == (1 - (l & 1)) as i8
                })
            })
    }
    /// Stream DIMACS without materializing the clauses or the output string.
    pub fn write_dimacs(&self, mut out: impl std::io::Write) -> std::io::Result<()> {
        writeln!(out, "p cnf {} {}", self.n(), self.m())?;
        for c in 0..self.m() {
            for e in self.0.clause_range(c) {
                let l = self.0.literal(e);
                let v = ((l >> 1) + 1) as i32;
                write!(out, "{} ", if l & 1 == 0 { v } else { -v })?;
            }
            writeln!(out, "0")?;
        }
        Ok(())
    }
    pub fn from_dimacs(text: &str) -> Result<Self> {
        Self::read_dimacs(std::io::Cursor::new(text.as_bytes()), false)
    }

    /// Parse DIMACS while retaining the first occurrence of each literal in file order.
    pub fn from_dimacs_preserving_order(text: &str) -> Result<Self> {
        Self::read_dimacs(std::io::Cursor::new(text.as_bytes()), true)
    }

    /// Parse buffered DIMACS input without materializing the complete text or clause list.
    pub fn from_dimacs_reader(reader: impl std::io::BufRead) -> Result<Self> {
        Self::read_dimacs(reader, false)
    }

    /// Parse buffered DIMACS while retaining the first occurrence of each literal.
    pub fn from_dimacs_reader_preserving_order(reader: impl std::io::BufRead) -> Result<Self> {
        Self::read_dimacs(reader, true)
    }

    fn read_dimacs(mut reader: impl std::io::BufRead, preserve_order: bool) -> Result<Self> {
        let mut line = String::new();
        let (n, m) = loop {
            line.clear();
            if reader
                .read_line(&mut line)
                .map_err(|e| Error(e.to_string()))?
                == 0
            {
                return Err(Error("missing DIMACS header".into()));
            }
            let text = line.trim();
            if text.is_empty() || text.starts_with('c') {
                continue;
            }
            if !text.starts_with('p') {
                return Err(Error("DIMACS literals before header".into()));
            }
            let parts: Vec<_> = text.split_whitespace().collect();
            if parts.len() != 4 || parts[0] != "p" || parts[1] != "cnf" {
                return Err(Error(
                    "expected one 'p cnf N M' header before literals".into(),
                ));
            }
            let parse = |s: &str| {
                s.parse::<usize>()
                    .map_err(|_| Error("invalid DIMACS header count".into()))
            };
            break (parse(parts[2])?, parse(parts[3])?);
        };
        if n > i32::MAX as usize {
            return Err(Error("too many variables for packed literals".into()));
        }
        let mut pos = line.len();
        let mut count = 0usize;
        Self::build(n, preserve_order, (0, 0), |packed| loop {
            if pos >= line.len() {
                line.clear();
                if reader
                    .read_line(&mut line)
                    .map_err(|e| Error(e.to_string()))?
                    == 0
                {
                    if !packed.is_empty() || count != m {
                        return Err(Error("DIMACS clause count or terminator mismatch".into()));
                    }
                    return Ok(false);
                }
                if line.trim_start().starts_with('c') {
                    pos = line.len();
                    continue;
                }
                pos = 0;
            }
            let remaining = &line[pos..];
            let Some(token) = remaining.split_whitespace().next() else {
                pos = line.len();
                continue;
            };
            let offset = token.as_ptr() as usize - line.as_ptr() as usize;
            let lit = token
                .parse::<i32>()
                .map_err(|_| Error(format!("invalid DIMACS literal: {token}")))?;
            pos = offset + token.len();
            if lit == 0 {
                count += 1;
                if count > m {
                    return Err(Error("DIMACS clause count or terminator mismatch".into()));
                }
                return Ok(true);
            }
            let var = lit.unsigned_abs() as usize;
            if var > n {
                return Err(Error(format!("literal {lit} outside 1..={n}")));
            }
            packed.push((((var - 1) as u32) << 1) | u32::from(lit < 0));
        })
    }
    pub fn to_dimacs(&self) -> String {
        let mut bytes = Vec::new();
        self.write_dimacs(&mut bytes)
            .expect("writing to Vec cannot fail");
        String::from_utf8(bytes).expect("DIMACS is ASCII")
    }
    pub fn random(k: usize, alpha: f64, n: usize, seed: u64) -> Result<Self> {
        if k == 0 || k > n || n > i32::MAX as usize || !alpha.is_finite() || alpha < 0.0 {
            return Err(Error(
                "require 1 <= k <= n <= i32::MAX and finite alpha >= 0".into(),
            ));
        }
        let m = (alpha * n as f64).floor();
        if !m.is_finite() || m * k as f64 >= u32::MAX as f64 {
            return Err(Error("random formula exceeds u32 capacity".into()));
        }
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut pool: Vec<i32> = (1..=n as i32).collect();
        let m = m as usize;
        let mut remaining = m;
        Self::build(n, false, (m, k), |packed| {
            if remaining == 0 {
                return Ok(false);
            }
            remaining -= 1;
            // Same partial Fisher-Yates draws, written directly to packed CSR.
            packed.reserve(k);
            for i in 0..k {
                let j = rng.random_range(i..n);
                pool.swap(i, j);
                let v = (pool[i] - 1) as u32;
                packed.push((v << 1) | u32::from(!rng.random::<bool>()));
            }
            Ok(true)
        })
    }
}

// A byte covers every mask word when all clauses have at most eight literals.
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
