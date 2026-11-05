use crate::{Error, FeistelTopology, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Formula(pub(crate) Arc<Data>);

#[derive(Debug)]
pub(crate) struct Data {
    pub n: usize,
    pub feistel: Option<FeistelTopology>,
    // Clause-major CSR; literal = (zero-based variable << 1) | negative.
    pub offsets: Vec<u32>,
    pub literals: Vec<u32>,
    pub edge_clause: Vec<u32>,
    // Sign-separated variable CSR: group 2*v is positive, 2*v+1 negative.
    pub var_offsets: Vec<u32>,
    pub var_edges: Vec<u32>,
    pub mask_offsets: Vec<u32>,
    pub masks: Vec<u64>,
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
        let mut offsets = vec![0];
        let mut literals = Vec::new();
        let mut edge_clause = Vec::new();
        let mut masks = Vec::new();
        let mut mask_offsets = vec![0];
        for clause in clauses {
            let mut packed = Vec::with_capacity(clause.len());
            for lit in clause {
                let var = lit.unsigned_abs() as usize;
                if var == 0 || var > n {
                    return Err(Error(format!("literal {lit} outside 1..={n}")));
                }
                packed.push((((var - 1) as u32) << 1) | u32::from(lit < 0));
            }
            if preserve_order {
                let mut seen = std::collections::HashSet::with_capacity(packed.len());
                let mut tautology = false;
                packed.retain(|&lit| {
                    tautology |= seen.contains(&(lit ^ 1));
                    seen.insert(lit)
                });
                if tautology {
                    continue;
                }
            } else {
                packed.sort_unstable();
                packed.dedup();
                if packed.windows(2).any(|w| w[0] ^ w[1] == 1) {
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
            let c = (offsets.len() - 1) as u32;
            let len = packed.len();
            literals.extend(packed);
            edge_clause.resize(literals.len(), c);
            offsets.push(literals.len() as u32);
            for start in (0..len).step_by(64) {
                let width = (len - start).min(64);
                masks.push(u64::MAX >> (64 - width));
            }
            mask_offsets
                .push(u32::try_from(masks.len()).map_err(|_| Error("too many mask words".into()))?);
        }
        let mut var_offsets = vec![0u32; 2 * n + 1];
        for &lit in &literals {
            var_offsets[lit as usize + 1] += 1;
        }
        for i in 1..var_offsets.len() {
            var_offsets[i] += var_offsets[i - 1];
        }
        let mut cursor = var_offsets[..2 * n].to_vec();
        let mut var_edges = vec![0; literals.len()];
        for (e, &lit) in literals.iter().enumerate() {
            let slot = &mut cursor[lit as usize];
            var_edges[*slot as usize] = e as u32;
            *slot += 1;
        }
        Ok(Self(Arc::new(Data {
            n,
            feistel: None,
            offsets,
            literals,
            edge_clause,
            var_offsets,
            var_edges,
            mask_offsets,
            masks,
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
            feistel: Some(topology),
            offsets: Vec::new(),
            literals: Vec::new(),
            edge_clause: Vec::new(),
            var_offsets: Vec::new(),
            var_edges: Vec::new(),
            mask_offsets: Vec::new(),
            masks: Vec::new(),
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
            + 8 * f.masks.capacity()
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
        Self::parse_dimacs(text, false)
    }

    /// Parse DIMACS while retaining the first occurrence of each literal in file order.
    pub fn from_dimacs_preserving_order(text: &str) -> Result<Self> {
        Self::parse_dimacs(text, true)
    }

    fn parse_dimacs(text: &str, preserve_order: bool) -> Result<Self> {
        let mut header = None;
        let mut clauses = Vec::new();
        let mut clause = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('c') {
                continue;
            }
            if line.starts_with('p') {
                let p: Vec<_> = line.split_whitespace().collect();
                if header.is_some()
                    || !clauses.is_empty()
                    || !clause.is_empty()
                    || p.len() != 4
                    || p[1] != "cnf"
                {
                    return Err(Error(
                        "expected one 'p cnf N M' header before literals".into(),
                    ));
                }
                let parse = |s: &str| {
                    s.parse::<usize>()
                        .map_err(|_| Error("invalid DIMACS header count".into()))
                };
                header = Some((parse(p[2])?, parse(p[3])?));
                continue;
            }
            if header.is_none() {
                return Err(Error("DIMACS literals before header".into()));
            }
            for token in line.split_whitespace() {
                let lit = token
                    .parse::<i32>()
                    .map_err(|_| Error(format!("invalid DIMACS literal: {token}")))?;
                if lit == 0 {
                    clauses.push(std::mem::take(&mut clause));
                } else {
                    clause.push(lit);
                }
            }
        }
        let (n, m) = header.ok_or_else(|| Error("missing DIMACS header".into()))?;
        if !clause.is_empty() || clauses.len() != m {
            return Err(Error("DIMACS clause count or terminator mismatch".into()));
        }
        Self::from_clauses(n, clauses, preserve_order)
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
        let mut clauses = Vec::with_capacity(m as usize);
        for _ in 0..m as usize {
            // Partial Fisher-Yates; O(k), including dense k/n, without rejection.
            let mut clause = Vec::with_capacity(k);
            for i in 0..k {
                let j = rng.random_range(i..n);
                pool.swap(i, j);
                let v = pool[i];
                clause.push(if rng.random::<bool>() { v } else { -v });
            }
            clauses.push(clause);
        }
        Self::new(n, clauses)
    }
}

impl Data {
    pub fn m(&self) -> usize {
        self.feistel
            .as_ref()
            .map_or_else(|| self.offsets.len() - 1, |f| f.m() as usize)
    }
    pub fn edges(&self) -> usize {
        self.feistel
            .as_ref()
            .map_or_else(|| self.literals.len(), |f| f.edges() as usize)
    }
    #[inline]
    pub fn offset(&self, c: usize) -> usize {
        self.feistel
            .as_ref()
            .map_or_else(|| self.offsets[c] as usize, |f| c * f.k() as usize)
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
        self.feistel
            .as_ref()
            .map_or_else(|| self.edge_clause[e] as usize, |f| e / f.k() as usize)
    }
    #[inline]
    pub fn mask_offset(&self, c: usize) -> usize {
        self.feistel
            .as_ref()
            .map_or_else(|| self.mask_offsets[c] as usize, |_| c)
    }
    pub fn initial_masks(&self) -> Vec<u64> {
        self.feistel.as_ref().map_or_else(
            || self.masks.clone(),
            |f| vec![u64::MAX >> (64 - f.k()); f.m() as usize],
        )
    }
    pub fn degree(&self, v: usize) -> u32 {
        self.feistel.as_ref().map_or_else(
            || self.var_offsets[2 * v + 2] - self.var_offsets[2 * v],
            |f| f.degree(v as u64) as u32,
        )
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
