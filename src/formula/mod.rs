mod data;
mod dimacs;

use crate::{Error, FeistelTopology, Result};
pub(crate) use data::{ClauseMasks, Data};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Formula(pub(crate) Arc<Data>);

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
    /// Construct an implicit partitioned formula for CPU or Metal solving.
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
    pub(crate) fn data(&self) -> &Data {
        &self.0
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
