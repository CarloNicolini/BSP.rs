use crate::{Error, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Formula(pub(crate) Arc<Data>);

#[derive(Debug)]
pub(crate) struct Data {
    pub n: usize,
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
            packed.sort_unstable();
            packed.dedup();
            if packed.windows(2).any(|w| w[0] ^ w[1] == 1) {
                continue;
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
            offsets,
            literals,
            edge_clause,
            var_offsets,
            var_edges,
            mask_offsets,
            masks,
        })))
    }
    pub fn n(&self) -> usize {
        self.0.n
    }
    pub fn m(&self) -> usize {
        self.0.offsets.len() - 1
    }
    pub fn edges(&self) -> usize {
        self.0.literals.len()
    }
    pub fn clauses(&self) -> Vec<Vec<i32>> {
        self.0
            .offsets
            .windows(2)
            .map(|w| {
                self.0.literals[w[0] as usize..w[1] as usize]
                    .iter()
                    .map(|&l| {
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
            && self.0.offsets.windows(2).all(|w| {
                self.0.literals[w[0] as usize..w[1] as usize]
                    .iter()
                    .any(|&l| assignment[(l >> 1) as usize] == (1 - (l & 1)) as i8)
            })
    }
    pub fn from_dimacs(text: &str) -> Result<Self> {
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
        Self::new(n, clauses)
    }
    pub fn to_dimacs(&self) -> String {
        use std::fmt::Write;
        let mut text = format!("p cnf {} {}\n", self.n(), self.m());
        for clause in self.clauses() {
            for lit in clause {
                write!(text, "{lit} ").unwrap();
            }
            text.push_str("0\n");
        }
        text
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
