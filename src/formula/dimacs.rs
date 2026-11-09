use super::Formula;
use crate::{Error, Result};

impl Formula {
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
            if reader.read_line(&mut line)? == 0 {
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
                if reader.read_line(&mut line)? == 0 {
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
}
