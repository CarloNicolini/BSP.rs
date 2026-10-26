use bsp_rust::Formula;

// Independent, deliberately quadratic f32 definition; no prefix/suffix code.
pub fn f32_oracle(f: &Formula, eta: &[f64], fixed: &[i8], damping: f32) -> Vec<f64> {
    let clauses = f.clauses();
    let mut offsets = vec![0];
    for cl in &clauses {
        offsets.push(offsets.last().unwrap() + cl.len());
    }
    let active: Vec<_> = clauses
        .iter()
        .map(|cl| {
            !cl.iter()
                .any(|&l| fixed[l.unsigned_abs() as usize - 1] == i8::from(l > 0))
        })
        .collect();
    let mut out = vec![0.0; eta.len()];
    for (c, cl) in clauses.iter().enumerate() {
        if !active[c] {
            continue;
        }
        for (target, &lit) in cl.iter().enumerate() {
            if fixed[lit.unsigned_abs() as usize - 1] >= 0 {
                continue;
            }
            let mut product = 1.0f32;
            for (j, &neighbor) in cl.iter().enumerate() {
                if target == j || fixed[neighbor.unsigned_abs() as usize - 1] >= 0 {
                    continue;
                }
                let mut same = 1.0f32;
                let mut opposite = 1.0f32;
                for (b, other) in clauses.iter().enumerate() {
                    if b == c || !active[b] {
                        continue;
                    }
                    for (p, &l) in other.iter().enumerate() {
                        if l == neighbor {
                            same *= 1.0 - eta[offsets[b] + p] as f32;
                        }
                        if l == -neighbor {
                            opposite *= 1.0 - eta[offsets[b] + p] as f32;
                        }
                    }
                }
                product *= (1.0 - opposite) * same / (opposite + (1.0 - opposite) * same);
            }
            let raw = if product < 1e-16 { 0.0 } else { product };
            let e = offsets[c] + target;
            out[e] = (raw + (eta[e] as f32 - raw) * damping) as f64;
        }
    }
    out
}

pub fn excluded_underflow_case() -> (Formula, Vec<f64>) {
    // A single exact zero hides underflow of its excluded product. The
    // opposite group remains subnormal, so the lost ratio is significant.
    let mut clauses = Vec::new();
    for v in 2..=12 {
        clauses.push(vec![1, v]);
    }
    for v in 13..=21 {
        clauses.push(vec![-1, v]);
    }
    let f = Formula::new(21, clauses).unwrap();
    let mut values = vec![0.0; f.edges()];
    for e in (0..values.len()).step_by(2) {
        values[e] = 1.0 - 2f64.powi(-16);
    }
    values[0] = 1.0;
    (f, values)
}
