use super::mutation::FreeLiterals;
use super::*;

pub(super) fn check_indexes(s: &Solver) {
    let f = s.formula.data();
    let mut degree = vec![0u32; f.n];
    for c in 0..s.formula.m() {
        let mut free = 0usize;
        let mut true_count = 0;
        for e in f.offset(c)..f.offset(c + 1) {
            let lit = f.literal(e);
            let v = (lit >> 1) as usize;
            let is_free = s.assignment[v] < 0;
            let local = e - f.offset(c);
            let mask_free =
                s.masks.get(f.mask_offset(c) + local / 64) & (1u64 << (local % 64)) != 0;
            assert_eq!(is_free, mask_free);
            if is_free {
                free += 1;
            } else if s.assignment[v] == (1 - (lit & 1)) as i8 {
                true_count += 1;
            }
        }
        let expected = match free {
            0 => FreeLiterals::None,
            1 => FreeLiterals::One(
                (f.offset(c)..f.offset(c + 1))
                    .find(|&e| s.assignment[(f.literal(e) >> 1) as usize] < 0)
                    .unwrap(),
            ),
            _ => FreeLiterals::Many,
        };
        assert_eq!(s.free_literals(c), expected);
        assert_eq!(s.true_count.get(c), true_count);
        assert_eq!(s.active.contains(c), true_count == 0);
        if true_count == 0 {
            for e in f.offset(c)..f.offset(c + 1) {
                let v = (f.literal(e) >> 1) as usize;
                if s.assignment[v] < 0 {
                    degree[v] += 1;
                }
            }
        }
    }
    assert_eq!(s.degree, degree);
    let eligible: Vec<_> = (0..f.n)
        .filter(|&v| s.assignment[v] >= 0 && !s.forced[v])
        .collect();
    let best = eligible
        .iter()
        .copied()
        .min_by(|&a, &b| s.cert[a].total_cmp(&s.cert[b]).then(a.cmp(&b)));
    assert_eq!(s.decisions.len(), eligible.len());
    assert_eq!(s.decisions.peek().map(|v| v as usize), best);
}
#[test]
fn generated_width_indexes_match_generic_indexes() {
    for (k, alpha) in [(1, 1.0), (3, 4.0), (64, 1.0), (65, 1.0), (3, 0.0)] {
        let generated = Formula::random(k, alpha, 130, 17).unwrap();
        let generic = Formula::new(generated.n(), generated.clauses()).unwrap();
        let f = generated.data();
        assert!(f.offsets.is_empty() && f.edge_clause.is_empty() && f.mask_offsets.is_empty());
        assert_eq!(f.var_offsets, generic.data().var_offsets);
        assert_eq!(f.var_edges, generic.data().var_edges);
        assert_eq!(generated.to_dimacs(), generic.to_dimacs());
        for e in 0..f.edges() {
            assert_eq!(f.edge_clause(e), generic.data().edge_clause(e));
        }
        for c in 0..=f.m() {
            assert_eq!(f.mask_offset(c), generic.data().mask_offset(c));
        }

        let mut a = Solver::new(generated, Options::default()).unwrap();
        let mut b = Solver::new(generic, Options::default()).unwrap();
        assert_eq!(a.unit_queue, b.unit_queue);
        for v in [0, 63, 64, 129] {
            a.fix(v, true).unwrap();
            b.fix(v, true).unwrap();
            check_indexes(&a);
            assert_eq!(a.degree, b.degree);
        }
    }
}
#[test]
fn cached_indexes_match_full_recounts_through_mixed_mutations() {
    let mut rng = SmallRng::seed_from_u64(91);
    for seed in 0..5 {
        let formula = Formula::random(5, 3.0, 130, seed).unwrap();
        let mut s = Solver::new(formula, Options::default()).unwrap();
        check_indexes(&s);
        for step in 0usize..1200 {
            let v = rng.random_range(0..130);
            if s.assignment[v] < 0 {
                s.fix(v, rng.random()).unwrap();
            } else if !s.forced[v] {
                s.release(v).unwrap();
            }
            if step.is_multiple_of(41) {
                let _ = s.propagate();
            }
            check_indexes(&s);
        }
    }
}
#[test]
fn heap_backtracking_matches_sorted_frozen_scores_for_batches() {
    let mut s = Solver::new(Formula::random(4, 1.0, 130, 7).unwrap(), Options::default()).unwrap();
    for v in 0..100 {
        s.cert[v] = (v % 7) as f64 / 7.0;
        s.fix(v, v.is_multiple_of(2)).unwrap();
    }
    for v in [0, 12, 63, 64, 91] {
        s.release(v).unwrap();
        s.fix(v, false).unwrap();
    }
    s.last_batch = 9;
    s.pending_units = 3;
    let mut expected: Vec<_> = (0..100).collect();
    expected.sort_unstable_by(|&a, &b| s.cert[a].total_cmp(&s.cert[b]).then(a.cmp(&b)));
    assert_eq!(s.backtrack(), 12);
    for v in 0..100 {
        assert_eq!(s.assignment[v] < 0, expected[..12].contains(&v));
    }
    check_indexes(&s);
}
