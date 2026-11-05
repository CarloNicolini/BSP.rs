use bsp_rust::{FeistelTopology, Formula, Options, Solver, Status};
use std::collections::BTreeMap;

#[test]
fn scans_and_repairs_preserve_the_exact_reverse_index() {
    let mut repaired = 0;
    for (n, k, m) in [
        (1, 1, 20),
        (4, 3, 17),
        (8, 8, 20),
        (31, 5, 0),
        (31, 3, 1),
        (31, 5, 7),
        (64, 2, 127),
        (65, 3, 100),
        (99, 3, 396),
        (100, 3, 400),
        (257, 3, 1028),
        (1000, 3, 4000),
        (64, 64, 12),
    ] {
        for seed in 1..=12 {
            let t = FeistelTopology::new_fluctuating(k, m, n, seed).unwrap();
            repaired += t.repair_count();
            assert_eq!((t.k(), t.m(), t.n(), t.edges()), (k, m, n, k * m));

            let mut expected = vec![Vec::new(); n as usize];
            for c in 0..m {
                let mut clause = Vec::new();
                for q in 0..k {
                    let e = c * k + q;
                    let (v, _) = t.literal(e);
                    assert!(v < n);
                    assert!(!clause.contains(&v), "duplicate variable in clause {c}");
                    clause.push(v);
                    expected[v as usize].push(e);
                }
            }

            let mut histogram = BTreeMap::new();
            for (v, expected) in expected.iter().enumerate() {
                let v = v as u64;
                let mut actual: Vec<_> = t.variable_edges(v).collect();
                assert!(t.variable_edges(v).rev().eq(actual.iter().rev().copied()));
                actual.sort_unstable();
                assert_eq!(&actual, expected);
                assert_eq!(actual.len() as u64, t.degree(v));
                *histogram.entry(t.degree(v)).or_default() += 1;
            }
            assert_eq!(histogram, t.degree_counts());
        }
    }
    assert!(repaired > 100);
    assert!(FeistelTopology::new_fluctuating(3, 1 << 53, 100, 0).is_err());
    assert!(FeistelTopology::new_fluctuating(3, 1, 1 << 54, 0).is_err());
}

#[test]
fn repairs_handle_dense_and_empty_formulas_without_changing_clause_count() {
    for k in [1, 2, 3, 8, 64] {
        let n = k;
        let f = Formula::feistel_fluctuating(k, 2.0, n, 5).unwrap();
        assert_eq!(f.m(), 2 * n);
        assert!(f.clauses().iter().all(|c| c.len() == k));
        let parsed = Formula::from_dimacs(&f.to_dimacs()).unwrap();
        assert_eq!(parsed.m(), f.m());
        assert_eq!(parsed.edges(), f.edges());
    }
    let empty = Formula::feistel_fluctuating(3, 0.0, 100, 5).unwrap();
    assert_eq!(empty.edges(), 0);
    assert_eq!(
        Solver::new(empty, Options::default())
            .unwrap()
            .solve()
            .status,
        Status::Sat
    );
}

#[test]
fn paired_sweeps_and_mutations_match_explicit_normalized_graph() {
    let mut verified = 0;
    for seed in 1..=20 {
        let f = Formula::feistel_fluctuating(3, 2.5, 31, seed).unwrap();
        let clauses = f.clauses();
        let explicit = Formula::new(31, clauses.clone()).unwrap();
        let mut map = Vec::new();
        for (c, clause) in clauses.iter().enumerate() {
            let mut sorted = clause.clone();
            sorted.sort_unstable_by_key(|l| l.abs());
            map.extend(
                clause
                    .iter()
                    .map(|l| 3 * c + sorted.iter().position(|t| t == l).unwrap()),
            );
        }
        let messages: Vec<_> = (0..f.edges()).map(|e| (e % 47) as f64 / 53.0).collect();
        let mut materialized = messages.clone();
        for (e, &slot) in map.iter().enumerate() {
            materialized[slot] = messages[e];
        }
        let opts = Options {
            seed,
            damping: 0.1,
            max_steps: 64,
            max_sweeps: 128,
            max_flips: 20_000,
            ..Options::default()
        };
        let mut a = Solver::new(f.clone(), opts.clone()).unwrap();
        let mut b = Solver::new(explicit.clone(), opts.clone()).unwrap();
        a.set_messages(&messages).unwrap();
        b.set_messages(&materialized).unwrap();
        let compare = |a: &Solver, b: &Solver| {
            assert_eq!(a.active_clauses(), b.active_clauses());
            for (i, &j) in map.iter().enumerate() {
                assert!((a.messages()[i] - b.messages()[j]).abs() < 3e-12);
            }
        };
        a.fix(0, false).unwrap();
        b.fix(0, false).unwrap();
        compare(&a, &b);
        for _ in 0..8 {
            assert!((a.sweep().unwrap() - b.sweep().unwrap()).abs() < 3e-12);
            compare(&a, &b);
        }
        a.release(0).unwrap();
        b.release(0).unwrap();
        compare(&a, &b);
        let result = Solver::new(f.clone(), opts).unwrap().solve();
        if let Some(x) = result.assignment {
            assert_eq!(result.status, Status::Sat);
            assert!(f.is_satisfied(&x) && explicit.is_satisfied(&x));
            verified += 1;
        } else {
            assert_ne!(result.status, Status::Sat);
        }
    }
    assert!(verified > 0);
}

#[test]
fn fluctuating_v1_seed_reproduces_clause_and_slot_order() {
    let formula = Formula::feistel_fluctuating(3, 1.125, 8, 7).unwrap();
    let topology = formula.feistel_topology().unwrap();
    assert_eq!(topology.version(), "fluctuating-feistel-v1");
    assert!(topology.repair_count() > 0);
    assert_eq!(
        formula.to_dimacs(),
        "p cnf 8 9\n-2 3 7 0\n6 8 5 0\n-5 -4 1 0\n-5 2 -1 0\n-2 6 4 0\n-6 -3 8 0\n2 6 -1 0\n3 -7 8 0\n7 -6 -3 0\n"
    );
}

#[test]
fn cpu_parallel_preserves_messages_and_search_order() {
    use bsp_rust::{Backend, ExecutionOptions};
    type Generate = fn(usize, f64, usize, u64) -> bsp_rust::Result<Formula>;
    let cases: [(Generate, usize); 3] = [
        (Formula::feistel_fluctuating, 500),
        (Formula::feistel, 500),
        (Formula::random, 2000),
    ];
    for (generate, n) in cases {
        let f = generate(3, 4.0, n, 7).unwrap();
        let opts = Options {
            seed: 7,
            damping: 0.1,
            max_steps: 64,
            max_sweeps: 128,
            ..Options::default()
        };
        let mut scalar = Solver::new(f.clone(), opts.clone()).unwrap();
        let mut parallel = Solver::with_execution(
            f.clone(),
            opts,
            ExecutionOptions {
                backend: Backend::CpuParallel,
                threads: 4,
                ..Default::default()
            },
        )
        .unwrap();
        for v in (0..19).step_by(2) {
            scalar.fix(v, v % 4 == 0).unwrap();
            parallel.fix(v, v % 4 == 0).unwrap();
        }
        for v in [None, Some(4), Some(12), Some(16)] {
            if let Some(v) = v {
                scalar.release(v).unwrap();
                parallel.release(v).unwrap();
            }
            assert_eq!(
                scalar.sweep_many(8).unwrap().to_bits(),
                parallel.sweep_many(8).unwrap().to_bits()
            );
            assert_eq!(scalar.messages(), parallel.messages());
        }
        assert!(parallel.execution_report().parallel_sweeps > 0);
        if f.feistel_topology().is_some() {
            assert!(parallel.execution_report().allocated_bytes <= 48 * 1024);
        }
        let a = scalar.solve();
        let b = parallel.solve();
        assert_eq!(a.status, b.status);
        assert_eq!(a.assignment, b.assignment);
        assert_eq!(a.partial_assignment, b.partial_assignment);
        assert_eq!(
            serde_json::to_value(&a.steps).unwrap(),
            serde_json::to_value(&b.steps).unwrap()
        );
        if let Some(x) = a.assignment {
            assert!(f.is_satisfied(&x));
        }
    }
}
