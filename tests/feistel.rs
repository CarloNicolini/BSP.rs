use bsp_rust::{Backend, ExecutionOptions, FeistelTopology, Formula, Options, Solver, Status};

#[test]
fn exhaustive_incidence_and_degrees() {
    for n in [99, 100, 200, 500, 999, 1000] {
        for seed in 1..=20 {
            let f = Formula::feistel(3, 4.0, n, seed).unwrap();
            let t = f.feistel_topology().unwrap();
            let mut seen = vec![false; f.edges()];
            let mut degree = vec![0; n];
            for v in 0..n as u64 {
                for e in t.variable_edges(v) {
                    assert!(!seen[e as usize]);
                    seen[e as usize] = true;
                    assert_eq!(t.literal(e).0, v);
                    degree[v as usize] += 1;
                }
                assert_eq!(degree[v as usize], t.degree(v));
            }
            assert!(seen.into_iter().all(|v| v));
            for c in 0..t.m() {
                let mut previous = None;
                for q in 0..3 {
                    let (v, _) = t.literal(c * 3 + q);
                    let (start, size) = t.partition(q);
                    assert!((start..start + size).contains(&v));
                    assert!(previous.is_none_or(|p| p < v));
                    previous = Some(v);
                }
            }
            let explicit = Formula::new(n, f.clauses()).unwrap();
            assert_eq!(f.clauses(), explicit.clauses());
            assert!(f.topology_bytes() < explicit.topology_bytes());
        }
    }
}

#[test]
fn billion_variable_descriptor_and_solver_guard() {
    let f = Formula::feistel(3, 4.0, 1_000_000_000, 7).unwrap();
    let t = f.feistel_topology().unwrap();
    assert_eq!((t.m(), t.edges()), (4_000_000_000, 12_000_000_000));
    assert_eq!(
        t.degree_counts().into_iter().collect::<Vec<_>>(),
        vec![(11, 8), (12, 999_999_984), (13, 8)]
    );
    assert_eq!(
        f.topology_bytes(),
        Formula::feistel(3, 4.0, 100, 7).unwrap().topology_bytes()
    );
    for c in [0, 1, 1_431_655_765, 2_147_483_648, 3_999_999_999] {
        for q in 0..3 {
            let e = c * 3 + q;
            let (v, _) = t.literal(e);
            assert_eq!(t.variable_edges(v).filter(|&x| x == e).count(), 1);
        }
    }
    assert!(Solver::new(f.clone(), Options::default())
        .err()
        .unwrap()
        .0
        .contains("capacity"));
    for backend in [
        Backend::CpuParallel,
        Backend::Metal,
        Backend::Cuda,
        Backend::CudaF32,
    ] {
        let error = Solver::with_execution(
            f.clone(),
            Options::default(),
            ExecutionOptions {
                backend,
                ..Default::default()
            },
        )
        .err()
        .unwrap();
        assert!(error.0.contains("require --backend cpu"));
    }
}

#[test]
fn small_edge_cases_and_invalid_inputs() {
    for (k, n, alpha) in [(1, 1, 1.0), (3, 100, 0.01), (64, 64, 0.015625), (3, 4, 4.0)] {
        let f = Formula::feistel(k, alpha, n, 2).unwrap();
        let t = f.feistel_topology().unwrap();
        for v in 0..n as u64 {
            assert_eq!(t.variable_edges(v).count() as u64, t.degree(v));
            for e in t.variable_edges(v) {
                assert_eq!(t.literal(e).0, v);
            }
        }
        let mut out = Vec::new();
        f.write_dimacs(&mut out).unwrap();
        assert_eq!(
            Formula::from_dimacs(std::str::from_utf8(&out).unwrap())
                .unwrap()
                .clauses(),
            f.clauses()
        );
    }
    for (k, a, n) in [
        (0, 4.0, 100),
        (65, 4.0, 100),
        (3, 4.0, 2),
        (3, 0.0, 100),
        (3, -1.0, 100),
        (3, f64::NAN, 100),
        (3, f64::INFINITY, 100),
        (3, 1e30, 100),
        (3, 0.001, 100),
    ] {
        assert!(Formula::feistel(k, a, n, 1).is_err());
    }
    assert!(FeistelTopology::new(3, u64::MAX / 3 + 1, 100, 0).is_err());
}

fn close(a: &[f64], b: &[f64]) {
    for (&x, &y) in a.iter().zip(b) {
        assert!((x - y).abs() < 3e-12, "{x} != {y}");
    }
    assert_eq!(a.len(), b.len());
}
#[test]
fn paired_solver_mutations_observables_and_solve() {
    for seed in 1..=20 {
        let f = Formula::feistel(3, 4.0, 100, seed).unwrap();
        let explicit = Formula::new(100, f.clauses()).unwrap();
        let opts = Options {
            seed,
            damping: 0.2,
            max_steps: 100,
            max_sweeps: 128,
            max_flips: 100_000,
            ..Default::default()
        };
        let mut a = Solver::new(f.clone(), opts.clone()).unwrap();
        let mut b = Solver::new(explicit.clone(), opts.clone()).unwrap();
        let before = a.messages().to_vec();
        for v in [0, 37, 82] {
            a.fix(v, true).unwrap();
            b.fix(v, true).unwrap();
        }
        assert_eq!(a.messages(), b.messages());
        for v in [82, 0, 37] {
            a.release(v).unwrap();
            b.release(v).unwrap();
        }
        assert_eq!(a.messages(), before);
        for _ in 0..5 {
            assert!((a.sweep().unwrap() - b.sweep().unwrap()).abs() < 3e-12);
            close(a.messages(), b.messages());
        }
        let ar = a.converge();
        let br = b.converge();
        match (ar, br) {
            (Ok(x), Ok(y)) => {
                assert!((x.complexity - y.complexity).abs() < 1e-9);
                close(a.messages(), b.messages());
            }
            (Err(x), Err(y)) => assert_eq!(x, y),
            other => panic!("convergence mismatch: {other:?}"),
        }
        for formula in [f.clone(), explicit.clone()] {
            let result = Solver::new(formula, opts.clone()).unwrap().solve();
            if result.status == Status::Sat {
                let assignment = result.assignment.unwrap();
                assert!(f.is_satisfied(&assignment) && explicit.is_satisfied(&assignment));
            }
        }
    }
}

#[test]
fn unit_propagation_matches_and_walksat_verifies() {
    for seed in 0..20 {
        for (k, a, n) in [(1, 1.0, 10), (2, 0.1, 20), (3, 0.5, 100)] {
            let f = Formula::feistel(k, a, n, seed).unwrap();
            let explicit = Formula::new(n, f.clauses()).unwrap();
            let mut x = Solver::new(f.clone(), Options::default()).unwrap();
            let mut y = Solver::new(explicit.clone(), Options::default()).unwrap();
            assert_eq!(x.propagate(), y.propagate());
            assert_eq!(x.assignment(), y.assignment());
            let r = x.solve();
            assert_eq!(r.status, Status::Sat);
            assert!(f.is_satisfied(r.assignment.as_ref().unwrap()));
            assert!(explicit.is_satisfied(r.assignment.as_ref().unwrap()));
        }
    }
}

#[test]
fn zero_and_subnormal_messages_match_explicit() {
    for zero in [false, true] {
        let f = Formula::feistel(2, 1024.0, 2, 7).unwrap();
        let explicit = Formula::new(2, f.clauses()).unwrap();
        let mut x = Solver::new(f.clone(), Options::default()).unwrap();
        let mut y = Solver::new(explicit, Options::default()).unwrap();
        let mut eta = vec![0.5; f.edges()];
        if zero {
            eta[0] = 1.0;
            eta[4] = 0.0;
        }
        x.set_messages(&eta).unwrap();
        y.set_messages(&eta).unwrap();
        match (x.sweep(), y.sweep()) {
            (Ok(a), Ok(b)) => {
                assert!((a - b).abs() < 3e-12);
                close(x.messages(), y.messages());
            }
            (Err(a), Err(b)) => assert_eq!(a, b),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn version_one_golden_vectors() {
    let t = FeistelTopology::new(3, 400, 100, 7).unwrap();
    assert_eq!(t.literal(0), (4, false));
    assert_eq!(t.literal(1), (62, false));
    assert_eq!(t.literal(2), (87, true));
    assert_eq!(t.literal(3), (7, true));
    assert_eq!(t.literal(4), (38, true));
    assert_eq!(t.literal(5), (90, false));
    assert_eq!(t.literal(51), (21, false));
    assert_eq!(t.literal(52), (55, true));
    assert_eq!(t.literal(53), (68, false));
    assert_eq!(t.literal(1197), (3, true));
    assert_eq!(t.literal(1198), (37, false));
    assert_eq!(t.literal(1199), (93, true));
    let t = FeistelTopology::new(1, u64::MAX, u64::MAX, 0).unwrap();
    for e in [0, 1, u64::MAX - 1] {
        let v = t.literal(e).0;
        assert_eq!(t.variable_edges(v).next(), Some(e));
    }
}
