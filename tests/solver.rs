use bsp_rust::{Formula, Options, Solver, Status};

// Deliberately slow independent definition: explicitly remove the target clause
// from every variable cavity, then remove the target literal from its clause.
fn scalar_step(f: &Formula, eta: &[f64], fixed: &[i8]) -> Vec<f64> {
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
    let mut result = vec![0.0; eta.len()];
    for (c, cl) in clauses.iter().enumerate() {
        if !active[c] {
            continue;
        }
        for (i, &target) in cl.iter().enumerate() {
            if fixed[target.unsigned_abs() as usize - 1] >= 0 {
                continue;
            }
            let mut product = 1.0;
            for (j, &neighbor) in cl.iter().enumerate() {
                if i == j || fixed[neighbor.unsigned_abs() as usize - 1] >= 0 {
                    continue;
                }
                let mut same = 1.0;
                let mut opposite = 1.0;
                for (b, other) in clauses.iter().enumerate() {
                    if b == c || !active[b] {
                        continue;
                    }
                    for (p, &lit) in other.iter().enumerate() {
                        if lit == neighbor {
                            same *= 1.0 - eta[offsets[b] + p];
                        }
                        if lit == -neighbor {
                            opposite *= 1.0 - eta[offsets[b] + p];
                        }
                    }
                }
                product *= (1.0 - opposite) * same / (same + opposite - same * opposite);
            }
            result[offsets[c] + i] = if product < 1e-16 { 0.0 } else { product };
        }
    }
    result
}

fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < 3e-13, "edge {i}: {x} != {y}");
    }
}

#[test]
fn sweeps_match_scalar_definition() {
    for seed in 0..12 {
        let f = Formula::random(3, 3.8, 18, seed).unwrap();
        let mut solver = Solver::new(
            f.clone(),
            Options {
                seed,
                ..Options::default()
            },
        )
        .unwrap();
        for _ in 0..5 {
            let expected = scalar_step(&f, solver.messages(), solver.assignment());
            solver.sweep().unwrap();
            close(solver.messages(), &expected);
        }
    }
}

#[test]
fn residual_sweep_matches_scalar_after_fix_and_release() {
    let f = Formula::random(4, 4.0, 24, 23).unwrap();
    let mut s = Solver::new(f.clone(), Options::default()).unwrap();
    for (v, value) in [(0, true), (1, false), (4, true), (9, false)] {
        s.fix(v, value).unwrap();
    }
    let expected = scalar_step(&f, s.messages(), s.assignment());
    s.sweep().unwrap();
    close(s.messages(), &expected);
    for v in [1, 9, 0, 4] {
        s.release(v).unwrap();
    }
    let expected = scalar_step(&f, s.messages(), s.assignment());
    s.sweep().unwrap();
    close(s.messages(), &expected);
}

#[test]
fn fix_release_restores_masks_and_frozen_messages_in_any_order() {
    for seed in 0..16 {
        let f = Formula::random(5, 3.0, 130, seed).unwrap();
        let mut s = Solver::new(f.clone(), Options::default()).unwrap();
        let before = s.messages().to_vec();
        for v in [0, 63, 64, 65, 129, 7, 44] {
            s.fix(v, v % 2 == 0).unwrap();
        }
        for v in [64, 44, 129, 0, 65, 7, 63] {
            s.release(v).unwrap();
        }
        assert_eq!(s.free_variables(), 130);
        assert_eq!(s.active_clauses(), f.m());
        assert_eq!(s.residual_formula().to_dimacs(), f.to_dimacs());
        assert_eq!(s.messages(), before);
    }
}

#[test]
fn long_clauses_and_word_boundaries() {
    for k in [1, 2, 3, 63, 64, 65, 129, 130] {
        let f = Formula::new(130, vec![(1..=k).collect()]).unwrap();
        let mut s = Solver::new(f.clone(), Options::default()).unwrap();
        for v in 0..k as usize - 1 {
            s.fix(v, false).unwrap();
        }
        assert_eq!(s.propagate(), Ok(1));
        assert_eq!(s.assignment()[k as usize - 1], 1);
        let result = s.solve();
        assert_eq!(result.status, Status::Sat);
        assert!(f.is_satisfied(result.assignment.as_ref().unwrap()));
    }
}

#[test]
fn dimacs_normalization_and_validation() {
    let f = Formula::from_dimacs("c comment\np cnf 4 3\n1 1\n-2 0 3 -3 0\n0\n").unwrap();
    assert_eq!(f.clauses(), vec![vec![1, -2], vec![]]);
    assert_eq!(
        Formula::from_dimacs(&f.to_dimacs()).unwrap().clauses(),
        f.clauses()
    );
    for text in [
        "p cnf 1 1\n2 0",
        "p cnf 1 1\n1",
        "1 0",
        "p cnf 1 0\n0",
        "p cnf 1 0\np cnf 1 0",
    ] {
        assert!(Formula::from_dimacs(text).is_err(), "{text}");
    }
    assert!(Formula::new(1, vec![vec![i32::MIN]]).is_err());
    assert!(Formula::new(1, vec![vec![0]]).is_err());
}

#[test]
fn unit_propagation_and_contradictions_never_claim_sat() {
    for cl in [
        vec![vec![]],
        vec![vec![1], vec![-1]],
        vec![vec![1], vec![-1, 2], vec![-2]],
    ] {
        let mut s = Solver::new(Formula::new(2, cl).unwrap(), Options::default()).unwrap();
        assert_eq!(s.solve().status, Status::Contradiction);
    }
    let mut s = Solver::new(
        Formula::new(3, vec![vec![1], vec![-1, 2], vec![-2, 3]]).unwrap(),
        Options::default(),
    )
    .unwrap();
    assert_eq!(s.propagate(), Ok(3));
    assert_eq!(s.assignment(), [1, 1, 1]);
    assert!(s.release(0).is_err());
    assert_eq!(s.solve().status, Status::Sat);
}

#[test]
fn empty_formulas_and_unused_variables() {
    for n in [0, 1, 130] {
        let f = Formula::new(n, vec![]).unwrap();
        let result = Solver::new(f.clone(), Options::default()).unwrap().solve();
        assert_eq!(result.status, Status::Sat);
        assert!(f.is_satisfied(result.assignment.as_ref().unwrap()));
    }
}

#[test]
fn solves_easy_random_instances_and_is_reproducible() {
    for seed in 0..30 {
        let f = Formula::random(3, 2.5, 40, seed).unwrap();
        let opts = Options {
            seed,
            max_flips: 100_000,
            ..Options::default()
        };
        let a = Solver::new(f.clone(), opts.clone()).unwrap().solve();
        let b = Solver::new(f.clone(), opts).unwrap().solve();
        assert_eq!(a.status, Status::Sat, "seed {seed}: {a:?}");
        assert!(f.is_satisfied(a.assignment.as_ref().unwrap()));
        assert_eq!(a.assignment, b.assignment);
        assert_eq!(a.flips, b.flips);
        assert_eq!(a.sp_sweeps, b.sp_sweeps);
    }
}

#[test]
fn concurrent_solvers_have_independent_state() {
    let f = Formula::random(3, 2.7, 80, 7).unwrap();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let f = f.clone();
            std::thread::spawn(move || Solver::new(f, Options::default()).unwrap().solve())
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|t| t.join().unwrap()).collect();
    for r in &results {
        assert_eq!(r.status, Status::Sat);
        assert_eq!(r.assignment, results[0].assignment);
    }
}

#[test]
fn explicit_failures_and_option_validation() {
    let f = Formula::random(3, 4.0, 80, 1).unwrap();
    let mut s = Solver::new(f.clone(), Options::default()).unwrap();
    s.cancellation_token().cancel();
    assert_eq!(s.solve().status, Status::Cancelled);
    let mut s = Solver::new(
        f.clone(),
        Options {
            max_sweeps: 1,
            epsilon: 1e-12,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(s.solve().status, Status::NonConverged);
    let mut s = Solver::new(
        f.clone(),
        Options {
            timeout_seconds: Some(1e-20),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(s.solve().status, Status::Timeout);
    for options in [
        Options {
            damping: f64::NAN,
            ..Options::default()
        },
        Options {
            backtrack_ratio: 1.0,
            ..Options::default()
        },
        Options {
            batch_fraction: 0.0,
            ..Options::default()
        },
        Options {
            max_sweeps: 0,
            ..Options::default()
        },
    ] {
        assert!(Solver::new(f.clone(), options).is_err());
    }
    let mut s = Solver::new(f, Options::default()).unwrap();
    assert!(s.set_messages(&[f64::NAN]).is_err());
    assert!(s.fix(80, true).is_err());
    assert!(s.release(0).is_err());
}

#[test]
fn zero_and_unit_messages_use_exclusion_products() {
    let f = Formula::new(3, vec![vec![1, 2], vec![-1, 3], vec![-2, -3]]).unwrap();
    let mut s = Solver::new(f.clone(), Options::default()).unwrap();
    let mut eta = vec![0.0; f.edges()];
    eta[0] = 1.0;
    s.set_messages(&eta).unwrap();
    let expected = scalar_step(&f, &eta, s.assignment());
    s.sweep().unwrap();
    close(s.messages(), &expected);
    s.set_messages(&vec![0.0; f.edges()]).unwrap();
    assert_eq!(s.sweep(), Ok(0.0));
}

#[test]
fn damping_does_not_hide_large_fixed_point_residuals() {
    let f = Formula::random(3, 3.8, 32, 8).unwrap();
    let mut a = Solver::new(f.clone(), Options::default()).unwrap();
    let mut b = Solver::new(
        f,
        Options {
            damping: 0.999999,
            max_sweeps: 1,
            ..Options::default()
        },
    )
    .unwrap();
    let raw = a.sweep().unwrap();
    let damped = b.sweep().unwrap();
    assert_eq!(raw, damped);
    assert!(damped > 0.01);
}

#[test]
fn subnormal_products_preserve_the_excluded_group() {
    let mut clauses = Vec::new();
    for v in 2..=21 {
        clauses.push(vec![1, v]);
    }
    for v in 22..=40 {
        clauses.push(vec![-1, v]);
    }
    let f = Formula::new(40, clauses).unwrap();
    let mut s = Solver::new(f.clone(), Options::default()).unwrap();
    let mut eta = vec![0.0; f.edges()];
    for e in (0..eta.len()).step_by(2) {
        eta[e] = 1.0 - f64::EPSILON;
    }
    s.set_messages(&eta).unwrap();
    let expected = scalar_step(&f, &eta, s.assignment());
    s.sweep().unwrap();
    close(s.messages(), &expected);
}

#[test]
fn walksat_respects_the_total_budget_across_restarts() {
    let f = Formula::new(2, vec![vec![1, 2], vec![1, -2], vec![-1, 2], vec![-1, -2]]).unwrap();
    for budget in [0, 1, 3, 4, 5, 100, 103] {
        let mut s = Solver::new(
            f.clone(),
            Options {
                max_flips: budget,
                walksat_restarts: 4,
                ..Options::default()
            },
        )
        .unwrap();
        s.set_messages(&vec![0.0; f.edges()]).unwrap();
        let result = s.solve();
        assert_eq!(result.status, Status::FlipLimit);
        assert_eq!(result.flips, budget);
        assert!(result.assignment.is_none());
    }
}

#[test]
fn bsp_decimates_and_releases_before_a_verified_solution() {
    let f = Formula::random(3, 4.15, 400, 3).unwrap();
    let options = Options {
        seed: 3,
        max_flips: 100_000,
        ..Options::default()
    };
    let result = Solver::new(f.clone(), options.clone()).unwrap().solve();
    assert_eq!(result.status, Status::Sat);
    assert!(result.decimations > 0 && result.releases > 0);
    assert!(f.is_satisfied(result.assignment.as_ref().unwrap()));
    let result = Solver::new(
        f,
        Options {
            max_steps: 1,
            ..options
        },
    )
    .unwrap()
    .solve();
    assert_eq!(result.status, Status::StepLimit);
}

#[test]
fn mixed_width_sweeps_and_complexity_match_direct_products() {
    let mut clauses = Formula::random(3, 2.0, 80, 3).unwrap().clauses();
    clauses.extend(Formula::random(4, 2.0, 80, 5).unwrap().clauses());
    let f = Formula::new(80, clauses).unwrap();
    let mut s = Solver::new(f.clone(), Options::default()).unwrap();
    let expected = scalar_step(&f, s.messages(), s.assignment());
    s.sweep().unwrap();
    close(s.messages(), &expected);

    let f = Formula::random(3, 4.15, 400, 3).unwrap();
    let mut s = Solver::new(
        f.clone(),
        Options {
            seed: 3,
            ..Options::default()
        },
    )
    .unwrap();
    let report = s.converge().unwrap();
    assert!(!report.paramagnetic);
    let clauses = f.clauses();
    let mut occurrences = vec![Vec::new(); f.n()];
    let mut e = 0;
    for (c, cl) in clauses.iter().enumerate() {
        for &lit in cl {
            occurrences[lit.unsigned_abs() as usize - 1].push((c, e, lit > 0));
            e += 1;
        }
    }
    let mut clause_term = 0.0;
    for (c, cl) in clauses.iter().enumerate() {
        let mut prod_z = 1.0;
        let mut prod_u = 1.0;
        for &lit in cl {
            let mut same = 1.0;
            let mut opposite = 1.0;
            for &(b, e, positive) in &occurrences[lit.unsigned_abs() as usize - 1] {
                if b == c {
                    continue;
                }
                if positive == (lit > 0) {
                    same *= 1.0 - s.messages()[e];
                } else {
                    opposite *= 1.0 - s.messages()[e];
                }
            }
            prod_z *= same + opposite - same * opposite;
            prod_u *= (1.0 - opposite) * same;
        }
        clause_term += (prod_z - prod_u).ln();
    }
    let mut variable_term = 0.0;
    for edges in occurrences {
        let mut plus = 1.0;
        let mut minus = 1.0;
        for &(_, e, positive) in &edges {
            if positive {
                plus *= 1.0 - s.messages()[e];
            } else {
                minus *= 1.0 - s.messages()[e];
            }
        }
        variable_term += (edges.len() as f64 - 1.0) * (plus + minus - plus * minus).ln();
    }
    assert!((report.complexity - (clause_term - variable_term)).abs() < 1e-9);
}
