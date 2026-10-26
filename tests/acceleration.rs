use bsp_rust::{solve_many, Backend, ExecutionOptions, Formula, Options, SolveJob, Solver, Status};

fn execution(backend: Backend) -> ExecutionOptions {
    ExecutionOptions {
        backend,
        threads: 4,
        profile: None,
    }
}
fn trajectory(s: &mut Solver) -> serde_json::Value {
    let mut value = serde_json::to_value(s.solve()).unwrap();
    value.as_object_mut().unwrap().remove("wall_seconds");
    value.as_object_mut().unwrap().remove("execution");
    value
}
#[test]
fn parallel_sweeps_and_mutations_match_every_f64_bit() {
    let mut clauses = Formula::random(4, 9.5, 1000, 7).unwrap().clauses();
    clauses.push((1..=150).collect());
    let f = Formula::new(1000, clauses).unwrap();
    let mut a = Solver::new(f.clone(), Options::default()).unwrap();
    let mut b =
        Solver::with_execution(f, Options::default(), execution(Backend::CpuParallel)).unwrap();
    for _ in 0..8 {
        assert_eq!(a.sweep(), b.sweep());
        assert_eq!(a.messages(), b.messages());
    }
    for (v, value) in [(0, true), (1, false), (78, true), (800, false)] {
        a.fix(v, value).unwrap();
        b.fix(v, value).unwrap();
    }
    assert_eq!(a.sweep_many(5), b.sweep_many(5));
    assert_eq!(a.messages(), b.messages());
    for v in [1, 800, 0, 78] {
        a.release(v).unwrap();
        b.release(v).unwrap();
    }
    assert_eq!(a.sweep(), b.sweep());
    assert_eq!(a.messages(), b.messages());
}

#[test]
fn parallel_zero_factors_and_subnormals_match_scalar() {
    let mut clauses = Vec::new();
    for v in 2..=21 {
        clauses.push(vec![1, v]);
    }
    for v in 22..=40 {
        clauses.push(vec![-1, v]);
    }
    clauses.extend(
        Formula::random(3, 4.2, 3000, 7)
            .unwrap()
            .clauses()
            .into_iter()
            .map(|c| {
                c.into_iter()
                    .map(|l| if l > 0 { l + 40 } else { l - 40 })
                    .collect()
            }),
    );
    let f = Formula::new(3040, clauses).unwrap();
    for subnormal in [true, false] {
        let mut values = vec![0.05; f.edges()];
        if subnormal {
            for e in (0..78).step_by(2) {
                values[e] = 1.0 - f64::EPSILON;
            }
        } else {
            values[0] = 1.0;
            values[2] = 1.0;
        }
        let mut a = Solver::new(f.clone(), Options::default()).unwrap();
        let mut b = Solver::with_execution(
            f.clone(),
            Options::default(),
            execution(Backend::CpuParallel),
        )
        .unwrap();
        a.set_messages(&values).unwrap();
        b.set_messages(&values).unwrap();
        let expected = a.sweep();
        assert!(expected.is_ok());
        assert_eq!(expected, b.sweep());
        assert_eq!(a.messages(), b.messages());
        assert_eq!(b.execution_report().parallel_sweeps, 1);
    }
}
#[test]
fn parallel_full_solver_trajectory_and_batch_order_match_reference() {
    let f = Formula::random(3, 4.15, 4000, 7).unwrap();
    let mut a = Solver::new(f.clone(), Options::default()).unwrap();
    let mut b =
        Solver::with_execution(f, Options::default(), execution(Backend::CpuParallel)).unwrap();
    let left = a.converge();
    let right = b.converge();
    assert!(left.is_ok());
    assert_eq!(format!("{left:?}"), format!("{right:?}"));
    assert_eq!(a.surveys(), b.surveys());
    assert!(b.execution_report().parallel_sweeps > 0);
    let f = Formula::random(3, 4.15, 400, 3).unwrap();
    let options = Options {
        seed: 3,
        max_flips: 100000,
        ..Options::default()
    };
    let mut a = Solver::new(f.clone(), options.clone()).unwrap();
    let mut b = Solver::with_execution(f, options, execution(Backend::CpuParallel)).unwrap();
    assert_eq!(trajectory(&mut a), trajectory(&mut b));
    let fs: Vec<_> = (0..8)
        .map(|seed| Formula::random(3, 2.5, 80, seed).unwrap())
        .collect();
    let jobs = fs
        .iter()
        .enumerate()
        .map(|(seed, f)| SolveJob {
            formula: f.clone(),
            options: Options {
                seed: seed as u64,
                ..Options::default()
            },
        })
        .collect();
    let results = solve_many(jobs, execution(Backend::CpuParallel)).unwrap();
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r.seed, i as u64);
        assert_eq!(r.status, Status::Sat);
        assert!(fs[i].is_satisfied(r.assignment.as_ref().unwrap()));
    }
    assert!(solve_many(Vec::new(), ExecutionOptions::default())
        .unwrap()
        .is_empty());
}
#[test]
fn execution_validation_and_profile_output() {
    let f = Formula::random(3, 2.0, 32, 5).unwrap();
    assert!(Solver::with_execution(
        f.clone(),
        Options::default(),
        ExecutionOptions {
            backend: Backend::CpuParallel,
            threads: 1025,
            profile: None
        }
    )
    .is_err());
    let path = std::env::temp_dir().join(format!("bsp-profile-{}.json", std::process::id()));
    let mut s = Solver::with_execution(
        f,
        Options::default(),
        ExecutionOptions {
            profile: Some(path.clone()),
            ..ExecutionOptions::default()
        },
    )
    .unwrap();
    s.solve();
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(
        report["execution"]["phases"]["cavities"]["calls"]
            .as_u64()
            .unwrap()
            > 0
    );
    std::fs::remove_file(path).unwrap();
}
#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[test]
fn unavailable_metal_is_an_explicit_error() {
    assert!(Solver::with_execution(
        Formula::new(0, vec![]).unwrap(),
        Options::default(),
        execution(Backend::Metal)
    )
    .is_err());
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    #[test]
    fn invalid_gpu_topologies_are_rejected_before_device_access() {
        for (offsets, var_offsets, var_edges) in [
            (vec![1, 1], vec![0, 1, 1], vec![0]),
            (vec![0, 1], vec![0, 2, 1], vec![0]),
            (vec![0, 1], vec![0, 1, 1], vec![1]),
            (vec![0, 1], vec![0, 0, 1], vec![0]),
        ] {
            let result = bsp_metal::MetalSolver::new(
                bsp_metal::Topology {
                    n: 1,
                    offsets: &offsets,
                    literals: &[0],
                    edge_clause: &[0],
                    var_offsets: &var_offsets,
                    var_edges: &var_edges,
                },
                &[0.2],
                false,
            );
            assert!(result.is_err());
        }
    }
    #[test]
    fn unsafe_f32_options_record_a_cpu_fallback() {
        let f = Formula::random(3, 2.0, 32, 7).unwrap();
        for options in [
            Options {
                damping: 1.0 - f64::EPSILON,
                ..Options::default()
            },
            Options {
                epsilon: 1e-320,
                ..Options::default()
            },
        ] {
            let reference = Solver::new(f.clone(), options.clone()).unwrap();
            let s = Solver::with_execution(f.clone(), options, execution(Backend::Metal)).unwrap();
            assert_eq!(s.execution_report().backend, Backend::Cpu);
            assert_eq!(s.execution_report().fallbacks.len(), 1);
            assert_eq!(s.messages(), reference.messages());
        }
    }
    // Independent, deliberately quadratic f32 definition; no prefix/suffix code.
    fn oracle(f: &Formula, eta: &[f64], fixed: &[i8], damping: f32) -> Vec<f64> {
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
    fn close(a: &[f64], b: &[f64]) {
        for (e, (a, b)) in a.iter().zip(b).enumerate() {
            assert!((a - b).abs() < 3e-6, "edge {e}: {a} != {b}");
        }
    }
    fn direct_complexity(f: &Formula, eta: &[f64]) -> f64 {
        let clauses = f.clauses();
        let mut occurrences = vec![Vec::new(); f.n()];
        let mut edge = 0;
        for (c, clause) in clauses.iter().enumerate() {
            for &lit in clause {
                occurrences[lit.unsigned_abs() as usize - 1].push((c, edge, lit > 0));
                edge += 1;
            }
        }
        let mut clause_term = 0.0;
        for (c, clause) in clauses.iter().enumerate() {
            let mut z_product = 1.0;
            let mut u_product = 1.0;
            for &lit in clause {
                let mut same = 1.0;
                let mut opposite = 1.0;
                for &(other, e, positive) in &occurrences[lit.unsigned_abs() as usize - 1] {
                    if other == c {
                        continue;
                    }
                    if positive == (lit > 0) {
                        same *= 1.0 - eta[e];
                    } else {
                        opposite *= 1.0 - eta[e];
                    }
                }
                z_product *= opposite + (1.0 - opposite) * same;
                u_product *= (1.0 - opposite) * same;
            }
            clause_term += (z_product - u_product).ln();
        }
        let mut variable_term = 0.0;
        for edges in occurrences {
            let mut plus = 1.0;
            let mut minus = 1.0;
            for &(_, e, positive) in &edges {
                if positive {
                    plus *= 1.0 - eta[e];
                } else {
                    minus *= 1.0 - eta[e];
                }
            }
            variable_term += (edges.len() as f64 - 1.0) * (plus + (1.0 - plus) * minus).ln();
        }
        clause_term - variable_term
    }
    #[test]
    #[ignore = "requires an available Apple GPU; run explicitly with --ignored"]
    fn metal_sweeps_match_independent_f32_definition_and_restore_messages() {
        for k in [3, 4, 6, 12, 65] {
            let f = Formula::random(k, 2.0, 80, 9).unwrap();
            let options = Options {
                damping: 0.2,
                ..Options::default()
            };
            let mut s =
                Solver::with_execution(f.clone(), options, execution(Backend::Metal)).unwrap();
            let eta: Vec<_> = (0..f.edges())
                .map(|e| 0.05 + 0.25 * ((e * 37) % 997) as f64 / 997.0)
                .collect();
            s.set_messages(&eta).unwrap();
            for _ in 0..4 {
                let expected = oracle(&f, s.messages(), s.assignment(), 0.2);
                s.sweep().unwrap();
                close(s.messages(), &expected);
            }
            let before = s.messages().to_vec();
            for (v, value) in [(0, true), (1, false), (3, true)] {
                s.fix(v, value).unwrap();
            }
            let expected = oracle(&f, s.messages(), s.assignment(), 0.2);
            s.sweep().unwrap();
            close(s.messages(), &expected);
            for v in [1, 0, 3] {
                s.release(v).unwrap();
            }
            let expected = oracle(&f, s.messages(), s.assignment(), 0.2);
            s.sweep().unwrap();
            close(s.messages(), &expected);
            assert_eq!(before.len(), s.messages().len());
            assert!(s.execution_report().fallbacks.is_empty());
        }
        let f = Formula::random(3, 4.0, 1000, 5).unwrap();
        let mut a =
            Solver::with_execution(f.clone(), Options::default(), execution(Backend::Metal))
                .unwrap();
        let mut b =
            Solver::with_execution(f, Options::default(), execution(Backend::Metal)).unwrap();
        for _ in 0..9 {
            a.sweep().unwrap();
        }
        b.sweep_many(9).unwrap();
        assert_eq!(a.messages(), b.messages());
    }
    #[test]
    #[ignore = "requires an available Apple GPU; run explicitly with --ignored"]
    fn metal_handles_limits_numerical_fallback_and_verified_solutions() {
        let mut clauses = Vec::new();
        for v in 2..=600 {
            clauses.push(vec![if v % 2 == 0 { 1 } else { -1 }, v]);
        }
        let f = Formula::new(600, clauses).unwrap();
        let mut high =
            Solver::with_execution(f.clone(), Options::default(), execution(Backend::Metal))
                .unwrap();
        high.set_messages(&vec![0.002; f.edges()]).unwrap();
        let expected = oracle(&f, high.messages(), high.assignment(), 0.0);
        high.sweep().unwrap();
        close(high.messages(), &expected);
        assert_eq!(high.execution_report().backend, Backend::Metal);
        high.set_messages(&vec![0.8; f.edges()]).unwrap();
        let mut reference = Solver::new(f, Options::default()).unwrap();
        reference.set_messages(high.messages()).unwrap();
        assert_eq!(high.sweep(), reference.sweep());
        assert_eq!(high.messages(), reference.messages());
        assert_eq!(high.execution_report().backend, Backend::Cpu);
        assert!(!high.execution_report().fallbacks.is_empty());
        let f = Formula::random(3, 3.8, 80, 9).unwrap();
        let mut s = Solver::with_execution(
            f.clone(),
            Options {
                max_sweeps: 1,
                epsilon: 1e-10,
                ..Options::default()
            },
            execution(Backend::Metal),
        )
        .unwrap();
        assert_eq!(s.converge().unwrap_err(), Status::NonConverged);
        s.set_messages(&vec![1.0 - f64::EPSILON; f.edges()])
            .unwrap();
        assert_eq!(s.execution_report().backend, Backend::Cpu);
        assert_eq!(s.messages()[0], 1.0 - f64::EPSILON);
        let f = Formula::new(3, vec![vec![1, 2], vec![-1, 3], vec![-2, -3]]).unwrap();
        let mut s =
            Solver::with_execution(f.clone(), Options::default(), execution(Backend::Metal))
                .unwrap();
        s.set_messages(&vec![0.0; f.edges()]).unwrap();
        assert_eq!(s.sweep().unwrap(), 0.0);
        let r = s.solve();
        assert_eq!(r.status, Status::Sat);
        assert!(f.is_satisfied(r.assignment.as_ref().unwrap()));
        let f = Formula::new(1, vec![vec![]]).unwrap();
        let mut s =
            Solver::with_execution(f, Options::default(), execution(Backend::Metal)).unwrap();
        assert_eq!(s.solve().status, Status::Contradiction);
        let f = Formula::random(3, 4.2, 10000, 3).unwrap();
        let mut s =
            Solver::with_execution(f, Options::default(), execution(Backend::Metal)).unwrap();
        s.cancellation_token().cancel();
        assert_eq!(s.sweep(), Err(Status::Cancelled));
    }
    #[test]
    #[ignore = "requires an available Apple GPU; run explicitly with --ignored"]
    fn metal_stops_at_first_fixed_point_and_during_long_batches() {
        let f = Formula::random(3, 4.15, 400, 3).unwrap();
        let mut observed = Solver::with_execution(
            f.clone(),
            Options {
                seed: 3,
                ..Options::default()
            },
            execution(Backend::Metal),
        )
        .unwrap();
        let report = observed.converge().unwrap();
        assert!(!report.paramagnetic);
        assert!((report.complexity - direct_complexity(&f, observed.messages())).abs() < 0.002);
        let f = Formula::random(3, 3.8, 80, 11).unwrap();
        let options = Options {
            damping: 0.2,
            ..Options::default()
        };
        let mut a =
            Solver::with_execution(f.clone(), options.clone(), execution(Backend::Metal)).unwrap();
        let mut b = Solver::with_execution(f, options, execution(Backend::Metal)).unwrap();
        let mut count = 0;
        loop {
            count += 1;
            if a.sweep().unwrap() <= 0.01 {
                break;
            }
            assert!(count < 1024);
        }
        let report = b.converge().unwrap();
        assert_eq!(report.sweeps, count);
        assert_eq!(a.messages(), b.messages());
        let f = Formula::random(3, 4.2, 10000, 7).unwrap();
        let mut s =
            Solver::with_execution(f.clone(), Options::default(), execution(Backend::Metal))
                .unwrap();
        let token = s.cancellation_token();
        let worker = std::thread::spawn(move || s.sweep_many(100000));
        std::thread::sleep(std::time::Duration::from_millis(10));
        token.cancel();
        assert_eq!(worker.join().unwrap(), Err(Status::Cancelled));
        let mut s = Solver::with_execution(
            f,
            Options {
                timeout_seconds: Some(0.01),
                ..Options::default()
            },
            execution(Backend::Metal),
        )
        .unwrap();
        assert_eq!(s.sweep_many(100000), Err(Status::Timeout));
    }
}
