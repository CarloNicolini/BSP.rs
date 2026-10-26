#![cfg(all(feature = "cuda", target_os = "linux"))]
mod common;
use bsp_rust::{solve_many, Backend, ExecutionOptions, Formula, Options, SolveJob, Solver, Status};
fn execution(backend: Backend) -> ExecutionOptions {
    ExecutionOptions {
        backend,
        threads: 4,
        profile: None,
    }
}
fn compare(a: &[f64], b: &[f64], tolerance: f64) {
    assert_eq!(a.len(), b.len());
    for (e, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= tolerance, "edge {e}: {a} != {b}");
    }
}
#[test]
fn invalid_cuda_topology_is_rejected_without_device_access() {
    let bad = bsp_cuda::Topology {
        n: 1,
        offsets: &[0, 1],
        literals: &[0],
        edge_clause: &[0],
        var_offsets: &[0, 2, 1],
        var_edges: &[0],
    };
    assert!(bsp_cuda::CudaSolver::new(bad, &[0.2], false, false).is_err());
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn cuda_batch_stops_after_the_first_failed_sweep() {
    for fp32 in [false, true] {
        // A contradictory cavity on variable 1, plus an independent component
        // whose messages would keep changing if later sweeps were executed.
        let topology = || bsp_cuda::Topology {
            n: 3,
            offsets: &[0, 1, 2, 3, 5, 7, 9],
            literals: &[0, 0, 1, 2, 4, 3, 4, 2, 5],
            edge_clause: &[0, 1, 2, 3, 3, 4, 4, 5, 5],
            var_offsets: &[0, 2, 3, 5, 6, 8, 9],
            var_edges: &[0, 1, 2, 3, 7, 5, 4, 6, 8],
        };
        let values = [1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2];
        let mut one = bsp_cuda::CudaSolver::new(topology(), &values, fp32, false).unwrap();
        let mut many = bsp_cuda::CudaSolver::new(topology(), &values, fp32, false).unwrap();
        assert!(one.sweeps(1, -1.0, 0.2, false).unwrap().numerical_error);
        let report = many.sweeps(16, -1.0, 0.2, false).unwrap();
        assert!(report.numerical_error);
        assert_eq!(report.sweeps, 0);
        let (mut a, mut b, mut saved) = ([0.0; 9], [0.0; 9], [0.0; 9]);
        one.snapshot(&mut a, &mut saved).unwrap();
        many.snapshot(&mut b, &mut saved).unwrap();
        assert_eq!(a, b);
    }
}
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn cuda_sweeps_and_graph_mutations_match_cpu() {
    for backend in [Backend::Cuda, Backend::CudaF32] {
        for k in [3, 4, 6, 12, 65] {
            let f = Formula::random(k, 2.0, 80, 9).unwrap();
            let options = Options {
                damping: 0.2,
                ..Options::default()
            };
            let mut a = Solver::new(f.clone(), options.clone()).unwrap();
            let mut b = Solver::with_execution(f.clone(), options, execution(backend)).unwrap();
            let values: Vec<_> = (0..f.edges())
                .map(|e| 0.05 + 0.25 * ((e * 37) % 997) as f64 / 997.0)
                .collect();
            a.set_messages(&values).unwrap();
            b.set_messages(&values).unwrap();
            let tolerance = if backend == Backend::Cuda {
                3e-13
            } else {
                3e-6
            };
            for _ in 0..5 {
                let oracle = (backend == Backend::CudaF32)
                    .then(|| common::f32_oracle(&f, b.messages(), b.assignment(), 0.2));
                let ar = a.sweep().unwrap();
                let br = b.sweep().unwrap();
                assert!((ar - br).abs() < tolerance);
                compare(a.messages(), b.messages(), tolerance);
                if let Some(oracle) = oracle {
                    compare(&oracle, b.messages(), 3e-6);
                }
            }
            for (v, value) in [(0, true), (1, false), (4, true)] {
                a.fix(v, value).unwrap();
                b.fix(v, value).unwrap();
            }
            a.sweep_many(3).unwrap();
            b.sweep_many(3).unwrap();
            compare(a.messages(), b.messages(), tolerance);
            for v in [1, 0, 4] {
                a.release(v).unwrap();
                b.release(v).unwrap();
            }
            a.sweep().unwrap();
            b.sweep().unwrap();
            compare(a.messages(), b.messages(), tolerance);
            assert_eq!(b.execution_report().backend, backend);
            assert!(b.execution_report().device_error.is_none());
        }
    }
}
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn cuda_full_solves_and_batches_are_verified() {
    let f = Formula::random(3, 4.15, 400, 3).unwrap();
    let options = Options {
        seed: 3,
        max_flips: 100000,
        ..Options::default()
    };
    let mut cpu = Solver::new(f.clone(), options.clone()).unwrap();
    let reference = cpu.solve();
    let mut gpu = Solver::with_execution(f.clone(), options, execution(Backend::Cuda)).unwrap();
    let result = gpu.solve();
    assert_eq!(result.status, reference.status);
    assert_eq!(result.assignment, reference.assignment);
    assert_eq!(result.sp_sweeps, reference.sp_sweeps);
    assert!(f.is_satisfied(result.assignment.as_ref().unwrap()));
    assert!(result.execution.device_error.is_none());
    for backend in [Backend::Cuda, Backend::CudaF32] {
        let f = Formula::random(3, 2.5, 80, 3).unwrap();
        let jobs = (0..4)
            .map(|seed| SolveJob {
                formula: f.clone(),
                options: Options {
                    seed,
                    ..Options::default()
                },
            })
            .collect();
        let results = solve_many(jobs, execution(backend)).unwrap();
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.seed, i as u64);
            assert_eq!(r.status, Status::Sat);
            assert!(f.is_satisfied(r.assignment.as_ref().unwrap()));
        }
    }
}
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn cuda_limits_zero_factors_and_precision_recovery() {
    let (f, values) = common::excluded_underflow_case();
    let mut cpu = Solver::new(f.clone(), Options::default()).unwrap();
    let mut gpu =
        Solver::with_execution(f, Options::default(), execution(Backend::CudaF32)).unwrap();
    cpu.set_messages(&values).unwrap();
    gpu.set_messages(&values).unwrap();
    assert_eq!(cpu.sweep(), gpu.sweep());
    compare(cpu.messages(), gpu.messages(), 0.0);
    assert_eq!(gpu.execution_report().backend, Backend::Cpu);
    let mut clauses = Vec::new();
    for v in 2..=21 {
        clauses.push(vec![1, v]);
    }
    for v in 22..=40 {
        clauses.push(vec![-1, v]);
    }
    let subnormal = Formula::new(40, clauses).unwrap();
    let mut values = vec![0.0; subnormal.edges()];
    for e in (0..values.len()).step_by(2) {
        values[e] = 1.0 - f64::EPSILON;
    }
    let mut cpu = Solver::new(subnormal.clone(), Options::default()).unwrap();
    let mut gpu =
        Solver::with_execution(subnormal, Options::default(), execution(Backend::Cuda)).unwrap();
    cpu.set_messages(&values).unwrap();
    gpu.set_messages(&values).unwrap();
    assert_eq!(cpu.sweep(), gpu.sweep());
    compare(cpu.messages(), gpu.messages(), 0.0);
    assert_eq!(gpu.execution_report().backend, Backend::Cuda);
    for backend in [Backend::Cuda, Backend::CudaF32] {
        let f = Formula::new(3, vec![vec![1, 2], vec![-1, 3], vec![-2, -3]]).unwrap();
        let mut s =
            Solver::with_execution(f.clone(), Options::default(), execution(backend)).unwrap();
        s.set_messages(&vec![0.0; f.edges()]).unwrap();
        assert_eq!(s.sweep(), Ok(0.0));
        assert_eq!(s.converge().unwrap().sweeps, 1);
        s.set_messages(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        s.sweep().unwrap();
        assert!(s.messages().iter().all(|x| x.is_finite()));
        let mut empty = Solver::with_execution(
            Formula::new(0, vec![]).unwrap(),
            Options::default(),
            execution(backend),
        )
        .unwrap();
        assert_eq!(empty.solve().status, Status::Sat);
        let mut bad = Solver::with_execution(
            Formula::new(1, vec![vec![]]).unwrap(),
            Options::default(),
            execution(backend),
        )
        .unwrap();
        assert_eq!(bad.solve().status, Status::Contradiction);
        let f = Formula::random(3, 3.8, 80, 7).unwrap();
        let mut s = Solver::with_execution(
            f,
            Options {
                max_sweeps: 1,
                epsilon: 1e-10,
                ..Options::default()
            },
            execution(backend),
        )
        .unwrap();
        assert_eq!(s.converge().unwrap_err(), Status::NonConverged);
    }
    let clauses = (2..=600)
        .map(|v| vec![if v % 2 == 0 { 1 } else { -1 }, v])
        .collect();
    let f = Formula::new(600, clauses).unwrap();
    let mut s =
        Solver::with_execution(f.clone(), Options::default(), execution(Backend::CudaF32)).unwrap();
    s.set_messages(&vec![0.002; f.edges()]).unwrap();
    s.sweep().unwrap();
    assert_eq!(s.execution_report().backend, Backend::CudaF32);
    s.set_messages(&vec![0.8; f.edges()]).unwrap();
    let mut reference = Solver::new(f, Options::default()).unwrap();
    reference.set_messages(s.messages()).unwrap();
    assert_eq!(s.sweep(), reference.sweep());
    assert_eq!(s.execution_report().backend, Backend::Cpu);
    compare(s.messages(), reference.messages(), 0.0);
}
#[test]
#[ignore = "requires an NVIDIA GPU"]
fn cuda_cooperative_cancellation_and_timeout() {
    let f = Formula::random(3, 4.2, 10000, 7).unwrap();
    let mut s =
        Solver::with_execution(f.clone(), Options::default(), execution(Backend::Cuda)).unwrap();
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
        execution(Backend::CudaF32),
    )
    .unwrap();
    assert_eq!(s.sweep_many(100000), Err(Status::Timeout));
}
