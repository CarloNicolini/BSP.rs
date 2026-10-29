use bsp_rust::{Backend, ExecutionOptions, Formula, Options, Solver};

#[test]
fn observable_scratch_does_not_overwrite_frozen_messages() {
    let mut backends = vec![Backend::Cpu, Backend::CpuParallel];
    if cfg!(all(feature = "metal", target_os = "macos")) {
        backends.push(Backend::Metal);
    }
    for backend in backends {
        let f = Formula::random(5, 3.0, 3000, 7).unwrap();
        let mut solver = Solver::with_execution(
            f.clone(),
            Options::default(),
            ExecutionOptions {
                backend,
                threads: 4,
                profile: None,
            },
        )
        .unwrap();
        solver.set_messages(&vec![0.13; f.edges()]).unwrap();
        let initial = solver.messages().to_vec();
        solver.fix(0, true).unwrap();
        let frozen: Vec<_> = solver.messages().iter().map(|&x| x == 0.0).collect();
        assert!(frozen.iter().any(|&v| v));
        solver.converge().unwrap();
        assert_eq!(solver.assignment()[0], 1);
        solver.release(0).unwrap();
        for (e, &was_frozen) in frozen.iter().enumerate() {
            if was_frozen {
                assert_eq!(solver.messages()[e], initial[e], "{backend:?}: edge {e}");
            }
        }
    }
}
