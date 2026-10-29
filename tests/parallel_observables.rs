use bsp_rust::{Backend, ExecutionOptions, Formula, Options, Solver};

#[test]
fn masked_parallel_observables_match_scalar_before_and_after_release() {
    let formula = Formula::random(4, 9.5, 1000, 7).unwrap();
    // A loose tolerance evaluates a nontrivial observable after a short, paired
    // relaxation. This tests the observable, not a claim of SP convergence.
    let options = Options {
        epsilon: 0.99,
        ..Options::default()
    };
    let mut scalar = Solver::new(formula.clone(), options.clone()).unwrap();
    let mut parallel = Solver::with_execution(
        formula.clone(),
        options,
        ExecutionOptions {
            backend: Backend::CpuParallel,
            threads: 4,
            profile: None,
        },
    )
    .unwrap();
    for solver in [&mut scalar, &mut parallel] {
        solver.set_messages(&vec![0.1; formula.edges()]).unwrap();
        solver.fix(0, true).unwrap();
    }
    assert!(scalar.active_clauses() < formula.m());
    for release in [false, true] {
        if release {
            scalar.release(0).unwrap();
            parallel.release(0).unwrap();
        }
        let a = scalar.converge().unwrap();
        let b = parallel.converge().unwrap();
        assert!(!a.paramagnetic);
        assert_eq!(a.complexity.to_bits(), b.complexity.to_bits());
        assert_eq!(a.residual.to_bits(), b.residual.to_bits());
        assert_eq!(scalar.messages(), parallel.messages());
        assert_eq!(scalar.surveys(), parallel.surveys());
    }
}
