use super::tests::check_indexes;
use super::*;

/// Largest undamped change of one independent synchronous sweep. Only eta is restored,
/// so q holds a full recomputation afterwards.
fn sweep_residual(s: &mut Solver) -> f64 {
    let eta = s.eta.clone();
    s.scalar_cavities(false).unwrap();
    let residual = s.scalar_clauses().unwrap();
    s.eta = eta;
    residual
}

#[test]
fn residual_schedule_meets_the_sweep_criterion_through_fixes_and_releases() {
    for (seed, damping) in [(1, 0.0), (2, 0.0), (3, 0.5)] {
        let formula = Formula::random(3, 4.0, 2000, seed).unwrap();
        let options = Options {
            seed,
            damping,
            batch_fraction: 0.01,
            ..Options::default()
        };
        let mut s = Solver::new(formula, options).unwrap();
        s.enable_residual_schedule();
        let mut converged = 0;
        for step in 0..60 {
            if s.propagate_inner().is_err() {
                break;
            }
            let mut queue = s.residual.take().unwrap();
            let settled = s.settle(&mut queue, u64::MAX);
            s.residual = Some(queue);
            if settled.is_err() {
                break;
            }
            let q = s.q.clone();
            let residual = sweep_residual(&mut s);
            let f = s.formula.data();
            for c in s.active.ones() {
                for e in s.masks.free_edges(f, c) {
                    assert_eq!(
                        q[e].to_bits(),
                        s.q[e].to_bits(),
                        "seed {seed}, step {step}, edge {e}"
                    );
                }
            }
            assert!(
                residual <= s.options.epsilon,
                "seed {seed}, step {step}: {residual}"
            );
            let report = s.converge().unwrap();
            converged += 1;
            check_indexes(&s);
            if report.paramagnetic {
                break;
            }
            if step % 3 == 2 {
                s.backtrack();
            } else {
                s.decimate();
            }
        }
        assert!(converged >= 20, "seed {seed}: {converged} rounds");
    }
}

#[test]
fn residual_schedule_returns_verified_solutions() {
    for seed in [1, 2, 3] {
        let formula = Formula::random(3, 4.0, 2000, seed).unwrap();
        let options = Options {
            seed,
            ..Options::default()
        };
        let mut s = Solver::new(formula.clone(), options).unwrap();
        s.enable_residual_schedule();
        let result = s.solve();
        assert_eq!(result.status, Status::Sat);
        assert!(formula.is_satisfied(result.assignment.as_ref().unwrap()));
    }
}
