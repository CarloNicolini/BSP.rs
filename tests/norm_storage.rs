use bsp_rust::{Backend, ExecutionOptions, Formula, Options, Solver};

fn check_frozen_messages(backend: Backend) {
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

#[test]
fn observable_scratch_does_not_overwrite_frozen_messages() {
    for backend in [Backend::Cpu, Backend::CpuParallel] {
        check_frozen_messages(backend);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires an available Apple GPU; run explicitly with --ignored"]
fn metal_observables_preserve_frozen_messages() {
    check_frozen_messages(Backend::Metal);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires an available Apple GPU; run explicitly with --ignored"]
fn generic_metal_prefix_storage_is_retained_only_when_needed() {
    let offsets = [0, 3, 6];
    let general_offsets = [0, 1, 6];
    let literals = [0, 2, 4, 6, 8, 10];
    let edge_clause = [0, 0, 0, 1, 1, 1];
    let general_edge_clause = [0, 1, 1, 1, 1, 1];
    let var_offsets = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6];
    let var_edges = [0, 1, 2, 3, 4, 5];
    let allocate = |offsets: &[u32], edge_clause: &[u32]| {
        bsp_metal::MetalSolver::new(
            bsp_metal::Topology {
                n: 6,
                offsets,
                literals: &literals,
                edge_clause,
                var_offsets: &var_offsets,
                var_edges: &var_edges,
            },
            &[0.2; 6],
            false,
        )
        .unwrap()
    };
    let specialized = allocate(&offsets, &edge_clause);
    let generic = allocate(&general_offsets, &general_edge_clause);
    assert_eq!(
        generic.allocated_bytes - specialized.allocated_bytes,
        6 * 4 - 4
    );
}
