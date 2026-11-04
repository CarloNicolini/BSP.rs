//! CUDA-event sweep/observable timings with setup and snapshots outside timers.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn main() {
    use bsp_cuda::{CudaSolver, Topology};
    use bsp_rust::Formula;
    let args: Vec<_> = std::env::args().collect();
    let n: usize = args.get(1).expect("N").parse().unwrap();
    let seed: u64 = args.get(2).expect("seed").parse().unwrap();
    let trials: usize = args.get(3).map_or(9, |s| s.parse().unwrap());
    let count: usize = args.get(4).map_or(100, |s| s.parse().unwrap());
    assert!(count > 0, "batch count must be positive");
    let observables = args.get(5).is_some_and(|mode| mode == "obs");
    let formula = Formula::random(3, 4.2, n, seed).unwrap();
    let clauses = formula.clauses();
    let mut offsets = vec![0u32];
    let mut literals = Vec::with_capacity(formula.edges());
    let mut edge_clause = Vec::with_capacity(formula.edges());
    for (c, clause) in clauses.iter().enumerate() {
        for &literal in clause {
            literals.push(2 * (literal.unsigned_abs() - 1) + u32::from(literal < 0));
            edge_clause.push(c as u32);
        }
        offsets.push(literals.len() as u32);
    }
    let mut var_offsets = vec![0u32; 2 * n + 1];
    for &literal in &literals {
        var_offsets[literal as usize + 1] += 1;
    }
    for i in 1..var_offsets.len() {
        var_offsets[i] += var_offsets[i - 1];
    }
    let mut next = var_offsets.clone();
    let mut var_edges = vec![0u32; literals.len()];
    for (e, &literal) in literals.iter().enumerate() {
        var_edges[next[literal as usize] as usize] = e as u32;
        next[literal as usize] += 1;
    }
    let eta: Vec<f64> = (0..literals.len())
        .map(|e| 0.05 + 0.1 * ((e * 37) % 997) as f64 / 997.0)
        .collect();
    let mut device = CudaSolver::new(
        Topology {
            n,
            offsets: &offsets,
            literals: &literals,
            edge_clause: &edge_clause,
            var_offsets: &var_offsets,
            var_edges: &var_edges,
        },
        &eta,
        true,
        false,
    )
    .unwrap();
    for trial in 0..trials + 2 {
        device.upload(&eta, &eta).unwrap();
        let (report, gpu_seconds, wall_seconds) = if observables {
            let setup = device.sweeps(2, -1.0, 0.0, false).unwrap();
            assert!(!setup.numerical_error && !setup.contradiction);
            let started = std::time::Instant::now();
            let mut gpu_seconds = 0.0;
            let mut last = None;
            for _ in 0..count {
                let report = device.observables().unwrap();
                assert!(!report.numerical_error && !report.contradiction);
                gpu_seconds += report.gpu_seconds;
                last = Some(report);
            }
            (last.unwrap(), gpu_seconds, started.elapsed().as_secs_f64())
        } else {
            let report = device.sweeps(count, -1.0, 0.0, false).unwrap();
            assert_eq!(report.sweeps, count);
            let gpu_seconds = report.gpu_seconds;
            let wall_seconds = report.wall_seconds;
            (report, gpu_seconds, wall_seconds)
        };
        assert!(!report.numerical_error && !report.contradiction);
        if trial >= 2 {
            println!(
                "{}",
                serde_json::json!({"n":n,"seed":seed,"trial":trial-2,"sweeps":if observables {0} else {count},"passes":if observables {count} else {0},"gpu_seconds":gpu_seconds,"wall_seconds":wall_seconds,"residual":report.residual,"device_bytes":device.allocated_bytes})
            );
        }
    }
    // State identity includes every message, observable, and survey, outside timing.
    let obs = device.observables().unwrap();
    assert!(!obs.numerical_error && !obs.contradiction);
    let mut final_eta = vec![0.0; eta.len()];
    let mut saved = vec![0.0; eta.len()];
    device.snapshot(&mut final_eta, &mut saved).unwrap();
    let mut surveys = vec![[0.0; 3]; n];
    let mut cert = vec![0.0; n];
    device.surveys(&mut surveys, &mut cert).unwrap();
    let mut hash = 14695981039346656037u64;
    for value in final_eta
        .iter()
        .chain(&saved)
        .chain(surveys.iter().flatten())
        .chain(&cert)
    {
        hash = (hash ^ value.to_bits()).wrapping_mul(1099511628211);
    }
    println!(
        "{}",
        serde_json::json!({"n":n,"seed":seed,"hash":format!("{hash:016x}"),"complexity":obs.complexity,"paramagnetic":obs.paramagnetic})
    );
}
#[cfg(not(all(feature = "cuda", target_os = "linux")))]
fn main() {
    eprintln!("requires Linux and --features cuda");
    std::process::exit(2);
}
