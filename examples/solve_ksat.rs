use bsp_rust::{Formula, Options, Solver, Status};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let formula = match std::env::args().nth(1) {
        Some(path) => Formula::from_dimacs(&std::fs::read_to_string(path)?)?,
        None => Formula::random(3, 2.5, 128, 3)?,
    };
    let mut solver = Solver::new(formula.clone(), Options::default())?;
    let result = solver.solve();
    if result.status != Status::Sat {
        return Err(format!("solver stopped: {}", result.status).into());
    }
    let assignment = result.assignment.as_ref().unwrap();
    assert!(formula.is_satisfied(assignment));
    println!(
        "SAT: {} variables, {} clauses, {} SP sweeps",
        formula.n(),
        formula.m(),
        result.sp_sweeps
    );
    println!("assignment (0=false, 1=true): {assignment:?}");
    Ok(())
}
