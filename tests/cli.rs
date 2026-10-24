use std::process::Command;

#[test]
fn cli_solves_the_shipped_dimacs_example() {
    let output = Command::new(env!("CARGO_BIN_EXE_bsp-rust"))
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/3sat.cnf"))
        .output()
        .unwrap();
    assert!(output.status.success());
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["status"], "sat");
    let assignment: Vec<i8> = serde_json::from_value(result["assignment"].clone()).unwrap();
    let formula = bsp_rust::Formula::from_dimacs(include_str!("../examples/3sat.cnf")).unwrap();
    assert!(formula.is_satisfied(&assignment));
}

#[test]
fn cli_documents_options_and_rejects_invalid_input() {
    let output = Command::new(env!("CARGO_BIN_EXE_bsp-rust"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8(output.stdout)
        .unwrap()
        .contains("--generate K ALPHA N"));
    let output = Command::new(env!("CARGO_BIN_EXE_bsp-rust"))
        .arg("--invalid")
        .arg("1")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}
