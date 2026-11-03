use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("bsp-cli-{label}-{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(dir: &PathBuf, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_bsp-rust"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap()
}

fn stdout(output: &std::process::Output) -> String {
    String::from_utf8(output.stdout.clone()).unwrap()
}

#[test]
fn version_matches_package_without_a_formula() {
    for flag in ["--version", "-V"] {
        let output = Command::new(env!("CARGO_BIN_EXE_bsp-rust"))
            .arg(flag)
            .output()
            .unwrap();
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("bsp-rust {}\n", env!("CARGO_PKG_VERSION"))
        );
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn cli_solves_the_shipped_dimacs_example() {
    let dir = temp_dir("solve");
    let output = run(
        &dir,
        &[
            "--no-progress",
            "--output",
            "result.json",
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/3sat.cnf"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(stdout(&output), "result.json\n");
    let result: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["status"], "sat");
    let assignment: Vec<i8> = serde_json::from_value(result["assignment"].clone()).unwrap();
    let formula = bsp_rust::Formula::from_dimacs(include_str!("../examples/3sat.cnf")).unwrap();
    assert!(formula.is_satisfied(&assignment));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn cli_default_output_name_contains_the_run_variables() {
    let dir = temp_dir("name");
    let output = run(
        &dir,
        &[
            "--no-progress",
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/3sat.cnf"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let name = stdout(&output);
    let name = name.trim();
    assert!(name.starts_with("output_3sat_seed1_r0.9_cpu_"));
    assert!(name.ends_with("Z.json"));
    let result: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join(name)).unwrap()).unwrap();
    assert_eq!(result["status"], "sat");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn cli_progress_line_names_the_three_phases() {
    let dir = temp_dir("progress");
    let output = run(
        &dir,
        &[
            "--progress",
            "--output",
            "result.json",
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/3sat.cnf"),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = stdout(&output);
    assert!(text.contains("formula"));
    assert!(text.contains("bsp"));
    assert!(text.contains("walksat"));
    assert!(text.contains("result.json"));
    let result: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["status"], "sat");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn cli_documents_options_and_rejects_invalid_input() {
    let output = Command::new(env!("CARGO_BIN_EXE_bsp-rust"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("--generate K ALPHA N"));
    assert!(help.contains("Default: 0.9."));
    assert!(help.contains("Default: 0.01."));
    assert!(help.contains("--output PATH"));
    assert!(help.contains("--progress"));
    assert!(help.contains("--no-progress"));
    let output = Command::new(env!("CARGO_BIN_EXE_bsp-rust"))
        .arg("--invalid")
        .arg("1")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}
