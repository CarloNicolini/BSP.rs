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

fn summary(output: &std::process::Output) -> serde_json::Value {
    let text = stdout(output);
    serde_json::from_str(text.lines().last().unwrap()).unwrap()
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
    let brief = summary(&output);
    assert_eq!(brief["status"], "sat");
    assert_eq!(brief["output"], "result.json");
    assert!(brief.get("assignment").is_none());
    assert!(brief.get("steps").is_none());
    let result: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["status"], brief["status"]);
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
    let name = summary(&output)["output"].as_str().unwrap().to_string();
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
    let brief = summary(&output);
    assert_eq!(brief["status"], "sat");
    assert_eq!(brief["output"], "result.json");
    assert!(brief.get("assignment").is_none());
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
    assert!(help.contains("--generate"));
    assert!(help.contains("<K> <ALPHA> <N>"));
    assert!(help.contains("[default: 0.9]"));
    assert!(help.contains("[default: 0.01]"));
    assert!(help.contains("--output"));
    assert!(help.contains("--config"));
    assert!(help.contains("--progress"));
    assert!(help.contains("--no-progress"));
    let output = Command::new(env!("CARGO_BIN_EXE_bsp-rust"))
        .arg("--invalid")
        .arg("1")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn cli_loads_parameters_from_toml_config() {
    let dir = temp_dir("toml");
    let config = dir.join("run.toml");
    fs::write(
        &config,
        format!(
            "formula = {:?}\nseed = 3\nprogress = false\noutput = \"result.json\"\n",
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/3sat.cnf")
        ),
    )
    .unwrap();
    let output = run(&dir, &["--config", config.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let brief = summary(&output);
    assert_eq!(brief["status"], "sat");
    assert_eq!(brief["output"], "result.json");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn cli_flags_override_yaml_config() {
    let dir = temp_dir("yaml");
    let config = dir.join("run.yaml");
    fs::write(
        &config,
        format!(
            "formula: {}\nseed: 1\nprogress: false\noutput: from-config.json\n",
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/3sat.cnf")
        ),
    )
    .unwrap();
    let output = run(
        &dir,
        &[
            "--config",
            config.to_str().unwrap(),
            "--seed",
            "9",
            "--output",
            "from-cli.json",
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let brief = summary(&output);
    assert_eq!(brief["output"], "from-cli.json");
    let result: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("from-cli.json")).unwrap()).unwrap();
    assert_eq!(result["seed"], 9);
    let _ = fs::remove_dir_all(dir);
}
