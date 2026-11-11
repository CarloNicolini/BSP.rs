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
    assert!(help.contains("--stream-table"));
    let output = Command::new(env!("CARGO_BIN_EXE_bsp-rust"))
        .arg("--invalid")
        .arg("1")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn cli_stream_table_writes_tsv_rows() {
    let dir = temp_dir("stream-table");
    let output = run(
        &dir,
        &[
            "--no-progress",
            "--update-frequency",
            "0.1",
            "--stream-table",
            "stats.tsv",
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
    let text = fs::read_to_string(dir.join("stats.tsv")).unwrap();
    let mut lines = text.lines().filter(|line| !line.is_empty());
    let header = lines.next().unwrap();
    assert_eq!(
        header,
        "time\tstep\taction\tfree\tclauses\tsweeps\tresidual\tcomplexity\tchanged"
    );
    let row = lines.next().expect("at least one streamed data row");
    let cells: Vec<_> = row.split('\t').collect();
    assert_eq!(cells.len(), 9);
    assert!(cells[0].parse::<f64>().is_ok(), "time={}", cells[0]);
    assert!(cells[1].parse::<usize>().is_ok(), "step={}", cells[1]);
    assert!(!cells[2].is_empty(), "action");
    assert!(cells[7].parse::<f64>().is_ok(), "complexity={}", cells[7]);
    let brief = summary(&output);
    assert_eq!(brief["status"], "sat");
    let _ = fs::remove_dir_all(dir);
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

#[test]
fn feistel_config_export_messages_and_conflicts() {
    let dir = temp_dir("feistel");
    fs::write(
        dir.join("run.toml"),
        "seed = 7\nsweeps = 2\ndump_messages = true\n[feistel]\nk = 3\nalpha = 4.0\nn = 100\n",
    )
    .unwrap();
    let output = run(&dir, &["--config", "run.toml", "--output", "result.json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    assert_eq!(summary(&output)["output"], "result.json");
    let report: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("result.json")).unwrap()).unwrap();
    assert_eq!(report["topology"]["generator"], "fluctuating-feistel-v1");
    assert_eq!(report["topology"]["ensemble"], "random-k-sat-target");
    assert!(report["topology"]["repaired_clauses"].is_u64());

    let partitioned = run(
        &dir,
        &[
            "--config",
            "run.toml",
            "--feistel-ensemble",
            "partitioned",
            "--output",
            "partitioned.json",
        ],
    );
    assert!(partitioned.status.success());
    let warning = String::from_utf8_lossy(&partitioned.stderr);
    assert_eq!(warning.lines().count(), 1);
    assert!(warning.contains("different random k-SAT ensemble from --generate"));
    assert!(warning.contains("depending on instance size"));
    let report_partitioned: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("partitioned.json")).unwrap()).unwrap();
    assert_eq!(
        report_partitioned["topology"]["generator"],
        "partitioned-feistel-v1"
    );
    assert_eq!(report_partitioned["topology"]["degree_counts"]["12"], 84);

    fs::write(
        dir.join("messages.json"),
        serde_json::to_string(&report["messages"]).unwrap(),
    )
    .unwrap();
    assert!(run(
        &dir,
        &[
            "--feistel",
            "3",
            "4",
            "100",
            "--seed",
            "7",
            "--sweeps",
            "1",
            "--messages",
            "messages.json"
        ]
    )
    .status
    .success());
    let export = run(
        &dir,
        &[
            "--feistel",
            "3",
            "4",
            "100",
            "--seed",
            "7",
            "--write-formula",
            "f.cnf",
        ],
    );
    assert!(export.status.success());
    assert_eq!(export.stderr, output.stderr);
    assert!(export.stdout.is_empty());
    let standard = run(
        &dir,
        &[
            "--generate",
            "3",
            "4",
            "100",
            "--write-formula",
            "standard.cnf",
        ],
    );
    assert!(standard.status.success());
    assert!(standard.stderr.is_empty());
    let f =
        bsp_rust::Formula::from_dimacs(&fs::read_to_string(dir.join("f.cnf")).unwrap()).unwrap();
    assert_eq!(
        f.clauses(),
        bsp_rust::Formula::new(
            100,
            bsp_rust::Formula::feistel_fluctuating(3, 4.0, 100, 7)
                .unwrap()
                .clauses(),
        )
        .unwrap()
        .clauses()
    );
    for extra in [
        vec!["--generate", "3", "4", "100"],
        vec!["f.cnf"],
        vec!["--backend", "cuda"],
        vec!["--backend", "cuda-f32"],
    ] {
        let mut args = vec!["--config", "run.toml"];
        args.extend(extra);
        assert_eq!(run(&dir, &args).status.code(), Some(2));
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let output = run(&dir, &["--config", "run.toml", "--backend", "metal"]);
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("build with --features metal"));
    }
    for text in [
        "feistel: {k: 3, alpha: 4, n: 100}\nsweeps: 1\n",
        "{\"feistel\":{\"k\":3,\"alpha\":4,\"n\":100},\"sweeps\":1}",
    ] {
        let path = if text.starts_with('{') {
            "run.json"
        } else {
            "run.yaml"
        };
        fs::write(dir.join(path), text).unwrap();
        assert!(run(&dir, &["--config", path]).status.success());
    }
    assert_eq!(
        run(&dir, &["--feistel", "3", "4", "1000000000"])
            .status
            .code(),
        Some(2)
    );
    fs::write(
        dir.join("partitioned.toml"),
        "[feistel]\nk = 3\nalpha = 4.0\nn = 100\nensemble = \"partitioned\"\n",
    )
    .unwrap();
    let export_partitioned = run(
        &dir,
        &["--config", "partitioned.toml", "--write-formula", "p.cnf"],
    );
    assert!(export_partitioned.status.success());
    assert_eq!(export_partitioned.stderr, partitioned.stderr);
    let override_mode = run(
        &dir,
        &[
            "--config",
            "partitioned.toml",
            "--feistel-ensemble",
            "fluctuating",
            "--write-formula",
            "f.cnf",
        ],
    );
    assert!(override_mode.status.success());
    assert!(override_mode.stderr.is_empty());
    assert_eq!(
        run(
            &dir,
            &[
                "--generate",
                "3",
                "4",
                "100",
                "--feistel-ensemble",
                "partitioned"
            ]
        )
        .status
        .code(),
        Some(2)
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn feistel_dimacs_replays_the_same_seed_on_scalar_cpu() {
    let dir = temp_dir("feistel-replay");
    for seed in ["2", "7"] {
        let source = ["--feistel", "3", "4", "100", "--seed", seed];
        let mut export = source.to_vec();
        export.extend(["--write-formula", "replay.cnf"]);
        assert!(run(&dir, &export).status.success());

        let common = [
            "--max-steps",
            "512",
            "--max-sweeps",
            "256",
            "--frac",
            "0.01",
        ];
        let mut implicit = source.to_vec();
        implicit.extend(common);
        implicit.extend(["--output", "implicit.json"]);
        let a = run(&dir, &implicit);
        let mut explicit = vec!["replay.cnf", "--preserve-literal-order", "--seed", seed];
        explicit.extend(common);
        explicit.extend(["--output", "explicit.json"]);
        let b = run(&dir, &explicit);
        assert_eq!(a.status.code(), b.status.code());
        assert!(matches!(a.status.code(), Some(0 | 1)));
        let load = |name| {
            serde_json::from_str::<serde_json::Value>(&fs::read_to_string(dir.join(name)).unwrap())
                .unwrap()
        };
        let (a, b) = (load("implicit.json"), load("explicit.json"));
        for key in [
            "status",
            "assignment",
            "partial_assignment",
            "steps",
            "sp_sweeps",
            "flips",
            "unit_assignments",
        ] {
            assert_eq!(a[key], b[key], "{seed}: {key}");
        }
        if let Some(assignment) = a["assignment"].as_array() {
            let x: Vec<_> = assignment
                .iter()
                .map(|v| v.as_i64().unwrap() as i8)
                .collect();
            let f = bsp_rust::Formula::from_dimacs(
                &fs::read_to_string(dir.join("replay.cnf")).unwrap(),
            )
            .unwrap();
            assert!(f.is_satisfied(&x));
        }
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn generated_and_feistel_cli_sweeps_use_the_requested_cpu_pool() {
    let dir = temp_dir("parallel-sources");
    for (flag, n) in [("--generate", "2000"), ("--feistel", "500")] {
        let out = run(
            &dir,
            &[
                flag,
                "3",
                "4",
                n,
                "--sweeps",
                "2",
                "--backend",
                "cpu-parallel",
                "--threads",
                "4",
                "--output",
                "parallel.json",
            ],
        );
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let report: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("parallel.json")).unwrap()).unwrap();
        assert_eq!(report["execution"]["parallel_sweeps"], 2);
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn concurrent_default_reports_do_not_overwrite() {
    let dir = temp_dir("collision");
    let mut children: Vec<_> = (0..12)
        .map(|_| {
            Command::new(env!("CARGO_BIN_EXE_bsp-rust"))
                .current_dir(&dir)
                .args(["--generate", "3", "0.1", "10", "--no-progress"])
                .stdout(std::process::Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    let mut names = std::collections::HashSet::new();
    for child in children.drain(..) {
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success());
        let name = summary(&out)["output"].as_str().unwrap().to_owned();
        assert!(names.insert(name.clone()));
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join(name)).unwrap()).unwrap();
        assert_eq!(report["status"], "sat");
    }
    assert_eq!(names.len(), 12);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn partial_options_preserve_config_and_null_clears_timeout() {
    let dir = temp_dir("options-merge");
    fs::write(
        dir.join("run.toml"),
        "seed = 123\ntimeout_seconds = 0.00000000001\n",
    )
    .unwrap();
    fs::write(
        dir.join("options.json"),
        r#"{"damping":0.2,"timeout_seconds":null}"#,
    )
    .unwrap();
    let args = [
        "--generate",
        "3",
        "0.1",
        "10",
        "--config",
        "run.toml",
        "--options",
        "options.json",
        "--output",
        "result.json",
    ];
    let output = run(&dir, &args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(summary(&output)["seed"], 123);
    fs::write(dir.join("options.json"), r#"{"misspelled":1}"#).unwrap();
    assert_eq!(run(&dir, &args).status.code(), Some(2));
    fs::write(dir.join("options.json"), r#"{"seed":456}"#).unwrap();
    let mut override_args = args.to_vec();
    override_args.extend(["--seed", "9", "--timeout", "10"]);
    assert_eq!(summary(&run(&dir, &override_args))["seed"], 9);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn profile_write_error_preserves_solve_result() {
    let dir = temp_dir("profile-error");
    let output = run(
        &dir,
        &[
            "--generate",
            "3",
            "0.1",
            "10",
            "--profile",
            "missing/profile.json",
            "--output",
            "result.json",
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("result.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "sat");
    assert!(report["execution"]["profile_error"].is_string());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn batch_directory_processes_sorted_inputs_and_continues_after_errors() {
    let dir = temp_dir("batch-dir");
    fs::create_dir(dir.join("formulas")).unwrap();
    fs::create_dir(dir.join("formulas/nested")).unwrap();
    fs::write(dir.join("formulas/a.cnf"), "p cnf 1 1\n1 0\n").unwrap();
    fs::write(dir.join("formulas/b.CNF"), "p cnf 1 2\n1 0\n-1 0\n").unwrap();
    fs::write(dir.join("formulas/c.cnf"), "p cnf 1 1\n2 0\n").unwrap();
    fs::write(dir.join("formulas/z.txt"), "not a formula").unwrap();
    fs::write(dir.join("formulas/nested/d.cnf"), "p cnf 1 1\n1 0\n").unwrap();
    let output = run(&dir, &["formulas", "--no-progress"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let text = stdout(&output);
    let records: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 4);
    assert_eq!(records[0]["input"], "formulas/a.cnf");
    assert_eq!(records[0]["status"], "sat");
    assert_eq!(records[1]["input"], "formulas/b.CNF");
    assert_eq!(records[1]["status"], "contradiction");
    assert_eq!(records[2]["input"], "formulas/c.cnf");
    assert_eq!(records[2]["status"], "error");
    assert_eq!(
        records[3],
        serde_json::json!({"batch":true,"total":3,"succeeded":1,"failed":1,"errors":1})
    );
    for record in &records[..2] {
        let path = dir.join(record["output"].as_str().unwrap());
        let report: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(report["status"], record["status"]);
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn batch_files_keep_distinct_reports_and_return_failure_for_unsolved() {
    let dir = temp_dir("batch-files");
    fs::create_dir(dir.join("a")).unwrap();
    fs::create_dir(dir.join("b")).unwrap();
    fs::write(dir.join("a/same.cnf"), "p cnf 1 1\n1 0\n").unwrap();
    fs::write(dir.join("b/same.cnf"), "p cnf 1 2\n1 0\n-1 0\n").unwrap();
    let output = run(&dir, &["b/same.cnf", "a/same.cnf", "--progress"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty()); // no terminal: no escape sequences or progress line
    let text = stdout(&output);
    let records: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records[0]["input"], "b/same.cnf");
    assert_eq!(records[1]["input"], "a/same.cnf");
    assert_ne!(records[0]["output"], records[1]["output"]);
    assert_eq!(
        records[2],
        serde_json::json!({"batch":true,"total":2,"succeeded":1,"failed":1,"errors":0})
    );
    for record in &records[..2] {
        assert!(dir.join(record["output"].as_str().unwrap()).exists());
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn batch_rejects_ambiguous_options_and_empty_directories() {
    let dir = temp_dir("batch-options");
    fs::create_dir(dir.join("empty")).unwrap();
    assert_eq!(run(&dir, &["empty"]).status.code(), Some(2));
    for option in [
        vec!["--output", "same.json"],
        vec!["--profile", "same.json"],
        vec!["--write-formula", "same.cnf"],
        vec!["--sweeps", "1"],
        vec!["--messages", "messages.json"],
        vec!["--dump-messages"],
        vec!["--stream-table", "stats.tsv"],
        vec!["--generate", "3", "2", "10"],
    ] {
        let mut args = vec!["a.cnf", "b.cnf"];
        args.extend(option);
        assert_eq!(run(&dir, &args).status.code(), Some(2), "{args:?}");
    }
    fs::write(dir.join("config.toml"), "formula = \"empty\"\n").unwrap();
    assert_eq!(
        run(&dir, &["--config", "config.toml"]).status.code(),
        Some(2)
    );
    let _ = fs::remove_dir_all(dir);
}
