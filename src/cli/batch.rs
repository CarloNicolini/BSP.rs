use super::config::Resolved;
use super::single::run_single;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::{fs, process::ExitCode};

#[derive(Clone, Copy)]
pub struct BatchContext<'a> {
    pub input: &'a Path,
    pub bar: Option<&'a indicatif::ProgressBar>,
}

pub fn run(
    (mut resolved, mut inputs): (Resolved, Vec<PathBuf>),
) -> Result<ExitCode, Box<dyn std::error::Error>> {
    if inputs.is_empty() {
        if let Some(path) = resolved.formula.take() {
            inputs.push(path);
        }
    }
    if inputs.len() <= 1 && inputs.first().is_none_or(|p| !p.is_dir()) {
        resolved.formula = inputs.pop();
        return run_single(resolved, None);
    }
    if resolved.generate.is_some()
        || resolved.feistel.is_some()
        || resolved.feistel_ensemble.is_some()
    {
        return Err("batch DIMACS inputs conflict with formula generation".into());
    }
    if resolved.output.is_some()
        || resolved.execution.profile.is_some()
        || resolved.write_formula.is_some()
        || resolved.sweeps.is_some()
        || resolved.messages.is_some()
        || resolved.dump_messages
        || resolved.stream_table.is_some()
    {
        return Err("batch mode does not support --output, --profile, --write-formula, --sweeps, --messages, --dump-messages, or --stream-table".into());
    }
    resolved.options.validate()?;
    let mut files = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut children = Vec::new();
            for entry in fs::read_dir(&input)? {
                let entry = entry?;
                let path = entry.path();
                if entry.file_type()?.is_file()
                    && path
                        .extension()
                        .and_then(|v| v.to_str())
                        .is_some_and(|v| v.eq_ignore_ascii_case("cnf"))
                {
                    children.push(path);
                }
            }
            children.sort();
            files.extend(children);
        } else {
            files.push(input);
        }
    }
    if files.is_empty() {
        return Err("no .cnf files found in batch inputs".into());
    }
    let bar = if resolved.progress.unwrap_or(std::io::stderr().is_terminal())
        && std::io::stderr().is_terminal()
    {
        let bar = indicatif::ProgressBar::with_draw_target(
            Some(files.len() as u64),
            indicatif::ProgressDrawTarget::stderr(),
        );
        bar.set_style(
            indicatif::ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} {msg}")
                .expect("valid progress template"),
        );
        bar.enable_steady_tick(std::time::Duration::from_millis(100));
        Some(bar)
    } else {
        None
    };
    let total = files.len();
    let (mut succeeded, mut failed, mut errors) = (0usize, 0usize, 0usize);
    for path in files {
        if let Some(bar) = &bar {
            bar.set_message(path.display().to_string());
        }
        let mut job = resolved.clone();
        job.formula = Some(path.clone());
        job.progress = Some(false);
        match run_single(
            job,
            Some(BatchContext {
                input: &path,
                bar: bar.as_ref(),
            }),
        ) {
            Ok(code) if code == ExitCode::SUCCESS => succeeded += 1,
            Ok(_) => failed += 1,
            Err(error) => {
                errors += 1;
                let record = serde_json::json!({"input":path.to_string_lossy(), "status":"error", "error":error.to_string()});
                if let Some(bar) = &bar {
                    bar.suspend(|| println!("{record}"));
                } else {
                    println!("{record}");
                }
            }
        }
        if let Some(bar) = &bar {
            bar.inc(1);
        }
    }
    if let Some(bar) = bar {
        bar.finish_and_clear();
    }
    println!(
        "{}",
        serde_json::json!({"batch":true,"total":total,"succeeded":succeeded,"failed":failed,"errors":errors})
    );
    std::io::stdout().flush()?;
    Ok(if errors > 0 {
        ExitCode::from(2)
    } else if failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}
