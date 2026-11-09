mod cli;

use crate::cli::args::Cli;
use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let code = if error.use_stderr() {
                ExitCode::from(2)
            } else {
                ExitCode::SUCCESS
            };
            let _ = error.print();
            return code;
        }
    };
    if let Some(level) = cli.enable_tracing {
        tracing_subscriber::fmt()
            .json()
            .with_max_level(tracing::Level::from(level))
            .with_writer(std::io::stderr)
            .init();
    }
    match cli::resolve::resolve(cli).and_then(cli::batch::run) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("bsp-rust: {error}");
            ExitCode::from(2)
        }
    }
}
