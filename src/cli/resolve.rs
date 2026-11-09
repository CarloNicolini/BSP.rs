use super::args::Cli;
use super::config::{FileConfig, Resolved};
use std::fs;
use std::path::PathBuf;

fn parse_generate(values: &[String]) -> Result<(usize, f64, usize), Box<dyn std::error::Error>> {
    if values.len() != 3 {
        return Err("formula generation needs K ALPHA N".into());
    }
    Ok((values[0].parse()?, values[1].parse()?, values[2].parse()?))
}

pub fn resolve(cli: Cli) -> Result<(Resolved, Vec<PathBuf>), Box<dyn std::error::Error>> {
    let mut resolved = Resolved::default();
    if let Some(path) = &cli.config {
        FileConfig::load(path)?.apply(&mut resolved);
    }
    if let Some(path) = &cli.options {
        let patch: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        let patch = patch.as_object().ok_or("options must be a JSON object")?;
        let mut merged = serde_json::to_value(&resolved.options)?;
        merged
            .as_object_mut()
            .expect("Options serializes as an object")
            .extend(patch.clone());
        resolved.options = serde_json::from_value(merged)?;
    }
    FileConfig::from(&cli).apply(&mut resolved);
    let inputs = cli.formula;
    if !inputs.is_empty() {
        resolved.formula = None;
    }
    if let Some(values) = &cli.generate {
        resolved.generate = Some(parse_generate(values)?);
    }
    if let Some(values) = &cli.feistel {
        resolved.feistel = Some(parse_generate(values)?);
    }
    if let Some(ensemble) = cli.feistel_ensemble {
        resolved.feistel_ensemble = Some(ensemble);
    }
    Ok((resolved, inputs))
}
