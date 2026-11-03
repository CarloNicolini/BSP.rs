use bsp_rust::{Backend, ExecutionOptions, Options};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateSpec {
    pub k: usize,
    pub alpha: f64,
    pub n: usize,
}

/// Values loaded from a TOML, YAML, or JSON run file.
///
/// Absent fields leave the previous layer unchanged. Field names match the
/// library `Options` and `ExecutionOptions` serde names, plus run controls.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FileConfig {
    pub formula: Option<PathBuf>,
    pub generate: Option<GenerateSpec>,
    pub seed: Option<u64>,
    pub backtrack_ratio: Option<f64>,
    pub batch_fraction: Option<f64>,
    pub epsilon: Option<f64>,
    pub damping: Option<f64>,
    pub max_sweeps: Option<usize>,
    pub max_steps: Option<usize>,
    pub max_flips: Option<u64>,
    pub walksat_restarts: Option<usize>,
    pub walksat_noise: Option<f64>,
    pub timeout_seconds: Option<f64>,
    pub backend: Option<Backend>,
    pub threads: Option<usize>,
    pub profile: Option<PathBuf>,
    pub sweeps: Option<usize>,
    pub messages: Option<PathBuf>,
    pub dump_messages: Option<bool>,
    pub write_formula: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub progress: Option<bool>,
}

#[derive(Clone, Debug, Default)]
pub struct Resolved {
    pub formula: Option<PathBuf>,
    pub generate: Option<(usize, f64, usize)>,
    pub options: Options,
    pub execution: ExecutionOptions,
    pub sweeps: Option<usize>,
    pub messages: Option<PathBuf>,
    pub dump_messages: bool,
    pub write_formula: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub progress: Option<bool>,
}

impl FileConfig {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = fs::read_to_string(path)?;
        let ext = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "toml" => Ok(toml::from_str(&text)?),
            "yaml" | "yml" => Ok(serde_yaml::from_str(&text)?),
            "json" => Ok(serde_json::from_str(&text)?),
            _ => Err("config path must end in .toml, .yaml, .yml, or .json".into()),
        }
    }

    pub fn apply(&self, resolved: &mut Resolved) {
        if let Some(path) = &self.formula {
            resolved.formula = Some(path.clone());
        }
        if let Some(generate) = &self.generate {
            resolved.generate = Some((generate.k, generate.alpha, generate.n));
        }
        if let Some(value) = self.seed {
            resolved.options.seed = value;
        }
        if let Some(value) = self.backtrack_ratio {
            resolved.options.backtrack_ratio = value;
        }
        if let Some(value) = self.batch_fraction {
            resolved.options.batch_fraction = value;
        }
        if let Some(value) = self.epsilon {
            resolved.options.epsilon = value;
        }
        if let Some(value) = self.damping {
            resolved.options.damping = value;
        }
        if let Some(value) = self.max_sweeps {
            resolved.options.max_sweeps = value;
        }
        if let Some(value) = self.max_steps {
            resolved.options.max_steps = value;
        }
        if let Some(value) = self.max_flips {
            resolved.options.max_flips = value;
        }
        if let Some(value) = self.walksat_restarts {
            resolved.options.walksat_restarts = value;
        }
        if let Some(value) = self.walksat_noise {
            resolved.options.walksat_noise = value;
        }
        if let Some(value) = self.timeout_seconds {
            resolved.options.timeout_seconds = Some(value);
        }
        if let Some(value) = self.backend {
            resolved.execution.backend = value;
        }
        if let Some(value) = self.threads {
            resolved.execution.threads = value;
        }
        if let Some(path) = &self.profile {
            resolved.execution.profile = Some(path.clone());
        }
        if let Some(value) = self.sweeps {
            resolved.sweeps = Some(value);
        }
        if let Some(path) = &self.messages {
            resolved.messages = Some(path.clone());
        }
        if let Some(value) = self.dump_messages {
            resolved.dump_messages = value;
        }
        if let Some(path) = &self.write_formula {
            resolved.write_formula = Some(path.clone());
        }
        if let Some(path) = &self.output {
            resolved.output = Some(path.clone());
        }
        if let Some(value) = self.progress {
            resolved.progress = Some(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(label: &str, name: &str, body: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bsp-config-{label}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(body.as_bytes()).unwrap();
        path
    }

    #[test]
    fn toml_and_yaml_override_selected_fields() {
        let toml_path = write_temp(
            "toml",
            "run.toml",
            r#"
formula = "examples/3sat.cnf"
seed = 7
backtrack_ratio = 0.5
backend = "cpu-parallel"
threads = 4
progress = false
dump_messages = true

[generate]
k = 3
alpha = 2.5
n = 128
"#,
        );
        let yaml_path = write_temp(
            "yaml",
            "run.yaml",
            "seed: 9\nepsilon: 0.02\nbackend: metal\n",
        );

        let mut resolved = Resolved::default();
        FileConfig::load(&toml_path).unwrap().apply(&mut resolved);
        assert_eq!(
            resolved.formula.as_deref(),
            Some(Path::new("examples/3sat.cnf"))
        );
        assert_eq!(resolved.generate, Some((3, 2.5, 128)));
        assert_eq!(resolved.options.seed, 7);
        assert_eq!(resolved.options.backtrack_ratio, 0.5);
        assert_eq!(resolved.options.epsilon, Options::default().epsilon);
        assert_eq!(resolved.execution.backend, Backend::CpuParallel);
        assert_eq!(resolved.execution.threads, 4);
        assert_eq!(resolved.progress, Some(false));
        assert!(resolved.dump_messages);

        FileConfig::load(&yaml_path).unwrap().apply(&mut resolved);
        assert_eq!(resolved.options.seed, 9);
        assert_eq!(resolved.options.epsilon, 0.02);
        assert_eq!(resolved.execution.backend, Backend::Metal);
        assert_eq!(resolved.options.backtrack_ratio, 0.5);

        let _ = fs::remove_dir_all(toml_path.parent().unwrap());
        let _ = fs::remove_dir_all(yaml_path.parent().unwrap());
    }

    #[test]
    fn unknown_fields_and_extensions_are_rejected() {
        let path = write_temp("bad", "run.toml", "not_a_field = 1\n");
        assert!(FileConfig::load(&path).is_err());
        let path = write_temp("ext", "run.cfg", "seed = 1\n");
        assert!(FileConfig::load(&path)
            .unwrap_err()
            .to_string()
            .contains(".toml"));
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }
}
