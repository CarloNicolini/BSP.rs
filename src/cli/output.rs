use bsp_rust::Backend;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::CpuParallel => "cpu-parallel",
        Backend::Metal => "metal",
        Backend::Cuda => "cuda",
        Backend::CudaF32 => "cuda-f32",
    }
}

pub fn formula_token(path: &str) -> String {
    let stem = Path::new(path)
        .file_stem()
        .and_then(|text| text.to_str())
        .unwrap_or("formula");
    let mut out = String::new();
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "formula".to_string()
    } else {
        out
    }
}

pub fn generated_token(k: usize, alpha: f64, n: usize) -> String {
    format!("k{k}_alpha{alpha}_n{n}")
}

pub fn output_name(formula: &str, seed: u64, ratio: f64, backend: &str, stamp: &str) -> String {
    format!("output_{formula}_seed{seed}_r{ratio}_{backend}_{stamp}.json")
}

pub fn utc_stamp_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    utc_compact(secs)
}

pub fn utc_compact(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let time = (secs % 86_400) as u32;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        time / 3600,
        (time % 3600) / 60,
        time % 60
    )
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    if month <= 2 {
        year += 1;
    }
    (year, month, day)
}

pub fn write_json(
    path: &Path,
    value: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(path)?;
    serde_json::to_writer(&mut file, value)?;
    use std::io::Write;
    file.write_all(b"\n")?;
    Ok(())
}

const SUMMARY_KEYS: &[&str] = &[
    "status",
    "seed",
    "wall_seconds",
    "sp_sweeps",
    "flips",
    "decimations",
    "releases",
    "unit_assignments",
    "n",
    "m",
    "edges",
    "sweeps",
    "seconds",
    "residual",
];

pub fn summary(path: &Path, report: &serde_json::Value, times: [f64; 3]) -> serde_json::Value {
    let mut out = serde_json::json!({
        "output": path.display().to_string(),
        "formula_seconds": times[0],
        "bsp_seconds": times[1],
        "walksat_seconds": times[2],
    });
    for key in SUMMARY_KEYS {
        if let Some(value) = report.get(*key) {
            out[*key] = value.clone();
        }
    }
    out
}

pub fn default_path(formula: &str, seed: u64, ratio: f64, backend: Backend) -> PathBuf {
    PathBuf::from(output_name(
        formula,
        seed,
        ratio,
        backend_name(backend),
        &utc_stamp_now(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_stamps_match_known_instants() {
        assert_eq!(utc_compact(0), "19700101T000000Z");
        assert_eq!(utc_compact(1793715633), "20261103T142033Z");
    }

    #[test]
    fn names_keep_bash_field_separators() {
        assert_eq!(formula_token("examples/3sat.cnf"), "3sat");
        assert_eq!(formula_token("my formula.cnf"), "my_formula");
        assert_eq!(
            output_name("k3_alpha2.5_n128", 1, 0.9, "cpu", "20261103T142033Z"),
            "output_k3_alpha2.5_n128_seed1_r0.9_cpu_20261103T142033Z.json"
        );
    }
}
