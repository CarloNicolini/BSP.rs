use std::{env, process::Command};
fn main() {
    let compiler = Command::new(env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .arg("--version")
        .output()
        .expect("rustc version");
    println!(
        "cargo:rustc-env=BSP_RUSTC={}",
        String::from_utf8_lossy(&compiler.stdout).trim()
    );
    println!(
        "cargo:rustc-env=BSP_BUILD_PROFILE={}",
        env::var("PROFILE").unwrap()
    );
    println!(
        "cargo:rustc-env=BSP_RUSTFLAGS={}",
        env::var("CARGO_ENCODED_RUSTFLAGS")
            .unwrap_or_default()
            .replace('\u{1f}', " ")
    );
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
}
