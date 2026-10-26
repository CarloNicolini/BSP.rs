use std::{env, path::PathBuf, process::Command};

fn main() {
    assert_eq!(env::var("CARGO_CFG_TARGET_OS").unwrap(), "macos");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let status = Command::new("xcrun")
        .args(["clang", "-O3", "-fobjc-arc", "-c", "src/bridge.m", "-o"])
        .arg(out.join("bridge.o"))
        .status()
        .expect("Xcode Command Line Tools are required");
    assert!(status.success(), "Metal bridge compilation failed");
    assert!(Command::new("ar")
        .arg("crs")
        .arg(out.join("libbsp_metal_bridge.a"))
        .arg(out.join("bridge.o"))
        .status()
        .unwrap()
        .success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=bsp_metal_bridge");
    for framework in ["Metal", "Foundation", "CoreGraphics"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    println!("cargo:rerun-if-changed=src/bridge.m");
    println!("cargo:rerun-if-changed=src/kernels.metal");
}
