use std::{env, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let arch = env::var("BSP_CUDA_ARCH").unwrap_or_else(|_| "61".into());
    assert!(
        arch.bytes().all(|c| c.is_ascii_digit()) && !arch.is_empty(),
        "BSP_CUDA_ARCH must contain digits, e.g. 61 or 86"
    );
    let status = Command::new(&nvcc)
        .args([
            "-O3",
            "-lineinfo",
            "-std=c++14",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "--ftz=false",
            "-Xcompiler",
            "-fPIC",
            "-gencode",
        ])
        .arg(format!(
            "arch=compute_{arch},code=[sm_{arch},compute_{arch}]"
        ))
        .args(["-c", "src/kernels.cu", "-o"])
        .arg(out.join("kernels.o"))
        .status()
        .expect("CUDA feature requires nvcc; install a CUDA toolkit or set NVCC");
    assert!(status.success(), "CUDA compilation failed. Pascal requires CUDA <=12.9; set BSP_CUDA_ARCH for another architecture.");
    assert!(Command::new("ar")
        .arg("crs")
        .arg(out.join("libbsp_cuda_kernels.a"))
        .arg(out.join("kernels.o"))
        .status()
        .unwrap()
        .success());
    println!("cargo:rustc-link-search=native={}", out.display());
    if let Some(root) = env::var_os("CUDA_HOME") {
        println!(
            "cargo:rustc-link-search=native={}/lib64",
            PathBuf::from(root).display()
        );
    }
    println!("cargo:rustc-link-lib=static=bsp_cuda_kernels");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rerun-if-changed=src/kernels.cu");
    for name in ["NVCC", "CUDA_HOME", "BSP_CUDA_ARCH"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
}
