"""Build isolated tuning binaries from the production source; restore it afterwards."""
import pathlib
import shutil
import subprocess
root = pathlib.Path(__file__).resolve().parents[2]
out = pathlib.Path(__file__).resolve().parent
kernel = root / 'crates/bsp-cuda/src/kernels.cu'
layout = root / 'crates/bsp-cuda/src/layout.rs'
original_kernel, original_layout = kernel.read_text(), layout.read_text()
baseline_kernel = subprocess.check_output(['git', 'show', 'ad6a0d3c496adc5fcf7e01c0ea44cffaba2e02eb:crates/bsp-cuda/src/kernels.cu'], cwd=root, text=True)
variants = {
    'baseline': (baseline_kernel, original_layout),
    'block128': (baseline_kernel.replace('static_cast<unsigned long long>(threads) + 255) / 256', 'static_cast<unsigned long long>(threads) + 127) / 128').replace('cavities_simd<T><<<nblocks, 256,', 'cavities_simd<T><<<nblocks, 128,'), original_layout),
    'q14': (baseline_kernel, original_layout.replace('Q_WINDOW: u32 = 1 << 16', 'Q_WINDOW: u32 = 1 << 14')),
    'q18': (baseline_kernel, original_layout.replace('Q_WINDOW: u32 = 1 << 16', 'Q_WINDOW: u32 = 1 << 18')),
}
try:
    for name, (cu, rs) in variants.items():
        assert name == "baseline" or cu != baseline_kernel or rs != original_layout, name
        kernel.write_text(cu)
        layout.write_text(rs)
        subprocess.run(['cargo', 'build', '--release', '--locked', '--features', 'cuda', '--example', 'cuda_layout_bench'], cwd=root, check=True)
        shutil.copy2(root / 'target/release/examples/cuda_layout_bench', out / (name + '-bench'))
        print('built', name, flush=True)
finally:
    kernel.write_text(original_kernel)
    layout.write_text(original_layout)
