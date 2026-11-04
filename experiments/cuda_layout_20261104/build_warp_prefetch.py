"""Coalesce block-record reads across one warp; lane zero sums in original order."""
import pathlib
import shutil
import subprocess
root = pathlib.Path(__file__).resolve().parents[2]
out = pathlib.Path(__file__).resolve().parent
p = root / 'crates/bsp-cuda/src/kernels.cu'
original = p.read_text()
base = subprocess.check_output(['git', 'show', 'ad6a0d3c496adc5fcf7e01c0ea44cffaba2e02eb:crates/bsp-cuda/src/kernels.cu'], cwd=root, text=True)
a = base.index('template <class T> __global__ void observable_final(Params p) {')
b = base.index('__device__ unsigned domain(', a)
replacement = '''template <class T> __global__ void observable_final(Params p) {
  unsigned lane = tid();
  if (lane >= 32) return;
  T vs = 0, cs = 0, vc = 0, cc = 0;
  unsigned flag = 1;
  unsigned count = (maximum(p.n, p.m) + 255) / 256;
  for (unsigned i = 0; i < count; i += 32) {
    Quad<T> record = {};
    if (i + lane < count) record = buf<Quad<T>>(p, 23)[i + lane];
    #pragma unroll 1
    for (unsigned j = 0; j < 32; ++j) {
      T vx = __shfl_sync(0xffffffffu, record.x, j);
      T cx = __shfl_sync(0xffffffffu, record.y, j);
      T px = __shfl_sync(0xffffffffu, record.z, j);
      if (!lane && i + j < count) {
        T y = vx - vc, t = vs + y;
        vc = (t - vs) - y;
        vs = t;
        y = cx - cc;
        t = cs + y;
        cc = (t - cs) - y;
        cs = t;
        flag &= px != 0;
      }
    }
  }
  if (!lane) {
    Control &s = *buf<Control>(p, 19);
    s.paramagnetic = flag;
    s.complexity = flag ? 0 : double(cs - vs);
    if (!isfinite(s.complexity)) atomicOr(&s.error, 1u);
  }
}
'''
try:
    p.write_text(base[:a] + replacement + base[b:])
    subprocess.run(['cargo', 'build', '--release', '--locked', '--features', 'cuda', '--example', 'cuda_layout_bench'], cwd=root, check=True)
    shutil.copy2(root / 'target/release/examples/cuda_layout_bench', out / 'warp32-bench')
finally:
    p.write_text(original)
