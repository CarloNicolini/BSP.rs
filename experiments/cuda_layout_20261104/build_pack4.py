"""Test four-edge packing to reduce index/output instructions; f64 stays unchanged."""
import pathlib
import shutil
import subprocess
root = pathlib.Path(__file__).resolve().parents[2]
p = root / 'crates/bsp-cuda/src/kernels.cu'
original = p.read_text()
base = subprocess.check_output(['git', 'show', 'ad6a0d3c496adc5fcf7e01c0ea44cffaba2e02eb:crates/bsp-cuda/src/kernels.cu'], cwd=root, text=True)
a = base.index('template <class T> __global__ void pack_eta(Params p) {')
b = base.index('template <class T> __global__ void cavities(Params p) {', a)
replacement = '''template <class T> __global__ void pack_eta(Params p) {
  unsigned k = tid() * (sizeof(T) == 4 ? 4u : 1u);
  if (k >= p.count || buf<Control>(p, 19)->done) return;
  if (sizeof(T) == 4 && k + 4 <= p.count) {
    uint4 e = buf<uint4>(p, 28)[k >> 2];
    float4 x = make_float4(buf<float>(p, 8)[e.x], buf<float>(p, 8)[e.y],
                          buf<float>(p, 8)[e.z], buf<float>(p, 8)[e.w]);
    if (p.observables) {
      unsigned *active = buf<unsigned>(p, 6), *ec = buf<unsigned>(p, 4);
      if (!active[ec[e.x]]) x.x = -x.x;
      if (!active[ec[e.y]]) x.y = -x.y;
      if (!active[ec[e.z]]) x.z = -x.z;
      if (!active[ec[e.w]]) x.w = -x.w;
    }
    buf<float4>(p, 32)[k >> 2] = x;
  } else {
    unsigned end = sizeof(T) == 4 ? min(k + 4u, p.count) : k + 1;
    for (; k < end; ++k) {
      unsigned e = buf<unsigned>(p, 28)[k];
      T x = buf<T>(p, 8)[e];
      if (p.observables && !buf<unsigned>(p, 6)[buf<unsigned>(p, 4)[e]]) x = -x;
      buf<T>(p, 32)[k] = x;
    }
  }
}
'''
source = base[:a] + replacement + base[b:]
source = source.replace('case 15:\n    pack_eta<T><<<blocks, 256,', 'case 15:\n    if (sizeof(T) == 4) blocks = unsigned((static_cast<unsigned long long>(count) + 1023) / 1024);\n    pack_eta<T><<<blocks, 256,')
assert source != base
try:
    p.write_text(source)
    subprocess.run(['cargo', 'build', '--release', '--locked', '--features', 'cuda', '--example', 'cuda_layout_bench'], cwd=root, check=True)
    shutil.copy2(root / 'target/release/examples/cuda_layout_bench', pathlib.Path(__file__).resolve().parent / 'pack4-bench')
finally:
    p.write_text(original)
