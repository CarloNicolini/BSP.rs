"""Prefetch records while retaining the exact serial Kahan summation order."""
import pathlib
import shutil
import subprocess
root = pathlib.Path(__file__).resolve().parents[2]
out = pathlib.Path(__file__).resolve().parent
p = root / 'crates/bsp-cuda/src/kernels.cu'
original = p.read_text()
base = subprocess.check_output(['git', 'show', 'ad6a0d3c496adc5fcf7e01c0ea44cffaba2e02eb:crates/bsp-cuda/src/kernels.cu'], cwd=root, text=True)
a = base.index('  for (unsigned i = 0; i < (maximum(p.n, p.m) + 255) / 256; ++i) {')
b = base.index('\n  Control &s =', a)
body = base[a:b]
consume = body[body.index('    T y ='):body.rindex('\n  }')]
try:
    for name, width in [('obsbase', 0), ('prefetch4', 4), ('prefetch8', 8)]:
        if width:
            replacement = f'''  unsigned count = (maximum(p.n, p.m) + 255) / 256;
  for (unsigned i = 0; i < count; i += {width}) {{
    Quad<T> records[{width}];
    #pragma unroll
    for (unsigned j = 0; j < {width}; ++j)
      if (i + j < count) records[j] = buf<Quad<T>>(p, 23)[i + j];
    asm volatile("" ::: "memory");
    #pragma unroll
    for (unsigned j = 0; j < {width}; ++j) {{
      if (i + j < count) {{
        Quad<T> a = records[j];
{consume}
      }}
    }}
  }}'''
            p.write_text(base[:a] + replacement + base[b:])
        else:
            p.write_text(base)
        subprocess.run(['cargo', 'build', '--release', '--locked', '--features', 'cuda', '--example', 'cuda_layout_bench'], cwd=root, check=True)
        shutil.copy2(root / 'target/release/examples/cuda_layout_bench', out / (name + '-bench'))
        print('built', name, flush=True)
finally:
    p.write_text(original)
