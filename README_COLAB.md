# Quick CUDA test on Google Colab (T4)

This guide builds the Python library with CUDA and checks that kernels actually
run on the GPU. Start with an active T4 runtime. A small SAT solve alone may
finish on CPU before reaching SP, so the test also performs explicit GPU sweeps.
Colab runtimes are temporary: repeat setup after a runtime reset.

## 1. From a local terminal over SSH

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) locally if needed, then install the Colab CLI:

```sh
uv tool install google-colab-cli
colab sessions
```

Complete the CLI's Google sign-in flow if prompted.
Use the same Google account as your notebook.
Discover the session with **Hardware: T4** in the output.
The value in square brackets is the CLI's local session name; the long value after it is the remote endpoint.
With multiple sessions, specify the local name:

```sh
export COLAB_SESSION='REPLACE_WITH_LOCAL_T4_SESSION_NAME'
colab ssh -s "$COLAB_SESSION"
```

### Create a T4 GPU remote session
Create a separately named CLI runtime if you want a new SSH-accessible session (availability and compute-unit charges depend on your Colab plan):
```sh
colab new -s bsp-t4 --gpu T4
colab sessions
colab ssh -s bsp-t4
```

SSH requires a local key; if the CLI reports none, generate one with `ssh-keygen -t ed25519` or pass an existing key with `colab ssh -i PATH -s NAME`.
If a browser-created runtime reports that the SSH endpoint is unavailable, use notebook cells or a CLI-created runtime. A notebook URL is not an SSH address.

The remaining commands run **inside Colab**, not on your laptop.

## 2. Install the build tools and clone

```sh
set -eu
nvidia-smi
apt-get update
apt-get install -y nvidia-cuda-toolkit
/usr/bin/nvcc --version
apt-get install -y build-essential curl git
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.local/bin/env"
uv tool install nvitop
. "$HOME/.cargo/env"
rustc --version
cd /content
if [ ! -d BSP.rs ]; then
  git clone https://github.com/CarloNicolini/BSP.rs.git
fi
```

Rust must be 1.88 or newer. If `/content/BSP.rs` already exists, reuse that checkout instead of cloning again. Use `git rev-parse HEAD` there to record the revision being tested.

The driver and compiler are separate dependencies: a working `nvidia-smi` does not mean `nvcc` is installed.
This guide explicitly installs Ubuntu's `nvidia-cuda-toolkit`.
The tested SSH runtime runs as root, so `sudo` is unnecessary; if `id -u` is nonzero, prefix the `apt-get` commands with `sudo`.
Do not replace Colab's working NVIDIA driver.

## 3. Build the CUDA Python package

```sh
set -eu
. "$HOME/.local/bin/env"
. "$HOME/.cargo/env"
cd /content/BSP.rs
export CUDA_HOME=/usr
export NVCC=/usr/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BSP_CUDA_ARCH=75
"$NVCC" --version
uv sync --python 3.12 --reinstall-package pybsp-rust \
  --config-settings 'build-args=--features extension-module,cuda'
```

These paths deliberately select the Ubuntu-packaged compiler and runtime.
Some images also contain `/usr/local/cuda`; mixing that compiler with the distro
runtime can produce version mismatches. `CUDA_HOME=/usr` works here because the
Ubuntu linker already searches `/usr/lib/x86_64-linux-gnu`.

T4 has compute capability **7.5**, hence `75`; it is supported by CUDA 12 and
13. For another GPU, discover its capability with
`nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader` and remove the dot.
The initial build can take several minutes. uv installs Python and NumPy and
builds the Rust extension; no PyTorch installation is required.

## 4. Verify GPU sweeps and a SAT assignment

```sh
set -eu
. "$HOME/.local/bin/env"
cd /content/BSP.rs
export CUDA_HOME=/usr
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run --no-sync python - <<'PY'
from pybsp_rust import Formula, Solver

# Force SP work on a moderately sized graph; don't rely on a tiny solve
# exercising the GPU before WalkSAT completes it.
formula = Formula.random(k=3, alpha=4.2, n=10000, seed=7)
for backend in ("cuda", "cuda-f32"):
    solver = Solver(formula=formula, backend=backend)
    solver.sweep_many(10)
    report = solver.execution_report
    print(backend, report)
    assert report["backend"] == backend, report
    assert report["cuda_sweeps"] > 0, report
    assert not report["fallbacks"], report
    assert report["device_error"] is None, report

# Independently check the returned assignment against the original formula.
easy = Formula.random(k=3, alpha=2.5, n=128, seed=3)
solver = Solver(
    formula=easy,
    seed=3,
    timeout_seconds=10,
    backend="cuda-f32",
)
result = solver.solve()
print(result.status, result.execution)
assert result.assignment is not None, result.status
assert easy.is_satisfied(result.assignment)
print("PASS: CUDA f64/f32 sweeps and a verified SAT assignment")
PY
```

Expect a device name identifying the T4, nonzero `cuda_sweeps`, empty `fallbacks`,
and the final `PASS` line. This is a smoke test, not a performance comparison.
Use `uv run --no-sync` after the GPU build to preserve its feature selection.

## Verified run

On 2026-10-29, the setup, build, and test blocks were run on a Colab Tesla T4
with Ubuntu 24.04, driver 580.82.07, and the Ubuntu-packaged CUDA 12.0 compiler.
The runtime already had the toolkit installed; the explicit package installation
completed successfully as an idempotent step. An existing checkout was reused.
Both precision modes reported `Tesla T4 (sm_75)`, 10 CUDA sweeps, no fallbacks,
and no device errors. The final solve returned `sat`, completed 9 CUDA sweeps,
and passed the formula's assignment check.

The blocks were executed through `colab exec` because an existing SSH connection
occupied the runtime's SSH endpoint. This checks the shell commands on the actual
GPU runtime without requiring a second SSH connection. Fresh Google sign-in and
allocation of a new runtime were not repeated.

## Troubleshooting and finishing

| Symptom | Action |
| --- | --- |
| CLI requests Google authentication | Complete sign-in locally; do not put tokens in this repository. |
| T4 listed as `[?]` / session not found | Use the existing notebook, or create a named CLI session as above. |
| `nvidia-smi` finds no GPU | Connect a GPU runtime; check you did not select the CPU session. |
| Unsupported `compute_61` | Export `BSP_CUDA_ARCH=75` and repeat the CUDA sync with `--reinstall-package`. |
| `nvcc: command not found` | Run `apt-get install -y nvidia-cuda-toolkit`, then check `/usr/bin/nvcc --version`. |
| SSH reports HTTP 429 / already-active SSH session | Exit the other SSH connection, or run these blocks as notebook `%%bash` cells. |
| Cannot find `-lcudart` / `libcudart.so` | For the Ubuntu package, use `/usr/lib/x86_64-linux-gnu`; vendor toolkits usually use `$CUDA_HOME/lib64`. |
| CUDA backend unavailable | Rebuild with the CUDA sync command and run with `--no-sync`. |
| Driver/toolkit incompatibility | Use a toolkit supported by the Colab driver; consult `nvidia-smi` and `nvcc --version`. |

Exit SSH with `exit`. Disconnecting SSH does not necessarily release the runtime.
When finished, use **Runtime → Disconnect and delete runtime** in Colab, or run
`colab stop -s "$COLAB_SESSION"` locally for a CLI-managed session you intend to
release. Save any results before deleting a runtime.

See [CUDA.md](CUDA.md) for the full device test suite and numerical contract.
