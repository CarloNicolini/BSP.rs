# BSP.rs

[![Build and test](https://github.com/CarloNicolini/BSP.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/CarloNicolini/BSP.rs/actions/workflows/ci.yml)
[![License: GPL v2 or later](https://img.shields.io/badge/license-GPL--2.0--or--later-blue.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/Rust-1.88%2B-orange.svg)](Cargo.toml)

BSP.rs ships the `bsp-rust` executable for Backtracking Survey Propagation on
CNF SAT. Generate random k-SAT formulas or load DIMACS files, then write a JSON
result. An optional Python API shares the same Rust core. CPU execution works on
macOS and Linux; optional GPU backends support Apple silicon (Metal) and NVIDIA
GPUs on Linux (CUDA).

BSP is a heuristic: returned SAT assignments are verified, but an unsuccessful
run does not prove UNSAT.

## Quick start: `bsp-rust`

You need Rust 1.88+ and the usual platform compiler tools. Python is not required
for the executable. From a clone, `make` builds the release binary (same as
`make rust`). The Makefile picks Metal on Apple silicon with Xcode tools, CUDA
on Linux when an NVIDIA GPU and `nvcc` are present, and CPU otherwise:

```sh
git clone https://github.com/CarloNicolini/BSP.rs.git
cd BSP.rs
make
target/release/bsp-rust --version
target/release/bsp-rust examples/3sat.cnf
target/release/bsp-rust --generate 3 2.5 128 --seed 3 --timeout 10
target/release/bsp-rust --config examples/solve.toml
```

Override detection with `BACKEND=cpu`, `metal`, or `cuda` (for example
`make BACKEND=cpu`), or call `make rust-metal` / `make rust-cuda`. Run
`make help` for every target. `make clean` removes Cargo build outputs,
`docs/_build`, and `dist`.

Install on PATH with `cargo install --path . --locked`, then use `bsp-rust --help`.
Pass shared parameters with `--config PATH` (`.toml`, `.yaml`, `.yml`, or
`.json`); explicit flags override the file. The program writes the full JSON
result to a file. The default file name contains the main run variables and the
UTC time. `--output PATH` selects another file. `--progress` prints one stdout
line with timers for formula setup, BSP, and WalkSAT. Stdout then prints one
JSON summary with `status` and `output`. Exit code 0 means a verified SAT
assignment. Exit code 1 means a heuristic failure or a limit. Exit code 2 means
invalid input. After a GPU-capable `make` build, select `--backend metal`,
`cuda`, or `cuda-f32` at run time.

Other Makefile targets: `make python` installs the Python package with the same
backend detection, `make docs` builds Sphinx HTML, and `make docs-serve` serves
it (`PORT=8001` changes the port). After a GPU Python build, use
`uv run --no-sync ...` so a plain `uv run` does not rebuild with CPU defaults.
GitHub Actions exercises the Makefile builds on Linux, macOS, and Windows; the
hosted CUDA job uses an explicit override because that runner has no GPU.

## Experimental implicit topology

`bsp-rust --feistel 3 4 1000 --seed 7 --backend cpu` reconstructs the factor
graph on demand. The default targets the random k-SAT ensemble of
`--generate`, with fluctuating variable degrees and distinct variables per
clause. Choose `--feistel-ensemble partitioned` for the faster fixed-group
construction, which samples a different ensemble. Both modes still allocate
messages and mutable solver state.

Python exposes the same generators through
`Formula.feistel(k=3, alpha=4.0, n=1000, seed=7)`. Pass
`ensemble="partitioned"` to select the fixed-group ensemble; solve with
`backend="cpu"` or `backend="cpu-parallel"`.

See the [generator and limits](docs/guide/feistel.rst).

## Use the Rust crate

To use the library from another Rust project, add the Git dependency. This
package is not yet published on crates.io:

```toml
[dependencies]
bsp-rust = { git = "https://github.com/CarloNicolini/BSP.rs.git" }
```

```rust
use bsp_rust::{Formula, Options, Solver, Status};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let formula = Formula::random(3, 2.5, 128, 3)?;
    let mut solver = Solver::new(formula.clone(), Options::default())?;
    let result = solver.solve();
    if result.status == Status::Sat {
        let assignment = result.assignment.as_ref().unwrap();
        assert!(formula.is_satisfied(assignment));
        println!("{assignment:?}");
    } else {
        println!("Solver stopped: {}", result.status);
    }
    Ok(())
}
```

Use `Formula::from_dimacs(&text)` to load a CNF file. `cargo doc --open` builds
API documentation. Pin a Git revision in your dependency for reproducible runs.
The executable and the Python API run the same Rust solver.

## Python SDK: install the build tools

After the `bsp-rust` executable works, you can install the optional Python API.
The package builds a native extension from source. Install the tools for your
platform once; no separate Python, NumPy, or maturin installation is needed.

**macOS / Apple silicon** — install Apple's compiler tools, then wait for the
installer to finish before continuing:

```sh
xcode-select --install
```

**Linux / Ubuntu** — install a C/C++ compiler, linker, and Git:

```sh
sudo apt-get update
sudo apt-get install -y build-essential curl git
```

**Both platforms** — install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and [Rust](https://rustup.rs):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.local/bin/env"
. "$HOME/.cargo/env"
uv --version
rustc --version
```

Rust must be **1.88 or newer**. If an existing installation is older, run
`rustup update stable`. On macOS, Git is included in the Command Line Tools.

## Python SDK: clone and sync

```sh
git clone https://github.com/CarloNicolini/BSP.rs.git
cd BSP.rs
uv sync --python 3.12
```

This downloads Python 3.12 if needed, creates `.venv`, installs NumPy, and builds
`pybsp-rust` in release mode. The import name is `pybsp_rust`.
Run commands with `uv run`; activating the environment is unnecessary.

## Python SDK: solve a formula

```sh
uv run python examples/solve.py examples/3sat.cnf
```

Or use the library directly:

```python
from pybsp_rust import Formula, solve

formula = Formula.random(k=3, alpha=2.5, n=128, seed=3)
result = solve(formula=formula, seed=3)
print(result.status)
if result.assignment is not None:
    assert formula.is_satisfied(result.assignment)
    print(result.assignment)
```

The solver has no wall-time limit by default (`timeout_seconds=None`).

Pass algorithm and execution settings directly as keywords to `solve`, `Solver`,
or `solve_many`; the [parameter guide](docs/guide/parameters.rst) lists every
setting and default. For batches, use `solve_many(formulas=jobs, seeds=range(4),
threads=4)` to assign one seed per job. Reuse a settings dictionary with
`**settings`. Optional `Options` and `ExecutionOptions` objects remain supported. Mixing an object with direct keywords for the same settings raises
`TypeError`.

Save this as `quickstart.py` and run `uv run python quickstart.py`.
To load a file, use `Formula.from_dimacs(Path("instance.cnf").read_text())`
with `from pathlib import Path`.

## Python SDK: enable your GPU (optional)

### Apple silicon: Metal

The Command Line Tools installed above are sufficient; full Xcode and PyTorch
are unnecessary. From the repository root:

```sh
uv sync --reinstall-package pybsp-rust --config-settings 'build-args=--features extension-module,metal'
```

Select `backend="metal"` in the example below. Metal uses single precision;
small formulas can run faster on CPU.

### NVIDIA: CUDA on Linux

Install a compatible NVIDIA driver and the CUDA toolkit using
[NVIDIA's Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
Choose the toolkit for your distribution and GPU; install the **toolkit** as well
as the driver. Check both:

```sh
nvidia-smi
nvcc --version
```

`nvidia-smi` reports the driver's supported CUDA version; `nvcc` reports the
installed compiler. If `nvcc` is missing, the toolkit is missing or not on `PATH`.
For a toolkit installed at `/usr/local/cuda`:

```sh
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Find your GPU's compute capability:

```sh
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

Set `BSP_CUDA_ARCH` to that capability **without the dot**, then build:

```sh
export BSP_CUDA_ARCH=89  # example: 8.9; replace with your GPU's capability
uv sync --reinstall-package pybsp-rust --config-settings 'build-args=--features extension-module,cuda'
```

Common targets from [NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus):

| GPU | `BSP_CUDA_ARCH` | Toolkit requirement |
| --- | --- | --- |
| T4 | `75` | CUDA 12 or newer |
| A100 | `80` | CUDA 12 or newer |
| RTX 30 series / A10 | `86` | CUDA 12 or newer |
| RTX 40 series / L4 | `89` | CUDA 12 or newer |
| H100 / H200 | `90` | CUDA 12 or newer |
| B200 | `100` | CUDA 12.8 or newer |
| RTX 50 series | `120` | CUDA 12.8 or newer |

Blackwell compilation support starts with [CUDA 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).
Use a toolkit supported by your installed driver. Always set the architecture:
the project's legacy default is `61`, which CUDA 13 cannot compile.
If your `nvidia-smi` lacks `compute_cap`, look up the model in the linked table.

### Select and check the backend

```python
from pybsp_rust import Formula, Solver

formula = Formula.random(k=3, alpha=2.5, n=128, seed=3)
solver = Solver(
    formula=formula,
    seed=3,
    backend="metal",  # NVIDIA: "cuda-f32" or "cuda"
)
result = solver.solve()
print(result.status, result.execution)
if result.assignment is not None:
    assert formula.is_satisfied(result.assignment)
```

`cuda-f32` uses single precision; `cuda` uses double precision. Inspect execution
metadata for the actual backend, device, GPU sweep count, and numerical fallbacks.
GPU kernels accelerate SP; BSP decisions and WalkSAT still run on CPU.

After changing build features, repeat the corresponding `uv sync` command with
`--reinstall-package`. Use `uv run --no-sync` for a GPU-built environment to avoid
automatic synchronization rebuilding it with default CPU features, for example:

```sh
uv run --no-sync python quickstart.py
```

## Why can a GPU make BSP faster?

A SAT formula is a graph: variable nodes connect to clause nodes wherever a
literal occurs. Survey propagation (SP) repeatedly sends a small numerical
message along each connection. These messages estimate how strongly clauses
constrain variables. A sweep updates the whole graph; solving can require many
sweeps between decisions.

For example, random 3-SAT with 100,000 variables and four clauses per variable
has 400,000 clauses and 1.2 million literal occurrences. Most of the work in a
sweep consists of similar arithmetic on these connections. The scalar CPU
backend visits variables and clauses in sequence. Metal and CUDA distribute
that work across many GPU threads:

| Stage of an SP sweep | Work performed in parallel |
| --- | --- |
| Variable cavities | For each variable, combine incoming messages to compute outgoing probabilities, excluding the recipient clause. Variables can be processed independently using the previous sweep's messages. |
| Clause messages | Each clause combines its variables' cavity probabilities and updates its outgoing warnings. Different clauses can be processed independently. |
| Convergence check | Compute message changes across clauses, then reduce them to a global maximum. |
| Observables | Compute variable surveys and clause contributions, then reduce the quantities needed for BSP decisions. |

The stages run in order: clause updates wait for the variable stage, and the
next sweep waits for the current one. Within a stage, thousands of variables or
clauses can be in flight. Short clauses use specialized kernels; variable
updates also use cooperating SIMD groups or thread blocks for suitable degrees.
Prefix/suffix products reuse intermediate arithmetic when excluding each edge,
so the implementation avoids recomputing a whole neighborhood for every message.

Keeping data near the arithmetic matters too. The graph, messages, and scratch
arrays stay in GPU allocations across sweeps. Changes caused by CPU decisions
are queued as edits. This avoids uploading and downloading the entire graph on
every iteration, although inspecting arrays from Python requires host snapshots.
Apple's unified memory helps with access, but GPU submission and synchronization
still cost time. Batching sweeps amortizes those costs.

BSP's decisions, backtracking, unit propagation, and WalkSAT completion run on
the CPU. The GPU accelerates the repeated SP calculation rather than trying
independent full SAT searches for every thread. If SP accounts for 90% of a run
and becomes ten times faster, the whole run becomes about 5.3 times faster:
`1 / (0.10 + 0.90 / 10)`. This illustrative limit explains why kernel throughput
and time to a complete solution are different measurements.

Large formulas provide enough parallel work to amortize GPU overhead; small
ones can be faster on CPU. Precision matters: Metal uses `f32`, CUDA offers both
`f32` and `f64`, and CPU uses `f64`. Single precision reduces memory traffic and
can be much faster on GPUs with limited double-precision throughput, but it can
also change convergence and the search trajectory. Numerical recovery is
reported in the execution metadata.

Grounded speedups describe specific hardware, inputs, precision, and baselines.
They should not be treated as a fixed multiplier for every SAT problem.

## Versions and contribution workflow

The current version is **0.4.0**; **0.1.0** remains the initial tagged baseline.
See the [changelog](CHANGELOG.md),
[release policy](RELEASING.md), and [contributor/agent workflow](AGENTS.md).
Changes go through issues and PRs on topic branches. Version tags produce
validated, checksummed draft releases for separate publication review.

## Where to go next

- [Python API and algorithm documentation](docs/index.rst)
- [Contributing and validation](CONTRIBUTING.md)
- [Release policy](RELEASING.md)

Build the Python documentation locally:

```sh
uv sync --extra docs
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
uv run python -m http.server 8000 --directory docs/_build/html
```

Open <http://localhost:8000>. MathJax and prebuilt diagrams are included.

## License and attribution

GPL-2.0-or-later. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The algorithm and
baseline behavior follow [Raffaele Marino's original C++ BSP code](https://github.com/raffaelemarino/The-Backtracking-Survey-Propagation-Algorithm-code).
[Carlo Nicolini's pybsp](https://github.com/CarloNicolini/pybsp) provides the C++
solver and Python bindings used as the benchmark counterpart. Carlo Nicolini
maintains this repository and its Python interface.
