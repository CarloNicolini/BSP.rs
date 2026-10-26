# BSP.rs

[![Build and test](https://github.com/CarloNicolini/BSP.rs/actions/workflows/ci.yml/badge.svg)](https://github.com/CarloNicolini/BSP.rs/actions/workflows/ci.yml)

BSP.rs solves CNF SAT problems with a native Rust implementation of Backtracking
Survey Propagation (BSP), followed by native WalkSAT completion. It supports
random k-SAT generation, DIMACS files, a Rust library, a JSON command-line tool,
and Python bindings with NumPy state access. It requires no C++ solver or
external SAT executable.

Large graphs can use a multicore `f64` backend, Apple Metal `f32`, or NVIDIA CUDA
in `f64` or `f32`. Both the CLI and Python API expose hardware selection, worker
limits, batch solving, and phase profiling. See [CPU and GPU execution](ACCELERATION.md)
for builds, numerical behavior, and benchmark commands.

BSP is a heuristic. Every returned SAT assignment is checked against the formula.
A failed run or a contradictory residual does not prove the input UNSAT.

## Python documentation

The Sphinx documentation includes a full Python API reference, parameter and
hardware guides, and research-level chapters developing belief propagation,
survey propagation, and BSP, with arXiv references and TikZ/SVG factor graphs.
Build the PyData-themed site locally:

```sh
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open `http://localhost:8000`. MathJax is bundled locally, and prebuilt SVG diagrams
are included; HTML builds require neither a TeX installation nor an external CDN.
See [documentation development](docs/building.rst) for executable examples and
TikZ regeneration. CI also provides a `bsp-python-documentation` HTML artifact.

## Quick start: command line

Install Rust 1.88 or newer with [rustup](https://rustup.rs), then clone and build:

```sh
git clone https://github.com/CarloNicolini/BSP.rs.git
cd BSP.rs
cargo build --release --locked
cargo run --release -- examples/3sat.cnf
cargo run --release -- --generate 3 2.5 128 --seed 3 --timeout 10
```

The executable is `target/release/bsp-rust` (`bsp-rust.exe` on Windows). Install it
on your PATH with `cargo install --path . --locked`. Run `bsp-rust --help` for all
options. GitHub Actions also builds downloadable CLI artifacts on Linux, macOS,
and Windows. Artifact binaries target the architecture of their build runner.

Input uses DIMACS, for example:

```text
p cnf 3 4
1 2 3 0
-1 2 -3 0
1 -2 3 0
-1 -2 -3 0
```

Each output is a JSON result with `status`, `assignment`, `sp_sweeps`, `flips`,
and step history. Assignments are ordered by variable, with 0=false and 1=true.
Partial assignments use -1 for free variables. Exit codes are 0 for SAT, 1 for a
heuristic failure or a limit, and 2 for invalid input.

Generate an instance for later use:

```sh
bsp-rust --generate 3 4.15 400 --seed 3 --write-formula instance.cnf
bsp-rust --timeout 10 --max-flips 1000000 instance.cnf
```

Random generation uses distinct variables per clause and `floor(alpha*N)`
clauses. General DIMACS formulas may have mixed clause widths. Construction
removes duplicate literals and tautological clauses, while keeping empty clauses.

## Quick start: Apple silicon GPU (Metal/MPS)

Install Rust 1.88 or newer and Xcode Command Line Tools (`xcode-select --install`).
The Apple GPU backend is named `metal`: it uses custom Metal compute kernels
directly, with no PyTorch or MPS framework dependency. Full Xcode is unnecessary.

From the repository root, compile:

```sh
cargo build --release --locked --features metal
```

Solve a DIMACS file or generate an instance:

```sh
target/release/bsp-rust --backend metal examples/3sat.cnf
target/release/bsp-rust --backend metal --generate 3 2.5 128 --seed 3 --timeout 10
```

Compare scalar CPU and Apple GPU throughput on the same seeded formula:

```sh
target/release/bsp-rust --backend cpu --generate 3 4.2 100000 --seed 7 --damping 0.2 --sweeps 300
target/release/bsp-rust --backend metal --generate 3 4.2 100000 --seed 7 --damping 0.2 --sweeps 300
```

Compare the JSON `seconds` values. Metal uses `f32`; scalar CPU uses `f64`.
These commands measure sweeps, including the final host snapshot, and exclude
construction and backend setup. Small complete solves can be faster on CPU.

## Quick start: NVIDIA GPU (CUDA, Linux)

Install Rust 1.88 or newer, an NVIDIA driver, and a CUDA toolkit containing
`nvcc`, runtime headers, and `libcudart`. On the tested Ubuntu machine:

```sh
sudo apt install nvidia-cuda-toolkit
nvidia-smi
nvcc --version
```

Build from the repository root for your GPU's compute capability. This example
targets the GTX 1060 (6.1); use `86` for a GPU with compute capability 8.6:

```sh
BSP_CUDA_ARCH=61 cargo build --release --locked --features cuda
```

The GTX 1060 requires CUDA 12.9 or earlier for compilation. The tested toolkit
is CUDA 12.0. The CUDA version shown by `nvidia-smi` is the driver's capability;
check `nvcc --version` for the installed compiler.

Solve with double precision (`cuda`) or single precision (`cuda-f32`):

```sh
target/release/bsp-rust --backend cuda examples/3sat.cnf
target/release/bsp-rust --backend cuda-f32 --generate 3 2.5 128 --seed 3 --timeout 10
```

Compare sweep throughput with the same input and damping:

```sh
target/release/bsp-rust --backend cpu --generate 3 4.2 100000 --seed 7 --damping 0.2 --sweeps 300
target/release/bsp-rust --backend cuda --generate 3 4.2 100000 --seed 7 --damping 0.2 --sweeps 300
target/release/bsp-rust --backend cuda-f32 --generate 3 4.2 100000 --seed 7 --damping 0.2 --sweeps 300
```

Single precision can follow a different search trajectory. Every returned SAT
assignment is verified on CPU. Execution metadata reports the actual backend,
precision, device, and any numerical fallback. See [CUDA execution](CUDA.md) for
toolkit configuration and [measured performance](PERFORMANCE.md) for results.

### Python with an Apple or NVIDIA GPU

Use Python 3.12 or newer, create a virtual environment, then build the extension
with the feature for your platform:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install maturin numpy
```

On Apple silicon:

```sh
maturin develop --release --features extension-module,metal
```

On NVIDIA/Linux:

```sh
python -m pip install patchelf
BSP_CUDA_ARCH=61 maturin develop --release --features extension-module,cuda
```

Select the compiled backend explicitly in Python:

```python
from pybsp_rust import ExecutionOptions, Formula, Options, Solver

formula = Formula.random(3, 2.5, 128, seed=3)
solver = Solver(
    formula,
    Options(seed=3, timeout_seconds=10),
    execution=ExecutionOptions(backend="metal"),  # Linux: "cuda" or "cuda-f32"
)
result = solver.solve()
print(result.status, result.execution)
```

## Quick start: Rust library

Inside this repository, run the complete example:

```sh
cargo run --release --example solve_ksat -- examples/3sat.cnf
cargo run --release --example solve_ksat
```

To use it from another Rust project, add the Git dependency. This package is not
yet published on crates.io:

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

## Quick start: Python

Use Python 3.12 or newer. A source installation also needs Rust and the platform's
linker/build tools (Xcode Command Line Tools on macOS, or Visual Studio Build
Tools on Windows):

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
python examples/solve.py examples/3sat.cnf
```

Or install directly from GitHub:

```sh
python -m pip install "git+https://github.com/CarloNicolini/BSP.rs.git"
```

The Python distribution is `pybsp-rust`; the import name is `pybsp_rust`.
It is not yet published on PyPI. CI builds wheels for Python 3.12+ and source
archives as downloadable Actions artifacts. To use a downloaded wheel, run
`python -m pip install path/to/pybsp_rust-*.whl` with a wheel matching your OS and
CPU architecture.

```python
from pathlib import Path
from pybsp_rust import Formula, Options, solve

formula = Formula.from_dimacs(Path("examples/3sat.cnf").read_text())
result = solve(formula, Options(seed=3, timeout_seconds=10))
print(result.status)
if result.assignment is not None:
    assert formula.is_satisfied(result.assignment)
    print(result.assignment)

random_formula = Formula.random(k=3, alpha=2.5, n=128, seed=3)
random_result = solve(random_formula)
```

Each result includes an owned `int8` NumPy assignment, partial assignment, status,
SP/WalkSAT counters, and step history. Python and Rust run the same native core.
The package does not launch a subprocess or write implicit solver files.

## Inspect and control the solver

```python
from pybsp_rust import Formula, Solver

formula = Formula([[1, 2, 3], [-1, 2, 4], [1, -3, -4]], n=4)
solver = Solver(formula)
token = solver.cancellation_token
report = solver.converge()
messages = solver.messages       # owned float64 snapshot
surveys = solver.surveys         # N x 3: true, false, unfrozen
solver.fix(0, True)              # zero-based variable ID
residual = solver.residual_formula()
solver.release(0)
solver.set_messages(messages)
solver.sweep()                   # one synchronous SP sweep
step = solver.advance()          # one fixed point and BSP action
result = solver.solve()          # continue from current state
```

Call `converge()` before manual `decimate()` calls. `fix()` does not propagate
implicitly; use `propagate()` when needed. Unit-propagated assignments stay fixed.
Message order follows the normalized `Formula.clauses`. Array snapshots own their
storage; use `set_messages()` to update native state.

Long native calls release the GIL. Independent solvers can run in Python threads.
Get the token before starting a native call, then call `token.cancel()` from
another thread. Cancellation is permanent for that solver. Timeout and
cancellation checks are cooperative: once per SP sweep and every 256 WalkSAT
flips. Python signal handling resumes when the native call returns.

## Algorithm, optimization, and limits

The solver provides zero-temperature SP, certitude-ranked batch decimation,
stored-certitude backtracking, unit propagation, and min-break WalkSAT.
Research extensions in the original C++ code, including ThermoSP, 1RSB/2RSB,
oracle policies, whitening, and multiple-solution collection, are outside this
repository. The original C++ solver remains at [pybsp](https://github.com/CarloNicolini/pybsp).

Storage uses packed 32-bit literals/indices, contiguous CSR adjacency, and 64-bit
masks. Prefix/suffix products make clause updates linear in clause width and
handle zero messages without dividing by `1-eta`. Partial selection ranks free
variables; an indexed min-heap ranks fixed decisions in logarithmic time. Active
degrees and the sole free literal of a unit clause have constant-time cached
lookups. The [complexity audit](COMPLEXITY.md) covers remaining scans and measured
tradeoffs. Probabilities use `f64`; release builds use optimization
level 3, ThinLTO, and one code-generation unit. The core forbids unsafe Rust.

Use `RUSTFLAGS='-C target-cpu=native' cargo build --release --locked` for a local
CPU-specific binary on macOS/Linux. Portable builds remain the default. The
[performance report](PERFORMANCE.md) records a 1.23-2.18x SP-kernel improvement on
four matched ARM64 workloads. It does not establish a general complete-solver
speedup or zero cache misses. The comparison harness requires a separate C++
checkout; normal builds and tests do not.

Statuses distinguish SAT, residual contradiction, SP nonconvergence, negative
complexity, numerical failure, step/flip limits, cancellation, and timeout.
Extreme degrees and incompatible hard warnings can produce zero normalizers.
Reproducibility requires the same formula, options, dependency lock, compiler,
CPU target, and operation order.

## Development and CI

```sh
cargo test --locked
cargo test --release --locked
cargo clippy --all-targets -- -D warnings
cargo fmt --check
uv sync --extra test
uv run pytest -q tests/python
```

CI tests Rust and the CLI on Linux, macOS, and Windows. It builds and installs
Python wheels in clean environments, runs the Python examples and tests, and
uploads binaries, wheels, and source archives. See [CONTRIBUTING.md](CONTRIBUTING.md)
for validation rules. The tests include scalar SP comparisons, subnormal
products, reversible masks, bounded WalkSAT, and real BSP backtracking.

## License and attribution

GPL-2.0-or-later. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The algorithm and
baseline behavior follow Raffaele Marino's original C++ BSP code. Carlo Nicolini
maintains this repository and the Python interface.
