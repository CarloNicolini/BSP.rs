# Further usage and development

Start with the [Python installation guide](README.md). GPU setup is in
[the acceleration guide](ACCELERATION.md).

## Quick start: command line

Install Rust 1.88 or newer with [rustup](https://rustup.rs), then clone and build:

```sh
git clone https://github.com/CarloNicolini/BSP.rs.git
cd BSP.rs
cargo build --release --locked
target/release/bsp-rust --version
target/release/bsp-rust examples/3sat.cnf
target/release/bsp-rust --generate 3 2.5 128 --seed 3 --timeout 10
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


## Inspect and control the solver

```python
from pybsp_rust import Formula, Solver

formula = Formula([[1, 2, 3], [-1, 2, 4], [1, -3, -4]], n=4)
solver = Solver(formula=formula)
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

CI runs on every push and pull request, and can also be started manually:

| Workflow | Platforms / features | Validation and delivery |
| --- | --- | --- |
| Build and test | Linux, macOS, Windows; no GPU features | Formatting, Clippy, debug/release/doc tests, CLI example, release binaries |
| Python tests and wheels | Linux, macOS, Windows; Python 3.12/3.13 | Installed-wheel tests, examples, wheels and source archives |
| CUDA and Metal builds | Linux x64 + CUDA 12.6 (sm61); macOS ARM64 + Metal | Clippy, debug/release/doc tests, CPU smoke test, GPU-enabled release binaries |
| Python documentation | Linux; installed native extension | Strict HTML build and executable doctests |
| Repository policy | Linux; Python 3.12 | Synchronized versions, changelog, and release-policy regression tests |
| Release artifacts | `v*` tags; all build/test workflows | Checksummed archives attached to a draft GitHub release |
| GPU hardware tests (manual) | Self-hosted NVIDIA/Linux or Apple silicon/macOS | Device tests, numerical comparisons, verified GPU solves |

CUDA is supported on Linux; Metal is supported on macOS. The default builds
on both platforms require neither toolkit. Standard hosted CI compiles and links
the GPU backends and runs tests that do not require a GPU. Device tests are
explicitly ignored there; a green build badge does not certify GPU execution.
CUDA binaries require a compatible CUDA 12 runtime on the destination machine.
Download binaries and Python packages from the corresponding run's **Artifacts**.

To run device tests, register a self-hosted runner with labels
`self-hosted,linux,x64,cuda` (driver, CUDA toolkit, and Rust installed) or
`self-hosted,macOS,ARM64,metal` (Apple GPU, Xcode Command Line Tools, and Rust).
Set `CUDA_HOME` and the toolkit library path on the CUDA runner if needed.
Then use **Actions → GPU hardware tests → Run workflow**, or:

```sh
gh workflow run gpu.yml -f backend=cuda -f cuda_arch=61
gh workflow run gpu.yml -f backend=metal
```

No self-hosted runners are required for the automatic workflows. Pushing a
version tag such as `v0.1.0` validates the annotated tag, versions, release notes,
and main ancestry, runs the platform and documentation workflows, and creates a
draft GitHub release only after every job passes. The release contains platform
archives and `SHA256SUMS`; review the draft in **Releases** before publishing.
This workflow does not publish packages to PyPI or crates.io.

Follow [AGENTS.md](AGENTS.md) for branch and PR rules and
[RELEASING.md](RELEASING.md) for version selection and release approval.

CI also tests Rust and the CLI on Linux, macOS, and Windows. It builds and installs
Python wheels in clean environments, runs the Python examples and tests, and
uploads binaries, wheels, and source archives. See [CONTRIBUTING.md](CONTRIBUTING.md)
for validation rules. The tests include scalar SP comparisons, subnormal
products, reversible masks, bounded WalkSAT, and real BSP backtracking.
