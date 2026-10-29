# Changelog

All notable public changes are recorded here. Versions follow the compatibility
policy in [RELEASING.md](RELEASING.md).

## [Unreleased]

## [0.2.0]

Draft release; publication awaits maintainer review.

### Python SDK

- Expose every solver setting as an explicit keyword on `Solver`, `solve`, and
  `solve_many`; serialize direct settings without constructing configuration
  dataclasses. Existing configuration-object calls remain supported, with lazy
  compatibility snapshots for keyword callers.
- Accept `seeds` for per-job batch seeds alongside shared algorithm keywords.
- Document each public parameter and native adapter with NumPy-style contracts;
  bind documentation examples against the installed Python API and compile the
  documented Rust examples in CI.

### Documentation

- Reorganize the README around installation and standalone CLI use; move the
  Apple benchmark and detailed usage into linked guides and add the Colab T4
  setup and GPU verification guide.

### Memory

- Reuse active/free frozen-message slots for CPU and Metal cavity norms, and
  allocate Metal's generic prefix buffer only for clause widths that need it.
  Paired experiments report allocation savings with workload-dependent timing
  costs; these changes do not establish billion-variable capacity or a speedup.

### Compatibility and limits

- Restore CLI executable permissions at both supported artifact layouts before
  packaging release archives, including the nested `target/release` path.
- Existing `Options` and `ExecutionOptions` calls remain supported. New code can
  use named keywords, plain settings dictionaries, and per-job `seeds`.
- The memory reductions preserve the tested per-backend trajectories, with
  workload-dependent timing costs. No general speedup, billion-variable support,
  implicit topology, or quantized production backend is claimed.
- Hosted CUDA CI builds and links the backend; NVIDIA hardware execution was not
  rerun for this release. Metal device tests are run explicitly on supported Macs.
- `v0.1.0` remains the immutable initial baseline; it is not moved to this release.

## [0.1.0]

Initial versioned release. Publication date will be recorded after draft review.

### Added

- Native Rust BSP and WalkSAT solver with verified SAT assignments, random
  k-SAT generation, DIMACS input, JSON CLI, and inspectable NumPy Python API.
- Direct Python algorithm and execution keywords on `solve`, `Solver`, and
  `solve_many`, such as `solve(formula=formula, seed=3, backend="cpu")`.
  Existing `Options` and `ExecutionOptions` calls remain supported; ambiguous
  mixed settings and unknown keywords raise `TypeError`.
- Optional multicore CPU, Apple Metal, and NVIDIA CUDA execution, bounded batch
  solving, cancellation, cooperative timeouts, and execution profiling.
- Standalone CLI installation and quickstart, including `bsp-rust --version`.
- PR/issue templates, contributor/agent workflow, synchronized-version checks,
  and validated, checksummed draft releases from immutable annotated tags.

### Limitations

- BSP is heuristic and cannot prove UNSAT. Failed searches are not UNSAT claims.
- GPU single precision can change numerical behavior and search trajectories.
  Hosted CUDA CI validates compilation/linking, not NVIDIA device execution.
- Packages are distributed as GitHub release assets; no automated PyPI or
  crates.io publication is configured.
