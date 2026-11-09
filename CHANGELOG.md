# Changelog

All notable public changes are recorded here. Versions follow the compatibility
policy in [RELEASING.md](RELEASING.md).

## [Unreleased]

- Support both implicit Feistel ensembles on Metal without materializing the
  packed factor graph. This compatible backend addition requires a minor
  release under the 0.x policy.

- Add sequential CLI solving of multiple DIMACS paths and directories, with a
  terminal progress bar, per-input JSON results, and aggregate status. This
  compatible CLI addition requires a minor release under the 0.x policy.

- Fix automatic report-name collisions, layered CLI options, profile-write result
  loss, and late execution-option validation. Existing explicit output paths keep
  their overwrite behavior; partial options files now preserve earlier settings.
- Stream DIMACS files through the CLI and Python `from_file` without retaining
  the input text and all unpacked clauses. Rust adds buffered-reader constructors,
  a compatible new API requiring a minor increment under the 0.x policy.

- Expose both implicit Feistel ensembles and their topology metadata in the
  Python `Formula` API. This is compatible new functionality; a future 0.x
  release containing it requires a minor version increment.

## [0.5.0]

- Add experimental CPU and CPU-parallel `--feistel K ALPHA N` and configuration
  `feistel`. The default fluctuating ensemble targets ordinary random K-SAT.
  `--feistel-ensemble partitioned` selects the different fixed-group ensemble
  and emits a warning. Rust exposes both through `Formula::feistel_fluctuating`
  and `Formula::feistel`, with versioned u64 topology queries.
- Add streaming `Formula::write_dimacs` and Feistel-only CLI topology metadata.
  Solver state remains linear and retains u32 capacity limits.
- Add same-seed DIMACS replay with `--preserve-literal-order`.
- Reduce CPU memory use during explicit generation, Feistel traversal, and
  residual WalkSAT, and reuse scratch space on CPU-parallel.

- Breaking: the default WalkSAT budget now scales with residual variables and
  restarts. `max_flips=0` selects automatic mode instead of forbidding flips;
  positive explicit budgets retain their meaning.

## [0.4.1]

### CUDA

- Construct packed transport windows with linear bucket counters instead of
  stable sorts, preserving all mapping entries and stable tie ordering. Reuse
  the temporary inverse permutation for q slots to reduce peak host allocation.
- Prefetch eight observable block records before consuming them in the original
  serial Kahan order. T4 paired trials show a modest improvement to the complete
  observable pass; the exact-order regression covers f32/f64 cancellation and
  partial prefetch groups.
- Run packed-layout regression tests on CPU-only CI, including randomized
  multi-window permutations, inverse mappings, degree boundaries, and sparse
  clause headers.

### Compatibility and limits

- Compatible internal optimizations with no public API, default, output, device
  buffer, or numerical summation-order changes.
- Performance measurements use a Colab Xeon host and Tesla T4 at alpha=4.2.
  They cover layout setup and observable passes, rather than complete solves;
  no other GPU architecture or f64 performance improvement is claimed.

## [0.4.0] - 2026-11-04

### CLI

- Parse flags with Clap instead of a handmade argv loop. Existing flag names,
  positional formula path, exit codes, and JSON stdout/file behavior stay the
  same. Help text now follows Clap formatting.
- Add `--config PATH` to load run parameters from TOML, YAML, or JSON via serde.
  Absent fields keep the previous layer. Merge order is defaults, then
  `--config`, then `--options` (Options JSON), then explicit CLI flags.
  Examples: `examples/solve.toml`, `examples/generate.yaml`.
- The CLI writes the JSON result to a file. Stdout then prints one JSON
  object with `status` and the main counters, plus `output` for the file path.
  The default name contains the main run variables and a UTC timestamp, for
  example `output_k3_alpha2.5_n128_seed1_r0.9_cpu_20261103T142033Z.json`.
  `--output PATH` selects the file. Assignments and the step list stay in the file.
- `--progress` shows one stdout line with timers for formula setup, BSP, and
  WalkSAT. The timer does not run inside a device sweep. `--no-progress` hides
  that line. The default is on when stdout is a terminal, and off otherwise.
- `--help` prints the default value of each option. The JSON field `residual`
  remains the largest undamped message change in the last sweep.

### Build and documentation

- Plain `make` builds the release `bsp-rust` executable (same as `make rust`).
  `make clean` removes Cargo outputs, Sphinx `docs/_build`, and `dist`.
  The README leads with the executable and Makefile workflow; the Python SDK
  follows.

### Metal

- Pack edge messages into source-ordered buffers so that the SP sweep reads
  them with local gathers. A sweep at N=10^6, alpha=4 takes 8.6 ms instead of
  52 ms on an M1 Pro, with bit-identical messages. Device memory grows by
  about 190 MB at that size.

### CUDA

- Pack f32 survey messages into windowed buffers and run several small
  variables in each warp. On a Tesla T4, one sweep at N=10^6, K=3, alpha=4,
  seed 3 takes 3.10 ms instead of 21.83 ms (median of 9 trials). Messages stay
  bit-identical with the previous kernel. Device buffers grow by 118 MB at
  that size. The f64 path packs messages but does not use the warp kernel.
  The window size and the two-record warp are tuned on the T4.

### Compatibility and limits

- Clap changes help formatting only; flag names, exit codes, and JSON stdout/file
  behavior stay the same. `--options` JSON loading remains supported.
- Version jumps from `0.2.0` to `0.4.0` by maintainer choice; there is no
  published `0.3.0` tag.
- Hosted CUDA CI builds and links the backend; NVIDIA hardware execution for the
  packed f32 kernel was measured on a Tesla T4. Metal timings are from an M1 Pro.


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
