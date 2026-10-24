# Contributing

Use Rust 1.88 or newer and Python 3.12 or newer. From a fresh clone:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
cargo test --release --locked
cargo test --doc --locked
uv sync --extra test
uv run pytest -q tests/python
uv run python examples/solve.py examples/3sat.cnf
```

Changes to SP must agree with the independent scalar recurrence, including zero
messages and subnormal products. Preserve fix/release reversibility and verify
every returned SAT assignment. Report nonconvergence separately from UNSAT.
Include formula, seed, options, compiler, target CPU, and build flags with a
performance claim. Keep benchmark setup and output serialization outside timers.
Do not compare complete runtimes without stating differences in search policy,
solution count, and postprocessing.
