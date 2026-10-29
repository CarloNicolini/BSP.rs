# Contributing

Start with [AGENTS.md](AGENTS.md) for branch, issue, commit, and PR rules.
Develop on a topic branch and submit a PR; `main` is protected. Describe public
API compatibility and link the relevant issue in every PR. Version selection,
release PRs, annotated tags, and draft review follow [RELEASING.md](RELEASING.md).

Use Rust 1.88 or newer and Python 3.12 or newer. From a fresh clone:

```sh
python3 tools/check_release.py
python3 -m unittest discover -s tests/release_policy -v
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
cargo test --release --locked
cargo test --doc --locked
uv sync --extra test
uv run pytest -q tests/python
uv run python examples/solve.py examples/3sat.cnf
uv sync --extra docs --extra test
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
uv run sphinx-build -W --keep-going -b doctest docs docs/_build/doctest
```

Changes to SP must agree with the independent scalar recurrence, including zero
messages and subnormal products. Preserve fix/release reversibility and verify
every returned SAT assignment. Report nonconvergence separately from UNSAT.
Include formula, seed, options, compiler, target CPU, and build flags with a
performance claim. Keep benchmark setup and output serialization outside timers.
Do not compare complete runtimes without stating differences in search policy,
solution count, and postprocessing.
