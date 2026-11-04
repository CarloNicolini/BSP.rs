# Contributing

Start with [AGENTS.md](AGENTS.md) for branch, issue, commit, and PR rules.
Develop on a topic branch and submit a PR; `main` is protected. Describe public
API compatibility and link the relevant issue in every PR. Version selection,
release PRs, annotated tags, and draft review follow [RELEASING.md](RELEASING.md).

Use Rust 1.88 or newer. CI uses Python 3.12 and three jobs: Linux CPU,
Linux CUDA, and macOS CPU. Each runs the same Makefile entry point:

```sh
make ci BACKEND=cpu
# On a host with the corresponding toolkit:
make ci BACKEND=metal
make ci BACKEND=cuda
```

This builds the release CLI, runs Rust release tests, builds and installs a
Python wheel, and runs Python tests plus an example. Use `make test-rust` or
`make test-python` to rerun tests after building. CUDA CI checks compilation,
linking, and CPU execution; NVIDIA device tests remain ignored on hosted runners.
Linux CUDA wheel builds also need `patchelf` to bundle the CUDA runtime library.

Additional local checks, when relevant:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
make release-check
make docs BACKEND=cpu
uv run --no-sync python tools/check_doc_examples.py --rust
uv run --no-sync sphinx-build -W --keep-going -b doctest docs docs/_build/doctest
```

CI runs on pull requests and pushes to `main`, avoiding duplicate branch-push
and PR runs. Windows, alternate Python versions, standalone docs/policy jobs,
and repeated debug/release validation are outside the CI matrix.

Changes to SP must agree with the independent scalar recurrence, including zero
messages and subnormal products. Preserve fix/release reversibility and verify
every returned SAT assignment. Report nonconvergence separately from UNSAT.
Include formula, seed, options, compiler, target CPU, and build flags with a
performance claim. Keep benchmark setup and output serialization outside timers.
Do not compare complete runtimes without stating differences in search policy,
solution count, and postprocessing.
