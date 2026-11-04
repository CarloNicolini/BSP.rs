.PHONY: help rust rust-cpu rust-metal rust-cuda python python-cpu python-metal python-cuda docs docs-serve clean test-rust python-wheel test-python ci release-check

.DEFAULT_GOAL := rust

PORT ?= 8000
BACKEND ?= $(shell sh tools/detect_backend.sh)
DOC_FEATURES = extension-module$(if $(filter metal cuda,$(BACKEND)),$(comma)$(BACKEND))
comma := ,
FEATURES = $(if $(filter metal cuda,$(BACKEND)),--features $(BACKEND))
PYTHON_FEATURES = extension-module$(if $(filter metal cuda,$(BACKEND)),$(comma)$(BACKEND))
PYTHON ?= 3.12
RELEASE_ARGS ?=

help:
	@printf '%s\n' 'make / make rust                     Build the release executable (auto backend)' \
	  'make python                          Install the Python API (auto backend)' \
	  'make rust-cpu|rust-metal|rust-cuda   Build for a specific backend' \
	  'make python-cpu|python-metal|python-cuda  Install for a specific backend' \
	  'make docs                            Build the Sphinx HTML documentation' \
	  'make docs-serve [PORT=8000]          Build and serve the documentation' \
	  'make ci BACKEND=cpu|metal|cuda      Build and test Rust and the Python wheel (Python 3.12)' \
	  'make test-rust / make test-python   Run the corresponding release tests' \
	  'make clean                           Remove build and documentation artifacts'

rust:
	@echo 'Selected build backend: $(BACKEND)'
	@$(MAKE) rust-$(BACKEND)

rust-cpu:
	cargo build --release --locked --bin bsp-rust

rust-metal:
	cargo build --release --locked --bin bsp-rust --features metal

rust-cuda:
	cargo build --release --locked --bin bsp-rust --features cuda

python:
	@echo 'Selected build backend: $(BACKEND)'
	@$(MAKE) python-$(BACKEND)

python-cpu:
	uv sync --reinstall-package pybsp-rust

python-metal:
	uv sync --reinstall-package pybsp-rust --config-settings 'build-args=--features extension-module,metal'

python-cuda:
	uv sync --reinstall-package pybsp-rust --config-settings 'build-args=--features extension-module,cuda'

docs:
	uv sync --extra docs --config-settings 'build-args=--features $(DOC_FEATURES)'
	uv run --no-sync sphinx-build -W --keep-going -b html docs docs/_build/html

docs-serve: docs
	uv run --no-sync python -m http.server $(PORT) --directory docs/_build/html

clean:
	cargo clean
	rm -rf docs/_build dist

# CI and local validation share these commands; acceleration is opt-in via BACKEND.
test-rust:
	cargo test --release --locked $(FEATURES)

test-python:
	uv run --no-sync pytest -q tests/python
	uv run --no-sync python examples/solve.py examples/3sat.cnf

python-wheel:
	uv sync --no-install-project --extra test --python $(PYTHON)
	uv run --no-sync maturin build --release --locked --features $(PYTHON_FEATURES) --out dist
	uv pip install --reinstall --no-deps dist/*.whl

ci:
	$(MAKE) rust BACKEND=$(BACKEND)
	$(MAKE) test-rust BACKEND=$(BACKEND)
	$(MAKE) python-wheel BACKEND=$(BACKEND)
	$(MAKE) test-python

release-check:
	uv run --no-project --python $(PYTHON) python tools/check_release.py $(RELEASE_ARGS)
	uv run --no-project --python $(PYTHON) python -m unittest discover -s tests/release_policy -q
