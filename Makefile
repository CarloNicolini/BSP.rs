.PHONY: help rust rust-metal rust-cuda python python-metal python-cuda docs docs-serve

PORT ?= 8000

help:
	@printf '%s\n' 'make rust [rust-metal|rust-cuda]     Build the release executable' \
	  'make python [python-metal|python-cuda] Install the Python API with uv' \
	  'make docs                            Build the Sphinx HTML documentation' \
	  'make docs-serve [PORT=8000]          Build and serve the documentation'

rust:
	cargo build --release --locked --bin bsp-rust

rust-metal:
	cargo build --release --locked --bin bsp-rust --features metal

rust-cuda:
	cargo build --release --locked --bin bsp-rust --features cuda

python:
	uv sync --reinstall-package pybsp-rust

python-metal:
	uv sync --reinstall-package pybsp-rust --config-settings 'build-args=--features extension-module,metal'

python-cuda:
	uv sync --reinstall-package pybsp-rust --config-settings 'build-args=--features extension-module,cuda'

docs:
	uv sync --extra docs
	uv run sphinx-build -W --keep-going -b html docs docs/_build/html

docs-serve: docs
	uv run python -m http.server $(PORT) --directory docs/_build/html
