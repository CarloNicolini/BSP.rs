.PHONY: help rust rust-cpu rust-metal rust-cuda python python-cpu python-metal python-cuda docs docs-serve clean

PORT ?= 8000
BACKEND ?= $(shell sh tools/detect_backend.sh)
DOC_FEATURES = extension-module$(if $(filter metal cuda,$(BACKEND)),$(comma)$(BACKEND))
comma := ,

help:
	@printf '%s\n' 'make rust                            Build the release executable (auto backend)' \
	  'make python                          Install the Python API (auto backend)' \
	  'make rust-cpu|rust-metal|rust-cuda   Build for a specific backend' \
	  'make python-cpu|python-metal|python-cuda  Install for a specific backend' \
	  'make docs                            Build the Sphinx HTML documentation' \
	  'make docs-serve [PORT=8000]          Build and serve the documentation' \
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
