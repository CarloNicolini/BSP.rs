SPHINXBUILD ?= python -m sphinx

.PHONY: html doctest diagrams clean
html:
	$(SPHINXBUILD) -W --keep-going -b html . _build/html
doctest:
	$(SPHINXBUILD) -W --keep-going -b doctest . _build/doctest
diagrams:
	python ../tools/render_diagrams.py
clean:
	$(SPHINXBUILD) -M clean . _build
