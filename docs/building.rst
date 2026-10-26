Documentation development
=========================

Strict HTML and executable examples
-----------------------------------

Install the real native extension and the docs extra from the repository root:

.. code-block:: sh

   python -m venv .venv-docs
   source .venv-docs/bin/activate
   python -m pip install -e '.[docs,test]'
   python -m sphinx -W --keep-going -b html docs docs/_build/html
   python -m sphinx -W --keep-going -b doctest docs docs/_build/doctest
   python -m http.server 8000 --directory docs/_build/html

Open ``http://localhost:8000``. The PyData theme supplies navigation, search,
light/dark mode, and API sidebars. The strict build resolves actual objects from
the installed ``pybsp_rust`` extension; it does not mock native classes.
Reinstall the editable package after modifying Rust doc comments, so the
introspected PyO3 docstrings reflect the current source.

CI builds strict HTML, executes the doctests, and uploads the HTML directory as
an artifact. It does not require a GPU or publish a website automatically.
GPU examples are marked as code examples, while portable small examples execute
as doctests. A separate linkcheck run can check outbound references; network
failures should not be confused with documentation syntax errors.

MathJax without an external service
-----------------------------------

``sphinx.ext.mathjax`` uses the pinned MathJax 3.2.2 ``tex-svg.js`` bundle under
``docs/_static/mathjax``. Its SVG font data is included, so readers do not need
a CDN or a TeX install. The package's Apache license and asset checksum are
stored beside it. The configured TeX packages are base, AMS, newcommand, and
noundefined; equations use ordinary Sphinx math directives and roles.

This avoids requiring both MathJax and KaTeX. Update the bundle and its checksum
together, preserve the upstream license, and verify rendering after an update.
The upstream sources are documented in the asset README.

TikZ source and SVG regeneration
--------------------------------

Diagrams have editable sources under ``docs/diagrams`` and checked-in SVGs under
``docs/_static/diagrams``. Ordinary HTML builds use the SVGs directly. To change
a diagram, install a TeX distribution with TikZ, standalone, and dvisvgm, then run:

.. code-block:: sh

   python tools/render_diagrams.py

The script compiles LaTeX to DVI in a temporary directory, converts glyphs to
SVG paths, adds accessible title/description metadata, and writes only the
finished SVGs. No shell escape is used. Review the diagrams at normal reading
width and in light/dark mode after regeneration.

All bipartite figures put clauses on the left and variables on the right.
White and grey variable fills encode True and False; hatching marks unfrozen
survey states. Positive literals use solid edges, negated literals dashed
edges. Arrow direction indicates message flow, not sign. Captions distinguish
illustrative assignments from probability distributions and measured data.

Updating the scientific narrative
---------------------------------

Check parameter claims and policy descriptions against the owning implementation:
``src/solver.rs`` for SP and BSP, ``src/walksat.rs`` for completion,
``src/formula.rs`` for normalization, and ``src/execution.rs`` for backends.
Keep the difference between ordinary marginals and cluster-forcing surveys
explicit. Cite the original arXiv records for algorithmic motivation, and label
implementation choices and empirical measurements as such.
