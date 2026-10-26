"""Sphinx configuration; importing the real extension is intentional."""
from importlib.metadata import version as package_version

project = "BSP.rs"
author = "BSP.rs contributors"
copyright = "2026, BSP.rs contributors"
release = package_version("pybsp-rust")
version = release
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_design",
]
html_theme = "pydata_sphinx_theme"
html_title = "BSP.rs Python documentation"
html_static_path = ["_static"]
html_css_files = ["bsp.css"]
html_theme_options = {
    "logo": {"text": "BSP.rs"},
    "github_url": "https://github.com/CarloNicolini/BSP.rs",
    "navigation_depth": 3,
    "show_toc_level": 2,
    "header_links_before_dropdown": 3,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_context = {
    "github_user": "CarloNicolini",
    "github_repo": "BSP.rs",
    "github_version": "main",
    "doc_path": "docs",
}
exclude_patterns = ["_build", "README.md", "diagrams"]
nitpicky = True
nitpick_ignore = [
    ("py:class", "pathlib.Path"), ("py:class", "numpy.ndarray"),
    ("py:class", "typing.Sequence"), ("py:class", "typing.Iterable"),
]
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"
autodoc_inherit_docstrings = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "sequence": "typing.Sequence",
    "sequences": "typing.Sequence",
    "iterable": "typing.Iterable",
    "array_like": "numpy.ndarray",
}
autosummary_generate = True
numfig = True
math_numfig = True
math_number_all = False

# Pinned, locally shipped bundle: equations need no CDN, TeX install, or fonts.
# Use MathJax 3's SVG renderer, whose font data is included in tex-svg.js.
mathjax_path = "mathjax/tex-svg.js"
mathjax3_config = {
    "tex": {"packages": ["base", "ams", "newcommand", "noundefined"]},
    "svg": {"fontCache": "global"},
    "options": {"enableMenu": False},
}
doctest_global_setup = "from pybsp_rust import Formula, Options, ExecutionOptions, Solver, solve, solve_many"
doctest_test_doctest_blocks = "default"
