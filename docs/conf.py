# Sphinx configuration for the ``fitsy`` Python bindings.
#
# Build locally with:
#
#     pip install -e '.[docs]'
#     maturin develop --features python
#     sphinx-build -b html docs docs/_build/html

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the hand-written ``fitsy.pyi`` stub discoverable by tooling.
# The compiled extension must be installed separately via
# ``maturin develop --features python``.
sys.path.insert(0, os.path.abspath(os.path.join("..", "stubs")))

# -- Project information ----------------------------------------------------

project = "fitsy"
author = "fitsy contributors"
copyright = f"{datetime.now():%Y}, {author}"

# Pull the version straight from the installed package when possible.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("fitsy")
except Exception:  # pragma: no cover - docs build fallback
    release = "0.0.0"

version = ".".join(release.split(".")[:2])

# -- General configuration --------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = True
nitpick_ignore = [
    # Suppress complaints about types we deliberately leave loose.
    ("py:class", "numpy.ndarray"),
    ("py:class", "os.PathLike"),
]
# Napoleon parses NumPy-style type fields and hands the raw tokens to
# Sphinx as class cross-references.  Several common tokens are not
# resolvable Python class names:
#
#   - ``optional`` / ``sequence`` -- NumPy modifier/type keywords
#   - ``{'value', ...}`` -- set-literal choices parsed token-by-token;
#     Sphinx 9 sees each space-separated fragment as a separate ref
#
# Use regex patterns to suppress the resulting false-positive warnings.
nitpick_ignore_regex = [
    ("py:class", r"optional"),  # NumPy "optional" modifier
    ("py:class", r"sequence"),  # NumPy "sequence" pseudo-type
    ("py:class", r"\{.*"),  # opening fragment of a set literal
    ("py:class", r"'[^']*'\}?"),  # quoted string token (with optional "}")
    ("py:class", r"'[^']*',?"),  # quoted string followed by comma
]

# -- HTML output ------------------------------------------------------------

html_theme = "furo"
html_title = f"fitsy {release}"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/icon.svg"
html_theme_options = {
    "sidebar_hide_name": True,  # logo already contains the project name
}
