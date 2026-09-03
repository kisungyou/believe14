"""Sphinx configuration for the believe14 documentation."""

from __future__ import annotations

import os
import sys
import tempfile
from importlib.metadata import version as distribution_version
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_ROOT.parent
sys.path.insert(0, str(DOCS_ROOT / "_ext"))
os.environ.setdefault("MPLBACKEND", "Agg")
_runtime_root = Path(tempfile.gettempdir()) / "believe14-docs-runtime"
_runtime_root.mkdir(parents=True, exist_ok=True)
for _directory in ("ipython", "jupyter-config", "jupyter-runtime"):
    (_runtime_root / _directory).mkdir(exist_ok=True)
os.environ.setdefault("IPYTHONDIR", str(_runtime_root / "ipython"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(_runtime_root / "jupyter-config"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(_runtime_root / "jupyter-runtime"))

project = "believe14"
author = "Kisung You"
copyright = "2026, Kisung You"
release = distribution_version("believe14")
version = release

extensions = [
    "catalog",
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]
source_suffix = {".md": "myst-nb", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "validation/ledger-template.md",
]

nitpicky = True
nitpick_ignore_regex = [
    (
        "py:class",
        r"(?:array(?:-like)?|ndarray array|shape|n_.*|default(?:=.*)?|0\..*)",
    )
]
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = True

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

nb_execution_mode = "force"
nb_execution_timeout = 180
nb_execution_raise_on_error = True
nb_merge_streams = True

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

html_theme = "pydata_sphinx_theme"
html_title = f"believe14 {release}"
html_baseurl = "https://kisungyou.github.io/believe14/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "header_links_before_dropdown": 5,
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/kisungyou/believe14",
            "icon": "fa-brands fa-github",
        }
    ],
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_sidebars = {
    "index": [],
    "getting_started/index": [],
    "**": ["search-field", "sidebar-nav-bs"],
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


def _require_installed_wheel() -> None:
    """Fail release documentation if believe14 resolves to the source tree."""

    if os.environ.get("BELIEVE14_DOCS_REQUIRE_WHEEL") != "1":
        return
    import believe14

    package_path = Path(believe14.__file__).resolve()
    source_path = (PROJECT_ROOT / "src").resolve()
    if package_path.is_relative_to(source_path):
        raise RuntimeError(
            "BELIEVE14_DOCS_REQUIRE_WHEEL=1, but believe14 was imported from "
            f"the source tree: {package_path}"
        )
    if "site-packages" not in package_path.parts:
        raise RuntimeError(
            "Release documentation must import believe14 from an installed wheel; "
            f"resolved {package_path}"
        )


_require_installed_wheel()
