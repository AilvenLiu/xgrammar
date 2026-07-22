# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime

import tomli

# -- General configuration ------------------------------------------------

os.environ["XGRAMMAR_BUILD_DOCS"] = "1"
sys.path.insert(0, os.path.abspath("../python"))
sys.path.insert(0, os.path.abspath("../"))

# Load version from pyproject.toml
with open("../pyproject.toml", "rb") as f:
    pyproject_data = tomli.load(f)
__version__ = pyproject_data["project"]["version"]

project = "XGrammar"
author = "XGrammar Contributors"
copyright = f"2024-{datetime.now().year}, {author}"

version = __version__
release = __version__

# -- Extensions and extension configurations --------------------------------

extensions = [
    "myst_parser",
    "nbsphinx",
    "autodocsumm",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.httpdomain",
    "sphinxcontrib.mermaid",
]

nbsphinx_allow_errors = True
nbsphinx_execute = "never"

autosectionlabel_prefix_document = True
nbsphinx_allow_directives = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "linkify",
    "substitution",
]

myst_heading_anchors = 5
myst_ref_domains = ["std", "py"]
myst_all_links_external = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

autodoc_mock_imports = ["torch", "safetensors", "transformers", "tvm_ffi"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
    "member-order": "bysource",
}

autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_json = True
autodoc_pydantic_settings_show_json = False

# -- Other Options --------------------------------------------------------

templates_path = []

redirects = {}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

suppress_warnings = ["misc.highlighting_failure", "autodoc.mocked_object"]

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"

html_title = f"XGrammar {__version__}"

html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "img/logo.png",
    "dark_logo": "img/logo_dark.svg",
    "source_repository": "https://github.com/mlc-ai/xgrammar",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/mlc-ai/xgrammar",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0"'
                ' viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0'
                " 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38"
                " 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01"
                " 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95"
                " 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18"
                " 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44"
                " 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65"
                " 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013"
                ' 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        }
    ],
}
