import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "VERSUS"
copyright = "2026, Nathan Findlay"
author = "Nathan Findlay"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autodoc_mock_imports = [
    "pyfftw",
    "pmesh",
    "mpsort",
    "pfft-python",
    "pyrecon",
]

master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"
