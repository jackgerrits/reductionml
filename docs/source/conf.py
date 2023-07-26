# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
from reductionml import __version__
from pathlib import Path


# -- Project information -----------------------------------------------------

project = 'ReductionML'
copyright = '2023, Jack Gerrits'
author = 'Jack Gerrits'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

sys.path.append(str((Path(__file__).parent / "_ext").resolve()))

print(sys.path)
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_nb",
    "sphinx_copybutton",
    "reduction_info"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme = "furo"
html_static_path = []

html_theme_options = {
    "source_repository": "https://github.com/jackgerrits/reductionml",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

html_title = f"📚 ReductionML {release}"

nb_execution_raise_on_error = True
nb_execution_timeout = 60
nb_execution_mode = "cache"
nitpicky = True
myst_heading_anchors = 3