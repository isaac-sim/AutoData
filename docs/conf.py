# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# TODO: Get this programmatically from setup.py (AUTODATA_VERSION_NUMBER).
AUTODATA_VERSION_NUMBER = "0.1"

# -- Project information -----------------------------------------------------

project = "Autodata"
copyright = "2026, NVIDIA"
author = "NVIDIA"
released = False  # Indicates if this is a public or internal version of the repo.

# -- General configuration ---------------------------------------------------

sys.path.append(os.path.abspath("_ext"))

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_multiversion",
    "autodata_doc_tools",
]

# Render `.. todo::` directives in the built docs. The doc templates use them as
# placeholders, so keep this enabled until the docs are fully written.
todo_include_todos = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # Re-add numpy/torch inventories when API (autodoc) pages that reference them land:
    # "numpy": ("https://numpy.org/doc/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store", "venv_docs"]

# Be picky about missing references
nitpicky = True  # warns on broken references
nitpick_ignore: list[str] = []  # can exclude known bad refs

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "nvidia_sphinx_theme"
html_title = f"Autodata {AUTODATA_VERSION_NUMBER}"
html_show_sphinx = False
html_theme_options = {
    "copyright_override": {"start": 2026},
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "footer_links": {},
    "github_url": "https://github.com/isaac-sim/AutoData",
    "show_nav_level": 1,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Versioning
smv_remote_whitelist = r"^.*$"
smv_branch_whitelist = r"^(main|release/.*)$"
smv_tag_whitelist = r"^v.*$"
html_sidebars = {"**": ["versioning.html", "sidebar-nav-bs"]}

# Linkcheck
linkcheck_ignore: list[str] = []

#####################################
#  Macros dependent on release state
#####################################

autodata_docs_config = {
    "released": released,
    "internal_git_url": "git@github.com:isaac-sim/AutoData.git",
    "external_git_url": "UNDECIDED",
    "internal_code_link_base_url": "https://github.com/isaac-sim/AutoData/blob/main",
    "external_code_link_base_url": "UNDECIDED",
}
