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
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from datetime import date

# -- Project information -----------------------------------------------------
project = 'CADET'
copyright = f'2008-{date.today().year}'
author = 'CADET Authors'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'sphinx_sitemap',
    'sphinxcontrib.bibtex',
    'sphinx_multiversion',
    'breathe',
    'exhale',
]

# Bibliography
bibtex_bibfiles = ['literature.bib']

# Multiversion
smv_released_pattern = r'^refs/tags/.*$'    # Tags only
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'     # Include tags like "v2.1.1"
smv_branch_whitelist = r"^master$"          # Only use master branch
smv_remote_whitelist = r"^origin$"          # Use branches from remote origin
smv_outputdir_format = '{ref.name}'         # Use the branch/tag name

# Setup the breathe extension
breathe_projects = {
    "CADET": "./doxyoutput/xml"
}
breathe_default_project = "CADET"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "rootFileTitle":         "Library API",
    "doxygenStripFromPath":  "..",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    "INPUT = ../include",
    "verboseBuild": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Support for labelling and referencing equations
numfig = True
math_numfig = True
numfig_secnum_depth = 2
math_eqref_format = '{number}'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
html_theme_options = {
    'description': 'CADET - An open platform for integrated process modelling and simulation',
    # 'logo': 'cadet_logo.png',
    'sidebar_collapse': True,
    'fixed_sidebar': True,
    'show_powered_by': False,
}

html_favicon = '_static/cadet_icon.png'
html_title = 'CADET'
html_baseurl = 'https://cadet.github.io/master/'
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
        'versioning.html'
    ]
}

html_style = 'css/custom.css'

