# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyLensLib'
copyright = '2025, Massimo Meneghetti'
author = 'Massimo Meneghetti'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = [
    'matplotlib.backends.backend_qtagg',
    'lens_interact_qt',
    'g3read',
    'geopandas',
    'sphviewer',
    'lmfit',
    'skimage',
    'numba',
    'PyQt6',
    'uncertainties',
    'noise',
]

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = 'pyLensLib Documentation'
html_show_sphinx = False
html_logo = "_static/branding/pylenslib_logo_horizontal.svg"
html_favicon = "_static/branding/pylenslib_logo_mark.svg"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    # Update with your final public repository URL.
    "source_repository": "https://github.com/maxmen3/pyLensLib/",
    "source_branch": "master",
    "source_directory": "docs/",
    "prefers-color-scheme": "dark"
}

html_static_path = ['_static']
html_css_files = ['custom.css']

pygments_style = "friendly"
pygments_dark_style = "monokai"
