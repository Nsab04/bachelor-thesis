Getting Started
===============

This section covers the fastest way to install and run ``pyLensLib``.

Requirements
------------

- Python 3.10+
- ``pip`` and ``venv`` (or Conda)

Install from local source
-------------------------

From the project root:

.. code-block:: bash

   cd /path/to/pyLensLib
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .

Install with Conda (example)
----------------------------

.. code-block:: bash

   conda create -n pylenslib python=3.10 -y
   conda activate pylenslib
   cd /path/to/pyLensLib
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .

Quick import check
------------------

.. code-block:: bash

   python -c "import pyLensLib; print('pyLensLib import OK')"

Build the documentation locally
-------------------------------

.. code-block:: bash

   pip install sphinx furo
   PYTHONPATH=$(pwd) make -C docs clean html

Then open:

``docs/_build/html/index.html``
