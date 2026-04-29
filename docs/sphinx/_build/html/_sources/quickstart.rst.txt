Quickstart
==========

Repository setup
----------------

1. Create and activate the virtual environment.
2. Install runtime dependencies:

   .. code-block:: powershell

      .\.venv\Scripts\python.exe -m pip install -r requirements.txt

3. Run tests:

   .. code-block:: powershell

      .\.venv\Scripts\python.exe -m pytest tests/ -q

4. Run a deterministic pod smoke simulation:

   .. code-block:: powershell

      .\.venv\Scripts\python.exe examples\play_edh_pod.py --max-turns 6 --seed 7 --model none

Build this Sphinx site
----------------------

Install docs dependencies:

.. code-block:: powershell

   .\.venv\Scripts\python.exe -m pip install -r docs\sphinx\requirements.txt

Build HTML docs:

.. code-block:: powershell

   .\.venv\Scripts\python.exe -m sphinx -b html docs\sphinx docs\sphinx\_build\html

The generated entry page is:

``docs/sphinx/_build/html/index.html``
