.. _installation:

Installation
============

A CADET installation consists of two parts: The CADET core simulator and a frontend.

Install CADET core simulator
----------------------------

The core simulator can be compiled from source, or you can download pre-built binaries.
If you want to extend or modify CADET (e.g., add a custom binding model), you will need to build CADET from source.

Install pre-built binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^
CADET can be installed via `conda <https://docs.anaconda.com/free/miniconda/>`_ from the ``conda-forge channel``.

``conda install -c conda-forge cadet``

Optionally, use `mamba <https://github.com/mamba-org/mamba>`_ which uses a faster dependency solver than ``conda``.

``mamba install -c conda-forge cadet``

Install from source
^^^^^^^^^^^^^^^^^^^

- :ref:`build_linux`
- :ref:`build_windows`
- :ref:`build_osx`

.. _cadet_process:

Install a frontend
------------------

CADET provides a Python API, called ``CADET-Python``, which can be used to set the model input according to the Interface specifications section.
Setting up the model using ``CADET-Python`` can become very tedious, especially for systems, and is outdated now that an actual frontend is available:

We recommend using the ``CADET-Process`` frontend, which facilitates modeling processes using an object oriented model builder.
This interface layer provides convenient access to all model parameters in the system.
It automatically checks validity of the parameter values and sets reasonable default values where possible.

Install CADET-Process
^^^^^^^^^^^^^^^^^^^^

To install ``CADET-Process``, open an `anaconda shell` or `mamba shell` and execute:

.. code-block:: bash

    pip install CADET-Process

If you want to use ``CADET-Python``, open an `anaconda shell` or `mamba shell` and execute:

.. code-block:: bash

    pip install CADET-Python 