.. _cadet_core_architecture

Software Architecture
=====================

This section gives a general overview of the object-oriented c++ implementation of CADET-Core and is partially based on the Software Architecture chapter 3.2 in `Samuel Lewekes PhD thesis <https://publications.rwth-aachen.de/record/840314>`_.

The CADET-Core software is designed to be modular, extendable, maintainable and computationally efficient and robust.
To promote these goals, CADET-Core is implemented in C++, a highly performant, object-oriented language that supports cross-platform compatibility.

Features and Capabilities
^^^^^^^^^^^^^^^^^^^^^^^^^

CADET-Core 

- Builds on Windows, Linux, MacOs

- Allows to easily add new frontends as the core simulator is implemented in the library ``libcadet``, which can be called from different frontends.
  Currently, CADET-Core features the command line interface ``cadet-cli`` and CADET-Python, which includes a ``Cadet`` class that serves as a generic HDF5 frontend and calls the ``cadet-cli`` interface.
  The CADET-Process frontend is a separate software that wraps CADET-Python and supports additional functionality, as described in the `CADET-Process documentation <https://cadet-process.readthedocs.io/en/latest/index.html>`_. 
  The Matlab frontend was deprecated in `commit 4b34e0d5 <https://github.com/cadet/CADET-Core/commits/4b34e0d5fcabee2ff84ff422acac75a6982d6df7/>`_ due to its maintenance overhead.

- Can compute parameter sensitivities via a custom Algorithmic Differentiation (AD) implementation. The AD infrastructure allows for easy extension of new models to support parameter sensitivities, as described in the AD section of the :ref:`model_expansion` chapter.

- Allows arbitrary systems of unit operations, including cycles.

- Implements a method of lines approach to solve the model equations: Custom implementations of model equations and, if required, spatial discretization, is complemented by the well-established and publicly available time integration software library SUNDIALS.
  The IDAS solver of the `SUNDIALS software <https://sundials.readthedocs.io/en/latest/index.html>`_, implements a variable order (up to fifth order) BDF with adaptive time stepping and has proven to be stable and robust.
  Additionally, IDAS supports the forward computation of parameter sensitivities, i.e. solves the sensitivity system as `described in their documentation <https://sundials.readthedocs.io/en/latest/idas/Mathematics_link.html#forward-sensitivity-analysis>`_.


Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

Classes and realationships
Include figure 3.4 and/or 3.6 from Sams Diss

