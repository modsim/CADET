.. _overview:

CADET Overview
==============

Performing a forward simulation comprises several steps:

.. toctree::
   :maxdepth: 1

For this example, we will use CADET-Python (see :ref:`cadet_python`)
CADET-Python is a file based interface for CADET.
CADET still must be downloaded and built from https://github.com/modsim/CADET

CADET-Python almost exactly maps to the documented CADET interface except that all dataset names are lowercase.
This simplifies using the interface.

This package includes the Cadet class and H5 class.
H5 can be used as a simple generic HDF5 interface.

As an example look at setting column porosity for column 1.
From the CADET manual the path for this is /input/model/unit_001/COL_POROSITY

In the python interface this becomes
``
sim = Cadet() sim.root.input.model.unit_001.col_porosity = 0.33
``
Once the simulation has been created it must be saved before it can be run
``
sim.filename = "/path/to/where/you/want/the/file.hdf5" sim.save()
``

Define unit operation parameters
--------------------------------
See also: :ref:`unit_operation_models`

Define adsorption and reaction paramters
----------------------------------------
See also: :ref:`binding_models`, and :ref:`FFAdsorption`

See also: :ref:`reaction_models`, and :ref:`FFReaction`


Setup connections and switches
------------------------------
See also: :ref:`simulation`, and :ref:`networks`,

See Tables :ref:`FFModelSystemConnections` and :ref:`FFModelConnectionSwitch`.


.. image:: sections.png


Configure solver
----------------


Call solver and read results
----------------------------

``
sim.load()
``

