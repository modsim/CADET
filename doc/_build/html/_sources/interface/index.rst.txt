.. _file_format:

Interface specifications
========================

The CADET framework is designed to work on a file format structured into groups and datasets.
This concept may be implemented by different file formats.
At the moment, CADET natively supports HDF5 and XML as file formats.
The choice is not limited to those two formats but can be extended as needed.
In this section the general layout and structure of the file format is described.

.. topic::  File format versions

    The file format may change and evolve over time as new features are added to the simulator.
    This manual describes the most recent file format version that is also set as default value in ``/meta/FILE_FORMAT`` (see Tab. :ref:`FFMeta`).
    The simulator assumes that the input file uses the most recent format version and does not update old files to the current standard.


.. toctree::
   :maxdepth: 3

   introduction
   input_group
   output_group
   meta_group



