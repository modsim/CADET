.. _unit_operation_models:

Unit operation models
=====================

A short comparison of the most prominent unit operation model features
is given in :numref:`table_features_unit_operations`.

.. _table_features_unit_operations:
.. list-table:: Supported features of the different unit operations models
   :widths: 30 14 14 14 14 14
   :header-rows: 1

   * - Unit operation model
     - Radial dispersion
     - Pore diffusion
     - Film diffusion
     - Particle geometries
     - Multiple particle types
   * - :ref:`general_rate_model_model`
     - ×
     - ✓
     - ✓
     - ✓
     - ✓
   * - :ref:`lumped_rate_model_with_pores_model`
     - ×
     - ×
     - ✓
     - ✓
     - ✓
   * - :ref:`lumped_rate_model_without_pores_model`
     - ×
     - ×
     - ×
     - ×
     - ×
   * - :ref:`2d_general_rate_model_model`
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - :ref:`cstr_model`
     - ×
     - ×
     - ×
     - ×
     - ✓
   * - :ref:`multi_channel_transport_model`
     - ×
     - ×
     - ×
     - ×
     - ×


Moreover, the pseudo unit operations :ref:`inlet_model`, and :ref:`outlet_model` act as sources and sinks for the system. 


.. toctree::
    :hidden:
    :glob:

    general_rate_model
    lumped_rate_model_without_pores
    lumped_rate_model_with_pores
    2d_general_rate_model
    multi_channel_transport_model
    cstr
    inlet
    outlet
