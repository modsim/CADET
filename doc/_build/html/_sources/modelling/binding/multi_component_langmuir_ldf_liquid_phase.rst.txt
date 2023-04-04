.. _multi_component_langmuir_model_ldf_liquid_phase:

Multi Component Langmuir LDF Liquid Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This a linear driving force model variant of the :ref:`multi_component_langmuir_model` model.
It is based on the equilibrium concentration :math:`c^*` for a given solid phase concentration :math:`q` (see also :ref:`ldf_model`).

.. math::

    \begin{aligned}
        c_i^*=\frac{q_{i}}{k_{eq,i} q_{m,i} \left(1 - \sum_{j=1}^{N_{\text{comp}}} \frac{q_j}{q_{m,j}}\right) } && i = 0, \dots, N_{\text{comp}} - 1.
    \end{aligned}


For more information on model parameters required to define in CADET file format, see :ref:`multi_component_langmuir_ldf_liquid_phase_config`.
