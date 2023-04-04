.. _multi_component_langmuir_model_ldf:

Multi Component Langmuir LDF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This a linear driving force model variant of the :ref:`multi_component_langmuir_model` model.
It is based on the equilibrium concentration :math:`q^*` for a given liquid phase concentration :math:`c` (see also :ref:`ldf_model`).

.. math::

    \begin{aligned}
        q_i^*=\frac{q_{m,i} k_{eq,i} c_i}{1 + \sum_{j=1}^{n_{comp}}{k_{eq,j} c_j}} && i = 0, \dots, N_{\text{comp}} - 1.
    \end{aligned}

For more information on model parameters required to define in CADET file format, see :ref:`multi_component_langmuir_ldf_config`.
