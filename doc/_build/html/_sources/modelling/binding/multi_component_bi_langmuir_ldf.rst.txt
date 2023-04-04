.. _multi_component_bi_langmuir_model_ldf:

Multi Component Bi-Langmuir LDF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This a linear driving force model variant of the :ref:`multi_component_bi_langmuir_model` model.
It is based on the equilibrium concentration :math:`q^*` for a given liquid phase concentration :math:`c` (see also :ref:`ldf_model`).

.. math::
    \begin{aligned}
        q_{i,j}^*=\frac{q_{m,i,j} k_{eq,i,j} c_i}{1 + \sum_{k=1}^{N_{comp}}{k_{eq,k,j} c_k}} && i = 0, \dots, N_{\text{comp}} - 1, \: j = 0, \dots, M - 1.% 	           (0 \leq i \leq N_{\text{comp}} - 1, \: 0 \leq j \leq M - 1). 
    \end{aligned}


For more information on model parameters required to define in CADET file format, see :ref:`multi_component_bi_langmuir_ldf_config`.
