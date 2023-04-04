.. _multi_component_bi_langmuir_model:

Multi Component Bi-Langmuir
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multi component Bi-Langmuir model :cite:`Guiochon2006` adds :math:`M - 1` additional types of binding sites :math:`q_{i,j}` (:math:`0 \leq j \leq M - 1`) to the Langmuir model (see Section :ref:`multi_component_langmuir_model`) without allowing an exchange between the different sites :math:`q_{i,j}` and :math:`q_{i,k}` (:math:`k \neq j`).
Therefore, there are no competitivity effects between the different types of binding sites and they have independent capacities.

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_{i,j}}{\mathrm{d} t} &=  k_{a,i}^{(j)}\: c_{p,i}\: q_{\text{max},i}^{(j)} \left( 1 - \sum_{k=0}^{N_{\text{comp}} - 1} \frac{q_{k,j}}{q_{\text{max},k}^{(j)}}\right) - k_{d,i}^{(j)} q_{i,j} & i = 0, \dots, N_{\text{comp}} - 1, \: j = 0, \dots, M - 1.% (0 \leq i \leq N_{\text{comp}} - 1, \: 0 \leq j \leq M - 1).
    \end{aligned}

Note that all binding components must have exactly the same number of binding site types :math:`M \geq 1`.
See the Section :ref:`multi_component_langmuir_model`.

Originally, the Bi-Langmuir model is limited to two different binding site types.
Here, the model has been extended to arbitrary many binding site types.

For more information on model parameters required to define in CADET file format, see :ref:`multi_component_bi_langmuir_config`.
