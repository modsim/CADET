.. _multi_component_spreading_model:

Multi Component Spreading
~~~~~~~~~~~~~~~~~~~~~~~~~

The multi component spreading model adds a second bound state :math:`q_{i,2}` to the Langmuir model (see Section :ref:`multi_component_langmuir_model`) and allows the exchange between the two bound states :math:`q_{i,1}` and :math:`q_{i,2}`.
In the spreading model a second state of the bound molecule (e.g., a different orientation on the surface or a different folding state) is added.
The exchange of molecules between the two states is allowed and, since the molecules can potentially bind in both states at the same binding site, competitivity effects are present.
This is different to the Bi-Langmuir model in which another type of binding sites is added and no exchange between the different bound states is considered (see Section :ref:`multi_component_bi_langmuir_model`).
For all components :math:`i = 0, \dots, N_{\text{comp}} - 1` the equations are given by

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_{i,1}}{\mathrm{d} t} &= \left( k_a^A\: c_{p,i} - k_{12} q_{i,1} \right) q_{\text{max},i}^A \left( 1 - \sum_{j=0}^{N_{\text{comp}} - 1} \frac{q_j^A}{q_{\text{max},j}^A} - \sum_{j=0}^{N_{\text{comp}} - 1} \frac{q_j^B}{q_{\text{max},j}^B} \right) - k_d^A q_{i,1} + k_{21} q_{i,2}, \\
        \frac{\mathrm{d} q_{i,2}}{\mathrm{d} t} &= \left( k_a^B\: c_{p,i} + k_{12} q_{i,1} \right) q_{\text{max},i}^A \left( 1 - \sum_{j=0}^{N_{\text{comp}} - 1} \frac{q_j^A}{q_{\text{max},j}^A} - \sum_{j=0}^{N_{\text{comp}} - 1} \frac{q_j^B}{q_{\text{max},j}^B} \right) - \left( k_d^B + k_{21} \right) q_{i,2}.
    \end{aligned}


For more information on model parameters required to define in CADET file format, see :ref:`multi_component_spreading_config`.
