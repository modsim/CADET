.. _bi_steric_mass_action_model:

Bi Steric Mass Action
~~~~~~~~~~~~~~~~~~~~~

Similar to the Bi-Langmuir model (see Section :ref:`multi_component_bi_langmuir_model`), the Bi-SMA model adds :math:`M - 1` *additional* types of binding sites :math:`q_{i,j}` (:math:`0 \leq j \leq M - 1`) to the SMA model (see Section :ref:`steric_mass_action_model`) without allowing an exchange between the different sites :math:`q_{i,j}` and :math:`q_{i,k}` (:math:`k \neq j`).
Therefore, there are no competitivity effects between the two types of binding sites and they have independent capacities.

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_{i,j}}{\mathrm{d} t} &= k_{a,i,j} c_{p,i} \left(\frac{\bar{q}_{0,j}}{q_{\text{ref},j}} \right)^{\nu_{i,j}} - k_{d,i,j}\: q_{i,j}\: \left(\frac{c_{p,0}}{c_{\text{ref},j}}\right)^{\nu_{i,j}} & i = 1, \dots, N_{\text{comp}} - 1, \quad j = 0, \dots, M - 1,
    \end{aligned}

where :math:`c_{p,0}` and :math:`q_{0,j}` (:math:`0 \leq j \leq M - 1`) denote the salt concentrations in the liquid and solid phases of the beads respectively.
The number of available salt ions :math:`\bar{q}_{0,j}` for each binding site type :math:`0 \leq j \leq M - 1` is given by

.. math::

    \begin{aligned}
        \bar{q}_{0,j} &= \Lambda_j - \sum_{k=1}^{N_{\text{comp}} - 1} \left( \nu_{k,j} + \sigma_{k,j} \right) q_{k,j}.
    \end{aligned}

Electro-neutrality conditions compensating for the missing equations for :math:`\frac{\mathrm{d} q_{0,j}}{\mathrm{d}t}` are required:

.. math::

    \begin{aligned}
        q_{0,j} &= \Lambda_j - \sum_{k=1}^{N_{\text{comp}} - 1} \nu_{k,j} q_{k,j} & j = 0, \dots, M - 1.
    \end{aligned}

Note that all binding components must have exactly the same number of binding site types :math:`M \geq 1`.

The reference concentrations :math:`c_{\text{ref},j}` and :math:`q_{\text{ref},j}` can be specified for each binding site type :math:`0 \leq j \leq M - 1`.
The concept of reference concentrations is explained in the respective paragraph in Section :ref:`reference_concentrations`.

Originally, the Bi-SMA model is limited to two different binding site types.
Here, the model has been extended to arbitrary many binding site types.


For more information on model parameters required to define in CADET file format, see :ref:`bi_steric_mass_action_config`.
