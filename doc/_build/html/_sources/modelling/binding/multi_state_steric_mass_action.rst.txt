.. _multi_state_steric_mass_action_model:

Multi-State Steric Mass Action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multi-state steric mass action model adds :math:`M_i-1` *additional* bound states :math:`q_{i,j}` (:math:`j = 0, \dots, M_i - 1`) for each component :math:`i` to the steric mass action model (see Section :ref:`steric_mass_action_model`) and allows the exchange between the different bound states :math:`q_{i,0}, \dots, q_{i,M-1}` of each component.
In the multi-state SMA model a variable number of states of the bound molecule (e.g., different orientations on the surface, binding strength of tentacle adsorbers) is added which are more and more strongly bound, i.e.,

.. math::

    \begin{aligned}
        \nu_{i,j} \leq \nu_{i,j+1} \qquad i = 1, \dots, N_{\text{comp}} - 1, \quad j = 0,\dots, M_i - 1.
    \end{aligned}

The exchange between the different states of each component is allowed and, since the molecules can potentially bind in all states at the same binding site, competitive effects are present.

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_{i,j}}{\mathrm{d} t} =& \phantom{+} k_{a,i}^{(j)} c_{p,i} \left(\frac{\bar{q}_0}{q_{\text{ref}}}\right)^{\nu_{i,j}} - k_{d,i}^{(j)}\: q_{i,j}\: \left(\frac{c_{p,0}}{c_{\text{ref}}}\right)^{\nu_{i,j}} \\
        &- \sum_{\ell = 0}^{j-1} \underbrace{k^{(i)}_{j\ell}\: q_{i,j}\: \left(\frac{c_{p,0}}{c_{\text{ref}}}\right)^{\left(\nu_{i,j} - \nu_{i,\ell}\right)}}_{\text{to weak state}} - \sum_{\ell = j+1}^{M_i - 1} \underbrace{k^{(i)}_{j\ell}\: q_{i,j}\: \left(\frac{\bar{q}_0}{q_{\text{ref}}}\right)^{\left(\nu_{i,\ell} - \nu_{i,j}\right)}}_{\text{to strong state}} \\
        &+ \sum_{\ell = 0}^{j-1} \underbrace{k^{(i)}_{\ell j}\: q_{i,\ell}\: \left(\frac{\bar{q}_0}{q_{\text{ref}}}\right)^{\left(\nu_{i,j} - \nu_{i,\ell}\right)}}_{\text{from weak state}} + \sum_{\ell = j+1}^{M_i - 1} \underbrace{k^{(i)}_{\ell j}\: q_{i,\ell}\: \left(\frac{c_{p,0}}{c_{\text{ref}}}\right)^{\left(\nu_{i,\ell} - \nu_{i,j}\right)}}_{\text{from strong state}} & \begin{aligned}
        i &= 1, \dots, N_{\text{comp}} - 1, \\ j &= 0, \dots, M_i - 1, \end{aligned}
    \end{aligned}

where :math:`c_{p,0}` and :math:`q_0` denote the salt concentrations in the liquid and solid phase of the beads respectively.
The number of available salt ions :math:`\bar{q}_0` is given by

.. math::

    \begin{aligned}
        \bar{q}_0 = \Lambda - \sum_{j=1}^{N_{\text{comp}} - 1} \sum_{\ell=0}^{M_j - 1} \left( \nu_{j,\ell} + \sigma_{j,\ell} \right) q_{j,\ell}.
    \end{aligned}

A neutrality condition compensating for the missing equation for :math:`\frac{\mathrm{d} q_0}{\mathrm{d}t}` is required:

.. math::

    \begin{aligned}
        q_0 = \Lambda - \sum_{j=1}^{N_{\text{comp}} - 1} \sum_{\ell=0}^{M_j - 1} \nu_{j,\ell} q_{j,\ell}.
    \end{aligned}


The concept of reference concentrations (:math:`c_{\text{ref}}` and :math:`q_{\text{ref}}`) is explained in the respective paragraph in Section :ref:`reference_concentrations`.


For more information on model parameters required to define in CADET file format, see :ref:`multi_state_steric_mass_action_config`.
