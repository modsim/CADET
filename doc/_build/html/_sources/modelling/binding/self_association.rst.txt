.. _self_association_model:

Self Association
~~~~~~~~~~~~~~~~

This binding model is similar to the steric mass action model (see Section :ref:`steric_mass_action_model`) but is also capable of describing dimerization :cite:`Mollerup2008,Westerberg2012`.
The dimerization, which is the immobilization of protein at some already bound protein, is also termed “self-association”.
It is modeled by adding a quadratic (in :math:`c_{p,i}`) term to the adsorption part of the equation.

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} &= c_{p,i}\left( \frac{\bar{q}_0}{q_{\text{ref}}} \right)^{\nu_i} \left[ k_{a,i,1} + k_{a,i,2} c_{p,i} \right] - k_{d,i}\: q_i\: \left(\frac{c_{p,0}}{c_{\text{ref}}}\right)^{\nu_i} && i = 1, \dots, N_{\text{comp}} - 1, \\
        q_0 &= \Lambda - \sum_{j=1}^{N_{\text{comp}} - 1} \nu_j q_j,
    \end{aligned}

where the number of available binding sites is given by

.. math::

    \begin{aligned}
        \bar{q}_0 = \Lambda - \sum_{j=1}^{N_{\text{comp}} - 1} \left( \nu_j + \sigma_j \right) q_j = q_0 - \sum_{j=1}^{N_{\text{comp}} - 1} \sigma_j q_j.
    \end{aligned}

The concept of reference concentrations (:math:`c_{\text{ref}}` and :math:`q_{\text{ref}}`) is explained in the respective paragraph in Section :ref:`reference_concentrations`.


For more information on model parameters required to define in CADET file format, see :ref:`self_association_config`.
