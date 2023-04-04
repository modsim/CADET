.. _steric_mass_action_model:

Steric Mass Action
~~~~~~~~~~~~~~~~~~

The steric mass action model takes charges of the molecules into account :cite:`Brooks1992` and is, thus, often used in ion-exchange chromatography.
Each component has a characteristic charge :math:`\nu` that determines the number of available binding sites :math:`\Lambda` (ionic capacity) used up by a molecule.
Due to the molecule’s shape, some additional binding sites (steric shielding factor :math:`\sigma`) may be shielded from other molecules and are not available for binding.

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} = k_{a,i} c_{p,i}\left( \frac{\bar{q}_0 }{q_{\text{ref}}} \right)^{\nu_i} - k_{d,i}\: q_i\: \left(\frac{c_{p,0}}{c_{\text{ref}}}\right)^{\nu_i} && i = 1, \dots, N_{\text{comp}} - 1,
    \end{aligned}

where :math:`c_{p,0}` and :math:`q_0` denote the salt concentrations in the liquid and solid phase of the beads, respectively.
The number of free binding sites

.. math::

    \begin{aligned}
        \bar{q}_0 = \Lambda - \sum_{j=1}^{N_{\text{comp}} - 1} \left( \nu_j + \sigma_j \right) q_j = q_0 - \sum_{j=1}^{N_{\text{comp}} - 1} \sigma_j q_j
    \end{aligned}

is calculated from the number of bound counter ions :math:`q_0` by taking steric shielding into account.
In turn, the number of bound counter ions :math:`q_0` (electro-neutrality condition) is given by

.. math::

    \begin{aligned}
        q_0 = \Lambda - \sum_{j=1}^{N_{\text{comp}} - 1} \nu_j q_j,
    \end{aligned}

which also compensates for the missing equation for :math:`\frac{\mathrm{d} q_0}{\mathrm{d}t}`.

The concept of reference concentrations (:math:`c_{\text{ref}}` and :math:`q_{\text{ref}}`) is explained in the respective paragraph in Section :ref:`reference_concentrations`.


For more information on model parameters required to define in CADET file format, see :ref:`steric_mass_action_config`.
