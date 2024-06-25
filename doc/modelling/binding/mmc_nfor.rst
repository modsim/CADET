.. _hic_water_on_hydrophobic_surfaces_model:

MMC Nfor 2010
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model implements the mixed mode chromatography isotherm described by Nfor et al 2010 :cite:`Nfor2010`.

.. math::

		\begin{aligned}
		 \frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i} \widetilde{\gamma_i} \left[ \Lambda - \sum_j\left( \nu_j + \sigma_j \right) q_j \right]^{\nu_i}\left[ \Lambda - \sum_j\left( n_j + s_j \right) q_j \right]^{n_i} - k_{d,i} q_i c_{s}^{\nu_i} \\
		                               c_s &= c_{p,0}\\
		              \widetilde{\gamma_i} &= e^{K_{p,i}c_{p,i} + K_{s,i}c_{s}}
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


For more information on model parameters required to define in CADET file format, see :ref:`hic_water_on_hydrophobic_surfaces_config`.
