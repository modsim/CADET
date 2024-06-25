.. _mmc_nfor_model:

HIC Water on Hydrophobic Surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model implements a slightly modified version of the HIC isotherm by Wang et al. based on their 2016 paper :cite:`Wang2016`.
A naive multicomponent version was added that reduces to the original formulation if only 1 binding species is present.

.. math::

    \begin{align}
		\beta &= \beta_0 e^{c_{p,0}\beta_1} \\
		\frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i} \left( 1 - \sum_j \frac{q_j}{q_{max,j}} \right)^{\nu_i} - k_{d,i} q_i  \left(\sum_j q_j \right)^{\nu_i \beta}
    \end{align}
   
- Component :math:`c_0` is assumed to be salt without a bound state.
- Multiple bound states are not supported.
- Components without bound state (i.e., salt and non-binding components) are supported.

For more information on model parameters required to define in CADET file format, see :ref:`hic_water_on_hydrophobic_surfaces_config`.
