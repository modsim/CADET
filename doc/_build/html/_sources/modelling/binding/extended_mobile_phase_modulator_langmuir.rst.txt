.. _extended_mobile_phase_modulator_langmuir_model:

Extended Mobile Phase Modulator Langmuir
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model is an extension of the mobile phase modulator Langmuir model (see Section :ref:`multi_component_langmuir_model`), which allows linear binding of some selected components.
A modifier component :math:`c_{p,\mathrm{mod}}` is selected and the remaining components are divided into the index sets :math:`\mathcal{I}_{\mathrm{lin}}` and :math:`\mathcal{I}_{\mathrm{lang}}`.

.. math::

    \begin{aligned}
		\frac{\mathrm{d} q_i}{\mathrm{d} t} &= k_{a,i} e^{\gamma_i c_{p,\mathrm{mod}}} c_{p,i}\: q_{\text{max},i} \left( 1 - \sum_{j=1}^{N_{\text{comp}} - 1} \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} \: c_{p,\mathrm{mod}}^{\beta_i} \: q_i && i \in \mathcal{I}_{\mathrm{lang}}, \\
		\frac{\mathrm{d} q_i}{\mathrm{d} t} &= k_{a,i} c_{p,i} - k_{d,i} \: q_i && i \in \mathcal{I}_{\mathrm{lin}}.
	\end{aligned}

The modifier component is considered to be inert, therefore either

.. math::

	\frac{\mathrm{d} q_{\mathrm{mod}}}{\mathrm{d} t} = 0

is used if the modifier component has a bound state, or it can be used without a bound state.

The model can also be used without a modifier component.
In this case, the equations are given by

.. math::

    \begin{aligned}
		\frac{\mathrm{d} q_i}{\mathrm{d} t} &= k_{a,i} c_{p,i}\: q_{\text{max},i} \left( 1 - \sum_{j=1}^{N_{\text{comp}} - 1} \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} \: q_i && i \in \mathcal{I}_{\mathrm{lang}}, \\
		\frac{\mathrm{d} q_i}{\mathrm{d} t} &= k_{a,i} c_{p,i} - k_{d,i} \: q_i && i \in \mathcal{I}_{\mathrm{lin}}.
	\end{aligned}

For more information on model parameters required to define in CADET file format, see :ref:`extended_mobile_phase_modulator_langmuir_config`.
