.. _mobile_phase_modulator_langmuir_model:

Mobile Phase Modulator Langmuir
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model is a modified Langmuir model (see Section :ref:`multi_component_langmuir_model`) which can be used to describe hydrophobic interaction chromatography :cite:`Melander1989,Karlsson2004`.
A modulator component (termed “salt”, :math:`c_{p,0}` and :math:`q_0`) influences ad- and desorption processes:

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} = k_{a,i} e^{\gamma_i c_{p,0}} c_{p,i}\: q_{\text{max},i} \left( 1 - \sum_{j=1}^{N_{\text{comp}} - 1} \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} \: c_{p,0}^{\beta_i} \: q_i && i = 1, \dots, N_{\text{comp}} - 1.
    \end{aligned}

where :math:`c_{p,0}` and :math:`q_0` denote the salt concentrations in the liquid and solid phase of the beads respectively.
Salt is considered to be inert, therefore either

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_0}{\mathrm{d} t} = 0
    \end{aligned}

is used if salt has one bound state, or salt can be used without a bound state.
The parameter :math:`\gamma` describes the hydrophobicity and :math:`\beta` the ion-exchange characteristics.


For more information on model parameters required to define in CADET file format, see :ref:`mobile_phase_modulator_langmuir_config`.
