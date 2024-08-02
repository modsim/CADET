.. _multi_component_sips_model:

Multi Component Sips
~~~~~~~~~~~~~~~~~~~~~~~~

The Sips binding model is a combination of the :ref:`Freundlich<freundlich_ldf_model>` and the :ref:`Langmuir adsorption model<multi_component_langmuir_model>`.

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} = k_{a,i}\: \left( \frac{c_{p,i}}{ c_{\text{ref}} }\right)^{n_i}\: q_{\text{max},i} \left( 1 - \sum_{j=0}^{N_{\text{comp}} - 1} \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} \left( \frac{q_i}{q_{\text{ref}}} \right) && i = 0, \dots, N_{\text{comp}} - 1.
    \end{aligned}


For more information on model parameters required to define in CADET file format, see :ref:`multi_component_sips_config`.
