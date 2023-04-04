.. _multi_component_langmuir_model:

Multi Component Langmuir
~~~~~~~~~~~~~~~~~~~~~~~~

The Langmuir binding model includes a saturation term and takes into account the capacity of the resin :cite:`Langmuir1916,Guiochon2006`.
All components compete for the same binding sites.

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} = k_{a,i}\: c_{p,i}\: q_{\text{max},i} \left( 1 - \sum_{j=0}^{N_{\text{comp}} - 1} \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} q_i && i = 0, \dots, N_{\text{comp}} - 1.
    \end{aligned}


For more information on model parameters required to define in CADET file format, see :ref:`multi_component_langmuir_config`.
