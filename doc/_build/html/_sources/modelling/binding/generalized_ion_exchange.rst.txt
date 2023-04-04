.. _generalized_ion_exchange_model:

Generalized Ion Exchange
~~~~~~~~~~~~~~~~~~~~~~~~

The generalized ion exchange model is based on the steric mass action model :cite:`Huuk2017,Mollerup2008`.
In addition to the first component :math:`c_{p,0}`, which represents salt, the second component :math:`c_{p,1}` represents another non-binding modifier (e.g., pH).
In comparison to the SMA model, the characteristic charge :math:`\nu` and the adsorption and desorption rate constants are modified:

.. math::

    \begin{aligned}
        q_0 &= \Lambda - \sum_{j=2}^{N_{\text{comp}} - 1} \nu_j(c_{p,1}) q_j \\
         \frac{\partial q_i}{\partial t} &= k_{a,i}(c_{p,0},c_{p,1}) \: c_{p,i} \: \left( \frac{\bar{q}_0 }{q_{\text{ref}}} \right)^{\nu_i(c_{p,1})} - k_{d,i}(c_{p,0},c_{p,1}) \: q_i \: \left( \frac{c_{p,0}}{c_{\text{ref}}} \right)^{\nu_i(c_{p,1})} & &i = 2, \dots, N_{\text{comp}} - 1,
    \end{aligned}

where

.. math::

    \begin{aligned}
        \bar{q}_0 &= \Lambda - \sum_{j=2}^{N_{\text{comp}} - 1} \left( \nu_j(c_{p,1}) + \sigma_j \right) q_j = q_0 - \sum_{j=2}^{N_{\text{comp}} - 1} \sigma_j q_j
    \end{aligned}

The dependence of the parameters on :math:`c_{p,0}` and :math:`c_{p,1}` is given for :math:`i = 2, \dots, N_{\text{comp}} - 1` by

.. math::

    \begin{aligned}
        \nu_i(c_{p,1}) &= \nu_{i,\mathrm{base}} + c_{p,1} \nu_{i,\mathrm{lin}} + c_{p,1}^2 \nu_{i,\mathrm{quad}} \\
        k_{a,i}\left(c_{p,0}, c_{p,1}\right) &= k_{a,i,\mathrm{base}} \exp\left(k_{a,i,\mathrm{lin}} c_{p,1} + k_{a,i,\mathrm{quad}} c_{p,1}^2 + k_{a,i,\mathrm{salt}} \frac{c_{p,0}}{c_{\text{ref}}} + k_{a,i,\mathrm{prot}} c_{p,i}\right) \\
        k_{d,i}\left(c_{p,0}, c_{p,1}\right) &= k_{d,i,\mathrm{base}} \exp\left(k_{d,i,\mathrm{lin}} c_{p,1} + k_{d,i,\mathrm{quad}} c_{p,1}^2 + k_{d,i,\mathrm{salt}} \frac{c_{p,0}}{c_{\text{ref}}} + k_{d,i,\mathrm{prot}} c_{p,i}\right)
    \end{aligned}


The concept of reference concentrations (:math:`c_{\text{ref}}` and :math:`q_{\text{ref}}`) is explained in the respective paragraph in Section :ref:`reference_concentrations`.


For more information on model parameters required to define in CADET file format, see :ref:`generalized_ion_exchange_config`.
