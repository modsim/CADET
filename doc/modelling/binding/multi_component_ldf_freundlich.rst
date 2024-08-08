.. _multi_component_ldf_freundlich_model:

Multi Component Linear Driving Force Freundlich
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A multi-component extension to the classical Freundlich adsorption model.
A linear driving force approach is applied to obtain a kinetic form.

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} &= k_{\text{ldf},i}\: \left(q_i^* - q_i \right) & i = 0, \dots, N_{\text{comp}} - 1 \\
        q_i^* &= k_{f,i} c_{p,i} \left( \sum_j a_{ij} c_{p,j} + \tau \right)^{n_i - 1}.
    \end{aligned}

Here, :math:`\tau > 0` is a small constant that ensures numerical stability.
In a rapid-equilibrium setting with a diagonal matrix (i.e., :math:`a_{ii} = 1` and :math:`a_{ij} = 0` for :math:`j \neq i`), the traditional Freundlich isotherm is recovered.

For more information on model parameters required to define in CADET file format, see :ref:`multi_component_ldf_freundlich_config`.
