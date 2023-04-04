.. _saska_model:

Saska
~~~~~

In this binding model an additional quadratic term is added to the linear model :cite:`Saska1992`.
The quadratic term allows to take interactions of liquid phase components into account.

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} = H_i c_{p,i} + \sum_{j=0}^{N_{\text{comp}} - 1} k_{ij} c_{p,i} c_{p,j} - q_i && i = 0, \dots, N_{\text{comp}} - 1
    \end{aligned}


For more information on model parameters required to define in CADET file format, see :ref:`saska_config`.
