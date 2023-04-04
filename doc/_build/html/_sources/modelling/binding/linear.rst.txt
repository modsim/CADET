.. _linear_model:

Linear
~~~~~~

A linear binding model, which is often employed for low concentrations or in analytic settings :cite:`Guiochon2006`.

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} = k_{a,i} c_{p,i} - k_{d,i} q_i && i = 0, \dots, N_{\text{comp}} - 1.
    \end{aligned}


For more information on model parameters required to define in CADET file format, see :ref:`linear_config`.
