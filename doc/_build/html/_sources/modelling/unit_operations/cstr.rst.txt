.. _cstr_model:

Continuous stirred tank reactor model (CSTR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuous stirred tank reactor model is a basic building block in unit operation networks and often used to model holdup volume.
When combined with a binding model, it can be used to model batch uptake experiments.

Assuming that the fluid inside the tank is well-mixed and that the volume can vary, the governing equations are given by

.. math::

    \begin{aligned}
        \frac{\mathrm{d}}{\mathrm{d}t} \left(\left[ c_i + \frac{1-\varepsilon}{\varepsilon} \sum_j d_j \sum_{m_{j,i}} c^s_{j,i,m_{j,i}} \right] V\right) &= F_{\text{in}} c_{\text{in},i} - F_{\text{out}} c_i + V f_{\text{react},i}^l\left( c \right) \\
    &+ V \frac{1-\varepsilon}{\varepsilon}\sum_j d_j f_{\text{react},j,i}^s\left( c, c_j^s \right),
    \end{aligned}

which balances the mass, the binding equation

.. math::

    \begin{aligned}
        \text{quasi-stationary: }& & 0 &= f_{\text{ads},j}\left( c, c^s_j\right), \\
        \text{dynamic: }& & \frac{\partial c^s_j}{\partial t} &= f_{\text{ads},j}\left( c, c^s_j\right) + f_{\text{react},j}^s\left( c, c_j^s \right),
    \end{aligned}

depending on whether quasi-stationary or dynamic binding is used, and the evolution of volume

.. math::

    \begin{aligned}
        \frac{\mathrm{d}V}{\mathrm{d}t} &= F_{\text{in}} - F_{\text{out}} - F_{\text{filter}}.
    \end{aligned}

The porosity :math:`\varepsilon` denotes the ratio of liquid phase volume to total tank volume.
Thus, setting :math:`\varepsilon = 1`, removing all bound states by setting :math:`N_{\text{bnd},j,i} = 0` for all components :math:`i` and particle types :math:`j`, and applying no binding model results in a simple tank.
The additional parameter :math:`F_{\text{filter}}`, which denotes the flow rate of pure liquid (without any components) out of the tank, can be used to model a filtering unit.

Note that it is the user’s duty to make sure that the volume of the CSTR does not fall below 0. If it does, the simulation may fail to run or may produce unreasonable (e.g., unphysical) results.

See :ref:`cstr_config`.


