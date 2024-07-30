.. _cstr_model:
.. include:: ../equations.rst

Continuous stirred tank reactor model (CSTR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuous stirred tank reactor model is a basic building block in unit operation networks and often used to model holdup volume.
When combined with a binding model, it can be used to model batch uptake experiments.

Assuming that the fluid inside the tank is well-mixed and that the liquid volume :math:`V^{\ell}` can vary, the governing equations are given by

.. math::

    \begin{aligned}
        \frac{\mathrm{d}}{\mathrm{d}t} (V^{\ell} c^\ell_i) + V^{s} \sum_j d_j \sum_{m_{j,i}}^{N_{\text{bnd},j,i}}  \frac{\mathrm{d}}{\mathrm{d}t} (c^s_{j,i,m_{j,i}}) &= F_{\text{in}} c^\ell_{\text{in},i} - F_{\text{out}} c^\ell_i + V^{\ell} f_{\text{react},i}^l\left( c^\ell \right) \\
        &+V^{s}\sum_j d_j f_{\text{react},j,i}^s\left( c^\ell, c_j^s \right)
    \end{aligned}
where:

- |component_li|
- |component_sij| 
- |volume_liquid|
- |volume_solid|
- |flow_in_out|
- |reaction|
- |volume_fac_pat|

Depending on whether quasi-stationary or dynamic binding is used the binding behavior of :math:`c_i^{\ell}` and :math:`c^s_j` is described by 

.. math::

    \begin{aligned}
        \text{quasi-stationary: }& & 0 &= f_{\text{ads},j}\left( c_i^\ell, c^s_j\right), \\
        \text{dynamic: }& & \frac{\partial c^s_j}{\partial t} &= f_{\text{ads},j}\left( c_i^\ell, c^s_j\right) + f_{\text{react},j}^s\left( c_i^\ell, c_j^s \right),
    \end{aligned}

where :math:`f_{\text{ads},j}` is the adsorption rate in the solid phase of particle type :math:`j`.

The evolution of volume of the liquid volume

.. math::

    \begin{aligned}
       \frac{\mathrm{d}V^{\ell}}{\mathrm{d}t}= F_{\text{in}} - F_{\text{out}} - F_{\text{filter}}.
    \end{aligned}

Removing all bound states by setting :math:`N_{\text{bnd},j,i} = 0` for all components :math:`i` and particle types :math:`j`, and applying no binding model results in a simple tank.
The additional parameter :math:`F_{\text{filter}}`, which denotes the flow rate of pure liquid (without any components) out of the tank, can be used to model a filtering unit.

Note that it is the user’s duty to make sure that the volume of the CSTR does not fall below 0. If it does, the simulation may fail to run or may produce unreasonable (e.g., unphysical) results.

See :ref:`cstr_config`.


