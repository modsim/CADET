.. _lumped_rate_model_with_pores_model:

Lumped rate model with pores (LRMP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lumped rate model with pores :cite:`Guiochon2006,Felinger2004` deviates from the general rate model (see Section :ref:`general_rate_model_model`) by neglecting pore diffusion.
The particle phase :math:`c^p_j` is still there, but no mass transfer happens except for binding and film diffusion.
Hence, the model equations are given by

.. math::

    \begin{aligned}
        \frac{\partial c^l_i}{\partial t} &= -u \frac{\partial c^l_i}{\partial z} + D_{\text{ax},i} \frac{\partial^2 c^l_i}{\partial z^2} - \frac{1}{\beta_c} \sum_{j} d_j \frac{3}{r_{p,j}} k_{f,j,i}\left[ c^l_i - c^p_{j,i} \right] + f_{\text{react},i}^l\left(c^l\right),
    \end{aligned}

.. math::

    \begin{aligned}
        \frac{\partial c^p_{j,i}}{\partial t} + \frac{1 - \varepsilon_{p,j}}{F_{\text{acc},j,i} \varepsilon_{p,j}} \frac{\partial}{\partial t} \sum_{m_{j,i}} c^s_{j,i,m_{j,i}} &= \frac{3}{F_{\text{acc},j,i} \varepsilon_{p,j} r_{p,j}}k_{f,j,i}\left[ c^l_i - c^p_{j,i} \right] \\
        &+ f_{\text{react},j,i}^p\left( c_j^p, c_j^s \right) + \frac{1 - \varepsilon_{p,j}}{F_{\text{acc},j,i} \varepsilon_{p,j}} f_{\text{react},j,i}^s\left( c_j^p, c_j^s \right)
    \end{aligned}

with the same meanings of variables and parameters as in the general rate model.
The equations are complemented by Danckwerts boundary conditions :cite:`Danckwerts1953`

.. math::

    \begin{aligned}
        u c_{\text{in},i}(t) &= u c^l_i(t,0) - D_{\text{ax},i} \frac{\partial c^l_i}{\partial z}(t, 0) & \forall t > 0,\\
        \frac{\partial c^l_i}{\partial z}(t, L) &= 0 & \forall t > 0.
    \end{aligned}

As for the general rate model, both quasi-stationary and dynamic binding models are supported:

.. math::

    \begin{aligned}
        \text{quasi-stationary: }& & 0 &= f_{\text{ads},j}\left( c^p_j, c^s_j\right), \\
        \text{dynamic: }& & \frac{\partial c^s_j}{\partial t} &= f_{\text{ads},j}\left( c^p_j, c^s_j\right) + f_{\text{react},j}^s\left( c_j^p, c_j^s \right).
    \end{aligned}

By default, the following initial conditions are applied for all :math:`z \in [0,L]`:

.. math::

    \begin{aligned}
        c^l_i(0, z) &= 0, & c^p_{j,i}(0, z) &= 0, & c^s_{j,i,m_{j,i}}(0,z) &= 0.
    \end{aligned}

:ref:`MUOPGRMMultiParticleTypes` types are supported.
This model can also be used to simulate :ref:`MUOPGRMSizeExclusion`.
For the specification of flow rate and direction, the same holds as for the general rate model (see Section :ref:`MUOPGRMflow`).

For information on model parameters see :ref:`lumpded_rate_model_with_pores_config`.
