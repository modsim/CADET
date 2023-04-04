.. _lumped_rate_model_without_pores_model:

Lumped rate model without pores (LRM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lumped rate model without pores :cite:`Guiochon2006,Felinger2004` deviates from the lumped rate model with pores (see Section :ref:`lumped_rate_model_with_pores_model`) by neglecting pores completely.
The particle phase :math:`c^p` is removed and the porosity :math:`\varepsilon_t` is taken as total porosity 

.. math::
    :label: TotalPorosity

    \begin{aligned}
        \varepsilon_t = \varepsilon_c + \left( 1 - \varepsilon_c \right) \varepsilon_p. 
    \end{aligned}

The phase ratio is denoted by :math:`\beta_t = \varepsilon_t / (1 - \varepsilon_t)` accordingly.
The model equations are given by

.. math::

    \begin{aligned}
        \frac{\partial c^l_i}{\partial t} + \frac{1}{\beta_t} \frac{\partial}{\partial t} \sum_{m_i} c^s_{i,m_i} &= -u \frac{\partial c^l_i}{\partial z} + D_{\text{ax},i} \frac{\partial^2 c^l_i}{\partial z^2} + f_{\text{react},i}^l\left( c^l, c^s \right) + \frac{1}{\beta_t} f_{\text{react},i}^s\left( c^l, c^s \right),
    \end{aligned}

where :math:`\beta_t = \varepsilon_t / (1 - \varepsilon_t)` denotes the (total) phase ratio.
The equations are complemented by Danckwerts boundary conditions :cite:`Danckwerts1953`

.. math::

    \begin{aligned}
        u c_{\text{in},i}(t) &= u c^l_i(t,0) - D_{\text{ax},i} \frac{\partial c^l_i}{\partial z}(t, 0) & \forall t > 0,\\
        \frac{\partial c^l_i}{\partial z}(t, L) &= 0 & \forall t > 0.
    \end{aligned}

Both quasi-stationary and dynamic binding models are supported:

.. math::

    \begin{aligned}
        \text{quasi-stationary: }& & 0 &= f_{\text{ads}}\left( c^l, c^s\right), \\
        \text{dynamic: }& & \frac{\partial q}{\partial t} &= f_{\text{ads}}\left( c^l, c^s\right) + f_{\text{react}}^s\left( c^l, c^s \right).
    \end{aligned}

By default, the following initial conditions are applied for all :math:`z \in [0,L]`:

.. math::

    \begin{aligned}
        c^l_i(0, z) &= 0, & c^s_{i,m_i}(0,z) &= 0.
    \end{aligned}

Note that by setting :math:`\varepsilon_t = 1`, removing all bound states by setting :math:`N_{\text{bnd},i} = 0` for all components :math:`i`, and applying no binding model, a dispersive plug flow reactor (DPFR) is obtained.
For the specification of flow rate and direction, the same holds as for the general rate model (see Section :ref:`MUOPGRMflow`).

For information on model parameters see :ref:`lumpded_rate_model_without_pores_config`.
