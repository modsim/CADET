.. _lumped_rate_model_with_pores_model:

Lumped rate model with pores (LRMP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lumped rate model with pores :cite:`Guiochon2006,Felinger2004` deviates from the general rate model (see Section :ref:`general_rate_model_model`) by neglecting pore diffusion.
The particle phase :math:`c^p_j` is still there, but no mass transfer happens except for binding and film diffusion.
Hence, the model equations are given by

.. math::

    \begin{aligned}
        \frac{\partial c^\ell_i}{\partial t} &= -u \frac{\partial c^\ell_i}{\partial z} + D_{\text{ax},i} \frac{\partial^2 c^\ell_i}{\partial z^2} - \frac{1}{\beta_c} \sum_{j} d_j \frac{3}{r_{p,j}} k_{f,j,i}\left[ c^\ell_i - c^p_{j,i} \right] + f_{\text{react},i}^\ell\left(c^\ell\right),
    \end{aligned}

.. math::
    :label: ModelParticleLRMP

    \begin{aligned}
        \frac{\partial c^p_{j,i}}{\partial t} + \frac{1 - \varepsilon_{p,j}}{F_{\text{acc},j,i} \varepsilon_{p,j}} \frac{\partial}{\partial t} \sum_{m_{j,i}} c^s_{j,i,m_{j,i}} &= \frac{3}{F_{\text{acc},j,i} \varepsilon_{p,j} r_{p,j}}k_{f,j,i}\left[ c^\ell_i - c^p_{j,i} \right] \\
        &+ f_{\text{react},j,i}^p\left( c_j^p, c_j^s \right) + \frac{1 - \varepsilon_{p,j}}{F_{\text{acc},j,i} \varepsilon_{p,j}} f_{\text{react},j,i}^s\left( c_j^p, c_j^s \right)
    \end{aligned}

with the same meanings of variables and parameters as in the general rate model.
The equations are complemented by Danckwerts boundary conditions :cite:`Danckwerts1953`

.. math::

    \begin{aligned}
        u c_{\text{in},i}(t) &= u c^\ell_i(t,0) - D_{\text{ax},i} \frac{\partial c^\ell_i}{\partial z}(t, 0) & \forall t > 0,\\
        \frac{\partial c^\ell_i}{\partial z}(t, L) &= 0 & \forall t > 0.
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
        c^\ell_i(0, z) &= 0, & c^p_{j,i}(0, z) &= 0, & c^s_{j,i,m_{j,i}}(0,z) &= 0.
    \end{aligned}

:ref:`MUOPGRMMultiParticleTypes` types are supported.
This model can also be used to simulate :ref:`MUOPGRMSizeExclusion`.
For the specification of flow rate and direction, the same holds as for the general rate model (see Section :ref:`MUOPGRMflow`).

For information on model parameters see :ref:`lumped_rate_model_with_pores_config`.

Radial flow LRMP
^^^^^^^^^^^^^^^^

The radial flow LRMP describes transport of solute molecules through the interstitial column volume by radial convective flow, band broadening caused by radial dispersion, mass transfer resistance through a stagnant film around the beads, and adsorption to the inner bead surfaces.

The main assumptions are:

- The shells of the column are homogenous in terms of interstitial volume, fluid flow, and distribution of components.
  Thus, only one spatial coordinate in radial direction :math:`\rho` is needed and axial transport is neglected in the column bulk volume.

- The bead radii :math:`r_{p}` are much smaller than the column radius :math:`\mathrm{P}-\mathrm{P}_c`, with :math:`\mathrm{P}` and :math:`\mathrm{P}_c` being the inner and outer column radius respectively, and the column length :math:`L`.
  Therefore, the beads can be seen as continuously distributed inside the column (i.e., at each point there is interstitial and bead volume).

- The fluids are incompressible, i.e. the velocity field :math:`\mathrm{V} \colon \mathbb{R}^3 \to \mathbb{R}^3` submits to :math:`\operatorname{div}\left( \mathrm{V} \right) \equiv 0`.
  That is, the volumetric flow rate at the inner and outer column radius are the same.

Consider a hollow (double walled) column with inner column diameter :math:`\mathrm{P}_c>0` and outer diameter :math:`\mathrm{P}>\mathrm{P}_c`, filled with spherical beads of (possibly) multiple types with radius :math:`r_{p,j} \ll L` (see :numref:`ModelGRMColumn`), where :math:`j` is the particle type index. The mass balance in the interstitial column volume is described by

.. math::

    \begin{aligned}
        \frac{\partial c^\ell_i}{\partial t} &= -\frac{u}{\rho} \frac{\partial c^\ell_i}{\partial \rho} + D_{\text{rad},i} \frac{1}{\rho} \frac{\partial}{\partial \rho} \left( \rho \frac{\partial c^\ell_i}{\partial \rho} \right) - \frac{1}{\beta_c} \sum_{j} d_j \frac{3}{r_{p,j}} k_{f,j,i}\left[ c^\ell_i - c^p_{j,i} \right] + f_{\text{react},i}^\ell\left(c^\ell\right),
    \end{aligned}

The equations are complemented by Eq. :ref:`ModelParticleLRMP` and the Danckwerts boundary conditions :cite:`Danckwerts1953`

.. math::

    \begin{aligned}
        u c_{\text{in},i}(t) &= u c^\ell_i(t,0) - D_{\text{rad},i} \frac{\partial c^\ell_i}{\partial \rho}(t, 0) & \forall t > 0,\\
        \frac{\partial c^\ell_i}{\partial \rho}(t, \mathrm{P}) &= 0 & \forall t > 0.
    \end{aligned}

The complementing binding equations are described by the same equations as for the axial LRMP.

For information on model parameters see :ref:`radial_flow_models_config` in addition to :ref:`lumped_rate_model_with_pores_config`.
