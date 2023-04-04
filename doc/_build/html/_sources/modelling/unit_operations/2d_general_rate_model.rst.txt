.. _2d_general_rate_model_model:

Two Dimensional General rate model (GRM2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The general rate model as introduced in Section :ref:`general_rate_model_model` assumes homogeneity in the cross sections of the column.
This allows to consider transport along the axial dimension only.
However, due to packing irregularity and inhomogeneous flow at the inlet (i.e., frits), this assumption may be a crude approximation.
This model can be improved by introducing a radial coordinate :math:`\rho \in [0, R]`, where :math:`R` is the column radius, in the interstitial volume Eq. :eq:`ModelColumn`:

.. math::
   :label: ModelColumn2D

   	\varepsilon_c \frac{\partial c^l_i}{\partial t} = &-\varepsilon_c u \frac{\partial c^l_i}{\partial z} + \varepsilon_c D_{\text{ax},i} \frac{\partial^2 c^l_i}{\partial z^2} + \frac{1}{\rho} \frac{\partial}{\partial \rho} \left( \rho D_{\text{rad},i} \frac{\partial}{\partial \rho} \left( \varepsilon_c c^l_i \right) \right) \\ 
    &- \left(1 - \varepsilon_c\right) \sum_j d_j \frac{ 3 k_{f,j,i} }{r_{p,j}} \left[ c^l_i - c^p_{j,i}(\cdot, \cdot, \cdot, r_{p,j}) \right] + \varepsilon_c f_{\text{react},i}^l\left(c^l\right). 

Here, 

  - :math:`c^l_i\colon \left[0, T_{\text{end}}\right] \times [0, L] \times [0, R] \rightarrow \mathbb{R}^{\geq 0}`,
  - :math:`c^p_{j,i}\colon \left[0, T_{\text{end}}\right] \times [0, L] \times [0, R] \times [r_{c,j}, r_{p,j}] \rightarrow \mathbb{R}^{\geq 0}`, and 
  - :math:`c^s_{j,i,m_{j,i}}\colon \left[0, T_{\text{end}}\right] \times [0, L] \times [0, R] \times [r_{c,j}, r_{p,j}] \rightarrow \mathbb{R}^{\geq 0}` 
  
depend on :math:`\rho`.
Additionally, the porosity :math:`\varepsilon_c`, axial dispersion coefficient :math:`D_{\text{ax},i}`, radial dispersion coefficient :math:`D_{\text{rad},i}`, and interstitial velocity :math:`u` may depend on :math:`\rho`.

The dependence of the parameters on :math:`\rho` is not arbitrary.
For simplicity, it is assumed that the parameters are piecewise constant, that is, the range :math:`[0, R]` is divided into disjoint zones in which all parameters are constant.
These zones are used for radial discretization and can be supplied to the simulator.
Continuous dependence of the parameters can be realized by piecewise constant approximation.

The Danckwerts boundary conditions at the column in- and outlet, Eq. :eq:`BCInlet` and :eq:`BCOutlet`, are modified to account for the radial coordinate:

.. math::
    :label: BCInlet2D 

    \begin{aligned}
        u(\rho) c_{\text{in},i}(t,\rho) &= u(\rho) c^l_i(t,0,\rho) - D_{\text{ax},i}(\rho) \frac{\partial c^l_i}{\partial z}(t, 0, \rho) & \forall t > 0, \rho \in (0,R),
    \end{aligned}

.. math::
    :label: BCOutlet2D

    \begin{aligned}
        \frac{\partial c^l_i}{\partial z}(t, L, \rho) &= 0 & \forall t > 0, \rho \in (0,R). 
    \end{aligned}

Conditions for the radial direction are added:

.. math::
    :label: BCRadial2DInner

    \begin{aligned}
        \frac{\partial{c^l_i}}{\partial \rho}(\cdot, \cdot, 0) &= 0,  \\
    \end{aligned}

.. math::
   :label: BCRadial2DOuter

    \begin{aligned}
        \frac{\partial{c^l_i}}{\partial \rho}(\cdot, \cdot, R) &= 0. 
    \end{aligned}

While the inner condition Eq.\ :eq:`BCRadial2DInner` represents symmetry at the column center, the outer condition Eq. :eq:`BCRadial2DOuter` is a no-flux condition.

Using the inlet boundary condition Eq. :eq:`BCInlet2D`, each radial zone is equipped with its own inlet and outlet port.
That is, this unit operation has as many inlet and outlet ports as it has radial zones (parameter ``NRAD`` in the ``discretization`` group).
This allows each radial zone to have its own inlet profile, which enables modeling of flow distribution in the frits by sending the feed through varying hold-up volumes before injecting it into a radial zone.


.. _MUOPGRMflow2D:

Specification of flow rate / velocity and direction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since the column radius :math:`R` and the zones :math:`(\rho_k, \rho_{k+1})`, :math:`k = 0, \dots, N_{\text{rad}} - 1`, are known, the interstitial velocities :math:`u_k` are inferred from the volumetric flow rates via

.. math::

    \begin{aligned}
        u_k = u_{\text{int},k} = \frac{F_{\text{in},k}}{\pi \left( \rho_{k+1}^2 - \rho_k^2 \right) \varepsilon_{c,k}},
    \end{aligned}

where :math:`F_{\text{in},k}` denotes the volumetric flow rate into zone :math:`k`.

The direction of flow inside the radial zone of the unit operation is governed by the sign of the interstitial velocity :math:`u_k`.
A positive sign results in (standard) forward flow, whereas a negative sign reverses the flow direction.
Note that in case of reversed flow, the chromatogram is returned at the unit operation’s *inlet* port, which may not be returned from simulation by default.

Note that, contrary to the standard general rate model as presented in Section :ref:`general_rate_model_model`, the interstitial flow rate is always given by the volumetric flow rate.
The velocity parameter only determines the flow direction.

For information on model parameters see :ref:`2d_general_rate_model_config`.
