.. _pbm_model:

Population balance models
~~~~~~~~~~~~~~~~~~~~~~~~~

The population balance equation is a particle-number continuity equation which describes the evolution of the number density :math:`n` of the particles in the time and space domains. 
The particles of interest have both internal and external coordinates: the internal coordinate can be chosen as any property of the particles such as the particle size or volume 
while the external coordinate can be a characteristic dimension of the reactor itself, including the axial position. 

The current release considers the nucleation, growth and growth rate dispersion. Aggregation and fragmentation processes will be included later. The equations can be solved in CSTR or DPFR (LRM) to model batch or continuous processes. 

One-dimensional Population balance model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assuming a well-mixed tank, the population balance equation chosen the particle size :math:`x` as the internal coodinate reads:

.. math::

    \begin{aligned}
        \frac{\partial (n V)}{\partial t} = F_{in}n_{in} - F_{out}n - V \left( \frac{\partial (v_{G}n)}{\partial x} - D_g \frac{\partial^2 n}{\partial x^2} - B_0 \delta (x-x_c) \right),
    \end{aligned}

where :math:`F_{in}` and :math:`F_{out}` are the volumetric inflow and
outflow rates, :math:`V` is the reactor volume, :math:`n` is
the number density, :math:`n_{in}` is the inlet number density
distribution, :math:`v_{G}` is the particle growth rate,
:math:`D_g` is the growth dispersion rate, :math:`B_0` is the
nucleation kinetics and :math:`\delta` is the Dirac delta function representing particle nucleations of size :math:`x_c`. 


The upper boundary condition is called the regularity boundary condition:

.. math::
    :label: RegularityBC

    \begin{aligned}
        \left. \left( nv_{G} - D_g \frac{\partial n}{\partial x} \right) \right|_{x \to \infty}=0.
    \end{aligned}

The nucleation source :math:`B_0 \delta (x-x_c)` is introduced as the lower boundary condition :

.. math::
    :label: NucleationBC

    \begin{aligned}
        \left. \left( nv_{G}-D_g \frac{\partial n}{\partial x} \right)\right|_{x=x_c} = B_0.
    \end{aligned}

The mass balance equation accounts for the mass transfer between the particle phase and the solute phase:

.. math::

    \begin{aligned}
        \frac{\partial (cV)}{\partial t} = F_{in}c_{in} - F_{out}c -\rho k_v  V \left( B_0x^3_c + 3\int_{x_c}^{\infty} v_{G}n\ x^2 \;\mathrm{d}x \right),
    \end{aligned}

where :math:`c` is the solte concentration in the bulk phase, :math:`c_{in}` is the inlet solute mass concentration, :math:`\rho` is the nuclei mass density and :math:`k_v` is the volumetric shape factor of the particles.

Evolution of the reactor's volume is governed by:

.. math::

    \begin{aligned}
        \frac{\mathrm{d}V}{\mathrm{d}t} &= F_{\text{in}} - F_{\text{out}}.
    \end{aligned}


Two-dimensional Population balance model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PBM can also be formulated in DPFR (LRM) to model continuous processes. If we choose the axial position within a plug
flow reactor as the external coordinate :math:`z`, the :math:`2D` PBM incorporating axial dispersion reads:

.. math::

    \begin{aligned}
        \frac{\partial n}{\partial t} = -v_{ax} \frac{\partial n}{\partial z} +D_{ax} \frac{\partial^2 n}{\partial z^2}  - \frac{\partial (v_{G}n)}{\partial x} + D_g \frac{\partial^2 n}{\partial x^2} + B_0 \delta (x-x_c),
    \end{aligned}

where :math:`v_{ax}` is the axial velocity and :math:`D_{ax}` is the axial dispersion coefficient. Boundary conditions for the internal coordinate are Eq. :eq:`RegularityBC` and Eq. :eq:`NucleationBC`.

For the external coordinate :math:`z`, Danckwerts boundary conditions are applied:

.. math::

    \begin{aligned}
        \left. \left( n v_{ax}-D_{ax}\frac{\partial n}{\partial z} \right) \right|_{z=0}=v_{ax}n_{in,x}, \qquad \left.\frac{\partial n}{\partial z}\right|_{z=L}=0, 
    \end{aligned}

where :math:`L` is the length of the DPFR. 

The solute mass balance equation is:

.. math::

    \begin{aligned}
        \frac{\partial c}{\partial t} = -v_{ax} \frac{\partial c}{\partial z} +D_{ax} \frac{\partial^2 c}{\partial z^2} -\rho k_v \left( B_0x^3_c + 3\int_{x_c}^{\infty} v_{G}n x^2 \;\mathrm{d}x \right).
    \end{aligned}

As with the particle phase, the solute mass concentration :math:`c` is also subject to the Danckwerts boundary conditions:

.. math::

    \begin{aligned}
        \left.\left( c v_{ax}-D_{ax}\frac{\partial c}{\partial z} \right) \right|_{z=0}=v_{ax}c_{in}, \qquad \left.\frac{\partial c}{\partial z}\right|_{z=L}=0.
    \end{aligned}


Constitutive equations
^^^^^^^^^^^^^^^^^^^^^^

Constitutive equations describe the kinetic processes in the governing equations. The relative supersaturation :math:`s` is:

.. math::

    \begin{aligned}
        s=\frac{c-c_{eq}}{c_{eq}},
    \end{aligned}

where :math:`c_{eq}` is the solute solubility in the solvent. An empirical equation for primary nucleation is given by:

.. math::

    \begin{aligned}
        B_p=k_ps^u,
    \end{aligned}

where :math:`k_p` is the primary nucleation rate constant and :math:`u` is a constant. An empirical power-law expression is used for the secondary nucleation:

.. math::

    \begin{aligned}
        B_s=k_bs^bM^k,
    \end{aligned}

where :math:`k_b` is the secondary nucleation rate constant, :math:`b` and :math:`k` are system-related parameters and :math:`M` is the suspension density defined as

.. math::

    \begin{aligned}
        M=k_v\rho\int_{0}^{\infty}n\ x^3\;\mathrm{d}x.
    \end{aligned}

The following expression for the growth rate is implemented:

.. math::

    \begin{aligned}
        v_{G}=k_gs^g(a+\gamma x^p),
    \end{aligned}

where :math:`k_g` is the growth rate constant, :math:`\gamma` quantifies the size dependence, and :math:`g`, :math:`a` and :math:`p` are system-related constants.

If you want to customize the constitutive equations for your applications, please contact us and we can figure something out.

The PBM is currently implemented in the Reaction module. To configure the code, see :ref:`pbm_config`.