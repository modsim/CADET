.. _binding_models:

Binding models
==============

The following binding models are presented in dynamic binding mode.
By replacing all occurrences of :math:`\mathrm{d}q / \mathrm{d}t` with :math:`0`, quasi-stationary (rapid-equlibrium) binding mode is achieved.
In quasi-stationary binding, it is assumed that ad- and desorption take place on a much faster time scale than the other transport processes such that bead liquid phase :math:`c_{p,i}` (or bulk liquid phase :math:`c_i` for certain unit operation models) are always in equilibrium with the solid phase :math:`q_i`.

**Equilibrium constants:** For the quasi-stationary binding mode, adsorption and desorption rate are no longer separate entities.
Instead, the quotient :math:`k_{\text{eq}} = k_a / k_d` of adsorption and desorption coefficient is the relevant parameter as shown for the linear binding model (see Section :ref:`linear_model`):

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} &= k_{a,i} c_{p,i} - k_{d,i} q_i \qquad \Rightarrow 0 = k_{a,i} c_{p,i} - k_{d,i} q_i \qquad \Leftrightarrow q_i = \frac{k_{a,i}}{k_{d,i}} c_{p,i} = k_{\text{eq},i} c_{p,i}.
    \end{aligned}

The equilibrium constant :math:`k_{\text{eq},i}` is used in CADET by setting :math:`k_{d,i} = 1` and :math:`k_{a,i} = k_{\text{eq},i}`.

**Correlation of ad- and desorption rates:** Note that adsorption rate :math:`k_{a,i}` and desorption rate :math:`k_{d,i}` are linearly correlated in both binding modes due to the form of the equilibrium constant :math:`k_{\text{eq}}`:

.. math::

    \begin{aligned}
        k_{a,i} = k_{\text{eq}} k_{d,i}.
    \end{aligned}

This correlation can potentially degrade performance of some optimization algorithms.
While in quasi-stationary binding mode this is prevented by using the technique above, a dynamic binding model has to be reparameterized in order to decouple parameters:

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} &= k_{a,i} c_{p,i} - k_{d,i} q_i = k_{d,i} \left[ k_{\text{eq},i} c_{p,i} - q_i \right] = k_{a,i} \left[ c_{p,i} - \frac{1}{k_{\text{eq},i}} q_i \right].
    \end{aligned}

This can be achieved by a (nonlinear) parameter transform

.. math::

    \begin{aligned}
        F\left( k_{\text{eq},i}, k_{d,i} \right) = \begin{pmatrix} k_{\text{eq},i} k_{d,i} \\ k_{d,i} \end{pmatrix} \text{ with Jacobian } J_F\left( k_{\text{eq},i}, k_{d,i} \right) = \begin{pmatrix} k_{d,i} & k_{\text{eq},i} \\ 0 & 1 \end{pmatrix}.
    \end{aligned}

**Dependence on external function:** A binding model may depend on an external function or profile :math:`T\colon \left[ 0, T_{\text{end}}\right] \times [0, L] \to \mathbb{R}`, where :math:`L` denotes the physical length of the unit operation, or :math:`T\colon \left[0, T_{\text{end}}\right] \to \mathbb{R}` if the unit operation model has no axial length.
By using an external profile, it is possible to account for effects that are not directly modeled in CADET (e.g., temperature).
The dependence of each parameter is modeled by a polynomial of third degree. For example, the adsorption rate :math:`k_a` is really given by

.. math::

    \begin{aligned}
        k_a(T) &= k_{a,3} T^3 + k_{a,2} T^2 + k_{a,1} T + k_{a,0}.
    \end{aligned}

While :math:`k_{a,0}` is set by the original parameter ``XXX_KA`` of the file format (``XXX`` being a placeholder for the binding model), the parameters :math:`k_{a,3}`, :math:`k_{a,2}`, and :math:`k_{a,1}` are given by ``XXX_KA_TTT``, ``XXX_KA_TT``, and ``XXX_KA_T``, respectively.
The identifier of the externally dependent binding model is constructed from the original identifier by prepending ``EXT_`` (e.g., ``MULTI_COMPONENT_LANGMUIR`` is changed into ``EXT_MULTI_COMPONENT_LANGMUIR``).
This pattern applies to all parameters and supporting binding models (see :numref:`MBFeatureMatrix`).
Note that the parameter units have to be adapted to the unit of the external profile by dividing with an appropriate power.

Each parameter of the externally dependent binding model can depend on a different external source.
The 0-based indices of the external source for each parameter is given in the dataset ``EXTFUN``.
By assigning only one index to ``EXTFUN``, all parameters use the same source.
The ordering of the parameters in ``EXTFUN`` is given by the ordering in the file format specification in Section :ref:`FFAdsorption`.

**Binding model feature matrix:** A short comparison of the most prominent binding model features is given in :numref:`MBFeatureMatrix`.
The implemented binding models can be divided into two main classes: Single-state and multi-state binding.
While single-state models only have one bound state per component (or less), multi-state models provide multiple (possibly different) bound states for each component, which may correspond to different binding orientations or binding site types.
The models also differ in whether a mobile phase modifier (e.g., salt) is supported to modulate the binding behavior.

.. _MBFeatureMatrix:
.. list-table:: Supported features of the different binding models
   :widths: 30 15 25 15 15
   :header-rows: 1

   * - Binding model
     - Competitive
     - Mobile phase modifier
     - External function
     - Multi-state
   * - :ref:`linear_model`
     - ×
     - ×
     - ✓
     - ×
   * - :ref:`multi_component_langmuir_model`
     - ✓
     - ×
     - ✓
     - ×
   * - :ref:`multi_component_anti_langmuir_model`
     - ✓
     - ×
     - ✓
     - ×
   * - :ref:`steric_mass_action_model`
     - ✓
     - ✓
     - ✓
     - ×
   * - :ref:`generalized_ion_exchange_model`
     - ✓
     - ✓
     - ✓
     - ×
   * - :ref:`self_association_model`
     - ✓
     - ✓
     - ✓
     - ×
   * - :ref:`mobile_phase_modulator_langmuir_model`
     - ✓
     - ✓
     - ✓
     - ×
   * - :ref:`extended_mobile_phase_modulator_langmuir_model`
     - ✓
     - ✓
     - ✓
     - ×
   * - :ref:`kumar_langmuir_model`
     - ✓
     - ✓
     - ✓
     - ×
   * - :ref:`saska_model`
     - ×
     - ×
     - ✓
     - ×
   * - :ref:`multi_component_bi_langmuir_model`
     - ✓
     - ×
     - ✓
     - ✓
   * - :ref:`multi_component_spreading_model`
     - ✓
     - ×
     - ✓
     - ✓
   * - :ref:`multi_state_steric_mass_action_model`
     - ✓
     - ✓
     - ✓
     - ✓
   * - :ref:`simplified_multi_state_steric_mass_action_model`
     - ✓
     - ✓
     - ×
     - ✓
   * - :ref:`bi_steric_mass_action_model`
     - ✓
     - ✓
     - ✓
     - ✓


.. toctree::
    :hidden:
    :glob:

    *

