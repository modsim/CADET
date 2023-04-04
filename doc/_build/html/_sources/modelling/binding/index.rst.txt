.. _binding_models:

Binding models
==============

The following binding models are presented in dynamic binding mode.
By replacing all occurrences of :math:`\mathrm{d}q / \mathrm{d}t` with :math:`0`, quasi-stationary (rapid-equlibrium) binding mode is achieved.
In quasi-stationary binding, it is assumed that ad- and desorption take place on a much faster time scale than the other transport processes such that bead liquid phase :math:`c_{p,i}` (or bulk liquid phase :math:`c_i` for certain unit operation models) are always in equilibrium with the solid phase :math:`q_i`.

Equilibrium constants
---------------------

For the quasi-stationary binding mode, adsorption and desorption rate are no longer separate entities.
Instead, the quotient :math:`k_{\text{eq}} = k_a / k_d` of adsorption and desorption coefficient is the relevant parameter as shown for the linear binding model (see Section :ref:`linear_model`):

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} &= k_{a,i} c_{p,i} - k_{d,i} q_i \qquad \Rightarrow 0 = k_{a,i} c_{p,i} - k_{d,i} q_i \qquad \Leftrightarrow q_i = \frac{k_{a,i}}{k_{d,i}} c_{p,i} = k_{\text{eq},i} c_{p,i}.
    \end{aligned}

The equilibrium constant :math:`k_{\text{eq},i}` is used in CADET by setting :math:`k_{d,i} = 1` and :math:`k_{a,i} = k_{\text{eq},i}`.

Note that adsorption rate :math:`k_{a,i}` and desorption rate :math:`k_{d,i}` are linearly correlated in both binding modes due to the form of the equilibrium constant :math:`k_{\text{eq}}`:

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


.. _ldf_model:

Linear Driving Force (LDF)
---------------------------
Some authors use the linear driving force (LDF) approximation instead of the native kinetic form of an isotherm.
All three approaches are equivalent in rapid equilibrium (``IS_KINETIC = 0``) but not equivalent when binding kinetics are considered (``IS_KINETIC = 1``).

1. In the native approach, :math:`\frac{dq}{dt}` is an explicit funtion of :math:`c` and :math:`q`. For example :math:`\frac{dq}{dt}=k_a c (q_m - q)-k_d q` in the Langmuir model.

2. A linear driving force approximation is based on the equilibrium concentration :math:`q^*` for given :math:`c`.
For example :math:`q^*= \frac{q_m k_{eq} c}{1 + k_{eq} c}` in the Langmuir model.
Here, :math:`\frac{dq}{dt}` is proportional to the actual difference from equilibrium, i.e. :math:`\frac{dq}{dt} = k_{kin}(q^*-q)`.
Note that, the sign of :math:`\frac{dq}{dt}` causes the resulting flux to act towards the equilibrium.
In CADET, this approach is denoted by ``LDF``, for example in ``MULTI_COMPONENT_LANGMUIR_LDF``.

3. An alterniative linear driving force approximation is based on the equilibrium concentration :math:`c^*` for given :math:`q`.
For example :math:`c^*=\frac{q}{k_{eq} \left(q_{m}-q\right)}` in the Langmuir model.
Here, :math:`\frac{dq}{dt}` is proportional to the actual difference from equilibrium concentrations, i.e. :math:`\frac{dq}{dt} = k_{kin}(c-c^*)`.
Note that, the sign of :math:`\frac{dq}{dt}` causes the resulting flux to act towards the equilibrium.
In CADET, this approach is denoted by ``LDF_LIQUID_PHASE``, for example in ``MULTI_COMPONENT_LANGMUIR_LDF_LIQUID_PHASE``.

In both LDF examples, the original rate constants :math:`k_a` and :math:`k_d` are replaced by the equilibrium contant :math:`k_{eq}=\frac{k_a}{k_d}`.
The linear driving force approximations are based on a new kinetic constant, :math:`k_{kin}`.

Note that some isotherms, such as the Freundlich model, don't have a native representation in the above sense.
LDF versions are availabe for some but not all binding models implemented in CADET.

.. _reference_concentrations:

Reference concentrations
------------------------

Some binding models use reference concentrations :math:`c_{\text{ref}}` and :math:`q_{\text{ref}}` of the mobile phase modulator (e.g., salt) in the particle liquid and solid phase, respectively.
The reference values are mainly used for normalizing adsorption and desorption rates, but also for other parameters that appear with those concentrations.
They amount to a simple parameter transformation that is exemplified at one equation of the steric mass action binding model

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} = k_{a,i} c_{p,i}\bar{q}_0^{\nu_i} - k_{d,i} q_i c_{p,0}^{\nu_i},
    \end{aligned}

where :math:`c_{p,0}` denotes the mobile phase salt concentration and

.. math::

    \begin{aligned}
        \bar{q}_0 = \Lambda - \sum_{j=1}^{N_{\text{comp}} - 1} \left( \nu_j + \sigma_j \right) q_j
    \end{aligned}

is the number of available binding sites which is related to the number of bound salt ions.
Using the parameter transformation

.. math::

    \begin{aligned}
        k_{a,i} &= \tilde{k}_{a,i} q_{\text{ref}}^{-\nu_i}, \\
        k_{d,i} &= \tilde{k}_{d,i} c_{\text{ref}}^{-\nu_i},
    \end{aligned}

we obtain the modified model equation

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} = \tilde{k}_{a,i} c_{p,i} \left(\frac{\bar{q}_0}{q_{\text{ref}}}\right)^{\nu_i} - \tilde{k}_{d,i} q_i \left(\frac{c_{p,0}}{c_{\text{ref}}}\right)^{\nu_i}.
    \end{aligned}

This transformation serves as a (partial) nondimensionalization of the adsorption and desorption rates and, by properly choosing the reference concentrations :math:`c_{\text{ref}}` and :math:`q_{\text{ref}}`, may improve the optimizer performance.

Recommended choices for :math:`c_{\text{ref}}` are the average or maximum inlet concentration of the mobile phase modifier :math:`c_0`, and for :math:`q_{\text{ref}}` the ionic capacity :math:`\Lambda`.
Note that setting the reference concentrations to :math:`1.0` each results in the original binding model.


.. _dependence-on-external-function_bind:

Dependence on external function
-------------------------------

A binding model may depend on an external function or profile :math:`T\colon \left[ 0, T_{\text{end}}\right] \times [0, L] \to \mathbb{R}`, where :math:`L` denotes the physical length of the unit operation, or :math:`T\colon \left[0, T_{\text{end}}\right] \to \mathbb{R}` if the unit operation model has no axial length.
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


.. _binding_model_feature:

Binding model feature matrix
----------------------------

A short comparison of the most prominent binding model features is given in :numref:`MBFeatureMatrix`.
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
   * - :ref:`freundlich_ldf_model`
     - ×
     - ×
     - ✓
     - ×
   * - :ref:`multi_component_langmuir_model`
     - ✓
     - ×
     - ✓
     - ×
   * - :ref:`multi_component_langmuir_model_ldf`
     - ✓
     - ×
     - ✓
     - ×
   * - :ref:`multi_component_langmuir_model_ldf_liquid_phase`
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
   * - :ref:`multi_component_bi_langmuir_model_ldf`
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

