.. _FFAdsorption:

Binding models
==============

Externally dependent binding models
-----------------------------------

Some binding models have a variant that can use external sources as specified in :ref:`/input/model/external/<FFModelExternalSourceLinInterp>` (also see Section :ref:`dependence-on-external-function_bind` for more information, and Section :ref:`binding_model_feature` on which binding models support this feature).
For the sake of brevity, only the standard variant of those binding models is specified below.
In order to obtain the format for the externally dependent variant, first replace the binding model name ``XXX`` by ``EXT_XXX``.
Each parameter :math:`p` (except for reference concentrations ``XXX_REFC0`` and ``XXX_REFQ``) depends on a (possibly distinct) external source in a polynomial way: 

.. math::

    \begin{aligned}
        p(T) &= p_{\texttt{TTT}} T^3 + p_{\texttt{TT}} T^2 + p_{\texttt{T}} T + p.
    \end{aligned}

Thus, a parameter ``XXX_YYY`` of the standard binding model variant is replaced by the four parameters ``EXT_XXX_YYY``, ``EXT_XXX_YYY_T``, ``EXT_XXX_YYY_TT``, and ``EXT_XXX_YYY_TTT``.
Since each parameter can depend on a different external source, the dataset ``EXTFUN`` (not listed in the standard variants below) should contain a vector of 0-based integer indices of the external source of each parameter.
The ordering of the parameters in ``EXTFUN`` is given by the ordering in the standard variant.
However, if only one index is passed in ``EXTFUN``, this external source is used for all parameters.

Note that parameter sensitivities with respect to column radius, column length, particle core radius, and particle radius may be wrong when using externally dependent binding models.
This is caused by not taking into account the derivative of the external profile with respect to column position.


Non-binding components
----------------------

For binding models that do not support multiple bound states, many parameters can vary per component and their length is taken as ``NCOMP``.
However, these models still support non-binding components.
In this case, the entries in their parameters that correspond to non-binding components are simply ignored.


.. _multiple-particle-types_binding:

Multiple particle types
-----------------------

The group that contains the parameters of a binding model in unit operation with index ``XXX`` reads ``/input/model/unit_XXX/adsorption``.
This is valid for models with a single particle type.
If a model has multiple particle types, it may have a different binding model in each type.
The parameters are then placed in the group ``/input/model/unit_XXX/adsorption_YYY`` instead, where ``YYY`` denotes the index of the particle type.

Note that, in any case, ``/input/model/unit_XXX/adsorption_000`` contains the parameters of the first (and possibly sole) particle type.
This group also takes precedence over a possibly existing ``/input/model/unit_XXX/adsorption`` group.

.. toctree::
    :maxdepth: 2

    linear
    multi_component_langmuir
    multi_component_bi_langmuir
    multi_component_anti_langmuir
    mobile_phase_modulator_langmuir
    extended_mobile_phase_modulator_langmuir
    multi_component_spreading
    steric_mass_action
    multi_state_steric_mass_action
    simplified_multi_state_steric_mass_action
    bi_steric_mass_action
    generalized_ion_exchange
    saska
    self_association
    kumar_langmuir

