.. _FFReaction:

Reaction models
===============

Externally dependent reaction models
------------------------------------

Some reaction models have a variant that can use external sources as specified :ref:`/input/model/external/<FFModelExternalSourceLinInterp>` (also see Section :ref:`dependence-on-external-function_react`).
For the sake of brevity, only the standard variant of those reaction models is specified below.
In order to obtain the format for the externally dependent variant, first replace the reaction model name ``XXX`` by ``EXT_XXX``.
Each parameter :math:`p` (except for stoichiometric and exponent matrices) depends on a (possibly distinct) external source in a polynomial way:

.. math::

    \begin{aligned}
        p(T) &= p_{\texttt{TTT}} T^3 + p_{\texttt{TT}} T^2 + p_{\texttt{T}} T + p.
    \end{aligned}

Thus, a parameter ``XXX_YYY`` of the standard reaction model variant is replaced by the four parameters ``EXT_XXX_YYY``, ``EXT_XXX_YYY_T``, ``EXT_XXX_YYY_TT``, and ``EXT_XXX_YYY_TTT``.
Since each parameter can depend on a different external source, the dataset ``EXTFUN`` (not listed in the standard variants below) should contain a vector of 0-based integer indices of the external source of each parameter.
The ordering of the parameters in ``EXTFUN`` is given by the ordering in the standard variant.
However, if only one index is passed in ``EXTFUN``, this external source is used for all parameters.

Note that parameter sensitivities with respect to column radius, column length, particle core radius, and particle radius may be wrong when using externally dependent reaction models.
This is caused by not taking into account the derivative of the external profile with respect to column position.


.. _multiple-particle-types_reactions:

Multiple particle types
-----------------------

The group that contains the parameters of a reaction model in unit operation with index ``XXX`` reads ``/input/model/unit_XXX/reaction_particle``.
This is valid for models with a single particle type.
If a model has multiple particle types, it may have a different reaction model in each type.
The parameters are then placed in the group ``/input/model/unit_XXX/reaction_particle_YYY`` instead, where ``YYY`` denotes the index of the particle type.

Note that, in any case, ``/input/model/unit_XXX/reaction_particle_000`` contains the parameters of the first (and possibly sole) particle type.
This group also takes precedence over a possibly existing ``/input/model/unit_XXX/adsorption_particle`` group.

.. _FFReactionMassActionLaw:

Group /input/model/unit_XXX/reaction - REACTION_MODEL = MASS_ACTION_LAW
-----------------------------------------------------------------------

``MAL_KFWD_BULK``

   Forward rate constants for bulk volume reactions (available for external functions)
   
   ================  =========================  ===================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NREACT}`
   ================  =========================  ===================================
   
``MAL_KBWD_BULK``

   Backward rate constants for bulk volume reactions (available for external functions)
   
   ================  =========================  ===================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NREACT}`
   ================  =========================  ===================================
   
``MAL_KFWD_LIQUID``

   Forward rate constants for particle liquid phase reactions (available for external functions)
   
   ================  =========================  ===================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NREACT}`
   ================  =========================  ===================================
   
``MAL_KBWD_LIQUID``

   Backward rate constants for particle liquid phase reactions (available for external functions)
   
   ================  =========================  ===================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NREACT}`
   ================  =========================  ===================================
   
``MAL_KFWD_SOLID``

   Forward rate constants for particle solid phase reactions (available for external functions)
   
   ================  =========================  ===================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NREACT}`
   ================  =========================  ===================================
   
``MAL_KBWD_SOLID``

   Backward rate constants for particle solid phase reactions (available for external functions)
   
   ================  =========================  ===================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NREACT}`
   ================  =========================  ===================================
   
``MAL_STOICHIOMETRY_BULK``

   Stoichiometric matrix of bulk volume reactions as :math:`\texttt{NCOMP} \times \texttt{NREACT}` matrix in row-major storage
   
   ================  ========================================================
   **Type:** double  **Length:** :math:`\texttt{NCOMP} \cdot \texttt{NREACT}`
   ================  ========================================================
   
``MAL_EXPONENTS_BULK_FWD``

   Forward exponent matrix of bulk volume reactions as :math:`\texttt{NCOMP} \times \texttt{NREACT}` matrix in row-major storage (optional, calculated from :math:`\texttt{MAL_STOICHIOMETRY_BULK}` by default)
   
   ================  ========================================================
   **Type:** double  **Length:** :math:`\texttt{NCOMP} \cdot \texttt{NREACT}`
   ================  ========================================================
   
``MAL_EXPONENTS_BULK_BWD``

   Backward exponent matrix of bulk volume reactions as :math:`\texttt{NCOMP} \times \texttt{NREACT}` matrix in row-major storage (optional, calculated from :math:`\texttt{MAL_STOICHIOMETRY_BULK}` by default)
   
   ================  ========================================================
   **Type:** double  **Length:** :math:`\texttt{NCOMP} \cdot \texttt{NREACT}`
   ================  ========================================================
   
``MAL_STOICHIOMETRY_LIQUID``

   Stoichiometric matrix of particle liquid phase reactions as :math:`\texttt{NCOMP} \times \texttt{NREACT}` matrix in row-major storage
   
   ================  ========================================================
   **Type:** double  **Length:** :math:`\texttt{NCOMP} \cdot \texttt{NREACT}`
   ================  ========================================================
   
``MAL_EXPONENTS_LIQUID_FWD``

   Forward exponent matrix of particle liquid phase reactions as :math:`\texttt{NCOMP} \times \texttt{NREACT}` matrix in row-major storage (optional, calculated from :math:`\texttt{MAL_STOICHIOMETRY_LIQUID}` by default)
   
   ================  ========================================================
   **Type:** double  **Length:** :math:`\texttt{NCOMP} \cdot \texttt{NREACT}`
   ================  ========================================================
   
``MAL_EXPONENTS_LIQUID_BWD``

   Backward exponent matrix of particle liquid phase reactions as :math:`\texttt{NCOMP} \times \texttt{NREACT}` matrix in row-major storage (optional, calculated from :math:`\texttt{MAL_STOICHIOMETRY_LIQUID}` by default)
   
   ================  ========================================================
   **Type:** double  **Length:** :math:`\texttt{NCOMP} \cdot \texttt{NREACT}`
   ================  ========================================================
   
``MAL_EXPONENTS_LIQUID_FWD_MODSOLID``

   Forward solid phase modifier exponent matrix of particle liquid phase reactions as :math:`\texttt{NTOTALBND} \times \texttt{NREACT}` matrix in row-major storage (optional, defaults to all 0)
   
   ================  ============================================================
   **Type:** double  **Length:** :math:`\texttt{NTOTALBND} \cdot \texttt{NREACT}`
   ================  ============================================================
   
``MAL_EXPONENTS_LIQUID_BWD_MODSOLID``

   Backward solid phase modifier exponent matrix of particle liquid phase reactions as :math:`\texttt{NTOTALBND} \times \texttt{NREACT}` matrix in row-major storage (optional, defaults to all 0)
   
   ================  ============================================================
   **Type:** double  **Length:** :math:`\texttt{NTOTALBND} \cdot \texttt{NREACT}`
   ================  ============================================================
   
``MAL_STOICHIOMETRY_SOLID``

   Stoichiometric matrix of particle solid phase reactions as :math:`\texttt{NTOTALBND} \times \texttt{NREACT}` matrix in row-major storage
   
   ================  ============================================================
   **Type:** double  **Length:** :math:`\texttt{NTOTALBND} \cdot \texttt{NREACT}`
   ================  ============================================================
   
``MAL_EXPONENTS_SOLID_FWD``

   Forward exponent matrix of particle solid phase reactions as :math:`\texttt{NTOTALBND} \times \texttt{NREACT}` matrix in row-major storage (optional, calculated from :math:`\texttt{MAL_STOICHIOMETRY_SOLID}` by default)
   
   ================  ============================================================
   **Type:** double  **Length:** :math:`\texttt{NTOTALBND} \cdot \texttt{NREACT}`
   ================  ============================================================
   
``MAL_EXPONENTS_SOLID_BWD``

   Backward exponent matrix of particle solid phase reactions as :math:`\texttt{NTOTALBND} \times \texttt{NREACT}` matrix in row-major storage (optional, calculated from :math:`\texttt{MAL_STOICHIOMETRY_SOLID}` by default)
   
   ================  ============================================================
   **Type:** double  **Length:** :math:`\texttt{NTOTALBND} \cdot \texttt{NREACT}`
   ================  ============================================================
   
``MAL_EXPONENTS_SOLID_FWD_MODLIQUID``

   Forward liquid phase modifier exponent matrix of particle solid phase reactions as :math:`\texttt{NCOMP} \times \texttt{NREACT}` matrix in row-major storage (optional, defaults to all 0)
   
   ================  ========================================================
   **Type:** double  **Length:** :math:`\texttt{NCOMP} \cdot \texttt{NREACT}`
   ================  ========================================================
   
``MAL_EXPONENTS_SOLID_BWD_MODLIQUID``

   Backward liquid phase modifier exponent matrix of particle solid phase reactions as :math:`\texttt{NCOMP} \times \texttt{NREACT}` matrix in row-major storage (optional, defaults to all 0)
   
   ================  ========================================================
   **Type:** double  **Length:** :math:`\texttt{NCOMP} \cdot \texttt{NREACT}`
   ================  ========================================================
