.. _lumpded_rate_model_with_pores_config:

Lumped rate model with pores
============================


Group /input/model/unit_XXX - UNIT_TYPE = LUMPED_RATE_MODEL_WITH_PORES
----------------------------------------------------------------------

``UNIT_TYPE``
   Specifies the type of unit operation model
   
   ================  ========================================================  =============
   **Type:** string  **Range:** :math:`\texttt{LUMPED_RATE_MODEL_WITH_PORES}`  **Length:** 1
   ================  ========================================================  =============
   
``NCOMP``
   Number of chemical components in the chromatographic medium
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``ADSORPTION_MODEL``
   Specifies the type of binding model of each particle type (or of all particle types if length is :math:`1`)
   
   ================  ==========================================  =========================================
   **Type:** string  **Range:** See Section :ref:`FFAdsorption`  **Length:** :math:`1 / \texttt{NPARTYPE}`
   ================  ==========================================  =========================================
   
``ADSORPTION_MODEL_MULTIPLEX``
   Multiplexing mode of :math:`\texttt{ADSORPTION_MODEL}`. If set to :math:`0`, each particle type has a different binding model and the length of :math:`\texttt{ADSORPTION_MODEL}` is :math:`\texttt{NPARTYPE}`. If set to :math:`1`, all particle types share the same binding model and the length of :math:`\texttt{ADSORPTION_MODEL}` is :math:`1`.  This field is optional and inferred from the length of :math:`\texttt{ADSORPTION_MODEL}` if left out.
   
   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 1\}`  **Length:** 1
   =============  ===========================  =============
   
``REACTION_MODEL_BULK``
   Specifies the type of reaction model of the bulk volume. The model is configured in the subgroup :math:`\texttt{reaction_bulk}`.
   
   ================  ================================  =============
   **Type:** string  **Range:** See :ref:`FFReaction`  **Length:** 1
   ================  ================================  =============
   
``REACTION_MODEL_PARTICLES``
   Specifies the type of reaction model of each particle type (or of all particle types if length is :math:`1`). The model is configured in the subgroup :math:`\texttt{reaction_particle}`, or :math:`\texttt{reaction_particle_XXX}` in case of disabled multiplexing.
   
   ================  ========================================  ===================================
   **Type:** string  **Range:** See Section :ref:`FFReaction`  **Length:** :math:`1` / :math:`\texttt{NPARTYPE}`
   ================  ========================================  ===================================
   
``REACTION_MODEL_PARTICLES_MULTIPLEX``
   Multiplexing mode of :math:`\texttt{REACTION_MODEL_PARTICLES}`. If set to :math:`0`, each particle type has a different reaction model and the length of :math:`\texttt{REACTION_MODEL_PARTICLES}` is :math:`\texttt{NPARTYPE}`. If set to :math:`1`, all particle types share the same reaction model and the length of :math:`\texttt{REACTION_MODEL_PARTICLES}` is :math:`1`.  This field is optional and inferred from the length of :math:`\texttt{REACTION_MODEL_PARTICLES}` if left out.
   
   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 1\}`  **Length:** 1
   =============  ===========================  =============
   
``INIT_C``
   Initial concentrations for each component in the bulk mobile phase

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   ================  =========================  ==================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NCOMP}`
   ================  =========================  ==================================
   
``INIT_CP``
   Initial concentrations for each component in the bead liquid phase (optional, :math:`\texttt{INIT_C}` is used if left out). The length of this field can be :math:`\texttt{NCOMP}` (same values for each particle type) or :math:`\texttt{NPARTYPE} \cdot \texttt{NCOMP}`  Values for each particle type can only be given when :math:`\texttt{ADSORPTION_MODEL_MULTIPLEX}` is :math:`0`. The ordering is type-major.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}`
   
   ================  =========================  ===========================================================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NCOMP} / \texttt{NPARTYPE} \cdot \texttt{NCOMP}`
   ================  =========================  ===========================================================================
   
``INIT_Q``
   Initial concentrations for each bound state of each component in the bead solid phase. If :math:`\texttt{ADSORPTION_MODEL_MULTIPLEX}` is :math:`0`, values for each particle type are required in type-component-major ordering (length is :math:`\texttt{NTOTALBND}`). If :math:`\texttt{ADSORPTION_MODEL_MULTIPLEX}` is :math:`1`, values for one particle type are required in component-major ordering (length is :math:`\sum_{i = 0}^{\texttt{NCOMP} - 1} \texttt{NBND}_i`).

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{SP}}^{-3}`
   
   ================  =========================
   **Type:** double  **Range:** :math:`\geq 0`
   ================  =========================
   
``INIT_STATE``
   Full state vector for initialization (optional, :math:`\texttt{INIT_C}`, :math:`\texttt{INIT_CP}`, and :math:`\texttt{INIT_Q}` will be ignored; if length is :math:`2\texttt{NDOF}`, then the second half is used for time derivatives)

   **Unit:** :math:`various`
   
   ================  =============================  ==================================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NDOF} / 2\texttt{NDOF}`
   ================  =============================  ==================================================
   
``COL_DISPERSION``
   Axial dispersion coefficient

   **Unit:** :math:`\mathrm{m}_{\mathrm{IV}}^{2}\,\mathrm{s}^{-1}`
   
   ================  =========================  =========================================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** see :math:`\texttt{COL_DISPERSION_MULTIPLEX}`
   ================  =========================  =========================================================
   
``COL_DISPERSION_MULTIPLEX``
   Multiplexing mode of :math:`\texttt{COL_DISPERSION}`. Determines whether :math:`\texttt{COL_DISPERSION}` is treated as component- and/or section-independent.  This field is optional. When left out, multiplexing behavior is inferred from the length of :math:`\texttt{COL_DISPERSION}`.  Valid modes are: 

  0. Component-independent, section-independent; length of :math:`\texttt{COL_DISPERSION}` is :math:`1` 
  1. Component-dependent, section-independent; length of :math:`\texttt{COL_DISPERSION}` is :math:`\texttt{NCOMP}` 
  2. Component-independent, section-dependent; length of :math:`\texttt{COL_DISPERSION}` is :math:`\texttt{NSEC}` 
  3. Component-dependent, section-dependent; length of :math:`\texttt{COL_DISPERSION}` is :math:`\texttt{NCOMP} \cdot \texttt{NSEC}`; ordering is section-major 
   
   =============  ===================================  =============
   **Type:** int  **Range:** :math:`\{0, \dots, 3 \}`  **Length:** 1
   =============  ===================================  =============
   
``COL_LENGTH``
   Column length

   **Unit:** :math:`\mathrm{m}`
   
   ================  ======================  =============
   **Type:** double  **Range:** :math:`> 0`  **Length:** 1
   ================  ======================  =============
   
``COL_POROSITY``

   Column porosity
   
   ================  ========================  =============
   **Type:** double  **Range:** :math:`(0,1]`  **Length:** 1
   ================  ========================  =============
   
``FILM_DIFFUSION``

   Film diffusion coefficients for each component of each particle type

   **Unit:** :math:`\mathrm{m}\,\mathrm{s}^{-1}`
   
   ================  =========================  =========================================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** see :math:`\texttt{FILM_DIFFUSION_MULTIPLEX}`
   ================  =========================  =========================================================
   
``FILM_DIFFUSION_MULTIPLEX``

   Multiplexing mode of :math:`\texttt{FILM_DIFFUSION}`. Determines whether :math:`\texttt{FILM_DIFFUSION}` is treated as component-, type-, and/or section-independent.  This field is optional. When left out, multiplexing behavior is inferred from the length of :math:`\texttt{FILM_DIFFUSION}`.  Valid modes are: 

  0. Component-dependent, type-independent, section-independent; length of :math:`\texttt{FILM_DIFFUSION}` is :math:`\texttt{NCOMP}` 
  1. Component-dependent, type-independent, section-dependent; length of :math:`\texttt{FILM_DIFFUSION}` is :math:`\texttt{NCOMP} \cdot \texttt{NSEC}`; ordering is section-major 
  2. Component-dependent, type-dependent, section-independent; length of :math:`\texttt{FILM_DIFFUSION}` is :math:`\texttt{NCOMP} \cdot \texttt{NPARTYPE}`; ordering is type-major 
  3. Component-dependent, type-dependent, section-dependent; length of :math:`\texttt{FILM_DIFFUSION}` is :math:`\texttt{NCOMP} \cdot \texttt{NPARTYPE} \cdot \texttt{NSEC}`; ordering is section-type-major 
   
   =============  ===================================  =============
   **Type:** int  **Range:** :math:`\{0, \dots, 3 \}`  **Length:** 1
   =============  ===================================  =============
   
``PAR_POROSITY``

   Particle porosity of all particle types or for each particle type
   
   ================  ========================  =========================================
   **Type:** double  **Range:** :math:`(0,1]`  **Length:** :math:`1 / \texttt{NPARTYPE}`
   ================  ========================  =========================================
   
``PAR_RADIUS``

   Particle radius of all particle types or for each particle type

   **Unit:** :math:`\mathrm{m}`
   
   ================  =====================  =========================================
   **Type:** double  **Range:** :math:`>0`  **Length:** :math:`1 / \texttt{NPARTYPE}`
   ================  =====================  =========================================
   
``PORE_ACCESSIBILITY``

   Pore accessibility factor of each component in each particle type (optional, defaults to :math:`1`).
   Note: Should not be used in combination with any binding model!

   
   ================  =========================  =============================================================
   **Type:** double  **Range:** :math:`(0, 1]`  **Length:** see :math:`\texttt{PORE_ACCESSIBILITY_MULTIPLEX}`
   ================  =========================  =============================================================
   
``PORE_ACCESSIBILITY_MULTIPLEX``
   Multiplexing mode of :math:`\texttt{PORE_ACCESSIBILITY}`. Determines whether :math:`\texttt{PORE_ACCESSIBILITY}` is treated as component-, type-, and/or section-independent.  This field is optional. When left out, multiplexing behavior is inferred from the length of :math:`\texttt{PORE_ACCESSIBILITY}`.  Valid modes are: 

  0. Component-dependent, type-independent, section-independent; length of :math:`\texttt{PORE_ACCESSIBILITY}` is :math:`\texttt{NCOMP}` 
  1. Component-dependent, type-independent, section-dependent; length of :math:`\texttt{PORE_ACCESSIBILITY}` is :math:`\texttt{NCOMP} \cdot \texttt{NSEC}`; ordering is section-major 
  2. Component-dependent, type-dependent, section-independent; length of :math:`\texttt{PORE_ACCESSIBILITY}` is :math:`\texttt{NCOMP} \cdot \texttt{NPARTYPE}`; ordering is type-major 
  3. Component-dependent, type-dependent, section-dependent; length of :math:`\texttt{PORE_ACCESSIBILITY}` is :math:`\texttt{NCOMP} \cdot \texttt{NPARTYPE} \cdot \texttt{NSEC}`; ordering is section-type-major 
   
   =============  ===================================  =============
   **Type:** int  **Range:** :math:`\{0, \dots, 3 \}`  **Length:** 1
   =============  ===================================  =============
   
``VELOCITY``
   Interstitial velocity of the mobile phase (optional if :math:`\texttt{CROSS_SECTION_AREA}` is present, see Section :ref:`MUOPGRMflow`)

   **Unit:** :math:`\mathrm{m}\,\mathrm{s}^{-1}`
   
   ================  =============================  =====================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`1 / \texttt{NSEC}`
   ================  =============================  =====================================
   
``CROSS_SECTION_AREA``

   Cross section area of the column (optional if :math:`\texttt{VELOCITY}` is present, see Section :ref:`MUOPGRMflow`)

   **Unit:** :math:`\mathrm{m}^{2}`
   
   ================  =====================  =============
   **Type:** double  **Range:** :math:`>0`  **Length:** 1
   ================  =====================  =============
   
``PAR_TYPE_VOLFRAC``

   Volume fractions of the particle types. The volume fractions can be set for all axial cells together or for each individual axial cell. For each cell, the volume fractions have to sum to :math:`1`. In case of a spatially inhomogeneous setting, the data is expected in cell-major ordering and the :math:`\texttt{SENS_SECTION}` field is used for indexing the axial cell when specifying parameter sensitivities.  This field is optional in case of only one particle type.
   
   ================  ========================  =======================================================================
   **Type:** double  **Range:** :math:`[0,1]`  **Length:** :math:`\texttt{NPARTYPE}` / :math:`\texttt{NCOL} \cdot \texttt{NPARTYPE}`
   ================  ========================  =======================================================================


Group /input/model/unit_XXX/discretization - UNIT_TYPE = LUMPED_RATE_MODEL_WITH_PORES
-------------------------------------------------------------------------------------

   
``NCOL``

   Number of axial column discretization cells
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``NPARTYPE``

   Number of particle types. Optional, inferred from the length of :math:`\texttt{NBOUND}` if left out.
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``NBOUND``

   Number of bound states for each component in each particle type in type-major ordering
   
   =============  =========================  ===========================================================================
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NCOMP} / \texttt{NPARTYPE} \cdot \texttt{NCOMP}`
   =============  =========================  ===========================================================================
   
``PAR_GEOM``

   Specifies the particle geometry for all or each particle type. Valid values are :math:`\texttt{SPHERE}`, :math:`\texttt{CYLINDER}`, :math:`\texttt{SLAB}`. Optional, defaults to :math:`\texttt{SPHERE}`.
   
   ================  =================================================
   **Type:** string  **Length:** :math:`1` / :math:`\texttt{NPARTYPE}`
   ================  =================================================
   
``USE_ANALYTIC_JACOBIAN``

   Determines whether analytically computed Jacobian matrix (faster) is used (value is :math:`1`) instead of Jacobians generated by algorithmic differentiation (slower, value is :math:`0`)
   
   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 1\}`  **Length:** 1
   =============  ===========================  =============
   
``RECONSTRUCTION``

   Type of reconstruction method for fluxes
   
   ================  ================================  =============
   **Type:** string  **Range:** :math:`\texttt{WENO}`  **Length:** 1
   ================  ================================  =============
   
``GS_TYPE``

   Type of Gram-Schmidt orthogonalization, see IDAS guide Section~4.5.7.3, p.~41f. A value of :math:`0` enables classical Gram-Schmidt, a value of 1 uses modified Gram-Schmidt.
   
   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 1\}`  **Length:** 1
   =============  ===========================  =============
   
``MAX_KRYLOV``

   Defines the size of the Krylov subspace in the iterative linear GMRES solver (0: :math:`\texttt{MAX_KRYLOV} = \texttt{NCOL} \cdot \texttt{NCOMP} \cdot \texttt{NPARTYPE}`)
   
   =============  ============================================================================================  =============
   **Type:** int  **Range:** :math:`\{0, \dots, \texttt{NCOL} \cdot \texttt{NCOMP} \cdot \texttt{NPARTYPE} \}`  **Length:** 1
   =============  ============================================================================================  =============
   
``MAX_RESTARTS``

   Maximum number of restarts in the GMRES algorithm. If lack of memory is not an issue, better use a larger Krylov space than restarts.
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** 1
   =============  =========================  =============
   
``SCHUR_SAFETY``

   Schur safety factor; Influences the tradeoff between linear iterations and nonlinear error control; see IDAS guide Section~2.1 and 5.
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============

For further discretization parameters, see also :ref:`flux_restruction_methods`, and :ref:`non_consistency_solver_parameters`.
