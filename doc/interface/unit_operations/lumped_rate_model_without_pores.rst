.. _lumped_rate_model_without_pores_config:

Lumped Rate Model Without Pores
===============================

Group /input/model/unit_XXX - UNIT_TYPE = LUMPED_RATE_MODEL_WITHOUT_PORES
-------------------------------------------------------------------------

For information on model equations, refer to :ref:`lumped_rate_model_without_pores_model`.


``UNIT_TYPE``

   Specifies the type of unit operation model
   
   ================  ===========================================================  =============
   **Type:** string  **Range:** :math:`\texttt{LUMPED_RATE_MODEL_WITHOUT_PORES}`  **Length:** 1
   ================  ===========================================================  =============
   
``NCOMP``

   Number of chemical components in the chromatographic medium
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``ADSORPTION_MODEL``

   Specifies the type of binding model
   
   ================  ==========================================  =============
   **Type:** string  **Range:** See Section :ref:`FFAdsorption`  **Length:** 1
   ================  ==========================================  =============
   
``NBOUND``

   Number of bound states for each component
   
   =============  =========================  ==================================
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NCOMP}`
   =============  =========================  ==================================
   
``REACTION_MODEL``

   Specifies the type of reaction model of the combined bulk and particle volume. The model is configured in the subgroup :math:`\texttt{reaction}`.
   
   ================  ========================================  =============
   **Type:** string  **Range:** See Section :ref:`FFReaction`  **Length:** 1
   ================  ========================================  =============
   
``INIT_C``

   Initial concentrations for each component in the bulk mobile phase

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   ================  =========================  ===================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NCOMP}`
   ================  =========================  ===================================
   
``INIT_Q``

   Initial concentrations for each bound state of each component in the bead solid phase in component-major ordering

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{SP}}^{-3}`
   
   ================  =========================  =======================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NTOTALBND}`
   ================  =========================  =======================================
   
``INIT_STATE``

   Full state vector for initialization (optional, :math:`\texttt{INIT_C}` and :math:`\texttt{INIT_Q}` will be ignored; if length is :math:`2\texttt{NDOF}`, then the second half is used for time derivatives)

   **Unit:** :math:`various`
   
   ================  =============================  ===================================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NDOF} / 2\texttt{NDOF}`
   ================  =============================  ===================================================
   
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
   
``TOTAL_POROSITY``

   Total porosity
   
   ================  ========================  =============
   **Type:** double  **Range:** :math:`[0,1]`  **Length:** 1
   ================  ========================  =============
   
``VELOCITY``

   Interstitial velocity of the mobile phase (optional if :math:`\texttt{CROSS_SECTION_AREA}` is present, see Section :ref:`MUOPGRMflow`)

   **Unit:** :math:`\mathrm{m}\,\mathrm{s}^{-1}`
   
   ================  =============================  ======================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`1 / \texttt{NSEC}`
   ================  =============================  ======================================
   
``CROSS_SECTION_AREA``

   Cross section area of the column (optional if :math:`\texttt{VELOCITY}` is present, see Section :ref:`MUOPGRMflow`)

   **Unit:** :math:`\mathrm{m}^{2}`
   
   ================  =====================  =============
   **Type:** double  **Range:** :math:`>0`  **Length:** 1
   ================  =====================  =============
   

Group /input/model/unit_XXX/discretization - UNIT_TYPE = LUMPED_RATE_MODEL_WITHOUT_PORES
----------------------------------------------------------------------------------------
   
``USE_ANALYTIC_JACOBIAN``

   Determines whether analytically computed Jacobian matrix (faster) is used (value is 1) instead of Jacobians generated by algorithmic differentiation (slower, value is 0)
   
   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 1\}`  **Length:** 1
   =============  ===========================  =============

Spatial discretization - Numerical Methods
------------------------------------------

CADET offers two spatial discretization methods: Finite Volumes (FV) and Discontinuous Galerkin (DG). Each method has it's own set of input fields.
While both methods approximate the same solution to the same underlying model, they may differ in terms of computational performance.
With our currently implemented variants of FV and DG, FV perform better for solutions with steep gradients or discontinuities, while DG can be much faster for rather smooth solutions.
For the same number of discrete points, DG will generally be slower but often more accurate.

For further information on the choice of discretization methods and their parameters, see :ref:`spatial_discretization_methods`.

``SPATIAL_METHOD``

   Spatial discretization method. Optional, defaults to :math:`\texttt{FV}`

   ================  ===============================================  =============
   **Type:** string  **Range:** :math:`\{\texttt{FV}, \texttt{DG}\}`  **Length:** 1
   ================  ===============================================  =============

Finite Volumes (Default)
------------------------

``NCOL``

   Number of axial column discretization points
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``RECONSTRUCTION``

   Type of reconstruction method for fluxes only (only needs to be specified for FV)
   
   ================  ================================  =============
   **Type:** string  **Range:** :math:`\texttt{WENO}`  **Length:** 1
   ================  ================================  =============
   
For further discretization parameters, see also :ref:`flux_reconstruction_methods` (FV specific)), and :ref:`non_consistency_solver_parameters`.


Discontinuous Galerkin
----------------------

``POLYDEG``

   DG polynomial degree. Optional, defaults to 4 and :math:`N_d \in \{3, 4, 5\}` is recommended. The total number of axial discrete points is given by (``POLYDEG`` + 1 ) * ``NELEM``
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============

``NELEM``

   Number of axial column discretization DG cells\elements. The total number of axial discrete points is given by (``POLYDEG`` + 1 ) * ``NELEM``
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============

``NCOL``

   Number of axial discrete points. Optional and ignored if ``NELEM`` is defined. Otherwise, used to calculate ``NELEM`` = :math:`\lfloor` ``NCOL`` / (``POLYDEG`` + 1 ) :math:`\rfloor`
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============

``EXACT_INTEGRATION``

   Specifies the DG integration variant. Optional, defaults to 0
   
   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 1\}`  **Length:** 1
   =============  ===========================  =============
   
   For further discretization parameters, see also :ref:`non_consistency_solver_parameters`.
