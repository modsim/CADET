.. _multi_channel_transport_model_config:

Multi Channel Transport model (MCT) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``UNIT_TYPE``

   Specifies the type of unit operation model
   
   ================  ===================================================  =============
   **Type:** string  **Range:** :math:`\texttt{MULTI_CHANNEL_TRANSPORT}`  **Length:** 1
   ================  ===================================================  =============
   
``NCOMP``

   Number of chemical components in the chromatographic medium
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``REACTION_MODEL_BULK``

   Specifies the type of reaction model of the bulk volume. The model is configured in the subgroup :math:`\texttt{reaction_bulk}`.
   
   ================  ========================================  =============
   **Type:** string  **Range:** See Section :ref:`FFReaction`  **Length:** 1
   ================  ========================================  =============
   
``INIT_C``

   Initial concentrations for each component in all radial zones the bulk mobile phase (length :math:`\texttt{NCOMP}`), or for each component in each radial zone (length :math:`\texttt{NCOMP} \cdot \texttt{NRAD}`, ordering radial-major)

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   ================  =========================  =========================================================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NCOMP} / \texttt{NCOMP} \cdot \texttt{NRAD}`
   ================  =========================  =========================================================================
   
``INIT_STATE``

   Full state vector for initialization (optional, :math:`\texttt{INIT_C}` will be ignored; if length is :math:`2\texttt{NDOF}`, then the second half is used for time derivatives)

   **Unit:** :math:`various`
   
   ================  =============================  ==================================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NDOF} / 2\texttt{NDOF}`
   ================  =============================  ==================================================
   
``COL_DISPERSION``

   Axial dispersion coefficient.  In case of a spatially inhomogeneous setting, the :math:`\texttt{SENS_PARTYPE}` field is used for indexing the radial zone when specifying parameter sensitivities.

   **Unit:** :math:`\mathrm{m}_{\mathrm{IV}}^{2}\,\mathrm{s}^{-1}`
   
   ================  =========================  =========================================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** see :math:`\texttt{COL_DISPERSION_MULTIPLEX}`
   ================  =========================  =========================================================
   
``COL_DISPERSION_MULTIPLEX``

   Multiplexing mode of :math:`\texttt{COL_DISPERSION}`. Determines whether :math:`\texttt{COL_DISPERSION}` is treated as component-, radial-, and/or section-independent.  This field is optional. When left out, multiplexing behavior is inferred from the length of :math:`\texttt{COL_DISPERSION}`.  Valid modes are: 

  1. Component-independent, radial-independent, section-independent; length of :math:`\texttt{COL_DISPERSION}` is 1 
  2. Component-independent, radial-dependent, section-independent; length of :math:`\texttt{COL_DISPERSION}` is :math:`\texttt{NRAD}` 
  3. Component-dependent, radial-independent, section-independent; length of :math:`\texttt{COL_DISPERSION}` is :math:`\texttt{NCOMP}` 
  4. Component-dependent, radial-dependent, section-independent; length of :math:`\texttt{COL_DISPERSION}` is :math:`\texttt{NCOMP} \cdot \texttt{NRAD}`; ordering is radial-major 
  5. Component-independent, radial-independent, section-dependent; length of :math:`\texttt{COL_DISPERSION}` is :math:`\texttt{NSEC}` 
  6. Component-independent, radial-dependent, section-dependent; length of :math:`\texttt{COL_DISPERSION}` is :math:`\texttt{NRAD} \cdot \texttt{NSEC}`; ordering is section-major 
  7. Component-dependent, radial-independent, section-independent; length of :math:`\texttt{COL_DISPERSION}` is :math:`\texttt{NCOMP} \cdot \texttt{NSEC}`; ordering is section-major 
  8. Component-dependent, radial-dependent, section-dependent; length of :math:`\texttt{COL_DISPERSION}` is :math:`\texttt{NCOMP} \cdot \texttt{NRAD} \cdot \texttt{NSEC}`; ordering is section-radial-major 
   
   =============  ===================================  =============
   **Type:** int  **Range:** :math:`\{0, \dots, 7 \}`  **Length:** 1
   =============  ===================================  =============
   
``COL_LENGTH``

   Column length

   **Unit:** :math:`\mathrm{m}`
   
   ================  ======================  =============
   **Type:** double  **Range:** :math:`> 0`  **Length:** 1
   ================  ======================  =============
   
``COL_RADIUS``

   Column radius

   **Unit:** :math:`\mathrm{m}`
   
   ================  ======================  =============
   **Type:** double  **Range:** :math:`> 0`  **Length:** 1
   ================  ======================  =============
   
``COL_POROSITY``

   Column porosity, either constant (length is 1) or for each radial zone (length is :math:`\texttt{NRAD}`).  In case of a spatially inhomogeneous setting, the :math:`\texttt{SENS_PARTYPE}` field is used for indexing the radial zone when specifying parameter sensitivities.
   
   ================  ========================  =====================================
   **Type:** double  **Range:** :math:`(0,1]`  **Length:** :math:`1 / \texttt{NRAD}`
   ================  ========================  =====================================
   
``FILM_DIFFUSION``

   Film diffusion coefficients for each component of each particle type

   **Unit:** :math:`\mathrm{m}\,\mathrm{s}^{-1}`
   
   ================  =========================  =======================================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** see :math:`\texttt{FILM_DIFFUSION_MULTIPLEX}`
   ================  =========================  =======================================================
   
``FILM_DIFFUSION_MULTIPLEX``

   Multiplexing mode of :math:`\texttt{FILM_DIFFUSION}`. Determines whether :math:`\texttt{FILM_DIFFUSION}` is treated as component-, type-, and/or section-independent.  This field is optional. When left out, multiplexing behavior is inferred from the length of :math:`\texttt{FILM_DIFFUSION}`.  Valid modes are: 

  0. Component-dependent, type-independent, section-independent; length of :math:`\texttt{FILM_DIFFUSION}` is :math:`\texttt{NCOMP}` 
  1. Component-dependent, type-independent, section-dependent; length of :math:`\texttt{FILM_DIFFUSION}` is :math:`\texttt{NCOMP} \cdot \texttt{NSEC}`; ordering is section-major 
  2. Component-dependent, type-dependent, section-independent; length of :math:`\texttt{FILM_DIFFUSION}` is :math:`\texttt{NCOMP} \cdot \texttt{NPARTYPE}`; ordering is type-major 
  3. Component-dependent, type-dependent, section-dependent; length of :math:`\texttt{FILM_DIFFUSION}` is :math:`\texttt{NCOMP} \cdot \texttt{NPARTYPE} \cdot \texttt{NSEC}`; ordering is section-type-major 
   
   =============  ===================================  =============
   **Type:** int  **Range:** :math:`\{0, \dots, 3 \}`  **Length:** 1
   =============  ===================================  =============
   
``VELOCITY``

   Indicates flow direction in each radial zone (forward if value is positive, backward if value is negative), see Section :ref:`MUOPGRMflow2D`).  In case of a spatially inhomogeneous setting, the :math:`\texttt{SENS_PARTYPE}` field is used for indexing the radial cell when specifying parameter sensitivities.
   
   ================  =============================  ===================================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** see :math:`\texttt{VELOCITY_MULTIPLEX}`
   ================  =============================  ===================================================
   
``VELOCITY_MULTIPLEX``

   Multiplexing mode of :math:`\texttt{VELOCITY}`. Determines whether :math:`\texttt{VELOCITY}` is treated as radial- and/or section-independent.  This field is optional. When left out, multiplexing behavior is inferred from the length of :math:`\texttt{VELOCITY}`.  Valid modes are: 

  1. Radial-independent, section-independent; length of :math:`\texttt{VELOCITY}` is 1 
  2. Radial-dependent, section-independent; length of :math:`\texttt{VELOCITY}` is :math:`\texttt{NRAD}` 
  3. Section-dependent; length of :math:`\texttt{VELOCITY}` is :math:`\texttt{NSEC}` 
  4. Radial-dependent, section-dependent; length of :math:`\texttt{VELOCITY}` is :math:`\texttt{NRAD} \cdot \texttt{NSEC}`; ordering is section-major 
   
   =============  ===================================  =============
   **Type:** int  **Range:** :math:`\{0, \dots, 3 \}`  **Length:** 1
   =============  ===================================  =============

``EXCHANGE_MATRIX``

   Ordered list containing all exchange rates :math:`e_{ij}` from compartment :math:`i` to :math:`j` based on the exchange matrix :math:`E`. One row after the other is noted in the list. 

   .. math::
    
    E=\begin{bmatrix} 
    0 & e_{12} & \dots & e_{1N} \\
    e_{21} & \ddots & & \vdots\\
    \vdots & & \ddots & e_{(N-1)N}\\
    e_{N1} & \dots & e_{N(N-1)} & 0 
    \end{bmatrix}    

   ================  ========================  ===============================================
   **Type:** double  **Range:** :math:`[0,1]`  **Length:** :math:`\texttt{NRAD}*\texttt{NRAD}`
   ================  ========================  ===============================================

``NCOL``

   Number of axial column discretization cells
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``NRAD``

   Number of radial column discretization cells
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
      
``RADIAL_DISC_TYPE``

   Specifies the radial discretization scheme. Valid values are :math:`\texttt{EQUIDISTANT}`, :math:`\texttt{EQUIVOLUME}`, and :math:`\texttt{USER_DEFINED}`.
   
   ================  =============
   **Type:** string  **Length:** 1
   ================  =============
   
``RADIAL_COMPARTMENTS``

   Coordinates for the radial compartment boundaries (ignored if :math:`\texttt{RADIAL_DISC_TYPE} \neq \texttt{USER_DEFINED}`). The coordinates are absolute and have to include the endpoints 0 and :math:`\texttt{COLUMN_RADIUS}`. The values are expected in ascending order (i.e., 0 is the first and :math:`\texttt{COLUMN_RADIUS}` the last value in the array).

   **Unit:** :math:`\mathrm{m}`
   
   ================  =============================================  ====================================
   **Type:** double  **Range:** :math:`[0,\texttt{COLUMN_RADIUS}]`  **Length:** :math:`\texttt{NRAD} + 1`
   ================  =============================================  ====================================
   
``USE_ANALYTIC_JACOBIAN``

   Determines whether analytically computed Jacobian matrix (faster) is used (value is 1) instead of Jacobians generated by algorithmic differentiation (slower, value is 0)
   
   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 1\}`  **Length:** 1
   =============  ===========================  =============
   
``LINEAR_SOLVER_BULK``

   Linear solver used for the sparse column bulk block. This field is optional, the best available method is selected (i.e., sparse direct solver if possible).  Valid values are: 

  \.exttt{DENSE}] Converts the sparse matrix into a banded matrix and uses regular LAPACK. Slow and memory intensive, but always available. 
  \.exttt{UMFPACK}] Uses the UMFPACK sparse direct solver (LU decomposition) from SuiteSparse. Fast, but has to be enabled when compiling and requires UMFPACK library. 
  \.exttt{SUPERLU}] Uses the SuperLU sparse direct solver (LU decomposition). Fast, but has to be enabled when compiling and requires SuperLU library. 
   
   ================  =======================================================================  =============
   **Type:** string  **Range:** :math:`\{\texttt{DENSE},\texttt{UMFPACK},\texttt{SUPERLU}\}`  **Length:** 1
   ================  =======================================================================  =============
   
``RECONSTRUCTION``

   Type of reconstruction method for fluxes
   
   ================  ================================  =============
   **Type:** string  **Range:** :math:`\texttt{WENO}`  **Length:** 1
   ================  ================================  =============
   
``GS_TYPE``

   Type of Gram-Schmidt orthogonalization, see IDAS guide Section~4.5.7.3, p.~41f. A value of 0 enables classical Gram-Schmidt, a value of 1 uses modified Gram-Schmidt.
   
   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 1\}`  **Length:** 1
   =============  ===========================  =============
   
``MAX_KRYLOV``

   Defines the size of the Krylov subspace in the iterative linear GMRES solver (0: :math:`\texttt{MAX_KRYLOV} = \texttt{NCOL} \cdot \texttt{NRAD} \cdot \texttt{NCOMP} \cdot \texttt{NPARTYPE}`)
   
   =============  ================================================================================================================  =============
   **Type:** int  **Range:** :math:`\{0, \dots, \texttt{NCOL} \cdot \texttt{NRAD} \cdot \texttt{NCOMP} \cdot \texttt{NPARTYPE} \}`  **Length:** 1
   =============  ================================================================================================================  =============
   
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


