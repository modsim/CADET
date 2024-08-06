.. _parameter_dependencies:

Parameter Dependencies
======================

Some parameters depend on other parameters (parameter-parameter dependency) or the solution variables (parameter-state dependency).
Parameter dependencies are defined in the unit operation scope.

Parameter-Parameter Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Group /input/model/unit_XXX
---------------------------

``COL_DISPERSION_DEP``

   Parameter dependence of column dispersion on the interstitial velocity. Available for the LRM, LRMP and GRM units (with FV discretization only at the moment)
   
   ================  =====================================  =============
   **Type:** string  **Range:** :math:`\texttt{POWER_LAW}`  **Length:** 1
   ================  =====================================  =============

``FILM_DIFFUSION_DEP``

   Parameter dependence of film diffusion on the interstitial velocity. Available for the LRMP unit (with FV discretization only at the moment)
   
   ================  =====================================  =============
   **Type:** string  **Range:** :math:`\texttt{POWER_LAW}`  **Length:** 1
   ================  =====================================  =============


**Correlations**
""""""""""""""""

Different types of parameter correlations are can be applied.
The following correlations can be used for all parameter-parameter dependencies, but we specify the required input fields only for ``COL_DISPERSION_DEP``, for the sake of conciseness.

**Power Law**

.. math::

    \begin{aligned}
        p_{dep} &= p_{dep} \cdot b \ |p_{on}^x|
    \end{aligned}

Here, :math:`p_{dep}` is the dependent parameter and :math:`p_{on}` is the parameter it depends on.

``COL_DISPERSION_DEP_BASE``

   Base :math:`b` of the power law parameter dependence. Optional, defaults to :math:`1.0`
   
   ================  =============================  =============
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** 1
   ================  =============================  =============

``COL_DISPERSION_DEP_EXPONENT``

   Exponent :math:`x` of the power law parameter dependence
   
   ================  =============================  =============
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** 1
   ================  =============================  =============

``COL_DISPERSION_DEP_ABS``

   Specifies whether or not the absolute value should be computed. Optional, defaults to :math:`1`
   
   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 1\}`  **Length:** 1
   =============  ===========================  =============


Parameter-State Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, the dependence of surface diffusion on the particle liquid salt component is the only available parameter-state dependence.

Group /input/model/unit_XXX
---------------------------

``PAR_SURFDIFFUSION_DEP``

   Parameter dependence of :math:`\texttt{PAR_SURFDIFFUSION}` on the particle liquid salt component (i.e. component with index 0). Valid dependencies are:

   - :math:`\texttt{NONE}` Original parameter is used unmodified.
   - :math:`\texttt{LIQUID_SALT_EXPONENTIAL}` Original parameter is modified by exponential law of liquid phase salt concentration.
   - :math:`\texttt{LIQUID_SALT_POWER}` Original parameter is modified by power law of liquid phase salt concentration.
   - :math:`\texttt{LIQUID_SALT_COLLOIDAL_AFFINITY}` Original parameter is modified by colloidal binding affinity based on liquid phase salt concentration.

   Optional: If left out, no parameter dependence is assumed and the original surface diffusion coefficients are used unmodified.
   
   ================  =========================================
   **Type:** string  **Length:** :math:`1 / \texttt{NPARTYPE}`
   ================  =========================================

``PAR_SURFDIFFUSION_EXPFACTOR``

   Factor :math:`\texttt{p1}` in exponential law particle surface diffusion relation
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} p_{1, i, m} exp \left(p_{2, i, m} c_{0}^{p} \right)`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient. Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_EXPONENTIAL}`.
   
   ================  =========================  ===================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NBOUND}`
   ================  =========================  ===================================

``PAR_SURFDIFFUSION_EXPARGMULT``

   Factor :math:`\texttt{p2}` in exponential law particle surface diffusion relation
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} p_{1, i, m} exp \left(p_{2, i, m} c_{0}^{p} \right)`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient. Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_EXPONENTIAL}`.
   
   ================  =============================  ===================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NBOUND}`
   ================  =============================  ===================================

``PAR_SURFDIFFUSION_POWFACTOR``

   Factor :math:`\texttt{p1}` in power law particle surface diffusion relation
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} p_{1, i, m} \left( c_{0}^{p} \right)^{p_{2, i, m}}`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient. Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_POWER}`.
   
   ================  =========================  ===================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NBOUND}`
   ================  =========================  ===================================

``PAR_SURFDIFFUSION_POWEXP``

   Fjactor :math:`\texttt{p2}` in power law particle surface diffusion relation
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} p_{1, i, m} \left( c_{0}^{p} \right)^{p_{2, i, m}}`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient. Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_POWER}`.
   
   ================  =============================  ===================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NBOUND}`
   ================  =============================  ===================================

``PAR_SURFDIFFUSION_LOGKEQFACTOR``

   Factor :math:`\texttt{p1}` in colloidal affinity law particle surface diffusion relation
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} \left[  p_{4, i, m} \left( k_{i, m} \left( c_{0}^{p} \right) \right)^{p_{5, i, m}} p_{6, i, m} exp \left( p_{7, i, m} k_{i, m} \left( c_{0}^{p} \right) \right) \right]`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient and 
   :math:`k_{i, m} \left( c_{0}^{p} \right) = p_{1, i, m}\left( c_{0}^{p} \right)^{p_{2, i, m}} + p_{3, i, m}`.
   Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_COLLOIDAL_AFFINITY}`.
   
   ================  =============================  ===================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NBOUND}`
   ================  =============================  ===================================

``PAR_SURFDIFFUSION_LOGKEQEXP``

   Factor :math:`\texttt{p2}` in colloidal affinity law particle surface diffusion relation
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} \left[  p_{4, i, m} \left( k_{i, m} \left( c_{0}^{p} \right) \right)^{p_{5, i, m}} p_{6, i, m} exp \left( p_{7, i, m} k_{i, m} \left( c_{0}^{p} \right) \right) \right]`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient and 
   :math:`k_{i, m} \left( c_{0}^{p} \right) = p_{1, i, m}\left( c_{0}^{p} \right)^{p_{2, i, m}} + p_{3, i, m}`.
   Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_COLLOIDAL_AFFINITY}`.
   
   ================  =============================  ===================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NBOUND}`
   ================  =============================  ===================================

``PAR_SURFDIFFUSION_LOGKEQCONST``

   Factor :math:`\texttt{p3}` in colloidal affinity law particle surface diffusion relation
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} \left[  p_{4, i, m} \left( k_{i, m} \left( c_{0}^{p} \right) \right)^{p_{5, i, m}} p_{6, i, m} exp \left( p_{7, i, m} k_{i, m} \left( c_{0}^{p} \right) \right) \right]`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient and 
   :math:`k_{i, m} \left( c_{0}^{p} \right) = p_{1, i, m}\left( c_{0}^{p} \right)^{p_{2, i, m}} + p_{3, i, m}`.
   Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_COLLOIDAL_AFFINITY}`.
   
   ================  =============================  ===================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NBOUND}`
   ================  =============================  ===================================

``PAR_SURFDIFFUSION_POWFACTOR``

   Factor :math:`\texttt{p4}` in colloidal affinity law particle surface diffusion relation
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} \left[  p_{4, i, m} \left( k_{i, m} \left( c_{0}^{p} \right) \right)^{p_{5, i, m}} p_{6, i, m} exp \left( p_{7, i, m} k_{i, m} \left( c_{0}^{p} \right) \right) \right]`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient and 
   :math:`k_{i, m} \left( c_{0}^{p} \right) = p_{1, i, m}\left( c_{0}^{p} \right)^{p_{2, i, m}} + p_{3, i, m}`.
   Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_COLLOIDAL_AFFINITY}`.
   
   ================  =============================  ===================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NBOUND}`
   ================  =============================  ===================================

``PAR_SURFDIFFUSION_POWEXP``

   Factor :math:`\texttt{p5}` in colloidal affinity law particle surface diffusion relation
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} \left[  p_{4, i, m} \left( k_{i, m} \left( c_{0}^{p} \right) \right)^{p_{5, i, m}} p_{6, i, m} exp \left( p_{7, i, m} k_{i, m} \left( c_{0}^{p} \right) \right) \right]`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient and 
   :math:`k_{i, m} \left( c_{0}^{p} \right) = p_{1, i, m}\left( c_{0}^{p} \right)^{p_{2, i, m}} + p_{3, i, m}`.
   Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_COLLOIDAL_AFFINITY}`.
   
   ================  =============================  ===================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NBOUND}`
   ================  =============================  ===================================

   ``PAR_SURFDIFFUSION_EXPFACTOR``
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} \left[  p_{4, i, m} \left( k_{i, m} \left( c_{0}^{p} \right) \right)^{p_{5, i, m}} p_{6, i, m} exp \left( p_{7, i, m} k_{i, m} \left( c_{0}^{p} \right) \right) \right]`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient and 
   :math:`k_{i, m} \left( c_{0}^{p} \right) = p_{1, i, m}\left( c_{0}^{p} \right)^{p_{2, i, m}} + p_{3, i, m}`.
   Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_COLLOIDAL_AFFINITY}`.
   
   ================  =============================  ===================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NBOUND}`
   ================  =============================  ===================================

``PAR_SURFDIFFUSION_POWEXP``

   Factor :math:`\texttt{p5}` in colloidal affinity law particle surface diffusion relation
   :math:`D_{s, i, m} = \tilde{D}_{s, i, m} \left[  p_{4, i, m} \left( k_{i, m} \left( c_{0}^{p} \right) \right)^{p_{5, i, m}} p_{6, i, m} exp \left( p_{7, i, m} k_{i, m} \left( c_{0}^{p} \right) \right) \right]`
   where :math:`\tilde{D}_{s, i, m}` is the original surface diffusion coefficient and 
   :math:`k_{i, m} \left( c_{0}^{p} \right) = p_{1, i, m}\left( c_{0}^{p} \right)^{p_{2, i, m}} + p_{3, i, m}`.
   Only required if :math:`\texttt{PAR_SURFDIFFUSION_DEP}` is :math:`\texttt{LIQUID_SALT_COLLOIDAL_AFFINITY}`.
   
   ================  =============================  ===================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NBOUND}`
   ================  =============================  ===================================
