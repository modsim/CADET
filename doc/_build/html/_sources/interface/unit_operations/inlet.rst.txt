.. _inlet_config:


Inlet
=====

Group /input/model/unit_XXX - UNIT-TYPE = INLET
-----------------------------------------------


``UNIT_TYPE``

   Specifies the type of unit operation model
   
   ================  =================================  =============
   **Type:** string  **Range:** :math:`\texttt{INLET}`  **Length:** 1
   ================  =================================  =============
   
``NCOMP``

   Number of chemical components in the chromatographic medium
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``INLET_TYPE``

   Specifies the type of inlet profile
   
   ================  ================================================  =============
   **Type:** string  **Range:** :math:`\texttt{PIECEWISE_CUBIC_POLY}`  **Length:** 1
   ================  ================================================  =============

Group /input/model/unit_XXX/sec_XXX
-----------------------------------

``CONST_COEFF``

   Constant coefficients for inlet concentrations

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   ================  =============================  ==================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NCOMP}`
   ================  =============================  ==================================
   
``LIN_COEFF``

   Linear coefficients for inlet concentrations

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}`
   
   ================  =============================  ==================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NCOMP}`
   ================  =============================  ==================================
   
``QUAD_COEFF``

   Quadratic coefficients for inlet concentrations

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-2}`
   
   ================  =============================  ==================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NCOMP}`
   ================  =============================  ==================================
   
``CUBE_COEFF``

   Cubic coefficients for inlet concentrations

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-3}`
   
   ================  =============================  ==================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\texttt{NCOMP}`
   ================  =============================  ==================================


