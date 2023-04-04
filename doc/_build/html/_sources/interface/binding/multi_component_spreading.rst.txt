.. _multi_component_spreading_config:

Multi Component Spreading
~~~~~~~~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = MULTI_COMPONENT_SPREADING**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}  		**Length:** 1/NTOTALBND
===================  =========================  =========================================

``MCSPR_KA``
   Adsorption rate constants in state-major ordering

**Unit:** :math:`m_{MP}^3~mol^{-1}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``MCSPR_KD``
   Desorption rate constants in state-major ordering

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``MCSPR_QMAX``
   Maximum adsorption capacities in state-major ordering

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``MCSPR_K12``
   Exchange rates from the first to the second bound state

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``MCSPR_K21``
   Exchange rates from the second to the first bound state

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================
