.. _multi_component_anti_langmuir_config:

Multi Component Anti-Langmuir
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = MULTI_COMPONENT_ANTILANGMUIR**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}           **Length:** 1/NTOTALBND
===================  =========================  =========================================

``MCAL_KA``
   Adsorption rate constants

**Unit:** :math:`m_{MP}^3~mol^{-1}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``MCAL_KD``
   Desorption rate constants

**Unit:** :math:`s^{-1}`

===================  =========================  ==================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  ================================== 

``MCAL_QMAX``
   Maximum adsorption capacities

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  ==================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** NCOMP
===================  =========================  ================================== 

``MCAL_ANTILANGMUIR``
   Anti-Langmuir coefficients (optional)

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  ==================================
**Type:** double     **Range:** {-1,1}          **Length:** NCOMP
===================  =========================  ================================== 
