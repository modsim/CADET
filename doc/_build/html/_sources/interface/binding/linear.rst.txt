.. _linear_config:

Linear
~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = LINEAR**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}           **Length:** 1/NTOTALBND
===================  =========================  =========================================

``LIN_KA``
   Adsorption rate constants for each component


**Unit:** :math:`m_{MP}^3~m_{SP}^{-3}~s^{-1}`

===================  =========================  ==================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1/NTOTALBND
===================  =========================  ==================================  


``LIN_KD``
   Desorption rate constants for each component

**Unit:** :math:`s^{-1}`

===================  =========================  ==================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1/NTOTALBND
===================  =========================  ==================================  




     
