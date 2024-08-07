.. _multi_component_sips_config:

Multi Component Sips
~~~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = MULTI_COMPONENT_SIPS**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}  		    **Length:** 1/NTOTALBND
===================  =========================  =========================================

``SIPS_KA``
   Adsorption rate constants

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SIPS_KD``
   Desorption rate constants

**Unit:** :math:`mol~m_{SP}^{-3}~s^{-1}`

===================  =========================  ==================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  ================================== 

``SIPS_QMAX``
   Maximum adsorption capacities

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  ==================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** NCOMP
===================  =========================  ================================== 

``SIPS_EXP``
   Freundlich-type exponent

===================  =========================  ==================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** NCOMP
===================  =========================  ================================== 

``SIPS_REFC0``
   Reference liquid phase concentration (optional, defaults to
   :math:`1.0`)


**Unit:** :math:`mol~m_{MP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================

``SIPS_REFQ``
   Reference solid phase concentration (optional, defaults to
   :math:`1.0`)


**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================

``SIPS_LINEAR_THRESHOLD``
   Threshold for linearization. Originally, the liquid phase concentration
   enters the adsorption rate via a power term. Below this threshold, a
   quadratic approximation is used instead. This ensures numerical stability.

**Unit:** :math:`mol~m_{MP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================
