.. _multi_component_colloidal_config:

Multi Component Colloidal
~~~~~~~~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = MULTI_COMPONENT_COLLOIDAL**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}  		    **Length:** 1/NTOTALBND
===================  =========================  =========================================

``COL_PHI``
   Phase ratio

**Unit:** :math:`m^{2} m_{s}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================

``COL_KAPPA_EXP``
   Screening term exponent factor

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================

``COL_KAPPA_FACT``
   Screening term factor

**Unit:** :math:`nm \cdot mM^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================

``COL_KAPPA_CONST``
   Screening term constant

**Unit:** :math:`nm`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================

``COL_CORDNUM``
   Coordination number

===================  =========================  =========================================
**Type:** int        **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================

``COL_LOGKEQ_PH_EXP``
   Equilibrium constant factor exponent term for pH

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_LOGKEQ_SALT_POWEXP``
   Equilibrium constant power exponent term for salt

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_LOGKEQ_SALT_POWFAC``
   Equilibrium constant power factor term for salt

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_LOGKEQ_SALT_EXPFAC``
   Equilibrium constant exponent factor term for salt

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_LOGKEQ_SALT_EXPARGMULT``
   Equilibrium constant exponent multiplier term for salt

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_BPP_PH_EXP``
   BPP constant exponent factor term for pH

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_BPP_SALT_POWEX``
   BPP constant power exponent term for salt

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_BPP_SALT_POWFACT``
   BPP constant power factor term for salt

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_BPP_SALT_EXPFACT``
   BPP constant exponent factor term for salt

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_BPP_SALT_EXPARGMUL``
   BPP constant exponent multiplier term for salt

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_RADIUS``
   Protein radius

**Unit:** :math:`m`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_KKIN``
   Adsorption rate constants in state-major ordering

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``COL_LINEAR_THRESHOLD``
   Threshold concentration for switching to linear implementation

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================

``COL_USE_PH``
   Selects if pH is included in the model or not: 1 = yes, 0 = no.

===================  =========================  =========================================
**Type:** int        **Range:** :math:`{0,1}`   **Length:** 1
===================  =========================  =========================================

