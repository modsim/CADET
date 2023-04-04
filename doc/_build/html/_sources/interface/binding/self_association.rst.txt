.. _self_association_config:

Self Association
~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = SELF_ASSOCIATION**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}           **Length:** 1/NTOTALBND
===================  =========================  =========================================

``SAI_KA1``
   Adsorption rate constants

**Unit:** :math:`m_{MP}^3~m_{SP}^{-3}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SAI_KA2``
   Adsorption rate constants

**Unit:** :math:`m_{MP}^6~m_{SP}^{-3}~mol^{-1}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SAI_KD``
   Desorption rate constants

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SAI_NU``
   Characteristic charges :math:`\nu` of the protein

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SAI_SIGMA``
   Steric factors :math:`\sigma` of the protein

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SAI_LAMBDA``
   Stationary phase capacity (monovalent salt counterions); The total
   number of binding sites available on the resin surface

**Unit:** :math:`mol m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================


``SAI_REFC0``
   Reference liquid phase concentration (optional, defaults to
   :math:`1.0`)

**Unit:** :math:`mol m_{MP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================


``SAI_REFQ``
   Reference solid phase concentration (optional, defaults to
   :math:`1.0`)

**Unit:** :math:`mol m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================

