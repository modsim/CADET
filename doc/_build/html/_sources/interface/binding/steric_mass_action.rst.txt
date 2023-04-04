.. _steric_mass_action_config:

Steric Mass Action
~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = STERIC_MASS_ACTION**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}           **Length:** 1/NTOTALBND
===================  =========================  =========================================

``SMA_KA``
   Adsorption rate constants

**Unit:** :math:`m_{MP}^3~m_{SP}^{-3}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SMA_KD``
   Desorption rate constants


**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SMA_NU``
   Characteristic charges of the protein; The number of sites
   :math:`\nu` that the protein interacts with on the resin surface

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SMA_SIGMA``
   Steric factors of the protein; The number of sites :math:`\sigma` on
   the surface that are shielded by the protein and prevented from
   exchange with the salt counterions in solution

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SMA_LAMBDA``
   Stationary phase capacity (monovalent salt counterions); The total
   number of binding sites available on the resin surface

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================

``SMA_REFC0``
   Reference liquid phase concentration (optional, defaults to
   :math:`1.0`)


**Unit:** :math:`mol~m_{MP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================

``SMA_REFQ``
   Reference solid phase concentration (optional, defaults to
   :math:`1.0`)


**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================
