.. _multi_state_steric_mass_action_config:

Multi-State Steric Mass Action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = MULTISTATE_STERIC_MASS_ACTION**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}  		**Length:** 1/NTOTALBND
===================  =========================  =========================================

``MSSMA_KA``
   Adsorption rate constants of the components to the different bound
   states in component-major ordering

**Unit:** :math:`m_{MP}^3~m_{SP}^{-3}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``MSSMA_KD``
   Desorption rate constants of the components in the different bound
   states in component-major ordering

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``MSSMA_NU``
   Characteristic charges of the components in the different bound
   states in component-major ordering

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``MSSMA_SIGMA``
   Steric factors of the components in the different bound states in
   component-major ordering

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``MSSMA_RATES``
   Conversion rates between different bound states in
   component-row-major ordering

**Unit:** :math:`s^{-1}`

===================  =========================  ===================================================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** :math:`\sum_{i=0}^{\text{NCOMP}-1} \text{NBND}_{i}^{2}`
===================  =========================  ===================================================================

``MSSMA_LAMBDA``
   Stationary phase capacity (monovalent salt counterions); The total
   number of binding sites available on the resin surface

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================

``MSSMA_REFC0``
   Reference liquid phase concentration (optional, defaults to
   :math:`1.0`)

**Unit:** :math:`mol~m_{MP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================

``MSSMA_REFQ``
   Reference solid phase concentration (optional, defaults to
   :math:`1.0`)

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================
