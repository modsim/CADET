.. _simplified_multi_state_steric_mass_action_config:

Simplified Multi-State Steric Mass Action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = SIMPLIFIED_MULTISTATE_STERIC_MASS_ACTION**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}  		**Length:** 1/NTOTALBND
===================  =========================  =========================================

``SMSSMA_LAMBDA``
   Stationary phase capacity (monovalent salt counterions); The total
   number of binding sites available on the resin surface

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** 1
===================  =========================  =========================================

``SMSSMA_KA``
   Adsorption rate constants of the components to the different bound
   states in component-major ordering

**Unit:** :math:`m_{MP}^{3}~m_{SP}^{-3}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``SMSSMA_KD``
   Desorption rate constants of the components to the different bound
   states in component-major ordering


**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NTOTALBND
===================  =========================  =========================================

``SMSSMA_NU_MIN``
   Characteristic charges of the components in the first (weakest) bound
   state

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SMSSMA_NU_MAX``
   Characteristic charges of the components in the last (strongest)
   bound state

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SMSSMA_NU_QUAD``
   Quadratic modifiers of the characteristic charges of the different
   components depending on the index of the bound state

===================  ==============================  =========================================
**Type:** double     **Range:** :math:`\mathbb{R}`   **Length:** NCOMP
===================  ==============================  =========================================

``SMSSMA_SIGMA_MIN``
   Steric factors of the components in the first (weakest) bound state

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SMSSMA_SIGMA_MAX``
   Steric factors of the components in the last (strongest) bound state

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SMSSMA_SIGMA_QUAD``
   Quadratic modifiers of steric factors of the different components
   depending on the index of the bound state

===================  ==============================  =========================================
**Type:** double     **Range:** :math:`\mathbb{R}`   **Length:** NCOMP
===================  ==============================  =========================================

``SMSSMA_KWS``
   Exchange rates from a weakly bound state to the next stronger bound
   state

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SMSSMA_KWS_LIN``
   Linear exchange rate coefficients from a weakly bound state to the
   next stronger bound state

**Unit:** :math:`s^{-1}`

===================  ==============================  =========================================
**Type:** double     **Range:** :math:`\mathbb{R}`   **Length:** NCOMP
===================  ==============================  =========================================

``SMSSMA_KWS_QUAD``
   Quadratic exchange rate coefficients from a weakly bound state to the
   next stronger bound state

**Unit:** :math:`s^{-1}`

===================  ==============================  =========================================
**Type:** double     **Range:** :math:`\mathbb{R}`   **Length:** NCOMP
===================  ==============================  =========================================

``SMSSMA_KSW``
   Exchange rates from a strongly bound state to the next weaker bound
   state

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``SMSSMA_KSW_LIN``
   Linear exchange rate coefficients from a strongly bound state to the
   next weaker bound state

**Unit:** :math:`s^{-1}`

===================  ==============================  =========================================
**Type:** double     **Range:** :math:`\mathbb{R}`   **Length:** NCOMP
===================  ==============================  =========================================

``SMSSMA_KSW_QUAD``
   Quadratic exchange rate coefficients from a strongly bound state to
   the next weaker bound state

**Unit:** :math:`s^{-1}`

===================  ==============================  =========================================
**Type:** double     **Range:** :math:`\mathbb{R}`   **Length:** NCOMP
===================  ==============================  =========================================

``SMSSMA_REFC0``
   Reference liquid phase concentration (optional, defaults to
   :math:`1.0`)

**Unit:** :math:`mol~m_{MP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================

``SMSSMA_REFQ``
   Reference solid phase concentration (optional, defaults to
   :math:`1.0`)

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** 1
===================  =========================  =========================================
