.. _extended_mobile_phase_modulator_langmuir_config:

Extended Mobile Phase Modulator Langmuir
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = EXTENDED_MOBILE_PHASE_MODULATOR**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}           **Length:** 1/NTOTALBND
===================  =========================  =========================================

``EMPM_COMP_MODE``
   Determines the mode of each component (:math:`0` denotes the modifier
   component, :math:`1` is linear binding, :math:`2` is modified Langmuir
   binding). At most one modifier component is allowed, that is, a
   modifier is not required.

   Note that this field has the same name for the externally dependent
   variant of the model.

===================  ============================  =========================================
**Type:** int        **Range:** :math:`\{0,1,2\}`   **Length:** NCOMP
===================  ============================  =========================================

``EMPM_KA``
   Adsorption rate constants

**Unit:** :math:`m_{MP}^3~mol^{-1}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``EMPM_KD``
   Desorption rate constants

**Unit:** :math:`m_{MP}^{3\beta}~mol^{-\beta}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``EMPM_QMAX``
   Maximum adsorption capacities


**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``EMPM_BETA``
   Parameters describing the ion-exchange characteristics (IEX)

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``EMPM_GAMMA``
   Parameters describing the hydrophobicity (HIC)

**Unit:** :math:`m_{MP}^{3} mol^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================
