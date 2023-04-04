.. _bi_steric_mass_action_config:

Bi Steric Mass Action
~~~~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = BI_STERIC_MASS_ACTION**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =======================
**Type:** int        **Range:** {0,1}           **Length:** 1/NTOTALBND
===================  =========================  =======================

``BISMA_KA``
   Adsorption rate constants in state-major ordering

**Unit:** :math:`m_{MP}^{3}~m_{SP}^{-3}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NSTATES :math:`\cdot` NCOMP
===================  =========================  =========================================

``BISMA_KD``
   Desorption rate constants in state-major ordering

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NSTATES :math:`\cdot` NCOMP
===================  =========================  =========================================

``BISMA_NU``
   Characteristic charges :math:`\nu_{i,j}` of the :math:`i`\ th protein
   with respect to the :math:`j`\ th binding site type in state-major
   ordering

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NSTATES :math:`\cdot` NCOMP
===================  =========================  =========================================

``BISMA_SIGMA``
   Steric factors :math:`\sigma_{i,j}` of the :math:`i`\ th protein with
   respect to the :math:`j`\ th binding site type in state-major
   ordering

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NSTATES :math:`\cdot` NCOMP
===================  =========================  =========================================

``BISMA_LAMBDA``
   Stationary phase capacity (monovalent salt counterions) of the
   different binding site types :math:`\lambda_j`

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  ===============================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NSTATES 
===================  =========================  ===============================

``BISMA_REFC0``
   Reference liquid phase concentration for each binding site type or
   one value for all types (optional, defaults to :math:`1.0`)

**Unit:** :math:`mol~m_{MP}^{-3}`

===================  =========================  ===============================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** {1,NSTATES} 
===================  =========================  ===============================

``BISMA_REFQ``
   Reference solid phase concentration for each binding site type or one
   value for all types (optional, defaults to :math:`1.0`)

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  ===============================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** {1,NSTATES} 
===================  =========================  ===============================
