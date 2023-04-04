.. _multi_component_bi_langmuir_config:

Multi Component Bi-Langmuir
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = MULTI_COMPONENT_BILANGMUIR**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}           **Length:** 1/NTOTALBND
===================  =========================  =========================================

``MCBL_KA``
   Adsorption rate constants in state-major ordering (see :ref:`ordering_multi_dimensional_data`)

**Unit:** :math:`m_{MP}^3~mol^{-1}~s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NSTATES :math:`\cdot` NCOMP
===================  =========================  =========================================

``MCBL_KD``
   Desorption rate constants in state-major ordering

**Unit:** :math:`s^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NSTATES :math:`\cdot` NCOMP
===================  =========================  =========================================

``MCBL_QMAX``
   Maximum adsorption capacities in state-major ordering

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NSTATES :math:`\cdot` NCOMP
===================  =========================  =========================================

