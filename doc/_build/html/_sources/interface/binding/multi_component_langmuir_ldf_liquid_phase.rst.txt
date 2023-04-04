.. _multi_component_langmuir_ldf_liquid_phase_config:

Multi Component Langmuir LDF Liquid Phase
==========================================

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = MULTI_COMPONENT_LANGMUIR_LDF_LIQUID_PHASE**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}  		    **Length:** 1/NTOTALBND
===================  =========================  =========================================

``MCLLDFC_KEQ``
   Equillibrium loading constants

**Unit:** :math:`m_{MP}^3~mol^{-1}`

===================  =========================  =========================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  =========================================

``MCLLDFC_KKIN``
   Linear driving force coefficients

**Unit:** :math:`s^{-1}`

===================  =========================  ==================================
**Type:** double     **Range:** :math:`\ge 0`   **Length:** NCOMP
===================  =========================  ================================== 

``MCLLDFC_QMAX``
   Maximum adsorption capacities

**Unit:** :math:`mol~m_{SP}^{-3}`

===================  =========================  ==================================
**Type:** double     **Range:** :math:`\gt 0`   **Length:** NCOMP
===================  =========================  ================================== 
