.. _saska_config:

Saska
~~~~~

**Group /input/model/unit_XXX/adsorption – ADSORPTION_MODEL = SASKA**


``IS_KINETIC``
   Selects kinetic or quasi-stationary adsorption mode: 1 = kinetic, 0 =
   quasi-stationary. If a single value is given, the mode is set for all
   bound states. Otherwise, the adsorption mode is set for each bound
   state separately.

===================  =========================  =========================================
**Type:** int        **Range:** {0,1}           **Length:** 1/NTOTALBND
===================  =========================  =========================================


``SASKA_H``
   Henry coefficient

**Unit:** :math:`m_{MP}^3~m_{SP}^{-3}~s^{-1}`

===================  =================================  =========================================
**Type:** double     **Range:** :math:`\mathbb {R}`     **Length:** NCOMP
===================  =================================  =========================================


``SASKA_K``
   Quadratic factors

**Unit:** :math:`m_{MP}^6~m_{SP}^{-3}~s^{-1}`

===================  ================================  =========================================
**Type:** double     **Range:** :math:`\mathbb {R}`    **Length:** :math:`\text{NCOMP}^2`
===================  ================================  =========================================



