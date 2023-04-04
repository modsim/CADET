.. _FFModelSystem:

System of unit operations
=========================

Group /input/model
------------------

``NUNITS``

   Number of unit operations in the system
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``INIT_STATE_Y``

   Initial full state vector (optional, unit operation specific initial data is ignored)
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``INIT_STATE_YDOT``

   Initial full time derivative state vector (optional, unit operation specific initial data is ignored)
   
   ================  ==================================
   **Type:** double  **Length:** :math:`\texttt{NDOF}`
   ================  ==================================
   
``INIT_STATE_SENSY_XXX``

   Number of unit operations in the system
   
   ================  ==================================
   **Type:** double  **Length:** :math:`\texttt{NDOF}`
   ================  ==================================
   
``INIT_STATE_SENSYDOT_XXX``

   Initial full state vector of the :math:`\texttt{XXX}` th sensitivity system (optional, unit operation specific initial data is ignored)
   
   ================  ==================================
   **Type:** double  **Length:** :math:`\texttt{NDOF}`
   ================  ==================================
   
``NUNITS``

   Initial full time derivative state vector of the :math:`\texttt{XXX}` th sensitivity system (optional, unit operation specific initial data is ignored)
   
   ================  ==================================
   **Type:** double  **Length:** :math:`\texttt{NDOF}`
   ================  ==================================
   
.. _FFModelSystemConnections:

Group /input/model/connections
------------------------------

``NSWITCHES``

   Number of valve switches
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``CONNECTIONS_INCLUDE_PORTS``

   Determines whether the :math:`\texttt{CONNECTIONS}` table includes ports (:math:`1`) or not (:math:`0`). Optional, defaults to 0 unless a unit operation model with multiple ports is present.
   
   =============  ============================  =============
   **Type:** int  **Range:** :math:`\{ 0,1 \}`  **Length:** 1
   =============  ============================  =============
   
``CONNECTIONS_INCLUDE_DYNAMIC_FLOW``

   Determines whether the :math:`\texttt{CONNECTIONS}` table includes linear, quadratic, and cubic flow rate coefficients (1) or not (0). Optional, defaults to 0.
   
   =============  ============================  =============
   **Type:** int  **Range:** :math:`\{ 0,1 \}`  **Length:** 1
   =============  ============================  =============


.. _FFModelConnectionSwitch:

Group /input/model/connections/switch_XXX
-----------------------------------------

``SECTION``

   Index of the section that activates this connection set
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** 1
   =============  =========================  =============
   
``CONNECTIONS``

   Matrix with list of connections in row-major storage. Columns are *UnitOpID from*, *UnitOpID to*, *Port from*, *Port to*, *Component from*, *Component to*, *volumetric flow rate*, *linear flow rate coefficient*, *quadratic flow rate coefficient*, *cubic flow rate coefficient*. 
   If both port indices are :math:`-1`, all ports are connected. 
   If both component indices are :math:`-1`, all components are connected.  

   The flow rate is a cubic function of time,

   .. math::
      Q = Q_0 + Q_1(t - t_s) + Q_2(t-t_s)^2 + Q_3(t-t_s)^3,

   where :math:`t_s` is the beginning of the section that activates the switch (i.e., :math:`\texttt{SECTION_TIMES}` at index :math:`\texttt{SECTION}`).

   The port indices are left out if :math:`\texttt{CONNECTIONS_INCLUDE_PORTS}` is set to :math:`0` and no unit operation with multiple ports is present in the system. If a unit operation with multiple ports is present, :math:`\texttt{CONNECTIONS_INCLUDE_PORTS}` is ignored and port indices are mandatory.  

   The last three flow rate coefficients are left out if :math:`\texttt{CONNECTIONS_INCLUDE_DYNAMIC_FLOW}` is set to :math:`0`.
   Contrary to the constant coefficient, which has the parameter name :math:`\texttt{CONNECTION}`, the other coefficients are named :math:`\texttt{CONNECTION_LIN}`, :math:`\texttt{CONNECTION_QUAD}`, and :math:`\texttt{CONNECTION_CUB}`, respectively.

   For addressing the flow rates as a parameter senstivity, the mapping is as follows:

  - :math:`\texttt{SENS_UNIT}` Unused, always set to :math:`-1` 
  - :math:`\texttt{SENS_BOUNDPHASE}` *UnitOpID from* 
  - :math:`\texttt{SENS_REACTION}` *UnitOpID to* 
  - :math:`\texttt{SENS_COMP}` *Port from* 
  - :math:`\texttt{SENS_PARTYPE}` *Port to* 
  - :math:`\texttt{SENS_SECTION}` :math:`\texttt{SECTION}` that activates the valve switch 
   
   ================  ==========================  ============================================================
   **Type:** double  **Range:** :math:`\geq -1`  **Length:** :math:`\{5,7,8,10\} \cdot \texttt{NCONNECTIONS}`
   ================  ==========================  ============================================================
   
.. _FFModelExternalSourceLinInterp:

Group /input/model/external/source_XXX - EXTFUN_TYPE = LINEAR_INTERP_DATA
-------------------------------------------------------------------------

``VELOCITY``

   Velocity of the external profile in positive column axial direction.
   The velocity is normalized to a column with length 1, hence the unit :math:`\mathrm{s}^{-1}`.

   **Unit:** :math:`\mathrm{s}^{-1}`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``DATA``

   Function values :math:`T` at the data points

   **Unit:** :math:`[\mathrm{Ext}]`
   
   ================  =============================  =====================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** Arbitrary
   ================  =============================  =====================
   
``TIME``
   Time of the data points

   **Unit:** :math:`\mathrm{s}`
   
   ================  ===========================  =========================================
   **Type:** double  **Range:** :math:`\geq 0.0`  **Length:** Same as :math:`\texttt{DATA}`
   ================  ===========================  =========================================
   

.. _FFModelExternalSourcePieceCubicPoly:

Group /input/model/external/source_XXX - EXTFUN_TYPE = PIECEWISE_CUBIC_POLY
---------------------------------------------------------------------------

``VELOCITY``

   Velocity of the external profile in positive column axial direction.
   The velocity is normalized to a column with length 1, hence the unit :math:`\mathrm{s}^{-1}`.

   **Unit:** :math:`\mathrm{s}^{-1}`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CONST_COEFF``

   Constant coefficients of piecewise cubic polynomial

   **Unit:** :math:`[\mathrm{Ext}]`
   
   ================  =============================  =====================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** Arbitrary
   ================  =============================  =====================
   
``LIN_COEFF``

   Linear coefficients of piecewise cubic polynomial

   **Unit:** :math:`[\mathrm{Ext}]\,\mathrm{s}^{-1}`
   
   ================  =============================  ================================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** Same as :math:`\texttt{CONST_COEFF}`
   ================  =============================  ================================================
   
``QUAD_COEFF``

   Quadratic coefficients of piecewise cubic polynomial

   **Unit:** :math:`[\mathrm{Ext}]\,\mathrm{s}^{-2}`
   
   ================  =============================  ================================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** Same as :math:`\texttt{CONST_COEFF}`
   ================  =============================  ================================================
   
``CUBE_COEFF``

   Cubic coefficients of piecewise cubic polynomial

   **Unit:** :math:`[\mathrm{Ext}]\,\mathrm{s}^{-3}`
   
   ================  =============================  ================================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** Same as :math:`\texttt{CONST_COEFF}`
   ================  =============================  ================================================
   
``SECTION_TIMES``

   Simulation times at which a new piece begins (breaks of the piecewise polynomial)

   **Unit:** :math:`\mathrm{s}`
   
   ================  ===========================  ==========================================
   **Type:** double  **Range:** :math:`\geq 0.0`  **Length:** :math:`\texttt{CONST_COEFF}+1`
   ================  ===========================  ==========================================
   
.. _FFModelSolver:

Group /input/model/solver
-------------------------

``GS_TYPE``

   Type of Gram-Schmidt orthogonalization, see IDAS guide Section~4.5.7.3, p.~41f. A value of :math:`0` enables classical Gram-Schmidt, a value of 1 uses modified Gram-Schmidt.
   
   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 1\}`  **Length:** 1
   =============  ===========================  =============
   
``MAX_KRYLOV``

   Defines the size of the Krylov subspace in the iterative linear GMRES solver (0: :math:`\texttt{MAX_KRYLOV} = \texttt{NDOF}`)
   
   =============  ==============================================  =============
   **Type:** int  **Range:** :math:`\{0, \dots, \texttt{NDOF}\}`  **Length:** 1
   =============  ==============================================  =============
   
``MAX_RESTARTS``

   Maximum number of restarts in the GMRES algorithm. If lack of memory is not an issue, better use a larger Krylov space than restarts.
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** 1
   =============  =========================  =============
   
``SCHUR_SAFETY``

   Schur safety factor; Influences the tradeoff between linear iterations and nonlinear error control; see IDAS guide Section~2.1 and 5.
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``LINEAR_SOLUTION_MODE``

   Determines whether the system of models is solved in parallel (1) or sequentially (2). A sequential solution is only possible for systems without cyclic connections. The setting can be chosen automatically (0) based on a heuristic (less than 25 unit operations and acyclic network selects sequential mode). Optional, defaults to automatic (0).
   
   =============  ==============================  =============
   **Type:** int  **Range:** :math:`\{ 0,1,2 \}`  **Length:** 1
   =============  ==============================  =============
