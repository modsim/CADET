.. _solver:

Solver Configuration
====================

Group /input/solver
-------------------

``NTHREADS``

   Number of used threads
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``USER_SOLUTION_TIMES``

   Vector with timepoints at which the solution is evaluated

   **Unit:** :math:`\mathrm{s}`
   
   ================  =========================  =====================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** Arbitrary
   ================  =========================  =====================
   
``CONSISTENT_INIT_MODE``

   Consistent initialization mode (optional, defaults to :math:`1`). Valid values are: 
   
  0. None 
  1. Full 
  2. Once, full 
  3. Lean 
  4. Once, lean 
  5. Full once, then lean 
  6. None once, then full 
  7. None once, then lean 
   
   =============  ===================================  =============
   **Type:** int  **Range:** :math:`\{ 0, \dots, 7\}`  **Length:** 1
   =============  ===================================  =============
   
``CONSISTENT_INIT_MODE_SENS``

   Consistent initialization mode for parameter sensitivities (optional, defaults to :math:`1`). Valid values are:

  0. None 
  1. Full 
  2. Once, full 
  3. Lean 
  4. Once, lean 
  5. Full once, then lean 
  6. None once, then full 
  7. None once, then lean 
   
   =============  ===================================  =============
   **Type:** int  **Range:** :math:`\{ 0, \dots, 7\}`  **Length:** 1
   =============  ===================================  =============
   
.. _FFSolverTime:

Group /solver/time_integrator
-----------------------------

``ABSTOL``

   Absolute tolerance in the solution of the original system
   
   ================  ======================  =============
   **Type:** double  **Range:** :math:`> 0`  **Length:** 1
   ================  ======================  =============
   
``RELTOL``

   Relative tolerance in the solution of the original system
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``ALGTOL``

   Tolerance in the solution of the nonlinear consistency equations
   
   ================  ======================  =============
   **Type:** double  **Range:** :math:`> 0`  **Length:** 1
   ================  ======================  =============
   
``RELTOL_SENS``

   Relative tolerance in the solution of the sensitivity systems
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``INIT_STEP_SIZE``

   Initial time integrator step size for each section or one value for all sections (0.0: IDAS default value), see IDAS guide 4.5, p.\ 36f.

   **Unit:** :math:`\mathrm{s}`
   
   ================  =========================  =====================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`1 / \texttt{NSEC}`
   ================  =========================  =====================================
   
``MAX_STEPS``

   Maximum number of timesteps taken by IDAS (0: IDAS default = 500), see IDAS guide Sec.~4.5
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** 1
   =============  =========================  =============
   
``MAX_STEP_SIZE``

   Maximum size of timesteps taken by IDAS (optional, defaults to unlimited 0.0), see IDAS guide Sec.~4.5

   **Unit:** :math:`\mathrm{s}`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``ERRORTEST_SENS``

   Determines whether (forward) sensitivities take part in local error test (optional, defaults to 1)
   
   =============  ==========================  =============
   **Type:** int  **Range:** :math:`\{0,1\}`  **Length:** 1
   =============  ==========================  =============
   
``MAX_NEWTON_ITER``

   Maximum number of Newton iterations in time step (optional, defaults to 3)
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** 1
   =============  =========================  =============
   
``MAX_ERRTEST_FAIL``

   Maximum number of local error test failures in time step (optional, defaults to 7)
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** 1
   =============  =========================  =============
   
``MAX_CONVTEST_FAIL``

   Maximum number of Newton convergence test failures (optional, defaults to 10)
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** 1
   =============  =========================  =============
   
``MAX_NEWTON_ITER_SENS``

   Maximum number of Newton iterations in forward sensitivity time step (optional, defaults to 3)
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** 1
   =============  =========================  =============
   
.. _FFSolverSections:

Group /solver/sections
----------------------

``NSEC``

   Number of sections
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============
   
``SECTION_TIMES``

   Simulation times at which the model changes or behaves discontinously; including start and end times

   **Unit:** :math:`\mathrm{s}`
   
   ================  =========================  ===================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** :math:`\texttt{NSEC}+1`
   ================  =========================  ===================================
   
``SECTION_CONTINUITY``

   Continuity indicator for each section transition: 0 (discontinuous) or 1 (continuous).
   
   =============  ==========================  ===================================
   **Type:** int  **Range:** :math:`\{0,1\}`  **Length:** :math:`\texttt{NSEC}-1`
   =============  ==========================  ===================================
   
