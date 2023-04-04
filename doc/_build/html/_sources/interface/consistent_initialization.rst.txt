.. _non_consistency_solver_parameters:

Nonlinear solver for consistent initialization
===============================================

Group /input/model/unit_XXX/discretization/consistency_solver - Nonlinear consistency solver paramters
------------------------------------------------------------------------------------------------------
``SOLVER_NAME``

Name of the solver. Available solvers are ``LEVMAR``, ``ATRN_RES``, ``ATRN_ERR``, and ``COMPOSITE``.

  ================== =======================
   **Type:** string  **Length:** :math:`1`     
  ================== =======================

``INIT_DAMPING``

Initial damping factor (default is :math:`0.01`)

   ================  =============================  ==================================
   **Type:** double  **Range:** :math:`\ge 0`       **Length:** :math:`1`
   ================  =============================  ==================================

``MIN_DAMPING``

Minimal damping factor (default is :math:`0.0001`; ignored by ``LEVMAR``)

   ================  =============================  ==================================
   **Type:** double  **Range:** :math:`\ge 0`       **Length:** :math:`1`
   ================  =============================  ==================================

``SUBSOLVERS``

Vector with names of solvers for the composite solver (only required for composite solver). See ``SOLVER_NAME`` for available solvers.

  ================== ==========================
   **Type:** string  **Length:** :math:`\gt 1`     
  ================== ==========================
