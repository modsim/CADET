.. _flux_restruction_methods:

Flux reconstruction methods
===========================

Group /input/model/unit_XXX/discretization/weno - WENO Parameters
-----------------------------------------------------------------

``BOUNDARY_MODEL``

   Boundary model type:

   0. Lower WENO order (stable)
   1. Zero weights (unstable for small :math:`D_{\mathrm{ax}}`)
   2. Zero weights for :math:`p \neq 0` (stable?)
   3. Large ghost points

   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{0, 3\}`  **Length:** 1
   =============  ===========================  =============

``WENO_EPS``

   WENO :math:`\varepsilon`

   ================  =========================  =============
   **Type:** double  **Range:** :math:`\qe 0\`  **Length:** 1
   ================  =========================  =============

``WENO_ORDER``

   WENO order, also called WENO :math:`k`:

   1. Standard upwind scheme (order 1)
   2. WENO 2 (order 3)
   3. WENO 3 (order 5)

   =============  ===========================  =============
   **Type:** int  **Range:** :math:`\{1, 3\}`  **Length:** 1
   =============  ===========================  =============
