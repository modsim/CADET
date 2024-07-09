.. _pbm_config:

Population balance model
========================

The PBM in CADET is implemented as part of the reaction module and can thus be used in any unit operation that includes reactions.
Typical applications consider crystallization in a :ref:`cstr_config` or, to model continuous processes, in a Dispersive Plug-Flow Reactor (DPFR), which is modelled by a :ref:`lumped_rate_model_without_pores_config`.

The particle size domain (internal coordinate) is discretized by the FV method, giving us a finite set of particle sizes under consideration :math:`\{x_1, \dots, x_{N_x}\}`.
Every particle size considered is treated as an individual component of the unit operation and the field ``NCOMP`` of that unit operation in which the crystallization happens, must be specified accordingly as :math:`N_x + 2`.
The two additional components account for the solute :math:`c` and solubility :math:`c_\text{eq}`.
*That is, by setting the ``NCOMP`` field, you specify the number of FV cells for the internal coordinate.*

Group /input/model/unit_XXX
---------------------------

``NCOMP``

   Number of components, which is given by two plus the number discrete particle sizes, which is given by the number of FV cells discretizing the internal coordinate
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 3`  **Length:** 1
   =============  =========================  =============
   
``REACTION_MODEL``

   The crystallization code is implemented as a reaction module, which is why crystallization needs to be specified here
   
   ================  ========================================  =============
   **Type:** String  **Range:** :math:`\{ CRYSTALLIZATION \}`  **Length:** 1
   ================  ========================================  =============

Group /input/model/unit_XXX/reaction_bulk - REACTION_MODEL = CRYSTALLIZATION - UNIT_TYPE = CSTR
-----------------------------------------------------------------------------------------------

*The following parameters need to be specified under Group /input/model/unit_XXX/reaction/reaction_bulk/ for CSTR units, and Group /input/model/unit_XXX/reaction/reaction/ for transport units like the LRM.*

``CRY_BINS``

   Coordinates of the cell faces, e.g. equidistant or logarithmic discretization of the internal coordinate :math:`x \in [x_c, x_\text{end}]`, including the end points.
   
   ================  =========================  =====================================
   **Type:** double  **Range:** :math:`\geq 1`   **Length:** :math:`\mathrm{N_x} + 1`
   ================  =========================  =====================================
   
``CRY_NUCLEI_MASS_DENSITY``

   Nulcei mass density
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_VOL_SHAPE_FACTOR``

   Volumetric shape factor of the particles
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_PRIMARY_NUCLEATION_RATE``

   Primary nucleation rate constant :math:`k_p`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_SECONDARY_NUCLEATION_RATE``

   Secondary nucleation rate :math:`k_b`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_GROWTH_RATE_CONSTANT``

   Growth rate constant :math:`k_g`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_GROWTH_CONSTANT``

   Growth constant :math:`\gamma`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_A``

   Defines constant :math:`a` used to determine the growth rate
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_G``

   Defines constant :math:`g` used to determine the growth rate
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_P``

   Defines constant :math:`p`  used to determine the growth rate
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_GROWTH_DISPERSION_RATE``

   Growth dispersion rate :math:`D_g`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_U``

   Defines constant :math:`u` used to determine the primary nucleation
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_B``

   Defines constant :math:`b` used to determine the secondary nucleation
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_K``

   Defines constant :math:`k` used to determine the secondary nucleation
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_GROWTH_SCHEME_ORDER``

   Defines the growth flux FV reconstruction scheme. It can only be :math:`1`: upwind scheme; :math:`2`: HR Koren scheme; :math:`3`: WENO23 scheme; :math:`4`: WENO35 scheme.
   We recommend using the HR Koren scheme, which showed to be the most performant in our benchmarks.
   
   =============  ================================  =============
   **Type:** int  **Range:** :math:`[1, \dots, 4]`  **Length:** 1
   =============  ================================  =============
