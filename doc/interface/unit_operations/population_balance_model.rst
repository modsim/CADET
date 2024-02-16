.. _pbm_config:

Population balance model
=====================================

One-dimensional (1D) PBM
^^^^^^^^^^^^^^^^^^^^^^^^

The 1D PBM is implemented in a CSTR as reactions. To configure the CSTR, see :ref:`cstr_config`.
Add crystallization reactions:

Group /input/model/unit_XXX - REACTION_MODEL = CRYSTALLIZATION
--------------------------------------------------------------

``NCOMP``

   Number of discretization cells
   
   =============  =================================  =============
   **Type:** int  **Range:** :math:`\mathrm{N_X}-1`  **Length:** 1
   =============  =================================  =============

Constitutive equations are configured under:
--------------------------------------------

Group /input/model/unit_XXX/reaction/reaction_bulk/
---------------------------------------------------

``CRY_BINS``

   Define cell faces
   
   ================  =========================  ================================
   **Type:** double  **Range:** :math:`\geq 1`   **Length:** :math:`\mathrm{N_X}`
   ================  =========================  ================================
   
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

    Define :math:`k_p`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_SECONDARY_NUCLEATION_RATE``

    Define :math:`k_s`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_GROWTH_RATE_CONSTANT``

    Define :math:`k_g`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_GROWTH_CONSTANT``

    Define :math:`\gamma`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_A``

    Define :math:`a`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_B``

    Define :math:`b`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_G``

    Define :math:`g`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_P``

    Define :math:`g`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_K``

    Define :math:`k`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_U``

    Define :math:`u`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_GROWTH_DISPERSION_RATE``

    Define the growth dispersion rate :math:`D_g`
   
   ================  =========================  =============
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** 1
   ================  =========================  =============
   
``CRY_GROWTH_SCHEME_ORDER``

    Define growth flux reconstruction scheme. It can only be 1, 2, 3, 4. Values other than these will give undefined behaviors.
    1: upwind scheme; 2: HR Koren scheme; 3: WENO23 scheme; 4: WENO35 scheme
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`[1, 4]`  **Length:** 1
   =============  =========================  =============

Two-dimensional (2D) PBM
^^^^^^^^^^^^^^^^^^^^^^^^

The 2D PBM is implemented in a DPFR (LRM). To configure the LRM, see :ref:`lumpded_rate_model_without_pores_config`.
The constitutive equations can be configured similarly to the 1D PBM above under

Group /input/model/unit_XXX/reaction/
-------------------------------------

Note the change from '../reaction_bulk/' to '../reaction/'.
