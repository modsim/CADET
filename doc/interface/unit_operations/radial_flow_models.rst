.. _radial_flow_models_config:

Radial Flow Models
==================

Radial flow models are available for the LRM, LRMP and GRM.
The configurations specified here complement the description for the respective model, i.e. please additionally refer to :ref:`lumped_rate_model_without_pores_config` or :ref:`lumped_rate_model_with_pores_config` or :ref:`general_rate_model_config`, respectively.
If input variables are described in both files, then only the description provided here applies for radial flow models.

The unit type input must be specified with the prefix :math:`\texttt{RADIAL_}` followed by the respective transport model name.
In this document, we specify the unit type as the radial GRM, but the LRM and LRMP are also available and the changes to input variables are the same for all three models.

Group /input/model/unit_XXX - UNIT_TYPE - RADIAL_GENERAL_RATE_MODEL
-------------------------------------------------------------------

For information on model equations, refer to :ref:`lumped_rate_model_without_pores_model` or :ref:`lumped_rate_model_with_pores_model` or :ref:`general_rate_model_model`, respectively.


``UNIT_TYPE``

   Specifies the type of unit operation model
   
   ================  =====================================================  =============
   **Type:** string  **Range:** :math:`\texttt{RADIAL_GENERAL_RATE_MODEL}`  **Length:** 1
   ================  =====================================================  =============

``COL_DISPERSION``

   Radial dispersion coefficient

   **Unit:** :math:`\mathrm{m}_{\mathrm{IV}}^{2}\,\mathrm{s}^{-1}`
   
   ================  =========================  =========================================================
   **Type:** double  **Range:** :math:`\geq 0`  **Length:** see :math:`\texttt{COL_DISPERSION_MULTIPLEX}`
   ================  =========================  =========================================================

	In addition to the multiplex specification (e.g. component dependency, see :ref:`general_rate_model_model`), the dispersion coefficient for radial flow model usually depends on other parameters.
	Parameter dependencies are described here :ref:`parameter_dependencies`.


``COL_RADIUS_INNER``

   Inner column radius 

   **Unit:** :math:`\mathrm{m}`
   
   ================  ======================  =============
   **Type:** double  **Range:** :math:`> 0`  **Length:** 1
   ================  ======================  =============

``COL_RADIUS_OUTER``

   Outer column radius 

   **Unit:** :math:`\mathrm{m}`
   
   ================  ======================  =============
   **Type:** double  **Range:** :math:`> 0`  **Length:** 1
   ================  ======================  =============

``CROSS_SECTION_AREA``

   Is not explicitly specified. Both the inner and outer cross section areas are implicitly given by the volumetric flow rates and either the velocity coefficient or column length

``COL_LENGTH``

   Column length/height (optional if :math:`\texttt{VELOCITY_COEFF}` is present, see Section :ref:`MUOPGRMflow`) 

   **Unit:** :math:`\mathrm{m}`
   
   ================  ======================  =============
   **Type:** double  **Range:** :math:`> 0`  **Length:** 1
   ================  ======================  =============

``VELOCITY_COEFF``

   Interstitial velocity coefficient of the mobile phase (optional :math:`\texttt{COL_LENGTH}` is present, see Section :ref:`MUOPGRMflow`).
   This input replaces the ``VELOCITY`` field, which is used for axial flow models. The distinction is made to emphasize that radial flow models do not incorporate a global velocity but a variable velocity field that depends on the spatial position, for details see Section :ref:`MUOPGRMradialFlow`.
   
   **Unit:** :math:`\mathrm{m}\,\mathrm{s}^{-1}`
   
   ================  =============================  =======================================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`1 / \texttt{NSEC}`
   ================  =============================  =======================================


Group /input/model/unit_XXX/discretization - UNIT_TYPE - RADIAL_GENERAL_RATE_MODEL
----------------------------------------------------------------------------------------

``NCOL``

   Number of radial column discretization points
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 1`  **Length:** 1
   =============  =========================  =============

Currently, there is only a first order FV spatial discretization available. Higher order spatial discretizations are planned for the future.
Accordingly, the following specifications can be left out for radial flow models.

``RECONSTRUCTION``

   Type of reconstruction method for fluxes
   
   ================  ================================  =============
   **Type:** string  **Range:** :math:`\texttt{NONE}`  **Length:** 1
   ================  ================================  =============

Parameters specified under :ref:`flux_restruction_methods` can also be ignored.
