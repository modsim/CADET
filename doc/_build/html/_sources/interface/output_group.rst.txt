.. _FFOutput:

Output Group
===============

Group /output
-------------

``LAST_STATE_Y``

   Full state vector at the last time point of the time integrator if :math:`\texttt{WRITE_SOLUTION_LAST}` in :math:`\texttt{/input/return}` is enabled
   
   **Type:** double
   
``LAST_STATE_YDOT``

   Full time derivative state vector at the last time point of the time integrator if :math:`\texttt{WRITE_SOLUTION_LAST}` in :math:`\texttt{/input/return}` is enabled
   
   **Type:** double
   
``LAST_STATE_SENSY_XXX``

   Full state vector of the ``XXX`` th sensitivity system at the last time point of the time integrator if :math:`\texttt{WRITE_SENS_LAST}` in :math:`\texttt{/input/return}` is enabled
   
   **Type:** double
   
``LAST_STATE_SENSYDOT_XXX``

   Full time derivative state vector of the ``XXX`` th sensitivity system at the last time point of the time integrator if :math:`\texttt{WRITE_SENS_LAST}` in :math:`\texttt{/input/return}` is enabled
   
   **Type:** double
   
Group /output/solution
----------------------

``SOLUTION_TIMES``

   Time points at which the solution is written if :math:`\texttt{WRITE_SOLUTION_TIMES}` in :math:`\texttt{/input/return}` is enabled

   **Unit:** :math:`\mathrm{s}`
   
   **Type:** double
   

Group /output/solution/unit_XXX
-------------------------------

``SOLUTION_BULK``

   Interstitial solution as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   **Type:** double
   
``SOLUTION_PARTICLE``

   Mobile phase solution inside the particles as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if just one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}`

   **Type:** double

``SOLUTION_PARTICLE_PARTYPE_XXX``

   Mobile phase solution inside the particles of type ``XXX`` as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if more than one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}`
   
   **Type:** double
   
``SOLUTION_SOLID``

   Solid phase solution inside the particles as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if just one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{SP}}^{-3}`
   
   **Type:** double
   
``SOLUTION_SOLID_PARTYPE_XXX``

   Solid phase solution inside the particles of type ``XXX`` as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if more than one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{SP}}^{-3}`
   
   **Type:** double
   
``SOLUTION_FLUX``

   Flux solution as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}^{-2}\,\mathrm{s}^{-1}`

   **Type:** double
   
``SOLUTION_VOLUME``

   Volume solution

   **Unit:** :math:`\mathrm{m}^{3}`
   
   **Type:** double
   
``SOLUTION_OUTLET``

   Tensor of solutions at the unit operation outlet with components as columns in time-port-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are both disabled. If the unit operation only has a single port, the port-dimension is removed if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   **Type:** double
   
``SOLUTION_INLET``

   Tensor of solutions at the unit operation inlet with components as columns in time-port-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are both disabled. If the unit operation only has a single port, the port-dimension is removed if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   **Type:** double
   
``SOLUTION_OUTLET_COMP_XXX``

   Component ``XXX`` of the solution at all outlet ports of the unit operation as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is enabled and :math:`\texttt{SPLIT_PORTS_DATA}` is disabled. If the unit operation only has a single port, a vector (1D array) is returned instead of a matrix if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   **Type:** double
   
``SOLUTION_INLET_COMP_XXX``

   Component ``XXX`` of the solution at all inlet ports of the unit operation inlet as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is enabled and :math:`\texttt{SPLIT_PORTS_DATA}` is disabled. If the unit operation only has a single port, a vector (1D array) is returned instead of a matrix if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   **Type:** double
   
``SOLUTION_OUTLET_PORT_XXX``

   All components at outlet port ``XXX`` of the solution of the unit operation as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is disabled and :math:`\texttt{SPLIT_PORTS_DATA}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   **Type:** double
   
``SOLUTION_INLET_PORT_XXX``

   All components at inlet port ``XXX`` of the solution of the unit operation inlet as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is disabled and :math:`\texttt{SPLIT_PORTS_DATA}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   **Type:** double
   
``SOLUTION_OUTLET_PORT_XXX_COMP_YYY``

   Component ``YYY`` at outlet port ``XXX`` of the solution of the unit operation. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are enabled, and the unit operation has multiple outlet ports. If the unit operation only has a single port, the field is created if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   **Type:** double
   
``SOLUTION_INLET_PORT_XXX_COMP_YYY``

   Component ``YYY`` at inlet port ``XXX`` of the solution of the unit operation. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are enabled, and the unit operation has multiple inlet ports. If the unit operation only has a single port, the field is created if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}`
   
   **Type:** double
   
``SOLDOT_BULK``

   Interstitial solution time derivative as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_PARTICLE``

   Mobile phase solution time derivative inside the particles as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if just one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_PARTICLE_PARTYPE_XXX``

   Mobile phase solution time derivative inside the particles of type ``XXX`` as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if more than one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_SOLID``

   Solid phase solution time derivative inside the particles as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if just one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_SOLID_PARTYPE_XXX``

   Solid phase solution time derivative inside the particles of type ``XXX`` as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if more than one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{SP}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_FLUX``

   Flux solution time derivative as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}^{-2}\,\mathrm{s}^{-2}`
   
   **Type:** double
   
``SOLDOT_VOLUME``

   Volume solution time derivative

   **Unit:** :math:`\mathrm{m}^{3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_OUTLET``

   Tensor of solution time derivatives at the unit operation outlet with components as columns in time-port-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are both disabled. If the unit operation only has a single port, the port-dimension is removed if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_INLET``

   Tensor of solution time derivatives at the unit operation inlet with components as columns in time-port-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are both disabled. If the unit operation only has a single port, the port-dimension is removed if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_OUTLET_COMP_XXX``

   Component ``XXX`` of the solution time derivative at all outlet ports of the unit operation as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is enabled and :math:`\texttt{SPLIT_PORTS_DATA}` is disabled. If the unit operation only has a single port, a vector (1D array) is returned instead of a matrix if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_INLET_COMP_XXX``

   Component ``XXX`` of the solution time derivative at all inlet ports of the unit operation inlet as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is enabled and :math:`\texttt{SPLIT_PORTS_DATA}` is disabled. If the unit operation only has a single port, a vector (1D array) is returned instead of a matrix if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_OUTLET_PORT_XXX``

   All components at outlet port ``XXX`` of the solution time derivative of the unit operation as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is disabled and :math:`\texttt{SPLIT_PORTS_DATA}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_INLET_PORT_XXX``

   All components at inlet port ``XXX`` of the solution time derivative of the unit operation inlet as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is disabled and :math:`\texttt{SPLIT_PORTS_DATA}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_OUTLET_PORT_XXX_COMP_YYY``

   Component ``YYY`` at outlet port ``XXX`` of the solution time derivative of the unit operation. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are enabled, and the unit operation has multiple outlet ports. If the unit operation only has a single port, the field is created if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``SOLDOT_INLET_PORT_XXX_COMP_YYY``

   Component ``YYY`` at inlet port ``XXX`` of the solution time derivative of the unit operation. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are enabled, and the unit operation has multiple inlet ports. If the unit operation only has a single port, the field is created if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}`
   
   **Type:** double
   
``LAST_STATE_Y``

   State vector of this unit at the last time point of the time integrator if :math:`\texttt{WRITE_SOLUTION_LAST_UNIT}` in :math:`\texttt{/input/return/unit_XXX}` is enabled.
   Note that the vector includes the dedicated inlet DOFs at the beginning (length: :math:`\texttt{NCOMP} \cdot \texttt{NPORT}`).
   
   **Type:** double
   
``LAST_STATE_YDOT``

   Time derivative state vector of this unit at the last time point of the time integrator if :math:`\texttt{WRITE_SOLUTION_LAST_UNIT}` in :math:`\texttt{/input/return/unit_XXX}` is enabled.
   Note that the vector includes the dedicated inlet DOFs at the beginning (length: :math:`\texttt{NCOMP} \cdot \texttt{NPORT}`).
   
   **Type:** double


Group /output/sensitivity/param_XXX/unit_YYY
--------------------------------------------

``SENS_BULK``

   Interstitial sensitivity as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_PARTICLE``

   Mobile phase sensitivity inside the particles as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if just one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_PARTICLE_PARTYPE_XXX``

   Mobile phase sensitivity inside the particles of type ``XXX`` as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if more than one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_SOLID``

   Solid phase sensitivity inside the particles as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if just one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_SOLID_PARTYPE_XXX``

   Solid phase sensitivity inside the particles of type ``XXX`` as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if more than one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{SP}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_FLUX``

   Flux sensitivity as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}^{-2}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_VOLUME``

   Volume sensitivity

   **Unit:** :math:`\mathrm{m}^{3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_OUTLET``

   Tensor of sensitivities at the unit operation outlet with components as columns in time-port-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are both disabled. If the unit operation only has a single port, the port-dimension is removed if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_INLET``

   Tensor of sensitivities at the unit operation inlet with components as columns in time-port-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are both disabled. If the unit operation only has a single port, the port-dimension is removed if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_OUTLET_COMP_XXX``

   Component ``XXX`` of the sensitivity at all outlet ports of the unit operation as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is enabled and :math:`\texttt{SPLIT_PORTS_DATA}` is disabled. If the unit operation only has a single port, a vector (1D array) is returned instead of a matrix if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_INLET_COMP_XXX``

   Component ``XXX`` of the sensitivity at all inlet ports of the unit operation inlet as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is enabled and :math:`\texttt{SPLIT_PORTS_DATA}` is disabled. If the unit operation only has a single port, a vector (1D array) is returned instead of a matrix if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_OUTLET_PORT_XXX``

   All components at outlet port ``XXX`` of the sensitivity of the unit operation as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is disabled and :math:`\texttt{SPLIT_PORTS_DATA}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_INLET_PORT_XXX``

   All components at inlet port ``XXX`` of the sensitivity of the unit operation inlet as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is disabled and :math:`\texttt{SPLIT_PORTS_DATA}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_OUTLET_PORT_XXX_COMP_YYY``

   Component ``YYY`` at outlet port ``XXX`` of the sensitivity of the unit operation. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are enabled, and the unit operation has multiple outlet ports. If the unit operation only has a single port, the field is created if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENS_INLET_PORT_XXX_COMP_YYY``

   Component ``YYY`` at inlet port ``XXX`` of the sensitivity of the unit operation. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are enabled, and the unit operation has multiple inlet ports. If the unit operation only has a single port, the field is created if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_BULK``

   Interstitial sensitivity time derivative as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_PARTICLE``

   Mobile phase sensitivity time derivative inside the particles as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if just one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_PARTICLE_PARTYPE_XXX``

   Mobile phase sensitivity time derivative inside the particles of type ``XXX`` as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if more than one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_SOLID``

   Solid phase sensitivity time derivative inside the particles as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if just one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{MP}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_SOLID_PARTYPE_XXX``

   Solid phase sensitivity time derivative inside the particles of type ``XXX`` as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage. Only present if more than one particle type is defined.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{SP}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_FLUX``

   Flux sensitivity time derivative as :math:`n_{\text{Time}} \times \texttt{UNITOPORDERING}` tensor in row-major storage

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}^{-2}\,\mathrm{s}^{-2}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_VOLUME``

   Volume sensitivity time derivative

   **Unit:** :math:`^{3}\,\mathrm{s}\mathrm{m}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_OUTLET``

   Tensor of sensitivity time derivatives at the unit operation outlet with components as columns in time-port-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are both disabled. If the unit operation only has a single port, the port-dimension is removed if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_INLET``

   Tensor of sensitivity time derivatives at the unit operation inlet with components as columns in time-port-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are both disabled. If the unit operation only has a single port, the port-dimension is removed if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_OUTLET_COMP_XXX``

   Component ``XXX`` of the sensitivity time derivative at all outlet ports of the unit operation as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is enabled and :math:`\texttt{SPLIT_PORTS_DATA}` is disabled. If the unit operation only has a single port, a vector (1D array) is returned instead of a matrix if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_INLET_COMP_XXX``

   Component ``XXX`` of the sensitivity time derivative at all inlet ports of the unit operation inlet as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is enabled and :math:`\texttt{SPLIT_PORTS_DATA}` is disabled. If the unit operation only has a single port, a vector (1D array) is returned instead of a matrix if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is disabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_OUTLET_PORT_XXX``

   All components at outlet port ``XXX`` of the sensitivity time derivative of the unit operation as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is disabled and :math:`\texttt{SPLIT_PORTS_DATA}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_INLET_PORT_XXX``

   All components at inlet port ``XXX`` of the sensitivity time derivative of the unit operation inlet as matrix in time-major storage. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` is disabled and :math:`\texttt{SPLIT_PORTS_DATA}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_OUTLET_PORT_XXX_COMP_YYY``

   Component ``YYY`` at outlet port ``XXX`` of the sensitivity time derivative of the unit operation. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are enabled, and the unit operation has multiple outlet ports. If the unit operation only has a single port, the field is created if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double
   
``SENSDOT_INLET_PORT_XXX_COMP_YYY``

   Component ``YYY`` at inlet port ``XXX`` of the sensitivity time derivative of the unit operation. Only present if :math:`\texttt{SPLIT_COMPONENTS_DATA}` and :math:`\texttt{SPLIT_PORTS_DATA}` are enabled, and the unit operation has multiple inlet ports. If the unit operation only has a single port, the field is created if :math:`\texttt{SINGLE_AS_MULTI_PORT}` is enabled.

   **Unit:** :math:`\mathrm{mol}\,\mathrm{m}_{\mathrm{IV}}^{-3}\,\mathrm{s}^{-1}\,[\mathrm{Param}]^{-1}`
   
   **Type:** double 
   

/output/coordinates/unit_XXX
------------------------------

``AXIAL_COORDINATES``

   Axial coordinates of the bulk discretization nodes

   **Unit:** :math:`\mathrm{m}`
   
   ================  =================================
   **Type:** double  **Length:** :math:`\texttt{NCOL}`
   ================  =================================
   
``RADIAL_COORDINATES``

   Radial coordinates of the bulk discretization nodes (only for 2D unit operations)

   **Unit:** :math:`\mathrm{m}`
   
   ================  =================================
   **Type:** double  **Length:** :math:`\texttt{NRAD}`
   ================  =================================
   
``PARTICLE_COORDINATES_XXX``

   Coordinates of the particle discretization nodes in particles of type ``XXX``

   **Unit:** :math:`\mathrm{m}`
   
   ================  =================================
   **Type:** double  **Length:** :math:`\texttt{NPAR}`
   ================  =================================
