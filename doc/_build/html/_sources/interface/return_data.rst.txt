.. _return:

Return data
===========

Group /input/return
-------------------

``WRITE_SOLUTION_TIMES``

   Write times at which a solution was produced (optional, defaults to 1)
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLUTION_LAST``

   Write full solution state vector at last time point (optional, defaults to 0)
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENS_LAST``

   Write full sensitivity state vectors at last time point (optional, defaults to 0)
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``SPLIT_COMPONENTS_DATA``

   Determines whether a joint dataset (matrix or tensor) for all components is created or if each component is put in a separate dataset (:math:`\texttt{XXX_COMP_000}`, :math:`\texttt{XXX_COMP_001}`, etc.) (optional, defaults to 1)
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``SPLIT_PORTS_DATA``

   Determines whether a joint dataset (matrix or tensor) for all inlet/outlet ports is created or if each port is put in a separate dataset (:math:`\texttt{XXX_PORT_000}`, :math:`\texttt{XXX_PORT_001}`, etc.) (optional, defaults to 1)
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``SINGLE_AS_MULTI_PORT``

   Determines whether single port unit operations are treated as multi port unit operations in the output naming scheme (i.e., :math:`\texttt{_PORT_XYZ_}` is added to the name) (optional, defaults to 0)
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   

Group /input/return/unit_XXX
----------------------------

``WRITE_COORDINATES``

   Write coordinates of discretization nodes
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLUTION_INLET``

   Write solutions at unit operation inlet :math:`c^l_i(t,0)`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLUTION_OUTLET``

   Write solutions at unit operation outlet (chromatograms) :math:`c^l_i(t,L)`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLUTION_BULK``

   Write solutions of the bulk volume :math:`c^l_i`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLUTION_PARTICLE``

   Write solutions of the particle mobile phase :math:`c^p_{j,i}`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLUTION_SOLID``

   Write solutions of the solid phase :math:`c^s_{j,i,m_{j,i}}`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLUTION_FLUX``

   Write solutions of the bead fluxes :math:`j_{f,i}`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLUTION_VOLUME``

   Write solutions of the volume V
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLDOT_INLET``

   Write solution time derivatives at unit operation inlet :math:`\partial c^l_i(t,0) / \partial t`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLDOT_OUTLET``

   Write solution time derivatives at unit operation outlet (chromatograms) :math:`\partial c^l_i(t,L) / \partial t`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLDOT_BULK``

   Write solution time derivatives of the bulk volume :math:`\partial c^l_i / \partial t`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLDOT_PARTICLE``

   Write solution time derivatives of the particle mobile phase :math:`\partial c^p_{j,i} / \partial t`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLDOT_SOLID``

   Write solution time derivatives of the solid phase :math:`\partial c^s_{j,i,m_{j,i}} / \partial t`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLDOT_FLUX``

   Write solution time derivatives of the bead fluxes :math:`\partial j_{f,i} / \partial t`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLDOT_VOLUME``

   Write solution time derivatives of the volume :math:`\partial V / \partial t`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENS_INLET``

   Write sensitivities at unit operation inlet :math:`\partial c^l_i(t,0) / \partial p`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENS_OUTLET``

   Write sensitivities at unit operation outlet (chromatograms) :math:`\partial c^l_i(t,L) / \partial p`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENS_BULK``

   Write sensitivities of the bulk volume :math:`\partial c^l_i / \partial p`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENS_PARTICLE``

   Write sensitivities of the particle mobile phase :math:`\partial c^p_{j,i} / \partial p`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENS_SOLID``

   Write sensitivities of the solid phase :math:`\partial c^s_{j,i,m_{j,i}} / \partial p`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENS_FLUX``

   Write sensitivities of the bead fluxes :math:`\partial j_{f,i} / \partial p`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENS_VOLUME``

   Write sensitivities of the volume :math:`\partial V / \partial p`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENSDOT_INLET``

   Write sensitivity time derivatives at unit operation inlet :math:`\partial^2 c^l_i(t,0) / (\partial p, \partial t)`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENSDOT_OUTLET``

   Write sensitivity time derivatives at unit operation outlet (chromatograms) :math:`\partial^2 c^l_i(t,L) / (\partial p, \partial t)`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENSDOT_BULK``

   Write sensitivity time derivatives of the bulk volume :math:`\partial^2 c^l_i / (\partial p, \partial t)`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENSDOT_PARTICLE``

   Write sensitivity time derivatives of the particle mobile phase :math:`\partial^2 c^p_{j,i} / (\partial p, \partial t)`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENSDOT_SOLID``

   Write sensitivity time derivatives of the solid phase :math:`\partial^2 c^s_{j,i,m_{j,i}} / (\partial p, \partial t)`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENSDOT_FLUX``

   Write sensitivity time derivatives of the bead fluxes :math:`\partial^2 j_{f,i} / (\partial p, \partial t)`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SENSDOT_VOLUME``

   Write sensitivity time derivatives of the volume :math:`\partial^2 V / (\partial p, \partial t)`
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================
   
``WRITE_SOLUTION_LAST_UNIT``

   Write solution state vector of this unit at last time point
   
   =============  ==========================
   **Type:** int  **Range:** :math:`\{0,1\}`
   =============  ==========================

