.. _sensitivity:

Parameter Sensitivities
=======================

.. _FFSensitivity:

Group /input/sensitivity
------------------------

``NSENS``

   Number of sensitivities to be computed
   
   =============  =========================  =============
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** 1
   =============  =========================  =============
   
``SENS_METHOD``

   Method used for computation of sensitivities (algorithmic differentiation)
   
   ================  ===============================  =============
   **Type:** string  **Range:** :math:`\texttt{ad1}`  **Length:** 1
   ================  ===============================  =============
   
.. _FFSensitivityParam:

Group /input/sensitivity/param_XXX
----------------------------------

``SENS_UNIT``

   Unit operation index
   
   =============  =========================  ==========================
   **Type:** int  **Range:** :math:`\geq 0`  **Length:** :math:`\geq 1`
   =============  =========================  ==========================
   
``SENS_NAME``

   Name of the parameter
   
   ================  ===========================
   **Type:** string  **Length:** :math:`\geq 1`
   ================  ===========================
   
``SENS_COMP``

   Component index (:math:`-1` if parameter is independent of components)
   
   =============  ==========================  ============================
   **Type:** int  **Range:** :math:`\geq -1`  **Length:** :math:`\geq 1`
   =============  ==========================  ============================
   
``SENS_PARTYPE``

   Particle type index (:math:`-1` if parameter is independent of particle types)
   
   =============  ==========================  ===========================
   **Type:** int  **Range:** :math:`\geq -1`  **Length:** :math:`\geq 1`
   =============  ==========================  ===========================
   
``SENS_REACTION``

   Reaction index (:math:`-1` if parameter is independent of reactions)
   
   =============  ==========================  ===========================
   **Type:** int  **Range:** :math:`\geq -1`  **Length:** :math:`\geq 1`
   =============  ==========================  ===========================
   
``SENS_BOUNDPHASE``

   Bound phase index (:math:`-1` if parameter is independent of bound phases)
   
   =============  ==========================  ==========================
   **Type:** int  **Range:** :math:`\geq -1`  **Length:** :math:`\geq 1`
   =============  ==========================  ==========================
   
``SENS_SECTION``

   Section index (:math:`-1` if parameter is independent of sections)
   
   =============  ==========================  ==========================
   **Type:** int  **Range:** :math:`\geq -1`  **Length:** :math:`\geq 1`
   =============  ==========================  ==========================
   
``SENS_ABSTOL``

   Absolute tolerance used in the computation of the sensitivities (optional). Rule of thumb: :math:`\texttt{ABSTOL} / \texttt{PARAM_VALUE}`
   
   ================  ===========================  ==========================
   **Type:** double  **Range:** :math:`\geq 0.0`  **Length:** :math:`\geq 1`
   ================  ===========================  ==========================
   
``SENS_FACTOR``

   Linear factor of the combined sensitivity (optional, taken as :math:`1.0` if left out)
   
   ================  =============================  ==========================
   **Type:** double  **Range:** :math:`\mathbb{R}`  **Length:** :math:`\geq 1`
   ================  =============================  ==========================
