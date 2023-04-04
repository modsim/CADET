.. _multi_component_anti_langmuir_model:

Multi Component Anti-Langmuir
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Anti-Langmuir (or generalized Langmuir) binding model extends the Langmuir model (see Section :ref:`multi_component_langmuir_model`).
The factor :math:`p_j \in \{ -1, 1 \}` determines the shape of the isotherm.
For :math:`p_j = 1` (standard Langmuir) the chromatograms have sharp fronts and a dispersed tail (isotherm is concave).
In case of the Anti-Langmuir (:math:`p_j = -1`) it is the other way around (isotherm is convex).

.. math::

    \begin{aligned}
        \frac{\mathrm{d} q_i}{\mathrm{d} t} = k_{a,i} c_{p,i} q_{\text{max},i} \left( 1 - \sum_{j=0}^{N_{\text{comp}} - 1} p_j \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} q_i && i = 0, \dots, N_{\text{comp}} - 1.
    \end{aligned}


For more information on model parameters required to define in CADET file format, see :ref:`multi_component_anti_langmuir_config`.
