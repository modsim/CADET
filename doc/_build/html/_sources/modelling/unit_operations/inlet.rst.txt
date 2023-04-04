.. _inlet_model:

Inlet
~~~~~

A system inlet unit operation is a pseudo unit operation since there is no physical correspondence.
The inlet serves as a mass source in the network of unit operations.
Consequently, it only possesses an outlet port and no inlet port.
Note that an inlet unit operation can provide arbitrary many components and there can be arbitrary many inlet unit operations in a network.

An inlet unit operation provides a feed in which the concentration of each component is given by a profile.
The most common profile is a piecewise cubic polynomial, which can both represent discontinuous signals (e.g., pulse or step) and smooth :math:`C^2` signals (cubic spline):

.. math::

    \begin{aligned}
        c_i(t) = \sum_{k = 1}^{N_{\text{sect}}} \mathbb{R}_{\left[t_k, t_{k+1} \right)}(t) \left[ a_{k,i} \left( t - t_k \right)^3 + b_{k,i} \left( t - t_k \right)^2 + d_{k,i} \left( t - t_k \right) + f_{k,i} \right],
    \end{aligned}

where :math:`0 \leq t_1 < t_2 < \dots < t_{N_{\text{sect}} + 1} \leq T_{\text{sim}}` is a decomposition of the simulation time interval :math:`\left[0, T_{\text{sim}}\right]` into pieces :math:`\left[t_k, t_{k+1} \right)`.
On each piece, the profile is given by a cubic (fourth order) polynomial shifted to the beginning :math:`t_k` of the piece.

For information on model parameters see :ref:`inlet_config`.
