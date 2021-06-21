.. _multi_component_colloidal_model:

Multi Component Colloidal
~~~~~~~~~~~~~~~~~~~~~~~~~
[add some general introduction]

.. math::

    \frac{{dc}_{i}^{s}}{dt} = &k_{kin,i} \left( c_{i}^{p} - c_{i}^{s} \cdot \exp \left[ \frac{n}{{4c}_{tot}^{s}} \sum_{j}^{N_{bnd}} {c_{j}^{s} \sqrt{b_{pp,i}}} b_{pp,j} \cdot \frac{r_{i} + r_{j}}{2R} \exp \left( \right. \right. \right. \\
    &\left. \left. \left. - \kappa \left[ R - \left( r_{i} + r_{j} \right) \right] \cdot \left( 3 + \kappa R \right) \right) - ln K_{e,i} \right] \right)

where :math:`n` is the coordination number (6 for hexagonal packing), :math:`r_{i}` is the radius of the protein, and :math:`K_{kin}` is the kinetic rate of adsorption. :math:`N_{bnd}` is the number of bound components.


For the surface coverage factor :math:`R`, the following equation is used:

.. math::

    R = \sqrt{\frac{2 \phi}{6.023 \cdot 10^{23} \cot \sqrt{3} \cdot c_{tot}^{s}}},

where :math:`\phi` is the phase ratio (surface area/solid phase volume), and with

.. math::

    c_{tot}^{s} = \sum c_{i}^{s},


For the screening term :math:`\kappa`, the following equation is used:

.. math::

    \kappa = \frac{10^{9}}{\kappa_f c_{0}^{\kappa_{ef}} + \kappa{c}}.

:math:`\kappa_{c}`, :math:`\kappa_{ef}`, and :math:`\kappa_{f}` are constants and :math:`c_{0}` is the total ionic strength.

The terms for protein-resin interaction, :math:`K_{e,i}`, and protein-protein interaction, :math:`b_{pp,i}`, are varied as function of ionic strength (:math:`c_0`) and pH (:math:`c_1`): 

.. math::

    ln K_{e, i} &= pH^{k_{e,i}} \left( k_{a,i} c_{0}^{-k_{b,i}} + k_{c,i} exp \left( k_{d,i} c_{0} \right) \right) \\
    b_{pp,i} &= pH^{b_{e,i}} \left( b_{a,i} c_{0}^{b_{b,i}} + b_{c,i} exp \left( b_{d,i} c_{0} \right) \right),

where :math:`k_{a-e}`, :math:`b_{a-e}` are fitting constants. 


If the surface concentration is close to zero, the model switches to a linear implementation:

.. math::

    \frac{{dc}_{i}^{s}}{dt} = k_{kin,i} \left(c_{i}^{p} - c_{i}^{s} \cdot \exp \left[ - ln K_{e,i} \right] \right).

More details about the model can be found in the `corresponding paper <https://doi.org/10.1016/j.chroma.2009.06.082>`_.

For more information on model parameters required to define in CADET file format, see :ref:`multi_component_colloidal_config`.

