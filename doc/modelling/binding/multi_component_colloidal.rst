.. _multi_component_colloidal_model:

Multi Component Colloidal
~~~~~~~~~~~~~~~~~~~~~~~~~

The colloidal isotherm assumes that adsorbed protein molecules are evenly distributed in a hexagonal arrangement at equilibrium, with the coverage varying via adjustment of the lattice size :cite:`Xu2009`. 
Protein-surface interactions are captured in a parameter :math:`K_{kin}`.
The surface coverage is modulated at higher coverage, described by a Yukawa potential.
The Yukawa constant :math:`b_{pp}` characterizes protein-protein interactions.
Since these are long-ranged, they are assumed to be primarily electrostatic for ion-exchange chromatography, with the range of interactions determined by the Debye parameter, :math:`\kappa`.

.. math::

    \frac{{dq}_{i}}{dt} = &k_{kin,i} \left( c_{p,i} - q_{i} \cdot \exp \left[ \frac{n}{4 q_{tot}} \sum_{j=0}^{N_{bound}} {q_{j} \sqrt{b_{pp,i}}} b_{pp,j} \cdot \frac{r_{i} + r_{j}}{2R} \exp \left( \right. \right. \right. \\
    &\left. \left. \left. - \kappa \left[ R - \left( r_{i} + r_{j} \right) \right] \cdot \left( 3 + \kappa R \right) \right) - ln K_{e,i} \right] \right) \\
    &\qquad \qquad i = 0, \dots, N_{comp} - 1, 

where :math:`n` is the coordination number describing the two-dimensional lattice agreement of protein molecules on the resin surface (:math:`n=6` for hexagonal arrangement).
:math:`r_{i}` is the radius of the protein, and :math:`K_{kin}` is the kinetic rate of adsorption.


For the surface coverage factor :math:`R`, the following equation is used:

.. math::

    R = \sqrt{\frac{2 \phi}{6.023 \cdot 10^{23} \cdot \sqrt{3} \cdot q_{tot}}},

where :math:`\phi` is the phase ratio (surface area/solid phase volume), and with

.. math::

    q_{tot} = \sum q_{i}.


The screening term :math:`\kappa` is a Debye parameter based on the ideal colloidal phase behavior of the protein.
In the case of ion exchange chromatography, this is the inverse of the Debye length.
For other interaction mechanisms, this term can be used in general to describe protein-protein interaction terms:

.. math::

    \kappa = \frac{10^{9}}{\kappa_f c_{p,0}^{\kappa_{ef}} + \kappa_{c}}.

:math:`\kappa_{c}`, :math:`\kappa_{ef}`, and :math:`\kappa_{f}` are fitting constants which can be used to custom define the protein-protein interaction behavior based on additives in the mobile phase.
:math:`c_{p,0}` is the total ionic strength, represented by the first component.

The terms for protein-resin interaction, :math:`K_{e,i}`, and protein-protein interaction, :math:`b_{pp,i}`, are varied as a function of the ionic strength (first component, :math:`c_{p,0}`) and pH (second component, :math:`c_{p,1}`). Both (pseudo) components are assumed to be non-binding:

.. math::

    ln K_{e, i} &= c_{p,1}^{k_{e,i}} \left( k_{a,i} c_{p, 0}^{-k_{b,i}} + k_{c,i} exp \left( k_{d,i} c_{p,0} \right) \right) \\
    b_{pp,i} &= c_{p,1}^{b_{e,i}} \left( b_{a,i} c_{p,0}^{b_{b,i}} + b_{c,i} exp \left( b_{d,i} c_{p,0} \right) \right),

where :math:`k_{a-e}`, :math:`b_{a-e}` are fitting constants. 


Because the model becomes mathematically singular at zero concentration, the original equation is replaced by its mathematical limit below a threshold concentration.

.. math::

    \frac{{dq}_{i}}{dt} = k_{kin,i} \left(c_{p,i} - q_{i} \cdot \exp \left[ - ln K_{e,i} \right] \right).



For more information on model parameters required to define in CADET file format, see :ref:`multi_component_colloidal_config`.

