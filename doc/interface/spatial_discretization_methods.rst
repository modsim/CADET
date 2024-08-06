.. _spatial_discretization_methods:

Spatial discretization methods
==============================

Group /input/model/unit_XXX/discretization/SPATIAL_METHOD - Details on the methods
----------------------------------------------------------------------------------

CADET offers two spatial discretization methods: Finite Volumes (FV) and Discontinuous Galerkin (DG).
While both methods approximate the same solution to the underlying models, they may differ in terms of computational performance.
Generally, FV can perform better for small problem sizes and solutions with steep gradients, while DG excels for large problem sizes and smooth solutions.

In the following, we give a brief introduction to the numerical theory w.r.t. the computational performance of the methods.
Based on that theory and our experience, we give advice on which method to use in which scenario, how to identify the computationally best performing method, and how to specify discretization parameters.
For a comprehensive description of the FV and DG variants that are implemented in CADET, we refer to our publications on `CADET-FV <https://doi.org/10.1016/j.compchemeng.2010.03.008>`_  and `CADET-DG <https://doi.org/10.1016/j.compchemeng.2023.108340>`_.

Discrete system size
--------------------

FV and DG discretize the continuous spatial domain of the partial differential algebraic equations (PDAE) into a finite set of discrete points.
Then, a system of (semi-discrete) equations is formulated on those points, resulting in a system of ordinary differential algebraic equations (ODAE).
The system size and number of unknowns, are also referred to as degrees of freedom (DoF), is given by the number of (spatial) discrete points (times the number of components for multi-component systems).

The numerical solution becomes more accurate with more discrete points used, but the compute time required to solve the equations also increases.
That is, we trade compute time for approximation accuracy.

For the FV method, the number of axial discrete points in the column is specified by the number of volume cells ``NCOL``.
For the DG method, the number of axial discrete points in the column is specified by the number of polynomial interpolation nodes (= ``POLYDEG`` + 1) times the number of DG elements ``NELEM``.
The GRM additionally comprises a transport equation along the radial coordinate of the particles which is also spatially discretized and the particle discrete points are correspondingly given by ``NPAR`` for FV and for DG via ``PAR_POLYDEG`` and ``PAR_NELEM``.

Order of convergence
--------------------

The computational performance of a numerical method depends on its theoretical order of convergence.
The order of convergence refers to the rate at which the numerical approximation approaches the exact solution under refinement of the spatial grid.
Consequently, higher order methods often require less spatial discrete points to compute an approximation of the desired accuracy and can thus be computationally more efficient.

The theoretical order of convergence for the CADET-FV method is globally limited to 2.
It is locally (except column boundaries) limited to 3, with an improved order of up to 5 for smooth solutions, and can be varied by specifying the parameters described in the :ref:`flux_reconstruction_methods` section.
For the CADET-DG method, the theoretical order of convergence is :math:`N_d + 1` with :math:`N_d` denoting the polynomial degree, and can thus be user-defined by specifying the field ``POLYDEG`` (and ``PAR_POLYDEG`` for the particles in the GRM).

The theoretical order of convergence is an asymptotic property, i.e. holds for infinite refinements, but can be numerivcally observed for finite refinements as well.
Having the exact solution, we can compute an experimental order of convergence (EOC) via the formula

.. math::
    :label: ModelColumn

    \begin{aligned}
        EOC_k = \frac{log(\varepsilon_{k+1} / \varepsilon_{k}) }{log(n_{k} / n_{k+1})},
    \end{aligned}

with :math:`\varepsilon_{k}` and :math:`n_{k}` denoting some error norm and the degrees of freedom of the $k$th approximation.
The EOC approaches the theoretical order of convergence for :math:`k \rightarrow \infty` but is typically lower for underresolved problems.
High-order methods typically do typically not exhibit their theoretical order on very coarse grids.
That is, increasing the number of discrete points from, e.g., 2 to 4 does often not improve the solution according to the theoretical order of convergence but by a much smaller EOC.

For smooth solutions, we typically observe an EOC of around 2.5 for the default CADET-FV method and around :math:`N_d` for the CADET-DG method.
To our experience, DG with :math:`N_d>6` does usually not realize an EOC of :math:`>6` for approximation errors within engineering tolerances, i.e. higher rates are achieved only for excessively small error tolerances that are not relevant for most applications.
We thus recommend to choose :math:`3 \leq N_d \leq 5` for the DG method.

One could still think that the higher the order of the method, the better the performance, but that is unfortunately not generally true, as the numerical solver performance can strongly depend on the smoothness of the approximated solution.

Smooth solutions
----------------

The smoothness of a function indicates the absence of sudden changes, reflected in the continuity and differentiability of the function and its derivatives.
In numerical simulation, smoothness indicators are often based on derivatives or moments of the solution.
That is, the occurrence of strong gradients and high frequencies are used to identify non-smooth parts of the solution.
Godunov's order barrier theorem shows why the concept of smoothness plays a crucial role in the deployment of numerical methods.
It states that linear high-order methods that are monotonous are at most first-order accurate.
Linear higher-order (:math:`\gt 1`) methods thus suffer from artificial oscillations at non-smooth parts of the solution, specifically at discontinuities and strong gradients.
Some higher-order methods, such as the FV variants implemented in CADET, contain a non-linear mechanism to suppress these oscillations.
The non-linear WENO mechanism employed in CADET-FV can be fine-tuned via the parameters specified in the :ref:`flux_reconstruction_methods` section.
Unfortunately, non-linear higher-order methods (here :math:`\geq 3`) are either not applicable (e.g., undefined boundary treatment) or have other shortcomings, such as parameters whose optimal values can be highly proplem-specific.

The DG variant that is implemented in CADET is a linear high-order method (arbitrary order) and can thus exhibit oscillatory behaviour at strong gradients, which increases the approximation error and results in a smaller EOC for lower resolutions.
Since strong gradients are local phenomena which can be captured by employing more discrete points, DG performs better again for higher spatial resolutions.
This, however, might happen after the engineering error tolerance is by far surpassed.
Hence, CADET-FV as a stabilized lower-order method can perform better, depending on the smoothness of the approximated solution.
The DG method reduces its oscillatory behaviour by adding artificial numerical dispersion at element interfaces.
Thus, the use of a lower polynomial degree and more elements is recommended for non-smooth solutions.

In Chromatography, mathematical discontinuities never happen, as there are always some dispersive effects in reality.
Chromatography models, however, allow for discontinuities if dispersion parameters are set to zero.
Moreover, steep and self-sharpening concentration fronts might appear due to competitive adsorption.
Examples that can cause self-sharpening concentration fronts are often associated with competitive Langmuir type isotherms for components with differently strong binding properties.
Nonetheless, many chromatography settings yield rather smooth concentration profiles, for which DG is the better choice in terms of computational performance.

Recommendations on the choice of spatial discretization methods
---------------------------------------------------------------

We recommend the FV method for

- Small problem sizes, e.g., low spatial resolution with the LRM
- Problems with strong gradients, e.g., no or low dispersion and binding model parameters that create sharp fronts
- Binding models that mathematically require positive values or exhibit strange behaviour with negative concentration values

We recommend the DG method for

- Large problem sizes, e.g., high spatial resolutions and more complex models (i.e. the LRMP and specifically the GRM)
- Smooth problems, i.e., sufficient band broadening

Recommendations on DG discretization parameters
-----------------------------------------------

- Employ an axial polynomial degree between 3 and 5
- Select a lower axial polynomial degree for approximating functions that tend to be less smooth and employ more elements instead. Converse choice for smooth problems
- Adjust the DG particle polynomial degree to control approximation accuracy; leave the number of elements at one. Make exceptions if very steep gradients occur inside the particles or when specific regions of the particle domain are more interesting (the spatial resolution of certain regions can be refined by a user-defined spacing of multiple elements)
- The field ``EXACT_INTEGRATION`` specifies the DG polynomial integration method. The default value of $0$ (collocation DG) is expected to be slightly more performant in most settings

Refinement strategy
-------------------

A common problem in numerical simulation is that the number of discrete points required to yield an accurate approximation within a specific tolerance is unknown.
We thus recommend determining the approximation error via comparison with a refined reference approximation.
Both the theoretical order of convergence and the EOC can be used to estimate the required number of discrete points.
An EOC that is significantly lower than the theoretical order indicates that the problem is numerically underresolved.

Note on DG solution vector
--------------------------

Any liquid or solid concentration within the column or particles is reported on the discrete points that are employed by the method.
That is, DG yields a piece-wise polynomial approximation on Lagrange-Gauss-Lobatto nodes.
If the solution is desired on a different grid, element-wise polynomial interpolation should be applied, and element interface values must be averaged.
