.. spatial_discretization_methods:

Spatial discretization methods
==============================

Group /input/model/unit_XXX/discretization/SPATIAL_METHOD - Details on the methods
----------------------------------------------------------------------------------

CADET offers two spatial discretization methods: Finite Volumes (FV) and Discontinuous Galerkin (DG).
While both methods approximate the same solution to the underlying models, they may differ in terms of computational performance.
Generally, FV can be more performant for small problem sizes and solutions with steep gradients, while DG excels for large problem sizes and smooth solutions.

In the following, we give a brief introduction to the numerical theory that is most relevant for the computational performance of the methods.
Based on that theory and our experience, we give advice on which method to use in which scenario, how to identify the more performant method, and how to specify the discretization parameters.
For a comprehensive description on the FV and DG methods as they are implemented in CADET, we refer to our publications on `CADET-FV <https://doi.org/10.1016/j.compchemeng.2010.03.008>`_  and `CADET-DG <https://doi.org/10.1016/j.compchemeng.2023.108340>`_.

Discrete system size
--------------------

Numerical methods discretize the continuous (here: spatial) domain of the equations into a finite set of discrete points.
Then, a system of equations is formulated for those points, and both system size and number of unknowns / degrees of freedom (DoF) are given by the number of discrete points.
This system can be linear or non-linear, depending on the method (this will become important again in the section on smooth solutions).
The wall clock time to compute the solution depends on the system size and, thus, on the number of discrete points.
Conversely, the numerical solution is more accurate with more discrete points.
Thus, we trade computation time for approximation accuracy by specifying the parameters that determine the number of discrete points.
For the FV scheme, the number of axial discrete points in the column is given by the number of volume cells ``NCOL``.
For the DG scheme, the number of axial discrete points in the column is given by the number of polynomial interpolation nodes (= ``POLYDEG`` + 1) times the number of DG elements ``NELEM``.
The LRMP and GRM additionally consider particle equations that are also discretized.
In the spatially discretized equations, a single particle is incorporated at each axial discrete point, which increases the total number of DOF per axial point, especially for the GRM where particles are fully resolved.
The parameters for the GRM particle discretization are given for FV in ``NPAR`` and for DG in ``PAR_POLYDEG`` and ``PAR_NELEM``.

Order of convergence
--------------------

The computational performance of a numerical method depends on its theoretical order of convergence.
The order of convergence refers to the rate at which the method's approximation approaches the exact solution under refinement of the spatial grid.

A higher-order method can be faster than a low-order method:
Imagine a high- and a low-order method's approximation to exhibit similarly bad approximation accuracy due to a coarse spatial resolution.
Refining the grid for both methods by the same number of discrete points improves the approximation accuracy of the higher-order method more than the other one.
Thus, the low-order method requires more DOFs and ultimately more compute time to compute a solution of the same accuracy.

The theoretical order of convergence is an asymptotic property, however.
Having the exact solution, we can compute an experimental order of convergence (EOC) via the formula

.. math::
    :label: ModelColumn

    \begin{aligned}
        EOC_k = \frac{log(\varepsilon_{k+1} / \varepsilon_{k}) }{log(n_{k} / n_{k+1})},
    \end{aligned}

with :math:`\varepsilon_{k}` and :math:`n_{k}` denoting some error norm and the degrees of freedom of the kth approximation.
The EOC approaches the theoretical order of convergence for :math:`k \rightarrow \infty` but is typically lower for underresolved problems.
High-order methods typically suffer from start-off problems, i.e. they typically won't exhibit their high order until the grid is fine enough and a certain accuracy is already reached.
That is, increasing the number of discrete points from, e.g., 2 to 4 typically does not improve the solution according to the theoretical order of convergence but by a much smaller EOC.
The EOC is highly problem-dependent, and it is generally unknown when a high-order method will actually be faster than a lower-order method.
Experience shows that higher-order methods work well for smooth solutions.

The theoretical order of convergence for the CADET-FV scheme is fixed at 2.
For the CADET-DG scheme, it is :math:`N_d + 1` with :math:`N_d` denoting the polynomial degree, and can thus be user-defined by specifying the field ``POLYDEG`` (and ``PAR_POLYDEG`` for the GRM).
As a convergence order of :math:`\gt 6` is hardly realized within the approximation error of engineering tolerance (due to start-off problems), we recommend a maximum polynomial order of 5.
As the FV scheme oftentimes yields an EOC of around 2.5 and is computationally more enhanced (less arithmetic operations per DOF and customized factorization) than the DG code, we recommend a polynomial degree of at least 3 to top this.

Smooth solutions
----------------

Smoothness in functions is characterized by the absence of sudden changes, reflected in the continuity and differentiability of the function and its derivatives.
In numerical simulation, smoothness indicators are often based on derivatives or moments of the solution.
That is, strong gradients and high frequencies are used to identify non-smooth parts of the solution.
Godunov's order barrier theorem shows why the concept of smoothness plays a crucial role in the deployment of numerical methods.
It states that linear high-order methods that are monotonous are at most first-order accurate.
Linear higher-order (:math:`\gt 1`) methods thus suffer from artificial oscillations at non-smooth parts of the solution, specifically at discontinuities and strong gradients.
Some higher-order methods, such as CADET-FV (2nd order), contain a non-linear mechanism to suppress these oscillations.
The non-linear WENO mechanism employed in CADET-FV can be fine-tuned via the fields specified here :ref:`flux_restruction_methods`.
Unfortunately, non-linear higher-order methods (order :math:`\geq 3`) are either not applicable (e.g., undefined boundary treatment) or have other shortcomings, such as more highly problem-dependent parameters.

CADET-DG is a linear high-order method (arbitrary order) and thus exhibits oscillatory behaviour at strong gradients, which increases the approximation error and results in a smaller EOC for lower resolutions.
As strong gradients are a local phenomena that can be captured by employing more discrete points, DG becomes more performant again for higher resolutions.
This, however, might happen after the engineering error tolerance is by far surpassed.
Hence, CADET-FV as a stabilized lower-order method can be more performant, depending on the setting.
The DG scheme reduces its oscillatory behaviour by adding artificial numerical dispersion at element interfaces.
Thus, the use of a lower polynomial degree and more elements is recommended for rather non-smooth problems.

In Chromatography, mathematical discontinuities never happen, as there are always some dispersive effects in reality.
Chromatography models, however, allow for discontinuities if dispersion parameters are set to zero.
Moreover, steep and self-sharpening concentration fronts might appear due to binding.
Binding models that might cause self-sharpening concentration fronts are often associated with competitive Langmuir type isotherms for components with differently strong binding properties.
Nonetheless, a lot of chromatography settings yield rather smooth concentration profiles, for which DG is the better choice in terms of computational performance.

Recommendations on the choice of spatial discretization methods
---------------------------------------------------------------

We recommend the FV method for

- Small problem sizes, e.g., low spatial resolution with the LRM
- Problems with strong gradients, e.g., no or low dispersion and bindings that create sharp fronts
- Bindings that mathematically require positive values or exhibit strange behaviour with negative concentration values

We recommend the DG method for

- Large problem sizes, e.g., high resolutions and more complex models (i.e. the LRMP and specifically the GRM)
- Smooth problems, e.g., sufficient dispersion

Recommendations on DG discretization parameters
-----------------------------------------------

- Employ an axial polynomial degree between 3 and 5
- Select a lower axial polynomial degree for non smooth tendency and employ more elements instead. Converse choice for smooth problems
- Adjust the DG particle polynomial degree to control approximation accuracy; leave the number of elements at one. Make exceptions if very steep gradients occur inside the particles or when specific parts of the particle domain are more interesting (here, you can resolve more interesting parts by a user-defined spacing of multiple elements)
- The field ``EXACT_INTEGRATION`` specifies the DG polynomial integration method. The default value of 0 (collocation DG) is expected to be slightly more performant in most settings

Refinement strategy
-------------------

A common problem in numerical simulation is that the number of discrete points required to yield an accurate approximation within a specific tolerance is unknown.
We thus recommend determining the approximation error via comparison with a refined reference approximation.
Both the theoretical order of convergence and the EOC can be used to estimate the required number of discrete points.

Note on DG solution vector
--------------------------

Any liquid or solid concentration within the column or particles is reported on the discrete points that are employed by the method.
That is, DG yields a piece-wise polynomial approximation on Lagrange-Gauss-Lobatto nodes.
If the solution is desired on a different grid, element-wise polynomial interpolation should be applied, and element interface values must be averaged.
