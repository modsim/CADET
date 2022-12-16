.. _multi_channel_transport_model:

Multi Channel Transport model (MCT) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Multi Channel Transport (MCT) model in CADET is based on a class of compartment models introduced by Jonas Bühler et al. :cite:`BUHLER2014131`.
This model class was originally developed in the field of plant science.
There it is used to determine transport and storage parameters of radioactive labelled tracer molecules from positron emission tomography (PET) or magnetic resonance imaging (MRI) based experimental data.
The model represents main functions of vascular transport pathways: axial transport of the tracer, diffusion in axial direction, lateral exchange between compartments and storage of tracer in compartments.
Here, the axial direction is represents the length of the stem of the plant and the lateral dimension its cross section.
The MCT model naturally describes other biological and technical processes, such as cross flow filtration or extraction, in particular when combined with chemical reactions.

The model class consists of :math:`N` one-dimensional spatially parallel compartments (see :numref:`fig-model-class`).
In each compartment tracer can be transported with flux velocities :math:`v_i` while undergoing axial diffusion: :math:`d_i`.
Tracer can be laterally exchanged between any pair of compartments :math:`i` and :math:`j` with rate constant :math:`e_{ij}`.

.. _fig-model-class:
.. figure:: multi_channel_transport_model_class.png

    Illustration of the Multi Channel Transport model class and relevant parameters. 
    Figure taken from Jonas Bühler et al. :cite:`BUHLER2014131`.

FixMe: Add dispersion coefficients and decay rates with corresponding arrows to the figure.

The model class is defined by a system of partial differential equations:

.. math::

	\frac{\partial \overrightarrow{\rho}(x,t)}{\partial t} = \left(\boldsymbol{A}^T-\boldsymbol{V}\frac{\partial}{\partial x}+\boldsymbol{D}\frac{\partial^2}{\partial x^2} \right){\overrightarrow{\rho}(x,t)}

- :math:`\overrightarrow{\rho}=({\rho}_1 \dots {\rho}_N)^T` is the tracer density within each compartment :math:`N` at position :math:`x` and time :math:`t`.
- The coupling matrix :math:`\boldsymbol{A}` contains exchange rates :math:`e_{ij}` describing the lateral tracer transport from compartment :math:`i` to compartment :math:`j`. All diagonal elements :math:`e_{ii}` in the first term are zero indicating there is no tracer exchange of one compartment with itself. The second term ensures mass conservation and removes exchanged tracer from each compartment respectively, by subtracting the sum of all exchange rates of a row (and therefore compartment) from the diagonal. The third term describes the decay of a radioactive tracer at a tracer specific rate :math:`\lambda`.

.. math::

    A=\begin{bmatrix}
    0 & e_{12} & \dots & e_{1N} \\
    e_{21} & \ddots & & \vdots\\
    \vdots & & \ddots & e_{(N-1)N}\\
    e_{N1} & \dots & e_{N(N-1)} & 0
    \end{bmatrix}-   
    \begin{bmatrix}
    {\sum_{k=1}^{N} e_{1k}} &  & 0 \\
     & \ddots & \\
     0 &  & {\sum_{k=1}^{N} e_{Nk}}
    \end{bmatrix}-
    \lambda \boldsymbol{I}

- The diagonal matrix :math:`\boldsymbol{V}` contains the flux velocities :math:`v_{i}` for each compartment.

.. math::

    V=\begin{bmatrix}
    v_1 &  & 0 \\
     & \ddots & \\
     0 &  & v_N
    \end{bmatrix}

- The diagolal matrix :math:`\boldsymbol{D}` contains the dispersion coefficients :math:`d_{i}` for each compartment.

.. math::

    D=\begin{bmatrix}
    d_1 &  & 0 \\
     & \ddots & \\
     0 &  & d_N
    \end{bmatrix}


All parameters can be zero to exclude the respective mechanism from the model.
A chart of all resulting valid models of the model family can be found in Bühler et al. :cite:`BUHLER2014131`.

FixMe: Ideally add an example from another field.

For information on model parameters see :ref:`multi_channel_transport_model_config`.
