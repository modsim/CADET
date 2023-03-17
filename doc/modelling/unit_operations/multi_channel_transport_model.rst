.. _multi_channel_transport_model:

Multi Channel Transport model (MCT model) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The Multi Channel Transport (MCT) model in CADET is based on a class of compartment models introduced by Jonas Bühler et al. :cite:`BUHLER2014131`, which was originally developed in the field of plant sciences.
There it is used to determine transport and storage parameters of radioactive labelled tracer molecules from positron emission tomography (PET) or magnetic resonance imaging (MRI) based experimental data.
The model represents main functions of vascular transport pathways: axial transport of the tracer, diffusion in axial direction, lateral exchange between compartments and storage of tracer in compartments. Here, the axial direction represents the length of the stem of the plant and the lateral dimension its cross section. In the MCT context, the compartments of the model class are also referred to as channels.

The same model equations arise in describing other biological and technical processes outside of the field of plant sciences, where solutes are transported and exchanged between spatially separated compartments, for example liquid-liquid chromatography (LLC). Here, components in a mixture are separated based on their interactions with two immiscible phases of a biphasic solvent system :cite:`MORLEY2020460479`. The MCT model represents these phases by channels with respective transport and exchange properties. While the current implementation only covers linear driving forces for the exchange processes, the reaction module in CADET allows to add non-linear driving forces for the exchange processes and other chemical reactions in the channels. 

The model class consists of :math:`N` one-dimensional spatially parallel channels (see :numref:`fig-model-class`).
In each channel tracer can be transported with flux velocities :math:`v_i` while undergoing axial diffusion: :math:`d_i`.
Tracer can be laterally exchanged between any pair of channels :math:`i` and :math:`j` with rate constant :math:`e_{ij}`.

.. _fig-model-class:
.. figure:: multi_channel_transport_model_class.png

    Illustration of the Multi Channel Transport model class and relevant parameters. 
    Figure taken from Jonas Bühler et al. :cite:`BUHLER2014131`.

The model class is defined by a system of partial differential equations:

.. math::

	\frac{\partial \boldsymbol{\rho}(x,t)}{\partial t} = \left({A}^T-{V}\frac{\partial}{\partial x}+{D}\frac{\partial^2}{\partial x^2} \right){\boldsymbol{\rho}(x,t)}

- :math:`\boldsymbol{\rho}=({\rho}_1 \dots {\rho}_N)^T` is the tracer density within each channel :math:`N` at position :math:`x` and time :math:`t`.
- The coupling matrix :math:`A` contains exchange rates :math:`e_{ij}` describing the lateral tracer transport from channel :math:`i` to channel :math:`j`. All diagonal elements :math:`e_{ii}` in the first term are zero indicating there is no tracer exchange of one channel with itself. The second term ensures mass conservation and removes exchanged tracer from each channel respectively, by subtracting the sum of all exchange rates of a row (and therefore channel) from the diagonal. The third term describes the decay of a radioactive tracer at a tracer specific rate :math:`\lambda`.
.. math::

    A=\begin{bmatrix}
    0 & e_{12} & \dots & e_{1N} \\
    e_{21} & \ddots & & \vdots\\
    \vdots & & \ddots & e_{(N-1)N}\\
    e_{N1} & \dots & e_{N(N-1)} & 0
    \end{bmatrix}-   
    \begin{bmatrix}
    v_1 &  & 0 \\
     & \ddots & \\
     0 &  & v_N
    \end{bmatrix}

- The diagolal matrix :math:`D` contains the dispersion coefficients :math:`d_{i}` for each channel.

.. math::

    D=\begin{bmatrix}
    d_1 &  & 0 \\
     & \ddots & \\
     0 &  & d_N
    \end{bmatrix}


All parameters can be zero to exclude the respective mechanism from the model.
A chart of all resulting valid models of the model family can be found in Bühler et al. :cite:`BUHLER2014131`.

By default...

For each channel a seperate cross section area :math:`A_N` can be defined (see :numref:`fig-variable-areas`). 


The cross-section area :math:`A_N` is individually specified for each channel (see :numref:`fig-variable-areas`). The MCT is agnostic to the shape of these cross sections, while their ratio determines the distribution of the volumetric flow.

FixMe: Ideally add an example from another field.

.. _fig-variable-areas:
.. figure:: mct_variable_areas.png

    Scheme of exemplary channels with different cross section areas and arbitrary exchange between channels.


For information on model parameters see :ref:`multi_channel_transport_model_config`.
