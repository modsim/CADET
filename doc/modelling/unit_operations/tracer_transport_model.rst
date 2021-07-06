.. _tracer_transport_model:

Tracer Transport model 
~~~~~~~~~~~~~~~~~~~~~~

The Tracer Transport model in CADET is based on the class of mechanistical compartment models introduced by Jonas Bühler et al. 2013. 
Its purpose is the determination of transport parameters from PET (positron emission tomography) or MRI (magnetic resonance imgaing) based tracer transport studies.
For that the model represents main functions of vascular transport pathways: axial transport of the tracer, diffusion in axial direction, lateral exchange between compartments and storage of tracer in compartments.



.. figure:: tracer_transport_model.png

    The model class consists of :math:`N` one-dimensional spatially parallel comparments. 
    In each comparment tracer can be transported with flux velocities :math:`v_i` while undergoing axial diffusion. 
    Between each compartment exchange of the tracer can take place. The exchange rates :math:`e_{ij}` specify the lateral exchange between to compartments :math:`i` and :math:`j`.


The model class is defined by a system of partial differential equations:

.. math::
   	\frac{\partial \overrightarrow{\rho}(x,t)}{\partial t} = \left(\boldsymbol{A}^T-\boldsymbol{V}\frac{\partial}{\partial X}+\boldsymbol{D}\frac{\partial^2}{\partial X^2} \right){\overrightarrow{\rho}(x,t)} 


- :math:`\overrightarrow{\rho}=({\rho}_1 \dots {\rho}_N)^T` is the tracer density distribution within each compartment :math:`N` at all spatial points :math:`x` and time points :math:`t`.
- The coupling matrix :math:`A` contains exchange rates :math:`e_{ij}` :math:`[s^{-1}]` describing the lateral tracer transport from compartment :math:`i` to compartment :math:`j`. All diagonal elements :math:`e_{ii}` in the first term are zero indicating there is no tracer exchange of one compartment with itself. The second term ensures mass conservation and removes exchanged tracer from each compartment respectively, by subtracting the sum of all exchange rates of a row (and therfore compartment) from the diagonal. The third term describes the decay of a radioactive tracer at a tracer specific rate :math:`\lambda`.
- The matrix :math:`V` contains the flux velocities :math:`v_{N}` for each compartment.


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
    \lambda I


.. math::

    V=\begin{bmatrix} 
    v_1 &  & 0 \\
     & \ddots & \\
     0 &  & v_N
    \end{bmatrix}


The velocities as well as the exchange rates can be zero. A chart of all resulting valid models of the model family can be found in Bühler et al. 2013.

Python Interface 
~~~~~~~~~~~~~~~~
(This section will probably be moved to doc/interface)

- EXCHANGE_MATRIX: Matrix containing all exchange rates :math:`e_{ij}` between the compartments. 

.. math::
    
    E=\begin{bmatrix} 
    0 & e_{12} & \dots & e_{1N} \\
    e_{21} & \ddots & & \vdots\\
    \vdots & & \ddots & e_{(N-1)N}\\
    e_{N1} & \dots & e_{N(N-1)} & 0 
    \end{bmatrix}-    

- FLUX_VECTOR: Vector with all flux rates :math:`v_N` of each comparment.
- DECAY_RATE: The material specific value for the radioactive tracer. 
  




