.. _multi_channel_transport_model:

Multi Channel Transport model (MCT) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Multi Channel Transport model in CADET is based on the class of mechanistical compartment models introduced by Jonas Bühler et al. 2013 [1]_. 
Its original purpose is the determination of transport parameters from PET (positron emission tomography) or MRI (magnetic resonance imgaing) based tracer transport studies.
For that the model represents main functions of vascular transport pathways: axial transport of the tracer, diffusion in axial direction, lateral exchange between compartments and storage of tracer in compartments. 
Naturally, the Multi Channel Transport model can be applied to any similar transport processes. 

The model class consists of :math:`N` one-dimensional spatially parallel comparments (see :numref:`fig-model-class`). 
In each comparment tracer can be transported with flux velocities :math:`v_i` while undergoing axial diffusion. 
Between each compartment exchange of the tracer can take place. The exchange rates :math:`e_{ij}` specify the lateral exchange between to compartments :math:`i` and :math:`j`.

.. _fig-model-class:
.. figure:: multi_channel_transport_model_class.png

    Illustration of the Multi Channel Transport model class and relevant parameters. 
    Figure taken from Jonas Bühler et al. 2013 [1]_.

    

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


The velocities as well as the exchange rates can be zero. A chart of all resulting valid models of the model family can be found in Bühler et al. 2013 [1]_.

.. [1] 
Jonas Bühler, Eric von Lieres, Gregor Huber,
A class of compartmental models for long-distance tracer transport in plants,
Journal of Theoretical Biology,
Volume 341,
2014,
Pages 131-142,
ISSN 0022-5193,
License Number 5103570248093,
License date Jul 07 2021,
https://doi.org/10.1016/j.jtbi.2013.09.023.
(https://www.sciencedirect.com/science/article/pii/S0022519313004530)






