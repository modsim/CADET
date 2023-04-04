.. _freundlich_ldf_model:

Freundlich LDF
~~~~~~~~~~~~~~~

The Freundlich isotherm model is an empirical model that considers changes in the equilibrium constant of the binding process due to the heterogeneity of the surface and the variation of the interaction strength :cite:`Benedikt2019,Singh2016`.
This variant of the model is based on the linear driving force approximation (see section :ref:`ldf_model`) and is given as

.. math::
	\begin{aligned} 
		q^*_i= k_{F,i}c_{p,i}^{\frac{1}{n_{i}}}  && i = 0, \dots, N_{\text{comp}} - 1.
	\end{aligned}

No interaction between the components is considered when the model has multiple components. 
One of the limitation of this isotherm is the first order Jacobian :math:`\left(\frac{dq^*}{dc_p}\right)` tends to infinity as :math:`c_{p} \rightarrow 0` for :math:`n>1`.
To address this issue an approximation of isotherm is considered near the origin.
This approximation matches the isotherm in such a way that  :math:`q=0` at :math:`c_p=0` and also matches the first derivative of the istotherm at :math:`c_p = \epsilon`, where :math:`\epsilon` is a very small number, for example :math:`1e-14`.
The form of approximation and its derivative is given below for :math:`c_p < \varepsilon` and :math:`n>1`:

.. math::

	\begin{aligned} 
		q^* = \alpha_0+\alpha_1 c+\alpha_2 c_p^2  
	\end{aligned}
	
	\begin{aligned} 
		\frac{dq^*}{dc_p} = \alpha_1+ 2 \alpha_2 c_p 
	\end{aligned}

where :math:`\alpha_0=0` and :math:`\alpha_1` and :math:`\alpha_2` are determined from :math:`\alpha_1 \varepsilon+\alpha_2 \varepsilon^2 = k_f \varepsilon^{1/n}` and :math:`\alpha_1 + 2 \alpha_2 \varepsilon = \frac{1}{n}k_f \varepsilon^{\frac{1-n}{n}}`.
	
.. math::
	\begin{aligned}
		\alpha_1 = \frac{2 n-1}{n}k_f \varepsilon^{\frac{1-n}{n}}
	\end{aligned}
.. math::
	\begin{aligned}
		\alpha_2 = \frac{1-n}{n}k_f \varepsilon^{\frac{1-2 n}{n}}
	\end{aligned}

This approximation can be used for any pore phase cocentration :math:`c_p < \epsilon` given :math:`n>1`.
For the case, when :math:`n \le 1` no special treament near the origin is required.
For more information on model parameters required to define in CADET file format, see :ref:`freundlich_ldf_config`.

