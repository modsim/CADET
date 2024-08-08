.. _model_expansion:

Model Expansion
===============

There are 7+ transport models and 20+ adsorption models implemented in CADET.
Please make sure that the one you are interested in is not implemented.
Additionally, some models can be mimicked by or are even equivalent to already implemented models when specific parameters are chosen, see e.g. the LRM use-case section for the :ref:`multi_channel_transport_model_model`.

The implementation of a new model follows three main steps:
1. Create a template binding/unit/reaction model from existing code
2. Register the binding/unit/reaction model in the respective CADET binding/unit/reaction model factory and add it to cmake
3. Implementation of model equations and Jacobian matrix
4. Testing

Binding Model
^^^^^^^^^^^^^

An extensive description on how to add a new binding model to CADET-Core is given in a `forum post <https://forum.cadet-web.de/t/registration-implementation-and-testing-of-new-binding-model-in-cadet/533>`_, where such an extension is described using the example of a Langmuir binding.
A more concise description is given in the following:

1.
Use the `binding model template <https://github.com/cadet/CADET-Core/tree/master/doc/developer_guide/TemplateBinding.cpp>`_ to create a new binding model file in the CADET binding model source directory

``\src\libcadet\model\binding\YourModelNameBinding.cpp``

and rename the model in the file (i.e. Template to YoureModelName).
Note that the provided template binding model is an implementation of the multi-component Langmuir binding.

2.
Register the binding model by adding

``void registerYourModelNameModel(std::unordered_map<std::string, std::function<model::IBindingModel* ()>>& bindings);``

to the ``src\libcadet\BindingModelFactory.cpp`` at the end of all the other registering functions such as ``void registerLinearModel``.
Then add

``model::binding::registerExampleModel(_bindingModels);``

to the ``BindingModelFactory.cpp`` at the end of all the other registering functions such as ``model::binding::registerLinearModel(_bindingModels)``.
The final step to register your model is to add your model to the ``\src\libcadet\CMakeLists.txt`` (again look for similar statements for the other binding models) by adding

``${CMAKE_SOURCE_DIR} /src/libcadet/model/binding/ExampleBinding.cpp``

Before continuing with the third step, you should rebuild CADET-Core to verify that the first two steps went well.

3.
The actual implementation of the new binding model follows two main steps: the configuration of the relevant mechanistic parameters and implementation of adsorption flux and Jacobian.
To set up the configuration of isotherm parameters a macro (.json script) has been included in the code, which generates the relevant code section when the user defines the parameters in the scope of this script.
To modify the script go to Line 30 in the provided file template and adjust the parameters to your needs, a more detailed description for that is provided in the `aforementioned forum post <https://forum.cadet-web.de/t/registration-implementation-and-testing-of-new-binding-model-in-cadet/533>`_.

Next, the adsorption flux equations need to be implemented int the corresponding function

``int fluxImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* y, CpStateType const* yCp, ResidualType* res, LinearBufferAllocator workSpace) const``

For an explanation of ``active`` types for the template arguments ``StateType``, ``ParamType``, ``ResidualType``, please refer to the Algorithmic Differentiation section.

Finally, the Jacobian needs to be implemented in the function

``void jacobianImpl()``

We note that the Jacobian implementation is optional but highly recommended to speed up the simulation.
If you have trouble with deriving the Jacobian or if you want to test you model first, modify the
``implementsAnalyticJacobian()``
function to return false.
By doing so, CADET-Core defaults to computing the binding `Jacobian via Algorithmic differentiation (AD) <https://doi.org/10.1016/j.ces.2015.08.050>`_.


Unit Operation
^^^^^^^^^^^^^^

The easiest way to create a new unit operation in CADET-Core is to start from an existing unit operation (e.g. ``src/libcadet/StirredTankModel``, but optimally the most similar one) and make adjustments from there.
The first, model independent steps are:

1. Add ``NewModel.cpp`` and ``NewModel.hpp`` files in ``src\libcadet\model`` folder (copy and rename existing ones). Note that we'll call the new model "NewModel" from here on, so make sure to substitute this with the actual name of your model.
2. Add ``${CMAKE_SOURCE_DIR}/src/libcadet/model/NewModel.cpp`` to the ``src/libcadet/CMakeLists.txt``
3. (Optional) In the root/CMakeLists.txt file, you can add a build option to make building your new model optional. This enables building CADET versions with and without the new extension. This can be done, e.g., when additional dependencies are required or when build time should be reduced. See e.g. ``ENABLE_GRM_2D`` or ``ENABLE_DG``.
4. Add new model to ModelBuilder (in ModelBuilderImpl.cpp, just like the other models are included):
5. Rename everything to the new model
   a. Change the ``identifier()`` function in the ``NewModel.hpp`` to return a new unique model name (here: "NewModel")
   b. Adjust the ``registerNewModel()`` function in the ``NewModel.cpp`` accordingly
   c. rename all functions, i.e. substitute the previous (copied) model name by your model name. Also do this for the header guards in the hpp file and the include in the cpp (essentially every occurrence of the old model name has to be changed).
6. Try to build (and maybe even run) the new unit operation, which at this point is essentially still the same as the copied one, to make sure everything is set up correctly.

You are now ready to implement your model in CADET-Core.
Note that you will probably have to clean up a lot of things from the copied unit operation and that you also need to change the documentation comments to match your model and implementation.

Most important functionality to be implemented:

1. configure(): Model configuration parameters are read from the .h5 file and exceptions are thrown if parameters are missing or have the wrong size or format. Furthermore, if required, implicitly given parameters are determined and memory is allocated.
2. configureModelDiscretization(): Similar to the function above but treats the numerical discretization part.
3. residualImpl(): Implements the residual formulation (i.e. function :math:`F = 0`) of the equations. Triggers updates of the (possibly state dependent) system Jacobian.
4. System Jacobian: Owned by the unit operation. Defined given by :math:`J := \frac{\partial F}{\partial y} + \alpha \frac{\partial F}{\partial \dot{y}}`, i.e. both the state and state derivative Jacobian need to be implemented.
5. Linear solve: Solves the system :math:`J x = b` with given :math:`b`.
6. Algorithmic differentiation (AD): 
   a. Parameter sensitivities: Use ``ParamType`` for all parameters and ``ResidualType`` for the residual. 
   b. Jacobian calculation via AD (can be used to verify the analytical implementation): Use ``StateType`` for the state and ``ResidualType`` for the residual. Additionally, you need to implement the following functions to enable the AD Jacobian: ``requiredADdirs()``, ``prepareADvectors``, ``extractJacobianFromAD()``, ``useAnalyticJacobian()``. For details please refer to `Püttmann et al. <https://doi.org/10.1016/j.compchemeng.2013.04.021>`_.

Testing and Publication
^^^^^^^^^^^^^^^^^^^^^^^
Many extensions of CADET-Core result in both software and paper publications.
Over the years we have established standard procedures to ensure good quality of the publication, including research data management and reproducability of the results.
An important part of the publication procedure on the software side is the implementation of tests:
Every model or method extension of CADET-Core has to be tested adequatly before it can be merged into the master branch.
Even if your extension is not planned to become a contribution to the master branch, rigorous testing should still be implemented as it is essential to ensure confidence in your code.
Please refer to the :ref:`testing` section for more technical information on the implementation of tests in CADET-Core.

We highly recommend reading the reference tests subsection within the :ref:`testing` section before writing the paper, as the testing procedure that we describe strongly overlaps with the model/method validation part, which should be part of the publication.

Algorithmic differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

AD in CADET-Core can be used to compute parameter sensitivities and/or the Jacobian of the system.
The custom AD implementation introduces an ``active`` type (see `AutoDiff.hpp <https://github.com/cadet/CADET-Core/blob/master/src/libcadet/AutoDiff.hpp>`_), which is treated as a scalar but holds multiple double values.
The first ``active`` entry holds the actual double value of the variable.
The latter ``active`` entries hold the derivatives of that variable w.r.t different directions.
Directions can either be the parameter(s) whose sensitivity we want to calculate or, if we compute the Jacobian via AD, an entry of the discrete state vector.

To use AD for a new binding model, you only need to use the template types properly:
Use ``ParamType`` and ``ResidualType`` for parameters and residual ``res`` to enable parameter sensitivities; that is, all parameters must be defined as actives in the binding model and used as ParamType in the residual function.
Use ``StateType`` and ``ResidualType`` for the state ``y`` and residual ``res`` to enable the AD Jacobian. 

To use AD for a new unit operation, you can either apply dense AD or, in case of a model with many states or spatial resolution, you need to think of the shape of the Jacobian and apply sparse AD. 
