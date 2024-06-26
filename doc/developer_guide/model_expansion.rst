.. _model_expansion:

Model Expansion
===============

Binding Model
^^^^^^^^^^^^^

New binding models can be implemented to extend the chromatography model family implemented in CADET-Core.
For now, please refer to the `forum post <https://forum.cadet-web.de/t/registration-implementation-and-testing-of-new-binding-model-in-cadet/533>`_.

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
   a. Parameter sensitivities: Use AD (active) type in the residual implementation, i.e. ``ParamType`` and ``ResidualType`` (template types). 
   b. Jacobian calculation via AD (can be used to verify the analytical implementation): Use AD (active) type in the residual implementation, i.e. ``StateType``. Additionally, you need to implement the following functions to enable the AD Jacobian: ``requiredADdirs()``, ``prepareADvectors``, ``extractJacobianFromAD()``, ``useAnalyticJacobian()``.

Publication
^^^^^^^^^^^
Many extensions of CADET-Core result in both software and paper publications.
Over the years we have established standard procedures to ensure good quality of the publication, including research data management and reproducability of the results.
An important part of the publication procedure on the software side is the implementation of tests:
Every model or numerical extension made to CADET-Core has to be tested adequatly before it can be merged into the master branch.
Even if your extension is not planned to become a contribution to the master branch, rigorous testing should still be implemented as it is essential to ensure confidence in your code.
Please refer to the :ref:`testing` section for more technical information on the implementation of tests in CADET-Core.

We highly recommend reading the reference tests subsection in the :ref:`testing` section before writing the paper, as the testing procedure that we describe strongly overlaps with the model/method validation part, which should be part of the publication.
