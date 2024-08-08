.. _build_options

CADET build options
===================

For builds on MS Windows, the CMake command arguments are specified for every build configuration in the cmakeSettings.json.
For builds on Linux and MacOs, the CMake command arguments must be specified from the command line.

cmakeSettings.json
------------------

The cmakeSettings.json provides standard build configurations for a ``DEBUG`` build, a ``RELEASE`` build and a ``RELEASE_with_Debug_Info`` build.

The ``DEBUG`` build compiles code suited for debugging, i.e. the code is compiled such that every line of code is individually executed.
This way, it is possible to go through the code line by line and to set breakpoints in an IDE such as MS Visual Studio.
Additionally, the ``DEBUG`` build will print additional information during the simulation, such as the current step size and time point of the simulation.

The ``RELEASE`` build compiles optimized code and should be used to actually use CADET for simulations.

The ``RELEASE_with_Debug_Info`` build compiles optimized code but some debug information will still be printed.

The ``RELEASE_with_Tests`` and ``DEBUG_with_Test`` build options additionally build the ``testRunner``, which is required to run the integrated tests in ``CADET-Core``.

Options
-------

The following build arguments can be set in the cmakeSettings.json or from the cmake command line:

- ``DCMAKE_INSTALL_PREFIX``: location for the installed ``CADET-Core`` framework.
- ``DENABLE_STATIC_LINK_LAPACK`` Prefer static over dynamic linking of LAPACK and BLAS into the ``CADET-Core`` framework. Static linking incorporates all necessary libraries into the final executable at compile time, while dynamic linking loads libraries at runtime. Static linking produces larger executables that are less dependent on changes in the operating system. Dynamic linking allows for dynamic updates of underlying libraries and smaller compiled software.
- ``DENABLE_STATIC_LINK_DEPS``: Prefer static over dynamic linking of dependencies into the ``CADET-Core`` framework.
- ``DENABLE_STATIC_LINK_CLI``: Prefers static over dynamic linking for CADET CLI.
- ``DENABLE_TESTS``: Build the ``restRunner`` executable to evaluate the integrated tests in ``CADET-Core``.
- ``DENABLE_ANALYTIC_JACOBIAN_CHECK``: Computes both the analytical and AD Jacobian and compares them for testing purpose.
- ``DENABLE_THREADING``: Enables multi-threading capabilities. Parallelized code will be compiled, using the TBB library. Note that the non-parallelized code is faster compared to the parallelized code when only one thread is being used. The number of threads is specified in the filed ``N_THREADS``.
- ``DBLA_VENDOR``: Vendor for the BLAS & LAPACK library. If unset, the system library will be used. By default on Windows we use the Intel OneApi library, specified with ``Intel10_64lp_seq``. If a parallelized build is generated, this should be set to ``Intel10_64lp``.
- ``DENABLE_LOGGING``: Enables logging functionality.
- ``DENABLE_BENCHMARK``: Activates benchmark mode for fine-grained timing.
- ``DENABLE_PLATFORM_TIMER``: Utilizes a platform-dependent timer.
- ``DENABLE_DEBUG_THREADING``: Activates multi-threading in debug builds.
- ``DENABLE_2D_MODELS``: Builds 2D models such as the 2D general rate model and multichannel transport.
- ``DENABLE_DG``: Constructs DG variants of models.
- ``DENABLE_SUNDIALS_OPENMP``: Prefers the OpenMP vector implementation of SUNDIALS for large problems if available.
- ``DENABLE_CADET_CLI``: Builds the CADET command line interface.
- ``DENABLE_CADET_TOOLS``: Constructs CADET tools.
- ``DENABLE_PACKAGED_SUNDIALS``: Uses packaged SUNDIALS code.
- ``DENABLE_IPO``: Enables interprocedural optimization if the compiler supports it.
- ``DCMAKE_INSTALL_RPATH_USE_LINK_PATH``: Adds paths to linker search and installed rpath.

The following build arguments are exclusive to builds on MS windows:

- ``DVCPKG_TARGET_TRIPLET``: We use ``vcpkg`` to manage our dependencies. This triplet specifies which version of the dependencies should be installed. It takes the form of ``architecture-os-linking``, so ``x64-windows-static`` for our use cases.

