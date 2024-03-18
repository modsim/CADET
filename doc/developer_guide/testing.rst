.. _testing

CADET-Core testing
==================

Run the tests
^^^^^^^^^^^^^

If you want to run tests in CADET-Core you need to ensure that in the `CMakeSettings.json` file:

1. the ``cmakeCommandArgs`` contain:
     1. ``-DENABLE_TESTS=ON`` to enable building the test runner
     2. ``-DENABLE_STATIC_LINK_LAPACK=ON -DENABLE_STATIC_LINK_DEPS=ON`` to create statically linked dependencies
2. `variables` contains:
```
  {
    "name": "HDF5_USE_STATIC_LIBRARIES",
    "value": "1",
    "type": "STRING"
  },
  {
    "name": "BUILD_SHARED_LIBS",
    "value": "0",
    "type": "STRING"
  },
```

Then you can find the testRunner(.exe) in ``CADET-root/build/test/Release``, to run tests in debug mode, in ``CADET-root/build/test/Debug``.

To debug specific tests (with flag [testHere]) from the Visual Studio IDE, you can add the following configuration to the launch.vs.json file mentioned in the :ref:`debugging` section:

```
    {
      "type": "default",
      "project": "CMakeLists.txt",
      "projectTarget": "testRunner.exe (test\\Debug\\testRunner.exe)",
      "name": "testRunner.exe (test\\Debug\\testRunner.exe)",
      "args": [
        "[testHere]"
      ]
    }
```


Expand the tests
^^^^^^^^^^^^^^^^

Every model or numerical extension made to CADET-Core has to be tested adequatly.
Tests to verify the Jacobian implementation and the convergence to a reference solution have to be implemented.
Every major functionality of the model/numerical extension should be tested in a separate test case.

If no analytical reference solutions are available, you can use numerical reference solution.
To this end, we have utilized CADET-Database and CADET-Reference to ensure reproducibility of the tests.
That is, model setups should be defined in CADET-Database to ensure that we are always testing the same setting.
A high resolution and a specific low resolution reference solution as well as convergence tables should be generated in CADET-Reference.
One test set should should recompute and compare to the low resolution reference solution, to ensure that the solver has not changed.
This test should be included in the CI pipeline by adding the [CI] flag (make sure that this simulation does not take too long).
The convergence test should not be added to the standard CI but only be rerun on release.

Maintain the tests
^^^^^^^^^^^^^^^^^^

Some tests might break over time, so here are some notes on how to maintain them properly.
