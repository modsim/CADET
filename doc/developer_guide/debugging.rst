.. _debugging

CADET-Core debugging
====================

We advice to use an IDE like MS Visual Studio to debug CADET-Core.

If the debugger does not stop at a breakpoint even though it is set in the correct source file, this might be because a corresponding libcadet file was build and is being run. You can find these files under `build\src\libcadet`. Alternatively, you can set a breakpoint in an early stage, e.g. in the driver file and manually step through the code.

To run a specific simulation with the Visual Studio debugguer, you can add the launch.vs.json file provided here to the .vs folder

.. literalinclude:: launch.vs.json
   :language: json