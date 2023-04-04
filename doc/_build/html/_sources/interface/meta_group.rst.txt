.. _FFMeta:

Meta Group
==========

``FILE_FORMAT``

   Version of the file format (defaults to 040000 = 4.0.0 if omitted) with two digits per part (Major.Minor.Patch)
   
   ================  =========================
   **In/out:** In    **Type:** int
   ================  =========================
   
``CADET_VERSION``

   Version of the executed :math:`\texttt{CADET}` simulator
   
   ================  =========================
   **In/out:** Out   **Type:** string
   ================  =========================
   
``CADET_COMMIT``

   Git commit SHA1 from which the :math:`\texttt{CADET}` simulator was built
   
   ================  =========================
   **In/out:** Out   **Type:** string
   ================  =========================
   
``CADET_BRANCH``

   Git branch from which the :math:`\texttt{CADET}` simulator was built
   
   ================  =========================
   **In/out:** Out   **Type:** string
   ================  =========================
   
``TIME_SIM``

   Time that the time integration took (excluding any preparations and postprocessing)

   **Unit:** :math:`\mathrm{s}`
   
   ================  =========================
   **In/out:** Out   **Type:** double
   ================  =========================
