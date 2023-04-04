.. _build_windows:

Build for MS Windows
====================


Prerequisites
-------------

-  CMake (>= 3.12.0)
-  Microsoft Visual Studio 2015 (Community Edition, or better)
-  Optional: MATLAB R2009a or greater
-  Optional: Git
-  Optional: Intel MKL
-  Optional: TBB

Assumed directory structure:

.. raw:: html

   <pre>
   &lt;ROOT&gt;
      |-Libs
      |   |- sundials
      |   |- hdf5
      |   |- lapack
      |   |- suitesparse
      |   |- tbb
      |-code
      |-cadet
      |-build
   </pre>

Note that the version numbers of the files and packages below are
subject to change and will not always reflect the most recent version.
In the following, we will use Visual Studio 2019 and CMake 3.17.0. We
also assume that Intel MKL is installed.

Build dependencies
------------------

HDF5
~~~~

-  Make sure that no HDF5 libraries are installed (remove already
   existing HDF5 installations)
-  Download CMake-enabled source from
   https://www.hdfgroup.org/downloads/hdf5/source-code/
-  Unzip and make sure that the directory path does not contain blank
   spaces
-  Open VS2019x64 Command Prompt and change to the unzipped directory
-  Execute
   ``ctest -S HDF5config.cmake,BUILD_GENERATOR=VS201964,INSTALLDIR="<ROOT>\Libs\hdf5" -C Release -V``
-  Extract the created ``HDF5-1.12.0-win64.zip`` file to
   ``<ROOT>/Libs/hdf5`` such that you have ``<ROOT>/Libs/hdf5/lib``

SUNDIALS
~~~~~~~~

-  Download SUNDIALS source from
   http://computation.llnl.gov/projects/sundials/sundials-software
   (version <= 3.2.1)
-  Unzip
-  Open VS2019x64 Command Prompt and change to the parent directory of
   the unzipped directory
-  Create a new folder ``sundialsbuild`` and change to it
-  Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>\Libs\sundials" -DEXAMPLES_ENABLE=OFF -DOPENMP_ENABLE=ON -DBUILD_SHARED_LIBS=OFF -G "Visual Studio 16 2019" -A x64 -DCMAKE_C_FLAGS="/GL" -DCMAKE_STATIC_LINKER_FLAGS="/LTCG" -DCMAKE_BUILD_TYPE=Release ..\sundials-3.2.1\``
-  Execute
   ``msbuild.exe INSTALL.vcxproj /p:Configuration=Release;Platform=x64``

LAPACK
~~~~~~

In the following, CLAPACK is built and used. You can also install the
freely available `Intel
MKL <https://software.intel.com/sites/campaigns/nest/>`__ which is very
fast and certainly much faster than CLAPACK. If MKL is used, skip this
step.

-  Download ``clapack-3.2.1-CMAKE.tgz`` from
   https://icl.cs.utk.edu/lapack-for-windows/clapack/index.html#build
   and unzip
-  Open VS2019x64 Command Prompt and change to the parent directory of
   the unzipped directory
-  Create a new folder ``clapackbuild`` and change to it
-  Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>\Libs\clapack" -G "Visual Studio 16 2019" -A x64 -DCMAKE_C_FLAGS="/GL" -DCMAKE_CXX_FLAGS="/GL" -DCMAKE_STATIC_LINKER_FLAGS="/LTCG" ..\clapack-3.2.1-``
-  Execute
   ``msbuild.exe CLAPACK.sln /p:Configuration=Release;Platform=x64``
-  Execute
   ``msbuild.exe INSTALL.vcxproj /p:Configuration=Release;Platform=x64``
-  Rename ``<ROOT>\Libs\clapack\lib\libf2c.lib`` to
   ``<ROOT>\Libs\clapack\lib\f2c.lib``

TBB
~~~

-  Download a release from https://github.com/oneapi-src/oneTBB/releases
-  Unzip to ``<ROOT>/Libs/tbb`` such that you have
   ``<ROOT>/Libs/tbb/README``

SuiteSparse
~~~~~~~~~~~

-  Download SuiteSparse source from
   http://faculty.cse.tamu.edu/davis/suitesparse.html

-  Download CMake build system source from
   https://github.com/jlblancoc/suitesparse-metis-for-windows

-  Unzip both into the same directory such that there are
   ``<DIR>\SuiteSparse-5.7.1`` and
   ``<DIR>\suitesparse-metis-for-windows-master``

-  Open VS2019x64 Command Prompt and change to the directory ``<DIR>``
   that contains both unzipped directories

-  Copy all SuiteSparse source files from ``<DIR>\SuiteSparse-5.7.1`` to
   ``<DIR>\suitesparse-metis-for-windows-master\SuiteSparse`` via
   ``xcopy .\SuiteSparse-5.7.1 .\suitesparse-metis-for-windows-master\SuiteSparse /s /e /y``

-  Delete the directory
   ``<DIR>\suitesparse-metis-for-windows-master\lapack_windows``

-  Create a new folder ``suitesparsebuild`` and change to it

-  If using MKL, execute
   ``set MKLROOT=C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl``

-  Using CLAPACK: Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>\Libs\suitesparse" -DCMAKE_LIBRARY_PATH="<ROOT>\Libs\clapack\lib" -DBLA_STATIC=ON -G "Visual Studio 16 2019" -A x64 -DCMAKE_C_FLAGS="/GL" -DCMAKE_STATIC_LINKER_FLAGS="/LTCG" -DCMAKE_BUILD_TYPE=Release -DBUILD_METIS=OFF ..\suitesparse-metis-for-windows-master\``

   Using MKL (sequential): Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>\Libs\suitesparse" -DBLA_VENDOR=Intel10_64lp_seq -DBLA_STATIC=ON -G "Visual Studio 16 2019" -A x64 -DCMAKE_C_FLAGS="/GL" -DCMAKE_STATIC_LINKER_FLAGS="/LTCG" -DCMAKE_BUILD_TYPE=Release -DBUILD_METIS=OFF ..\suitesparse-metis-for-windows-master\``

-  Execute
   ``msbuild.exe INSTALL.vcxproj /p:Configuration=Release;Platform=x64``

Build CADET
-----------

-  Download release of CADET or checkout from git

-  Place the source in ``<ROOT>\code`` and create the directory
   ``<ROOT>\build``

-  Open VS2019x64 Command Prompt and change to ``<ROOT>\build``

-  Execute ``set SUNDIALS_ROOT=<ROOT>\Libs\sundials``

-  Execute ``set UMFPACK_ROOT=<ROOT>\Libs\suitesparse``

-  Execute ``set TBB_ROOT=<ROOT>\Libs\tbb``

-  Execute ``set HDF5_ROOT=<ROOT>\Libs\hdf5``

-  If using MKL, execute
   ``set MKLROOT=C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2020.0.166\windows\mkl``

-  Using CLAPACK: Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>\cadet" -DBUILD_TESTS=OFF -DCMAKE_LIBRARY_PATH="<ROOT>\Libs\clapack\lib" -G "Visual Studio 16 2019" -A x64 -DBLA_VENDOR=CLAPACK -DENABLE_STATIC_LINK_LAPACK=ON ..\code\``

   Using MKL (sequential): Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>\cadet" -DBUILD_TESTS=OFF -G "Visual Studio 16 2019" -A x64 -DBLA_VENDOR=Intel10_64lp_seq -DENABLE_STATIC_LINK_LAPACK=ON ..\code\``

   Using MKL (parallel): Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>\cadet" -DBUILD_TESTS=OFF -G "Visual Studio 16 2019" -A x64 -DBLA_VENDOR=Intel10_64lp -DENABLE_STATIC_LINK_LAPACK=ON ..\code\``

-  Execute
   ``msbuild.exe INSTALL.vcxproj /p:Configuration=Release;Platform=x64``

-  If CADET does not start (i.e., ``cadet-cli`` does not run), try
   copying ``<ROOT>\Libs\tbb\bin\intel64\vc14\tbb_preview.dll`` to the
   directory that contains ``cadet-cli`` (i.e., ``<ROOT>\cadet\bin\``)

-  If the MATLAB interface does not work, try copying
   ``<ROOT>\Libs\tbb\bin\intel64\vc14\tbb_preview.dll`` to the directory
   that contains ``CadetMex.mexw64`` (i.e.,
   ``<ROOT>\cadet\matlab\bin\``)
