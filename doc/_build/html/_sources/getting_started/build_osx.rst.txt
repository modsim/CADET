.. _build_osx:

Build for OSX
=============

Prerequisites
-------------

-  CMake (>= 3.12.0)
-  GCC >= 5.0 or Clang >= 3.4
-  Optional: MATLAB R2009a or greater
-  Optional: Git

Assumed directory structure:

.. raw:: html

   <pre>
   &lt;ROOT&gt;
      |-libs
      |   |- sundials
      |   |- hdf5
      |   |- superlu
      |   |- suitesparse
      |-code
      |-cadet
      |-build
   </pre>

Note that the version numbers of the files and packages below are
subject to change and will not always reflect the most recent version.

Also note that you have to use the same compiler for all packages. This
is especially important if some of the packages are installed via a
package manager such as `Homebrew <http://brew.sh/>`__ which uses the
system compiler (Clang).

Build dependencies
------------------

HDF5
~~~~

You can either build HDF5 yourself or rely on the ``hdf5`` package of
`Homebrew <http://brew.sh/>`__.

Obtain HDF5 from Homebrew: \* Open a terminal and execute
``brew install hdf5`` \* Homebrew has installed HDF5 to ``/usr/local``

Build HDF5 yourself: \* Download CMake-enabled source from
https://www.hdfgroup.org/downloads/hdf5/source-code/ \* Unzip and make
sure that the directory path does not contain blank spaces \* Open a
terminal and change to the unzipped directory \* Execute
``ctest -S HDF5config.cmake,BUILD_GENERATOR=Unix,INSTALLDIR="<ROOT>/Libs/hdf5" -C Release -V``
\* Extract the created ``HDF5-1.10.0-patch1-Darwin.tar.gz`` file to
``<ROOT>/Libs/hdf5`` such that you have ``<ROOT>/Libs/hdf5/lib``

SUNDIALS
~~~~~~~~

You can either build SUNDIALS yourself or rely on the ``sundials``
package of `Homebrew <http://brew.sh/>`__. Note that a version <= 3.2.1
is required whereas the current version provided by Homebrew is >=
5.1.0.

Obtain SUNDIALS from Homebrew: \* Open a terminal and execute
``brew install sundials`` \* Homebrew has installed SUNDIALS to
``/usr/local``

Build SUNDIALS yourself: \* Download SUNDIALS source from
http://computation.llnl.gov/projects/sundials/sundials-software \* Unzip
\* Open a terminal and change to the parent directory of the unzipped
directory \* Create a new folder ``sundialsbuild`` and change to it \*
Execute
``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>/Libs/sundials" -DEXAMPLES_ENABLE=OFF -DOPENMP_ENABLE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release ../sundials-3.2.1/``
\* Execute ``make install`` \* Delete the folder ``sundialsbuild``
(e.g., execute ``rm -rf sundialsbuild`` in the parent directory of
``sundialsbuild``)

LAPACK
~~~~~~

You can either use the native LAPACK implementation provided by Mac OS X
(vecLib, Accelerate) or install the freely available `Intel
MKL <https://software.intel.com/sites/campaigns/nest/>`__ which is very
fast and probably faster than Accelerate.

SuperLU
~~~~~~~

-  Download SuperLU source from https://github.com/xiaoyeli/superlu
-  Unzip
-  Open a terminal and change to the parent directory of the unzipped
   directory
-  Create a new folder ``superlubuild`` and change to it
-  Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>/Libs/superlu" -Denable_complex=OFF -Denable_complex16=OFF -Denable_blaslib=OFF -DCMAKE_C_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release ../SuperLU_5.2.1/``
-  Execute ``make install``
-  Delete the folder ``superlubuild`` (e.g., execute
   ``rm -rf superlubuild`` in the parent directory of ``superlubuild``)

UMFPACK
~~~~~~~

-  Download SuiteSparse source from
   http://faculty.cse.tamu.edu/davis/suitesparse.html
-  Unzip
-  Open a terminal and change to the unzipped directory
-  Execute
   ``make install INSTALL="<ROOT>/Libs/suitesparse" CHOLMOD_CONFIG=-DNPARTITION``
   or
   ``make install INSTALL="<ROOT>/Libs/suitesparse" CHOLMOD_CONFIG=-DNPARTITION AUTOCC=no CC=<COMPILER> CXX=<C++COMPILER>``
   if you want to manually select the compiler

Build CADET
-----------

-  Download release of CADET or checkout from git

-  Place the source in ``<ROOT>/code`` and create the directory
   ``<ROOT>/build``

-  Open a terminal and change to ``<ROOT>/build``

-  If you have built HDF5 yourself, execute
   ``export HDF5_ROOT=<ROOT>/Libs/hdf5``

-  If you have built SUNDIALS yourself, execute
   ``export SUNDIALS_ROOT=<ROOT>/Libs/sundials``

-  Execute ``export SUPERLU_ROOT=<ROOT>/Libs/superlu``

-  Execute ``export UMFPACK_ROOT=<ROOT>/Libs/suitesparse``

-  If using MKL, execute ``export MKLROOT=/opt/intel/mkl``

-  Using standard LAPACK: Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>/cadet" ../code/``

   Using MKL (sequential): Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>/cadet" -DBLA_VENDOR=Intel10_64lp_seq ../code/``

   Using MKL (parallel): Execute
   ``cmake -DCMAKE_INSTALL_PREFIX="<ROOT>/cadet" -DBLA_VENDOR=Intel10_64lp ../code/``

-  Execute ``make``

-  Execute ``make install``

Before running cadet, make sure to

-  add ``<ROOT>/cadet/bin`` to your ``PATH``, by executing 
   ``export PATH=<ROOT>/cadet/bin:$PATH``, and

-  add ``<ROOT>/cadet/lib`` to your ``LD_LIBRARY_PATH``, by executing 
   ``export LD_LIBRARY_PATH=<ROOT>/cadet/lib:$LD_LIBRARY_PATH``.
