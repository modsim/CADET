# ---------------------------------------------------------------
# Programmer:  Daniel R. Reynolds @ SMU
# ---------------------------------------------------------------
# LLNS/SMU Copyright Start
# Copyright (c) 2017, Southern Methodist University and 
# Lawrence Livermore National Security
#
# This work was performed under the auspices of the U.S. Department 
# of Energy by Southern Methodist University and Lawrence Livermore 
# National Laboratory under Contract DE-AC52-07NA27344.
# Produced at Southern Methodist University and the Lawrence 
# Livermore National Laboratory.
#
# All rights reserved.
# For details, see the LICENSE file.
# LLNS/SMU Copyright End
# ---------------------------------------------------------------
# CMakeLists.txt file for the SPGMR SUNLinearSolver library

# Add variable sunlinsolspgmr_SOURCES with the sources for the SUNLINSOLSPGMR lib
SET(sunlinsolspgmr_SOURCES sunlinsol_spgmr.c)

# Add variable shared_SOURCES with the common SUNDIALS sources which will
# also be included in the SUNLINSOLSPGMR library
SET(shared_SOURCES
  ${sundials_SOURCE_DIR}/src/sundials/sundials_math.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_nvector.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_linearsolver.c
  ${sundials_SOURCE_DIR}/src/sundials/sundials_iterative.c
  )

# Add variable sunlinsolspgmr_HEADERS with the exported SUNLINSOLSPGMR header files
SET(sunlinsolspgmr_HEADERS 
  ${sundials_SOURCE_DIR}/include/sunlinsol/sunlinsol_spgmr.h
  )

# Add source directory to include directories
INCLUDE_DIRECTORIES(.)

# Define C preprocessor flag -DBUILD_SUNDIALS_LIBRARY 
ADD_DEFINITIONS(-DBUILD_SUNDIALS_LIBRARY)

# Rules for building and installing the static library:
#  - Add the build target for the SUNLINSOLSPGMR library
#  - Set the library name and make sure it is not deleted
#  - Install the SUNLINSOLSPGMR library
ADD_LIBRARY(sundials_sunlinsolspgmr_static STATIC ${sunlinsolspgmr_SOURCES} ${shared_SOURCES})
SET_TARGET_PROPERTIES(sundials_sunlinsolspgmr_static
  PROPERTIES OUTPUT_NAME sundials_sunlinsolspgmr CLEAN_DIRECT_OUTPUT 1)
