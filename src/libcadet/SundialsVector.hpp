// =============================================================================
//  CADET
//
//  Copyright © 2008-2022: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

/**
 * @file
 * Handles different NVector implementations of SUNDIALS and provides uniform access to them.
 * Note that, according to the SUNDIALS manual, the OpenMP implementation only improves performance
 * for state vectors of 100.000 entries or more because of threading overhead.
 */

#ifndef LIBCADET_SUNDIALSVECTOR_HPP_
#define LIBCADET_SUNDIALSVECTOR_HPP_

#ifdef CADET_SUNDIALS_OPENMP
#include <nvector/nvector_openmp.h>
#include <omp.h>

#define NVEC_DATA(x) NV_DATA_OMP(x)
#define NVEC_LENGTH(x) NV_LENGTH_OMP(x)

#define NVec_New(x) N_VNew_OpenMP(x, omp_get_max_threads())
#define NVec_Destroy N_VDestroy_OpenMP
#define NVec_DestroyArray N_VDestroyVectorArray_OpenMP
#define NVec_CloneArray N_VCloneVectorArray_OpenMP
#define NVec_NewEmpty(x) N_VNewEmpty_OpenMP(x, omp_get_max_threads())
#define NVec_SetThreads(x, nThreads) NV_NUM_THREADS_OMP(x) = nThreads
#else
#include <nvector/nvector_serial.h>

#define NVEC_DATA(x) NV_DATA_S(x)
#define NVEC_LENGTH(x) NV_LENGTH_S(x)

#define NVec_New(x) N_VNew_Serial(x)
#define NVec_Destroy N_VDestroy_Serial
#define NVec_DestroyArray N_VDestroyVectorArray_Serial
#define NVec_CloneArray N_VCloneVectorArray_Serial
#define NVec_NewEmpty N_VNewEmpty_Serial
#define NVec_SetThreads(x, nThreads)
#endif

#define NVec_Const N_VConst

#endif // LIBCADET_SUNDIALSVECTOR_HPP_
