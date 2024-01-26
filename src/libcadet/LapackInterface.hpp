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

#if (defined(CADET_MATLABMEX) && defined(CADET_MATLAB_LAPACK)) || defined(CADET_LAPACK_64BIT_INT)
#include <cstdint>
#endif

namespace cadet
{

#ifdef CADET_MATLAB_LAPACK
// The Intel MKL shipped with Matlab uses 64 bit integers (ILP64 variant)
// See http://www.mathworks.com/help/matlab/matlab_external/calling-lapack-and-blas-functions-from-mex-files.html
typedef int64_t lapackInt_t;
#else
#ifdef CADET_LAPACK_64BIT_INT
typedef int64_t lapackInt_t;
#else
typedef int lapackInt_t;
#endif
#endif

// Determine LAPACK function names
#ifdef CADET_LAPACK_TRAILING_UNDERSCORE
#ifdef CADET_LAPACK_UPPERCASE
#define LAPACK_FUNC(nameLower, nameUpper) nameUpper##_
#else
#define LAPACK_FUNC(nameLower, nameUpper) nameLower##_
#endif
#else
#ifdef CADET_LAPACK_PRECEDING_UNDERSCORE
#ifdef CADET_LAPACK_UPPERCASE
#define LAPACK_FUNC(nameLower, nameUpper) _##nameUpper
#else
#define LAPACK_FUNC(nameLower, nameUpper) _##nameLower
#endif
#else
#ifdef CADET_LAPACK_UPPERCASE
#define LAPACK_FUNC(nameLower, nameUpper) nameUpper
#else
#define LAPACK_FUNC(nameLower, nameUpper) nameLower
#endif
#endif
#endif

extern "C" void LAPACK_FUNC(dgbtrf, DGBTRF)(lapackInt_t* m, lapackInt_t* n, lapackInt_t* kl, lapackInt_t* ku,
											double* ab, lapackInt_t* ldab, lapackInt_t* ipiv, lapackInt_t* info);

extern "C" void LAPACK_FUNC(dgbtrs, DGBTRS)(char* trans, lapackInt_t* n, lapackInt_t* kl, lapackInt_t* ku,
											lapackInt_t* nrhs, double* ab, lapackInt_t* ldab, lapackInt_t* ipiv,
											double* b, lapackInt_t* ldb, lapackInt_t* info);

extern "C" void LAPACK_FUNC(dgbmv, DGBMV)(char* trans, lapackInt_t* m, lapackInt_t* n, lapackInt_t* kl, lapackInt_t* ku,
										  double* alpha, double* a, lapackInt_t* lda, double* x, lapackInt_t* incx,
										  double* beta, double* y, lapackInt_t* incy);

extern "C" void LAPACK_FUNC(dgetrf, DGETRF)(lapackInt_t* m, lapackInt_t* n, double* A, lapackInt_t* lda,
											lapackInt_t* ipiv, lapackInt_t* info);

extern "C" void LAPACK_FUNC(dgetrs, DGETRS)(char* trans, lapackInt_t* n, lapackInt_t* nrhs, double* a, lapackInt_t* lda,
											lapackInt_t* ipiv, double* b, lapackInt_t* ldb, lapackInt_t* info);

extern "C" void LAPACK_FUNC(dgemv, DGEMV)(char* trans, lapackInt_t* m, lapackInt_t* n, double* alpha, double* a,
										  lapackInt_t* lda, double* x, lapackInt_t* incx, double* beta, double* y,
										  lapackInt_t* incy);

extern "C" void LAPACK_FUNC(dgels, DGELS)(char* trans, lapackInt_t* M, lapackInt_t* N, lapackInt_t* NRHS, double* A,
										  lapackInt_t* LDA, double* B, lapackInt_t* LDB, double* WORK,
										  lapackInt_t* LWORK, lapackInt_t* INFO);

extern "C" void LAPACK_FUNC(dgelqf, DGELQF)(lapackInt_t* M, lapackInt_t* N, double* A, lapackInt_t* LDA, double* TAU,
											double* WORK, lapackInt_t* LWORK, lapackInt_t* INFO);

extern "C" void LAPACK_FUNC(dormlq, DORMLQ)(char* SIDE, char* TRANS, lapackInt_t* M, lapackInt_t* N, lapackInt_t* K,
											double* A, lapackInt_t* LDA, double* TAU, double* C, lapackInt_t* LDC,
											double* WORK, lapackInt_t* LWORK, lapackInt_t* INFO);

extern "C" void LAPACK_FUNC(dtrtrs, DTRTRS)(char* UPLO, char* TRANS, char* DIAG, lapackInt_t* N, lapackInt_t* NRHS,
											double* A, lapackInt_t* LDA, double* B, lapackInt_t* LDB,
											lapackInt_t* INFO);

#ifdef CADET_LAPACK_TRAILING_UNDERSCORE
#ifdef CADET_LAPACK_UPPERCASE
#define LapackFactorDenseBanded DGBTRF_
#define LapackSolveDenseBanded DGBTRS_
#define LapackMultiplyDenseBanded DGBMV_
#define LapackFactorDense DGETRF_
#define LapackSolveDense DGETRS_
#define LapackMultiplyDense DGEMV_
#define LapackDenseLeastSquares DGELS_
#define LapackFactorLQDense DGELQF_
#define LapackMultiplyFactorizedQ DORMLQ_
#define LapackSolveTriangular DTRTRS_
#else
#define LapackFactorDenseBanded dgbtrf_
#define LapackSolveDenseBanded dgbtrs_
#define LapackMultiplyDenseBanded dgbmv_
#define LapackFactorDense dgetrf_
#define LapackSolveDense dgetrs_
#define LapackMultiplyDense dgemv_
#define LapackDenseLeastSquares dgels_
#define LapackFactorLQDense dgelqf_
#define LapackMultiplyFactorizedQ dormlq_
#define LapackSolveTriangular dtrtrs_
#endif
#else
#ifdef CADET_LAPACK_PRECEDING_UNDERSCORE
#ifdef CADET_LAPACK_UPPERCASE
#define LapackFactorDenseBanded _DGBTRF
#define LapackSolveDenseBanded _DGBTRS
#define LapackMultiplyDenseBanded _DGBMV
#define LapackFactorDense _DGETRF
#define LapackSolveDense _DGETRS
#define LapackMultiplyDense _DGEMV
#define LapackDenseLeastSquares _DGELS
#define LapackFactorLQDense _DGELQF
#define LapackMultiplyFactorizedQ _DORMLQ
#define LapackSolveTriangular _DTRTRS
#else
#define LapackFactorDenseBanded _dgbtrf
#define LapackSolveDenseBanded _dgbtrs
#define LapackMultiplyDenseBanded _dgbmv
#define LapackFactorDense _dgetrf
#define LapackSolveDense _dgetrs
#define LapackMultiplyDense _dgemv
#define LapackDenseLeastSquares _dgels
#define LapackFactorLQDense _dgelqf
#define LapackMultiplyFactorizedQ _dormlq
#define LapackSolveTriangular _dtrtrs
#endif
#else
#ifdef CADET_LAPACK_UPPERCASE
#define LapackFactorDenseBanded DGBTRF
#define LapackSolveDenseBanded DGBTRS
#define LapackMultiplyDenseBanded DGBMV
#define LapackFactorDense DGETRF
#define LapackSolveDense DGETRS
#define LapackMultiplyDense DGEMV
#define LapackDenseLeastSquares DGELS
#define LapackFactorLQDense DGELQF
#define LapackMultiplyFactorizedQ DORMLQ
#define LapackSolveTriangular DTRTRS
#else
#define LapackFactorDenseBanded dgbtrf
#define LapackSolveDenseBanded dgbtrs
#define LapackMultiplyDenseBanded dgbmv
#define LapackFactorDense dgetrf
#define LapackSolveDense dgetrs
#define LapackMultiplyDense dgemv
#define LapackDenseLeastSquares dgels
#define LapackFactorLQDense dgelqf
#define LapackMultiplyFactorizedQ dormlq
#define LapackSolveTriangular dtrtrs
#endif
#endif
#endif

#undef LAPACK_FUNC
} // namespace cadet
