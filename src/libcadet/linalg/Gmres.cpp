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

#include "cadet/cadetCompilerInfo.hpp"
#include "linalg/Gmres.hpp"

#if CADET_SUNDIALS_IFACE == 2
#include <sundials/sundials_spgmr.h>
#elif CADET_SUNDIALS_IFACE == 3
#include <sunlinsol/sunlinsol_spgmr.h>
#endif

#include "SundialsVector.hpp"

#include <type_traits>

namespace cadet
{

namespace linalg
{

// Wrapper function that calls the user function with the supplied user data
int gmresCallback(void* userData, N_Vector v, N_Vector z)
{
	Gmres* const g = static_cast<Gmres*>(userData);

#ifdef CADET_BENCHMARK_MODE
	++g->_numIter;
#endif

	Gmres::MatrixVectorMultFun callback = g->matrixVectorMultiplier();
	return callback(g->userData(), NVEC_DATA(v), NVEC_DATA(z));
}

Gmres::Gmres() CADET_NOEXCEPT :
#if CADET_SUNDIALS_IFACE == 2
	_mem(nullptr),
#elif CADET_SUNDIALS_IFACE == 3
	_linearSolver(nullptr),
#endif
	_ortho(Orthogonalization::ModifiedGramSchmidt),
	_maxRestarts(0),
	_matrixSize(0),
	_matVecMul(nullptr),
	_userData(nullptr)
{
#ifdef CADET_BENCHMARK_MODE
	_numIter = 0;
#endif
}

Gmres::~Gmres() CADET_NOEXCEPT
{
#if CADET_SUNDIALS_IFACE == 2
	if (_mem)
		SpgmrFree(_mem);
#elif CADET_SUNDIALS_IFACE == 3
	if (_linearSolver)
		SUNLinSolFree(_linearSolver);
#endif
}

void Gmres::initialize(unsigned int matrixSize, unsigned int maxKrylov)
{
	initialize(matrixSize, maxKrylov, Orthogonalization::ModifiedGramSchmidt, 10);
}

void Gmres::initialize(unsigned int matrixSize, unsigned int maxKrylov, Orthogonalization om, unsigned int maxRestarts)
{
	_matrixSize = matrixSize;
	if (maxKrylov == 0)
		maxKrylov = _matrixSize;

	_maxRestarts = maxRestarts;
	_ortho = om;

	// Create a template vector for the malloc routine of SPGMR
	N_Vector NV_tmpl = NVec_New(matrixSize);
	NVec_Const(0.0, NV_tmpl);

	// Size of allocated memory is either _maxKrylov or _cc.neq_bnd()
#if CADET_SUNDIALS_IFACE == 2
	_mem = SpgmrMalloc(maxKrylov, NV_tmpl);
#elif CADET_SUNDIALS_IFACE == 3
	_linearSolver = SUNSPGMR(NV_tmpl, PREC_NONE, maxKrylov);
	SUNLinSolSetATimes(_linearSolver, this, &gmresCallback);
	SUNLinSolInitialize_SPGMR(_linearSolver);
#endif

	NVec_Destroy(NV_tmpl);
}

int Gmres::solve(double tolerance, double const* weight, double const* rhs, double* sol)
{
	// Create init-guess/solution vector by bending pointer
	N_Vector NV_sol = NVec_NewEmpty(_matrixSize);
	NVEC_DATA(NV_sol) = sol;

	// Create weight vector by bending pointer
	N_Vector NV_weight = NVec_NewEmpty(_matrixSize);
	NVEC_DATA(NV_weight) = const_cast<double*>(weight);

	// Create right hand side vector by pointer bending
	N_Vector NV_rhs = NVec_NewEmpty(_matrixSize);
	NVEC_DATA(NV_rhs) = const_cast<double*>(rhs);

	const int gsType = static_cast<typename std::underlying_type<Orthogonalization>::type>(_ortho);

#if CADET_SUNDIALS_IFACE == 2
	int nIter = 0;
	int nPrecondSolve = 0;
	double resNorm = -1.0;
	const int flag = SpgmrSolve(_mem, this, NV_sol, NV_rhs, PREC_NONE, gsType, tolerance, _maxRestarts, NULL, NV_weight,
								NV_weight, &gmresCallback, NULL, &resNorm, &nIter, &nPrecondSolve);
#elif CADET_SUNDIALS_IFACE == 3
	SUNSPGMRSetGSType(_linearSolver, gsType);
	SUNSPGMRSetMaxRestarts(_linearSolver, _maxRestarts);
	SUNLinSolSetScalingVectors(_linearSolver, NV_weight, NV_weight);
	SUNLinSolSetup(_linearSolver, nullptr);
	const int flag = SUNLinSolSolve(_linearSolver, nullptr, NV_sol, NV_rhs, tolerance);

#if defined(CADET_DEBUG) || defined(CADET_BENCHMARK_MODE)
	const int nIter = SUNLinSolNumIters(_linearSolver);
	const double resNorm = SUNLinSolResNorm(_linearSolver);
#endif
#endif

	// Free NVector memory space
	NVec_Destroy(NV_rhs);
	NVec_Destroy(NV_weight);
	NVec_Destroy(NV_sol);

	return flag;
}

#if CADET_SUNDIALS_IFACE == 2
const char* Gmres::getReturnFlagName(int flag) const CADET_NOEXCEPT
{
	switch (flag)
	{
	case 0:
		return "SPGMR_SUCCESS"; // Converged
	case 1:
		return "SPGMR_RES_REDUCED"; // Did not converge, but reduced
									// norm of residual

	case 2:
		return "SPGMR_CONV_FAIL"; // Failed to converge
	case 3:
		return "SPGMR_QRFACT_FAIL"; // QRfact found singular matrix
	case 4:
		return "SPGMR_PSOLVE_FAIL_REC"; // psolve failed recoverably
	case 5:
		return "SPGMR_ATIMES_FAIL_REC"; // atimes failed recoverably
	case 6:
		return "SPGMR_PSET_FAIL_REC"; // pset faild recoverably

	case -1:
		return "SPGMR_MEM_NULL"; // mem argument is NULL
	case -2:
		return "SPGMR_ATIMES_FAIL_UNREC"; // atimes returned failure flag
	case -3:
		return "SPGMR_PSOLVE_FAIL_UNREC"; // psolve failed unrecoverably
	case -4:
		return "SPGMR_GS_FAIL"; // Gram-Schmidt routine faiuled
	case -5:
		return "SPGMR_QRSOL_FAIL"; // QRsol found singular R
	case -6:
		return "SPGMR_PSET_FAIL_UNREC"; // pset failed unrecoverably
	default:
		return "NO_VALID_FLAG";
	}
}
#elif CADET_SUNDIALS_IFACE == 3
const char* Gmres::getReturnFlagName(int flag) const CADET_NOEXCEPT
{
	switch (flag)
	{
	case 0:
		return "SUNLS_SUCCESS"; // successful/converged

	case -1:
		return "SUNLS_MEM_NULL"; // mem argument is NULL
	case -2:
		return "SUNLS_ILL_INPUT"; // illegal function input
	case -3:
		return "SUNLS_MEM_FAIL"; // failed memory access
	case -4:
		return "SUNLS_ATIMES_FAIL_UNREC"; // atimes unrecoverable failure
	case -5:
		return "SUNLS_PSET_FAIL_UNREC"; // pset unrecoverable failure
	case -6:
		return "SUNLS_PSOLVE_FAIL_UNREC"; // psolve unrecoverable failure
	case -7:
		return "SUNLS_PACKAGE_FAIL_UNREC"; // external package unrec. fail
	case -8:
		return "SUNLS_GS_FAIL"; // Gram-Schmidt failure
	case -9:
		return "SUNLS_QRSOL_FAIL"; // QRsol found singular R

	case 1:
		return "SUNLS_RES_REDUCED"; // nonconv. solve, resid reduced
	case 2:
		return "SUNLS_CONV_FAIL"; // nonconvergent solve
	case 3:
		return "SUNLS_ATIMES_FAIL_REC"; // atimes failed recoverably
	case 4:
		return "SUNLS_PSET_FAIL_REC"; // pset failed recoverably
	case 5:
		return "SUNLS_PSOLVE_FAIL_REC"; // psolve failed recoverably
	case 6:
		return "SUNLS_PACKAGE_FAIL_REC"; // external package recov. fail
	case 7:
		return "SUNLS_QRFACT_FAIL"; // QRfact found singular matrix
	case 8:
		return "SUNLS_LUFACT_FAIL"; // LUfact found singular matrix

	default:
		return "NO_VALID_FLAG";
	}
}
#endif

} // namespace linalg

} // namespace cadet
