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

#include "nonlin/LevenbergMarquardt.hpp"
#include "cadet/ParameterProvider.hpp"
#include "linalg/DenseMatrix.hpp"

namespace cadet
{

namespace nonlin
{

LevenbergMarquardtSolver::LevenbergMarquardtSolver() : _initDamping(1e-2), _maxIter(50)
{
}
LevenbergMarquardtSolver::~LevenbergMarquardtSolver()
{
}

bool LevenbergMarquardtSolver::configure(IParameterProvider& paramProvider)
{
	if (paramProvider.exists("INIT_DAMPING"))
		_initDamping = paramProvider.getDouble("INIT_DAMPING");
	if (paramProvider.exists("MAX_ITERATIONS"))
		_maxIter = paramProvider.getInt("MAX_ITERATIONS");
	return true;
}

bool LevenbergMarquardtSolver::solve(
	std::function<bool(double const* const, double* const)> residual,
	std::function<bool(double const* const, linalg::detail::DenseMatrixBase& jac)> jacobian, double tol,
	double* const point, double* const workingMemory, linalg::detail::DenseMatrixBase& jacMatrix,
	unsigned int size) const
{
	return levenbergMarquardt(residual, jacobian, _maxIter, tol, _initDamping, point, workingMemory, jacMatrix, size);
}

} // namespace nonlin

} // namespace cadet
