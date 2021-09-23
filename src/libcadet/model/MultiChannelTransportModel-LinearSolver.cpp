// =============================================================================
//  CADET
//
//  Copyright © 2008-2021: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include "model/MultiChannelTransportModel.hpp"
#include "AdUtils.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

namespace cadet
{

namespace model
{

/**
 * @brief Computes the solution of the linear system involving the system Jacobian
 * @details The system \f[ \left( \frac{\partial F}{\partial y} + \alpha \frac{\partial F}{\partial \dot{y}} \right) x = b \f]
 *          has to be solved. The right hand side \f$ b \f$ is given by @p rhs, the Jacobians are evaluated at the
 *          point \f$(y, \dot{y})\f$ given by @p y and @p yDot. The residual @p res at this point, \f$ F(t, y, \dot{y}) \f$,
 *          may help with this. Error weights (see IDAS guide) are given in @p weight. The solution is returned in @p rhs.
 *
 * @param [in] t Current time point
 * @param [in] alpha Value of \f$ \alpha \f$ (arises from BDF time discretization)
 * @param [in] outerTol Error tolerance for the solution of the linear system from outer Newton iteration
 * @param [in,out] rhs On entry the right hand side of the linear equation system, on exit the solution
 * @param [in] weight Vector with error weights
 * @param [in] simState State of the simulation (state vector and its time derivatives) at which the Jacobian is evaluated
 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
 */
int MultiChannelTransportModel::linearSolve(double t, double alpha, double outerTol, double* const rhs, double const* const weight,
	const ConstSimulationState& simState)
{
	BENCH_SCOPE(_timerLinearSolve);

	Indexer idxr(_disc);

	// ==== Step 1: Factorize diagonal Jacobian blocks

	// Factorize partial Jacobians only if required

	if (_factorizeJacobian)
	{
		// Assemble and factorize discretized bulk Jacobian
		const bool result = _convDispOp.assembleAndFactorizeDiscretizedJacobian(alpha);
		if (cadet_unlikely(!result))
		{
			LOG(Error) << "Factorize() failed for bulk block";
		}

		// Do not factorize again at next call without changed Jacobians
		_factorizeJacobian = false;
	} // if (_factorizeJacobian)

	// ====== Step 1.5: Solve J c_uo = b_uo - A * c_in = b_uo - A*b_in

	_jacInlet.multiplySubtract(rhs, rhs + idxr.offsetC());

	// ==== Step 2: Solve diagonal Jacobian blocks J_i to get y_i = J_i^{-1} b_i
	// The result is stored in rhs (in-place solution)

	const bool result = _convDispOp.solveDiscretizedJacobian(rhs + idxr.offsetC(), weight + idxr.offsetC(), nullptr, outerTol);
	if (cadet_unlikely(!result))
	{
		LOG(Error) << "Solve() failed for bulk block";
	}

	// The full solution is now stored in rhs
	return 0;
}

}  // namespace model

}  // namespace cadet
