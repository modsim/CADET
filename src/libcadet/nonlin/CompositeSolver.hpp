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
 * Provides composite solvers that subsequently apply several solvers
 */

#ifndef LIBCADET_COMPOSITESOLVER_HPP_
#define LIBCADET_COMPOSITESOLVER_HPP_

#include "nonlin/Solver.hpp"
#include <vector>

namespace cadet
{

namespace nonlin
{

/**
 * @brief Applies multiple solvers subsequently
 * @details The CompositeSolver owns all its subsolvers and destroys them when it is destroyed.
 */
class CompositeSolver : public Solver
{
public:
	CompositeSolver();
	virtual ~CompositeSolver();

	static const char* identifier()
	{
		return "COMPOSITE";
	}
	virtual const char* name() const
	{
		return CompositeSolver::identifier();
	}

	virtual bool configure(IParameterProvider& paramProvider);

	virtual unsigned int workspaceSize(unsigned int problemSize) const;

	virtual unsigned int numTuningParameters() const;

	virtual bool solve(std::function<bool(double const* const, double* const)> residual,
					   std::function<bool(double const* const, linalg::detail::DenseMatrixBase& jac)> jacobian,
					   double tol, double* const point, double* const workingMemory,
					   linalg::detail::DenseMatrixBase& jacMatrix, unsigned int size) const;

	virtual void addSubsolver(Solver* const solver);

protected:
	std::vector<Solver*> _solvers;
};

} // namespace nonlin

} // namespace cadet

#endif // LIBCADET_COMPOSITESOLVER_HPP_
