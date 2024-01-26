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

#include "nonlin/CompositeSolver.hpp"
#include "cadet/ParameterProvider.hpp"

#include <string>
#include <algorithm>

namespace cadet
{

namespace nonlin
{

CompositeSolver::CompositeSolver()
{
}
CompositeSolver::~CompositeSolver()
{
	for (std::vector<Solver*>::iterator it = _solvers.begin(); it != _solvers.end(); ++it)
		delete (*it);
}

bool CompositeSolver::configure(IParameterProvider& paramProvider)
{
	// Add solvers
	if (paramProvider.exists("SUBSOLVERS"))
	{
		const std::vector<std::string> subSolvers = paramProvider.getStringArray("SUBSOLVERS");
		for (std::vector<std::string>::const_iterator it = subSolvers.begin(); it != subSolvers.end(); ++it)
		{
			Solver* const s = createSolver(*it);
			if (s)
				_solvers.push_back(s);
		}
	}

	// Configure all solvers (including previously added solvers)
	bool success = true;
	for (std::vector<Solver*>::iterator it = _solvers.begin(); it != _solvers.end(); ++it)
	{
		success = (*it)->configure(paramProvider) && success;
	}

	return success;
}

bool CompositeSolver::solve(std::function<bool(double const* const, double* const)> residual,
							std::function<bool(double const* const, linalg::detail::DenseMatrixBase& jac)> jacobian,
							double tol, double* const point, double* const workingMemory,
							linalg::detail::DenseMatrixBase& jacMatrix, unsigned int size) const
{
	bool success = true;
	for (std::vector<Solver*>::const_iterator it = _solvers.begin(); it != _solvers.end(); ++it)
		success = (*it)->solve(residual, jacobian, tol, point, workingMemory, jacMatrix, size) && success;

	return success;
}

unsigned int CompositeSolver::workspaceSize(unsigned int problemSize) const
{
	unsigned int ws = 0;
	for (std::vector<Solver*>::const_iterator it = _solvers.begin(); it != _solvers.end(); ++it)
		ws = std::max(ws, (*it)->workspaceSize(problemSize));
	return ws;
}

unsigned int CompositeSolver::numTuningParameters() const
{
	unsigned int ntp = 0;
	for (std::vector<Solver*>::const_iterator it = _solvers.begin(); it != _solvers.end(); ++it)
		ntp += (*it)->numTuningParameters();
	return ntp;
}

void CompositeSolver::addSubsolver(Solver* const solver)
{
	if (solver)
	{
		_solvers.push_back(solver);
	}
}

} // namespace nonlin

} // namespace cadet
