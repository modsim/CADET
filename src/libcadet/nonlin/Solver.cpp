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

#include "nonlin/Solver.hpp"
#include "nonlin/AdaptiveTrustRegionNewton.hpp"
#include "nonlin/LevenbergMarquardt.hpp"
#include "nonlin/CompositeSolver.hpp"

namespace cadet
{

namespace nonlin
{

Solver* createSolver(const std::string& name)
{
	if (name == AdaptiveTrustRegionNewtonSolver::identifier())
		return new AdaptiveTrustRegionNewtonSolver();
	if (name == RobustAdaptiveTrustRegionNewtonSolver::identifier())
		return new RobustAdaptiveTrustRegionNewtonSolver();
	if (name == LevenbergMarquardtSolver::identifier())
		return new LevenbergMarquardtSolver();
	if (name == CompositeSolver::identifier())
		return new CompositeSolver();

	// Default solver
	CompositeSolver* cs = new CompositeSolver();
	cs->addSubsolver(new RobustAdaptiveTrustRegionNewtonSolver());
	cs->addSubsolver(new LevenbergMarquardtSolver());
	return cs;
}

} // namespace nonlin

} // namespace cadet
