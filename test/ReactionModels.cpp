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

#include <catch.hpp>

#include "ReactionModelTests.hpp"

TEST_CASE("MassActionLaw kinetic analytic Jacobian vs AD", "[MassActionLaw],[ReactionModel],[Jacobian],[AD]")
{
	const unsigned int nBound[] = {1, 2, 1};
	const double point[] = {1.0, 2.0, 1.4, 2.1, 0.2, 1.1, 1.8};
	cadet::test::reaction::testDynamicJacobianAD("MASS_ACTION_LAW", 3, nBound,
												 R"json({
			"MAL_KFWD_BULK": [1.0, 2.0, 0.4],
			"MAL_KBWD_BULK": [0.0, 0.2, 1.5],
			"MAL_STOICHIOMETRY_BULK": [ 1.0, -2.0,  3.0,
			                           -1.0,  0.0, -2.0,
			                            0.0,  1.0,  1.0],
			"MAL_EXPONENTS_BULK_FWD": [ 1.2,  0.0,  0.0,
			                            0.0,  1.3,  2.2,
			                            0.0,  1.0,  1.1],
			"MAL_EXPONENTS_BULK_BWD": [ 0.8,  2.1,  1.0,
			                            1.3,  1.0,  0.0,
			                            0.0,  0.0,  1.4],

			"MAL_KFWD_LIQUID": [1.0, 2.0, 0.8, 1.2, 2.1],
			"MAL_KBWD_LIQUID": [0.1, 0.2, 2.4, 1.9, 0.8],
			"MAL_STOICHIOMETRY_LIQUID": [ 1.0, -2.0,  3.0,  1.0, -2.0,
			                             -1.0,  0.0, -2.0, -3.0, -1.0, 
			                              0.0,  1.0,  1.0,  2.0,  3.0],
			"MAL_EXPONENTS_LIQUID_FWD": [ 0.4,  0.0,  2.0,  1.2,  0.0,
			                              1.0,  1.0,  2.0,  0.0,  1.0, 
			                              0.0,  2.0,  0.0,  2.4,  1.4],
			"MAL_EXPONENTS_LIQUID_BWD": [ 1.0,  2.0,  0.0,  0.2,  2.0,
			                              0.0,  1.0,  2.0,  3.0,  1.0, 
			                              1.0,  0.0,  0.0,  1.0,  0.0],
			"MAL_EXPONENTS_LIQUID_FWD_MODSOLID": [ 1.0,  0.0,  0.0,  1.2,  0.0,
			                                       1.0,  1.6,  2.1,  0.0,  1.0, 
			                                       0.0,  0.0,  0.0,  0.0,  1.0, 
			                                       0.0,  0.0,  0.0,  2.4,  0.0],
			"MAL_EXPONENTS_LIQUID_BWD_MODSOLID": [ 0.0,  2.0,  0.0,  0.0,  0.0,
			                                       0.0,  0.0,  0.0,  1.0,  0.0, 
			                                       0.0,  1.0,  2.0,  1.2,  0.0, 
			                                       1.0,  0.0,  0.0,  0.0,  1.4],

			"MAL_KFWD_SOLID": [1.0, 2.0],
			"MAL_KBWD_SOLID": [0.8, 1.2],
			"MAL_STOICHIOMETRY_SOLID": [ 1.0, -2.0,
			                            -1.0,  0.0,
			                            -1.0,  2.0,
			                             0.0,  1.0],
			"MAL_EXPONENTS_SOLID_FWD": [ 1.0,  0.0,
			                             0.0,  0.2,
			                             1.2,  1.8,
			                             0.0,  1.0],
			"MAL_EXPONENTS_SOLID_BWD": [ 0.0,  2.0,
			                             2.0,  0.0,
			                             0.0,  0.0,
			                             1.0,  0.8],
			"MAL_EXPONENTS_SOLID_FWD_MODLIQUID": [ 1.0, 0.0,
			                                       0.0, 2.0,
			                                       0.0, 1.5],
			"MAL_EXPONENTS_SOLID_BWD_MODLIQUID": [ 1.2, 1.6,
			                                       2.0, 0.0,
			                                       1.8, 0.0]
		})json",
												 point, 1e-15, 1e-15);
}
