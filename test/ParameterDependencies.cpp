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

#include "ParamDepTests.hpp"
#include "ParameterDependencies.hpp"

CADET_PARAMDEPTEST("LIQUID_SALT_EXPONENTIAL", (1, 0, 2, 1), (1.1, 1.8, 0.9, 1.4, 1.7, 0.8, 2.1, 1.5),
				   R"json( "PD_EXPFACTOR": [1.0, 2.0, 1.5, 1.3],
	        "PD_EXPARGMULT": [0.1, 0.2, 0.3, 0.4]
	)json")

CADET_PARAMDEPTEST("LIQUID_SALT_POWER", (1, 0, 2, 1), (1.1, 1.8, 0.9, 1.4, 1.7, 0.8, 2.1, 1.5),
				   R"json( "PD_POWFACTOR": [1.0, 2.0, 1.5, 1.3],
	        "PD_POWEXP": [0.1, 0.2, 0.3, 0.4]
	)json")

CADET_PARAMDEPTEST("LIQUID_SALT_COLLOIDAL_AFFINITY", (1, 0, 2, 1), (1.1, 1.8, 0.9, 1.4, 1.7, 0.8, 2.1, 1.5),
				   R"json( "PD_LOGKEQEXP": [1.0, 2.0, 1.5, 1.3],
	        "PD_LOGKEQFACTOR": [0.1, 0.2, 0.3, 0.4],
	        "PD_LOGKEQCONST": [0.8, 1.1, 1.8, 1.6],
	        "PD_POWFACTOR": [0.9, 1.3, 1.7, 2.2],
	        "PD_POWEXP": [2.1, 1.9, 0.5, 2.1, 3.3],
	        "PD_EXPFACTOR": [0.9, 1.1, 1.4, 1.25],
	        "PD_EXPARGMULT": [1.8, 2.9, 2.2, 1.1]
	)json")
