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

#include "Weno.hpp"

namespace cadet
{

// Initialization of static WENO coefficients
const double Weno::_wenoD2[2] = {2.0 / 3.0, 1.0 / 3.0};

const double Weno::_wenoC2[2 * 2] = {0.5, -0.5, 0.5, 1.5};
const double Weno::_wenoJbvv2[2 * 3 * 3] = {0, 2, 0, -2, 0, 0, 0, -2, 2, 2, -2, 0, 0, 0, -2, 0, 2, 0};

const double Weno::_wenoD3[3] = {0.3, 0.6, 0.1};

const double Weno::_wenoC3[3 * 3] = {1.0 / 3.0,  -1.0 / 6.0, 1.0 / 3.0, 5.0 / 6.0, 5.0 / 6.0,
									 -7.0 / 6.0, -1.0 / 6.0, 1.0 / 3.0, 11.0 / 6.0};
const double Weno::_wenoJbvv3[3 * 5 * 5] = // Used to generate Jbv: vec(Jbv) = A*v
	{0,         0,         8.0 / 3,   0,         0,         -19.0 / 3, 0,        0,         11.0 / 3, 0,
	 0,         0,         0,         0,         0,         0,         0,        -19.0 / 3, 0,        8.0 / 3,
	 50.0 / 3,  0,         -13.0 / 3, -31.0 / 3, 0,         5.0 / 3,   0,        0,         0,        0,
	 0,         0,         11.0 / 3,  0,         -13.0 / 3, -31.0 / 3, 20.0 / 3, 26.0 / 3,  20.0 / 3, -31.0 / 3,
	 -13.0 / 3, 0,         11.0 / 3,  0,         0,         0,         0,        0,         0,        5.0 / 3,
	 0,         -31.0 / 3, -13.0 / 3, 0,         50.0 / 3,  8.0 / 3,   0,        -19.0 / 3, 0,        0,
	 0,         0,         0,         0,         0,         0,         11.0 / 3, 0,         0,        -19.0 / 3,
	 0,         0,         8.0 / 3,   0,         0};

} // namespace cadet
