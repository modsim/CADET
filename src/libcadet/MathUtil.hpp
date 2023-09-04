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

#ifndef LIBCADET_MATHUTIL_HPP_
#define LIBCADET_MATHUTIL_HPP_

#include <cmath>

#include "cadet/cadetCompilerInfo.hpp"

namespace cadet
{

/**
 * @brief Squares the given value
 * @details Calculates @f$ x * x @f$
 * @param [in] x Value that is squared
 * @return Squared value
 */
inline double sqr(const double x) CADET_NOEXCEPT
{
	return x * x;
}

#if defined(ACTIVE_SFAD) || defined(ACTIVE_SETFAD)
#endif

} // namespace cadet

#endif // LIBCADET_MATHUTIL_HPP_
