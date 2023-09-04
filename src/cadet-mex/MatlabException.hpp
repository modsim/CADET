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
 * Provides Matlab related exceptions
 */

#ifndef CADET_MEX_MATLABEXCEPTION_HPP_
#define CADET_MEX_MATLABEXCEPTION_HPP_

#include <string>
#include <stdexcept>

namespace cadet
{

namespace mex
{

class MatlabException : public std::runtime_error
{
public:
	MatlabException(const std::string& message) : std::runtime_error(message)
	{
	}
};

} // namespace mex
} // namespace cadet

#endif // CADET_MEX_MATLABEXCEPTION_HPP_
