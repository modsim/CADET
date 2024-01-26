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

#ifndef LIBCADET_IOEXCEPTION_HPP_
#define LIBCADET_IOEXCEPTION_HPP_

#include <string>
#include <stdexcept>

namespace cadet
{

namespace io
{

class IOException : public std::runtime_error
{
public:
	IOException(const std::string& message) : std::runtime_error(message)
	{
	}
};

} // namespace io

} // namespace cadet

#endif /* LIBCADET_IOEXCEPTION_HPP_ */
