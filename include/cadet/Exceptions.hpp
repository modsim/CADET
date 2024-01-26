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
 * Defines exceptions.
 */

#ifndef LIBCADET_EXCEPTIONS_HPP_
#define LIBCADET_EXCEPTIONS_HPP_

#include <stdexcept>

#include "cadet/LibExportImport.hpp"

namespace cadet
{

/**
 * @brief Signals invalid parameter or option values
 */
class CADET_API InvalidParameterException : public std::domain_error
{
public:
	explicit InvalidParameterException(const std::string& what_arg) : std::domain_error(what_arg)
	{
	}
	explicit InvalidParameterException(const char* what_arg) : std::domain_error(what_arg)
	{
	}
};

/**
 * @brief Signals errors during the time integration process
 */
class CADET_API IntegrationException : public std::runtime_error
{
public:
	explicit IntegrationException(const std::string& what_arg) : std::runtime_error(what_arg)
	{
	}
	explicit IntegrationException(const char* what_arg) : std::runtime_error(what_arg)
	{
	}
};

} // namespace cadet

#endif // LIBCADET_EXCEPTIONS_HPP_
