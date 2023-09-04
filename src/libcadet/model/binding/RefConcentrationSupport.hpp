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
 * Provides support functions for reading reference concentrations.
 */

#ifndef LIBCADET_BINDINGMODELREFCONCSUPPORT_HPP_
#define LIBCADET_BINDINGMODELREFCONCSUPPORT_HPP_

#include "cadet/ParameterProvider.hpp"
#include "cadet/Exceptions.hpp"

#include <vector>
#include <string>

namespace cadet
{

template <typename ValType>
inline bool readScalarParameterOrArray(std::vector<ValType>& dest, IParameterProvider& paramProvider,
									   const std::string& dataSet, unsigned int nExpand);

/**
 * @brief Reads multiple reference concentrations from the given parameter provider
 * @details The reference concentrations are read from the datasets "<PREFIX>REFC0" and "<PREFIX>REFQ"
 *          of the given @p paramProvider. If no such datasets are available, the reference
 *          concentrations are set to @c 1.0 each.
 * @param [in] paramProvider Parameter provider to read from
 * @param [in] numRefConc Number of reference concentrations to read
 * @param [in] prefix Prefix string used for assembling the parameter name
 * @param [out] refC0 Liquid phase reference concentrations
 * @param [out] refQ Solid phase reference concentrations
 * @tparam ValType Type of the parameter, such as @c active or @c double
 */
template <typename ValType>
inline void readReferenceConcentrations(IParameterProvider& paramProvider, unsigned int numRefConc,
										const std::string& prefix, std::vector<ValType>& refC0,
										std::vector<ValType>& refQ)
{
	if (paramProvider.exists(prefix + "REFC0"))
		readScalarParameterOrArray(refC0, paramProvider, prefix + "REFC0", 1);

	if (refC0.size() < numRefConc)
	{
		if (refC0.size() >= 1)
		{
			// Use first value to fill
			const ValType& val = refC0[0];
			for (unsigned int i = refC0.size(); i < numRefConc; ++i)
				refC0.push_back(val);
		}
		else
		{
			// Fill with 1.0
			for (unsigned int i = 0; i < numRefConc; ++i)
				refC0.push_back(ValType(1.0));
		}
	}

	if (paramProvider.exists(prefix + "REFQ"))
		readScalarParameterOrArray(refQ, paramProvider, prefix + "REFQ", 1);

	if (refQ.size() < numRefConc)
	{
		if (refQ.size() >= 1)
		{
			// Use first value to fill
			const ValType& val = refQ[0];
			for (unsigned int i = refQ.size(); i < numRefConc; ++i)
				refQ.push_back(val);
		}
		else
		{
			// Fill with 1.0
			for (unsigned int i = 0; i < numRefConc; ++i)
				refQ.push_back(ValType(1.0));
		}
	}

	if (refC0.size() != numRefConc)
		throw InvalidParameterException(prefix + "REFC0 has to have as many elements as there are binding sites");
	if (refQ.size() != numRefConc)
		throw InvalidParameterException(prefix + "REFQ has to have as many elements as there are binding sites");
}

/**
 * @brief Reads single reference concentrations from the given parameter provider
 * @details The reference concentrations are read from the datasets "<PREFIX>REFC0" and "<PREFIX>REFQ"
 *          of the given @p paramProvider. If no such datasets are available, the reference
 *          concentrations are set to @c 1.0 each.
 * @param [in] paramProvider Parameter provider to read from
 * @param [in] prefix Prefix string used for assembling the parameter name
 * @param [out] refC0 Liquid phase reference concentration
 * @param [out] refQ Solid phase reference concentration
 * @tparam ValType Type of the parameter, such as @c active or @c double
 */
template <typename ValType>
inline void readReferenceConcentrations(IParameterProvider& paramProvider, const std::string& prefix, ValType& refC0,
										ValType& refQ)
{
	if (paramProvider.exists(prefix + "REFC0"))
		refC0 = paramProvider.getDouble(prefix + "REFC0");
	else
		refC0 = 1.0;

	if (paramProvider.exists(prefix + "REFQ"))
		refQ = paramProvider.getDouble(prefix + "REFQ");
	else
		refQ = 1.0;
}

} // namespace cadet

#endif // LIBCADET_BINDINGMODELREFCONCSUPPORT_HPP_
