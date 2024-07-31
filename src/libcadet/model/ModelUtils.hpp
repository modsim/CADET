// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

/**
 * @file 
 * Defines helper functions for models.
 */

#ifndef LIBCADET_MODELUTILS_HPP_
#define LIBCADET_MODELUTILS_HPP_

#include "cadet/ParameterId.hpp"
#include "ParamIdUtil.hpp"
#include "AutoDiff.hpp"
#include "SensParamUtil.hpp"

#include <unordered_set>
#include <unordered_map>

namespace cadet
{

namespace model
{

/**
 * @brief Returns whether the associated model has multiple bound states
 * @details Components with multiple bound states have several entries in the solid phase.
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] nComp Number of components
 * @return @c true if multiple bound states are present, otherwise @c false
 */
inline bool hasMultipleBoundStates(unsigned int const* const nBound, unsigned int nComp)
{
	for (unsigned int i = 0; i < nComp; ++i)
	{
		if (nBound[i] > 1)
			return true;
	}
	return false;
}

/**
 * @brief Returns whether the associated model has non-binding components
 * @details Non-binding components do not have a solid phase representation.
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] nComp Number of components
 * @return @c true if non-binding components are present, otherwise @c false
 */
inline bool hasNonBindingComponents(unsigned int const* const nBound, unsigned int nComp)
{
	for (unsigned int i = 0; i < nComp; ++i)
	{
		if (nBound[i] == 0)
			return true;
	}
	return false;
}

/**
 * @brief Returns the number of bound states
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] nComp Number of components
 * @return Number of bound states
 */
inline unsigned int numBoundStates(unsigned int const* const nBound, unsigned int nComp)
{
	unsigned int totalBound = 0;
	for (unsigned int i = 0; i < nComp; ++i)
	{
		totalBound += nBound[i];
	}
	return totalBound;
}

/**
 * @brief Returns the number of bound states
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] nComp Number of components
 * @return Number of bound states
 */
inline unsigned int numExchangeStates(unsigned int nChannel, unsigned int nComp)
{
	return nChannel * nComp;
}

/**
 * @brief Returns the number of binding components (i.e., components that have at least @c 1 bound state)
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] nComp Number of components
 * @return Number of binding components
 */
inline unsigned int numBindingComponents(unsigned int const* const nBound, unsigned int nComp)
{
	unsigned int totalBound = 0;
	for (unsigned int i = 0; i < nComp; ++i)
	{
		if (nBound[i] > 0)
			++totalBound;
	}
	return totalBound;
}

/**
 * @brief Returns the number of bound states of the first component with non-zero bound states
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] nComp Number of components
 * @return Number of first non-zero bound states
 */
inline unsigned int firstNonEmptyBoundStates(unsigned int const* const nBound, unsigned int nComp)
{
	unsigned int nStates = 0;
	for (unsigned int i = 0; i < nComp; ++i)
	{
		if (nBound[i] == 0)
			continue;

		nStates = nBound[i];
		break;
	}
	return nStates;
}




template <typename T>
void getAllParameterValues(std::unordered_map<ParameterId, double>& data, const std::vector<T*>& items, bool singleItem)
{
	if (items.empty())
		return;

	if (singleItem)
	{
		if (!items[0])
			return;

		const std::unordered_map<ParameterId, double> localData = items[0]->getAllParameterValues();
		for (const std::pair<const ParameterId, double>& val : localData)
			data[val.first] = val.second;
	}
	else
	{
		for (T const* bm : items)
		{
            if (!bm)
                continue;

            const std::unordered_map<ParameterId, double> localData = bm->getAllParameterValues();
			for (const std::pair<const ParameterId, double>& val : localData)
				data[val.first] = val.second;
		}
	}
}

template <typename T>
bool getParameterDouble(const ParameterId& pId, const std::vector<T*>& items, bool singleItem, double& out)
{
	// Check binding model parameters
	if (items.empty())
		return false;

	if (singleItem)
	{
		if (!items[0])
			return false;

		active const* const val = items[0]->getParameter(pId);
		if (val)
		{
			out = static_cast<double>(*val);
			return true;
		}
	}
	else
	{
		for (T* bm : items)
		{
            if (!bm)
                continue;

            active const* const val = bm->getParameter(pId);
			if (val)
			{
				out = static_cast<double>(*val);
				return true;
			}
		}
	}

	// Not found
	return false;
}

template <typename T>
bool hasParameter(const ParameterId& pId, const std::vector<T*>& items, bool singleItem)
{
	if (items.empty())
		return false;

	if (singleItem)
	{
		if (items[0] && items[0]->hasParameter(pId))
			return true;
	}
	else
	{
		for (T const* bm : items)
		{
			if (bm && bm->hasParameter(pId))
				return true;
		}
	}

	return false;
}

template <typename T, typename param_t>
bool setParameter(const ParameterId& pId, param_t value, const std::vector<T*>& items, bool singleItem)
{
	if (items.empty())
		return false;

	if (singleItem)
	{
		if (items[0] && items[0]->setParameter(pId, value))
			return true;
	}
	else
	{
		for (T* bm : items)
		{
			if (bm && bm->setParameter(pId, value))
				return true;
		}
	}

	return false;
}

template <typename T>
bool setSensitiveParameterValue(const ParameterId& pId, double value, const std::unordered_set<active*>& sensParams, const std::vector<T*>& items, bool singleItem)
{
	if (items.empty())
		return false;

	if (singleItem)
	{
		if (!items[0])
			return false;

		active* const val = items[0]->getParameter(pId);
		if (val && contains(sensParams, val))
		{
			val->setValue(value);
			return true;
		}
	}
	else
	{
		for (T* bm : items)
		{
            if (!bm)
                continue;

			active* const val = bm->getParameter(pId);
			if (val && contains(sensParams, val))
			{
				val->setValue(value);
				return true;
			}
		}
	}

	return false;
}

template <typename T>
bool setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue, std::unordered_set<active*>& sensParams, const std::vector<T*>& items, bool singleItem)
{
	if (items.empty())
		return false;

	if (singleItem)
	{
		if (!items[0])
			return false;

		active* const paramBinding = items[0]->getParameter(pId);
		if (paramBinding)
		{
			// Register parameter and set AD seed / direction
			sensParams.insert(paramBinding);
			paramBinding->setADValue(adDirection, adValue);
			return true;
		}
	}
	else
	{
		for (T* bm : items)
		{
			if (!bm)
				continue;

			active* const paramBinding = bm->getParameter(pId);
			if (paramBinding)
			{
				// Register parameter and set AD seed / direction
				sensParams.insert(paramBinding);
				paramBinding->setADValue(adDirection, adValue);
				return true;
			}
		}
	}

	return false;
}


} // namespace model
} // namespace cadet

#endif  // LIBCADET_MODELUTILS_HPP_
