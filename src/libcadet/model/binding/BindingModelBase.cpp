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

#include "model/binding/BindingModelBase.hpp"
#include "model/ModelUtils.hpp"
#include "cadet/Exceptions.hpp"
#include "ParamReaderHelper.hpp"

#include "AdUtils.hpp"
#include "linalg/Norms.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include <iterator>
#include <algorithm>

namespace cadet
{

namespace model
{

BindingModelBase::BindingModelBase()
	: _nComp(0), _nBoundStates(nullptr), _reactionQuasistationarity(0, false), _hasQuasiStationary(false),
	  _hasDynamic(true)
{
}
BindingModelBase::~BindingModelBase() CADET_NOEXCEPT
{
}

bool BindingModelBase::configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp,
													unsigned int const* nBound, unsigned int const* boundOffset)
{
	_nComp = nComp;
	_nBoundStates = nBound;
	if (hasMultipleBoundStates(nBound, nComp) && !supportsMultistate())
		throw InvalidParameterException("Binding model does not support multiple bound states");

	_reactionQuasistationarity.resize(numBoundStates(nBound, nComp), false);

	// Read binding dynamics (quasi-stationary, kinetic)
	if (paramProvider.isArray("IS_KINETIC"))
	{
		const std::vector<int> vecKin = paramProvider.getIntArray("IS_KINETIC");
		if (vecKin.size() == 1)
		{
			// Treat an array with a single element as scalar
			std::fill(_reactionQuasistationarity.begin(), _reactionQuasistationarity.end(),
					  !static_cast<bool>(vecKin[0]));
		}
		else if (vecKin.size() < _reactionQuasistationarity.size())
		{
			// Error on too few elements
			throw InvalidParameterException("IS_KINETIC has to have at least " +
											std::to_string(_reactionQuasistationarity.size()) + " elements");
		}
		else
		{
			// Copy what we need (ignore excess values)
			std::copy_n(vecKin.begin(), _reactionQuasistationarity.size(), _reactionQuasistationarity.begin());
		}
	}
	else
	{
		const bool kineticBinding = paramProvider.getInt("IS_KINETIC");
		std::fill(_reactionQuasistationarity.begin(), _reactionQuasistationarity.end(), !kineticBinding);
	}

	_hasQuasiStationary = std::any_of(_reactionQuasistationarity.begin(), _reactionQuasistationarity.end(),
									  [](int i) -> bool { return i; });
	_hasDynamic = std::any_of(_reactionQuasistationarity.begin(), _reactionQuasistationarity.end(),
							  [](int i) -> bool { return !static_cast<bool>(i); });

	return true;
}

bool BindingModelBase::configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx)
{
	// Clear all parameters and reconfigure
	_parameters.clear();
	return configureImpl(paramProvider, unitOpIdx, parTypeIdx);
}

void BindingModelBase::fillBoundPhaseInitialParameters(ParameterId* params, UnitOpIdx unitOpIdx,
													   ParticleTypeIdx parTypeIdx) const CADET_NOEXCEPT
{
	unsigned int ctr = 0;
	for (int c = 0; c < _nComp; ++c)
	{
		for (unsigned int bp = 0; bp < _nBoundStates[c]; ++bp, ++ctr)
			params[ctr] = makeParamId(hashString("INIT_Q"), unitOpIdx, c, parTypeIdx, bp, ReactionIndep, SectionIndep);
	}
}

std::unordered_map<ParameterId, double> BindingModelBase::getAllParameterValues() const
{
	std::unordered_map<ParameterId, double> data;
	std::transform(_parameters.begin(), _parameters.end(), std::inserter(data, data.end()),
				   [](const std::pair<const ParameterId, active*>& p) {
					   return std::make_pair(p.first, static_cast<double>(*p.second));
				   });
	return data;
}

bool BindingModelBase::hasParameter(const ParameterId& pId) const
{
	return _parameters.find(pId) != _parameters.end();
}

bool BindingModelBase::setParameter(const ParameterId& pId, int value)
{
	return false;
}

bool BindingModelBase::setParameter(const ParameterId& pId, double value)
{
	auto paramHandle = _parameters.find(pId);
	if (paramHandle != _parameters.end())
	{
		paramHandle->second->setValue(value);
		return true;
	}

	return false;
}

bool BindingModelBase::setParameter(const ParameterId& pId, bool value)
{
	return false;
}

active* BindingModelBase::getParameter(const ParameterId& pId)
{
	auto paramHandle = _parameters.find(pId);
	if (paramHandle != _parameters.end())
	{
		return paramHandle->second;
	}

	return nullptr;
}

} // namespace model

} // namespace cadet
