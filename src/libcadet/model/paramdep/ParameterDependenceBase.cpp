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

#include "model/paramdep/ParameterDependenceBase.hpp"
#include "cadet/Exceptions.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include <iterator>
#include <algorithm>
#include <numeric>

namespace cadet
{

namespace model
{

ParameterStateDependenceBase::ParameterStateDependenceBase() : _nComp(0), _nBoundStates(nullptr) { }
ParameterStateDependenceBase::~ParameterStateDependenceBase() CADET_NOEXCEPT
{
}

bool ParameterStateDependenceBase::configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBound, unsigned int const* boundOffset)
{
	_nComp = nComp;
	_nBoundStates = nBound;
	_boundOffset = boundOffset;

	if (nBound)
		_nTotalBoundStates = std::accumulate(nBound, nBound + nComp, 0u);
	else
		_nTotalBoundStates = 0;

	return true;
}

bool ParameterStateDependenceBase::configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, const std::string& name)
{
	// Clear all parameters and reconfigure
	_parameters.clear();
	return configureImpl(paramProvider, unitOpIdx, parTypeIdx, name);
}

std::unordered_map<ParameterId, double> ParameterStateDependenceBase::getAllParameterValues() const
{
	std::unordered_map<ParameterId, double> data;
	std::transform(_parameters.begin(), _parameters.end(), std::inserter(data, data.end()),
	               [](const std::pair<const ParameterId, active*>& p) { return std::make_pair(p.first, static_cast<double>(*p.second)); });
	return data;
}

bool ParameterStateDependenceBase::hasParameter(const ParameterId& pId) const
{
	return _parameters.find(pId) != _parameters.end();
}

bool ParameterStateDependenceBase::setParameter(const ParameterId& pId, int value)
{
	return false;
}

bool ParameterStateDependenceBase::setParameter(const ParameterId& pId, double value)
{
	auto paramHandle = _parameters.find(pId);
	if (paramHandle != _parameters.end())
	{
		paramHandle->second->setValue(value);
		return true;
	}

	return false;
}

bool ParameterStateDependenceBase::setParameter(const ParameterId& pId, bool value)
{
	return false;
}

active* ParameterStateDependenceBase::getParameter(const ParameterId& pId)
{
	auto paramHandle = _parameters.find(pId);
	if (paramHandle != _parameters.end())
	{
		return paramHandle->second;
	}

	return nullptr;
}


ParameterParameterDependenceBase::ParameterParameterDependenceBase() { }
ParameterParameterDependenceBase::~ParameterParameterDependenceBase() CADET_NOEXCEPT
{
}

bool ParameterParameterDependenceBase::configureModelDiscretization(IParameterProvider& paramProvider)
{
	return true;
}

bool ParameterParameterDependenceBase::configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, BoundStateIdx bndIdx, const std::string& name)
{
	// Clear all parameters and reconfigure
	_parameters.clear();
	return configureImpl(paramProvider, unitOpIdx, parTypeIdx, bndIdx, name);
}

std::unordered_map<ParameterId, double> ParameterParameterDependenceBase::getAllParameterValues() const
{
	std::unordered_map<ParameterId, double> data;
	std::transform(_parameters.begin(), _parameters.end(), std::inserter(data, data.end()),
	               [](const std::pair<const ParameterId, active*>& p) { return std::make_pair(p.first, static_cast<double>(*p.second)); });
	return data;
}

bool ParameterParameterDependenceBase::hasParameter(const ParameterId& pId) const
{
	return _parameters.find(pId) != _parameters.end();
}

bool ParameterParameterDependenceBase::setParameter(const ParameterId& pId, int value)
{
	return false;
}

bool ParameterParameterDependenceBase::setParameter(const ParameterId& pId, double value)
{
	auto paramHandle = _parameters.find(pId);
	if (paramHandle != _parameters.end())
	{
		paramHandle->second->setValue(value);
		return true;
	}

	return false;
}

bool ParameterParameterDependenceBase::setParameter(const ParameterId& pId, bool value)
{
	return false;
}

active* ParameterParameterDependenceBase::getParameter(const ParameterId& pId)
{
	auto paramHandle = _parameters.find(pId);
	if (paramHandle != _parameters.end())
	{
		return paramHandle->second;
	}

	return nullptr;
}

}  // namespace model

}  // namespace cadet
