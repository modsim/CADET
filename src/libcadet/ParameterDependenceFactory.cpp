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

#include "ParameterDependenceFactory.hpp"
#include "cadet/Exceptions.hpp"

namespace cadet
{
namespace model
{
namespace paramdep
{
void registerLiquidSaltSolidParamDependence(
	std::unordered_map<std::string, std::function<model::IParameterDependence*()>>& paramDeps);
void registerDummyParamDependence(
	std::unordered_map<std::string, std::function<model::IParameterDependence*()>>& paramDeps);
} // namespace paramdep
} // namespace model

ParameterDependenceFactory::ParameterDependenceFactory()
{
	// Register all ParameterDependencies here
	model::paramdep::registerLiquidSaltSolidParamDependence(_paramDeps);
	model::paramdep::registerDummyParamDependence(_paramDeps);
}

ParameterDependenceFactory::~ParameterDependenceFactory()
{
}

template <class ParamDep_t> void ParameterDependenceFactory::registerModel(const std::string& name)
{
	_paramDeps[name] = []() { return new ParamDep_t(); };
}

template <class ParamDep_t> void ParameterDependenceFactory::registerModel()
{
	registerModel<ParamDep_t>(ParamDep_t::identifier());
}

model::IParameterDependence* ParameterDependenceFactory::create(const std::string& name) const
{
	const auto it = _paramDeps.find(name);
	if (it == _paramDeps.end())
	{
		// ParameterDependencieswas not found
		return nullptr;
	}

	// Call factory function (thanks to type erasure of std::function we can store
	// all factory functions in one container)
	return it->second();
}

void ParameterDependenceFactory::registerModel(const std::string& name,
											   std::function<model::IParameterDependence*()> factory)
{
	if (_paramDeps.find(name) == _paramDeps.end())
		_paramDeps[name] = factory;
	else
		throw InvalidParameterException("IParameterDependence implementation with the name " + name +
										" is already registered and cannot be overwritten");
}

bool ParameterDependenceFactory::exists(const std::string& name) const
{
	return _paramDeps.find(name) != _paramDeps.end();
}
} // namespace cadet
