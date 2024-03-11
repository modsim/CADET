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
			void registerLiquidSaltSolidParamDependence(std::unordered_map<std::string, std::function<model::IParameterStateDependence*()>>& paramDeps);
			void registerDummyParamDependence(std::unordered_map<std::string, std::function<model::IParameterStateDependence*()>>& paramDeps);
			void registerDummyParamDependence(std::unordered_map<std::string, std::function<model::IParameterParameterDependence*()>>& paramDeps);
			void registerIdentityParamDependence(std::unordered_map<std::string, std::function<model::IParameterStateDependence*()>>& paramDeps);
			void registerIdentityParamDependence(std::unordered_map<std::string, std::function<model::IParameterParameterDependence*()>>& paramDeps);
			void registerPowerLawParamDependence(std::unordered_map<std::string, std::function<model::IParameterParameterDependence*()>>& paramDeps);
		}
	}

	ParameterDependenceFactory::ParameterDependenceFactory()
	{
		// Register all ParamState dependencies
		model::paramdep::registerLiquidSaltSolidParamDependence(_paramStateDeps);
		model::paramdep::registerDummyParamDependence(_paramStateDeps);
		model::paramdep::registerIdentityParamDependence(_paramStateDeps);

		// Register all ParamParam dependencies
		model::paramdep::registerDummyParamDependence(_paramParamDeps);
		model::paramdep::registerIdentityParamDependence(_paramParamDeps);
		model::paramdep::registerPowerLawParamDependence(_paramParamDeps);
	}

	ParameterDependenceFactory::~ParameterDependenceFactory() { }

	template <class ParamDep_t>
	void ParameterDependenceFactory::registerStateDependence(const std::string& name)
	{
		_paramStateDeps[name] = []() { return new ParamDep_t(); };
	}

	template <class ParamDep_t>
	void ParameterDependenceFactory::registerStateDependence()
	{
		registerStateDependence<ParamDep_t>(ParamDep_t::identifier());
	}

	template <class ParamDep_t>
	void ParameterDependenceFactory::registerParameterDependence(const std::string& name)
	{
		_paramParamDeps[name] = []() { return new ParamDep_t(); };
	}

	template <class ParamDep_t>
	void ParameterDependenceFactory::registerParameterDependence()
	{
		registerParameterDependence<ParamDep_t>(ParamDep_t::identifier());
	}

	model::IParameterStateDependence* ParameterDependenceFactory::createStateDependence(const std::string& name) const
	{
		const auto it = _paramStateDeps.find(name);
		if (it == _paramStateDeps.end())
		{
			// ParameterDependence was not found
			return nullptr;
		}

		// Call factory function (thanks to type erasure of std::function we can store 
		// all factory functions in one container)
		return it->second();
	}

	model::IParameterParameterDependence* ParameterDependenceFactory::createParameterDependence(const std::string& name) const
	{
		const auto it = _paramParamDeps.find(name);
		if (it == _paramParamDeps.end())
		{
			// ParameterDependence was not found
			return nullptr;
		}

		// Call factory function (thanks to type erasure of std::function we can store 
		// all factory functions in one container)
		return it->second();
	}

	void ParameterDependenceFactory::registerModel(const std::string& name, std::function<model::IParameterStateDependence*()> factory)
	{
		if (_paramStateDeps.find(name) == _paramStateDeps.end())
			_paramStateDeps[name] = factory;
		else
			throw InvalidParameterException("IParameterStateDependence implementation with the name " + name + " is already registered and cannot be overwritten");
	}

	void ParameterDependenceFactory::registerModel(const std::string& name, std::function<model::IParameterParameterDependence*()> factory)
	{
		if (_paramParamDeps.find(name) == _paramParamDeps.end())
			_paramParamDeps[name] = factory;
		else
			throw InvalidParameterException("IParameterParameterDependence implementation with the name " + name + " is already registered and cannot be overwritten");
	}

	bool ParameterDependenceFactory::stateDependenceExists(const std::string& name) const
	{
		return _paramStateDeps.find(name) != _paramStateDeps.end();
	}

	bool ParameterDependenceFactory::parameterDependenceExists(const std::string& name) const
	{
		return _paramParamDeps.find(name) != _paramParamDeps.end();
	}
} // namespace cadet
