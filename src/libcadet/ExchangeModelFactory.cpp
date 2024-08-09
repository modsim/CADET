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

#include "ExchangeModelFactory.hpp"
#include "cadet/Exceptions.hpp"

#include "model/binding/SimplifiedMultiStateStericMassActionBinding.hpp"

namespace cadet
{
	namespace model
	{
		namespace exchange
		{
			void registerLinearExModel(std::unordered_map<std::string, std::function<model::IPhaseTransitionModel* ()>>& exchange);
		}
	}

	ExchangeModelFactory::ExchangeModelFactory()
	{
		// Register all ExchangeModels here
		model::exchange::registerLinearExModel(_exchangeModels);

	}

	ExchangeModelFactory::~ExchangeModelFactory() { }

	template <class ExchangeModel_t>
	void ExchangeModelFactory::registerModel(const std::string& name)
	{
		_exchangeModels[name] = []() { return new ExchangeModel_t(); };
	}

	template <class ExchangeModel_t>
	void ExchangeModelFactory::registerModel()
	{
		registerModel<ExchangeModel_t>(ExchangeModel_t::identifier());
	}

	model::IPhaseTransitionModel* ExchangeModelFactory::create(const std::string& name) const
	{
		const auto it = _exchangeModels.find(name);
		if (it == _exchangeModels.end())
		{
			// BindingModel was not found
			return nullptr;
		}

		// Call factory function (thanks to type erasure of std::function we can store 
		// all factory functions in one container)
		return it->second();
	}

	void ExchangeModelFactory::registerModel(const std::string& name, std::function<model::IPhaseTransitionModel*()> factory)
	{
		if (_exchangeModels.find(name) == _exchangeModels.end())
			_exchangeModels[name] = factory;
		else
			throw InvalidParameterException("IPhaseTransition implementation with the name " + name + " is already registered and cannot be overwritten");
	}

	bool ExchangeModelFactory::exists(const std::string& name) const
	{
		return _exchangeModels.find(name) != _exchangeModels.end();
	}
} // namespace cadet
