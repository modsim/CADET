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

#include "ReactionModelFactory.hpp"
#include "cadet/Exceptions.hpp"

namespace cadet
{
	namespace model
	{
		namespace reaction
		{
			void registerMassActionLawReaction(std::unordered_map<std::string, std::function<model::IDynamicReactionModel*()>>& reactions);
			void registerDummyReaction(std::unordered_map<std::string, std::function<model::IDynamicReactionModel*()>>& reactions);
			void registerCrystallizationReaction(std::unordered_map<std::string, std::function<model::IDynamicReactionModel*()>>& reactions);
		}
	}

	ReactionModelFactory::ReactionModelFactory()
	{
		// Register all reaction models here
		model::reaction::registerDummyReaction(_dynamicModels);
		model::reaction::registerMassActionLawReaction(_dynamicModels);
		model::reaction::registerCrystallizationReaction(_dynamicModels);
	}

	ReactionModelFactory::~ReactionModelFactory() { }

	template <class ReactionModel_t>
	void ReactionModelFactory::registerDynamicModel(const std::string& name)
	{
		_dynamicModels[name] = []() { return new ReactionModel_t(); };
	}

	template <class ReactionModel_t>
	void ReactionModelFactory::registerDynamicModel()
	{
		registerDynamicModel<ReactionModel_t>(ReactionModel_t::identifier());
	}

	model::IDynamicReactionModel* ReactionModelFactory::createDynamic(const std::string& name) const
	{
		const auto it = _dynamicModels.find(name);
		if (it == _dynamicModels.end())
		{
			// Reaction model was not found
			return nullptr;
		}

		// Call factory function (thanks to type erasure of std::function we can store 
		// all factory functions in one container)
		return it->second();
	}

	void ReactionModelFactory::registerModel(const std::string& name, std::function<model::IDynamicReactionModel*()> factory)
	{
		if (_dynamicModels.find(name) == _dynamicModels.end())
			_dynamicModels[name] = factory;
		else
			throw InvalidParameterException("IDynamicReactionModel implementation with the name " + name + " is already registered and cannot be overwritten");
	}

	bool ReactionModelFactory::existsDynamic(const std::string& name) const
	{
		return _dynamicModels.find(name) != _dynamicModels.end();
	}

} // namespace cadet
