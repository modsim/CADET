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

#include "BindingModelFactory.hpp"
#include "cadet/Exceptions.hpp"

#include "model/binding/SimplifiedMultiStateStericMassActionBinding.hpp"

namespace cadet
{
	namespace model
	{
		namespace binding
		{
			void registerDummyModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerLinearModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerLangmuirModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerAntiLangmuirModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerBiLangmuirModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerKumarLangmuirModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerMobilePhaseModulatorLangmuirModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerExtendedMobilePhaseModulatorLangmuirModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerStericMassActionModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerBiStericMassActionModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerMultiStateStericMassActionModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
//			void registerSimplifiedMultiStateStericMassActionModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerSelfAssociationModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerSaskaModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerMultiComponentSpreadingModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerGeneralizedIonExchangeModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerColloidalModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerFreundlichLDFModel(std::unordered_map<std::string, std::function<model::IBindingModel* ()>>& bindings);
			void registerLangmuirLDFModel(std::unordered_map<std::string, std::function<model::IBindingModel* ()>>& bindings);
			void registerLangmuirLDFCModel(std::unordered_map<std::string, std::function<model::IBindingModel* ()>>& bindings);
			void registerBiLangmuirLDFModel(std::unordered_map<std::string, std::function<model::IBindingModel* ()>>& bindings);
			void registerHICWaterOnHydrophobicSurfacesModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerHICConstantWaterActivityModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
			void registerHICUnifiedModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings);
		}
	}

	BindingModelFactory::BindingModelFactory()
	{
		// Register all BindingModels here
		model::binding::registerDummyModel(_bindingModels);
		model::binding::registerLinearModel(_bindingModels);
		model::binding::registerLangmuirModel(_bindingModels);
		model::binding::registerAntiLangmuirModel(_bindingModels);
		model::binding::registerBiLangmuirModel(_bindingModels);
		model::binding::registerKumarLangmuirModel(_bindingModels);
		model::binding::registerMobilePhaseModulatorLangmuirModel(_bindingModels);
		model::binding::registerExtendedMobilePhaseModulatorLangmuirModel(_bindingModels);
		model::binding::registerStericMassActionModel(_bindingModels);
		model::binding::registerBiStericMassActionModel(_bindingModels);
		model::binding::registerMultiStateStericMassActionModel(_bindingModels);
//		model::binding::registerSimplifiedMultiStateStericMassActionModel(_bindingModels);
		model::binding::registerSelfAssociationModel(_bindingModels);
		model::binding::registerSaskaModel(_bindingModels);
		model::binding::registerMultiComponentSpreadingModel(_bindingModels);
		model::binding::registerGeneralizedIonExchangeModel(_bindingModels);
		model::binding::registerColloidalModel(_bindingModels);
		model::binding::registerFreundlichLDFModel(_bindingModels);
		model::binding::registerLangmuirLDFModel(_bindingModels);
		model::binding::registerLangmuirLDFCModel(_bindingModels);
		model::binding::registerBiLangmuirLDFModel(_bindingModels);
		model::binding::registerHICWaterOnHydrophobicSurfacesModel(_bindingModels);
		model::binding::registerHICConstantWaterActivityModel(_bindingModels);
		model::binding::registerHICUnifiedModel(_bindingModels);
		registerModel<model::SimplifiedMultiStateStericMassActionBinding>();
	}

	BindingModelFactory::~BindingModelFactory() { }

	template <class BindingModel_t>
	void BindingModelFactory::registerModel(const std::string& name)
	{
		_bindingModels[name] = []() { return new BindingModel_t(); };
	}

	template <class BindingModel_t>
	void BindingModelFactory::registerModel()
	{
		registerModel<BindingModel_t>(BindingModel_t::identifier());
	}

	model::IBindingModel* BindingModelFactory::create(const std::string& name) const
	{
		const auto it = _bindingModels.find(name);
		if (it == _bindingModels.end())
		{
			// BindingModel was not found
			return nullptr;
		}

		// Call factory function (thanks to type erasure of std::function we can store 
		// all factory functions in one container)
		return it->second();
	}

	void BindingModelFactory::registerModel(const std::string& name, std::function<model::IBindingModel*()> factory)
	{
		if (_bindingModels.find(name) == _bindingModels.end())
			_bindingModels[name] = factory;
		else
			throw InvalidParameterException("IBindingModel implementation with the name " + name + " is already registered and cannot be overwritten");
	}

	bool BindingModelFactory::exists(const std::string& name) const
	{
		return _bindingModels.find(name) != _bindingModels.end();
	}
} // namespace cadet
