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

#include "ModelBuilderImpl.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/ModelSystem.hpp"
#include "cadet/Exceptions.hpp"
#include "cadet/InletProfile.hpp"
#include "cadet/ExternalFunction.hpp"

#include "model/ModelSystemImpl.hpp"
#include "CompileTimeConfig.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include <sstream>
#include <iomanip>

namespace cadet
{
	namespace model
	{
		void registerInletModel(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models);
		void registerOutletModel(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models);

		void registerGeneralRateModel(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models);
		void registerLumpedRateModelWithPores(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models);
		void registerLumpedRateModelWithoutPores(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models);
		void registerCSTRModel(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models);
#ifdef ENABLE_2D_MODELS
		void registerGeneralRateModel2D(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models);
		void registerMultiChannelTransportModel(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models);
#endif

		namespace inlet
		{
			void registerPiecewiseCubicPoly(std::unordered_map<std::string, std::function<IInletProfile*()>>& inlets);
		} // namespace inlet

		namespace extfun
		{
			void registerLinearInterpolation(std::unordered_map<std::string, std::function<IExternalFunction*()>>& extFuns);
			void registerPiecewiseCubicPoly(std::unordered_map<std::string, std::function<IExternalFunction*()>>& extFuns);
		} // namespace extfun
	} // namespace model

	ModelBuilder::ModelBuilder()
	{
		// Register all available models
		model::registerInletModel(_modelCreators);
		model::registerOutletModel(_modelCreators);
		model::registerGeneralRateModel(_modelCreators);
		model::registerLumpedRateModelWithPores(_modelCreators);
		model::registerLumpedRateModelWithoutPores(_modelCreators);
		model::registerCSTRModel(_modelCreators);

#ifdef ENABLE_2D_MODELS
		model::registerGeneralRateModel2D(_modelCreators);
		model::registerMultiChannelTransportModel(_modelCreators);
#endif

		// Register all available inlet profiles
		model::inlet::registerPiecewiseCubicPoly(_inletCreators);

		// Register all available external functions
		model::extfun::registerLinearInterpolation(_extFunCreators);
		model::extfun::registerPiecewiseCubicPoly(_extFunCreators);
	}

	ModelBuilder::~ModelBuilder() CADET_NOEXCEPT
	{
		for (IModelSystem* model : _models)
			delete model;
	}

	template <class UnitOpModel_t>
	void ModelBuilder::registerModel(const std::string& name)
	{
		_modelCreators[name] = [](UnitOpIdx uoId) { return new UnitOpModel_t(uoId); };
	}

	template <class UnitOpModel_t>
	void ModelBuilder::registerModel()
	{
		registerModel<UnitOpModel_t>(UnitOpModel_t::identifier());
	}

	IModelSystem* ModelBuilder::createSystem(IParameterProvider& paramProvider)
	{
		model::ModelSystem* sys = new model::ModelSystem();
		
		// Create and configure all unit operations
		bool success = true;
		unsigned int i = 0;
		std::ostringstream oss;
		oss << "unit_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
		while (paramProvider.exists(oss.str()))
		{
			// Create and configure unit operation
			paramProvider.pushScope(oss.str());

			IModel* const unitOp = createUnitOperation(paramProvider, i);

			paramProvider.popScope();

			if (unitOp)
			{
				// Model correctly created and configured -> add to system
				sys->addModel(unitOp);
			}
			else
			{
				// Something went wrong -> abort and exit
				success = false;
				break;
			}

			++i;
			oss.str("");
			oss << "unit_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
		}

		// Configure the whole system
		success = success && sys->configureModelDiscretization(paramProvider, *this) && sys->configure(paramProvider);

		if (success)
		{
			_models.push_back(sys);
			return sys;
		}
		else
		{
			delete sys;
			return nullptr;
		}
	}

	IModelSystem* ModelBuilder::createSystem()
	{
		IModelSystem* sys = new model::ModelSystem();
		_models.push_back(sys);
		return sys;
	}

	void ModelBuilder::detachSystem(IModelSystem const* sys)
	{
		for (std::vector<IModelSystem*>::iterator it = _models.begin(); it != _models.end(); ++it)
		{
			if (*it == sys)
			{
				_models.erase(it);
				break;
			}
		}
	}

	void ModelBuilder::destroySystem(IModelSystem* sys)
	{
		delete sys;
	}

	IModel* ModelBuilder::createUnitOperation(IParameterProvider& paramProvider, UnitOpIdx uoId)
	{
		const std::string uoType = paramProvider.getString("UNIT_TYPE");
		const auto it = _modelCreators.find(uoType);
		if (it == _modelCreators.end())
		{
			// Model was not found
			LOG(Error) << "Unknown unit type " << uoType << " for unit " << uoId;
			return nullptr;
		}

		// Call factory function (thanks to type erasure of std::function we can store 
		// all factory functions in one container)
		IUnitOperation* const model = it->second(uoId);

		if (!model->configureModelDiscretization(paramProvider, *this) || !model->configure(paramProvider))
		{
			LOG(Error) << "Configuration of unit " << uoId << "(" << uoType << ") failed";
			delete model;
			return nullptr;
		}

		return model;
	}

	IModel* ModelBuilder::createUnitOperation(const std::string& uoType, UnitOpIdx uoId)
	{
		const auto it = _modelCreators.find(uoType);
		if (it == _modelCreators.end())
		{
			// Model was not found
			LOG(Error) << "Unknown unit type " << uoType << " for unit " << uoId;
			return nullptr;
		}

		IUnitOperation* const model = it->second(uoId);
		return model;
	}

	void ModelBuilder::destroyUnitOperation(IModel* unitOp)
	{
		delete unitOp;
	}

	void ModelBuilder::registerInletType(const std::string& name, std::function<IInletProfile*(void)> factory)
	{
		if (_inletCreators.find(name) == _inletCreators.end())
			_inletCreators[name] = factory;
		else
			throw std::invalid_argument("INLET_TYPE " + name + " is already registered and cannot be overwritten");
	}

	void ModelBuilder::registerExternalFunctionType(const std::string& name, std::function<IExternalFunction*(void)> factory)
	{
		if (_extFunCreators.find(name) == _extFunCreators.end())
			_extFunCreators[name] = factory;
		else
			throw std::invalid_argument("EXTFUN_TYPE " + name + " is already registered and cannot be overwritten");
	}

	IInletProfile* ModelBuilder::createInletProfile(const std::string& type) const
	{
		const InletFactoryContainer_t::const_iterator it = _inletCreators.find(type);
		if (it != _inletCreators.end())
			return (it->second)();

		return nullptr;		
	}

	IExternalFunction* ModelBuilder::createExternalFunction(const std::string& type) const
	{
		const ExternalFunctionFactoryContainer_t::const_iterator it = _extFunCreators.find(type);
		if (it != _extFunCreators.end())
			return (it->second)();

		return nullptr;		
	}

	model::IBindingModel* ModelBuilder::createBindingModel(const std::string& name) const
	{
		return _bindingModels.create(name);
	}

	bool ModelBuilder::isValidBindingModel(const std::string& name) const
	{
		return _bindingModels.exists(name);
	}

	model::IDynamicReactionModel* ModelBuilder::createDynamicReactionModel(const std::string& name) const
	{
		return _reactionModels.createDynamic(name);
	}

	bool ModelBuilder::isValidDynamicReactionModel(const std::string& name) const
	{
		return _reactionModels.existsDynamic(name);
	}

	model::IParameterDependence* ModelBuilder::createParameterDependence(const std::string& name) const
	{
		return _paramDeps.create(name);
	}

	bool ModelBuilder::isValidParameterDependence(const std::string& name) const
	{
		return _paramDeps.exists(name);
	}

} // namespace cadet
