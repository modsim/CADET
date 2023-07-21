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
 * ModelBuilder implementation
 */

#ifndef LIBCADET_MODELBUILDER_IMPL_HPP_
#define LIBCADET_MODELBUILDER_IMPL_HPP_

#include "cadet/ModelBuilder.hpp"
#include "BindingModelFactory.hpp"
#include "ReactionModelFactory.hpp"
#include "ParameterDependenceFactory.hpp"
#include "ConfigurationHelper.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace cadet
{

class IUnitOperation;
class IInletProfile;

/**
 * @brief Provides functionality to build a model
 */
class ModelBuilder : public IModelBuilder, public IConfigHelper
{
public:

	ModelBuilder();

	virtual ~ModelBuilder() CADET_NOEXCEPT;

	virtual IModelSystem* createSystem(IParameterProvider& paramProvider);
	virtual IModelSystem* createSystem();
	virtual void detachSystem(IModelSystem const* sys);
	virtual void destroySystem(IModelSystem* sys);

	virtual IModel* createUnitOperation(IParameterProvider& paramProvider, UnitOpIdx uoId);
	virtual IModel* createUnitOperation(const std::string& uoType, UnitOpIdx uoId);
	virtual void destroyUnitOperation(IModel* unitOp);

	virtual void registerInletType(const std::string& name, std::function<IInletProfile*(void)> factory);
	virtual void registerExternalFunctionType(const std::string& name, std::function<IExternalFunction*(void)> factory);

	virtual IInletProfile* createInletProfile(const std::string& type) const;
	virtual model::IBindingModel* createBindingModel(const std::string& name) const;
	virtual bool isValidBindingModel(const std::string& name) const;
	virtual model::IDynamicReactionModel* createDynamicReactionModel(const std::string& name) const;
	virtual bool isValidDynamicReactionModel(const std::string& name) const;
	virtual model::IParameterStateDependence* createParameterStateDependence(const std::string& name) const;
	virtual bool isValidParameterStateDependence(const std::string& name) const;
	virtual model::IParameterParameterDependence* createParameterParameterDependence(const std::string& name) const;
	virtual bool isValidParameterParameterDependence(const std::string& name) const;
	virtual IExternalFunction* createExternalFunction(const std::string& type) const;

protected:

	/**
	 * @brief Registers an IUnitOperation
	 * @param [in] name Name of the model
	 * @tparam UnitOpModel_t Type of the model
	 */
	template <class UnitOpModel_t>
	void registerModel(const std::string& name);

	/**
	 * @brief Registers an IUnitOperation
	 * @details The name of the model is inferred from the static function IUnitOperation::identifier().
	 * @tparam UnitOpModel_t Type of the model
	 */
	template <class UnitOpModel_t>
	void registerModel();

	BindingModelFactory _bindingModels; //!< Factory for IBindingModel implementations
	ReactionModelFactory _reactionModels; //!< Factory for IDynamicReactionModel implementations
	ParameterDependenceFactory _paramDeps; //!< Factory for IParameterStateDependence implementations

	typedef std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>> ModelFactoryContainer_t;
	typedef std::unordered_map<std::string, std::function<IInletProfile*(void)>> InletFactoryContainer_t;
	typedef std::unordered_map<std::string, std::function<IExternalFunction*(void)>> ExternalFunctionFactoryContainer_t;

	ModelFactoryContainer_t _modelCreators; //!< Map with factory functions for models
	InletFactoryContainer_t _inletCreators; //!< Map with factory functions for inlet profiles
	ExternalFunctionFactoryContainer_t _extFunCreators; //!< Map with factory functions for external functions

	std::vector<IModelSystem*> _models; //!< Models
};

} // namespace cadet

#endif  // LIBCADET_MODELBUILDER_IMPL_HPP_
