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
 * Defines the ModelBuilder interface.
 */

#ifndef LIBCADET_MODELBUILDER_HPP_
#define LIBCADET_MODELBUILDER_HPP_

#include <string>
#include <functional>

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"
#include "cadet/ParameterId.hpp"

namespace cadet
{

class IParameterProvider;
class IModel;
class IInletProfile;
class IExternalFunction;
class IModelSystem;

/**
 * @brief Provides functionality to build a model
 * @details Builds a system of unit operation models or creates single unit operation models
 *          from given parameters.
 *
 *          The IModelBuilder owns all IModelSystem it created. The ownership of a specific system
 *          can be released by calling detachSystem(). Detached IModelSystem objects can be destroyed
 *          by calling destroySystem().
 *
 *          Note that the IModelBuilder does not own the created IModel objects. Unit operation
 *          models are supposed to be assigned to an IModelSystem which owns them. However, wrongfully
 *          created IModel objects can be destroyed by calling destroyUnitOperation().
 *
 *          Once registered, external functions are also owned by the IModelBuilder and can be released
 *          via detachExternalFunction(). Otherwise they will be deleted when the IModelBuilder is
 *          deleted. The external functions are indexed by the IModelBuilder, which serves as their ID.
 */
class CADET_API IModelBuilder
{
public:
	virtual ~IModelBuilder() CADET_NOEXCEPT
	{
	}

	/**
	 * @brief Automatically constructs a system of unit operations and populates it from provided parameters
	 * @details Creates a unit operation system and uses the provided parameters to populate it with
	 *          configured unit operation models that are also initialized. If one of the unit operation
	 *          models could not be constructed from the parameters or an error occurred, @c nullptr
	 *          is returned.
	 *
	 *          The created IModelSystem is owned by the creating instance of IModelBuilder.
	 *          It is automatically destroyed when the IModelBuilder that created it is destroyed.
	 *          Ownership can be transferred by calling detachSystem().
	 *
	 * @param [in] paramProvider ParameterProvider from which all necessary information is read
	 * @return A fully populated system of unit operation models or @c nullptr if an error occurred
	 */
	virtual IModelSystem* createSystem(IParameterProvider& paramProvider) = 0;

	/**
	 * @brief Creates an empty IModelSystem
	 * @details The created IModelSystem is empty and not configured
	 *
	 *          The created IModelSystem is owned by the creating instance of IModelBuilder.
	 *          It is automatically destroyed when the IModelBuilder that created it is destroyed.
	 *          Ownership can be transferred by calling detachSystem().
	 * @return Empty IModelSystem
	 */
	virtual IModelSystem* createSystem() = 0;

	/**
	 * @brief Detaches the given IModelSystem from this IModelBuilder instance
	 * @details After the call this IModelBuilder object does not longer own the
	 *          given IModelSystem.
	 * @param [in] sys IModelSystem object to be detached
	 */
	virtual void detachSystem(IModelSystem const* sys) = 0;

	/**
	 * @brief Destroys the given IModelSystem
	 * @details Frees all allocated memory corresponding to the given IModelSystem.
	 *          Note that the given @p sys must not be owned by some IModelBuilder object,
	 *          otherwise delete will be called on its memory location multiple times.
	 * @param [in] sys Ownerless IModelSystem object to be deleted
	 */
	virtual void destroySystem(IModelSystem* sys) = 0;

	/**
	 * @brief Automatically constructs a unit operation model from provided parameters
	 * @details Uses the provided parameters to create the correct unit operation model and
	 *          initializes it with the corresponding parameter values. If the model could
	 *          not be constructed from the provided parameters or an error occurred, @c nullptr is
	 *          returned.
	 *
	 *          The created unit operation model is not owned by the IModelBuilder.
	 *          Ownership of unit operation models is handled by IModelSystem.
	 *          Unit operation models can be destroyed by calling destroyUnitOperation().
	 *
	 * @param [in] paramProvider ParameterProvider from which all necessary information is read
	 * @param [in] uoId Unit operation index assigned to the created unit operation
	 * @return A fully populated unit operation model or @c nullptr if an error occurred
	 */
	virtual IModel* createUnitOperation(IParameterProvider& paramProvider, UnitOpIdx uoId) = 0;

	/**
	 * @brief Creates a unit operation model
	 * @details The created unit operation model is not owned by the IModelBuilder.
	 *          Ownership of unit operation models is handled by IModelSystem.
	 *          Unit operation models can be destroyed by calling destroyUnitOperation().
	 *
	 * @param [in] uoType Name of the unit operation model
	 * @param [in] uoId Unit operation index assigned to the created unit operation
	 * @return Uninitialized unit operation model or @c nullptr if an error occurred
	 */
	virtual IModel* createUnitOperation(const std::string& uoType, UnitOpIdx uoId) = 0;

	/**
	 * @brief Destroys the given IModel
	 * @details Frees all allocated memory corresponding to the given IModel.
	 *          Note that the given @p unitOp must not be owned by some IModelSystem object,
	 *          otherwise delete will be called on its memory location multiple times.
	 * @param [in] unitOp Ownerless IModel object to be deleted
	 */
	virtual void destroyUnitOperation(IModel* unitOp) = 0;

	/**
	 * @brief Registers an inlet profile type
	 * @param [in] name Name of the inlet profile type
	 * @param [in] factory Factory function that creates the registered IInletProfile
	 */
	virtual void registerInletType(const std::string& name, std::function<IInletProfile*(void)> factory) = 0;

	/**
	 * @brief Registers an external function type
	 * @param [in] name Name of the external function type
	 * @param [in] factory Factory function that creates the registered IExternalFunction
	 */
	virtual void registerExternalFunctionType(const std::string& name,
											  std::function<IExternalFunction*(void)> factory) = 0;
};

} // namespace cadet

#endif // LIBCADET_MODELBUILDER_HPP_
