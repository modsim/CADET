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
 * Defines the ReactionModelFactory
 */

#ifndef LIBCADET_REACTIONMODELFACTORY_HPP_
#define LIBCADET_REACTIONMODELFACTORY_HPP_

#include <string>
#include <unordered_map>
#include <functional>

namespace cadet
{

namespace model
{
class IDynamicReactionModel;
}

/**
 * @brief Creates reaction models
 */
class ReactionModelFactory
{
public:
	/**
	 * @brief Construct the ReactionModelFactory
	 * @details All internal reaction models are registered here.
	 */
	ReactionModelFactory();

	~ReactionModelFactory();

	/**
	 * @brief Creates dynamic reaction models with the given @p name
	 * @param [in] name Name of the dynamic reaction model
	 * @return The dynamic reaction model or @c NULL if a dynamic reaction model with this name does not exist
	 */
	model::IDynamicReactionModel* createDynamic(const std::string& name) const;

	/**
	 * @brief Registers the given dynamic reaction model implementation
	 * @param [in] name Name of the IBindingModel implementation
	 * @param [in] factory Function that creates an object of the IBindingModel class
	 */
	void registerModel(const std::string& name, std::function<model::IDynamicReactionModel*()> factory);

	/**
	 * @brief Returns whether a dynamic reaction model of the given name @p name exists
	 * @param [in] name Name of the dynamic reaction model
	 * @return @c true if a dynamic reaction model of this name exists, otherwise @c false
	 */
	bool existsDynamic(const std::string& name) const;

protected:
	/**
	 * @brief Registers an IDynamicReactionModel
	 * @param [in] name Name of the binding model
	 * @tparam ReactionModel_t Type of the binding model
	 */
	template <class ReactionModel_t> void registerDynamicModel(const std::string& name);

	/**
	 * @brief Registers an IDynamicReactionModel
	 * @details The name of the binding model is inferred from the static function IDynamicReactionModel::identifier().
	 * @tparam ReactionModel_t Type of the binding model
	 */
	template <class ReactionModel_t> void registerDynamicModel();

	std::unordered_map<std::string, std::function<model::IDynamicReactionModel*()>>
		_dynamicModels; //!< Map with factory functions
};

} // namespace cadet

#endif // LIBCADET_REACTIONMODELFACTORY_HPP_
