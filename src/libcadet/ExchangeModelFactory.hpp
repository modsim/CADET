// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

/**
 * @file 
 * Defines the ExchangeModelFactory
 */

#ifndef LIBCADET_EXCHANGEMODELFACTORY_HPP_
#define LIBCADET_EXCHANGEMODELFACTORY_HPP_

#include <string>
#include <unordered_map>
#include <functional>

namespace cadet
{

	namespace model
	{
		class IPhaseTransitionModel;
	}

	/**
	 * @brief Creates binding models
	 */
	class ExchangeModelFactory
	{
	public:
		/**
		 * @brief Construct the BindingModelFactory
		 * @details All internal binding models are registered here.
		 */
		ExchangeModelFactory();

		~ExchangeModelFactory();

		/**
		 * @brief Creates binding models with the given @p name
		 * @param [in] name Name of the binding model
		 * @return The binding model or @c NULL if a binding model with this name does not exist
		 */
		model::IPhaseTransitionModel* create(const std::string& name) const;

		/**
		 * @brief Registers the given binding model implementation
		 * @param [in] name Name of the IBindingModel implementation
		 * @param [in] factory Function that creates an object of the IBindingModel class
		 */
		void registerModel(const std::string& name, std::function<model::IPhaseTransitionModel*()> factory);

		/**
		 * @brief Returns whether a binding model of the given name @p name exists
		 * @param [in] name Name of the binding model
		 * @return @c true if a binding model of this name exists, otherwise @c false
		 */
		bool exists(const std::string& name) const;
	protected:

		/**
		 * @brief Registers an IBindingModel
		 * @param [in] name Name of the binding model
		 * @tparam BindingModel_t Type of the binding model
		 */
		template <class ExchangeModel_t>
		void registerModel(const std::string& name);

		/**
		 * @brief Registers an IBindingModel
		 * @details The name of the binding model is inferred from the static function IBindingModel::identifier().
		 * @tparam BindingModel_t Type of the binding model
		 */
		template <class ExchangeModel_t>
		void registerModel();

		std::unordered_map<std::string, std::function<model::IPhaseTransitionModel* ()>> _exchangeModels; //!< Map with factory functions
	};

} // namespace cadet

#endif  // LIBCADET_BINDINGMODELFACTORY_HPP_
