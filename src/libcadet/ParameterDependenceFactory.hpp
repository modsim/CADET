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
 * Defines the ParameterDependenceFactory
 */

#ifndef LIBCADET_PARAMETERDEPENDENCYFACTORY_HPP_
#define LIBCADET_PARAMETERDEPENDENCYFACTORY_HPP_

#include <string>
#include <unordered_map>
#include <functional>

namespace cadet
{

	namespace model
	{
		class IParameterStateDependence;
		class IParameterParameterDependence;
	}

	/**
	 * @brief Creates parameter dependency objects
	 */
	class ParameterDependenceFactory
	{
	public:
		/**
		 * @brief Construct the ParameterDependenceFactory
		 * @details All internal parameter dependencies are registered here.
		 */
		ParameterDependenceFactory();

		~ParameterDependenceFactory();

		/**
		 * @brief Creates parameter dependencies with the given @p name
		 * @param [in] name Name of the parameter dependence
		 * @return The parameter dependence or @c NULL if a parameter dependence with this name does not exist
		 */
		model::IParameterStateDependence* createStateDependence(const std::string& name) const;

		/**
		 * @brief Creates parameter dependencies with the given @p name
		 * @param [in] name Name of the parameter dependence
		 * @return The parameter dependence or @c NULL if a parameter dependence with this name does not exist
		 */
		model::IParameterParameterDependence* createParameterDependence(const std::string& name) const;

		/**
		 * @brief Registers the given parameter dependence implementation
		 * @param [in] name Name of the IParameterStateDependence implementation
		 * @param [in] factory Function that creates an object of the IParameterStateDependence class
		 */
		void registerModel(const std::string& name, std::function<model::IParameterStateDependence*()> factory);

		/**
		 * @brief Registers the given parameter dependence implementation
		 * @param [in] name Name of the IParameterParameterDependence implementation
		 * @param [in] factory Function that creates an object of the IParameterParameterDependence class
		 */
		void registerModel(const std::string& name, std::function<model::IParameterParameterDependence*()> factory);

		/**
		 * @brief Returns whether a parameter dependence of the given name @p name exists
		 * @param [in] name Name of the parameter dependence
		 * @return @c true if a parameter dependence of this name exists, otherwise @c false
		 */
		bool stateDependenceExists(const std::string& name) const;

		/**
		 * @brief Returns whether a parameter dependence of the given name @p name exists
		 * @param [in] name Name of the parameter dependence
		 * @return @c true if a parameter dependence of this name exists, otherwise @c false
		 */
		bool parameterDependenceExists(const std::string& name) const;
	protected:

		/**
		 * @brief Registers an IParameterStateDependence
		 * @param [in] name Name of the parameter dependence
		 * @tparam ParamDep_t Type of the parameter dependence
		 */
		template <class ParamDep_t>
		void registerStateDependence(const std::string& name);

		/**
		 * @brief Registers an IParameterStateDependence
		 * @details The name of the parameter dependence is inferred from the static function IParameterStateDependence::identifier().
		 * @tparam ParamDep_t Type of the parameter dependence
		 */
		template <class ParamDep_t>
		void registerStateDependence();

		/**
		 * @brief Registers an IParameterParameterDependence
		 * @param [in] name Name of the parameter dependence
		 * @tparam ParamDep_t Type of the parameter dependence
		 */
		template <class ParamDep_t>
		void registerParameterDependence(const std::string& name);

		/**
		 * @brief Registers an IParameterParameterDependence
		 * @details The name of the parameter dependence is inferred from the static function IParameterParameterDependence::identifier().
		 * @tparam ParamDep_t Type of the parameter dependence
		 */
		template <class ParamDep_t>
		void registerParameterDependence();

		std::unordered_map<std::string, std::function<model::IParameterStateDependence*()>> _paramStateDeps; //!< Map with factory functions
		std::unordered_map<std::string, std::function<model::IParameterParameterDependence*()>> _paramParamDeps; //!< Map with factory functions
	};

} // namespace cadet

#endif  // LIBCADET_PARAMETERDEPENDENCYFACTORY_HPP_
