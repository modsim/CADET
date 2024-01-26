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
class IParameterDependence;
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
	model::IParameterDependence* create(const std::string& name) const;

	/**
	 * @brief Registers the given parameter dependence implementation
	 * @param [in] name Name of the IParameterDependence implementation
	 * @param [in] factory Function that creates an object of the IParameterDependence class
	 */
	void registerModel(const std::string& name, std::function<model::IParameterDependence*()> factory);

	/**
	 * @brief Returns whether a parameter dependence of the given name @p name exists
	 * @param [in] name Name of the parameter dependence
	 * @return @c true if a parameter dependence of this name exists, otherwise @c false
	 */
	bool exists(const std::string& name) const;

protected:
	/**
	 * @brief Registers an IParameterDependence
	 * @param [in] name Name of the parameter dependence
	 * @tparam ParamDep_t Type of the parameter dependence
	 */
	template <class ParamDep_t> void registerModel(const std::string& name);

	/**
	 * @brief Registers an IParameterDependence
	 * @details The name of the parameter dependence is inferred from the static function
	 * IParameterDependence::identifier().
	 * @tparam ParamDep_t Type of the parameter dependence
	 */
	template <class ParamDep_t> void registerModel();

	std::unordered_map<std::string, std::function<model::IParameterDependence*()>>
		_paramDeps; //!< Map with factory functions
};

} // namespace cadet

#endif // LIBCADET_PARAMETERDEPENDENCYFACTORY_HPP_
