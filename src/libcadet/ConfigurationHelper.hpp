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
 * Provides an interface to create different subentitites (e.g., IInletProfile) and provide
 * some aids for configuring models.
 */

#ifndef LIBCADET_CONFIGURATIONHELPER_HPP_
#define LIBCADET_CONFIGURATIONHELPER_HPP_

#include <string>

namespace cadet
{

class IInletProfile;
class IExternalFunction;

	namespace model
	{
		class IBindingModel;
		class IDynamicReactionModel;
		class IParameterStateDependence;
		class IParameterParameterDependence;
	}

/**
 * @brief Provides means to create subentities (e.g., IInletProfile, IBindingModel)
 */
class IConfigHelper
{
public:

	/**
	 * @brief Creates an IInletProfile object of the given @p type
	 * @details The caller owns the returned IInletProfile object.
	 * @param [in] type Type of the IInletProfile object
	 * @return Object of the given IInletProfile @p type or @c nullptr if that type does not exist
	 */
	virtual IInletProfile* createInletProfile(const std::string& type) const = 0;

	/**
	 * @brief Creates an IBindingModel object of the given @p name
	 * @details The caller owns the returned IBindingModel object.
	 * @param [in] name Name of the IBindingModel object
	 * @return Object of the given IBindingModel @p name or @c nullptr if that name does not exist
	 */
	virtual model::IBindingModel* createBindingModel(const std::string& name) const = 0;

	/**
	 * @brief Checks if there is an IBindingModel of the given @p name
	 * @param [in] name Name of the IBindingModel object
	 * @return @c true if a binding model of this name exists, otherwise @c false
	 */
	virtual bool isValidBindingModel(const std::string& name) const = 0;

	/**
	 * @brief Creates an IDynamicReactionModel object of the given @p name
	 * @details The caller owns the returned IDynamicReactionModel object.
	 * @param [in] name Name of the IDynamicReactionModel object
	 * @return Object of the given IDynamicReactionModel @p name or @c nullptr if that name does not exist
	 */
	virtual model::IDynamicReactionModel* createDynamicReactionModel(const std::string& name) const = 0;

	/**
	 * @brief Checks if there is an IDynamicReactionModel of the given @p name
	 * @param [in] name Name of the IDynamicReactionModel object
	 * @return @c true if a dynamic reaction model of this name exists, otherwise @c false
	 */
	virtual bool isValidDynamicReactionModel(const std::string& name) const = 0;

	/**
	 * @brief Creates an IParameterStateDependence object of the given @p name
	 * @details The caller owns the returned IParameterStateDependence object.
	 * @param [in] name Name of the IParameterStateDependence object
	 * @return Object of the given IParameterStateDependence @p name or @c nullptr if that name does not exist
	 */
	virtual model::IParameterStateDependence* createParameterStateDependence(const std::string& name) const = 0;

	/**
	 * @brief Checks if there is an IParameterStateDependence of the given @p name
	 * @param [in] name Name of the IParameterStateDependence object
	 * @return @c true if a dynamic reaction model of this name exists, otherwise @c false
	 */
	virtual bool isValidParameterStateDependence(const std::string& name) const = 0;

	/**
	 * @brief Creates an IParameterParameterDependence object of the given @p name
	 * @details The caller owns the returned IParameterParameterDependence object.
	 * @param [in] name Name of the IParameterParameterDependence object
	 * @return Object of the given IParameterParameterDependence @p name or @c nullptr if that name does not exist
	 */
	virtual model::IParameterParameterDependence* createParameterParameterDependence(const std::string& name) const = 0;

	/**
	 * @brief Checks if there is an IParameterParameterDependence of the given @p name
	 * @param [in] name Name of the IParameterParameterDependence object
	 * @return @c true if a dynamic reaction model of this name exists, otherwise @c false
	 */
	virtual bool isValidParameterParameterDependence(const std::string& name) const = 0;

	/**
	 * @brief Creates an IExternalFunction object of the given @p type
	 * @details The caller owns the returned IExternalFunction object.
	 * @param [in] type Type of the IExternalFunction object
	 * @return Object of the given IExternalFunction @p type or @c nullptr if that type does not exist
	 */
	virtual IExternalFunction* createExternalFunction(const std::string& type) const = 0;
};

} // namespace cadet

#endif  // LIBCADET_CONFIGURATIONHELPER_HPP_
