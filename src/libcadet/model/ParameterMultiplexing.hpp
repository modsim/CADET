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
 * Provides helper functions for parameter multiplexing
 */

#ifndef LIBCADET_PARAMMULTIPLEXING_HPP_
#define LIBCADET_PARAMMULTIPLEXING_HPP_

#include "cadet/ParameterId.hpp"
#include "AutoDiff.hpp"

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace cadet
{

class IParameterProvider;

namespace model
{

enum class MultiplexMode : int
{
	Independent,
	Radial,
	RadialSection,
	Component,
	ComponentRadial,
	ComponentRadialSection,
	ComponentSection,
	Section,
	Axial,
	AxialRadial,
	Type,
	ComponentType,
	ComponentSectionType
};

/**
 * @brief Checks whether the given multiplexing mode is section-dependent
 * @param [in] mode MultiplexMode to check
 * @return @c true if the mode is section-dependent, otherwise @c false
 */
inline bool isSectionDependent(MultiplexMode mode) CADET_NOEXCEPT
{
	return (mode == MultiplexMode::RadialSection) || (mode == MultiplexMode::ComponentRadialSection) ||
		   (mode == MultiplexMode::ComponentSection) || (mode == MultiplexMode::Section) ||
		   (mode == MultiplexMode::ComponentSectionType);
}

/**
 * @brief Reads, multiplexes, and registers a parameter that may depend on particle type
 * @details Reads the fiels NAME from the IParameterProvider. The multiplexing behavior is inferred from
 *          the length of the field NAME.
 *
 *          Depending on the (inferred) mode, the read parameter values are multiplexed to a size of nParType.
 *          In case of multiplexing, only the first instance of the parameter is registered.
 *
 * @param [in] paramProvider ParameterProvider
 * @param [in] parameters Map to register the parameters in
 * @param [out] values Array to store the read parameters in
 * @param [in] name Name of the parameter
 * @param [in] nParType Number of particle types
 * @param [in] uoi Unit operation index
 * @return @c true if the parameter is multiplexed (single value), otherwise @c false
 */
bool readAndRegisterMultiplexTypeParam(IParameterProvider& paramProvider,
									   std::unordered_map<ParameterId, active*>& parameters,
									   std::vector<active>& values, const std::string& name, unsigned int nParType,
									   UnitOpIdx uoi);

/**
 * @brief Sets the value of a multiplexed parameter that may depend on particle type
 * @details Sets the value of a parameter and multiplexes the value onto all parameter instances.
 *          The parameter array is expected to have a size of nParType.
 *
 *          In case of multiplexing, the given value is applied to all parameter instances.
 *
 *          This function optionally checks whether the specified parameter is listed as sensitive. It uses the first
 *          parameter group / instance to check the @p sensParams set.
 *
 * @param [in] pId ParameterID
 * @param [in] nameHash Hash of the parameter name
 * @param [in] mode Multiplexing mode as obtained by readAndRegisterMultiplexTypeParam()
 * @param [in,out] data Array with parameters whose values are updated
 * @param [in] val Value to apply to the parameter(s)
 * @param [in] sensParams If not @c nullptr, the set is checked for the specified parameter.
 *                        If it is not contained in the set, the value is not applied to the parameter.
 * @return @c true if the value has been applied, or @c false otherwise
 */
bool multiplexTypeParameterValue(const ParameterId& pId, StringHash nameHash, bool mode, std::vector<active>& data,
								 double value, std::unordered_set<active*> const* sensParams);

/**
 * @brief Sets AD info of a multiplexed parameter that may depend on particle type
 * @details Sets the AD direction and seed value of a parameter and multiplexes the info onto all parameter instances.
 *          The parameter array is expected to have a size of nParType.
 *
 *          In case of multiplexing, the given AD info is applied to all parameter instances.
 *
 *          The first parameter group / instance is added to the @p sensParams set in order to mark the parameter as
 * sensitive.
 *
 * @param [in] pId ParameterID
 * @param [in] nameHash Hash of the parameter name
 * @param [in] mode Multiplexing mode as obtained by readAndRegisterMultiplexTypeParam()
 * @param [in,out] data Array with parameters whose AD info are updated
 * @param [in] adDirection AD direction
 * @param [in] adValue AD seed value
 * @param [in,out] sensParams The parameter(s) are marked sensitive by adding them to this set
 * @return @c true if the parameter has been found, or @c false otherwise
 */
bool multiplexTypeParameterAD(const ParameterId& pId, StringHash nameHash, bool mode, std::vector<active>& data,
							  unsigned int adDirection, double adValue, std::unordered_set<active*>& sensParams);

/**
 * @brief Reads, multiplexes, and registers a parameter that depends on particle type, component, and (optionally)
 * section
 * @details Reads the fiels NAME and NAME_MULTIPLEX (if available) from the IParameterProvider. If NAME_MULTIPLEX is
 * missing, the multiplexing behavior is inferred from the length of the field NAME.
 *
 *          Depending on the (inferred) mode, the read parameter values are multiplexed to a size of nComp * nParType,
 * or nComp * nParType * nSections. The ordering is section-type-major.
 *
 *          In case of multiplexing, only the first group of parameters is registered. For example, if one value for all
 *          components in each particle type is given (read array of size nParType), the first component of each
 * particle type is registered.
 *
 * @param [in] paramProvider ParameterProvider
 * @param [in] parameters Map to register the parameters in
 * @param [out] values Array to store the read parameters in
 * @param [in] name Name of the parameter
 * @param [in] nParType Number of particle types
 * @param [in] nComp Number of components
 * @param [in] uoi Unit operation index
 * @return Inferred or read multiplexing mode
 */
MultiplexMode readAndRegisterMultiplexCompTypeSecParam(IParameterProvider& paramProvider,
													   std::unordered_map<ParameterId, active*>& parameters,
													   std::vector<active>& values, const std::string& name,
													   unsigned int nParType, unsigned int nComp, UnitOpIdx uoi);

/**
 * @brief Sets the value of a multiplexed parameter that depends on particle type, component, and (optionally) section
 * @details Sets the value of a parameter and multiplexes the value onto all parameter instances.
 *          The parameter array is expected to have a size of nComp * nParType, or nComp * nParType * nSections.
 *          The ordering is section-type-major.
 *
 *          In case of multiplexing, the given value is applied to all parameter instances. For example, if one
 * parameter for all components in each particle type is given, the value is applied to all components of the specified
 * particle type.
 *
 *          This function optionally checks whether the specified parameter is listed as sensitive. It uses the first
 *          parameter group / instance to check the @p sensParams set.
 *
 * @param [in] pId ParameterID
 * @param [in] nameHash Hash of the parameter name
 * @param [in] mode Multiplexing mode as obtained by readAndRegisterMultiplexCompTypeSecParam()
 * @param [in,out] data Array with parameters whose values are updated
 * @param [in] nParType Number of particle types
 * @param [in] nComp Number of components
 * @param [in] val Value to apply to the parameter(s)
 * @param [in] sensParams If not @c nullptr, the set is checked for the specified parameter.
 *                        If it is not contained in the set, the value is not applied to the parameter.
 * @return @c true if the value has been applied, or @c false otherwise
 */
bool multiplexCompTypeSecParameterValue(const ParameterId& pId, StringHash nameHash, MultiplexMode mode,
										std::vector<active>& data, unsigned int nParType, unsigned int nComp,
										double value, std::unordered_set<active*> const* sensParams);

/**
 * @brief Sets AD info of a multiplexed parameter that depends on particle type, component, and (optionally) section
 * @details Sets the AD direction and seed value of a parameter and multiplexes the info onto all parameter instances.
 *          The parameter array is expected to have a size of nComp * nParType, or nComp * nParType * nSections.
 *          The ordering is section-type-major.
 *
 *          In case of multiplexing, the given AD info is applied to all parameter instances. For example, if one
 * parameter for all components in each particle type is given, the AD info is applied to all components of the
 * specified particle type.
 *
 *          The first parameter group / instance is added to the @p sensParams set in order to mark the parameter as
 * sensitive.
 *
 * @param [in] pId ParameterID
 * @param [in] nameHash Hash of the parameter name
 * @param [in] mode Multiplexing mode as obtained by readAndRegisterMultiplexCompTypeSecParam()
 * @param [in,out] data Array with parameters whose AD info are updated
 * @param [in] nParType Number of particle types
 * @param [in] nComp Number of components
 * @param [in] adDirection AD direction
 * @param [in] adValue AD seed value
 * @param [in,out] sensParams The parameter(s) are marked sensitive by adding them to this set
 * @return @c true if the parameter has been found, or @c false otherwise
 */
bool multiplexCompTypeSecParameterAD(const ParameterId& pId, StringHash nameHash, MultiplexMode mode,
									 std::vector<active>& data, unsigned int nParType, unsigned int nComp,
									 unsigned int adDirection, double adValue, std::unordered_set<active*>& sensParams);

/**
 * @brief Reads, multiplexes, and registers a parameter that depends on particle type, component, bound state, and
 * (optionally) section
 * @details Reads the fiels NAME and NAME_MULTIPLEX (if available) from the IParameterProvider. If NAME_MULTIPLEX is
 * missing, the multiplexing behavior is inferred from the length of the field NAME.
 *
 *          Depending on the (inferred) mode, the read parameter values are multiplexed to a size of nTotalBound, or
 *          nTotalBound * nSections. The ordering is section-type-component-major.
 *
 *          In case of multiplexing, only the first group of parameters is registered. See
 * readAndRegisterMultiplexCompTypeSecParam().
 *
 *          Components and bound states are treated together, that is, a parameter cannot be set independent of bound
 * state but dependent on component (and vice versa). If the parameter is particle type independent, the same number of
 *          bound states per component is expected in all particle types (i.e., same binding model in all types).
 *
 * @param [in] paramProvider ParameterProvider
 * @param [in] parameters Map to register the parameters in
 * @param [out] values Array to store the read parameters in
 * @param [in] name Name of the parameter
 * @param [in] nParType Number of particle types
 * @param [in] nComp Number of components
 * @param [in] strideBound Array with number of bound states per particle type (additional last element is total number
 * of bound states)
 * @param [in] nBound Array with number of bound states per component and particle type in type-major ordering
 * @param [in] uoi Unit operation index
 * @return Inferred or read multiplexing mode
 */
MultiplexMode readAndRegisterMultiplexBndCompTypeSecParam(IParameterProvider& paramProvider,
														  std::unordered_map<ParameterId, active*>& parameters,
														  std::vector<active>& values, const std::string& name,
														  unsigned int nParType, unsigned int nComp,
														  unsigned int const* strideBound, unsigned int const* nBound,
														  UnitOpIdx uoi);

/**
 * @brief Sets the value of a multiplexed parameter that depends on particle type, component, bound state, and
 * (optionally) section
 * @details Sets the value of a parameter and multiplexes the value onto all parameter instances.
 *          The parameter array is expected to have a size of nTotalBound, or nTotalBound * nSections.
 *          The ordering is section-type-component-major.
 *
 *          In case of multiplexing, the given value is applied to all parameter instances. See
 * multiplexCompTypeSecParameterValue().
 *
 *          This function optionally checks whether the specified parameter is listed as sensitive. It uses the first
 *          parameter group / instance to check the @p sensParams set.
 *
 * @param [in] pId ParameterID
 * @param [in] nameHash Hash of the parameter name
 * @param [in] mode Multiplexing mode as obtained by readAndRegisterMultiplexCompTypeSecParam()
 * @param [in,out] data Array with parameters whose values are updated
 * @param [in] nParType Number of particle types
 * @param [in] nComp Number of components
 * @param [in] strideBound Array with number of bound states per particle type (additional last element is total number
 * of bound states)
 * @param [in] nBound Array with number of bound states per component and particle type in type-major ordering
 * @param [in] boundOffset Array with offset to component in bound-phase (cumulative sum of nBound per particle type)
 * per particle type in type-major ordering
 * @param [in] val Value to apply to the parameter(s)
 * @param [in] sensParams If not @c nullptr, the set is checked for the specified parameter.
 *                        If it is not contained in the set, the value is not applied to the parameter.
 * @return @c true if the value has been applied, or @c false otherwise
 */
bool multiplexBndCompTypeSecParameterValue(const ParameterId& pId, StringHash nameHash, MultiplexMode mode,
										   std::vector<active>& data, unsigned int nParType, unsigned int nComp,
										   unsigned int const* strideBound, unsigned int const* nBound,
										   unsigned int const* boundOffset, double value,
										   std::unordered_set<active*> const* sensParams);

/**
 * @brief Sets AD info of a multiplexed parameter that depends on particle type, component, bound state, and
 * (optionally) section
 * @details Sets the AD direction and seed value of a parameter and multiplexes the info onto all parameter instances.
 *          The parameter array is expected to have a size of nTotalBound, or nTotalBound * nSections.
 *          The ordering is section-type-component-major.
 *
 *          In case of multiplexing, the given AD info is applied to all parameter instances. See
 * multiplexCompTypeSecParameterAD().
 *
 *          The first parameter group / instance is added to the @p sensParams set in order to mark the parameter as
 * sensitive.
 *
 * @param [in] pId ParameterID
 * @param [in] nameHash Hash of the parameter name
 * @param [in] mode Multiplexing mode as obtained by readAndRegisterMultiplexCompTypeSecParam()
 * @param [in,out] data Array with parameters whose AD info are updated
 * @param [in] nParType Number of particle types
 * @param [in] nComp Number of components
 * @param [in] strideBound Array with number of bound states per particle type (additional last element is total number
 * of bound states)
 * @param [in] nBound Array with number of bound states per component and particle type in type-major ordering
 * @param [in] boundOffset Array with offset to component in bound-phase (cumulative sum of nBound per particle type)
 * per particle type in type-major ordering
 * @param [in] adDirection AD direction
 * @param [in] adValue AD seed value
 * @param [in,out] sensParams The parameter(s) are marked sensitive by adding them to this set
 * @return @c true if the parameter has been found, or @c false otherwise
 */
bool multiplexBndCompTypeSecParameterAD(const ParameterId& pId, StringHash nameHash, MultiplexMode mode,
										std::vector<active>& data, unsigned int nParType, unsigned int nComp,
										unsigned int const* strideBound, unsigned int const* nBound,
										unsigned int const* boundOffset, unsigned int adDirection, double adValue,
										std::unordered_set<active*>& sensParams);

} // namespace model

} // namespace cadet

#endif // LIBCADET_PARAMMULTIPLEXING_HPP_
