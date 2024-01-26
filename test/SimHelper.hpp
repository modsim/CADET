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
 * Defines helper functions for building simulations.
 */

#ifndef CADETTEST_SIMHELPER_HPP_
#define CADETTEST_SIMHELPER_HPP_

#include "JsonTestModels.hpp"
#include "cadet/ParameterId.hpp"

#include <vector>

namespace cadet
{

namespace test
{

/**
 * @brief Sets the number of components for a unit operation
 * @details Overwrites the NCOMP field of the given ParameterProvider.
 * @param [in,out] jpp ParameterProvider
 * @param [in] unit Index of unit operation
 * @param [in] nComp Number of components
 */
void setNumberOfComponents(cadet::JsonParameterProvider& jpp, UnitOpIdx unit, int nComp);

/**
 * @brief Sets the initial conditions of a CSTR
 * @details Overwrites the INIT_C, INIT_Q, and INIT_VOLUME fields of the given ParameterProvider.
 * @param [in,out] jpp ParameterProvider
 * @param [in] c Initial liquid phase concentration
 * @param [in] q Initial bound phase concentration
 * @param [in] v Initial volume
 */
void setInitialConditions(cadet::JsonParameterProvider& jpp, const std::vector<double>& c, const std::vector<double>& q,
						  double v);

/**
 * @brief Sets the initial conditions of a column
 * @details Overwrites the INIT_C, INIT_CP, and INIT_Q fields of the given ParameterProvider.
 * @param [in,out] jpp ParameterProvider
 * @param [in] c Initial liquid phase concentration
 * @param [in] cp Initial bead liquid phase concentration
 * @param [in] q Initial bound phase concentration
 */
void setInitialConditions(cadet::JsonParameterProvider& jpp, const std::vector<double>& c,
						  const std::vector<double>& cp, const std::vector<double>& q);

/**
 * @brief Sets all flow rates of a section
 * @details Sets inflow and outflow as well as filter flow rate.
 * @param [in,out] jpp ParameterProvider
 * @param [in] secIdx Section index
 * @param [in] in Inflow rate
 * @param [in] out Outflow rate
 * @param [in] filter Filter flow rate
 */
void setFlowRates(cadet::JsonParameterProvider& jpp, unsigned int secIdx, double in, double out, double filter);

/**
 * @brief Sets the filter flow rate of a section
 * @details Sets filter flow rate.
 * @param [in,out] jpp ParameterProvider
 * @param [in] secIdx Section index
 * @param [in] filter Filter flow rate
 */
void setFlowRates(cadet::JsonParameterProvider& jpp, unsigned int secIdx, double filter);

/**
 * @brief Sets the inlet profile of a section and component
 * @details Overwrites the spline piece of the given section and component.
 * @param [in,out] jpp ParameterProvider
 * @param [in] secIdx Section index
 * @param [in] comp Component index
 * @param [in] con Constant polynomial coefficient
 * @param [in] lin Linear polynomial coefficient
 * @param [in] quad Quadratic polynomial coefficient
 * @param [in] cub Cubic polynomial coefficient
 */
void setInletProfile(cadet::JsonParameterProvider& jpp, unsigned int secIdx, unsigned int comp, double con, double lin,
					 double quad, double cub);

/**
 * @brief Sets the section times
 * @details Overwrites the SECTION_TIMES field of the given ParameterProvider.
 * @param [in,out] jpp ParameterProvider
 * @param [in] secTimes Section times vector
 */
void setSectionTimes(cadet::JsonParameterProvider& jpp, const std::vector<double>& secTimes);

/**
 * @brief Adds bound states to a CSTR model
 * @param [in,out] jpp ParameterProvider
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] porosity Porosity
 */
void addBoundStates(cadet::JsonParameterProvider& jpp, const std::vector<int>& nBound, double porosity);

/**
 * @brief Adds dummy binding model to a CSTR model
 * @param [in,out] jpp ParameterProvider
 */
void addDummyBindingModel(cadet::JsonParameterProvider& jpp);

/**
 * @brief Adds linear binding model to a CSTR model
 * @param [in,out] jpp ParameterProvider
 * @param [in] kinetic Determines whether kinetic or quasi-stationary binding is used
 * @param [in] kA Vector with kA rates
 * @param [in] kD Vector with kD rates
 */
void addLinearBindingModel(cadet::JsonParameterProvider& jpp, bool kinetic, const std::vector<double>& kA,
						   const std::vector<double>& kD);

/**
 * @brief Adds SMA binding model to a CSTR model
 * @param [in,out] jpp ParameterProvider
 * @param [in] kinetic Determines whether kinetic or quasi-stationary binding is used
 * @param [in] lambda Ionic capacity Lambda
 * @param [in] kA Vector with kA rates
 * @param [in] kD Vector with kD rates
 * @param [in] nu Vector with characteristic charges nu
 * @param [in] sigma Vector with steric factors sigma
 */
void addSMABindingModel(cadet::JsonParameterProvider& jpp, bool kinetic, double lambda, const std::vector<double>& kA,
						const std::vector<double>& kD, const std::vector<double>& nu, const std::vector<double>& sigma);

/**
 * @brief Adds Langmuir binding model to a CSTR model
 * @param [in,out] jpp ParameterProvider
 * @param [in] kinetic Determines whether kinetic or quasi-stationary binding is used
 * @param [in] kA Vector with kA rates
 * @param [in] kD Vector with kD rates
 * @param [in] qMax Vector with maximum capacities qMax
 */
void addLangmuirBindingModel(cadet::JsonParameterProvider& jpp, bool kinetic, const std::vector<double>& kA,
							 const std::vector<double>& kD, const std::vector<double>& qMax);

/**
 * @brief Sets the binding mode of a binding model
 * @details Changes IS_KINETIC field in the adsorption group
 * @param [in,out] jpp ParameterProvider
 * @param [in] isKinetic Determines whether kinetic or quasi-stationary binding is used
 */
void setBindingMode(cadet::JsonParameterProvider& jpp, bool isKinetic);

/**
 * @brief Adds a parameter sensitivity
 * @param [in,out] jpp ParameterProvider
 * @param [in] name Parameter name
 * @param [in] id Parameter ID
 * @param [in] absTol Absolute tolerance
 */
void addSensitivity(cadet::JsonParameterProvider& jpp, const std::string& name, const ParameterId& id, double absTol);

/**
 * @brief Adds return info for parameter sensitivities to parameter provider
 * @param [in,out] jpp ParameterProvider
 * @param [in] unit Index of unit operation
 * @param [in] inlet Determines whether sensitivity at inlet is returned (@c true) or not (@c false)
 */
void returnSensitivities(cadet::JsonParameterProvider& jpp, UnitOpIdx unit, bool inlet = false);

/**
 * @brief Disables error test of sensitivities
 * @details Sensitivities are included in the local error test by default.
 * @param [in,out] jpp ParameterProvider
 * @param [in] isDisabled Determines whether the error test for sensitivities is disabled (@c true) or not (@c false)
 */
void disableSensitivityErrorTest(cadet::JsonParameterProvider& jpp, bool isDisabled = true);

/**
 * @brief Sets maximum step size in time integration
 * @details Default is @c 0 which means unlimited step size.
 * @param [in,out] jpp ParameterProvider
 * @param [in] maxStepSize Maximum time step size
 */
void setMaxStepSize(cadet::JsonParameterProvider& jpp, double maxStepSize);

} // namespace test
} // namespace cadet

#endif // CADETTEST_SIMHELPER_HPP_
