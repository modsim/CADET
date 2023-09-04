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
 * Defines an inlet profile.
 */

#ifndef LIBCADET_INLETPROFILE_HPP_
#define LIBCADET_INLETPROFILE_HPP_

#include <vector>

#include "cadet/cadetCompilerInfo.hpp"
#include "cadet/LibExportImport.hpp"
#include "cadet/ParameterId.hpp"

namespace cadet
{

class IParameterProvider;

/**
 * @brief Interface for inlet profiles
 */
class CADET_API IInletProfile
{
public:
	virtual ~IInletProfile() CADET_NOEXCEPT
	{
	}

	/**
	 * @brief Configures the inlet profile by extracting all parameters from the given @p paramProvider
	 * @details The scope of the cadet::IParameterProvider is left unchanged on return.
	 *
	 * @param [in] paramProvider Pointer to parameter provider (may be @c nullptr)
	 * @param [in] nComp Number of components
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configure(IParameterProvider* paramProvider, unsigned int nComp) = 0;

	/**
	 * @brief Sets the number of components for which inlet data has to be provided
	 * @param [in] nComp Number of components
	 */
	virtual void numComponents(unsigned int nComp) CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the name of the inlet profile
	 * @return Name of the inlet profile
	 */
	virtual const char* name() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns a vector with all available parameters
	 * @details Intrinsic and global parameters, such as SECTION_TIMES, should be returned.
	 *
	 * @param [in] unitOpIdx Index of the unit operation that owns the inlet profile
	 * @return Vector with all available parameters
	 */
	virtual std::vector<ParameterId> availableParameters(unsigned int unitOpIdx) CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the inlet concentration at a given time for all components
	 *
	 * @param [in]  t         Absolute simulation time
	 * @param [in]  sec       Index of the current time section
	 * @param [out] inletConc Pointer to first element of contiguous array receiving the inlet concentration of all
	 * components
	 */
	virtual void inletConcentration(double t, unsigned int sec, double* inletConc) = 0;

	/**
	 * @brief Returns the derivative of all components with respect to a given parameter
	 * @details The given parameter @p id matches one of the availableParameters() (when unit operation id is ignored).
	 *          In other words, this function is only called for parameters that belong to this IInletProfile.
	 *
	 * @param [in]  t          Absolute simulation time
	 * @param [in]  sec        Index of the current time section
	 * @param [in]  pId        ID of the parameter to be differentiated with respect to
	 * @param [out] paramDeriv Pointer to first element of contiguous array receiving the parameter derivative of all
	 * components
	 */
	virtual void parameterDerivative(double t, unsigned int sec, const ParameterId& pId, double* paramDeriv) = 0;

	/**
	 * @brief Returns the time derivative of all components
	 *
	 * @param [in]  t              Absolute simulation time
	 * @param [in]  sec            Index of the current time section
	 * @param [out] timeDerivative Pointer to first element of contiguous array receiving the time derivative of all
	 * components
	 */
	virtual void timeDerivative(double t, unsigned int sec, double* timeDerivative) = 0;

	/**
	 * @brief Returns the second derivative of all components with respect to a given parameter and time
	 * @details The given parameter @p id matches one of the availableParameters() (when unit operation id is ignored).
	 *          In other words, this function is only called for parameters that belong to this IInletProfile.
	 *
	 * @param [in]  t     Absolute simulation time
	 * @param [in]  sec   Index of the current time section
	 * @param [in]  pId   ID of the parameter to be differentiated with respect to
	 * @param [out] deriv Pointer to first element of contiguous array receiving the derivative of all components
	 */
	virtual void timeParameterDerivative(double t, unsigned int sec, const ParameterId& pId, double* deriv) = 0;

	/**
	 * @brief Returns the value of the given parameter
	 * @details The given parameter @p id matches one of the availableParameters() (when unit operation id is ignored).
	 *
	 * @param [in] id    Parameter ID of the parameter to be returned
	 * @return Value of the queried parameter
	 */
	virtual double getParameterValue(const ParameterId& id) = 0;

	/**
	 * @brief Sets the value of a parameter
	 * @details The given parameter @p id matches one of the availableParameters() (when unit operation id is ignored).
	 *
	 * @param [in] id    Parameter ID of the parameter to be manipulated
	 * @param [in] value Value of the parameter
	 */
	virtual void setParameterValue(const ParameterId& id, double value) = 0;

	/**
	 * @brief Sets the section time vector
	 * @details The integration time is partitioned into sections. All parameters and
	 *          equations are assumed continuous inside one section. Thus, sections
	 *          provide means to implement discontinuous behavior (e.g., pulse injection profiles,
	 *          switching of valves). After initialization, the simulator notifies all entities
	 *          such as models or data sources of its section times.
	 *
	 *          The vector of section times consists of strictly increasing time points
	 *          @f[ t_0 < t_1 < t_2 < \dots t_N @f]
	 *          which mark the beginning and end of a section. The @f$ i@f$-th section is given by
	 *          @f[ \left[ t_i, t_{i+1} \right]. @f]
	 *          If a transition from one section to the next is continuous, the @p secContinuity flag
	 *          for that transition is @c true. In this case, the time integrator will not stop at the
	 *          transition time point and reinitialize consistently (which will be done for discontinuous
	 *          transitions).
	 *
	 * @param [in] secTimes Vector with section time points (length is @p nSections + 1)
	 * @param [in] secContinuity Vector of flags that indicate a continuous (@c true) or discontinuous (@c false)
	 *             transition from the current section to the next one (length is @p nSections - 1). For instance,
	 *             the first element indicates whether the transition from section @c 0 to @c 1 is continuous.
	 * @param [in] nSections Number of sections
	 */
	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity,
								 unsigned int nSections) CADET_NOEXCEPT = 0;
};

} // namespace cadet

#endif // LIBCADET_INLETPROFILE_HPP_
