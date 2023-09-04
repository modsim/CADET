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
 * Defines an external function as data source.
 */

#ifndef LIBCADET_EXTERNALFUNCTION_HPP_
#define LIBCADET_EXTERNALFUNCTION_HPP_

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"

namespace cadet
{

class IParameterProvider;

/**
 * @brief Interface for external functions
 */
class CADET_API IExternalFunction
{
public:
	virtual ~IExternalFunction() CADET_NOEXCEPT
	{
	}

	/**
	 * @brief Configures the external function by extracting all parameters from the given @p paramProvider
	 * @details The scope of the cadet::IParameterProvider is left unchanged on return.
	 *
	 * @param [in] paramProvider Pointer to parameter provider (may be @c nullptr)
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configure(IParameterProvider* paramProvider) = 0;

	/**
	 * @brief Returns the name of the external function
	 * @return Name of the external function
	 */
	virtual const char* name() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the function value at a given time and spatial position
	 *
	 * @param [in]  t       Absolute simulation time
	 * @param [in]  z       Normalized axial position in the column in [0,1]
	 * @param [in]  rho     Normalized radial position in the column in [0,1]
	 * @param [in]  r       Normalized radial position in the particle in [0,1]
	 * @param [in]  sec     Index of the current time section
	 * @return Function value
	 */
	virtual double externalProfile(double t, double z, double rho, double r, unsigned int sec) = 0;

	/**
	 * @brief Returns the time derivative of the function at a given time and spatial position
	 *
	 * @param [in]  t              Absolute simulation time
	 * @param [in]  z              Normalized axial position in the column in [0,1]
	 * @param [in]  rho            Normalized radial position in the column in [0,1]
	 * @param [in]  r              Normalized radial position in the particle in [0,1]
	 * @param [in]  sec            Index of the current time section
	 * @return Time derivative of the external function
	 */
	virtual double timeDerivative(double t, double z, double rho, double r, unsigned int sec) = 0;

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
	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections) = 0;
};

} // namespace cadet

#endif // LIBCADET_EXTERNALFUNCTION_HPP_
