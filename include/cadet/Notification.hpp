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
 * Defines interfaces for (progress) notification.
 */

#ifndef LIBCADET_NOTIFICATION_HPP_
#define LIBCADET_NOTIFICATION_HPP_

#include "cadet/cadetCompilerInfo.hpp"

namespace cadet
{

/**
 * @brief Defines callback functions that are called from an ISimulator
 */
class CADET_API INotificationCallback
{
public:
	virtual ~INotificationCallback() CADET_NOEXCEPT
	{
	}

	/**
	 * @brief Called when time integration starts
	 * @details This function is called before any consistent initialization has taken place.
	 */
	virtual void timeIntegrationStart() = 0;

	/**
	 * @brief Called when time integration has finished successfully
	 */
	virtual void timeIntegrationEnd() = 0;

	/**
	 * @brief Called when time integration has stopped because of an error
	 * @details This function is called before an IntegrationException is raised.
	 * @param[in]  message   Error message
	 * @param[in]  section   Index of the current time section
	 * @param[in]  time      Current process time
	 * @param[in]  progress  Progress in percent (between @c 0.0 and @c 1.0)
	 */
	virtual void timeIntegrationError(char const* message, unsigned int section, double time, double progress) = 0;

	/**
	 * @brief Called when the time integrator enters a new time section
	 * @details This function is called after consistent initialization has been performed.
	 *
	 * @param[in]  section   Index of the current time section
	 * @param[in]  time      Current process time
	 * @param[in]  state     Current state vector
	 * @param[in]  stateDot  Current time derivative of the state vector
	 * @param[in]  progress  Progress in percent (between @c 0.0 and @c 1.0)
	 * @return @c true if time integrator should continue, otherwise @c false
	 */
	virtual bool timeIntegrationSection(unsigned int section, double time, double const* state, double const* stateDot,
										double progress) = 0;

	/**
	 * @brief Called when the time integrator has finished a single time step
	 *
	 * @param[in]  section   Index of the current time section
	 * @param[in]  time      Current process time
	 * @param[in]  state     Current state vector
	 * @param[in]  stateDot  Current time derivative of the state vector
	 * @param[in]  progress  Progress in percent (between @c 0.0 and @c 1.0)
	 * @return @c true if time integrator should continue, otherwise @c false
	 */
	virtual bool timeIntegrationStep(unsigned int section, double time, double const* state, double const* stateDot,
									 double progress) = 0;
};

} // namespace cadet

#endif // LIBCADET_NOTIFICATION_HPP_
