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
 * Signal handling for cadet-cli.
 */

#ifndef CADETCLI_SIGNALHANDLER_HPP_
#define CADETCLI_SIGNALHANDLER_HPP_

namespace cadet
{
/**
 * @brief  Installs the signal handler in the OS
 * @return @c true if the signal handler was installed successfully, otherwise @c false
 */
bool installSignalHandler();

/**
 * @brief   Checks whether the user has requested the program to stop
 * @details The user can request stopping the execution by pressing CTRL+C or sending SIGINT.
 * @return  @c true if the user request stopping the program, otherwise @c false
 */
bool stopExecutionRequested();

} // namespace cadet

#endif // CADETCLI_SIGNALHANDLER_HPP_
