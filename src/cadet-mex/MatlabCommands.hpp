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
 * Provides commands that can be executed on a cadet::Driver object.
 */

#ifndef CADET_MEX_MATLABCOMMANDS_HPP_
#define CADET_MEX_MATLABCOMMANDS_HPP_

#include <unordered_map>
#include <mex.h>

namespace cadet
{

class Driver;

namespace mex
{

class MatlabReaderWriter;

/**
 * @brief Runs a full CADET simulation cycle
 * @details Builds and configures a new model, runs it, and writes the results back.
 * @param [in] drv Driver
 * @param [in] input Matlab variable to read from
 * @param [out] output Matlab variable to write to
 */
void runFullSimulation(cadet::Driver& drv, mxArray const*& input, mxArray*& output);

/**
 * @brief Command map type
 */
typedef std::unordered_map<std::string, void (*)(cadet::Driver&, int, mxArray**, int, const mxArray**)> CommandMap;

/**
 * @brief Returns a map with all available commands
 * @details A command is identified by a string (ID) and is given by function pointer.
 * @return Map with all available commands
 */
CommandMap registeredCommands();

} // namespace mex

} // namespace cadet

#endif // CADET_MEX_MATLABCOMMANDS_HPP_
