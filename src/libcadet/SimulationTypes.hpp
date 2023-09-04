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
 * Defines useful types for the simulation.
 */

#ifndef LIBCADET_SIMULATIONTYPES_HPP_
#define LIBCADET_SIMULATIONTYPES_HPP_

#include "AutoDiff.hpp"

namespace cadet
{

/**
 * @brief Position inside a column unit operation
 */
struct ColumnPosition
{
	double axial;    //!< Axial bulk coordinate z
	double radial;   //!< Radial bulk coordinate rho
	double particle; //!< Radial particle coordinate r
};

/**
 * @brief Common parameters for Jacobians via AD
 */
struct AdJacobianParams
{
	active* adRes;            //!< Residual vector
	active* adY;              //!< State vector
	unsigned int adDirOffset; //!< Number of reserved AD directions (e.g., for sensitivities)
};

/**
 * @brief Simulation time point
 */
struct SimulationTime
{
	double t;            //!< Time of the simulation
	unsigned int secIdx; //!< Index of the current section
};

/**
 * @brief State vectors of the simulation that can be updated
 */
struct SimulationState
{
	double* vecStateY;    //!< State vector
	double* vecStateYdot; //!< Time derivative of state vector
};

/**
 * @brief State vectors of the simulation that cannot be changed
 */
struct ConstSimulationState
{
	double const* vecStateY;    //!< State vector
	double const* vecStateYdot; //!< Time derivative of state vector
};

/**
 * @brief Converts a mutable simulation state to a constant one
 * @param [in] simState Mutable simulation state
 * @return Constant simulation state
 */
inline ConstSimulationState toConst(const SimulationState& simState)
{
	return ConstSimulationState{simState.vecStateY, simState.vecStateYdot};
}

/**
 * @brief Type tag for including parameter sensitivities
 */
struct WithParamSensitivity
{
};

/**
 * @brief Type tag for excluding parameter sensitivities
 */
struct WithoutParamSensitivity
{
};

/**
 * @brief Type used for choosing the parameter sensitivity tag type
 * @details Chooses WithoutParamSensitivity if @p T is @c double and
 *          WithParamSensitivity if @p T is @c active.
 * @tparam T Parameter data type
 */
template <typename T> struct ParamSens
{
};

template <> struct ParamSens<double>
{
	typedef WithoutParamSensitivity enabled;
};

template <> struct ParamSens<active>
{
	typedef WithParamSensitivity enabled;
};

} // namespace cadet

#endif // LIBCADET_SIMULATIONTYPES_HPP_
