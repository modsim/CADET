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
 * Defines tests for general unit operations.
 */

#ifndef CADETTEST_UNITOPTESTS_HPP_
#define CADETTEST_UNITOPTESTS_HPP_

namespace cadet
{

class IUnitOperation;
class IModelBuilder;
class JsonParameterProvider;

namespace test
{

namespace unitoperation
{
/**
 * @brief Creates a runnable unit operation model
 * @details Creates a unit operation model and configures it using the given IParameterProvider @p jpp.
 * @param [in] jpp Configuration of the model
 * @param [in] mb ModelBuilder
 * @return Runnable unit operation model
 */
cadet::IUnitOperation* createAndConfigureUnit(cadet::JsonParameterProvider& jpp, cadet::IModelBuilder& mb);

/**
 * @brief Creates a runnable unit operation model
 * @details Creates a unit operation model and configures it using the given IParameterProvider @p jpp.
 * @param [in] uoType Unit operation type
 * @param [in] mb ModelBuilder
 * @param [in] jpp Configuration of the model
 * @return Runnable unit operation model
 */
cadet::IUnitOperation* createAndConfigureUnit(const std::string& uoType, cadet::IModelBuilder& mb,
											  cadet::JsonParameterProvider& jpp);

/**
 * @brief Checks the full analytic Jacobian against AD for a given model
 * @param [in] jpp Unit operation configuration
 */
void testJacobianAD(cadet::JsonParameterProvider& jpp);

/**
 * @brief Checks the (analytic) time derivative Jacobian against FD for a given model
 * @details Uses centered finite differences.
 * @param [in] jpp Unit operation configuration
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testTimeDerivativeJacobianFD(cadet::JsonParameterProvider& jpp, double h, double absTol, double relTol);

/**
 * @brief Checks the consistent initialization of a generic unit operation
 * @param [in] unit Configured unit operation
 * @param [in] adEnabled Determines whether AD is used for calculating the Jacobian
 * @param [in] y Initial state to start consistent initialization from
 * @param [in] consTol Tolerance for the consistent initialization algorithm
 * @param [in] absTol Allowed maximum residual error after consistent initialization
 */
void testConsistentInitialization(IUnitOperation* const unit, bool adEnabled, double* const y, double consTol,
								  double absTol);

/**
 * @brief Checks the consistent initialization of the sensitivies of a generic unit operation
 * @param [in] unit Configured unit operation
 * @param [in] adEnabled Determines whether AD is used for calculating the Jacobian
 * @param [in] y State of original system
 * @param [in] yDot Time derivative of state of original system
 * @param [in] absTol Allowed maximum residual error after consistent initialization
 */
void testConsistentInitializationSensitivity(cadet::IUnitOperation* const unit, bool adEnabled, double const* const y,
											 double const* const yDot, double absTol);

/**
 * @brief Checks whether the inlet DOFs produce the identity matrix in the Jacobian of the unit operation
 * @param [in] unit Configured unit operation
 * @param [in] adEnabled Determines whether AD is used for calculating the Jacobian
 */
void testInletDofJacobian(cadet::IUnitOperation* const unit, bool adEnabled);

} // namespace unitoperation
} // namespace test
} // namespace cadet

#endif // CADETTEST_UNITOPTESTS_HPP_
