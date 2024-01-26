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
 * Defines simulation tests for column unit operations.
 */

#ifndef CADETTEST_COLUMNSIMTEST_HPP_
#define CADETTEST_COLUMNSIMTEST_HPP_

#include <limits>
#include <string>

namespace cadet
{

class JsonParameterProvider;
class IUnitOperation;

namespace test
{

namespace column
{

/**
 * @brief Sets the number of axial cells in a configuration of a column-like unit operation
 * @details Overwrites the NCOL field in the discretization group of the given ParameterProvider.
 * @param [in,out] jpp ParameterProvider to change the number of axial cells in
 * @param [in] nCol Number of axial cells
 */
void setNumAxialCells(cadet::JsonParameterProvider& jpp, unsigned int nCol);

/**
 * @brief Sets the WENO order in a configuration of a column-like unit operation
 * @details Overwrites the WENO_ORDER field in the weno group of the given ParameterProvider.
 * @param [in,out] jpp ParameterProvider to change the WENO order in
 * @param [in] order Target order
 */
void setWenoOrder(cadet::JsonParameterProvider& jpp, int order);

/**
 * @brief Reverses the flow of a column-like unit operation
 * @param [in,out] jpp ParameterProvider to change the flow direction in
 */
void reverseFlow(cadet::JsonParameterProvider& jpp);

/**
 * @brief Infers cross section area of column model from interstitial velocity
 * @details Uses interstitial velocity and porosity to calculate cross section area.
 *          To this end, a volumetric flow rate of 1.0 m^3/s is assumed. Depending on
 *          @p dir, the velocity field is removed or set to @c +1.0 or @c -1.0 indicating
 *          direction of the flow inside the unit operation.
 *
 * @param [in,out] jpp ParameterProvider to add cross section area to
 * @param [in] useTotalPorosity Determines whether TOTAL_POROSITY is used (@c true) or COL_POROSITY (@c false)
 * @param [in] dir Flow direction in unit operation (@c 0 removes field, @c 1 standard direction, @c -1 flow reversal)
 */
void setCrossSectionArea(cadet::JsonParameterProvider& jpp, bool useTotalPorosity, int dir);

/**
 * @brief Returns the offset to the flux part in the local state vector
 * @param [in] unit Unit operation
 * @return Offset to the flux part
 */
unsigned int fluxOffsetOfColumnUnitOp(cadet::IUnitOperation* unit);

/**
 * @brief Runs a simulation test comparing against (semi-)analytic single component pulse injection reference data
 * @details Linear binding model is used in the column-like unit operation.
 * @param [in] uoType Unit operation type
 * @param [in] refFileRelPath Path to the reference data file from the directory of this file
 * @param [in] forwardFlow Determines whether the unit operates in forward flow (@c true) or backwards flow (@c false)
 * @param [in] dynamicBinding Determines whether dynamic binding (@c true) or rapid equilibrium (@c false) is used
 * @param [in] nCol Number of axial cells
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testAnalyticBenchmark(const char* uoType, const char* refFileRelPath, bool forwardFlow, bool dynamicBinding,
						   unsigned int nCol, double absTol, double relTol);

/**
 * @brief Runs a simulation test comparing against (semi-)analytic single component pulse injection reference data
 * @details The component is assumed to be non-binding.
 * @param [in] uoType Unit operation type
 * @param [in] refFileRelPath Path to the reference data file from the directory of this file
 * @param [in] forwardFlow Determines whether the unit operates in forward flow (@c true) or backwards flow (@c false)
 * @param [in] nCol Number of axial cells
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testAnalyticNonBindingBenchmark(const char* uoType, const char* refFileRelPath, bool forwardFlow,
									 unsigned int nCol, double absTol, double relTol);

/**
 * @brief Runs a simulation test comparing forward and backwards flow in the load-wash-elution example
 * @param [in] uoType Unit operation type
 * @param [in] wenoOrder Order of the WENO method
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testWenoForwardBackward(const char* uoType, int wenoOrder, double absTol, double relTol);

/**
 * @brief Checks the full Jacobian against AD and FD pattern switching
 * @details Checks the analytic Jacobian against the AD Jacobian and checks both against the FD pattern.
 * @param [in] jpp Configured column model
 */
void testJacobianAD(cadet::JsonParameterProvider& jpp);

/**
 * @brief Checks the full Jacobian against AD and FD pattern switching in case of variable surface diffusion coefficient
 * @details Checks the analytic Jacobian against the AD Jacobian and checks both against the FD pattern.
 * @param [in] uoType Unit operation type
 * @param [in] dynamicBinding Determines whether dynamic binding is used
 */
void testJacobianADVariableParSurfDiff(const std::string& uoType, bool dynamicBinding);

/**
 * @brief Checks the full Jacobian against AD and FD pattern switching from forward to backward flow and back
 * @details Checks the analytic Jacobian against the AD Jacobian and checks both against the FD pattern.
 *          Checks both forward and backward flow mode as well as switching between them.
 *
 * @param [in] uoType Unit operation type
 * @param [in] wenoOrder Order of the WENO method
 */
void testJacobianWenoForwardBackward(const std::string& uoType, int wenoOrder);

/**
 * @brief Checks the full Jacobian against FD switching from forward to backward flow and back
 * @details Checks the analytic Jacobian against the finite difference Jacobian.
 *          Checks both forward and backward flow mode as well as switching between them.
 *          Uses centered finite differences.
 *
 * @param [in] uoType Unit operation type
 * @param [in] wenoOrder Order of the WENO method
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testJacobianWenoForwardBackwardFD(const std::string& uoType, int wenoOrder, double h = 1e-6, double absTol = 0.0,
									   double relTol = std::numeric_limits<float>::epsilon() * 100.0);

/**
 * @brief Checks the (analytic) time derivative Jacobian against FD
 * @details Uses centered finite differences.
 * @param [in] uoType Unit operation type
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testTimeDerivativeJacobianFD(const std::string& uoType, double h = 1e-6, double absTol = 0.0,
								  double relTol = std::numeric_limits<float>::epsilon() * 100.0);

/**
 * @brief Checks the bottom macro row and right macro column of the Jacobian against FD
 * @details Uses centered finite differences to check the flux part of the Jacobian.
 * @param [in] uoType Unit operation type
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testArrowHeadJacobianFD(const std::string& uoType, double h = 1e-6, double absTol = 0.0,
							 double relTol = std::numeric_limits<float>::epsilon() * 100.0);

/**
 * @brief Checks the bottom macro row and right macro column of the Jacobian against FD
 * @details Uses centered finite differences to check the flux part of the Jacobian.
 * @param [in] uoType Unit operation type
 * @param [in] dynamicBinding Determines whether dynamic binding is used
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testArrowHeadJacobianFD(const std::string& uoType, bool dynamicBinding, double h = 1e-6, double absTol = 0.0,
							 double relTol = std::numeric_limits<float>::epsilon() * 100.0);

/**
 * @brief Checks the bottom macro row and right macro column of the Jacobian against FD
 * @details Uses centered finite differences to check the flux part of the Jacobian.
 * @param [in] jpp Configured column model
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testArrowHeadJacobianFD(cadet::JsonParameterProvider& jpp, double h = 1e-6, double absTol = 0.0,
							 double relTol = std::numeric_limits<float>::epsilon() * 100.0);

/**
 * @brief Checks the bottom macro row and right macro column of the Jacobian against FD in case of variable surface
 * diffusion coefficient
 * @details Uses centered finite differences to check the flux part of the Jacobian.
 * @param [in] uoType Unit operation type
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testArrowHeadJacobianFDVariableParSurfDiff(const std::string& uoType, double h = 1e-6, double absTol = 0.0,
												double relTol = std::numeric_limits<float>::epsilon() * 100.0);

/**
 * @brief Checks the forward sensitivity residual using analytic Jacobians
 * @details Uses centered finite differences.
 * @param [in] uoType Unit operation type
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testFwdSensJacobians(const std::string& uoType, double h = 1e-6, double absTol = 0.0,
						  double relTol = std::numeric_limits<float>::epsilon() * 100.0);

/**
 * @brief Checks the forward sensitivity solution against finite differences
 * @details Assumes column-like unit models and uses centered finite differences. Checks 4 parameters in
 *          the standard load-wash-elution test case:
 *          COL_DISPERSION, CONST_COEFF (salt, loading), SMA_KA (first protein), CONNECTION (volumetric flow rate).
 *          Each sensitivity is checked for quasi-stationary and dynamic binding, which means that
 *          in total 8 checks are performed.
 * @param [in] uoType Unit operation type
 * @param [in] disableSensErrorTest Determines whether sensitivities take part in local error test
 * @param [in] fdStepSize Array with step sizes of centered finite differences
 * @param [in] absTols Array with absolute error tolerances
 * @param [in] relTols Array with relative error tolerances
 * @param [in] passRates Array with rates of relative error test passes
 */
void testFwdSensSolutionFD(const std::string& uoType, bool disableSensErrorTest, double const* fdStepSize,
						   double const* absTols, double const* relTols, double const* passRates);

/**
 * @brief Checks the forward sensitivity solution with forward flow against the one using backward flow
 * @details Assumes column-like unit models and checks 4 parameters in the standard load-wash-elution test case:
 *          COL_DISPERSION, CONST_COEFF (salt, loading), SMA_KA (first protein), CONNECTION (volumetric flow rate).
 *          Each sensitivity is checked for quasi-stationary and dynamic binding, which means that
 *          in total 8 checks are performed.
 * @param [in] uoType Unit operation type
 * @param [in] absTols Array with absolute error tolerances
 * @param [in] relTols Array with relative error tolerances
 * @param [in] passRates Array with rates of relative error test passes
 */
void testFwdSensSolutionForwardBackward(const std::string& uoType, double const* absTols, double const* relTols,
										double const* passRates);

/**
 * @brief Checks consistent initialization using a model with linear binding
 * @details Assumes column-like unit models and checks the residual of the model equations after
 *          consistent initialization. A linear binding model is applied. Both binding modes and
 *          both AD and analytic Jacobians are checked.
 * @param [in] uoType Unit operation type
 * @param [in] consTol Error tolerance for consistent initialization solver
 * @param [in] absTol Error tolerance for checking whether algebraic residual is 0
 */
void testConsistentInitializationLinearBinding(const std::string& uoType, double consTol, double absTol);

/**
 * @brief Checks consistent initialization using a model with SMA binding
 * @details Assumes column-like unit models and checks the residual of the model equations after
 *          consistent initialization. Both binding modes and both AD and analytic Jacobians are checked.
 * @param [in] uoType Unit operation type
 * @param [in] initState Initial state vector to start process from
 * @param [in] consTol Error tolerance for consistent initialization solver
 * @param [in] absTol Error tolerance for checking whether algebraic residual is 0
 */
void testConsistentInitializationSMABinding(const std::string& uoType, double const* const initState, double consTol,
											double absTol);

/**
 * @brief Checks consistent initialization of sensitivities in a column-like model
 * @details Assumes column-like unit models and checks the residual of the sensitivity equations after
 *          consistent initialization. Both binding modes and both AD and analytic Jacobians are checked.
 * @param [in] uoType Unit operation type
 * @param [in] y State vector of original system
 * @param [in] yDot Time derivative of state vector of original system
 * @param [in] linearBinding Determines whether linear binding or SMA binding model is used
 * @param [in] absTol Error tolerance for checking whether sensitivity residual is 0
 */
void testConsistentInitializationSensitivity(const std::string& uoType, double const* const y, double const* const yDot,
											 bool linearBinding, double absTol);

/**
 * @brief Checks whether the inlet DOFs produce the identity matrix in the Jacobian of the unit operation
 * @details Assumes column-like unit models. Both AD and analytic Jacobians are checked.
 * @param [in] uoType Unit operation type
 */
void testInletDofJacobian(const std::string& uoType);

} // namespace column
} // namespace test
} // namespace cadet

#endif // CADETTEST_COLUMNSIMTEST_HPP_
