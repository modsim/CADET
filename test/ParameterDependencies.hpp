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
 * Defines tests for parameter dependencies.
 */

#ifndef CADETTEST_PARAMDEPENDENCIES_HPP_
#define CADETTEST_PARAMDEPENDENCIES_HPP_

/**
 * @brief Puts the given arguments in curly braces
 * @details This macro is required for taking initializer lists in parentheses as macro arguments
 */
#define BRACED_INIT_LIST(...)                                                                                          \
	{                                                                                                                  \
		__VA_ARGS__                                                                                                    \
	}

/**
 * @brief Emits tests for a parameter dependence having a non-binding and an all-binding variant
 * @param modelName Identifier of the model as string (e.g. "LINEAR")
 * @param tagName Tags added to the tests as string (e.g., "[SOMETAG]")
 * @param postFix String appended to the test description / name
 * @param nBound Array with number of bound states in parentheses (e.g., (1, 1, 2))
 * @param state Array with full state vector (liquid and solid phase) in parentheses
 * @param config Interior of a JSON object block with parameters (prefix "PD")
 */
#define CADET_PARAMDEPTEST_IMPL(modelName, tagName, postFix, nBound, state, config)                                    \
	TEST_CASE(modelName " param dep liquid cell analytic Jacobian vs AD" postFix,                                      \
			  "[Jacobian],[AD],[ParameterDependence]," tagName)                                                        \
	{                                                                                                                  \
		const unsigned int nBound2[] = BRACED_INIT_LIST nBound;                                                        \
		const double state2[] = BRACED_INIT_LIST state;                                                                \
		cadet::test::paramdep::testLiquidJacobianAD(modelName, sizeof(nBound2) / sizeof(unsigned int), nBound2,        \
													"{" config "}", state2);                                           \
	}                                                                                                                  \
	TEST_CASE(modelName " param dep combined cell analytic Jacobian vs AD" postFix,                                    \
			  "[Jacobian],[AD],[ParameterDependence]," tagName)                                                        \
	{                                                                                                                  \
		const unsigned int nBound2[] = BRACED_INIT_LIST nBound;                                                        \
		const double state2[] = BRACED_INIT_LIST state;                                                                \
		cadet::test::paramdep::testCombinedJacobianAD(modelName, sizeof(nBound2) / sizeof(unsigned int), nBound2,      \
													  "{" config "}", state2);                                         \
	}

/**
 * @brief Emits tests for a parameter dependence
 * @param modelName Identifier of the model as string (e.g. "LINEAR")
 * @param nBound Array with number of bound states in parentheses (e.g., (1, 1, 2))
 * @param state Array with full state vector (liquid and solid phase) in parentheses
 * @param config Interior of a JSON object block with parameters (prefix "PD")
 */
#define CADET_PARAMDEPTEST(modelName, nBound, state, config)                                                           \
	CADET_PARAMDEPTEST_IMPL(modelName, "[" modelName "]", "", nBound, state, config)

#endif // CADETTEST_PARAMDEPENDENCIES_HPP_
