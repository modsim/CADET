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
 * Defines tests for binding models.
 */

#ifndef CADETTEST_BINDINGMODELS_HPP_
#define CADETTEST_BINDINGMODELS_HPP_

/**
 * @brief Puts the given arguments in curly braces
 * @details This macro is required for taking initializer lists in parentheses as macro arguments
 */
#define BRACED_INIT_LIST(...)                                                                                          \
	{                                                                                                                  \
		__VA_ARGS__                                                                                                    \
	}

/**
 * @brief Emits code for a test that checks whether Jacobian columns belonging to non-binding liquid phase components
 * are all zero
 * @details The correct macro, CADET_BINDINGTEST_SINGLE_IMPL_NONBNDJACCONST_false or
 * CADET_BINDINGTEST_SINGLE_IMPL_NONBNDJACCONST_true, is selected by concatentation with a variable:
 * CADET_BINDINGTEST_SINGLE_IMPL_NONBNDJACCONST_##SomeVar(...)
 */
#define CADET_BINDINGTEST_SINGLE_IMPL_NONBNDJACCONST_false(modelName, tagName, postFix, someNonBinding, stateSomeNon,  \
														   configSomeNon)

#define CADET_BINDINGTEST_SINGLE_IMPL_NONBNDJACCONST_true(modelName, tagName, postFix, someNonBinding, stateSomeNon,   \
														  configSomeNon)                                               \
	TEST_CASE(modelName " binding model consistency of non-binding Jacobian columns" postFix,                          \
			  "[BindingModel],[Jacobian]," tagName)                                                                    \
	{                                                                                                                  \
		const unsigned int nBound3[] = BRACED_INIT_LIST someNonBinding;                                                \
		const double state3[] = BRACED_INIT_LIST stateSomeNon;                                                         \
		for (int bindMode = 0; bindMode < 2; ++bindMode)                                                               \
		{                                                                                                              \
			const bool isKinetic = bindMode;                                                                           \
			SECTION(std::string("Binding mode ") + (isKinetic ? "dynamic" : "quasi-stationary"))                       \
			{                                                                                                          \
				for (int adMode = 0; adMode < 2; ++adMode)                                                             \
				{                                                                                                      \
					const bool useAD = adMode;                                                                         \
					SECTION(std::string("Use AD ") + (useAD ? "yes" : "no"))                                           \
					{                                                                                                  \
						cadet::test::binding::testNonBindingConsistency(                                               \
							modelName, sizeof(nBound3) / sizeof(unsigned int), nBound3, isKinetic,                     \
							"{" configSomeNon "}", useAD, state3);                                                     \
					}                                                                                                  \
				}                                                                                                      \
			}                                                                                                          \
		}                                                                                                              \
	}

/**
 * @brief Used to make the macro argument more verbose and for increasing readability
 */
#define CADET_NONBINDING_LIQUIDPHASE_COMP_UNUSED true
#define CADET_NONBINDING_LIQUIDPHASE_COMP_USED false

/**
 * @brief Emits code for a test that checks residual and Jacobian of non-binding model variants against all-binding ones
 * @details The correct macro, CADET_BINDINGTEST_SINGLE_IMPL_BNDVSNONBND_false or
 * CADET_BINDINGTEST_SINGLE_IMPL_BNDVSNONBND_true, is selected by concatentation with a variable:
 * CADET_BINDINGTEST_SINGLE_IMPL_BNDVSNONBND_##SomeVar(...)
 */
#define CADET_BINDINGTEST_SINGLE_IMPL_BNDVSNONBND_false(modelName, tagName, postFix, allBinding, someNonBinding,       \
														stateAll, stateSomeNon, configAll, configSomeNon)

#define CADET_BINDINGTEST_SINGLE_IMPL_BNDVSNONBND_true(modelName, tagName, postFix, allBinding, someNonBinding,        \
													   stateAll, stateSomeNon, configAll, configSomeNon)               \
	TEST_CASE(modelName " binding model consistency of non-binding vs binding" postFix,                                \
			  "[BindingModel],[Jacobian]," tagName)                                                                    \
	{                                                                                                                  \
		const unsigned int nBound2[] = BRACED_INIT_LIST allBinding;                                                    \
		const unsigned int nBound3[] = BRACED_INIT_LIST someNonBinding;                                                \
		const double state2[] = BRACED_INIT_LIST stateAll;                                                             \
		const double state3[] = BRACED_INIT_LIST stateSomeNon;                                                         \
		for (int bindMode = 0; bindMode < 2; ++bindMode)                                                               \
		{                                                                                                              \
			const bool isKinetic = bindMode;                                                                           \
			SECTION(std::string("Binding mode ") + (isKinetic ? "dynamic" : "quasi-stationary"))                       \
			{                                                                                                          \
				for (int adMode = 0; adMode < 2; ++adMode)                                                             \
				{                                                                                                      \
					const bool useAD = adMode;                                                                         \
					SECTION(std::string("Use AD ") + (useAD ? "yes" : "no"))                                           \
					{                                                                                                  \
						cadet::test::binding::testNonbindingBindingConsistency(                                        \
							modelName, sizeof(nBound2) / sizeof(unsigned int), sizeof(nBound3) / sizeof(unsigned int), \
							nBound2, nBound3, isKinetic, "{" configAll "}", "{" configSomeNon "}", useAD, state2,      \
							state3);                                                                                   \
					}                                                                                                  \
				}                                                                                                      \
			}                                                                                                          \
		}                                                                                                              \
	}

/**
 * @brief Used to make the macro argument more verbose and for increasing readability
 */
#define CADET_COMPARE_BINDING_VS_NONBINDING true
#define CADET_DONT_COMPARE_BINDING_VS_NONBINDING false

/**
 * @brief Emits tests for a binding model having a non-binding and an all-binding variant
 * @param modelName Identifier of the model as string (e.g. "LINEAR")
 * @param tagName Tags added to the tests as string (e.g., "[SOMETAG]")
 * @param postFix String appended to the test description / name
 * @param allBinding Array with number of bound states of the all-binding variant in parentheses (e.g., (1, 1, 2))
 * @param someNonBinding Array with number of bound states of the non-binding variant in parentheses (e.g., (1, 0, 2))
 * @param stateAll Array with full state vector (liquid and solid phase) of the all-binding variant in parentheses
 * @param stateSomeNon Array with full state vector (liquid and solid phase) of the non-binding variant in parentheses
 * @param configAll Interior of a JSON object block with parameters for the all-binding variant
 * @param configSomeNon Interior of a JSON object block with parameters for the non-binding variant
 * @param consInitTol Error tolerance for nonlinear solvers in consistent initialization
 * @param consInitCheckTol Error tolerance for residual check in consistent initialization
 * @param usesNonBindingLiquidPhase Determines whether a test for all-zero Jacobian columns belonging to non-binding
 * components is created
 * @param cmpBndVsNonbnd Determines whether a test for all-binding vs non-binding variant (Jacobian and residual) is
 * created
 */
#define CADET_BINDINGTEST_SINGLE_IMPL(modelName, tagName, postFix, allBinding, someNonBinding, stateAll, stateSomeNon, \
									  configAll, configSomeNon, consInitTol, consInitCheckTol,                         \
									  usesNonBindingLiquidPhase, cmpBndVsNonbnd)                                       \
	TEST_CASE(modelName " binding model analytic Jacobian vs AD" postFix, "[Jacobian],[AD],[BindingModel]," tagName)   \
	{                                                                                                                  \
		const unsigned int nBound2[] = BRACED_INIT_LIST allBinding;                                                    \
		const unsigned int nBound3[] = BRACED_INIT_LIST someNonBinding;                                                \
		const double state2[] = BRACED_INIT_LIST stateAll;                                                             \
		const double state3[] = BRACED_INIT_LIST stateSomeNon;                                                         \
		for (int bindMode = 0; bindMode < 2; ++bindMode)                                                               \
		{                                                                                                              \
			const bool isKinetic = bindMode;                                                                           \
			SECTION(std::string("Binding mode ") + (isKinetic ? "dynamic" : "quasi-stationary"))                       \
			{                                                                                                          \
				SECTION("Without nonbinding components")                                                               \
				{                                                                                                      \
					cadet::test::binding::testJacobianAD(modelName, sizeof(nBound2) / sizeof(unsigned int), nBound2,   \
														 isKinetic, "{" configAll "}", state2);                        \
				}                                                                                                      \
				SECTION("With nonbinding components")                                                                  \
				{                                                                                                      \
					cadet::test::binding::testJacobianAD(modelName, sizeof(nBound3) / sizeof(unsigned int), nBound3,   \
														 isKinetic, "{" configSomeNon "}", state3);                    \
				}                                                                                                      \
			}                                                                                                          \
		}                                                                                                              \
	}                                                                                                                  \
	CADET_BINDINGTEST_SINGLE_IMPL_NONBNDJACCONST_##usesNonBindingLiquidPhase(                                          \
		modelName, tagName, postFix, someNonBinding, stateSomeNon, configSomeNon)                                      \
		CADET_BINDINGTEST_SINGLE_IMPL_BNDVSNONBND_##cmpBndVsNonbnd(                                                    \
			modelName, tagName, postFix, allBinding, someNonBinding, stateAll, stateSomeNon, configAll, configSomeNon)

/**
 * @brief Emits tests for a binding model having a non-binding and an all-binding variant
 * @param modelName Identifier of the model as string (e.g. "LINEAR")
 * @param allBinding Array with number of bound states of the all-binding variant in parentheses (e.g., (1, 1, 2))
 * @param someNonBinding Array with number of bound states of the non-binding variant in parentheses (e.g., (1, 0, 2))
 * @param stateAll Array with full state vector (liquid and solid phase) of the all-binding variant in parentheses
 * @param stateSomeNon Array with full state vector (liquid and solid phase) of the non-binding variant in parentheses
 * @param configAll Interior of a JSON object block with parameters for the all-binding variant
 * @param configSomeNon Interior of a JSON object block with parameters for the non-binding variant
 * @param consInitTol Error tolerance for nonlinear solvers in consistent initialization
 * @param consInitCheckTol Error tolerance for residual check in consistent initialization
 * @param usesNonBindingLiquidPhase Determines whether a test for all-zero Jacobian columns belonging to non-binding
 * components is created
 * @param cmpBndVsNonbnd Determines whether a test for all-binding vs non-binding variant (Jacobian and residual) is
 * created
 */
#define CADET_BINDINGTEST_SINGLE(modelName, allBinding, someNonBinding, stateAll, stateSomeNon, configAll,             \
								 configSomeNon, consInitTol, consInitCheckTol, usesNonBindingLiquidPhase,              \
								 cmpBndVsNonbnd)                                                                       \
	CADET_BINDINGTEST_SINGLE_IMPL(modelName, "[" modelName "]", "", allBinding, someNonBinding, stateAll,              \
								  stateSomeNon, configAll, configSomeNon, consInitTol, consInitCheckTol,               \
								  usesNonBindingLiquidPhase, cmpBndVsNonbnd)

/**
 * @brief Emits tests for a binding model that can have external function dependence and has a non-binding and an
 * all-binding variant
 * @param modelName Identifier of the model as string (e.g. "LINEAR")
 * @param extModelName Identifier of the externally dependent model as string (e.g. "EXT_LINEAR")
 * @param postFix String appended to the test description / name
 * @param allBinding Array with number of bound states of the all-binding variant in parentheses (e.g., (1, 1, 2))
 * @param someNonBinding Array with number of bound states of the non-binding variant in parentheses (e.g., (1, 0, 2))
 * @param stateAll Array with full state vector (liquid and solid phase) of the all-binding variant in parentheses
 * @param stateSomeNon Array with full state vector (liquid and solid phase) of the non-binding variant in parentheses
 * @param configAll Interior of a JSON object block with parameters for the all-binding variant
 * @param configSomeNon Interior of a JSON object block with parameters for the non-binding variant
 * @param extConfigAll Interior of a JSON object block with parameters for the externally dependent all-binding variant
 * @param extConfigSomeNon Interior of a JSON object block with parameters for the externally dependent non-binding
 * variant
 * @param consInitTol Error tolerance for nonlinear solvers in consistent initialization
 * @param consInitCheckTol Error tolerance for residual check in consistent initialization
 * @param usesNonBindingLiquidPhase Determines whether a test for all-zero Jacobian columns belonging to non-binding
 * components is created
 * @param cmpBndVsNonbnd Determines whether a test for all-binding vs non-binding variant (Jacobian and residual) is
 * created
 */
#define CADET_BINDINGTEST_MULTI(modelName, extModelName, postFix, allBinding, someNonBinding, stateAll, stateSomeNon,  \
								configAll, configSomeNon, extConfigAll, extConfigSomeNon, consInitTol,                 \
								consInitCheckTol, usesNonBindingLiquidPhase, cmpBndVsNonbnd)                           \
	CADET_BINDINGTEST_SINGLE_IMPL(modelName, "[" modelName "]", postFix, allBinding, someNonBinding, stateAll,         \
								  stateSomeNon, configAll, configSomeNon, consInitTol, consInitCheckTol,               \
								  usesNonBindingLiquidPhase, cmpBndVsNonbnd)                                           \
	CADET_BINDINGTEST_SINGLE_IMPL(extModelName, "[ExternalFunction],[" extModelName "]", postFix, allBinding,          \
								  someNonBinding, stateAll, stateSomeNon, extConfigAll, extConfigSomeNon, consInitTol, \
								  consInitCheckTol, usesNonBindingLiquidPhase, cmpBndVsNonbnd)                         \
	TEST_CASE(modelName " binding model consistent with externally dependent variant" postFix,                         \
			  "[BindingModel],[ExternalFunction],[" modelName "],[" extModelName "]")                                  \
	{                                                                                                                  \
		const unsigned int nBound2[] = BRACED_INIT_LIST allBinding;                                                    \
		const unsigned int nBound3[] = BRACED_INIT_LIST someNonBinding;                                                \
		const double state2[] = BRACED_INIT_LIST stateAll;                                                             \
		const double state3[] = BRACED_INIT_LIST stateSomeNon;                                                         \
		for (int bindMode = 0; bindMode < 2; ++bindMode)                                                               \
		{                                                                                                              \
			const bool isKinetic = bindMode;                                                                           \
			SECTION(std::string("Binding mode ") + (isKinetic ? "dynamic" : "quasi-stationary"))                       \
			{                                                                                                          \
				SECTION("Without nonbinding components")                                                               \
				{                                                                                                      \
					cadet::test::binding::testNormalExternalConsistency(                                               \
						modelName, extModelName, sizeof(nBound2) / sizeof(unsigned int), nBound2, isKinetic,           \
						"{" configAll "," extConfigAll "}", state2);                                                   \
				}                                                                                                      \
				SECTION("With nonbinding components")                                                                  \
				{                                                                                                      \
					cadet::test::binding::testNormalExternalConsistency(                                               \
						modelName, extModelName, sizeof(nBound3) / sizeof(unsigned int), nBound3, isKinetic,           \
						"{" configSomeNon "," extConfigSomeNon "}", state3);                                           \
				}                                                                                                      \
			}                                                                                                          \
		}                                                                                                              \
	}

/**
 * @brief Emits tests for a binding model that can have external function dependence and has a non-binding and an
 * all-binding variant
 * @param modelName Identifier of the model as string (e.g. "LINEAR")
 * @param extModelName Identifier of the externally dependent model as string (e.g. "EXT_LINEAR")
 * @param allBinding Array with number of bound states of the all-binding variant in parentheses (e.g., (1, 1, 2))
 * @param someNonBinding Array with number of bound states of the non-binding variant in parentheses (e.g., (1, 0, 2))
 * @param stateAll Array with full state vector (liquid and solid phase) of the all-binding variant in parentheses
 * @param stateSomeNon Array with full state vector (liquid and solid phase) of the non-binding variant in parentheses
 * @param configAll Interior of a JSON object block with parameters for the all-binding variant
 * @param configSomeNon Interior of a JSON object block with parameters for the non-binding variant
 * @param extConfigAll Interior of a JSON object block with parameters for the externally dependent all-binding variant
 * @param extConfigSomeNon Interior of a JSON object block with parameters for the externally dependent non-binding
 * variant
 * @param consInitTol Error tolerance for nonlinear solvers in consistent initialization
 * @param consInitCheckTol Error tolerance for residual check in consistent initialization
 * @param usesNonBindingLiquidPhase Determines whether a test for all-zero Jacobian columns belonging to non-binding
 * components is created
 * @param cmpBndVsNonbnd Determines whether a test for all-binding vs non-binding variant (Jacobian and residual) is
 * created
 */
#define CADET_BINDINGTEST(modelName, extModelName, allBinding, someNonBinding, stateAll, stateSomeNon, configAll,      \
						  configSomeNon, extConfigAll, extConfigSomeNon, consInitTol, consInitCheckTol,                \
						  usesNonBindingLiquidPhase, cmpBndVsNonbnd)                                                   \
	CADET_BINDINGTEST_MULTI(modelName, extModelName, "", allBinding, someNonBinding, stateAll, stateSomeNon,           \
							configAll, configSomeNon, extConfigAll, extConfigSomeNon, consInitTol, consInitCheckTol,   \
							usesNonBindingLiquidPhase, cmpBndVsNonbnd)

/**
 * @brief Emits tests for a binding model that does not allow non-binding components
 * @param modelName Identifier of the model as string (e.g. "LINEAR")
 * @param tagName Tags added to the tests as string (e.g., "[SOMETAG]")
 * @param allBinding Array with number of bound states in parentheses (e.g., (1, 1, 2))
 * @param stateAll Array with full state vector (liquid and solid phase) in parentheses
 * @param configAll Interior of a JSON object block with parameters for the
 * @param consInitTol Error tolerance for nonlinear solvers in consistent initialization
 * @param consInitCheckTol Error tolerance for residual check in consistent initialization
 */
#define CADET_BINDINGTEST_ALLBINDING_SINGLE_IMPL(modelName, tagName, allBinding, stateAll, configAll, consInitTol,     \
												 consInitCheckTol)                                                     \
	TEST_CASE(modelName " binding model analytic Jacobian vs AD", "[Jacobian],[AD],[BindingModel]," tagName)           \
	{                                                                                                                  \
		const unsigned int nBound2[] = BRACED_INIT_LIST allBinding;                                                    \
		const double state2[] = BRACED_INIT_LIST stateAll;                                                             \
		for (int bindMode = 0; bindMode < 2; ++bindMode)                                                               \
		{                                                                                                              \
			const bool isKinetic = bindMode;                                                                           \
			SECTION(std::string("Binding mode ") + (isKinetic ? "dynamic" : "quasi-stationary"))                       \
			{                                                                                                          \
				cadet::test::binding::testJacobianAD(modelName, sizeof(nBound2) / sizeof(unsigned int), nBound2,       \
													 isKinetic, "{" configAll "}", state2);                            \
			}                                                                                                          \
		}                                                                                                              \
	}

/**
 * @brief Emits tests for a binding model that does not allow non-binding components
 * @param modelName Identifier of the model as string (e.g. "LINEAR")
 * @param allBinding Array with number of bound states in parentheses (e.g., (1, 1, 2))
 * @param stateAll Array with full state vector (liquid and solid phase) in parentheses
 * @param configAll Interior of a JSON object block with parameters for the
 * @param consInitTol Error tolerance for nonlinear solvers in consistent initialization
 * @param consInitCheckTol Error tolerance for residual check in consistent initialization
 */
#define CADET_BINDINGTEST_ALLBINDING_SINGLE(modelName, allBinding, stateAll, configAll, consInitTol, consInitCheckTol) \
	CADET_BINDINGTEST_ALLBINDING_SINGLE_IMPL(modelName, "[" modelName "]", allBinding, stateAll, configAll,            \
											 consInitTol, consInitCheckTol)

/**
 * @brief Emits tests for a binding model that can have external function dependence and does not allow non-binding
 * components
 * @param modelName Identifier of the model as string (e.g. "LINEAR")
 * @param extModelName Identifier of the externally dependent model as string (e.g. "EXT_LINEAR")
 * @param allBinding Array with number of bound states in parentheses (e.g., (1, 1, 2))
 * @param stateAll Array with full state vector (liquid and solid phase) in parentheses
 * @param configAll Interior of a JSON object block with parameters
 * @param extConfigAll Interior of a JSON object block with parameters for the externally dependent variant
 * @param consInitTol Error tolerance for nonlinear solvers in consistent initialization
 * @param consInitCheckTol Error tolerance for residual check in consistent initialization
 */
#define CADET_BINDINGTEST_ALLBINDING(modelName, extModelName, allBinding, stateAll, configAll, extConfigAll,           \
									 consInitTol, consInitCheckTol)                                                    \
	CADET_BINDINGTEST_ALLBINDING_SINGLE_IMPL(modelName, "[" modelName "]", allBinding, stateAll, configAll,            \
											 consInitTol, consInitCheckTol)                                            \
	CADET_BINDINGTEST_ALLBINDING_SINGLE_IMPL(extModelName, "[ExternalFunction],[" extModelName "]", allBinding,        \
											 stateAll, extConfigAll, consInitTol, consInitCheckTol)                    \
	TEST_CASE(modelName " binding model consistent with externally dependent variant",                                 \
			  "[BindingModel],[ExternalFunction],[" modelName "],[" extModelName "]")                                  \
	{                                                                                                                  \
		const unsigned int nBound2[] = BRACED_INIT_LIST allBinding;                                                    \
		const double state2[] = BRACED_INIT_LIST stateAll;                                                             \
		for (int bindMode = 0; bindMode < 2; ++bindMode)                                                               \
		{                                                                                                              \
			const bool isKinetic = bindMode;                                                                           \
			SECTION(std::string("Binding mode ") + (isKinetic ? "dynamic" : "quasi-stationary"))                       \
			{                                                                                                          \
				cadet::test::binding::testNormalExternalConsistency(                                                   \
					modelName, extModelName, sizeof(nBound2) / sizeof(unsigned int), nBound2, isKinetic,               \
					"{" configAll "," extConfigAll "}", state2);                                                       \
			}                                                                                                          \
		}                                                                                                              \
	}

#endif // CADETTEST_BINDINGMODELS_HPP_
