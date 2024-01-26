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

#include <catch.hpp>

#include "ColumnTests.hpp"
#include "ParticleHelper.hpp"
#include "ReactionModelTests.hpp"
#include "JsonTestModels.hpp"
#include "Weno.hpp"
#include "Utils.hpp"

TEST_CASE("LRMP LWE forward vs backward flow", "[LRMP],[Simulation]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		cadet::test::column::testWenoForwardBackward("LUMPED_RATE_MODEL_WITH_PORES", i, 6e-8, 4e-6);
}

TEST_CASE("LRMP linear pulse vs analytic solution", "[LRMP],[Simulation],[Analytic]")
{
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-pulseBenchmark.data", true,
											   true, 512, 6e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-pulseBenchmark.data", true,
											   false, 512, 6e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-pulseBenchmark.data", false,
											   true, 512, 6e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-pulseBenchmark.data", false,
											   false, 512, 6e-5, 1e-7);
}

TEST_CASE("LRMP non-binding linear pulse vs analytic solution", "[LRMP],[Simulation],[Analytic],[NonBinding]")
{
	cadet::test::column::testAnalyticNonBindingBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-nonBinding.data",
														 true, 512, 6e-5, 1e-7);
	cadet::test::column::testAnalyticNonBindingBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-nonBinding.data",
														 false, 512, 6e-5, 1e-7);
}

TEST_CASE("LRMP Jacobian forward vs backward flow", "[LRMP],[UnitOp],[Residual],[Jacobian],[AD]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		cadet::test::column::testJacobianWenoForwardBackward("LUMPED_RATE_MODEL_WITH_PORES", i);
}

TEST_CASE("LRMP time derivative Jacobian vs FD", "[LRMP],[UnitOp],[Residual],[Jacobian]")
{
	cadet::test::column::testTimeDerivativeJacobianFD("LUMPED_RATE_MODEL_WITH_PORES");
}

TEST_CASE("LRMP flux Jacobian vs FD", "[LRMP],[UnitOp],[Residual],[Jacobian]")
{
	cadet::test::column::testArrowHeadJacobianFD("LUMPED_RATE_MODEL_WITH_PORES");
}

TEST_CASE("LRMP sensitivity Jacobians", "[LRMP],[UnitOp],[Sensitivity]")
{
	cadet::test::column::testFwdSensJacobians("LUMPED_RATE_MODEL_WITH_PORES", 1e-4, 6e-7);
}

TEST_CASE("LRMP forward sensitivity vs FD", "[LRMP],[Sensitivity],[Simulation]")
{
	// Relative error is checked first, we use high absolute error for letting
	// some points that are far off pass the error test, too. This is required
	// due to errors in finite differences.
	const double fdStepSize[] = {1e-5, 1e-6, 1e-3, 1e-5};
	const double absTols[] = {6e5, 2e-2, 2e-2, 1.0};
	const double relTols[] = {5e-3, 1e-1, 5e-1, 6e-3};
	const double passRatio[] = {0.87, 0.84, 0.88, 0.95};
	cadet::test::column::testFwdSensSolutionFD("LUMPED_RATE_MODEL_WITH_PORES", true, fdStepSize, absTols, relTols,
											   passRatio);
}

TEST_CASE("LRMP forward sensitivity forward vs backward flow", "[LRMP],[Sensitivity],[Simulation]")
{
	const double absTols[] = {50.0, 2e-10, 1.0, 5e-7};
	const double relTols[] = {2e-4, 9e-6, 5e-7, 1e-7};
	const double passRatio[] = {1.0, 0.99, 0.98, 0.99};
	cadet::test::column::testFwdSensSolutionForwardBackward("LUMPED_RATE_MODEL_WITH_PORES", absTols, relTols,
															passRatio);
}

TEST_CASE("LRMP consistent initialization with linear binding", "[LRMP],[ConsistentInit]")
{
	cadet::test::column::testConsistentInitializationLinearBinding("LUMPED_RATE_MODEL_WITH_PORES", 1e-12, 1e-14);
}

TEST_CASE("LRMP consistent initialization with SMA binding", "[LRMP],[ConsistentInit]")
{
	std::vector<double> y(4 + 4 * 16 + 16 * (4 + 4) + 4 * 16, 0.0);
	// Optimal values:
	//	const double bindingCell[] = {1.2, 2.0, 1.0, 1.5, 858.034, 66.7896, 3.53273, 2.53153,
	//		1.0, 1.8, 1.5, 1.6, 856.173, 64.457, 5.73227, 2.85286};
	const double bindingCell[] = {1.2, 2.0, 1.0, 1.5, 840.0, 63.0, 3.0, 3.0, 1.0, 1.8, 1.5, 1.6, 840.0, 63.0, 6.0, 3.0};
	cadet::test::util::populate(
		y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4 + 4 * 16);
	cadet::test::util::repeat(y.data() + 4 + 4 * 16, bindingCell, 16, 8);
	cadet::test::util::populate(
		y.data() + 4 + 4 * 16 + 16 * (4 + 4), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; },
		4 * 16);

	cadet::test::column::testConsistentInitializationSMABinding("LUMPED_RATE_MODEL_WITH_PORES", y.data(), 1e-14, 1e-5);
}

TEST_CASE("LRMP consistent sensitivity initialization with linear binding", "[LRMP],[ConsistentInit],[Sensitivity]")
{
	// Fill state vector with given initial values
	const unsigned int numDofs = 4 + 4 * 16 + 16 * (4 + 4) + 4 * 16;
	std::vector<double> y(numDofs, 0.0);
	std::vector<double> yDot(numDofs, 0.0);
	cadet::test::util::populate(
		y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, numDofs);
	cadet::test::util::populate(
		yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, numDofs);

	cadet::test::column::testConsistentInitializationSensitivity("LUMPED_RATE_MODEL_WITH_PORES", y.data(), yDot.data(),
																 true, 1e-14);
}

TEST_CASE("LRMP consistent sensitivity initialization with SMA binding", "[LRMP],[ConsistentInit],[Sensitivity]")
{
	// Fill state vector with given initial values
	const unsigned int numDofs = 4 + 4 * 16 + 16 * (4 + 4) + 4 * 16;
	std::vector<double> y(numDofs, 0.0);
	std::vector<double> yDot(numDofs, 0.0);

	const double bindingCell[] = {1.0, 1.8, 1.5, 1.6, 840.0, 63.0, 6.0, 3.0};
	cadet::test::util::populate(
		y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4 + 4 * 16);
	cadet::test::util::repeat(y.data() + 4 + 4 * 16, bindingCell, 8, 16);
	cadet::test::util::populate(
		y.data() + 4 + 4 * 16 + 16 * (4 + 4), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; },
		4 * 16);

	cadet::test::util::populate(
		yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, numDofs);

	cadet::test::column::testConsistentInitializationSensitivity("LUMPED_RATE_MODEL_WITH_PORES", y.data(), yDot.data(),
																 false, 1e-10);
}

TEST_CASE("LRMP inlet DOF Jacobian", "[LRMP],[UnitOp],[Jacobian],[Inlet]")
{
	cadet::test::column::testInletDofJacobian("LUMPED_RATE_MODEL_WITH_PORES");
}

TEST_CASE("LRMP LWE one vs two identical particle types match", "[LRMP],[Simulation],[ParticleType]")
{
	cadet::test::particle::testOneVsTwoIdenticalParticleTypes("LUMPED_RATE_MODEL_WITH_PORES", 2.2e-8, 6e-5);
}

TEST_CASE("LRMP LWE separate identical particle types match", "[LRMP],[Simulation],[ParticleType]")
{
	cadet::test::particle::testSeparateIdenticalParticleTypes("LUMPED_RATE_MODEL_WITH_PORES", 1e-15, 1e-15);
}

TEST_CASE("LRMP linear binding single particle matches particle distribution", "[LRMP],[Simulation],[ParticleType]")
{
	cadet::test::particle::testLinearMixedParticleTypes("LUMPED_RATE_MODEL_WITH_PORES", 5e-8, 5e-5);
}

TEST_CASE("LRMP multiple particle types Jacobian analytic vs AD", "[LRMP],[Jacobian],[AD],[ParticleType]")
{
	cadet::test::particle::testJacobianMixedParticleTypes("LUMPED_RATE_MODEL_WITH_PORES");
}

TEST_CASE("LRMP multiple particle types time derivative Jacobian vs FD",
		  "[LRMP],[UnitOp],[Residual],[Jacobian],[ParticleType]")
{
	cadet::test::particle::testTimeDerivativeJacobianMixedParticleTypesFD("LUMPED_RATE_MODEL_WITH_PORES", 1e-6, 0.0,
																		  9e-4);
}

TEST_CASE("LRMP multiple spatially dependent particle types Jacobian analytic vs AD",
		  "[LRMP],[Jacobian],[AD],[ParticleType]")
{
	cadet::test::particle::testJacobianSpatiallyMixedParticleTypes("LUMPED_RATE_MODEL_WITH_PORES");
}

TEST_CASE("LRMP linear binding single particle matches spatially dependent particle distribution",
		  "[LRMP],[Simulation],[ParticleType]")
{
	cadet::test::particle::testLinearSpatiallyMixedParticleTypes("LUMPED_RATE_MODEL_WITH_PORES", 5e-8, 5e-5);
}

TEST_CASE("LRMP multiple spatially dependent particle types flux Jacobian vs FD",
		  "[LRMP],[UnitOp],[Residual],[Jacobian],[ParticleType]")
{
	cadet::test::particle::testArrowHeadJacobianSpatiallyMixedParticleTypes("LUMPED_RATE_MODEL_WITH_PORES", 1e-6, 1e-8,
																			1e-5);
}

TEST_CASE("LRMP dynamic reactions Jacobian vs AD bulk", "[LRMP],[Jacobian],[AD],[ReactionModel]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("LUMPED_RATE_MODEL_WITH_PORES", true, false, false);
}

TEST_CASE("LRMP dynamic reactions Jacobian vs AD particle", "[LRMP],[Jacobian],[AD],[ReactionModel]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("LUMPED_RATE_MODEL_WITH_PORES", false, true, false);
}

TEST_CASE("LRMP dynamic reactions Jacobian vs AD modified particle", "[LRMP],[Jacobian],[AD],[ReactionModel]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("LUMPED_RATE_MODEL_WITH_PORES", false, true, true);
}

TEST_CASE("LRMP dynamic reactions Jacobian vs AD bulk and particle", "[LRMP],[Jacobian],[AD],[ReactionModel]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("LUMPED_RATE_MODEL_WITH_PORES", true, true, false);
}

TEST_CASE("LRMP dynamic reactions Jacobian vs AD bulk and modified particle", "[LRMP],[Jacobian],[AD],[ReactionModel]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("LUMPED_RATE_MODEL_WITH_PORES", true, true, true);
}

TEST_CASE("LRMP dynamic reactions time derivative Jacobian vs FD bulk", "[LRMP],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("LUMPED_RATE_MODEL_WITH_PORES", true, false,
																		false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("LRMP dynamic reactions time derivative Jacobian vs FD particle",
		  "[LRMP],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("LUMPED_RATE_MODEL_WITH_PORES", false, true,
																		false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("LRMP dynamic reactions time derivative Jacobian vs FD modified particle",
		  "[LRMP],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("LUMPED_RATE_MODEL_WITH_PORES", false, true,
																		true, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("LRMP dynamic reactions time derivative Jacobian vs FD bulk and particle",
		  "[LRMP],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("LUMPED_RATE_MODEL_WITH_PORES", true, true,
																		false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("LRMP dynamic reactions time derivative Jacobian vs FD bulk and modified particle",
		  "[LRMP],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("LUMPED_RATE_MODEL_WITH_PORES", true, true,
																		true, 1e-6, 1e-14, 8e-4);
}

inline cadet::JsonParameterProvider createColumnWithTwoCompLinearBindingThreeParticleTypes()
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding("LUMPED_RATE_MODEL_WITH_PORES");

	const double parVolFrac[] = {0.3, 0.6, 0.1};
	const double parFactor[] = {0.9, 0.8};
	cadet::test::particle::extendModelToManyParticleTypes(jpp, 3, parFactor, parVolFrac);

	return jpp;
}

TEST_CASE("LRMP multi particle types dynamic reactions Jacobian vs AD bulk",
		  "[LRMP],[Jacobian],[AD],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, false, false);
}

TEST_CASE("LRMP multi particle types dynamic reactions Jacobian vs AD particle",
		  "[LRMP],[Jacobian],[AD],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, false, true, false);
}

TEST_CASE("LRMP multi particle types dynamic reactions Jacobian vs AD modified particle",
		  "[LRMP],[Jacobian],[AD],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, false, true, true);
}

TEST_CASE("LRMP multi particle types dynamic reactions Jacobian vs AD bulk and particle",
		  "[LRMP],[Jacobian],[AD],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, true, false);
}

TEST_CASE("LRMP multi particle types dynamic reactions Jacobian vs AD bulk and modified particle",
		  "[LRMP],[Jacobian],[AD],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, true, true);
}

TEST_CASE("LRMP multi particle types dynamic reactions time derivative Jacobian vs FD bulk",
		  "[LRMP],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, false, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("LRMP multi particle types dynamic reactions time derivative Jacobian vs FD particle",
		  "[LRMP],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, false, true, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("LRMP multi particle types dynamic reactions time derivative Jacobian vs FD modified particle",
		  "[LRMP],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, false, true, true, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("LRMP multi particle types dynamic reactions time derivative Jacobian vs FD bulk and particle",
		  "[LRMP],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, true, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("LRMP multi particle types dynamic reactions time derivative Jacobian vs FD bulk and modified particle",
		  "[LRMP],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, true, true, 1e-6, 1e-14, 8e-4);
}
