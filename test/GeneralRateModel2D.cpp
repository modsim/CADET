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
#include "Weno.hpp"
#include "Utils.hpp"
#include "JsonTestModels.hpp"
#include "JacobianHelper.hpp"
#include "cadet/ModelBuilder.hpp"
#include "ModelBuilderImpl.hpp"
#include "cadet/FactoryFuncs.hpp"
#include "ParallelSupport.hpp"

TEST_CASE("GRM2D LWE forward vs backward flow", "[GRM2D],[Simulation]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		cadet::test::column::testWenoForwardBackward("GENERAL_RATE_MODEL_2D", i, 1e-9, 2e-4);
}

TEST_CASE("GRM2D Jacobian forward vs backward flow", "[GRM2D],[UnitOp],[Residual],[Jacobian]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		cadet::test::column::testJacobianWenoForwardBackwardFD("GENERAL_RATE_MODEL_2D", i, 1e-6, 0.0, 1e-3);
}

TEST_CASE("GRM2D time derivative Jacobian vs FD", "[GRM2D],[UnitOp],[Residual],[Jacobian]")
{
	cadet::test::column::testTimeDerivativeJacobianFD("GENERAL_RATE_MODEL_2D", 1e-6, 0.0, 9e-4);
}

TEST_CASE("GRM2D rapid-equilibrium binding flux Jacobian vs FD", "[GRM2D],[UnitOp],[Residual],[Jacobian]")
{
	cadet::test::column::testArrowHeadJacobianFD("GENERAL_RATE_MODEL_2D", false, 1e-6, 2e-9);
}

TEST_CASE("GRM2D dynamic binding flux Jacobian vs FD", "[GRM2D],[UnitOp],[Residual],[Jacobian]")
{
	cadet::test::column::testArrowHeadJacobianFD("GENERAL_RATE_MODEL_2D", true, 1e-6, 2e-9);
}

TEST_CASE("GRM2D sensitivity Jacobians", "[GRM2D],[UnitOp],[Sensitivity]")
{
	cadet::test::column::testFwdSensJacobians("GENERAL_RATE_MODEL_2D", 1e-4, 6e-7);
}

TEST_CASE("GRM2D forward sensitivity vs FD", "[GRM2D],[Sensitivity],[Simulation]")
{
	// Relative error is checked first, we use high absolute error for letting
	// some points that are far off pass the error test, too. This is required
	// due to errors in finite differences.
	const double fdStepSize[] = {5e-5, 1e-4, 1e-4, 1e-3};
	const double absTols[] = {3e5, 2e-3, 2e-4, 5.0};
	const double relTols[] = {5e-3, 7e-2, 8e-2, 1e-4};
	const double passRatio[] = {0.95, 0.9, 0.91, 0.83};
	cadet::test::column::testFwdSensSolutionFD("GENERAL_RATE_MODEL_2D", false, fdStepSize, absTols, relTols, passRatio);
}

TEST_CASE("GRM2D forward sensitivity forward vs backward flow", "[GRM2D],[Sensitivity],[Simulation]")
{
	const double absTols[] = {4e-5, 1e-11, 1e-11, 8e-9};
	const double relTols[] = {6e-9, 5e-8, 5e-6, 5e-10};
	const double passRatio[] = {0.99, 0.95, 0.98, 0.98};
	cadet::test::column::testFwdSensSolutionForwardBackward("GENERAL_RATE_MODEL_2D", absTols, relTols, passRatio);
}

TEST_CASE("GRM2D consistent initialization with linear binding", "[GRM2D],[ConsistentInit]")
{
	cadet::test::column::testConsistentInitializationLinearBinding("GENERAL_RATE_MODEL_2D", 1e-12, 1e-14);
}

TEST_CASE("GRM2D consistent initialization with SMA binding", "[GRM2D],[ConsistentInit]")
{
	std::vector<double> y(4 * 3 + 4 * 8 * 3 + 8 * 3 * 3 * (4 + 4) + 4 * 8 * 3, 0.0);
	// Optimal values:
	//	const double bindingCell[] = {1.2, 2.0, 1.0, 1.5, 858.034, 66.7896, 3.53273, 2.53153,
	//		1.0, 1.8, 1.5, 1.6, 856.173, 64.457, 5.73227, 2.85286};
	const double bindingCell[] = {1.2, 2.0, 1.0, 1.5, 840.0, 63.0, 3.0, 3.0, 1.0, 1.8, 1.5, 1.6, 840.0, 63.0, 6.0, 3.0};
	cadet::test::util::populate(
		y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4 * 3 + 4 * 8 * 3);
	cadet::test::util::repeat(y.data() + 4 * 3 + 4 * 8 * 3, bindingCell, 16, 3 * 8 * 3 / 2);
	cadet::test::util::populate(
		y.data() + 4 * 3 + 4 * 8 * 3 + 8 * 3 * 3 * (4 + 4),
		[](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4 * 8 * 3);

	cadet::test::column::testConsistentInitializationSMABinding("GENERAL_RATE_MODEL_2D", y.data(), 1e-14, 1e-5);
}

TEST_CASE("GRM2D consistent sensitivity initialization with linear binding", "[GRM2D],[ConsistentInit],[Sensitivity]")
{
	// Fill state vector with given initial values
	const unsigned int numDofs = 2 * 3 + 2 * 8 * 3 + 8 * 3 * 3 * (2 + 2) + 2 * 8 * 3;
	std::vector<double> y(numDofs, 0.0);
	std::vector<double> yDot(numDofs, 0.0);
	cadet::test::util::populate(
		y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, numDofs);
	cadet::test::util::populate(
		yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, numDofs);

	cadet::test::column::testConsistentInitializationSensitivity("GENERAL_RATE_MODEL_2D", y.data(), yDot.data(), true,
																 1e-14);
}

TEST_CASE("GRM2D consistent sensitivity initialization with SMA binding", "[GRM2D],[ConsistentInit],[Sensitivity]")
{
	// Fill state vector with given initial values
	const unsigned int numDofs = 4 * 3 + 4 * 8 * 3 + 8 * 3 * 3 * (4 + 4) + 4 * 8 * 3;
	std::vector<double> y(numDofs, 0.0);
	std::vector<double> yDot(numDofs, 0.0);

	const double bindingCell[] = {1.0, 1.8, 1.5, 1.6, 840.0, 63.0, 6.0, 3.0};
	cadet::test::util::populate(
		y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4 * 3 + 4 * 8 * 3);
	cadet::test::util::repeat(y.data() + 4 * 3 + 4 * 8 * 3, bindingCell, 8, 3 * 8 * 3);
	cadet::test::util::populate(
		y.data() + 4 * 3 + 4 * 8 * 3 + 8 * 3 * 3 * (4 + 4),
		[](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4 * 8 * 3);

	cadet::test::util::populate(
		yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, numDofs);

	cadet::test::column::testConsistentInitializationSensitivity("GENERAL_RATE_MODEL_2D", y.data(), yDot.data(), false,
																 1e-9);
}

TEST_CASE("GRM2D inlet DOF Jacobian", "[GRM2D],[UnitOp],[Jacobian],[Inlet]")
{
	cadet::test::column::testInletDofJacobian("GENERAL_RATE_MODEL_2D");
}

TEST_CASE("GRM2D LWE one vs two identical particle types match", "[GRM2D],[Simulation],[ParticleType]")
{
	cadet::test::particle::testOneVsTwoIdenticalParticleTypes("GENERAL_RATE_MODEL_2D", 2e-8, 5e-5);
}

TEST_CASE("GRM2D LWE separate identical particle types match", "[GRM2D],[Simulation],[ParticleType]")
{
	cadet::test::particle::testSeparateIdenticalParticleTypes("GENERAL_RATE_MODEL_2D", 1e-15, 1e-15);
}

TEST_CASE("GRM2D linear binding single particle matches particle distribution", "[GRM2D],[Simulation],[ParticleType]")
{
	cadet::test::particle::testLinearMixedParticleTypes("GENERAL_RATE_MODEL_2D", 5e-8, 5e-5);
}

TEST_CASE("GRM2D multiple particle types Jacobian analytic vs AD", "[GRM2D],[Jacobian],[AD],[ParticleType]")
{
	cadet::test::particle::testJacobianMixedParticleTypes("GENERAL_RATE_MODEL_2D");
}

TEST_CASE("GRM2D multiple particle types time derivative Jacobian vs FD",
		  "[GRM2D],[UnitOp],[Residual],[Jacobian],[ParticleType]")
{
	cadet::test::particle::testTimeDerivativeJacobianMixedParticleTypesFD("GENERAL_RATE_MODEL_2D", 1e-6, 0.0, 9e-4);
}

TEST_CASE("GRM2D linear binding single particle matches spatially dependent particle distribution",
		  "[GRM2D],[Simulation],[ParticleType]")
{
	cadet::test::particle::testLinearSpatiallyMixedParticleTypes("GENERAL_RATE_MODEL_2D", 5e-8, 5e-5);
}

TEST_CASE("GRM2D multiple spatially dependent particle types flux Jacobian vs FD",
		  "[GRM2D],[UnitOp],[Residual],[Jacobian],[ParticleType]")
{
	cadet::test::particle::testArrowHeadJacobianSpatiallyMixedParticleTypes("GENERAL_RATE_MODEL_2D", 1e-6, 1e-8, 1e-5);
}

TEST_CASE("GRM2D dynamic reactions time derivative Jacobian vs FD bulk",
		  "[GRM2D],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("GENERAL_RATE_MODEL_2D", true, false, false,
																		1e-6, 1e-14, 8e-4);
}

TEST_CASE("GRM2D dynamic reactions time derivative Jacobian vs FD particle",
		  "[GRM2D],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("GENERAL_RATE_MODEL_2D", false, true, false,
																		1e-6, 1e-14, 8e-4);
}

TEST_CASE("GRM2D dynamic reactions time derivative Jacobian vs FD modified particle",
		  "[GRM2D],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("GENERAL_RATE_MODEL_2D", false, true, true,
																		1e-6, 1e-14, 8e-4);
}

TEST_CASE("GRM2D dynamic reactions time derivative Jacobian vs FD bulk and particle",
		  "[GRM2D],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("GENERAL_RATE_MODEL_2D", true, true, false,
																		1e-6, 1e-14, 8e-4);
}

TEST_CASE("GRM2D dynamic reactions time derivative Jacobian vs FD bulk and modified particle",
		  "[GRM2D],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("GENERAL_RATE_MODEL_2D", true, true, true, 1e-6,
																		1e-14, 8e-4);
}

inline cadet::JsonParameterProvider createColumnWithTwoCompLinearBindingThreeParticleTypes()
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding("GENERAL_RATE_MODEL_2D");

	const double parVolFrac[] = {0.3, 0.6, 0.1};
	const double parFactor[] = {0.9, 0.8};
	cadet::test::particle::extendModelToManyParticleTypes(jpp, 3, parFactor, parVolFrac);

	return jpp;
}

TEST_CASE("GRM2D multi particle types dynamic reactions time derivative Jacobian vs FD bulk",
		  "[GRM2D],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, false, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("GRM2D multi particle types dynamic reactions time derivative Jacobian vs FD particle",
		  "[GRM2D],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, false, true, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("GRM2D multi particle types dynamic reactions time derivative Jacobian vs FD modified particle",
		  "[GRM2D],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, false, true, true, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("GRM2D multi particle types dynamic reactions time derivative Jacobian vs FD bulk and particle",
		  "[GRM2D],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, true, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("GRM2D multi particle types dynamic reactions time derivative Jacobian vs FD bulk and modified particle",
		  "[GRM2D],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, true, true, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("GRM2D with 1 radial zone matches GRM", "[GRM],[GRM2D],[UnitOp],[Jacobian]")
{
	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	// Create a unit
	cadet::IModel* const iUnitGrm = mb->createUnitOperation("GENERAL_RATE_MODEL", 0);
	cadet::IModel* const iUnitGrm2d = mb->createUnitOperation("GENERAL_RATE_MODEL_2D", 0);
	REQUIRE(nullptr != iUnitGrm);
	REQUIRE(nullptr != iUnitGrm2d);

	cadet::IUnitOperation* const grm = reinterpret_cast<cadet::IUnitOperation*>(iUnitGrm);
	cadet::IUnitOperation* const grm2d = reinterpret_cast<cadet::IUnitOperation*>(iUnitGrm2d);

	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding("GENERAL_RATE_MODEL");
	const double velocity = jpp.getDouble("VELOCITY");
	const double colRadius = jpp.getDouble("COL_RADIUS");
	const double colPorosity = jpp.getDouble("COL_POROSITY");
	const double crossSectionArea = 3.1415926535897932384626434 * (colRadius * colRadius - 0.0 * 0.0);

	jpp.set("COL_DISPERSION_RADIAL", 0.0);
	jpp.set("CROSS_SECTION_AREA", crossSectionArea);

	jpp.pushScope("discretization");
	jpp.set("NRAD", 1);
	jpp.set("RADIAL_DISC_TYPE", "EQUIDISTANT");
	jpp.set("LINEAR_SOLVER_BULK", "DENSE");
	jpp.popScope();

	// Set WENO order
	const int wenoOrder = 3;
	cadet::test::column::setWenoOrder(jpp, wenoOrder);

	// Configure
	cadet::ModelBuilder& temp = *reinterpret_cast<cadet::ModelBuilder*>(mb);
	REQUIRE(grm->configureModelDiscretization(jpp, temp));
	REQUIRE(grm->configure(jpp));
	REQUIRE(grm2d->configureModelDiscretization(jpp, temp));
	REQUIRE(grm2d->configure(jpp));

	REQUIRE(grm2d->numDofs() == grm->numDofs());

	const cadet::active flowIn[] = {velocity * colPorosity * crossSectionArea};
	const cadet::active flowOut[] = {velocity * colPorosity * crossSectionArea};
	grm->setFlowRates(flowIn, flowOut);
	grm2d->setFlowRates(flowIn, flowOut);

	// Obtain memory for state, Jacobian multiply direction, Jacobian column
	const unsigned int nDof = grm->numDofs();
	std::vector<double> y(nDof, 0.0);
	std::vector<double> yDot(nDof, 0.0);
	std::vector<double> jacDir(nDof, 0.0);
	std::vector<double> jacCol1(nDof, 0.0);
	std::vector<double> jacCol2(nDof, 0.0);
	cadet::util::ThreadLocalStorage tls;
	tls.resize(std::max(grm2d->threadLocalMemorySize(), grm->threadLocalMemorySize()));

	// Fill state vectors with some values
	cadet::test::util::populate(
		y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
	cadet::test::util::populate(
		yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);

	// Setup matrices
	const cadet::AdJacobianParams noAdParams{nullptr, nullptr, 0u};
	grm->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, noAdParams);
	grm2d->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, noAdParams);

	// Compute Jacobian
	const cadet::SimulationTime simTime{0.0, 0u};
	const cadet::ConstSimulationState simState{y.data(), yDot.data()};
	grm->residualWithJacobian(simTime, simState, jacCol1.data(), noAdParams, tls);
	grm2d->residualWithJacobian(simTime, simState, jacCol2.data(), noAdParams, tls);

	for (unsigned int i = 0; i < nDof; ++i)
	{
		CAPTURE(i);
		CHECK(jacCol1[i] == jacCol2[i]);
	}

	// Compare Jacobian
	cadet::test::compareJacobian(grm, grm2d, y.data(), yDot.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), 0.0,
								 1e-15);

	// Compare Jacobian solutions
	cadet::test::util::populate(
		jacCol1.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + 2 * nDof) * 0.17)) + 1e-4; }, nDof);
	std::copy(jacCol1.begin(), jacCol1.end(), jacCol2.begin());
	std::fill(jacDir.begin(), jacDir.end(), 1.0);

	REQUIRE(grm->linearSolve(0.0, 1.0, 0.33, jacCol1.data(), jacDir.data(), simState) == 0);
	REQUIRE(grm2d->linearSolve(0.0, 1.0, 0.33, jacCol2.data(), jacDir.data(), simState) == 0);

	for (unsigned int i = 0; i < nDof; ++i)
	{
		CAPTURE(i);
		CHECK(jacCol1[i] == jacCol2[i]);
	}

	mb->destroyUnitOperation(iUnitGrm2d);
	mb->destroyUnitOperation(iUnitGrm);
	cadet::destroyModelBuilder(mb);
}
