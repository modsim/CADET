// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

// we have the following tests:
// fixGRM: test that does not work and is not based on FD
// CI: test that works and is not based on FD
// FDtestGRM: Test based on FD that works on my machine
// failedFDtestGRM: Test based on FD that does not work on my machine

#include <catch.hpp>

#include "ColumnTests.hpp"
#include "ParticleHelper.hpp"
#include "ReactionModelTests.hpp"
#include "JsonTestModels.hpp"
#include "Weno.hpp"
#include "Utils.hpp"

TEST_CASE("GRM LWE forward vs backward flow", "[GRM],[Simulation],[CI]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		cadet::test::column::testWenoForwardBackward("GENERAL_RATE_MODEL", i, 1e-9, 2e-4);
}

TEST_CASE("GRM linear pulse vs analytic solution", "[GRM],[Simulation],[Analytic],[CI]")
{
	cadet::test::column::testAnalyticBenchmark("GENERAL_RATE_MODEL", "/data/grm-pulseBenchmark.data", true, true, 512, 6e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("GENERAL_RATE_MODEL", "/data/grm-pulseBenchmark.data", true, false, 512, 6e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("GENERAL_RATE_MODEL", "/data/grm-pulseBenchmark.data", false, true, 512, 6e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("GENERAL_RATE_MODEL", "/data/grm-pulseBenchmark.data", false, false, 512, 6e-5, 1e-7);
}

TEST_CASE("GRM non-binding linear pulse vs analytic solution", "[GRM],[Simulation],[Analytic],[NonBinding],[CI]")
{
	cadet::test::column::testAnalyticNonBindingBenchmark("GENERAL_RATE_MODEL", "/data/grm-nonBinding.data", true, 512, 6e-5, 1e-7);
	cadet::test::column::testAnalyticNonBindingBenchmark("GENERAL_RATE_MODEL", "/data/grm-nonBinding.data", false, 512, 6e-5, 1e-7);
}

TEST_CASE("GRM Jacobian forward vs backward flow", "[GRM],[UnitOp],[Residual],[Jacobian],[AD],[CI]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		cadet::test::column::testJacobianWenoForwardBackward("GENERAL_RATE_MODEL", i);
}

TEST_CASE("GRM numerical Benchmark with parameter sensitivities for linear case", "[GRM],[Simulation],[Reference],[Sensitivity],[CI]")
{
	const std::string& modelFilePath = std::string("/data/model_GRM_dynLin_1comp_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_GRM_dynLin_1comp_sensbenchmark1_FV_Z32parZ4.h5");
	const std::vector<double> absTol = { 1e-12, 1e-12, 1e-12, 1e-12 };
	const std::vector<double> relTol = { 1.0, 1.0, 1.0, 1.0 };
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, 32, 4, true);
}

TEST_CASE("GRM numerical Benchmark with parameter sensitivities for SMA LWE case", "[GRM],[Simulation],[Reference],[Sensitivity],[CI]")
{
	const std::string& modelFilePath = std::string("/data/model_GRM_reqSMA_4comp_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_GRM_reqSMA_4comp_sensbenchmark1_FV_Z16parZ2.h5");
	const std::vector<double> absTol = { 1e-12, 1e-12, 1e-12, 1e-12 };
	const std::vector<double> relTol = { 1.0, 1.0, 1.0, 1.0 };
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "000", absTol, relTol, 16, 2, true);
}

TEST_CASE("GRM numerical EOC Benchmark with parameter sensitivities for linear case", "[releaseCI],[EOC],[EOC_GRM_FV]")
{
	const std::string& modelFilePath = std::string("/data/model_GRM_dynLin_1comp_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_GRM_dynLin_1comp_sensbenchmark1_FV_Z1024parZ128.h5");
	const std::string& convFilePath = std::string("/data/convergence_GRM_dynLin_1comp_sensbenchmark1.json");
	const std::vector<double> absTol = { 1e-12, 1e-12, 1e-12, 1e-12 };
	const std::vector<double> relTol = { 1.0, 1.0, 1.0, 1.0 };
	cadet::test::column::testEOCReferenceBenchmark(modelFilePath, refFilePath, convFilePath, "001", absTol, relTol, 3, 8, 1, true);
}

TEST_CASE("GRM numerical EOC Benchmark with parameter sensitivities for SMA LWE case", "[releaseCI],[EOC],[EOC_GRM_FV]")
{
	const std::string& modelFilePath = std::string("/data/model_GRM_reqSMA_4comp_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_GRM_reqSMA_4comp_sensbenchmark1_FV_Z512parZ64.h5");
	const std::string& convFilePath = std::string("/data/convergence_GRM_reqSMA_4comp_sensbenchmark1.json");
	const std::vector<double> absTol = { 1e-12, 1e-12, 1e-12, 1e-12 };
	const std::vector<double> relTol = { 1.0, 1.0, 1.0, 1.0 };
	cadet::test::column::testEOCReferenceBenchmark(modelFilePath, refFilePath, convFilePath, "000", absTol, relTol, 2, 8, 1, true);
}

TEST_CASE("GRM time derivative Jacobian vs FD", "[GRM],[UnitOp],[Residual],[Jacobian],[CI],[FD]")
{
	cadet::test::column::testTimeDerivativeJacobianFD("GENERAL_RATE_MODEL", 1e-6, 0.0, 9e-4);
}

TEST_CASE("GRM rapid-equilibrium binding flux Jacobian vs FD", "[GRM],[UnitOp],[Residual],[Jacobian],[CI],[FD]")
{
	cadet::test::column::testArrowHeadJacobianFD("GENERAL_RATE_MODEL", false, 1e-6, 2e-9);
}

TEST_CASE("GRM rapid-equilibrium binding with surf diff par dep flux Jacobian vs FD", "[GRM],[UnitOp],[Residual],[Jacobian],[ParameterDependence],[CI],[FD]")
{
	cadet::test::column::testArrowHeadJacobianFDVariableParSurfDiff("GENERAL_RATE_MODEL", 1e-6, 5e-9);
}

TEST_CASE("GRM dynamic binding with surf diff par dep Jacobian vs AD", "[GRM],[UnitOp],[Residual],[Jacobian],[ParameterDependence],[CI]")
{
	cadet::test::column::testJacobianADVariableParSurfDiff("GENERAL_RATE_MODEL", true);
}

TEST_CASE("GRM rapid-equilibrium binding with surf diff par dep Jacobian vs AD", "[GRM],[UnitOp],[Residual],[Jacobian],[ParameterDependence],[CI]")
{
	cadet::test::column::testJacobianADVariableParSurfDiff("GENERAL_RATE_MODEL", false);
}

TEST_CASE("GRM dynamic binding flux Jacobian vs FD", "[GRM],[UnitOp],[Residual],[Jacobian],[CI],[FD]")
{
	cadet::test::column::testArrowHeadJacobianFD("GENERAL_RATE_MODEL", true, 1e-6, 2e-9);
}

TEST_CASE("GRM sensitivity Jacobians", "[GRM],[UnitOp],[Sensitivity],[CI]")
{
	cadet::test::column::testFwdSensJacobians("GENERAL_RATE_MODEL", 1e-4, 6e-7);
}

//TEST_CASE("GRM forward sensitivity vs FD", "[GRM],[Sensitivity],[Simulation],[failedFDtestGRM]") // todo fix. fails for SMA_KA for both binding modes
//{
//	// todo comment
//	// Relative error is checked first, we use high absolute error for letting
//	// some points that are far off pass the error test, too. This is required
//	// due to errors in finite differences.
//	const double fdStepSize[] = {5e-5, 1e-4, 1e-4, 1e-3};
//	const double absTols[] = {3e5, 2e-3, 2e-4, 5.0};
//	const double relTols[] = {5e-3, 7e-2, 8e-2, 1e-4};
//	const double passRatio[] = {0.95, 0.9, 0.91, 0.83};
//	cadet::test::column::testFwdSensSolutionFD("GENERAL_RATE_MODEL", false, fdStepSize, absTols, relTols, passRatio);
//}
//
//TEST_CASE("GRM forward sensitivity forward vs backward flow", "[GRM],[Sensitivity],[Simulation],[fixGRM]") // todo fix. fails for COL_DISPERION for both binding modes
//{
//	const double absTols[] = {4e-5, 1e-11, 1e-11, 8e-9};
//	const double relTols[] = {6e-9, 5e-8, 5e-6, 5e-10};
//	const double passRatio[] = {0.99, 0.95, 0.98, 0.98};
//	cadet::test::column::testFwdSensSolutionForwardBackward("GENERAL_RATE_MODEL", absTols, relTols, passRatio);
//}

TEST_CASE("GRM consistent initialization with linear binding", "[GRM],[ConsistentInit],[CI]")
{
	cadet::test::column::testConsistentInitializationLinearBinding("GENERAL_RATE_MODEL", 1e-12, 1e-12);
}

//TEST_CASE("GRM consistent initialization with SMA binding", "[GRM],[ConsistentInit],[fixGRM]") // todo fix
//{
//	std::vector<double> y(4 + 4 * 16 + 16 * 4 * (4 + 4) + 4 * 16, 0.0);
//// Optimal values:
////	const double bindingCell[] = {1.2, 2.0, 1.0, 1.5, 858.034, 66.7896, 3.53273, 2.53153, 
////		1.0, 1.8, 1.5, 1.6, 856.173, 64.457, 5.73227, 2.85286};
//	const double bindingCell[] = {1.2, 2.0, 1.0, 1.5, 840.0, 63.0, 3.0, 3.0, 
//		1.0, 1.8, 1.5, 1.6, 840.0, 63.0, 6.0, 3.0};
//	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4 + 4 * 16);
//	cadet::test::util::repeat(y.data() + 4 + 4 * 16, bindingCell, 16, 4 * 16 / 2);
//	cadet::test::util::populate(y.data() + 4 + 4 * 16 + 16 * 4 * (4 + 4), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4 * 16);
//
//	cadet::test::column::testConsistentInitializationSMABinding("GENERAL_RATE_MODEL", y.data(), 1e-14, 1e-5);
//}

TEST_CASE("GRM consistent sensitivity initialization with linear binding", "[GRM],[ConsistentInit],[Sensitivity],[CI]")
{
	// Fill state vector with given initial values
	const unsigned int numDofs = 4 + 4 * 16 + 16 * 4 * (4 + 4) + 4 * 16;
	std::vector<double> y(numDofs, 0.0);
	std::vector<double> yDot(numDofs, 0.0);
	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, numDofs);
	cadet::test::util::populate(yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, numDofs);

	cadet::test::column::testConsistentInitializationSensitivity("GENERAL_RATE_MODEL", y.data(), yDot.data(), true, 1e-14);
}

TEST_CASE("GRM consistent sensitivity initialization with SMA binding", "[GRM],[ConsistentInit],[Sensitivity],[CI]")
{
	// Fill state vector with given initial values
	const unsigned int numDofs = 4 + 4 * 16 + 16 * 4 * (4 + 4) + 4 * 16;
	std::vector<double> y(numDofs, 0.0);
	std::vector<double> yDot(numDofs, 0.0);

	const double bindingCell[] = {1.0, 1.8, 1.5, 1.6, 840.0, 63.0, 6.0, 3.0};
	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4 + 4 * 16);
	cadet::test::util::repeat(y.data() + 4 + 4 * 16, bindingCell, 8, 4 * 16);
	cadet::test::util::populate(y.data() + 4 + 4 * 16 + 16 * 4 * (4 + 4), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4 * 16);

	cadet::test::util::populate(yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, numDofs);

	cadet::test::column::testConsistentInitializationSensitivity("GENERAL_RATE_MODEL", y.data(), yDot.data(), false, 1e-9);
}

TEST_CASE("GRM inlet DOF Jacobian", "[GRM],[UnitOp],[Jacobian],[Inlet],[CI]")
{
	cadet::test::column::testInletDofJacobian("GENERAL_RATE_MODEL");
}

TEST_CASE("GRM transport Jacobian", "[GRM],[UnitOp],[Jacobian],[CI]")
{
	cadet::JsonParameterProvider jpp = createColumnLinearBenchmark(false, true, "GENERAL_RATE_MODEL");
	cadet::test::column::testJacobianAD(jpp);
}

TEST_CASE("GRM with two component linear binding Jacobian", "[GRM],[UnitOp],[Jacobian],[CI]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding("GENERAL_RATE_MODEL");
	cadet::test::column::testJacobianAD(jpp);
}

TEST_CASE("GRM LWE one vs two identical particle types match", "[GRM],[Simulation],[ParticleType],[CI]")
{
	cadet::test::particle::testOneVsTwoIdenticalParticleTypes("GENERAL_RATE_MODEL", 2e-8, 5e-5);
}

TEST_CASE("GRM LWE separate identical particle types match", "[GRM],[Simulation],[ParticleType],[CI]")
{
	cadet::test::particle::testSeparateIdenticalParticleTypes("GENERAL_RATE_MODEL", 1e-15, 1e-15);
}

TEST_CASE("GRM linear binding single particle matches particle distribution", "[GRM],[Simulation],[ParticleType],[CI]")
{
	cadet::test::particle::testLinearMixedParticleTypes("GENERAL_RATE_MODEL", 5e-8, 5e-5);
}

TEST_CASE("GRM multiple particle types Jacobian analytic vs AD", "[GRM],[Jacobian],[AD],[ParticleType],[CI]")
{
	cadet::test::particle::testJacobianMixedParticleTypes("GENERAL_RATE_MODEL");
}

TEST_CASE("GRM multiple particle types time derivative Jacobian vs FD", "[GRM],[UnitOp],[Residual],[Jacobian],[ParticleType],[CI]")
{
	cadet::test::particle::testTimeDerivativeJacobianMixedParticleTypesFD("GENERAL_RATE_MODEL", 1e-6, 0.0, 9e-4);
}

TEST_CASE("GRM multiple spatially dependent particle types Jacobian analytic vs AD", "[GRM],[Jacobian],[AD],[ParticleType],[CI]")
{
	cadet::test::particle::testJacobianSpatiallyMixedParticleTypes("GENERAL_RATE_MODEL");
}

TEST_CASE("GRM linear binding single particle matches spatially dependent particle distribution", "[GRM],[Simulation],[ParticleType],[CI]")
{
	cadet::test::particle::testLinearSpatiallyMixedParticleTypes("GENERAL_RATE_MODEL", 5e-8, 5e-5);
}

TEST_CASE("GRM multiple spatially dependent particle types flux Jacobian vs FD", "[GRM],[UnitOp],[Residual],[Jacobian],[ParticleType],[CI],[FD]")
{
	cadet::test::particle::testArrowHeadJacobianSpatiallyMixedParticleTypes("GENERAL_RATE_MODEL", 1e-6, 1e-8, 1e-5);
}

TEST_CASE("GRM dynamic reactions Jacobian vs AD bulk", "[GRM],[Jacobian],[AD],[ReactionModel],[CI]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("GENERAL_RATE_MODEL", true, false, false);
}

TEST_CASE("GRM dynamic reactions Jacobian vs AD particle", "[GRM],[Jacobian],[AD],[ReactionModel],[CI]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("GENERAL_RATE_MODEL", false, true, false);
}

TEST_CASE("GRM dynamic reactions Jacobian vs AD modified particle", "[GRM],[Jacobian],[AD],[ReactionModel],[CI]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("GENERAL_RATE_MODEL", false, true, true);
}

TEST_CASE("GRM dynamic reactions Jacobian vs AD bulk and particle", "[GRM],[Jacobian],[AD],[ReactionModel],[CI]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("GENERAL_RATE_MODEL", true, true, false);
}

TEST_CASE("GRM dynamic reactions Jacobian vs AD bulk and modified particle", "[GRM],[Jacobian],[AD],[ReactionModel],[CI]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("GENERAL_RATE_MODEL", true, true, true);
}

TEST_CASE("GRM dynamic reactions time derivative Jacobian vs FD bulk", "[GRM],[Jacobian],[Residual],[ReactionModel],[CI],[FD]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("GENERAL_RATE_MODEL", true, false, false, 1e-6, 1e-14, 9e-4);
}

TEST_CASE("GRM dynamic reactions time derivative Jacobian vs FD particle", "[GRM],[Jacobian],[Residual],[ReactionModel],[CI],[FD]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("GENERAL_RATE_MODEL", false, true, false, 1e-6, 1e-14, 9e-4);
}

TEST_CASE("GRM dynamic reactions time derivative Jacobian vs FD modified particle", "[GRM],[Jacobian],[Residual],[ReactionModel],[CI],[FD]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("GENERAL_RATE_MODEL", false, true, true, 1e-6, 1e-14, 9e-4);
}

TEST_CASE("GRM dynamic reactions time derivative Jacobian vs FD bulk and particle", "[GRM],[Jacobian],[Residual],[ReactionModel],[CI],[FD]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("GENERAL_RATE_MODEL", true, true, false, 1e-6, 1e-14, 9e-4);
}

TEST_CASE("GRM dynamic reactions time derivative Jacobian vs FD bulk and modified particle", "[GRM],[Jacobian],[Residual],[ReactionModel],[CI],[FD]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("GENERAL_RATE_MODEL", true, true, true, 1e-6, 1e-14, 9e-4);
}

inline cadet::JsonParameterProvider createColumnWithTwoCompLinearBindingThreeParticleTypes()
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding("GENERAL_RATE_MODEL");

	const double parVolFrac[] = {0.3, 0.6, 0.1};
	const double parFactor[] = {0.9, 0.8};
	cadet::test::particle::extendModelToManyParticleTypes(jpp, 3, parFactor, parVolFrac);

	return jpp;
}

TEST_CASE("GRM multi particle types dynamic reactions Jacobian vs AD bulk", "[GRM],[Jacobian],[AD],[ReactionModel],[ParticleType],[CI]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, false, false);
}

TEST_CASE("GRM multi particle types dynamic reactions Jacobian vs AD particle", "[GRM],[Jacobian],[AD],[ReactionModel],[ParticleType],[CI]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, false, true, false);
}

TEST_CASE("GRM multi particle types dynamic reactions Jacobian vs AD modified particle", "[GRM],[Jacobian],[AD],[ReactionModel],[ParticleType],[CI]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, false, true, true);
}

TEST_CASE("GRM multi particle types dynamic reactions Jacobian vs AD bulk and particle", "[GRM],[Jacobian],[AD],[ReactionModel],[ParticleType],[CI]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, true, false);
}

TEST_CASE("GRM multi particle types dynamic reactions Jacobian vs AD bulk and modified particle", "[GRM],[Jacobian],[AD],[ReactionModel],[ParticleType],[CI]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, true, true);
}

TEST_CASE("GRM multi particle types dynamic reactions time derivative Jacobian vs FD bulk", "[GRM],[Jacobian],[Residual],[ReactionModel],[ParticleType],[CI],[FD]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, false, false, 1e-6, 1e-14, 9e-4);
}

TEST_CASE("GRM multi particle types dynamic reactions time derivative Jacobian vs FD particle", "[GRM],[Jacobian],[Residual],[ReactionModel],[ParticleType],[CI],[FD]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, false, true, false, 1e-6, 1e-14, 9e-4);
}

TEST_CASE("GRM multi particle types dynamic reactions time derivative Jacobian vs FD modified particle", "[GRM],[Jacobian],[Residual],[ReactionModel],[ParticleType],[CI],[FD]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, false, true, true, 1e-6, 1e-14, 9e-4);
}

TEST_CASE("GRM multi particle types dynamic reactions time derivative Jacobian vs FD bulk and particle", "[GRM],[Jacobian],[Residual],[ReactionModel],[ParticleType],[CI],[FD]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, true, false, 1e-6, 1e-14, 9e-4);
}

TEST_CASE("GRM multi particle types dynamic reactions time derivative Jacobian vs FD bulk and modified particle", "[GRM],[Jacobian],[Residual],[ReactionModel],[ParticleType],[CI],[FD]")
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBindingThreeParticleTypes();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, true, true, 1e-6, 1e-14, 9e-4);
}
