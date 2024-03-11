// =============================================================================
//  CADET
//  
//  Copyright © 2008-2023: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include <catch.hpp>

#include "ColumnTests.hpp"
#include "ReactionModelTests.hpp"
#include "Utils.hpp"

TEST_CASE("LRM_DG LWE forward vs backward flow", "[LRM],[DG],[Simulation]")
{
	cadet::test::column::DGparams disc;

	// Test all integration modes
	for (int i = 0; i <= 1; i++)
	{
		disc.setIntegrationMode(i);
		cadet::test::column::testForwardBackward("LUMPED_RATE_MODEL_WITHOUT_PORES", disc, 6e-9, 6e-2);
	}
}

TEST_CASE("LRM_DG linear pulse vs analytic solution", "[LRM],[DG],[Simulation],[Analytic]")
{
	cadet::test::column::DGparams disc;
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITHOUT_PORES", "/data/lrm-pulseBenchmark.data", true, true, disc, 2e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITHOUT_PORES", "/data/lrm-pulseBenchmark.data", true, false, disc, 2e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITHOUT_PORES", "/data/lrm-pulseBenchmark.data", false, true, disc, 2e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITHOUT_PORES", "/data/lrm-pulseBenchmark.data", false, false, disc, 2e-5, 1e-7);
}

TEST_CASE("LRM_DG non-binding linear pulse vs analytic solution", "[LRM],[DG],[Simulation],[Analytic],[NonBinding]")
{
	cadet::test::column::DGparams disc;
	cadet::test::column::testAnalyticNonBindingBenchmark("LUMPED_RATE_MODEL_WITHOUT_PORES", "/data/lrm-nonBinding.data", true, disc, 2e-5, 1e-7);
	cadet::test::column::testAnalyticNonBindingBenchmark("LUMPED_RATE_MODEL_WITHOUT_PORES", "/data/lrm-nonBinding.data", false, disc, 2e-5, 1e-7);
}

//// todo FIX! (AD)
//TEST_CASE("LRM_DG Jacobian forward vs backward flow", "[LRM],[DG],[UnitOp],[Residual],[Jacobian],[AD]")
//{
//	cadet::test::column::DGparams disc;
//
//	// Test all integration modes
//	for (int i = 0; i <= 1; i++)
//	{
//		disc.setIntegrationMode(i);
//		cadet::test::column::testJacobianForwardBackward("LUMPED_RATE_MODEL_WITHOUT_PORES", disc);
//	}
//}

TEST_CASE("LRM_DG time derivative Jacobian vs FD", "[LRM],[DG],[UnitOp],[Residual],[Jacobian]")
{
	cadet::test::column::testTimeDerivativeJacobianFD("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG");
}

//// todo FIX! (AD)
//TEST_CASE("LRM_DG sensitivity Jacobians", "[LRM],[DG],[UnitOp],[Sensitivity]")
//{
//	cadet::test::column::testFwdSensJacobians("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", 1e-4, 3e-7, 5e-5);
//}

//// todo FIX! (Also fails for for FV. DG error seems to be different)
//TEST_CASE("LRM_DG forward sensitivity vs FD", "[LRM],[DG],[Sensitivity],[Simulation]")
//{
//	// Relative error is checked first, we use high absolute error for letting
//	// some points that are far off pass the error test, too. This is required
//	// due to errors in finite differences.
//	const double fdStepSize[] = { 5e-3, 5e-3, 5e-3, 1e-3 };
//	const double absTols[] = { 2e8, 8e-3, 2e-2, 3e-1 };
//	const double relTols[] = { 1e-1, 5e-1, 5e-2, 1e-2 };
//	const double passRatio[] = { 0.88, 0.84, 0.73, 0.87 };
//	cadet::test::column::testFwdSensSolutionFD("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", false, fdStepSize, absTols, relTols, passRatio);
//}

//// todo FIX! (Also fails for for FV)
//TEST_CASE("LRM_DG forward sensitivity forward vs backward flow", "[LRM],[DG],[Sensitivity],[Simulation]")
//{
//	const double absTols[] = { 500.0, 8e-7, 9e-7, 2e-3 };
//	const double relTols[] = { 7e-3, 5e-5, 5e-5, 9e-4 };
//	const double passRatio[] = { 0.99, 0.97, 0.97, 0.98 };
//	cadet::test::column::testFwdSensSolutionForwardBackward("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", absTols, relTols, passRatio);
//}

//// todo FIX! (Also fails for for FV)
//TEST_CASE("LRM_DG consistent initialization with linear binding", "[LRM],[DG],[ConsistentInit]")
//{
//	cadet::test::column::testConsistentInitializationLinearBinding("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", 1e-12, 1e-14);
//}

//// todo FIX! (Also fails for for FV)
//TEST_CASE("LRM_DG consistent initialization with SMA binding", "[LRM],[DG],[ConsistentInit]")
//{
//	std::vector<double> y(4 + 16 * (4 + 4), 0.0);
//	// Optimal values:
//	//	const double bindingCell[] = {1.2, 2.0, 1.0, 1.5, 858.034, 66.7896, 3.53273, 2.53153, 
//	//		1.0, 1.8, 1.5, 1.6, 856.173, 64.457, 5.73227, 2.85286};
//	const double bindingCell[] = { 1.2, 2.0, 1.0, 1.5, 840.0, 63.0, 3.0, 3.0,
//		1.0, 1.8, 1.5, 1.6, 840.0, 63.0, 6.0, 3.0 };
//	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4);
//	cadet::test::util::repeat(y.data() + 4, bindingCell, 16, 8);
//
//	cadet::test::column::testConsistentInitializationSMABinding("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", y.data(), 1e-14, 1e-5);
//}

//// todo FIX! (works for for FV. DG has NAN errors)
//TEST_CASE("LRM_DG consistent sensitivity initialization with linear binding", "[LRM],[DG],[ConsistentInit],[Sensitivity]")
//{
//	// Fill state vector with given initial values
//	const unsigned int numDofs = 4 + 16 * (4 + 4);
//	std::vector<double> y(numDofs, 0.0);
//	std::vector<double> yDot(numDofs, 0.0);
//	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, numDofs);
//	cadet::test::util::populate(yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, numDofs);
//
//	cadet::test::column::testConsistentInitializationSensitivity("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", y.data(), yDot.data(), true, 1e-14);
//}

//// todo FIX! (Also fails for for FV)
//TEST_CASE("LRM_DG consistent sensitivity initialization with SMA binding", "[LRM],[DG],[ConsistentInit],[Sensitivity],[TestHere2]")
//{
//	// Fill state vector with given initial values
//	const unsigned int numDofs = 4 + 16 * (4 + 4);
//	std::vector<double> y(numDofs, 0.0);
//	std::vector<double> yDot(numDofs, 0.0);
//
//	const double bindingCell[] = { 1.0, 1.8, 1.5, 1.6, 840.0, 63.0, 6.0, 3.0 };
//	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4);
//	cadet::test::util::repeat(y.data() + 4, bindingCell, 8, 16);
//
//	cadet::test::util::populate(yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, numDofs);
//
//	cadet::test::column::testConsistentInitializationSensitivity("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", y.data(), yDot.data(), false, 1e-10);
//}

TEST_CASE("LRM_DG inlet DOF Jacobian", "[LRM],[DG],[UnitOp],[Jacobian],[Inlet]")
{
	cadet::test::column::testInletDofJacobian("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG");
}

//// todo FIX! (works for for FV. DG raises NAN errors, potentially AD issue!)
//TEST_CASE("LRM_DG dynamic reactions Jacobian vs AD bulk", "[LRM],[DG],[Jacobian],[AD],[ReactionModel],[TestHere1]")
//{
//	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", true, false, false);
//}
//
//// todo FIX! (works for for FV. DG raises NAN errors, potentially AD issue!)
//TEST_CASE("LRM_DG dynamic reactions Jacobian vs AD modified bulk", "[LRM],[DG],[Jacobian],[AD],[ReactionModel],[TestHere2]")
//{
//	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", true, false, true);
//}

TEST_CASE("LRM_DG dynamic reactions time derivative Jacobian vs FD bulk", "[LRM],[DG],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", true, false, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("LRM_DG dynamic reactions time derivative Jacobian vs FD modified bulk", "[LRM],[DG],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("LUMPED_RATE_MODEL_WITHOUT_PORES", "DG", true, false, true, 1e-6, 1e-14, 8e-4);
}