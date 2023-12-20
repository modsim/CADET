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
#include "ParticleHelper.hpp"
#include "ReactionModelTests.hpp"
#include "JsonTestModels.hpp"
#include "Utils.hpp"

TEST_CASE("LRMP_DG LWE forward vs backward flow", "[LRMP],[DG],[Simulation]")
{
	cadet::test::column::DGparams disc;

	// Test all integration modes
	for (int i = 0; i <= 1; i++)
	{
		disc.setIntegrationMode(i);
		cadet::test::column::testForwardBackward("LUMPED_RATE_MODEL_WITH_PORES", disc, 6e-8, 4e-6);
	}
}

TEST_CASE("LRMP_DG linear pulse vs analytic solution", "[LRMP],[DG],[Simulation],[Analytic]")
{
	cadet::test::column::DGparams disc;
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-pulseBenchmark.data", true, true, disc, 6e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-pulseBenchmark.data", true, false, disc, 6e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-pulseBenchmark.data", false, true, disc, 6e-5, 1e-7);
	cadet::test::column::testAnalyticBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-pulseBenchmark.data", false, false, disc, 6e-5, 1e-7);
}

TEST_CASE("LRMP_DG non-binding linear pulse vs analytic solution", "[LRMP],[DG],[Simulation],[Analytic],[NonBinding]")
{
	cadet::test::column::DGparams disc;
	cadet::test::column::testAnalyticNonBindingBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-nonBinding.data", true, disc, 6e-5, 1e-7);
	cadet::test::column::testAnalyticNonBindingBenchmark("LUMPED_RATE_MODEL_WITH_PORES", "/data/lrmp-nonBinding.data", false, disc, 6e-5, 1e-7);
}
