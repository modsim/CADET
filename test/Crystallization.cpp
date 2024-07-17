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

#include <catch.hpp>

#include "ColumnTests.hpp"
#include "ReactionModelTests.hpp"
#include "Weno.hpp"
#include "Utils.hpp"
#include "JsonTestModels.hpp"

/**
 * @brief Returns the absolute path to the test/ folder of the project
 * @details Absolute path to the test/ folder of the project without trailing slash
 * @return Absolute path to the test/ folder
 */
const char* getTestDirectory();

TEST_CASE("Crystallization in a CSTR with initial distribution and growth", "[testHere],[Crystallization],[Simulation],[Reference],[CI]")
{
	const std::string& modelFilePath = std::string("/data/configuration_PBM_CSTR_growth_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_PBM_CSTR_growth_benchmark1.h5");
	const std::vector<double> absTol = { 1e-10 };
	const std::vector<double> relTol = { 1e-10 };

	cadet::test::column::Dummyparams disc; // CSTR, so no spatial resolution
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, true);
}

TEST_CASE("Crystallization in a CSTR with initial distribution and size-dependent growth", "[Crystallization],[Simulation],[Reference],[CI]")
{
	const std::string& modelFilePath = std::string("/data/configuration_PBM_CSTR_growthSizeDep_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_PBM_CSTR_growthSizeDep_benchmark1.h5");
	const std::vector<double> absTol = { 1e-10 };
	const std::vector<double> relTol = { 1e-10 };

	cadet::test::column::Dummyparams disc; // CSTR, so no spatial resolution
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, true);
}

TEST_CASE("Crystallization in a CSTR with primary nucleation and growth", "[Crystallization],[Simulation],[Reference],[CI]")
{
	const std::string& modelFilePath = std::string("/data/configuration_PBM_CSTR_primaryNucleationAndGrowth_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_PBM_CSTR_primaryNucleationAndGrowth_benchmark1.h5");
	const std::vector<double> absTol = { 1e-10 };
	const std::vector<double> relTol = { 1e-10 };

	cadet::test::column::Dummyparams disc; // CSTR, so no spatial resolution
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, true);
}

TEST_CASE("Crystallization in a CSTR with primary nucleation, growth and growth rate dispersion", "[Crystallization],[Simulation],[Reference],[CI]")
{
	const std::string& modelFilePath = std::string("/data/configuration_PBM_CSTR_primaryNucleationGrowthGrowthRateDispersion_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_PBM_CSTR_primaryNucleationGrowthGrowthRateDispersion_benchmark1.h5");
	const std::vector<double> absTol = { 1e-10 };
	const std::vector<double> relTol = { 1e-10 };

	cadet::test::column::Dummyparams disc; // CSTR, so no spatial resolution
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, true);
}

TEST_CASE("Crystallization in a CSTR with primary and secondary nucleation and growth", "[Crystallization],[Simulation],[Reference],[CI]")
{
	const std::string& modelFilePath = std::string("/data/configuration_PBM_CSTR_primarySecondaryNucleationAndGrowth_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_PBM_CSTR_primarySecondaryNucleationAndGrowth_benchmark1.h5");
	const std::vector<double> absTol = { 1e-10 };
	const std::vector<double> relTol = { 1e-10 };

	cadet::test::column::Dummyparams disc; // CSTR, so no spatial resolution
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, true);
}

TEST_CASE("Crystallization in a DPFR/LRM with primary and secondary nucleation and growth", "[Crystallization],[Simulation],[Reference],[CI]")
{
	const std::string& modelFilePath = std::string("/data/configuration_PBM_DPFR_primarySecondaryNucleationGrowth_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_PBM_DPFR_primarySecondaryNucleationGrowth_benchmark1.h5");
	const std::vector<double> absTol = { 7e+6 };
	const std::vector<double> relTol = { 1e-10 };

	cadet::test::column::FVparams disc(25);
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, true);
}

// Note: two entries fail, given in (row,col): (204, 190), (204, 191)
TEST_CASE("Crystallization Jacobian verification for a CSTR with initial distribution and growth", "[CrysToFix1]") // "[Crystallization],[UnitOp],[Jacobian]")
{
	// read json model setup file
	const std::string& modelFileRelPath = std::string("/data/configuration_PBM_CSTR_growth_benchmark1.json");
	const std::string setupFile = std::string(getTestDirectory()) + modelFileRelPath;
	cadet::JsonParameterProvider pp_setup(cadet::JsonParameterProvider::fromFile(setupFile));

	pp_setup.pushScope("model");
	pp_setup.pushScope("unit_001");

	cadet::test::column::testJacobianAD(pp_setup, std::numeric_limits<float>::epsilon() * 100.0);
}

// Note: two entries fail, given in (row,col): (204, 190), (204, 191)
TEST_CASE("Crystallization Jacobian verification for a CSTR with initial distribution and size-dependent growth", "[CrysToFix2]") // "[Crystallization],[UnitOp],[Jacobian]")
{
	// read json model setup file
	const std::string& modelFileRelPath = std::string("/data/configuration_PBM_CSTR_growthSizeDep_benchmark1.json");
	const std::string setupFile = std::string(getTestDirectory()) + modelFileRelPath;
	cadet::JsonParameterProvider pp_setup(cadet::JsonParameterProvider::fromFile(setupFile));

	pp_setup.pushScope("model");
	pp_setup.pushScope("unit_001");

	cadet::test::column::testJacobianAD(pp_setup, std::numeric_limits<float>::epsilon() * 100.0);
}

// Note: two entries fail, given in (row,col): (204, 190), (204, 191)
TEST_CASE("Crystallization Jacobian verification for a CSTR with primary nucleation and growth", "[CrysToFix3]") // "[Crystallization],[UnitOp],[Jacobian]")
{
	// read json model setup file
	const std::string& modelFileRelPath = std::string("/data/configuration_PBM_CSTR_primaryNucleationAndGrowth_benchmark1.json");
	const std::string setupFile = std::string(getTestDirectory()) + modelFileRelPath;
	cadet::JsonParameterProvider pp_setup(cadet::JsonParameterProvider::fromFile(setupFile));

	pp_setup.pushScope("model");
	pp_setup.pushScope("unit_001");

	cadet::test::column::testJacobianAD(pp_setup, std::numeric_limits<float>::epsilon() * 100.0);
}

// Some more comparisons fail
TEST_CASE("Crystallization Jacobian verification for a CSTR with primary nucleation, growth and growth rate dispersion", "[CrysToFix4]") // "[Crystallization],[UnitOp],[Jacobian]")
{
	// read json model setup file
	const std::string& modelFileRelPath = std::string("/data/configuration_PBM_CSTR_primaryNucleationGrowthGrowthRateDispersion_benchmark1.json");
	const std::string setupFile = std::string(getTestDirectory()) + modelFileRelPath;
	cadet::JsonParameterProvider pp_setup(cadet::JsonParameterProvider::fromFile(setupFile));

	pp_setup.pushScope("model");
	pp_setup.pushScope("unit_001");

	cadet::test::column::testJacobianAD(pp_setup, std::numeric_limits<float>::epsilon() * 100.0);
}

// Note: two entries fail, given in (row,col): (204, 190), (204, 191)
TEST_CASE("Crystallization Jacobian verification for a CSTR with primary and secondary nucleation and growth", "[CrysToFix5]") // "[Crystallization],[UnitOp],[Jacobian]")
{
	// read json model setup file
	const std::string& modelFileRelPath = std::string("/data/configuration_PBM_CSTR_primarySecondaryNucleationAndGrowth_benchmark1.json");
	const std::string setupFile = std::string(getTestDirectory()) + modelFileRelPath;
	cadet::JsonParameterProvider pp_setup(cadet::JsonParameterProvider::fromFile(setupFile));

	pp_setup.pushScope("model");
	pp_setup.pushScope("unit_001");

	cadet::test::column::testJacobianAD(pp_setup, std::numeric_limits<float>::epsilon() * 100.0);
}

// FD pattern tests fail due to non-equal entries
// AD tests fail: access violation
TEST_CASE("Crystallization Jacobian verification for a DPFR/LRM with primary and secondary nucleation and growth", "[CrysToFix6]") // "[Crystallization],[UnitOp],[Jacobian]")
{
	// read json model setup file
	const std::string& modelFileRelPath = std::string("/data/configuration_PBM_DPFR_primarySecondaryNucleationGrowth_benchmark1.json");
	const std::string setupFile = std::string(getTestDirectory()) + modelFileRelPath;
	cadet::JsonParameterProvider pp_setup(cadet::JsonParameterProvider::fromFile(setupFile));

	pp_setup.pushScope("model");
	pp_setup.pushScope("unit_001");

	cadet::test::column::testJacobianAD(pp_setup, std::numeric_limits<float>::epsilon() * 100.0);
}
