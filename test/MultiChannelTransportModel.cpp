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
#include <json.hpp>

#include "Approx.hpp"
#include "Logging.hpp"
#include "common/Driver.hpp"
#include "common/JsonParameterProvider.hpp"
#include "ColumnTests.hpp"
#include "ReactionModelTests.hpp"
#include "Weno.hpp"
#include "Utils.hpp"
#include "JacobianHelper.hpp"
#include "cadet/ModelBuilder.hpp"
#include "ModelBuilderImpl.hpp"
#include "cadet/FactoryFuncs.hpp"
#include "UnitOperationTests.hpp"

/**
 * @brief Returns the absolute path to the test/ folder of the project
 * @details Absolute path to the test/ folder of the project without trailing slash
 * @return Absolute path to the test/ folder
 */
const char* getTestDirectory();

namespace
{
	using json = nlohmann::json;

	json createMCTModelJson(const std::vector<double>& crossSectionAreas, const std::vector<double>& velocity, const std::vector<double>& exchangeMatrix, const double colDisp = 5.75e-8, const double colLength = 200, const int nCol = 16)
	{
		json config;
		config["UNIT_TYPE"] = "MULTI_CHANNEL_TRANSPORT";
		config["NCOMP"] = 1;
		config["COL_DISPERSION"] = colDisp;
		config["EXCHANGE_MATRIX"] = exchangeMatrix;
		config["NCHANNEL"] = crossSectionAreas.size();

		// Geometry
		config["COL_LENGTH"] = colLength;
		config["CHANNEL_CROSS_SECTION_AREAS"] = crossSectionAreas;
		config["VELOCITY"] = velocity;

		// Initial conditions
		config["INIT_C"] = {0.0};

		// Discretization
		{
			json disc;

			disc["NCOL"] = nCol;

			disc["USE_ANALYTIC_JACOBIAN"] = true;

			// WENO
			{
				json weno;

				weno["WENO_ORDER"] = 3;
				weno["BOUNDARY_MODEL"] = 0;
				weno["WENO_EPS"] = 1e-10;
				disc["weno"] = weno;
			}
			config["discretization"] = disc;
		}

		return config;
	}

	json createMCTJson(const std::vector<double>& volFlowRate, const std::vector<double>& velocity, const std::vector<double>& concentrationIn, const std::vector<double>& crossSectionAreas, const std::vector<double>& exchangeMatrix, const double colDisp = 5.75e-8)
	{
		json config;
		// Model
		{
			json model;
			model["NUNITS"] = 2;
			model["unit_000"] = createMCTModelJson(crossSectionAreas, velocity, exchangeMatrix, colDisp);

			// Inlet - unit 001 ... unitXXX
			for (unsigned int i = 0; i < volFlowRate.size(); ++i)
			{
				json inlet;

				inlet["UNIT_TYPE"] = std::string("INLET");
				inlet["INLET_TYPE"] = std::string("PIECEWISE_CUBIC_POLY");
				inlet["NCOMP"] = 1;

				{
					json sec;

					sec["CONST_COEFF"] = {concentrationIn[i]};
					sec["LIN_COEFF"] = {0.0};
					sec["QUAD_COEFF"] = {0.0};
					sec["CUBE_COEFF"] = {0.0};

					inlet["sec_000"] = sec;
				}

				{
					json sec;

					sec["CONST_COEFF"] = {0.0};
					sec["LIN_COEFF"] = {0.0};
					sec["QUAD_COEFF"] = {0.0};
					sec["CUBE_COEFF"] = {0.0};

					inlet["sec_001"] = sec;
				}

				model[std::string("unit_00") + std::to_string(i+1)] = inlet;
			}

			// Valve switches
			{
				json con;
				con["NSWITCHES"] = 1;
				con["CONNECTIONS_INCLUDE_PORTS"] = true;

				{
					json sw;

					// This switch occurs at beginning of section 0 (initial configuration)
					sw["SECTION"] = 0;

					// Connection list is 3x7 since we have 1 connection between
					// the two unit operations with 3 ports (and we need to have 7 columns)
					std::vector<double> conn(volFlowRate.size() * 7, 0.0);
					for (unsigned int i = 0; i < volFlowRate.size(); ++i)
					{
						conn[i * 7 + 0] = i + 1;
						conn[i * 7 + 1] = 0.0;
						conn[i * 7 + 2] = 0.0;
						conn[i * 7 + 3] = i;
						conn[i * 7 + 4] = -1.0;
						conn[i * 7 + 5] = -1.0;
						conn[i * 7 + 6] = volFlowRate[i];
					}

					sw["CONNECTIONS"] = conn;
					// Connections: From unit operation,
					//              to unit operation,
					//              from port,
					//              to port,
					//              connect all components -1 (i.e., all components),
					//              to all components -1 (i.e., all components),
					//              volumetric flow rate

					con["switch_000"] = sw;
				}
				model["connections"] = con;
			}

			// Solver settings
			{
				json solver;

				solver["MAX_KRYLOV"] = 0;
				solver["GS_TYPE"] = 1;
				solver["MAX_RESTARTS"] = 10;
				solver["SCHUR_SAFETY"] = 1e-8;
				model["solver"] = solver;
			}

			config["model"] = model;
		}

		// Return
		{
			json ret;
			ret["WRITE_SOLUTION_TIMES"] = true;

			json mct;
			mct["WRITE_SOLUTION_BULK"] = true;
			mct["WRITE_SOLUTION_PARTICLE"] = false;
			mct["WRITE_SOLUTION_FLUX"] = false;
			mct["WRITE_SOLUTION_INLET"] = true;
			mct["WRITE_SOLUTION_OUTLET"] = true;

			ret["unit_000"] = mct;
			config["return"] = ret;
		}

		// Solver
		{
			json solver;

			{
				std::vector<double> solTimes;

				solTimes.reserve(1501);
				for (double t = 0.0; t <= 1500.0; t += 1.0)
					solTimes.push_back(t);

				solver["USER_SOLUTION_TIMES"] = solTimes;
			}

			solver["NTHREADS"] = 1;

			// Sections
			{
				json sec;

				sec["NSEC"] = 2;
				sec["SECTION_TIMES"] = {0.0, 10.0, 1500.0};
				sec["SECTION_CONTINUITY"] = std::vector<bool>{false};

				solver["sections"] = sec;
			}

			// Time integrator
			{
				json ti;

				ti["ABSTOL"] = 1e-8;
				ti["RELTOL"] = 1e-6;
				ti["ALGTOL"] = 1e-12;
				ti["INIT_STEP_SIZE"] = 1e-6;
				ti["MAX_STEPS"] = 10000;
				ti["MAX_STEP_SIZE"] = 0.0;
				ti["RELTOL_SENS"] = 1e-6;
				ti["ERRORTEST_SENS"] = true;
				ti["MAX_NEWTON_ITER"] = 3;
				ti["MAX_ERRTEST_FAIL"] = 7;
				ti["MAX_CONVTEST_FAIL"] = 10;
				ti["MAX_NEWTON_ITER_SENS"] = 3;
				ti["CONSISTENT_INIT_MODE"] = 1;
				ti["CONSISTENT_INIT_MODE_SENS"] = 1;

				solver["time_integrator"] = ti;
			}

			config["solver"] = solver;
		}
		return config;
	}

	cadet::JsonParameterProvider createMCT(const std::vector<double>& volFlowRate, const std::vector<double>& velocity, const std::vector<double>& concentrationIn, const std::vector<double>& crossSectionAreas, const std::vector<double>& exchangeMatrix, const double colDisp = 5.75e-8)
	{
		return cadet::JsonParameterProvider(createMCTJson(volFlowRate, velocity, concentrationIn, crossSectionAreas, exchangeMatrix, colDisp));
	}

	/**
	 * @brief Creates a runnable column model with given WENO order
	 * @details Creates a column model and configures it using the given IParameterProvider @p jpp.
	 * @param [in] uoType Unit operation type
	 * @param [in] mb ModelBuilder
	 * @param [in] jpp Configuration of the model
	 * @param [in] wenoOrder WENO order
	 * @return Runnable column model
	 */
	inline cadet::IUnitOperation* createAndConfigureUnit(cadet::IModelBuilder& mb, cadet::JsonParameterProvider& jpp, int wenoOrder)
	{
		// Create a unit
		cadet::IModel* const iUnit = mb.createUnitOperation(jpp, 0);
		REQUIRE(nullptr != iUnit);

		cadet::IUnitOperation* const unit = reinterpret_cast<cadet::IUnitOperation*>(iUnit);

		// Set WENO order
		cadet::test::column::setWenoOrder(jpp, wenoOrder);

		// Configure
		cadet::ModelBuilder& temp = *reinterpret_cast<cadet::ModelBuilder*>(&mb);
		REQUIRE(unit->configureModelDiscretization(jpp, temp));
		REQUIRE(unit->configure(jpp));

		// Do some checks
		const unsigned int nComp = jpp.getInt("NCOMP");
		REQUIRE(unit->numComponents() == nComp);

		return unit;
	}
}

TEST_CASE("MCT with two channels and without exchange yields same result on both ports", "[MCT],[Simulation],[CI]")
{
	const double relTol = 1e-10;
	const double absTol = 1e-14;

	// Setup simulation
	cadet::JsonParameterProvider jpp = createMCT({ 1.0, 1.0 }, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {0.0, 0.0, 0.0, 0.0});

	// Run simulation
	cadet::Driver drv;
	drv.configure(jpp);
	drv.run();

	// Get data from simulation
	cadet::InternalStorageUnitOpRecorder const* const fwdData = drv.solution()->unitOperation(0);

	const unsigned int nComp = fwdData->numComponents();
	const unsigned int nPorts = fwdData->numInletPorts();

	CHECK(nComp == 1);
	CHECK(nPorts == 2);

	const unsigned int nDataPoints = fwdData->numDataPoints() * nComp * nPorts;
	double const* const time = drv.solution()->time();
	double const* outlet = fwdData->outlet();
	for (int i = 0; i < fwdData->numDataPoints(); ++i, outlet += 2)
	{
		INFO("Time " << time[i] << " time point idx " << i);
		CHECK(outlet[1] == cadet::test::makeApprox(outlet[0], relTol, absTol));
	}
}

TEST_CASE("Two MCTs with two channels and forward/backward exchange yield same output in opposite ports", "[MCT],[Simulation],[CI]")
{
	const double relTol = 1e-6;
	const double absTol = 1e-10;

	// Setup forward exchange simulation
	cadet::JsonParameterProvider jppFwdEx = createMCT({ 1.0, 1.0 }, { 1.0, 1.0 }, { 1.0, 0.2 }, { 1.0, 1.0 }, { 0.0, 0.01, 0.0, 0.0 });

	// Run simulation
	cadet::Driver drvFwdEx;
	drvFwdEx.configure(jppFwdEx);
	drvFwdEx.run();

	// Get data from simulation
	cadet::InternalStorageUnitOpRecorder const* const FwdExData = drvFwdEx.solution()->unitOperation(0);
	
	// Setup backward exchange simulation
	cadet::JsonParameterProvider jppBwdEx = createMCT({ 1.0, 1.0 }, { 1.0, 1.0 }, { 0.2, 1.0 }, { 1.0, 1.0 }, { 0.0, 0.0, 0.01, 0.0 });

	// Run simulation
	cadet::Driver drvBwdEx;
	drvBwdEx.configure(jppBwdEx);
	drvBwdEx.run();

	// Get data from simulation
	cadet::InternalStorageUnitOpRecorder const* const BwdExData = drvBwdEx.solution()->unitOperation(0);

	const unsigned int nComp = FwdExData->numComponents();
	const unsigned int nPorts = FwdExData->numInletPorts();

	CHECK(nComp == 1);
	CHECK(nPorts == 2);

	const unsigned int nDataPoints = FwdExData->numDataPoints() * nComp * nPorts;
	double const* const time = drvFwdEx.solution()->time();
	double const* FwdOutlet = FwdExData->outlet();
	double const* BwdOutlet = BwdExData->outlet();

	for (int i = 0; i < FwdExData->numDataPoints(); ++i, FwdOutlet += 2, BwdOutlet += 2)
	{
		INFO("Time " << time[i] << " time point idx " << i);
		CHECK(FwdOutlet[0] == cadet::test::makeApprox(BwdOutlet[1], relTol, absTol));
		CHECK(FwdOutlet[1] == cadet::test::makeApprox(BwdOutlet[0], relTol, absTol));
	}
}

TEST_CASE("MCT with two channels and forward/backward flow yields same result at both ports", "[MCT],[Simulation],[CI]")
{
	const double absTol = 6e-9;
	const double relTol = 6e-4;

	// Setup forward exchange simulation
	cadet::JsonParameterProvider jppMixFlow = createMCT({ 1.0, 1.0 }, { 1.0, -1.0 }, { 1.0, 1.0 }, { 1.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 });

	// Run simulation
	cadet::Driver drvMixFlow;
	drvMixFlow.configure(jppMixFlow);
	drvMixFlow.run();

	// Get data from simulation
	cadet::InternalStorageUnitOpRecorder const* const MixFlowData = drvMixFlow.solution()->unitOperation(0);

	const unsigned int nComp = MixFlowData->numComponents();
	const unsigned int nPorts = MixFlowData->numInletPorts();

	CHECK(nComp == 1);
	CHECK(nPorts == 2);

	const unsigned int nDataPoints = MixFlowData->numDataPoints() * nComp * nPorts;
	double const* const time = drvMixFlow.solution()->time();
	double const* MixFlowOutlet = MixFlowData->outlet();

	for (int i = 0; i < MixFlowData->numDataPoints(); ++i, MixFlowOutlet += 2)
	{
		INFO("Time " << time[i] << " time point idx " << i);
		CHECK(MixFlowOutlet[0] == cadet::test::makeApprox(MixFlowOutlet[1], relTol, absTol));
	}
}

TEST_CASE("Two MCT's with forward/backward flow yield same result", "[MCT],[Simulation],[CI]")
{
	const double relTol = 1e-6;
	const double absTol = 1e-10;

	// Setup forward exchange simulation
	cadet::JsonParameterProvider jppFwdFlow = createMCT({ 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 0.0 });

	// Run simulation
	cadet::Driver drvFwdFlow;
	drvFwdFlow.configure(jppFwdFlow);
	drvFwdFlow.run();

	// Get data from simulation
	cadet::InternalStorageUnitOpRecorder const* const FwdFlowData = drvFwdFlow.solution()->unitOperation(0);

	// Setup backward exchange simulation
	cadet::JsonParameterProvider jppBwdFlow = createMCT({ 1.0 }, { -1.0 }, { 1.0 }, { 1.0 }, { 0.0 });

	// Run simulation
	cadet::Driver drvBwdFlow;
	drvBwdFlow.configure(jppBwdFlow);
	drvBwdFlow.run();

	// Get data from simulation
	cadet::InternalStorageUnitOpRecorder const* const BwdFlowData = drvBwdFlow.solution()->unitOperation(0);

	const unsigned int nComp = FwdFlowData->numComponents();
	const unsigned int nPorts = FwdFlowData->numInletPorts();

	CHECK(nComp == 1);
	CHECK(nPorts == 1);

	const unsigned int nDataPoints = FwdFlowData->numDataPoints() * nComp * nPorts;
	double const* const time = drvFwdFlow.solution()->time();
	double const* FwdFlowOutlet = FwdFlowData->outlet();
	double const* BwdFlowOutlet = BwdFlowData->outlet();

	double const* BwdFlowBulk = BwdFlowData->bulk();
	double const* FwdFlowBulk = FwdFlowData->bulk();

	for (int i = 0; i < FwdFlowData->numDataPoints(); ++i, ++FwdFlowOutlet, ++BwdFlowOutlet)
	{
		INFO("Time " << time[i] << " time point idx " << i);
		CHECK(FwdFlowOutlet[0] == cadet::test::makeApprox(BwdFlowOutlet[0], relTol, absTol));
	}
}

TEST_CASE("MCT inlet DOF Jacobian", "[MCT],[UnitOp],[Jacobian],[Inlet],[CI]")
{
	cadet::test::column::testInletDofJacobian("MULTI_CHANNEL_TRANSPORT", "FV");
}

TEST_CASE("MCT numerical Benchmark for 1 channel no exchange, no reaction case", "[MCT],[Simulation],[Reference],[mctReference]") // todo CI flag: currently only runs locally but fails on server
{
	const std::string& modelFilePath = std::string("/data/model_MCT1ch_noEx_noReac_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_MCT1ch_noEx_noReac_benchmark1_FV_Z256.h5");
	const std::vector<double> absTol = { RelApprox::defaultEpsilon() };
	const std::vector<double> relTol = { RelApprox::defaultMargin() };
	cadet::test::column::FVparams disc(256);
	disc.setNRad(1); // will be used as NCHANNEL
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, false);
}

TEST_CASE("MCT numerical Benchmark comparison with LRM (1 channel no exchange, no reaction case)", "[MCT],[Simulation],[Reference],[mctReference]") // todo CI flag: currently only runs locally but fails on server
{
	const std::string& modelFilePath = std::string("/data/model_LRM_noBnd_1comp_MCTbenchmark.json");
	const std::string& refFilePath = std::string("/data/ref_MCT1ch_noEx_noReac_benchmark1_FV_Z256.h5");
	const std::vector<double> absTol = { RelApprox::defaultEpsilon() };
	const std::vector<double> relTol = { RelApprox::defaultMargin() };
	cadet::test::column::FVparams disc(256);
	disc.setNRad(1); // will be used as NCHANNEL
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, false);
}

TEST_CASE("MCT numerical Benchmark for 1 channel no exchange, with reaction case", "[MCT],[Simulation],[Reference],[mctReference]") // todo CI flag: currently only runs locally but fails on server
{
	const std::string& modelFilePath = std::string("/data/model_MCT1ch_noEx_reac_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_MCT1ch_noEx_reac_benchmark1_FV_Z256.h5");
	const std::vector<double> absTol = { RelApprox::defaultEpsilon() };
	const std::vector<double> relTol = { RelApprox::defaultMargin() };
	cadet::test::column::FVparams disc(256);
	disc.setNRad(1); // will be used as NCHANNEL
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, false);
}

TEST_CASE("MCT numerical Benchmark for 2 channels with one-way-exchange and reaction case", "[MCT],[Simulation],[Reference],[mctReference]") // todo CI flag: currently only runs locally but fails on server
{
	const std::string& modelFilePath = std::string("/data/model_MCT2ch_oneWayEx_reac_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_MCT2ch_oneWayEx_reac_benchmark1_FV_Z256.h5");
	const std::vector<double> absTol = { RelApprox::defaultEpsilon() };
	const std::vector<double> relTol = { RelApprox::defaultMargin() };
	cadet::test::column::FVparams disc(256);
	disc.setNRad(2); // will be used as NCHANNEL
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, false);
}

TEST_CASE("MCT numerical Benchmark for 3 channels with two-way-exchange and reaction case", "[MCT],[Simulation],[Reference],[mctReference]") // todo CI flag: currently only runs locally but fails on server
{
	const std::string& modelFilePath = std::string("/data/model_MCT3ch_twoWayExc_reac_benchmark1.json");
	const std::string& refFilePath = std::string("/data/ref_MCT3ch_twoWayExc_reac_benchmark1_FV_Z256.h5");
	const std::vector<double> absTol = { RelApprox::defaultEpsilon() };
	const std::vector<double> relTol = { RelApprox::defaultMargin() };
	cadet::test::column::FVparams disc(256);
	disc.setNRad(3); // will be used as NCHANNEL
	cadet::test::column::testReferenceBenchmark(modelFilePath, refFilePath, "001", absTol, relTol, disc, false);
}

TEST_CASE("MCT compare AD with analytical Jacobian for 1 channel without exchange", "[MCT],[UnitOp],[Jacobian],[CI]")
{
	cadet::JsonParameterProvider jpp = createMCT({ 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 0.0 }, 1e-4); // increased col dispersion so that jacobian entries are above tolerances
	jpp.pushScope("model");
	jpp.pushScope("unit_000");
	cadet::test::column::testJacobianAD(jpp, std::numeric_limits<float>::epsilon());
}

TEST_CASE("MCT compare AD with analytical Jacobian for 2 channels and exchange", "[MCT],[UnitOp],[Jacobian],[CI]")
{
	cadet::JsonParameterProvider jpp = createMCT({ 1.0, 1.0 }, { 1.0, 1.0 }, { 1.0, 0.2 }, { 1.0, 1.0 }, { 0.0, 0.01, 0.0, 0.0 }, 1e-4); // increased col dispersion so that jacobian entries are above tolerances
	jpp.pushScope("model");
	jpp.pushScope("unit_000");
	const double FDtolerance = 0.02; // large tolerance to effectively disable FD pattern check, which fails with 0.0 != -0.01
	cadet::test::column::testJacobianAD(jpp, FDtolerance);
}

TEST_CASE("MCT compare AD with analytical Jacobian for 2 channels with opposing flow directions and exchange", "[MCT],[UnitOp],[Jacobian],[CI]")
{
	cadet::JsonParameterProvider jpp = createMCT({ 1.0, 1.0 }, { 1.0, -1.0 }, { 1.0, 0.2 }, { 1.0, 1.0 }, { 0.0, 0.01, 0.0, 0.0 }, 1e-4); // increased col dispersion so that jacobian entries are above tolerances
	jpp.pushScope("model");
	jpp.pushScope("unit_000");
	const double FDtolerance = 0.02; // large tolerance to effectively disable FD pattern check, which fails with 0.0 != -0.01
	cadet::test::column::testJacobianAD(jpp, FDtolerance);
}

TEST_CASE("MCT time derivative Jacobian vs FD", "[MCT],[UnitOp],[Residual],[Jacobian],[CI],[FD]")
{
	// Use some test case parameters
	cadet::JsonParameterProvider jpp = createMCTModelJson({ 1.0, 1.0 }, { 1.0, 1.0 }, { 0.0, 0.01, 0.0, 0.0 });
	cadet::test::unitoperation::testTimeDerivativeJacobianFD(jpp, 1e-7, 0.0, 1e-3);
}

TEST_CASE("MCT sensitivity Jacobians", "[MCT],[UnitOp],[Sensitivity],[CI]")
{
	// Use some test case parameters
	cadet::JsonParameterProvider jpp = createMCTModelJson({ 1.0, 1.0 }, { 1.0, 1.0 }, { 0.0, 0.01, 0.0, 0.0 });
	cadet::test::column::testFwdSensJacobians(jpp, 1e-4, 6e-7, std::numeric_limits<float>::epsilon() * 100.0, false);
}

TEST_CASE("MCT consistent sensitivity initialization", "[MCT],[ConsistentInit],[Sensitivity],[CI]")
{
	// Fill state vector with given initial values
	const unsigned int numDofs = 1 * 2 + 1 * 16 * 2; // nComp * nChannel + nComp * nCol * nChannel = (inletDof + unitDof)
	std::vector<double> y(numDofs, 0.0);
	std::vector<double> yDot(numDofs, 0.0);
	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, numDofs);
	cadet::test::util::populate(yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, numDofs);

	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	for (int adMode = 0; adMode < 2; ++adMode)
	{
		const bool adEnabled = (adMode > 0);
		SECTION("AD " + adEnabled ? "enabled" : "disabled")
		{
			// Use some test case parameters
			cadet::JsonParameterProvider jpp = createMCT({ 1.0, 1.0 }, { 1.0, 1.0 }, { 1.0, 1.0 }, { 1.0, 1.0 }, { 0.0, 0.0, 0.0, 0.0 });
			jpp.pushScope("model");
			jpp.pushScope("unit_000");
			cadet::IUnitOperation* const unit = createAndConfigureUnit(*mb, jpp, cadet::Weno::maxOrder());

			unit->setSensitiveParameter(cadet::makeParamId("INIT_C", 0, 0, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 0, 1.0);
			unit->setSensitiveParameter(cadet::makeParamId("COL_LENGTH", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 2, 1.0);

			REQUIRE(unit->numSensParams() == 2);
			cadet::test::unitoperation::testConsistentInitializationSensitivity(unit, adEnabled, y.data(), yDot.data(), 1e-14);

			mb->destroyUnitOperation(unit);
		}
	}
	destroyModelBuilder(mb);
}

TEST_CASE("MCT dynamic reactions time derivative Jacobian vs FD bulk", "[MCT],[Jacobian],[Residual],[ReactionModel],[CI],[FD]")
{
	cadet::JsonParameterProvider jpp = createMCTModelJson({ 1.0, 1.0 }, { 1.0, 1.0 }, { 0.0, 0.01, 0.0, 0.0 });
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, false, false, 1e-7, 1e-14, 8e-4);
}

TEST_CASE("MCT dynamic reactions time derivative Jacobian vs FD modified bulk", "[MCT],[Jacobian],[Residual],[ReactionModel],[CI],[FD]")
{
	cadet::JsonParameterProvider jpp = createMCTModelJson({ 1.0, 1.0 }, { 1.0, 1.0 }, { 0.0, 0.01, 0.0, 0.0 });
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, false, true, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("MCT dynamic reactions Jacobian vs AD bulk", "[MCT],[Jacobian],[AD],[ReactionModel],[CI]")
{
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD("MULTI_CHANNEL_TRANSPORT", "FV", true, false, false, std::numeric_limits<float>::epsilon() * 100.0);
}