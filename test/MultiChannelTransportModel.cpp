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
#include "common/Driver.hpp"
#include "common/JsonParameterProvider.hpp"
#include "ColumnTests.hpp"
#include "ParticleHelper.hpp"
#include "ReactionModelTests.hpp"
#include "Weno.hpp"
#include "Utils.hpp"
#include "JacobianHelper.hpp"
#include "cadet/ModelBuilder.hpp"
#include "ModelBuilderImpl.hpp"
#include "cadet/FactoryFuncs.hpp"
#include "ParallelSupport.hpp"

namespace
{
	using json = nlohmann::json;

	json createMCTModelJson(const std::vector<double>& crossSectionAreas, const std::vector<double>& exchangeMatrix)
	{
		json config;
		config["UNIT_TYPE"] = "MULTI_CHANNEL_TRANSPORT";
		config["NCOMP"] = 1;
		config["COL_DISPERSION"] = 5.75e-8;
		config["EXCHANGE_MATRIX"] = exchangeMatrix;

		// Geometry
		config["COL_LENGTH"] = 200;
		config["CHANNEL_CROSS_SECTION_AREAS"] = crossSectionAreas;

		// Initial conditions
		config["INIT_C"] = {0.0};

		// Discretization
		{
			json disc;

			disc["NCOL"] = 16;
			disc["NCHANNEL"] = crossSectionAreas.size();

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

	json createMCTJson(const std::vector<double>& volFlowRate, const std::vector<double>& concentrationIn, const std::vector<double>& crossSectionAreas, const std::vector<double>& exchangeMatrix)
	{
		json config;
		// Model
		{
			json model;
			model["NUNITS"] = 2;
			model["unit_000"] = createMCTModelJson(crossSectionAreas, exchangeMatrix);

			// Inlet - unit 001
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
			mct["WRITE_SOLUTION_BULK"] = false;
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

	cadet::JsonParameterProvider createMCT(const std::vector<double>& volFlowRate, const std::vector<double>& concentrationIn, const std::vector<double>& crossSectionAreas, const std::vector<double>& exchangeMatrix)
	{
		return cadet::JsonParameterProvider(createMCTJson(volFlowRate, concentrationIn, crossSectionAreas, exchangeMatrix));
	}
}

TEST_CASE("MCT two channels without exchange yield same result", "[MCT],[Simulation]")
{
//	const double relTol = 1e-6;
//	const double absTol = 1e-10;
	const double relTol = 1e-10;
	const double absTol = 1e-14;

	// Setup simulation
	cadet::JsonParameterProvider jpp = createMCT({1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {0.0, 0.0, 0.0, 0.0});

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

//		const bool relativeOK = std::abs(*bwdInlet - *fwdOutlet) <= relTol * std::abs(*fwdOutlet);
//		if (relativeOK)
//			++numPassed;
//		CHECK()
	}
}

/*
TEST_CASE("MCT LWE forward vs backward flow", "[MCT],[Simulation]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		cadet::test::column::testWenoForwardBackward("MULTI_CHANNEL_TRANSPORT", i, 1e-9, 2e-4);
}

TEST_CASE("MCT Jacobian forward vs backward flow", "[MCT],[UnitOp],[Residual],[Jacobian]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		cadet::test::column::testJacobianWenoForwardBackwardFD("MULTI_CHANNEL_TRANSPORT", i, 1e-6, 0.0, 1e-3);
}

TEST_CASE("MCT time derivative Jacobian vs FD", "[MCT],[UnitOp],[Residual],[Jacobian]")
{
	cadet::test::column::testTimeDerivativeJacobianFD("MULTI_CHANNEL_TRANSPORT", 1e-6, 0.0, 9e-4);
}

TEST_CASE("MCT sensitivity Jacobians", "[MCT],[UnitOp],[Sensitivity]")
{
	cadet::test::column::testFwdSensJacobians("MULTI_CHANNEL_TRANSPORT", 1e-4, 6e-7);
}

TEST_CASE("MCT forward sensitivity vs FD", "[MCT],[Sensitivity],[Simulation]")
{
	// Relative error is checked first, we use high absolute error for letting
	// some points that are far off pass the error test, too. This is required
	// due to errors in finite differences.
	const double fdStepSize[] = {5e-5, 1e-4, 1e-4, 1e-3};
	const double absTols[] = {3e5, 2e-3, 2e-4, 5.0};
	const double relTols[] = {5e-3, 7e-2, 8e-2, 1e-4};
	const double passRatio[] = {0.95, 0.9, 0.91, 0.83};
	cadet::test::column::testFwdSensSolutionFD("MULTI_CHANNEL_TRANSPORT", false, fdStepSize, absTols, relTols, passRatio);
}

TEST_CASE("MCT forward sensitivity forward vs backward flow", "[MCT],[Sensitivity],[Simulation]")
{
	const double absTols[] = {4e-5, 1e-11, 1e-11, 8e-9};
	const double relTols[] = {6e-9, 5e-8, 5e-6, 5e-10};
	const double passRatio[] = {0.99, 0.95, 0.98, 0.98};
	cadet::test::column::testFwdSensSolutionForwardBackward("MULTI_CHANNEL_TRANSPORT", absTols, relTols, passRatio);
}

TEST_CASE("MCT inlet DOF Jacobian", "[MCT],[UnitOp],[Jacobian],[Inlet]")
{
	cadet::test::column::testInletDofJacobian("MULTI_CHANNEL_TRANSPORT");
}

TEST_CASE("MCT dynamic reactions time derivative Jacobian vs FD bulk", "[MCT],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("MULTI_CHANNEL_TRANSPORT", true, false, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("MCT dynamic reactions time derivative Jacobian vs FD particle", "[MCT],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("MULTI_CHANNEL_TRANSPORT", false, true, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("MCT dynamic reactions time derivative Jacobian vs FD modified particle", "[MCT],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("MULTI_CHANNEL_TRANSPORT", false, true, true, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("MCT dynamic reactions time derivative Jacobian vs FD bulk and particle", "[MCT],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("MULTI_CHANNEL_TRANSPORT", true, true, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("MCT dynamic reactions time derivative Jacobian vs FD bulk and modified particle", "[MCT],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD("MULTI_CHANNEL_TRANSPORT", true, true, true, 1e-6, 1e-14, 8e-4);
}
*/
