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
#include "Approx.hpp"
#include "cadet/cadet.hpp"

#define CADET_LOGGING_DISABLE
#include "Logging.hpp"

#include "ModelBuilderImpl.hpp"
#include "JsonTestModels.hpp"
#include "SimHelper.hpp"
#include "ParticleHelper.hpp"
#include "common/Driver.hpp"
#include "model/UnitOperation.hpp"
#include "SimulationTypes.hpp"

#include <cmath>
#include <functional>
#include <vector>
#include <algorithm>
#include <iterator>

inline void setFlowRateFilter(cadet::JsonParameterProvider& jpp, double filter)
{
	jpp.pushScope("model");
	jpp.pushScope("unit_000");

	jpp.set("FLOWRATE_FILTER", filter);

	jpp.popScope();
	jpp.popScope();
}

inline cadet::JsonParameterProvider createMultiParticleTypesTestCase()
{
	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(2, 100.0, 1.0);
	cadet::test::setNumberOfComponents(jpp, 0, 2);
	cadet::test::setNumberOfComponents(jpp, 1, 2);
	cadet::test::setNumberOfComponents(jpp, 2, 2);
	cadet::test::setSectionTimes(jpp, {0.0, 100.0});
	cadet::test::addBoundStatesConstSolidVolume(jpp, {1, 1}, 0.5);
	cadet::test::addLinearBindingModel(jpp, true, {0.1, 0.2}, {1.0, 0.9});
	cadet::test::setInitialConditions(jpp, {1.0, 2.0}, {3.0, 4.0}, 6.0);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setInletProfile(jpp, 1, 0, 0.0, 0.0, 0.0, 0.0);
	cadet::test::setInletProfile(jpp, 0, 1, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setInletProfile(jpp, 1, 1, 0.0, 0.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 0.1, 0.1, 0.0);

	return jpp;
}

inline void runSimVarPor(cadet::JsonParameterProvider& jpp, std::function<double(double)> solC, std::function<double(double)> solV)
{
	// Run simulation
	cadet::Driver drv;
	drv.configure(jpp);
	drv.run();

	// Get data from simulation
	cadet::InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(0);
	double const* outlet = simData->outlet();
	double const* volume = simData->volume();
	double const* time = drv.solution()->time();

	// Compare
	for (unsigned int i = 0; i < simData->numDataPoints(); ++i, ++outlet, ++volume, ++time)
	{
		// Compare with relative error 1e-6 and absolute error 4e-5
		CAPTURE(*time);
		CHECK((*outlet) == cadet::test::makeApprox(solC(*time), 1e-6, 4e-5));
		CHECK((*volume) == cadet::test::makeApprox(solV(*time), 1e-6, 4e-5));
	}
}

inline void runSimVarPor(cadet::JsonParameterProvider& jpp, std::function<double(double)> solC, std::function<double(double)> solQ, std::function<double(double)> solV, double absTol = 4e-5, double relTol = 1e-6)
{
	// Run simulation
	cadet::Driver drv;
	drv.configure(jpp);
	drv.run();

	// Get data from simulation
	cadet::InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(0);
	double const* outlet = simData->outlet();
	double const* volume = simData->volume();
	double const* solid = simData->solid();
	double const* time = drv.solution()->time();

	// Compare
	for (unsigned int i = 0; i < simData->numDataPoints(); ++i, ++outlet, ++volume, ++solid, ++time)
	{
		CAPTURE(*time);
		CHECK((*outlet) == cadet::test::makeApprox(solC(*time), relTol, absTol));
		CHECK((*volume) == cadet::test::makeApprox(solV(*time), relTol, absTol));
		CHECK((*solid) == cadet::test::makeApprox(solQ(*time), relTol, absTol));
	}
}

inline void runSensSim(cadet::JsonParameterProvider& jpp, std::function<double(double)> solC, std::function<double(double)> solV, double absTol = 1e-5, double relTol = 1e-8)
{
	// Run simulation
	cadet::Driver drv;
	drv.configure(jpp);
	drv.run();

	REQUIRE(drv.simulator()->numSensParams() == 1);

	// Get data from simulation
	cadet::InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(0);
	double const* outlet = simData->sensOutlet(0);
	double const* volume = simData->sensVolume(0);
	double const* time = drv.solution()->time();

	// Compare
	for (unsigned int i = 0; i < simData->numDataPoints(); ++i, ++outlet, ++volume, ++time)
	{
		CAPTURE(*time);
		CHECK((*outlet) == cadet::test::makeApprox(solC(*time), relTol, absTol));
		CHECK((*volume) == cadet::test::makeApprox(solV(*time), relTol, absTol));
	}
}

TEST_CASE("CSTRVarPor vs analytic solution (V constant) w/o binding model", "[CSTRVarPor],[Simulation]")
{
	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(3, 119.0, 1.0);
	cadet::test::setSectionTimes(jpp, {0.0, 10.0, 100.0, 119.0});
	cadet::test::setInitialConditions(jpp, {0.0}, {}, 10.0);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setInletProfile(jpp, 1, 0, 1.0, -1.0 / 90.0, 0.0, 0.0);
	cadet::test::setInletProfile(jpp, 2, 0, 0.0, 0.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 1.0, 0.5, 0.5);
	cadet::test::setFlowRates(jpp, 1, 1.0, 0.5, 0.5);
	cadet::test::setFlowRates(jpp, 2, 1.0, 0.5, 0.5);

	const double temp = 10.0 * (9.0 + 2.0 * std::sqrt(std::exp(1.0)));
	const double temp2 = 2.0 / 9.0 * (-9.0 - 2.0 * std::sqrt(std::exp(1.0)) + 2 * std::exp(5));
	runSimVarPor(jpp, [=](double t) {
			if (t <= 10.0)
				return -2.0 * std::expm1(-t / 20.0);
			else if (t <= 100.0)
				return (120.0 - temp * std::exp(-t / 20.0) - t)  / 45.0;
			else
				return std::exp(-5.0 - (t - 100.0) / 20.0) * temp2;
		}, 
		[](double t) {
			return 10.0;
	});
}

TEST_CASE("CSTRVarPor vs analytic solution (V increasing) w/o binding model", "[CSTRVarPor],[Simulation]")
{
	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(1, 100.0, 1.0);
	cadet::test::setSectionTimes(jpp, {0.0, 100.0});
	cadet::test::setInitialConditions(jpp, {1.0}, {}, 10.0);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 2.0, 1.0, 0.5);

	runSimVarPor(jpp, [=](double t) {
			return 4.0 * (6000.0 + t * (1200.0 + t * (60.0 + t))) / (3.0 * std::pow(20.0 + t, 3.0));
		}, 
		[](double t) {
			return 10.0 + 0.5 * t;
	});
}

TEST_CASE("CSTRVarPor vs analytic solution (V decreasing) w/o binding model", "[CSTRVarPor],[Simulation]")
{
	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(1, 100.0, 1.0);
	cadet::test::setSectionTimes(jpp, {0.0, 100.0});
	cadet::test::setInitialConditions(jpp, {1.0}, {}, 10.0);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 1.5, 1.5, 0.5);

	runSimVarPor(jpp, [=](double t) {
			return 1.0 + t * (1.0 / 20.0 - t / 800.0);
		}, 
		[](double t) {
			return 10.0 - 0.5 * t;
	});
}

TEST_CASE("CSTRVarPor vs analytic solution (V constant) with dynamic linear binding", "[CSTRVarPor],[Simulation]")
{
	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(1, 100.0, 1.0);
	cadet::test::setSectionTimes(jpp, {0.0, 100.0});
	cadet::test::addBoundStatesConstSolidVolume(jpp, {1}, 1.0);
	cadet::test::setInitialConditions(jpp, {0.0}, {0.0}, 1.0);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 0.1, 0.1, 0.0);
	cadet::test::addLinearBindingModel(jpp, true, {0.1}, {10.0});

	const double sqrt2501 = std::sqrt(2501.0);
	runSimVarPor(jpp, [=](double t) {
			return 1.0 - std::exp(-5.1 * t) * (2501.0 * std::cosh(sqrt2501 * t / 10.0) + 50.0 * sqrt2501 * std::sinh(sqrt2501 * t / 10.0)) / 2501.0;
		}, 
		[=](double t) {
			return 0.01 - std::exp(-5.1 * t) * (2501.0 * std::cosh(sqrt2501 * t / 10.0) + 51.0 * sqrt2501 * std::sinh(sqrt2501 * t / 10.0)) / 250100.0;
		}, 
		[](double t) {
			return 1.0;
	});
}

TEST_CASE("CSTRVarPor vs analytic solution (V constant) with quasi-stationary linear binding", "[CSTRVarPor],[Simulation]")
{
	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(1, 100.0, 1.0);
	cadet::test::setSectionTimes(jpp, {0.0, 100.0});
	cadet::test::addBoundStatesConstSolidVolume(jpp, {1}, 1.0);
	cadet::test::setInitialConditions(jpp, {0.0}, {0.0}, 1.0);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 0.1, 0.1, 0.0);
	cadet::test::addLinearBindingModel(jpp, false, {0.1}, {10.0});

	const double sqrt2501 = std::sqrt(2501.0);
	runSimVarPor(jpp, [=](double t) {  
			return -std::expm1(-10.0 / 101.0 * t);
		}, 
		[=](double t) {
			return -std::expm1(-10.0 / 101.0 * t) * 0.01;
		}, 
		[](double t) {
			return 1.0;
	});
}

TEST_CASE("CSTRVarPor filter flowrate sensitivity vs analytic solution (V constant) w/o binding model", "[CSTRVarPor],[Simulation],[AD],[Sensitivity]")
{
	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(3, 119.0, 1.0);
	cadet::test::setSectionTimes(jpp, {0.0, 10.0, 100.0, 119.0});
	cadet::test::setInitialConditions(jpp, {0.0}, {}, 10.0);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setInletProfile(jpp, 1, 0, 1.0, -1.0 / 90.0, 0.0, 0.0);
	cadet::test::setInletProfile(jpp, 2, 0, 0.0, 0.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 1.0, 0.5, 0.5);
	cadet::test::setFlowRates(jpp, 1, 1.0, 0.5, 0.5);
	cadet::test::setFlowRates(jpp, 2, 1.0, 0.5, 0.5);
	setFlowRateFilter(jpp, 0.5);
	cadet::test::addSensitivity(jpp, "FLOWRATE_FILTER", cadet::makeParamId("FLOWRATE_FILTER", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 1e-6);
	cadet::test::returnSensitivities(jpp, 0);

	const double sqrtE = std::sqrt(std::exp(1.0));
	runSensSim(jpp, [=](double t) {
			if (t <= 10.0)
				return 4.0 + 1.0/200.0 * std::exp(-t / 20.0) * (-800.0 + (-40.0 + t) * t);
			else if (t <= 100.0)
				return -160.0 * (-80.0 + t) / 1800.0 + std::exp(-t / 20.0) * (-4.0 + (9.0 * (-40.0 + t) * t + 2 * sqrtE * (-1700.0 + (-40.0 + t) * t)) / 1800.0);
			else
				return std::exp(-t / 20.0) * (-4.0 + (9.0 * (-40.0 + t) * t + std::exp(5.0) * (8800.0 - 2.0 * (-40.0 + t) * t) + 2.0 * sqrtE * (-1700.0 + (-40.0 + t) * t)) / 1800.0);
		}, 
		[](double t) {
			return -t;
	});
}

TEST_CASE("CSTRVarPor LIN_COEFF sensitivity vs analytic solution (V constant) w/o binding model", "[CSTRVarPor],[Simulation],[AD],[Sensitivity]")
{
	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(1, 100.0, 1.0);
	cadet::test::setSectionTimes(jpp, {0.0, 100.0});
	cadet::test::setInitialConditions(jpp, {0.0}, {}, 10.0);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 1.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 1.0, 0.5, 0.5);
	cadet::test::addSensitivity(jpp, "LIN_COEFF", cadet::makeParamId("LIN_COEFF", 1, 0, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, 0), 1e-6);
	cadet::test::returnSensitivities(jpp, 0);

	runSensSim(jpp, [=](double t) { return 2.0 * (20.0 * std::expm1(-t / 20.0) + t); }, [](double t) { return 0.0; }, 2e-5, 6e-7);
}

TEST_CASE("CSTRVarPor initial volume sensitivity vs analytic solution (V constant) w/o binding model", "[CSTRVarPor],[Simulation],[Sensitivity]")
{
	const double V = 5.0;

	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(2, 100.0, 1.0);
	cadet::test::setSectionTimes(jpp, {0.0, 50.0, 100.0});
	cadet::test::setInitialConditions(jpp, {0.0}, {}, V);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setInletProfile(jpp, 1, 0, 0.0, 0.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 1.0, 1.0, 0.0);
	cadet::test::addSensitivity(jpp, "INIT_VOLUME", cadet::makeParamId("INIT_VOLUME", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 1e-6);
	cadet::test::returnSensitivities(jpp, 0);

	const double invV2 = 1.0 / (V * V);
	const double e50overV = std::exp(50.0 / V);
	runSensSim(jpp, [=](double t) {
			if (t <= 50.0)
				return -std::exp(-t / V) * t * invV2;
			else
				return std::exp(-t / V) * invV2 * (e50overV * (t - 50.0) - t);
		}, 
		[](double t) {
			return 1.0;
	});
}

TEST_CASE("CSTRVarPor initial condition read and apply correctly", "[CSTRVarPor],[InitialConditions]]")
{
	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(1, 100.0, 1.0);
	cadet::test::setSectionTimes(jpp, {0.0, 100.0});
	cadet::test::addBoundStatesConstSolidVolume(jpp, {1}, 0.5);
	cadet::test::setInitialConditions(jpp, {1.0}, {2.0}, 3.0);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 0.1, 0.1, 0.0);
	cadet::test::addLinearBindingModel(jpp, true, {0.1}, {10.0});

	// Create and configure model
	cadet::Driver drv;
	drv.configure(jpp);

	// Fetch CSTR unit operation model
	cadet::IUnitOperation* const uo = reinterpret_cast<cadet::IUnitOperation*>(drv.model()->getUnitOperationModel(0));
	std::vector<double> vecStateY(uo->numDofs(), 0.0);
	std::vector<double> vecStateYdot(uo->numDofs(), 0.0);

	// Apply initial conditions to state vector
	uo->applyInitialCondition(cadet::SimulationState{vecStateY.data(), vecStateYdot.data()});
	CHECK(vecStateY[0] == 0.0);
	CHECK(vecStateY[1] == 1.0);
	CHECK(vecStateY[2] == 2.0);
	CHECK(vecStateY[3] == 3.0);
	CHECK(vecStateYdot[0] == 0.0);
	CHECK(vecStateYdot[1] == 0.0);
	CHECK(vecStateYdot[2] == 0.0);
	CHECK(vecStateYdot[3] == 0.0);

	// Check precedence of INIT_STATE if present
	cadet::JsonParameterProvider jppState("{}");
	jppState.set("INIT_STATE", std::vector<double>{-1.0, -2.0, -3.0});
	jppState.set("INIT_C", 4.0);
	jppState.set("INIT_Q", 5.0);
	jppState.set("INIT_VOLUME", 6.0);
	std::fill(vecStateY.begin(), vecStateY.end(), 0.0);
	std::fill(vecStateYdot.begin(), vecStateYdot.end(), 0.0);

	uo->readInitialCondition(jppState);
	uo->applyInitialCondition(cadet::SimulationState{vecStateY.data(), vecStateYdot.data()});

	CHECK(vecStateY[0] == 0.0);
	CHECK(vecStateY[1] == -1.0);
	CHECK(vecStateY[2] == -2.0);
	CHECK(vecStateY[3] == -3.0);
	CHECK(vecStateYdot[0] == 0.0);
	CHECK(vecStateYdot[1] == 0.0);
	CHECK(vecStateYdot[2] == 0.0);
	CHECK(vecStateYdot[3] == 0.0);

	// Check whether we revert to single values if INIT_STATE is misssing
	jppState.remove("INIT_STATE");
	std::fill(vecStateY.begin(), vecStateY.end(), 0.0);
	std::fill(vecStateYdot.begin(), vecStateYdot.end(), 0.0);

	uo->readInitialCondition(jppState);
	uo->applyInitialCondition(cadet::SimulationState{vecStateY.data(), vecStateYdot.data()});

	CHECK(vecStateY[0] == 0.0);
	CHECK(vecStateY[1] == 4.0);
	CHECK(vecStateY[2] == 5.0);
	CHECK(vecStateY[3] == 6.0);
	CHECK(vecStateYdot[0] == 0.0);
	CHECK(vecStateYdot[1] == 0.0);
	CHECK(vecStateYdot[2] == 0.0);
	CHECK(vecStateYdot[3] == 0.0);

	// Check whether state and time derivative are set if INIT_STATE has correct size
	jppState.set("INIT_STATE", std::vector<double>{-5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0});
	std::fill(vecStateY.begin(), vecStateY.end(), 0.0);
	std::fill(vecStateYdot.begin(), vecStateYdot.end(), 0.0);

	uo->readInitialCondition(jppState);
	uo->applyInitialCondition(cadet::SimulationState{vecStateY.data(), vecStateYdot.data()});

	CHECK(vecStateY[0] == 0.0);
	CHECK(vecStateY[1] == -5.0);
	CHECK(vecStateY[2] == -6.0);
	CHECK(vecStateY[3] == -7.0);
	CHECK(vecStateYdot[0] == 0.0);
	CHECK(vecStateYdot[1] == -8.0);
	CHECK(vecStateYdot[2] == -9.0);
	CHECK(vecStateYdot[3] == -10.0);
}

TEST_CASE("CSTRVarPor initial condition behave like standard parameters", "[CSTRVarPor],[InitialConditions]")
{
	cadet::JsonParameterProvider jpp = createCSTRVarPorBenchmark(1, 100.0, 1.0);
	cadet::test::setNumberOfComponents(jpp, 0, 2);
	cadet::test::setNumberOfComponents(jpp, 1, 2);
	cadet::test::setNumberOfComponents(jpp, 2, 2);
	cadet::test::setSectionTimes(jpp, {0.0, 100.0});
	cadet::test::addBoundStatesConstSolidVolume(jpp, {1, 2}, 0.5);
	cadet::test::addDummyBindingModel(jpp);
	cadet::test::setInitialConditions(jpp, {1.0, 2.0}, {3.0, 4.0, 5.0}, 6.0);
	cadet::test::setInletProfile(jpp, 0, 0, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setInletProfile(jpp, 0, 1, 1.0, 0.0, 0.0, 0.0);
	cadet::test::setFlowRates(jpp, 0, 0.1, 0.1, 0.0);

	// Create and configure model
	cadet::Driver drv;
	drv.configure(jpp);

	// Fetch CSTR unit operation model
	cadet::IUnitOperation* const uo = reinterpret_cast<cadet::IUnitOperation*>(drv.model()->getUnitOperationModel(0));
	std::vector<double> vecStateY(uo->numDofs(), 0.0);
	std::vector<double> vecStateYdot(uo->numDofs(), 0.0);

	// Apply initial conditions to state vector
	uo->applyInitialCondition(cadet::SimulationState{vecStateY.data(), vecStateYdot.data()});
	CHECK(vecStateY[0] == 0.0);
	CHECK(vecStateY[1] == 0.0);
	CHECK(vecStateY[2] == 1.0);
	CHECK(vecStateY[3] == 2.0);
	CHECK(vecStateY[4] == 3.0);
	CHECK(vecStateY[5] == 4.0);
	CHECK(vecStateY[6] == 5.0);
	CHECK(vecStateY[7] == 6.0);

	// Get parameter values
	CHECK(uo->getParameterDouble(cadet::makeParamId("INIT_C", 0, 0, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep)) == 1.0);
	CHECK(uo->getParameterDouble(cadet::makeParamId("INIT_C", 0, 1, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep)) == 2.0);
	CHECK(uo->getParameterDouble(cadet::makeParamId("INIT_Q", 0, 0, 0, 0, cadet::ReactionIndep, cadet::SectionIndep)) == 3.0);
	CHECK(uo->getParameterDouble(cadet::makeParamId("INIT_Q", 0, 1, 0, 0, cadet::ReactionIndep, cadet::SectionIndep)) == 4.0);
	CHECK(uo->getParameterDouble(cadet::makeParamId("INIT_Q", 0, 1, 0, 1, cadet::ReactionIndep, cadet::SectionIndep)) == 5.0);
	CHECK(uo->getParameterDouble(cadet::makeParamId("INIT_VOLUME", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep)) == 6.0);

	// Set parameter values
	uo->setParameter(cadet::makeParamId("INIT_C", 0, 0, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), -1.0);
	uo->setParameter(cadet::makeParamId("INIT_C", 0, 1, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), -2.0);
	uo->setParameter(cadet::makeParamId("INIT_Q", 0, 0, 0, 0, cadet::ReactionIndep, cadet::SectionIndep), -3.0);
	uo->setParameter(cadet::makeParamId("INIT_Q", 0, 1, 0, 0, cadet::ReactionIndep, cadet::SectionIndep), -4.0);
	uo->setParameter(cadet::makeParamId("INIT_Q", 0, 1, 0, 1, cadet::ReactionIndep, cadet::SectionIndep), -5.0);
	uo->setParameter(cadet::makeParamId("INIT_VOLUME", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), -6.0);

	// Apply initial conditions to state vector
	std::fill(vecStateY.begin(), vecStateY.end(), 0.0);
	std::fill(vecStateYdot.begin(), vecStateYdot.end(), 0.0);
	uo->applyInitialCondition(cadet::SimulationState{vecStateY.data(), vecStateYdot.data()});

	CHECK(vecStateY[0] == 0.0);
	CHECK(vecStateY[1] == 0.0);
	CHECK(vecStateY[2] == -1.0);
	CHECK(vecStateY[3] == -2.0);
	CHECK(vecStateY[4] == -3.0);
	CHECK(vecStateY[5] == -4.0);
	CHECK(vecStateY[6] == -5.0);
	CHECK(vecStateY[7] == -6.0);
}

TEST_CASE("CSTRVarPor one vs two identical particle types match", "[CSTRVarPor],[Simulation],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createMultiParticleTypesTestCase();
	cadet::test::particle::testOneVsTwoIdenticalParticleTypes(jpp, 2e-8, 5e-5);
}

TEST_CASE("CSTRVarPor separate identical particle types match", "[CSTRVarPor],[Simulation],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createMultiParticleTypesTestCase();
	cadet::test::particle::testSeparateIdenticalParticleTypes(jpp, 1e-15, 1e-15);
}

TEST_CASE("CSTRVarPor linear binding single particle matches particle distribution", "[CSTRVarPor],[Simulation],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createMultiParticleTypesTestCase();
	cadet::test::particle::testLinearMixedParticleTypes(jpp, 5e-8, 5e-5);
}