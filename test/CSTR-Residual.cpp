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
#include "cadet/cadet.hpp"

#include "model/StirredTankModel.hpp"
#include "ModelBuilderImpl.hpp"
#include "SimulationTypes.hpp"
#include "ParallelSupport.hpp"

#include "JsonTestModels.hpp"
#include "JacobianHelper.hpp"
#include "ParticleHelper.hpp"
#include "ReactionModelTests.hpp"
#include "SimHelper.hpp"
#include "UnitOperationTests.hpp"
#include "Utils.hpp"

#include <cmath>
#include <functional>
#include <algorithm>

/**
 * @brief Creates a runnable CSTR model with given WENO order
 * @details Creates a CSTR model and configures it using the given IParameterProvider @p jpp.
 * @param [in] mb ModelBuilder
 * @param [in] jpp Configuration of the CSTR
 * @return Runnable CSTR
 */
cadet::model::CSTRModel* createAndConfigureCSTR(cadet::IModelBuilder& mb, cadet::JsonParameterProvider& jpp)
{
	// Create a CSTR
	cadet::IModel* const iCstr = mb.createUnitOperation(jpp, 0);
	REQUIRE(nullptr != iCstr);

	cadet::model::CSTRModel* const cstr = reinterpret_cast<cadet::model::CSTRModel*>(iCstr);

	// Configure
	cadet::ModelBuilder& temp = *reinterpret_cast<cadet::ModelBuilder*>(&mb);
	REQUIRE(cstr->configureModelDiscretization(jpp, temp));
	REQUIRE(cstr->configure(jpp));

	return cstr;
}

/**
 * @brief Creates a CSTR model with two components and linear binding model
 * @return Two component CSTR model with linear binding model
 */
inline cadet::JsonParameterProvider createTwoComponentLinearTestCase()
{
	cadet::JsonParameterProvider jpp = createCSTR(2);
	cadet::test::addBoundStates(jpp, {1, 1}, 0.5);
	cadet::test::addLinearBindingModel(jpp, true, {0.1, 0.2}, {1.0, 0.9});
	cadet::test::setInitialConditions(jpp, {1.0, 2.0}, {3.0, 4.0}, 6.0);
	cadet::test::setFlowRates(jpp, 0, 1.0);

	return jpp;
}

/**
 * @brief Creates a CSTR model with two components, linear binding model, and three particle types
 * @return Two component CSTR model with linear binding model and three particle types
 */
inline cadet::JsonParameterProvider createTwoComponentLinearThreeParticleTypesTestCase()
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();

	const double parVolFrac[] = {0.3, 0.6, 0.1};
	const double parFactor[] = {0.9, 0.8};
	cadet::test::particle::extendModelToManyParticleTypes(jpp, 3, parFactor, parVolFrac);

	return jpp;
}

inline void checkJacobianAD(double flowRateIn, double flowRateOut, double flowRateFilter, std::function<void(cadet::JsonParameterProvider&, unsigned int)> modelRefiner)
{
	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	// CSTR with 2 components
	const unsigned int nComp = 2;
	cadet::JsonParameterProvider jpp = createCSTR(nComp);
	modelRefiner(jpp, nComp);

	cadet::model::CSTRModel* const cstrAna = createAndConfigureCSTR(*mb, jpp);
	cadet::model::CSTRModel* const cstrAD = createAndConfigureCSTR(*mb, jpp);
	if (flowRateFilter > 0.0)
	{
		cstrAna->flowRateFilter().push_back(flowRateFilter);
		cstrAD->flowRateFilter().push_back(flowRateFilter);
	}

	cstrAna->setFlowRates(flowRateIn, flowRateOut);
	cstrAD->setFlowRates(flowRateIn, flowRateOut);

	// Enable AD
	cadet::ad::setDirections(cadet::ad::getMaxDirections());
	cstrAD->useAnalyticJacobian(false);

	cadet::active* adRes = new cadet::active[cstrAD->numDofs()];
	cadet::active* adY = new cadet::active[cstrAD->numDofs()];

	const cadet::AdJacobianParams noParams{nullptr, nullptr, 0u};
	const cadet::AdJacobianParams adParams{adRes, adY, 0u};
	cstrAD->prepareADvectors(adParams);

	// Obtain memory for state, Jacobian multiply direction, Jacobian column
	const unsigned int nDof = cstrAna->numDofs();
	std::vector<double> y(nDof, 0.0);
	std::vector<double> yDot(nDof, 0.0);
	std::vector<double> jacDir(nDof, 0.0);
	std::vector<double> jacCol1(nDof, 0.0);
	std::vector<double> jacCol2(nDof, 0.0);
	cadet::util::ThreadLocalStorage tls;
	tls.resize(cstrAna->threadLocalMemorySize());

	// Fill state vectors with some values
	cadet::test::util::populate(y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
	cadet::test::util::populate(yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);

	// Setup matrices
	cstrAna->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, noParams);
	cstrAD->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, adParams);

	// Compute state Jacobian
	cstrAna->residualWithJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), nullptr}, jacDir.data(), noParams, tls);
	cstrAD->residualWithJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), nullptr}, jacDir.data(), adParams, tls);
	std::fill(jacDir.begin(), jacDir.end(), 0.0);

	// Compare Jacobians
	cadet::test::compareJacobian(cstrAna, cstrAD, y.data(), yDot.data(), jacDir.data(), jacCol1.data(), jacCol2.data());

	delete[] adRes;
	delete[] adY;
	mb->destroyUnitOperation(cstrAna);
	mb->destroyUnitOperation(cstrAD);
	destroyModelBuilder(mb);	
}

inline void checkJacobianFD(double flowRateIn, double flowRateOut, double flowRateFilter, std::function<void(cadet::JsonParameterProvider&, unsigned int)> modelRefiner)
{
	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	// CSTR with 2 components
	const unsigned int nComp = 2;
	cadet::JsonParameterProvider jpp = createCSTR(nComp);
	modelRefiner(jpp, nComp);

	cadet::model::CSTRModel* const cstr = createAndConfigureCSTR(*mb, jpp);
	if (flowRateFilter > 0.0)
		cstr->flowRateFilter().push_back(flowRateFilter);

	cstr->setFlowRates(flowRateIn, flowRateOut);

	// Obtain memory for state, Jacobian multiply direction, Jacobian column
	const unsigned int nDof = cstr->numDofs();
	std::vector<double> y(nDof, 0.0);
	std::vector<double> yDot(nDof, 0.0);
	std::vector<double> jacDir(nDof, 0.0);
	std::vector<double> jacCol1(nDof, 0.0);
	std::vector<double> jacCol2(nDof, 0.0);
	cadet::util::ThreadLocalStorage tls;
	tls.resize(cstr->threadLocalMemorySize());

	// Fill state vectors with some values
	cadet::test::util::populate(y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
	cadet::test::util::populate(yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);

	// Setup matrices
	cstr->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, cadet::AdJacobianParams{nullptr, nullptr, 0u});

	// Compute state Jacobian
	cstr->residualWithJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, jacDir.data(), cadet::AdJacobianParams{nullptr, nullptr, 0u}, tls);
	std::fill(jacDir.begin(), jacDir.end(), 0.0);

	// Compare Jacobians
//	cadet::test::checkJacobianPatternFD(cstr, cstr, y.data(), yDot.data(), jacDir.data(), jacCol1.data(), jacCol2.data());
	cadet::test::compareJacobianFD(cstr, cstr, y.data(), yDot.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), tls);

	mb->destroyUnitOperation(cstr);
	destroyModelBuilder(mb);	
}

inline void checkTimeDerivativeJacobianFD(double flowRateIn, double flowRateOut, double flowRateFilter, std::function<void(cadet::JsonParameterProvider&, unsigned int)> modelRefiner)
{
	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	// CSTR with 2 components
	const unsigned int nComp = 2;
	cadet::JsonParameterProvider jpp = createCSTR(nComp);
	modelRefiner(jpp, nComp);

	cadet::model::CSTRModel* const cstr = createAndConfigureCSTR(*mb, jpp);
	if (flowRateFilter > 0.0)
		cstr->flowRateFilter().push_back(flowRateFilter);

	cstr->setFlowRates(flowRateIn, flowRateOut);

	// Obtain memory for state, Jacobian multiply direction, Jacobian column
	const unsigned int nDof = cstr->numDofs();
	std::vector<double> y(nDof, 0.0);
	std::vector<double> yDot(nDof, 0.0);
	std::vector<double> jacDir(nDof, 0.0);
	std::vector<double> jacCol1(nDof, 0.0);
	std::vector<double> jacCol2(nDof, 0.0);
	cadet::util::ThreadLocalStorage tls;
	tls.resize(cstr->threadLocalMemorySize());

	// Fill state vectors with some values
	cadet::test::util::populate(y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
	cadet::test::util::populate(yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);

	// Setup matrices
	cstr->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, cadet::AdJacobianParams{nullptr, nullptr, 0u});

	// Compare Jacobians
	cadet::test::compareTimeDerivativeJacobianFD(cstr, cstr, y.data(), yDot.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), tls);

	mb->destroyUnitOperation(cstr);
	destroyModelBuilder(mb);
}

inline void checkConsistentInitialization(const std::function<cadet::model::CSTRModel*(cadet::IModelBuilder&, bool)>& modelCreator, double const* initState, double consTol, double absTol)
{
	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	for (int bindingMode = 0; bindingMode < 2; ++bindingMode)
	{
		const bool isKinetic = (bindingMode == 0);
		for (int adMode = 0; adMode < 2; ++adMode)
		{
			const bool adEnabled = (adMode > 0);
			SECTION(std::string(isKinetic ? " Kinetic binding" : " Quasi-stationary binding") + " with AD " + (adEnabled ? "enabled" : "disabled"))
			{
				cadet::model::CSTRModel* const cstr = modelCreator(*mb, isKinetic);
				std::vector<double> y(initState, initState + cstr->numDofs());
				cadet::test::unitoperation::testConsistentInitialization(cstr, adEnabled, y.data(), consTol, absTol);
				mb->destroyUnitOperation(cstr);
			}
		}
	}
	destroyModelBuilder(mb);
}

inline void checkSensitivityConsistentInitialization(const std::function<cadet::model::CSTRModel*(cadet::IModelBuilder&, bool)>& modelCreator, double const* y, double const* yDot, double absTol)
{
	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	for (int bindingMode = 0; bindingMode < 2; ++bindingMode)
	{
		const bool isKinetic = (bindingMode == 0);
		for (int adMode = 0; adMode < 2; ++adMode)
		{
			const bool adEnabled = (adMode > 0);
			SECTION(std::string(isKinetic ? " Kinetic binding" : " Quasi-stationary binding") + " with AD " + (adEnabled ? "enabled" : "disabled"))
			{
				cadet::model::CSTRModel* const cstr = modelCreator(*mb, isKinetic);
				cadet::test::unitoperation::testConsistentInitializationSensitivity(cstr, adEnabled, y, yDot, absTol);
				mb->destroyUnitOperation(cstr);
			}
		}
	}
	destroyModelBuilder(mb);
}

TEST_CASE("StirredTankModel Jacobian vs FD w/o binding model", "[CSTR],[UnitOp],[Residual],[Jacobian],[CI]")
{
	const double rateList[] = {0.0, 0.0, 0.0,
	                           1.0, 0.0, 0.0,
	                           0.0, 1.0, 0.0,
	                           0.0, 0.0, 1.0,
	                           1.0, 1.0, 0.0,
	                           1.0, 0.0, 1.0,
	                           0.0, 1.0, 1.0,
	                           1.0, 1.0, 1.0,
	                           1.0, 2.0, 3.0,
	                           2.0, 3.0, 1.0,
	                           3.0, 1.0, 2.0,
	                           3.0, 2.0, 1.0,
	                           2.0, 1.0, 3.0,
	                           1.0, 3.0, 2.0,
	                           1.0, 2.0, 0.0,
	                           2.0, 1.0, 0.0};
	double const* row = &rateList[0];
	for (unsigned int i = 0; i < sizeof(rateList) / sizeof(double) / 3; ++i, row += 3)
	{
		SECTION("Fin = " + std::to_string(row[0]) + " Fout = " + std::to_string(row[1]) + " Ffilter = " + std::to_string(row[2]))
		{
			checkJacobianFD(row[0], row[1], row[2], [](cadet::JsonParameterProvider& jpp, unsigned int nComp) { });
		}
	}
}

TEST_CASE("StirredTankModel Jacobian vs AD w/o binding model", "[CSTR],[UnitOp],[Residual],[Jacobian],[AD],[CI]")
{
	const double rateList[] = {0.0, 0.0, 0.0,
	                           1.0, 0.0, 0.0,
	                           0.0, 1.0, 0.0,
	                           0.0, 0.0, 1.0,
	                           1.0, 1.0, 0.0,
	                           1.0, 0.0, 1.0,
	                           0.0, 1.0, 1.0,
	                           1.0, 1.0, 1.0,
	                           1.0, 2.0, 3.0,
	                           2.0, 3.0, 1.0,
	                           3.0, 1.0, 2.0,
	                           3.0, 2.0, 1.0,
	                           2.0, 1.0, 3.0,
	                           1.0, 3.0, 2.0,
	                           1.0, 2.0, 0.0,
	                           2.0, 1.0, 0.0};
	double const* row = &rateList[0];
	for (unsigned int i = 0; i < sizeof(rateList) / sizeof(double) / 3; ++i, row += 3)
	{
		SECTION("Fin = " + std::to_string(row[0]) + " Fout = " + std::to_string(row[1]) + " Ffilter = " + std::to_string(row[2]))
		{
			checkJacobianAD(row[0], row[1], row[2], [](cadet::JsonParameterProvider& jpp, unsigned int nComp) { });
		}
	}
}

TEST_CASE("StirredTankModel time derivative Jacobian vs FD w/o binding model", "[CSTR],[UnitOp],[Residual],[Jacobian],[CI]")
{
	const double rateList[] = {0.0, 0.0, 0.0,
	                           1.0, 0.0, 0.0,
	                           0.0, 1.0, 0.0,
	                           0.0, 0.0, 1.0,
	                           1.0, 1.0, 0.0,
	                           1.0, 0.0, 1.0,
	                           0.0, 1.0, 1.0,
	                           1.0, 1.0, 1.0,
	                           1.0, 2.0, 3.0,
	                           2.0, 3.0, 1.0,
	                           3.0, 1.0, 2.0,
	                           3.0, 2.0, 1.0,
	                           2.0, 1.0, 3.0,
	                           1.0, 3.0, 2.0,
	                           1.0, 2.0, 0.0,
	                           2.0, 1.0, 0.0};
	double const* row = &rateList[0];
	for (unsigned int i = 0; i < sizeof(rateList) / sizeof(double) / 3; ++i, row += 3)
	{
		SECTION("Fin = " + std::to_string(row[0]) + " Fout = " + std::to_string(row[1]) + " Ffilter = " + std::to_string(row[2]))
		{
			checkTimeDerivativeJacobianFD(row[0], row[1], row[2], [](cadet::JsonParameterProvider& jpp, unsigned int nComp) { });
		}
	}
}

TEST_CASE("StirredTankModel Jacobian vs FD with linear binding", "[CSTR],[UnitOp],[Residual],[Jacobian],[CI]")
{
	const double rateList[] = {0.0, 0.0, 0.0,
	                           1.0, 0.0, 0.0,
	                           0.0, 1.0, 0.0,
	                           0.0, 0.0, 1.0,
	                           1.0, 1.0, 0.0,
	                           1.0, 0.0, 1.0,
	                           0.0, 1.0, 1.0,
	                           1.0, 1.0, 1.0,
	                           1.0, 2.0, 3.0,
	                           2.0, 3.0, 1.0,
	                           3.0, 1.0, 2.0,
	                           3.0, 2.0, 1.0,
	                           2.0, 1.0, 3.0,
	                           1.0, 3.0, 2.0,
	                           1.0, 2.0, 0.0,
	                           2.0, 1.0, 0.0};
	for (int j = 0; j < 2; ++j)
	{
		const bool dynamic = (j == 0);
		const std::string bndMode = dynamic ? " dynamic" : " quasi-stationary";
		double const* row = &rateList[0];
		for (unsigned int i = 0; i < sizeof(rateList) / sizeof(double) / 3; ++i, row += 3)
		{
			SECTION("Fin = " + std::to_string(row[0]) + " Fout = " + std::to_string(row[1]) + " Ffilter = " + std::to_string(row[2]) + bndMode)
			{
				checkJacobianFD(row[0], row[1], row[2], [=](cadet::JsonParameterProvider& jpp, unsigned int nComp) {
					cadet::test::addBoundStates(jpp, {1, 1}, 0.5);
					cadet::test::addLinearBindingModel(jpp, dynamic, {5.0, 4.0}, {2.0, 3.0});
				});
			}
		}
	}
}

TEST_CASE("StirredTankModel Jacobian vs AD with linear binding", "[CSTR],[UnitOp],[Residual],[Jacobian],[AD],[CI]")
{
	const double rateList[] = {0.0, 0.0, 0.0,
	                           1.0, 0.0, 0.0,
	                           0.0, 1.0, 0.0,
	                           0.0, 0.0, 1.0,
	                           1.0, 1.0, 0.0,
	                           1.0, 0.0, 1.0,
	                           0.0, 1.0, 1.0,
	                           1.0, 1.0, 1.0,
	                           1.0, 2.0, 3.0,
	                           2.0, 3.0, 1.0,
	                           3.0, 1.0, 2.0,
	                           3.0, 2.0, 1.0,
	                           2.0, 1.0, 3.0,
	                           1.0, 3.0, 2.0,
	                           1.0, 2.0, 0.0,
	                           2.0, 1.0, 0.0};
	for (int j = 0; j < 2; ++j)
	{
		const bool dynamic = (j == 0);
		const std::string bndMode = dynamic ? " dynamic" : " quasi-stationary";
		double const* row = &rateList[0];
		for (unsigned int i = 0; i < sizeof(rateList) / sizeof(double) / 3; ++i, row += 3)
		{
			SECTION("Fin = " + std::to_string(row[0]) + " Fout = " + std::to_string(row[1]) + " Ffilter = " + std::to_string(row[2]) + bndMode)
			{
				checkJacobianAD(row[0], row[1], row[2], [=](cadet::JsonParameterProvider& jpp, unsigned int nComp) {
					cadet::test::addBoundStates(jpp, {1, 1}, 0.5);
					cadet::test::addLinearBindingModel(jpp, dynamic, {5.0, 4.0}, {2.0, 3.0});
				});
			}
		}
	}
}

TEST_CASE("StirredTankModel time derivative Jacobian vs FD with linear binding", "[CSTR],[UnitOp],[Residual],[Jacobian],[CI]")
{
	const double rateList[] = {0.0, 0.0, 0.0,
	                           1.0, 0.0, 0.0,
	                           0.0, 1.0, 0.0,
	                           0.0, 0.0, 1.0,
	                           1.0, 1.0, 0.0,
	                           1.0, 0.0, 1.0,
	                           0.0, 1.0, 1.0,
	                           1.0, 1.0, 1.0,
	                           1.0, 2.0, 3.0,
	                           2.0, 3.0, 1.0,
	                           3.0, 1.0, 2.0,
	                           3.0, 2.0, 1.0,
	                           2.0, 1.0, 3.0,
	                           1.0, 3.0, 2.0,
	                           1.0, 2.0, 0.0,
	                           2.0, 1.0, 0.0};
	for (int j = 0; j < 2; ++j)
	{
		const bool dynamic = (j == 0);
		const std::string bndMode = dynamic ? " dynamic" : " quasi-stationary";
		double const* row = &rateList[0];
		for (unsigned int i = 0; i < sizeof(rateList) / sizeof(double) / 3; ++i, row += 3)
		{
			SECTION("Fin = " + std::to_string(row[0]) + " Fout = " + std::to_string(row[1]) + " Ffilter = " + std::to_string(row[2]) + bndMode)
			{
				checkTimeDerivativeJacobianFD(row[0], row[1], row[2], [=](cadet::JsonParameterProvider& jpp, unsigned int nComp) {
					cadet::test::addBoundStates(jpp, {1, 1}, 0.5);
					cadet::test::addLinearBindingModel(jpp, dynamic, {5.0, 4.0}, {2.0, 3.0});
				});
			}
		}
	}
}

TEST_CASE("StirredTankModel consistent initialization with linear binding", "[CSTR],[ConsistentInit],[CI]")
{
	// Fill state vector with initial values
	std::vector<double> y(2 + 2 * 2 + 1, 0.0);
	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, y.size());

	checkConsistentInitialization([](cadet::IModelBuilder& mb, bool dynamic) -> cadet::model::CSTRModel* {
		cadet::JsonParameterProvider jpp = createCSTR(2);
		cadet::test::addBoundStates(jpp, {1, 1}, 0.5);
		cadet::test::addLinearBindingModel(jpp, dynamic, {5.0, 4.0}, {2.0, 3.0});

		cadet::model::CSTRModel* const cstr = createAndConfigureCSTR(mb, jpp);
		cstr->setFlowRates(1.0, 1.0);
		return cstr;
	}, y.data(), 1e-14, 1e-14);
}

TEST_CASE("StirredTankModel consistent initialization with SMA binding", "[CSTR],[ConsistentInit]")
{
// Optimal values:
//	const double bindingCell[] = {1.2, 2.0, 1.0, 1.5, 858.034, 66.7896, 3.53273, 2.53153};
	const double bindingCell[] = {1.2, 2.0, 1.0, 1.5, 860.0, 66.0, 3.5, 2.5};

	// Fill state vector with initial values
	std::vector<double> y(4 + 2 * 4 + 1, 0.0);
	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4);
	std::copy(bindingCell, bindingCell + 8, y.data() + 4);
	y[4 + 8] = 1.0;

	checkConsistentInitialization([](cadet::IModelBuilder& mb, bool dynamic) -> cadet::model::CSTRModel* {
		cadet::JsonParameterProvider jpp = createCSTR(4);
		cadet::test::addBoundStates(jpp, {1, 1, 1, 1}, 0.5);
		cadet::test::addSMABindingModel(jpp, dynamic, 1.2e3, {0.0, 35.5, 1.59, 7.7}, {0.0, 1000.0, 1000.0, 1000.0}, {0.0, 4.7, 5.29, 3.7}, {0.0, 11.83, 10.6, 10.0});

		cadet::model::CSTRModel* const cstr = createAndConfigureCSTR(mb, jpp);
		cstr->setFlowRates(1.0, 1.0);
		return cstr;
	}, y.data(), 1e-14, 1e-5);
}

TEST_CASE("StirredTankModel consistent sensitivity initialization with linear binding", "[CSTR],[ConsistentInit],[Sensitivity]")
{
	// Fill state vector with initial values
	std::vector<double> y(2 + 2 * 2 + 1, 0.0);
	std::vector<double> yDot(2 + 2 * 2 + 1, 0.0);
	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, y.size());
	cadet::test::util::populate(yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, yDot.size());

	// Note that checkSensitivityConsistentInitialization() applies non-zero initial values for the whole
	// sensitivity state vector. Hence, this is more strict than usual as most initial sensitivity state
	// vectors are all zero (only sensitivities wrt. initial conditions produce non-zero initial values).

	checkSensitivityConsistentInitialization([](cadet::IModelBuilder& mb, bool dynamic) -> cadet::model::CSTRModel* {
		cadet::JsonParameterProvider jpp = createCSTR(2);
		cadet::test::addBoundStates(jpp, {1, 1}, 0.5);
		cadet::test::addLinearBindingModel(jpp, dynamic, {5.0, 4.0}, {2.0, 3.0});

		cadet::model::CSTRModel* const cstr = createAndConfigureCSTR(mb, jpp);
		cstr->setFlowRates(1.0, 1.0);
		cstr->setSensitiveParameter(cadet::makeParamId("INIT_VOLUME", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 0, 1.0);
		cstr->setSensitiveParameter(cadet::makeParamId("LIN_KA", 0, 0, cadet::ParTypeIndep, 0, cadet::ReactionIndep, cadet::SectionIndep), 1, 1.0);
		cstr->setSensitiveParameter(cadet::makeParamId("POROSITY", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 2, 1.0);

		return cstr;
	}, y.data(), yDot.data(), 1e-14);
}

TEST_CASE("StirredTankModel consistent sensitivity initialization with SMA binding", "[CSTR],[ConsistentInit],[Sensitivity]")
{
	// Fill state vector with initial values
	std::vector<double> y(4 + 2 * 4 + 1, 0.0);
	std::vector<double> yDot(4 + 2 * 4 + 1, 0.0);

	const double bindingCell[] = {1.0, 1.8, 1.5, 1.6, 840.0, 63.0, 6.0, 3.0};
	cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, 4);
	std::copy(bindingCell, bindingCell + 8, y.data() + 4);
	y[4 + 8] = 1.0;

	cadet::test::util::populate(yDot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, yDot.size());

	// Note that checkSensitivityConsistentInitialization() applies non-zero initial values for the whole
	// sensitivity state vector. Hence, this is more strict than usual as most initial sensitivity state
	// vectors are all zero (only sensitivities wrt. initial conditions produce non-zero initial values).

	checkSensitivityConsistentInitialization([](cadet::IModelBuilder& mb, bool dynamic) -> cadet::model::CSTRModel* {
		cadet::JsonParameterProvider jpp = createCSTR(4);
		cadet::test::addBoundStates(jpp, {1, 1, 1, 1}, 0.5);
		cadet::test::addSMABindingModel(jpp, dynamic, 1.2e3, {0.0, 35.5, 1.59, 7.7}, {0.0, 1000.0, 1000.0, 1000.0}, {0.0, 4.7, 5.29, 3.7}, {0.0, 11.83, 10.6, 10.0});

		cadet::model::CSTRModel* const cstr = createAndConfigureCSTR(mb, jpp);
		cstr->setFlowRates(1.0, 1.0);
		cstr->setSensitiveParameter(cadet::makeParamId("INIT_VOLUME", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 0, 1.0);
		cstr->setSensitiveParameter(cadet::makeParamId("SMA_NU", 0, 1, cadet::ParTypeIndep, 0, cadet::ReactionIndep, cadet::SectionIndep), 1, 1.0);
		cstr->setSensitiveParameter(cadet::makeParamId("POROSITY", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 2, 1.0);

		return cstr;
	}, y.data(), yDot.data(), 1e-8);
}

TEST_CASE("StirredTankModel inlet DOF Jacobian", "[CSTR],[UnitOp],[Jacobian],[Inlet],[CI]")
{
	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	for (int adMode = 0; adMode < 2; ++adMode)
	{
		const bool adEnabled = (adMode > 0);
		SECTION(std::string("AD ") + (adEnabled ? "enabled" : "disabled"))
		{
			cadet::JsonParameterProvider jpp = createCSTR(4);
			cadet::model::CSTRModel* const cstr = createAndConfigureCSTR(*mb, jpp);
			cadet::test::unitoperation::testInletDofJacobian(cstr, adEnabled);
			mb->destroyUnitOperation(cstr);
		}
	}
	destroyModelBuilder(mb);
}

TEST_CASE("CSTR multiple particle types Jacobian analytic vs AD", "[CSTR],[Jacobian],[AD],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::particle::testJacobianMixedParticleTypes(jpp);
}

TEST_CASE("CSTR multiple particle types time derivative Jacobian vs FD", "[CSTR],[UnitOp],[Residual],[Jacobian],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::particle::testTimeDerivativeJacobianMixedParticleTypesFD(jpp, 1e-6, 0.0, 9e-4);
}

TEST_CASE("CSTR dynamic reactions Jacobian vs AD bulk", "[CSTR],[Jacobian],[AD],[ReactionModel]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, false, false);
}

TEST_CASE("CSTR dynamic reactions Jacobian vs AD particle", "[CSTR],[Jacobian],[AD],[ReactionModel]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, false, true, false);
}

TEST_CASE("CSTR dynamic reactions Jacobian vs AD modified particle", "[CSTR],[Jacobian],[AD],[ReactionModel]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, false, true, true);
}

TEST_CASE("CSTR dynamic reactions Jacobian vs AD bulk and particle", "[CSTR],[Jacobian],[AD],[ReactionModel]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, true, false);
}

TEST_CASE("CSTR dynamic reactions Jacobian vs AD bulk and modified particle", "[CSTR],[Jacobian],[AD],[ReactionModel]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, true, true);
}

TEST_CASE("CSTR dynamic reactions time derivative Jacobian vs FD bulk", "[CSTR],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, false, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("CSTR dynamic reactions time derivative Jacobian vs FD particle", "[CSTR],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, false, true, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("CSTR dynamic reactions time derivative Jacobian vs FD modified particle", "[CSTR],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, false, true, true, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("CSTR dynamic reactions time derivative Jacobian vs FD bulk and particle", "[CSTR],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, true, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("CSTR dynamic reactions time derivative Jacobian vs FD bulk and modified particle", "[CSTR],[Jacobian],[Residual],[ReactionModel]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearTestCase();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, true, true, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("CSTR multi particle types dynamic reactions Jacobian vs AD bulk", "[CSTR],[Jacobian],[AD],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearThreeParticleTypesTestCase();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, false, false);
}

TEST_CASE("CSTR multi particle types dynamic reactions Jacobian vs AD particle", "[CSTR],[Jacobian],[AD],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearThreeParticleTypesTestCase();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, false, true, false);
}

TEST_CASE("CSTR multi particle types dynamic reactions Jacobian vs AD modified particle", "[CSTR],[Jacobian],[AD],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearThreeParticleTypesTestCase();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, false, true, true);
}

TEST_CASE("CSTR multi particle types dynamic reactions Jacobian vs AD bulk and particle", "[CSTR],[Jacobian],[AD],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearThreeParticleTypesTestCase();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, true, false);
}

TEST_CASE("CSTR multi particle types dynamic reactions Jacobian vs AD bulk and modified particle", "[CSTR],[Jacobian],[AD],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearThreeParticleTypesTestCase();
	cadet::test::reaction::testUnitJacobianDynamicReactionsAD(jpp, true, true, true);
}

TEST_CASE("CSTR multi particle types dynamic reactions time derivative Jacobian vs FD bulk", "[CSTR],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearThreeParticleTypesTestCase();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, false, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("CSTR multi particle types dynamic reactions time derivative Jacobian vs FD particle", "[CSTR],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearThreeParticleTypesTestCase();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, false, true, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("CSTR multi particle types dynamic reactions time derivative Jacobian vs FD modified particle", "[CSTR],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearThreeParticleTypesTestCase();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, false, true, true, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("CSTR multi particle types dynamic reactions time derivative Jacobian vs FD bulk and particle", "[CSTR],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearThreeParticleTypesTestCase();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, true, false, 1e-6, 1e-14, 8e-4);
}

TEST_CASE("CSTR multi particle types dynamic reactions time derivative Jacobian vs FD bulk and modified particle", "[CSTR],[Jacobian],[Residual],[ReactionModel],[ParticleType]")
{
	cadet::JsonParameterProvider jpp = createTwoComponentLinearThreeParticleTypesTestCase();
	cadet::test::reaction::testTimeDerivativeJacobianDynamicReactionsFD(jpp, true, true, true, 1e-6, 1e-14, 8e-4);
}
