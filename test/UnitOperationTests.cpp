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

#define CADET_LOGGING_DISABLE
#include "Logging.hpp"

#include "cadet/ModelBuilder.hpp"
#include "cadet/FactoryFuncs.hpp"
#include "common/JsonParameterProvider.hpp"

#include "ModelBuilderImpl.hpp"
#include "AutoDiff.hpp"
#include "linalg/Norms.hpp"
#include "model/UnitOperation.hpp"
#include "UnitOperationTests.hpp"
#include "JacobianHelper.hpp"
#include "SimulationTypes.hpp"
#include "ParallelSupport.hpp"

#include "Utils.hpp"

#include <vector>

namespace cadet
{

namespace test
{

namespace unitoperation
{

cadet::IUnitOperation* createAndConfigureUnit(cadet::JsonParameterProvider& jpp, cadet::IModelBuilder& mb)
{
	return createAndConfigureUnit(jpp.getString("UNIT_TYPE"), mb, jpp);
}

cadet::IUnitOperation* createAndConfigureUnit(const std::string& uoType, cadet::IModelBuilder& mb,
											  cadet::JsonParameterProvider& jpp)
{
	// Create a unit
	cadet::IModel* const iUnit = mb.createUnitOperation(uoType, 0);
	REQUIRE(nullptr != iUnit);

	cadet::IUnitOperation* const unit = reinterpret_cast<cadet::IUnitOperation*>(iUnit);

	// Configure
	cadet::ModelBuilder& temp = *reinterpret_cast<cadet::ModelBuilder*>(&mb);
	REQUIRE(unit->configureModelDiscretization(jpp, temp));
	REQUIRE(unit->configure(jpp));

	return unit;
}

void testJacobianAD(cadet::JsonParameterProvider& jpp)
{
	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	cadet::IUnitOperation* const unitAna = createAndConfigureUnit(jpp, *mb);
	cadet::IUnitOperation* const unitAD = createAndConfigureUnit(jpp, *mb);

	// Enable AD
	cadet::ad::setDirections(cadet::ad::getMaxDirections());
	unitAD->useAnalyticJacobian(false);

	cadet::active* adRes = new cadet::active[unitAD->numDofs()];
	cadet::active* adY = new cadet::active[unitAD->numDofs()];

	const AdJacobianParams noParams{nullptr, nullptr, 0u};
	const AdJacobianParams adParams{adRes, adY, 0u};

	unitAD->prepareADvectors(adParams);

	// Obtain memory for state, Jacobian multiply direction, Jacobian column
	std::vector<double> y(unitAD->numDofs(), 0.0);
	std::vector<double> jacDir(unitAD->numDofs(), 0.0);
	std::vector<double> jacCol1(unitAD->numDofs(), 0.0);
	std::vector<double> jacCol2(unitAD->numDofs(), 0.0);
	cadet::util::ThreadLocalStorage tls;
	tls.resize(unitAna->threadLocalMemorySize());

	// Fill state vector with some values
	util::populate(
		y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, unitAna->numDofs());
	//		util::populate(y.data(), [](unsigned int idx) { return 1.0; }, unitAna->numDofs());

	// Setup matrices
	unitAna->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, noParams);
	unitAD->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, adParams);

	// Compute state Jacobian
	unitAna->residualWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), nullptr}, jacDir.data(),
								  noParams, tls);
	unitAD->residualWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), nullptr}, jacDir.data(),
								 adParams, tls);
	std::fill(jacDir.begin(), jacDir.end(), 0.0);

	// Compare Jacobians
	cadet::test::checkJacobianPatternFD(unitAna, unitAD, y.data(), nullptr, jacDir.data(), jacCol1.data(),
										jacCol2.data(), tls);
	cadet::test::checkJacobianPatternFD(unitAna, unitAna, y.data(), nullptr, jacDir.data(), jacCol1.data(),
										jacCol2.data(), tls);
	cadet::test::compareJacobian(unitAna, unitAD, nullptr, nullptr, jacDir.data(), jacCol1.data(), jacCol2.data());
	//		cadet::test::compareJacobianFD(unitAna, unitAD, y.data(), nullptr, jacDir.data(), jacCol1.data(),
	// jacCol2.data());

	delete[] adRes;
	delete[] adY;
	mb->destroyUnitOperation(unitAna);
	mb->destroyUnitOperation(unitAD);
	destroyModelBuilder(mb);
}

void testTimeDerivativeJacobianFD(cadet::JsonParameterProvider& jpp, double h, double absTol, double relTol)
{
	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	cadet::IUnitOperation* const unit = createAndConfigureUnit(jpp, *mb);

	// Obtain memory for state, Jacobian multiply direction, Jacobian column
	const unsigned int nDof = unit->numDofs();
	std::vector<double> y(nDof, 0.0);
	std::vector<double> yDot(nDof, 0.0);
	std::vector<double> jacDir(nDof, 0.0);
	std::vector<double> jacCol1(nDof, 0.0);
	std::vector<double> jacCol2(nDof, 0.0);
	cadet::util::ThreadLocalStorage tls;
	tls.resize(unit->threadLocalMemorySize());

	// Fill state vectors with some values
	util::populate(
		y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
	util::populate(
		yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);

	// Setup matrices
	unit->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()},
											   AdJacobianParams{nullptr, nullptr, 0u});

	// Compare Jacobians
	cadet::test::compareTimeDerivativeJacobianFD(unit, unit, y.data(), yDot.data(), jacDir.data(), jacCol1.data(),
												 jacCol2.data(), tls, h, absTol, relTol);

	mb->destroyUnitOperation(unit);
	destroyModelBuilder(mb);
}

void testConsistentInitialization(cadet::IUnitOperation* const unit, bool adEnabled, double* const y, double consTol,
								  double absTol)
{
	cadet::active* adRes = nullptr;
	cadet::active* adY = nullptr;

	// Enable AD
	if (adEnabled)
	{
		cadet::ad::setDirections(cadet::ad::getMaxDirections());
		unit->useAnalyticJacobian(false);

		adRes = new cadet::active[unit->numDofs()];
		adY = new cadet::active[unit->numDofs()];
		unit->prepareADvectors(AdJacobianParams{adRes, adY, 0});
	}
	else
	{
		unit->useAnalyticJacobian(true);
	}

	const AdJacobianParams adParams{adRes, adY, 0};

	// Obtain memory
	std::vector<double> yIn(y, y + unit->numComponents() * unit->numInletPorts());
	std::vector<double> yDot(unit->numDofs(), 0.0);
	std::vector<double> res(unit->numDofs(), 0.0);
	cadet::util::ThreadLocalStorage tls;
	tls.resize(unit->threadLocalMemorySize());

	// Setup matrices
	unit->notifyDiscontinuousSectionTransition(0.0, 0u, {y, yDot.data()}, adParams);

	// Initialize algebraic variables in state vector
	unit->consistentInitialState(SimulationTime{0.0, 0u}, y, adParams, consTol, tls);

	// Make sure inlet DOFs have not been touched
	for (std::size_t i = 0; i < yIn.size(); ++i)
		CHECK(yIn[i] == y[i]);

	// Evaluate residual without yDot and update Jacobians
	// Store residual in yDot
	unit->residualWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y, nullptr}, yDot.data(), adParams, tls);
	CAPTURE(yDot);

	// Initialize time derivative of state
	unit->consistentInitialTimeDerivative(SimulationTime{0.0, 0u}, y, yDot.data(), tls);

	// Check norm of residual (but skip over connection DOFs at the beginning of the local state vector slice)
	unit->residual(SimulationTime{0.0, 0u}, ConstSimulationState{y, yDot.data()}, res.data(), tls);
	INFO("Residual " << linalg::linfNorm(res.data() + unit->numComponents() * unit->numInletPorts(),
										 unit->numDofs() - unit->numComponents() * unit->numInletPorts()));
	CAPTURE(res);
	CHECK(linalg::linfNorm(res.data() + unit->numComponents() * unit->numInletPorts(),
						   unit->numDofs() - unit->numComponents() * unit->numInletPorts()) <= absTol);

	// Clean up
	delete[] adRes;
	delete[] adY;
}

void testConsistentInitializationSensitivity(cadet::IUnitOperation* const unit, bool adEnabled, double const* const y,
											 double const* const yDot, double absTol)
{
	const unsigned int nSens = unit->numSensParams();
	REQUIRE(nSens > 0);

	cadet::active* adRes = new cadet::active[unit->numDofs()];
	cadet::active* adY = nullptr;

	// Enable AD
	if (adEnabled)
	{
		cadet::ad::setDirections(cadet::ad::getMaxDirections());
		unit->useAnalyticJacobian(false);

		adY = new cadet::active[unit->numDofs()];
		unit->prepareADvectors(AdJacobianParams{adRes, adY, nSens});
	}
	else
	{
		unit->useAnalyticJacobian(true);
	}

	const AdJacobianParams adParams{adRes, adY, nSens};

	cadet::util::ThreadLocalStorage tls;
	tls.resize(unit->threadLocalMemorySize());

	// Setup matrices
	unit->notifyDiscontinuousSectionTransition(0.0, 0u, {nullptr, nullptr}, adParams);

	// Calculate dres / dp and Jacobians
	const SimulationTime simTime{0.0, 0u};
	unit->residualSensFwdWithJacobian(simTime, ConstSimulationState{y, yDot}, adParams, tls);

	// Obtain memory
	std::vector<double> sensY(unit->numDofs() * nSens, 0.0);
	std::vector<double> sensYdot(unit->numDofs() * nSens, 0.0);
	std::vector<double> res(unit->numDofs() * nSens, 0.0);

	std::vector<double*> vecSensY(nSens, nullptr);
	std::vector<double*> vecSensYdot(nSens, nullptr);
	std::vector<double const*> cVecSensY(nSens, nullptr);
	std::vector<double const*> cVecSensYdot(nSens, nullptr);
	std::vector<double*> vecRes(nSens, nullptr);

	for (unsigned int i = 0; i < nSens; ++i)
	{
		vecSensY[i] = sensY.data() + i * unit->numDofs();
		vecSensYdot[i] = sensYdot.data() + i * unit->numDofs();
		cVecSensY[i] = sensY.data() + i * unit->numDofs();
		cVecSensYdot[i] = sensYdot.data() + i * unit->numDofs();
		vecRes[i] = res.data() + i * unit->numDofs();
	}

	std::vector<double> tmp1(unit->numDofs(), 0.0);
	std::vector<double> tmp2(unit->numDofs(), 0.0);
	std::vector<double> tmp3(unit->numDofs(), 0.0);

	// Fill sensY and sensYdot with some values
	util::populate(
		sensY.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, sensY.size());
	util::populate(
		sensYdot.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.9)) + 1e-4; }, sensYdot.size());

	// Initialize sensitivity state vectors
	unit->consistentInitialSensitivity(simTime, ConstSimulationState{y, yDot}, vecSensY, vecSensYdot, adRes, tls);

	// Check norm of residual (but skip over connection DOFs at the beginning of the local state vector slice)
	unit->residualSensFwdAdOnly(simTime, ConstSimulationState{y, yDot}, adRes, tls);
	unit->residualSensFwdCombine(simTime, ConstSimulationState{y, yDot}, cVecSensY, cVecSensYdot, vecRes, adRes,
								 tmp1.data(), tmp2.data(), tmp3.data());

	for (unsigned int i = 0; i < nSens; ++i)
	{
		INFO("Residual " << i << ": "
						 << linalg::linfNorm(vecRes[i] + unit->numComponents() * unit->numInletPorts(),
											 unit->numDofs() - unit->numComponents() * unit->numInletPorts()));
		CHECK(linalg::linfNorm(vecRes[i] + unit->numComponents() * unit->numInletPorts(),
							   unit->numDofs() - unit->numComponents() * unit->numInletPorts()) <= absTol);
	}

	// Clean up
	delete[] adRes;
	delete[] adY;
}

void testInletDofJacobian(cadet::IUnitOperation* const unit, bool adEnabled)
{
	cadet::active* adRes = nullptr;
	cadet::active* adY = nullptr;

	// Enable AD
	if (adEnabled)
	{
		cadet::ad::setDirections(cadet::ad::getMaxDirections());
		unit->useAnalyticJacobian(false);

		adRes = new cadet::active[unit->numDofs()];
		adY = new cadet::active[unit->numDofs()];
		unit->prepareADvectors(AdJacobianParams{adRes, adY, 0});
	}
	else
	{
		unit->useAnalyticJacobian(true);
	}

	const AdJacobianParams adParams{adRes, adY, 0};

	// Obtain memory
	std::vector<double> y(unit->numDofs(), 0.0);
	std::vector<double> jac(unit->numDofs(), 0.0);
	cadet::util::ThreadLocalStorage tls;
	tls.resize(unit->threadLocalMemorySize());

	// Setup matrices
	unit->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, adParams);

	// Assemble Jacobian
	const SimulationTime simTime{0.0, 0u};
	const ConstSimulationState simState{y.data(), nullptr};
	unit->residualWithJacobian(simTime, simState, jac.data(), adParams, tls);

	// Check ports and components
	const unsigned int inletBlockSize = unit->numInletPorts() * unit->numComponents();
	CHECK(inletBlockSize == unit->numDofs() - unit->numPureDofs());

	for (unsigned int port = 0; port < unit->numInletPorts(); ++port)
	{
		const unsigned int inletOffset = unit->localInletComponentIndex(port);
		const unsigned int inletStride = unit->localInletComponentStride(port);

		for (unsigned int comp = 0; comp < unit->numComponents(); ++comp)
		{
			// Get Jacobian column
			const unsigned int curItem = inletOffset + comp * inletStride;

			y[curItem] = 1.0;
			unit->multiplyWithJacobian(SimulationTime{0.0, 0u}, simState, y.data(), 1.0, 0.0, jac.data());
			y[curItem] = 0.0;

			// Check inlet block of Jacobian column
			for (unsigned int i = 0; i < inletBlockSize; ++i)
			{
				CAPTURE(port);
				CAPTURE(comp);
				CAPTURE(curItem);
				if (i == curItem)
					CHECK(jac[i] == 1.0);
				else
					CHECK(jac[i] == 0.0);
			}
		}
	}

	// Clean up
	delete[] adRes;
	delete[] adY;
}

} // namespace unitoperation
} // namespace test
} // namespace cadet
