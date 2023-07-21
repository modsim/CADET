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

#include "model/OutletModel.hpp"
#include "ParamReaderHelper.hpp"
#include "cadet/Exceptions.hpp"
#include "cadet/SolutionRecorder.hpp"
#include "SimulationTypes.hpp"

#include "ConfigurationHelper.hpp"
#include "ParamIdUtil.hpp"

#include <algorithm>
#include <functional>
#include <limits>

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include "AdUtils.hpp"


inline void residual(double const* y, unsigned int nComp, double* res)
{
	std::copy(y, y + nComp, res);
}


namespace cadet
{

namespace model
{

OutletModel::OutletModel(UnitOpIdx unitOpIdx) : _unitOpIdx(unitOpIdx)
{
}

OutletModel::~OutletModel() CADET_NOEXCEPT
{
}

unsigned int OutletModel::numDofs() const CADET_NOEXCEPT
{
	return _nComp;
}

unsigned int OutletModel::numPureDofs() const CADET_NOEXCEPT
{
	return 0;
}

bool OutletModel::usesAD() const CADET_NOEXCEPT
{
	return false;
}

bool OutletModel::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper)
{
	_nComp = paramProvider.getInt("NCOMP");
	return true;
}

bool OutletModel::configure(IParameterProvider& paramProvider)
{
	return true;
}

void OutletModel::setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections) { }

std::unordered_map<ParameterId, double> OutletModel::getAllParameterValues() const
{
	return std::unordered_map<ParameterId, double>();
}

bool OutletModel::hasParameter(const ParameterId& pId) const
{
	return false;
}

double OutletModel::getParameterDouble(const ParameterId& pId) const
{
	return std::numeric_limits<double>::quiet_NaN();
}

bool OutletModel::setParameter(const ParameterId& pId, int value)
{
	return false;
}

bool OutletModel::setParameter(const ParameterId& pId, double value)
{
	return false;
}

bool OutletModel::setParameter(const ParameterId& pId, bool value)
{
	return false;
}

void OutletModel::setSensitiveParameterValue(const ParameterId& pId, double value)
{
}

bool OutletModel::setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue)
{
	return false;
}

void OutletModel::clearSensParams()
{
}

unsigned int OutletModel::numSensParams() const
{
	return 0;
}

void OutletModel::useAnalyticJacobian(const bool analyticJac) { }
void OutletModel::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac) { }

void OutletModel::reportSolution(ISolutionRecorder& recorder, double const* const solution) const
{
	Exporter expr(_nComp, solution);
	recorder.beginUnitOperation(_unitOpIdx, *this, expr);
	recorder.endUnitOperation();
}

void OutletModel::reportSolutionStructure(ISolutionRecorder& recorder) const
{
	Exporter expr(_nComp, nullptr);
	recorder.unitOperationStructure(_unitOpIdx, *this, expr);
}

unsigned int OutletModel::requiredADdirs() const CADET_NOEXCEPT
{
	return 0;
}

void OutletModel::prepareADvectors(const AdJacobianParams& adJac) const { }

void OutletModel::applyInitialCondition(const SimulationState& simState) const
{
	std::fill(simState.vecStateY, simState.vecStateY + _nComp, 0.0);
	std::fill(simState.vecStateYdot, simState.vecStateYdot + _nComp, 0.0);
}

void OutletModel::readInitialCondition(IParameterProvider& paramProvider) { }

int OutletModel::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	::residual(simState.vecStateY, _nComp, res);
	return 0;
}

int OutletModel::residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, 
	double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	// Jacobian is always identity
	::residual(simState.vecStateY, _nComp, res);
	return 0;
}

int OutletModel::residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	for (unsigned int i = 0; i < _nComp; ++i)
		adRes[i] = simState.vecStateY[i];

	return 0;
}

int OutletModel::residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState, 
	const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS, active const* adRes, 
	double* const tmp1, double* const tmp2, double* const tmp3)
{
	// Directional derivative (dF / dy) * s does nothing since dF / dy = I (identity)
	for (std::size_t param = 0; param < resS.size(); ++param)
	{
		double const* const y = yS[param];
		double* const res = resS[param];
		std::copy(y, y + _nComp, res);
	}
	return 0;
}

int OutletModel::residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	for (unsigned int i = 0; i < _nComp; ++i)
		adJac.adRes[i] = simState.vecStateY[i];

	return 0;
}

void OutletModel::initializeSensitivityStates(const std::vector<double*>& vecSensY) const { }

void OutletModel::consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
	std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	// Nothing to do here as inlet DOFs are initialized by ModelSystem
}

void OutletModel::leanConsistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
	std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	// Nothing to do here as inlet DOFs are initialized by ModelSystem
}

void OutletModel::multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret)
{
	// dF / dy = I (identity matrix)
	for (unsigned int i = 0; i < _nComp; ++i)
	{
		ret[i] = alpha * yS[i] + beta * ret[i];
	}
}

void OutletModel::multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret)
{
	std::fill_n(ret, numDofs(), 0.0);
}


int OutletModel::Exporter::writeInlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);
	std::copy_n(_data, _nComp, buffer);
	return _nComp;
}

int OutletModel::Exporter::writeInlet(double* buffer) const
{
	std::copy_n(_data, _nComp, buffer);
	return _nComp;
}

int OutletModel::Exporter::writeOutlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);
	std::copy_n(_data, _nComp, buffer);
	return _nComp;
}

int OutletModel::Exporter::writeOutlet(double* buffer) const
{
	std::copy_n(_data, _nComp, buffer);
	return _nComp;
}


void registerOutletModel(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models)
{
	models[OutletModel::identifier()] = [](UnitOpIdx uoId) { return new OutletModel(uoId); };
}

}  // namespace model

}  // namespace cadet
