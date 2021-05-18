// =============================================================================
//  CADET
//
//  Copyright © 2008-2021: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include "model/MixerSplitterModel.hpp"
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

MixerSplitterModel::MixerSplitterModel(UnitOpIdx unitOpIdx) : _unitOpIdx(unitOpIdx)
{
}

MixerSplitterModel::~MixerSplitterModel() CADET_NOEXCEPT
{
}

unsigned int MixerSplitterModel::numDofs() const CADET_NOEXCEPT
{
	return 2 * _nComp;
}

unsigned int MixerSplitterModel::numPureDofs() const CADET_NOEXCEPT
{
	return _nComp;
}

bool MixerSplitterModel::usesAD() const CADET_NOEXCEPT
{
	return false;
}

bool MixerSplitterModel::configureModelDiscretization(IParameterProvider& paramProvider, IConfigHelper& helper)
{
	_nComp = paramProvider.getInt("NCOMP");
	return true;
}

bool MixerSplitterModel::configure(IParameterProvider& paramProvider)
{
	return true;
}

void MixerSplitterModel::setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections) { }

std::unordered_map<ParameterId, double> MixerSplitterModel::getAllParameterValues() const
{
	return std::unordered_map<ParameterId, double>();
}

bool MixerSplitterModel::hasParameter(const ParameterId& pId) const
{
	return false;
}

double MixerSplitterModel::getParameterDouble(const ParameterId& pId) const
{
	return std::numeric_limits<double>::quiet_NaN();
}

bool MixerSplitterModel::setParameter(const ParameterId& pId, int value)
{
	return false;
}

bool MixerSplitterModel::setParameter(const ParameterId& pId, double value)
{
	return false;
}

bool MixerSplitterModel::setParameter(const ParameterId& pId, bool value)
{
	return false;
}

void MixerSplitterModel::setSensitiveParameterValue(const ParameterId& pId, double value)
{
}

bool MixerSplitterModel::setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue)
{
	return false;
}

void MixerSplitterModel::clearSensParams()
{
}

unsigned int MixerSplitterModel::numSensParams() const
{
	return 0;
}

void MixerSplitterModel::useAnalyticJacobian(const bool analyticJac) { }
void MixerSplitterModel::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac) { }

void MixerSplitterModel::reportSolution(ISolutionRecorder& recorder, double const* const solution) const
{
	Exporter expr(_nComp, solution);
	recorder.beginUnitOperation(_unitOpIdx, *this, expr);
	recorder.endUnitOperation();
}

void MixerSplitterModel::reportSolutionStructure(ISolutionRecorder& recorder) const
{
	Exporter expr(_nComp, nullptr);
	recorder.unitOperationStructure(_unitOpIdx, *this, expr);
}

unsigned int MixerSplitterModel::requiredADdirs() const CADET_NOEXCEPT
{
	return 0;
}

void MixerSplitterModel::prepareADvectors(const AdJacobianParams& adJac) const { }

void MixerSplitterModel::applyInitialCondition(const SimulationState& simState) const
{
	std::fill(simState.vecStateY, simState.vecStateY + _nComp, 0.0);
	std::fill(simState.vecStateYdot, simState.vecStateYdot + _nComp, 0.0);
}

void MixerSplitterModel::readInitialCondition(IParameterProvider& paramProvider) { }

int MixerSplitterModel::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	::residual(simState.vecStateY, _nComp, res);
	return 0;
}

int MixerSplitterModel::residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, 
	double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	// Jacobian is always identity
	::residual(simState.vecStateY, _nComp, res);
	return 0;
}

int MixerSplitterModel::residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	for (unsigned int i = 0; i < _nComp; ++i)
		adRes[i] = simState.vecStateY[i];

	return 0;
}

int MixerSplitterModel::residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState, 
	const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS, active const* adRes, 
	double* const tmp1, double* const tmp2, double* const tmp3)
{
	// Directional derivative (dF / dy) * s does nothing since dF / dy = I (identity)
	for (unsigned int param = 0; param < resS.size(); ++param)
	{
		double const* const y = yS[param];
		double* const res = resS[param];
		std::copy(y, y + _nComp, res);
	}
	return 0;
}

int MixerSplitterModel::residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	for (unsigned int i = 0; i < _nComp; ++i)
		adJac.adRes[i] = simState.vecStateY[i];

	return 0;
}

void MixerSplitterModel::initializeSensitivityStates(const std::vector<double*>& vecSensY) const { }

void MixerSplitterModel::consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
	std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	// Nothing to do here as inlet DOFs are initialized by ModelSystem
}

void MixerSplitterModel::leanConsistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
	std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	// Nothing to do here as inlet DOFs are initialized by ModelSystem
}

void MixerSplitterModel::multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret)
{
	// dF / dy = I (identity matrix)
	for (unsigned int i = 0; i < _nComp; ++i)
	{
		ret[i] = alpha * yS[i] + beta * ret[i];
	}
}

void MixerSplitterModel::multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret)
{
	std::fill_n(ret, numDofs(), 0.0);
}

int MixerSplitterModel::linearSolve(double t, double alpha, double tol, double* const rhs, double const* const weight,
	const ConstSimulationState& simState)
{
	// Handle inlet equations by backsubstitution
	for (unsigned int i = 0; i < _nComp; ++i)
	{
		rhs[i + _nComp] += rhs[i];
	}
}

void registerMixerSplitterModel(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models)
{
	models[MixerSplitterModel::identifier()] = [](UnitOpIdx uoId) { return new MixerSplitterModel(uoId); };
}

}  // namespace model

}  // namespace cadet
