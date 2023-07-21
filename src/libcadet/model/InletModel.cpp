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

#include "model/InletModel.hpp"
#include "ParamReaderHelper.hpp"
#include "cadet/Exceptions.hpp"
#include "cadet/InletProfile.hpp"
#include "cadet/SolutionRecorder.hpp"
#include "SimulationTypes.hpp"

#include "ConfigurationHelper.hpp"
#include "ParamIdUtil.hpp"

#include <algorithm>
#include <functional>

#include "LoggingUtils.hpp"
#include "Logging.hpp"
#include "AdUtils.hpp"

namespace cadet
{

namespace model
{

InletModel::InletModel(UnitOpIdx unitOpIdx) : _unitOpIdx(unitOpIdx), _inlet(nullptr),
	_inletConcentrationsRaw(nullptr), _inletDerivatives(nullptr), _inletConcentrations(nullptr)
{
}

InletModel::~InletModel() CADET_NOEXCEPT
{
	delete[] _inletConcentrationsRaw;
	// Do not delete _inletDerivatives since its memory is owned by _inletConcentrationsRaw
	delete[] _inletConcentrations;

	delete _inlet;
}

unsigned int InletModel::numDofs() const CADET_NOEXCEPT
{
	return _nComp;
}

unsigned int InletModel::numPureDofs() const CADET_NOEXCEPT
{
	return _nComp;
}

bool InletModel::usesAD() const CADET_NOEXCEPT
{
	return false;
}

bool InletModel::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper)
{
	_nComp = paramProvider.getInt("NCOMP");
	const std::string inType = paramProvider.getString("INLET_TYPE");

	_inlet = helper.createInletProfile(inType);
	if (!_inlet)
		throw InvalidParameterException("INLET_TYPE " + inType + " is unknown");

	// Allocate memory
	_inletConcentrationsRaw = new double[3 * _nComp]; // Thrice the size because _inletDerivatives uses two thirds
	_inletDerivatives = _inletConcentrationsRaw + _nComp;
	_inletConcentrations = new active[_nComp];

	return true;
}

bool InletModel::configure(IParameterProvider& paramProvider)
{
	return _inlet->configure(&paramProvider, _nComp);
}

void InletModel::setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections)
{
	if (_inlet)
		_inlet->setSectionTimes(secTimes, secContinuity, nSections);
}

std::unordered_map<ParameterId, double> InletModel::getAllParameterValues() const
{
	std::unordered_map<ParameterId, double> data;

	// Collect inlet profile
	if (!_inlet)
		return data;

	const std::vector<ParameterId> inletParams = _inlet->availableParameters(_unitOpIdx);
	for (const ParameterId& pId : inletParams)
	{
		data[pId] = _inlet->getParameterValue(pId);
	}

	return data;
}

bool InletModel::hasParameter(const ParameterId& pId) const
{
	// Check inlet profile
	if (!_inlet || ((pId.unitOperation != _unitOpIdx) && (pId.unitOperation != UnitOpIndep)))
		return false;
	
	const std::vector<ParameterId> inletParams = _inlet->availableParameters(_unitOpIdx);

	// Search for parameter
	for (const ParameterId& ipid : inletParams)
	{
		if (ipid == pId)
			return true;
	}
	return false;
}

double InletModel::getParameterDouble(const ParameterId& pId) const
{
	// Check inlet profile
	if (!_inlet || ((pId.unitOperation != _unitOpIdx) && (pId.unitOperation != UnitOpIndep)))
		return std::numeric_limits<double>::quiet_NaN();

	const std::vector<ParameterId> inletParams = _inlet->availableParameters(_unitOpIdx);

	// Search for parameter
	for (const ParameterId& ipid : inletParams)
	{
		if (ipid == pId)
			return _inlet->getParameterValue(pId);
	}
	return std::numeric_limits<double>::quiet_NaN();
}

bool InletModel::setParameter(const ParameterId& pId, int value)
{
	return false;
}

bool InletModel::setParameter(const ParameterId& pId, double value)
{
	// Check inlet and filter parameters
	if (_inlet && ((pId.unitOperation == _unitOpIdx) || (pId.unitOperation == UnitOpIndep)))
	{
		_inlet->setParameterValue(pId, value);
		return true;
	}

	return false;
}

bool InletModel::setParameter(const ParameterId& pId, bool value)
{
	return false;
}

void InletModel::setSensitiveParameterValue(const ParameterId& pId, double value)
{
	// Check inlet and filter parameters
	if (_inlet && ((pId.unitOperation == _unitOpIdx) || (pId.unitOperation == UnitOpIndep)) && (_sensParamsInlet.find(pId) != _sensParamsInlet.end()))
		_inlet->setParameterValue(pId, value);
}

bool InletModel::setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue)
{
	// Check inlet profile
	if (_inlet && ((pId.unitOperation == _unitOpIdx) || (pId.unitOperation == UnitOpIndep)))
	{
		const std::vector<ParameterId> inletParams = _inlet->availableParameters(_unitOpIdx);

		// Search for parameter
		bool found = false;
		for (ParameterId ipid : inletParams)
		{
			if (ipid == pId)
			{
				found = true;
				break;
			}
		}

		if (found)
		{
			if (_sensParamsInlet.find(pId) == _sensParamsInlet.end())
			{
				LOG(Debug) << "Found parameter " << pId << " in Inlet: Dir " << adDirection << " is set to " << adValue;
			}
			
			// Register parameter and reserve AD direction
			_sensParamsInlet[pId] = std::make_tuple(adDirection, adValue);
			return true;
		}
	}

	return false;
}

void InletModel::clearSensParams()
{
	_sensParamsInlet.clear();
}

unsigned int InletModel::numSensParams() const
{
	return _sensParamsInlet.size();
}

template<> double const* InletModel::moveInletValues(double const* const rawValues, double t, unsigned int secIdx) const
{
	// No parameter derivatives required, return raw values
	return rawValues;
}

template<> active const* InletModel::moveInletValues(double const* const rawValues, double t, unsigned int secIdx) const
{
	// Convert to active
	for (unsigned int i = 0; i < _nComp; ++i)
		_inletConcentrations[i] = rawValues[i];

	// Get time derivative
//	_inlet->timeDerivative(t, secIdx, _inletDerivatives + _nComp);
//	LOG(Debug) << "dInlet / dt = " << cadet::log::VectorPtr<double>(_inletDerivatives + _nComp, _nComp); 

	// Retrieve parameter derivatives
	for (auto sp : _sensParamsInlet)
	{
		const unsigned int adDir = std::get<0>(sp.second);
		const double adValue = std::get<1>(sp.second);

		_inlet->parameterDerivative(t, secIdx, sp.first, _inletDerivatives);
//		LOG(Debug) << "dInlet / dp = " << cadet::log::VectorPtr<double>(_inletDerivatives, _nComp); 

		// Copy derivatives into AD datatypes
		for (unsigned int i = 0; i < _nComp; ++i)
			_inletConcentrations[i].setADValue(adDir, _inletConcentrations[i].getADValue(adDir) + _inletDerivatives[i] * adValue);

//		LOG(Debug) << "totalInlet " << adDir << " " << cadet::log::VectorPtr<cadet::active>(_inletConcentrations, _nComp);
	}

	return _inletConcentrations;
}

template<> double const* InletModel::getData() const
{
	return _inletConcentrationsRaw;
}

template<> active const* InletModel::getData() const
{
	return _inletConcentrations;
}

void InletModel::useAnalyticJacobian(const bool analyticJac) { }
void InletModel::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac) { }

void InletModel::reportSolution(ISolutionRecorder& recorder, double const* const solution) const
{
	Exporter expr(_nComp, solution);
	recorder.beginUnitOperation(_unitOpIdx, *this, expr);
	recorder.endUnitOperation();
}

void InletModel::reportSolutionStructure(ISolutionRecorder& recorder) const
{
	Exporter expr(_nComp, nullptr);
	recorder.unitOperationStructure(_unitOpIdx, *this, expr);
}

unsigned int InletModel::requiredADdirs() const CADET_NOEXCEPT
{
	return 0;
}

void InletModel::prepareADvectors(const AdJacobianParams& adJac) const { }

void InletModel::applyInitialCondition(const SimulationState& simState) const { }

void InletModel::readInitialCondition(IParameterProvider& paramProvider) { }

void InletModel::consistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem)
{ 
	_inlet->inletConcentration(simTime.t, simTime.secIdx, _inletConcentrationsRaw);
	std::copy_n(_inletConcentrationsRaw, _nComp, vecStateY);
}

void InletModel::leanConsistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem)
{
	consistentInitialState(simTime, vecStateY, adJac, errorTol, threadLocalMem);
	_inlet->timeDerivative(simTime.t, simTime.secIdx, _inletDerivatives);
}

void InletModel::consistentInitialTimeDerivative(const SimulationTime& simTime, double const* vecStateY, double* const vecStateYdot, util::ThreadLocalStorage& threadLocalMem)
{ 
	_inlet->timeDerivative(simTime.t, simTime.secIdx, _inletDerivatives);
	std::copy_n(_inletDerivatives, _nComp, vecStateYdot);
}

void InletModel::leanConsistentInitialTimeDerivative(double t, double const* const vecStateY, double* const vecStateYdot, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	std::copy_n(_inletDerivatives, _nComp, vecStateYdot);
}

int InletModel::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	return residualImpl<double, double>(simTime.t, simTime.secIdx, simState, res, threadLocalMem);
}

template <typename ResidualType, typename ParamType>
int InletModel::residualImpl(double t, unsigned int secIdx, const ConstSimulationState& simState, ResidualType* const res, util::ThreadLocalStorage& threadLocalMem)
{
	// Evaluate the user-specified function for the inlet concentration
	_inlet->inletConcentration(t, secIdx, _inletConcentrationsRaw);

	// Copy inlet concentrations over to active types if necessary
	moveInletValues<ResidualType>(_inletConcentrationsRaw, t, secIdx);
	ResidualType const* const fromData = getData<ResidualType>();

	for (unsigned int i = 0; i < _nComp; ++i)
	{
		res[i] = simState.vecStateY[i] - fromData[i];
	}

	return 0;
}

int InletModel::residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	return residualImpl<double, double>(simTime.t, simTime.secIdx, simState, res, threadLocalMem);
}

int InletModel::residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	// Evaluate residual for all parameters using AD in vector mode
	return residualImpl<active, active>(simTime.t, simTime.secIdx, simState, adRes, threadLocalMem);
}

int InletModel::residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState, 
	const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS, active const* adRes, 
	double* const tmp1, double* const tmp2, double* const tmp3)
{
	for (std::size_t param = 0; param < yS.size(); ++param)
	{
		double* const ptrResS = resS[param];
		double const* const ptrYs = yS[param];

		// Compute (dF / dy) * s + dF / dp = s + dF / dp
		for (unsigned int i = 0; i < numDofs(); ++i)
			ptrResS[i] = ptrYs[i] + adRes[i].getADValue(param);
	}
	return 0;
}

int InletModel::residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	// Evaluate residual for all parameters using AD in vector mode, Jacobian is always analytic (identity matrix)
	return residualImpl<active, active>(simTime.t, simTime.secIdx, simState, adJac.adRes, threadLocalMem);
}

void InletModel::initializeSensitivityStates(const std::vector<double*>& vecSensY) const
{
	// The work is actually done in consistentInitialSensitivity()
}

void InletModel::consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
	std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	for (std::size_t param = 0; param < vecSensY.size(); ++param)
	{
		// Copy parameter derivative from AD to tempState and negate it
		for (unsigned int i = 0; i < numDofs(); ++i)
			vecSensY[param][i] = -adRes[i].getADValue(param);
		std::fill_n(vecSensYdot[param], _nComp, 0.0);
	}

	// Calculate second order derivatives and assign to vecSensYdot
	for (auto sp : _sensParamsInlet)
	{
		const unsigned int adDir = std::get<0>(sp.second);
		const double adValue = std::get<1>(sp.second);

		_inlet->timeParameterDerivative(simTime.t, simTime.secIdx, sp.first, _inletDerivatives);

		// Copy derivatives into vecSensYdot
		for (unsigned int i = 0; i < _nComp; ++i)
		{
			vecSensYdot[adDir][i] += adValue * _inletDerivatives[i];
		}
	}
}

void InletModel::leanConsistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
	std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	consistentInitialSensitivity(simTime, simState, vecSensY, vecSensYdot, adRes, threadLocalMem);
}

void InletModel::multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret)
{
	for (unsigned int i = 0; i < numDofs(); ++i)
	{
		ret[i] = alpha * yS[i] + beta * ret[i];
	}
}

void InletModel::multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret)
{
	std::fill_n(ret, numDofs(), 0.0);
}


int InletModel::Exporter::writeInlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);
	std::copy_n(_data, _nComp, buffer);
	return _nComp;
}

int InletModel::Exporter::writeInlet(double* buffer) const
{
	std::copy_n(_data, _nComp, buffer);
	return _nComp;
}

int InletModel::Exporter::writeOutlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);
	std::copy_n(_data, _nComp, buffer);
	return _nComp;
}

int InletModel::Exporter::writeOutlet(double* buffer) const
{
	std::copy_n(_data, _nComp, buffer);
	return _nComp;
}


void registerInletModel(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models)
{
	models[InletModel::identifier()] = [](UnitOpIdx uoId) { return new InletModel(uoId); };
}

}  // namespace model

}  // namespace cadet
