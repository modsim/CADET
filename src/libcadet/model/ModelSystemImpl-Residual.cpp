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

#include "model/ModelSystemImpl.hpp"

#include "linalg/Norms.hpp"
#include "AdUtils.hpp"
#include "SimulationTypes.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include "ParallelSupport.hpp"
#ifdef CADET_PARALLELIZE
#include <tbb/parallel_for.h>
#endif

#include "model/ModelSystemImpl-Helper.hpp"

namespace
{
/**
 * @brief Selects either double or active SparseMatrix based on template argument
 * @details Helper function that returns either @p a or @p b depending on the template argument.
 * @param [in] a SparseMatrix of double elements
 * @param [in] b SparseMatrix of active elements
 * @tparam selector_t One of @c double or @c active
 * @return Either @p a or @p b depending on the template argument
 */
template <class selector_t>
const cadet::linalg::SparseMatrix<selector_t>& select(const cadet::linalg::SparseMatrix<double>& a,
													  const cadet::linalg::SparseMatrix<cadet::active>& b)
{
	cadet_assert(false);
}

template <>
const cadet::linalg::SparseMatrix<double>& select<double>(const cadet::linalg::SparseMatrix<double>& a,
														  const cadet::linalg::SparseMatrix<cadet::active>& b)
{
	return a;
}

template <>
const cadet::linalg::SparseMatrix<cadet::active>& select<cadet::active>(
	const cadet::linalg::SparseMatrix<double>& a, const cadet::linalg::SparseMatrix<cadet::active>& b)
{
	return b;
}

struct FullTag
{
};
struct LeanTag
{
};

template <bool evalJacobian> struct ResidualSensCaller
{
};

template <> struct ResidualSensCaller<true>
{
	static inline int call(cadet::IUnitOperation* model, const cadet::SimulationTime& simTime,
						   const cadet::ConstSimulationState& simState, const cadet::AdJacobianParams& adJac,
						   cadet::util::ThreadLocalStorage& threadLocalMem)
	{
		return model->residualSensFwdWithJacobian(simTime, simState, adJac, threadLocalMem);
	}
};

template <> struct ResidualSensCaller<false>
{
	static inline int call(cadet::IUnitOperation* model, const cadet::SimulationTime& simTime,
						   const cadet::ConstSimulationState& simState, const cadet::AdJacobianParams& adJac,
						   cadet::util::ThreadLocalStorage& threadLocalMem)
	{
		return model->residualSensFwdAdOnly(simTime, simState, adJac.adRes, threadLocalMem);
	}
};
} // namespace

namespace cadet
{

namespace model
{

void ModelSystem::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx,
													   const ConstSimulationState& simState,
													   const AdJacobianParams& adJac)
{
	// Check if simulation is (re-)starting from the very beginning
	if (secIdx == 0)
		_curSwitchIndex = 0;

	const unsigned int wrapSec = secIdx % _switchSectionIndex.size();
	const unsigned int prevSwitch = _curSwitchIndex;

	// If there are still some switches left and the next switch occurs in this section, advance index
	if ((_curSwitchIndex < _switchSectionIndex.size() - 1) && (_switchSectionIndex[_curSwitchIndex + 1] <= wrapSec))
	{
		++_curSwitchIndex;
	}
	else if (_curSwitchIndex == _switchSectionIndex.size() - 1)
	{
		// We're in the last valve configuration, let's check if we should cycle back to the first one
		if (_switchSectionIndex[0] == wrapSec)
			_curSwitchIndex = 0;
	}

	const bool switchOccurred = (0 == secIdx) || (prevSwitch != _curSwitchIndex);
	if (switchOccurred)
	{
		// A switch has occurred -> Compute flow rate coefficients
		_switchStartTime = t;
		calcUnitFlowRateCoefficients();
	}

	// Notify models that a discontinuous section transition has happened
	for (std::size_t i = 0; i < _models.size(); ++i)
	{
		const unsigned int offset = _dofOffset[i];

		updateModelFlowRates(t, i);
		_models[i]->setFlowRates(_flowRateIn[i], _flowRateOut[i]);
		_models[i]->notifyDiscontinuousSectionTransition(t, secIdx, simState, applyOffset(adJac, offset));
	}

	if (cadet_likely(switchOccurred && !_hasDynamicFlowRates))
	{
		// Update bottom macro row *after* models have changed their flow directions due to updating their internal
		// velocities
		assembleBottomMacroRow(t);
	}

#ifdef CADET_DEBUG
	int const* ptrConn = _connections[_curSwitchIndex];

	LOG(Debug) << "Switching from valve configuration " << prevSwitch << " to " << _curSwitchIndex
			   << " (sec = " << secIdx << " wrapSec = " << wrapSec << ")";
	for (unsigned int i = 0; i < _connections.sliceSize(_curSwitchIndex) / 6; ++i, ptrConn += 6)
	{
		// Extract current connection
		const int uoSource = ptrConn[0];
		const int uoDest = ptrConn[1];
		const int portSource = ptrConn[2];
		const int portDest = ptrConn[3];
		const int compSource = ptrConn[4];
		const int compDest = ptrConn[5];

		// Number of components was already verified so assume they are all correct

		LOG(Debug) << "Unit op " << uoSource << " (" << _models[uoSource]->unitOperationName() << ") port "
				   << portSource << " comp " << compSource << " => " << uoDest << " ("
				   << _models[uoDest]->unitOperationName() << ") port " << portDest << " comp " << compDest;
	}
#endif
}

/**
 * @brief Updates inlet and outlet flow rates of the given unit operation
 * @details Updates the corresponding slice of _flowRateIn and _flowRateOut.
 * @param[in] t Time
 * @param[in] idxUnit Unit operation index
 */
void ModelSystem::updateModelFlowRates(double t, unsigned int idxUnit)
{
	active* const in = _flowRateIn[idxUnit];
	active* const in0 = _totalInletFlow[idxUnit];

	active* const out = _flowRateOut[idxUnit];
	active* const out0 = _totalOutletFlow[idxUnit];

	if (cadet_unlikely(_hasDynamicFlowRates))
	{
		// Convert to time since start of section
		const double secT = t - _switchStartTime;

		active* const in1 = _totalInletFlowLin[idxUnit];
		active* const in2 = _totalInletFlowQuad[idxUnit];
		active* const in3 = _totalInletFlowCub[idxUnit];
		for (unsigned int i = 0; i < _models[idxUnit]->numInletPorts(); ++i)
			in[i] = cubicPoly<active>(in0, in1, in2, in3, i, secT);

		active* const out1 = _totalOutletFlowLin[idxUnit];
		active* const out2 = _totalOutletFlowQuad[idxUnit];
		active* const out3 = _totalOutletFlowCub[idxUnit];
		for (unsigned int i = 0; i < _models[idxUnit]->numOutletPorts(); ++i)
			out[i] = cubicPoly<active>(out0, out1, out2, out3, i, secT);
	}
	else
	{
		std::copy(in0, in0 + _models[idxUnit]->numInletPorts(), in);
		std::copy(out0, out0 + _models[idxUnit]->numOutletPorts(), out);
	}
}

/**
 * @brief Updates inlet and outlet flow rates of the given unit operation
 * @details Updates the corresponding slice of _flowRateIn and _flowRateOut.
 * @param[in] t Time
 * @param[in] idxUnit Unit operation index
 */
void ModelSystem::updateDynamicModelFlowRates(double t, unsigned int idxUnit)
{
	// Convert to time since start of section
	const double secT = t - _switchStartTime;

	active* const in = _flowRateIn[idxUnit];
	active* const in0 = _totalInletFlow[idxUnit];
	active* const in1 = _totalInletFlowLin[idxUnit];
	active* const in2 = _totalInletFlowQuad[idxUnit];
	active* const in3 = _totalInletFlowCub[idxUnit];
	for (unsigned int i = 0; i < _models[idxUnit]->numInletPorts(); ++i)
	{
		in[i] = cubicPoly<active>(in0, in1, in2, in3, i, secT);
		LOG(Debug) << "Flow in unit " << idxUnit << " port " << i << ": " << static_cast<double>(in[i]);
	}

	active* const out = _flowRateOut[idxUnit];
	active* const out0 = _totalOutletFlow[idxUnit];
	active* const out1 = _totalOutletFlowLin[idxUnit];
	active* const out2 = _totalOutletFlowQuad[idxUnit];
	active* const out3 = _totalOutletFlowCub[idxUnit];
	for (unsigned int i = 0; i < _models[idxUnit]->numOutletPorts(); ++i)
	{
		out[i] = cubicPoly<active>(out0, out1, out2, out3, i, secT);
		LOG(Debug) << "Flow out unit " << idxUnit << " port " << i << ": " << static_cast<double>(out[i]);
	}
}

/**
 * @brief Calculate inlet and outlet flow rate coefficients for each unit operation in current section
 */
void ModelSystem::calcUnitFlowRateCoefficients()
{
	// Calculate total flow rate for each inlet
	int const* const ptrConn = _connections[_curSwitchIndex];
	active const* const ptrRate = _flowRates[_curSwitchIndex];
	active const* const ptrRateLin = _flowRatesLin[_curSwitchIndex];
	active const* const ptrRateQuad = _flowRatesQuad[_curSwitchIndex];
	active const* const ptrRateCub = _flowRatesCub[_curSwitchIndex];

	// Reset total flows back to zero
	_totalInletFlow.fill(0.0);
	_totalInletFlowLin.fill(0.0);
	_totalInletFlowQuad.fill(0.0);
	_totalInletFlowCub.fill(0.0);

	_totalOutletFlow.fill(0.0);
	_totalOutletFlowLin.fill(0.0);
	_totalOutletFlowQuad.fill(0.0);
	_totalOutletFlowCub.fill(0.0);

	// Compute total volumetric inflow for each unit operation port
	for (unsigned int i = 0; i < _connections.sliceSize(_curSwitchIndex) / 6; ++i)
	{
		// Extract current connection
		const int uoSource = ptrConn[6 * i];
		const int uoDest = ptrConn[6 * i + 1];
		const int portSource = ptrConn[6 * i + 2];
		const int portDest = ptrConn[6 * i + 3];

		// Check if the same connection has appeared before (with different components)
		bool skip = false;
		for (unsigned int j = 0; j < i; ++j)
		{
			if ((ptrConn[6 * j] == uoSource) && (ptrConn[6 * j + 1] == uoDest) && (ptrConn[6 * j + 2] == portSource) &&
				(ptrConn[6 * j + 3] == portDest))
			{
				skip = true;
				break;
			}
		}

		// Skip this row in connection list if there was an identical previous connection (except for component indices)
		if (skip)
			continue;

		// Use the first flow rate from uoSource to uoDest

		if (portDest < 0)
		{
			for (unsigned int j = 0; j < _models[uoDest]->numInletPorts(); ++j)
			{
				_totalInletFlow(uoDest, j) += ptrRate[i];
				_totalInletFlowLin(uoDest, j) += ptrRateLin[i];
				_totalInletFlowQuad(uoDest, j) += ptrRateQuad[i];
				_totalInletFlowCub(uoDest, j) += ptrRateCub[i];
			}
		}
		else
		{
			_totalInletFlow(uoDest, portDest) += ptrRate[i];
			_totalInletFlowLin(uoDest, portDest) += ptrRateLin[i];
			_totalInletFlowQuad(uoDest, portDest) += ptrRateQuad[i];
			_totalInletFlowCub(uoDest, portDest) += ptrRateCub[i];
		}

		if (portSource < 0)
		{
			for (unsigned int j = 0; j < _models[uoSource]->numOutletPorts(); ++j)
			{
				_totalOutletFlow(uoSource, j) += ptrRate[i];
				_totalOutletFlowLin(uoSource, j) += ptrRateLin[i];
				_totalOutletFlowQuad(uoSource, j) += ptrRateQuad[i];
				_totalOutletFlowCub(uoSource, j) += ptrRateCub[i];
			}
		}
		else
		{
			_totalOutletFlow(uoSource, portSource) += ptrRate[i];
			_totalOutletFlowLin(uoSource, portSource) += ptrRateLin[i];
			_totalOutletFlowQuad(uoSource, portSource) += ptrRateQuad[i];
			_totalOutletFlowCub(uoSource, portSource) += ptrRateCub[i];
		}
	}
}

/**
 * @brief Assembles the right macro column handling the connections
 * @details Only depends on models and their ports / components.
 *          Does not depend on connections between units.
 */
void ModelSystem::assembleRightMacroColumn()
{
	// Clear the matrices before we set new entries
	for (unsigned int i = 0; i < numModels(); ++i)
		_jacNF[i].clear();

	// Assemble Jacobian submatrices

	// Right macro-column
	// NF
	unsigned int couplingIdx = 0;
	for (unsigned int i = 0; i < numModels(); ++i)
	{
		IUnitOperation const* const model = _models[i];

		// Only items with an inlet have non-zero entries in the NF matrices
		if (model->hasInlet())
		{
			for (unsigned int port = 0; port < model->numInletPorts(); ++port)
			{
				// Each component generates a -1 for its inlet in the NF[i] matrix and increases the couplingIdx by 1
				const unsigned int localInletComponentIndex = model->localInletComponentIndex(port);
				const unsigned int localInletComponentStride = model->localInletComponentStride(port);
				for (unsigned int comp = 0; comp < model->numComponents(); ++comp)
				{
					_jacNF[i].addElement(localInletComponentIndex + comp * localInletComponentStride, couplingIdx,
										 -1.0);
					++couplingIdx;
				}
			}
		}
	}
}

/**
 * @brief Assembles the bottom macro row handling the connections
 * @details Computes flow rates and ratios for coupling unit operations.
 * @param[in] t Simulation time
 */
void ModelSystem::assembleBottomMacroRow(double t)
{
	// Convert to time since start of section
	const double secT = t - _switchStartTime;

	// Clear the matrices before we set new entries
	for (unsigned int i = 0; i < numModels(); ++i)
		_jacActiveFN[i].clear();

	int const* const ptrConn = _connections[_curSwitchIndex];
	active const* const ptrRate = _flowRates[_curSwitchIndex];
	active const* const ptrRateLin = _flowRatesLin[_curSwitchIndex];
	active const* const ptrRateQuad = _flowRatesQuad[_curSwitchIndex];
	active const* const ptrRateCub = _flowRatesCub[_curSwitchIndex];

	// Bottom macro-row
	// FN
	for (unsigned int i = 0; i < _connections.sliceSize(_curSwitchIndex) / 6; ++i)
	{
		// Extract current connection
		const int uoSource = ptrConn[6 * i];
		const int uoDest = ptrConn[6 * i + 1];
		const int portSource = ptrConn[6 * i + 2];
		const int portDest = ptrConn[6 * i + 3];
		const int compSource = ptrConn[6 * i + 4];
		const int compDest = ptrConn[6 * i + 5];

		// Obtain index of first connection from uoSource to uoDest
		unsigned int idx = i;
		for (unsigned int j = 0; j < i; ++j)
		{
			if ((ptrConn[6 * j] == uoSource) && (ptrConn[6 * j + 1] == uoDest) && (ptrConn[6 * j + 2] == portSource) &&
				(ptrConn[6 * j + 3] == portDest))
			{
				idx = j;
				break;
			}
		}

		// idx contains the index of the first connection from uoSource to uoDest
		// Hence, ptrRate[idx] is the flow rate to use for this connection

		IUnitOperation const* const modelSource = _models[uoSource];

		// The outlet column is the outlet index + component number * outlet stride

		if (portSource == -1)
		{
			for (unsigned int j = 0; j < modelSource->numOutletPorts(); ++j)
			{
				const active totInFlow =
					cubicPoly<active>(_totalInletFlow(uoDest, j), _totalInletFlowLin(uoDest, j),
									  _totalInletFlowQuad(uoDest, j), _totalInletFlowCub(uoDest, j), secT);

				// Ignore ports with incoming flow rate 0
				if (totInFlow <= 0.0)
					continue;

				const unsigned int outletIndex = modelSource->localOutletComponentIndex(j);
				const unsigned int outletStride = modelSource->localOutletComponentStride(j);

				const active inFlow =
					-cubicPoly<active>(ptrRate, ptrRateLin, ptrRateQuad, ptrRateCub, idx, secT) / totInFlow;

				if (compSource == -1)
				{

					// Connect all components with the same flow rate
					for (unsigned int comp = 0; comp < modelSource->numComponents(); ++comp)
					{
						const unsigned int row =
							_couplingIdxMap[std::make_tuple(uoDest, j, comp)]; // destination coupling DOF
						const unsigned int col = outletIndex + outletStride * comp;
						_jacActiveFN[uoSource].addElement(row, col, inFlow);
					}
				}
				else
				{
					const unsigned int row =
						_couplingIdxMap[std::make_tuple(uoDest, j, compDest)]; // destination coupling DOF
					const unsigned int col = outletIndex + outletStride * compSource;
					_jacActiveFN[uoSource].addElement(row, col, inFlow);
				}
			}
		}
		else
		{
			const active totInFlow =
				cubicPoly<active>(_totalInletFlow(uoDest, portDest), _totalInletFlowLin(uoDest, portDest),
								  _totalInletFlowQuad(uoDest, portDest), _totalInletFlowCub(uoDest, portDest), secT);

			// Ignore ports with incoming flow rate 0
			if (totInFlow <= 0.0)
				continue;

			const unsigned int outletIndex = modelSource->localOutletComponentIndex(portSource);
			const unsigned int outletStride = modelSource->localOutletComponentStride(portSource);

			const active inFlow =
				-cubicPoly<active>(ptrRate, ptrRateLin, ptrRateQuad, ptrRateCub, idx, secT) / totInFlow;

			if (compSource == -1)
			{
				// Connect all components with the same flow rate
				for (unsigned int comp = 0; comp < modelSource->numComponents(); ++comp)
				{
					const unsigned int row =
						_couplingIdxMap[std::make_tuple(uoDest, portDest, comp)]; // destination coupling DOF
					const unsigned int col = outletIndex + outletStride * comp;
					_jacActiveFN[uoSource].addElement(row, col, inFlow);
				}
			}
			else
			{
				const unsigned int row =
					_couplingIdxMap[std::make_tuple(uoDest, portDest, compDest)]; // destination coupling DOF
				const unsigned int col = outletIndex + outletStride * compSource;
				_jacActiveFN[uoSource].addElement(row, col, inFlow);
			}
		}
	}

	// Copy active sparse matrices to their double pendants
	for (unsigned int i = 0; i < numModels(); ++i)
		_jacFN[i].copyFrom(_jacActiveFN[i]);
}

double ModelSystem::residualNorm(const SimulationTime& simTime, const ConstSimulationState& simState)
{
	residual(simTime, simState, _tempState);
	LOG(Debug) << "Residual: " << log::VectorPtr<double>(_tempState, numDofs());
	return linalg::linfNorm(_tempState, numDofs());
}

int ModelSystem::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res)
{
	BENCH_START(_timerResidual);

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), _models.size(), [&](std::size_t i)
#else
	for (std::size_t i = 0; i < _models.size(); ++i)
#endif
	{
		IUnitOperation* const m = _models[i];
		const unsigned int offset = _dofOffset[i];

		if (cadet_unlikely(_hasDynamicFlowRates))
		{
			updateDynamicModelFlowRates(simTime.t, i);
			m->setFlowRates(_flowRateIn[i], _flowRateOut[i]);
		}

		_errorIndicator[i] = m->residual(simTime, applyOffset(simState, offset), res + offset, _threadLocalStorage);
	} CADET_PARFOR_END;

	// Handle connections
	if (cadet_unlikely(_hasDynamicFlowRates))
		assembleBottomMacroRow(simTime.t);

	residualConnectUnitOps<double, double, double>(simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res);

	BENCH_STOP(_timerResidual);
	return totalErrorIndicatorFromLocal(_errorIndicator);
}

int ModelSystem::residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState,
									  double* const res, const AdJacobianParams& adJac)
{
	BENCH_START(_timerResidual);

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), _models.size(), [&](std::size_t i)
#else
	for (std::size_t i = 0; i < _models.size(); ++i)
#endif
	{
		IUnitOperation* const m = _models[i];
		const unsigned int offset = _dofOffset[i];

		if (cadet_unlikely(_hasDynamicFlowRates))
		{
			updateDynamicModelFlowRates(simTime.t, i);
			m->setFlowRates(_flowRateIn[i], _flowRateOut[i]);
		}

		_errorIndicator[i] = m->residualWithJacobian(simTime, applyOffset(simState, offset), res + offset,
													 applyOffset(adJac, offset), _threadLocalStorage);

	} CADET_PARFOR_END;

	// Handle connections
	if (cadet_unlikely(_hasDynamicFlowRates))
		assembleBottomMacroRow(simTime.t);

	residualConnectUnitOps<double, double, double>(simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res);

	BENCH_STOP(_timerResidual);
	return totalErrorIndicatorFromLocal(_errorIndicator);
}

/**
 * @brief Calculate coupling DOF residual
 * @param [in] secIdx  Section ID
 * @param [in] y State vector
 * @param [in] yDot Derivative state vector
 * @param [in,out] res Residual vector
 * @tparam StateType Type of the state vector
 * @tparam ResidualType Type of the residual vector
 * @tparam ParamType Type of the parameters
 */
template <typename StateType, typename ResidualType, typename ParamType>
void ModelSystem::residualConnectUnitOps(unsigned int secIdx, StateType const* const y, double const* const yDot,
										 ResidualType* const res) CADET_NOEXCEPT
{
	// Use connection matrices for the residual
	const unsigned int finalOffset = _dofOffset.back();

	// N_f (Inlets to Inlets) Lower Right diagonal (Identity matrix)
	// The lower right matrix is Identity so residual equals y value
	for (unsigned int i = finalOffset; i < numDofs(); ++i)
		res[i] = y[i];

	// These could technically be done in parallel but from profiling no time is spent here
	// and the parallelization has more overhead than can be gained.

	// N_{x,f} Inlets (Right) matrices; Right macro-column
	unsigned int offset;
	for (std::size_t i = 0; i < _models.size(); ++i)
	{
		offset = _dofOffset[i];
		_jacNF[i].multiplyAdd(y + finalOffset, res + offset);
	}

	// N_{f,x} Outlet (Lower) matrices; Bottom macro-row
	for (std::size_t i = 0; i < _models.size(); ++i)
	{
		offset = _dofOffset[i];
		select<ParamType>(_jacFN[i], _jacActiveFN[i]).multiplyAdd(y + offset, res + finalOffset);
	}
}

int ModelSystem::residualSensFwd(unsigned int nSens, const SimulationTime& simTime,
								 const ConstSimulationState& simState, double const* const res,
								 const std::vector<const double*>& yS, const std::vector<const double*>& ySdot,
								 const std::vector<double*>& resS, active* const adRes, double* const tmp1,
								 double* const tmp2, double* const tmp3)
{
	BENCH_SCOPE(_timerResidualSens);
	return residualSensFwdWithJacobianAlgorithm<false>(nSens, simTime, simState, res, yS, ySdot, resS,
													   AdJacobianParams{adRes, nullptr, 0}, tmp1, tmp2, tmp3);
}

void ModelSystem::multiplyWithMacroJacobian(double const* yS, double alpha, double beta, double* ret)
{
	const unsigned int finalOffset = _dofOffset.back();

	// Set ret_con = yS_con
	// This applies the identity matrix in the bottom right corner of the Jaocbian (network coupling equation)

	for (unsigned int i = finalOffset; i < numDofs(); ++i)
	{
		ret[i] = alpha * yS[i] + beta * ret[i];
	}

	// N_{x,f} Inlets (Right) matrices
	for (std::size_t i = 0; i < _models.size(); ++i)
	{
		const unsigned int offset = _dofOffset[i];
		_jacNF[i].multiplyAdd(yS + finalOffset, ret + offset, alpha);
	}

	// N_{f,x} Outlet (Lower) matrices
	for (std::size_t i = 0; i < _models.size(); ++i)
	{
		const unsigned int offset = _dofOffset[i];
		_jacFN[i].multiplyAdd(yS + offset, ret + finalOffset, alpha);
	}
}

void ModelSystem::residualSensFwdNorm(unsigned int nSens, const SimulationTime& simTime,
									  const ConstSimulationState& simState, const std::vector<const double*>& yS,
									  const std::vector<const double*>& ySdot, double* const norms, active* const adRes,
									  double* const tmp)
{
	const unsigned int nDOFs = numDofs();

	// Reserve memory for nSens residual vectors
	util::SlicedVector<double> tempRes;
	tempRes.reserve(nSens * nDOFs, nSens);

	std::vector<double*> resPtr(nSens, nullptr);
	for (std::size_t i = 0; i < resPtr.size(); ++i)
	{
		tempRes.pushBackSlice(nDOFs);
		resPtr[i] = tempRes[i];
	}

	// Reserve some more temporary memory
	std::vector<double> tempMem(nDOFs * 2, 0.0);

	// Evaluate all the sensitivity system residuals at once
	residualSensFwd(nSens, simTime, simState, nullptr, yS, ySdot, resPtr, adRes, tmp, tempMem.data(),
					tempMem.data() + nDOFs);

	// Calculate norms
	for (unsigned int i = 0; i < nSens; ++i)
		norms[i] = linalg::linfNorm(tempRes[i], nDOFs);
}

int ModelSystem::residualSensFwdWithJacobian(unsigned int nSens, const SimulationTime& simTime,
											 const ConstSimulationState& simState, double const* const res,
											 const std::vector<const double*>& yS,
											 const std::vector<const double*>& ySdot, const std::vector<double*>& resS,
											 const AdJacobianParams& adJac, double* const tmp1, double* const tmp2,
											 double* const tmp3)
{
	return residualSensFwdWithJacobianAlgorithm<true>(nSens, simTime, simState, res, yS, ySdot, resS, adJac, tmp1, tmp2,
													  tmp3);
}

template <bool evalJacobian>
int ModelSystem::residualSensFwdWithJacobianAlgorithm(unsigned int nSens, const SimulationTime& simTime,
													  const ConstSimulationState& simState, double const* const res,
													  const std::vector<const double*>& yS,
													  const std::vector<const double*>& ySdot,
													  const std::vector<double*>& resS, const AdJacobianParams& adJac,
													  double* const tmp1, double* const tmp2, double* const tmp3)
{
	BENCH_START(_timerResidualSens);

	const unsigned int nModels = _models.size();

	// Resize yStemp and yStempDot (this should be a noop except for the first time)
	_yStemp.resize(nModels);
	_yStempDot.resize(nModels);
	_resSTemp.resize(nModels);

	for (unsigned int i = 0; i < nModels; ++i)
	{
		_yStemp[i].resize(yS.size());
		_yStempDot[i].resize(ySdot.size());
		_resSTemp[i].resize(resS.size());
	}

	// Step 1: Calculate sensitivities using AD in vector mode

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(nModels), [&](std::size_t i)
#else
	for (unsigned int i = 0; i < nModels; ++i)
#endif
	{
		IUnitOperation* const m = _models[i];
		const unsigned int offset = _dofOffset[i];

		if (cadet_unlikely(_hasDynamicFlowRates))
		{
			updateDynamicModelFlowRates(simTime.t, i);
			m->setFlowRates(_flowRateIn[i], _flowRateOut[i]);
		}

		_errorIndicator[i] = ResidualSensCaller<evalJacobian>::call(m, simTime, applyOffset(simState, offset),
																	applyOffset(adJac, offset), _threadLocalStorage);
	} CADET_PARFOR_END;

	// Connect units
	if (cadet_unlikely(_hasDynamicFlowRates))
		assembleBottomMacroRow(simTime.t);

	residualConnectUnitOps<double, active, active>(simTime.secIdx, simState.vecStateY, simState.vecStateYdot, adJac.adRes);

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(nModels), [&](std::size_t i)
#else
	for (unsigned int i = 0; i < nModels; ++i)
#endif
	{
		// Step 2: Compute forward sensitivity residuals by multiplying with system Jacobians
		IUnitOperation* const m = _models[i];
		const unsigned int offset = _dofOffset[i];

		// Move this outside the loop, these are memory addresses and should never change
		// Use correct offset in sensitivity state vectors
		for (std::size_t j = 0; j < yS.size(); ++j)
		{
			_yStemp[i][j] = yS[j] + offset;
			_yStempDot[i][j] = ySdot[j] + offset;
			_resSTemp[i][j] = resS[j] + offset;
		}

		const int intermediateRes =
			m->residualSensFwdCombine(simTime, applyOffset(simState, offset), _yStemp[i], _yStempDot[i], _resSTemp[i],
									  adJac.adRes + offset, tmp1 + offset, tmp2 + offset, tmp3 + offset);
		_errorIndicator[i] = updateErrorIndicator(_errorIndicator[i], intermediateRes);
	} CADET_PARFOR_END;

	// tmp1 stores result of (dF / dy) * s
	// tmp2 stores result of (dF / dyDot) * sDot

	const unsigned int finalOffset = _dofOffset.back();

	// Handle super structure (i.e., right macro column and lower macro row)

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), yS.size(), [&](std::size_t param)
#else
	for (std::size_t param = 0; param < yS.size(); ++param)
#endif
	{
		double* const ptrResS = resS[param];

		// Directional derivative: res_{con} = (dF / dy) * s
		// Also adds contribution of the right macro column blocks
		multiplyWithMacroJacobian(yS[param], ptrResS);

		// Directional derivative (dF / dyDot) * sDot  (always zero so ignore it)

		// The other adRes values have already been taken care of in the unit operations
		for (unsigned int i = finalOffset; i < numDofs(); ++i)
		{
			ptrResS[i] += adJac.adRes[i].getADValue(param);
		}
	} CADET_PARFOR_END;

	BENCH_STOP(_timerResidualSens);
	return totalErrorIndicatorFromLocal(_errorIndicator);
}

} // namespace model

} // namespace cadet
