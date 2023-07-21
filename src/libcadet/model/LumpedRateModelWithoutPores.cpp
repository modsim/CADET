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

#include "model/LumpedRateModelWithoutPores.hpp"
#include "BindingModelFactory.hpp"
#include "ParamReaderHelper.hpp"
#include "cadet/Exceptions.hpp"
#include "cadet/ExternalFunction.hpp"
#include "cadet/SolutionRecorder.hpp"
#include "ConfigurationHelper.hpp"
#include "model/BindingModel.hpp"
#include "SimulationTypes.hpp"
#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"
#include "linalg/Norms.hpp"
#include "linalg/Subset.hpp"
#include "model/parts/BindingCellKernel.hpp"

#include "Stencil.hpp"
#include "Weno.hpp"
#include "AdUtils.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include <algorithm>
#include <functional>

#include "ParallelSupport.hpp"
#ifdef CADET_PARALLELIZE
	#include <tbb/parallel_for.h>
#endif

namespace
{
	template <typename ConvDispOperator, typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	class ConvOpResidual
	{
	public:
		static inline void call(cadet::IModel* const model, ConvDispOperator& op, double t, unsigned int secIdx, StateType const* const y, double const* const yDot, ResidualType* const res, cadet::linalg::BandMatrix& jac)
		{
			// This should not be reached
			cadet_assert(false);
		}
	};

	template <typename ConvDispOperator, typename ResidualType, typename ParamType>
	class ConvOpResidual<ConvDispOperator, double, ResidualType, ParamType, true>
	{
	public:
		static inline void call(cadet::IModel* const model, ConvDispOperator& op, double t, unsigned int secIdx, double const* const y, double const* const yDot, ResidualType* const res, cadet::linalg::BandMatrix& jac)
		{
			op.residual(*model, t, secIdx, y, yDot, res, jac);
		}
	};

	template <typename ConvDispOperator, typename ResidualType, typename ParamType>
	class ConvOpResidual<ConvDispOperator, double, ResidualType, ParamType, false>
	{
	public:
		static inline void call(cadet::IModel* const model, ConvDispOperator& op, double t, unsigned int secIdx, double const* const y, double const* const yDot, ResidualType* const res, cadet::linalg::BandMatrix& jac)
		{
			op.residual(*model, t, secIdx, y, yDot, res, typename cadet::ParamSens<ParamType>::enabled());
		}
	};

	template <typename ConvDispOperator, typename ResidualType, typename ParamType>
	class ConvOpResidual<ConvDispOperator, cadet::active, ResidualType, ParamType, false>
	{
	public:
		static inline void call(cadet::IModel* const model, ConvDispOperator& op, double t, unsigned int secIdx, cadet::active const* const y, double const* const yDot, ResidualType* const res, cadet::linalg::BandMatrix& jac)
		{
			op.residual(*model, t, secIdx, y, yDot, res, typename cadet::ParamSens<ParamType>::enabled());
		}
	};
}

namespace cadet
{

namespace model
{

template <typename ConvDispOperator>
LumpedRateModelWithoutPores<ConvDispOperator>::LumpedRateModelWithoutPores(UnitOpIdx unitOpIdx) : UnitOperationBase(unitOpIdx),
	_jacInlet(), _analyticJac(true), _jacobianAdDirs(0), _factorizeJacobian(false), _tempState(nullptr), _initC(0),
	_initQ(0), _initState(0), _initStateDot(0)
{
	// Multiple particle types are not supported
	_singleBinding = true;
	_singleDynReaction = true;
}

template <typename ConvDispOperator>
LumpedRateModelWithoutPores<ConvDispOperator>::~LumpedRateModelWithoutPores() CADET_NOEXCEPT
{
	delete[] _tempState;

	delete[] _disc.nBound;
	delete[] _disc.boundOffset;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithoutPores<ConvDispOperator>::numDofs() const CADET_NOEXCEPT
{
	// Column bulk DOFs: nCol * nComp mobile phase and nCol * (sum boundStates) solid phase
	// Inlet DOFs: nComp
	return _disc.nCol * (_disc.nComp + _disc.strideBound) + _disc.nComp;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithoutPores<ConvDispOperator>::numPureDofs() const CADET_NOEXCEPT
{
	// Column bulk DOFs: nCol * nComp mobile phase and nCol * (sum boundStates) solid phase
	return _disc.nCol * (_disc.nComp + _disc.strideBound);
}


template <typename ConvDispOperator>
bool LumpedRateModelWithoutPores<ConvDispOperator>::usesAD() const CADET_NOEXCEPT
{
#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
	// We always need AD if we want to check the analytical Jacobian
	return true;
#else
	// We only need AD if we are not computing the Jacobian analytically
	return !_analyticJac;
#endif
}

template <typename ConvDispOperator>
bool LumpedRateModelWithoutPores<ConvDispOperator>::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper)
{
	// ==== Read discretization
	_disc.nComp = paramProvider.getInt("NCOMP");

	paramProvider.pushScope("discretization");

	_disc.nCol = paramProvider.getInt("NCOL");

	const std::vector<int> nBound = paramProvider.getIntArray("NBOUND");
	if (nBound.size() < _disc.nComp)
		throw InvalidParameterException("Field NBOUND contains too few elements (NCOMP = " + std::to_string(_disc.nComp) + " required)");

	_disc.nBound = new unsigned int[_disc.nComp];
	std::copy_n(nBound.begin(), _disc.nComp, _disc.nBound);

	// Precompute offsets and total number of bound states (DOFs in solid phase)
	_disc.boundOffset = new unsigned int[_disc.nComp];
	_disc.boundOffset[0] = 0;
	for (unsigned int i = 1; i < _disc.nComp; ++i)
	{
		_disc.boundOffset[i] = _disc.boundOffset[i-1] + _disc.nBound[i-1];
	}
	_disc.strideBound = _disc.boundOffset[_disc.nComp-1] + _disc.nBound[_disc.nComp - 1];

	// Determine whether analytic Jacobian should be used but don't set it right now.
	// We need to setup Jacobian matrices first.
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	const bool analyticJac = paramProvider.getBool("USE_ANALYTIC_JACOBIAN");
#else
	const bool analyticJac = false;
#endif

	// Allocate space for initial conditions
	_initC.resize(_disc.nComp);
	_initQ.resize(_disc.strideBound);

	// Create nonlinear solver for consistent initialization
	configureNonlinearSolver(paramProvider);

	paramProvider.popScope();

	const unsigned int strideCell = _disc.nComp + _disc.strideBound;
	const bool transportSuccess = _convDispOp.configureModelDiscretization(paramProvider, helper, _disc.nComp, _disc.nCol, strideCell);

	// Allocate memory
	Indexer idxr(_disc);

	_jacInlet.resize(_disc.nComp);

	const unsigned int lb = _convDispOp.jacobianLowerBandwidth();
	const unsigned int ub = _convDispOp.jacobianUpperBandwidth();
	const unsigned int mb = _convDispOp.jacobianDiscretizedBandwidth();

	// Allocate matrices such that bandwidths can be switched (backwards flow support)
	_jac.resize(_disc.nCol * strideCell, lb, ub);

	_jacDisc.resize(_disc.nCol * strideCell, mb, mb);
	_jacDisc.repartition(lb, ub);

	// Set whether analytic Jacobian is used
	useAnalyticJacobian(analyticJac);

	// ==== Construct and configure binding model
	clearBindingModels();
	_binding.push_back(nullptr);

	if (paramProvider.exists("ADSORPTION_MODEL"))
		_binding[0] = helper.createBindingModel(paramProvider.getString("ADSORPTION_MODEL"));
	else
		_binding[0] = helper.createBindingModel("NONE");

	if (!_binding[0])
		throw InvalidParameterException("Unknown binding model " + paramProvider.getString("ADSORPTION_MODEL"));

	bool bindingConfSuccess = true;
	if (_binding[0]->usesParamProviderInDiscretizationConfig())
		paramProvider.pushScope("adsorption");

	bindingConfSuccess = _binding[0]->configureModelDiscretization(paramProvider, _disc.nComp, _disc.nBound, _disc.boundOffset);

	if (_binding[0]->usesParamProviderInDiscretizationConfig())
		paramProvider.popScope();

	// ==== Construct and configure dynamic reaction model
	bool reactionConfSuccess = true;
	clearDynamicReactionModels();
	_dynReaction.push_back(nullptr);

	if (paramProvider.exists("REACTION_MODEL"))
	{
		_dynReaction[0] = helper.createDynamicReactionModel(paramProvider.getString("REACTION_MODEL"));
		if (!_dynReaction[0])
			throw InvalidParameterException("Unknown dynamic reaction model " + paramProvider.getString("REACTION_MODEL"));

		if (_dynReaction[0]->usesParamProviderInDiscretizationConfig())
			paramProvider.pushScope("reaction");

		reactionConfSuccess = _dynReaction[0]->configureModelDiscretization(paramProvider, _disc.nComp, _disc.nBound, _disc.boundOffset);

		if (_dynReaction[0]->usesParamProviderInDiscretizationConfig())
			paramProvider.popScope();
	}

	// Setup the memory for tempState based on state vector
	_tempState = new double[numDofs()];

	return transportSuccess && bindingConfSuccess && reactionConfSuccess;
}

template <typename ConvDispOperator>
bool LumpedRateModelWithoutPores<ConvDispOperator>::configure(IParameterProvider& paramProvider)
{
	_parameters.clear();

	const bool transportSuccess = _convDispOp.configure(_unitOpIdx, paramProvider, _parameters);

	// Read geometry parameters
	_totalPorosity = paramProvider.getDouble("TOTAL_POROSITY");

	// Add parameters to map
	_parameters[makeParamId(hashString("TOTAL_POROSITY"), _unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_totalPorosity;

	// Register initial conditions parameters
	for (unsigned int i = 0; i < _disc.nComp; ++i)
		_parameters[makeParamId(hashString("INIT_C"), _unitOpIdx, i, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = _initC.data() + i;

	if (_binding[0])
	{
		std::vector<ParameterId> initParams(_disc.strideBound);
		_binding[0]->fillBoundPhaseInitialParameters(initParams.data(), _unitOpIdx, cadet::ParTypeIndep);

		for (unsigned int i = 0; i < _disc.strideBound; ++i)
			_parameters[initParams[i]] = _initQ.data() + i;
	}

	// Reconfigure binding model
	bool bindingConfSuccess = true;
	if (_binding[0] && paramProvider.exists("adsorption") && _binding[0]->requiresConfiguration())
	{
		paramProvider.pushScope("adsorption");
		bindingConfSuccess = _binding[0]->configure(paramProvider, _unitOpIdx, cadet::ParTypeIndep);
		paramProvider.popScope();
	}

	// Reconfigure dynamic reaction model
	bool reactionConfSuccess = true;
	if (_dynReaction[0] && paramProvider.exists("reaction") && _dynReaction[0]->requiresConfiguration())
	{
		paramProvider.pushScope("reaction");
		reactionConfSuccess = _dynReaction[0]->configure(paramProvider, _unitOpIdx, cadet::ParTypeIndep);
		paramProvider.popScope();
	}

	return transportSuccess && bindingConfSuccess && reactionConfSuccess;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithoutPores<ConvDispOperator>::threadLocalMemorySize() const CADET_NOEXCEPT
{
	LinearMemorySizer lms;

	// Memory for parts::cell::residualKernel = residualImpl()
	if (_binding[0] && _binding[0]->requiresWorkspace())
		lms.addBlock(_binding[0]->workspaceSize(_disc.nComp, _disc.strideBound, _disc.nBound));

	if (_dynReaction[0])
	{
		lms.addBlock(_dynReaction[0]->workspaceSize(_disc.nComp, _disc.strideBound, _disc.nBound));
		lms.add<active>(_disc.strideBound);
		lms.add<double>(_disc.strideBound * (_disc.strideBound + _disc.nComp));
	}

	lms.commit();
	const std::size_t resKernelSize = lms.bufferSize();

	// Memory for consistentInitialSensitivity()
	lms.add<double>(_disc.strideBound);
	lms.add<double>(_disc.strideBound);

	lms.commit();

	// Memory for consistentInitialState()
	lms.add<double>(_nonlinearSolver->workspaceSize(_disc.strideBound) * sizeof(double));
	lms.add<double>(_disc.strideBound);
	lms.add<double>(_disc.strideBound + _disc.nComp);
	lms.add<double>(_disc.strideBound + _disc.nComp);
	lms.add<double>(_disc.strideBound * _disc.strideBound);
	lms.add<double>(_disc.nComp);
	lms.addBlock(_binding[0]->workspaceSize(_disc.nComp, _disc.strideBound, _disc.nBound));
	lms.addBlock(resKernelSize);

	lms.commit();

	return lms.bufferSize();
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::useAnalyticJacobian(const bool analyticJac)
{
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	_analyticJac = analyticJac;
	if (!_analyticJac)
		_jacobianAdDirs = _jac.stride();
	else
		_jacobianAdDirs = 0;
#else
	_analyticJac = false;
	_jacobianAdDirs = _jac.stride();
#endif
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac)
{
	Indexer idxr(_disc);

	// ConvectionDispersionOperator tells us whether flow direction has changed
	if (!_convDispOp.notifyDiscontinuousSectionTransition(t, secIdx) && (secIdx != 0))
		return;

	// Setup the matrix connecting inlet DOFs to first column cells
	_jacInlet.clear();
	const double v = _convDispOp.inletJacobianFactor();

	const unsigned int lb = _convDispOp.jacobianLowerBandwidth();
	const unsigned int ub = _convDispOp.jacobianUpperBandwidth();
	if (_convDispOp.forwardFlow())
	{
		// Forwards flow

		// Place entries for inlet DOF to first column cell conversion
		for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			_jacInlet.addElement(comp * idxr.strideColComp(), comp, -v);

		// Repartition Jacobians
		_jac.repartition(lb, ub);
		_jacDisc.repartition(lb, ub);
	}
	else
	{
		// Backwards flow

		// Place entries for inlet DOF to last column cell conversion
		const unsigned int offset = (_disc.nCol - 1) * idxr.strideColCell();
		for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			_jacInlet.addElement(offset + comp * idxr.strideColComp(), comp, v);

		// Repartition Jacobians
		_jac.repartition(ub, lb);
		_jacDisc.repartition(ub, lb);
	}

	prepareADvectors(adJac);
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::setFlowRates(active const* in, active const* out) CADET_NOEXCEPT
{
	_convDispOp.setFlowRates(in[0], out[0], _totalPorosity);
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::reportSolution(ISolutionRecorder& recorder, double const* const solution) const
{
	Exporter expr(_disc, *this, solution);
	recorder.beginUnitOperation(_unitOpIdx, *this, expr);
	recorder.endUnitOperation();
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::reportSolutionStructure(ISolutionRecorder& recorder) const
{
	Exporter expr(_disc, *this, nullptr);
	recorder.unitOperationStructure(_unitOpIdx, *this, expr);
}


template <typename ConvDispOperator>
unsigned int LumpedRateModelWithoutPores<ConvDispOperator>::requiredADdirs() const CADET_NOEXCEPT
{
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	return _jacobianAdDirs;
#else
	// If CADET_CHECK_ANALYTIC_JACOBIAN is active, we always need the AD directions for the Jacobian
	return _jac.stride();
#endif
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::prepareADvectors(const AdJacobianParams& adJac) const
{
	// Early out if AD is disabled
	if (!adJac.adY)
		return;

	Indexer idxr(_disc);

	// Get bandwidths
	const unsigned int lowerBandwidth = _jac.lowerBandwidth();
	const unsigned int upperBandwidth = _jac.upperBandwidth();

	ad::prepareAdVectorSeedsForBandMatrix(adJac.adY + _disc.nComp, adJac.adDirOffset, _jac.rows(), lowerBandwidth, upperBandwidth, lowerBandwidth);
}

/**
 * @brief Extracts the system Jacobian from band compressed AD seed vectors
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset)
{
	Indexer idxr(_disc);
	ad::extractBandedJacobianFromAd(adRes + idxr.offsetC(), adDirOffset, _jac.lowerBandwidth(), _jac);
}

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN

/**
 * @brief Compares the analytical Jacobian with a Jacobian derived by AD
 * @details The analytical Jacobian is assumed to be stored in the corresponding band matrices.
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const
{
	Indexer idxr(_disc);

	const double maxDiff = ad::compareBandedJacobianWithAd(adRes + idxr.offsetC(), adDirOffset, _jac.lowerBandwidth(), _jac);
	LOG(Debug) << "AD dir offset: " << adDirOffset << " DiagDirCol: " << _jac.lowerBandwidth() << " MaxDiff: " << maxDiff;
}

#endif

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidual);

	// Evaluate residual do not compute Jacobian or parameter sensitivities
	return residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem);
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidual);

	// Evaluate residual, use AD for Jacobian if required but do not evaluate parameter derivatives
	return residual(simTime, simState, res, adJac, threadLocalMem, true, false);
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res,
	const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem, bool updateJacobian, bool paramSensitivity)
{
	if (updateJacobian)
	{
		_factorizeJacobian = true;

#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
		if (_analyticJac)
		{
			if (paramSensitivity)
			{
				const int retCode = residualImpl<double, active, active, true>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, adJac.adRes, threadLocalMem);

				// Copy AD residuals to original residuals vector
				if (res)
					ad::copyFromAd(adJac.adRes, res, numDofs());

				return retCode;
			}
			else
				return residualImpl<double, double, double, true>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem);
		}
		else
		{
			// Compute Jacobian via AD

			// Copy over state vector to AD state vector (without changing directional values to keep seed vectors)
			// and initialize residuals with zero (also resetting directional values)
			ad::copyToAd(simState.vecStateY, adJac.adY, numDofs());
			// @todo Check if this is necessary
			ad::resetAd(adJac.adRes, numDofs());

			// Evaluate with AD enabled
			int retCode = 0;
			if (paramSensitivity)
				retCode = residualImpl<active, active, active, false>(simTime.t, simTime.secIdx, adJac.adY, simState.vecStateYdot, adJac.adRes, threadLocalMem);
			else
				retCode = residualImpl<active, active, double, false>(simTime.t, simTime.secIdx, adJac.adY, simState.vecStateYdot, adJac.adRes, threadLocalMem);

			// Copy AD residuals to original residuals vector
			if (res)
				ad::copyFromAd(adJac.adRes, res, numDofs());

			// Extract Jacobian
			extractJacobianFromAD(adJac.adRes, adJac.adDirOffset);

			return retCode;
		}
#else
		// Compute Jacobian via AD

		// Copy over state vector to AD state vector (without changing directional values to keep seed vectors)
		// and initialize residuals with zero (also resetting directional values)
		ad::copyToAd(simState.vecStateY, adJac.adY, numDofs());
		// @todo Check if this is necessary
		ad::resetAd(adJac.adRes, numDofs());

		// Evaluate with AD enabled
		int retCode = 0;
		if (paramSensitivity)
			retCode = residualImpl<active, active, active, false>(simTime.t, simTime.secIdx, adJac.adY, simState.vecStateYdot, adJac.adRes, threadLocalMem);
		else
			retCode = residualImpl<active, active, double, false>(simTime.t, simTime.secIdx, adJac.adY, simState.vecStateYdot, adJac.adRes, threadLocalMem);

		// Only do comparison if we have a residuals vector (which is not always the case)
		if (res)
		{
			// Evaluate with analytical Jacobian which is stored in the band matrices
			retCode = residualImpl<double, double, double, true>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem);

			// Compare AD with anaytic Jacobian
			checkAnalyticJacobianAgainstAd(adJac.adRes, adJac.adDirOffset);
		}

		// Extract Jacobian
		extractJacobianFromAD(adJac.adRes, adJac.adDirOffset);

		return retCode;
#endif
	}
	else
	{
		if (paramSensitivity)
		{
			// initialize residuals with zero
			// @todo Check if this is necessary
			ad::resetAd(adJac.adRes, numDofs());

			const int retCode = residualImpl<double, active, active, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, adJac.adRes, threadLocalMem);

			// Copy AD residuals to original residuals vector
			if (res)
				ad::copyFromAd(adJac.adRes, res, numDofs());

			return retCode;
		}
		else
			return residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem);
	}
}

template <typename ConvDispOperator>
template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int LumpedRateModelWithoutPores<ConvDispOperator>::residualImpl(double t, unsigned int secIdx, StateType const* const y, double const* const yDot, ResidualType* const res, util::ThreadLocalStorage& threadLocalMem)
{
	ConvOpResidual<ConvDispOperator, StateType, ResidualType, ParamType, wantJac>::call(this, _convDispOp, t, secIdx, y, yDot, res, _jac);

	Indexer idxr(_disc);

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(_disc.nCol), [&](std::size_t col)
#else
	for (unsigned int col = 0; col < _disc.nCol; ++col)
#endif
	{
		StateType const* const localY = y + idxr.offsetC() + idxr.strideColCell() * col;
		ResidualType* const localRes = res + idxr.offsetC() + idxr.strideColCell() * col;
		double const* const localYdot = yDot ? yDot + idxr.offsetC() + idxr.strideColCell() * col : nullptr;

		const parts::cell::CellParameters cellResParams
			{
				_disc.nComp,
				_disc.nBound,
				_disc.boundOffset,
				_disc.strideBound,
				_binding[0]->reactionQuasiStationarity(),
				_totalPorosity,
				nullptr,
				_binding[0],
				(_dynReaction[0] && (_dynReaction[0]->numReactionsCombined() > 0)) ? _dynReaction[0] : nullptr
			};

		// Midpoint of current column cell (z coordinate) - needed in externally dependent adsorption kinetic
		const double z = _convDispOp.relativeCoordinate(col);

		parts::cell::residualKernel<StateType, ResidualType, ParamType, parts::cell::CellParameters, linalg::BandMatrix::RowIterator, wantJac, false>(
			t, secIdx, ColumnPosition{z, 0.0, 0.0}, localY, localYdot, localRes, _jac.row(col * idxr.strideColCell()), cellResParams, threadLocalMem.get()
		);

	} CADET_PARFOR_END;

	BENCH_STOP(_timerResidualPar);

	// Handle inlet DOFs, which are simply copied to res
	for (unsigned int i = 0; i < _disc.nComp; ++i)
	{
		res[i] = y[i];
	}

	return 0;
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidualSens);

	// Evaluate residual for all parameters using AD in vector mode and at the same time update the
	// Jacobian (in one AD run, if analytic Jacobians are disabled)
	return residual(simTime, simState, nullptr, adJac, threadLocalMem, true, true);
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidualSens);

	// Evaluate residual for all parameters using AD in vector mode
	return residualImpl<double, active, active, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, adRes, threadLocalMem);
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState,
	const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS, active const* adRes,
	double* const tmp1, double* const tmp2, double* const tmp3)
{
	BENCH_SCOPE(_timerResidualSens);

	// tmp1 stores result of (dF / dy) * s
	// tmp2 stores result of (dF / dyDot) * sDot

	for (std::size_t param = 0; param < yS.size(); ++param)
	{
		// Directional derivative (dF / dy) * s
		multiplyWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{nullptr, nullptr}, yS[param], 1.0, 0.0, tmp1);

		// Directional derivative (dF / dyDot) * sDot
		multiplyWithDerivativeJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{nullptr, nullptr}, ySdot[param], tmp2);

		double* const ptrResS = resS[param];

		BENCH_START(_timerResidualSensPar);

		// Complete sens residual is the sum:
		// TODO: Chunk TBB loop
#ifdef CADET_PARALLELIZE
		tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(numDofs()), [&](std::size_t i)
#else
		for (unsigned int i = 0; i < numDofs(); ++i)
#endif
		{
			ptrResS[i] = tmp1[i] + tmp2[i] + adRes[i].getADValue(param);
		} CADET_PARFOR_END;

		BENCH_STOP(_timerResidualSensPar);
	}

	return 0;
}

/**
 * @brief Multiplies the given vector with the system Jacobian (i.e., @f$ \frac{\partial F}{\partial y}\left(t, y, \dot{y}\right) @f$)
 * @details Actually, the operation @f$ z = \alpha \frac{\partial F}{\partial y} x + \beta z @f$ is performed.
 *
 *          Note that residual() or one of its cousins has to be called with the requested point @f$ (t, y, \dot{y}) @f$ once
 *          before calling multiplyWithJacobian() as this implementation ignores the given @f$ (t, y, \dot{y}) @f$.
 * @param [in] simTime Current simulation time point
 * @param [in] simState Simulation state vectors
 * @param [in] yS Vector @f$ x @f$ that is transformed by the Jacobian @f$ \frac{\partial F}{\partial y} @f$
 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ \frac{\partial F}{\partial y} @f$
 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ z @f$
 * @param [in,out] ret Vector @f$ z @f$ which stores the result of the operation
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret)
{
	Indexer idxr(_disc);

	// Handle identity matrix of inlet DOFs
	for (unsigned int i = 0; i < _disc.nComp; ++i)
	{
		ret[i] = alpha * yS[i] + beta * ret[i];
	}

	// Main Jacobian
	_jac.multiplyVector(yS + idxr.offsetC(), alpha, beta, ret + idxr.offsetC());

	// Map inlet DOFs to the column inlet (first bulk cells)
	_jacInlet.multiplyAdd(yS, ret + idxr.offsetC(), alpha);
}

/**
 * @brief Multiplies the time derivative Jacobian @f$ \frac{\partial F}{\partial \dot{y}}\left(t, y, \dot{y}\right) @f$ with a given vector
 * @details The operation @f$ z = \frac{\partial F}{\partial \dot{y}} x @f$ is performed.
 *          The matrix-vector multiplication is performed matrix-free (i.e., no matrix is explicitly formed).
 * @param [in] simTime Current simulation time point
 * @param [in] simState Simulation state vectors
 * @param [in] sDot Vector @f$ x @f$ that is transformed by the Jacobian @f$ \frac{\partial F}{\partial \dot{y}} @f$
 * @param [out] ret Vector @f$ z @f$ which stores the result of the operation
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret)
{
	Indexer idxr(_disc);
	const double invBeta = (1.0 / static_cast<double>(_totalPorosity) - 1.0);

	_convDispOp.multiplyWithDerivativeJacobian(simTime, sDot, ret);
	for (unsigned int col = 0; col < _disc.nCol; ++col)
	{
		const unsigned int localOffset = idxr.offsetC() + col * idxr.strideColCell();
		double const* const localSdot = sDot + localOffset;
		double* const localRet = ret + localOffset;

		parts::cell::multiplyWithDerivativeJacobianKernel<false>(localSdot, localRet, _disc.nComp, _disc.nBound, _disc.boundOffset, _disc.strideBound, _binding[0]->reactionQuasiStationarity(), 1.0, invBeta);
	}

	// Handle inlet DOFs (all algebraic)
	std::fill_n(ret, _disc.nComp, 0.0);
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::setExternalFunctions(IExternalFunction** extFuns, unsigned int size)
{
	if (_binding[0])
		_binding[0]->setExternalFunctions(extFuns, size);
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithoutPores<ConvDispOperator>::localOutletComponentIndex(unsigned int port) const CADET_NOEXCEPT
{
	// Inlets are duplicated so need to be accounted for
	if (_convDispOp.forwardFlow())
		// Forward Flow: outlet is last cell
		return _disc.nComp + (_disc.nCol - 1) * (_disc.nComp + _disc.strideBound);
	else
		// Backward flow: Outlet is first cell
		return _disc.nComp;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithoutPores<ConvDispOperator>::localInletComponentIndex(unsigned int port) const CADET_NOEXCEPT
{
	return 0;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithoutPores<ConvDispOperator>::localOutletComponentStride(unsigned int port) const CADET_NOEXCEPT
{
	return 1;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithoutPores<ConvDispOperator>::localInletComponentStride(unsigned int port) const CADET_NOEXCEPT
{
	return 1;
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut)
{
	// @todo Write this function
}

/**
 * @brief Computes the solution of the linear system involving the system Jacobian
 * @details The system \f[ \left( \frac{\partial F}{\partial y} + \alpha \frac{\partial F}{\partial \dot{y}} \right) x = b \f]
 *          has to be solved. The right hand side \f$ b \f$ is given by @p rhs, the Jacobians are evaluated at the
 *          point \f$(y, \dot{y})\f$ given by @p y and @p yDot. The residual @p res at this point, \f$ F(t, y, \dot{y}) \f$,
 *          may help with this. Error weights (see IDAS guide) are given in @p weight. The solution is returned in @p rhs.
 *
 * @param [in] t Current time point
 * @param [in] alpha Value of \f$ \alpha \f$ (arises from BDF time discretization)
 * @param [in] outerTol Error tolerance for the solution of the linear system from outer Newton iteration
 * @param [in,out] rhs On entry the right hand side of the linear equation system, on exit the solution
 * @param [in] weight Vector with error weights
 * @param [in] simState State of the simulation (state vector and its time derivatives) at which the Jacobian is evaluated
 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
 */
template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::linearSolve(double t, double alpha, double outerTol, double* const rhs, double const* const weight,
	const ConstSimulationState& simState)
{
	BENCH_SCOPE(_timerLinearSolve);

	Indexer idxr(_disc);

	bool success = true;

	// Factorize Jacobian only if required
	if (_factorizeJacobian)
	{
		// Assemble
		assembleDiscretizedJacobian(alpha, idxr);

		// Factorize
		success = _jacDisc.factorize();
		if (cadet_unlikely(!success))
		{
			LOG(Error) << "Factorize() failed for par block";
		}

		// Do not factorize again at next call without changed Jacobians
		_factorizeJacobian = false;
	}

	// Handle inlet DOFs
	_jacInlet.multiplySubtract(rhs, rhs + idxr.offsetC());

	// Solve
	const bool result = _jacDisc.solve(rhs + idxr.offsetC());
	if (cadet_unlikely(!result))
	{
		LOG(Error) << "Solve() failed for bulk block";
	}

	return (success && result) ? 0 : 1;;
}

/**
 * @brief Assembles the Jacobian of the time-discretized equations
 * @details The system \f[ \left( \frac{\partial F}{\partial y} + \alpha \frac{\partial F}{\partial \dot{y}} \right) x = b \f]
 *          has to be solved. The system Jacobian of the original equations,
 *          \f[ \frac{\partial F}{\partial y}, \f]
 *          is already computed (by AD or manually in residualImpl() with @c wantJac = true). This function is responsible
 *          for adding
 *          \f[ \alpha \frac{\partial F}{\partial \dot{y}} \f]
 *          to the system Jacobian, which yields the Jacobian of the time-discretized equations
 *          \f[ F\left(t, y_0, \sum_{k=0}^N \alpha_k y_k \right) = 0 \f]
 *          when a BDF method is used. The time integrator needs to solve this equation for @f$ y_0 @f$, which requires
 *          the solution of the linear system mentioned above (@f$ \alpha_0 = \alpha @f$ given in @p alpha).
 *
 * @param [in] alpha Value of \f$ \alpha \f$ (arises from BDF time discretization)
 * @param [in] idxr Indexer
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::assembleDiscretizedJacobian(double alpha, const Indexer& idxr)
{
	// Copy normal matrix over to factorizable matrix
	_jacDisc.copyOver(_jac);

	// Handle transport equations (dc_i / dt terms)
	_convDispOp.addTimeDerivativeToJacobian(alpha, _jacDisc);

	// Add time derivatives to cells
	const double invBeta = 1.0 / static_cast<double>(_totalPorosity) - 1.0;
	linalg::FactorizableBandMatrix::RowIterator jac = _jacDisc.row(0);
	for (unsigned int j = 0; j < _disc.nCol; ++j)
	{
		addTimeDerivativeToJacobianCell(jac, idxr, alpha, invBeta);
	}
}

/**
 * @brief Adds Jacobian @f$ \frac{\partial F}{\partial \dot{y}} @f$ to cell of system Jacobian
 * @details Actually adds @f$ \alpha \frac{\partial F}{\partial \dot{y}} @f$, which is useful
 *          for constructing the linear system in BDF time discretization.
 * @param [in,out] jac On entry, RowIterator pointing to the beginning of a cell;
 *                     on exit, the iterator points to the end of the cell
 * @param [in] idxr Indexer
 * @param [in] alpha Value of \f$ \alpha \f$ (arises from BDF time discretization)
 * @param [in] invBeta Inverse porosity term @f$\frac{1}{\beta}@f$
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::addTimeDerivativeToJacobianCell(linalg::FactorizableBandMatrix::RowIterator& jac, const Indexer& idxr, double alpha, double invBeta) const
{
	// Mobile phase
	for (int comp = 0; comp < static_cast<int>(_disc.nComp); ++comp, ++jac)
	{
		// dc / dt is handled by convection dispersion operator

		// Add derivative with respect to dq / dt to Jacobian
		for (int i = 0; i < static_cast<int>(_disc.nBound[comp]); ++i)
		{
			// Index explanation:
			//   -comp -> go back to beginning of liquid phase
			//   + strideColLiquid() skip to solid phase
			//   + offsetBoundComp() jump to component (skips all bound states of previous components)
			//   + i go to current bound state
			jac[idxr.strideColLiquid() - comp + idxr.offsetBoundComp(comp) + i] += alpha * invBeta;
		}
	}

	// Solid phase
	int const* const qsReaction = _binding[0]->reactionQuasiStationarity();
	for (unsigned int bnd = 0; bnd < _disc.strideBound; ++bnd, ++jac)
	{
		// Add derivative with respect to dynamic states to Jacobian
		if (qsReaction[bnd])
			continue;

		// Add derivative with respect to dq / dt to Jacobian
		jac[0] += alpha;
	}
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::applyInitialCondition(const SimulationState& simState) const
{
	Indexer idxr(_disc);

	// Check whether full state vector is available as initial condition
	if (!_initState.empty())
	{
		std::fill(simState.vecStateY, simState.vecStateY + idxr.offsetC(), 0.0);
		std::copy(_initState.data(), _initState.data() + numPureDofs(), simState.vecStateY + idxr.offsetC());

		if (!_initStateDot.empty())
		{
			std::fill(simState.vecStateYdot, simState.vecStateYdot + idxr.offsetC(), 0.0);
			std::copy(_initStateDot.data(), _initStateDot.data() + numPureDofs(), simState.vecStateYdot + idxr.offsetC());
		}
		else
			std::fill(simState.vecStateYdot, simState.vecStateYdot + numDofs(), 0.0);

		return;
	}

	double* const stateYbulk = simState.vecStateY + idxr.offsetC();

	// Loop over column cells
	for (unsigned int col = 0; col < _disc.nCol; ++col)
	{
		const unsigned int localOffset = col * idxr.strideColCell();

		// Loop over components in cell
		for (unsigned comp = 0; comp < _disc.nComp; ++comp)
			stateYbulk[localOffset + comp * idxr.strideColComp()] = static_cast<double>(_initC[comp]);

		// Initialize q
		for (unsigned int bnd = 0; bnd < _disc.strideBound; ++bnd)
			stateYbulk[localOffset + idxr.strideColLiquid() + bnd] = static_cast<double>(_initQ[bnd]);
	}
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::readInitialCondition(IParameterProvider& paramProvider)
{
	_initState.clear();
	_initStateDot.clear();

	// Check if INIT_STATE is present
	if (paramProvider.exists("INIT_STATE"))
	{
		const std::vector<double> initState = paramProvider.getDoubleArray("INIT_STATE");
		_initState = std::vector<double>(initState.begin(), initState.begin() + numPureDofs());

		// Check if INIT_STATE contains the full state and its time derivative
		if (initState.size() >= 2 * numPureDofs())
			_initStateDot = std::vector<double>(initState.begin() + numPureDofs(), initState.begin() + 2 * numPureDofs());
		return;
	}

	const std::vector<double> initC = paramProvider.getDoubleArray("INIT_C");
	std::vector<double> initQ;

	if (paramProvider.exists("INIT_Q"))
		initQ = paramProvider.getDoubleArray("INIT_Q");

	if (initC.size() < _disc.nComp)
		throw InvalidParameterException("INIT_C does not contain enough values for all components");

	if ((_disc.strideBound > 0) && (initQ.size() < _disc.strideBound))
		throw InvalidParameterException("INIT_Q does not contain enough values for all bound states");

	ad::copyToAd(initC.data(), _initC.data(), _disc.nComp);
	if (!initQ.empty())
		ad::copyToAd(initQ.data(), _initQ.data(), _disc.strideBound);
}

/**
 * @brief Computes consistent initial values (state variables without their time derivatives)
 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] the initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ have
 *          to be consistent. This functions updates the initial state \f$ y_0 \f$ and overwrites the time
 *          derivative \f$ \dot{y}_0 \f$ such that they are consistent.
 *
 *          The process works in two steps:
 *          <ol>
 *              <li>Solve all algebraic equations in the model (e.g., quasi-stationary isotherms, reaction equilibria).</li>
 *              <li>Compute the time derivatives of the state @f$ \dot{y} @f$ such that the residual is 0.
 *                 However, because of the algebraic equations, we need additional conditions to fully determine
 *                 @f$ \dot{y}@f$. By differentiating the algebraic equations with respect to time, we get the
 *                 missing linear equations (recall that the state vector @f$ y @f$ is fixed).
 *
 *     The right hand side of the linear system is given by the negative residual without contribution
 *     of @f$ \dot{y} @f$ for differential equations and 0 for algebraic equations
 *     (@f$ -\frac{\partial F}{\partial t}@f$, to be more precise).</li>
 *          </ol>
 *
 *     This function performs step 1. See consistentInitialTimeDerivative() for step 2.
 *
 * 	   This function is to be used with consistentInitialTimeDerivative(). Do not mix normal and lean
 *     consistent initialization!
 *
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in,out] vecStateY State vector with initial values that are to be updated for consistency
 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
 * @param [in] errorTol Error tolerance for algebraic equations
 * @todo Decrease amount of allocated memory by partially using temporary vectors (state and Schur complement)
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::consistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerConsistentInit);

	Indexer idxr(_disc);

	// Step 1: Solve algebraic equations
	if (!_binding[0]->hasQuasiStationaryReactions())
		return;

	// Copy quasi-stationary binding mask to a local array that also includes the mobile phase
	std::vector<int> qsMask(_disc.nComp + _disc.strideBound, false);
	int const* const qsMaskSrc = _binding[0]->reactionQuasiStationarity();
	std::copy_n(qsMaskSrc, _disc.strideBound, qsMask.data() + _disc.nComp);

	// Activate mobile phase components that have at least one active bound state
	unsigned int bndStartIdx = 0;
	unsigned int numActiveComp = 0;
	for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
	{
		for (unsigned int bnd = 0; bnd < _disc.nBound[comp]; ++bnd)
		{
			if (qsMaskSrc[bndStartIdx + bnd])
			{
				++numActiveComp;
				qsMask[comp] = true;
				break;
			}
		}

		bndStartIdx += _disc.nBound[comp];
	}

	const linalg::ConstMaskArray mask{qsMask.data(), static_cast<int>(_disc.nComp + _disc.strideBound)};
	const int probSize = linalg::numMaskActive(mask);

	//Problem capturing variables here
#ifdef CADET_PARALLELIZE
	BENCH_SCOPE(_timerConsistentInitPar);
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(_disc.nCol), [&](std::size_t col)
#else
	for (unsigned int col = 0; col < _disc.nCol; ++col)
#endif
	{
		LinearBufferAllocator tlmAlloc = threadLocalMem.get();

		// Reuse memory of band matrix for dense matrix
		linalg::DenseMatrixView fullJacobianMatrix(_jacDisc.data() + col * _disc.strideBound * _disc.strideBound, nullptr, mask.len, mask.len);

		// Midpoint of current column cell (z coordinate) - needed in externally dependent adsorption kinetic
		const double z = _convDispOp.relativeCoordinate(col);

		// Get workspace memory
		BufferedArray<double> nonlinMemBuffer = tlmAlloc.array<double>(_nonlinearSolver->workspaceSize(probSize));
		double* const nonlinMem = static_cast<double*>(nonlinMemBuffer);

		BufferedArray<double> solutionBuffer = tlmAlloc.array<double>(probSize);
		double* const solution = static_cast<double*>(solutionBuffer);

		BufferedArray<double> fullResidualBuffer = tlmAlloc.array<double>(mask.len);
		double* const fullResidual = static_cast<double*>(fullResidualBuffer);

		BufferedArray<double> fullXBuffer = tlmAlloc.array<double>(mask.len);
		double* const fullX = static_cast<double*>(fullXBuffer);

		BufferedArray<double> jacobianMemBuffer = tlmAlloc.array<double>(probSize * probSize);
		double* const jacobianMem = static_cast<double*>(jacobianMemBuffer);

		BufferedArray<double> conservedQuantsBuffer = tlmAlloc.array<double>(numActiveComp);
		double* const conservedQuants = static_cast<double*>(conservedQuantsBuffer);

		linalg::DenseMatrixView jacobianMatrix(jacobianMem, _jacDisc.pivot() + col * _disc.strideBound, probSize, probSize);
		const parts::cell::CellParameters cellResParams
			{
				_disc.nComp,
				_disc.nBound,
				_disc.boundOffset,
				_disc.strideBound,
				_binding[0]->reactionQuasiStationarity(),
				_totalPorosity,
				nullptr,
				_binding[0],
				(_dynReaction[0] && (_dynReaction[0]->numReactionsCombined() > 0)) ? _dynReaction[0] : nullptr
			};

		const int localOffsetToCell = idxr.offsetC() + col * idxr.strideColCell();
		const int localOffsetInCell = idxr.strideColLiquid();

		// Get pointer to q variables in cell
		double* const qShell = vecStateY + localOffsetToCell + localOffsetInCell;
		active* const localAdRes = adJac.adRes ? adJac.adRes + localOffsetToCell : nullptr;
		active* const localAdY = adJac.adY ? adJac.adY + localOffsetToCell : nullptr;

		const ColumnPosition colPos{z, 0.0, 0.0};

		// Determine whether nonlinear solver is required
		if (!_binding[0]->preConsistentInitialState(simTime.t, simTime.secIdx, colPos, qShell, qShell - localOffsetInCell, tlmAlloc))
			CADET_PAR_CONTINUE;

		// Extract initial values from current state
		linalg::selectVectorSubset(qShell - _disc.nComp, mask, solution);

		// Save values of conserved moieties
		const double epsQ = 1.0 - static_cast<double>(_totalPorosity);
		linalg::conservedMoietiesFromPartitionedMask(mask, _disc.nBound, _disc.nComp, qShell - _disc.nComp, conservedQuants, static_cast<double>(_totalPorosity), epsQ);

		std::function<bool(double const* const, linalg::detail::DenseMatrixBase&)> jacFunc;
		if (localAdY && localAdRes)
		{
			jacFunc = [&](double const* const x, linalg::detail::DenseMatrixBase& mat)
			{
				// Copy over state vector to AD state vector (without changing directional values to keep seed vectors)
				// and initialize residuals with zero (also resetting directional values)
				ad::copyToAd(qShell - _disc.nComp, localAdY, mask.len);
				// @todo Check if this is necessary
				ad::resetAd(localAdRes, mask.len);

				// Prepare input vector by overwriting masked items
				linalg::applyVectorSubset(x, mask, localAdY);

				// Call residual function
				parts::cell::residualKernel<active, active, double, parts::cell::CellParameters, linalg::DenseBandedRowIterator, false, true>(
					simTime.t, simTime.secIdx, colPos, localAdY, nullptr, localAdRes, fullJacobianMatrix.row(0), cellResParams, tlmAlloc
				);

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
				std::copy_n(qShell - _disc.nComp, mask.len, fullX);
				linalg::applyVectorSubset(x, mask, fullX);

				// Compute analytic Jacobian
				parts::cell::residualKernel<double, double, double, parts::cell::CellParameters, linalg::DenseBandedRowIterator, true, true>(
					simTime.t, simTime.secIdx, colPos, fullX, nullptr, fullResidual, fullJacobianMatrix.row(0), cellResParams, tlmAlloc
				);

				// Compare
				const double diff = ad::compareDenseJacobianWithBandedAd(
					adJac.adRes + idxr.offsetC(), col * idxr.strideColCell(), adJac.adDirOffset, _jac.lowerBandwidth(),
					_jac.lowerBandwidth(), _jac.upperBandwidth(), fullJacobianMatrix
				);
				LOG(Debug) << "MaxDiff: " << diff;
#endif

				// Extract Jacobian from AD
				ad::extractDenseJacobianFromBandedAd(
					adJac.adRes + idxr.offsetC(), col * idxr.strideColCell(), adJac.adDirOffset, _jac.lowerBandwidth(),
					_jac.lowerBandwidth(), _jac.upperBandwidth(), fullJacobianMatrix
				);

				// Extract Jacobian from full Jacobian
				mat.setAll(0.0);
				linalg::copyMatrixSubset(fullJacobianMatrix, mask, mask, mat);

				// Replace upper part with conservation relations
				mat.submatrixSetAll(0.0, 0, 0, numActiveComp, probSize);

				unsigned int bndIdx = 0;
				unsigned int rIdx = 0;
				unsigned int bIdx = 0;
				for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				{
					if (!mask.mask[comp])
					{
						bndIdx += _disc.nBound[comp];
						continue;
					}

					mat.native(rIdx, rIdx) = static_cast<double>(_totalPorosity);

					for (unsigned int bnd = 0; bnd < _disc.nBound[comp]; ++bnd, ++bndIdx)
					{
						if (mask.mask[bndIdx])
						{
							mat.native(rIdx, bIdx + numActiveComp) = epsQ;
							++bIdx;
						}
					}

					++rIdx;
				}

				return true;
			};
		}
		else
		{
			jacFunc = [&](double const* const x, linalg::detail::DenseMatrixBase& mat)
			{
				// Prepare input vector by overwriting masked items
				std::copy_n(qShell - _disc.nComp, mask.len, fullX);
				linalg::applyVectorSubset(x, mask, fullX);

				// Call residual function
				parts::cell::residualKernel<double, double, double, parts::cell::CellParameters, linalg::DenseBandedRowIterator, true, true>(
					simTime.t, simTime.secIdx, colPos, fullX, nullptr, fullResidual, fullJacobianMatrix.row(0), cellResParams, tlmAlloc
				);

				// Extract Jacobian from full Jacobian
				mat.setAll(0.0);
				linalg::copyMatrixSubset(fullJacobianMatrix, mask, mask, mat);

				// Replace upper part with conservation relations
				mat.submatrixSetAll(0.0, 0, 0, numActiveComp, probSize);

				unsigned int bndIdx = 0;
				unsigned int rIdx = 0;
				unsigned int bIdx = 0;
				for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				{
					if (!mask.mask[comp])
					{
						bndIdx += _disc.nBound[comp];
						continue;
					}

					mat.native(rIdx, rIdx) = static_cast<double>(_totalPorosity);

					for (unsigned int bnd = 0; bnd < _disc.nBound[comp]; ++bnd, ++bndIdx)
					{
						if (mask.mask[bndIdx])
						{
							mat.native(rIdx, bIdx + numActiveComp) = epsQ;
							++bIdx;
						}
					}

					++rIdx;
				}

				return true;
			};
		}

		// Apply nonlinear solver
		_nonlinearSolver->solve(
			[&](double const* const x, double* const r)
			{
				// Prepare input vector by overwriting masked items
				std::copy_n(qShell - _disc.nComp, mask.len, fullX);
				linalg::applyVectorSubset(x, mask, fullX);

				// Call residual function
				parts::cell::residualKernel<double, double, double, parts::cell::CellParameters, linalg::DenseBandedRowIterator, false, true>(
					simTime.t, simTime.secIdx, colPos, fullX, nullptr, fullResidual, fullJacobianMatrix.row(0), cellResParams, tlmAlloc
				);

				// Extract values from residual
				linalg::selectVectorSubset(fullResidual, mask, r);

				// Calculate residual of conserved moieties
				std::fill_n(r, numActiveComp, 0.0);
				unsigned int bndIdx = _disc.nComp;
				unsigned int rIdx = 0;
				unsigned int bIdx = 0;
				for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				{
					if (!mask.mask[comp])
					{
						bndIdx += _disc.nBound[comp];
						continue;
					}

					r[rIdx] = static_cast<double>(_totalPorosity) * x[rIdx] - conservedQuants[rIdx];

					for (unsigned int bnd = 0; bnd < _disc.nBound[comp]; ++bnd, ++bndIdx)
					{
						if (mask.mask[bndIdx])
						{
							r[rIdx] += epsQ * x[bIdx + numActiveComp];
							++bIdx;
						}
					}

					++rIdx;
				}

				return true;
			},
			jacFunc, errorTol, solution, nonlinMem, jacobianMatrix, probSize);

		// Apply solution
		linalg::applyVectorSubset(solution, mask, qShell - idxr.strideColLiquid());

		// Refine / correct solution
		_binding[0]->postConsistentInitialState(simTime.t, simTime.secIdx, colPos, qShell, qShell - idxr.strideColLiquid(), tlmAlloc);
	} CADET_PARFOR_END;
}

/**
 * @brief Computes consistent initial time derivatives
 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] the initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ have
 *          to be consistent. This functions updates the initial state \f$ y_0 \f$ and overwrites the time
 *          derivative \f$ \dot{y}_0 \f$ such that they are consistent.
 *
 *          The process works in two steps:
 *          <ol>
 *              <li>Solve all algebraic equations in the model (e.g., quasi-stationary isotherms, reaction equilibria).</li>
 *              <li>Compute the time derivatives of the state @f$ \dot{y} @f$ such that the residual is 0.
 *                 However, because of the algebraic equations, we need additional conditions to fully determine
 *                 @f$ \dot{y}@f$. By differentiating the algebraic equations with respect to time, we get the
 *                 missing linear equations (recall that the state vector @f$ y @f$ is fixed).
 *
 *     The right hand side of the linear system is given by the negative residual without contribution
 *     of @f$ \dot{y} @f$ for differential equations and 0 for algebraic equations
 *     (@f$ -\frac{\partial F}{\partial t}@f$, to be more precise).</li>
 *          </ol>
 *
 *     This function performs step 2. See consistentInitialState() for step 1.
 *
 * 	   This function is to be used with consistentInitialState(). Do not mix normal and lean
 *     consistent initialization!
 *
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in] vecStateY Consistently initialized state vector
 * @param [in,out] vecStateYdot On entry, residual without taking time derivatives into account. On exit, consistent state time derivatives.
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::consistentInitialTimeDerivative(const SimulationTime& simTime, double const* vecStateY, double* const vecStateYdot, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerConsistentInit);

	Indexer idxr(_disc);

	// Step 2: Compute the correct time derivative of the state vector

	// Note that the residual has not been negated, yet. We will do that now.
	for (unsigned int i = 0; i < numDofs(); ++i)
		vecStateYdot[i] = -vecStateYdot[i];

	_jacDisc.setAll(0.0);

	// Handle transport equations (dc_i / dt terms)
	_convDispOp.addTimeDerivativeToJacobian(1.0, _jacDisc);

	const double invBeta = 1.0 / static_cast<double>(_totalPorosity) - 1.0;
	double* const dFluxDt = _tempState + idxr.offsetC();
	LinearBufferAllocator tlmAlloc = threadLocalMem.get();
	for (unsigned int col = 0; col < _disc.nCol; ++col)
	{
		// Assemble
		linalg::FactorizableBandMatrix::RowIterator jac = _jacDisc.row(idxr.strideColCell() * col);

		// Mobile and solid phase (advances jac accordingly)
		addTimeDerivativeToJacobianCell(jac, idxr, 1.0, invBeta);

		// Stationary phase
		if (!_binding[0]->hasQuasiStationaryReactions())
			continue;

		// Midpoint of current column cell (z coordinate) - needed in externally dependent adsorption kinetic
		const double z = _convDispOp.relativeCoordinate(col);

		// Get iterators to beginning of solid phase
		linalg::BandMatrix::RowIterator jacSolidOrig = _jac.row(idxr.strideColCell() * col + idxr.strideColLiquid());
		linalg::FactorizableBandMatrix::RowIterator jacSolid = jac - idxr.strideColBound();

		int const* const mask = _binding[0]->reactionQuasiStationarity();
		double* const qShellDot = vecStateYdot + idxr.offsetC() + col * idxr.strideColCell() + idxr.strideColLiquid();

		// Obtain derivative of fluxes wrt. time
		std::fill_n(dFluxDt, _disc.strideBound, 0.0);
		if (_binding[0]->dependsOnTime())
		{
			_binding[0]->timeDerivativeQuasiStationaryFluxes(simTime.t, simTime.secIdx, ColumnPosition{z, 0.0, 0.0},
				qShellDot - _disc.nComp, qShellDot, dFluxDt, tlmAlloc);
		}

		// Copy row from original Jacobian and set right hand side
		for (unsigned int i = 0; i < _disc.strideBound; ++i, ++jacSolid, ++jacSolidOrig)
		{
			if (!mask[i])
				continue;

			jacSolid.copyRowFrom(jacSolidOrig);
			qShellDot[i] = -dFluxDt[i];
		}
	}

	// Precondition
	double* const scaleFactors = _tempState + idxr.offsetC();
	_jacDisc.rowScaleFactors(scaleFactors);
	_jacDisc.scaleRows(scaleFactors);

	// Factorize
	const bool result = _jacDisc.factorize();
	if (!result)
	{
		LOG(Error) << "Factorize() failed for par block";
	}

	const bool result2 = _jacDisc.solve(scaleFactors, vecStateYdot + idxr.offsetC());
	if (!result2)
	{
		LOG(Error) << "Solve() failed for par block";
	}
}

/**
 * @brief Computes approximately / partially consistent initial values (state variables without their time derivatives)
 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] the initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ have
 *          to be consistent. This functions updates the initial state \f$ y_0 \f$ and overwrites the time
 *          derivative \f$ \dot{y}_0 \f$ such that they are consistent.
 *
 *          This function performs a relaxed consistent initialization: Only parts of the vectors are updated
 *          and, hence, consistency is not guaranteed. Since there is less work to do, it is probably faster than
 *          the standard process represented by consistentInitialState().
 *
 *          The process works in two steps:
 *          <ol>
 *              <li>Keep state and time derivative vectors as they are (i.e., do not solve algebraic equations).</li>
 *              <li>Compute the time derivatives of the state @f$ \dot{y} @f$ such that the residual is 0 for the
 *              mobile phase variables.</li>
 *          </ol>
 *     This function performs step 1. See leanConsistentInitialTimeDerivative() for step 2.
 *
 * 	   This function is to be used with leanConsistentInitialTimeDerivative(). Do not mix normal and lean
 *     consistent initialization!
 *
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in,out] vecStateY State vector with initial values that are to be updated for consistency
 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
 * @param [in] errorTol Error tolerance for algebraic equations
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::leanConsistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem)
{
}

/**
 * @brief Computes approximately / partially consistent initial time derivatives
 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] the initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ have
 *          to be consistent. This functions updates the initial state \f$ y_0 \f$ and overwrites the time
 *          derivative \f$ \dot{y}_0 \f$ such that they are consistent.
 *
 *          This function performs a relaxed consistent initialization: Only parts of the vectors are updated
 *          and, hence, consistency is not guaranteed. Since there is less work to do, it is probably faster than
 *          the standard process represented by consistentInitialState().
 *
 *          The process works in two steps:
 *          <ol>
 *              <li>Keep state and time derivative vectors as they are (i.e., do not solve algebraic equations).</li>
 *              <li>Compute the time derivatives of the state @f$ \dot{y} @f$ such that the residual is 0 for the
 *              mobile phase variables.</li>
 *          </ol>
 *     This function performs step 2. See leanConsistentInitialState() for step 1.
 *
 * 	   This function is to be used with leanConsistentInitialState(). Do not mix normal and lean
 *     consistent initialization!
 *
 * @param [in] t Current time point
 * @param [in] vecStateY (Lean) consistently initialized state vector
 * @param [in,out] vecStateYdot On entry, inconsistent state time derivatives. On exit, partially consistent state time derivatives.
 * @param [in] res On entry, residual without taking time derivatives into account. The data is overwritten during execution of the function.
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::leanConsistentInitialTimeDerivative(double t, double const* const vecStateY, double* const vecStateYdot, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerConsistentInit);

	Indexer idxr(_disc);

	// Step 2: Compute the correct time derivative of the state vector (only mobile phase DOFs)

	const double invBeta = (1.0 / static_cast<double>(_totalPorosity) - 1.0);
	for (unsigned int col = 0; col < _disc.nCol; ++col)
	{
		// Offset to current cell's c and q variables
		const unsigned int localOffset = idxr.offsetC() + col * idxr.strideColCell();
		const unsigned int localOffsetQ = localOffset + idxr.strideColLiquid();

		for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
		{
			// dq_{i,j} / dt is assumed to be fixed, so bring it on the right hand side
			for (unsigned int i = 0; i < _disc.nBound[comp]; ++i)
			{
				res[localOffset + comp] += invBeta * vecStateYdot[localOffsetQ + _disc.boundOffset[comp] + i];
			}

			vecStateYdot[localOffset + comp] = -res[localOffset + comp];
		}
	}
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::initializeSensitivityStates(const std::vector<double*>& vecSensY) const
{
	Indexer idxr(_disc);
	for (std::size_t param = 0; param < vecSensY.size(); ++param)
	{
		double* const stateYbulk = vecSensY[param] + idxr.offsetC();

		// Loop over column cells
		for (unsigned int col = 0; col < _disc.nCol; ++col)
		{
			const unsigned int localOffset = col * idxr.strideColCell();

			// Loop over components in cell
			for (unsigned comp = 0; comp < _disc.nComp; ++comp)
				stateYbulk[localOffset + comp * idxr.strideColComp()] = _initC[comp].getADValue(param);

			// Initialize q
			for (unsigned int bnd = 0; bnd < _disc.strideBound; ++bnd)
				stateYbulk[localOffset + idxr.strideColLiquid() + bnd] = _initQ[bnd].getADValue(param);
		}
	}
}

/**
 * @brief Computes consistent initial values and time derivatives of sensitivity subsystems
 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] and initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$,
 *          the sensitivity system for a parameter @f$ p @f$ reads
 *          \f[ \frac{\partial F}{\partial y}(t, y, \dot{y}) s + \frac{\partial F}{\partial \dot{y}}(t, y, \dot{y}) \dot{s} + \frac{\partial F}{\partial p}(t, y, \dot{y}) = 0. \f]
 *          The initial values of this linear DAE, @f$ s_0 = \frac{\partial y_0}{\partial p} @f$ and @f$ \dot{s}_0 = \frac{\partial \dot{y}_0}{\partial p} @f$
 *          have to be consistent with the sensitivity DAE. This functions updates the initial sensitivity\f$ s_0 \f$ and overwrites the time
 *          derivative \f$ \dot{s}_0 \f$ such that they are consistent.
 *
 *          The process follows closely the one of consistentInitialConditions() and, in fact, is a linearized version of it.
 *          This is necessary because the initial conditions of the sensitivity system \f$ s_0 \f$ and \f$ \dot{s}_0 \f$ are
 *          related to the initial conditions \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ of the original DAE by differentiating them
 *          with respect to @f$ p @f$: @f$ s_0 = \frac{\partial y_0}{\partial p} @f$ and @f$ \dot{s}_0 = \frac{\partial \dot{y}_0}{\partial p}. @f$
 *          <ol>
 *              <li>Solve all algebraic equations in the model (e.g., quasi-stationary isotherms, reaction equilibria).
 *                 Let @f$ \mathcal{I}_a @f$ be the index set of algebraic equations, then, at this point, we have
 *                 \f[ \left( \frac{\partial F}{\partial y}(t, y_0, \dot{y}_0) s + \frac{\partial F}{\partial p}(t, y_0, \dot{y}_0) \right)_{\mathcal{I}_a} = 0. \f]</li>
 *              <li>Compute the time derivatives of the sensitivity @f$ \dot{s} @f$ such that the differential equations hold.
 *                 However, because of the algebraic equations, we need additional conditions to fully determine
 *                 @f$ \dot{s}@f$. By differentiating the algebraic equations with respect to time, we get the
 *                 missing linear equations (recall that the sensitivity vector @f$ s @f$ is fixed).
 *
 *     Let @f$ \mathcal{I}_d @f$ denote the index set of differential equations.
 *     The right hand side of the linear system is given by @f[ -\frac{\partial F}{\partial y}(t, y, \dot{y}) s - \frac{\partial F}{\partial p}(t, y, \dot{y}), @f]
 *     which is 0 for algebraic equations (@f$ -\frac{\partial^2 F}{\partial t \partial p}@f$, to be more precise).</li>
 *          </ol>
 *     This function requires the parameter sensitivities to be computed beforehand and up-to-date Jacobians.
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in] simState Consistent state of the simulation (state vector and its time derivative)
 * @param [in,out] vecSensY Sensitivity subsystem state vectors
 * @param [in,out] vecSensYdot Time derivative state vectors of the sensitivity subsystems to be initialized
 * @param [in] adRes Pointer to residual vector of AD datatypes with parameter sensitivities
 * @todo Decrease amount of allocated memory by partially using temporary vectors (state and Schur complement)
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
	std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerConsistentInit);

	Indexer idxr(_disc);

	for (std::size_t param = 0; param < vecSensY.size(); ++param)
	{
		double* const sensY = vecSensY[param];
		double* const sensYdot = vecSensYdot[param];

		// Copy parameter derivative dF / dp from AD and negate it
		for (unsigned int i = _disc.nComp; i < numDofs(); ++i)
			sensYdot[i] = -adRes[i].getADValue(param);

		// Step 1: Solve algebraic equations

		if (_binding[0]->hasQuasiStationaryReactions())
		{
			int const* const qsMask = _binding[0]->reactionQuasiStationarity();
			const linalg::ConstMaskArray mask{qsMask, static_cast<int>(_disc.strideBound)};
			const int probSize = linalg::numMaskActive(mask);

#ifdef CADET_PARALLELIZE
			BENCH_SCOPE(_timerConsistentInitPar);
			tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(_disc.nCol), [&](std::size_t col)
#else
			for (unsigned int col = 0; col < _disc.nCol; ++col)
#endif
			{
				const unsigned int jacRowOffset = idxr.strideColCell() * col + static_cast<unsigned int>(idxr.strideColLiquid());
				const int localQOffset = idxr.offsetC() + col * idxr.strideColCell() + idxr.strideColLiquid();

				// Reuse memory of band matrix for dense matrix
				linalg::DenseMatrixView jacobianMatrix(_jacDisc.data() + col * _disc.strideBound * _disc.strideBound, _jacDisc.pivot() + col * _disc.strideBound, probSize, probSize);

				// Get workspace memory
				LinearBufferAllocator tlmAlloc = threadLocalMem.get();

				BufferedArray<double> rhsBuffer = tlmAlloc.array<double>(probSize);
				double* const rhs = static_cast<double*>(rhsBuffer);

				BufferedArray<double> rhsUnmaskedBuffer = tlmAlloc.array<double>(_disc.strideBound);
				double* const rhsUnmasked = static_cast<double*>(rhsUnmaskedBuffer);

				double* const maskedMultiplier = _tempState + idxr.offsetC() + col * idxr.strideColCell();

				// Extract subproblem Jacobian from full Jacobian
				jacobianMatrix.setAll(0.0);
				linalg::copyMatrixSubset(_jac, mask, mask, jacRowOffset, 0, jacobianMatrix);

				// Construct right hand side
				linalg::selectVectorSubset(sensYdot + localQOffset, mask, rhs);

				// Zero out masked elements
				std::copy_n(sensY + localQOffset - idxr.strideColLiquid(), idxr.strideColCell(), maskedMultiplier);
				linalg::fillVectorSubset(maskedMultiplier + _disc.nComp, mask, 0.0);

				// Assemble right hand side
				_jac.submatrixMultiplyVector(maskedMultiplier, jacRowOffset, -static_cast<int>(_disc.nComp), _disc.strideBound, idxr.strideColCell(), rhsUnmasked);
				linalg::vectorSubsetAdd(rhsUnmasked, mask, -1.0, 1.0, rhs);

				// Precondition
				double* const scaleFactors = _tempState + idxr.offsetC() + col * idxr.strideColCell();
				jacobianMatrix.rowScaleFactors(scaleFactors);
				jacobianMatrix.scaleRows(scaleFactors);

				// Solve
				jacobianMatrix.factorize();
				jacobianMatrix.solve(scaleFactors, rhs);

				// Write back
				linalg::applyVectorSubset(rhs, mask, sensY + localQOffset);
			} CADET_PARFOR_END;
		}

		// Step 2: Compute the correct time derivative of the state vector

		// Compute right hand side by adding -dF / dy * s = -J * s to -dF / dp which is already stored in sensYdot
		multiplyWithJacobian(simTime, simState, sensY, -1.0, 1.0, sensYdot);

		// Note that we have correctly negated the right hand side

		_jacDisc.setAll(0.0);

		// Handle transport equations (dc_i / dt terms)
		_convDispOp.addTimeDerivativeToJacobian(1.0, _jacDisc);

		const double invBeta = 1.0 / static_cast<double>(_totalPorosity) - 1.0;
		for (unsigned int col = 0; col < _disc.nCol; ++col)
		{
			// Assemble
			linalg::FactorizableBandMatrix::RowIterator jac = _jacDisc.row(idxr.strideColCell() * col);

			// Mobile and solid phase (advances jac accordingly)
			addTimeDerivativeToJacobianCell(jac, idxr, 1.0, invBeta);

			// Iterator jac has already been advanced to next shell

			// Overwrite rows corresponding to algebraic equations with the Jacobian and set right hand side to 0
			if (_binding[0]->hasQuasiStationaryReactions())
			{
				// Get iterators to beginning of solid phase
				linalg::BandMatrix::RowIterator jacSolidOrig = _jac.row(idxr.strideColCell() * col + idxr.strideColLiquid());
				linalg::FactorizableBandMatrix::RowIterator jacSolid = jac - idxr.strideColBound();

				int const* const mask = _binding[0]->reactionQuasiStationarity();
				double* const qShellDot = sensYdot + idxr.offsetC() + idxr.strideColCell() * col + idxr.strideColLiquid();

				// Copy row from original Jacobian and set right hand side
				for (unsigned int i = 0; i < _disc.strideBound; ++i, ++jacSolid, ++jacSolidOrig)
				{
					if (!mask[i])
						continue;

					jacSolid.copyRowFrom(jacSolidOrig);

					// Right hand side is -\frac{\partial^2 res(t, y, \dot{y})}{\partial p \partial t}
					// If the residual is not explicitly depending on time, this expression is 0
					// @todo This is wrong if external functions are used. Take that into account!
					qShellDot[i] = 0.0;
				}
			}
		}

		// Precondition
		double* const scaleFactors = _tempState + idxr.offsetC();
		_jacDisc.rowScaleFactors(scaleFactors);
		_jacDisc.scaleRows(scaleFactors);

		// Factorize
		const bool result = _jacDisc.factorize();
		if (!result)
		{
			LOG(Error) << "Factorize() failed for par block";
		}

		const bool result2 = _jacDisc.solve(scaleFactors, sensYdot + idxr.offsetC());
		if (!result2)
		{
			LOG(Error) << "Solve() failed for par block";
		}
	}
}

/**
 * @brief Computes approximately / partially consistent initial values and time derivatives of sensitivity subsystems
 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] and initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$,
 *          the sensitivity system for a parameter @f$ p @f$ reads
 *          \f[ \frac{\partial F}{\partial y}(t, y, \dot{y}) s + \frac{\partial F}{\partial \dot{y}}(t, y, \dot{y}) \dot{s} + \frac{\partial F}{\partial p}(t, y, \dot{y}) = 0. \f]
 *          The initial values of this linear DAE, @f$ s_0 = \frac{\partial y_0}{\partial p} @f$ and @f$ \dot{s}_0 = \frac{\partial \dot{y}_0}{\partial p} @f$
 *          have to be consistent with the sensitivity DAE. This functions updates the initial sensitivity\f$ s_0 \f$ and overwrites the time
 *          derivative \f$ \dot{s}_0 \f$ such that they are consistent.
 *
 *          The process follows closely the one of leanConsistentInitialConditions() and, in fact, is a linearized version of it.
 *          This is necessary because the initial conditions of the sensitivity system \f$ s_0 \f$ and \f$ \dot{s}_0 \f$ are
 *          related to the initial conditions \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ of the original DAE by differentiating them
 *          with respect to @f$ p @f$: @f$ s_0 = \frac{\partial y_0}{\partial p} @f$ and @f$ \dot{s}_0 = \frac{\partial \dot{y}_0}{\partial p}. @f$
 *          <ol>
 *              <li>Keep state and time derivative vectors as they are (i.e., do not solve algebraic equations).</li>
 *              <li>Compute the time derivatives of the state @f$ \dot{y} @f$ such that the residual is 0 for the
 *              mobile phase variables.</li>
 *          </ol>
 *     This function requires the parameter sensitivities to be computed beforehand and up-to-date Jacobians.
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in] simState Consistent state of the simulation (state vector and its time derivative)
 * @param [in,out] vecSensY Sensitivity subsystem state vectors
 * @param [in,out] vecSensYdot Time derivative state vectors of the sensitivity subsystems to be initialized
 * @param [in] adRes Pointer to residual vector of AD datatypes with parameter sensitivities
 * @todo Decrease amount of allocated memory by partially using temporary vectors (state and Schur complement)
 */
template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::leanConsistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
	std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerConsistentInit);

	Indexer idxr(_disc);

	for (std::size_t param = 0; param < vecSensY.size(); ++param)
	{
		double* const sensY = vecSensY[param];
		double* const sensYdot = vecSensYdot[param];

		// Copy parameter derivative from AD to tempState and negate it
		for (unsigned int col = 0; col < _disc.nCol; ++col)
		{
			const unsigned int localOffset = idxr.offsetC() + col * idxr.strideColCell();
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				_tempState[localOffset + comp] = -adRes[localOffset + comp].getADValue(param);
			}
		}

		// Step 2: Compute the correct time derivative of the state vector

		// Compute right hand side by adding -dF / dy * s = -J * s to -dF / dp which is already stored in _tempState
		multiplyWithJacobian(simTime, simState, sensY, -1.0, 1.0, _tempState);

		const double invBeta = (1.0 / static_cast<double>(_totalPorosity) - 1.0);
		for (unsigned int col = 0; col < _disc.nCol; ++col)
		{
			// Offset to current cell's c and q variables
			const unsigned int localOffset = idxr.offsetC() + col * idxr.strideColCell();
			const unsigned int localOffsetQ = localOffset + idxr.strideColLiquid();

			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				// dq_{i,j} / dt is assumed to be fixed, so bring it on the right hand side
				for (unsigned int i = 0; i < _disc.nBound[comp]; ++i)
				{
					_tempState[localOffset + comp] -= invBeta * sensYdot[localOffsetQ + _disc.boundOffset[comp] + i];
				}

				sensYdot[localOffset + comp] = _tempState[localOffset + comp];
			}
		}
	}
}

template <typename ConvDispOperator>
bool LumpedRateModelWithoutPores<ConvDispOperator>::setParameter(const ParameterId& pId, double value)
{
	if (_convDispOp.setParameter(pId, value))
		return true;

	return UnitOperationBase::setParameter(pId, value);
}

template <typename ConvDispOperator>
void LumpedRateModelWithoutPores<ConvDispOperator>::setSensitiveParameterValue(const ParameterId& pId, double value)
{
	if (_convDispOp.setSensitiveParameterValue(_sensParams, pId, value))
		return;

	UnitOperationBase::setSensitiveParameterValue(pId, value);
}

template <typename ConvDispOperator>
bool LumpedRateModelWithoutPores<ConvDispOperator>::setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue)
{
	if (_convDispOp.setSensitiveParameter(_sensParams, pId, adDirection, adValue))
	{
		LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
		return true;
	}

	return UnitOperationBase::setSensitiveParameter(pId, adDirection, adValue);
}


template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::Exporter::writeMobilePhase(double* buffer) const
{
	const int stride = _idx.strideColCell();
	double const* ptr = _data + _idx.offsetC();
	for (unsigned int i = 0; i < _disc.nCol; ++i)
	{
		std::copy_n(ptr, _disc.nComp, buffer);
		buffer += _disc.nComp;
		ptr += stride;
	}
	return _disc.nCol * _disc.nComp;
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::Exporter::writeSolidPhase(double* buffer) const
{
	const int stride = _idx.strideColCell();
	double const* ptr = _data + _idx.offsetC() + _idx.strideColLiquid();
	for (unsigned int i = 0; i < _disc.nCol; ++i)
	{
		std::copy_n(ptr, _disc.strideBound, buffer);
		buffer += _disc.strideBound;
		ptr += stride;
	}
	return _disc.nCol * _disc.strideBound;
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::Exporter::writeSolidPhase(unsigned int parType, double* buffer) const
{
	cadet_assert(parType == 0);
	return writeSolidPhase(buffer);
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::Exporter::writeInlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);
	std::copy_n(_data, _disc.nComp, buffer);
	return _disc.nComp;
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::Exporter::writeInlet(double* buffer) const
{
	std::copy_n(_data, _disc.nComp, buffer);
	return _disc.nComp;
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::Exporter::writeOutlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);

	if (_model._convDispOp.forwardFlow())
		std::copy_n(&_idx.c(_data, _disc.nCol - 1, 0), _disc.nComp, buffer);
	else
		std::copy_n(&_idx.c(_data, 0, 0), _disc.nComp, buffer);

	return _disc.nComp;
}

template <typename ConvDispOperator>
int LumpedRateModelWithoutPores<ConvDispOperator>::Exporter::writeOutlet(double* buffer) const
{
	if (_model._convDispOp.forwardFlow())
		std::copy_n(&_idx.c(_data, _disc.nCol - 1, 0), _disc.nComp, buffer);
	else
		std::copy_n(&_idx.c(_data, 0, 0), _disc.nComp, buffer);

	return _disc.nComp;
}


void registerLumpedRateModelWithoutPores(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models)
{
	typedef LumpedRateModelWithoutPores<parts::AxialConvectionDispersionOperatorBase> AxialLRM;
	typedef LumpedRateModelWithoutPores<parts::RadialConvectionDispersionOperatorBase> RadialLRM;

	models[AxialLRM::identifier()] = [](UnitOpIdx uoId) { return new AxialLRM(uoId); };
	models["LRM"] = [](UnitOpIdx uoId) { return new AxialLRM(uoId); };
	models["DPFR"] = [](UnitOpIdx uoId) { return new AxialLRM(uoId); };

	models[RadialLRM::identifier()] = [](UnitOpIdx uoId) { return new RadialLRM(uoId); };
	models["RLRM"] = [](UnitOpIdx uoId) { return new RadialLRM(uoId); };
}

}  // namespace model

}  // namespace cadet
