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

#include "model/StirredTankModel.hpp"
#include "ParamReaderHelper.hpp"
#include "ParamReaderScopes.hpp"
#include "cadet/Exceptions.hpp"
#include "cadet/SolutionRecorder.hpp"
#include "model/BindingModel.hpp"
#include "model/ReactionModel.hpp"
#include "SimulationTypes.hpp"
#include "ParallelSupport.hpp"
#include "linalg/Subset.hpp"

#include "ConfigurationHelper.hpp"
#include "linalg/Norms.hpp"
#include "AdUtils.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include <algorithm>
#include <functional>

namespace cadet
{

namespace model
{

namespace
{
	inline void bindingFlux(IBindingModel* binding, double t, unsigned int secIdx, const ColumnPosition& colPos, active const* yCp, active const* yQ, active* res, cadet::LinearBufferAllocator buffer, WithParamSensitivity)
	{
		binding->flux(t, secIdx, colPos, yQ, yCp, res, buffer, WithParamSensitivity());
	}

	inline void bindingFlux(IBindingModel* binding, double t, unsigned int secIdx, const ColumnPosition& colPos, active const* yCp, active const* yQ, active* res, cadet::LinearBufferAllocator buffer, WithoutParamSensitivity)
	{
		binding->flux(t, secIdx, colPos, yQ, yCp, res, buffer, WithoutParamSensitivity());
	}

	inline void bindingFlux(IBindingModel* binding, double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yCp, double const* yQ, active* res, cadet::LinearBufferAllocator buffer, WithParamSensitivity)
	{
		binding->flux(t, secIdx, colPos, yQ, yCp, res, buffer);
	}

	inline void bindingFlux(IBindingModel* binding, double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yCp, double const* yQ, double* res, cadet::LinearBufferAllocator buffer, WithoutParamSensitivity)
	{
		binding->flux(t, secIdx, colPos, yQ, yCp, res, buffer);
	}
}


CSTRModel::CSTRModel(UnitOpIdx unitOpIdx) : UnitOperationBase(unitOpIdx), _nComp(0), _nParType(0), _nBound(nullptr), _boundOffset(nullptr), _strideBound(nullptr), _offsetParType(nullptr), 
	_totalBound(0), _analyticJac(true), _jac(), _jacFact(), _factorizeJac(false), _initConditions(0), _initConditionsDot(0), _dynReactionBulk(nullptr)
{
	// Mutliplexed binding and reaction models make no sense in CSTR
	_singleBinding = false;
	_singleDynReaction = false;
}

CSTRModel::~CSTRModel() CADET_NOEXCEPT
{
	delete[] _boundOffset;
	delete[] _nBound;
	delete[] _strideBound;
	delete[] _offsetParType;

	delete _dynReactionBulk;
}

unsigned int CSTRModel::numDofs() const CADET_NOEXCEPT
{
	return 2 * _nComp + _totalBound + 1;
}

unsigned int CSTRModel::numPureDofs() const CADET_NOEXCEPT
{
	return _nComp + _totalBound + 1;
}

bool CSTRModel::usesAD() const CADET_NOEXCEPT
{
#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
	// We always need AD if we want to check the analytical Jacobian
	return true;
#else
	// We only need AD if we are not computing the Jacobian analytically
	return !_analyticJac;
#endif
}

void CSTRModel::setFlowRates(active const* in, active const* out) CADET_NOEXCEPT 
{ 
	_flowRateIn = in[0];
	_flowRateOut = out[0];
}

bool CSTRModel::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper)
{
	_nComp = paramProvider.getInt("NCOMP");

	if (paramProvider.exists("NBOUND"))
	{
		const std::vector<int> nBound = paramProvider.getIntArray("NBOUND");
		_nParType = nBound.size() / _nComp;
		
		_nBound = new unsigned int[_nComp * _nParType];
		std::copy(nBound.begin(), nBound.begin() + _nComp * _nParType, _nBound);

		// Precompute offsets and total number of bound states (DOFs in solid phase)
		_boundOffset = new unsigned int[_nComp * _nParType];
		_offsetParType = new unsigned int[_nParType];
		_strideBound = new unsigned int[_nParType];
		_totalBound = 0;
		_offsetParType[0] = 0;
		for (unsigned int j = 0; j < _nParType; ++j)
		{
			unsigned int* const bo = _boundOffset + j * _nComp;
			unsigned int* const nb = _nBound + j * _nComp;

			bo[0] = 0;
			for (unsigned int i = 1; i < _nComp; ++i)
			{
				bo[i] = bo[i - 1] + nb[i - 1];
			}
			_strideBound[j] = bo[_nComp-1] + nb[_nComp - 1];
			_totalBound += _strideBound[j];

			if (j < _nParType - 1)
				_offsetParType[j + 1] = _totalBound;
		}
	}
	else
	{
		_nParType = 0;
		_totalBound = 0;
		_nBound = nullptr;
		_boundOffset = nullptr;
		_strideBound = nullptr;
		_offsetParType = nullptr;
	}

	if ((_nParType > 1) && (_totalBound == 0))
		throw InvalidParameterException("Multiple particle types but not a single bound state detected");

	// Make sure each particle type has at least one bound state
	for (unsigned int i = 0; i < _nParType; ++i)
	{
		if (_strideBound[i] == 0)
			throw InvalidParameterException("Particle type " + std::to_string(i) + " does not have a bound state");
	}

	// Allocate Jacobian
	const unsigned int nVar = _nComp + _totalBound + 1;
	_jac.resize(nVar, nVar);
	_jacFact.resize(nVar, nVar);

	// Allocate space for initial conditions
	_initConditions.resize(nVar);
	_initConditionsDot.resize(nVar, 0.0);

	// Determine whether analytic Jacobian should be used
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	bool analyticJac = true;
	if (paramProvider.exists("USE_ANALYTIC_JACOBIAN"))
		analyticJac = paramProvider.getBool("USE_ANALYTIC_JACOBIAN");
#else
	// Default to AD Jacobian when analytic Jacobian is to be checked
	const bool analyticJac = false;
#endif
	useAnalyticJacobian(analyticJac);

	// Create nonlinear solver for consistent initialization
	if (paramProvider.exists("discretization"))
	{
		paramProvider.pushScope("discretization");
		configureNonlinearSolver(paramProvider);
		paramProvider.popScope();
	}
	else
		configureNonlinearSolver();

	// ==== Construct and configure binding models
	clearBindingModels();
	_binding = std::vector<IBindingModel*>(_nParType, nullptr);

	if ((_nParType > 0) && !paramProvider.exists("ADSORPTION_MODEL"))
		throw InvalidParameterException("Binding model required when using at least one particle type");

	bool bindingConfSuccess = true;
	if (_nParType > 0)
	{
		const std::vector<std::string> bindModelNames = paramProvider.getStringArray("ADSORPTION_MODEL");
		if (bindModelNames.size() < _nParType)
			throw InvalidParameterException("Field ADSORPTION_MODEL contains too few elements (" + std::to_string(_nParType) + " required)");

		for (unsigned int i = 0; i < _nParType; ++i)
		{
			_binding[i] = helper.createBindingModel(bindModelNames[i]);
			if (!_binding[i])
				throw InvalidParameterException("Unknown binding model " + bindModelNames[i]);

			MultiplexedScopeSelector scopeGuard(paramProvider, "adsorption", _nParType == 1, i, _nParType == 1, _binding[i]->usesParamProviderInDiscretizationConfig());
			bindingConfSuccess = _binding[i]->configureModelDiscretization(paramProvider, _nComp, _nBound + i * _nComp, _boundOffset + i * _nComp) && bindingConfSuccess;
		}
	}

	// ==== Construct and configure dynamic reaction model
	clearDynamicReactionModels();
	bool reactionConfSuccess = true;

	_dynReactionBulk = nullptr;
	if (paramProvider.exists("REACTION_MODEL"))
	{
		const std::string dynReactName = paramProvider.getString("REACTION_MODEL");
		_dynReactionBulk = helper.createDynamicReactionModel(dynReactName);
		if (!_dynReactionBulk)
			throw InvalidParameterException("Unknown dynamic reaction model " + dynReactName);

		if (_dynReactionBulk->usesParamProviderInDiscretizationConfig())
			paramProvider.pushScope("reaction_bulk");

		reactionConfSuccess = _dynReactionBulk->configureModelDiscretization(paramProvider, _nComp, nullptr, nullptr);

		if (_dynReactionBulk->usesParamProviderInDiscretizationConfig())
			paramProvider.popScope();
	}

	_dynReaction = std::vector<IDynamicReactionModel*>(_nParType, nullptr);

	if (paramProvider.exists("REACTION_MODEL_PARTICLES"))
	{
		const std::vector<std::string> dynReactModelNames = paramProvider.getStringArray("REACTION_MODEL_PARTICLES");
		if (dynReactModelNames.size() < _nParType)
			throw InvalidParameterException("Field REACTION_MODEL_PARTICLES contains too few elements (" + std::to_string(_nParType) + " required)");

		for (unsigned int i = 0; i < _nParType; ++i)
		{
			_dynReaction[i] = helper.createDynamicReactionModel(dynReactModelNames[i]);
			if (!_dynReaction[i])
				throw InvalidParameterException("Unknown dynamic reaction model " + dynReactModelNames[i]);

			MultiplexedScopeSelector scopeGuard(paramProvider, "reaction_particle", _nParType == 1, i, _nParType == 1, _dynReaction[i]->usesParamProviderInDiscretizationConfig());
			reactionConfSuccess = _dynReaction[i]->configureModelDiscretization(paramProvider, _nComp, _nBound + i * _nComp, _boundOffset + i * _nComp) && reactionConfSuccess;
		}
	}

	return bindingConfSuccess && reactionConfSuccess;
}

bool CSTRModel::configure(IParameterProvider& paramProvider)
{
	_curFlowRateFilter = 0.0;
	_flowRateFilter.clear();
	const bool hasFlowrateFilter = paramProvider.exists("FLOWRATE_FILTER");
	if (hasFlowrateFilter)
		readScalarParameterOrArray(_flowRateFilter, paramProvider, "FLOWRATE_FILTER", 1);

	_porosity = 1.0;
	if (paramProvider.exists("POROSITY"))
		_porosity = paramProvider.getDouble("POROSITY");

	if (_totalBound > 0)
	{
		// Let PAR_TYPE_VOLFRAC default to 1.0 for backwards compatibility
		if (paramProvider.exists("PAR_TYPE_VOLFRAC"))
			readScalarParameterOrArray(_parTypeVolFrac, paramProvider, "PAR_TYPE_VOLFRAC", 1);
		else
			_parTypeVolFrac.resize(1, 1.0);

		if (_nParType != _parTypeVolFrac.size())
			throw InvalidParameterException("Number of elements in field PAR_TYPE_VOLFRAC does not match number of particle types");

		// Check that particle volume fractions sum to 1.0
		const double volFracSum = std::accumulate(_parTypeVolFrac.begin(), _parTypeVolFrac.end(), 0.0, 
			[](double a, const active& b) -> double { return a + static_cast<double>(b); });
		if (std::abs(1.0 - volFracSum) > 1e-10)
			throw InvalidParameterException("Sum of field PAR_TYPE_VOLFRAC differs from 1.0 (is " + std::to_string(volFracSum) + ")");
	}

	_parameters.clear();
	if (hasFlowrateFilter)
		registerScalarSectionDependentParam(hashString("FLOWRATE_FILTER"), _parameters, _flowRateFilter, _unitOpIdx, ParTypeIndep);
	_parameters[makeParamId(hashString("POROSITY"), _unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_porosity;
	if (_totalBound > 0)
		registerParam1DArray(_parameters, _parTypeVolFrac, [=](bool multi, unsigned int type) { return makeParamId(hashString("PAR_TYPE_VOLFRAC"), _unitOpIdx, CompIndep, type, BoundStateIndep, ReactionIndep, SectionIndep); });

	// Register initial conditions parameters
	for (unsigned int i = 0; i < _nComp; ++i)
		_parameters[makeParamId(hashString("INIT_C"), _unitOpIdx, i, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = _initConditions.data() + i;

	for (unsigned int type = 0; type < _nParType; ++type)
	{
		if (!_binding[type])
			continue;

		std::vector<ParameterId> initParams(_strideBound[type]);
		_binding[type]->fillBoundPhaseInitialParameters(initParams.data(), _unitOpIdx, type);

		active* const ic = _initConditions.data() + _nComp + _offsetParType[type];
		for (unsigned int i = 0; i < _strideBound[type]; ++i)
			_parameters[initParams[i]] = ic + i;
	}

	_parameters[makeParamId(hashString("INIT_VOLUME"), _unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = _initConditions.data() + _nComp + _totalBound;

	// Reconfigure binding model
	bool bindingConfSuccess = true;
	if (!_binding.empty())
	{
		for (unsigned int type = 0; type < _nParType; ++type)
		{
 			if (!_binding[type] || !_binding[type]->requiresConfiguration())
 				continue;

			MultiplexedScopeSelector scopeGuard(paramProvider, "adsorption", type, _nParType == 1, true);
			bindingConfSuccess = _binding[type]->configure(paramProvider, _unitOpIdx, type) && bindingConfSuccess;
		}
	}

	// Reconfigure reaction model
	bool dynReactionConfSuccess = true;
	if (_dynReactionBulk && _dynReactionBulk->requiresConfiguration())
	{
		paramProvider.pushScope("reaction_bulk");
		dynReactionConfSuccess = _dynReactionBulk->configure(paramProvider, _unitOpIdx, ParTypeIndep);
		paramProvider.popScope();
	}

	for (unsigned int type = 0; type < _nParType; ++type)
	{
		if (!_dynReaction[type] || !_dynReaction[type]->requiresConfiguration())
			continue;

		MultiplexedScopeSelector scopeGuard(paramProvider, "reaction_particle", type, _nParType == 1, true);
		dynReactionConfSuccess = _dynReaction[type]->configure(paramProvider, _unitOpIdx, type) && dynReactionConfSuccess;
	}

	return bindingConfSuccess && dynReactionConfSuccess;
}

unsigned int CSTRModel::threadLocalMemorySize() const CADET_NOEXCEPT
{
	LinearMemorySizer lms;

	// Memory for residualImpl()
	for (unsigned int i = 0; i < _nParType; ++i)
	{
		if (_binding[i] && _binding[i]->requiresWorkspace())
			lms.fitBlock(_binding[i]->workspaceSize(_nComp, _strideBound[i], _nBound + i * _nComp));
		if (_dynReaction[i] && _dynReaction[i]->requiresWorkspace())
			lms.fitBlock(_dynReaction[i]->workspaceSize(_nComp, _strideBound[i], _nBound + i * _nComp));
	}

	if (_dynReactionBulk && _dynReactionBulk->requiresWorkspace())
		lms.fitBlock(_dynReactionBulk->workspaceSize(_nComp, 0, nullptr));

	const unsigned int maxStrideBound = _strideBound ? *std::max_element(_strideBound, _strideBound + _nParType) : 0;
	lms.add<active>(_nComp + maxStrideBound);
	lms.add<double>((maxStrideBound + _nComp) * (maxStrideBound + _nComp));

	lms.commit();
	const std::size_t resImplSize = lms.bufferSize();

	// Memory for consistentInitialTimeDerivative()
	for (unsigned int i = 0; i < _nParType; ++i)
	{
		if (_binding[i] && _binding[i]->requiresWorkspace())
			lms.fitBlock(_binding[i]->workspaceSize(_nComp, _strideBound[i], _nBound + i * _nComp));
	}
	lms.add<double>(_nComp + _totalBound + 1);
	lms.commit();

	// Memory for consistentInitialState()
	lms.add<int>(_nComp + _totalBound);
	lms.add<double>(_nComp + _totalBound);
	lms.add<double>(_nComp);
	lms.add<double>(_nComp + _totalBound);
	lms.add<double>(numDofs());
	lms.add<double>(_nonlinearSolver->workspaceSize(_nComp + _totalBound) * sizeof(double));
	lms.addBlock(resImplSize);
	lms.commit();

	// Memory for consistentInitialSensitivity
	lms.add<int>(_totalBound);
	lms.add<double>(_nComp + _totalBound + 1);
	lms.add<double>(_nComp + _totalBound);
	lms.add<double>(_totalBound);
	lms.commit();

	return lms.bufferSize();
}

void CSTRModel::setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections)
{
}

void CSTRModel::useAnalyticJacobian(const bool analyticJac)
{
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	_analyticJac = analyticJac;
#else
	// Use AD Jacobian if analytic Jacobian is to be checked
	_analyticJac = false;
#endif
}

void CSTRModel::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac)
{
	if (_flowRateFilter.size() > 1)
	{
		_curFlowRateFilter = _flowRateFilter[secIdx];
	}
	else if (_flowRateFilter.size() == 1)
	{
		_curFlowRateFilter = _flowRateFilter[0];
	}
}

void CSTRModel::reportSolution(ISolutionRecorder& recorder, double const* const solution) const
{
	Exporter expr(_nComp, _nParType, _nBound, _strideBound, _boundOffset, _totalBound, solution);
	recorder.beginUnitOperation(_unitOpIdx, *this, expr);
	recorder.endUnitOperation();
}

void CSTRModel::reportSolutionStructure(ISolutionRecorder& recorder) const
{
	Exporter expr(_nComp, _nParType, _nBound, _strideBound, _boundOffset, _totalBound, nullptr);
	recorder.unitOperationStructure(_unitOpIdx, *this, expr);
}

unsigned int CSTRModel::requiredADdirs() const CADET_NOEXCEPT
{
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	// We don't need AD if analytic Jaocbian is enabled
	if (_analyticJac)
		return 0;
#endif

	// We always need AD if CADET_CHECK_ANALYTIC_JACOBIAN and if analytic Jacobian is disabled
	return _nComp + _totalBound + 1;
}

void CSTRModel::prepareADvectors(const AdJacobianParams& adJac) const
{
	// Early out if AD is disabled
	if (!adJac.adY)
		return;

	ad::prepareAdVectorSeedsForDenseMatrix(adJac.adY + _nComp, adJac.adDirOffset, _jac.columns());
}

void CSTRModel::applyInitialCondition(const SimulationState& simState) const
{
	// Inlet DOFs
	std::fill(simState.vecStateY, simState.vecStateY + _nComp, 0.0);
	std::fill(simState.vecStateYdot, simState.vecStateYdot + _nComp, 0.0);

	const unsigned int nDof = numPureDofs();
	ad::copyFromAd(_initConditions.data(), simState.vecStateY + _nComp, nDof);

	std::copy(_initConditionsDot.data(), _initConditionsDot.data() + nDof, simState.vecStateYdot + _nComp);
}

void CSTRModel::readInitialCondition(IParameterProvider& paramProvider)
{
	// Clear time derivative
	std::fill(_initConditionsDot.begin(), _initConditionsDot.end(), 0.0);

	// Check if INIT_STATE is present
	if (paramProvider.exists("INIT_STATE"))
	{
		const unsigned int nDof = numPureDofs();
		const std::vector<double> initState = paramProvider.getDoubleArray("INIT_STATE");

		ad::copyToAd(initState.data(), _initConditions.data(), nDof);

		// Check if INIT_STATE contains the full state and its time derivative
		if (initState.size() >= 2 * nDof)
			std::copy(initState.data() + nDof, initState.data() + 2 * nDof, _initConditionsDot.data());

		return;
	}

	const std::vector<double> initC = paramProvider.getDoubleArray("INIT_C");

	if (initC.size() < _nComp)
		throw InvalidParameterException("INIT_C does not contain enough values for all components");
	
	ad::copyToAd(initC.data(), _initConditions.data(), _nComp);

	if (paramProvider.exists("INIT_Q"))
	{
		const std::vector<double> initQ = paramProvider.getDoubleArray("INIT_Q");
		ad::copyToAd(initQ.data(), _initConditions.data() + _nComp, _totalBound);
	}
	else
		ad::fillAd(_initConditions.data() + _nComp, _totalBound, 0.0);

	if (paramProvider.exists("INIT_VOLUME"))
		_initConditions[_nComp + _totalBound].setValue(paramProvider.getDouble("INIT_VOLUME"));
	else
		_initConditions[_nComp + _totalBound].setValue(0.0);
}

void CSTRModel::consistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem)
{
	double * const c = vecStateY + _nComp;
	const double v = c[_nComp + _totalBound];

	// Check if volume is 0
	if (v == 0.0)
	{
		const double flowIn = static_cast<double>(_flowRateIn);
		const double flowOut = static_cast<double>(_flowRateOut);

		// Volume: \dot{V} = F_{in} - F_{out} - F_{filter}
		const double vDot = flowIn - flowOut - static_cast<double>(_curFlowRateFilter);

		// We have the equation
		//    \dot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]) + V * (\dot{c}_i + 1 / beta * [sum_j sum_m d_j \dot{q}_{i,m}]) = c_{in,i} * F_in + c_i * F_out
		// which is now algebraic wrt. c due to V = 0:
		//    \dot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]) = c_{in,i} * F_in + c_i * F_out
		// Separating knowns from unknowns gives
		//    (\dot{V} + F_out) * c + \dot{V} / beta * [sum_j sum_m d_j q_{i,m}] = c_in * F_in

		// We assume that all binding models are fully dynamic (i.e., they do not have
		// algebraic equations). 
		// TODO: Figure out what to do when at least one binding model has algebraic equations.

		// If the binding models do not have algebraic equations, the
		// bound states q_{i,j} are dynamic and, thus, already determined
		//    (\dot{V} + F_out) * c = c_in * F_in - \dot{V} / beta * [sum_j sum_m d_j q_{j,i,m}]
		// This finally leads to
		//    c = (c_in * F_in - \dot{V} / beta * [sum_j sum_m d_j q_{j,i,m}]) / (\dot{V} + F_out)

		// Note that if the denominator (\dot{V} + F_out) were 0, we had
		//    0 = \dot{V} + F_out = F_{in} - F_{filter}
		// which leads to
		//    F_in = F_filter
		// Since F_out >= 0 and \dot{V} = -F_out, we get
		//    \dot{V} <= 0
		// Assuming a valid configuration, we obtain \dot{V} = 0
		// as the tank would get a negative volume otherwise.
		// Concluding, we arrive at \dot{V} = F_out = 0.
		// In this situation, F_in = F_filter = 0 has to hold
		// as otherwise the liquid (solvent) is immediately and
		// fully taken out, leaving only the pure dry components.
		// We, hence, assume that this doesn't happen and simply
		// do nothing leaving the initial conditions in place.

		const double denom = vDot + flowOut;
		if (denom == 0.0)
			return;

		for (unsigned int i = 0; i < _nComp; ++i)
			c[i] = vecStateY[i] * flowIn / denom;;

		const double qFactor = vDot * (1.0 / static_cast<double>(_porosity) - 1.0) / denom;
		for (unsigned int type = 0; type < _nParType; ++type)
		{
			unsigned int const* const bo = _boundOffset + type * _nComp;
			unsigned int const* const nb = _nBound + type * _nComp;
			double const* const typeQ = c + _nComp + _offsetParType[type];
			for (unsigned int i = 0; i < _nComp; ++i)
			{
				double qSum = 0.0;
				double const* const localQ = typeQ + bo[i];
				for (unsigned int j = 0; j < nb[i]; ++j)
					qSum += localQ[j];

				c[i] -= qFactor * static_cast<double>(_parTypeVolFrac[type]) * qSum;
			}
		}
	}
	else
	{
		LinearBufferAllocator tlmAlloc = threadLocalMem.get();
		BufferedArray<int> qsMask = tlmAlloc.array<int>(_nComp + _totalBound);

		// Check whether quasi-stationary reactions are present and construct mask array
		bool hasNoQSreactions = true;
		std::fill_n(static_cast<int*>(qsMask), _nComp + _totalBound, false);
		int* localMask = static_cast<int*>(qsMask) + _nComp;
		for (unsigned int type = 0; type < _nParType; localMask += _strideBound[type], ++type)
		{
			if (!_binding[type]->hasQuasiStationaryReactions())
				continue;

			// Determine whether nonlinear solver is required
			const ColumnPosition colPos{0.0, 0.0, 0.0};
			if (!_binding[type]->preConsistentInitialState(simTime.t, simTime.secIdx, colPos, c + _nComp + _offsetParType[type], c, tlmAlloc))
				continue;

			hasNoQSreactions = false;

			// Construct mask
			int const* const qsMaskSrc = _binding[type]->reactionQuasiStationarity();
			std::copy_n(qsMaskSrc, _strideBound[type], localMask);

			// Mark component for conserved moiety if it has a quasi-stationary bound state
			unsigned int idx = 0;
			for (unsigned int comp = 0; comp < _nComp; ++ comp)
			{
				// Skip components that are already marked
				if (qsMask[comp])
				{
					idx += _nBound[type * _nComp + comp];
					continue;
				}

				for (unsigned int state = 0; state < _nBound[type * _nComp + comp]; ++state, ++idx)
				{
					if (qsMaskSrc[idx])
					{
						qsMask[comp] = true;
						break;
					}
				}
			}
		}

		// Consistent init is not required as we do not have quasi-stationary reactions
		if (hasNoQSreactions)
			return;

		const linalg::ConstMaskArray mask{static_cast<int*>(qsMask), static_cast<int>(_nComp + _totalBound)};
		const int probSize = linalg::numMaskActive(mask);

		// Extract initial values from current state
		BufferedArray<double> solution = tlmAlloc.array<double>(probSize);
		linalg::selectVectorSubset(c, mask, static_cast<double*>(solution));

		// Save values of conserved moieties
		const unsigned int numActiveComp = numMaskActive(mask, _nComp);
		BufferedArray<double> conservedQuants = tlmAlloc.array<double>(numActiveComp);
		const double epsQ = 1.0 - static_cast<double>(_porosity);
		unsigned int idx = 0;
		for (unsigned int comp = 0; comp < _nComp; ++comp)
		{
			if (!qsMask[comp])
				continue;

			conservedQuants[idx] = static_cast<double>(_porosity) * c[comp];
			for (unsigned int type = 0; type < _nParType; ++type)
			{
				const double factor = epsQ * static_cast<double>(_parTypeVolFrac[type]);
				for (unsigned int state = 0; state < _nBound[type * _nComp + comp]; ++state)
				{
					const unsigned int bndIdx = _offsetParType[type] + _boundOffset[type * _nComp + comp] + state;
					if (!qsMask[_nComp + bndIdx])
						continue;

					conservedQuants[idx] += factor * c[_nComp + bndIdx];
				}
			}

			++idx;
		}

		linalg::DenseMatrixView jacobianMatrix(_jacFact.data(), _jacFact.pivotData(), probSize, probSize);

		BufferedArray<double> baFullX = tlmAlloc.array<double>(numDofs());
		double* const fullX = static_cast<double*>(baFullX);

		BufferedArray<double> baFullResidual = tlmAlloc.array<double>(numDofs());
		double* const fullResidual = static_cast<double*>(baFullResidual);

		BufferedArray<double> baNonlinMem = tlmAlloc.array<double>(_nonlinearSolver->workspaceSize(probSize));
		double* const nonlinMem = static_cast<double*>(baNonlinMem);

		std::function<bool(double const* const, linalg::detail::DenseMatrixBase&)> jacFunc;
		if (adJac.adY && adJac.adRes)
		{
			ad::copyToAd(vecStateY, adJac.adY, _nComp);

			jacFunc = [&](double const* const x, linalg::detail::DenseMatrixBase& mat)
			{
				active* const localAdY = adJac.adY + _nComp;
				active* const localAdRes = adJac.adRes + _nComp;

				// Copy over state vector to AD state vector (without changing directional values to keep seed vectors)
				// and initialize residuals with zero (also resetting directional values)
				ad::copyToAd(c, localAdY, mask.len);
				// @todo Check if this is necessary
				ad::resetAd(localAdRes, mask.len);

				// Prepare input vector by overwriting masked items
				linalg::applyVectorSubset(x, mask, localAdY);

				// Call residual function
				residualImpl<active, active, double, false>(simTime.t, simTime.secIdx, adJac.adY, nullptr, adJac.adRes, tlmAlloc.manageRemainingMemory());

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
				std::copy_n(c, mask.len, fullX);
				linalg::applyVectorSubset(x, mask, fullX);

				// Compute analytic Jacobian
				residualImpl<double, double, double, true>(simTime.t, simTime.secIdx, fullX, nullptr, fullResidual, tlmAlloc.manageRemainingMemory());

				// Compare
				const double diff = ad::compareDenseJacobianWithAd(localAdRes, adJac.adDirOffset, _jac);
				LOG(Debug) << "MaxDiff: " << diff;
#endif

				// Extract Jacobian from AD
				ad::extractDenseJacobianFromAd(localAdRes, adJac.adDirOffset, _jac);

				// Extract Jacobian from full Jacobian
				mat.setAll(0.0);
				linalg::copyMatrixSubset(_jac, mask, mask, mat);

				// Replace upper part with conservation relations
				mat.submatrixSetAll(0.0, 0, 0, numActiveComp, probSize);

				unsigned int rIdx = 0;
				const double epsQ = 1.0 - static_cast<double>(_porosity);

				for (unsigned int comp = 0; comp < _nComp; ++comp)
				{
					if (!mask.mask[comp])
						continue;

					mat.native(rIdx, rIdx) = static_cast<double>(_porosity);

					for (unsigned int type = 0; type < _nParType; ++type)
					{
						const double factor = epsQ * static_cast<double>(_parTypeVolFrac[type]);;

						const unsigned int offset = _nComp + _offsetParType[type] + _boundOffset[type * _nComp + comp];
						unsigned int bIdx = numActiveComp + numMaskActive(mask, _nComp, _boundOffset[type * _nComp + comp]);
						for (unsigned int state = 0; state < _nBound[type * _nComp + comp]; ++state)
						{
							if (!mask.mask[offset + state])
								continue;

							mat.native(rIdx, bIdx) = factor;
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
				std::copy_n(c, mask.len, fullX + _nComp);
				linalg::applyVectorSubset(x, mask, fullX + _nComp);

				// Call residual function
				residualImpl<double, double, double, true>(simTime.t, simTime.secIdx, fullX, nullptr, fullResidual, tlmAlloc.manageRemainingMemory());

				// Extract Jacobian from full Jacobian
				mat.setAll(0.0);
				linalg::copyMatrixSubset(_jac, mask, mask, mat);

				// Replace upper part with conservation relations
				mat.submatrixSetAll(0.0, 0, 0, numActiveComp, probSize);

				unsigned int rIdx = 0;
				const double epsQ = 1.0 - static_cast<double>(_porosity);

				for (unsigned int comp = 0; comp < _nComp; ++comp)
				{
					if (!mask.mask[comp])
						continue;

					mat.native(rIdx, rIdx) = static_cast<double>(_porosity);

					for (unsigned int type = 0; type < _nParType; ++type)
					{
						const double factor = epsQ * static_cast<double>(_parTypeVolFrac[type]);;

						const unsigned int offset = _nComp + _offsetParType[type] + _boundOffset[type * _nComp + comp];
						unsigned int bIdx = numActiveComp + numMaskActive(mask, _nComp, _boundOffset[type * _nComp + comp]);
						for (unsigned int state = 0; state < _nBound[type * _nComp + comp]; ++state)
						{
							if (!mask.mask[offset + state])
								continue;

							mat.native(rIdx, bIdx) = factor;
							++bIdx;
						}
					}

					++rIdx;
				}

				return true;
			};
		}

		// Copy inlet values
		std::copy_n(vecStateY, _nComp, fullX);

		// Apply nonlinear solver
		_nonlinearSolver->solve(
			[&](double const* const x, double* const r)
			{
				// Prepare input vector by overwriting masked items
				std::copy_n(c, mask.len, fullX + _nComp);
				linalg::applyVectorSubset(x, mask, fullX + _nComp);

				// Call residual function
				residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, fullX, nullptr, fullResidual, tlmAlloc.manageRemainingMemory());

				// Extract values from residual
				linalg::selectVectorSubset(fullResidual + _nComp, mask, r);

				// Calculate residual of conserved moieties
				std::fill_n(r, numActiveComp, 0.0);
				unsigned int rIdx = 0;
				const double epsQ = 1.0 - static_cast<double>(_porosity);

				for (unsigned int comp = 0; comp < _nComp; ++comp)
				{
					if (!mask.mask[comp])
						continue;

					r[rIdx] = static_cast<double>(_porosity) * x[rIdx] - conservedQuants[rIdx];

					for (unsigned int type = 0; type < _nParType; ++type)
					{
						const double factor = epsQ * static_cast<double>(_parTypeVolFrac[type]);;

						const unsigned int offset = _nComp + _offsetParType[type] + _boundOffset[type * _nComp + comp];
						unsigned int bIdx = numActiveComp + numMaskActive(mask, _nComp, _boundOffset[type * _nComp + comp]);
						for (unsigned int state = 0; state < _nBound[type * _nComp + comp]; ++state)
						{
							if (!mask.mask[offset + state])
								continue;

							r[rIdx] += factor * x[bIdx];
							++bIdx;
						}
					}

					++rIdx;
				}

				return true;
			},
			jacFunc, errorTol, static_cast<double*>(solution), nonlinMem, jacobianMatrix, probSize);

		// Apply solution
		linalg::applyVectorSubset(static_cast<double*>(solution), mask, c);

		// Refine / correct solution
		for (unsigned int type = 0; type < _nParType; ++type)
		{
			const ColumnPosition colPos{0.0, 0.0, 0.0};
			_binding[type]->postConsistentInitialState(simTime.t, simTime.secIdx, colPos, c + _nComp + _offsetParType[type], c, tlmAlloc);
		}
	}
}

void CSTRModel::consistentInitialTimeDerivative(const SimulationTime& simTime, double const* vecStateY, double* const vecStateYdot, util::ThreadLocalStorage& threadLocalMem) 
{
	double const* const c = vecStateY + _nComp;
	double* const cDot = vecStateYdot + _nComp;
	const double v = c[_nComp + _totalBound];

	const double flowIn = static_cast<double>(_flowRateIn);
	const double flowOut = static_cast<double>(_flowRateOut);

	// Note that the residual has not been negated, yet. We will do that now.
	for (unsigned int i = 0; i < numDofs(); ++i)
		vecStateYdot[i] = -vecStateYdot[i];

	// Assemble time derivative Jacobian
	_jacFact.setAll(0.0);
	addTimeDerivativeJacobian(simTime.t, 1.0, ConstSimulationState{vecStateY, nullptr}, _jacFact);

	// Check if volume is 0
	if (v == 0.0)
	{
		// Volume: \dot{V} = F_{in} - F_{out} - F_{filter}
		const double vDot = flowIn - flowOut - static_cast<double>(_curFlowRateFilter);
		_jacFact.native(_nComp + _totalBound, _nComp + _totalBound) = 1.0;
		cDot[_nComp + _totalBound] = vDot;

		// We have the equation
		//    V * \dot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} + \dot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]) = c_{in,i} * F_in - c_i * F_out
		// which is now algebraic wrt. c due to V = 0:
		//    \dot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]) = c_{in,i} * F_in - c_i * F_out
		// So we take the derivative wrt. to time t on both sides
		//    2 * \dot{V} * \dot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} + V * \ddot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} + \ddot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]) = \dot{c}_{in,i} * F_in - \dot{c} * F_out
		// and use the fact that \ddot{V} = 0 and V = 0 to arrive at
		//    2 * \dot{V} * \dot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} = \dot{c}_{in,i} * F_in - \dot{c} * F_out
		// Separating knowns from unknowns gives
		//    (2 * \dot{V} + F_out) * \dot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} = \dot{c}_{in,i} * F_in
		// which finally yields
		//    \dot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} = \dot{c}_{in,i} * F_in / (2 * \dot{V} + F_out)

		// Note that if the denominator were 0, we had
		//    0 = 2 * \dot{V} + F_out = 2 * F_in - 2 * F_filter - F_out
		// which leads to
		//    F_out = 2 * F_in - 2 * F_filter                       (*)
		// Plugging this back into the \dot{V} equation gives
		//    \dot{V} = F_in - F_out - F_filter = -F_in + F_filter
		// Since V = 0, a valid choice of parameters has to ensure
		// \dot{V} >= 0. In this case, we obtain
		//    F_in <= F_filter
		// On the other hand, we infer from (*) that
		//    0 <= F_out <= 0   =>   F_out = 0   =>   F_in = F_filter
		// This, in turn, concludes \dot{V} = 0. In this situation, 
		//    F_in = F_filter = 0
		// has to hold as otherwise the liquid (solvent) is immediately
		// and fully taken out, leaving only the pure dry components.
		// We, hence, assume that this doesn't happen. Summarizing, we
		// have
		//    \dot{V} = 0, F_in = F_filter = F_out = 0
		// and nothing can happen or change. Therefore, \dot{c} is set
		// to 0.0.

		typename linalg::DenseMatrix::RowIterator itRow = _jacFact.row(0);

		const double denom = 2.0 * vDot + flowOut;
		if (denom == 0.0)
		{
			// Assume F_in = F_filter = 0, set cDot to 0
			for (unsigned int i = 0; i < _nComp; ++i, ++itRow)
			{
				itRow.setAll(0.0);
				itRow[0] = 1.0;
			}
			std::fill(cDot, cDot + _nComp, 0.0);
		}
		else
		{
			const double factor = flowIn / denom;
			const double invBeta = 1.0 / static_cast<double>(_porosity) - 1.0;
			for (unsigned int i = 0; i < _nComp; ++i, ++itRow)
			{
				itRow.setAll(0.0);

				// Jacobian of c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]

				// d/dc_i
				itRow[0] = 1.0;

				// Advance to bound states
				const int bndIdx = _nComp - i;
				for (unsigned int type = 0; type < _nParType; ++type)
				{
					const int innerIdx = bndIdx + _offsetParType[type] + _boundOffset[type * _nComp + i];
					const double innerFactor = static_cast<double>(_parTypeVolFrac[type]) * invBeta;
					for (unsigned int j = 0; j < _nBound[type * _nComp + i]; ++j)
					{
						itRow[innerIdx + j] = innerFactor;
					}
				}

				// TODO: This is wrong as vecStateYdot does not contain \dot{c}_in (on entry)
				// This scenario violates the assumption that every outlet DOF is dynamic
				// which is key to the consistent initialization algorithm. Fixing this problem
				// requires fundamental changes to the consistent initialization concept
				// implemented so far.
				vecStateYdot[i] = 0.0;
				cDot[i] = vecStateYdot[i] * factor;
			}
		}
	}

	LinearBufferAllocator tlmAlloc = threadLocalMem.get();

	// Overwrite rows corresponding to algebraic equations with the Jacobian and set right hand side to 0
	BufferedArray<double> dFluxDt = tlmAlloc.array<double>(_nComp + _totalBound + 1);
	unsigned int idx = _nComp;
	for (unsigned int type = 0; type < _nParType; ++type)
	{
		if (!_binding[type]->hasQuasiStationaryReactions())
		{
			idx += _strideBound[type];
			continue;
		}

		int const* const mask = _binding[type]->reactionQuasiStationarity();

		// Obtain derivative of fluxes wrt. time
		std::fill_n(static_cast<double*>(dFluxDt), _strideBound[type], 0.0);
		if (_binding[type]->dependsOnTime())
		{
			_binding[type]->timeDerivativeQuasiStationaryFluxes(simTime.t, simTime.secIdx,
				ColumnPosition{0.0, 0.0, 0.0},
				c, c + _nComp + _offsetParType[type], static_cast<double*>(dFluxDt), tlmAlloc);
		}

		// Copy row from original Jacobian and set right hand side
		double* const qShellDot = cDot + _nComp + _offsetParType[type];
		for (unsigned int i = 0; i < _strideBound[type]; ++i, ++idx)
		{
			if (!mask[i])
				continue;

			_jacFact.copyRowFrom(_jac, idx, idx);
			qShellDot[i] = -dFluxDt[i];
		}
	}

	// Factorize
	const bool result = _jacFact.robustFactorize(static_cast<double*>(dFluxDt));
	if (!result)
	{
		LOG(Error) << "Factorize() failed";
	}

	// Solve
	const bool result2 = _jacFact.robustSolve(vecStateYdot + _nComp, static_cast<double*>(dFluxDt));
	if (!result2)
	{
		LOG(Error) << "Solve() failed";
	}
}

void CSTRModel::leanConsistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem)
{
	// It is assumed that the bound states are (approximately) correctly initialized.
	// Thus, only the liquid phase has to be initialized

	double * const c = vecStateY + _nComp;
	const double v = c[_nComp];

	// Check if volume is 0
	if (v == 0.0)
	{
		const double flowIn = static_cast<double>(_flowRateIn);
		const double flowOut = static_cast<double>(_flowRateOut);

		// Volume: \dot{V} = F_{in} - F_{out} - F_{filter}
		const double vDot = flowIn - flowOut - static_cast<double>(_curFlowRateFilter);

		// We have the equation
		//    \dot{V} * (c + 1 / beta * [sum_j q_j]) + V * (\dot{c} + 1 / beta * [sum_j \dot{q}_j]) = c_in * F_in + c * F_out
		// which is now algebraic wrt. c due to V = 0:
		//    \dot{V} * (c + 1 / beta * [sum_j q_j]) = c_in * F_in + c * F_out
		// Separating knowns from unknowns gives
		//    (\dot{V} + F_out) * c = c_in * F_in - \dot{V} / beta * [sum_j q_j]
		// Hence, we obtain
		//    c = (c_in * F_in - \dot{V} / beta * [sum_j q_j]) / (\dot{V} + F_out)

		// Note that if the denominator were 0, we had
		//    0 = \dot{V} + F_out = F_in - F_filter
		// which leads to
		//    F_in = F_filter
		// Since F_out >= 0 and \dot{V} = -F_out, we get
		//    \dot{V} <= 0
		// Assuming a valid configuration, we obtain \dot{V} = 0
		// as the tank would get a negative volume otherwise.
		// Concluding, we arrive at \dot{V} = F_out = 0.
		// In this situation, F_in = F_filter = 0 has to hold
		// as otherwise the liquid (solvent) is immediately and
		// fully taken out, leaving only the pure dry components.
		// We, hence, assume that this doesn't happen and simply
		// do nothing leaving the initial conditions in place.

		const double denom = vDot + flowOut;
		if (denom == 0.0)
			return;

		for (unsigned int i = 0; i < _nComp; ++i)
			c[i] = vecStateY[i] * flowIn / denom;;

		const double qFactor = vDot * (1.0 / static_cast<double>(_porosity) - 1.0) / denom;
		for (unsigned int type = 0; type < _nParType; ++type)
		{
			unsigned int const* const bo = _boundOffset + type * _nComp;
			unsigned int const* const nb = _nBound + type * _nComp;
			double const* const typeQ = c + _nComp + _offsetParType[type];
			for (unsigned int i = 0; i < _nComp; ++i)
			{
				double qSum = 0.0;
				double const* const localQ = typeQ + bo[i];
				for (unsigned int j = 0; j < nb[i]; ++j)
					qSum += localQ[j];

				c[i] -= qFactor * static_cast<double>(_parTypeVolFrac[type]) * qSum;
			}
		}
	}
}

void CSTRModel::leanConsistentInitialTimeDerivative(double t, double const* const vecStateY, double* const vecStateYdot, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	// It is assumed that the bound states are (approximately) correctly initialized.
	// Thus, only the liquid phase has to be initialized

	double const* const c = vecStateY + _nComp;
	double* const cDot = vecStateYdot + _nComp;
	double* const resC = res + _nComp;
	const double v = c[_nComp];

	const double flowIn = static_cast<double>(_flowRateIn);
	const double flowOut = static_cast<double>(_flowRateOut);

	// Volume: \dot{V} = F_{in} - F_{out} - F_{filter}
	const double vDot = flowIn - flowOut - static_cast<double>(_curFlowRateFilter);
	cDot[_nComp] = vDot;

	// Check if volume is 0
	if (v == 0.0)
	{
		// We have the equation
		//    V * \dot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} + \dot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]) = c_{in,i} * F_in - c_i * F_out
		// which is now algebraic wrt. c due to V = 0:
		//    \dot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]) = c_{in,i} * F_in - c_i * F_out
		// So we take the derivative wrt. to time t on both sides
		//    2 * \dot{V} * \dot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} + V * \ddot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} + \ddot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]) = \dot{c}_{in,i} * F_in - \dot{c} * F_out
		// and use the fact that \ddot{V} = 0 and V = 0 to arrive at
		//    2 * \dot{V} * \dot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} = \dot{c}_{in,i} * F_in - \dot{c} * F_out
		// Separating knowns from unknowns gives
		//    (2 * \dot{V} + F_out) * \dot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} = \dot{c}_{in,i} * F_in
		// which finally yields
		//    \dot{c_i + 1 / beta * [sum_j sum_m d_j q_{i,m}]} = \dot{c}_{in,i} * F_in / (2 * \dot{V} + F_out)

		// Note that if the denominator were 0, we had
		//    0 = 2 * \dot{V} + F_out = 2 * F_in - 2 * F_filter - F_out
		// which leads to
		//    F_out = 2 * F_in - 2 * F_filter                       (*)
		// Plugging this back into the \dot{V} equation gives
		//    \dot{V} = F_in - F_out - F_filter = -F_in + F_filter
		// Since V = 0, a valid choice of parameters has to ensure
		// \dot{V} >= 0. In this case, we obtain
		//    F_in <= F_filter
		// On the other hand, we infer from (*) that
		//    0 <= F_out <= 0   =>   F_out = 0   =>   F_in = F_filter
		// This, in turn, concludes \dot{V} = 0. In this situation, 
		//    F_in = F_filter = 0
		// has to hold as otherwise the liquid (solvent) is immediately
		// and fully taken out, leaving only the pure dry components.
		// We, hence, assume that this doesn't happen. Summarizing, we
		// have
		//    \dot{V} = 0, F_in = F_filter = F_out = 0
		// and nothing can happen or change. Therefore, \dot{c} is set
		// to 0.0.

		const double denom = 2.0 * vDot + flowOut;
		if (denom == 0.0)
		{
			// Assume F_in = F_filter = 0
			std::fill(cDot, cDot + _nComp, 0.0);
		}
		else
		{
			const double qFactor = 2.0 * vDot * (1.0 / static_cast<double>(_porosity) - 1.0) / denom;
			const double factor = flowIn / denom;

			for (unsigned int i = 0; i < _nComp; ++i)
			{
				// TODO: This is wrong as vecStateYdot does not contain \dot{c}_in (on entry)
				// This scenario violates the assumption that every outlet DOF is dynamic
				// which is key to the consistent initialization algorithm. Fixing this problem
				// requires fundamental changes to the consistent initialization concept
				// implemented so far.
				vecStateYdot[i] = 0.0;

				cDot[i] = vecStateYdot[i] * factor;

				for (unsigned int type = 0; type < _nParType; ++type)
				{
					unsigned int const* const bo = _boundOffset + type * _nComp;
					unsigned int const* const nb = _nBound + type * _nComp;
					double const* const localQdot = cDot + _nComp + _offsetParType[type] + bo[i];

					double qDotSum = 0.0;
					for (unsigned int j = 0; j < nb[i]; ++j)
						qDotSum += localQdot[j];

					cDot[i] -= qFactor * static_cast<double>(_parTypeVolFrac[type]) * qDotSum;
				}
			}
		}
	}
	else
	{
		const double invBeta = 1.0 / static_cast<double>(_porosity) - 1.0;

		// Concentrations: V * (\dot{c} + 1 / beta * [sum_j sum_m d_j \dot{q}_{j,m}]) = c_in * F_in + c * F_out - \dot{V} * (c + 1 / beta * [sum_j sum_m d_j q_{j,m}])
		//             <=> V * \dot{c} = c_in * F_in + c * F_out - \dot{V} * (c + 1 / beta * [sum_j sum_m d_j q_{j,m}]) - V / beta * [sum_j sum_m d_j \dot{q}_{j,m}]
		//                             = -res - \dot{V} * (c + 1 / beta * [sum_j sum_m d_j q_{j,m}]) - V / beta * [sum_j sum_m d_j \dot{q}_{j,m}]
		// => \dot{c} = (-res - \dot{V} * (c + 1 / beta * [sum_j sum_m d_j q_{j,m}]) - V / beta * [sum_j sum_m d_j \dot{q}_m]) / V
		for (unsigned int i = 0; i < _nComp; ++i)
		{
			double qSum = 0.0;
			double qDotSum = 0.0;
			for (unsigned int type = 0; type < _nParType; ++type)
			{
				double const* const localQ = c + _nComp + _offsetParType[type] + _boundOffset[type * _nComp + i];
				double const* const localQdot = cDot + _nComp +  _offsetParType[type] +_boundOffset[type * _nComp + i];
				double qSumType = 0.0;
				double qDotSumType = 0.0;
				for (unsigned int j = 0; j < _nBound[type * _nComp + i]; ++j)
				{
					qSumType += localQ[j];
					qDotSumType += localQdot[j];
				}
				qSum += static_cast<double>(_parTypeVolFrac[type]) * qSumType;
				qDotSum += static_cast<double>(_parTypeVolFrac[type]) * qDotSumType;
			}

			cDot[i] = (-resC[i] - vDot * (c[i] + invBeta * qSum) - v * invBeta * qDotSum) / v;
		}
	}
}

void CSTRModel::leanConsistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
	std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	consistentInitialSensitivity(simTime, simState, vecSensY, vecSensYdot, adRes, threadLocalMem);
}

int CSTRModel::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	return residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem.get());
}

template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int CSTRModel::residualImpl(double t, unsigned int secIdx, StateType const* const y, double const* const yDot, ResidualType* const res, LinearBufferAllocator tlmAlloc)
{
	StateType const* const cIn = y;
	StateType const* const c = y + _nComp;
	const StateType& v = y[2 * _nComp + _totalBound];

	double const* const cDot = yDot ? yDot + _nComp : nullptr;
	const double vDot = yDot ? yDot[2 * _nComp + _totalBound] : 0.0;

	const ParamType flowIn = static_cast<ParamType>(_flowRateIn);
	const ParamType flowOut = static_cast<ParamType>(_flowRateOut);

	// Inlet DOF
	for (unsigned int i = 0; i < _nComp; ++i)
	{
		res[i] = cIn[i];
	}

	// Concentrations: \dot{V} * (c + 1 / beta * [sum_j q_j]) + V * (\dot{c} + 1 / beta * [sum_j \dot{q}_j]) = c_in * F_in - c * F_out
	const ParamType invBeta = 1.0 / static_cast<ParamType>(_porosity) - 1.0;
	ResidualType* const resC = res + _nComp;
	for (unsigned int i = 0; i < _nComp; ++i)
	{
		resC[i] = 0.0;

		// Add time derivatives
		if (cadet_likely(yDot))
		{
			// Ultimately, we need (dc_{i} / dt + 1 / beta * [ sum_j sum_m d_j dq_{j,i,m} / dt ]) * V
			// and (c_{i} + 1 / beta * [ sum_j sum_m d_j q_{i,m} ]) * dV / dt
			// Compute the sum in the brackets first, then divide by beta and add remaining term

			// Sum q_{i,1} + q_{i,2} + ... + q_{i,N_i}
			// and sum dq_{i,1} / dt + dq_{i,2} / dt + ... + dq_{i,N_i} / dt
			typename DoubleActivePromoter<StateType, ResidualType>::type qSum = 0.0;
			typename ActivePromoter<ParamType>::type qDotSum = 0.0;

			for (unsigned int type = 0; type < _nParType; ++type)
			{
				StateType const* const q = c + _nComp + _offsetParType[type] + _boundOffset[type * _nComp + i];
				double const* const qDot = cDot + _nComp + _offsetParType[type] + _boundOffset[type * _nComp + i];
				
				StateType qSumType = 0.0;
				double qDotSumType = 0.0;
				for (unsigned int j = 0; j < _nBound[type * _nComp + i]; ++j)
				{
					qSumType += q[j];
					qDotSumType += qDot[j];
				}

				qSum += static_cast<ParamType>(_parTypeVolFrac[type]) * qSumType;
				qDotSum += static_cast<ParamType>(_parTypeVolFrac[type]) * qDotSumType;
			}

			// Divide by beta and add c_i and dc_i / dt
			resC[i] = ((cDot[i] + invBeta * qDotSum) * v + vDot * (c[i] + invBeta * qSum));
		}

		resC[i] += -flowIn * cIn[i] + flowOut * c[i];
	}

	if (wantJac)
	{
		_jac.setAll(0.0);

		// Assemble Jacobian: Liquid phase

		// Concentrations: \dot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{j,i,m}]) + V * (\dot{c}_i + 1 / beta * [sum_j sum_m d_j \dot{q}_{j,i,m}]) - c_{in,i} * F_in + c_i * F_out == 0
		const double vDotTimeFactor = static_cast<double>(vDot);
		for (unsigned int i = 0; i < _nComp; ++i)
		{
			_jac.native(i, i) = vDotTimeFactor + static_cast<double>(flowOut);

			if (cadet_likely(yDot))
			{
				double qDotSum = 0.0;
				const double vDotInvBeta = vDotTimeFactor * static_cast<double>(invBeta);
				for (unsigned int type = 0; type < _nParType; ++type)
				{
					double const* const qiDot = cDot + _nComp + _offsetParType[type] + _boundOffset[type * _nComp + i];
					const unsigned int localOffset = _nComp + _offsetParType[type] + _boundOffset[type * _nComp + i];

					const double vDotInvBetaParVolFrac = vDotInvBeta * static_cast<double>(_parTypeVolFrac[type]);
					double qDotSumType = 0.0;
					for (unsigned int j = 0; j < _nBound[type * _nComp + i]; ++j)
					{
						_jac.native(i, localOffset + j) = vDotInvBetaParVolFrac;
						// + _nComp: Moves over liquid phase components
						// + _offsetParType[type]: Moves to particle type
						// + _boundOffset[i]: Moves over bound states of previous components
						// + j: Moves to current bound state j of component i

						qDotSumType += qiDot[j];
					}

					qDotSum += static_cast<double>(_parTypeVolFrac[type]) * qDotSumType;
				}

				_jac.native(i, _nComp + _totalBound) = (cDot[i] + static_cast<double>(invBeta) * qDotSum);
			}
		}
	}

	// Reactions in liquid phase
	const ColumnPosition colPos{0.0, 0.0, 0.0};

	if (_dynReactionBulk && (_dynReactionBulk->numReactionsLiquid() > 0))
	{
		LinearBufferAllocator subAlloc = tlmAlloc.manageRemainingMemory();
		BufferedArray<ResidualType> flux = subAlloc.array<ResidualType>(_nComp);

		std::fill_n(static_cast<ResidualType*>(flux), _nComp, 0.0);
		_dynReactionBulk->residualLiquidAdd(t, secIdx, colPos, c, static_cast<ResidualType*>(flux), -1.0, subAlloc);

		for (unsigned int comp = 0; comp < _nComp; ++comp)
			resC[comp] += v * flux[comp];

		if (wantJac)
		{
			for (unsigned int comp = 0; comp < _nComp; ++comp)
				_jac.native(comp, _nComp + _totalBound) += static_cast<double>(flux[comp]);

			_dynReactionBulk->analyticJacobianLiquidAdd(t, secIdx, colPos, reinterpret_cast<double const*>(c), -static_cast<double>(v), _jac.row(0), subAlloc);
		}
	}

	// Bound states
	for (unsigned int type = 0; type < _nParType; ++type)
	{
		// Binding
		IBindingModel* const binding = _binding[type];
		bindingFlux(binding, t, secIdx, colPos, c, c + _nComp + _offsetParType[type], res + 2 * _nComp + _offsetParType[type], tlmAlloc, typename ParamSens<ParamType>::enabled());

		int const* const qsReaction = binding->reactionQuasiStationarity();

		if (binding->hasDynamicReactions() && yDot)
		{
			double const* const qDot = yDot + 2 * _nComp + _offsetParType[type];
			ResidualType* const resQ = resC + _nComp + _offsetParType[type];
			unsigned int idx = 0;
			for (unsigned int comp = 0; comp < _nComp; ++comp)
			{
				for (unsigned int state = 0; state < _nBound[type * _nComp + comp]; ++state, ++idx)
				{
					// Skip quasi-stationary fluxes
					if (qsReaction[idx])
						continue;

					// Add time derivative to solid phase
					resQ[idx] += qDot[idx];
				}
			}
		}

		if (wantJac)
		{
			// Assemble Jacobian: Binding
			_binding[type]->analyticJacobian(t, secIdx, colPos, reinterpret_cast<double const*>(y) + 2 * _nComp + _offsetParType[type], _nComp + _offsetParType[type], _jac.row(_nComp + _offsetParType[type]), tlmAlloc);
		}

		// Reaction
		IDynamicReactionModel* const dynReaction = _dynReaction[type];
		if (dynReaction && (dynReaction->numReactionsCombined() > 0))
		{
			LinearBufferAllocator subAlloc = tlmAlloc.manageRemainingMemory();

			ResidualType* const resQ = resC + _nComp + _offsetParType[type];
			BufferedArray<ResidualType> fluxBuffer = subAlloc.array<ResidualType>(_nComp + _strideBound[type]);
			ResidualType* const fluxLiquid = static_cast<ResidualType*>(fluxBuffer);
			ResidualType* const fluxSolid = fluxLiquid + _nComp;

			std::fill_n(fluxLiquid, _nComp, 0.0);
			std::fill_n(fluxSolid, _strideBound[type], 0.0);
			dynReaction->residualCombinedAdd(t, secIdx, colPos, c, c + _nComp + _offsetParType[type], fluxLiquid, fluxSolid, -1.0, subAlloc);

			for (unsigned int comp = 0; comp < _nComp; ++comp)
				resC[comp] += v * fluxLiquid[comp];

			typedef typename DoubleActivePromoter<StateType, ParamType>::type FactorType;
			const FactorType liquidFactor = v * invBeta * static_cast<ParamType>(_parTypeVolFrac[type]);
			unsigned int idx = 0;
			for (unsigned int comp = 0; comp < _nComp; ++comp)
			{
				for (unsigned int bnd = 0; bnd < _nBound[type * _nComp + comp]; ++bnd, ++idx)
				{
					// Add reaction term to mobile phase
					resC[comp] += static_cast<typename DoubleActiveDemoter<FactorType, ResidualType>::type>(liquidFactor) * fluxSolid[idx];

					if (!qsReaction[idx])
					{
						// Add reaction term to solid phase
						resQ[idx] += fluxSolid[idx];
					}
				}
			}

			if (wantJac)
			{
				// Assemble Jacobian: Reaction

				// dRes / dV
				idx = 0;
				for (unsigned int comp = 0; comp < _nComp; ++comp)
				{
					double sum = 0;
					for (unsigned int bnd = 0; bnd < _nBound[type * _nComp + comp]; ++bnd, ++idx)
						sum += static_cast<double>(fluxSolid[idx]);

					_jac.native(comp, _nComp + _totalBound) += static_cast<double>(fluxLiquid[comp]) + static_cast<double>(invBeta) * static_cast<double>(_parTypeVolFrac[type]) * sum;
				}

				// dRes / dC and dRes / dQ
				BufferedArray<double> fluxJacobianMem = subAlloc.array<double>((_strideBound[type] + _nComp) * (_strideBound[type] + _nComp));
				linalg::DenseMatrixView jacFlux(static_cast<double*>(fluxJacobianMem), nullptr, _strideBound[type] + _nComp, _strideBound[type] + _nComp);
				dynReaction->analyticJacobianCombinedAdd(t, secIdx, colPos, reinterpret_cast<double const*>(c), reinterpret_cast<double const*>(c + _nComp + _offsetParType[type]),
					-1.0, jacFlux.row(0), jacFlux.row(_nComp), subAlloc);

				idx = 0;
				const double liquidFactor = static_cast<double>(v) * static_cast<double>(invBeta) * static_cast<double>(_parTypeVolFrac[type]);
				for (unsigned int comp = 0; comp < _nComp; ++comp)
				{
					// Add bulk part of reaction to mobile phase Jacobian
					jacFlux.addSubmatrixTo(_jac, static_cast<double>(v), comp, 0, 1, _nComp, comp, 0);
					jacFlux.addSubmatrixTo(_jac, static_cast<double>(v), comp, _nComp, 1, _strideBound[type], comp, _nComp + _offsetParType[type]);

					for (unsigned int bnd = 0; bnd < _nBound[type * _nComp + comp]; ++bnd, ++idx)
					{
						// Add Jacobian row to mobile phase
						jacFlux.addSubmatrixTo(_jac, liquidFactor, _nComp + idx, 0, 1, _nComp, comp, 0);
						jacFlux.addSubmatrixTo(_jac, liquidFactor, _nComp + idx, _nComp, 1, _strideBound[type], comp, _nComp + _offsetParType[type]);

						if (!qsReaction[idx])
						{
							// Add Jacobian row to solid phase
							jacFlux.addSubmatrixTo(_jac, 1.0, _nComp + idx, 0, 1, _nComp, _nComp + _offsetParType[type] + idx, 0);
							jacFlux.addSubmatrixTo(_jac, 1.0, _nComp + idx, _nComp, 1, _strideBound[type], _nComp + _offsetParType[type] + idx, _nComp + _offsetParType[type]);
						}
					}
				}
			}
		}
	}

	// Volume: \dot{V} = F_{in} - F_{out} - F_{filter}
	res[2 * _nComp + _totalBound] = vDot - flowIn + flowOut + static_cast<ParamType>(_curFlowRateFilter);

	return 0;
}

int CSTRModel::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res,
	const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem, bool updateJacobian, bool paramSensitivity)
{
	if (updateJacobian)
	{
		_factorizeJac = true;

#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
		if (_analyticJac)
		{
			if (paramSensitivity)
			{
				const int retCode = residualImpl<double, active, active, true>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, adJac.adRes, threadLocalMem.get());

				// Copy AD residuals to original residuals vector
				if (res)
					ad::copyFromAd(adJac.adRes, res, numDofs());

				return retCode;
			}
			else
				return residualImpl<double, double, double, true>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem.get());
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
				retCode = residualImpl<active, active, active, false>(simTime.t, simTime.secIdx, adJac.adY, simState.vecStateYdot, adJac.adRes, threadLocalMem.get());
			else
				retCode = residualImpl<active, active, double, false>(simTime.t, simTime.secIdx, adJac.adY, simState.vecStateYdot, adJac.adRes, threadLocalMem.get());

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
			retCode = residualImpl<active, active, active, false>(simTime.t, simTime.secIdx, adJac.adY, simState.vecStateYdot, adJac.adRes, threadLocalMem.get());
		else
			retCode = residualImpl<active, active, double, false>(simTime.t, simTime.secIdx, adJac.adY, simState.vecStateYdot, adJac.adRes, threadLocalMem.get());

		// Only do comparison if we have a residuals vector (which is not always the case)
		if (res)
		{
			// Evaluate with analytical Jacobian which is stored in the band matrices
			retCode = residualImpl<double, double, double, true>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem.get());

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

			const int retCode = residualImpl<double, active, active, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, adJac.adRes, threadLocalMem.get());

			// Copy AD residuals to original residuals vector
			if (res)
				ad::copyFromAd(adJac.adRes, res, numDofs());

			return retCode;
		}
		else
			return residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem.get());
	}
}

int CSTRModel::residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	return residual(simTime, simState, res, adJac, threadLocalMem, true, false);
}

int CSTRModel::residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	// Evaluate residual for all parameters using AD in vector mode
	return residualImpl<double, active, active, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, adRes, threadLocalMem.get());
}

int CSTRModel::residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	// Evaluate residual for all parameters using AD in vector mode and at the same time update the 
	// Jacobian (in one AD run, if analytic Jacobians are disabled)
	return residual(simTime, simState, nullptr, adJac, threadLocalMem, true, true);
}

void CSTRModel::initializeSensitivityStates(const std::vector<double*>& vecSensY) const
{
	const unsigned int nDof = numPureDofs();
	for (std::size_t param = 0; param < vecSensY.size(); ++param)
	{
		double* const sensY = vecSensY[param];
		ad::copyFromAdDirection(_initConditions.data(), sensY + _nComp, nDof, param);
	}
}

void CSTRModel::consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
	std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	// TODO: Handle v = 0

	LinearBufferAllocator tlmAlloc = threadLocalMem.get();
	BufferedArray<int> qsMaskBuffer = tlmAlloc.array<int>(_totalBound);
	int* const qsMask = static_cast<int*>(qsMaskBuffer);
	int* localMask = static_cast<int*>(qsMask);

	// Check whether quasi-stationary reactions are present and construct mask array
	bool hasQSreactions = false;
	std::fill_n(static_cast<int*>(qsMask), _totalBound, false);
	for (unsigned int type = 0; type < _nParType; localMask += _strideBound[type], ++type)
	{
		if (!_binding[type]->hasQuasiStationaryReactions())
			continue;

		hasQSreactions = true;

		// Construct mask
		int const* const qsMaskSrc = _binding[type]->reactionQuasiStationarity();
		std::copy_n(qsMaskSrc, _strideBound[type], localMask);
	}

	const linalg::ConstMaskArray mask{static_cast<int*>(qsMask), static_cast<int>(_totalBound)};
	const int probSize = linalg::numMaskActive(mask);
	BufferedArray<double> buffer = tlmAlloc.array<double>(_nComp + _totalBound + 1);
	BufferedArray<double> rhs = tlmAlloc.array<double>(probSize);
	BufferedArray<double> rhsUnmasked = tlmAlloc.array<double>(_totalBound);

	for (std::size_t param = 0; param < vecSensY.size(); ++param)
	{
		double* const sensY = vecSensY[param];
		double* const sensYdot = vecSensYdot[param];

		// Copy parameter derivative dF / dp from AD and negate it
		for (unsigned int i = _nComp; i < numDofs(); ++i)
			sensYdot[i] = -adRes[i].getADValue(param);

		// Step 1: Solve algebraic equations

		// Step 1a: Compute quasi-stationary binding model state
		if (hasQSreactions)
		{
			double* const maskedMultiplier = static_cast<double*>(buffer);

			linalg::DenseMatrixView jacobianMatrix(_jacFact.data(), _jacFact.pivotData(), probSize, probSize);
			jacobianMatrix.setAll(0.0);

			// Extract subproblem Jacobian from full Jacobian
			linalg::copyMatrixSubset(_jac, mask, mask, _nComp, _nComp, jacobianMatrix, 0);

			// Construct right hand side
			linalg::selectVectorSubset(sensYdot + 2 * _nComp, mask, static_cast<double*>(rhs));

			// Zero out masked elements
			std::copy_n(sensY + _nComp, _nComp + _totalBound + 1, maskedMultiplier);
			linalg::fillVectorSubset(maskedMultiplier + _nComp, mask, 0.0);

			// Assemble right hand side
			_jac.submatrixMultiplyVector(maskedMultiplier, _nComp, 0, _totalBound, _nComp + _totalBound + 1, static_cast<double*>(rhsUnmasked));
			linalg::vectorSubsetAdd(static_cast<double*>(rhsUnmasked), mask, -1.0, 1.0, static_cast<double*>(rhs));

			// Precondition
			double* const scaleFactors = static_cast<double*>(buffer);
			jacobianMatrix.rowScaleFactors(scaleFactors);
			jacobianMatrix.scaleRows(scaleFactors);

			// Solve
			jacobianMatrix.factorize();
			jacobianMatrix.solve(scaleFactors, static_cast<double*>(rhs));

			// Write back
			linalg::applyVectorSubset(static_cast<double*>(rhs), mask, sensY + 2 * _nComp);
		}

		// Step 2: Compute the correct time derivative of the state vector

		// Step 2a: Assemble, factorize, and solve diagonal blocks of linear system

		// Compute right hand side by adding -dF / dy * s = -J * s to -dF / dp which is already stored in sensYdot
		multiplyWithJacobian(simTime, simState, sensY, -1.0, 1.0, sensYdot);

		// Note that we have correctly negated the right hand side

		// Assemble dF / dYdot
		_jacFact.setAll(0.0);
		addTimeDerivativeJacobian(simTime.t, 1.0, simState, _jacFact);

		// Overwrite rows corresponding to algebraic equations with the Jacobian and set right hand side to 0
		for (unsigned int type = 0; type < _nParType; ++type)
		{
			if (!_binding[type]->hasQuasiStationaryReactions())
				continue;

			double* const localQdot = sensYdot + 2 * _nComp + _offsetParType[type];
			int const* const localMask = mask.mask + _offsetParType[type];
			for (unsigned int i = 0; i < _strideBound[type]; ++i)
			{
				if (!localMask[i])
					continue;

				const unsigned int idx = _nComp + _offsetParType[type] + i;
				_jacFact.copyRowFrom(_jac, idx, idx);

				// Right hand side is -\frac{\partial^2 res(t, y, \dot{y})}{\partial p \partial t}
				// If the residual is not explicitly depending on time, this expression is 0
				// @todo This is wrong if external functions are used. Take that into account!
				localQdot[i] = 0.0;
			}
		}

		// Factorize
		const bool result = _jacFact.robustFactorize(static_cast<double*>(buffer));
		if (!result)
		{
			LOG(Error) << "Factorize() failed";
		}

		// Solve
		const bool result2 = _jacFact.robustSolve(sensYdot + _nComp, static_cast<double*>(buffer));
		if (!result2)
		{
			LOG(Error) << "Solve() failed";
		}
	}
}

void CSTRModel::multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret)
{
	const double flowIn = static_cast<double>(_flowRateIn);
	double* const resTank = ret + _nComp;

	// Inlet DOFs
	for (unsigned int i = 0; i < _nComp; ++i)
	{
		ret[i] = alpha * yS[i] + beta * ret[i];
	}

	// Multiply with main body Jacobian: dRes / dy
	_jac.multiplyVector(yS + _nComp, alpha, beta, resTank);

	// Map inlet DOFs to the tank (tank cells)
	for (unsigned int i = 0; i < _nComp; ++i)
	{
		resTank[i] -= alpha * flowIn * yS[i];
	}
}

void CSTRModel::multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret)
{
	// Handle inlet DOFs (all algebraic)
	std::fill_n(ret, _nComp, 0.0);

	// Handle actual ODE DOFs
	double const* const c = simState.vecStateY + _nComp;
	double const* const q = simState.vecStateY + 2 * _nComp;
	const double v = simState.vecStateY[2 * _nComp + _totalBound];
	const double invBeta = 1.0 / static_cast<double>(_porosity) - 1.0;
	const double timeV = v;
	const double vInvBeta = timeV * invBeta;
	double* const r = ret + _nComp;
	double const* const s = sDot + _nComp;

	// Concentrations: \dot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{j,i,m}]) + V * (\dot{c}_i + 1 / beta * [sum_j sum_m d_j \dot{q}_{j,i,m}]) - c_{in,i} * F_in + c_i * F_out == 0
	for (unsigned int i = 0; i < _nComp; ++i)
	{
		r[i] = timeV * s[i];

		double qSum = 0.0;
		for (unsigned int type = 0; type < _nParType; ++type)
		{
			double const* const qi = q + _offsetParType[type] + _boundOffset[type * _nComp + i];
			const unsigned int localOffset = _nComp + _offsetParType[type] + _boundOffset[type * _nComp + i];
			const double vInvBetaParVolFrac = vInvBeta * static_cast<double>(_parTypeVolFrac[type]);
			double qSumType = 0.0;
			for (unsigned int j = 0; j < _nBound[type * _nComp + i]; ++j)
			{
				r[i] += vInvBetaParVolFrac * s[localOffset + j];
				// + _nComp: Moves over liquid phase components
				// + _offsetParType[type]: Moves to particle type
				// + _boundOffset[i]: Moves over bound states of previous components
				// + j: Moves to current bound state j of component i

				qSumType += qi[j];
			}

			qSum += static_cast<double>(_parTypeVolFrac[type]) * qSumType;
		}
		r[i] += (c[i] + invBeta * qSum) * s[_nComp + _totalBound];
	}

	// Bound states
	for (unsigned int type = 0; type < _nParType; ++type)
	{
		// Jump to solid phase of current particle type
		double* const rQ = r + _nComp + _offsetParType[type];
		double const* const sQ = s + _nComp + _offsetParType[type];
		std::fill_n(rQ, _strideBound[type], 0.0);

		// Skip binding models without dynamic binding fluxes
		IBindingModel* const binding = _binding[type];
		if (!binding->hasDynamicReactions())
			continue;

		int const* const qsReaction = binding->reactionQuasiStationarity();
		for (unsigned int idx = 0; idx < _strideBound[type]; ++idx)
		{
			// Skip quasi-stationary fluxes
			if (qsReaction[idx])
				continue;

			rQ[idx] = sQ[idx];
		}
	}

	// Volume: \dot{V} - F_{in} + F_{out} + F_{filter} == 0
	r[_nComp + _totalBound] = s[_nComp + _totalBound];
}

int CSTRModel::linearSolve(double t, double alpha, double tol, double* const rhs, double const* const weight,
	const ConstSimulationState& simState)
{
	const double flowIn = static_cast<double>(_flowRateIn);

	// Handle inlet equations by backsubstitution
	for (unsigned int i = 0; i < _nComp; ++i)
	{
		rhs[i + _nComp] += flowIn * rhs[i];
	}

	bool success = true;
	if (_factorizeJac)
	{
		// Factorization is necessary
		_factorizeJac = false;
		_jacFact.copyFrom(_jac);

		addTimeDerivativeJacobian(t, alpha, simState, _jacFact);
		success = _jacFact.factorize();
	}
	success = success && _jacFact.solve(rhs + _nComp);

	// Return 0 on success and 1 on failure
	return success ? 0 : 1;
}

template <typename MatrixType>
void CSTRModel::addTimeDerivativeJacobian(double t, double alpha, const ConstSimulationState& simState, MatrixType& mat)
{
	double const* const c = simState.vecStateY + _nComp;
	double const* const q = simState.vecStateY + 2 * _nComp;
	const double v = simState.vecStateY[2 * _nComp + _totalBound];
	const double invBeta = 1.0 / static_cast<double>(_porosity) - 1.0;
	const double timeV = v * alpha;
	const double vInvBeta = timeV * invBeta;

	// Assemble Jacobian: dRes / dyDot

	// Concentrations: \dot{V} * (c_i + 1 / beta * [sum_j sum_m d_j q_{j,i,m}]) + V * (\dot{c}_i + 1 / beta * [sum_j sum_m d_j \dot{q}_{j,i,m}]) - c_{in,i} * F_in + c_i * F_out == 0
	for (unsigned int i = 0; i < _nComp; ++i)
	{
		mat.native(i, i) += timeV;

		double qSum = 0.0;
		for (unsigned int type = 0; type < _nParType; ++type)
		{
			double const* const qi = q + _offsetParType[type] + _boundOffset[type * _nComp + i];
			const unsigned int localOffset = _nComp + _offsetParType[type] + _boundOffset[type * _nComp + i];
			const double vInvBetaParVolFrac = vInvBeta * static_cast<double>(_parTypeVolFrac[type]);
			double qSumType = 0.0;
			for (unsigned int j = 0; j < _nBound[type * _nComp + i]; ++j)
			{
				mat.native(i, localOffset + j) += vInvBetaParVolFrac;
				// + _nComp: Moves over liquid phase components
				// + _offsetParType[type]: Moves to particle type
				// + _boundOffset[i]: Moves over bound states of previous components
				// + j: Moves to current bound state j of component i

				qSumType += qi[j];
			}

			qSum += static_cast<double>(_parTypeVolFrac[type]) * qSumType;
		}
		mat.native(i, _nComp + _totalBound) += alpha * (c[i] + invBeta * qSum);
	}

	// Bound states
	unsigned int globalIdx = _nComp;
	for (unsigned int type = 0; type < _nParType; ++type)
	{
		IBindingModel* const binding = _binding[type];
		if (!binding->hasDynamicReactions())
		{
			// Skip binding models without dynamic binding fluxes
			globalIdx += _strideBound[type];
			continue;
		}

		int const* const qsReaction = binding->reactionQuasiStationarity();
		for (unsigned int idx = 0; idx < _strideBound[type]; ++idx, ++globalIdx)
		{
			// Skip quasi-stationary fluxes
			if (qsReaction[idx])
				continue;

			mat.native(globalIdx, globalIdx) += alpha;
		}
	}

	// Volume: \dot{V} - F_{in} + F_{out} + F_{filter} == 0
	mat.native(_nComp + _totalBound, _nComp + _totalBound) += alpha;
}

/**
 * @brief Extracts the system Jacobian from AD seed vectors
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
void CSTRModel::extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset)
{
	ad::extractDenseJacobianFromAd(adRes + _nComp, adDirOffset, _jac);
}

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN

/**
 * @brief Compares the analytical Jacobian with a Jacobian derived by AD
 * @details The analytical Jacobian is assumed to be stored in the dense matrix.
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
void CSTRModel::checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const
{
	const double diff = ad::compareDenseJacobianWithAd(adRes + _nComp, adDirOffset, _jac);
	LOG(Debug) << "AD dir offset: " << adDirOffset << " diff: " << diff;
}

#endif


unsigned int CSTRModel::Exporter::numSolidPhaseDofs() const CADET_NOEXCEPT
{
	return _totalBound;
}

int CSTRModel::Exporter::writeMobilePhase(double* buffer) const
{
	std::copy_n(_data + _nComp, _nComp, buffer);
	return _nComp;
}

int CSTRModel::Exporter::writeSolidPhase(double* buffer) const
{
	if (_totalBound == 0)
		return 0;

	std::copy_n(_data + 2 * _nComp, _totalBound, buffer);
	return _totalBound;
}

int CSTRModel::Exporter::writeSolidPhase(unsigned int parType, double* buffer) const
{
	if (_totalBound == 0)
		return 0;

	cadet_assert(parType < _nParType);

	unsigned int offset = 0;
	for (int i = 0; i < parType; ++i)
		offset += _strideBound[i];

	std::copy_n(_data + 2 * _nComp + offset, _strideBound[parType], buffer);
	return _strideBound[parType];
}

int CSTRModel::Exporter::writeVolume(double* buffer) const
{
	*buffer = _data[2 * _nComp + _totalBound];
	return 1;
}

int CSTRModel::Exporter::writeInlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);
	std::copy_n(_data, _nComp, buffer);
	return _nComp;
}

int CSTRModel::Exporter::writeInlet(double* buffer) const
{
	std::copy_n(_data, _nComp, buffer);
	return _nComp;
}

int CSTRModel::Exporter::writeOutlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);
	std::copy_n(_data + _nComp, _nComp, buffer);
	return _nComp;
}

int CSTRModel::Exporter::writeOutlet(double* buffer) const
{
	std::copy_n(_data + _nComp, _nComp, buffer);
	return _nComp;
}



void registerCSTRModel(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models)
{
	models[CSTRModel::identifier()] = [](UnitOpIdx uoId) { return new CSTRModel(uoId); };
}

}  // namespace model

}  // namespace cadet
