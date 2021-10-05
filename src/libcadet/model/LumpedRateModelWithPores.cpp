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

#include "model/LumpedRateModelWithPores.hpp"
#include "BindingModelFactory.hpp"
#include "ParamReaderHelper.hpp"
#include "ParamReaderScopes.hpp"
#include "cadet/Exceptions.hpp"
#include "cadet/ExternalFunction.hpp"
#include "cadet/SolutionRecorder.hpp"
#include "ConfigurationHelper.hpp"
#include "model/BindingModel.hpp"
#include "model/ReactionModel.hpp"
#include "model/parts/BindingCellKernel.hpp"
#include "model/ParameterDependence.hpp"
#include "SimulationTypes.hpp"
#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"
#include "linalg/Norms.hpp"

#include "Stencil.hpp"
#include "Weno.hpp"
#include "AdUtils.hpp"
#include "SensParamUtil.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include <algorithm>
#include <functional>

#include "ParallelSupport.hpp"
#ifdef CADET_PARALLELIZE
	#include <tbb/parallel_for.h>
#endif

namespace cadet
{

namespace model
{

constexpr double SurfVolRatioSphere = 3.0;
constexpr double SurfVolRatioCylinder = 2.0;
constexpr double SurfVolRatioSlab = 1.0;

template <typename Operator>
int schurComplementMultiplierLRMPores(void* userData, double const* x, double* z)
{
	typedef LumpedRateModelWithPores<Operator> LRMP;
	LRMP* const lrm = static_cast<LRMP*>(userData);
	return lrm->schurComplementMatrixVector(x, z);
}


template <typename ConvDispOperator>
LumpedRateModelWithPores<ConvDispOperator>::LumpedRateModelWithPores(UnitOpIdx unitOpIdx) : UnitOperationBase(unitOpIdx),
	_dynReactionBulk(nullptr), _filmDiffDep(nullptr), _jacP(0), _jacPdisc(0), _jacPF(0), _jacFP(0), _jacInlet(), _analyticJac(true),
	_jacobianAdDirs(0), _factorizeJacobian(false), _tempState(nullptr), _initC(0), _initCp(0), _initQ(0),
	_initState(0), _initStateDot(0)
{
}

template <typename ConvDispOperator>
LumpedRateModelWithPores<ConvDispOperator>::~LumpedRateModelWithPores() CADET_NOEXCEPT
{
	delete[] _tempState;

	delete _dynReactionBulk;
	delete _filmDiffDep;

	delete[] _disc.parTypeOffset;
	delete[] _disc.nBound;
	delete[] _disc.boundOffset;
	delete[] _disc.strideBound;
	delete[] _disc.nBoundBeforeType;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithPores<ConvDispOperator>::numDofs() const CADET_NOEXCEPT
{
	// Column bulk DOFs: nCol * nComp
	// Particle DOFs: nCol * nParType particles each having nComp (liquid phase) + sum boundStates (solid phase) DOFs
	// Flux DOFs: nCol * nComp * nParType
	// Inlet DOFs: nComp
	return _disc.nComp + _disc.nComp * _disc.nCol * (1 + _disc.nParType) + _disc.parTypeOffset[_disc.nParType];
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithPores<ConvDispOperator>::numPureDofs() const CADET_NOEXCEPT
{
	// Column bulk DOFs: nCol * nComp
	// Particle DOFs: nCol * nParType particles each having nComp (liquid phase) + sum boundStates (solid phase) DOFs
	// Flux DOFs: nCol * nComp * nParType
	return _disc.nComp * _disc.nCol * (1 + _disc.nParType) + _disc.parTypeOffset[_disc.nParType];
}


template <typename ConvDispOperator>
bool LumpedRateModelWithPores<ConvDispOperator>::usesAD() const CADET_NOEXCEPT
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
bool LumpedRateModelWithPores<ConvDispOperator>::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper)
{
	// ==== Read discretization
	_disc.nComp = paramProvider.getInt("NCOMP");

	paramProvider.pushScope("discretization");

	_disc.nCol = paramProvider.getInt("NCOL");

	const std::vector<int> nBound = paramProvider.getIntArray("NBOUND");
	if (nBound.size() < _disc.nComp)
		throw InvalidParameterException("Field NBOUND contains too few elements (NCOMP = " + std::to_string(_disc.nComp) + " required)");

	if (nBound.size() % _disc.nComp != 0)
		throw InvalidParameterException("Field NBOUND must have a size divisible by NCOMP (" + std::to_string(_disc.nComp) + ")");

	if (paramProvider.exists("NPARTYPE"))
	{
		_disc.nParType = paramProvider.getInt("NPARTYPE");
		_disc.nBound = new unsigned int[_disc.nComp * _disc.nParType];
		if (nBound.size() < _disc.nComp * _disc.nParType)
		{
			// Multiplex number of bound states to all particle types
			for (unsigned int i = 0; i < _disc.nParType; ++i)
				std::copy_n(nBound.begin(), _disc.nComp, _disc.nBound + i * _disc.nComp);
		}
		else
			std::copy_n(nBound.begin(), _disc.nComp * _disc.nParType, _disc.nBound);
	}
	else
	{
		// Infer number of particle types
		_disc.nParType = nBound.size() / _disc.nComp;
		_disc.nBound = new unsigned int[_disc.nComp * _disc.nParType];
		std::copy_n(nBound.begin(), _disc.nComp * _disc.nParType, _disc.nBound);
	}

	// Precompute offsets and total number of bound states (DOFs in solid phase)
	const unsigned int nTotalBound = std::accumulate(_disc.nBound, _disc.nBound + _disc.nComp * _disc.nParType, 0u);

	// Precompute offsets and total number of bound states (DOFs in solid phase)
	_disc.boundOffset = new unsigned int[_disc.nComp * _disc.nParType];
	_disc.strideBound = new unsigned int[_disc.nParType + 1];
	_disc.nBoundBeforeType = new unsigned int[_disc.nParType];
	_disc.strideBound[_disc.nParType] = nTotalBound;
	_disc.nBoundBeforeType[0] = 0;
	for (unsigned int j = 0; j < _disc.nParType; ++j)
	{
		unsigned int* const ptrOffset = _disc.boundOffset + j * _disc.nComp;
		unsigned int* const ptrBound = _disc.nBound + j * _disc.nComp;

		ptrOffset[0] = 0;
		for (unsigned int i = 1; i < _disc.nComp; ++i)
		{
			ptrOffset[i] = ptrOffset[i - 1] + ptrBound[i - 1];
		}
		_disc.strideBound[j] = ptrOffset[_disc.nComp - 1] + ptrBound[_disc.nComp - 1];

		if (j != _disc.nParType - 1)
			_disc.nBoundBeforeType[j + 1] = _disc.nBoundBeforeType[j] + _disc.strideBound[j];
	}

	// Precompute offsets of particle type DOFs
	_disc.parTypeOffset = new unsigned int[_disc.nParType + 1];
	_disc.parTypeOffset[0] = 0;
	for (unsigned int j = 1; j < _disc.nParType + 1; ++j)
	{
		_disc.parTypeOffset[j] = _disc.parTypeOffset[j-1] + (_disc.nComp + _disc.strideBound[j-1]) * _disc.nCol;
	}

	// Determine whether analytic Jacobian should be used but don't set it right now.
	// We need to setup Jacobian matrices first.
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	const bool analyticJac = paramProvider.getBool("USE_ANALYTIC_JACOBIAN");
#else
	const bool analyticJac = false;
#endif

	// Initialize and configure GMRES for solving the Schur-complement
	_gmres.initialize(_disc.nCol * _disc.nComp * _disc.nParType, paramProvider.getInt("MAX_KRYLOV"), linalg::toOrthogonalization(paramProvider.getInt("GS_TYPE")), paramProvider.getInt("MAX_RESTARTS"));
	_gmres.matrixVectorMultiplier(&schurComplementMultiplierLRMPores<ConvDispOperator>, this);
	_schurSafety = paramProvider.getDouble("SCHUR_SAFETY");

	// Allocate space for initial conditions
	_initC.resize(_disc.nComp);
	_initCp.resize(_disc.nComp * _disc.nParType);
	_initQ.resize(nTotalBound);

	// Create nonlinear solver for consistent initialization
	configureNonlinearSolver(paramProvider);

	// Read particle geometry and default to "SPHERICAL"
	_parGeomSurfToVol = std::vector<double>(_disc.nParType, SurfVolRatioSphere);
	if (paramProvider.exists("PAR_GEOM"))
	{
		std::vector<std::string> pg = paramProvider.getStringArray("PAR_GEOM");
		if ((pg.size() == 1) && (_disc.nParType > 1))
		{
			// Multiplex using first value
			pg.resize(_disc.nParType, pg[0]);
		}
		else if (pg.size() < _disc.nParType)
			throw InvalidParameterException("Field PAR_GEOM contains too few elements (" + std::to_string(_disc.nParType) + " required)");

		for (unsigned int i = 0; i < _disc.nParType; ++i)
		{
			if (pg[i] == "SPHERE")
				_parGeomSurfToVol[i] = SurfVolRatioSphere;
			else if (pg[i] == "CYLINDER")
				_parGeomSurfToVol[i] = SurfVolRatioCylinder;
			else if (pg[i] == "SLAB")
				_parGeomSurfToVol[i] = SurfVolRatioSlab;
			else
				throw InvalidParameterException("Unknown particle geometry type \"" + pg[i] + "\" at index " + std::to_string(i) + " of field PAR_GEOM");
		}
	}

	paramProvider.popScope();

	const bool transportSuccess = _convDispOp.configureModelDiscretization(paramProvider, helper, _disc.nComp, _disc.nCol);

	if (paramProvider.exists("FILM_DIFFUSION_DEP"))
	{
		const std::string paramDepName = paramProvider.getString("FILM_DIFFUSION_DEP");
		_filmDiffDep = helper.createParameterParameterDependence(paramDepName);
		if (!_filmDiffDep)
			throw InvalidParameterException("Unknown parameter dependence " + paramDepName + " in FILM_DIFFUSION_DEP");

		_filmDiffDep->configureModelDiscretization(paramProvider);
	}
	else
		_filmDiffDep = helper.createParameterParameterDependence("CONSTANT_ONE");

	// Allocate memory
	Indexer idxr(_disc);

	_jacInlet.resize(_disc.nComp);

	_jacP.resize(_disc.nParType);
	_jacPdisc.resize(_disc.nParType);
	for (unsigned int i = 0; i < _disc.nParType; ++i)
	{
		_jacPdisc[i].resize(_disc.nCol * (_disc.nComp + _disc.strideBound[i]), _disc.nComp + _disc.strideBound[i] - 1, _disc.nComp + _disc.strideBound[i] - 1);
		_jacP[i].resize(_disc.nCol * (_disc.nComp + _disc.strideBound[i]), _disc.nComp + _disc.strideBound[i] - 1, _disc.nComp + _disc.strideBound[i] - 1);
	}

	_jacPF.resize(_disc.nParType);
	_jacFP.resize(_disc.nParType);
	for (unsigned int i = 0; i < _disc.nParType; ++i)
	{
		_jacPF[i].resize(_disc.nComp * _disc.nCol);
		_jacFP[i].resize(_disc.nComp * _disc.nCol);
	}

	_jacCF.resize(_disc.nComp * _disc.nCol * _disc.nParType);
	_jacFC.resize(_disc.nComp * _disc.nCol * _disc.nParType);

	// Set whether analytic Jacobian is used
	useAnalyticJacobian(analyticJac);

	// ==== Construct and configure binding model
	clearBindingModels();
	_binding = std::vector<IBindingModel*>(_disc.nParType, nullptr);

	std::vector<std::string> bindModelNames = { "NONE" };
	if (paramProvider.exists("ADSORPTION_MODEL"))
		bindModelNames = paramProvider.getStringArray("ADSORPTION_MODEL");

	if (paramProvider.exists("ADSORPTION_MODEL_MULTIPLEX"))
		_singleBinding = (paramProvider.getInt("ADSORPTION_MODEL_MULTIPLEX") == 1);
	else
	{
		// Infer multiplex mode
		_singleBinding = (bindModelNames.size() == 1);
	}

	if (!_singleBinding && (bindModelNames.size() < _disc.nParType))
		throw InvalidParameterException("Field ADSORPTION_MODEL contains too few elements (" + std::to_string(_disc.nParType) + " required)");
	else if (_singleBinding && (bindModelNames.size() != 1))
		throw InvalidParameterException("Field ADSORPTION_MODEL requires (only) 1 element");

	bool bindingConfSuccess = true;
	for (unsigned int i = 0; i < _disc.nParType; ++i)
	{
		if (_singleBinding && (i > 0))
		{
			// Reuse first binding model
			_binding[i] = _binding[0];
		}
		else
		{
			_binding[i] = helper.createBindingModel(bindModelNames[i]);
			if (!_binding[i])
				throw InvalidParameterException("Unknown binding model " + bindModelNames[i]);

			MultiplexedScopeSelector scopeGuard(paramProvider, "adsorption", _singleBinding, i, _disc.nParType == 1, _binding[i]->usesParamProviderInDiscretizationConfig());
			bindingConfSuccess = _binding[i]->configureModelDiscretization(paramProvider, _disc.nComp, _disc.nBound + i * _disc.nComp, _disc.boundOffset + i * _disc.nComp) && bindingConfSuccess;
		}
	}

	// ==== Construct and configure dynamic reaction model
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

		reactionConfSuccess = _dynReactionBulk->configureModelDiscretization(paramProvider, _disc.nComp, nullptr, nullptr);

		if (_dynReactionBulk->usesParamProviderInDiscretizationConfig())
			paramProvider.popScope();
	}

	clearDynamicReactionModels();
	_dynReaction = std::vector<IDynamicReactionModel*>(_disc.nParType, nullptr);

	if (paramProvider.exists("REACTION_MODEL_PARTICLES"))
	{
		const std::vector<std::string> dynReactModelNames = paramProvider.getStringArray("REACTION_MODEL_PARTICLES");

		if (paramProvider.exists("REACTION_MODEL_PARTICLES_MULTIPLEX"))
			_singleDynReaction = (paramProvider.getInt("REACTION_MODEL_PARTICLES_MULTIPLEX") == 1);
		else
		{
			// Infer multiplex mode
			_singleDynReaction = (dynReactModelNames.size() == 1);
		}

		if (!_singleDynReaction && (dynReactModelNames.size() < _disc.nParType))
			throw InvalidParameterException("Field REACTION_MODEL_PARTICLES contains too few elements (" + std::to_string(_disc.nParType) + " required)");
		else if (_singleDynReaction && (dynReactModelNames.size() != 1))
			throw InvalidParameterException("Field REACTION_MODEL_PARTICLES requires (only) 1 element");

		for (unsigned int i = 0; i < _disc.nParType; ++i)
		{
			if (_singleDynReaction && (i > 0))
			{
				// Reuse first binding model
				_dynReaction[i] = _dynReaction[0];
			}
			else
			{
				_dynReaction[i] = helper.createDynamicReactionModel(dynReactModelNames[i]);
				if (!_dynReaction[i])
					throw InvalidParameterException("Unknown dynamic reaction model " + dynReactModelNames[i]);

				MultiplexedScopeSelector scopeGuard(paramProvider, "reaction_particle", _singleDynReaction, i, _disc.nParType == 1, _dynReaction[i]->usesParamProviderInDiscretizationConfig());
				reactionConfSuccess = _dynReaction[i]->configureModelDiscretization(paramProvider, _disc.nComp, _disc.nBound + i * _disc.nComp, _disc.boundOffset + i * _disc.nComp) && reactionConfSuccess;
			}
		}
	}

	// Setup the memory for tempState based on state vector
	_tempState = new double[numDofs()];

	return transportSuccess && bindingConfSuccess && reactionConfSuccess;
}

template <typename ConvDispOperator>
bool LumpedRateModelWithPores<ConvDispOperator>::configure(IParameterProvider& paramProvider)
{
	_parameters.clear();

	const bool transportSuccess = _convDispOp.configure(_unitOpIdx, paramProvider, _parameters);

	if (_filmDiffDep)
	{
		if (!_filmDiffDep->configure(paramProvider, _unitOpIdx, ParTypeIndep, BoundStateIndep, "FILM_DIFFUSION_DEP"))
			throw InvalidParameterException("Failed to configure film diffusion parameter dependency (FILM_DIFFUSION_DEP)");
	}

	// Read geometry parameters
	_colPorosity = paramProvider.getDouble("COL_POROSITY");
	_singleParRadius = readAndRegisterMultiplexTypeParam(paramProvider, _parameters, _parRadius, "PAR_RADIUS", _disc.nParType, _unitOpIdx);
	_singleParPorosity = readAndRegisterMultiplexTypeParam(paramProvider, _parameters, _parPorosity, "PAR_POROSITY", _disc.nParType, _unitOpIdx);

	// Read vectorial parameters (which may also be section dependent; transport)
	_filmDiffusionMode = readAndRegisterMultiplexCompTypeSecParam(paramProvider, _parameters, _filmDiffusion, "FILM_DIFFUSION", _disc.nParType, _disc.nComp, _unitOpIdx);

	if (paramProvider.exists("PORE_ACCESSIBILITY"))
		_poreAccessFactorMode = readAndRegisterMultiplexCompTypeSecParam(paramProvider, _parameters, _poreAccessFactor, "PORE_ACCESSIBILITY", _disc.nParType, _disc.nComp, _unitOpIdx);
	else
	{
		_poreAccessFactorMode = MultiplexMode::ComponentType;
		_poreAccessFactor = std::vector<cadet::active>(_disc.nComp * _disc.nParType, 1.0);
	}

	// Check whether PAR_TYPE_VOLFRAC is required or not
	if ((_disc.nParType > 1) && !paramProvider.exists("PAR_TYPE_VOLFRAC"))
		throw InvalidParameterException("The required parameter \"PAR_TYPE_VOLFRAC\" was not found");

	// Let PAR_TYPE_VOLFRAC default to 1.0 for backwards compatibility
	if (paramProvider.exists("PAR_TYPE_VOLFRAC"))
	{
		readScalarParameterOrArray(_parTypeVolFrac, paramProvider, "PAR_TYPE_VOLFRAC", 1);
		if (_parTypeVolFrac.size() == _disc.nParType)
		{
			_axiallyConstantParTypeVolFrac = true;

			// Expand to all axial cells
			_parTypeVolFrac.resize(_disc.nCol * _disc.nParType, 1.0);
			for (unsigned int i = 1; i < _disc.nCol; ++i)
				std::copy(_parTypeVolFrac.begin(), _parTypeVolFrac.begin() + _disc.nParType, _parTypeVolFrac.begin() + _disc.nParType * i);
		}
		else
			_axiallyConstantParTypeVolFrac = false;
	}
	else
	{
		_parTypeVolFrac.resize(_disc.nCol, 1.0);
		_axiallyConstantParTypeVolFrac = false;
	}

	// Check whether all sizes are matched
	if (_disc.nParType != _parRadius.size())
		throw InvalidParameterException("Number of elements in field PAR_RADIUS does not match number of particle types");
	if (_disc.nParType * _disc.nCol != _parTypeVolFrac.size())
		throw InvalidParameterException("Number of elements in field PAR_TYPE_VOLFRAC does not match number of particle types");
	if (_disc.nParType != _parPorosity.size())
		throw InvalidParameterException("Number of elements in field PAR_POROSITY does not match number of particle types");

	if ((_filmDiffusion.size() < _disc.nComp * _disc.nParType) || (_filmDiffusion.size() % (_disc.nComp * _disc.nParType) != 0))
		throw InvalidParameterException("Number of elements in field FILM_DIFFUSION is not a positive multiple of NCOMP * NPARTYPE (" + std::to_string(_disc.nComp * _disc.nParType) + ")");
	if (_disc.nComp * _disc.nParType != _poreAccessFactor.size())
		throw InvalidParameterException("Number of elements in field PORE_ACCESSIBILITY differs from NCOMP * NPARTYPE (" + std::to_string(_disc.nComp * _disc.nParType) + ")");

	// Check that particle volume fractions sum to 1.0
	for (unsigned int i = 0; i < _disc.nCol; ++i)
	{
		const double volFracSum = std::accumulate(_parTypeVolFrac.begin() + i * _disc.nParType, _parTypeVolFrac.begin() + (i+1) * _disc.nParType, 0.0,
			[](double a, const active& b) -> double { return a + static_cast<double>(b); });
		if (std::abs(1.0 - volFracSum) > 1e-10)
			throw InvalidParameterException("Sum of field PAR_TYPE_VOLFRAC differs from 1.0 (is " + std::to_string(volFracSum) + ") in axial cell " + std::to_string(i));
	}

	// Add parameters to map
	_parameters[makeParamId(hashString("COL_POROSITY"), _unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_colPorosity;

	if (_axiallyConstantParTypeVolFrac)
	{
		// Register only the first nParType items
		for (unsigned int i = 0; i < _disc.nParType; ++i)
			_parameters[makeParamId(hashString("PAR_TYPE_VOLFRAC"), _unitOpIdx, CompIndep, i, BoundStateIndep, ReactionIndep, SectionIndep)] = &_parTypeVolFrac[i];
	}
	else
		registerParam2DArray(_parameters, _parTypeVolFrac, [=](bool multi, unsigned cell, unsigned int type) { return makeParamId(hashString("PAR_TYPE_VOLFRAC"), _unitOpIdx, CompIndep, type, BoundStateIndep, ReactionIndep, cell); }, _disc.nParType);

	// Register initial conditions parameters
	registerParam1DArray(_parameters, _initC, [=](bool multi, unsigned int comp) { return makeParamId(hashString("INIT_C"), _unitOpIdx, comp, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep); });

	if (_singleBinding)
	{
		for (unsigned int c = 0; c < _disc.nComp; ++c)
			_parameters[makeParamId(hashString("INIT_CP"), _unitOpIdx, c, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_initCp[c];
	}
	else
		registerParam2DArray(_parameters, _initCp, [=](bool multi, unsigned int type, unsigned int comp) { return makeParamId(hashString("INIT_CP"), _unitOpIdx, comp, type, BoundStateIndep, ReactionIndep, SectionIndep); }, _disc.nComp);


	if (!_binding.empty())
	{
		const unsigned int maxBoundStates = *std::max_element(_disc.strideBound, _disc.strideBound + _disc.nParType);
		std::vector<ParameterId> initParams(maxBoundStates);

		if (_singleBinding)
		{
			_binding[0]->fillBoundPhaseInitialParameters(initParams.data(), _unitOpIdx, ParTypeIndep);

			active* const iq = _initQ.data() + _disc.nBoundBeforeType[0];
			for (unsigned int i = 0; i < _disc.strideBound[0]; ++i)
				_parameters[initParams[i]] = iq + i;
		}
		else
		{
			for (unsigned int type = 0; type < _disc.nParType; ++type)
			{
				_binding[type]->fillBoundPhaseInitialParameters(initParams.data(), _unitOpIdx, type);

				active* const iq = _initQ.data() + _disc.nBoundBeforeType[type];
				for (unsigned int i = 0; i < _disc.strideBound[type]; ++i)
					_parameters[initParams[i]] = iq + i;
			}
		}
	}

	bool bindingConfSuccess = true;
	if (!_binding.empty())
	{
		if (_singleBinding)
		{
			if (_binding[0] && _binding[0]->requiresConfiguration())
			{
				MultiplexedScopeSelector scopeGuard(paramProvider, "adsorption", true);
				bindingConfSuccess = _binding[0]->configure(paramProvider, _unitOpIdx, ParTypeIndep);
			}
		}
		else
		{
			for (unsigned int type = 0; type < _disc.nParType; ++type)
			{
	 			if (!_binding[type] || !_binding[type]->requiresConfiguration())
	 				continue;

	 			// Check whether required = true and no isActive() check should be performed
				MultiplexedScopeSelector scopeGuard(paramProvider, "adsorption", type, _disc.nParType == 1, false);
				if (!scopeGuard.isActive())
					continue;

				bindingConfSuccess = _binding[type]->configure(paramProvider, _unitOpIdx, type) && bindingConfSuccess;
			}
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

	if (_singleDynReaction)
	{
		if (_dynReaction[0] && _dynReaction[0]->requiresConfiguration())
		{
			MultiplexedScopeSelector scopeGuard(paramProvider, "reaction_particle", true);
			dynReactionConfSuccess = _dynReaction[0]->configure(paramProvider, _unitOpIdx, ParTypeIndep) && dynReactionConfSuccess;
		}
	}
	else
	{
		for (unsigned int type = 0; type < _disc.nParType; ++type)
		{
 			if (!_dynReaction[type] || !_dynReaction[type]->requiresConfiguration())
 				continue;

			MultiplexedScopeSelector scopeGuard(paramProvider, "reaction_particle", type, _disc.nParType == 1, true);
			dynReactionConfSuccess = _dynReaction[type]->configure(paramProvider, _unitOpIdx, type) && dynReactionConfSuccess;
		}
	}

	return transportSuccess && bindingConfSuccess && dynReactionConfSuccess;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithPores<ConvDispOperator>::threadLocalMemorySize() const CADET_NOEXCEPT
{
	LinearMemorySizer lms;

	// Memory for residualImpl()
	for (unsigned int i = 0; i < _disc.nParType; ++i)
	{
		if (_binding[i] && _binding[i]->requiresWorkspace())
			lms.fitBlock(_binding[i]->workspaceSize(_disc.nComp, _disc.strideBound[i], _disc.nBound + i * _disc.nComp));

		if (_dynReaction[i] && _dynReaction[i]->requiresWorkspace())
			lms.fitBlock(_dynReaction[i]->workspaceSize(_disc.nComp, _disc.strideBound[i], _disc.nBound + i * _disc.nComp));
	}

	if (_dynReactionBulk && _dynReactionBulk->requiresWorkspace())
		lms.fitBlock(_dynReactionBulk->workspaceSize(_disc.nComp, 0, nullptr));

	const unsigned int maxStrideBound = *std::max_element(_disc.strideBound, _disc.strideBound + _disc.nParType);
	lms.add<active>(_disc.nComp + maxStrideBound);
	lms.add<double>((maxStrideBound + _disc.nComp) * (maxStrideBound + _disc.nComp));

	lms.commit();
	const std::size_t resImplSize = lms.bufferSize();

	// Memory for consistentInitialState()
	lms.add<double>(_nonlinearSolver->workspaceSize(_disc.nComp + maxStrideBound) * sizeof(double));
	lms.add<double>(_disc.nComp + maxStrideBound);
	lms.add<double>(_disc.nComp + maxStrideBound);
	lms.add<double>(_disc.nComp + maxStrideBound);
	lms.add<double>((_disc.nComp + maxStrideBound) * (_disc.nComp + maxStrideBound));
	lms.add<double>(_disc.nComp);

	lms.addBlock(resImplSize);
	lms.commit();

	// Memory for consistentInitialSensitivity
	lms.add<double>(_disc.nComp + maxStrideBound);
	lms.add<double>(maxStrideBound);
	lms.commit();

	return lms.bufferSize();
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithPores<ConvDispOperator>::numAdDirsForJacobian() const CADET_NOEXCEPT
{
	// We need as many directions as the highest bandwidth of the diagonal blocks:
	// The bandwidth of the column block depends on the size of the WENO stencil, whereas
	// the bandwidth of the particle blocks are given by the number of components and bound states.

	// Get maximum stride of particle type blocks
	int maxStride = 0;
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		maxStride = std::max(maxStride, _jacP[type].stride());
	}

	return std::max(_convDispOp.requiredADdirs(), maxStride);
}

template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::useAnalyticJacobian(const bool analyticJac)
{
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	_analyticJac = analyticJac;
	if (!_analyticJac)
		_jacobianAdDirs = numAdDirsForJacobian();
	else
		_jacobianAdDirs = 0;
#else
	// If CADET_CHECK_ANALYTIC_JACOBIAN is active, we always enable AD for comparison and use it in simulation
	_analyticJac = false;
	_jacobianAdDirs = numAdDirsForJacobian();
#endif
}

template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac)
{
	// Setup flux Jacobian blocks at the beginning of the simulation or in case of
	// section dependent film or particle diffusion coefficients
	if ((secIdx == 0) || isSectionDependent(_filmDiffusionMode))
		assembleOffdiagJac(t, secIdx);

	Indexer idxr(_disc);

	// AxialConvectionDispersionOperator tells us whether flow direction has changed
	if (!_convDispOp.notifyDiscontinuousSectionTransition(t, secIdx, adJac))
		return;

	// Setup the matrix connecting inlet DOFs to first column cells
	_jacInlet.clear();
	const double v = _convDispOp.inletJacobianFactor();

	if (_convDispOp.forwardFlow())
	{
		// Forwards flow

		// Place entries for inlet DOF to first column cell conversion
		for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			_jacInlet.addElement(comp * idxr.strideColComp(), comp, -v);
	}
	else
	{
		// Backwards flow

		// Place entries for inlet DOF to last column cell conversion
		const unsigned int offset = (_disc.nCol - 1) * idxr.strideColCell();
		for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			_jacInlet.addElement(offset + comp * idxr.strideColComp(), comp, v);
	}
}

template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::setFlowRates(active const* in, active const* out) CADET_NOEXCEPT
{
	_convDispOp.setFlowRates(in[0], out[0], _colPorosity);
}

template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::reportSolution(ISolutionRecorder& recorder, double const* const solution) const
{
	Exporter expr(_disc, *this, solution);
	recorder.beginUnitOperation(_unitOpIdx, *this, expr);
	recorder.endUnitOperation();
}

template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::reportSolutionStructure(ISolutionRecorder& recorder) const
{
	Exporter expr(_disc, *this, nullptr);
	recorder.unitOperationStructure(_unitOpIdx, *this, expr);
}


template <typename ConvDispOperator>
unsigned int LumpedRateModelWithPores<ConvDispOperator>::requiredADdirs() const CADET_NOEXCEPT
{
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	return _jacobianAdDirs;
#else
	// If CADET_CHECK_ANALYTIC_JACOBIAN is active, we always need the AD directions for the Jacobian
	return numAdDirsForJacobian();
#endif
}

template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::prepareADvectors(const AdJacobianParams& adJac) const
{
	// Early out if AD is disabled
	if (!adJac.adY)
		return;

	Indexer idxr(_disc);

	// Column block
	_convDispOp.prepareADvectors(adJac);

	// Particle block
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		const unsigned int lowerParBandwidth = _jacP[type].lowerBandwidth();
		const unsigned int upperParBandwidth = _jacP[type].upperBandwidth();

		ad::prepareAdVectorSeedsForBandMatrix(adJac.adY + idxr.offsetCp(ParticleTypeIndex{type}), adJac.adDirOffset, idxr.strideParBlock(type) * _disc.nCol, lowerParBandwidth, upperParBandwidth, lowerParBandwidth);
	}
}

/**
 * @brief Extracts the system Jacobian from band compressed AD seed vectors
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset)
{
	Indexer idxr(_disc);

	// Column
	_convDispOp.extractJacobianFromAD(adRes, adDirOffset);

	// Particles
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		linalg::BandMatrix& jacMat = _jacP[type];
		ad::extractBandedJacobianFromAd(adRes + idxr.offsetCp(ParticleTypeIndex{type}), adDirOffset, jacMat.lowerBandwidth(), jacMat);
	}
}

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN

/**
 * @brief Compares the analytical Jacobian with a Jacobian derived by AD
 * @details The analytical Jacobian is assumed to be stored in the corresponding band matrices.
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const
{
	Indexer idxr(_disc);

	LOG(Debug) << "AD dir offset: " << adDirOffset << " DiagDirCol: " << _convDispOp.jacobian().lowerBandwidth();

	// Column
	const double maxDiffCol = _convDispOp.checkAnalyticJacobianAgainstAd(adRes, adDirOffset);

	// Particles
	double maxDiffPar = 0.0;
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		const linalg::BandMatrix& jacMat = _jacP[type];
		const double localDiff = ad::compareBandedJacobianWithAd(adRes + idxr.offsetCp(ParticleTypeIndex{type}), adDirOffset, jacMat.lowerBandwidth(), jacMat);
		LOG(Debug) << "-> Par type " << type << " diff: " << localDiff;
		maxDiffPar = std::max(maxDiffPar, localDiff);
	}
}

#endif

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidual);

	// Evaluate residual do not compute Jacobian or parameter sensitivities
	return residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem);
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidual);

	// Evaluate residual, use AD for Jacobian if required but do not evaluate parameter derivatives
	return residual(simTime, simState, res, adJac, threadLocalMem, true, false);
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res,
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
int LumpedRateModelWithPores<ConvDispOperator>::residualImpl(double t, unsigned int secIdx, StateType const* const y, double const* const yDot, ResidualType* const res, util::ThreadLocalStorage& threadLocalMem)
{
	// Reset Jacobian
	if (wantJac)
	{
		for (unsigned int type = 0; type < _disc.nParType; ++type)
			_jacP[type].setAll(0.0);
	}

	BENCH_START(_timerResidualPar);

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(_disc.nCol * _disc.nParType + 1), [&](std::size_t pblk)
#else
	for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nParType + 1; ++pblk)
#endif
	{
		if (cadet_unlikely(pblk == 0))
			residualBulk<StateType, ResidualType, ParamType, wantJac>(t, secIdx, y, yDot, res, threadLocalMem);
		else
		{
			const unsigned int type = (pblk - 1) / _disc.nCol;
			const unsigned int par = (pblk - 1) % _disc.nCol;
			residualParticle<StateType, ResidualType, ParamType, wantJac>(t, type, par, secIdx, y, yDot, res, threadLocalMem);
		}
	} CADET_PARFOR_END;

	BENCH_STOP(_timerResidualPar);

	residualFlux<StateType, ResidualType, ParamType>(t, secIdx, y, yDot, res);

	// Handle inlet DOFs, which are simply copied to res
	for (unsigned int i = 0; i < _disc.nComp; ++i)
	{
		res[i] = y[i];
	}

	return 0;
}

template <typename ConvDispOperator>
template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int LumpedRateModelWithPores<ConvDispOperator>::residualBulk(double t, unsigned int secIdx, StateType const* yBase, double const* yDotBase, ResidualType* resBase, util::ThreadLocalStorage& threadLocalMem)
{
	_convDispOp.residual(*this, t, secIdx, yBase, yDotBase, resBase, wantJac, typename ParamSens<ParamType>::enabled());
	if (!_dynReactionBulk || (_dynReactionBulk->numReactionsLiquid() == 0))
		return 0;

	// Get offsets
	Indexer idxr(_disc);
	StateType const* y = yBase + idxr.offsetC();
	ResidualType* res = resBase + idxr.offsetC();
	LinearBufferAllocator tlmAlloc = threadLocalMem.get();

	for (unsigned int col = 0; col < _disc.nCol; ++col, y += idxr.strideColCell(), res += idxr.strideColCell())
	{
		const ColumnPosition colPos{(0.5 + static_cast<double>(col)) / static_cast<double>(_disc.nCol), 0.0, 0.0};
		_dynReactionBulk->residualLiquidAdd(t, secIdx, colPos, y, res, -1.0, tlmAlloc);

		if (wantJac)
		{
			// static_cast should be sufficient here, but this statement is also analyzed when wantJac = false
			_dynReactionBulk->analyticJacobianLiquidAdd(t, secIdx, colPos, reinterpret_cast<double const*>(y), -1.0, _convDispOp.jacobian().row(col * idxr.strideColCell()), tlmAlloc);
		}
	}

	return 0;
}

template <typename ConvDispOperator>
template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int LumpedRateModelWithPores<ConvDispOperator>::residualParticle(double t, unsigned int parType, unsigned int colCell, unsigned int secIdx, StateType const* yBase, double const* yDotBase, ResidualType* resBase, util::ThreadLocalStorage& threadLocalMem)
{
	Indexer idxr(_disc);

	// Go to the particle block of the given type and column cell
	StateType const* y = yBase + idxr.offsetCp(ParticleTypeIndex{parType}, ParticleIndex{colCell});
	double const* yDot = yDotBase + idxr.offsetCp(ParticleTypeIndex{parType}, ParticleIndex{colCell});
	ResidualType* res = resBase + idxr.offsetCp(ParticleTypeIndex{parType}, ParticleIndex{colCell});

	// Prepare parameters
	const ParamType radius = static_cast<ParamType>(_parRadius[parType]);

	// Midpoint of current column cell (z coordinate) - needed in externally dependent adsorption kinetic
	const double z = _convDispOp.relativeCoordinate(colCell);

	const parts::cell::CellParameters cellResParams
		{
			_disc.nComp,
			_disc.nBound + _disc.nComp * parType,
			_disc.boundOffset + _disc.nComp * parType,
			_disc.strideBound[parType],
			_binding[parType]->reactionQuasiStationarity(),
			_parPorosity[parType],
			_poreAccessFactor.data() + _disc.nComp * parType,
			_binding[parType],
			(_dynReaction[parType] && (_dynReaction[parType]->numReactionsCombined() > 0)) ? _dynReaction[parType] : nullptr
		};

	// Handle time derivatives, binding, dynamic reactions
	parts::cell::residualKernel<StateType, ResidualType, ParamType, parts::cell::CellParameters, linalg::BandMatrix::RowIterator, wantJac, true>(
		t, secIdx, ColumnPosition{z, 0.0, static_cast<double>(radius) * 0.5}, y, yDotBase ? yDot : nullptr, res,
		_jacP[parType].row(colCell * idxr.strideParBlock(parType)), cellResParams, threadLocalMem.get()
	);

	return 0;
}

template <typename ConvDispOperator>
template <typename StateType, typename ResidualType, typename ParamType>
int LumpedRateModelWithPores<ConvDispOperator>::residualFlux(double t, unsigned int secIdx, StateType const* yBase, double const* yDotBase, ResidualType* resBase)
{
	Indexer idxr(_disc);

	const ParamType invBetaC = 1.0 / static_cast<ParamType>(_colPorosity) - 1.0;

	// Get offsets
	ResidualType* const resCol = resBase + idxr.offsetC();
	ResidualType* const resFlux = resBase + idxr.offsetJf();

	StateType const* const yCol = yBase + idxr.offsetC();
	StateType const* const yFlux = yBase + idxr.offsetJf();

	// J_f block (identity matrix), adds flux state to flux equation
	for (unsigned int i = 0; i < _disc.nComp * _disc.nCol * _disc.nParType; ++i)
		resFlux[i] = yFlux[i];

	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		ResidualType* const resParType = resBase + idxr.offsetCp(ParticleTypeIndex{type});
		ResidualType* const resFluxType = resBase + idxr.offsetJf(ParticleTypeIndex{type});

		StateType const* const yParType = yBase + idxr.offsetCp(ParticleTypeIndex{type});
		StateType const* const yFluxType = yBase + idxr.offsetJf(ParticleTypeIndex{type});

		const ParamType epsP = static_cast<ParamType>(_parPorosity[type]);
		const ParamType radius = static_cast<ParamType>(_parRadius[type]);
		active const* const filmDiff = getSectionDependentSlice(_filmDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;
		active const* const poreAccFactor = _poreAccessFactor.data() + type * _disc.nComp;

		const ParamType jacCF_val = invBetaC * _parGeomSurfToVol[type] / radius;
		const ParamType jacPF_val = -_parGeomSurfToVol[type] / (epsP * radius);

		// J_{0,f} block, adds flux to column void / bulk volume equations
		for (unsigned int i = 0; i < _disc.nCol * _disc.nComp; ++i)
		{
			const unsigned int colCell = i / _disc.nComp;
			const unsigned int comp = i % _disc.nComp;

			const double relPos = _convDispOp.relativeCoordinate(colCell);
			const active curVelocity = _convDispOp.currentVelocity(relPos);
			const active modifier = _filmDiffDep->getValue(*this, ColumnPosition{relPos, 0.0, 0.0}, comp, ParTypeIndep, BoundStateIndep, curVelocity);

			resCol[i] += jacCF_val * static_cast<ParamType>(filmDiff[comp]) * static_cast<ParamType>(modifier) * static_cast<ParamType>(_parTypeVolFrac[type + _disc.nParType * colCell]) * yFluxType[i];
		}

		// J_{f,0} block, adds bulk volume state c_i to flux equation
		for (unsigned int bnd = 0; bnd < _disc.nCol; ++bnd)
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				const unsigned int eq = bnd * idxr.strideColCell() + comp * idxr.strideColComp();
				resFluxType[eq] -= yCol[eq];
			}
		}

		// J_{p,f} block, adds flux to particle / bead volume equations
		for (unsigned int pblk = 0; pblk < _disc.nCol; ++pblk)
		{
			const double relPos = _convDispOp.relativeCoordinate(pblk);
			const active curVelocity = _convDispOp.currentVelocity(relPos);

			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				const active modifier = _filmDiffDep->getValue(*this, ColumnPosition{relPos, 0.0, 0.0}, comp, ParTypeIndep, BoundStateIndep, curVelocity);

				const unsigned int eq = pblk * idxr.strideColCell() + comp * idxr.strideColComp();
				resParType[pblk * idxr.strideParBlock(type) + comp] += jacPF_val / static_cast<ParamType>(poreAccFactor[comp]) * static_cast<ParamType>(filmDiff[comp]) * static_cast<ParamType>(modifier) * yFluxType[eq];
			}
		}

		// J_{f,p} block, adds outer bead shell state c_{p,i} to flux equation
		for (unsigned int pblk = 0; pblk < _disc.nCol; ++pblk)
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				const unsigned int eq = pblk * idxr.strideColCell() + comp * idxr.strideColComp();
				resFluxType[eq] += yParType[comp + pblk * idxr.strideParBlock(type)];
			}
		}
	}

	return 0;
}

/**
 * @brief Assembles off diagonal Jacobian blocks
 * @details Assembles the fixed blocks @f$ J_{0,f}, \dots, J_{N_p,f} @f$ and @f$ J_{f,0}, \dots, J_{f, N_p}. @f$
 *          The blocks are fixed for each section.
 * @param [in] t Current time
 * @param [in] secIdx Index of the current section
 */
template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::assembleOffdiagJac(double t, unsigned int secIdx)
{
	// Clear matrices for new assembly
	_jacCF.clear();
	_jacFC.clear();
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		_jacPF[type].clear();
		_jacFP[type].clear();
	}

	Indexer idxr(_disc);

	const double invBetaC = 1.0 / static_cast<double>(_colPorosity) - 1.0;

	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		const unsigned int typeOffset = type * _disc.nCol * _disc.nComp;

		const double epsP = static_cast<double>(_parPorosity[type]);
		const double radius = static_cast<double>(_parRadius[type]);
		const double jacCF_val = invBetaC * _parGeomSurfToVol[type] / radius;
		const double jacPF_val = -_parGeomSurfToVol[type] / (radius * epsP);

		active const* const filmDiff = getSectionDependentSlice(_filmDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;
		active const* const poreAccFactor = _poreAccessFactor.data() + type * _disc.nComp;

		// Note that the J_f block, which is the identity matrix, is treated in the linear solver

		// J_{0,f} block, adds flux to column void / bulk volume equations
		for (unsigned int eq = 0; eq < _disc.nCol * _disc.nComp; ++eq)
		{
			const unsigned int colCell = eq / _disc.nComp;
			const unsigned int comp = eq % _disc.nComp;

			const double relPos = _convDispOp.relativeCoordinate(colCell);
			const double curVelocity = static_cast<double>(_convDispOp.currentVelocity(relPos));
			const double modifier = _filmDiffDep->getValue(*this, ColumnPosition{relPos, 0.0, 0.0}, comp, ParTypeIndep, BoundStateIndep, curVelocity);

			// Main diagonal corresponds to j_{f,i} (flux) state variable
			_jacCF.addElement(eq, eq + typeOffset, jacCF_val * static_cast<double>(filmDiff[comp]) * modifier * static_cast<double>(_parTypeVolFrac[type + _disc.nParType * colCell]));
		}

		// J_{f,0} block, adds bulk volume state c_i to flux equation
		for (unsigned int eq = 0; eq < _disc.nCol * _disc.nComp; ++eq)
		{
			_jacFC.addElement(eq + typeOffset, eq, -1.0);
		}

		// J_{p,f} block, implements bead boundary condition in outer bead shell equation
		for (unsigned int pblk = 0; pblk < _disc.nCol; ++pblk)
		{
			const double relPos = _convDispOp.relativeCoordinate(pblk);
			const double curVelocity = static_cast<double>(_convDispOp.currentVelocity(relPos));

			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				const unsigned int eq = typeOffset + pblk * idxr.strideColCell() + comp * idxr.strideColComp();
				const unsigned int col = pblk * idxr.strideParBlock(type) + comp;

				const double modifier = _filmDiffDep->getValue(*this, ColumnPosition{relPos, 0.0, 0.0}, comp, ParTypeIndep, BoundStateIndep, curVelocity);
				_jacPF[type].addElement(col, eq, jacPF_val / static_cast<double>(poreAccFactor[comp]) * static_cast<double>(filmDiff[comp]) * modifier);
			}
		}

		// J_{f,p} block, adds outer bead shell state c_{p,i} to flux equation
		for (unsigned int pblk = 0; pblk < _disc.nCol; ++pblk)
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				const unsigned int eq = typeOffset + pblk * idxr.strideColCell() + comp * idxr.strideColComp();
				const unsigned int col = pblk * idxr.strideParBlock(type) + comp;
				_jacFP[type].addElement(eq, col, 1.0);
			}
		}
	}
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState,
	const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidualSens);

	// Evaluate residual for all parameters using AD in vector mode and at the same time update the
	// Jacobian (in one AD run, if analytic Jacobians are disabled)
	return residual(simTime, simState, nullptr, adJac, threadLocalMem, true, true);
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidualSens);

	// Evaluate residual for all parameters using AD in vector mode
	return residualImpl<double, active, active, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, adRes, threadLocalMem);
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState,
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
void LumpedRateModelWithPores<ConvDispOperator>::multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret)
{
	Indexer idxr(_disc);

	// Handle identity matrix of inlet DOFs
	for (unsigned int i = 0; i < _disc.nComp; ++i)
	{
		ret[i] = alpha * yS[i] + beta * ret[i];
	}

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(_disc.nParType + 1), [&](std::size_t idx)
#else
	for (unsigned int idx = 0; idx < _disc.nParType + 1; ++idx)
#endif
	{
		if (cadet_unlikely(idx == 0))
		{
			// Interstitial block
			_convDispOp.jacobian().multiplyVector(yS + idxr.offsetC(), alpha, beta, ret + idxr.offsetC());
			_jacCF.multiplyVector(yS + idxr.offsetJf(), alpha, 1.0, ret + idxr.offsetC());
		}
		else
		{
			// Particle blocks
			const unsigned int type = idx - 1;
			const int localOffset = idxr.offsetCp(ParticleTypeIndex{type});
			_jacP[type].multiplyVector(yS + localOffset, alpha, beta, ret + localOffset);
			_jacPF[type].multiplyVector(yS + idxr.offsetJf(), alpha, 1.0, ret + localOffset);
		}
	} CADET_PARFOR_END;

	// Handle flux equation

	// Set fluxes(ret) = fluxes(yS)
	// This applies the identity matrix in the bottom right corner of the Jaocbian (flux equation)
	for (unsigned int i = idxr.offsetJf(); i < numDofs(); ++i)
		ret[i] = alpha * yS[i] + beta * ret[i];

	double* const retJf = ret + idxr.offsetJf();
	_jacFC.multiplyVector(yS + idxr.offsetC(), alpha, 1.0, retJf);
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		_jacFP[type].multiplyVector(yS + idxr.offsetCp(ParticleTypeIndex{type}), alpha, 1.0, retJf);
	}

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
void LumpedRateModelWithPores<ConvDispOperator>::multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret)
{
	Indexer idxr(_disc);

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(_disc.nCol * _disc.nParType + 1), [&](std::size_t idx)
#else
	for (unsigned int idx = 0; idx < _disc.nCol * _disc.nParType + 1; ++idx)
#endif
	{
		if (cadet_unlikely(idx == 0))
		{
			_convDispOp.multiplyWithDerivativeJacobian(simTime, sDot, ret);
		}
		else
		{
			const unsigned int idxParLoop = idx - 1;
			const unsigned int pblk = idxParLoop % _disc.nCol;
			const unsigned int type = idxParLoop / _disc.nCol;

			// Particle
			double const* const localSdot = sDot + idxr.offsetCp(ParticleTypeIndex{type}, ParticleIndex{pblk});
			double* const localRet = ret + idxr.offsetCp(ParticleTypeIndex{type}, ParticleIndex{pblk});

			unsigned int const* const nBound = _disc.nBound + type * _disc.nComp;
			unsigned int const* const boundOffset = _disc.boundOffset + type * _disc.nComp;

			// Mobile phase
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				// Add derivative with respect to dc_p / dt to Jacobian
				localRet[comp] = localSdot[comp];

				const double invBetaP = (1.0 - static_cast<double>(_parPorosity[type])) / (static_cast<double>(_poreAccessFactor[type * _disc.nComp + comp]) * static_cast<double>(_parPorosity[type]));

				// Add derivative with respect to dq / dt to Jacobian (normal equations)
				for (unsigned int i = 0; i < nBound[comp]; ++i)
				{
					// Index explanation:
					//   nComp -> skip mobile phase
					//   + boundOffset[comp] skip bound states of all previous components
					//   + i go to current bound state
					localRet[comp] += invBetaP * localSdot[_disc.nComp + boundOffset[comp] + i];
				}
			}

			// Solid phase
			double const* const solidSdot = localSdot + _disc.nComp;
			double* const solidRet = localRet + _disc.nComp;
			int const* const qsReaction = _binding[type]->reactionQuasiStationarity();

			for (unsigned int bnd = 0; bnd < _disc.strideBound[type]; ++bnd)
			{
				// Add derivative with respect to dynamic states to Jacobian
				if (qsReaction[bnd])
					solidRet[bnd] = 0.0;
				else
					solidRet[bnd] = solidSdot[bnd];
			}
		}
	} CADET_PARFOR_END;

	// Handle fluxes (all algebraic)
	double* const dFdyDot = ret + idxr.offsetJf();
	std::fill(dFdyDot, dFdyDot + _disc.nCol * _disc.nComp * _disc.nParType, 0.0);

	// Handle inlet DOFs (all algebraic)
	std::fill_n(ret, _disc.nComp, 0.0);
}

template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::setExternalFunctions(IExternalFunction** extFuns, unsigned int size)
{
	for (IBindingModel* bm : _binding)
	{
		if (bm)
			bm->setExternalFunctions(extFuns, size);
	}
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithPores<ConvDispOperator>::localOutletComponentIndex(unsigned int port) const CADET_NOEXCEPT
{
	// Inlets are duplicated so need to be accounted for
	if (_convDispOp.forwardFlow())
		// Forward Flow: outlet is last cell
		return _disc.nComp + (_disc.nCol - 1) * _disc.nComp;
	else
		// Backward flow: Outlet is first cell
		return _disc.nComp;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithPores<ConvDispOperator>::localInletComponentIndex(unsigned int port) const CADET_NOEXCEPT
{
	return 0;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithPores<ConvDispOperator>::localOutletComponentStride(unsigned int port) const CADET_NOEXCEPT
{
	return 1;
}

template <typename ConvDispOperator>
unsigned int LumpedRateModelWithPores<ConvDispOperator>::localInletComponentStride(unsigned int port) const CADET_NOEXCEPT
{
	return 1;
}

template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut)
{
	// @todo Write this function
}

template <typename ConvDispOperator>
bool LumpedRateModelWithPores<ConvDispOperator>::setParameter(const ParameterId& pId, double value)
{
	if (pId.unitOperation == _unitOpIdx)
	{
		// Intercept changes to PAR_TYPE_VOLFRAC when not specified per axial cell (but once globally)
		if (_axiallyConstantParTypeVolFrac && (pId.name == hashString("PAR_TYPE_VOLFRAC")))
		{
			if ((pId.section != SectionIndep) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep))
				return false;
			if (pId.particleType >= _disc.nParType)
				return false;

			for (unsigned int i = 0; i < _disc.nCol; ++i)
				_parTypeVolFrac[i * _disc.nParType + pId.particleType].setValue(value);

			return true;
		}

		if (multiplexTypeParameterValue(pId, hashString("PAR_RADIUS"), _singleParRadius, _parRadius, value, nullptr))
			return true;
		if (multiplexTypeParameterValue(pId, hashString("PAR_POROSITY"), _singleParPorosity, _parPorosity, value, nullptr))
			return true;

		if (multiplexCompTypeSecParameterValue(pId, hashString("FILM_DIFFUSION"), _filmDiffusionMode, _filmDiffusion, _disc.nParType, _disc.nComp, value, nullptr))
			return true;
		if (multiplexCompTypeSecParameterValue(pId, hashString("PORE_ACCESSIBILITY"), _poreAccessFactorMode, _poreAccessFactor, _disc.nParType, _disc.nComp, value, nullptr))
			return true;

		const int mpIc = multiplexInitialConditions(pId, value, false);
		if (mpIc > 0)
			return true;
		else if (mpIc < 0)
			return false;

		if (_convDispOp.setParameter(pId, value))
			return true;

		if (_filmDiffDep)
		{
			if (_filmDiffDep->hasParameter(pId))
			{
				_filmDiffDep->setParameter(pId, value);
				return true;
			}
		}
	}

	return UnitOperationBase::setParameter(pId, value);
}

template <typename ConvDispOperator>
void LumpedRateModelWithPores<ConvDispOperator>::setSensitiveParameterValue(const ParameterId& pId, double value)
{
	if (pId.unitOperation == _unitOpIdx)
	{
		// Intercept changes to PAR_TYPE_VOLFRAC when not specified per axial cell (but once globally)
		if (_axiallyConstantParTypeVolFrac && (pId.name == hashString("PAR_TYPE_VOLFRAC")))
		{
			if ((pId.section != SectionIndep) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep))
				return;
			if (pId.particleType >= _disc.nParType)
				return;

			if (!contains(_sensParams, &_parTypeVolFrac[pId.particleType]))
				return;

			for (unsigned int i = 0; i < _disc.nCol; ++i)
				_parTypeVolFrac[i * _disc.nParType + pId.particleType].setValue(value);

			return;
		}

		if (multiplexTypeParameterValue(pId, hashString("PAR_RADIUS"), _singleParRadius, _parRadius, value, &_sensParams))
			return;
		if (multiplexTypeParameterValue(pId, hashString("PAR_POROSITY"), _singleParPorosity, _parPorosity, value, &_sensParams))
			return;

		if (multiplexCompTypeSecParameterValue(pId, hashString("FILM_DIFFUSION"), _filmDiffusionMode, _filmDiffusion, _disc.nParType, _disc.nComp, value, &_sensParams))
			return;
		if (multiplexCompTypeSecParameterValue(pId, hashString("PORE_ACCESSIBILITY"), _poreAccessFactorMode, _poreAccessFactor, _disc.nParType, _disc.nComp, value, &_sensParams))
			return;
		if (multiplexInitialConditions(pId, value, true) != 0)
			return;

		if (_convDispOp.setSensitiveParameterValue(_sensParams, pId, value))
			return;

		if (_filmDiffDep)
		{
			active* const param = _filmDiffDep->getParameter(pId);
			if (param)
			{
				param->setValue(value);
				return;
			}
		}
	}

	UnitOperationBase::setSensitiveParameterValue(pId, value);
}

template <typename ConvDispOperator>
bool LumpedRateModelWithPores<ConvDispOperator>::setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue)
{
	if (pId.unitOperation == _unitOpIdx)
	{
		// Intercept changes to PAR_TYPE_VOLFRAC when not specified per axial cell (but once globally)
		if (_axiallyConstantParTypeVolFrac && (pId.name == hashString("PAR_TYPE_VOLFRAC")))
		{
			if ((pId.section != SectionIndep) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep))
				return false;
			if (pId.particleType >= _disc.nParType)
				return false;

			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;

			// Register parameter and set AD seed / direction
			_sensParams.insert(&_parTypeVolFrac[pId.particleType]);
			for (unsigned int i = 0; i < _disc.nCol; ++i)
				_parTypeVolFrac[i * _disc.nParType + pId.particleType].setADValue(adDirection, adValue);

			return true;
		}

		if (multiplexTypeParameterAD(pId, hashString("PAR_RADIUS"), _singleParRadius, _parRadius, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (multiplexTypeParameterAD(pId, hashString("PAR_POROSITY"), _singleParPorosity, _parPorosity, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (multiplexCompTypeSecParameterAD(pId, hashString("FILM_DIFFUSION"), _filmDiffusionMode, _filmDiffusion, _disc.nParType, _disc.nComp, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (multiplexCompTypeSecParameterAD(pId, hashString("PORE_ACCESSIBILITY"), _poreAccessFactorMode, _poreAccessFactor, _disc.nParType, _disc.nComp, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		const int mpIc = multiplexInitialConditions(pId, adDirection, adValue);
		if (mpIc > 0)
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}
		else if (mpIc < 0)
			return false;

		if (_convDispOp.setSensitiveParameter(_sensParams, pId, adDirection, adValue))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (_filmDiffDep)
		{
			active* const param = _filmDiffDep->getParameter(pId);
			if (param)
			{
				LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
				param->setADValue(adDirection, adValue);
				return true;
			}
		}
	}

	return UnitOperationBase::setSensitiveParameter(pId, adDirection, adValue);
}


template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeMobilePhase(double* buffer) const
{
	const int blockSize = _disc.nComp * _disc.nCol;
	std::copy_n(_idx.c(_data), blockSize, buffer);
	return blockSize;
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeSolidPhase(double* buffer) const
{
	int numWritten = 0;
	for (unsigned int i = 0; i < _disc.nParType; ++i)
	{
		const int n = writeSolidPhase(i, buffer);
		buffer += n;
		numWritten += n;
	}
	return numWritten;
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeParticleMobilePhase(double* buffer) const
{
	int numWritten = 0;
	for (unsigned int i = 0; i < _disc.nParType; ++i)
	{
		const int n = writeParticleMobilePhase(i, buffer);
		buffer += n;
		numWritten += n;
	}
	return numWritten;
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeSolidPhase(unsigned int parType, double* buffer) const
{
	cadet_assert(parType < _disc.nParType);

	const unsigned int stride = _disc.nComp + _disc.strideBound[parType];
	double const* ptr = _data + _idx.offsetCp(ParticleTypeIndex{parType}) + _idx.strideParLiquid();
	for (unsigned int i = 0; i < _disc.nCol; ++i)
	{
		std::copy_n(ptr, _disc.strideBound[parType], buffer);
		buffer += _disc.strideBound[parType];
		ptr += stride;
	}
	return _disc.nCol * _disc.strideBound[parType];
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeParticleMobilePhase(unsigned int parType, double* buffer) const
{
	cadet_assert(parType < _disc.nParType);

	const unsigned int stride = _disc.nComp + _disc.strideBound[parType];
	double const* ptr = _data + _idx.offsetCp(ParticleTypeIndex{parType});
	for (unsigned int i = 0; i < _disc.nCol; ++i)
	{
		std::copy_n(ptr, _disc.nComp, buffer);
		buffer += _disc.nComp;
		ptr += stride;
	}
	return _disc.nCol * _disc.nComp;
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeParticleFlux(double* buffer) const
{
	const int blockSize = numParticleFluxDofs();
	std::copy_n(_idx.jf(_data), blockSize, buffer);
	return blockSize;
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeParticleFlux(unsigned int parType, double* buffer) const
{
	const unsigned int blockSize = _disc.nComp * _disc.nCol;
	std::copy_n(_idx.jf(_data) + blockSize * parType, blockSize, buffer);
	return blockSize;
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeInlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);
	std::copy_n(_data, _disc.nComp, buffer);
	return _disc.nComp;
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeInlet(double* buffer) const
{
	std::copy_n(_data, _disc.nComp, buffer);
	return _disc.nComp;
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeOutlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);

	if (_model._convDispOp.forwardFlow())
		std::copy_n(&_idx.c(_data, _disc.nCol - 1, 0), _disc.nComp, buffer);
	else
		std::copy_n(&_idx.c(_data, 0, 0), _disc.nComp, buffer);

	return _disc.nComp;
}

template <typename ConvDispOperator>
int LumpedRateModelWithPores<ConvDispOperator>::Exporter::writeOutlet(double* buffer) const
{
	if (_model._convDispOp.forwardFlow())
		std::copy_n(&_idx.c(_data, _disc.nCol - 1, 0), _disc.nComp, buffer);
	else
		std::copy_n(&_idx.c(_data, 0, 0), _disc.nComp, buffer);

	return _disc.nComp;
}

}  // namespace model

}  // namespace cadet

#include "model/LumpedRateModelWithPores-InitialConditions.cpp"
#include "model/LumpedRateModelWithPores-LinearSolver.cpp"

namespace cadet
{

namespace model
{

// Template instantiations
template class LumpedRateModelWithPores<parts::AxialConvectionDispersionOperator>;
template class LumpedRateModelWithPores<parts::RadialConvectionDispersionOperator>;

void registerLumpedRateModelWithPores(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models)
{
	typedef LumpedRateModelWithPores<parts::AxialConvectionDispersionOperator> AxialLRMP;
	typedef LumpedRateModelWithPores<parts::RadialConvectionDispersionOperator> RadialLRMP;

	models[AxialLRMP::identifier()] = [](UnitOpIdx uoId) { return new AxialLRMP(uoId); };
	models["LRMP"] = [](UnitOpIdx uoId) { return new AxialLRMP(uoId); };

	models[RadialLRMP::identifier()] = [](UnitOpIdx uoId) { return new RadialLRMP(uoId); };
	models["RLRMP"] = [](UnitOpIdx uoId) { return new RadialLRMP(uoId); };
}

}  // namespace model

}  // namespace cadet
