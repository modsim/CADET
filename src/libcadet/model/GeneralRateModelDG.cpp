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
//TODO: delete iostream, iomanip include
#include <iostream>
#include <iomanip>

#include "model/GeneralRateModelDG.hpp"
#include "BindingModelFactory.hpp"
#include "ReactionModelFactory.hpp"
#include "ParamReaderHelper.hpp"
#include "ParamReaderScopes.hpp"
#include "cadet/Exceptions.hpp"
#include "cadet/ExternalFunction.hpp"
#include "cadet/SolutionRecorder.hpp"
#include "ConfigurationHelper.hpp"
#include "model/BindingModel.hpp"
#include "model/ReactionModel.hpp"
#include "model/ParameterDependence.hpp"
#include "model/parts/BindingCellKernel.hpp"
#include "SimulationTypes.hpp"
#include "linalg/Norms.hpp"
#include "linalg/Subset.hpp"

#include "AdUtils.hpp"
#include "SensParamUtil.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <iterator>

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


GeneralRateModelDG::GeneralRateModelDG(UnitOpIdx unitOpIdx) : UnitOperationBase(unitOpIdx),
	_hasSurfaceDiffusion(0, false), _dynReactionBulk(nullptr),
	_globalJac(), _globalJacDisc(), _jacInlet(), _hasParDepSurfDiffusion(false),
	_analyticJac(true), _jacobianAdDirs(0), _factorizeJacobian(false), _tempState(nullptr),
	_initC(0), _initCp(0), _initQ(0), _initState(0), _initStateDot(0)
{
}

GeneralRateModelDG::~GeneralRateModelDG() CADET_NOEXCEPT
{
	delete[] _tempState;

	delete _dynReactionBulk;

	clearParDepSurfDiffusion();

	delete[] _disc.nParCell;
	delete[] _disc.parTypeOffset;
	delete[] _disc.nParPointsBeforeType;
	delete[] _disc.nBound;
	delete[] _disc.boundOffset;
	delete[] _disc.strideBound;
	delete[] _disc.nBoundBeforeType;
}

unsigned int GeneralRateModelDG::numDofs() const CADET_NOEXCEPT
{
	// Column bulk DOFs: nPoints * nComp
	// Particle DOFs: nPoints * nParType particles each having nComp (liquid phase) + sum boundStates (solid phase) DOFs
	//                in each shell; there are nParCell shells for each particle type
	// Inlet DOFs: nComp
	return _disc.nPoints * _disc.nComp + _disc.parTypeOffset[_disc.nParType] + _disc.nComp;
}

unsigned int GeneralRateModelDG::numPureDofs() const CADET_NOEXCEPT
{
	// Column bulk DOFs: nPoints * nComp
	// Particle DOFs: nPoints particles each having nComp (liquid phase) + sum boundStates (solid phase) DOFs
	//                in each shell; there are nPar shells
	return _disc.nPoints * _disc.nComp  + _disc.parTypeOffset[_disc.nParType];
}


bool GeneralRateModelDG::usesAD() const CADET_NOEXCEPT
{
#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
	// We always need AD if we want to check the analytical Jacobian
	return true;
#else
	// We only need AD if we are not computing the Jacobian analytically
	return !_analyticJac;
#endif
}

void GeneralRateModelDG::clearParDepSurfDiffusion()
{
	if (_singleParDepSurfDiffusion)
	{
		if (!_parDepSurfDiffusion.empty())
			delete _parDepSurfDiffusion[0];
	}
	else
	{
		for (IParameterStateDependence* pd : _parDepSurfDiffusion)
			delete pd;
	}

	_parDepSurfDiffusion.clear();
}

bool GeneralRateModelDG::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper)
{
	// ==== Read discretization
	_disc.nComp = paramProvider.getInt("NCOMP");

	paramProvider.pushScope("discretization");

	if (paramProvider.exists("POLYDEG"))
		_disc.polyDeg = paramProvider.getInt("POLYDEG");
	else
		_disc.polyDeg = 4u; // default value
	if (paramProvider.getInt("POLYDEG") < 1)
		throw InvalidParameterException("Polynomial degree must be at least 1!");
	else if (_disc.polyDeg < 3)
		LOG(Warning) << "Polynomial degree > 2 in bulk discretization (cf. POLYDEG) is always recommended for performance reasons.";

	_disc.nNodes = _disc.polyDeg + 1;

	if (paramProvider.exists("NELEM"))
		_disc.nCol = paramProvider.getInt("NELEM");
	else if (paramProvider.exists("NCOL"))
		_disc.nCol = std::max(1u, paramProvider.getInt("NCOL") / _disc.nNodes); // number of elements is rounded down
	else
		throw InvalidParameterException("Specify field NELEM (or NCOL)");

	if (_disc.nCol < 1)
		throw InvalidParameterException("Number of column elements must be at least 1!");

	_disc.nPoints = _disc.nNodes * _disc.nCol;

	int polynomial_integration_mode = 0;
	if (paramProvider.exists("EXACT_INTEGRATION"))
		polynomial_integration_mode = paramProvider.getInt("EXACT_INTEGRATION");
	_disc.exactInt = static_cast<bool>(polynomial_integration_mode); // only integration mode 0 applies the inexact collocated diagonal LGL mass matrix

	const std::vector<int> nBound = paramProvider.getIntArray("NBOUND");
	if (nBound.size() < _disc.nComp)
		throw InvalidParameterException("Field NBOUND contains too few elements (NCOMP = " + std::to_string(_disc.nComp) + " required)");
	if ((nBound.size() > _disc.nComp) && (nBound.size() < _disc.nComp * _disc.nParType))
		throw InvalidParameterException("Field NBOUND must have NCOMP (" + std::to_string(_disc.nComp) + ") or NCOMP * NPARTYPE (" + std::to_string(_disc.nComp * _disc.nParType) + ") entries");

	if (paramProvider.exists("NPARTYPE"))
		_disc.nParType = paramProvider.getInt("NPARTYPE");
	else // Infer number of particle types
		_disc.nParType = nBound.size() / _disc.nComp;

	std::vector<int> parPolyDeg(_disc.nParType);
	std::vector<int> ParNelem(_disc.nParType);
	std::vector<bool> parExactInt(_disc.nParType);
	_disc.parPolyDeg = new unsigned int[_disc.nParType];
	_disc.nParCell = new unsigned int[_disc.nParType];
	_disc.parExactInt = new bool[_disc.nParType];
	_disc.parGSM = new bool[_disc.nParType];

	if (paramProvider.exists("PAR_POLYDEG"))
	{
		parPolyDeg = paramProvider.getIntArray("PAR_POLYDEG");
		if ((std::any_of(parPolyDeg.begin(), parPolyDeg.end(), [](int value) { return value < 1; })))
			throw InvalidParameterException("Particle polynomial degrees must be at least 1!");
		ParNelem = paramProvider.getIntArray("PAR_NELEM");
		if ((std::any_of(ParNelem.begin(), ParNelem.end(), [](int value) { return value < 1; })))
			throw InvalidParameterException("Particle number of elements must be at least 1!");
		parExactInt = paramProvider.getBoolArray("PAR_EXACT_INTEGRATION");
		if ((std::any_of(parExactInt.begin(), parExactInt.end(), [](bool value) { return !value; })))
			LOG(Warning) << "Inexact integration method (cf. PAR_EXACT_INTEGRATION) in particles might add severe! stiffness to the system and disables consistent initialization!";

		if (parPolyDeg.size() == 1)
		{
			// Multiplex number of particle shells to all particle types
			for (unsigned int i = 0; i < _disc.nParType; ++i)
				std::fill(_disc.parPolyDeg, _disc.parPolyDeg + _disc.nParType, parPolyDeg[0]);
		}
		else if (parPolyDeg.size() < _disc.nParType)
			throw InvalidParameterException("Field PAR_POLYDEG must have 1 or NPARTYPE (" + std::to_string(_disc.nParType) + ") entries");
		else
			std::copy_n(parPolyDeg.begin(), _disc.nParType, _disc.parPolyDeg);
		if (ParNelem.size() == 1)
		{
			// Multiplex number of particle shells to all particle types
			for (unsigned int i = 0; i < _disc.nParType; ++i)
				std::fill(_disc.nParCell, _disc.nParCell + _disc.nParType, ParNelem[0]);
		}
		else if (ParNelem.size() < _disc.nParType)
			throw InvalidParameterException("Field PAR_NELEM must have 1 or NPARTYPE (" + std::to_string(_disc.nParType) + ") entries");
		else
			std::copy_n(ParNelem.begin(), _disc.nParType, _disc.nParCell);
		if (parExactInt.size() == 1)
		{
			// Multiplex number of particle shells to all particle types
			for (unsigned int i = 0; i < _disc.nParType; ++i)
				std::fill(_disc.parExactInt, _disc.parExactInt + _disc.nParType, parExactInt[0]);
		}
		else if (parExactInt.size() < _disc.nParType)
			throw InvalidParameterException("Field PAR_EXACT_INTEGRATION must have 1 or NPARTYPE (" + std::to_string(_disc.nParType) + ") entries");
		else
			std::copy_n(parExactInt.begin(), _disc.nParType, _disc.parExactInt);
	}
	else if (paramProvider.exists("NPAR"))
	{
		const std::vector<int> nParPoints = paramProvider.getIntArray("NPAR");
		if ((nParPoints.size() > 1) && (nParPoints.size() < _disc.nParType))
			throw InvalidParameterException("Field NPAR must have 1 or NPARTYPE (" + std::to_string(_disc.nParType) + ") entries");

		for (unsigned int par = 0; par < _disc.nParType; par++)
			parPolyDeg[par] = std::max(1, std::min(nParPoints[par] - 1, 4));
	}
	else
		throw InvalidParameterException("Specify field PAR_POLYDEG (or NPAR)");

	if (paramProvider.exists("PAR_GSM"))
	{
		std::vector<bool> parGSM = paramProvider.getBoolArray("PAR_GSM");
		if (parGSM.size() == 1)
			std::fill(_disc.parGSM, _disc.parGSM + _disc.nParType, parGSM[0]);
		else
			std::copy_n(parGSM.begin(), _disc.nParType, _disc.parGSM);
		for (int type = 0; type < _disc.nParType; type++)
			if (_disc.parGSM[type] && _disc.nParCell[type] != 1)
				throw InvalidParameterException("Field PAR_NELEM must equal one to use a GSM discretization in the corresponding particle type");
	}
	else // Use GSM as default for particle discretization
	{
		for (int type = 0; type < _disc.nParType; type++)
			_disc.parGSM[type] = (_disc.nParCell[type] == 1);
	}

	// Compute discretization operators and initialize containers
	_disc.initializeDG();

	_disc.nBound = new unsigned int[_disc.nComp * _disc.nParType];
	if (nBound.size() < _disc.nComp * _disc.nParType)
	{
		// Multiplex number of bound states to all particle types
		for (unsigned int i = 0; i < _disc.nParType; ++i)
			std::copy_n(nBound.begin(), _disc.nComp, _disc.nBound + i * _disc.nComp);
	}
	else
		std::copy_n(nBound.begin(), _disc.nComp * _disc.nParType, _disc.nBound);

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
	_disc.nParPointsBeforeType = new unsigned int[_disc.nParType + 1];
	_disc.parTypeOffset[0] = 0;
	_disc.nParPointsBeforeType[0] = 0;
	unsigned int nTotalParPoints = 0;
	for (unsigned int j = 1; j < _disc.nParType + 1; ++j)
	{
		_disc.parTypeOffset[j] = _disc.parTypeOffset[j-1] + (_disc.nComp + _disc.strideBound[j-1]) * _disc.nParPoints[j-1] * _disc.nPoints;
		_disc.nParPointsBeforeType[j] = _disc.nParPointsBeforeType[j-1] + _disc.nParPoints[j-1];
		nTotalParPoints += _disc.nParPoints[j-1];
	}
	_disc.nParPointsBeforeType[_disc.nParType] = nTotalParPoints;

	// Configure particle discretization
	_parCellSize.resize(_disc.offsetMetric[_disc.nParType]);
	_parCenterRadius.resize(_disc.offsetMetric[_disc.nParType]);
	_parOuterSurfAreaPerVolume.resize(_disc.offsetMetric[_disc.nParType]);
	_parInnerSurfAreaPerVolume.resize(_disc.offsetMetric[_disc.nParType]);

	// Read particle discretization mode and default to "EQUIDISTANT_PAR"
	_parDiscType = std::vector<ParticleDiscretizationMode>(_disc.nParType, ParticleDiscretizationMode::Equidistant);
	std::vector<std::string> pdt = paramProvider.getStringArray("PAR_DISC_TYPE");
	if ((pdt.size() == 1) && (_disc.nParType > 1))
	{
		// Multiplex using first value
		pdt.resize(_disc.nParType, pdt[0]);
	}
	else if (pdt.size() < _disc.nParType)
		throw InvalidParameterException("Field PAR_DISC_TYPE contains too few elements (" + std::to_string(_disc.nParType) + " required)");

	for (unsigned int i = 0; i < _disc.nParType; ++i)
	{
		if (pdt[i] == "EQUIVOLUME_PAR")
			_parDiscType[i] = ParticleDiscretizationMode::Equivolume;
		else if (pdt[i] == "USER_DEFINED_PAR")
			_parDiscType[i] = ParticleDiscretizationMode::UserDefined;
	}

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

	if (paramProvider.exists("PAR_DISC_VECTOR"))
	{
		_parDiscVector = paramProvider.getDoubleArray("PAR_DISC_VECTOR");
		if (_parDiscVector.size() < nTotalParPoints + _disc.nParType)
			throw InvalidParameterException("Field PAR_DISC_VECTOR contains too few elements (Sum [NPAR + 1] = " + std::to_string(nTotalParPoints + _disc.nParType) + " required)");
	}

	// Determine whether analytic Jacobian should be used but don't set it right now.
	// We need to setup Jacobian matrices first.
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	const bool analyticJac = paramProvider.getBool("USE_ANALYTIC_JACOBIAN");
#else
	const bool analyticJac = false;
#endif

	// Allocate space for initial conditions
	_initC.resize(_disc.nComp);
	_initCp.resize(_disc.nComp * _disc.nParType);
	_initQ.resize(nTotalBound);

	// Determine whether surface diffusion optimization is applied (decreases Jacobian size) //@TODO?
	const bool optimizeParticleJacobianBandwidth = paramProvider.exists("OPTIMIZE_PAR_BANDWIDTH") ? paramProvider.getBool("OPTIMIZE_PAR_BANDWIDTH") : true;

	// Create nonlinear solver for consistent initialization
	configureNonlinearSolver(paramProvider);

	paramProvider.popScope();

	// ==== Construct and configure parameter dependencies
	clearParDepSurfDiffusion();
	bool parSurfDiffDepConfSuccess = true;
	if (paramProvider.exists("PAR_SURFDIFFUSION_DEP"))
	{
		const std::vector<std::string> psdDepNames = paramProvider.getStringArray("PAR_SURFDIFFUSION_DEP");
		if ((psdDepNames.size() == 1) || (_disc.nParType == 1))
			_singleParDepSurfDiffusion = true;

		if (!_singleParDepSurfDiffusion && (psdDepNames.size() < _disc.nParType))
			throw InvalidParameterException("Field PAR_SURFDIFFUSION_DEP contains too few elements (" + std::to_string(_disc.nParType) + " required)");
		else if (_singleParDepSurfDiffusion && (psdDepNames.size() != 1))
			throw InvalidParameterException("Field PAR_SURFDIFFUSION_DEP requires (only) 1 element");

		if (_singleParDepSurfDiffusion)
		{
			if ((psdDepNames[0] == "") || (psdDepNames[0] == "NONE") || (psdDepNames[0] == "DUMMY"))
			{
				_hasParDepSurfDiffusion = false;
				_singleParDepSurfDiffusion = true;
				_parDepSurfDiffusion = std::vector<IParameterStateDependence*>(_disc.nParType, nullptr);
			}
			else
			{
				IParameterStateDependence* const pd = helper.createParameterStateDependence(psdDepNames[0]);
				if (!pd)
					throw InvalidParameterException("Unknown parameter dependence " + psdDepNames[0]);

				_parDepSurfDiffusion = std::vector<IParameterStateDependence*>(_disc.nParType, pd);
				parSurfDiffDepConfSuccess = pd->configureModelDiscretization(paramProvider, _disc.nComp, _disc.nBound, _disc.boundOffset);
				_hasParDepSurfDiffusion = true;
			}
		}
		else
		{
			_parDepSurfDiffusion = std::vector<IParameterStateDependence*>(_disc.nParType, nullptr);

			for (unsigned int i = 0; i < _disc.nParType; ++i)
			{
				if ((psdDepNames[0] == "") || (psdDepNames[0] == "NONE") || (psdDepNames[0] == "DUMMY"))
					continue;

				_parDepSurfDiffusion[i] = helper.createParameterStateDependence(psdDepNames[i]);
				if (!_parDepSurfDiffusion[i])
					throw InvalidParameterException("Unknown parameter dependence " + psdDepNames[i]);

				parSurfDiffDepConfSuccess = _parDepSurfDiffusion[i]->configureModelDiscretization(paramProvider, _disc.nComp, _disc.nBound + i * _disc.nComp, _disc.boundOffset + i * _disc.nComp) && parSurfDiffDepConfSuccess;
			}

			_hasParDepSurfDiffusion = std::any_of(_parDepSurfDiffusion.cbegin(), _parDepSurfDiffusion.cend(), [](IParameterStateDependence const* pd) -> bool { return pd; });
		}
	}
	else
	{
		_hasParDepSurfDiffusion = false;
		_singleParDepSurfDiffusion = true;
		_parDepSurfDiffusion = std::vector<IParameterStateDependence*>(_disc.nParType, nullptr);
	}

	if (optimizeParticleJacobianBandwidth)
	{
		// Check whether surface diffusion is present
		_hasSurfaceDiffusion = std::vector<bool>(_disc.nParType, false);
		if (paramProvider.exists("PAR_SURFDIFFUSION"))
		{
			const std::vector<double> surfDiff = paramProvider.getDoubleArray("PAR_SURFDIFFUSION");
			for (unsigned int i = 0; i < _disc.nParType; ++i)
			{
				// Assume particle surface diffusion if a parameter dependence is present
				if (_parDepSurfDiffusion[i])
				{
					_hasSurfaceDiffusion[i] = true;
					continue;
				}

				double const* const lsd = surfDiff.data() + _disc.nBoundBeforeType[i];

				// Check surface diffusion coefficients of each particle type
				for (unsigned int j = 0; j < _disc.strideBound[i]; ++j)
				{
					if (lsd[j] != 0.0)
					{
						_hasSurfaceDiffusion[i] = true;
						break;
					}
				}
			}
		}
	}
	else
	{
		// Assume that surface diffusion is present
		_hasSurfaceDiffusion = std::vector<bool>(_disc.nParType, true);
	}

	unsigned int strideColNode = _disc.nComp;
	const bool transportSuccess = _convDispOp.configureModelDiscretization(paramProvider, helper, _disc.nComp, polynomial_integration_mode, _disc.nCol, _disc.polyDeg, strideColNode);

	_disc.curSection = -1;

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

	// Allocate memory
	_tempState = new double[numDofs()];

	if (_disc.exactInt)
		_jacInlet.resize(_disc.nNodes, 1); // first cell depends on inlet concentration (same for every component)
	else
		_jacInlet.resize(1, 1); // first node depends on inlet concentration (same for every component)

	// set jacobian pattern
	_globalJacDisc.resize(numDofs(), numDofs());
	_globalJac.resize(numDofs(), numDofs());
	// pattern is set in configure(), after surface diffusion is read
	// FDJac = MatrixXd::Zero(numDofs(), numDofs()); // todo delete

	// Set whether analytic Jacobian is used
	useAnalyticJacobian(analyticJac);

	return transportSuccess && parSurfDiffDepConfSuccess && bindingConfSuccess && reactionConfSuccess;
}
 
bool GeneralRateModelDG::configure(IParameterProvider& paramProvider)
{
	_parameters.clear();

	const bool transportSuccess = _convDispOp.configure(_unitOpIdx, paramProvider, _parameters);

	// Read geometry parameters
	_colPorosity = paramProvider.getDouble("COL_POROSITY");
	_singleParRadius = readAndRegisterMultiplexTypeParam(paramProvider, _parameters, _parRadius, "PAR_RADIUS", _disc.nParType, _unitOpIdx);
	_singleParPorosity = readAndRegisterMultiplexTypeParam(paramProvider, _parameters, _parPorosity, "PAR_POROSITY", _disc.nParType, _unitOpIdx);

	// Let PAR_CORERADIUS default to 0.0 for backwards compatibility
	if (paramProvider.exists("PAR_CORERADIUS"))
		_singleParCoreRadius = readAndRegisterMultiplexTypeParam(paramProvider, _parameters, _parCoreRadius, "PAR_CORERADIUS", _disc.nParType, _unitOpIdx);
	else
	{
		_singleParCoreRadius = true;
		_parCoreRadius = std::vector<active>(_disc.nParType, 0.0);
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
			_parTypeVolFrac.resize(_disc.nPoints * _disc.nParType, 1.0);
			for (unsigned int i = 1; i < _disc.nPoints; ++i)
				std::copy(_parTypeVolFrac.begin(), _parTypeVolFrac.begin() + _disc.nParType, _parTypeVolFrac.begin() + _disc.nParType * i);
		}
		else
			_axiallyConstantParTypeVolFrac = false;
	}
	else
	{
		_parTypeVolFrac.resize(_disc.nPoints, 1.0);
		_axiallyConstantParTypeVolFrac = false;
	}

	// Check whether all sizes are matched
	if (_disc.nParType != _parRadius.size())
		throw InvalidParameterException("Number of elements in field PAR_RADIUS does not match number of particle types");
	if (_disc.nParType * _disc.nPoints != _parTypeVolFrac.size())
		throw InvalidParameterException("Number of elements in field PAR_TYPE_VOLFRAC does not match number of particle types times number of axial cells");
	if (_disc.nParType != _parPorosity.size())
		throw InvalidParameterException("Number of elements in field PAR_POROSITY does not match number of particle types");
	if (_disc.nParType != _parCoreRadius.size())
		throw InvalidParameterException("Number of elements in field PAR_CORERADIUS does not match number of particle types");

	// Check that particle volume fractions sum to 1.0
	for (unsigned int i = 0; i < _disc.nPoints; ++i)
	{
		const double volFracSum = std::accumulate(_parTypeVolFrac.begin() + i * _disc.nParType, _parTypeVolFrac.begin() + (i+1) * _disc.nParType, 0.0,
			[](double a, const active& b) -> double { return a + static_cast<double>(b); });
		if (std::abs(1.0 - volFracSum) > 1e-10)
			throw InvalidParameterException("Sum of field PAR_TYPE_VOLFRAC differs from 1.0 (is " + std::to_string(volFracSum) + ") in axial cell " + std::to_string(i));
	}

	// Read vectorial parameters (which may also be section dependent; transport)
	_filmDiffusionMode = readAndRegisterMultiplexCompTypeSecParam(paramProvider, _parameters, _filmDiffusion, "FILM_DIFFUSION", _disc.nParType, _disc.nComp, _unitOpIdx);
	_parDiffusionMode = readAndRegisterMultiplexCompTypeSecParam(paramProvider, _parameters, _parDiffusion, "PAR_DIFFUSION", _disc.nParType, _disc.nComp, _unitOpIdx);

	_disc.offsetSurfDiff = new unsigned int[_disc.strideBound[_disc.nParType]];
	if (paramProvider.exists("PAR_SURFDIFFUSION"))
		_parSurfDiffusionMode = readAndRegisterMultiplexBndCompTypeSecParam(paramProvider, _parameters, _parSurfDiffusion, "PAR_SURFDIFFUSION", _disc.nParType, _disc.nComp, _disc.strideBound, _disc.nBound, _unitOpIdx);
	else
	{
		_parSurfDiffusionMode = MultiplexMode::Component;
		_parSurfDiffusion.resize(_disc.strideBound[_disc.nParType], 0.0);
	}

	bool parSurfDiffDepConfSuccess = true;
	if (_hasParDepSurfDiffusion)
	{
		if (_singleParDepSurfDiffusion && _parDepSurfDiffusion[0])
		{
			parSurfDiffDepConfSuccess = _parDepSurfDiffusion[0]->configure(paramProvider, _unitOpIdx, ParTypeIndep, "PAR_SURFDIFFUSION");
		}
		else if (!_singleParDepSurfDiffusion)
		{
			for (unsigned int i = 0; i < _disc.nParType; ++i)
			{
				if (!_parDepSurfDiffusion[i])
					continue;

				parSurfDiffDepConfSuccess = _parDepSurfDiffusion[i]->configure(paramProvider, _unitOpIdx, i, "PAR_SURFDIFFUSION") && parSurfDiffDepConfSuccess;
			}
		}
	}

	if ((_filmDiffusion.size() < _disc.nComp * _disc.nParType) || (_filmDiffusion.size() % (_disc.nComp * _disc.nParType) != 0))
		throw InvalidParameterException("Number of elements in field FILM_DIFFUSION is not a positive multiple of NCOMP * NPARTYPE (" + std::to_string(_disc.nComp * _disc.nParType) + ")");
	if ((_parDiffusion.size() < _disc.nComp * _disc.nParType) || (_parDiffusion.size() % (_disc.nComp * _disc.nParType) != 0))
		throw InvalidParameterException("Number of elements in field PAR_DIFFUSION is not a positive multiple of NCOMP * NPARTYPE (" + std::to_string(_disc.nComp * _disc.nParType) + ")");
	if ((_parSurfDiffusion.size() < _disc.strideBound[_disc.nParType]) || ((_disc.strideBound[_disc.nParType] > 0) && (_parSurfDiffusion.size() % _disc.strideBound[_disc.nParType] != 0)))
		throw InvalidParameterException("Number of elements in field PAR_SURFDIFFUSION is not a positive multiple of NTOTALBND (" + std::to_string(_disc.strideBound[_disc.nParType]) + ")");

	if (paramProvider.exists("PORE_ACCESSIBILITY"))
		_poreAccessFactorMode = readAndRegisterMultiplexCompTypeSecParam(paramProvider, _parameters, _poreAccessFactor, "PORE_ACCESSIBILITY", _disc.nParType, _disc.nComp, _unitOpIdx);
	else
	{
		_poreAccessFactorMode = MultiplexMode::ComponentType;
		_poreAccessFactor = std::vector<cadet::active>(_disc.nComp * _disc.nParType, 1.0);
	}

	if (_disc.nComp * _disc.nParType != _poreAccessFactor.size())
		throw InvalidParameterException("Number of elements in field PORE_ACCESSIBILITY differs from NCOMP * NPARTYPE (" + std::to_string(_disc.nComp * _disc.nParType) + ")");

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

	// Calculate the particle radial discretization variables (_parCellSize, _parCenterRadius, etc.)
	updateRadialDisc();

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

	// Reconfigure binding model
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

				MultiplexedScopeSelector scopeGuard(paramProvider, "adsorption", type, _disc.nParType == 1, true);
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

	// jaobian pattern set after binding and particle surface diffusion are configured
	setJacobianPattern_GRM(_globalJac, 0, _dynReactionBulk);
	_globalJacDisc = _globalJac;
	// the solver repetitively solves the linear system with a static pattern of the jacobian (set above). 
	// The goal of analyzePattern() is to reorder the nonzero elements of the matrix, such that the factorization step creates less fill-in
	_globalSolver.analyzePattern(_globalJacDisc.block(_disc.nComp, _disc.nComp, numPureDofs(), numPureDofs()));

	return transportSuccess && parSurfDiffDepConfSuccess && bindingConfSuccess && dynReactionConfSuccess;
}

unsigned int GeneralRateModelDG::threadLocalMemorySize() const CADET_NOEXCEPT
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

unsigned int GeneralRateModelDG::numAdDirsForJacobian() const CADET_NOEXCEPT
{
	// The global DG Jacobian is banded around the main diagonal and has additional (also banded) entries for film diffusion.
	// To feasibly seed and reconstruct the Jacobian, we create dedicated active directions for the bulk and each particle type (see @ todo)
	Indexer idxr(_disc);
	
	int sumParBandwidth = 0;
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		sumParBandwidth += idxr.strideParBlock(type);
	}

	return _convDispOp.requiredADdirs() + sumParBandwidth;
}

void GeneralRateModelDG::useAnalyticJacobian(const bool analyticJac)
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

void GeneralRateModelDG::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac)
{
	// calculate offsets between surface diffusion storage and state vector order
	orderSurfDiff();

	Indexer idxr(_disc);

	// todo: only reset jacobian pattern if it changes, i.e. once in configuration and then only for changes in SurfDiff+kinetic binding.
 	setJacobianPattern_GRM(_globalJac, 0, _dynReactionBulk);
	_globalJacDisc = _globalJac;

	// ConvectionDispersionOperator tells us whether flow direction has changed
	if (!_convDispOp.notifyDiscontinuousSectionTransition(t, secIdx, _jacInlet)) {
		// (re)compute DG particle Jacobian blocks (can only be done after notify)
		updateSection(secIdx);
		_disc.initializeDGjac(_parGeomSurfToVol);
		return;
	}
	else {
		// (re)compute DG particle Jacobian blocks
		updateSection(secIdx);
		_disc.initializeDGjac(_parGeomSurfToVol);
	}
}

void GeneralRateModelDG::setFlowRates(active const* in, active const* out) CADET_NOEXCEPT
{
	_convDispOp.setFlowRates(in[0], out[0], _colPorosity);
}

void GeneralRateModelDG::reportSolution(ISolutionRecorder& recorder, double const* const solution) const
{
	Exporter expr(_disc, *this, solution);
	recorder.beginUnitOperation(_unitOpIdx, *this, expr);
	recorder.endUnitOperation();
}

void GeneralRateModelDG::reportSolutionStructure(ISolutionRecorder& recorder) const
{
	Exporter expr(_disc, *this, nullptr);
	recorder.unitOperationStructure(_unitOpIdx, *this, expr);
}

unsigned int GeneralRateModelDG::requiredADdirs() const CADET_NOEXCEPT
{
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	return _jacobianAdDirs;
#else
	// If CADET_CHECK_ANALYTIC_JACOBIAN is active, we always need the AD directions for the Jacobian
	return numAdDirsForJacobian();
#endif
}

void GeneralRateModelDG::prepareADvectors(const AdJacobianParams& adJac) const
{
	// Early out if AD is disabled
	if (!adJac.adY)
		return;

	Indexer idxr(_disc);

	// The global DG Jacobian is banded around the main diagonal and has additional (also banded, but offset) entries for film diffusion,
	// i.e. banded AD vector seeding is not sufficient (as it is for the FV Jacobians, see @puttmann2016 and the DG LRM Jacobian).
	// The compressed vectorial AD seeding and Jacobian construction is described in the following.
	// The global DG Jacobian is banded around the main diagonal and has additional (also banded) entries for film diffusion.
	// To feasibly seed and reconstruct the Jacobian (we need information for decompression), we create dedicated active directions for
	// the bulk and each particle type. Particle AD directions are treated as dense (per particle block) since only nCells > 6 would
	// justify band compression, which rarely ever happens.

	// We begin by seeding the (banded around main diagonal) bulk Jacobian block
	// We have differing Jacobian structures for exact integration and collocation DG scheme, i.e. we need different seed vectors
	// collocation DG: 2 * N_n * (N_c + N_q) + 1 = total bandwidth (main diagonal entries maximally depend on the next and last N_n liquid phase entries of same component)
	//    ex. int. DG: 4 * N_n * (N_c + N_q) + 1 = total bandwidth (main diagonal entries maximally depend on the next and last 2*N_n liquid phase entries of same component)
	const int lowerBandwidth = (_disc.exactInt) ? 2 * _disc.nNodes * idxr.strideColNode() : _disc.nNodes * idxr.strideColNode();
	const int upperBandwidth = lowerBandwidth;
	const int bulkRows = idxr.offsetCp() - idxr.offsetC();
	ad::prepareAdVectorSeedsForBandMatrix(adJac.adY + _disc.nComp, adJac.adDirOffset, bulkRows, lowerBandwidth, upperBandwidth, lowerBandwidth);

	// We now seed the particle Jacobian blocks using the individual AD directions for each particle type.
	unsigned int adDirOffset = adJac.adDirOffset + _convDispOp.requiredADdirs();

	for (unsigned int type = 0; type < _disc.nParType; type++)
	{
		for (unsigned int parBlock = 0; parBlock < _disc.nPoints; parBlock++)
		{
			// move adVec pointer to start of current particle block 
			active* _adVec = adJac.adY + idxr.offsetCp(ParticleTypeIndex{ type }, ParticleIndex{ parBlock });

			for (int eq = 0; eq < idxr.strideParBlock(type); ++eq)
			{
				// Clear previously set directions
				_adVec[eq].fillADValue(adJac.adDirOffset, 0.0);
				// Set direction
				_adVec[eq].setADValue(adDirOffset + eq, 1.0);

			}
		}
		if (type < _disc.nParType - 1u) // move to dedicated DoFs of next particle type
			adDirOffset += idxr.strideParBlock(type);
	}
}
/**
 * @brief Extracts the system Jacobian from band compressed AD seed vectors
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
void GeneralRateModelDG::extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset)
{
	Indexer idxr(_disc);

	const active* const adVec = adRes + idxr.offsetC();

	/* Extract bulk phase equations entries */
	const int lowerBandwidth = (_disc.exactInt) ? 2 * _disc.nNodes * idxr.strideColNode() : _disc.nNodes * idxr.strideColNode();
	const int upperBandwidth = lowerBandwidth;
	const int stride = lowerBandwidth + 1 + upperBandwidth;
	int diagDir = lowerBandwidth;
	const int bulkDoFs = idxr.offsetCp() - idxr.offsetC();
	const int eqOffset = 0;
	const int matOffset = idxr.offsetC();
	ad::extractBandedBlockEigenJacobianFromAd(adVec, adDirOffset, diagDir, lowerBandwidth, upperBandwidth, eqOffset, bulkDoFs, _globalJac, matOffset);

	/* Handle particle liquid and solid phase equations entries */
	// Read particle Jacobian entries from dedicated AD directions
	int offsetParticleTypeDirs = adDirOffset + _convDispOp.requiredADdirs();

	for (unsigned int type = 0; type < _disc.nParType; type++)
	{
		for (unsigned int par = 0; par < _disc.nPoints; par++)
		{
			const int eqOffset_res = idxr.offsetCp(ParticleTypeIndex{ type }, ParticleIndex{ par });
			const int eqOffset_mat = idxr.offsetCp(ParticleTypeIndex{ type }, ParticleIndex{ par });
			for (unsigned int phase = 0; phase < idxr.strideParBlock(type); phase++)
			{
				for (unsigned int phaseTo = 0; phaseTo < idxr.strideParBlock(type); phaseTo++)
				{
					_globalJac.coeffRef(eqOffset_mat + phase, eqOffset_mat + phaseTo) = adRes[eqOffset_res + phase].getADValue(offsetParticleTypeDirs + phaseTo);
				}
			}
		}
		offsetParticleTypeDirs += idxr.strideParBlock(type);
	}

	/* Add analytically derived flux entries (only those that are part of the outlier bands) */
	// todo extract these entries instead of analytical calculation?
	calcFluxJacobians(_disc.curSection, true);
}

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN

/**
 * @brief Compares the analytical Jacobian with a Jacobian derived by AD
 * @details The analytical Jacobian is assumed to be stored in the corresponding band matrices.
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
void GeneralRateModelDG::checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const
{
	// todo write this function
	//Indexer idxr(_disc);

	//LOG(Debug) << "AD dir offset: " << adDirOffset << " DiagDirCol: " << _convDispOp.jacobian().lowerBandwidth() << " DiagDirPar: " << _jacP[0].lowerBandwidth();

	//// Column
	//const double maxDiffCol = _convDispOp.checkAnalyticJacobianAgainstAd(adRes, adDirOffset);

	//// Particles
	//double maxDiffPar = 0.0;
	//for (unsigned int type = 0; type < _disc.nParType; ++type)
	//{
	//	for (unsigned int pblk = 0; pblk < _disc.nPoints; ++pblk)
	//	{
	//		linalg::BandMatrix& jacMat = _jacP[_disc.nPoints * type + pblk];
	//		const double localDiff = ad::compareBandedJacobianWithAd(adRes + idxr.offsetCp(ParticleTypeIndex{type}, ParticleIndex{pblk}), adDirOffset, jacMat.lowerBandwidth(), jacMat);
	//		LOG(Debug) << "-> Par type " << type << " block " << pblk << " diff: " << localDiff;
	//		maxDiffPar = std::max(maxDiffPar, localDiff);
	//	}
	//}
}

#endif

int GeneralRateModelDG::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidual);

	// Evaluate residual do not compute Jacobian or parameter sensitivities
	return residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem);
}

int GeneralRateModelDG::residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidual);

	//FDJac = calcFDJacobian(static_cast<const double*>(simState.vecStateY), static_cast<const double*>(simState.vecStateYdot), simTime, threadLocalMem, 2.0); // todo delete

	// Evaluate residual, use AD for Jacobian if required but do not evaluate parameter derivatives
	return residual(simTime, simState, res, adJac, threadLocalMem, true, false);
}

int GeneralRateModelDG::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res,
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

template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int GeneralRateModelDG::residualImpl(double t, unsigned int secIdx, StateType const* const y, double const* const yDot, ResidualType* const res, util::ThreadLocalStorage& threadLocalMem)
{
	
	// determine wether we have a section switch. If so, set velocity, dispersion, newStaticJac
	updateSection(secIdx);

	double* const resPtr = reinterpret_cast<double* const>(res);
	Eigen::Map<Eigen::VectorXd> resi(resPtr, numDofs());
	resi.setZero();


	if (wantJac && _disc.newStaticJac) {

		// estimate new static (per section) jacobian
		bool success = calcStaticAnaJacobian_GRM(secIdx);

		_disc.newStaticJac = false;

		if (cadet_unlikely(!success)) {
			LOG(Error) << "Jacobian pattern did not fit the Jacobian estimation";
		}
	}

	residualBulk<StateType, ResidualType, ParamType, wantJac>(t, secIdx, y, yDot, res, threadLocalMem);

	BENCH_START(_timerResidualPar);

	for (unsigned int pblk = 0; pblk < _disc.nPoints * _disc.nParType; ++pblk)
	{
		const unsigned int parType = pblk / _disc.nPoints;
		const unsigned int par = pblk % _disc.nPoints;
		residualParticle<StateType, ResidualType, ParamType, wantJac>(t, parType, par, secIdx, y, yDot, res, threadLocalMem);
	}

	// we need to add the DG discretized solid entries of the jacobian that get overwritten by the binding kernel.
	// These entries only exist for the GRM with surface diffusion
	if (wantJac) {
		for (unsigned int parType = 0; parType < _disc.nParType; parType++) {
			if (_binding[parType]->hasDynamicReactions() && _hasSurfaceDiffusion[parType]) {
				active const* const _parSurfDiff = getSectionDependentSlice(_parSurfDiffusion, _disc.strideBound[_disc.nParType], secIdx) + _disc.nBoundBeforeType[parType];
				addSolidDGentries(parType, _parSurfDiff);
			}
		}
	}

	BENCH_STOP(_timerResidualPar);

	residualFlux<StateType, ResidualType, ParamType>(t, secIdx, y, yDot, res);

	// Handle inlet DOFs, which are simply copied to the residual
	for (unsigned int i = 0; i < _disc.nComp; ++i)
	{
		res[i] = y[i];
	}

	return 0;
}

template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int GeneralRateModelDG::residualBulk(double t, unsigned int secIdx, StateType const* yBase, double const* yDotBase, ResidualType* resBase, util::ThreadLocalStorage& threadLocalMem)
{
	_convDispOp.residual(*this, t, secIdx, yBase, yDotBase, resBase, typename cadet::ParamSens<ParamType>::enabled());

	if (!_dynReactionBulk || (_dynReactionBulk->numReactionsLiquid() == 0))
		return 0;

	Indexer idxr(_disc);

	// Dynamic reactions
	if (_dynReactionBulk) {
		// Get offsets
		StateType const* y = yBase + idxr.offsetC();
		ResidualType* res = resBase + idxr.offsetC();
		LinearBufferAllocator tlmAlloc = threadLocalMem.get();

		for (unsigned int col = 0; col < _disc.nPoints; ++col, y += idxr.strideColNode(), res += idxr.strideColNode())
		{
			const ColumnPosition colPos{ (0.5 + static_cast<double>(col)) / static_cast<double>(_disc.nPoints), 0.0, 0.0 };
			_dynReactionBulk->residualLiquidAdd(t, secIdx, colPos, y, res, -1.0, tlmAlloc);

			if (wantJac)
			{
				linalg::BandedEigenSparseRowIterator jac(_globalJacDisc, col * idxr.strideColNode());
				// static_cast should be sufficient here, but this statement is also analyzed when wantJac = false
				_dynReactionBulk->analyticJacobianLiquidAdd(t, secIdx, colPos, reinterpret_cast<double const*>(y), -1.0, jac, tlmAlloc);
			}
		}
	}

	return 0;
}

template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int GeneralRateModelDG::residualParticle(double t, unsigned int parType, unsigned int colNode, unsigned int secIdx, StateType const* yBase,
	double const* yDotBase, ResidualType* resBase, util::ThreadLocalStorage& threadLocalMem)
{
	Indexer idxr(_disc);

	LinearBufferAllocator tlmAlloc = threadLocalMem.get();

	// Special case: individual treatment of time derivatives in particle mass balance at inner particle boundary node
	bool specialCase = !_disc.parExactInt[parType] && (_parGeomSurfToVol[parType] != _disc.SurfVolRatioSlab && _parCoreRadius[parType] == 0.0);

	// Prepare parameters
	active const* const parDiff = getSectionDependentSlice(_parDiffusion, _disc.nComp * _disc.nParType, secIdx) + parType * _disc.nComp;

	// Ordering of particle surface diffusion:
	// bnd0comp0, bnd0comp1, bnd0comp2, bnd1comp0, bnd1comp1, bnd1comp2
	active const* const parSurfDiff = getSectionDependentSlice(_parSurfDiffusion, _disc.strideBound[_disc.nParType], secIdx) + _disc.nBoundBeforeType[parType];

	// z coordinate (column length normed to 1) of current node - needed in externally dependent adsorption kinetic
	const double z = _convDispOp.relativeCoordinate(colNode);

	// The RowIterator is always centered on the main diagonal.
	// This means that jac[0] is the main diagonal, jac[-1] is the first lower diagonal,
	// and jac[1] is the first upper diagonal. We can also access the rows from left to
	// right beginning with the last lower diagonal moving towards the main diagonal and
	// continuing to the last upper diagonal by using the native() method.
	linalg::BandedEigenSparseRowIterator jac(_globalJac, idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode }));

	int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();
	const parts::cell::CellParameters cellResParams = makeCellResidualParams(parType, qsReaction);

	// Handle time derivatives, binding, dynamic reactions: residualKernel computes discrete point wise,
	// so we loop over each discrete particle point
	for (unsigned int par = 0; par < _disc.nParPoints[parType]; ++par)
	{
		int cell = std::floor(par / _disc.nParNode[parType]);
		// local Pointers to current particle node, needed in residualKernel
		StateType const* local_y = yBase + idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode }) + par * idxr.strideParNode(parType);
		double const* local_yDot = yDotBase ? yDotBase + idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode }) + par * idxr.strideParNode(parType) : nullptr;
		ResidualType* local_res = resBase + idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode }) + par * idxr.strideParNode(parType);

		// r (particle) coordinate of current node (particle radius normed to 1) - needed in externally dependent adsorption kinetic
		const double r = static_cast<double>(
			(_disc.deltaR[_disc.offsetMetric[parType] + cell] * cell
			+ 0.5 * _disc.deltaR[_disc.offsetMetric[parType] + cell] * (1 + _disc.parNodes[parType][par % _disc.nParNode[parType]]))
			/ (_parRadius[parType] - _parCoreRadius[parType])
			);
		const ColumnPosition colPos{ z, 0.0, r };

		// Handle time derivatives, binding, dynamic reactions.
		// Special case: Dont add time derivatives to inner boundary node for DG discretized mass balance equations.
		// This can be achieved by setting yDot pointer to null before passing to residual kernel, and adding only the time derivative for dynamic binding
		// TODO Check Treatment of reactions (do we need yDot then?)
		if (cadet_unlikely(par == 0 && specialCase))
		{
			parts::cell::residualKernel<StateType, ResidualType, ParamType, parts::cell::CellParameters, linalg::BandedEigenSparseRowIterator, wantJac, true>(
				t, secIdx, colPos, local_y, nullptr, local_res, jac, cellResParams, tlmAlloc // TODO Check Treatment of reactions (do we need yDot then?)
				);

			if (cellResParams.binding->hasDynamicReactions() && local_yDot)
			{
				unsigned int idx = 0;
				for (unsigned int comp = 0; comp < cellResParams.nComp; ++comp)
				{
					for (unsigned int state = 0; state < cellResParams.nBound[comp]; ++state, ++idx)
					{
						// Skip quasi-stationary fluxes
						if (cellResParams.qsReaction[idx])
							continue;

						// For kinetic bindings and surface diffusion, we have an additional DG-discretized mass balance eq.
						// -> add time derivate at inner bonudary node only without surface diffusion 
						else if (_hasSurfaceDiffusion[parType])
							continue;
						// Some bound states might still not be effected by surface diffusion
						else if (parSurfDiff[idx] != 0.0)
							continue;

						// Add time derivative to solid phase
						local_res[idxr.strideParLiquid() + idx] += local_yDot[idxr.strideParLiquid() + idx];
					}
				}
			}
		}
		else
		{
			parts::cell::residualKernel<StateType, ResidualType, ParamType, parts::cell::CellParameters, linalg::BandedEigenSparseRowIterator, wantJac, true>(
				t, secIdx, colPos, local_y, local_yDot, local_res, jac, cellResParams, tlmAlloc
				);
		}

		// Move rowiterator to next particle node
		jac += idxr.strideParNode(parType);
	}

	// We still need to handle transport/diffusion

	// Get pointers to the particle block of the current column node, particle type
	const StateType* c_p = yBase + idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });
	ResidualType* resC_p = resBase + idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });

	// Mobile phase RHS

	// Get film diffusion flux at current node to compute boundary condition
	active const* const filmDiff = getSectionDependentSlice(_filmDiffusion, _disc.nComp * _disc.nParType, secIdx) + parType * _disc.nComp;
	for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
		_disc.localFlux[comp] = filmDiff[comp] * (yBase[idxr.offsetC() + colNode * idxr.strideColNode() + comp]
			                  - yBase[idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode }) + (_disc.nParPoints[parType] - 1) * idxr.strideParNode(parType) + comp]);
	}

	const int nNodes = _disc.nParNode[parType];
	const int nCells = _disc.nParCell[parType];
	const int nPoints = _disc.nParPoints[parType];
	const int nComp = _disc.nComp;

	int strideParLiquid = idxr.strideParLiquid();
	int strideParNode = idxr.strideParNode(parType);

	if (_disc.parGSM[parType]) // GSM implementation
	{
		for (unsigned int comp = 0; comp < nComp; comp++)
		{

			// ====================================================================================//
			// solve GSM-discretized particle mass balance   									   //
			// ====================================================================================//

			Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>> resCp(resC_p + comp, nPoints, InnerStride<Dynamic>(idxr.strideParNode(parType)));
			Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>> Cp(c_p + comp, nPoints, InnerStride<Dynamic>(idxr.strideParNode(parType)));

			// Use auxiliary variable to get c^p + \sum 1 / \Beta_p c^s
			Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<>> sum_cp_cs(reinterpret_cast<ResidualType*>(&_disc.g_pSum[parType][0]), nPoints, InnerStride<>(1));
			sum_cp_cs = static_cast<ParamType>(parDiff[comp]) * Cp.template cast<ResidualType>();

			for (int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++)
			{
				if (parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) // some bound states might still not be effected by surface diffusion
				{
					Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>> c_s(c_p + _disc.nComp + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd, nPoints, InnerStride<Dynamic>(idxr.strideParNode(parType)));
					ParamType invBetaP = (1.0 - static_cast<ParamType>(_parPorosity[parType])) / (static_cast<ParamType>(_poreAccessFactor[_disc.nComp * parType + comp]) * static_cast<ParamType>(_parPorosity[parType]));
					sum_cp_cs += invBetaP * static_cast<ParamType>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) * c_s;

					/* For kinetic bindings with surface diffusion: add the additional DG-discretized particle mass balance equations to residual */

					if (!qsReaction[bnd])
					{
						// Eigen access to current bound state residual
						Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>> resCs(resBase + idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode }) + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd,
							nPoints, InnerStride<Dynamic>(idxr.strideParNode(parType)));

						// Use auxiliary variable to get \Beta_p D_s c^s
						Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<>> c_s_modified(reinterpret_cast<ResidualType*>(&_disc.g_p[parType][0]), nPoints, InnerStride<>(1));

						// Apply squared inverse mapping and surface diffusion
						c_s_modified = 2.0 / static_cast<ParamType>(_disc.deltaR[_disc.offsetMetric[parType]]) * 2.0 / static_cast<ParamType>(_disc.deltaR[_disc.offsetMetric[parType]]) *
							static_cast<ParamType>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) * c_s;

						Eigen::Map<const Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>> c_s_modified_const(&c_s_modified[0], nPoints, InnerStride<Dynamic>(1));
						parGSMVolumeIntegral<ResidualType, ResidualType>(parType, c_s_modified_const, resCs);

						// Leave out the surface integral as we only have one element, i.e. we apply BC with zeros
					}
				}
			}

			// Apply squared inverse mapping
			sum_cp_cs *= 2.0 / static_cast<ParamType>(_disc.deltaR[_disc.offsetMetric[parType]]) * 2.0 / static_cast<ParamType>(_disc.deltaR[_disc.offsetMetric[parType]]);

			Eigen::Map<const Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>> sum_cp_cs_const(&sum_cp_cs[0], nPoints, InnerStride<Dynamic>(1));
			parGSMVolumeIntegral<ResidualType, ResidualType>(parType, sum_cp_cs_const, resCp);

			// Pass sum_cp_cs_const to match the DGSEM interface; nullptr might also be feasible
			parSurfaceIntegral<ResidualType, ResidualType>(parType, sum_cp_cs_const, resCp, nNodes, 1u, false, comp);

		}
	}
	else // DGSEM implementation
	{
		for (unsigned int comp = 0; comp < nComp; comp++)
		{
			// Component dependent (through access factor) inverse Beta_P
			ParamType invBetaP = (1.0 - static_cast<ParamType>(_parPorosity[parType])) / (static_cast<ParamType>(_poreAccessFactor[_disc.nComp * parType + comp]) * static_cast<ParamType>(_parPorosity[parType]));

			// =====================================================================================================//
			// Solve auxiliary systems  d_p g_p + d_s beta_p sum g_s= d (d_p c_p + d_s beta_p sum c_s) / d xi		//
			// =====================================================================================================//
			// Component-wise! strides
			unsigned int strideCell = nNodes;
			unsigned int strideNode = 1u;
			// Reset cache for auxiliary variable
			Eigen::Map<Vector<StateType, Dynamic>, 0, InnerStride<>> _g_p(reinterpret_cast<StateType*>(&_disc.g_p[parType][0]), nPoints, InnerStride<>(1));
			Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<>> _g_pSum(reinterpret_cast<ResidualType*>(&_disc.g_pSum[parType][0]), nPoints, InnerStride<>(1));
			_g_p.setZero();
			_g_pSum.setZero();

			Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>> cp(c_p + comp, _disc.nParPoints[parType], InnerStride<Dynamic>(idxr.strideParNode(parType)));

			// Handle surface diffusion: Compute auxiliary variable; For kinetic bindings: add additional mass balance to residual of respective bound state
			if (_hasSurfaceDiffusion[parType])
			{
				for (int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++)
				{
					if (parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) // some bound states might still not be effected by surface diffusion
					{
						// Get solid phase vector
						Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>> c_s(c_p + strideParLiquid + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd,
							nPoints, InnerStride<Dynamic>(strideParNode));
						// Compute g_s = d c_s / d xi
						solve_auxiliary_DG<StateType>(parType, c_s, strideCell, strideNode, comp);
						// Apply invBeta_p, d_s and add to sum -> gSum += d_s * invBeta_p * (D c - M^-1 B [c - c^*])
						_g_pSum += _g_p.template cast<ResidualType>() * invBetaP * static_cast<ParamType>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]);

						/* For kinetic bindings with surface diffusion: add the additional DG-discretized particle mass balance equations to residual */

						if (!qsReaction[bnd])
						{
							// Eigen access to current bound state residual
							Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>> resCs(resBase + idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode }) + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd,
								nPoints, InnerStride<Dynamic>(idxr.strideParNode(parType)));

							// Promote auxiliary variable storage from double to active if required
							// @todo is there a more efficient or elegant solution?
							if (std::is_same<ResidualType, active>::value && std::is_same<StateType, double>::value)
								vectorPromoter(reinterpret_cast<double*>(&_g_p[0]), nPoints); // reinterpret_cast only required because statement is scanned also when StateType != double

							// Access auxiliary variable as ResidualType
							Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<>> _g_p_ResType(reinterpret_cast<ResidualType*>(&_disc.g_p[parType][0]), nPoints, InnerStride<>(1));

							applyParInvMap<ResidualType, ParamType>(_g_p_ResType, parType);
							_g_p_ResType *= static_cast<ParamType>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]);

							// Eigen access to auxiliary variable of current bound state
							Eigen::Map<const Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>> _g_p_ResType_const(&_g_p_ResType[0], nPoints, InnerStride<Dynamic>(1));

							// Add - D_r * gs to the residual, including metric part.->res = invMap^2* [ -D_r * (d_s c^s) ]
							parVolumeIntegral<ResidualType, ResidualType>(parType, false, _g_p_ResType_const, resCs);

							// Add M^-1 B (gs - gs^*) to the residual -> res =  invMap^2 * [ - D_r * (d_s c^s) + M^-1 B (gs - gs^*) ]
							parSurfaceIntegral<ResidualType, ResidualType>(parType, _g_p_ResType_const, resCs, strideCell, strideNode, false, comp, true);
						}
					}
				}
			}

			// Compute g_p = d c_p / d xi
			solve_auxiliary_DG<StateType>(parType, cp, strideCell, strideNode, comp);

			// Add particle diffusion part to auxiliary variable sum -> gSum += d_p * (D c - M^-1 B [c - c^*])
			_g_pSum += _g_p * static_cast<ParamType>(parDiff[comp]);

			// apply squared inverse mapping to sum of bound state auxiliary variables -> gSum = - invMap^2 * (d_p * c^p + sum_mi d_s invBeta_p c^s)
			applyParInvMap<ResidualType, ParamType>(_g_pSum, parType);

			// ====================================================================================//
			// solve DG-discretized particle mass balance   									     //
			// ====================================================================================//

			  /* Solve DG-discretized particle mass balance equation */

			Eigen::Map<const Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>> _g_pSum_const(&_g_pSum[0], nPoints, InnerStride<Dynamic>(1));

			// Eigen access to particle liquid residual
			Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>> resCp(resC_p + comp, nPoints, InnerStride<Dynamic>(idxr.strideParNode(parType)));

			// Add - D_r * (g_sum) to the residual, including metric part. -> res = - D_r * (d_p * c^p + invBeta_p sum_mi d_s c^s)
			parVolumeIntegral<ResidualType, ResidualType>(parType, false, _g_pSum_const, resCp);

			// Add M^-1 B (g_sum - g_sum^*) to the residual -> res = - D_r * (d_p * c^p + invBeta_p sum_mi d_s c^s) + M^-1 B (g_sum - g_sum^*)
			parSurfaceIntegral<ResidualType, ResidualType>(parType, _g_pSum_const, resCp, strideCell, strideNode, false, comp);

		}
	}

	return 0;
}

template <typename StateType, typename ResidualType, typename ParamType>
int GeneralRateModelDG::residualFlux(double t, unsigned int secIdx, StateType const* yBase, double const* yDotBase, ResidualType* resBase)
{
	Indexer idxr(_disc);

	const ParamType invBetaC = 1.0 / static_cast<ParamType>(_colPorosity) - 1.0;

	// Get offsets
	ResidualType* const resCol = resBase + idxr.offsetC();
	StateType const* const yCol = yBase + idxr.offsetC();

	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		ResidualType* const resParType = resBase + idxr.offsetCp(ParticleTypeIndex{type});
		StateType const* const yParType = yBase + idxr.offsetCp(ParticleTypeIndex{type});

		const ParamType epsP = static_cast<ParamType>(_parPorosity[type]);

		// Ordering of diffusion:
		// sec0type0comp0, sec0type0comp1, sec0type0comp2, sec0type1comp0, sec0type1comp1, sec0type1comp2,
		// sec1type0comp0, sec1type0comp1, sec1type0comp2, sec1type1comp0, sec1type1comp1, sec1type1comp2, ...
		active const* const filmDiff = getSectionDependentSlice(_filmDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;
		active const* const parDiff = getSectionDependentSlice(_parDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;

		const ParamType surfaceToVolumeRatio = _parGeomSurfToVol[type] / static_cast<ParamType>(_parRadius[type]);

		const ParamType jacCF_val = invBetaC * surfaceToVolumeRatio;
		const ParamType jacPF_val = -1.0 / epsP;

		// Add flux to column void / bulk volume
		for (unsigned int i = 0; i < _disc.nPoints * _disc.nComp; ++i)
		{
			const unsigned int colNode = i / _disc.nComp;
			const unsigned int comp = i - colNode * _disc.nComp;
			// + 1/Beta_c * (surfaceToVolumeRatio_{p,j}) * d_j * (k_f * [c_l - c_p])
			resCol[i] += static_cast<ParamType>(filmDiff[comp]) * jacCF_val * static_cast<ParamType>(_parTypeVolFrac[type + colNode * _disc.nParType])
				        * (yCol[i] - yParType[colNode * idxr.strideParBlock(type) + (_disc.nParPoints[type] - 1) * idxr.strideParNode(type) + comp]);
		}

		//  Bead boundary condition is computed in residualParticle().

	}

	return 0;
}

parts::cell::CellParameters GeneralRateModelDG::makeCellResidualParams(unsigned int parType, int const* qsReaction) const
{
	return parts::cell::CellParameters
		{
			_disc.nComp,
			_disc.nBound + _disc.nComp * parType,
			_disc.boundOffset + _disc.nComp * parType,
			_disc.strideBound[parType],
			qsReaction,
			_parPorosity[parType],
			_poreAccessFactor.data() + _disc.nComp * parType,
			_binding[parType],
			(_dynReaction[parType] && (_dynReaction[parType]->numReactionsCombined() > 0)) ? _dynReaction[parType] : nullptr
		};
}

int GeneralRateModelDG::residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidualSens);

	// Evaluate residual for all parameters using AD in vector mode and at the same time update the
	// Jacobian (in one AD run, if analytic Jacobians are disabled)
	return residual(simTime, simState, nullptr, adJac, threadLocalMem, true, true);
}

int GeneralRateModelDG::residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidualSens);

	// Evaluate residual for all parameters using AD in vector mode
	return residualImpl<double, active, active, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, adRes, threadLocalMem);
}

int GeneralRateModelDG::residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState,
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
void GeneralRateModelDG::multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret)
{
	Indexer idxr(_disc);

	// Handle identity matrix of inlet DOFs
	for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
	{
		ret[comp] = alpha * yS[comp] + beta * ret[comp];
	}

	// Main Jacobian
	Eigen::Map<Eigen::VectorXd> ret_vec(ret + idxr.offsetC(), numPureDofs());
	Eigen::Map<const Eigen::VectorXd> yS_vec(yS + idxr.offsetC(), numPureDofs());
	ret_vec = alpha * _globalJac.block(idxr.offsetC(), idxr.offsetC(), numPureDofs(), numPureDofs()) * yS_vec + beta * ret_vec;

	// Map inlet DOFs to the column inlet (first bulk cells)
	// Inlet at z = 0 for forward flow, at z = L for backward flow.
	unsigned int offInlet = _convDispOp.forwardFlow() ? 0 : (_disc.nCol - 1u) * idxr.strideColCell();

	for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
		for (unsigned int node = 0; node < (_disc.exactInt ? _disc.nNodes : 1); node++) {
			ret[idxr.offsetC() + offInlet + comp * idxr.strideColComp() + node * idxr.strideColNode()] += alpha * _jacInlet(node, 0) * yS[comp];
		}
	}
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
void GeneralRateModelDG::multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret)
{
	Indexer idxr(_disc);

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(_disc.nPoints * _disc.nParType + 1), [&](std::size_t idx)
#else
	for (unsigned int idx = 0; idx < _disc.nPoints * _disc.nParType + 1; ++idx)
#endif
	{
		if (cadet_unlikely(idx == 0))
		{
			_convDispOp.multiplyWithDerivativeJacobian(simTime, sDot, ret);
		}
		else
		{
			const unsigned int idxParLoop = idx - 1;
			const unsigned int pblk = idxParLoop % _disc.nPoints;
			const unsigned int type = idxParLoop / _disc.nPoints;

			const double invBetaP = (1.0 / static_cast<double>(_parPorosity[type]) - 1.0);
			unsigned int const* const nBound = _disc.nBound + type * _disc.nComp;
			unsigned int const* const boundOffset = _disc.boundOffset + type * _disc.nComp;
			int const* const qsReaction = _binding[type]->reactionQuasiStationarity();

			// Particle shells
			const int offsetCpType = idxr.offsetCp(ParticleTypeIndex{ type }, ParticleIndex{ pblk });
			for (unsigned int shell = 0; shell < _disc.nParPoints[type]; ++shell)
			{
				const int offsetCpShell = offsetCpType + shell * idxr.strideParNode(type);
				double const* const mobileSdot = sDot + offsetCpShell;
				double* const mobileRet = ret + offsetCpShell;

				parts::cell::multiplyWithDerivativeJacobianKernel<true>(mobileSdot, mobileRet, _disc.nComp, nBound, boundOffset, _disc.strideBound[type], qsReaction, 1.0, invBetaP);
			}
		}
	} CADET_PARFOR_END;

	// Handle inlet DOFs (all algebraic)
	std::fill_n(ret, _disc.nComp, 0.0);
}

void GeneralRateModelDG::setExternalFunctions(IExternalFunction** extFuns, unsigned int size)
{
	for (IBindingModel* bm : _binding)
	{
		if (bm)
			bm->setExternalFunctions(extFuns, size);
	}
}

unsigned int GeneralRateModelDG::localOutletComponentIndex(unsigned int port) const CADET_NOEXCEPT
{
	// Inlets are duplicated so need to be accounted for
	if (_convDispOp.forwardFlow())
		// Forward Flow: outlet is last cell
		return _disc.nComp + (_disc.nPoints - 1) * _disc.nComp;
	else
		// Backward flow: Outlet is first cell
		return _disc.nComp;
}

unsigned int GeneralRateModelDG::localInletComponentIndex(unsigned int port) const CADET_NOEXCEPT
{
	// Always 0 due to dedicated inlet DOFs
	return 0;
}

unsigned int GeneralRateModelDG::localOutletComponentStride(unsigned int port) const CADET_NOEXCEPT
{
	return 1;
}

unsigned int GeneralRateModelDG::localInletComponentStride(unsigned int port) const CADET_NOEXCEPT
{
	return 1;
}

void GeneralRateModelDG::expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut)
{
	// @todo Write this function
}

void GeneralRateModelDG::setEquidistantRadialDisc(unsigned int parType)
{
	active* const ptrCenterRadius = _parCenterRadius.data() + _disc.offsetMetric[parType];
	active* const ptrOuterSurfAreaPerVolume = _parOuterSurfAreaPerVolume.data() + _disc.offsetMetric[parType];
	active* const ptrInnerSurfAreaPerVolume = _parInnerSurfAreaPerVolume.data() + _disc.offsetMetric[parType];

	const active radius = _parRadius[parType] - _parCoreRadius[parType];
	const active dr = radius / static_cast<double>(_disc.nParCell[parType]);
	std::fill(_parCellSize.data() + _disc.offsetMetric[parType], _parCellSize.data() + _disc.offsetMetric[parType] + _disc.nParCell[parType], dr);

	if (_parGeomSurfToVol[parType] == SurfVolRatioSphere)
	{
		for (unsigned int cell = 0; cell < _disc.nParCell[parType]; ++cell)
		{
			const active r_out = _parRadius[parType] - static_cast<double>(cell) * dr;
			const active r_in = _parRadius[parType] - static_cast<double>(cell + 1) * dr;

			ptrCenterRadius[cell] = _parRadius[parType] - (0.5 + static_cast<double>(cell)) * dr;

			// Compute denominator -> corresponding to cell volume
			const active vol = pow(r_out, 3.0) - pow(r_in, 3.0);

			ptrOuterSurfAreaPerVolume[cell] = 3.0 * sqr(r_out) / vol;
			ptrInnerSurfAreaPerVolume[cell] = 3.0 * sqr(r_in) / vol;
		}
	}
	else if (_parGeomSurfToVol[parType] == SurfVolRatioCylinder)
	{
		for (unsigned int cell = 0; cell < _disc.nParCell[parType]; ++cell)
		{
			const active r_out = _parRadius[parType] - static_cast<double>(cell) * dr;
			const active r_in = _parRadius[parType] - static_cast<double>(cell + 1) * dr;

			ptrCenterRadius[cell] = _parRadius[parType] - (0.5 + static_cast<double>(cell)) * dr;

			// Compute denominator -> corresponding to cell volume
			const active vol = sqr(r_out) - sqr(r_in);

			ptrOuterSurfAreaPerVolume[cell] = 2.0 * r_out / vol;
			ptrInnerSurfAreaPerVolume[cell] = 2.0 * r_in / vol;
		}
	}
	else if (_parGeomSurfToVol[parType] == SurfVolRatioSlab)
	{
		for (unsigned int cell = 0; cell < _disc.nParCell[parType]; ++cell)
		{
			const active r_out = _parRadius[parType] - static_cast<double>(cell) * dr;
			const active r_in = _parRadius[parType] - static_cast<double>(cell + 1) * dr;

			ptrCenterRadius[cell] = _parRadius[parType] - (0.5 + static_cast<double>(cell)) * dr;

			// Compute denominator -> corresponding to cell volume
			const active vol = r_out - r_in;

			ptrOuterSurfAreaPerVolume[cell] = 1.0 / vol;
			ptrInnerSurfAreaPerVolume[cell] = 1.0 / vol;
		}
	}
}
/**
 * @brief Computes the radial nodes in the beads in such a way that all shells have the same volume
 */
void GeneralRateModelDG::setEquivolumeRadialDisc(unsigned int parType)
{
	active* const ptrCellSize = _parCellSize.data() + _disc.offsetMetric[parType];
	active* const ptrCenterRadius = _parCenterRadius.data() + _disc.offsetMetric[parType];
	active* const ptrOuterSurfAreaPerVolume = _parOuterSurfAreaPerVolume.data() + _disc.offsetMetric[parType];
	active* const ptrInnerSurfAreaPerVolume = _parInnerSurfAreaPerVolume.data() + _disc.offsetMetric[parType];

	if (_parGeomSurfToVol[parType] == SurfVolRatioSphere)
	{
		active r_out = _parRadius[parType];
		active r_in = _parCoreRadius[parType];
		const active volumePerShell = (pow(_parRadius[parType], 3.0) - pow(_parCoreRadius[parType], 3.0)) / static_cast<double>(_disc.nParCell[parType]);

		for (unsigned int cell = 0; cell < _disc.nParCell[parType]; ++cell)
		{
			if (cell != (_disc.nParCell[parType] - 1))
				r_in = pow(pow(r_out, 3.0) - volumePerShell, (1.0 / 3.0));
			else
				r_in = _parCoreRadius[parType];

			ptrCellSize[cell] = r_out - r_in;
			ptrCenterRadius[cell] = (r_out + r_in) * 0.5;

			ptrOuterSurfAreaPerVolume[cell] = 3.0 * sqr(r_out) / volumePerShell;
			ptrInnerSurfAreaPerVolume[cell] = 3.0 * sqr(r_in) / volumePerShell;
			// Note that the DG particle shells are oppositely ordered compared to the FV particle shells
			_disc.deltaR[_disc.offsetMetric[parType] + _disc.nParCell[parType] - (cell + 1)] = r_out - r_in;

			// For the next cell: r_out == r_in of the current cell
			r_out = r_in;
		}
	}
	else if (_parGeomSurfToVol[parType] == SurfVolRatioCylinder)
	{
		active r_out = _parRadius[parType];
		active r_in = _parCoreRadius[parType];
		const active volumePerShell = (sqr(_parRadius[parType]) - sqr(_parCoreRadius[parType])) / static_cast<double>(_disc.nParCell[parType]);

		for (unsigned int cell = 0; cell < _disc.nParCell[parType]; ++cell)
		{
			if (cell != (_disc.nParCell[parType] - 1))
				r_in = sqrt(sqr(r_out) - volumePerShell);
			else
				r_in = _parCoreRadius[parType];

			ptrCellSize[cell] = r_out - r_in;
			ptrCenterRadius[cell] = (r_out + r_in) * 0.5;

			ptrOuterSurfAreaPerVolume[cell] = 2.0 * r_out / volumePerShell;
			ptrInnerSurfAreaPerVolume[cell] = 2.0 * r_in / volumePerShell;
			// Note that the DG particle shells are oppositely ordered compared to the FV particle shells
			_disc.deltaR[_disc.offsetMetric[parType] + _disc.nParCell[parType] - (cell + 1)] = r_out - r_in;

			// For the next cell: r_out == r_in of the current cell
			r_out = r_in;
		}
	}
	else if (_parGeomSurfToVol[parType] == SurfVolRatioSlab)
	{
		active r_out = _parRadius[parType];
		active r_in = _parCoreRadius[parType];
		const active volumePerShell = (_parRadius[parType] - _parCoreRadius[parType]) / static_cast<double>(_disc.nParCell[parType]);

		for (unsigned int cell = 0; cell < _disc.nParCell[parType]; ++cell)
		{
			if (cell != (_disc.nParCell[parType] - 1))
				r_in = r_out - volumePerShell;
			else
				r_in = _parCoreRadius[parType];

			ptrCellSize[cell] = r_out - r_in;
			ptrCenterRadius[cell] = (r_out + r_in) * 0.5;

			ptrOuterSurfAreaPerVolume[cell] = 1.0 / volumePerShell;
			ptrInnerSurfAreaPerVolume[cell] = 1.0 / volumePerShell;
			// Note that the DG particle shells are oppositely ordered compared to the FV particle shells
			_disc.deltaR[_disc.offsetMetric[parType] + _disc.nParCell[parType] - (cell + 1)] = r_out - r_in;

			// For the next cell: r_out == r_in of the current cell
			r_out = r_in;
		}
	}
}

/**
 * @brief Computes all helper quantities for radial bead discretization from given radial cell boundaries
 * @details Calculates surface areas per volume for every shell and the radial shell centers.
 */
void GeneralRateModelDG::setUserdefinedRadialDisc(unsigned int parType)
{
	active* const ptrCellSize = _parCellSize.data() + _disc.offsetMetric[parType];
	active* const ptrCenterRadius = _parCenterRadius.data() + _disc.offsetMetric[parType];
	active* const ptrOuterSurfAreaPerVolume = _parOuterSurfAreaPerVolume.data() + _disc.offsetMetric[parType];
	active* const ptrInnerSurfAreaPerVolume = _parInnerSurfAreaPerVolume.data() + _disc.offsetMetric[parType];

	// Care for the right ordering and include 0.0 / 1.0 if not already in the vector.
	std::vector<active> orderedInterfaces = std::vector<active>(_parDiscVector.begin() + _disc.offsetMetric[parType] + parType,
		_parDiscVector.begin() + _disc.offsetMetric[parType] + parType + _disc.nParCell[parType] + 1);

	// Sort in descending order
	std::sort(orderedInterfaces.begin(), orderedInterfaces.end(), std::greater<active>());

	// Force first and last element to be 1.0 and 0.0, respectively
	orderedInterfaces[0] = 1.0;
	orderedInterfaces.back() = 0.0;

	// Map [0, 1] -> [core radius, particle radius] via linear interpolation
	for (unsigned int cell = 0; cell < _disc.nParCell[parType]; ++cell)
		orderedInterfaces[cell] = static_cast<double>(orderedInterfaces[cell]) * (_parRadius[parType] - _parCoreRadius[parType]) + _parCoreRadius[parType];

	if (_parGeomSurfToVol[parType] == SurfVolRatioSphere)
	{
		for (unsigned int cell = 0; cell < _disc.nParCell[parType]; ++cell)
		{
			ptrCellSize[cell] = orderedInterfaces[cell] - orderedInterfaces[cell + 1];
			ptrCenterRadius[cell] = (orderedInterfaces[cell] + orderedInterfaces[cell + 1]) * 0.5;

			// Compute denominator -> corresponding to cell volume
			const active vol = pow(orderedInterfaces[cell], 3.0) - pow(orderedInterfaces[cell + 1], 3.0);

			ptrOuterSurfAreaPerVolume[cell] = 3.0 * sqr(orderedInterfaces[cell]) / vol;
			ptrInnerSurfAreaPerVolume[cell] = 3.0 * sqr(orderedInterfaces[cell + 1]) / vol;
			// Note that the DG particle shells are oppositely ordered compared to the FV particle shells
			_disc.deltaR[_disc.offsetMetric[parType] + _disc.nParCell[parType] - (cell + 1)] = ptrOuterSurfAreaPerVolume[cell] - ptrInnerSurfAreaPerVolume[cell];
		}
	}
	else if (_parGeomSurfToVol[parType] == SurfVolRatioCylinder)
	{
		for (unsigned int cell = 0; cell < _disc.nParCell[parType]; ++cell)
		{
			ptrCellSize[cell] = orderedInterfaces[cell] - orderedInterfaces[cell + 1];
			ptrCenterRadius[cell] = (orderedInterfaces[cell] + orderedInterfaces[cell + 1]) * 0.5;

			// Compute denominator -> corresponding to cell volume
			const active vol = sqr(orderedInterfaces[cell]) - sqr(orderedInterfaces[cell + 1]);

			ptrOuterSurfAreaPerVolume[cell] = 2.0 * orderedInterfaces[cell] / vol;
			ptrInnerSurfAreaPerVolume[cell] = 2.0 * orderedInterfaces[cell + 1] / vol;
			// Note that the DG particle shells are oppositely ordered compared to the FV particle shells
			_disc.deltaR[_disc.offsetMetric[parType] + _disc.nParCell[parType] - (cell + 1)] = ptrOuterSurfAreaPerVolume[cell] - ptrInnerSurfAreaPerVolume[cell];
		}
	}
	else if (_parGeomSurfToVol[parType] == SurfVolRatioSlab)
	{
		for (unsigned int cell = 0; cell < _disc.nParCell[parType]; ++cell)
		{
			ptrCellSize[cell] = orderedInterfaces[cell] - orderedInterfaces[cell + 1];
			ptrCenterRadius[cell] = (orderedInterfaces[cell] + orderedInterfaces[cell + 1]) * 0.5;

			// Compute denominator -> corresponding to cell volume
			const active vol = orderedInterfaces[cell] - orderedInterfaces[cell + 1];

			ptrOuterSurfAreaPerVolume[cell] = 1.0 / vol;
			ptrInnerSurfAreaPerVolume[cell] = 1.0 / vol;
			// Note that the DG particle shells are oppositely ordered compared to the FV particle shells
			_disc.deltaR[_disc.offsetMetric[parType] + _disc.nParCell[parType] - (cell + 1)] = ptrOuterSurfAreaPerVolume[cell] - ptrInnerSurfAreaPerVolume[cell];
		}
	}
}

// todo: parameter sensitivities for particle radius. Here, we have the problem that every DG operator becomes an active type.
// alternatively (only for exact integration), we could store more matrices and compute the metric dependend calculations in the residual.
// inexact integration approach is deprecated anyways but would require active type DG operators since every entry is multiplied by an individual metric term,
// wherease for the exact integration approach we have sums of three matrices each multiplied by its own metric term, whcih could be applied iteratively to the residual/solution.
// Not needed for Slab, and only two matrices (exact integration approach) required for Cylinder.
// This approach should only be used when necessary, i.e. solely when particle radius parameter sensitivity is required.
void GeneralRateModelDG::updateRadialDisc()
{
	_disc.deltaR = new active[_disc.offsetMetric[_disc.nParType]];

	for (unsigned int parType = 0; parType < _disc.nParType; ++parType)
	{
		if (_parDiscType[parType] == ParticleDiscretizationMode::Equidistant)
		{
			for (int cell = 0; cell < _disc.nParCell[parType]; cell++)
			{
				_disc.deltaR[_disc.offsetMetric[parType] + cell] = (_parRadius[parType] - _parCoreRadius[parType]) / _disc.nParCell[parType];
			}
				setEquidistantRadialDisc(parType);
		}
		else if (_parDiscType[parType] == ParticleDiscretizationMode::Equivolume)
			setEquivolumeRadialDisc(parType);
		else if (_parDiscType[parType] == ParticleDiscretizationMode::UserDefined)
			setUserdefinedRadialDisc(parType);
	}

	/*		metrics		*/
	// estimate cell dependent D_r

	for (int parType = 0; parType < _disc.nParType; parType++)
	{
		for (int cell = 0; cell < _disc.nParCell[parType]; cell++)
		{
			_disc.Ir[_disc.offsetMetric[parType] + cell] = Vector<active, Dynamic>::Zero(_disc.nParNode[parType]);

			for (int node = 0; node < _disc.nParNode[parType]; node++)
				_disc.Ir[_disc.offsetMetric[parType] + cell][node] = _disc.deltaR[_disc.offsetMetric[parType] + cell] / 2.0 * (_disc.parNodes[parType][node] + 1.0);

			_disc.Dr[_disc.offsetMetric[parType] + cell].resize(_disc.nParNode[parType], _disc.nParNode[parType]);
			_disc.Dr[_disc.offsetMetric[parType] + cell].setZero();

			active r_L = _parCoreRadius[parType] + cell * _disc.deltaR[_disc.offsetMetric[parType] + cell]; // left boundary of current cell

			_disc.Ir[_disc.offsetMetric[parType] + cell] = _disc.Ir[_disc.offsetMetric[parType] + cell] + VectorXd::Ones(_disc.nParNode[parType]) * r_L;

			if (_parGeomSurfToVol[parType] == SurfVolRatioSphere)
				_disc.Ir[_disc.offsetMetric[parType] + cell] = _disc.Ir[_disc.offsetMetric[parType] + cell].array().square();
			else if (_parGeomSurfToVol[parType] == SurfVolRatioSlab)
				_disc.Ir[_disc.offsetMetric[parType] + cell] = Vector<active, Dynamic>::Ones(_disc.nParNode[parType]); // no metrics for slab

			// (D_r)_{i, j} = D_{i, j} * (r_j / r_i) [only needed for inexact integration]
			_disc.Dr[_disc.offsetMetric[parType] + cell] = _disc.parPolyDerM[parType];
			_disc.Dr[_disc.offsetMetric[parType] + cell].array().rowwise() *= _disc.Ir[_disc.offsetMetric[parType] + cell].array().template cast<double>().transpose();
			_disc.Dr[_disc.offsetMetric[parType] + cell].array().colwise() *= _disc.Ir[_disc.offsetMetric[parType] + cell].array().template cast<double>().cwiseInverse();

			// compute mass matrices for exact integration based on particle geometry, via transformation to normalized Jacobi polynomials with weight function w
			if (_parGeomSurfToVol[parType] == SurfVolRatioSphere) // r^2 =  r_i^2 + (1 + \xi) * r_i * DeltaR_i / 2.0 + (1 + \xi)^2 * (DeltaR_i / 2.0)^2
			{
				_disc.parInvMM[_disc.offsetMetric[parType] + cell] = parts::dgtoolbox::mMatrix(_disc.parPolyDeg[parType], _disc.parNodes[parType], 0.0, 2.0) * pow((static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType] + cell]) / 2.0), 2.0);
				if (cell > 0 || _parCoreRadius[parType] != 0.0) // following contributions are zero for first cell when R_c = 0 (no particle core)
					_disc.parInvMM[_disc.offsetMetric[parType] + cell] += parts::dgtoolbox::mMatrix(_disc.parPolyDeg[parType], _disc.parNodes[parType], 0.0, 1.0) * (static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType] + cell]) * static_cast<double>(r_L))
					+ parts::dgtoolbox::mMatrix(_disc.parPolyDeg[parType], _disc.parNodes[parType], 0.0, 0.0) * pow(static_cast<double>(r_L), 2.0);

				_disc.parInvMM[_disc.offsetMetric[parType] + cell] = _disc.parInvMM[_disc.offsetMetric[parType] + cell].inverse();
				_disc.minus_InvMM_ST[_disc.offsetMetric[parType] + cell] = -_disc.parInvMM[_disc.offsetMetric[parType] + cell] * _disc.parPolyDerM[parType].transpose() * _disc.parInvMM[_disc.offsetMetric[parType] + cell].inverse();

				// particle GSM specific second order stiffness matrix (single element, i.e. nParCell = 1)
				_disc.secondOrderStiffnessM[parType] = std::pow(static_cast<double>(_parCoreRadius[parType]), 2.0) * parts::dgtoolbox::secondOrderStiffnessMatrix(_disc.parPolyDeg[parType], 0.0, 0.0, _disc.parNodes[parType]);
				_disc.secondOrderStiffnessM[parType] += static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType]]) * static_cast<double>(_parCoreRadius[parType]) * parts::dgtoolbox::secondOrderStiffnessMatrix(_disc.parPolyDeg[parType], 0.0, 1.0, _disc.parNodes[parType]);
				_disc.secondOrderStiffnessM[parType] += std::pow(static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType]]) / 2.0, 2.0) * parts::dgtoolbox::secondOrderStiffnessMatrix(_disc.parPolyDeg[parType], 0.0, 2.0, _disc.parNodes[parType]);
			}
			else if (_parGeomSurfToVol[parType] == SurfVolRatioCylinder) // r = r_i + (1 + \xi) * DeltaR_i / 2.0
			{
				_disc.parInvMM[_disc.offsetMetric[parType] + cell] = parts::dgtoolbox::mMatrix(_disc.parPolyDeg[parType], _disc.parNodes[parType], 0.0, 1.0) * (static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType] + cell]) / 2.0);
				if (cell > 0 || _parCoreRadius[parType] != 0.0) // following contribution is zero for first cell when R_c = 0 (no particle core)
					_disc.parInvMM[_disc.offsetMetric[parType] + cell] += parts::dgtoolbox::mMatrix(_disc.parPolyDeg[parType], _disc.parNodes[parType], 0.0, 0.0) * static_cast<double>(r_L);

				_disc.parInvMM[_disc.offsetMetric[parType] + cell] = _disc.parInvMM[_disc.offsetMetric[parType] + cell].inverse();
				_disc.minus_InvMM_ST[_disc.offsetMetric[parType] + cell] = -_disc.parInvMM[_disc.offsetMetric[parType] + cell] * _disc.parPolyDerM[parType].transpose() * _disc.parInvMM[_disc.offsetMetric[parType] + cell].inverse();

				// particle GSM specific second order stiffness matrix (single element, i.e. nParCell = 1)
				_disc.secondOrderStiffnessM[parType] = static_cast<double>(_parCoreRadius[parType]) * parts::dgtoolbox::secondOrderStiffnessMatrix(_disc.parPolyDeg[parType], 0.0, 0.0, _disc.parNodes[parType]);
				_disc.secondOrderStiffnessM[parType] += static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType]]) / 2.0 * parts::dgtoolbox::secondOrderStiffnessMatrix(_disc.parPolyDeg[parType], 0.0, 1.0, _disc.parNodes[parType]);
			}
			else if (_parGeomSurfToVol[parType] == SurfVolRatioSlab) // r = 1
			{
				_disc.minus_InvMM_ST[_disc.offsetMetric[parType] + cell] = -_disc.parInvMM[_disc.offsetMetric[parType] + cell] * _disc.parPolyDerM[parType].transpose() * _disc.parInvMM[_disc.offsetMetric[parType] + cell].inverse();

				_disc.secondOrderStiffnessM[parType] = parts::dgtoolbox::secondOrderStiffnessMatrix(_disc.parPolyDeg[parType], 0.0, 0.0, _disc.parNodes[parType]);
				_disc.parInvMM[_disc.offsetMetric[parType] + cell] = parts::dgtoolbox::invMMatrix(_disc.parPolyDeg[parType], _disc.parNodes[parType], 0.0, 0.0);
			}
		}

		_disc.minus_parInvMM_Ar[parType] = -_disc.parInvMM[_disc.offsetMetric[parType]] * _disc.secondOrderStiffnessM[parType];
	}
}

bool GeneralRateModelDG::setParameter(const ParameterId& pId, double value)
{
	if (pId.unitOperation == _unitOpIdx)
	{
		if (multiplexCompTypeSecParameterValue(pId, hashString("PORE_ACCESSIBILITY"), _poreAccessFactorMode, _poreAccessFactor, _disc.nParType, _disc.nComp, value, nullptr))
			return true;
		if (multiplexCompTypeSecParameterValue(pId, hashString("FILM_DIFFUSION"), _filmDiffusionMode, _filmDiffusion, _disc.nParType, _disc.nComp, value, nullptr))
			return true;
		if (multiplexCompTypeSecParameterValue(pId, hashString("PAR_DIFFUSION"), _parDiffusionMode, _parDiffusion, _disc.nParType, _disc.nComp, value, nullptr))
			return true;
		if (multiplexBndCompTypeSecParameterValue(pId, hashString("PAR_SURFDIFFUSION"), _parSurfDiffusionMode, _parSurfDiffusion, _disc.nParType, _disc.nComp, _disc.strideBound, _disc.nBound, _disc.boundOffset, value, nullptr))
			return true;
		const int mpIc = multiplexInitialConditions(pId, value, false);
		if (mpIc > 0)
			return true;
		else if (mpIc < 0)
			return false;

		// Intercept changes to PAR_TYPE_VOLFRAC when not specified per axial cell (but once globally)
		if (_axiallyConstantParTypeVolFrac && (pId.name == hashString("PAR_TYPE_VOLFRAC")))
		{
			if ((pId.section != SectionIndep) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep))
				return false;
			if (pId.particleType >= _disc.nParType)
				return false;

			for (unsigned int i = 0; i < _disc.nPoints; ++i)
				_parTypeVolFrac[i * _disc.nParType + pId.particleType].setValue(value);

			return true;
		}

		if (multiplexTypeParameterValue(pId, hashString("PAR_RADIUS"), _singleParRadius, _parRadius, value, nullptr))
			return true;
		if (multiplexTypeParameterValue(pId, hashString("PAR_CORERADIUS"), _singleParCoreRadius, _parCoreRadius, value, nullptr))
			return true;
		if (multiplexTypeParameterValue(pId, hashString("PAR_POROSITY"), _singleParPorosity, _parPorosity, value, nullptr))
			return true;

		if (model::setParameter(pId, value, _parDepSurfDiffusion, _singleParDepSurfDiffusion))
			return true;

		if (_convDispOp.setParameter(pId, value))
			return true;
	}

	const bool result = UnitOperationBase::setParameter(pId, value);

	// Check whether particle radius or core radius has changed and update radial discretization if necessary
	if (result && ((pId.name == hashString("PAR_RADIUS")) || (pId.name == hashString("PAR_CORERADIUS"))))
		updateRadialDisc();

	return result;
}

bool GeneralRateModelDG::setParameter(const ParameterId& pId, int value)
{
	if ((pId.unitOperation != _unitOpIdx) && (pId.unitOperation != UnitOpIndep))
		return false;

	if (model::setParameter(pId, value, _parDepSurfDiffusion, _singleParDepSurfDiffusion))
		return true;

	return UnitOperationBase::setParameter(pId, value);
}

bool GeneralRateModelDG::setParameter(const ParameterId& pId, bool value)
{
	if ((pId.unitOperation != _unitOpIdx) && (pId.unitOperation != UnitOpIndep))
		return false;

	if (model::setParameter(pId, value, _parDepSurfDiffusion, _singleParDepSurfDiffusion))
		return true;

	return UnitOperationBase::setParameter(pId, value);
}

void GeneralRateModelDG::setSensitiveParameterValue(const ParameterId& pId, double value)
{
	if (pId.unitOperation == _unitOpIdx)
	{
		if (multiplexCompTypeSecParameterValue(pId, hashString("PORE_ACCESSIBILITY"), _poreAccessFactorMode, _poreAccessFactor, _disc.nParType, _disc.nComp, value, &_sensParams))
			return;
		if (multiplexCompTypeSecParameterValue(pId, hashString("FILM_DIFFUSION"), _filmDiffusionMode, _filmDiffusion, _disc.nParType, _disc.nComp, value, &_sensParams))
			return;
		if (multiplexCompTypeSecParameterValue(pId, hashString("PAR_DIFFUSION"), _parDiffusionMode, _parDiffusion, _disc.nParType, _disc.nComp, value, &_sensParams))
			return;
		if (multiplexBndCompTypeSecParameterValue(pId, hashString("PAR_SURFDIFFUSION"), _parSurfDiffusionMode, _parSurfDiffusion, _disc.nParType, _disc.nComp, _disc.strideBound, _disc.nBound, _disc.boundOffset, value, &_sensParams))
			return;
		if (multiplexInitialConditions(pId, value, true) != 0)
			return;

		// Intercept changes to PAR_TYPE_VOLFRAC when not specified per axial cell (but once globally)
		if (_axiallyConstantParTypeVolFrac && (pId.name == hashString("PAR_TYPE_VOLFRAC")))
		{
			if ((pId.section != SectionIndep) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep))
				return;
			if (pId.particleType >= _disc.nParType)
				return;

			if (!contains(_sensParams, &_parTypeVolFrac[pId.particleType]))
				return;

			for (unsigned int i = 0; i < _disc.nPoints; ++i)
				_parTypeVolFrac[i * _disc.nParType + pId.particleType].setValue(value);

			return;
		}

		if (multiplexTypeParameterValue(pId, hashString("PAR_RADIUS"), _singleParRadius, _parRadius, value, &_sensParams))
			return;
		if (multiplexTypeParameterValue(pId, hashString("PAR_CORERADIUS"), _singleParCoreRadius, _parCoreRadius, value, &_sensParams))
			return;
		if (multiplexTypeParameterValue(pId, hashString("PAR_POROSITY"), _singleParPorosity, _parPorosity, value, &_sensParams))
			return;

		if (model::setSensitiveParameterValue(pId, value, _sensParams, _parDepSurfDiffusion, _singleParDepSurfDiffusion))
			return;

		if (_convDispOp.setSensitiveParameterValue(_sensParams, pId, value))
			return;
	}

	UnitOperationBase::setSensitiveParameterValue(pId, value);

	// Check whether particle radius or core radius has changed and update radial discretization if necessary
	if ((pId.name == hashString("PAR_RADIUS")) || (pId.name == hashString("PAR_CORERADIUS")))
		updateRadialDisc();
}

bool GeneralRateModelDG::setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue)
{
	if (pId.unitOperation == _unitOpIdx)
	{
		if (multiplexCompTypeSecParameterAD(pId, hashString("PORE_ACCESSIBILITY"), _poreAccessFactorMode, _poreAccessFactor, _disc.nParType, _disc.nComp, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (multiplexCompTypeSecParameterAD(pId, hashString("FILM_DIFFUSION"), _filmDiffusionMode, _filmDiffusion, _disc.nParType, _disc.nComp, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (multiplexCompTypeSecParameterAD(pId, hashString("PAR_DIFFUSION"), _parDiffusionMode, _parDiffusion, _disc.nParType, _disc.nComp, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (multiplexBndCompTypeSecParameterAD(pId, hashString("PAR_SURFDIFFUSION"), _parSurfDiffusionMode, _parSurfDiffusion, _disc.nParType, _disc.nComp, _disc.strideBound, _disc.nBound, _disc.boundOffset, adDirection, adValue, _sensParams))
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
			for (unsigned int i = 0; i < _disc.nPoints; ++i)
				_parTypeVolFrac[i * _disc.nParType + pId.particleType].setADValue(adDirection, adValue);

			return true;
		}

		if (multiplexTypeParameterAD(pId, hashString("PAR_RADIUS"), _singleParRadius, _parRadius, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (multiplexTypeParameterAD(pId, hashString("PAR_CORERADIUS"), _singleParCoreRadius, _parCoreRadius, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (multiplexTypeParameterAD(pId, hashString("PAR_POROSITY"), _singleParPorosity, _parPorosity, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (model::setSensitiveParameter(pId, adDirection, adValue, _sensParams, _parDepSurfDiffusion, _singleParDepSurfDiffusion))
		{
			LOG(Debug) << "Found parameter " << pId << " in surface diffusion parameter dependence: Dir " << adDirection << " is set to " << adValue;
			return true;
		}

		if (_convDispOp.setSensitiveParameter(_sensParams, pId, adDirection, adValue))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}
	}

	const bool result = UnitOperationBase::setSensitiveParameter(pId, adDirection, adValue);

	// Check whether particle radius or core radius has been set active and update radial discretization if necessary
	// Note that we need to recompute the radial discretization variables (_parCellSize, _parCenterRadius, _parOuterSurfAreaPerVolume, _parInnerSurfAreaPerVolume)
	// because their gradient has changed (although their nominal value has not changed).
	if ((pId.name == hashString("PAR_RADIUS")) || (pId.name == hashString("PAR_CORERADIUS")))
		updateRadialDisc();

	return result;
}

std::unordered_map<ParameterId, double> GeneralRateModelDG::getAllParameterValues() const
{
	std::unordered_map<ParameterId, double> data = UnitOperationBase::getAllParameterValues();
	model::getAllParameterValues(data, _parDepSurfDiffusion, _singleParDepSurfDiffusion);

	return data;
}

double GeneralRateModelDG::getParameterDouble(const ParameterId& pId) const
{
	double val = 0.0;
	if (model::getParameterDouble(pId, _parDepSurfDiffusion, _singleParDepSurfDiffusion, val))
		return val;

	// Not found
	return UnitOperationBase::getParameterDouble(pId);
}

bool GeneralRateModelDG::hasParameter(const ParameterId& pId) const
{
	if (model::hasParameter(pId, _parDepSurfDiffusion, _singleParDepSurfDiffusion))
		return true;

	return UnitOperationBase::hasParameter(pId);
}

int GeneralRateModelDG::Exporter::writeMobilePhase(double* buffer) const
{
	const int blockSize = numMobilePhaseDofs();
	std::copy_n(_idx.c(_data), blockSize, buffer);
	return blockSize;
}

int GeneralRateModelDG::Exporter::writeSolidPhase(double* buffer) const
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

int GeneralRateModelDG::Exporter::writeParticleMobilePhase(double* buffer) const
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

int GeneralRateModelDG::Exporter::writeSolidPhase(unsigned int parType, double* buffer) const
{
	cadet_assert(parType < _disc.nParType);

	const unsigned int stride = _disc.nComp + _disc.strideBound[parType];
	double const* ptr = _data + _idx.offsetCp(ParticleTypeIndex{ parType }) + _disc.nComp;
	for (unsigned int i = 0; i < _disc.nPoints; ++i)
	{
		for (unsigned int j = 0; j < _disc.nParPoints[parType]; ++j)
		{
			std::copy_n(ptr, _disc.strideBound[parType], buffer);
			buffer += _disc.strideBound[parType];
			ptr += stride;
		}
	}
	return _disc.nPoints * _disc.nParPoints[parType] * _disc.strideBound[parType];
}

int GeneralRateModelDG::Exporter::writeParticleMobilePhase(unsigned int parType, double* buffer) const
{
	cadet_assert(parType < _disc.nParType);

	const unsigned int stride = _disc.nComp + _disc.strideBound[parType];
	double const* ptr = _data + _idx.offsetCp(ParticleTypeIndex{ parType });
	for (unsigned int i = 0; i < _disc.nPoints; ++i)
	{
		for (unsigned int j = 0; j < _disc.nParPoints[parType]; ++j)
		{
			std::copy_n(ptr, _disc.nComp, buffer);
			buffer += _disc.nComp;
			ptr += stride;
		}
	}
	return _disc.nPoints * _disc.nParPoints[parType] * _disc.nComp;
}

int GeneralRateModelDG::Exporter::writeParticleFlux(double* buffer) const
{
	return 0;
}

int GeneralRateModelDG::Exporter::writeParticleFlux(unsigned int parType, double* buffer) const
{
	return 0;
}

int GeneralRateModelDG::Exporter::writeInlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);
	std::copy_n(_data, _disc.nComp, buffer);
	return _disc.nComp;
}

int GeneralRateModelDG::Exporter::writeInlet(double* buffer) const
{
	std::copy_n(_data, _disc.nComp, buffer);
	return _disc.nComp;
}

int GeneralRateModelDG::Exporter::writeOutlet(unsigned int port, double* buffer) const
{
	cadet_assert(port == 0);

	if (_model._convDispOp.forwardFlow())
		std::copy_n(&_idx.c(_data, _disc.nPoints - 1, 0), _disc.nComp, buffer);
	else
		std::copy_n(&_idx.c(_data, 0, 0), _disc.nComp, buffer);

	return _disc.nComp;
}

int GeneralRateModelDG::Exporter::writeOutlet(double* buffer) const
{
	if (_model._convDispOp.forwardFlow())
		std::copy_n(&_idx.c(_data, _disc.nPoints - 1, 0), _disc.nComp, buffer);
	else
		std::copy_n(&_idx.c(_data, 0, 0), _disc.nComp, buffer);

	return _disc.nComp;
}

}  // namespace model

}  // namespace cadet

#include "model/GeneralRateModelDG-InitialConditions.cpp"
#include "model/GeneralRateModelDG-LinearSolver.cpp"

