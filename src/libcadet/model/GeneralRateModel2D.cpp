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

#include "model/GeneralRateModel2D.hpp"
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
#include <numeric>

#include "ParallelSupport.hpp"
#ifdef CADET_PARALLELIZE
	#include <tbb/parallel_for.h>
#endif

namespace
{

cadet::model::MultiplexMode readAndRegisterMultiplexParam(cadet::IParameterProvider& paramProvider, std::unordered_map<cadet::ParameterId, cadet::active*>& parameters, std::vector<cadet::active>& values, const std::string& name, unsigned int nAxial, unsigned int nRad, unsigned int nParType, cadet::UnitOpIdx uoi)
{
	cadet::model::MultiplexMode mode = cadet::model::MultiplexMode::Independent;
	readScalarParameterOrArray(values, paramProvider, name, 1);
	if (paramProvider.exists(name + "_MULTIPLEX"))
	{
		const int modeConfig = paramProvider.getInt(name + "_MULTIPLEX");
		if (modeConfig == 0)
		{
			mode = cadet::model::MultiplexMode::Independent;
			if (values.size() != nParType)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name + "_MULTIPLEX (should be " + std::to_string(nParType) + ")");
		}
		else if (modeConfig == 1)
		{
			mode = cadet::model::MultiplexMode::Radial;
			if (values.size() != nRad * nParType)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name + "_MULTIPLEX (should be " + std::to_string(nRad * nParType) + ")");
		}
		else if (modeConfig == 2)
		{
			mode = cadet::model::MultiplexMode::Axial;
			if (values.size() != nAxial * nParType)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name + "_MULTIPLEX (should be " + std::to_string(nAxial * nParType) + ")");
		}
		else if (modeConfig == 3)
		{
			mode = cadet::model::MultiplexMode::AxialRadial;
			if (values.size() != nAxial * nRad * nParType)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name + "_MULTIPLEX (should be " + std::to_string(nAxial * nRad * nParType) + ")");
		}
	}
	else
	{
		if (values.size() == nParType)
			mode = cadet::model::MultiplexMode::Independent;
		else if (values.size() == nRad * nParType)
			mode = cadet::model::MultiplexMode::Radial;
		else if (values.size() == nAxial * nParType)
			mode = cadet::model::MultiplexMode::Axial;
		else if (values.size() == nRad * nAxial * nParType)
			mode = cadet::model::MultiplexMode::AxialRadial;
		else
			throw cadet::InvalidParameterException("Could not infer multiplex mode of field " + name + ", set " + name + "_MULTIPLEX or change number of elements");
	}

	const cadet::StringHash nameHash = cadet::hashStringRuntime(name);
	switch (mode)
	{
		case cadet::model::MultiplexMode::Independent:
			{
				std::vector<cadet::active> p(nAxial * nRad * nParType);
				for (unsigned int s = 0; s < nAxial * nRad; ++s)
					std::copy(values.begin(), values.end(), p.begin() + s * nParType);

				values = std::move(p);

				for (unsigned int s = 0; s < nParType; ++s)
					parameters[cadet::makeParamId(nameHash, uoi, cadet::CompIndep, s, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep)] = &values[s];
			}
			break;
		case cadet::model::MultiplexMode::Radial:
			{
				std::vector<cadet::active> p(nAxial * nRad * nParType);
				for (unsigned int s = 0; s < nAxial; ++s)
					std::copy(values.begin(), values.end(), p.begin() + s * nParType * nRad);

				values = std::move(p);

				for (unsigned int s = 0; s < nRad; ++s)
				{
					for (unsigned int i = 0; i < nParType; ++i)
						parameters[cadet::makeParamId(nameHash, uoi, cadet::CompIndep, i, cadet::BoundStateIndep, s, cadet::SectionIndep)] = &values[s * nParType + i];
				}
			}
			break;
		case cadet::model::MultiplexMode::Axial:
			{
				std::vector<cadet::active> p(nAxial * nRad * nParType);
				for (unsigned int i = 0; i < nAxial; ++i)
				{
					for (unsigned int j = 0; j < nRad; ++j)
						std::copy(values.begin() + i * nParType, values.begin() + (i+1) * nParType, p.begin() + i * nRad * nParType + j * nParType);
				}

				values = std::move(p);

				for (unsigned int s = 0; s < nAxial; ++s)
				{
					for (unsigned int i = 0; i < nParType; ++i)
						parameters[cadet::makeParamId(nameHash, uoi, cadet::CompIndep, i, cadet::BoundStateIndep, cadet::ReactionIndep, s)] = &values[s * nParType * nRad + i];
				}
			}
			break;
		case cadet::model::MultiplexMode::AxialRadial:
			cadet::registerParam3DArray(parameters, values, [=](bool multi, unsigned int ax, unsigned int rad, unsigned int pt) { return cadet::makeParamId(nameHash, uoi, cadet::CompIndep, pt, cadet::BoundStateIndep, rad, ax); }, nParType, nRad);
			break;
		case cadet::model::MultiplexMode::RadialSection:
		case cadet::model::MultiplexMode::Component:
		case cadet::model::MultiplexMode::ComponentRadial:
		case cadet::model::MultiplexMode::ComponentRadialSection:
		case cadet::model::MultiplexMode::ComponentSection:
		case cadet::model::MultiplexMode::Section:
		case cadet::model::MultiplexMode::Type:
		case cadet::model::MultiplexMode::ComponentType:
		case cadet::model::MultiplexMode::ComponentSectionType:
			cadet_assert(false);
			break;
	}

	return mode;
}

bool multiplexParameterValue(const cadet::ParameterId& pId, cadet::StringHash nameHash, cadet::model::MultiplexMode mode, std::vector<cadet::active>& data, unsigned int nAxial, unsigned int nRad, unsigned int nParType, double value, std::unordered_set<cadet::active*> const* sensParams)
{
	if (pId.name != nameHash)
		return false;

	switch (mode)
	{
		case cadet::model::MultiplexMode::Independent:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.particleType]))
					return false;

				for (unsigned int i = 0; i < nAxial * nRad; ++i)
					data[i * nParType + pId.particleType].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::Radial:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction == cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.reaction * nParType + pId.particleType]))
					return false;

				for (unsigned int i = 0; i < nAxial; ++i)
					data[i * nRad * nParType + pId.reaction * nParType + pId.particleType].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::Axial:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.section * nParType * nRad + pId.particleType]))
					return false;

				for (unsigned int i = 0; i < nRad; ++i)
					data[pId.section * nParType * nRad + i * nParType + pId.particleType].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::AxialRadial:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction == cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.section * nParType * nRad + pId.reaction * nParType + pId.particleType]))
					return false;

				data[pId.section * nParType * nRad + pId.reaction * nParType + pId.particleType].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::RadialSection:
		case cadet::model::MultiplexMode::Component:
		case cadet::model::MultiplexMode::ComponentRadial:
		case cadet::model::MultiplexMode::ComponentRadialSection:
		case cadet::model::MultiplexMode::ComponentSection:
		case cadet::model::MultiplexMode::Section:
		case cadet::model::MultiplexMode::Type:
		case cadet::model::MultiplexMode::ComponentType:
		case cadet::model::MultiplexMode::ComponentSectionType:
			cadet_assert(false);
			break;
	}

	return false;
}

bool multiplexParameterAD(const cadet::ParameterId& pId, cadet::StringHash nameHash, cadet::model::MultiplexMode mode, std::vector<cadet::active>& data, unsigned int nAxial, unsigned int nRad, unsigned int nParType, unsigned int adDirection, double adValue, std::unordered_set<cadet::active*>& sensParams)
{
	if (pId.name != nameHash)
		return false;

	switch (mode)
	{
		case cadet::model::MultiplexMode::Independent:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.particleType]);

				for (unsigned int i = 0; i < nAxial * nRad; ++i)
					data[i * nParType + pId.particleType].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::Radial:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction == cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.reaction * nParType + pId.particleType]);

				for (unsigned int i = 0; i < nAxial; ++i)
					data[i * nRad * nParType + pId.reaction * nParType + pId.particleType].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::Axial:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.section * nParType * nRad + pId.particleType]);

				for (unsigned int i = 0; i < nRad; ++i)
					data[pId.section * nParType * nRad + i * nParType + pId.particleType].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::AxialRadial:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction == cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.section * nParType * nRad + pId.reaction * nParType + pId.particleType]);
				data[pId.section * nParType * nRad + pId.reaction * nParType + pId.particleType].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::RadialSection:
		case cadet::model::MultiplexMode::Component:
		case cadet::model::MultiplexMode::ComponentRadial:
		case cadet::model::MultiplexMode::ComponentRadialSection:
		case cadet::model::MultiplexMode::ComponentSection:
		case cadet::model::MultiplexMode::Section:
		case cadet::model::MultiplexMode::Type:
		case cadet::model::MultiplexMode::ComponentType:
		case cadet::model::MultiplexMode::ComponentSectionType:
			cadet_assert(false);
			break;
	}

	return false;
}


}  // namespace


namespace cadet
{

namespace model
{

constexpr double SurfVolRatioSphere = 3.0;
constexpr double SurfVolRatioCylinder = 2.0;
constexpr double SurfVolRatioSlab = 1.0;

int schurComplementMultiplierGRM2D(void* userData, double const* x, double* z)
{
	GeneralRateModel2D* const grm = static_cast<GeneralRateModel2D*>(userData);
	return grm->schurComplementMatrixVector(x, z);
}


GeneralRateModel2D::GeneralRateModel2D(UnitOpIdx unitOpIdx) : UnitOperationBase(unitOpIdx),
	_dynReactionBulk(nullptr), _jacP(nullptr), _jacPdisc(nullptr), _jacPF(nullptr), _jacFP(nullptr), _jacInlet(),
	_analyticJac(true), _jacobianAdDirs(0), _factorizeJacobian(false), _tempState(nullptr),
	_initC(0), _singleRadiusInitC(true), _initCp(0), _singleRadiusInitCp(true), _initQ(0), _singleRadiusInitQ(true), _initState(0), _initStateDot(0)
{
}

GeneralRateModel2D::~GeneralRateModel2D() CADET_NOEXCEPT
{
	delete[] _tempState;

	delete _dynReactionBulk;

	delete[] _jacPF;
	delete[] _jacFP;

	delete[] _jacP;
	delete[] _jacPdisc;

	delete[] _disc.nParCell;
	delete[] _disc.parTypeOffset;
	delete[] _disc.nParCellsBeforeType;
	delete[] _disc.nBound;
	delete[] _disc.boundOffset;
	delete[] _disc.strideBound;
	delete[] _disc.nBoundBeforeType;
}

unsigned int GeneralRateModel2D::numDofs() const CADET_NOEXCEPT
{
	// Column bulk DOFs: nCol * nComp * nRad
	// Particle DOFs: nCol * nRad * nParType particles each having nComp (liquid phase) + sum boundStates (solid phase) DOFs
	//                in each shell; there are nParCell shells for each particle type
	// Flux DOFs: nCol * nComp * nRad * nParType (as many as column bulk DOFs)
	// Inlet DOFs: nComp * nRad
	return _disc.nCol * _disc.nRad * (_disc.nComp * (1 + _disc.nParType)) + _disc.parTypeOffset[_disc.nParType] + _disc.nComp * _disc.nRad;
}

unsigned int GeneralRateModel2D::numPureDofs() const CADET_NOEXCEPT
{
	// Column bulk DOFs: nCol * nComp * nRad
	// Particle DOFs: nCol * nRad * nParType particles each having nComp (liquid phase) + sum boundStates (solid phase) DOFs
	//                in each shell; there are nParCell shells for each particle type
	// Flux DOFs: nCol * nComp * nRad * nParType (as many as column bulk DOFs)
	return _disc.nCol * _disc.nRad * (_disc.nComp * (1 + _disc.nParType)) + _disc.parTypeOffset[_disc.nParType];
}


bool GeneralRateModel2D::usesAD() const CADET_NOEXCEPT
{
#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
	// We always need AD if we want to check the analytical Jacobian
	return true;
#else
	// We only need AD if we are not computing the Jacobian analytically
	return !_analyticJac;
#endif
}

bool GeneralRateModel2D::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper)
{
	// ==== Read discretization
	_disc.nComp = paramProvider.getInt("NCOMP");

	paramProvider.pushScope("discretization");

	_disc.nCol = paramProvider.getInt("NCOL");
	_disc.nRad = paramProvider.getInt("NRAD");

	const std::vector<int> nParCell = paramProvider.getIntArray("NPAR");

	const std::vector<int> nBound = paramProvider.getIntArray("NBOUND");
	if (nBound.size() < _disc.nComp)
		throw InvalidParameterException("Field NBOUND contains too few elements (NCOMP = " + std::to_string(_disc.nComp) + " required)");

	if (paramProvider.exists("NPARTYPE"))
		_disc.nParType = paramProvider.getInt("NPARTYPE");
	else
	{
		// Infer number of particle types
		_disc.nParType = std::max(nBound.size() / _disc.nComp, nParCell.size());
	}

	if ((nParCell.size() > 1) && (nParCell.size() < _disc.nParType))
		throw InvalidParameterException("Field NPAR must have 1 or NPARTYPE (" + std::to_string(_disc.nParType) + ") entries");

	_disc.nParCell = new unsigned int[_disc.nParType];
	if (nParCell.size() < _disc.nParType)
	{
		// Multiplex number of particle shells to all particle types
		for (unsigned int i = 0; i < _disc.nParType; ++i)
			std::fill(_disc.nParCell, _disc.nParCell + _disc.nParType, nParCell[0]);
	}
	else
		std::copy_n(nParCell.begin(), _disc.nParType, _disc.nParCell);

	if ((nBound.size() > _disc.nComp) && (nBound.size() < _disc.nComp * _disc.nParType))
		throw InvalidParameterException("Field NBOUND must have NCOMP (" + std::to_string(_disc.nComp) + ") or NCOMP * NPARTYPE (" + std::to_string(_disc.nComp * _disc.nParType) + ") entries");

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
	_disc.nParCellsBeforeType = new unsigned int[_disc.nParType + 1];
	_disc.parTypeOffset[0] = 0;
	_disc.nParCellsBeforeType[0] = 0;
	unsigned int nTotalParCells = 0;
	for (unsigned int j = 1; j < _disc.nParType + 1; ++j)
	{
		_disc.parTypeOffset[j] = _disc.parTypeOffset[j-1] + (_disc.nComp + _disc.strideBound[j-1]) * _disc.nParCell[j-1] * _disc.nCol * _disc.nRad;
		_disc.nParCellsBeforeType[j] = _disc.nParCellsBeforeType[j-1] + _disc.nParCell[j-1];
		nTotalParCells += _disc.nParCell[j-1];
	}
	_disc.nParCellsBeforeType[_disc.nParType] = nTotalParCells;

	// Configure particle discretization
	_parCellSize.resize(nTotalParCells);
	_parCenterRadius.resize(nTotalParCells);
	_parOuterSurfAreaPerVolume.resize(nTotalParCells);
	_parInnerSurfAreaPerVolume.resize(nTotalParCells);

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
		if (_parDiscVector.size() < nTotalParCells + _disc.nParType)
			throw InvalidParameterException("Field PAR_DISC_VECTOR contains too few elements (Sum [NPAR + 1] = " + std::to_string(nTotalParCells + _disc.nParType) + " required)");
	}

	// Determine whether analytic Jacobian should be used but don't set it right now.
	// We need to setup Jacobian matrices first.
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	const bool analyticJac = paramProvider.getBool("USE_ANALYTIC_JACOBIAN");
#else
	const bool analyticJac = false;
#endif

	// Read bulk-particle interface discretization order
	// Default to second order
	_colParBoundaryOrder = 2;
	if (paramProvider.exists("PAR_BOUNDARY_ORDER"))
	{
		_colParBoundaryOrder = paramProvider.getInt("PAR_BOUNDARY_ORDER");
		if ((_colParBoundaryOrder < 1) || (_colParBoundaryOrder > 2))
			throw InvalidParameterException("Field PAR_BOUNDARY_ORDER is out of valid range (1 or 2)");
	}

	// Initialize and configure GMRES for solving the Schur-complement
	_gmres.initialize(_disc.nCol * _disc.nRad * _disc.nComp * _disc.nParType, paramProvider.getInt("MAX_KRYLOV"), linalg::toOrthogonalization(paramProvider.getInt("GS_TYPE")), paramProvider.getInt("MAX_RESTARTS"));
	_gmres.matrixVectorMultiplier(&schurComplementMultiplierGRM2D, this);
	_schurSafety = paramProvider.getDouble("SCHUR_SAFETY");

	// Allocate space for initial conditions
	_initC.resize(_disc.nComp * _disc.nRad);
	_initCp.resize(_disc.nComp * _disc.nRad * _disc.nParType);
	_initQ.resize(nTotalBound * _disc.nRad);

	// Create nonlinear solver for consistent initialization
	configureNonlinearSolver(paramProvider);

	paramProvider.popScope();

	// Allocate memory
	Indexer idxr(_disc);

	_jacInlet.resize(_disc.nComp * _disc.nRad);

	_jacP = new linalg::BandMatrix[_disc.nCol * _disc.nRad * _disc.nParType];
	_jacPdisc = new linalg::FactorizableBandMatrix[_disc.nCol * _disc.nRad * _disc.nParType];
	for (unsigned int j = 0; j < _disc.nParType; ++j)
	{
		linalg::BandMatrix* const ptrJac = _jacP + _disc.nCol * _disc.nRad * j;
		linalg::FactorizableBandMatrix* const ptrJacDisc = _jacPdisc + _disc.nCol * _disc.nRad * j;
		for (unsigned int i = 0; i < _disc.nCol * _disc.nRad; ++i)
		{
			ptrJacDisc[i].resize(_disc.nParCell[j] * (_disc.nComp + _disc.strideBound[j]), _disc.nComp + _disc.strideBound[j], _disc.nComp + 2 * _disc.strideBound[j]);
			ptrJac[i].resize(_disc.nParCell[j] * (_disc.nComp + _disc.strideBound[j]), _disc.nComp + _disc.strideBound[j], _disc.nComp + 2 * _disc.strideBound[j]);
		}
	}

	_jacPF = new linalg::DoubleSparseMatrix[_disc.nCol * _disc.nRad * _disc.nParType];
	_jacFP = new linalg::DoubleSparseMatrix[_disc.nCol * _disc.nRad * _disc.nParType];
	for (unsigned int i = 0; i < _disc.nCol * _disc.nRad * _disc.nParType; ++i)
	{
		_jacPF[i].resize(_disc.nComp);
		const int type = i / (_disc.nCol * _disc.nRad);
		_jacFP[i].resize(_disc.nComp + 2 * _disc.strideBound[type]);
	}

	_jacCF.resize(_disc.nComp * _disc.nCol * _disc.nRad * _disc.nParType);
	_jacFC.resize(_disc.nComp * _disc.nCol * _disc.nRad * _disc.nParType);

	_discParFlux.resize(sizeof(active) * _disc.nComp);

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

	const bool transportSuccess = _convDispOp.configureModelDiscretization(paramProvider, helper, _disc.nComp, _disc.nCol, _disc.nRad, _dynReactionBulk);

	// Setup the memory for tempState based on state vector
	_tempState = new double[numDofs()];

	return transportSuccess && bindingConfSuccess && reactionConfSuccess;
}

bool GeneralRateModel2D::configure(IParameterProvider& paramProvider)
{
	_parameters.clear();

	const bool transportSuccess = _convDispOp.configure(_unitOpIdx, paramProvider, _parameters);

	// Read geometry parameters
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
		_parTypeVolFracMode = readAndRegisterMultiplexParam(paramProvider, _parameters, _parTypeVolFrac, "PAR_TYPE_VOLFRAC", _disc.nCol, _disc.nRad, _disc.nParType, _unitOpIdx);
	else
	{
		// Only one particle type present
		_parTypeVolFrac.resize(_disc.nCol * _disc.nRad, 1.0);
		_parTypeVolFracMode = MultiplexMode::Independent;
	}

	// Check whether all sizes are matched
	if (_disc.nParType != _parRadius.size())
		throw InvalidParameterException("Number of elements in field PAR_RADIUS does not match number of particle types");
	if (_disc.nParType * _disc.nCol * _disc.nRad != _parTypeVolFrac.size())
		throw InvalidParameterException("Number of elements in field PAR_TYPE_VOLFRAC does not match number of particle types times number of bulk cells");
	if (_disc.nParType != _parPorosity.size())
		throw InvalidParameterException("Number of elements in field PAR_POROSITY does not match number of particle types");
	if (_disc.nParType != _parCoreRadius.size())
		throw InvalidParameterException("Number of elements in field PAR_CORERADIUS does not match number of particle types");

	// Check that particle volume fractions sum to 1.0
	for (unsigned int i = 0; i < _disc.nCol * _disc.nRad; ++i)
	{
		const double volFracSum = std::accumulate(_parTypeVolFrac.begin() + i * _disc.nParType, _parTypeVolFrac.begin() + (i+1) * _disc.nParType, 0.0,
			[](double a, const active& b) -> double { return a + static_cast<double>(b); });
		if (std::abs(1.0 - volFracSum) > 1e-10)
			throw InvalidParameterException("Sum of field PAR_TYPE_VOLFRAC differs from 1.0 (is " + std::to_string(volFracSum) + ") in axial cell " + std::to_string(i / _disc.nRad) + " radial cell " + std::to_string(i % _disc.nRad));
	}

	// Read vectorial parameters (which may also be section dependent; transport)
	_filmDiffusionMode = readAndRegisterMultiplexCompTypeSecParam(paramProvider, _parameters, _filmDiffusion, "FILM_DIFFUSION", _disc.nParType, _disc.nComp, _unitOpIdx);
	_parDiffusionMode = readAndRegisterMultiplexCompTypeSecParam(paramProvider, _parameters, _parDiffusion, "PAR_DIFFUSION", _disc.nParType, _disc.nComp, _unitOpIdx);

	if (paramProvider.exists("PAR_SURFDIFFUSION"))
		_parSurfDiffusionMode = readAndRegisterMultiplexBndCompTypeSecParam(paramProvider, _parameters, _parSurfDiffusion, "PAR_SURFDIFFUSION", _disc.nParType, _disc.nComp, _disc.strideBound, _disc.nBound, _unitOpIdx);
	else
	{
		_parSurfDiffusionMode = MultiplexMode::Component;
		_parSurfDiffusion.resize(_disc.strideBound[_disc.nParType], 0.0);
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

	// Calculate the particle radial discretization variables (_parCellSize, _parCenterRadius, etc.)
	updateRadialDisc();

	// Register initial conditions parameters
	registerParam1DArray(_parameters, _initC, [=](bool multi, unsigned int comp) { return makeParamId(hashString("INIT_C"), _unitOpIdx, comp, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep); });
	if (_disc.nRad > 1)
		registerParam2DArray(_parameters, _initC, [=](bool multi, unsigned int rad, unsigned int comp) { return makeParamId(hashString("INIT_C"), _unitOpIdx, comp, ParTypeIndep, BoundStateIndep, rad, SectionIndep); }, _disc.nComp);

	if (_singleBinding)
	{
		for (unsigned int c = 0; c < _disc.nComp; ++c)
			_parameters[makeParamId(hashString("INIT_CP"), _unitOpIdx, c, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_initCp[c];

		if (_disc.nRad > 1)
		{
			for (unsigned int r = 0; r < _disc.nRad; ++r)
			{
				for (unsigned int c = 0; c < _disc.nComp; ++c)
					_parameters[makeParamId(hashString("INIT_CP"), _unitOpIdx, c, ParTypeIndep, BoundStateIndep, r, SectionIndep)] = &_initCp[r * _disc.nComp * _disc.nParType + c];
			}
		}
	}
	else
	{
		registerParam2DArray(_parameters, _initCp, [=](bool multi, unsigned int type, unsigned int comp) { return makeParamId(hashString("INIT_CP"), _unitOpIdx, comp, type, BoundStateIndep, ReactionIndep, SectionIndep); }, _disc.nComp);
		if (_disc.nRad > 1)
			registerParam3DArray(_parameters, _initCp, [=](bool multi, unsigned int rad, unsigned int type, unsigned int comp) { return makeParamId(hashString("INIT_CP"), _unitOpIdx, comp, type, BoundStateIndep, rad, SectionIndep); }, _disc.nComp, _disc.nParType);
	}

	if (!_binding.empty())
	{
		const unsigned int maxBoundStates = *std::max_element(_disc.strideBound, _disc.strideBound + _disc.nParType);
		std::vector<ParameterId> initParams(maxBoundStates);

		// Register radially independent
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

		// Register radially dependent
		if (_disc.nRad > 1)
		{
			if (_singleBinding)
			{
				_binding[0]->fillBoundPhaseInitialParameters(initParams.data(), _unitOpIdx, ParTypeIndep);

				for (unsigned int r = 0; r < _disc.nRad; ++r)
				{
					for (ParameterId& pId : initParams)
						pId.reaction = r;

					active* const iq = _initQ.data() + _disc.nBoundBeforeType[0] + r * _disc.strideBound[_disc.nParType];
					for (unsigned int i = 0; i < _disc.strideBound[0]; ++i)
						_parameters[initParams[i]] = iq + i;
				}
			}
			else
			{
				for (unsigned int r = 0; r < _disc.nRad; ++r)
				{
					for (unsigned int type = 0; type < _disc.nParType; ++type)
					{
						_binding[type]->fillBoundPhaseInitialParameters(initParams.data(), _unitOpIdx, type);
						for (ParameterId& pId : initParams)
							pId.reaction = r;

						active* const iq = _initQ.data() + _disc.nBoundBeforeType[type] + r * _disc.strideBound[_disc.nParType];
						for (unsigned int i = 0; i < _disc.strideBound[type]; ++i)
							_parameters[initParams[i]] = iq + i;
					}
				}
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


unsigned int GeneralRateModel2D::threadLocalMemorySize() const CADET_NOEXCEPT
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

unsigned int GeneralRateModel2D::numAdDirsForJacobian() const CADET_NOEXCEPT
{
	// We need as many directions as the highest bandwidth of the diagonal blocks:
	// The bandwidth of the column block depends on the size of the WENO stencil, whereas
	// the bandwidth of the particle blocks are given by the number of components and bound states.

	// Get maximum stride of particle type blocks
	int maxStride = 0;
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		maxStride = std::max(maxStride, _jacP[type * _disc.nCol * _disc.nRad].stride());
	}

	return maxStride;
}

void GeneralRateModel2D::useAnalyticJacobian(const bool analyticJac)
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

void GeneralRateModel2D::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac)
{
	// Setup flux Jacobian blocks at the beginning of the simulation or in case of
	// section dependent film or particle diffusion coefficients
	if ((secIdx == 0) || isSectionDependent(_filmDiffusionMode) || isSectionDependent(_parDiffusionMode))
		assembleOffdiagJac(t, secIdx);

	Indexer idxr(_disc);

	// ConvectionDispersionOperator tells us whether flow direction has changed
	if (!_convDispOp.notifyDiscontinuousSectionTransition(t, secIdx))
		return;

	// Setup the matrix connecting inlet DOFs to first column cells
	_jacInlet.clear();

	for (unsigned int rad = 0; rad < _disc.nRad; ++rad)
	{
		const double f = _convDispOp.inletFactor(secIdx, rad);
		if (_convDispOp.currentVelocity(rad) >= 0.0)
		{
			// Forwards flow

			// Place entries for inlet DOF to first axial cell conversion
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				_jacInlet.addElement(comp * idxr.strideColComp() + rad * idxr.strideColRadialCell(), comp, f);
		}
		else
		{
			// Backwards flow

			// Place entries for inlet DOF to last column cell conversion
			const unsigned int offset = (_disc.nCol - 1) * idxr.strideColAxialCell();
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				_jacInlet.addElement(offset + comp * idxr.strideColComp() + rad * idxr.strideColRadialCell(), comp, f);
		}
	}
}

void GeneralRateModel2D::setFlowRates(active const* in, active const* out) CADET_NOEXCEPT
{
	_convDispOp.setFlowRates(in, out);
}

void GeneralRateModel2D::reportSolution(ISolutionRecorder& recorder, double const* const solution) const
{
	Exporter expr(_disc, *this, solution);
	recorder.beginUnitOperation(_unitOpIdx, *this, expr);
	recorder.endUnitOperation();
}

void GeneralRateModel2D::reportSolutionStructure(ISolutionRecorder& recorder) const
{
	Exporter expr(_disc, *this, nullptr);
	recorder.unitOperationStructure(_unitOpIdx, *this, expr);
}


unsigned int GeneralRateModel2D::requiredADdirs() const CADET_NOEXCEPT
{
#ifndef CADET_CHECK_ANALYTIC_JACOBIAN
	return _jacobianAdDirs;
#else
	// If CADET_CHECK_ANALYTIC_JACOBIAN is active, we always need the AD directions for the Jacobian
	return numAdDirsForJacobian();
#endif
}

void GeneralRateModel2D::prepareADvectors(const AdJacobianParams& adJac) const
{
	// Early out if AD is disabled
	if (!adJac.adY)
		return;

	Indexer idxr(_disc);

	// Particle blocks
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		const unsigned int lowerParBandwidth = _jacP[type * _disc.nCol * _disc.nRad].lowerBandwidth();
		const unsigned int upperParBandwidth = _jacP[type * _disc.nCol * _disc.nRad].upperBandwidth();

		for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad; ++pblk)
		{
			ad::prepareAdVectorSeedsForBandMatrix(adJac.adY + idxr.offsetCp(ParticleTypeIndex{type}, ParticleIndex{pblk}), adJac.adDirOffset, idxr.strideParBlock(type), lowerParBandwidth, upperParBandwidth, lowerParBandwidth);
		}
	}
}

/**
 * @brief Extracts the system Jacobian from band compressed AD seed vectors
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
void GeneralRateModel2D::extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset)
{
	Indexer idxr(_disc);

	// Particles
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad; ++pblk)
		{
			linalg::BandMatrix& jacMat = _jacP[_disc.nCol * _disc.nRad * type + pblk];
			ad::extractBandedJacobianFromAd(adRes + idxr.offsetCp(ParticleTypeIndex{type}, ParticleIndex{pblk}), adDirOffset, jacMat.lowerBandwidth(), jacMat);
		}
	}
}

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN

/**
 * @brief Compares the analytical Jacobian with a Jacobian derived by AD
 * @details The analytical Jacobian is assumed to be stored in the corresponding band matrices.
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
void GeneralRateModel2D::checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const
{
	Indexer idxr(_disc);

	LOG(Debug) << "AD dir offset: " << adDirOffset << " DiagDirPar: " << _jacP[0].lowerBandwidth();

	// Particles
	double maxDiffPar = 0.0;
	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad; ++pblk)
		{
			linalg::BandMatrix& jacMat = _jacP[_disc.nCol * _disc.nRad * type + pblk];
			const double localDiff = ad::compareBandedJacobianWithAd(adRes + idxr.offsetCp(ParticleTypeIndex{type}, ParticleIndex{pblk}), adDirOffset, jacMat.lowerBandwidth(), jacMat);
			LOG(Debug) << "-> Par type " << type << " block " << pblk << " diff: " << localDiff;
			maxDiffPar = std::max(maxDiffPar, localDiff);
		}
	}
}

#endif

int GeneralRateModel2D::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidual);

	// Evaluate residual do not compute Jacobian or parameter sensitivities
	return residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, res, threadLocalMem);
}

int GeneralRateModel2D::residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidual);

	// Evaluate residual, use AD for Jacobian if required but do not evaluate parameter derivatives
	return residual(simTime, simState, res, adJac, threadLocalMem, true, false);
}

int GeneralRateModel2D::residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res,
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

template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int GeneralRateModel2D::residualImpl(double t, unsigned int secIdx, StateType const* const y, double const* const yDot, ResidualType* const res, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_START(_timerResidualPar);

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(_disc.nCol * _disc.nRad * _disc.nParType + 1), [&](std::size_t pblk)
#else
	for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad * _disc.nParType + 1; ++pblk)
#endif
	{
		if (cadet_unlikely(pblk == 0))
			residualBulk<StateType, ResidualType, ParamType, wantJac>(t, secIdx, y, yDot, res, threadLocalMem);
		else
		{
			const unsigned int type = (pblk - 1) / (_disc.nCol * _disc.nRad);
			const unsigned int par = (pblk - 1) % (_disc.nCol * _disc.nRad);
			residualParticle<StateType, ResidualType, ParamType, wantJac>(t, type, par, secIdx, y, yDot, res, threadLocalMem);
		}
	} CADET_PARFOR_END;

	BENCH_STOP(_timerResidualPar);

	residualFlux<StateType, ResidualType, ParamType>(t, secIdx, y, yDot, res);

	// Handle inlet DOFs, which are simply copied to res
	for (unsigned int i = 0; i < _disc.nComp * _disc.nRad; ++i)
	{
		res[i] = y[i];
	}

	return 0;
}

template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int GeneralRateModel2D::residualBulk(double t, unsigned int secIdx, StateType const* yBase, double const* yDotBase, ResidualType* resBase, util::ThreadLocalStorage& threadLocalMem)
{
	_convDispOp.residual(*this, t, secIdx, yBase, yDotBase, resBase, wantJac, typename ParamSens<ParamType>::enabled());
	if (!_dynReactionBulk || (_dynReactionBulk->numReactionsLiquid() == 0))
		return 0;

	// Get offsets
	Indexer idxr(_disc);
	StateType const* y = yBase + idxr.offsetC();
	ResidualType* res = resBase + idxr.offsetC();
	LinearBufferAllocator tlmAlloc = threadLocalMem.get();

	for (unsigned int colCell = 0; colCell < _disc.nCol * _disc.nRad; ++colCell, y += idxr.strideColRadialCell(), res += idxr.strideColRadialCell())
	{
		const unsigned int axialCell = colCell / _disc.nRad;
		const unsigned int radialCell = colCell % _disc.nRad;
		const double r = static_cast<double>(_convDispOp.radialCenters()[radialCell]) / static_cast<double>(_convDispOp.columnRadius());
		const double z = (0.5 + static_cast<double>(axialCell)) / static_cast<double>(_disc.nCol);

		const ColumnPosition colPos{z, r, 0.0};
		_dynReactionBulk->residualLiquidAdd(t, secIdx, colPos, y, res, -1.0, tlmAlloc);

		if (wantJac)
		{
			// static_cast should be sufficient here, but this statement is also analyzed when wantJac = false
			_dynReactionBulk->analyticJacobianLiquidAdd(t, secIdx, colPos, reinterpret_cast<double const*>(y), -1.0, _convDispOp.jacobian().row(colCell * idxr.strideColRadialCell()), tlmAlloc);
		}
	}

	return 0;
}

template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int GeneralRateModel2D::residualParticle(double t, unsigned int parType, unsigned int colCell, unsigned int secIdx, StateType const* yBase, double const* yDotBase, ResidualType* resBase, util::ThreadLocalStorage& threadLocalMem)
{
	Indexer idxr(_disc);

	// Go to the particle block of the given column cell
	StateType const* y = yBase + idxr.offsetCp(ParticleTypeIndex{parType}, ParticleIndex{colCell});
	double const* yDot = yDotBase + idxr.offsetCp(ParticleTypeIndex{parType}, ParticleIndex{colCell});
	ResidualType* res = resBase + idxr.offsetCp(ParticleTypeIndex{parType}, ParticleIndex{colCell});

	LinearBufferAllocator tlmAlloc = threadLocalMem.get();

	// Prepare parameters
	active const* const parDiff = getSectionDependentSlice(_parDiffusion, _disc.nComp * _disc.nParType, secIdx) + parType * _disc.nComp;

	// Ordering of particle surface diffusion:
	// bnd0comp0, bnd0comp1, bnd0comp2, bnd1comp0, bnd1comp1, bnd1comp2
	active const* const parSurfDiff = getSectionDependentSlice(_parSurfDiffusion, _disc.strideBound[_disc.nParType], secIdx) + _disc.nBoundBeforeType[parType];

	// Midpoint of current column cell (z, rho coordinate) - needed in externally dependent adsorption kinetic
	const unsigned int axialCell = colCell / _disc.nRad;
	const unsigned int radialCell = colCell % _disc.nRad;
	const double r = static_cast<double>(_convDispOp.radialCenters()[radialCell]) / static_cast<double>(_convDispOp.columnRadius());
	const double z = (0.5 + static_cast<double>(axialCell)) / static_cast<double>(_disc.nCol);

	// Reset Jacobian
	if (wantJac)
		_jacP[_disc.nCol * _disc.nRad * parType + colCell].setAll(0.0);

	// The RowIterator is always centered on the main diagonal.
	// This means that jac[0] is the main diagonal, jac[-1] is the first lower diagonal,
	// and jac[1] is the first upper diagonal. We can also access the rows from left to
	// right beginning with the last lower diagonal moving towards the main diagonal and
	// continuing to the last upper diagonal by using the native() method.
	linalg::BandMatrix::RowIterator jac = _jacP[_disc.nCol * _disc.nRad * parType + colCell].row(0);

	active const* const outerSurfPerVol = _parOuterSurfAreaPerVolume.data() + _disc.nParCellsBeforeType[parType];
	active const* const innerSurfPerVol = _parInnerSurfAreaPerVolume.data() + _disc.nParCellsBeforeType[parType];
	active const* const parCenterRadius = _parCenterRadius.data() + _disc.nParCellsBeforeType[parType];

	int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

	const parts::cell::CellParameters cellResParams
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

	// Loop over particle cells
	for (unsigned int par = 0; par < _disc.nParCell[parType]; ++par)
	{
		const ColumnPosition colPos{z, r, static_cast<double>(parCenterRadius[par]) / static_cast<double>(_parRadius[parType])};

		// Handle time derivatives, binding, dynamic reactions
		parts::cell::residualKernel<StateType, ResidualType, ParamType, parts::cell::CellParameters, linalg::BandMatrix::RowIterator, wantJac, true>(
			t, secIdx, colPos, y, yDotBase ? yDot : nullptr, res, jac, cellResParams, tlmAlloc
		);

		// We still need to handle transport and quasi-stationary reactions

		// Geometry
		const ParamType outerAreaPerVolume = static_cast<ParamType>(outerSurfPerVol[par]);
		const ParamType innerAreaPerVolume = static_cast<ParamType>(innerSurfPerVol[par]);

		// Mobile phase
		for (unsigned int comp = 0; comp < _disc.nComp; ++comp, ++res, ++y, ++yDot, ++jac)
		{
			const unsigned int nBound = _disc.nBound[_disc.nComp * parType + comp];
			const ParamType invBetaP = (1.0 - static_cast<ParamType>(_parPorosity[parType])) / (static_cast<ParamType>(_poreAccessFactor[_disc.nComp * parType + comp]) * static_cast<ParamType>(_parPorosity[parType]));
			const ParamType dp = static_cast<ParamType>(parDiff[comp]);

			// Add flow through outer surface
			// Note that inflow boundary conditions are handled in residualFlux().
			if (cadet_likely(par != 0))
			{
				// Difference between two cell-centers
				const ParamType dr = static_cast<ParamType>(parCenterRadius[par - 1]) - static_cast<ParamType>(parCenterRadius[par]);

				// Molecular diffusion contribution
				const ResidualType gradCp = (y[-idxr.strideParShell(parType)] - y[0]) / dr;
				*res -= outerAreaPerVolume * dp * gradCp;

				// Surface diffusion contribution
				for (unsigned int i = 0; i < nBound; ++i)
				{
					// See above for explanation of curIdx value
					const int curIdx = idxr.strideParLiquid() - comp + idxr.offsetBoundComp(ParticleTypeIndex{parType}, ComponentIndex{comp}) + i;
					const ResidualType gradQ = (y[-idxr.strideParShell(parType) + curIdx] - y[curIdx]) / dr;
					*res -= outerAreaPerVolume * static_cast<ParamType>(parSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{parType}, ComponentIndex{comp}) + i]) * invBetaP * gradQ;
				}

				if (wantJac)
				{
					const double localInvBetaP = static_cast<double>(invBetaP);
					const double ouApV = static_cast<double>(outerAreaPerVolume);
					const double ldr = static_cast<double>(dr);

					// Liquid phase
					jac[0] += ouApV * static_cast<double>(dp) / ldr; // dres / dc_p,i^(p,j)
					jac[-idxr.strideParShell(parType)] = -ouApV * static_cast<double>(dp) / ldr; // dres / dc_p,i^(p,j-1)

					// Solid phase
					for (unsigned int i = 0; i < nBound; ++i)
					{
						// See above for explanation of curIdx value
						const int curIdx = idxr.strideParLiquid() - comp + idxr.offsetBoundComp(ParticleTypeIndex{parType}, ComponentIndex{comp}) + i;
						jac[curIdx] += ouApV * localInvBetaP * static_cast<double>(parSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{parType}, ComponentIndex{comp}) + i]) / ldr; // dres / dq_i^(p,j)
						jac[-idxr.strideParShell(parType) + curIdx] = -ouApV * localInvBetaP * static_cast<double>(parSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{parType}, ComponentIndex{comp}) + i]) / ldr; // dres / dq_i^(p,j-1)
					}
				}
			}

			// Add flow through inner surface
			// Note that this term vanishes for the most inner shell due to boundary conditions
			if (cadet_likely(par != _disc.nParCell[parType] - 1))
			{
				// Difference between two cell-centers
				const ParamType dr = static_cast<ParamType>(parCenterRadius[par]) - static_cast<ParamType>(parCenterRadius[par + 1]);

				// Molecular diffusion contribution
				const ResidualType gradCp = (y[0] - y[idxr.strideParShell(parType)]) / dr;
				*res += innerAreaPerVolume * dp * gradCp;

				// Surface diffusion contribution
				for (unsigned int i = 0; i < nBound; ++i)
				{
					// See above for explanation of curIdx value
					const unsigned int curIdx = idxr.strideParLiquid() - comp + idxr.offsetBoundComp(ParticleTypeIndex{parType}, ComponentIndex{comp}) + i;
					const ResidualType gradQ = (y[curIdx] - y[idxr.strideParShell(parType) + curIdx]) / dr;
					*res += innerAreaPerVolume * static_cast<ParamType>(parSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{parType}, ComponentIndex{comp}) + i]) * invBetaP * gradQ;
				}

				if (wantJac)
				{
					const double localInvBetaP = static_cast<double>(invBetaP);
					const double inApV = static_cast<double>(innerAreaPerVolume);
					const double ldr = static_cast<double>(dr);

					// Liquid phase
					jac[0] += inApV * static_cast<double>(dp) / ldr; // dres / dc_p,i^(p,j)
					jac[idxr.strideParShell(parType)] = -inApV * static_cast<double>(dp) / ldr; // dres / dc_p,i^(p,j+1)

					// Solid phase
					for (unsigned int i = 0; i < nBound; ++i)
					{
						// See above for explanation of curIdx value
						const int curIdx = idxr.strideParLiquid() - comp + idxr.offsetBoundComp(ParticleTypeIndex{parType}, ComponentIndex{comp}) + i;
						jac[curIdx] += inApV * localInvBetaP * static_cast<double>(parSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{parType}, ComponentIndex{comp}) + i]) / ldr; // dres / dq_i^(p,j)
						jac[idxr.strideParShell(parType) + curIdx] = -inApV * localInvBetaP * static_cast<double>(parSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{parType}, ComponentIndex{comp}) + i]) / ldr; // dres / dq_i^(p,j-1)
					}
				}
			}
		}

		// Bound phases
		for (unsigned int bnd = 0; bnd < _disc.strideBound[parType]; ++bnd, ++res, ++y, ++jac)
		{
			// Skip quasi-stationary bound states
			if (qsReaction[bnd])
				continue;

			// Add flow through outer surface
			// Note that this term vanishes for the most outer shell due to boundary conditions
			if (cadet_likely(par != 0))
			{
				// Difference between two cell-centers
				const ParamType dr = static_cast<ParamType>(parCenterRadius[par - 1]) - static_cast<ParamType>(parCenterRadius[par]);

				const ResidualType gradQ = (y[-idxr.strideParShell(parType)] - y[0]) / dr;
				*res -= outerAreaPerVolume * static_cast<ParamType>(parSurfDiff[bnd]) * gradQ;

				if (wantJac)
				{
					const double ouApV = static_cast<double>(outerAreaPerVolume);
					const double ldr = static_cast<double>(dr);

					jac[0] += ouApV * static_cast<double>(parSurfDiff[bnd]) / ldr; // dres / dq_i^(p,j)
					jac[-idxr.strideParShell(parType)] += -ouApV * static_cast<double>(parSurfDiff[bnd]) / ldr; // dres / dq_i^(p,j-1)
				}
			}

			// Add flow through inner surface
			// Note that this term vanishes for the most inner shell due to boundary conditions
			if (cadet_likely(par != _disc.nParCell[parType] - 1))
			{
				// Difference between two cell-centers
				const ParamType dr = static_cast<ParamType>(parCenterRadius[par]) - static_cast<ParamType>(parCenterRadius[par + 1]);

				const ResidualType gradQ = (y[0] - y[idxr.strideParShell(parType)]) / dr;
				*res += innerAreaPerVolume * static_cast<ParamType>(parSurfDiff[bnd]) * gradQ;

				if (wantJac)
				{
					const double inApV = static_cast<double>(innerAreaPerVolume);
					const double ldr = static_cast<double>(dr);

					jac[0] += inApV * static_cast<double>(parSurfDiff[bnd]) / ldr; // dres / dq_i^(p,j)
					jac[idxr.strideParShell(parType)] += -inApV * static_cast<double>(parSurfDiff[bnd]) / ldr; // dres / dq_i^(p,j-1)
				}
			}
		}

		// Advance pointers over all bound states
		yDot += idxr.strideParBound(parType);
	}
	return 0;
}

template <typename StateType, typename ResidualType, typename ParamType>
int GeneralRateModel2D::residualFlux(double t, unsigned int secIdx, StateType const* yBase, double const* yDotBase, ResidualType* resBase)
{
	Indexer idxr(_disc);

	// Get offsets
	ResidualType* const resCol = resBase + idxr.offsetC();
	ResidualType* const resFlux = resBase + idxr.offsetJf();

	StateType const* const yCol = yBase + idxr.offsetC();
	StateType const* const yFlux = yBase + idxr.offsetJf();

	// J_f block (identity matrix), adds flux state to flux equation
	for (unsigned int i = 0; i < _disc.nComp * _disc.nCol * _disc.nRad * _disc.nParType; ++i)
		resFlux[i] = yFlux[i];

	// Discretized film diffusion kf for finite volumes
	ParamType* const kf_FV = _discParFlux.create<ParamType>(_disc.nComp);

	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		ResidualType* const resParType = resBase + idxr.offsetCp(ParticleTypeIndex{type});
		ResidualType* const resFluxType = resBase + idxr.offsetJf(ParticleTypeIndex{type});

		StateType const* const yParType = yBase + idxr.offsetCp(ParticleTypeIndex{type});
		StateType const* const yFluxType = yBase + idxr.offsetJf(ParticleTypeIndex{type});

		const ParamType epsP = static_cast<ParamType>(_parPorosity[type]);

		// Ordering of diffusion:
		// sec0type0comp0, sec0type0comp1, sec0type0comp2, sec0type1comp0, sec0type1comp1, sec0type1comp2,
		// sec1type0comp0, sec1type0comp1, sec1type0comp2, sec1type1comp0, sec1type1comp1, sec1type1comp2, ...
		active const* const filmDiff = getSectionDependentSlice(_filmDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;
		active const* const parDiff = getSectionDependentSlice(_parDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;

		const ParamType surfaceToVolumeRatio = _parGeomSurfToVol[type] / static_cast<ParamType>(_parRadius[type]);
		const ParamType outerAreaPerVolume = static_cast<ParamType>(_parOuterSurfAreaPerVolume[_disc.nParCellsBeforeType[type]]);

		const ParamType jacPF_val = -outerAreaPerVolume / epsP;

		// Discretized film diffusion kf for finite volumes
		if (cadet_likely(_colParBoundaryOrder == 2))
		{
			const ParamType absOuterShellHalfRadius = 0.5 * static_cast<ParamType>(_parCellSize[_disc.nParCellsBeforeType[type]]);
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				kf_FV[comp] = 1.0 / (absOuterShellHalfRadius / epsP / static_cast<ParamType>(_poreAccessFactor[type * _disc.nComp + comp]) / static_cast<ParamType>(parDiff[comp]) + 1.0 / static_cast<ParamType>(filmDiff[comp]));
		}
		else
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				kf_FV[comp] = static_cast<ParamType>(filmDiff[comp]);
		}

		// J_{0,f} block, adds flux to column void / bulk volume equations
		unsigned int idx = 0;
		for (unsigned int i = 0; i < _disc.nCol; ++i)
		{
			for (unsigned int j = 0; j < _disc.nRad; ++j)
			{
				const ParamType invBetaC = 1.0 / static_cast<ParamType>(_convDispOp.columnPorosity(j)) - 1.0;
				const ParamType jacCF_val = invBetaC * surfaceToVolumeRatio;
				for (unsigned int k = 0; k < _disc.nComp; ++k, ++idx)
				{
					resCol[idx] += jacCF_val * static_cast<ParamType>(_parTypeVolFrac[type + i * _disc.nParType * _disc.nRad + j * _disc.nParType]) * yFluxType[idx];
				}
			}
		}

		// J_{f,0} block, adds bulk volume state c_i to flux equation
		for (unsigned int bnd = 0; bnd < _disc.nCol * _disc.nRad; ++bnd)
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				const unsigned int eq = bnd * idxr.strideColRadialCell() + comp * idxr.strideColComp();
				resFluxType[eq] -= kf_FV[comp] * yCol[eq];
			}
		}

		// J_{p,f} block, implements bead boundary condition in outer bead shell equation
		for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad; ++pblk)
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				const unsigned int eq = pblk * idxr.strideColRadialCell() + comp * idxr.strideColComp();
				resParType[pblk * idxr.strideParBlock(type) + comp] += jacPF_val / static_cast<ParamType>(_poreAccessFactor[type * _disc.nComp + comp]) * yFluxType[eq];
			}
		}

		// J_{f,p} block, adds outer bead shell state c_{p,i} to flux equation
		for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad; ++pblk)
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				const unsigned int eq = pblk * idxr.strideColRadialCell() + comp * idxr.strideColComp();
				resFluxType[eq] += kf_FV[comp] * yParType[comp + pblk * idxr.strideParBlock(type)];
			}
		}

		if (cadet_unlikely(_binding[type]->hasQuasiStationaryReactions() && (_disc.nParCell[type] > 1)))
		{
			int const* const qsReaction = _binding[type]->reactionQuasiStationarity();

			// Ordering of particle surface diffusion:
			// bnd0comp0, bnd0comp1, bnd0comp2, bnd1comp0, bnd1comp1, bnd1comp2
			active const* const parSurfDiff = getSectionDependentSlice(_parSurfDiffusion, _disc.strideBound[_disc.nParType], secIdx) + _disc.nBoundBeforeType[type];
			active const* const parCenterRadius = _parCenterRadius.data() + _disc.nParCellsBeforeType[type];
			const ParamType absOuterShellHalfRadius = 0.5 * static_cast<ParamType>(_parCellSize[_disc.nParCellsBeforeType[type]]);

			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				kf_FV[comp] = (1.0 - static_cast<ParamType>(_parPorosity[type])) / (1.0 + epsP * static_cast<ParamType>(_poreAccessFactor[type * _disc.nComp + comp]) * static_cast<ParamType>(parDiff[comp]) / (absOuterShellHalfRadius * static_cast<ParamType>(filmDiff[comp])));

			for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad; ++pblk)
			{
				const ParamType dr = static_cast<ParamType>(parCenterRadius[0]) - static_cast<ParamType>(parCenterRadius[1]);

				for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				{
					const unsigned int eq = pblk * idxr.strideColRadialCell() + comp * idxr.strideColComp();
					const unsigned int nBound = _disc.nBound[_disc.nComp * type + comp];

					for (unsigned int i = 0; i < nBound; ++i)
					{
						const int idxBnd = idxr.offsetBoundComp(ParticleTypeIndex{type}, ComponentIndex{comp}) + i;

						// Skip quasi-stationary bound states
						if (!qsReaction[idxBnd])
							continue;

						const int curIdx = pblk * idxr.strideParBlock(type) + idxr.strideParLiquid() + idxBnd;
						const ResidualType gradQ = (yParType[curIdx] - yParType[curIdx + idxr.strideParShell(type)]) / dr;
						resFluxType[eq] -= kf_FV[comp] * static_cast<ParamType>(parSurfDiff[idxBnd]) * gradQ;
					}
				}
			}
		}
	}

	_discParFlux.destroy<ParamType>();
	return 0;
}

/**
 * @brief Assembles off diagonal Jacobian blocks
 * @details Assembles the fixed blocks @f$ J_{0,f}, \dots, J_{N_p,f} @f$ and @f$ J_{f,0}, \dots, J_{f, N_p}. @f$
 *          The blocks are fixed for each section.
 * @param [in] t Current time
 * @param [in] secIdx Index of the current section
 */
void GeneralRateModel2D::assembleOffdiagJac(double t, unsigned int secIdx)
{
	// Clear matrices for new assembly
	_jacCF.clear();
	_jacFC.clear();
	for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad * _disc.nParType; ++pblk)
	{
		_jacPF[pblk].clear();
		_jacFP[pblk].clear();
	}

	// Note that the J_f block, which is the identity matrix, is treated in the linear solver

	Indexer idxr(_disc);

	// Discretized film diffusion kf for finite volumes
	double* const kf_FV = _discParFlux.create<double>(_disc.nComp);

	for (unsigned int type = 0; type < _disc.nParType; ++type)
	{
		const unsigned int typeOffset = type * _disc.nCol * _disc.nComp * _disc.nRad;
		const double epsP = static_cast<double>(_parPorosity[type]);

		// Ordering of diffusion:
		// sec0type0comp0, sec0type0comp1, sec0type0comp2, sec0type1comp0, sec0type1comp1, sec0type1comp2,
		// sec1type0comp0, sec1type0comp1, sec1type0comp2, sec1type1comp0, sec1type1comp1, sec1type1comp2, ...
		active const* const filmDiff = getSectionDependentSlice(_filmDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;
		active const* const parDiff = getSectionDependentSlice(_parDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;

		const double surfaceToVolumeRatio = _parGeomSurfToVol[type] / static_cast<double>(_parRadius[type]);
		const double outerAreaPerVolume = static_cast<double>(_parOuterSurfAreaPerVolume[_disc.nParCellsBeforeType[type]]);

		const double jacPF_val = -outerAreaPerVolume / epsP;

		// Discretized film diffusion kf for finite volumes
		if (cadet_likely(_colParBoundaryOrder == 2))
		{
			const double absOuterShellHalfRadius = 0.5 * static_cast<double>(_parCellSize[_disc.nParCellsBeforeType[type]]);
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				kf_FV[comp] = 1.0 / (absOuterShellHalfRadius / epsP / static_cast<double>(_poreAccessFactor[type * _disc.nComp + comp]) / static_cast<double>(parDiff[comp]) + 1.0 / static_cast<double>(filmDiff[comp]));
		}
		else
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				kf_FV[comp] = static_cast<double>(filmDiff[comp]);
		}

		// J_{0,f} block, adds flux to column void / bulk volume equations
		unsigned int idx = 0;
		for (unsigned int col = 0; col < _disc.nCol; ++col)
		{
			for (unsigned int rad = 0; rad < _disc.nRad; ++rad)
			{
				const double invBetaC = 1.0 / static_cast<double>(_convDispOp.columnPorosity(rad)) - 1.0;
				const double jacCF_val = invBetaC * surfaceToVolumeRatio;

				for (unsigned int comp = 0; comp < _disc.nComp; ++comp, ++idx)
				{
					// Main diagonal corresponds to j_{f,i} (flux) state variable
					_jacCF.addElement(idx, idx + typeOffset, jacCF_val * static_cast<double>(_parTypeVolFrac[type + col * _disc.nRad * _disc.nParType + rad * _disc.nParType]));
				}
			}
		}

		// J_{f,0} block, adds bulk volume state c_i to flux equation
		for (unsigned int col = 0; col < _disc.nCol * _disc.nRad; ++col)
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				// Main diagonal corresponds to c_i state variable in each column cell
				const unsigned int eq = col * idxr.strideColRadialCell() + comp * idxr.strideColComp();
				_jacFC.addElement(eq + typeOffset, eq, -kf_FV[comp]);
			}
		}

		// J_{p,f} block, implements bead boundary condition in outer bead shell equation
		linalg::DoubleSparseMatrix* const jacPFtype = _jacPF + type * _disc.nCol * _disc.nRad;
		for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad; ++pblk)
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				const unsigned int eq = typeOffset + pblk * idxr.strideColRadialCell() + comp * idxr.strideColComp();
				jacPFtype[pblk].addElement(comp, eq, jacPF_val / static_cast<double>(_poreAccessFactor[comp]));
			}
		}

		// J_{f,p} block, adds outer bead shell state c_{p,i} to flux equation
		linalg::DoubleSparseMatrix* const jacFPtype = _jacFP + type * _disc.nCol * _disc.nRad;
		for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad; ++pblk)
		{
			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
			{
				const unsigned int eq = typeOffset + pblk * idxr.strideColRadialCell() + comp * idxr.strideColComp();
				jacFPtype[pblk].addElement(eq, comp, kf_FV[comp]);
			}
		}

		if (cadet_unlikely(_binding[type]->hasQuasiStationaryReactions() && (_disc.nParCell[type] > 1)))
		{
			int const* const qsReaction = _binding[type]->reactionQuasiStationarity();

			// Ordering of particle surface diffusion:
			// bnd0comp0, bnd0comp1, bnd0comp2, bnd1comp0, bnd1comp1, bnd1comp2
			active const* const parSurfDiff = getSectionDependentSlice(_parSurfDiffusion, _disc.strideBound[_disc.nParType], secIdx) + _disc.nBoundBeforeType[type];
			active const* const parCenterRadius = _parCenterRadius.data() + _disc.nParCellsBeforeType[type];
			const double absOuterShellHalfRadius = 0.5 * static_cast<double>(_parCellSize[_disc.nParCellsBeforeType[type]]);

			for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				kf_FV[comp] = (1.0 - static_cast<double>(_parPorosity[type])) / (1.0 + epsP * static_cast<double>(_poreAccessFactor[type * _disc.nComp + comp]) * static_cast<double>(parDiff[comp]) / (absOuterShellHalfRadius * static_cast<double>(filmDiff[comp])));

			for (unsigned int pblk = 0; pblk < _disc.nCol * _disc.nRad; ++pblk)
			{
				const double dr = static_cast<double>(parCenterRadius[0]) - static_cast<double>(parCenterRadius[1]);

				for (unsigned int comp = 0; comp < _disc.nComp; ++comp)
				{
					const unsigned int eq = typeOffset + pblk * idxr.strideColRadialCell() + comp * idxr.strideColComp();
					const unsigned int nBound = _disc.nBound[_disc.nComp * type + comp];

					for (unsigned int i = 0; i < nBound; ++i)
					{
						const int idxBnd = idxr.offsetBoundComp(ParticleTypeIndex{type}, ComponentIndex{comp}) + i;

						// Skip quasi-stationary bound states
						if (!qsReaction[idxBnd])
							continue;

						const double v = kf_FV[comp] * static_cast<double>(parSurfDiff[idxBnd]) / dr;
						const int curIdx = idxr.strideParLiquid() + idxBnd;

						jacFPtype[pblk].addElement(eq, curIdx, -v);
						jacFPtype[pblk].addElement(eq, curIdx + idxr.strideParShell(type), v);
					}
				}
			}
		}
	}

	_discParFlux.destroy<double>();
}

int GeneralRateModel2D::residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidualSens);

	// Evaluate residual for all parameters using AD in vector mode and at the same time update the
	// Jacobian (in one AD run, if analytic Jacobians are disabled)
	return residual(simTime, simState, nullptr, adJac, threadLocalMem, true, true);
}

int GeneralRateModel2D::residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem)
{
	BENCH_SCOPE(_timerResidualSens);

	// Evaluate residual for all parameters using AD in vector mode
	return residualImpl<double, active, active, false>(simTime.t, simTime.secIdx, simState.vecStateY, simState.vecStateYdot, adRes, threadLocalMem);
}

int GeneralRateModel2D::residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState,
	const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS, active const* adRes,
	double* const tmp1, double* const tmp2, double* const tmp3)
{
	BENCH_SCOPE(_timerResidualSens);

	// tmp1 stores result of (dF / dy) * s
	// tmp2 stores result of (dF / dyDot) * sDot

	const SimulationTime cst{simTime.t, simTime.secIdx};
	const ConstSimulationState css{nullptr, nullptr};
	for (std::size_t param = 0; param < yS.size(); ++param)
	{

		// Directional derivative (dF / dy) * s
		multiplyWithJacobian(cst, css, yS[param], 1.0, 0.0, tmp1);

		// Directional derivative (dF / dyDot) * sDot
		multiplyWithDerivativeJacobian(cst, css, ySdot[param], tmp2);

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
void GeneralRateModel2D::multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret)
{
	Indexer idxr(_disc);

	// Handle identity matrix of inlet DOFs
	for (unsigned int i = 0; i < _disc.nComp * _disc.nRad; ++i)
	{
		ret[i] = alpha * yS[i] + beta * ret[i];
	}

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(_disc.nCol * _disc.nRad * _disc.nParType + 1), [&](std::size_t idx)
#else
	for (unsigned int idx = 0; idx < _disc.nCol * _disc.nRad * _disc.nParType + 1; ++idx)
#endif
	{
		if (cadet_unlikely(idx == 0))
		{
			_convDispOp.jacobian().multiplyVector(yS + idxr.offsetC(), alpha, beta, ret + idxr.offsetC());
			_jacCF.multiplyVector(yS + idxr.offsetJf(), alpha, 1.0, ret + idxr.offsetC());
		}
		else
		{
			const unsigned int pblk = idx - 1;
			const unsigned int type = pblk / (_disc.nCol * _disc.nRad);
			const unsigned int par = pblk % (_disc.nCol * _disc.nRad);

			const int localOffset = idxr.offsetCp(ParticleTypeIndex{type}, ParticleIndex{par});
			_jacP[pblk].multiplyVector(yS + localOffset, alpha, beta, ret + localOffset);
			_jacPF[pblk].multiplyVector(yS + idxr.offsetJf(), alpha, 1.0, ret + localOffset);
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
		for (unsigned int par = 0; par < _disc.nCol * _disc.nRad; ++par)
		{
			_jacFP[type * _disc.nCol * _disc.nRad + par].multiplyVector(yS + idxr.offsetCp(ParticleTypeIndex{type}, ParticleIndex{par}), alpha, 1.0, retJf);
		}
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
void GeneralRateModel2D::multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret)
{
	Indexer idxr(_disc);

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), static_cast<std::size_t>(_disc.nCol * _disc.nRad * _disc.nParType + 1), [&](std::size_t idx)
#else
	for (unsigned int idx = 0; idx < _disc.nCol * _disc.nRad * _disc.nParType + 1; ++idx)
#endif
	{
		if (cadet_unlikely(idx == 0))
		{
			_convDispOp.multiplyWithDerivativeJacobian(simTime, sDot, ret);
		}
		else
		{
			const unsigned int idxParLoop = idx - 1;
			const unsigned int pblk = idxParLoop % (_disc.nCol * _disc.nRad);
			const unsigned int type = idxParLoop / (_disc.nCol * _disc.nRad);

			const double invBetaP = (1.0 / static_cast<double>(_parPorosity[type]) - 1.0);
			unsigned int const* const nBound = _disc.nBound + type * _disc.nComp;
			unsigned int const* const boundOffset = _disc.boundOffset + type * _disc.nComp;
			int const* const qsReaction = _binding[type]->reactionQuasiStationarity();

			// Particle shells
			for (unsigned int shell = 0; shell < _disc.nParCell[type]; ++shell)
			{
				double const* const localSdot = sDot + idxr.offsetCp(ParticleTypeIndex{type}, ParticleIndex{pblk}) + shell * idxr.strideParShell(type);
				double* const localRet = ret + idxr.offsetCp(ParticleTypeIndex{type}, ParticleIndex{pblk}) + shell * idxr.strideParShell(type);
				parts::cell::multiplyWithDerivativeJacobianKernel<true>(localSdot, localRet, _disc.nComp, nBound, boundOffset, _disc.strideBound[type], qsReaction, 1.0, invBetaP);
			}
		}
	} CADET_PARFOR_END;

	// Handle fluxes (all algebraic)
	double* const dFdyDot = ret + idxr.offsetJf();
	std::fill(dFdyDot, dFdyDot + _disc.nCol * _disc.nComp * _disc.nRad * _disc.nParType, 0.0);

	// Handle inlet DOFs (all algebraic)
	std::fill_n(ret, _disc.nComp * _disc.nRad, 0.0);
}

void GeneralRateModel2D::setExternalFunctions(IExternalFunction** extFuns, unsigned int size)
{
	for (IBindingModel* bm : _binding)
	{
		if (bm)
			bm->setExternalFunctions(extFuns, size);
	}
}

unsigned int GeneralRateModel2D::localOutletComponentIndex(unsigned int port) const CADET_NOEXCEPT
{
	// Inlets are duplicated so need to be accounted for
	if (static_cast<double>(_convDispOp.currentVelocity(port)) >= 0.0)
		// Forward Flow: outlet is last cell
		return _disc.nComp * _disc.nRad + (_disc.nCol - 1) * _disc.nComp * _disc.nRad + port * _disc.nComp;
	else
		// Backward flow: Outlet is first cell
		return _disc.nComp * _disc.nRad + _disc.nComp * port;
}

unsigned int GeneralRateModel2D::localInletComponentIndex(unsigned int port) const CADET_NOEXCEPT
{
	// Always 0 due to dedicated inlet DOFs
	return _disc.nComp * port;
}

unsigned int GeneralRateModel2D::localOutletComponentStride(unsigned int port) const CADET_NOEXCEPT
{
	return 1;
}

unsigned int GeneralRateModel2D::localInletComponentStride(unsigned int port) const CADET_NOEXCEPT
{
	return 1;
}

void GeneralRateModel2D::expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut)
{
	// @todo Write this function
}

/**
 * @brief Computes equidistant radial nodes in the beads
 */
void GeneralRateModel2D::setEquidistantRadialDisc(unsigned int parType)
{
	active* const ptrCenterRadius = _parCenterRadius.data() + _disc.nParCellsBeforeType[parType];
	active* const ptrOuterSurfAreaPerVolume = _parOuterSurfAreaPerVolume.data() + _disc.nParCellsBeforeType[parType];
	active* const ptrInnerSurfAreaPerVolume = _parInnerSurfAreaPerVolume.data() + _disc.nParCellsBeforeType[parType];

	const active radius = _parRadius[parType] - _parCoreRadius[parType];
	const active dr = radius / static_cast<double>(_disc.nParCell[parType]);
	std::fill(_parCellSize.data() + _disc.nParCellsBeforeType[parType], _parCellSize.data() + _disc.nParCellsBeforeType[parType] + _disc.nParCell[parType], dr);

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
void GeneralRateModel2D::setEquivolumeRadialDisc(unsigned int parType)
{
	active* const ptrCellSize = _parCellSize.data() + _disc.nParCellsBeforeType[parType];
	active* const ptrCenterRadius = _parCenterRadius.data() + _disc.nParCellsBeforeType[parType];
	active* const ptrOuterSurfAreaPerVolume = _parOuterSurfAreaPerVolume.data() + _disc.nParCellsBeforeType[parType];
	active* const ptrInnerSurfAreaPerVolume = _parInnerSurfAreaPerVolume.data() + _disc.nParCellsBeforeType[parType];

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

			// For the next cell: r_out == r_in of the current cell
			r_out = r_in;
		}
	}
}

/**
 * @brief Computes all helper quantities for radial bead discretization from given radial cell boundaries
 * @details Calculates surface areas per volume for every shell and the radial shell centers.
 */
void GeneralRateModel2D::setUserdefinedRadialDisc(unsigned int parType)
{
	active* const ptrCellSize = _parCellSize.data() + _disc.nParCellsBeforeType[parType];
	active* const ptrCenterRadius = _parCenterRadius.data() + _disc.nParCellsBeforeType[parType];
	active* const ptrOuterSurfAreaPerVolume = _parOuterSurfAreaPerVolume.data() + _disc.nParCellsBeforeType[parType];
	active* const ptrInnerSurfAreaPerVolume = _parInnerSurfAreaPerVolume.data() + _disc.nParCellsBeforeType[parType];

	// Care for the right ordering and include 0.0 / 1.0 if not already in the vector.
	std::vector<active> orderedInterfaces = std::vector<active>(_parDiscVector.begin() + _disc.nParCellsBeforeType[parType] + parType,
		_parDiscVector.begin() + _disc.nParCellsBeforeType[parType] + parType + _disc.nParCell[parType] + 1);

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
		}
	}
}

void GeneralRateModel2D::updateRadialDisc()
{
	for (unsigned int i = 0; i < _disc.nParType; ++i)
	{
		if (_parDiscType[i] == ParticleDiscretizationMode::Equidistant)
			setEquidistantRadialDisc(i);
		else if (_parDiscType[i] == ParticleDiscretizationMode::Equivolume)
			setEquivolumeRadialDisc(i);
		else if (_parDiscType[i] == ParticleDiscretizationMode::UserDefined)
			setUserdefinedRadialDisc(i);
	}
}

bool GeneralRateModel2D::setParameter(const ParameterId& pId, double value)
{
	if (pId.unitOperation == _unitOpIdx)
	{
		if (multiplexParameterValue(pId, hashString("PAR_TYPE_VOLFRAC"), _parTypeVolFracMode, _parTypeVolFrac, _disc.nCol, _disc.nRad, _disc.nParType, value, nullptr))
			return true;
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

		if (multiplexTypeParameterValue(pId, hashString("PAR_RADIUS"), _singleParRadius, _parRadius, value, nullptr))
			return true;
		if (multiplexTypeParameterValue(pId, hashString("PAR_CORERADIUS"), _singleParCoreRadius, _parCoreRadius, value, nullptr))
			return true;
		if (multiplexTypeParameterValue(pId, hashString("PAR_POROSITY"), _singleParPorosity, _parPorosity, value, nullptr))
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

void GeneralRateModel2D::setSensitiveParameterValue(const ParameterId& pId, double value)
{
	if (pId.unitOperation == _unitOpIdx)
	{
		if (multiplexParameterValue(pId, hashString("PAR_TYPE_VOLFRAC"), _parTypeVolFracMode, _parTypeVolFrac, _disc.nCol, _disc.nRad, _disc.nParType, value, &_sensParams))
			return;
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

		if (multiplexTypeParameterValue(pId, hashString("PAR_RADIUS"), _singleParRadius, _parRadius, value, &_sensParams))
			return;
		if (multiplexTypeParameterValue(pId, hashString("PAR_CORERADIUS"), _singleParCoreRadius, _parCoreRadius, value, &_sensParams))
			return;
		if (multiplexTypeParameterValue(pId, hashString("PAR_POROSITY"), _singleParPorosity, _parPorosity, value, &_sensParams))
			return;

		if (_convDispOp.setSensitiveParameterValue(_sensParams, pId, value))
			return;
	}

	UnitOperationBase::setSensitiveParameterValue(pId, value);

	// Check whether particle radius or core radius has changed and update radial discretization if necessary
	if ((pId.name == hashString("PAR_RADIUS")) || (pId.name == hashString("PAR_CORERADIUS")))
		updateRadialDisc();
}

bool GeneralRateModel2D::setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue)
{
	if (pId.unitOperation == _unitOpIdx)
	{
		if (multiplexParameterAD(pId, hashString("PAR_TYPE_VOLFRAC"), _parTypeVolFracMode, _parTypeVolFrac, _disc.nCol, _disc.nRad, _disc.nParType, adDirection, adValue, _sensParams))
		{
			LOG(Debug) << "Found parameter " << pId << ": Dir " << adDirection << " is set to " << adValue;
			return true;
		}

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


int GeneralRateModel2D::Exporter::writeMobilePhase(double* buffer) const
{
	const int blockSize = numMobilePhaseDofs();
	std::copy_n(_idx.c(_data), blockSize, buffer);
	return blockSize;
}

int GeneralRateModel2D::Exporter::writeSolidPhase(double* buffer) const
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

int GeneralRateModel2D::Exporter::writeParticleMobilePhase(double* buffer) const
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

int GeneralRateModel2D::Exporter::writeSolidPhase(unsigned int parType, double* buffer) const
{
	cadet_assert(parType < _disc.nParType);

	const unsigned int stride = _disc.nComp + _disc.strideBound[parType];
	double const* ptr = _data + _idx.offsetCp(ParticleTypeIndex{parType}) + _disc.nComp;
	for (unsigned int i = 0; i < _disc.nCol * _disc.nRad; ++i)
	{
		for (unsigned int j = 0; j < _disc.nParCell[parType]; ++j)
		{
			std::copy_n(ptr, _disc.strideBound[parType], buffer);
			buffer += _disc.strideBound[parType];
			ptr += stride;
		}
	}
	return _disc.nCol * _disc.nRad * _disc.nParCell[parType] * _disc.strideBound[parType];
}

int GeneralRateModel2D::Exporter::writeParticleMobilePhase(unsigned int parType, double* buffer) const
{
	cadet_assert(parType < _disc.nParType);

	const unsigned int stride = _disc.nComp + _disc.strideBound[parType];
	double const* ptr = _data + _idx.offsetCp(ParticleTypeIndex{parType});
	for (unsigned int i = 0; i < _disc.nCol * _disc.nRad; ++i)
	{
		for (unsigned int j = 0; j < _disc.nParCell[parType]; ++j)
		{
			std::copy_n(ptr, _disc.nComp, buffer);
			buffer += _disc.nComp;
			ptr += stride;
		}
	}
	return _disc.nCol * _disc.nRad * _disc.nParCell[parType] * _disc.nComp;
}

int GeneralRateModel2D::Exporter::writeParticleFlux(double* buffer) const
{
	const int blockSize = numParticleFluxDofs();
	std::copy_n(_idx.jf(_data), blockSize, buffer);
	return blockSize;
}

int GeneralRateModel2D::Exporter::writeParticleFlux(unsigned int parType, double* buffer) const
{
	const unsigned int blockSize = _disc.nComp * _disc.nRad * _disc.nCol;
	std::copy_n(_idx.jf(_data) + blockSize * parType, blockSize, buffer);
	return blockSize;
}

int GeneralRateModel2D::Exporter::writeInlet(unsigned int port, double* buffer) const
{
	cadet_assert(port < _disc.nRad);
	std::copy_n(_data + port * _disc.nComp, _disc.nComp, buffer);
	return _disc.nComp;
}

int GeneralRateModel2D::Exporter::writeInlet(double* buffer) const
{
	std::copy_n(_data, _disc.nComp * _disc.nRad, buffer);
	return _disc.nComp * _disc.nRad;
}

int GeneralRateModel2D::Exporter::writeOutlet(unsigned int port, double* buffer) const
{
	cadet_assert(port < _disc.nRad);

	if (_model._convDispOp.currentVelocity(port) >= 0)
		std::copy_n(&_idx.c(_data, _disc.nCol - 1, port, 0), _disc.nComp, buffer);
	else
		std::copy_n(&_idx.c(_data, 0, port, 0), _disc.nComp, buffer);

	return _disc.nComp;
}

int GeneralRateModel2D::Exporter::writeOutlet(double* buffer) const
{
	for (int i = 0; i < _disc.nRad; ++i)
	{
		writeOutlet(i, buffer);
		buffer += _disc.nComp;
	}
	return _disc.nComp * _disc.nRad;
}


void registerGeneralRateModel2D(std::unordered_map<std::string, std::function<IUnitOperation*(UnitOpIdx)>>& models)
{
	models[GeneralRateModel2D::identifier()] = [](UnitOpIdx uoId) { return new GeneralRateModel2D(uoId); };
	models["GRM2D"] = [](UnitOpIdx uoId) { return new GeneralRateModel2D(uoId); };
}

}  // namespace model

}  // namespace cadet
