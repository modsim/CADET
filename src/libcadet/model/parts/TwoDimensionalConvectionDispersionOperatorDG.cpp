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

#include "model/parts/TwoDimensionalConvectionDispersionOperatorDG.hpp"
#include "cadet/Exceptions.hpp"

#include "Stencil.hpp"
#include "ParamReaderHelper.hpp"
#include "AdUtils.hpp"
#include "SensParamUtil.hpp"
#include "SimulationTypes.hpp"
#include "linalg/CompressedSparseMatrix.hpp"
#include "model/parts/AxialConvectionDispersionKernel.hpp"
#include "model/ParameterDependence.hpp"
#include "ConfigurationHelper.hpp"

#include "model/parts/DGToolbox.hpp"

#ifdef SUPERLU_FOUND
	#include "linalg/SuperLUSparseMatrix.hpp"
#endif
#ifdef UMFPACK_FOUND
	#include "linalg/UMFPackSparseMatrix.hpp"
#endif

#include "linalg/BandMatrix.hpp"
#include "linalg/Gmres.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include <algorithm>

using namespace Eigen;

namespace
{

cadet::model::MultiplexMode readAndRegisterMultiplexParam(cadet::IParameterProvider& paramProvider, std::unordered_map<cadet::ParameterId, cadet::active*>& parameters, std::vector<cadet::active>& values, const std::string& name, unsigned int nComp, unsigned int radNPoints, cadet::UnitOpIdx uoi)
{
	cadet::model::MultiplexMode mode = cadet::model::MultiplexMode::Independent;
	readParameterMatrix(values, paramProvider, name, nComp * radNPoints, 1);
	unsigned int nSec = 1;
	if (paramProvider.exists(name + "_MULTIPLEX"))
	{
		const int modeConfig = paramProvider.getInt(name + "_MULTIPLEX");
		if (modeConfig == 0)
		{
			mode = cadet::model::MultiplexMode::Independent;
			if (values.size() > 1)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name + "_MULTIPLEX (should be 1)");
		}
		else if (modeConfig == 1)
		{
			mode = cadet::model::MultiplexMode::Radial;
			if (values.size() != radNPoints)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name + "_MULTIPLEX (should be " + std::to_string(radNPoints) + ")");
		}
		else if (modeConfig == 2)
		{
			mode = cadet::model::MultiplexMode::Component;
			if (values.size() != nComp)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name + "_MULTIPLEX (should be " + std::to_string(nComp) + ")");
		}
		else if (modeConfig == 3)
		{
			mode = cadet::model::MultiplexMode::ComponentRadial;
			if (values.size() != nComp * radNPoints)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name + "_MULTIPLEX (should be " + std::to_string(nComp * radNPoints) + ")");
		}
		else if (modeConfig == 4)
		{
			mode = cadet::model::MultiplexMode::Section;
			nSec = values.size();
		}
		else if (modeConfig == 5)
		{
			mode = cadet::model::MultiplexMode::RadialSection;
			if (values.size() % radNPoints != 0)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " is not a positive multiple of radNPoints (" + std::to_string(radNPoints) + ")");

			nSec = values.size() / radNPoints;
		}
		else if (modeConfig == 6)
		{
			mode = cadet::model::MultiplexMode::ComponentSection;
			if (values.size() % nComp != 0)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " is not a positive multiple of NCOMP (" + std::to_string(nComp) + ")");

			nSec = values.size() / nComp;
		}
		else if (modeConfig == 7)
		{
			mode = cadet::model::MultiplexMode::ComponentRadialSection;
			if (values.size() % (nComp * radNPoints) != 0)
				throw cadet::InvalidParameterException("Number of elements in field " + name + " is not a positive multiple of NCOMP * radNPoints (" + std::to_string(nComp * radNPoints) + ")");

			nSec = values.size() / (nComp * radNPoints);
		}
	}
	else
	{
		if (values.size() == 1)
			mode = cadet::model::MultiplexMode::Independent;
		else if (values.size() == nComp)
			mode = cadet::model::MultiplexMode::Component;
		else if (values.size() == radNPoints)
			mode = cadet::model::MultiplexMode::Radial;
		else if (values.size() == radNPoints * nComp)
			mode = cadet::model::MultiplexMode::ComponentRadial;
		else if (values.size() % nComp == 0)
		{
			mode = cadet::model::MultiplexMode::ComponentSection;
			nSec = values.size() / nComp;
		}
		else if (values.size() % radNPoints == 0)
		{
			mode = cadet::model::MultiplexMode::RadialSection;
			nSec = values.size() / radNPoints;
		}
		else if (values.size() % (radNPoints * nComp) == 0)
		{
			mode = cadet::model::MultiplexMode::ComponentRadialSection;
			nSec = values.size() / (nComp * radNPoints);
		}
		else
			throw cadet::InvalidParameterException("Could not infer multiplex mode of field " + name + ", set " + name + "_MULTIPLEX or change number of elements");

		// Do not infer cadet::model::MultiplexMode::Section in case of no matches (might hide specification errors)
	}

	const cadet::StringHash nameHash = cadet::hashStringRuntime(name);
	switch (mode)
	{
		case cadet::model::MultiplexMode::Independent:
		case cadet::model::MultiplexMode::Section:
			{
				std::vector<cadet::active> p(nComp * radNPoints * nSec);
				for (unsigned int s = 0; s < nSec; ++s)
					std::fill(p.begin() + s * radNPoints * nComp, p.begin() + (s+1) * radNPoints * nComp, values[s]);

				values = std::move(p);

				for (unsigned int s = 0; s < nSec; ++s)
					parameters[cadet::makeParamId(nameHash, uoi, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, (mode == cadet::model::MultiplexMode::Independent) ? cadet::SectionIndep : s)] = &values[s * radNPoints * nComp];
			}
			break;
		case cadet::model::MultiplexMode::Component:
		case cadet::model::MultiplexMode::ComponentSection:
			{
				std::vector<cadet::active> p(nComp * radNPoints * nSec);
				for (unsigned int s = 0; s < nSec; ++s)
				{
					for (unsigned int i = 0; i < nComp; ++i)
						std::copy(values.begin() + s * nComp, values.begin() + (s+1) * nComp, p.begin() + i * nComp + s * nComp * radNPoints);
				}

				values = std::move(p);

				for (unsigned int s = 0; s < nSec; ++s)
				{
					for (unsigned int i = 0; i < nComp; ++i)
						parameters[cadet::makeParamId(nameHash, uoi, i, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, (mode == cadet::model::MultiplexMode::Component) ? cadet::SectionIndep : s)] = &values[s * radNPoints * nComp + i];
				}
			}
			break;
		case cadet::model::MultiplexMode::Radial:
		case cadet::model::MultiplexMode::RadialSection:
			{
				std::vector<cadet::active> p(nComp * radNPoints * nSec);
				for (unsigned int i = 0; i < radNPoints * nSec; ++i)
					std::fill(p.begin() + i * nComp, p.begin() + (i+1) * nComp, values[i]);

				values = std::move(p);

				for (unsigned int s = 0; s < nSec; ++s)
				{
					for (unsigned int i = 0; i < radNPoints; ++i)
						parameters[cadet::makeParamId(nameHash, uoi, cadet::CompIndep, i, cadet::BoundStateIndep, cadet::ReactionIndep, (mode == cadet::model::MultiplexMode::Radial) ? cadet::SectionIndep : s)] = &values[s * radNPoints * nComp + i * nComp];
				}
			}
			break;
		case cadet::model::MultiplexMode::ComponentRadial:
		case cadet::model::MultiplexMode::ComponentRadialSection:
			cadet::registerParam3DArray(parameters, values, [=](bool multi, unsigned int sec, unsigned int compartment, unsigned int comp) { return cadet::makeParamId(nameHash, uoi, comp, compartment, cadet::BoundStateIndep, cadet::ReactionIndep, multi ? sec : cadet::SectionIndep); }, nComp, radNPoints);
			break;
		case cadet::model::MultiplexMode::Axial:
		case cadet::model::MultiplexMode::AxialRadial:
		case cadet::model::MultiplexMode::Type:
		case cadet::model::MultiplexMode::ComponentType:
		case cadet::model::MultiplexMode::ComponentSectionType:
			cadet_assert(false);
			break;
	}

	return mode;
}

bool multiplexParameterValue(const cadet::ParameterId& pId, cadet::StringHash nameHash, cadet::model::MultiplexMode mode, std::vector<cadet::active>& data, unsigned int nComp, unsigned int radNPoints, double value, std::unordered_set<cadet::active*> const* sensParams)
{
	if (pId.name != nameHash)
		return false;

	switch (mode)
	{
		case cadet::model::MultiplexMode::Independent:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType != cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[0]))
					return false;

				for (std::size_t i = 0; i < data.size(); ++i)
					data[i].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::Section:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType != cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.section * nComp * radNPoints]))
					return false;

				for (unsigned int i = 0; i < nComp * radNPoints; ++i)
					data[i + pId.section * nComp * radNPoints].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::Component:
			{
				if ((pId.component == cadet::CompIndep) || (pId.particleType != cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.component]))
					return false;

				for (unsigned int i = 0; i < radNPoints; ++i)
					data[i * nComp + pId.component].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::ComponentSection:
			{
				if ((pId.component == cadet::CompIndep) || (pId.particleType != cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.component + pId.section * nComp * radNPoints]))
					return false;

				for (unsigned int i = 0; i < radNPoints; ++i)
					data[i * nComp + pId.component + pId.section * nComp * radNPoints].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::Radial:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.particleType * nComp]))
					return false;

				for (unsigned int i = 0; i < nComp; ++i)
					data[i + pId.particleType * nComp].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::RadialSection:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.particleType * nComp + pId.section * nComp * radNPoints]))
					return false;

				for (unsigned int i = 0; i < nComp; ++i)
					data[i + pId.particleType * nComp + pId.section * nComp * radNPoints].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::ComponentRadial:
			{
				if ((pId.component == cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.component + pId.particleType * nComp]))
					return false;

				data[pId.component + pId.particleType * nComp].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::ComponentRadialSection:
			{
				if ((pId.component == cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				if (sensParams && !cadet::contains(*sensParams, &data[pId.component + pId.particleType * nComp + pId.section * nComp * radNPoints]))
					return false;

				data[pId.component + pId.particleType * nComp + pId.section * nComp * radNPoints].setValue(value);

				return true;
			}
		case cadet::model::MultiplexMode::Axial:
		case cadet::model::MultiplexMode::AxialRadial:
		case cadet::model::MultiplexMode::Type:
		case cadet::model::MultiplexMode::ComponentType:
		case cadet::model::MultiplexMode::ComponentSectionType:
			cadet_assert(false);
			break;
	}

	return false;
}

bool multiplexParameterAD(const cadet::ParameterId& pId, cadet::StringHash nameHash, cadet::model::MultiplexMode mode, std::vector<cadet::active>& data, unsigned int nComp, unsigned int radNPoints, unsigned int adDirection, double adValue, std::unordered_set<cadet::active*>& sensParams)
{
	if (pId.name != nameHash)
		return false;

	switch (mode)
	{
		case cadet::model::MultiplexMode::Independent:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType != cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				sensParams.insert(&data[0]);

				for (std::size_t i = 0; i < data.size(); ++i)
					data[i].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::Section:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType != cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.section * nComp * radNPoints]);

				for (unsigned int i = 0; i < nComp * radNPoints; ++i)
					data[i + pId.section * nComp * radNPoints].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::Component:
			{
				if ((pId.component == cadet::CompIndep) || (pId.particleType != cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.component]);

				for (unsigned int i = 0; i < radNPoints; ++i)
					data[i * nComp + pId.component].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::ComponentSection:
			{
				if ((pId.component == cadet::CompIndep) || (pId.particleType != cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.component + pId.section * nComp * radNPoints]);

				for (unsigned int i = 0; i < radNPoints; ++i)
					data[i * nComp + pId.component + pId.section * nComp * radNPoints].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::Radial:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.particleType * nComp]);

				for (unsigned int i = 0; i < nComp; ++i)
					data[i + pId.particleType * nComp].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::RadialSection:
			{
				if ((pId.component != cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.particleType * nComp + pId.section * nComp * radNPoints]);

				for (unsigned int i = 0; i < nComp; ++i)
					data[i + pId.particleType * nComp + pId.section * nComp * radNPoints].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::ComponentRadial:
			{
				if ((pId.component == cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section != cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.component + pId.particleType * nComp]);

				data[pId.component + pId.particleType * nComp].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::ComponentRadialSection:
			{
				if ((pId.component == cadet::CompIndep) || (pId.particleType == cadet::ParTypeIndep) || (pId.boundState != cadet::BoundStateIndep)
					|| (pId.reaction != cadet::ReactionIndep) || (pId.section == cadet::SectionIndep))
					return false;

				sensParams.insert(&data[pId.component + pId.particleType * nComp + pId.section * nComp * radNPoints]);

				data[pId.component + pId.particleType * nComp + pId.section * nComp * radNPoints].setADValue(adDirection, adValue);

				return true;
			}
		case cadet::model::MultiplexMode::Axial:
		case cadet::model::MultiplexMode::AxialRadial:
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

namespace parts
{

class TwoDimensionalConvectionDispersionOperatorDG::LinearSolver
{
public:

	virtual ~LinearSolver() CADET_NOEXCEPT { }

	virtual bool initialize(IParameterProvider& paramProvider, unsigned int nComp, unsigned int axNPoints, unsigned int radNPoints, const Weno& weno) = 0;
	virtual void setSparsityPattern(const cadet::linalg::SparsityPattern& pattern) = 0;
	virtual void assembleDiscretizedJacobian(double alpha) = 0;
	virtual bool factorize() = 0;
	virtual bool solveDiscretizedJacobian(double* rhs, double const* weight, double const* init, double outerTol) const = 0;
};

int schurComplementMultiplier2DCDO_DG(void* userData, double const* x, double* z);

class TwoDimensionalConvectionDispersionOperatorDG::GmresSolver : public TwoDimensionalConvectionDispersionOperatorDG::LinearSolver
{
public:

	GmresSolver(linalg::CompressedSparseMatrix const* jacC) : _jacC(jacC) { }
	virtual ~GmresSolver() CADET_NOEXCEPT { }

	virtual bool initialize(IParameterProvider& paramProvider, unsigned int nComp, unsigned int axNPoints, unsigned int radNPoints, const Weno& weno)
	{
		_gmres.initialize(axNPoints * nComp * radNPoints, 0, linalg::toOrthogonalization(1), 0);
		_gmres.matrixVectorMultiplier(&schurComplementMultiplier2DCDO_DG, this);
		_cache.resize(axNPoints * nComp * radNPoints, 0.0);

		return true;
	}

	virtual void setSparsityPattern(const linalg::SparsityPattern& pattern) { }

	virtual void assembleDiscretizedJacobian(double alpha)
	{
		_alpha = alpha;
	}

	virtual bool factorize() { return true; }

	virtual bool solveDiscretizedJacobian(double* rhs, double const* weight, double const* init, double outerTol) const 
	{
		if (init)
			std::copy(init, init + _cache.size(), _cache.begin());

		const int gmresResult = _gmres.solve(0.05 * outerTol * std::sqrt(_jacC->rows()), weight, rhs, _cache.data());
		std::copy(_cache.begin(), _cache.end(), rhs);

		return gmresResult == 0;
	}

protected:
	linalg::CompressedSparseMatrix const* const _jacC;
	double _alpha;
	mutable linalg::Gmres _gmres; //!< GMRES algorithm for the Schur-complement in linearSolve()
	mutable std::vector<double> _cache; //!< GMRES cache for result

	int schurComplementMatrixVector(double const* x, double* z) const
	{
		std::fill(z, z + _jacC->rows(), _alpha);
		_jacC->multiplyVector(x, 1.0, 1.0, z);
		return 0;
	}

	// Wrapper for calling the corresponding function in this class
	friend int schurComplementMultiplier2DCDO_DG(void* userData, double const* x, double* z);
};

int schurComplementMultiplier2DCDO_DG(void* userData, double const* x, double* z)
{
	TwoDimensionalConvectionDispersionOperatorDG::GmresSolver* const cdo = static_cast<TwoDimensionalConvectionDispersionOperatorDG::GmresSolver*>(userData);
	return cdo->schurComplementMatrixVector(x, z);
}

#if defined(UMFPACK_FOUND) || defined(SUPERLU_FOUND) 

	template <typename sparse_t>
	class TwoDimensionalConvectionDispersionOperatorDG::SparseDirectSolver : public TwoDimensionalConvectionDispersionOperatorDG::LinearSolver
	{
	public:

		SparseDirectSolver(linalg::CompressedSparseMatrix const* jacC) : _jacC(jacC) { }
		virtual ~SparseDirectSolver() CADET_NOEXCEPT { }

		virtual bool initialize(IParameterProvider& paramProvider, unsigned int nComp, unsigned int axNPoints, unsigned int radNPoints, const Weno& weno)
		{
			return true;
		}

		virtual void setSparsityPattern(const linalg::SparsityPattern& pattern)
		{
			_jacCdisc.assignPattern(pattern);
			_jacCdisc.prepare();
		}

		virtual void assembleDiscretizedJacobian(double alpha)
		{
			// Copy normal matrix over to factorizable matrix
			_jacCdisc.copyFromSamePattern(*_jacC);

			for (int i = 0; i < _jacC->rows(); ++i)
				_jacCdisc.centered(i, 0) += alpha;
		}

		virtual bool factorize()
		{
			return _jacCdisc.factorize();
		}

		virtual bool solveDiscretizedJacobian(double* rhs, double const* weight, double const* init, double outerTol) const 
		{
			return _jacCdisc.solve(rhs);
		}

	protected:
		linalg::CompressedSparseMatrix const* const _jacC;
		sparse_t _jacCdisc;
	};

#endif

class TwoDimensionalConvectionDispersionOperatorDG::DenseDirectSolver : public TwoDimensionalConvectionDispersionOperatorDG::LinearSolver
{
public:

	DenseDirectSolver(linalg::CompressedSparseMatrix const* jacC) : _jacC(jacC) { }
	virtual ~DenseDirectSolver() CADET_NOEXCEPT { }

	virtual bool initialize(IParameterProvider& paramProvider, unsigned int nComp, unsigned int axNPoints, unsigned int radNPoints, const Weno& weno)
	{
		// Note that we have to increase the lower bandwidth by 1 because the WENO stencil is applied to the
		// right cell face (lower + 1 + upper) and to the left cell face (shift the stencil by -1 because influx of cell i
		// is outflux of cell i-1)
		// We also have to make sure that there's at least one sub and super diagonal for the dispersion term
		const unsigned int lb = std::max(weno.lowerBandwidth() + 1u, 1u) * nComp * radNPoints;

		// We have to make sure that there's at least one sub and super diagonal for the dispersion term
		const unsigned int ub = std::max(weno.upperBandwidth(), 1u) * nComp * radNPoints;

		// When flow direction is changed, the bandwidths of the Jacobian swap.
		// Hence, we have to reserve memory such that the swapped Jacobian can fit into the matrix.
		const unsigned int mb = std::max(lb, ub);

		// Allocate matrices such that bandwidths can be switched (backwards flow support)
		_jacCdisc.resize(axNPoints * nComp * radNPoints, mb, mb);
		return true;
	}

	virtual void setSparsityPattern(const linalg::SparsityPattern& pattern) { }

	virtual void assembleDiscretizedJacobian(double alpha)
	{
		// Copy normal matrix over to factorizable matrix
		_jacCdisc.setAll(0.0);

		linalg::FactorizableBandMatrix::RowIterator jac = _jacCdisc.row(0);
		for (int i = 0; i < _jacC->rows(); ++i, ++jac)
		{
			linalg::sparse_int_t const* const colIdx = _jacC->columnIndicesOfRow(i);
			double const* const vals = _jacC->valuesOfRow(i);
			const int nnz = _jacC->numNonZerosInRow(i);

			// Copy row from sparse matrix to banded matrix
			for (int c = 0; c < nnz; ++c)
			{
				const linalg::sparse_int_t diag = colIdx[c] - i;
				jac[diag] = vals[c];
			}

			// Add time derivative			
			jac[0] += alpha;
		}
	}

	virtual bool factorize()
	{
		return _jacCdisc.factorize();
	}

	virtual bool solveDiscretizedJacobian(double* rhs, double const* weight, double const* init, double outerTol) const 
	{
		return _jacCdisc.solve(rhs);
	}

protected:
	linalg::CompressedSparseMatrix const* const _jacC;
	linalg::FactorizableBandMatrix _jacCdisc;
};


/**
 * @brief Creates a TwoDimensionalConvectionDispersionOperatorDG
 */
TwoDimensionalConvectionDispersionOperatorDG::TwoDimensionalConvectionDispersionOperatorDG() : _colPorosities(0), _dir(0), _stencilMemory(sizeof(active) * Weno::maxStencilSize()), 
	_linearSolver(nullptr), _dispersionDep(nullptr)
{
}

TwoDimensionalConvectionDispersionOperatorDG::~TwoDimensionalConvectionDispersionOperatorDG() CADET_NOEXCEPT
{
	delete _linearSolver;
	delete _dispersionDep;
}


// todo: tildeMr, tildeMrDash, tildeSrDash // todo active operators?
MatrixXd TwoDimensionalConvectionDispersionOperatorDG::tildeMr(const unsigned int elemIdx)
{
	MatrixXd tildeMr = MatrixXd::Zero(_radNNodes, _qNNodes);
	MatrixXd ellEll = MatrixXd::Zero(_radNNodes, _qNNodes);

	for (unsigned int j = 0; j < _radNNodes; j++)
		ellEll.row(j) = dgtoolbox::evalLagrangeBasis(j, _radNodes, _qNodes);

	for (unsigned int j = 0; j < _radNNodes; j++)
	{
		for (unsigned int k = 0; k < _qNNodes; k++)
		{
			// todo allow active types in mapping
			tildeMr(j, k) = dgtoolbox::mapRefToPhys<active>(_radDelta, elemIdx, _qNodes[k]) * static_cast<double>(_curRadialDispersionTilde[elemIdx * _qNNodes + k]) * ellEll(j, k) * _qWeights[k];
		}
	}

	return tildeMr;
}

MatrixXd TwoDimensionalConvectionDispersionOperatorDG::tildeMrDash(const unsigned int elemIdx, const unsigned int secIdx)
{
	MatrixXd ellEll = MatrixXd::Zero(_radNNodes, _qNNodes);
	MatrixXd tildeMrDash = MatrixXd::Zero(_radNNodes, _radNNodes * _qNNodes);

	for (unsigned int j = 0; j < _radNNodes; j++)
		ellEll.row(j) = dgtoolbox::evalLagrangeBasis(j, _radNodes, _qNodes);

	for (unsigned int j = 0; j < _radNNodes; j++)
	{
		for (unsigned int m = 0; m < _radNNodes; m++)
		{
			for (unsigned int k = 0; k < _qNNodes; k++)
			{
				tildeMrDash(j, (m - 1) * _qNNodes + k) = dgtoolbox::mapRefToPhys<active>(_radDelta, elemIdx, _qNodes[k]) * static_cast<double>(_curRadialDispersionTilde[elemIdx * _qNNodes + k]) * ellEll(m, k) * ellEll(j, k) * _qWeights[k];
			}
		}
	}

	return tildeMrDash;
}

MatrixXd TwoDimensionalConvectionDispersionOperatorDG::tildeSrDash(const unsigned int elemIdx)
{
	MatrixXd ellEll = MatrixXd::Zero(_radNNodes, _qNNodes);
	MatrixXd tildeSrDash = MatrixXd::Zero(_radNNodes, _radNNodes * _qNNodes);

	for (unsigned int j = 0; j < _radNNodes; j++)
		ellEll.row(j) = dgtoolbox::evalLagrangeBasis(j, _radNodes, _qNodes);

	MatrixXd tildeDr = dgtoolbox::derivativeMatrix(_quadratureOrder, _qNodes);
	MatrixXd tildeSr = _qWeights.diagonal() * tildeDr;

	for (unsigned int j = 0; j < _radNNodes; j++)
	{
		for (unsigned int m = 0; m < _radNNodes; m++)
		{
			for (unsigned int k = 0; k < _qNNodes; k++)
			{
				tildeSrDash(j, (m - 1) * _qNNodes + k) = dgtoolbox::mapRefToPhys<active>(_radDelta, elemIdx, _qNodes) * _curRadialDispersionTilde[k] * tildeSr(j, k) * ellEll(m, k);
			}
		}
	}

	return tildeSrDash;
}

void TwoDimensionalConvectionDispersionOperatorDG::initializeDG()
{
	dgtoolbox::lglNodesWeights(_axPolyDeg, _axNodes, _axInvWeights, true);
	dgtoolbox::lglNodesWeights(_radPolyDeg, _radNodes, _radInvWeights, true);
	dgtoolbox::lglNodesWeights(_quadratureOrder, _qNodes, _qWeights, false);

	//_axPolyDerM = dgtoolbox::derivativeMatrix(_axPolyDeg, _axNodes);
	//_radPolyDerM = dgtoolbox::derivativeMatrix(_radPolyDeg, _radNodes);

	VectorXd radBaryWeights = dgtoolbox::barycentricWeights(_quadratureOrder, _qNodes);
	_interpolationM = dgtoolbox::polynomialInterpolationMatrix(_qNodes, _radNodes, radBaryWeights);

	// auxiliary equation
	_axG = VectorXd::Zero(_axNPoints);
	_radG = VectorXd::Zero(_radNPoints);

	_axStiffM = dgtoolbox::stiffnessMatrix(_axPolyDeg, _axNodes, 0.0, 0.0);
	_radStiffM = dgtoolbox::stiffnessMatrix(_radPolyDeg, _radNodes, 0.0, 0.0);

	_axLiftM = dgtoolbox::liftingMatrix(_axNNodes);
	_radLiftM = dgtoolbox::liftingMatrix(_radNNodes);

	// main equation operators: cylindrical coordinates and parameters
	_tildeMr = new MatrixXd[_radNElem];
	_tildeMrDash = new MatrixXd[_radNElem];
	_tildeSrDash = new MatrixXd[_radNElem];
	for (unsigned int rElem = 0; rElem < _radNElem; rElem++)
	{
		_tildeMr[rElem] = tildeMr(rElem);
		_tildeMrDash[rElem] = tildeMrDash(rElem);
		_tildeSrDash[rElem] = tildeMrDash(rElem);
	}

}

/**
 * @brief Reads parameters and allocates memory
 * @details Has to be called once before the operator is used.
 * @param [in] paramProvider Parameter provider for reading parameters
 * @param [in] nComp Number of components
 * @param [in] axNPoints Number of axial cells
 * @param [in] dynamicReactions Determines whether the sparsity pattern accounts for dynamic reactions
 * @return @c true if configuration went fine, @c false otherwise
 */
bool TwoDimensionalConvectionDispersionOperatorDG::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, unsigned int axNodeStride, unsigned int radNodeStride, bool dynamicReactions)
{
	_nComp = nComp;
	_hasDynamicReactions = dynamicReactions;

	// TODO: Add support for parameter dependent dispersion
	_dispersionDep = helper.createParameterParameterDependence("CONSTANT_ONE");

	paramProvider.pushScope("discretization");

	if (paramProvider.exists("AX_POLYDEG"))
		_axPolyDeg = paramProvider.getInt("AX_POLYDEG");
	else
		_axPolyDeg = 4u; // default value
	if (paramProvider.getInt("AX_POLYDEG") < 1)
		throw InvalidParameterException("Axial polynomial degree must be at least 1!");
	else if (_axPolyDeg < 3)
		LOG(Warning) << "Polynomial degree > 2 in axial bulk discretization (cf. AX_POLYDEG) is always recommended for performance reasons.";

	_axNNodes = _axPolyDeg + 1;

	if (paramProvider.exists("AX_NELEM"))
		_radNElem = paramProvider.getInt("AX_NELEM");
	else if (paramProvider.exists("NCOL"))
		_radNElem = std::max(1u, paramProvider.getInt("NCOL") / _radNNodes); // number of elements is rounded down
	else
		throw InvalidParameterException("Specify field AX_NELEM (or NCOL)");

	if (_radNElem < 1)
		throw InvalidParameterException("Number of column elements must be at least 1!");

	_radNPoints = _radNNodes * _radNElem;

	if (paramProvider.exists("RAD_POLYDEG"))
		_radPolyDeg = paramProvider.getInt("RAD_POLYDEG");
	else
		_radPolyDeg = 4u; // default value
	if (paramProvider.getInt("RAD_POLYDEG") < 1)
		throw InvalidParameterException("Radial polynomial degree must be at least 1!");

	_radNNodes = _radPolyDeg + 1;

	if (paramProvider.exists("RAD_NELEM"))
		_radNElem = paramProvider.getInt("RAD_NELEM");
	else if (paramProvider.exists("NRAD"))
		_radNElem = std::max(1u, paramProvider.getInt("NRAD") / _radNNodes); // number of elements is rounded down
	else
		throw InvalidParameterException("Specify field RAD_NELEM (or NRAD)");

	if (_radNElem < 1)
		throw InvalidParameterException("Number of column elements must be at least 1!");

	_radNPoints = _radNNodes * _radNElem;

	if (paramProvider.exists("QUADRATURE_RULE"))
	{
		const std::string quadratureRule = paramProvider.getString("QUADRATURE_RULE");
		if (quadratureRule == "LOBATTO")
			_quadratureRule = 0;
		else if (quadratureRule == "GAUSS")
			_quadratureRule = 1;
		else
			throw InvalidParameterException("Unknown quadrature rule " + quadratureRule);
	}
	else
		_quadratureRule = 1;

	if (paramProvider.exists("QUADRATURE_ORDER"))
		_quadratureOrder = paramProvider.getInt("QUADRATURE_ORDER");
	else
		_quadratureOrder = _radPolyDeg; // todo or nNodes?

	paramProvider.popScope();

	_axNodeStride = axNodeStride;
	_axElemStride = _axNNodes * _axNodeStride;
	_radNodeStride = radNodeStride;
	_radElemStride = _radNNodes * _radNodeStride;

	_radialCoordinates.resize(_radNPoints + 1);
	_radialElemInterfaces.resize(_radNElem + 1);
	_radDelta.resize(_radNElem + 1);
	_nodalCrossSections.resize(_radNPoints);
	_curVelocity.resize(_radNPoints);

	setSparsityPattern();

	return _linearSolver->initialize(paramProvider, nComp, _axNPoints, _radNPoints);
}

/**
 * @brief Reads model parameters
 * @details Only reads parameters that do not affect model structure (e.g., discretization).
 * @param [in] unitOpIdx Unit operation id of the owning unit operation model
 * @param [in] paramProvider Parameter provider for reading parameters
 * @param [out] parameters Map in which local parameters are inserted
 * @return @c true if configuration went fine, @c false otherwise
 */
bool TwoDimensionalConvectionDispersionOperatorDG::configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters)
{
	// Read geometry parameters
	_colLength = paramProvider.getDouble("COL_LENGTH");
	_colRadius = paramProvider.getDouble("COL_RADIUS");
	readScalarParameterOrArray(_colPorosities, paramProvider, "COL_POROSITY", 1);

	if ((_colPorosities.size() != 1) && (_colPorosities.size() != _radNPoints))
		throw InvalidParameterException("Number of elements in field COL_POROSITY is neither 1 nor radNPoints (" + std::to_string(_radNPoints) + ")");

	_singlePorosity = (_colPorosities.size() == 1);
	if (_singlePorosity)
		_colPorosities = std::vector<active>(_radNPoints, _colPorosities[0]);

	// Read radial discretization mode and default to "EQUIDISTANT"
	paramProvider.pushScope("discretization");
	const std::string rdt = paramProvider.getString("RADIAL_DISC_TYPE");
	if (rdt == "EQUIVOLUME")
		_radialDiscretizationMode = RadialDiscretizationMode::Equivolume;
	else if (rdt == "USER_DEFINED")
	{
		_radialDiscretizationMode = RadialDiscretizationMode::UserDefined;
		readScalarParameterOrArray(_radialElemInterfaces, paramProvider, "RADIAL_COMPARTMENTS", 1);

		if (_radialElemInterfaces.size() < _radNElem + 1)
			throw InvalidParameterException("Number of elements in field RADIAL_COMPARTMENTS is less than radNElem + 1 (" + std::to_string(_radNElem + 1) + ")");

		registerParam1DArray(parameters, _radialElemInterfaces, [=](bool multi, unsigned int i) { return makeParamId(hashString("RADIAL_COMPARTMENTS"), unitOpIdx, CompIndep, i, BoundStateIndep, ReactionIndep, SectionIndep); });
	}
	else
		_radialDiscretizationMode = RadialDiscretizationMode::Equidistant;
	paramProvider.popScope();

	updateRadialDisc();

	// Read section dependent parameters (transport)

	// Read VELOCITY
	_velocity.clear();
	if (paramProvider.exists("VELOCITY"))
	{
		readScalarParameterOrArray(_velocity, paramProvider, "VELOCITY", 1);

		if (paramProvider.exists("VELOCITY_MULTIPLEX"))
		{
			const int mode = paramProvider.getInt("VELOCITY_MULTIPLEX");
			if (mode == 0)
				// Rad-indep, sec-indep
				_singleVelocity = true;
			else if (mode == 1)
				// Rad-dep, sec-indep
				_singleVelocity = false;
			else if (mode == 2)
				// Rad-indep, sec-dep
				_singleVelocity = true;
			else if (mode == 3)
				// Rad-dep, sec-dep
				_singleVelocity = false;

			if (!_singleVelocity && (_velocity.size() % _radNPoints != 0))
				throw InvalidParameterException("Number of elements in field VELOCITY is not a positive multiple of radNPoints (" + std::to_string(_radNPoints) + ")");
			if ((mode == 0) && (_velocity.size() != 1))
				throw InvalidParameterException("Number of elements in field VELOCITY inconsistent with VELOCITY_MULTIPLEX (should be 1)");
			if ((mode == 1) && (_velocity.size() != _radNPoints))
				throw InvalidParameterException("Number of elements in field VELOCITY inconsistent with VELOCITY_MULTIPLEX (should be " + std::to_string(_radNPoints) + ")");
		}
		else
		{
			// Infer radial dependence of VELOCITY:
			//   size not divisible by radNPoints -> radial independent
			_singleVelocity = ((_velocity.size() % _radNPoints) != 0);
		}

		// Expand _velocity to make it component dependent
		if (_singleVelocity)
		{
			std::vector<active> expanded(_velocity.size() * _radNPoints);
			for (std::size_t i = 0; i < _velocity.size(); ++i)
				std::fill(expanded.begin() + i * _radNPoints, expanded.begin() + (i + 1) * _radNPoints, _velocity[i]);

			_velocity = std::move(expanded);
		}
	}
	else
	{
		_singleVelocity = false;
		_velocity.resize(_radNPoints, 1.0);
	}

	// Register VELOCITY
	if (_singleVelocity)
	{
		if (_velocity.size() > _radNPoints)
		{
			// Register only the first item in each section
			for (std::size_t i = 0; i < _velocity.size() / _radNPoints; ++i)
				parameters[makeParamId(hashString("VELOCITY"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, i)] = &_velocity[i * _radNPoints];
		}
		else
		{
			// We have only one parameter
			parameters[makeParamId(hashString("VELOCITY"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_velocity[0];
		}
	}
	else
		registerParam2DArray(parameters, _velocity, [=](bool multi, unsigned int sec, unsigned int compartment) { return makeParamId(hashString("VELOCITY"), unitOpIdx, CompIndep, compartment, BoundStateIndep, ReactionIndep, multi ? sec : SectionIndep); }, _radNPoints);

	_dir = std::vector<int>(_radNPoints, 1);

	_axialDispersionMode = readAndRegisterMultiplexParam(paramProvider, parameters, _axialDispersion, "COL_DISPERSION", _nComp, _radNPoints, unitOpIdx);
	_radialDispersionMode = readAndRegisterMultiplexParam(paramProvider, parameters, _radialDispersion, "COL_DISPERSION_RADIAL", _nComp, _radNPoints, unitOpIdx);

	// Add parameters to map
	parameters[makeParamId(hashString("COL_LENGTH"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_colLength;
	parameters[makeParamId(hashString("COL_RADIUS"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_colRadius;
	registerParam1DArray(parameters, _colPorosities, [=](bool multi, unsigned int i) { return makeParamId(hashString("COL_POROSITY"), unitOpIdx, CompIndep, multi ? i : ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep); });

	// configure DG operators
	initializeDG();
	_curRadialDispersionTilde = std::vector<active>(_radNElem * _qNNodes, 0.0);
	const active* const curRadialDispersion = getSectionDependentSlice(_radialDispersion, _radNPoints * _nComp, 0);
	for (unsigned int i = 0; i < _interpolationM.rows(); i++) {
		for (unsigned int j = 0; j < _interpolationM.cols(); j++) {
			_curRadialDispersionTilde[i] += _interpolationM(i, j) * curRadialDispersion[j];
		}
	}
	
	return true;
}

/**
 * @brief Notifies the operator that a discontinuous section transition is in progress
 * @details In addition to changing flow direction internally, if necessary, the function returns whether
 *          the flow direction has changed.
 * @param [in] t Current time point
 * @param [in] secIdx Index of the new section that is about to be integrated
 * @return @c true if flow direction has changed, otherwise @c false
 */
bool TwoDimensionalConvectionDispersionOperatorDG::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx)
{
	bool hasChanged = false;

	// todo adapt operators to section dependent parameters
	const active* const curRadialDispersion = getSectionDependentSlice(_radialDispersion, _radNPoints * _nComp, 0);
	for (unsigned int i = 0; i < _interpolationM.rows(); i++) {
		for (unsigned int j = 0; j < _interpolationM.cols(); j++) {
			_curRadialDispersionTilde[i] += _interpolationM(i, j) * curRadialDispersion[j];
		}
	}

	if (!_velocity.empty())
	{
		// _curVelocity has already been set to the network flow rate in setFlowRates()
		// the direction of the flow (i.e., sign of _curVelocity) is given by _velocity
		active const* const dirNew = getSectionDependentSlice(_velocity, _radNPoints, secIdx);

		for (unsigned int i = 0; i < _radNPoints; ++i)
		{
			const int newDir = (dirNew[i] >= 0) ? 1 : -1;
			if (_dir[i] * newDir < 0)
			{
				hasChanged = true;
				_curVelocity[i] *= -1;
			}
			_dir[i] = newDir;
		}
	}

	// Change the sparsity pattern if necessary
	if ((secIdx == 0) || hasChanged)
		setSparsityPattern();

	return hasChanged || (secIdx == 0);
}

/**
 * @brief Sets the flow rates for the current time section
 * @details The flow rates may change due to valve switches.
 * @param [in] compartment Index of the compartment
 * @param [in] in Total volumetric inlet flow rate
 * @param [in] out Total volumetric outlet flow rate
 */
void TwoDimensionalConvectionDispersionOperatorDG::setFlowRates(int compartment, const active& in, const active& out) CADET_NOEXCEPT
{
	_curVelocity[compartment] = _dir[compartment] * in / (_nodalCrossSections[compartment] * _colPorosities[compartment]);
}

void TwoDimensionalConvectionDispersionOperatorDG::setFlowRates(active const* in, active const* out) CADET_NOEXCEPT
{
	for (unsigned int compartment = 0; compartment < _radNPoints; ++compartment)
		_curVelocity[compartment] = in[compartment] / (_nodalCrossSections[compartment] * _colPorosities[compartment]);
}

double TwoDimensionalConvectionDispersionOperatorDG::inletFactor(unsigned int idxSec, int idxRad) const CADET_NOEXCEPT
{
	const double h = static_cast<double>(_colLength) / static_cast<double>(_axNPoints);
	return -std::abs(static_cast<double>(_curVelocity[idxRad])) / h;
}

const active& TwoDimensionalConvectionDispersionOperatorDG::axialDispersion(unsigned int idxSec, int idxRad, int idxComp) const CADET_NOEXCEPT
{
	return *(getSectionDependentSlice(_axialDispersion, _radNPoints * _nComp, idxSec) + idxRad * _nComp + idxComp);
}

const active& TwoDimensionalConvectionDispersionOperatorDG::radialDispersion(unsigned int idxSec, int idxRad, int idxComp) const CADET_NOEXCEPT
{
	return *(getSectionDependentSlice(_radialDispersion, _radNPoints * _nComp, idxSec) + idxRad * _nComp + idxComp);
}

/**
 * @brief Computes the residual of the transport equations
 * @param [in] t Current time point
 * @param [in] secIdx Index of the current section
 * @param [in] y Pointer to unit operation's state vector
 * @param [in] yDot Pointer to unit operation's time derivative state vector
 * @param [out] res Pointer to unit operation's residual vector
 * @param [in] wantJac Determines whether the Jacobian is computed or not
 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
 */
int TwoDimensionalConvectionDispersionOperatorDG::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, bool wantJac, WithoutParamSensitivity)
{
	if (wantJac)
		return residualImpl<double, double, double, true>(model, t, secIdx, y, yDot, res);
	else
		return residualImpl<double, double, double, false>(model, t, secIdx, y, yDot, res);
}

int TwoDimensionalConvectionDispersionOperatorDG::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithoutParamSensitivity)
{
	if (wantJac)
		return residualImpl<active, active, double, true>(model, t, secIdx, y, yDot, res);
	else
		return residualImpl<active, active, double, false>(model, t, secIdx, y, yDot, res);
}

int TwoDimensionalConvectionDispersionOperatorDG::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity)
{
	if (wantJac)
		return residualImpl<double, active, active, true>(model, t, secIdx, y, yDot, res);
	else
		return residualImpl<double, active, active, false>(model, t, secIdx, y, yDot, res);
}

int TwoDimensionalConvectionDispersionOperatorDG::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity)
{
	if (wantJac)
		return residualImpl<active, active, active, true>(model, t, secIdx, y, yDot, res);
	else
		return residualImpl<active, active, active, false>(model, t, secIdx, y, yDot, res);
}

template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
int TwoDimensionalConvectionDispersionOperatorDG::residualImpl(const IModel& model, double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res)
{

	//MatrixMap map4(&array[0], 4, 4, Stride<Dynamic, Dynamic>(2, 3));

	Eigen::Map<Vector<StateType, Dynamic>, 0, InnerStride<>> _g(reinterpret_cast<StateType*>(&_g[0]), _nPoints, InnerStride<>(1));
	Eigen::Map<Matrix<StateType, Dynamic, Dynamic>, 0, InnerStride<>> _g(reinterpret_cast<StateType*>(&_g[0]), _nPoints, InnerStride<>(1));


	if (wantJac)
	{
		// Reset Jacobian
		_jacC.setAll(0.0);
	}

	const unsigned int offsetC = _radNPoints * _nComp;

	active const* const d_rho = getSectionDependentSlice(_radialDispersion, _radNPoints * _nComp, secIdx);
	active const* const d_c = getSectionDependentSlice(_axialDispersion, _radNPoints * _nComp, secIdx) + i * _nComp;

	for (unsigned int zEidx = 0; zEidx < _axNElem; zEidx++)
	{
		for (unsigned int rEidx = 0; rEidx < _radNElem; rEidx++)
		{
			static_cast<ParamType>(_curVelocity[node]),
				_nComp* i,                        // Offset to the first component of the inlet DOFs in the local state vector
				_nComp* (_radNPoints + i),              // Offset to the first component of the first bulk cell in the local state vector

			// todo add jacobian blocks
			//if (wantJac)
			//else
		}
	}

	for (unsigned int col = 0; col < _axNPoints; ++col)
	{
		const unsigned int offsetColBlock = col * _radNPoints * _nComp;
		ResidualType* const resColBlock = res + offsetC + offsetColBlock;
		StateType const* const yColBlock = y + offsetC + offsetColBlock;

		for (unsigned int rad = 0; rad < _radNPoints; ++rad)
		{
			const unsigned int offsetToRadBlock = rad * _nComp;
			const unsigned int offsetColRadBlock = offsetColBlock + offsetToRadBlock;
			ResidualType* const resColRadBlock = resColBlock + offsetToRadBlock;
			StateType const* const yColRadBlock = yColBlock + offsetToRadBlock;

			active const* const localD_rho = d_rho + rad * _nComp;
			const ParamType invEps = 1.0 / static_cast<ParamType>(_colPorosities[rad]);
			const ParamType deltaRho = static_cast<ParamType>(_radialEdges[rad+1]) - static_cast<ParamType>(_radialEdges[rad]);
			const ParamType rho = static_cast<ParamType>(_radialCenters[rad]);
//			const ParamType rhoCentroid = static_cast<ParamType>(_radialCentroids[rad]);

			for (unsigned int comp = 0; comp < _nComp; ++comp)
			{
				const unsigned int offsetCur = offsetColRadBlock + comp;
				ResidualType* const resCur = resColRadBlock + comp;
				StateType const* const yCur = yColRadBlock + comp;
				const ParamType curFluxCoeff = static_cast<ParamType>(_colPorosities[rad]) * static_cast<ParamType>(localD_rho[comp]);

				if (cadet_unlikely(curFluxCoeff <= 0.0))
					continue;

				if (cadet_likely(rad > 0))
				{
					// Flow from inner cell
					const ParamType centerDiffRho = rho - static_cast<ParamType>(_radialCenters[rad-1]);
//					const ParamType centerDiffRho = rhoCentroid - static_cast<ParamType>(_radialCentroids[rad-1]);
					const ParamType innerFluxCoeff = static_cast<ParamType>(_colPorosities[rad-1]) * static_cast<ParamType>(localD_rho[-static_cast<int>(_nComp) + static_cast<int>(comp)]);

					if (cadet_likely(innerFluxCoeff > 0.0))
					{
//						const ParamType fluxCoeff = (centerDiffRho * curFluxCoeff * innerFluxCoeff) / (curFluxCoeff * (static_cast<ParamType>(_radialEdges[rad]) - static_cast<ParamType>(_radialCentroids[rad-1])) + innerFluxCoeff * (rhoCentroid - static_cast<ParamType>(_radialEdges[rad])));
						const ParamType fluxCoeff = (centerDiffRho * curFluxCoeff * innerFluxCoeff) / (curFluxCoeff * (static_cast<ParamType>(_radialEdges[rad]) - static_cast<ParamType>(_radialCenters[rad-1])) + innerFluxCoeff * (rho - static_cast<ParamType>(_radialEdges[rad])));
//						const ParamType fluxCoeff = static_cast<ParamType>(_colPorosities[0]) * static_cast<ParamType>(d_rho[0]);
						const ParamType finalFactor = invEps * (static_cast<ParamType>(_radialEdges[rad]) * fluxCoeff / centerDiffRho) / (rho * deltaRho);

						*resCur += finalFactor * (yCur[0] - yCur[-static_cast<int>(_nComp)]);
						if (wantJac)
						{
							_jacC.centered(offsetCur, 0) += static_cast<double>(finalFactor);
							_jacC.centered(offsetCur, -static_cast<int>(_nComp)) -= static_cast<double>(finalFactor);
						}
					}
				}

				if (cadet_likely(rad < _radNPoints - 1))
				{
					// Flow from outer cell
					const ParamType centerDiffRho = static_cast<ParamType>(_radialCenters[rad+1]) - rho;
//					const ParamType centerDiffRho = static_cast<ParamType>(_radialCentroids[rad+1]) - rhoCentroid;
					const ParamType outerFluxCoeff = static_cast<ParamType>(_colPorosities[rad+1]) * static_cast<ParamType>(localD_rho[_nComp + comp]);

					if (cadet_likely(outerFluxCoeff > 0.0))
					{
//						const ParamType fluxCoeff = (centerDiffRho * curFluxCoeff * outerFluxCoeff) / (curFluxCoeff * (static_cast<ParamType>(_radialCentroids[rad+1] - static_cast<ParamType>(_radialEdges[rad+1]))) + outerFluxCoeff * (static_cast<ParamType>(_radialEdges[rad+1]) - rhoCentroid));
						const ParamType fluxCoeff = (centerDiffRho * curFluxCoeff * outerFluxCoeff) / (curFluxCoeff * (static_cast<ParamType>(_radialCenters[rad+1] - static_cast<ParamType>(_radialEdges[rad+1]))) + outerFluxCoeff * (static_cast<ParamType>(_radialEdges[rad+1]) - rho));
//						const ParamType fluxCoeff = static_cast<ParamType>(_colPorosities[0]) * static_cast<ParamType>(d_rho[0]);
						const ParamType finalFactor = invEps * (static_cast<ParamType>(_radialEdges[rad+1]) * fluxCoeff / centerDiffRho) / (rho * deltaRho);

						*resCur += finalFactor * (yCur[0] - yCur[_nComp]);
						if (wantJac)
						{
							_jacC.centered(offsetCur, 0) += static_cast<double>(finalFactor);
							_jacC.centered(offsetCur, _nComp) -= static_cast<double>(finalFactor);
						}
					}
				}
			}
		}
	}

	return 0;
}

void TwoDimensionalConvectionDispersionOperatorDG::setSparsityPattern()
{
	// Note that we have to increase the lower non-zeros by 1 because the WENO stencil is applied to the
	// right cell face (lower + 1 + upper) and to the left cell face (shift the stencil by -1 because influx of cell i
	// is outflux of cell i-1)
	// We also have to make sure that there's at least one sub and super diagonal for the dispersion term
	const unsigned int lowerNonZeros = std::max(_weno.lowerBandwidth() + 1u, 1u);
	const unsigned int upperNonZeros = std::max(_weno.upperBandwidth(), 1u);
	// Total number of non-zeros per row is WENO stencil (lowerNonZeros + 1u + upperNonZeros) + radial dispersion (2)
	cadet::linalg::SparsityPattern pattern(_nComp * _axNPoints * _radNPoints, lowerNonZeros + 1u + upperNonZeros + 2u);

	// Handle convection, axial dispersion (WENO)
	for (unsigned int i = 0; i < _radNPoints; ++i)
		cadet::model::parts::convdisp::sparsityPatternAxial(pattern.row(i * _nComp), _nComp, _axNPoints, _nComp * _radNPoints, static_cast<double>(_curVelocity[i]), _weno);

	// Handle radial dispersion
	if (_radNPoints > 1)
	{
		for (unsigned int col = 0; col < _axNPoints; ++col)
		{
			const unsigned int idxColBlock = col * _radNPoints * _nComp;

			// First and last cell have only one term
			for (unsigned int comp = 0; comp < _nComp; ++comp)
			{
				// Radial cell 1
				pattern.add(idxColBlock + comp, idxColBlock + comp + _nComp);

				// Radial cell radNPoints - 1
				pattern.add(idxColBlock + (_radNPoints - 1) * _nComp + comp, idxColBlock + comp + (_radNPoints - 2) * _nComp);
			}

			// Middle cells have two terms
			for (unsigned int rad = 1; rad < _radNPoints - 1; ++rad)
			{
				const unsigned int idxColRadBlock = idxColBlock + rad * _nComp;
				for (unsigned int comp = 0; comp < _nComp; ++comp)
				{
					const unsigned int idxCur = idxColRadBlock + comp;
					pattern.add(idxCur, idxCur + _nComp);
					pattern.add(idxCur, idxCur - _nComp);
				}
			}
		}
	}

	// Add space for dynamic reactions
	if (_hasDynamicReactions)
	{
		// Add nComp x nComp diagonal blocks (everything can react with everything)
		for (unsigned int col = 0; col < _axNPoints; ++col)
		{
			const unsigned int idxColBlock = col * _radNPoints * _nComp;

			for (unsigned int rad = 0; rad < _radNPoints; ++rad)
			{
				const unsigned int idxColRadBlock = idxColBlock + rad * _nComp;

				for (unsigned int comp = 0; comp < _nComp; ++comp)
				{
					const unsigned int idxCur = idxColRadBlock + comp;
					for (unsigned int comp2 = 0; comp2 < _nComp; ++comp2)
						pattern.add(idxCur, idxColRadBlock + comp2);
				}
			}
		}
	}

	_jacC.assignPattern(pattern);
	_linearSolver->setSparsityPattern(pattern);
}

/**
 * @brief Multiplies the time derivative Jacobian @f$ \frac{\partial F}{\partial \dot{y}}\left(t, y, \dot{y}\right) @f$ with a given vector
 * @details The operation @f$ z = \frac{\partial F}{\partial \dot{y}} x @f$ is performed.
 *          The matrix-vector multiplication is performed matrix-free (i.e., no matrix is explicitly formed).
 *          
 *          Note that this function only performs multiplication with the Jacobian of the (axial) transport equations.
 *          The input vectors are assumed to point to the beginning (including inlet DOFs) of the respective unit operation's arrays.
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in] sDot Vector @f$ x @f$ that is transformed by the Jacobian @f$ \frac{\partial F}{\partial \dot{y}} @f$
 * @param [out] ret Vector @f$ z @f$ which stores the result of the operation
 */
void TwoDimensionalConvectionDispersionOperatorDG::multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const
{
	double* localRet = ret + _nComp * _radNPoints;
	double const* localSdot = sDot + _nComp * _radNPoints;
	for (unsigned int i = 0; i < _axNPoints * _nComp * _radNPoints; ++i)
		localRet[i] = localSdot[i];
}

/**
 * @brief Assembles the axial transport Jacobian @f$ J_0 @f$ of the time-discretized equations
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
 */
void TwoDimensionalConvectionDispersionOperatorDG::assembleDiscretizedJacobian(double alpha)
{
	_linearSolver->assembleDiscretizedJacobian(alpha);
}

/**
 * @brief Assembles and factorizes the time discretized Jacobian
 * @details See assembleDiscretizedJacobian() for assembly of the time discretized Jacobian.
 * @param [in] alpha Factor in front of @f$ \frac{\partial F}{\partial \dot{y}} @f$
 * @return @c true if factorization went fine, otherwise @c false
 */
bool TwoDimensionalConvectionDispersionOperatorDG::assembleAndFactorizeDiscretizedJacobian(double alpha)
{
	assembleDiscretizedJacobian(alpha);
	return _linearSolver->factorize();
}

/**
 * @brief Solves a (previously factorized) equation system
 * @details The (time discretized) Jacobian matrix has to be factorized before calling this function.
 *          Note that the given right hand side vector @p rhs is not shifted by the inlet DOFs. That
 *          is, it is assumed to point directly to the first axial DOF.
 * 
 * @param [in,out] rhs On entry, right hand side of the equation system. On exit, solution of the system.
 * @return @c true if the system was solved correctly, otherwise @c false
 */
bool TwoDimensionalConvectionDispersionOperatorDG::solveDiscretizedJacobian(double* rhs, double const* weight, double const* init, double outerTol) const
{
	return _linearSolver->solveDiscretizedJacobian(rhs, weight, init, outerTol);
}

/**
 * @brief Solves a system with the time derivative Jacobian and given right hand side
 * @details Note that the given right hand side vector @p rhs is not shifted by the inlet DOFs. That
 *          is, it is assumed to point directly to the first axial DOF.
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in,out] rhs On entry, right hand side. On exit, solution of the system.
 * @return @c true if the system was solved correctly, @c false otherwise
 */
bool TwoDimensionalConvectionDispersionOperatorDG::solveTimeDerivativeSystem(const SimulationTime& simTime, double* const rhs)
{
	return true;
}

void TwoDimensionalConvectionDispersionOperatorDG::setEquidistantRadialDisc()
{
	const active h = _colRadius / _radNElem;
	const double pi = 3.1415926535897932384626434;

	std::fill(_radDelta.begin(), _radDelta.end(), h);

	_radialElemInterfaces[0] = 0.0;
	for (unsigned int r = 0; r < _radNElem; ++r)
	{
		// Set last edge to _colRadius for exact geometry
		if (r == _radNElem - 1)
			_radialElemInterfaces[r+1] = _colRadius;
		else
			_radialElemInterfaces[r+1] = h * (r + 1);

		// todo not needed?
		//_elementCrossSections[r] = pi * (_radialElemInterfaces[r + 1] * _radialElemInterfaces[r + 1] - _radialElemInterfaces[r] * _radialElemInterfaces[r]);
		active leftEnd = dgtoolbox::mapRefToPhys(_radDelta, r, 0);
		for (unsigned int node = 0;node < _radNNodes; ++node)
			_nodalCrossSections[r * _radNNodes + node] = pi * (pow(leftEnd + (1.0 / _radInvWeights[node]), 2.0) - pow(leftEnd, 2.0));
	}
}

void TwoDimensionalConvectionDispersionOperatorDG::setEquivolumeRadialDisc()
{
	const active volPerElement = _colRadius * _colRadius / _radNElem;
	const double pi = 3.1415926535897932384626434;

	_radialElemInterfaces[0] = 0.0;
	for (unsigned int r = 0; r < _radNElem; ++r)
	{
		// Set last edge to _colRadius for exact geometry
		if (r == _radNElem - 1)
			_radialElemInterfaces[r+1] = _colRadius;
		else
			_radialElemInterfaces[r+1] = sqrt(volPerElement + _radialElemInterfaces[r] * _radialElemInterfaces[r]);

		_radDelta[r] = _radialElemInterfaces[r + 1] - _radialElemInterfaces[r];
		// todo not needed?
		//_elementCrossSections[r] = pi * volPerCompartment;
		active leftEnd = dgtoolbox::mapRefToPhys(_radDelta, r, 0);
		for (unsigned int node = 0; node < _radNNodes; ++node)
			_nodalCrossSections[r * _radNNodes + node] = pi * (pow(leftEnd + (1.0 / _radInvWeights[node]), 2.0) - pow(leftEnd, 2.0));
	}
}

void TwoDimensionalConvectionDispersionOperatorDG::setUserdefinedRadialDisc()
{
	const double pi = 3.1415926535897932384626434;
	for (unsigned int r = 0; r < _radNElem; ++r)
	{
		_radDelta[r] = _radialElemInterfaces[r + 1] - _radialElemInterfaces[r];
		// todo not needed?
		//_elementCrossSections[r] = pi * (_radialEdges[r+1] * _radialEdges[r+1] - _radialEdges[r] * _radialEdges[r]);
		active leftEnd = dgtoolbox::mapRefToPhys(_radDelta, r, 0);
		for (unsigned int node = 0; node < _radNNodes; ++node)
			_nodalCrossSections[r * _radNNodes + node] = pi * (pow(leftEnd + (1.0 / _radInvWeights[node]), 2.0) - pow(leftEnd, 2.0));
	}
}

void TwoDimensionalConvectionDispersionOperatorDG::updateRadialDisc()
{
	if (_radialDiscretizationMode == RadialDiscretizationMode::Equidistant)
		setEquidistantRadialDisc();
	else if (_radialDiscretizationMode == RadialDiscretizationMode::Equivolume)
		setEquivolumeRadialDisc();
	else if (_radialDiscretizationMode == RadialDiscretizationMode::UserDefined)
		setUserdefinedRadialDisc();
}

bool TwoDimensionalConvectionDispersionOperatorDG::setParameter(const ParameterId& pId, double value)
{
	if (_singlePorosity && (pId.name == hashString("COL_POROSITY")) && (pId.component == CompIndep) && (pId.boundState == BoundStateIndep)
		&& (pId.reaction == ReactionIndep) && (pId.section == SectionIndep) && (pId.particleType == ParTypeIndep))
	{
		for (unsigned int i = 0; i < _radNPoints; ++i)
			_colPorosities[i].setValue(value);
		return true;
	}

	if (_singleVelocity && (pId.name == hashString("VELOCITY")) && (pId.component == CompIndep) && (pId.boundState == BoundStateIndep) && (pId.reaction == ReactionIndep))
	{
		if (_velocity.size() > _radNPoints)
		{
			// Section dependent
			if (pId.section == SectionIndep)
				return false;

			for (unsigned int i = 0; i < _radNPoints; ++i)
				_velocity[pId.section * _radNPoints + i].setValue(value);
		}
		else
		{
			// Section independent
			if (pId.section != SectionIndep)
				return false;

			for (unsigned int i = 0; i < _radNPoints; ++i)
				_velocity[i].setValue(value);
		}
	}

	const bool ad = multiplexParameterValue(pId, hashString("COL_DISPERSION"), _axialDispersionMode, _axialDispersion, _nComp, _radNPoints, value, nullptr);
	if (ad)
		return true;

	const bool adr = multiplexParameterValue(pId, hashString("COL_DISPERSION_RADIAL"), _radialDispersionMode, _radialDispersion, _nComp, _radNPoints, value, nullptr);
	if (adr)
		return true;

	return false;
}

bool TwoDimensionalConvectionDispersionOperatorDG::setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& pId, double value)
{
	if (_singlePorosity && (pId.name == hashString("COL_POROSITY")) && (pId.component == CompIndep) && (pId.boundState == BoundStateIndep)
		&& (pId.reaction == ReactionIndep) && (pId.section == SectionIndep) && (pId.particleType == ParTypeIndep))
	{
		if (contains(sensParams, &_colPorosities[0]))
		{
			for (unsigned int i = 0; i < _radNPoints; ++i)
				_colPorosities[i].setValue(value);
			return true;
		}
	}

	if (_singleVelocity && (pId.name == hashString("VELOCITY")) && (pId.component == CompIndep) && (pId.boundState == BoundStateIndep) && (pId.reaction == ReactionIndep))
	{
		if (_velocity.size() > _radNPoints)
		{
			// Section dependent
			if ((pId.section == SectionIndep) || !contains(sensParams, &_velocity[pId.section * _radNPoints]))
				return false;

			for (unsigned int i = 0; i < _radNPoints; ++i)
				_velocity[pId.section * _radNPoints + i].setValue(value);
		}
		else
		{
			// Section independent
			if ((pId.section != SectionIndep) || !contains(sensParams, &_velocity[0]))
				return false;

			for (unsigned int i = 0; i < _radNPoints; ++i)
				_velocity[i].setValue(value);
		}
	}

	const bool ad = multiplexParameterValue(pId, hashString("COL_DISPERSION"), _axialDispersionMode, _axialDispersion, _nComp, _radNPoints, value, &sensParams);
	if (ad)
		return true;

	const bool adr = multiplexParameterValue(pId, hashString("COL_DISPERSION_RADIAL"), _radialDispersionMode, _radialDispersion, _nComp, _radNPoints, value, &sensParams);
	if (adr)
		return true;

	return false;
}

bool TwoDimensionalConvectionDispersionOperatorDG::setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue)
{
	if (_singlePorosity && (pId.name == hashString("COL_POROSITY")) && (pId.component == CompIndep) && (pId.boundState == BoundStateIndep)
		&& (pId.reaction == ReactionIndep) && (pId.section == SectionIndep) && (pId.particleType == ParTypeIndep))
	{
		sensParams.insert(&_colPorosities[0]);
		for (unsigned int i = 0; i < _radNPoints; ++i)
			_colPorosities[i].setADValue(adDirection, adValue);

		return true;
	}

	if (_singleVelocity && (pId.name == hashString("VELOCITY")) && (pId.component == CompIndep) && (pId.boundState == BoundStateIndep) && (pId.reaction == ReactionIndep))
	{
		if (_velocity.size() > _radNPoints)
		{
			// Section dependent
			if (pId.section == SectionIndep)
				return false;

			sensParams.insert(&_velocity[pId.section * _radNPoints]);
			for (unsigned int i = 0; i < _radNPoints; ++i)
				_velocity[pId.section * _radNPoints + i].setADValue(adDirection, adValue);
		}
		else
		{
			// Section independent
			if (pId.section != SectionIndep)
				return false;

			sensParams.insert(&_velocity[0]);
			for (unsigned int i = 0; i < _radNPoints; ++i)
				_velocity[i].setADValue(adDirection, adValue);
		}
	}

	const bool ad = multiplexParameterAD(pId, hashString("COL_DISPERSION"), _axialDispersionMode, _axialDispersion, _nComp, _radNPoints, adDirection, adValue, sensParams);
	if (ad)
		return true;

	const bool adr = multiplexParameterAD(pId, hashString("COL_DISPERSION_RADIAL"), _radialDispersionMode, _radialDispersion, _nComp, _radNPoints, adDirection, adValue, sensParams);
	if (adr)
		return true;

	return false;
}

}  // namespace parts

}  // namespace model

}  // namespace cadet
