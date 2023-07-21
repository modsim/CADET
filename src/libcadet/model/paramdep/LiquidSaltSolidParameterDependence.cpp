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

#include "model/paramdep/ParameterDependenceBase.hpp"
#include "ParamReaderHelper.hpp"
#include "cadet/Exceptions.hpp"
#include "SimulationTypes.hpp"
#include "cadet/ParameterId.hpp"

#include <sstream>
#include <vector>
#include <iomanip>

namespace cadet
{

namespace model
{

namespace
{
	inline void readAndRegisterParameter(std::unordered_map<ParameterId, active*>& params, std::vector<active>& out, const std::string& name, IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, int nElements)
	{
		const StringHash nameHash = hashStringRuntime(name);

		if (parTypeIdx == ParTypeIndep)
		{
			const std::vector<double> tmp = paramProvider.getDoubleArray(name);
			if (static_cast<int>(tmp.size()) < nElements)
				throw InvalidParameterException(name + " contains too few elements (" + std::to_string(nElements) + " required)");

			out.reserve(nElements);
			for (int i = 0; i < nElements; ++i)
				out.push_back(tmp[i]);

			registerParam1DArray(params, out, [=](bool multi, unsigned int comp) { return makeParamId(nameHash, unitOpIdx, comp, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep); });
		}
		else
		{
			std::ostringstream oss;
			oss << name << "_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << parTypeIdx;

			if (paramProvider.exists(oss.str()))
			{
				const std::vector<double> tmp = paramProvider.getDoubleArray(oss.str());
				if (static_cast<int>(tmp.size()) < nElements)
					throw InvalidParameterException(oss.str() + " contains too few elements (" + std::to_string(nElements) + " required)");

				out.reserve(nElements);
				for (int i = 0; i < nElements; ++i)
					out.push_back(tmp[i]);

				registerParam1DArray(params, out, [=](bool multi, unsigned int comp) { return makeParamId(nameHash, unitOpIdx, comp, parTypeIdx, BoundStateIndep, ReactionIndep, SectionIndep); });
			}
			else
				throw InvalidParameterException("Field " + oss.str() + " not found");
		}
	}
}


/**
 * @brief Defines an exponential parameter dependence for the solid phase of a combined cell based on the liquid phase salt concentration
 */
class ExpLiquidSaltSolidParameterStateDependence : public ParameterStateDependenceBase
{
public:

	ExpLiquidSaltSolidParameterStateDependence() : _factor(0), _multiplicator(0) { }
	virtual ~ExpLiquidSaltSolidParameterStateDependence() CADET_NOEXCEPT { }

	static const char* identifier() { return "LIQUID_SALT_EXPONENTIAL"; }
	virtual const char* name() const CADET_NOEXCEPT { return ExpLiquidSaltSolidParameterStateDependence::identifier(); }

	virtual int jacobianElementsPerRowLiquid() const CADET_NOEXCEPT { return 0; }
	virtual int jacobianElementsPerRowCombined() const CADET_NOEXCEPT { return 1; }

	virtual void analyticJacobianLiquidAdd(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }
	virtual void analyticJacobianCombinedAddLiquid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }

	virtual void analyticJacobianCombinedAddSolid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const
	{
		jac(row, offset) += factor * param * static_cast<double>(_factor[bnd]) * exp(yLiquid[0] * static_cast<double>(_multiplicator[bnd])) * static_cast<double>(_multiplicator[bnd]);
	}

	CADET_PARAMETERSTATEDEPENDENCE_BOILERPLATE

protected:
	std::vector<active> _factor;
	std::vector<active> _multiplicator;

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, const std::string& name)
	{
		readAndRegisterParameter(_parameters, _factor, name + "_EXPFACTOR", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);
		readAndRegisterParameter(_parameters, _multiplicator, name + "_EXPARGMULT", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);

		return true;
	}

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type liquidParameterImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* y, int comp) const
	{
		return 0.0;
	}

	template <typename RowIterator>
	void analyticJacobianLiquidAddImpl(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, RowIterator jac) const { }

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type combinedParameterLiquidImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* yLiquid, StateType const* ySolid, int comp) const
	{
		return 0.0;
	}

	template <typename RowIterator>
	void analyticJacobianCombinedAddLiquidImpl(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, RowIterator jac) const { }

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type combinedParameterSolidImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* yLiquid, StateType const* ySolid, int bnd) const
	{
		return param * static_cast<ParamType>(_factor[bnd]) * exp(yLiquid[0] * static_cast<ParamType>(_multiplicator[bnd]));
	}

	template <typename RowIterator>
	void analyticJacobianCombinedAddSolidImpl(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, RowIterator jac) const
	{
		jac[offset - bnd - _nComp] += factor * param * static_cast<double>(_factor[bnd]) * exp(yLiquid[0] * static_cast<double>(_multiplicator[bnd])) * static_cast<double>(_multiplicator[bnd]);
	}
};


/**
 * @brief Defines a power law parameter dependence for the solid phase of a combined cell based on the liquid phase salt concentration
 */
class PowerLiquidSaltSolidParameterStateDependence : public ParameterStateDependenceBase
{
public:

	PowerLiquidSaltSolidParameterStateDependence() : _factor(0), _exponent(0) { }
	virtual ~PowerLiquidSaltSolidParameterStateDependence() CADET_NOEXCEPT { }

	static const char* identifier() { return "LIQUID_SALT_POWER"; }
	virtual const char* name() const CADET_NOEXCEPT { return PowerLiquidSaltSolidParameterStateDependence::identifier(); }

	virtual int jacobianElementsPerRowLiquid() const CADET_NOEXCEPT { return 0; }
	virtual int jacobianElementsPerRowCombined() const CADET_NOEXCEPT { return 1; }

	virtual void analyticJacobianLiquidAdd(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }
	virtual void analyticJacobianCombinedAddLiquid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }

	virtual void analyticJacobianCombinedAddSolid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const
	{
		jac(row, offset) += factor * param * static_cast<double>(_factor[bnd]) * pow(yLiquid[0], static_cast<double>(_exponent[bnd]) - 1.0) * static_cast<double>(_exponent[bnd]);
	}

	CADET_PARAMETERSTATEDEPENDENCE_BOILERPLATE

protected:
	std::vector<active> _factor;
	std::vector<active> _exponent;

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, const std::string& name)
	{
		readAndRegisterParameter(_parameters, _factor, name + "_POWFACTOR", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);
		readAndRegisterParameter(_parameters, _exponent, name + "_POWEXP", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);

		return true;
	}

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type liquidParameterImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* y, int comp) const
	{
		return 0.0;
	}

	template <typename RowIterator>
	void analyticJacobianLiquidAddImpl(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, RowIterator jac) const { }

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type combinedParameterLiquidImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* yLiquid, StateType const* ySolid, int comp) const
	{
		return 0.0;
	}

	template <typename RowIterator>
	void analyticJacobianCombinedAddLiquidImpl(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, RowIterator jac) const { }

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type combinedParameterSolidImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* yLiquid, StateType const* ySolid, int bnd) const
	{
		return param * static_cast<ParamType>(_factor[bnd]) * pow(yLiquid[0], static_cast<ParamType>(_exponent[bnd]));
	}

	template <typename RowIterator>
	void analyticJacobianCombinedAddSolidImpl(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, RowIterator jac) const
	{
		jac[offset - bnd - _nComp] += factor * param * static_cast<double>(_factor[bnd]) * pow(yLiquid[0], static_cast<double>(_exponent[bnd]) - 1.0) * static_cast<double>(_exponent[bnd]);
	}
};


/**
 * @brief Defines a parameter dependence for the solid phase of a combined cell based on the liquid phase salt concentration using the binding affinity of a colloidal binding model
 */
class ColloidalAffinityLiquidSaltSolidParameterStateDependence : public ParameterStateDependenceBase
{
public:

	ColloidalAffinityLiquidSaltSolidParameterStateDependence() : _lnKeqExp(0), _lnKeqFactor(0), _lnKeqConst(0), _powFactor(0), _powExponent(0), _expFactor(0), _expExponent(0) { }
	virtual ~ColloidalAffinityLiquidSaltSolidParameterStateDependence() CADET_NOEXCEPT { }

	static const char* identifier() { return "LIQUID_SALT_COLLOIDAL_AFFINITY"; }
	virtual const char* name() const CADET_NOEXCEPT { return ColloidalAffinityLiquidSaltSolidParameterStateDependence::identifier(); }

	virtual int jacobianElementsPerRowLiquid() const CADET_NOEXCEPT { return 0; }
	virtual int jacobianElementsPerRowCombined() const CADET_NOEXCEPT { return 1; }

	virtual void analyticJacobianLiquidAdd(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }
	virtual void analyticJacobianCombinedAddLiquid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }

	virtual void analyticJacobianCombinedAddSolid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const
	{
		const double logKeq = static_cast<double>(_lnKeqFactor[bnd]) * pow(yLiquid[0], static_cast<double>(_lnKeqExp[bnd])) + static_cast<double>(_lnKeqConst[bnd]);
		const double dLogKeq_dy = static_cast<double>(_lnKeqFactor[bnd]) * pow(yLiquid[0], static_cast<double>(_lnKeqExp[bnd]) - 1.0) * static_cast<double>(_lnKeqExp[bnd]);
		const double dPow_dy = static_cast<double>(_powFactor[bnd]) * pow(logKeq, static_cast<double>(_powExponent[bnd]) - 1.0) * static_cast<double>(_powExponent[bnd]);
		const double dExp_dy = static_cast<double>(_expFactor[bnd]) * exp(logKeq * static_cast<double>(_expExponent[bnd])) * static_cast<double>(_expExponent[bnd]);
		jac(row, offset) += factor * param * (dPow_dy + dExp_dy) * dLogKeq_dy;
	}

	CADET_PARAMETERSTATEDEPENDENCE_BOILERPLATE

protected:
	std::vector<active> _lnKeqExp;
	std::vector<active> _lnKeqFactor;
	std::vector<active> _lnKeqConst;
	std::vector<active> _powFactor;
	std::vector<active> _powExponent;
	std::vector<active> _expFactor;
	std::vector<active> _expExponent;

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, const std::string& name)
	{
		readAndRegisterParameter(_parameters, _lnKeqExp, name + "_LOGKEQEXP", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);
		readAndRegisterParameter(_parameters, _lnKeqFactor, name + "_LOGKEQFACTOR", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);
		readAndRegisterParameter(_parameters, _lnKeqConst, name + "_LOGKEQCONST", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);

		readAndRegisterParameter(_parameters, _powFactor, name + "_POWFACTOR", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);
		readAndRegisterParameter(_parameters, _powExponent, name + "_POWEXP", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);

		readAndRegisterParameter(_parameters, _expFactor, name + "_EXPFACTOR", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);
		readAndRegisterParameter(_parameters, _expExponent, name + "_EXPARGMULT", paramProvider, unitOpIdx, parTypeIdx, _nTotalBoundStates);

		return true;
	}

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type liquidParameterImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* y, int comp) const
	{
		return 0.0;
	}

	template <typename RowIterator>
	void analyticJacobianLiquidAddImpl(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, RowIterator jac) const { }

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type combinedParameterLiquidImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* yLiquid, StateType const* ySolid, int comp) const
	{
		return 0.0;
	}

	template <typename RowIterator>
	void analyticJacobianCombinedAddLiquidImpl(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, RowIterator jac) const { }

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type combinedParameterSolidImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* yLiquid, StateType const* ySolid, int bnd) const
	{
		const typename DoubleActivePromoter<StateType, ParamType>::type logKeq = static_cast<ParamType>(_lnKeqFactor[bnd]) * pow(yLiquid[0], static_cast<ParamType>(_lnKeqExp[bnd])) + static_cast<ParamType>(_lnKeqConst[bnd]);
		return param * (static_cast<ParamType>(_powFactor[bnd]) * pow(logKeq, static_cast<ParamType>(_powExponent[bnd])) + static_cast<ParamType>(_expFactor[bnd]) * exp(logKeq * static_cast<ParamType>(_expExponent[bnd])));
	}

	template <typename RowIterator>
	void analyticJacobianCombinedAddSolidImpl(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, RowIterator jac) const
	{
		const double logKeq = static_cast<double>(_lnKeqFactor[bnd]) * pow(yLiquid[0], static_cast<double>(_lnKeqExp[bnd])) + static_cast<double>(_lnKeqConst[bnd]);
		const double dLogKeq_dy = static_cast<double>(_lnKeqFactor[bnd]) * pow(yLiquid[0], static_cast<double>(_lnKeqExp[bnd]) - 1.0) * static_cast<double>(_lnKeqExp[bnd]);
		const double dPow_dy = static_cast<double>(_powFactor[bnd]) * pow(logKeq, static_cast<double>(_powExponent[bnd]) - 1.0) * static_cast<double>(_powExponent[bnd]);
		const double dExp_dy = static_cast<double>(_expFactor[bnd]) * exp(logKeq * static_cast<double>(_expExponent[bnd])) * static_cast<double>(_expExponent[bnd]);
		jac[offset - bnd - _nComp] += factor * param * (dPow_dy + dExp_dy) * dLogKeq_dy;
	}
};


namespace paramdep
{
	void registerLiquidSaltSolidParamDependence(std::unordered_map<std::string, std::function<model::IParameterStateDependence*()>>& paramDeps)
	{
		paramDeps[ExpLiquidSaltSolidParameterStateDependence::identifier()] = []() { return new ExpLiquidSaltSolidParameterStateDependence(); };
		paramDeps[PowerLiquidSaltSolidParameterStateDependence::identifier()] = []() { return new PowerLiquidSaltSolidParameterStateDependence(); };
		paramDeps[ColloidalAffinityLiquidSaltSolidParameterStateDependence::identifier()] = []() { return new ColloidalAffinityLiquidSaltSolidParameterStateDependence(); };
	}
}  // namespace paramdep

}  // namespace model

}  // namespace cadet
