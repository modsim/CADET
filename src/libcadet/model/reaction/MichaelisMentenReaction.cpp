// =============================================================================
//  CADET - The Chromatography Analysis and Design Toolkit
//
//  Copyright © 2008-2020: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include "model/reaction/ReactionModelBase.hpp"
#include "model/ExternalFunctionSupport.hpp"
#include "model/ModelUtils.hpp"
#include "cadet/Exceptions.hpp"
#include "model/Parameters.hpp"
#include "LocalVector.hpp"
#include "SimulationTypes.hpp"
#include "linalg/ActiveDenseMatrix.hpp"
#include "MathUtil.hpp"
#include "Memory.hpp"

#include <algorithm>
#include <unordered_map>
#include <string>
#include <vector>

/*<codegen>
{
	"name": "MichaelisMentenParamHandler",
	"externalName": "ExtMichaelisMentenParamHandler",
	"parameters":
		[
			{ "type": "ScalarReactionDependentParameter", "varName": "vMax", "confName": "MM_VMAX"},
			{ "type": "ScalarReactionDependentParameter", "varName": "kMM", "confName": "MM_KMM"},
			{ "type": "ComponentDependentReactionDependentParameter", "varName": "kInhibit", "confName": "MM_KI"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
*/


namespace cadet
{

namespace model
{

inline const char* MichaelisMentenParamHandler::identifier() CADET_NOEXCEPT { return "MICHAELIS_MENTEN"; }

inline bool MichaelisMentenParamHandler::validateConfig(unsigned int nReactions, unsigned int nComp, unsigned int const* nBoundStates)
{
	return true;
}

inline const char* ExtMichaelisMentenParamHandler::identifier() CADET_NOEXCEPT { return "EXT_MICHAELIS_MENTEN"; }

inline bool ExtMichaelisMentenParamHandler::validateConfig(unsigned int nReactions, unsigned int nComp, unsigned int const* nBoundStates)
{
	return true;
}

namespace
{
	/**
	 * @brief Registers a matrix-valued parameter (row-major storage) with components as rows
	 * @details The matrix-valued parameter has as many rows as there are components in the system.
	 * @param [in,out] parameters Parameter map
	 * @param [in] unitOpIdx Unit operation id
	 * @param [in] parTypeIdx Particle type index
	 * @param [in] paramName Name of the parameter
	 * @param [in] mat Matrix to register
	 */
	inline void registerCompRowMatrix(std::unordered_map<ParameterId, active*>& parameters, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, const std::string& paramName, cadet::linalg::ActiveDenseMatrix& mat)
	{
		const cadet::StringHash hashName = cadet::hashStringRuntime(paramName);
		cadet::registerParam2DArray(parameters, mat.data(), mat.elements(), [=](bool multi, unsigned int row, unsigned int col)
			{
				return cadet::makeParamId(hashName, unitOpIdx, row, parTypeIdx, cadet::BoundStateIndep, col, cadet::SectionIndep);
			},
			mat.columns()
		);
	}
}


/**
 * @brief Defines a Michaelis-Menten reaction kinetic with simple inhibition
 * @details Implements the Michaelis-Menten kinetics: \f[ \begin{align}
 *              S \nu,
 *          \end{align} \f]
 *          where \f$ S \f$ is the stoichiometric matrix and the fluxes are given by
 *          \f[ \begin{align}
 *              \nu_i = \frac{\mu_{\mathrm{max},i} c_S}{k_{\mathrm{MM},i} + c_S}.
 *          \end{align} \f]
 *          The substrate component \f$ c_S \f$ is identified by the index of the
 *          first negative entry in the stoichiometry of this reaction.
 *
 *          In addition, the reaction might be inhibited by other components. In this
 *          case, the flux has the form
 *          \f[ \begin{align}
 *              \nu_i = \frac{\mu_{\mathrm{max},i} c_S}{k_{\mathrm{MM},i} + c_S} \prod_j \frac{k_{\mathrm{I},i,j}}{k_{\mathrm{I},i,j} + c_{\mathrm{I},j}}.
 *          \end{align} \f]
 *          The value of \f$ k_{\mathrm{I},i,j} \f$ decides whether component \f$ j \f$
 *          inhibits reaction \f$ i \f$. If \f$ k_{\mathrm{I},i,j} < 0 \f$, the component
 *          does not inhibit the reaction.
 *
 *          Only reactions in liquid phase are supported (no solid phase or cross-phase reactions).
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t>
class MichaelisMentenReactionBase : public DynamicReactionModelBase
{
public:

	MichaelisMentenReactionBase() : _idxSubstrate(0) { }
	virtual ~MichaelisMentenReactionBase() CADET_NOEXCEPT { }

	static const char* identifier() { return ParamHandler_t::identifier(); }
	virtual const char* name() const CADET_NOEXCEPT { return ParamHandler_t::identifier(); }

	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size) { _paramHandler.setExternalFunctions(extFuns, size); }
	virtual bool dependsOnTime() const CADET_NOEXCEPT { return ParamHandler_t::dependsOnTime(); }
	virtual bool requiresWorkspace() const CADET_NOEXCEPT { return true; }
	virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		return _paramHandler.cacheSize(_stoichiometryBulk.columns(), nComp, totalNumBoundStates) + std::max(_stoichiometryBulk.columns() * sizeof(active), 2 * (_nComp + totalNumBoundStates) * sizeof(double));
	}

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBound, unsigned int const* boundOffset)
	{
		DynamicReactionModelBase::configureModelDiscretization(paramProvider, nComp, nBound, boundOffset);

		if (paramProvider.exists("MM_STOICHIOMETRY_BULK"))
		{
			const std::size_t numElements = paramProvider.numElements("MM_STOICHIOMETRY_BULK");
			if (numElements % nComp != 0)
				throw InvalidParameterException("Size of field MM_STOICHIOMETRY_BULK must be a positive multiple of NCOMP (" + std::to_string(nComp) + ")");

			const unsigned int nReactions = numElements / nComp;

			_stoichiometryBulk.resize(nComp, nReactions);
			_idxSubstrate = std::vector<int>(nReactions, -1);
		}

		return true;
	}

	virtual unsigned int numReactionsLiquid() const CADET_NOEXCEPT { return _stoichiometryBulk.columns(); }
	virtual unsigned int numReactionsCombined() const CADET_NOEXCEPT { return 0; }

	CADET_DYNAMICREACTIONMODEL_BOILERPLATE

protected:
	ParamHandler_t _paramHandler; //!< Handles parameters and their dependence on external functions

	linalg::ActiveDenseMatrix _stoichiometryBulk;
	std::vector<int> _idxSubstrate;

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx)
	{
		_paramHandler.configure(paramProvider, _stoichiometryBulk.columns(), _nComp, _nBoundStates);
		_paramHandler.registerParameters(_parameters, unitOpIdx, parTypeIdx, _nComp, _nBoundStates);

		if ((_stoichiometryBulk.columns() > 0) && ((_paramHandler.vMax().size() < _stoichiometryBulk.columns()) || (_paramHandler.kInhibit().size() < _stoichiometryBulk.columns()) || (_paramHandler.kMM().size() < _stoichiometryBulk.columns())))
			throw InvalidParameterException("MM_VMAX, MM_KI, and MM_KMM have to have the same size (number of reactions)");

		if (paramProvider.exists("MM_STOICHIOMETRY_BULK"))
		{
			const std::vector<double> s = paramProvider.getDoubleArray("MM_STOICHIOMETRY_BULK");

			if (s.size() != _stoichiometryBulk.elements())
				throw InvalidParameterException("MM_STOICHIOMETRY_BULK has changed size (number of reactions changed)");

			std::copy(s.begin(), s.end(), _stoichiometryBulk.data());

			// Find substrate index (first educt)
			for (int i = 0; i < _stoichiometryBulk.columns(); ++i)
			{
				_idxSubstrate[i] = -1;

				for (int j = 0; j < _nComp; ++j)
				{
					if (_stoichiometryBulk.native(j, i) < 0.0)
					{
						_idxSubstrate[i] = j;
						break;
					}
				}
			}
		}
		registerCompRowMatrix(_parameters, unitOpIdx, parTypeIdx, "MM_STOICHIOMETRY_BULK", _stoichiometryBulk);

		return true;
	}

	template <typename StateType, typename ResidualType, typename ParamType, typename FactorType>
	int residualLiquidImpl(double t, unsigned int secIdx, const ColumnPosition& colPos,
		StateType const* y, ResidualType* res, const FactorType& factor, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Calculate fluxes
		typedef typename DoubleActivePromoter<StateType, ParamType>::type flux_t;
		BufferedArray<flux_t> fluxes = workSpace.array<flux_t>(_stoichiometryBulk.columns());
		for (unsigned int r = 0; r < _stoichiometryBulk.columns(); ++r)
		{
			const int idxSubs = _idxSubstrate[r];
			if (idxSubs == -1)
			{
				fluxes[r] = 0.0;
				continue;
			}

			fluxes[r] = static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(p->vMax[r]) * y[idxSubs] / (static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(p->kMM[r]) + y[idxSubs]);

			for (int comp = 0; comp < _nComp; ++comp)
			{
				const flux_t kI = static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(p->kInhibit[_nComp * r + comp]);
				if (kI < 0.0)
					continue;

				fluxes[r] *= kI / (kI + y[comp]);
			}
		}

		// Add reaction terms to residual
		_stoichiometryBulk.multiplyVector(static_cast<flux_t*>(fluxes), factor, res);

		return 0;
	}

	template <typename StateType, typename ResidualType, typename ParamType>
	int residualCombinedImpl(double t, unsigned int secIdx, const ColumnPosition& colPos,
		StateType const* yLiquid, StateType const* ySolid, ResidualType* resLiquid, ResidualType* resSolid, double factor, LinearBufferAllocator workSpace) const
	{
		std::fill_n(resLiquid, _nComp, 0.0);

		if (_nTotalBoundStates == 0)
			return 0;

		std::fill_n(resSolid, _nTotalBoundStates, 0.0);

		return 0;
	}

	template <typename RowIterator>
	void jacobianLiquidImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor, const RowIterator& jac, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		for (unsigned int r = 0; r < _stoichiometryBulk.columns(); ++r)
		{
			const int idxSubs = _idxSubstrate[r];
			if (idxSubs == -1)
				continue;

			double inhibit = 1.0;
			for (int comp = 0; comp < _nComp; ++comp)
			{
				const double kI = static_cast<double>(p->kInhibit[_nComp * r + comp]);
				if (kI < 0.0)
					continue;

				inhibit *= kI / (kI + y[comp]);
			}

			const double vMax = static_cast<double>(p->vMax[r]);
			const double kMM = static_cast<double>(p->kMM[r]);
			const double denom = kMM + y[idxSubs];
			const double fluxGrad = vMax / denom * (1.0 - y[idxSubs] / denom) * inhibit;
			const double flux = vMax * y[idxSubs] / denom * inhibit;

			// Add gradients to Jacobian
			RowIterator curJac = jac;
			for (unsigned int row = 0; row < _nComp; ++row, ++curJac)
			{
				const double colFactor = static_cast<double>(_stoichiometryBulk.native(row, r)) * factor;
				curJac[idxSubs - static_cast<int>(row)] += colFactor * fluxGrad;
			}

			curJac = jac;
			for (unsigned int row = 0; row < _nComp; ++row, ++curJac)
			{
				const double colFactor = static_cast<double>(_stoichiometryBulk.native(row, r)) * factor;
				for (int comp = 0; comp < _nComp; ++comp)
				{
					const double kI = static_cast<double>(p->kInhibit[_nComp * r + comp]);
					if (kI < 0.0)
						continue;

					curJac[comp - static_cast<int>(row)] -= colFactor * flux / (kI + y[comp]);
				}
			}
		}
	}

	template <typename RowIteratorLiquid, typename RowIteratorSolid>
	void jacobianCombinedImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid, double factor, const RowIteratorLiquid& jacLiquid, const RowIteratorSolid& jacSolid, LinearBufferAllocator workSpace) const
	{
	}
};

typedef MichaelisMentenReactionBase<MichaelisMentenParamHandler> MichaelisMentenReaction;
typedef MichaelisMentenReactionBase<ExtMichaelisMentenParamHandler> ExternalMichaelisMentenReaction;

namespace reaction
{
	void registerMichaelisMentenReaction(std::unordered_map<std::string, std::function<model::IDynamicReactionModel*()>>& reactions)
	{
		reactions[MichaelisMentenReaction::identifier()] = []() { return new MichaelisMentenReaction(); };
		reactions[ExternalMichaelisMentenReaction::identifier()] = []() { return new ExternalMichaelisMentenReaction(); };
	}
}  // namespace reaction

}  // namespace model

}  // namespace cadet
