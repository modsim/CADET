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

#include "model/binding/BindingModelBase.hpp"
#include "model/ExternalFunctionSupport.hpp"
#include "model/ModelUtils.hpp"
#include "cadet/Exceptions.hpp"
#include "model/Parameters.hpp"
#include "LocalVector.hpp"
#include "SimulationTypes.hpp"

#include <functional>
#include <unordered_map>
#include <string>
#include <vector>

/*<codegen>
{
	"name": "LangmuirLDFParamHandler",
	"externalName": "ExtLangmuirLDFParamHandler",
	"parameters":
		[
			{ "type": "ScalarComponentDependentParameter", "varName": "keq", "confName": "MCLLDF_KEQ"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kkin", "confName": "MCLLDF_KKIN"},
			{ "type": "ScalarComponentDependentParameter", "varName": "qMax", "confName": "MCLLDF_QMAX"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 keq = Equillibrium constant
 kkin = Linear driving force
 qMax = Capacity
*/

namespace cadet
{

namespace model
{

inline const char* LangmuirLDFParamHandler::identifier() CADET_NOEXCEPT
{
	return "MULTI_COMPONENT_LANGMUIR_LDF";
}

inline bool LangmuirLDFParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_keq.size() != _kkin.size()) || (_keq.size() != _qMax.size()) || (_keq.size() < nComp))
		throw InvalidParameterException("MCLLDF_KEQ, MCLLDF_KKIN, and MCLLDF_QMAX have to have the same size");

	return true;
}

inline const char* ExtLangmuirLDFParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_MULTI_COMPONENT_LANGMUIR_LDF";
}

inline bool ExtLangmuirLDFParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_keq.size() != _kkin.size()) || (_keq.size() != _qMax.size()) || (_keq.size() < nComp))
		throw InvalidParameterException(
			"EXT_MCLLDF_KEQ, EXT_MCLLDF_KKIN, and EXT_MCLLDF_QMAX have to have the same size");

	return true;
}

/**
 * @brief Defines the multi component Langmuir binding model
 * @details Implements the Langmuir adsorption model: \f[ \begin{align}
 *              \frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i} q_{\text{max},i} \left( 1 - \sum_j
 * \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} q_i \end{align} \f] Multiple bound states are not supported.
 *          Components without bound state (i.e., non-binding components) are supported.
 *
 *          See @cite Langmuir1916.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t> class LangmuirLDFBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:
	LangmuirLDFBindingBase()
	{
	}
	virtual ~LangmuirLDFBindingBase() CADET_NOEXCEPT
	{
	}

	static const char* identifier()
	{
		return ParamHandler_t::identifier();
	}

	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT
	{
		return true;
	}

	/*virtual void timeDerivativeQuasiStationaryFluxes(double t, unsigned int secIdx, const ColumnPosition& colPos,
	double const* yCp, double const* y, double* dResDt, LinearBufferAllocator workSpace) const
	{
		if (!this->hasQuasiStationaryReactions())
			return;

		if (!ParamHandler_t::dependsOnTime())
			return;

		// Update external function and compute time derivative of parameters
		typename ParamHandler_t::ParamsHandle p;
		typename ParamHandler_t::ParamsHandle dpDt;
		std::tie(p, dpDt) = _paramHandler.updateTimeDerivative(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Protein fluxes: k_{kin,i}q_i - k_{kin,i} \frac{q_{m,i}k_{eq,i}c_i}{1+\sum_{j=1}^{n_{comp}} k_{eq,j}c_j}
		double cpSum = 1.0;
		double cpSumT = 0.0;
		unsigned int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const double summand = y[bndIdx] / static_cast<double>(p->qMax[i]);
			cpSum -= summand;
			cpSumT += summand / static_cast<double>(p->qMax[i]) * static_cast<double>(dpDt->qMax[i]);

			// Next bound component
			++bndIdx;
		}

		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			// Residual
			dResDt[bndIdx] = static_cast<double>(dpDt->kD[i]) * y[bndIdx]
				- yCp[i] * (static_cast<double>(dpDt->kA[i]) * static_cast<double>(p->qMax[i]) * cpSum
						   + static_cast<double>(p->kA[i]) * static_cast<double>(dpDt->qMax[i]) * cpSum
						   + static_cast<double>(p->kA[i]) * static_cast<double>(p->qMax[i]) * cpSumT);

			// Next bound component
			++bndIdx;
		}
	}*/

	CADET_BINDINGMODELBASE_BOILERPLATE

protected:
	using ParamHandlerBindingModelBase<ParamHandler_t>::_paramHandler;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_reactionQuasistationarity;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_nComp;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_nBoundStates;

	template <typename StateType, typename CpStateType, typename ResidualType, typename ParamType>
	int fluxImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* y,
				 CpStateType const* yCp, ResidualType* res, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Protein fluxes: k_{kin,i}q_i - k_{kin,i} \frac{q_{m,i}k_{eq,i}c_i}{1+\sum_{j=1}^{n_{comp}} k_{eq,j}c_j}
		ResidualType cpSum = 1.0;
		unsigned int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			cpSum += yCp[i] * static_cast<ParamType>(p->keq[i]);

			// Next bound component
			++bndIdx;
		}

		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			// Residual
			res[bndIdx] =
				static_cast<ParamType>(p->kkin[i]) *
				(y[bndIdx] - static_cast<ParamType>(p->keq[i]) * yCp[i] * static_cast<ParamType>(p->qMax[i]) / cpSum);

			// Next bound component
			++bndIdx;
		}

		return 0;
	}

	template <typename RowIterator>
	void jacobianImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp,
					  int offsetCp, RowIterator jac, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Protein fluxes: k_{kin,i}q_i - k_{kin,i} \frac{q_{m,i}k_{eq,i}c_i}{1+\sum_{j=1}^{n_{comp}} k_{eq,j}c_j}
		double cpSum = 1.0;
		int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			cpSum += yCp[i] * static_cast<double>(p->keq[i]);

			// Next bound component
			++bndIdx;
		}

		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const double keq = static_cast<double>(p->keq[i]);
			const double kkin = static_cast<double>(p->kkin[i]);
			//(keq *keq *static_cast<double>(p->qMax[i]) * yCp[i] / (cpSum*cpSum))
			// dres_i / dc_{p,i}
			jac[i - bndIdx - offsetCp] = -kkin * keq * static_cast<double>(p->qMax[i]) / cpSum;
			// Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +i to c_{p,i}.
			//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.

			// Fill dres_i / dc_j
			const double commonFactor = keq * kkin * static_cast<double>(p->qMax[i]) * yCp[i] / (cpSum * cpSum);
			int bndIdx2 = 0;
			for (int j = 0; j < _nComp; ++j)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[j] == 0)
					continue;

				// dres_i / dc_j
				jac[j - bndIdx - offsetCp] += static_cast<double>(p->keq[j]) * commonFactor;
				// Getting to c_{p,j}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +j to c_{p,j}.
				//                     This means jac[j - bndIdx - offsetCp] corresponds to c_{p,j}.

				++bndIdx2;
			}

			// Add to dres_i / dq_i
			jac[0] += kkin;

			// Advance to next flux and Jacobian row
			++bndIdx;
			++jac;
		}
	}
};

typedef LangmuirLDFBindingBase<LangmuirLDFParamHandler> LangmuirLDFBinding;
typedef LangmuirLDFBindingBase<ExtLangmuirLDFParamHandler> ExternalLangmuirLDFBinding;

namespace binding
{
void registerLangmuirLDFModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
{
	bindings[LangmuirLDFBinding::identifier()] = []() { return new LangmuirLDFBinding(); };
	bindings[ExternalLangmuirLDFBinding::identifier()] = []() { return new ExternalLangmuirLDFBinding(); };
}
} // namespace binding

} // namespace model

} // namespace cadet
