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
#include "model/binding/BindingModelMacros.hpp"
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
	"name": "KumarLangmuirParamHandler",
	"externalName": "ExtKumarLangmuirParamHandler",
	"parameters":
		[
			{ "type": "ScalarParameter", "varName": "temperature", "confName": "KMCL_TEMP"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kA", "confName": "KMCL_KA"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kD", "confName": "KMCL_KD"},
			{ "type": "ScalarComponentDependentParameter", "varName": "qMax", "confName": "KMCL_QMAX"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kAct", "confName": "KMCL_KACT"},
			{ "type": "ScalarComponentDependentParameter", "varName": "nu", "confName": "KMCL_NU"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 temp = Temperature
 kA = Adsorption rate pre-exponential factor / frequency
 kD = Desorption rate
 qMax = Capacity
 kAct = Activation temperature @f$ k_{\text{act}} = \frac{E_a}{R} @f$
 nu = Salt exponents / characteristic charges @f$ \nu @f$
*/

namespace cadet
{

namespace model
{

inline const char* KumarLangmuirParamHandler::identifier() CADET_NOEXCEPT
{
	return "KUMAR_MULTI_COMPONENT_LANGMUIR";
}

inline bool KumarLangmuirParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kAct.size()) || (_kA.size() != _kD.size()) || (_kA.size() != _qMax.size()) ||
		(_kA.size() != _nu.size()) || (_kA.size() < nComp))
		throw InvalidParameterException(
			"KMCL_KA, KMCL_KD, KMCL_KACT, KMCL_NU, and KMCL_QMAX have to have the same size");

	return true;
}

inline const char* ExtKumarLangmuirParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_KUMAR_MULTI_COMPONENT_LANGMUIR";
}

inline bool ExtKumarLangmuirParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kAct.size()) || (_kA.size() != _kD.size()) || (_kA.size() != _qMax.size()) ||
		(_kA.size() != _nu.size()) || (_kA.size() < nComp))
		throw InvalidParameterException(
			"KMCL_KA, KMCL_KD, KMCL_KACT, KMCL_NU, and KMCL_QMAX have to have the same size");

	return true;
}

/**
 * @brief Defines the extended multi component Langmuir binding model used by Kumar et al.
 * @details Implements the extended Langmuir adsorption model: \f[ \begin{align}
 *              \frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} \exp\left( \frac{k_{\text{act},i}}{T} \right) c_{p,i}
 * q_{\text{max},i} \left( 1 - \sum_j \frac{q_j}{q_{\text{max},j}} \right) - \left( c_{p,0} \right)^{\nu_i} k_{d,i} q_i
 *          \end{align} \f]
 *          The first component @f$ c_{p,0} @f$ is assumed to be salt and should be set as non-binding (@c 0 bound
 * states). Multiple bound states are not supported. Components without bound state (i.e., non-binding components) are
 * supported.
 *
 *          In this model, the true adsorption rate @f$ k_{a,\text{true}} @f$ is governed by the Arrhenius law in order
 * to take temperature into account @f[ k_{a,\text{true}} = k_{a,i} \exp\left( \frac{k_{\text{act},i}}{T} \right). @f]
 *          Here, @f$ k_{a,i} @f$ is the frequency or pre-exponential factor and @f[ k_{\text{act},i} = \frac{E}{R} @f]
 * is the activation temperature (@f$ E @f$ denotes the activation energy and @f$ R @f$ the Boltzmann gas constant).
 *          Desorption is modified by salt (component @c 0) which does not bind. The characteristic charge @f$ \nu @f$
 *          of the protein is taken into account by the power law.
 *          See @cite Kumar2015 for details.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t> class KumarLangmuirBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:
	KumarLangmuirBindingBase()
	{
	}
	virtual ~KumarLangmuirBindingBase() CADET_NOEXCEPT
	{
	}

	static const char* identifier()
	{
		return ParamHandler_t::identifier();
	}

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp,
											  unsigned int const* nBound, unsigned int const* boundOffset)
	{
		const bool res = BindingModelBase::configureModelDiscretization(paramProvider, nComp, nBound, boundOffset);

		// Guarantee that salt has no bound state
		if (nBound[0] != 0)
			throw InvalidParameterException("Kumar-Langmuir binding model requires non-binding salt component");

		return res;
	}

	virtual bool hasSalt() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT
	{
		return true;
	}

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

		// Protein fluxes: -k_{a,i} * exp( k_{act,i} / T ) * c_{p,i} * q_{max,i} * (1 - \sum_j q_j / q_{max,j}) +
		// (c_{p,0})^{\nu_i} * k_{d,i} * q_i
		ResidualType qSum = 1.0;
		unsigned int bndIdx = 0;
		for (int i = 1; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			qSum -= y[bndIdx] / static_cast<ParamType>(p->qMax[i]);

			// Next bound component
			++bndIdx;
		}

		bndIdx = 0;
		for (int i = 1; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			// Residual
			const ResidualType ka = static_cast<ParamType>(p->kA[i]) *
									exp(static_cast<ParamType>(p->kAct[i]) / static_cast<ParamType>(p->temperature));
			const ResidualType kd = pow(yCp[0], static_cast<ParamType>(p->nu[i])) * static_cast<ParamType>(p->kD[i]);
			res[bndIdx] = kd * y[bndIdx] - ka * yCp[i] * static_cast<ParamType>(p->qMax[i]) * qSum;

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

		// Protein fluxes: -k_{a,i} * exp( k_{act,i} / T ) * c_{p,i} * q_{max,i} * (1 - \sum_j q_j / q_{max,j}) +
		// (c_{p,0})^{\nu_i} * k_{d,i} * q_i
		double qSum = 1.0;
		int bndIdx = 0;
		for (int i = 1; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			qSum -= y[bndIdx] / static_cast<double>(p->qMax[i]);

			// Next bound component
			++bndIdx;
		}

		bndIdx = 0;
		for (int i = 1; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const double ka = static_cast<double>(p->kA[i]) *
							  exp(static_cast<double>(p->kAct[i]) / static_cast<double>(p->temperature));
			const double kd = pow(yCp[0], static_cast<double>(p->nu[i])) * static_cast<double>(p->kD[i]);

			// dres_i / dc_{p,i}
			jac[i - bndIdx - offsetCp] = -ka * static_cast<double>(p->qMax[i]) * qSum;
			// Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +i to c_{p,i}.
			//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.

			// dres_i / dc_{p,0}
			jac[-bndIdx - offsetCp] = static_cast<double>(p->kD[i]) * static_cast<double>(p->nu[i]) *
									  pow(yCp[0], static_cast<double>(p->nu[i]) - 1.0) * y[bndIdx];

			// Fill dres_i / dq_j
			int bndIdx2 = 0;
			for (int j = 1; j < _nComp; ++j)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[j] == 0)
					continue;

				// dres_i / dq_j
				jac[bndIdx2 - bndIdx] = ka * yCp[i] * static_cast<double>(p->qMax[i]) / static_cast<double>(p->qMax[j]);
				// Getting to q_j: -bndIdx takes us to q_0, another +bndIdx2 to q_j. This means jac[bndIdx2 - bndIdx]
				// corresponds to q_j.

				++bndIdx2;
			}

			// Add to dres_i / dq_i
			jac[0] += kd;

			// Advance to next flux and Jacobian row
			++bndIdx;
			++jac;
		}
	}
};

typedef KumarLangmuirBindingBase<KumarLangmuirParamHandler> KumarLangmuirBinding;
typedef KumarLangmuirBindingBase<ExtKumarLangmuirParamHandler> ExternalKumarLangmuirBinding;

namespace binding
{
void registerKumarLangmuirModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
{
	bindings[KumarLangmuirBinding::identifier()] = []() { return new KumarLangmuirBinding(); };
	bindings[ExternalKumarLangmuirBinding::identifier()] = []() { return new ExternalKumarLangmuirBinding(); };
}
} // namespace binding

} // namespace model

} // namespace cadet
