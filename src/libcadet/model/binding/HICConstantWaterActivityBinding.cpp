// =============================================================================
//  CADET
//
//  Copyright © 2008-2023: The CADET Authors
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
#include "model/binding/RefConcentrationSupport.hpp"
#include "model/ModelUtils.hpp"
#include "cadet/Exceptions.hpp"
#include "model/Parameters.hpp"
#include "LocalVector.hpp"
#include "SimulationTypes.hpp"

#include <cmath>
#include <functional>
#include <unordered_map>
#include <string>
#include <vector>

/*<codegen>
{
	"name": "HICCWAParamHandler",
	"externalName": "ExtHICCWAParamHandler",
	"parameters":
		[
			{ "type": "ScalarParameter", "varName": "beta0", "confName": "HICCWA_BETA0"},
			{ "type": "ScalarParameter", "varName": "beta1", "confName": "HICCWA_BETA1"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kA", "confName": "HICCWA_KA"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kD", "confName": "HICCWA_KD"},
			{ "type": "ScalarComponentDependentParameter", "varName": "nu", "confName": "HICCWA_NU"},
			{ "type": "ScalarComponentDependentParameter", "varName": "qMax", "confName": "HICCWA_QMAX"}
		],
	"constantParameters":
		[
			{ "type": "ScalarParameter", "varName": "waterActivity", "default": 0.1, "confName":
"HICCWA_WATER_ACTIVITY"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 beta0 = bulk-like ordered water molecules at infinitely diluted salt concentration
 beta1 = influence of salt concentration on bulk-like ordered water molecules
 kA = Adsorption constant
 kD = Desorption constant
 nu = Number of binding sites
 qMax = Maximum binding capacity
*/

namespace cadet
{

namespace model
{

inline const char* HICCWAParamHandler::identifier() CADET_NOEXCEPT
{
	return "HIC_CONSTANT_WATER_ACTIVITY";
}

inline bool HICCWAParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _qMax.size()) ||
		(_kA.size() < nComp))
		throw InvalidParameterException("HICCWA_KA, HICCWA_KD, HICCWA_NU, and HICCWA_QMAX have to have the same size");

	return true;
}

inline const char* ExtHICCWAParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_HIC_CONSTANT_WATER_ACTIVITY";
}

inline bool ExtHICCWAParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _qMax.size()) ||
		(_kA.size() < nComp))
		throw InvalidParameterException("KA, KD, NU, and QMAX have to have the same size");

	return true;
}

/**
 * @brief Defines the HIC Isotherm assuming a constant water activity as described by Jäpel and Buyel, 2022
 * @details Implements the the HIC Isotherm assuming a constant water activity: \f[ \begin{align}
 *				\beta &= \beta_0 e^{c_{p,0}\beta_1}\\
 *				\frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i} \left( 1 - \sum_j \frac{q_j}{q_{max,j}}
 *\right)^{\nu_i}
 *- k_{d,i} q_i 0.1^{\nu_i \beta} \end{align}  \f] Component @c 0 is assumed to be salt without a bound state. Multiple
 *bound states are not supported. Components without bound state (i.e., salt and non-binding components) are supported.
 *
 *          See @cite Jaepel2022.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t>
class ConstantWaterActivityBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:
	ConstantWaterActivityBindingBase()
	{
	}
	virtual ~ConstantWaterActivityBindingBase() CADET_NOEXCEPT
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
			throw InvalidParameterException(
				"HICCWA binding model requires exactly zero bound states for salt component");

		// First flux is salt, which is always quasi-stationary
		_reactionQuasistationarity[0] = false;

		return res;
	}

	virtual bool hasSalt() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool supportsMultistate() const CADET_NOEXCEPT
	{
		return false;
	}
	virtual bool supportsNonBinding() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool hasQuasiStationaryReactions() const CADET_NOEXCEPT
	{
		return false;
	}
	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT
	{
		return true;
	}

	virtual bool preConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y,
										   double const* yCp, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);
		return true;
	}

	virtual void postConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y,
											double const* yCp, LinearBufferAllocator workSpace) const
	{
		preConsistentInitialState(t, secIdx, colPos, y, yCp, workSpace);
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
		using CpStateParamType = typename DoubleActivePromoter<CpStateType, ParamType>::type;
		using StateParamType = typename DoubleActivePromoter<StateType, ParamType>::type;
		using std::pow, std::exp;

		typename ParamHandler_t::ParamsHandle const _p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		const ParamType beta0 = static_cast<ParamType>(_p->beta0);
		const ParamType beta1 = static_cast<ParamType>(_p->beta1);

		const CpStateParamType beta = beta0 * exp(beta1 * yCp[0]);

		StateParamType qSum = 1.0;

		unsigned int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			qSum -= y[bndIdx] / static_cast<ParamType>(_p->qMax[i]);

			// Next bound component
			++bndIdx;
		}

		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const ParamType kD = static_cast<ParamType>(_p->kD[i]);
			const ParamType kA = static_cast<ParamType>(_p->kA[i]);
			const ParamType nu = static_cast<ParamType>(_p->nu[i]);

			const CpStateParamType bulkWater = pow(static_cast<ParamType>(_p->waterActivity), nu * beta);
			const StateParamType qSumPowNu = pow(qSum, nu);

			res[bndIdx] = kD * y[bndIdx] * bulkWater - kA * qSumPowNu * yCp[i];

			// Next bound component
			++bndIdx;
		}

		return 0;
	}

	template <typename RowIterator>
	void jacobianImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp,
					  int offsetCp, RowIterator jac, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const _p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		const double beta0 = static_cast<double>(_p->beta0);
		const double beta1 = static_cast<double>(_p->beta1);

		const double beta = beta0 * std::exp(beta1 * yCp[0]);

		double qSum = 1.0;

		unsigned int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			qSum -= y[bndIdx] / static_cast<double>(_p->qMax[i]);

			// Next bound component
			++bndIdx;
		}

		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const double kD = static_cast<double>(_p->kD[i]);
			const double kA = static_cast<double>(_p->kA[i]);
			const double nu = static_cast<double>(_p->nu[i]);

			const double bulkWater = std::pow(static_cast<double>(_p->waterActivity), nu * beta);

			// dres_i / dc_{p,0}
			jac[-bndIdx - offsetCp] = std::log(static_cast<double>(_p->waterActivity)) * beta0 * beta1 * kD *
									  y[bndIdx] * nu * std::exp(beta1 * yCp[0]) * bulkWater;

			// dres_i / dc_{p,i}
			jac[i - bndIdx - offsetCp] = -kA * std::pow(qSum, nu);
			// Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +i to c_{p,i}.
			//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.

			// Fill dres_i / dq_j
			// could start j at 1, because j=1&bndIdx2=0 is the first bound component anyways
			int bndIdx2 = 0;
			for (int j = 0; j < _nComp; ++j)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[j] == 0)
					continue;

				// dres_i / dq_j
				jac[bndIdx2 - bndIdx] = kA * yCp[i] * nu * std::pow(qSum, nu - 1) / (static_cast<double>(_p->qMax[j]));
				// Getting to q_j: -bndIdx takes us to q_0, another +bndIdx2 to q_j. This means jac[bndIdx2 - bndIdx]
				// corresponds to q_j.

				++bndIdx2;
			}

			// Add to dres_i / dq_i
			jac[0] += kD * bulkWater;

			// Advance to next flux and Jacobian row
			++bndIdx;
			++jac;
		}
	}
};

typedef ConstantWaterActivityBindingBase<HICCWAParamHandler> ConstantWaterActivityBinding;
typedef ConstantWaterActivityBindingBase<ExtHICCWAParamHandler> ExternalConstantWaterActivityBinding;

namespace binding
{
void registerHICConstantWaterActivityModel(
	std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
{
	bindings[ConstantWaterActivityBinding::identifier()] = []() { return new ConstantWaterActivityBinding(); };
	bindings[ExternalConstantWaterActivityBinding::identifier()] = []() {
		return new ExternalConstantWaterActivityBinding();
	};
}
} // namespace binding

} // namespace model

} // namespace cadet
