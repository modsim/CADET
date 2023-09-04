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
	"name": "EMPMLangmuirParamHandler",
	"externalName": "ExtEMPMLangmuirParamHandler",
	"parameters":
		[
			{ "type": "ScalarComponentDependentParameter", "varName": "kA", "confName": "EMPM_KA"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kD", "confName": "EMPM_KD"},
			{ "type": "ScalarComponentDependentParameter", "varName": "qMax", "confName": "EMPM_QMAX"},
			{ "type": "ScalarComponentDependentParameter", "varName": "gamma", "confName": "EMPM_GAMMA"},
			{ "type": "ScalarComponentDependentParameter", "varName": "beta", "confName": "EMPM_BETA"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 kA = Adsorption rate
 kD = Desorption rate
 qMax = Capacity
 gamma = Hydrophobicity
 beta = Ion exchange characteristics
*/

namespace cadet
{

namespace model
{

inline const char* EMPMLangmuirParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXTENDED_MOBILE_PHASE_MODULATOR";
}

inline bool EMPMLangmuirParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kD.size()) || (_kA.size() != _qMax.size()) || (_kA.size() != _gamma.size()) ||
		(_kA.size() != _beta.size()) || (_kA.size() < nComp))
		throw InvalidParameterException(
			"EMPM_KA, EMPM_KD, EMPM_QMAX, EMPM_GAMMA, and EMPM_BETA have to have the same size");

	return true;
}

inline const char* ExtEMPMLangmuirParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_EXTENDED_MOBILE_PHASE_MODULATOR";
}

inline bool ExtEMPMLangmuirParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kD.size()) || (_kA.size() != _qMax.size()) || (_kA.size() != _gamma.size()) ||
		(_kA.size() != _beta.size()) || (_kA.size() < nComp))
		throw InvalidParameterException(
			"EMPM_KA, EMPM_KD, EMPM_QMAX, EMPM_GAMMA, and EMPM_BETA have to have the same size");

	return true;
}

/**
 * @brief Defines the mobile phase modulator Langmuir binding model
 * @details Implements the mobile phase modulator Langmuir adsorption model: \f[ \begin{align}
 *              \frac{\mathrm{d}q_0}{\mathrm{d}t} &= 0 \\
 *              \frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i} e^{\gamma_i c_{p,0}} q_{\text{max},i} \left( 1 -
 * \sum_j \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} c_{p,0}^{\beta_i} q_i \end{align} \f] While @f$ \gamma @f$
 * describes hydrophobicity, @f$ \beta @f$ accounts for ion-exchange characteristics. Multiple bound states are not
 * supported. Component @c 0 is assumed to be salt, which is also assumed to be inert. Components without bound state
 * (i.e., non-binding components) are supported.
 *
 *          Note that the first flux is only used if salt (component @c 0) has a bound state.
 *          It is reasonable to set the number of bound states for salt to @c 0 in order to save time and memory.
 *
 *          See @cite Melander1989 and @cite Karlsson2004.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t>
class ExtendedMobilePhaseModulatorLangmuirBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:
	ExtendedMobilePhaseModulatorLangmuirBindingBase()
	{
	}
	virtual ~ExtendedMobilePhaseModulatorLangmuirBindingBase() CADET_NOEXCEPT
	{
	}

	static const char* identifier()
	{
		return ParamHandler_t::identifier();
	}

	virtual bool hasSalt() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT
	{
		return true;
	}

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp,
											  unsigned int const* nBound, unsigned int const* boundOffset)
	{
		const bool res = BindingModelBase::configureModelDiscretization(paramProvider, nComp, nBound, boundOffset);

		_mode = paramProvider.getIntArray("EMPM_COMP_MODE");
		if (_mode.size() < nComp)
			throw InvalidParameterException("Not enough elements in EMPM_COMP_MODE (expected " + std::to_string(nComp) +
											", got " + std::to_string(_mode.size()) + ")");

		_idxModifier = -1;
		for (int i = 0; i < static_cast<int>(_mode.size()); ++i)
		{
			if (_mode[i] == static_cast<int>(CompMode::Modifier))
			{
				if (_idxModifier >= 0)
					throw InvalidParameterException("EMPM: Component " + std::to_string(_idxModifier) +
													" already set as modifier (requested comp " + std::to_string(i) +
													")");

				_idxModifier = i;
			}
			else if ((_mode[i] != static_cast<int>(CompMode::Linear)) &&
					 (_mode[i] != static_cast<int>(CompMode::Langmuir)))
				throw InvalidParameterException("Unknown EMPM_COMP_MODE for component " + std::to_string(i));
		}

		if (_idxModifier >= 0)
		{
			// If salt has a bound state, a dynamic reaction is added with zero flux
			if (_nBoundStates[_idxModifier] == 1)
				_reactionQuasistationarity[_idxModifier] = false;
		}

		return res;
	}

	CADET_BINDINGMODELBASE_BOILERPLATE

protected:
	using ParamHandlerBindingModelBase<ParamHandler_t>::_paramHandler;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_reactionQuasistationarity;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_nComp;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_nBoundStates;

	enum class CompMode : int
	{
		Modifier = 0,
		Linear = 1,
		Langmuir = 2
	};

	std::vector<int> _mode; //!< Mode of each component (e.g., linear or modified Langmuir)
	int _idxModifier;       //!< Index of the modifier component

	template <typename StateType, typename CpStateType, typename ResidualType, typename ParamType>
	int fluxImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* y,
				 CpStateType const* yCp, ResidualType* res, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Salt flux: 0
		// Protein fluxes: -k_{a,i} * exp(\gamma_i * c_{p,0}) * c_{p,i} * q_{max,i} * (1 - \sum q_i / q_{max,i}) +
		// k_{d,i} * c_{p,0}^\beta_i * q_i)
		ResidualType qSum = 1.0;
		int bndIdx = 0;

		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			// Also skip non-Langmuir components
			if (_mode[i] != static_cast<int>(CompMode::Langmuir))
			{
				++bndIdx;
				continue;
			}

			qSum -= y[bndIdx] / static_cast<ParamType>(p->qMax[i]);

			// Next bound component
			++bndIdx;
		}

		// Handle fluxes
		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			// Residual
			if (_mode[i] == static_cast<int>(CompMode::Linear))
				res[bndIdx] = static_cast<ParamType>(p->kD[i]) * y[bndIdx] - static_cast<ParamType>(p->kA[i]) * yCp[i];
			else if (_mode[i] == static_cast<int>(CompMode::Langmuir))
			{
				if (_idxModifier >= 0)
					res[bndIdx] = static_cast<ParamType>(p->kD[i]) *
									  pow(yCp[_idxModifier], static_cast<ParamType>(p->beta[i])) * y[bndIdx] -
								  static_cast<ParamType>(p->kA[i]) *
									  exp(yCp[_idxModifier] * static_cast<ParamType>(p->gamma[i])) * yCp[i] *
									  static_cast<ParamType>(p->qMax[i]) * qSum;
				else
					res[bndIdx] = static_cast<ParamType>(p->kD[i]) * y[bndIdx] -
								  static_cast<ParamType>(p->kA[i]) * yCp[i] * static_cast<ParamType>(p->qMax[i]) * qSum;
			}
			else if (_mode[i] == static_cast<int>(CompMode::Modifier))
				res[bndIdx] = 0.0;

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

		double qSum = 1.0;
		int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			// Also skip non-Langmuir components
			if (_mode[i] != static_cast<int>(CompMode::Langmuir))
			{
				++bndIdx;
				continue;
			}

			qSum -= y[bndIdx] / static_cast<double>(p->qMax[i]);

			// Next bound component
			++bndIdx;
		}

		// Protein fluxes: -k_{a,i} * exp(\gamma_i * c_{p,0}) * c_{p,i} * q_{max,i} * (1 - \sum q_i / q_{max,i}) +
		// k_{d,i} * c_{p,0}^\beta_i * q_i)
		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			// Also skip modifier component
			if (_nBoundStates[i] == 0)
				continue;

			if (_mode[i] == static_cast<int>(CompMode::Modifier))
			{
				// Advance to next flux and Jacobian row
				++bndIdx;
				++jac;
				continue;
			}

			const double gamma = static_cast<double>(p->gamma[i]);
			const double beta = static_cast<double>(p->beta[i]);
			const double qMax = static_cast<double>(p->qMax[i]);
			const double ka = (_idxModifier >= 0) ? static_cast<double>(p->kA[i]) * exp(gamma * yCp[_idxModifier])
												  : static_cast<double>(p->kA[i]);
			const double kdRaw = static_cast<double>(p->kD[i]);

			if (_mode[i] == static_cast<int>(CompMode::Linear))
			{
				// dres_i / dc_{p,i}
				jac[i - bndIdx - offsetCp] = -static_cast<double>(p->kA[i]);
				// Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +i to c_{p,i}.
				//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.

				// dres_i / dq_i
				jac[0] = kdRaw;
			}
			else if (_mode[i] == static_cast<int>(CompMode::Langmuir))
			{
				// dres_i / dc_{p,i}
				jac[i - bndIdx - offsetCp] = -ka * qMax * qSum;
				// Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +i to c_{p,i}.
				//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.

				// Fill dres_i / dq_j
				int bndIdx2 = 0;
				for (int j = 0; j < _nComp; ++j)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[j] == 0)
						continue;

					// Skip bound states that are not Langmuir type
					if (_mode[j] != static_cast<int>(CompMode::Langmuir))
					{
						++bndIdx2;
						continue;
					}

					// dres_i / dq_j
					jac[bndIdx2 - bndIdx] = ka * yCp[i] * qMax / static_cast<double>(p->qMax[j]);
					// Getting to q_j: -bndIdx takes us to q_0, another +bndIdx2 to q_j. This means jac[bndIdx2 -
					// bndIdx] corresponds to q_j.

					++bndIdx2;
				}

				if (_idxModifier >= 0)
				{
					// dres_i / dc_{p,mod}
					jac[-bndIdx - offsetCp + _idxModifier] =
						-ka * yCp[i] * qMax * qSum * gamma +
						kdRaw * beta * y[bndIdx] * pow(yCp[_idxModifier], beta - 1.0);
					// Getting to c_{p,mod}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0}, and +_idxModifier to
					// c_{p,mod}.
					//                     This means jac[bndIdx - offsetCp + _idxModifier] corresponds to c_{p,mod}.

					// Add to dres_i / dq_i
					jac[0] += kdRaw * pow(yCp[_idxModifier], beta);
				}
				else
					jac[0] += kdRaw;
			}

			// Advance to next flux and Jacobian row
			++bndIdx;
			++jac;
		}
	}
};

typedef ExtendedMobilePhaseModulatorLangmuirBindingBase<EMPMLangmuirParamHandler>
	ExtendedMobilePhaseModulatorLangmuirBinding;
typedef ExtendedMobilePhaseModulatorLangmuirBindingBase<ExtEMPMLangmuirParamHandler>
	ExternalExtendedMobilePhaseModulatorLangmuirBinding;

namespace binding
{
void registerExtendedMobilePhaseModulatorLangmuirModel(
	std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
{
	bindings[ExtendedMobilePhaseModulatorLangmuirBinding::identifier()] = []() {
		return new ExtendedMobilePhaseModulatorLangmuirBinding();
	};
	bindings[ExternalExtendedMobilePhaseModulatorLangmuirBinding::identifier()] = []() {
		return new ExternalExtendedMobilePhaseModulatorLangmuirBinding();
	};
}
} // namespace binding

} // namespace model

} // namespace cadet
