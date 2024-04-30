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
	"name": "HICUNIFIEDParamHandler",
	"externalName": "ExtHICUNIFIEDParamHandler",
	"parameters":
		[
			{ "type": "ScalarParameter", "varName": "beta0", "confName": "HICUNI_BETA0"},
			{ "type": "ScalarParameter", "varName": "beta1", "confName": "HICUNI_BETA1"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kA", "confName": "HICUNI_KA"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kALin", "confName": "HICUNI_KA_LIN"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kD", "confName": "HICUNI_KD"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kP", "confName": "HICUNI_KP"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kS", "confName": "HICUNI_KS"},
			{ "type": "ScalarComponentDependentParameter", "varName": "epsilon", "confName": "HICUNI_EPSILON"},
			{ "type": "ScalarComponentDependentParameter", "varName": "nu", "confName": "HICUNI_NU"},
			{ "type": "ScalarComponentDependentParameter", "varName": "nuLin", "confName": "HICUNI_NU_LIN"},
			{ "type": "ScalarComponentDependentParameter", "varName": "qMax", "confName": "HICUNI_QMAX"}
		],
	"constantParameters":
		[
			{ "type": "ScalarParameter", "varName": "refPh", "default": 7.0, "objName": "referencePh", "confName": "HICUNI_PH_REF"},
			{ "type": "ScalarParameter", "varName": "rho", "default": 0.00003348, "objName": "osmoticEffect", "confName": "HICUNI_RHO"}

		]
}
</codegen>*/

/* Parameter description
 ------------------------
 beta0 = bulk-like ordered water molecules at infinitely diluted salt concentration
 beta1 = influence of salt concentration on bulk-like ordered water molecules
 kA = Adsorption constant
 kALin = Linear dependency of ka on pH
 kD = Desorption constant
 kP = Influence of protein concentration on protein activity
 kS = Influence of salt concentration on protein activity
 epsilon = Influence of bound protein concentration on bound protein activity
 nu = Number of binding sites
 nuLin = Linear dependency of nu on pH
 qMax = Maximum binding capacity
 rh0 = osmotic effect of salt concentration on water activity. Calculated as osmotic_coefficient * molar_weight_of_water * ion_number in
 refpH = reference pH (defaults to 7)
*/

namespace cadet
{

	namespace model
	{

		inline const char* HICUNIFIEDParamHandler::identifier() CADET_NOEXCEPT { return "HIC_UNIFIED"; }

		inline bool HICUNIFIEDParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
		{
			if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _qMax.size())
				|| (_kA.size() != _kP.size()) || (_kA.size() != _kS.size()) || (_kA.size() != _epsilon.size())
				|| (_kA.size() != _nuLin.size()) || (_kA.size() != _kALin.size())
				|| (_kA.size() < nComp))
				throw InvalidParameterException("HICUNI_KA, HICUNI_KD, HICUNI_NU, and HICUNI_QMAX, HICUNI_KA_LIN, HICUNI_NU_LIN have to have the same size");

			return true;
		}

		inline const char* ExtHICUNIFIEDParamHandler::identifier() CADET_NOEXCEPT { return "EXT_HIC_UNIFIED"; }

		inline bool ExtHICUNIFIEDParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
		{
			if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _qMax.size())
				|| (_kA.size() != _kP.size()) || (_kA.size() != _kS.size()) || (_kA.size() != _epsilon.size())
				|| (_kA.size() != _nuLin.size()) || (_kA.size() != _kALin.size())
				|| (_kA.size() < nComp))
				throw InvalidParameterException("HICUNI_KA, HICUNI_KD, HICUNI_NU, and HICUNI_QMAX, HICUNI_KA_LIN, HICUNI_NU_LIN have to have the same size");

			return true;
		}


		/**
		 * @brief Defines a unified HIC isotherm combining publications by Deichert et al 2010, Mollerup et al 2006, and Jäpel and Buyel 2022
		 * @details Implements a unified HIC isotherm : \f[ \begin{align}
		 *				\beta &= \beta_0 e^{c_{p,0}\beta_1}\\
		 *				k_{a,i} &= k_{a,i,0} \exp\left( k_{a,i,lin} (\mathrm{pH}-\mathrm{pH}_{ref})\right)\\
		 *				\nu_i &= \nu_{i,0}  + \nu_{i,lin} (\mathrm{pH}-\mathrm{pH}_{ref})\\
		 *				\frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i}e^{k_{p,i} c_{p,i}+k_{s,i} c_{s}}(1-\sum{\frac{q_p}{q_{max}}})^{\nu_i}
		 *													-k_{d,i} q_{p,i}(1+\epsilon q_{p,i}) (e^{\rho c_{s}})^{\nu_i \beta_0 e^{\beta_1 c_s}}
		 *			\end{align}  \f]
		 *          Component @c 0 is assumed to be salt without a bound state. Multiple bound states are not supported.
		 *          Components without bound state (i.e., salt and non-binding components) are supported.
		 *
		 * @tparam ParamHandler_t Type that can add support for external function dependence
		 */
		template <class ParamHandler_t>
		class HICUnifiedBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
		{
		public:

			HICUnifiedBindingBase() { }
			virtual ~HICUnifiedBindingBase() CADET_NOEXCEPT { }

			static const char* identifier() { return ParamHandler_t::identifier(); }

			virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBound, unsigned int const* boundOffset)
			{
				const bool res = BindingModelBase::configureModelDiscretization(paramProvider, nComp, nBound, boundOffset);

				// Guarantee that salt has no bound state
				if (nBound[0] != 0)
					throw InvalidParameterException("HICUNIFIED binding model requires exactly zero bound states for salt component");


				// Guarantee that modifier component is non-binding
				if (nBound[1] != 0)
					throw InvalidParameterException("HICUNIFIED binding model requires non-binding pH modifier component (NBOUND[1] = 0)");
				return res;
			}

			virtual bool hasSalt() const CADET_NOEXCEPT { return true; }
			virtual bool supportsMultistate() const CADET_NOEXCEPT { return false; }
			virtual bool supportsNonBinding() const CADET_NOEXCEPT { return true; }
			virtual bool hasQuasiStationaryReactions() const CADET_NOEXCEPT
			{
				return std::any_of(_reactionQuasistationarity.begin(), _reactionQuasistationarity.end(), [](int i) -> bool { return i; });
			}
			virtual bool hasDynamicReactions() const CADET_NOEXCEPT
			{
				return std::any_of(_reactionQuasistationarity.begin(), _reactionQuasistationarity.end(), [](int i) -> bool { return !static_cast<bool>(i); });
			}
			virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT { return true; }

			virtual bool preConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const
			{
				typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);
				return true;
			}

			virtual void postConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const
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

				typename ParamHandler_t::ParamsHandle const _p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

				// Pseudo component 1 is pH
				const CpStateType pH = yCp[1];
				const ParamType refPh = static_cast<ParamType>(_p->refPh);

				const ParamType beta0 = static_cast<ParamType>(_p->beta0);
				const ParamType beta1 = static_cast<ParamType>(_p->beta1);

				// For reference how rho is calculated:
				//const double osmotic_coefficient = 0.93;  // for NaCl, average of Partanen and Partanen 2020, difference can be compensated with beta0 & beta1 (Jäpel Dissertation 2023)
				//const double ion_number = 2.0; // for NaCl
				//const double molar_weight_water = 18.0 / 1000.0 / 1000.0;  // in kg per mmol -> to match yCp[0] in mmol / L (kg)
				//const double rho = osmotic_coefficient * ion_number * molar_weight_water

				const CpStateParamType water_activity = exp(-yCp[0] * static_cast<ParamType>(_p->rho));

				const CpStateParamType beta = beta0 * exp(beta1 * yCp[0]);

				StateParamType qSum = 1.0;

				unsigned int bndIdx = 0;
				for (int i = 2; i < _nComp; ++i)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[i] == 0)
						continue;

					qSum -= y[bndIdx] / static_cast<ParamType>(_p->qMax[i]);

					// Next bound component
					++bndIdx;
				}

				bndIdx = 0;
				for (int i = 2; i < _nComp; ++i)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[i] == 0)
						continue;

					const ParamType kD = static_cast<ParamType>(_p->kD[i]);
					const ParamType kA = static_cast<ParamType>(_p->kA[i]);
					const ParamType kALin = static_cast<ParamType>(_p->kALin[i]);
					const ParamType kP = static_cast<ParamType>(_p->kP[i]);
					const ParamType kS = static_cast<ParamType>(_p->kS[i]);
					const ParamType nu = static_cast<ParamType>(_p->nu[i]);
					const ParamType nuLin = static_cast<ParamType>(_p->nuLin[i]);
					const ParamType epsilon = static_cast<ParamType>(_p->epsilon[i]);

					const CpStateParamType kApH = kA * exp(kALin * (pH - refPh));
					const CpStateParamType nupH = nu + nuLin * (pH - refPh);

					res[bndIdx] = kD * y[bndIdx] * (1 + epsilon * y[bndIdx]) * pow(water_activity, nupH * beta)
						- kApH * pow(qSum, nupH) * yCp[i] * exp(kP * yCp[i] + kS * yCp[0]);

					// Next bound component
					++bndIdx;
				}

				return 0;
			}

			template <typename RowIterator>
			void jacobianImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp, int offsetCp, RowIterator jac, LinearBufferAllocator workSpace) const
			{
				typename ParamHandler_t::ParamsHandle const _p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

				using std::pow, std::exp;

				// Pseudo component 1 is pH
				const double pH = yCp[1];
				const double refPh = static_cast<double>(_p->refPh);

				const double beta0 = static_cast<double>(_p->beta0);
				const double beta1 = static_cast<double>(_p->beta1);

				const double beta = beta0 * exp(beta1 * yCp[0]);
				const double rho = static_cast<double>(_p->rho);

				const double water_activity = exp(-yCp[0] * rho);

				double qSum = 1.0;

				unsigned int bndIdx = 0;
				for (int i = 2; i < _nComp; ++i)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[i] == 0)
						continue;

					qSum -= y[bndIdx] / static_cast<double>(_p->qMax[i]);

					// Next bound component
					++bndIdx;
				}

				bndIdx = 0;
				for (int i = 2; i < _nComp; ++i)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[i] == 0)
						continue;

					const double kD = static_cast<double>(_p->kD[i]);
					const double kA = static_cast<double>(_p->kA[i]);
					const double kP = static_cast<double>(_p->kP[i]);
					const double kS = static_cast<double>(_p->kS[i]);
					const double nu = static_cast<double>(_p->nu[i]);
					const double epsilon = static_cast<double>(_p->epsilon[i]);
					const double kALin = static_cast<double>(_p->kALin[i]);
					const double nuLin = static_cast<double>(_p->nuLin[i]);

					const double kApH = kA * exp(kALin * (pH - refPh));
					const double nupH = nu + nuLin * (pH - refPh);


					// dres_i / dc_{p,0}
					jac[-bndIdx - offsetCp] = kD * y[bndIdx] * (1 + epsilon * y[bndIdx]) * (-rho) * nupH * beta0 * (beta1 * yCp[0] + 1) *
						exp(beta1 * yCp[0] - rho * nupH * beta0 * exp(beta1 * yCp[0]) * yCp[0])
						- kApH * yCp[i] * pow(qSum, nupH) * exp(kP * yCp[i] + kS * yCp[0]) * kS;

					// dres_i / dc_{p,i}
					jac[i - bndIdx - offsetCp] = -kApH * pow(qSum, nupH) * exp(kP * yCp[i] + kS * yCp[0])
						- kApH * yCp[i] * pow(qSum, nupH) * exp(kP * yCp[i] + kS * yCp[0]) * kP;
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
						jac[bndIdx2 - bndIdx] = kApH * yCp[i] * nupH * pow(qSum, nupH - 1) / (static_cast<double>(_p->qMax[j])) * exp(kP * yCp[i] + kS * yCp[0]);
						// Getting to q_j: -bndIdx takes us to q_0, another +bndIdx2 to q_j. This means jac[bndIdx2 - bndIdx] corresponds to q_j.

						++bndIdx2;
					}

					// Add to dres_i / dq_i
					jac[0] += (1 + 2 * epsilon * y[bndIdx]) * kD * pow(water_activity, nupH * beta);

					// Advance to next flux and Jacobian row
					++bndIdx;
					++jac;
				}
			}
		};

		typedef HICUnifiedBindingBase<HICUNIFIEDParamHandler> HICUnifiedBinding;
		typedef HICUnifiedBindingBase<ExtHICUNIFIEDParamHandler> ExternalHICUnifiedBinding;

		namespace binding
		{
			void registerHICUnifiedModel(std::unordered_map<std::string, std::function<model::IBindingModel* ()>>& bindings)
			{
				bindings[HICUnifiedBinding::identifier()] = []() { return new HICUnifiedBinding(); };
				bindings[ExternalHICUnifiedBinding::identifier()] = []() { return new ExternalHICUnifiedBinding(); };
			}
		}  // namespace binding

	}  // namespace model

}  // namespace cadet
