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
			{ "type": "ScalarComponentDependentParameter", "varName": "kD", "confName": "HICUNI_KD"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kP", "confName": "HICUNI_KP"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kS", "confName": "HICUNI_KS"},
			{ "type": "ScalarComponentDependentParameter", "varName": "epsilon", "confName": "HICUNI_EPSILON"},
			{ "type": "ScalarComponentDependentParameter", "varName": "nu", "confName": "HICUNI_NU"},
			{ "type": "ScalarComponentDependentParameter", "varName": "qMax", "confName": "HICUNI_QMAX"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 beta0 = bulk-like ordered water molecules at infinitely diluted salt concentration
 beta1 = influence of salt concentration on bulk-like ordered water molecules
 kA = Adsorption constant
 kD = Desorption constant
 kP = Influence of protein concentration on protein activity
 kS = Influence of salt concentration on protein activity
 epsilon = Influence of bound protein concentration on bound protein activity
 nu = Number of binding sites
 qMax = Maximum binding capacity
*/

namespace cadet
{

	namespace model
	{

		inline const char* HICUNIFIEDParamHandler::identifier() CADET_NOEXCEPT { return "HIC_UNIFIED"; }

		inline bool HICUNIFIEDParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
		{
			if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _qMax.size())
				|| (_kA.size() != _kP.size()) || (_kA.size() != _kS.size()) || (_kA.size() != _epsilon.size()) || (_kA.size() < nComp))
				throw InvalidParameterException("HICUNI_KA, HICUNI_KD, HICUNI_NU, and HICUNI_QMAX have to have the same size");

			return true;
		}

		inline const char* ExtHICUNIFIEDParamHandler::identifier() CADET_NOEXCEPT { return "EXT_HIC_UNIFIED"; }

		inline bool ExtHICUNIFIEDParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
		{
			if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _qMax.size())
				|| (_kA.size() != _kP.size()) || (_kA.size() != _kS.size()) || (_kA.size() != _epsilon.size()) || (_kA.size() < nComp))
				throw InvalidParameterException("KA, KD, NU, and QMAX have to have the same size");

			return true;
		}


		/**
		 * @brief Defines a unified HIC isotherm combining publications by Deichert et al 2010, Mollerup et al 2006, and Jäpel and Buyel 2022
		 * @details Implements a unified HIC isotherm : \f[ \begin{align}
		 *				\beta &= \beta_0 e^{c_{p,0}\beta_1}\\
		 *				\frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i}e^{k_{p,i} c_{p,i}+k_{s,i} c_{s}}(1-\sum{\frac{q_p}{q_{max}}})^{ν_i} -k_{d,i} q_{p,i}(1+\epsilon q_{p,i}) (e^{ρc_{s}})^{ν_i β_0 e^{β_1 c_s}}
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

				// First flux is salt, which is always quasi-stationary
				_reactionQuasistationarity[0] = false;

				return res;
			}

			virtual bool hasSalt() const CADET_NOEXCEPT { return true; }
			virtual bool supportsMultistate() const CADET_NOEXCEPT { return false; }
			virtual bool supportsNonBinding() const CADET_NOEXCEPT { return true; }
			virtual bool hasQuasiStationaryReactions() const CADET_NOEXCEPT { return false; }
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

				const ParamType beta0 = static_cast<ParamType>(_p->beta0);
				const ParamType beta1 = static_cast<ParamType>(_p->beta1);

				const double osmotic_coefficient = 0.93;  // for NaCl
				const double ion_number = 2.0; // for NaCl
				const double molar_weight_water = 18.0 / 1000.0 / 1000.0;  // kg per mmol
				const CpStateType water_activity = exp(-yCp[0] * osmotic_coefficient * ion_number * molar_weight_water );

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
					const ParamType kP = static_cast<ParamType>(_p->kP[i]);
					const ParamType kS = static_cast<ParamType>(_p->kS[i]);
					const ParamType nu = static_cast<ParamType>(_p->nu[i]);
					const ParamType epsilon = static_cast<ParamType>(_p->epsilon[i]);

					/*res[bndIdx] = kD * y[bndIdx] * pow(water_activity, nu * beta)
						- kA * pow(qSum, nu) * yCp[i];*/
					res[bndIdx] = kD * y[bndIdx] * (1 + epsilon * y[bndIdx]) * pow(water_activity, nu * beta)
						- kA * pow(qSum, nu) * yCp[i] * exp(kP * yCp[i] + kS * yCp[0]);

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

				const double beta0 = static_cast<double>(_p->beta0);
				const double beta1 = static_cast<double>(_p->beta1);

				const double beta = beta0 * exp(beta1 * yCp[0]);

				const double osmotic_coefficient = 0.93;  // for NaCl
				const double ion_number = 2.0; // for NaCl
				const double molar_weight_water = 18.0 / 1000.0 / 1000.0;  // kg per mmol
				const double water_activity = exp(-yCp[0] * osmotic_coefficient * ion_number * molar_weight_water);

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
					const double kP = static_cast<double>(_p->kP[i]);
					const double kS = static_cast<double>(_p->kS[i]);
					const double nu = static_cast<double>(_p->nu[i]);
					const double epsilon = static_cast<double>(_p->epsilon[i]);

					// dres_i / dc_{p,0}
					jac[-bndIdx - offsetCp] = kD * y[bndIdx] * (1 + epsilon * y[bndIdx])* (-osmotic_coefficient * ion_number * molar_weight_water) * nu * beta0 * (beta1 * yCp[0] + 1) *
						exp(beta1 * yCp[0] - osmotic_coefficient * ion_number * molar_weight_water * nu * beta0 * exp(beta1 * yCp[0]) * yCp[0]) 
						- kA * yCp[i] *pow(qSum, nu) * exp(kP * yCp[i] + kS * yCp[0]) * kS;

					// dres_i / dc_{p,i}
					jac[i - bndIdx - offsetCp] = -kA * pow(qSum, nu) * exp(kP * yCp[i] + kS * yCp[0])
						- kA * yCp[i] * pow(qSum, nu) * exp(kP * yCp[i] + kS * yCp[0]) * kP;
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
						jac[bndIdx2 - bndIdx] = kA * yCp[i] * nu * pow(qSum, nu - 1) / (static_cast<double>(_p->qMax[j])) * exp(kP * yCp[i] + kS * yCp[0]);
						// Getting to q_j: -bndIdx takes us to q_0, another +bndIdx2 to q_j. This means jac[bndIdx2 - bndIdx] corresponds to q_j.

						++bndIdx2;
					}

					// Add to dres_i / dq_i
					jac[0] += (1 + 2 * epsilon * y[bndIdx]) * kD * pow(water_activity, nu * beta);

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
