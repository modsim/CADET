// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
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

/* In the following lines of code the interface to the binding model is defined.
 In the interface the binding model specific parameters are defined. Please modify configuration name and 
 parameter names according to the naming convection explained in the tutorial.*/

/*<codegen>
{
	"name": "LangmuirParamHandler",
	"externalName": "ExtLangmuirParamHandler",
	"parameters":
		[
			{ "type": "ScalarComponentDependentParameter", "varName": "kA", "confName": "MCL_KA"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kD", "confName": "MCL_KD"},
			{ "type": "ScalarComponentDependentParameter", "varName": "qMax", "confName": "MCL_QMAX"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 kA = Adsorption rate
 kD = Desorption rate
 qMax = Capacity
*/

namespace cadet
{

	namespace model
	{

		inline const char* LangmuirParamHandler::identifier() CADET_NOEXCEPT { return "MULTI_COMPONENT_LANGMUIR"; }

		inline bool LangmuirParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
		{
			if ((_kA.size() != _kD.size()) || (_kA.size() != _qMax.size()) || (_kA.size() < nComp))
				throw InvalidParameterException("MCL_KA, MCL_KD, and MCL_QMAX have to have the same size");

			return true;
		}

		inline const char* ExtLangmuirParamHandler::identifier() CADET_NOEXCEPT { return "EXT_MULTI_COMPONENT_LANGMUIR"; }

		inline bool ExtLangmuirParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
		{
			if ((_kA.size() != _kD.size()) || (_kA.size() != _qMax.size()) || (_kA.size() < nComp))
				throw InvalidParameterException("EXT_MCL_KA, EXT_MCL_KD, and EXT_MCL_QMAX have to have the same size");

			return true;
		}


		/**
		 * @brief Defines the multi component Langmuir binding model
		 * @details Implements the Langmuir adsorption model: \f[ \begin{align}
		 *              \frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i} q_{\text{max},i} \left( 1 - \sum_j \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} q_i
		 *          \end{align} \f]
		 *          Multiple bound states are not supported.
		 *          Components without bound state (i.e., non-binding components) are supported.
		 *
		 *          See @cite Langmuir1916.
		 * @tparam ParamHandler_t Type that can add support for external function dependence
		 */
		template <class ParamHandler_t>
		class LangmuirBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
		{
		public:

			LangmuirBindingBase() { }
			virtual ~LangmuirBindingBase() CADET_NOEXCEPT { }

			static const char* identifier() { return ParamHandler_t::identifier(); }

			virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT { return true; }

			CADET_BINDINGMODELBASE_BOILERPLATE

		protected:
			using ParamHandlerBindingModelBase<ParamHandler_t>::_paramHandler;
			using ParamHandlerBindingModelBase<ParamHandler_t>::_reactionQuasistationarity;
			using ParamHandlerBindingModelBase<ParamHandler_t>::_nComp;
			using ParamHandlerBindingModelBase<ParamHandler_t>::_nBoundStates;

			// In the follwing the class method the binding model mass transfer behavior is implemented. 
			// Apart from the mentioned places in the tutorial, do not modify anything in this function. 
			template <typename StateType, typename CpStateType, typename ResidualType, typename ParamType>
			int fluxImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* y,
				CpStateType const* yCp, ResidualType* res, LinearBufferAllocator workSpace) const
			{
				typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

				// Protein fluxes: -k_{a,i} * c_{p,i} * q_{max,i} * (1 - \sum_j q_j / q_{max,j}) + k_{d,i} * q_i
				ResidualType qSum = 1.0;
				unsigned int bndIdx = 0;
				for (int i = 0; i < _nComp; ++i)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[i] == 0)
						continue;

					qSum -= y[bndIdx] / static_cast<ParamType>(p->qMax[i]);

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
					res[bndIdx] = static_cast<ParamType>(p->kD[i]) * y[bndIdx] - static_cast<ParamType>(p->kA[i]) * yCp[i] * static_cast<ParamType>(p->qMax[i]) * qSum;

					// Next bound component
					++bndIdx;
				}

				return 0;
			}

			// In the follwing the class method the anlytical Jacobian of the binding model should be implemented.
			// Apart from the mentioned places in the tutorial, do not modify anything in this function. 
			template <typename RowIterator>
			void jacobianImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp, int offsetCp, RowIterator jac, LinearBufferAllocator workSpace) const
			{
				typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

				// Protein fluxes: -k_{a,i} * c_{p,i} * q_{max,i} * (1 - \sum_j q_j / q_{max,j}) + k_{d,i} * q_i
				double qSum = 1.0;
				int bndIdx = 0;
				for (int i = 0; i < _nComp; ++i)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[i] == 0)
						continue;

					qSum -= y[bndIdx] / static_cast<double>(p->qMax[i]);

					// Next bound component
					++bndIdx;
				}

				bndIdx = 0;
				for (int i = 0; i < _nComp; ++i)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[i] == 0)
						continue;

					const double ka = static_cast<double>(p->kA[i]);
					const double kd = static_cast<double>(p->kD[i]);

					// dres_i / dc_{p,i}
					jac[i - bndIdx - offsetCp] = -static_cast<double>(p->kA[i]) * static_cast<double>(p->qMax[i]) * qSum;
					// Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +i to c_{p,i}.
					//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.

					// Fill dres_i / dq_j
					int bndIdx2 = 0;
					for (int j = 0; j < _nComp; ++j)
					{
						// Skip components without bound states (bound state index bndIdx is not advanced)
						if (_nBoundStates[j] == 0)
							continue;

						// dres_i / dq_j
						jac[bndIdx2 - bndIdx] = static_cast<double>(p->kA[i]) * yCp[i] * static_cast<double>(p->qMax[i]) / static_cast<double>(p->qMax[j]);
						// Getting to q_j: -bndIdx takes us to q_0, another +bndIdx2 to q_j. This means jac[bndIdx2 - bndIdx] corresponds to q_j.

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

		// Do not forget to make changes in the following lines of code according to the guidelines given in the tutorial.
		typedef LangmuirBindingBase<LangmuirParamHandler> LangmuirBinding;
		typedef LangmuirBindingBase<ExtLangmuirParamHandler> ExternalLangmuirBinding;

		namespace binding
		{
			void registerLangmuirModel(std::unordered_map<std::string, std::function<model::IBindingModel* ()>>& bindings)
			{
				bindings[LangmuirBinding::identifier()] = []() { return new LangmuirBinding(); };
				bindings[ExternalLangmuirBinding::identifier()] = []() { return new ExternalLangmuirBinding(); };
			}
		}  // namespace binding

	}  // namespace model

}  // namespace cadet
