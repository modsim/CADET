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
#include "MathUtil.hpp"

#include <functional>
#include <unordered_map>
#include <string>
#include <vector>

/*<codegen>
{
	"name": "SipsParamHandler",
	"externalName": "ExtSipsParamHandler",
	"parameters":
		[
			{ "type": "ScalarComponentDependentParameter", "varName": "kA", "confName": "SIPS_KA"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kD", "confName": "SIPS_KD"},
			{ "type": "ScalarComponentDependentParameter", "varName": "qMax", "confName": "SIPS_QMAX"},
			{ "type": "ScalarComponentDependentParameter", "varName": "n", "confName": "SIPS_EXP"}
		],
	"constantParameters":
		[
			{ "type": "ReferenceConcentrationParameter", "varName": ["refC", "refQ"], "objName": "refConcentration", "confPrefix": "SIPS_"},
			{ "type": "ScalarParameter", "varName": "linearThreshold", "confName": "SIPS_LINEAR_THRESHOLD"}
		]
}
</codegen>*/

namespace cadet
{

namespace model
{

inline const char* SipsParamHandler::identifier() CADET_NOEXCEPT { return "MULTI_COMPONENT_SIPS"; }

inline bool SipsParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kD.size()) || (_kA.size() != _qMax.size()) || (_kA.size() < nComp))
		throw InvalidParameterException("SIPS_KA, SIPS_KD, and SIPS_QMAX have to have the same size");

	return true;
}

inline const char* ExtSipsParamHandler::identifier() CADET_NOEXCEPT { return "EXT_MULTI_COMPONENT_SIPS"; }

inline bool ExtSipsParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kD.size()) || (_kA.size() != _qMax.size()) || (_kA.size() < nComp))
		throw InvalidParameterException("EXT_SIPS_KA, EXT_SIPS_KD, and EXT_SIPS_QMAX have to have the same size");

	return true;
}


/**
 * @brief Defines the multi component Sips binding model
 * @details Implements the Sips adsorption model: \f[ \begin{align} 
 *              \frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} \left(\frac{c_{p,i}}{c_{p,\text{ref}}} \right)^{n_i} q_{\text{max},i} \left( 1 - \sum_j \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} \frac{q_i}{q_{i,\text{ref}}}
 *          \end{align} \f]
 *          Multiple bound states are not supported. 
 *          Components without bound state (i.e., non-binding components) are supported.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t>
class SipsBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:

	SipsBindingBase() { }
	virtual ~SipsBindingBase() CADET_NOEXCEPT { }

	static const char* identifier() { return ParamHandler_t::identifier(); }

	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT { return true; }

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
		typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);
		using CpStateParamType = typename DoubleActivePromoter<CpStateType, ParamType>::type;

		// Protein fluxes: -k_{a,i} * (c_{p,i} / c_{p,ref})^n_i * q_{max,i} * (1 - \sum_j q_j / q_{max,j}) + k_{d,i} * q_i / q_{ref}
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

		const ParamType refC = static_cast<ParamType>(p->refC);
		const ParamType refQ = static_cast<ParamType>(p->refQ);
		const double linearThreshold = static_cast<double>(p->linearThreshold);

		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			// Residual
			if (abs(yCp[i]) <= linearThreshold)
			{
				// Linearize
				const CpStateParamType cpLin = pow(linearThreshold / static_cast<ParamType>(refC), static_cast<ParamType>(p->n[i]) - 2.0) * yCp[i] / sqr(static_cast<ParamType>(refC)) * ((2.0 - static_cast<ParamType>(p->n[i])) * linearThreshold + yCp[i] * (static_cast<ParamType>(p->n[i]) - 1.0));
				res[bndIdx] = static_cast<ParamType>(p->kD[i]) * y[bndIdx] / refQ - static_cast<ParamType>(p->kA[i]) * cpLin * static_cast<ParamType>(p->qMax[i]) * qSum;
			}
			else
			{
				res[bndIdx] = static_cast<ParamType>(p->kD[i]) * y[bndIdx] / refQ - static_cast<ParamType>(p->kA[i]) * pow(yCp[i] / refC, static_cast<ParamType>(p->n[i])) * static_cast<ParamType>(p->qMax[i]) * qSum;
			}

			// Next bound component
			++bndIdx;
		}

		return 0;
	}

	template <typename RowIterator>
	void jacobianImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp, int offsetCp, RowIterator jac, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Protein fluxes: -k_{a,i} * (c_{p,i} / c_{p,ref})^n_i * q_{max,i} * (1 - \sum_j q_j / q_{max,j}) + k_{d,i} * q_i / q_{ref}
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

		const double refC = static_cast<double>(p->refC);
		const double refQ = static_cast<double>(p->refQ);
		const double linearThreshold = static_cast<double>(p->linearThreshold);

		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const double ka = static_cast<double>(p->kA[i]);
			const double kd = static_cast<double>(p->kD[i]);
			const double n = static_cast<double>(p->n[i]);

			// dres_i / dc_{p,i}
			if (abs(yCp[i]) <= linearThreshold)
			{
				// Linearized
				jac[i - bndIdx - offsetCp] = -ka * static_cast<double>(p->qMax[i]) * qSum * pow(linearThreshold / refC, n) / linearThreshold * (2.0 - n + 2.0 * (n - 1.0) * yCp[i] / linearThreshold);
				// Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +i to c_{p,i}.
				//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.
			}
			else
			{
				jac[i - bndIdx - offsetCp] = -ka * n * static_cast<double>(p->qMax[i]) * qSum * pow(yCp[i] / refC, n - 1.0) / refC;
				// Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +i to c_{p,i}.
				//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.
			}

			// Fill dres_i / dq_j
			const double linearizedCp = (abs(yCp[i]) <= linearThreshold) ? 
				pow(linearThreshold / refC, n - 2.0) * yCp[i] / sqr(refC) * ((2.0 - n) * linearThreshold + yCp[i] * (n - 1.0)) : // Linearized 
				pow(yCp[i] / refC, n);

			const double factor = ka * linearizedCp * static_cast<double>(p->qMax[i]);

			int bndIdx2 = 0;
			for (int j = 0; j < _nComp; ++j)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[j] == 0)
					continue;

				// dres_i / dq_j
				jac[bndIdx2 - bndIdx] = factor / static_cast<double>(p->qMax[j]);
				// Getting to q_j: -bndIdx takes us to q_0, another +bndIdx2 to q_j. This means jac[bndIdx2 - bndIdx] corresponds to q_j.

				++bndIdx2;
			}

			// Add to dres_i / dq_i
			jac[0] += kd / refQ;

			// Advance to next flux and Jacobian row
			++bndIdx;
			++jac;
		}
	}	
};

typedef SipsBindingBase<SipsParamHandler> SipsBinding;
typedef SipsBindingBase<ExtSipsParamHandler> ExternalSipsBinding;

namespace binding
{
	void registerSipsModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
	{
		bindings[SipsBinding::identifier()] = []() { return new SipsBinding(); };
		bindings[ExternalSipsBinding::identifier()] = []() { return new ExternalSipsBinding(); };
	}
}  // namespace binding

}  // namespace model

}  // namespace cadet
