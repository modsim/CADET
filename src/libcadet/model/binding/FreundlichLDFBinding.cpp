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
#include <cmath>
#include <vector>
#include <unordered_map>
#include <functional>

/*<codegen>
{
	"name": "FreundlichLDFParamHandler",
	"externalName": "ExtFreundlichLDFParamHandler",
	"parameters":
		[
			{ "type": "ScalarComponentDependentParameter", "varName": "kkin", "confName": "FLDF_KKIN"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kF", "confName": "FLDF_KF"},
			{ "type": "ScalarComponentDependentParameter", "varName": "n", "confName": "FLDF_N"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 k_kin : Kinetic coefficient
 k_F   : Freundlich coefficient
 n     : Freundlich exponent
*/

namespace cadet
{

namespace model
{

inline const char* FreundlichLDFParamHandler::identifier() CADET_NOEXCEPT
{
	return "FREUNDLICH_LDF";
}

inline bool FreundlichLDFParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kkin.size() != _kF.size()) || (_kkin.size() != _n.size()) || (_kkin.size() < nComp))
		throw InvalidParameterException("FLDF_KKIN, FLDF_KF and FLDF_N,  have to have the same size");

	return true;
}

inline const char* ExtFreundlichLDFParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_FREUNDLICH_LDF";
}

inline bool ExtFreundlichLDFParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kkin.size() != _kF.size()) || (_kkin.size() != _n.size()) || (_kkin.size() < nComp))
		throw InvalidParameterException("FLDF_KKIN, FLDF_KF and FLDF_N,  have to have the same size");

	return true;
}

/**
 * @brief Defines the multi component Freundlich LDF (Linear Driving Force) isotherm
 * @details Implements the Freundlich LDF adsorption model: \f[ \begin{align}
 *              \frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{f,i} c_{p,i}^{\frac{1}{n_{i}}} \end{align} \f]
 *         	Components without bound state (i.e., non-binding components) are supported.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t> class FreundlichLDFBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:
	FreundlichLDFBindingBase()
	{
	}
	virtual ~FreundlichLDFBindingBase() CADET_NOEXCEPT
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

	CADET_BINDINGMODELBASE_BOILERPLATE

protected:
	using ParamHandlerBindingModelBase<ParamHandler_t>::_paramHandler;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_reactionQuasistationarity;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_nComp;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_nBoundStates;

	const double _threshold = 1e-14;

	template <typename StateType, typename CpStateType, typename ResidualType, typename ParamType>
	int fluxImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* y,
				 CpStateType const* yCp, ResidualType* res, LinearBufferAllocator workSpace) const
	{
		using std::abs;
		using std::pow;

		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Protein Flux: dq/dt = k_kin * (q* - q) with q* = k_F * c^(1/n)
		unsigned int bndIdx = 0;

		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const ParamType n_param = static_cast<ParamType>(p->n[i]);
			const ParamType kF = static_cast<ParamType>(p->kF[i]);
			const ParamType kkin = static_cast<ParamType>(p->kkin[i]);

			// Residual
			if ((n_param > 1) && (abs(yCp[i]) < _threshold))
			{
				const ParamType alpha_1 =
					((2.0 * n_param - 1.0) / n_param) * kF * pow(_threshold, (1.0 - n_param) / n_param);
				const ParamType alpha_2 =
					((1.0 - n_param) / n_param) * kF * pow(_threshold, (1.0 - 2.0 * n_param) / n_param);
				res[bndIdx] = kkin * (y[bndIdx] - yCp[i] * (alpha_1 + alpha_2 * yCp[i]));
			}
			else
			{
				res[bndIdx] = kkin * (y[bndIdx] - kF * pow(abs(yCp[i]), 1.0 / n_param));
			}

			// Next bound component
			++bndIdx;
		}

		return 0;
	}

	template <typename RowIterator>
	void jacobianImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp,
					  int offsetCp, RowIterator jac, LinearBufferAllocator workSpace) const
	{
		using std::abs;
		using std::pow;

		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const double n_param = static_cast<double>(p->n[i]);
			const double kF = static_cast<double>(p->kF[i]);
			const double kkin = static_cast<double>(p->kkin[i]);

			// dres / dq_i
			jac[0] = kkin;

			// dres / dc_{p,i}
			if ((n_param > 1) && (abs(yCp[i]) < _threshold))
			{
				double const alpha_1 =
					((2.0 * n_param - 1.0) / n_param) * kF * pow(_threshold, (1.0 - n_param) / n_param);
				double const alpha_2 =
					((1.0 - n_param) / n_param) * kF * pow(_threshold, (1.0 - 2.0 * n_param) / n_param);
				jac[i - bndIdx - offsetCp] = -kkin * (alpha_1 + 2.0 * alpha_2 * yCp[i]);
			}
			else
			{
				jac[i - bndIdx - offsetCp] = -(1.0 / n_param) * kkin * kF * pow(abs(yCp[i]), (1.0 - n_param) / n_param);
			}

			++bndIdx;
			++jac;
		}
	}
};

typedef FreundlichLDFBindingBase<FreundlichLDFParamHandler> FreundlichLDFBinding;
typedef FreundlichLDFBindingBase<ExtFreundlichLDFParamHandler> ExternalFreundlichLDFBinding;

namespace binding
{
void registerFreundlichLDFModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
{
	bindings[FreundlichLDFBinding::identifier()] = []() { return new FreundlichLDFBinding(); };
	bindings[ExternalFreundlichLDFBinding::identifier()] = []() { return new ExternalFreundlichLDFBinding(); };
}
} // namespace binding

} // namespace model

} // namespace cadet
