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
	"name": "HICWHSParamHandler",
	"externalName": "ExtHICWHSParamHandler",
	"parameters":
		[
			{ "type": "ScalarParameter", "varName": "beta0", "confName": "HICWHS_BETA0"},
			{ "type": "ScalarParameter", "varName": "beta1", "confName": "HICWHS_BETA1"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kA", "confName": "HICWHS_KA"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kD", "confName": "HICWHS_KD"},
			{ "type": "ScalarComponentDependentParameter", "varName": "nu", "confName": "HICWHS_NU"},
			{ "type": "ScalarComponentDependentParameter", "varName": "qMax", "confName": "HICWHS_QMAX"}
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

inline const char* HICWHSParamHandler::identifier() CADET_NOEXCEPT
{
	return "HIC_WATER_ON_HYDROPHOBIC_SURFACES";
}

inline bool HICWHSParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _qMax.size()) ||
		(_kA.size() < nComp))
		throw InvalidParameterException("HICWHS_KA, HICWHS_KD, HICWHS_NU, and HICWHS_QMAX have to have the same size");

	return true;
}

inline const char* ExtHICWHSParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_HIC_WATER_ON_HYDROPHOBIC_SURFACES";
}

inline bool ExtHICWHSParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _qMax.size()) ||
		(_kA.size() < nComp))
		throw InvalidParameterException("HICWHS_KA, HICWHS_KD, HICWHS_NU, and HICWHS_QMAX have to have the same size");

	return true;
}

/**
 * @brief Defines the HIC Isotherm as described by Wang et al., 2016
 * @details Implements the "water on hydrophobic surfaces" model: \f[ \begin{align}
 *				\beta &= \beta_0 e^{c_{p,0}\beta_1}\\
 *				\frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i} \left( 1 - \sum_j \frac{q_j}{q_{max,j}}
 *\right)^{\nu_i}
 *              - k_{d,i} q_i  \left(\sum_j q_j \right)^{\nu_i \beta}
 *			\end{align}  \f]
 *          Component @c 0 is assumed to be salt without a bound state. Multiple bound states are not supported.
 *          Components without bound state (i.e., salt and non-binding components) are supported.
 *
 *          See @cite Wang2016.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t> class HICWHSBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:
	HICWHSBase()
	{
	}
	virtual ~HICWHSBase() CADET_NOEXCEPT
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
			throw InvalidParameterException("HICWHS model requires exactly zero bound states for salt component");

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

		typename ParamHandler_t::ParamsHandle const _p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		const ParamType beta0 = static_cast<ParamType>(_p->beta0);
		const ParamType beta1 = static_cast<ParamType>(_p->beta1);

		const CpStateParamType beta = beta0 * exp(beta1 * yCp[0]);

		StateParamType freeBindingSites = 1.0;
		StateType qSum = 0.0;

		// qSumOverqMax could be calculated as 1-freeBindingSites, but is calculated explicitly for clarity
		StateParamType qSumOverqMax = 0.0;

		unsigned int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const StateParamType qOverQmax = y[bndIdx] / static_cast<ParamType>(_p->qMax[i]);

			qSum += y[bndIdx];
			qSumOverqMax += qOverQmax;
			freeBindingSites -= qOverQmax;

			// Next bound component
			++bndIdx;
		}

		// Clip free binding sites to 0.0
		if (freeBindingSites < 0)
			freeBindingSites = 0;

		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const ParamType qMax = static_cast<ParamType>(_p->qMax[i]);
			const ParamType kD = static_cast<ParamType>(_p->kD[i]);
			const ParamType kA = static_cast<ParamType>(_p->kA[i]);
			const ParamType nu = static_cast<ParamType>(_p->nu[i]);

			if (qSum <= 0.0)
			{
				// Use Taylor series approximation
				res[bndIdx] = kA * yCp[i] * nu * qSumOverqMax - kA * yCp[i];
			}
			else
			{
				res[bndIdx] = kD * y[bndIdx] * pow(qSum, nu * beta) - kA * pow(freeBindingSites, nu) * yCp[i];
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
		typename ParamHandler_t::ParamsHandle const _p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		auto beta0 = static_cast<double>(_p->beta0);
		auto beta1 = static_cast<double>(_p->beta1);

		auto beta = static_cast<double>(beta0 * exp(beta1 * yCp[0]));

		double freeBindingSites = 1.0;
		double qSum = 0.0;

		// qSumOverqMax could be calculated as 1-freeBindingSites, but is calculated explicitly for clarity
		double qSumOverqMax = 0.0;

		unsigned int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const double qOverQmax = y[bndIdx] / static_cast<double>(_p->qMax[i]);

			qSum += y[bndIdx];
			qSumOverqMax += qOverQmax;
			freeBindingSites -= qOverQmax;

			// Next bound component
			++bndIdx;
		}

		// Clip free binding sites to 0.0
		if (freeBindingSites < 0)
			freeBindingSites = 0;

		bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			const double kA = static_cast<double>(_p->kA[i]);
			const double kD = static_cast<double>(_p->kD[i]);
			const double nu = static_cast<double>(_p->nu[i]);
			const double qMax = static_cast<double>(_p->qMax[i]);

			if (qSum <= 0.0)
			{
				// Compute the Jacobian for the Taylor series defined above

				// dres_i / dc_{p,0}
				jac[-bndIdx - offsetCp] = 0;

				// dres_i / dc_{p,i}
				jac[i - bndIdx - offsetCp] = kA * nu * qSumOverqMax - kA;
			}
			else
			{
				// todo
				// dres_i / dc_{p,0}
				jac[-bndIdx - offsetCp] = kD * y[bndIdx] * beta * beta1 * nu * pow(qSum, nu * beta) * std::log(qSum);

				// dres_i / dc_{p,i}
				jac[i - bndIdx - offsetCp] = -kA * static_cast<double>(pow(freeBindingSites, nu));
			}

			// Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0} and a +i to c_{p,i}.
			//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.

			// Fill dres_i / dq_j
			int bndIdx2 = 0;
			for (int j = 0; j < _nComp; ++j)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[j] == 0)
					continue;

				if (qSum <= 0)
				{
					jac[bndIdx2 - bndIdx] = yCp[i] * kA * nu / static_cast<double>(_p->qMax[j]);
				}
				else
				{
					// dres_i / dq_j
					jac[bndIdx2 - bndIdx] =
						-kA * yCp[i] * nu * pow(freeBindingSites, nu - 1) / (-static_cast<double>(_p->qMax[j])) +
						beta * kD * nu * y[bndIdx] * pow(qSum, nu * beta - 1);
				}

				// Getting to q_j: -bndIdx takes us to q_0, another +bndIdx2 to q_j. This means jac[bndIdx2 - bndIdx]
				// corresponds to q_j.

				++bndIdx2;
			}

			// Add to dres_i / dq_i
			if (qSum <= 0)
			{
				// if qSum is below zero the jacobian is computed for the Taylor series defined above
				jac[0] = kA * yCp[i] * nu / qMax;
			}
			else
			{
				jac[0] += kD * pow(qSum, nu * beta);
			}

			// Advance to next flux and Jacobian row
			++bndIdx;
			++jac;
		}
	}
};

typedef HICWHSBase<HICWHSParamHandler> HICWHS;
typedef HICWHSBase<ExtHICWHSParamHandler> ExternalHICWHS;

namespace binding
{
void registerHICWaterOnHydrophobicSurfacesModel(
	std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
{
	bindings[HICWHS::identifier()] = []() { return new HICWHS(); };
	bindings[ExternalHICWHS::identifier()] = []() { return new ExternalHICWHS(); };
}
} // namespace binding

} // namespace model

} // namespace cadet
