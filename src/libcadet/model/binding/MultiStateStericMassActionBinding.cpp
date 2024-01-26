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
#include "model/binding/RefConcentrationSupport.hpp"
#include "model/ModelUtils.hpp"
#include "cadet/Exceptions.hpp"
#include "model/Parameters.hpp"
#include "SlicedVector.hpp"
#include "LocalVector.hpp"
#include "SimulationTypes.hpp"

#include <vector>
#include <unordered_map>
#include <functional>

/*<codegen>
{
	"name": "MSSMAParamHandler",
	"externalName": "ExtMSSMAParamHandler",
	"parameters":
		[
			{ "type": "ScalarParameter", "varName": "lambda", "confName": "MSSMA_LAMBDA"},
			{ "type": "ComponentMajorBoundStateDependentParameter", "varName": "kA", "confName": "MSSMA_KA"},
			{ "type": "ComponentMajorBoundStateDependentParameter", "varName": "kD", "confName": "MSSMA_KD"},
			{ "type": "ComponentMajorBoundStateDependentParameter", "varName": "nu", "confName": "MSSMA_NU"},
			{ "type": "ComponentMajorBoundStateDependentParameter", "varName": "sigma", "confName": "MSSMA_SIGMA"},
			{ "type": "ComponentDependentBoundStateMatrixParameter", "varName": "kRate", "confName": "MSSMA_RATES"}
		],
	"constantParameters":
		[
			{ "type": "ReferenceConcentrationParameter", "varName": ["refC0", "refQ"], "objName": "refConcentration",
"confPrefix": "MSSMA_"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 lambda = Ionic capacity
 kA = Adsorption rate in component-major ordering
 kD = Desorption rate in component-major ordering
 nu = Characteristic charge in component-major ordering
 sigma = Steric factor in component-major ordering
 kRate = State transition rates @f$ k_{\ell j}^{(i)} @f$ in component-row-major ordering
 refC0, refQ = Reference concentrations
*/

namespace cadet
{

namespace model
{

inline const char* MSSMAParamHandler::identifier() CADET_NOEXCEPT
{
	return "MULTISTATE_STERIC_MASS_ACTION";
}

inline bool MSSMAParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	const unsigned int totalBoundStates = numBoundStates(nBoundStates, nComp);

	if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _sigma.size()) ||
		(_kA.size() != totalBoundStates))
		throw InvalidParameterException("MSSMA_KA, MSSMA_KD, MSSMA_NU, and MSSMA_SIGMA have to have the same size");

	util::SlicedVector<active>& n = _nu.get();
	for (unsigned int i = 0; i < nComp; ++i)
	{
		active const* curNu = n[i];
		for (unsigned int j = 1; j < nBoundStates[i]; ++j)
		{
			if (curNu[j - 1] > curNu[j])
				throw InvalidParameterException("MSSMA_NU is not monotonically increasing for component " +
												std::to_string(i) + " at bound state " + std::to_string(j));
		}
	}

	// Assume monovalent salt ions by default
	if (n.native(0) <= 0.0)
		n.native(0) = 1.0;

	return true;
}

inline const char* ExtMSSMAParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_MULTISTATE_STERIC_MASS_ACTION";
}

inline bool ExtMSSMAParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	const unsigned int totalBoundStates = numBoundStates(nBoundStates, nComp);

	if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _sigma.size()) ||
		(_kA.size() != totalBoundStates))
		throw InvalidParameterException(
			"EXT_MSSMA_KA, EXT_MSSMA_KD, EXT_MSSMA_NU, and EXT_MSSMA_SIGMA have to have the same size");

	// Assume monovalent salt ions by default
	if (_nu.base().native(0) <= 0.0)
		_nu.base().native(0) = 1.0;

	return true;
}

/**
 * @brief Defines the Multi-State-SMA binding model
 * @details Implements the Multi-State-SMA adsorption model, which introduces @f$ M_i - 1 @f$ additional bound states
 *@f$ q_i^{(j)} @f$ for each component @f$ i @f$ with transitions between them: \f[ \begin{align}
 *              q_0 &= \Lambda - \sum_{i=1}^{n_{\text{comp}}} \sum_{j=1}^{M_i} \nu_i^{(j)} q_i^{(j)} \\
 *              \frac{\mathrm{d}q_i^{(j)}}{\mathrm{d}t} &= k_{a,i}^{(j)} c_{p,i} \bar{q}_0^{\nu_i^{(j)}} -
 *k_{d,i}^{(j)}\: q_i^{(j)}\: c_{p,0}^{\nu_i^{(j)}} + \sum_{\ell = 1}^{j-1} r^{(i)}_{j\ell} + \sum_{\ell = j+1}^{M}
 *r^{(i)}_{j\ell}, \end{align} \f] where @f$ i = 1, \dots, N_{\text{comp}} @f$ and @f$ j = 1, \dots, M_i @f$. The net
 *flux @f$ r^{(i)}_{j\ell} @f$ from @f$ q_i^{(\ell)} @f$ into @f$ q_i^{(j)} @f$ is given by \f[ \begin{align*}
 *               r_{j,\ell}^{(i)} = \begin{cases}
 *	                 k_{\ell j}^{(i)} q_i^{(\ell)} \bar{q}_0^{\nu_i^{(j)} - \nu_i^{(\ell)}} - k_{j \ell}^{(i)} q_i^{(j)}
 *c_{p,0}^{\nu_i^{(j)} - \nu_i^{(\ell)}}, & \ell < j, \\ k_{\ell j}^{(i)} q_i^{(\ell)} c_{p,0}^{\nu_i^{(\ell)} -
 *\nu_i^{(j)}} - k_{j \ell}^{(i)} q_i^{(j)} \bar{q}_0^{\nu_i^{(\ell)} - \nu_i^{(j)}}, & \ell > j \end{cases}
 *\end{align*} \f] for @f$ 1 \leq \ell, j \leq M_i @f$ and @f$ i = 1, \dots, N_{\text{comp}} @f$. Here, each component
 *possesses several different binding states with increasing strength: @f[ \nu_i^{(j)} \leq \nu_i^{(j+1)} @f] A molecule
 *of component @f$ i @f$ can transition from bound state @f$ \ell @f$ to state @f$ j @f$ with given rate @f$ k_{\ell
 *j}^{(i)} @f$.
 *
 *          Components without bound state (i.e., non-binding components) are supported and all components are allowed
 *to have varying numbers of bound states (except for salt which has only one bound state @f$ q_0 @f$). Component @c 0
 *is assumed to be salt.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t>
class MultiStateStericMassActionBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:
	MultiStateStericMassActionBindingBase()
	{
	}
	virtual ~MultiStateStericMassActionBindingBase() CADET_NOEXCEPT
	{
	}

	static const char* identifier()
	{
		return ParamHandler_t::identifier();
	}
	virtual const char* name() const CADET_NOEXCEPT
	{
		return ParamHandler_t::identifier();
	}

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp,
											  unsigned int const* nBound, unsigned int const* boundOffset)
	{
		const bool res = BindingModelBase::configureModelDiscretization(paramProvider, nComp, nBound, boundOffset);

		if (nBound[0] != 1)
			throw InvalidParameterException("Multi-State SMA binding model requires exactly one bound state for salt");

		const unsigned int totalBoundStates = numBoundStates(nBound, nComp);

		// Allocate space for parameters
		_paramHandler.reserve(totalBoundStates, nComp, nComp, nBound);

		// First flux is salt, which is always quasi-stationary
		_reactionQuasistationarity[0] = true;

		return res;
	}

	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool hasSalt() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool supportsMultistate() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool supportsNonBinding() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool hasQuasiStationaryReactions() const CADET_NOEXCEPT
	{
		return true;
	}

	virtual bool preConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y,
										   double const* yCp, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Compute salt component from given bound states q_i^j

		// Salt equation: q_0 * nu_0 - Lambda + Sum[Sum[nu_i^j * q_i^j, j], i] == 0
		//           <=>  q_0 == (Lambda - Sum[Sum[nu_i^j * q_i^j, j], i]) / nu_0
		y[0] = static_cast<double>(p->lambda);

		// Loop over all components i
		unsigned int bndIdx = 1;
		for (int i = 1; i < _nComp; ++i)
		{
			// Get nu slice for comp
			active const* const curNu = p->nu[i];

			// Loop over all bound states j
			for (unsigned int j = 0; j < _nBoundStates[i]; ++j)
			{
				y[0] -= static_cast<double>(curNu[j]) * y[bndIdx];

				// Next bound component
				++bndIdx;
			}
		}

		y[0] /= static_cast<double>(p->nu.native(0));

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

		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Salt flux: q_0 * nu_0 - Lambda + Sum[Sum[nu_i^j * q_i^j, j], i] == 0
		//       <=>  q_0 * nu_0 == Lambda - Sum[Sum[nu_i^j * q_i^j, j], i]
		// Also compute \bar{q}_0 = q_0 * nu_0 - Sum[Sum[sigma_i^j * q_i^j, j], i]
		res[0] = static_cast<ParamType>(p->nu.native(0)) * y[0] - static_cast<ParamType>(p->lambda);
		StateParamType q0_bar = static_cast<ParamType>(p->nu.native(0)) * y[0];

		unsigned int bndIdx = 1;

		// Loop over all components i
		for (int i = 1; i < _nComp; ++i)
		{
			active const* const curNu = p->nu[i];
			active const* const curSigma = p->sigma[i];

			// Loop over bound states j of component i
			for (unsigned int j = 0; j < _nBoundStates[i]; ++j)
			{
				res[0] += static_cast<ParamType>(curNu[j]) * y[bndIdx];
				q0_bar -= static_cast<ParamType>(curSigma[j]) * y[bndIdx];

				// Next bound component
				++bndIdx;
			}
		}

		const ParamType refC0 = static_cast<ParamType>(p->refC0);
		const ParamType refQ = static_cast<ParamType>(p->refQ);
		const CpStateParamType yCp0_divRef = yCp[0] / refC0;
		const StateParamType q0_bar_divRef = q0_bar / refQ;

		// Protein fluxes

		// Loop over all components i
		bndIdx = 1;
		for (int i = 1; i < _nComp; ++i)
		{
			active const* const curKa = p->kA[i];
			active const* const curKd = p->kD[i];
			active const* const curNu = p->nu[i];
			active const* const curRates = p->kRate[i];
			const unsigned int nBoundStatesI = _nBoundStates[i];

			// Loop over bound states j of component i
			for (unsigned int j = 0; j < nBoundStatesI; ++j)
			{
				const ParamType nu_over_nu0 =
					static_cast<ParamType>(curNu[j]) / static_cast<ParamType>(p->nu.native(0));
				const CpStateParamType c0_pow_nu = pow(yCp0_divRef, nu_over_nu0);
				const StateParamType q0_bar_pow_nu = pow(q0_bar_divRef, nu_over_nu0);

				// Calculate residual
				// Adsorption and desorption
				res[bndIdx] = static_cast<ParamType>(curKd[j]) * y[bndIdx] * c0_pow_nu -
							  static_cast<ParamType>(curKa[j]) * yCp[i] * q0_bar_pow_nu;

				// Conversion to and from weaker states
				for (unsigned int l = 0; l < j; ++l)
				{
					const ParamType nuDiff_over_nu0 =
						(static_cast<ParamType>(curNu[j]) - static_cast<ParamType>(curNu[l])) /
						static_cast<ParamType>(p->nu.native(0));
					// Conversion to weaker state
					res[bndIdx] += static_cast<ParamType>(curRates[nBoundStatesI * j + l]) * y[bndIdx] *
								   pow(yCp0_divRef, nuDiff_over_nu0);
					// Conversion from weaker state
					res[bndIdx] -= static_cast<ParamType>(curRates[nBoundStatesI * l + j]) * y[bndIdx - j + l] *
								   pow(q0_bar_divRef, nuDiff_over_nu0);
					// Getting to q_i^l: bndIdx takes us to q_i^j, subtracting j sets the pointer back
					//                   to q_i^0, and adding l brings us to q_i^l.
					//                   This means y[bndIdx - j + l] corresponds to q_i^l
				}

				// Conversion to and from stronger states
				for (unsigned int l = j + 1; l < nBoundStatesI; ++l)
				{
					const ParamType nuDiff_over_nu0 =
						(static_cast<ParamType>(curNu[l]) - static_cast<ParamType>(curNu[j])) /
						static_cast<ParamType>(p->nu.native(0));
					// Conversion to stronger state
					res[bndIdx] += static_cast<ParamType>(curRates[nBoundStatesI * j + l]) * y[bndIdx] *
								   pow(q0_bar_divRef, nuDiff_over_nu0);
					// Conversion from stronger state
					res[bndIdx] -= static_cast<ParamType>(curRates[nBoundStatesI * l + j]) * y[bndIdx - j + l] *
								   pow(yCp0_divRef, nuDiff_over_nu0);
				}

				// Next bound component
				++bndIdx;
			}
		}
		return 0;
	}

	template <typename RowIterator>
	void jacobianImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp,
					  int offsetCp, RowIterator jac, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		double q0_bar = static_cast<double>(p->nu.native(0)) * y[0];

		// Salt flux: nu_0 * q_0 - Lambda + Sum[Sum[nu_j * q_i^j, j], i] == 0
		jac[0] = static_cast<double>(p->nu.native(0));
		int bndIdx = 1;
		for (int i = 1; i < _nComp; ++i)
		{
			active const* const curNu = p->nu[i];
			active const* const curSigma = p->sigma[i];

			for (unsigned int j = 0; j < _nBoundStates[i]; ++j)
			{
				jac[bndIdx] = static_cast<double>(curNu[j]);

				// Calculate \bar{q}_0 = nu_0 * q_0 - Sum[Sum[sigma_j * q_i^j, j], i]
				q0_bar -= static_cast<double>(curSigma[j]) * y[bndIdx];

				// Next bound component
				++bndIdx;
			}
		}

		// Advance to protein fluxes
		++jac;

		const double refC0 = static_cast<double>(p->refC0);
		const double refQ = static_cast<double>(p->refQ);
		const double yCp0_divRef = yCp[0] / refC0;
		const double q0_bar_divRef = q0_bar / refQ;

		// Protein fluxes
		// We have already computed \bar{q}_0 in the loop above
		bndIdx = 1;
		for (int i = 1; i < _nComp; ++i)
		{
			active const* const curKa = p->kA[i];
			active const* const curKd = p->kD[i];
			active const* const curNu = p->nu[i];
			active const* const curRates = p->kRate[i];
			const double refC0 = static_cast<double>(p->refC0);
			const double refQ = static_cast<double>(p->refQ);
			const unsigned int nBoundStatesI = _nBoundStates[i];

			for (int j = 0; j < static_cast<int>(nBoundStatesI); ++j)
			{
				// Getting to c_{p,0}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0}. This means jac[-bndIdx -
				// offsetCp] corresponds to c_{p,0}. Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to
				// c_{p,0} and a +i to c_{p,i}.
				//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.

				const double ka = static_cast<double>(curKa[j]);
				const double kd = static_cast<double>(curKd[j]);
				const double nu = static_cast<double>(curNu[j]);
				const double nu_over_nu0 = nu / static_cast<double>(p->nu.native(0));

				const double c0_pow_nu = pow(yCp0_divRef, nu_over_nu0);
				const double q0_bar_pow_nu = pow(q0_bar_divRef, nu_over_nu0);
				const double c0_pow_nu_m1_divRef = pow(yCp0_divRef, nu_over_nu0 - 1.0) / refC0;
				const double q0_bar_pow_nu_m1_divRef = nu_over_nu0 * pow(q0_bar_divRef, nu_over_nu0 - 1.0) / refQ;

				// dres_i / dc_{p,0}
				jac[-bndIdx - offsetCp] = kd * y[bndIdx] * nu_over_nu0 * c0_pow_nu_m1_divRef;
				// dres_i / dc_{p,i}
				jac[i - bndIdx - offsetCp] = -ka * q0_bar_pow_nu;
				// dres_i / dq_0
				jac[-bndIdx] = -ka * yCp[i] * q0_bar_pow_nu_m1_divRef * static_cast<double>(p->nu.native(0));

				// Fill dres_i / dq_i^j (no flux terms, just handle \bar{q}_0^{nu_i} term)
				int bndIdx2 = 1;
				for (int i2 = 1; i2 < _nComp; ++i2)
				{
					active const* const curSigma = p->sigma[i2];
					for (unsigned int j2 = 0; j2 < _nBoundStates[i2]; ++j2)
					{
						// dres_i / dq_{i2}^{j2}
						jac[bndIdx2 - bndIdx] =
							ka * yCp[i] * q0_bar_pow_nu_m1_divRef * static_cast<double>(curSigma[j2]);
						// Getting to q_{i2}^{j2}: -bndIdx takes us to q_1^0, another +bndIdx2 to q_{i2}^{j2}.
						// This means jac[bndIdx2 - bndIdx] corresponds to q_{i2}^{j2}.

						++bndIdx2;
					}
				}

				// Add to dres_i / dq_i
				jac[0] += kd * c0_pow_nu;

				// Handle flux terms

				// Conversion to and from weaker states
				for (int l = 0; l < j; ++l)
				{
					const double nuDiff = (nu - static_cast<double>(curNu[l])) / static_cast<double>(p->nu.native(0));
					const double q0_bar_pow_nudiff_deriv = static_cast<double>(curRates[nBoundStatesI * l + j]) *
														   y[bndIdx - j + l] * nuDiff *
														   pow(q0_bar_divRef, nuDiff - 1.0) / refQ;

					// dres_i / dc_{p,0}
					jac[-bndIdx - offsetCp] += static_cast<double>(curRates[nBoundStatesI * j + l]) * y[bndIdx] *
											   nuDiff * pow(yCp0_divRef, nuDiff - 1.0) / refC0;
					// dres_i / dq_0
					jac[-bndIdx] -= q0_bar_pow_nudiff_deriv * static_cast<double>(p->nu.native(0));
					// dres_i / dq_i^j (current outer loop element)
					jac[0] += static_cast<double>(curRates[nBoundStatesI * j + l]) * pow(yCp0_divRef, nuDiff);
					// dres_i / dq_i^l (without dq0_bar / dq_i^l)
					jac[l - j] -= static_cast<double>(curRates[nBoundStatesI * l + j]) * pow(q0_bar_divRef, nuDiff);
					// dres_i / dq_{i2}^{j2} (accounts for all dq0_bar / dq_{i2}^{j2} including missing dq0_bar / dq_i^l
					// from above)
					bndIdx2 = 1;
					for (int i2 = 1; i2 < _nComp; ++i2)
					{
						active const* const curSigma = p->sigma[i2];
						for (int j2 = 0; j2 < static_cast<int>(_nBoundStates[i2]); ++j2, ++bndIdx2)
							jac[bndIdx2 - bndIdx] += q0_bar_pow_nudiff_deriv * static_cast<double>(curSigma[j2]);
					}
				}

				// Conversion to and from stronger states
				for (int l = j + 1; l < static_cast<int>(nBoundStatesI); ++l)
				{
					const double nuDiff = (static_cast<double>(curNu[l]) - nu) / static_cast<double>(p->nu.native(0));
					const double q0_bar_pow_nudiff_deriv = static_cast<double>(curRates[nBoundStatesI * j + l]) *
														   y[bndIdx] * nuDiff * pow(q0_bar_divRef, nuDiff - 1.0) / refQ;

					// dres_i / dc_{p,0}
					jac[-bndIdx - offsetCp] -= static_cast<double>(curRates[nBoundStatesI * l + j]) *
											   y[bndIdx - j + l] * nuDiff * pow(yCp0_divRef, nuDiff - 1.0) / refC0;
					// dres_i / dq_0
					jac[-bndIdx] += q0_bar_pow_nudiff_deriv * static_cast<double>(p->nu.native(0));
					// dres_i / dq_i^j (current outer loop element)
					jac[0] += static_cast<double>(curRates[nBoundStatesI * j + l]) * pow(q0_bar_divRef, nuDiff);
					// dres_i / dq_i^l
					jac[l - j] -= static_cast<double>(curRates[nBoundStatesI * l + j]) * pow(yCp0_divRef, nuDiff);
					// dres_i / dq_{i2}^{j2} (accounts for all dq0_bar / dq_{i2}^{j2})
					bndIdx2 = 1;
					for (int i2 = 1; i2 < _nComp; ++i2)
					{
						active const* const curSigma = p->sigma[i2];
						for (int j2 = 0; j2 < static_cast<int>(_nBoundStates[i2]); ++j2, ++bndIdx2)
							jac[bndIdx2 - bndIdx] -= q0_bar_pow_nudiff_deriv * static_cast<double>(curSigma[j2]);
					}
				}

				// Advance to next flux and Jacobian row
				++bndIdx;
				++jac;
			}
		}
	}
};

typedef MultiStateStericMassActionBindingBase<MSSMAParamHandler> MultiStateStericMassActionBinding;
typedef MultiStateStericMassActionBindingBase<ExtMSSMAParamHandler> ExternalMultiStateStericMassActionBinding;

namespace binding
{
void registerMultiStateStericMassActionModel(
	std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
{
	bindings[MultiStateStericMassActionBinding::identifier()] = []() {
		return new MultiStateStericMassActionBinding();
	};
	bindings[ExternalMultiStateStericMassActionBinding::identifier()] = []() {
		return new ExternalMultiStateStericMassActionBinding();
	};
}
} // namespace binding

} // namespace model

} // namespace cadet
