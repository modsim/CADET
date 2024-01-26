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
	"name": "BiSMAParamHandler",
	"externalName": "ExtBiSMAParamHandler",
	"parameters":
		[
			{ "type": "ScalarBoundStateDependentParameter", "varName": "lambda", "confName": "BISMA_LAMBDA"},
			{ "type": "ComponentBoundStateMajorDependentParameter", "varName": "kA", "confName": "BISMA_KA"},
			{ "type": "ComponentBoundStateMajorDependentParameter", "varName": "kD", "confName": "BISMA_KD"},
			{ "type": "ComponentBoundStateMajorDependentParameter", "varName": "nu", "confName": "BISMA_NU"},
			{ "type": "ComponentBoundStateMajorDependentParameter", "varName": "sigma", "confName": "BISMA_SIGMA"}
		],
	"constantParameters":
		[
			{ "type": "ReferenceConcentrationBoundStateDependentParameter", "varName": ["refC0", "refQ"], "objName":
"refConcentration", "confPrefix": "BISMA_"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 lambda = Ionic capacity in binding site-major ordering
 kA = Adsorption rate in binding site-major ordering
 kD = Desorption rate in binding site-major ordering
 nu = Characteristic charge in binding site-major ordering
 sigma = Steric factor in binding site-major ordering
 refC0, refQ = Reference concentrations
*/

namespace cadet
{

namespace model
{

inline const char* BiSMAParamHandler::identifier() CADET_NOEXCEPT
{
	return "BI_STERIC_MASS_ACTION";
}

inline bool BiSMAParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	const unsigned int numStates = firstNonEmptyBoundStates(nBoundStates, nComp);

	if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _sigma.size()) ||
		(_kA.size() < nComp * numStates))
		throw InvalidParameterException("BISMA_KA, BISMA_KD, BISMA_NU, and BISMA_SIGMA have to have the same size");
	if (_lambda.size() != numStates)
		throw InvalidParameterException("BISMA_LAMBDA has to have as many elements as there are binding sites");

	util::SlicedVector<active>& nu = _nu.get();
	for (unsigned int i = 0; i < numStates; ++i)
	{
		if (nu(i, 0) <= 0.0)
			nu(i, 0) = 1.0;
	}

	return true;
}

inline const char* ExtBiSMAParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_BI_STERIC_MASS_ACTION";
}

inline bool ExtBiSMAParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	const unsigned int numStates = firstNonEmptyBoundStates(nBoundStates, nComp);

	if ((_kA.size() != _kD.size()) || (_kA.size() != _nu.size()) || (_kA.size() != _sigma.size()) ||
		(_kA.size() < nComp * numStates))
		throw InvalidParameterException("BISMA_KA, BISMA_KD, BISMA_NU, and BISMA_SIGMA have to have the same size");
	if (_lambda.size() != numStates)
		throw InvalidParameterException("BISMA_LAMBDA has to have as many elements as there are binding sites");

	util::SlicedVector<active>& nu = _nu.base();
	for (unsigned int i = 0; i < numStates; ++i)
	{
		if (nu(i, 0) <= 0.0)
			nu(i, 0) = 1.0;
	}

	return true;
}

/**
 * @brief Defines the Bi-SMA binding model
 * @details Implements the Bi-SMA adsorption model, which introduces multiple different binding sites and assumes an SMA
 * model for each site: \f[ \begin{align}
 *              q_0^A &= \Lambda^A - \sum_{j} \nu_j^A q_j^A \\
 *              q_0^B &= \Lambda^B - \sum_{j} \nu_j^B q_j^B \\
 *              \vodts & \vdots
 *              \frac{\mathrm{d}q_i^A}{\mathrm{d}t} &= k_{a,i}^A c_{p,i} \left( \Lambda^A - \sum_j\left( \nu_j^A +
 * \sigma_j^A \right) q_j^A \right)^{\nu_i^A} - k_{d,i}^A q_i^A c_{p,0}^{\nu_i^A} \\
 *              \frac{\mathrm{d}q_i^B}{\mathrm{d}t} &= k_{a,i}^B c_{p,i} \left( \Lambda^B - \sum_j\left( \nu_j^B +
 * \sigma_j^B \right) q_j^B \right)^{\nu_i^B} - k_{d,i}^B q_i^B c_{p,0}^{\nu_i^B} \\ \vodts & \vdots \end{align} \f]
 *          Here, several different types of binding sites @f$ q^A @f$, @f$ q^B @f$, etc. are considered. A molecule can
 * either bind to site A or B (or C, etc.). A direct exchange between the different binding sites does not occur. While
 * components without bound state (i.e., non-binding components) are supported, all other components must have the same
 * number of bound states (i.e., binding sites). Component @c 0 is assumed to be salt.
 *
 *          Since all algebraic variables have to be in one contiguous block, the order of the bound state variables is
 * as follows: \f[ \begin{align} q_0^A, q_0^B, q_0^C, \dots, q_1^A, q_2^A, q_3^A, \dots, q_1^B, q_2^B, q_3^B, \dots
 *          \end{align} \f]
 *          First, all the salt components are collected in one block as they are always algebraic.
 *          Then the other components for each bound state follow.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t>
class BiStericMassActionBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:
	BiStericMassActionBindingBase()
	{
	}
	virtual ~BiStericMassActionBindingBase() CADET_NOEXCEPT
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

		unsigned int numSlices = 0;
		for (unsigned int i = 0; i < nComp; ++i)
		{
			if (nBound[i] == 0)
				continue;

			if (numSlices == 0)
				numSlices = nBound[i];

			if (nBound[i] != numSlices)
				throw InvalidParameterException(
					"BiSMA binding model requires exactly the same bound states for all components");
		}

		_numBindingComp = numBindingComponents(_nBoundStates, _nComp);

		// Allocate space for parameters
		_paramHandler.reserve(nComp * numSlices, numSlices, nComp, nBound);

		// Guarantee that salt has exactly one bound state per binding site
		if (nBound[0] != numSlices)
			throw InvalidParameterException(
				"BiSMA binding model requires exactly one bound state per binding site for salt component");

		// First fluxes are salt, which are always quasi-stationary
		std::fill_n(_reactionQuasistationarity.data(), numSlices, true);

		return res;
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
	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT
	{
		return true;
	}

	virtual bool preConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y,
										   double const* yCp, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);
		const unsigned int numStates = p->lambda.size();

		// Compute salt component from given bound states q_j

		// Loop over all binding site types
		for (unsigned int bndSite = 0; bndSite < numStates; ++bndSite)
		{
			// Salt equation: nu_0 * q_0 - Lambda + Sum[nu_j * q_j, j] == 0
			//           <=>  q_0 == (Lambda - Sum[nu_j * q_j, j]) / nu_0
			y[bndSite] = static_cast<double>(p->lambda[bndSite]);

			// Get nu slice for bound state bndSite
			active const* const curNu = p->nu[bndSite];

			unsigned int bndIdx = 1;
			for (int j = 1; j < _nComp; ++j)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[j] == 0)
					continue;

				y[bndSite] -= static_cast<double>(curNu[j]) * y[bndIdx * numStates];

				// Next bound component
				++bndIdx;
			}

			y[bndSite] /= static_cast<double>(curNu[0]);
		}

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

	unsigned int _numBindingComp; //!< Number of binding components

	template <typename StateType, typename CpStateType, typename ResidualType, typename ParamType>
	int fluxImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* y,
				 CpStateType const* yCp, ResidualType* res, LinearBufferAllocator workSpace) const
	{
		using CpStateParamType = typename DoubleActivePromoter<CpStateType, ParamType>::type;
		using StateParamType = typename DoubleActivePromoter<StateType, ParamType>::type;

		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		const unsigned int numStates = p->lambda.size();

		// Ordering of the states is (q_{comp,state})
		// q_{0,0}, q{0,1}, q_{0,2}, q_{1,0}, q_{1,1}, q_{1,2}, ...
		// A state corresponds to a type of binding site. It is assumed that all components have either 0
		// or the same number of states. Thus, a component is either non-binding or has nSites bound states.
		//
		// The same ordering is used for the fluxes. That is, q_{0,0}, q_{1,0} and q_{0,1}, q_{1,1} and ... each
		// form one SMA binding model system.

		// Loop over all binding site types
		for (unsigned int bndSite = 0; bndSite < numStates; ++bndSite, ++y, ++res)
		{
			// y, yDot, and res point to q_{0,site}

			active const* const curNu = p->nu[bndSite];
			active const* const curSigma = p->sigma[bndSite];
			active const* const curKa = p->kA[bndSite];
			active const* const curKd = p->kD[bndSite];

			const ParamType refC0 = static_cast<ParamType>(p->refC0[bndSite]);
			const ParamType refQ = static_cast<ParamType>(p->refQ[bndSite]);

			// Salt flux: q_0 - Lambda + Sum[nu_j * q_j, j] == 0
			//       <=>  q_0 == Lambda - Sum[nu_j * q_j, j]
			// Also compute \bar{q}_0 = q_0 - Sum[sigma_j * q_j, j]
			res[0] = static_cast<ParamType>(curNu[0]) * y[0] - static_cast<ParamType>(p->lambda[bndSite]);
			StateParamType q0_bar = static_cast<ParamType>(curNu[0]) * y[0];

			// bndIdx is used as a counter inside one binding site type
			// Getting from one component to another requires a step size of numStates (stride)
			unsigned int bndIdx = 1;
			for (int j = 1; j < _nComp; ++j)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[j] == 0)
					continue;

				res[0] += static_cast<ParamType>(curNu[j]) * y[bndIdx * numStates];
				q0_bar -= static_cast<ParamType>(curSigma[j]) * y[bndIdx * numStates];

				// Next bound component
				++bndIdx;
			}

			const ResidualType q0_bar_divRef = q0_bar / refQ;
			const ResidualType yCp0_divRef = yCp[0] / refC0;

			// Protein fluxes: -k_{a,i} * c_{p,i} * \bar{q}_0^{nu_i} + k_{d,i} * q_i * c_{p,0}^{nu_i}
			bndIdx = 1;
			for (int i = 1; i < _nComp; ++i)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[i] == 0)
					continue;

				const ParamType nu_over_nu0 = static_cast<ParamType>(curNu[i]) / static_cast<ParamType>(curNu[0]);
				const CpStateParamType c0_pow_nu_divRef = pow(yCp0_divRef, nu_over_nu0);
				const StateParamType q0_bar_pow_nu_divRef = pow(q0_bar_divRef, nu_over_nu0);

				// Residual
				res[bndIdx * numStates] = static_cast<ParamType>(curKd[i]) * y[bndIdx * numStates] * c0_pow_nu_divRef -
										  static_cast<ParamType>(curKa[i]) * yCp[i] * q0_bar_pow_nu_divRef;

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

		const unsigned int numStates = p->lambda.size();

		// Ordering of the states is (q_{comp,state}, example uses 2 components, 3 binding sites)
		// q_{0,0}, q{0,1}, q_{0,2}, q_{1,0}, q_{1,1}, q_{1,2}, ...
		// Ordering of the fluxes is the same, that is, we need nSites steps to jump from one flux of an SMA
		// binding model system to the next.

		// Loop over all binding site types
		for (unsigned int bndSite = 0; bndSite < numStates; ++bndSite, ++y)
		{
			// Jump from first row to current Salt row
			RowIterator curJac = jac + bndSite;

			active const* const curNu = p->nu[bndSite];
			active const* const curSigma = p->sigma[bndSite];
			active const* const curKa = p->kA[bndSite];
			active const* const curKd = p->kD[bndSite];
			const double refC0 = static_cast<double>(p->refC0[bndSite]);
			const double refQ = static_cast<double>(p->refQ[bndSite]);
			double q0_bar = static_cast<double>(curNu[0]) * y[0];

			// Salt flux: q_0 - Lambda + Sum[nu_j * q_j, j] == 0
			curJac[0] = static_cast<double>(curNu[0]);

			int bndIdx = 1;
			for (int j = 1; j < _nComp; ++j)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[j] == 0)
					continue;

				curJac[bndIdx * numStates] = static_cast<double>(curNu[j]);

				// Calculate \bar{q}_0 = q_0 - Sum[sigma_j * q_j, j]
				q0_bar -= static_cast<double>(curSigma[j]) * y[bndIdx * numStates];

				// Next bound component
				++bndIdx;
			}

			// Advance to protein fluxes
			curJac += numStates;

			const double q0_bar_divRef = q0_bar / refQ;
			const double yCp0_divRef = yCp[0] / refC0;

			// Protein fluxes: -k_{a,i} * c_{p,i} * \bar{q}_0^{nu_i} + k_{d,i} * q_i * c_{p,0}^{nu_i}
			// We have already computed \bar{q}_0 in the loop above
			bndIdx = 1;
			for (int i = 1; i < _nComp; ++i)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[i] == 0)
					continue;

				// Getting to c_{p,0}: -numStates * bndIdx takes us to q_{0,site}, another -bndSite to q_{0,0}. From
				// there, we
				//                     take a -offsetCp to reach c_{p,0}.
				//                     This means jac[-bndSite - offsetCp - numStates * bndIdx] corresponds to c_{p,0}.
				// Getting to c_{p,i}: Go to c_{p,0} and add i.
				//                     This means jac[i - bndSite - offsetCp - numStates * bndIdx] corresponds to
				//                     c_{p,i}.

				const double ka = static_cast<double>(curKa[i]);
				const double kd = static_cast<double>(curKd[i]);
				const double nu_over_nu0 = static_cast<double>(curNu[i]) / static_cast<double>(curNu[0]);

				const double c0_pow_nu = pow(yCp0_divRef, nu_over_nu0);
				const double q0_bar_pow_nu = pow(q0_bar_divRef, nu_over_nu0);
				const double c0_pow_nu_m1_divRef = pow(yCp0_divRef, nu_over_nu0 - 1.0);
				const double q0_bar_pow_nu_m1_divRef = pow(q0_bar_divRef, nu_over_nu0 - 1.0) / refQ;

				// dres_i / dc_{p,0}
				curJac[-bndSite - offsetCp - numStates * bndIdx] =
					kd * y[bndIdx * numStates] * nu_over_nu0 * c0_pow_nu_m1_divRef / refC0;
				// dres_i / dc_{p,i}
				curJac[i - bndSite - offsetCp - numStates * bndIdx] = -ka * q0_bar_pow_nu;
				// dres_i / dq_{0,bndSite}
				curJac[-bndIdx * numStates] =
					-ka * yCp[i] * nu_over_nu0 * q0_bar_pow_nu_m1_divRef * static_cast<double>(curNu[0]);

				// Fill dres_i / dq_{j,bndSite}
				int bndIdx2 = 1;
				for (int j = 1; j < _nComp; ++j)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[j] == 0)
						continue;

					// dres_i / dq_{j,bndSite}
					curJac[(bndIdx2 - bndIdx) * numStates] =
						-ka * yCp[i] * nu_over_nu0 * q0_bar_pow_nu_m1_divRef * (-static_cast<double>(curSigma[j]));
					// Getting to q_j: -bndIdx * numStates takes us to q_{0,bndSite}, another +bndIdx2 * numStates to
					// q_{j,bndSite}. This means curJac[(bndIdx2 - bndIdx) * numStates] corresponds to q_{j,bndSite}.

					++bndIdx2;
				}

				// Add to dres_i / dq_{i,bndSite}
				curJac[0] += kd * c0_pow_nu;

				// Advance to next flux
				++bndIdx;
				curJac += numStates;
				// Note that there is a spacing of numStates between the fluxes inside one binding site type
			}
		}
	}
};

typedef BiStericMassActionBindingBase<BiSMAParamHandler> BiStericMassActionBinding;
typedef BiStericMassActionBindingBase<ExtBiSMAParamHandler> ExternalBiStericMassActionBinding;

namespace binding
{
void registerBiStericMassActionModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
{
	bindings[BiStericMassActionBinding::identifier()] = []() { return new BiStericMassActionBinding(); };
	bindings[ExternalBiStericMassActionBinding::identifier()] = []() {
		return new ExternalBiStericMassActionBinding();
	};
}
} // namespace binding

} // namespace model

} // namespace cadet
