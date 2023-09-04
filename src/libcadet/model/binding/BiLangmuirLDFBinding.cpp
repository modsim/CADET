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
#include "SlicedVector.hpp"
#include "LocalVector.hpp"
#include "SimulationTypes.hpp"

#include <vector>
#include <unordered_map>
#include <functional>

/*<codegen>
{
	"name": "BiLangmuirLDFParamHandler",
	"externalName": "ExtBiLangmuirLDFParamHandler",
	"parameters":
		[
			{ "type": "ComponentBoundStateMajorDependentParameter", "varName": "keq", "confName": "MCBLLDF_KEQ"},
			{ "type": "ComponentBoundStateMajorDependentParameter", "varName": "kkin", "confName": "MCBLLDF_KKIN"},
			{ "type": "ComponentBoundStateMajorDependentParameter", "varName": "qMax", "confName": "MCBLLDF_QMAX"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 keq = Equillibrium constant in binding site-major ordering
 kkkin = Linear driving force coefficient in binding site-major ordering
 qMax = Capacity in binding site-major ordering
*/

namespace cadet
{

namespace model
{

inline const char* BiLangmuirLDFParamHandler::identifier() CADET_NOEXCEPT
{
	return "MULTI_COMPONENT_BILANGMUIR_LDF";
}

inline bool BiLangmuirLDFParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_keq.size() != _kkin.size()) || (_keq.size() != _qMax.size()) || (_keq.size() < nComp))
		throw InvalidParameterException("MCBLLDF_KEQ, MCBLLDF_KKIN, and MCBLLDF_QMAX have to have the same size");

	return true;
}

inline const char* ExtBiLangmuirLDFParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_MULTI_COMPONENT_BILANGMUIR_LDF";
}

inline bool ExtBiLangmuirLDFParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_keq.size() != _kkin.size()) || (_keq.size() != _qMax.size()) || (_keq.size() < nComp))
		throw InvalidParameterException(
			"EXT_MCBLLDF_KEQ, EXT_MCBLLDF_KKIN, and EXT_MCBLLDF_QMAX have to have the same size");

	return true;
}

/**
 * @brief Defines the multi component Bi-Langmuir binding model
 * @details Implements the Bi-Langmuir adsorption model: \f[ \begin{align}
 *              \frac{\mathrm{d}q_i^A}{\mathrm{d}t} &= k_{a,i}^A c_{p,i} q_{\text{max},i}^A \left( 1 - \sum_j
 * \frac{q_j^A}{q_{\text{max},j}^A} \right) - k_{d,i}^A q_i^A \\
 *              \frac{\mathrm{d}q_i^B}{\mathrm{d}t} &= k_{a,i}^B c_{p,i} q_{\text{max},i}^B \left( 1 - \sum_j
 * \frac{q_j^B}{q_{\text{max},j}^B} \right) - k_{d,i}^B q_i^B \\ \vodts & \vdots \end{align} \f] Here, several different
 * types of binding sites @f$ q^A @f$, @f$ q^B @f$, etc. are considered. A molecule can either bind to site A or B (or
 * C, etc.). A direct exchange between the different binding sites does not occur. While components without bound state
 * (i.e., non-binding components) are supported, all other components must have the same number of bound states (i.e.,
 * binding sites).
 *
 *          Internal state vector order is component-major. The state vector is composed of all components and within
 * each component all bound states are listed.
 *
 *          See @cite Guiochon2006.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t> class BiLangmuirLDFBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:
	BiLangmuirLDFBindingBase()
	{
	}
	virtual ~BiLangmuirLDFBindingBase() CADET_NOEXCEPT
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
				throw InvalidParameterException("Bi-Langmuir LDF binding model requires all components to have the "
												"same number of bound states or zero");
		}

		_numBindingComp = numBindingComponents(_nBoundStates, _nComp);

		// Allocate space for parameters
		_paramHandler.reserve(nComp * numSlices, numSlices, nComp, nBound);

		return res;
	}

	virtual bool hasSalt() const CADET_NOEXCEPT
	{
		return false;
	}
	virtual bool supportsMultistate() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool supportsNonBinding() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT
	{
		return true;
	}

	/*virtual void timeDerivativeQuasiStationaryFluxes(double t, unsigned int secIdx, const ColumnPosition& colPos,
	double const* yCp, double const* y, double* dResDt, LinearBufferAllocator workSpace) const
	{
		if (!this->hasQuasiStationaryReactions())
			return;

		if (!ParamHandler_t::dependsOnTime())
			return;

		// Update external function and compute time derivative of parameters
		typename ParamHandler_t::ParamsHandle p;
		typename ParamHandler_t::ParamsHandle dpDt;
		std::tie(p, dpDt) = _paramHandler.updateTimeDerivative(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Protein flux: -k_{a,i}^j * c_{p,i} * (1 - \sum q_i^j / q_{max,i}^j) + k_{d,i}^j * q_i^j

		const unsigned int nSites = p->kA.slices();

		// Ordering of the states is (q_{comp,state})
		// q_{0,0}, q{0,1}, q_{0,2}, q_{1,0}, q_{1,1}, q_{1,2}, ...
		// A state corresponds to a type of binding site. It is assumed that all components have either 0
		// or the same number of states. Thus, a component is either non-binding or has nSites bound states.
		//
		// The same ordering is used for the fluxes. That is, q_{0,0}, q_{1,0} and q_{0,1}, q_{1,1} and ... each
		// form one Langmuir binding model system.

		// Loop over all binding site types
		for (unsigned int site = 0; site < nSites; ++site, ++y, ++dResDt)
		{
			// y, and dResDt point to q_{0,site}

			// Get parameter slice for current binding site type
			active const* const localKa = p->kA[site];
			//			active const* const localKd = p->kD[site];
			active const* const localQmax = p->qMax[site];
			active const* const localKaT = dpDt->kA[site];
			active const* const localKdT = dpDt->kD[site];
			active const* const localQmaxT = dpDt->qMax[site];

			double qSum = 1.0;
			double qSumT = 0.0;
			unsigned int bndIdx = 0;

			// bndIdx is used as a counter inside one binding site type
			// Getting from one component to another requires a step size of nSites (stride)

			for (int i = 0; i < _nComp; ++i)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[i] == 0)
					continue;

				const double summand = y[bndIdx * nSites] / static_cast<double>(localQmax[i]);
				qSum -= summand;
				qSumT += summand / static_cast<double>(localQmax[i]) * static_cast<double>(localQmaxT[i]);

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
				dResDt[bndIdx * nSites] = static_cast<double>(localKdT[i]) * y[bndIdx * nSites]
					- yCp[i] * (static_cast<double>(localKaT[i]) * static_cast<double>(localQmax[i]) * qSum
						+ static_cast<double>(localKa[i]) * static_cast<double>(localQmaxT[i]) * qSum
						+ static_cast<double>(localKa[i]) * static_cast<double>(localQmax[i]) * qSumT);

				// Next bound component
				++bndIdx;
			}
		}
	}*/

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
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Protein flux: \frac{dq_{i,j}}{dt} = k_{kin,i,j}(q_{i,j}^*-q_{i,j}) with
		// q_{i,j}^*=\frac{q_{m,i,j} k_{eq,i,j} c_i}{1 + \sum_{k=1}^{n_{comp}}{k_{eq,k,j} c_k}}

		const unsigned int nSites = p->keq.slices();

		// Ordering of the states is (q_{comp,state})
		// q_{0,0}, q{0,1}, q_{0,2}, q_{1,0}, q_{1,1}, q_{1,2}, ...
		// A state corresponds to a type of binding site. It is assumed that all components have either 0
		// or the same number of states. Thus, a component is either non-binding or has nSites bound states.
		//
		// The same ordering is used for the fluxes. That is, q_{0,0}, q_{1,0} and q_{0,1}, q_{1,1} and ... each
		// form one Langmuir binding model system.

		// Loop over all binding site types
		for (unsigned int site = 0; site < nSites; ++site, ++y, ++res)
		{
			// y, and res point to q_{0,site}

			// Get parameter slice for current binding site type
			active const* const localKeq = p->keq[site];
			active const* const localKkin = p->kkin[site];
			active const* const localQmax = p->qMax[site];

			ResidualType cpSum = 1.0;
			unsigned int bndIdx = 0;

			// bndIdx is used as a counter inside one binding site type
			// Getting from one component to another requires a step size of nSites (stride)

			for (int i = 0; i < _nComp; ++i)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[i] == 0)
					continue;

				cpSum += yCp[i] * static_cast<ParamType>(localKeq[i]);

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
				res[bndIdx * nSites] = static_cast<ParamType>(localKkin[i]) *
									   (y[bndIdx * nSites] - static_cast<ParamType>(localKeq[i]) * yCp[i] *
																 static_cast<ParamType>(localQmax[i]) / cpSum);

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

		// Protein flux: \frac{dq_{i,j}}{dt} = k_{kin,i,j}(q_{i,j}^*-q_{i,j}) with
		// q_{i,j}^*=\frac{q_{m,i,j} k_{eq,i,j} c_i}{1 + \sum_{k=1}^{n_{comp}}{k_{eq,k,j} c_k}}

		// Ordering of the states is (q_{comp,state}, example uses 2 components, 3 binding sites)
		// q_{0,0}, q{0,1}, q_{0,2}, q_{1,0}, q_{1,1}, q_{1,2}, ...
		// Ordering of the fluxes is the same, that is, we need nSites steps to jump from one flux of a Langmuir
		// binding model system to the next.

		const int nSites = static_cast<int>(p->keq.slices());

		// Loop over all binding site types
		for (int site = 0; site < nSites; ++site, ++y)
		{
			// Get parameter slice for current binding site type
			active const* const localKeq = p->keq[site];
			active const* const localKkin = p->kkin[site];
			active const* const localQmax = p->qMax[site];

			double cpSum = 1.0;
			int bndIdx = 0;
			for (int i = 0; i < _nComp; ++i)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[i] == 0)
					continue;

				cpSum += yCp[i] * static_cast<double>(localKeq[i]);

				// Next bound component
				++bndIdx;
			}

			bndIdx = 0;
			for (int i = 0; i < _nComp; ++i)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[i] == 0)
					continue;

				const double keq = static_cast<double>(localKeq[i]);
				const double kkin = static_cast<double>(localKkin[i]);

				// dres_i / dc_{p,i}
				jac[i - site - offsetCp - nSites * bndIdx] = -kkin * keq * static_cast<double>(localQmax[i]) / cpSum;
				// Getting to c_{p,i}: -nSites * bndIdx takes us to q_{0,site}, another -site to q_{0,0}. From there, we
				//                     take a -offsetCp to reach c_{p,0} and a +i to arrive at c_{p,i}.
				//                     This means jac[i - site - offsetCp - nSites * bndIdx] corresponds to c_{p,i}.

				// Fill dres_i / dc_j
				const double commonFactor = kkin * keq * static_cast<double>(localQmax[i]) * yCp[i] / (cpSum * cpSum);
				int bndIdx2 = 0;
				for (int j = 0; j < _nComp; ++j)
				{
					// Skip components without bound states (bound state index bndIdx2 is not advanced)
					if (_nBoundStates[j] == 0)
						continue;

					// dres_i / dc_j
					jac[j - site - offsetCp - nSites * bndIdx] += static_cast<double>(localKeq[j]) * commonFactor;
					// Getting to c_{p,j}: -nSites * bndIdx takes us to q_{0,site}, another -site to q_{0,0}. From
					// there, we
					//                     take a -offsetCp to reach c_{p,0} and a +j to arrive at c_{p,j}.
					//                     This means jac[j - site - offsetCp - nSites * bndIdx] corresponds to c_{p,j}.

					++bndIdx2;
				}

				// Add to dres_i / dq_{i,site}
				jac[0] += kkin;

				// Advance to next flux and Jacobian row
				++bndIdx;
				jac += nSites;
				// Note that there is a spacing of nSites between the fluxes inside one binding site type
			}
			// We are at the end of the flux block q_{_numBindingComp,site} and need to jump back to the beginning
			// by using -_numBindingComp * nSites steps. From there, we take one step forward to arrive at q_{0,site+1}.
			jac -= _numBindingComp * nSites - 1;
		}
	}
};

typedef BiLangmuirLDFBindingBase<BiLangmuirLDFParamHandler> BiLangmuirLDFBinding;
typedef BiLangmuirLDFBindingBase<ExtBiLangmuirLDFParamHandler> ExternalBiLangmuirLDFBinding;

namespace binding
{
void registerBiLangmuirLDFModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
{
	bindings[BiLangmuirLDFBinding::identifier()] = []() { return new BiLangmuirLDFBinding(); };
	bindings[ExternalBiLangmuirLDFBinding::identifier()] = []() { return new ExternalBiLangmuirLDFBinding(); };
}
} // namespace binding

} // namespace model

} // namespace cadet
