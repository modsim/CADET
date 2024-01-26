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
	"name": "ColloidalParamHandler",
	"externalName": "ExtColloidalParamHandler",
	"parameters":
		[
			{ "type": "ScalarParameter", "varName": "phi", "confName": "COL_PHI"},
			{ "type": "ScalarParameter", "varName": "kappaExp", "confName": "COL_KAPPA_EXP"},
			{ "type": "ScalarParameter", "varName": "kappaFact", "confName": "COL_KAPPA_FACT"},
			{ "type": "ScalarParameter", "varName": "kappaConst", "confName": "COL_KAPPA_CONST"},
			{ "type": "ScalarParameter", "varName": "cordNum", "confName": "COL_CORDNUM"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kEqPhExp", "confName": "COL_LOGKEQ_PH_EXP"},
			{ "type": "ScalarComponentDependentParameter", "varName": "kEqSaltPowerExp", "confName":
"COL_LOGKEQ_SALT_POWEXP"}, { "type": "ScalarComponentDependentParameter", "varName": "kEqSaltPowerFact", "confName":
"COL_LOGKEQ_SALT_POWFACT"}, { "type": "ScalarComponentDependentParameter", "varName": "kEqSaltExpFact", "confName":
"COL_LOGKEQ_SALT_EXPFACT"}, { "type": "ScalarComponentDependentParameter", "varName": "kEqSaltExpExp", "confName":
"COL_LOGKEQ_SALT_EXPARGMULT"}, { "type": "ScalarComponentDependentParameter", "varName": "bppPhExp", "confName":
"COL_BPP_PH_EXP"}, { "type": "ScalarComponentDependentParameter", "varName": "bppSaltPowerExp", "confName":
"COL_BPP_SALT_POWEXP"}, { "type": "ScalarComponentDependentParameter", "varName": "bppSaltPowerFact", "confName":
"COL_BPP_SALT_POWFACT"}, { "type": "ScalarComponentDependentParameter", "varName": "bppSaltExpFact", "confName":
"COL_BPP_SALT_EXPFACT"}, { "type": "ScalarComponentDependentParameter", "varName": "bppSaltExpExp", "confName":
"COL_BPP_SALT_EXPARGMULT"}, { "type": "ScalarComponentDependentParameter", "varName": "radius", "confName":
"COL_RADIUS"}, { "type": "ScalarComponentDependentParameter", "varName": "kKin", "confName": "COL_KKIN"}
		],
	"constantParameters":
		[
			{ "type": "ScalarParameter", "varName": "linThreshold", "confName": "COL_LINEAR_THRESHOLD"},
			{ "type": "ScalarBoolParameter", "varName": "usePh", "confName": "COL_USE_PH"}
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

inline const char* ColloidalParamHandler::identifier() CADET_NOEXCEPT
{
	return "MULTI_COMPONENT_COLLOIDAL";
}

inline bool ColloidalParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kEqPhExp.size() != _kEqSaltPowerExp.size()) || (_kEqPhExp.size() != _kEqSaltPowerFact.size()) ||
		(_kEqPhExp.size() != _kEqSaltExpFact.size()) || (_kEqPhExp.size() != _kEqSaltExpExp.size()) ||
		(_kEqPhExp.size() != _radius.size()) || (_kEqPhExp.size() != _kKin.size()) || (_kEqPhExp.size() < nComp))
		throw InvalidParameterException(
			"COL_LOGKEQ_PH_EXP, COL_LOGKEQ_SALT_POWEXP, COL_LOGKEQ_SALT_POWFACT, COL_LOGKEQ_SALT_EXPFACT, "
			"COL_LOGKEQ_SALT_EXPMULT, COL_RADIUS, and COL_KKIN have to have the same size");

	if ((_kEqPhExp.size() != _bppPhExp.size()) || (_kEqPhExp.size() != _bppSaltPowerExp.size()) ||
		(_kEqPhExp.size() != _bppSaltPowerFact.size()) || (_kEqPhExp.size() != _bppSaltExpFact.size()) ||
		(_kEqPhExp.size() != _bppSaltExpExp.size()) || (_kEqPhExp.size() < nComp))
		throw InvalidParameterException("COL_LOGKEQ_PH_EXP, COL_BPP_PH_EXP, COL_BPP_SALT_POWEXP, COL_BPP_SALT_POWFACT, "
										"COL_BPP_SALT_EXPFACT, and COL_BPP_SALT_EXPMULT have to have the same size");

	return true;
}

inline const char* ExtColloidalParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_MULTI_COMPONENT_COLLOIDAL";
}

inline bool ExtColloidalParamHandler::validateConfig(unsigned int nComp, unsigned int const* nBoundStates)
{
	if ((_kEqPhExp.size() != _kEqSaltPowerExp.size()) || (_kEqPhExp.size() != _kEqSaltPowerFact.size()) ||
		(_kEqPhExp.size() != _kEqSaltExpFact.size()) || (_kEqPhExp.size() != _kEqSaltExpExp.size()) ||
		(_kEqPhExp.size() != _radius.size()) || (_kEqPhExp.size() != _kKin.size()) || (_kEqPhExp.size() < nComp))
		throw InvalidParameterException(
			"EXTCOL_LOGKEQ_PH_EXP, EXTCOL_LOGKEQ_SALT_POWEXP, EXTCOL_LOGKEQ_SALT_POWFACT, EXTCOL_LOGKEQ_SALT_EXPFACT, "
			"EXTCOL_LOGKEQ_SALT_EXPMULT, EXTCOL_RADIUS, and EXTCOL_KKIN have to have the same size");

	if ((_kEqPhExp.size() != _bppPhExp.size()) || (_kEqPhExp.size() != _bppSaltPowerExp.size()) ||
		(_kEqPhExp.size() != _bppSaltPowerFact.size()) || (_kEqPhExp.size() != _bppSaltExpFact.size()) ||
		(_kEqPhExp.size() != _bppSaltExpExp.size()) || (_kEqPhExp.size() < nComp))
		throw InvalidParameterException(
			"EXTCOL_LOGKEQ_PH_EXP, EXTCOL_BPP_PH_EXP, EXTCOL_BPP_SALT_POWEXP, EXTCOL_BPP_SALT_POWFACT, "
			"EXTCOL_BPP_SALT_EXPFACT, and EXTCOL_BPP_SALT_EXPMULT have to have the same size");

	return true;
}

/**
 * @brief Defines the multi component colloidal binding model
 * @details Implements the colloidal adsorption model: \f[ \begin{align}
 *              \frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i} q_{\text{max},i} \left( 1 - \sum_j
 * \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} q_i \end{align} \f] Multiple bound states are not supported.
 *          Components without bound state (i.e., non-binding components) are supported.
 *          The first component (index @c 0) is salt, the second component (index @c 1)
 *          can be pH. Both components have to be non-binding.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t> class ColloidalBindingBase : public ParamHandlerBindingModelBase<ParamHandler_t>
{
public:
	ColloidalBindingBase() : _startIdx(1), _rFactor(1.9174253548654209e-24)
	{
		// _rFactor: 1.9174253548654209e-24 = 2 / (N_Avogadro * sqrt(3))
	}
	virtual ~ColloidalBindingBase() CADET_NOEXCEPT
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

	virtual void timeDerivativeQuasiStationaryFluxes(double t, unsigned int secIdx, const ColumnPosition& colPos,
													 double const* yCp, double const* y, double* dResDt,
													 LinearBufferAllocator workSpace) const
	{
		if (!this->hasQuasiStationaryReactions())
			return;

		if (!ParamHandler_t::dependsOnTime())
			return;

		// TODO: Compute derivative
	}

	CADET_BINDINGMODELBASE_BOILERPLATE

protected:
	using ParamHandlerBindingModelBase<ParamHandler_t>::_paramHandler;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_reactionQuasistationarity;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_nComp;
	using ParamHandlerBindingModelBase<ParamHandler_t>::_nBoundStates;

	int _startIdx;
	const double _rFactor;

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx)
	{
		const bool res =
			ParamHandlerBindingModelBase<ParamHandler_t>::configureImpl(paramProvider, unitOpIdx, parTypeIdx);

		if (_nComp <= 1)
			throw InvalidParameterException("No protein component present");

		if (_nBoundStates[0] != 0)
			throw InvalidParameterException("Salt component (index 0) must be non-binding (NBOUND = 0)");

		if (_paramHandler.usePh().get() && (_nComp <= 2))
			throw InvalidParameterException("No protein component present (existing two components are salt and PH)");

		if (_paramHandler.usePh().get())
		{
			_startIdx = 2;
			if (_nBoundStates[1] != 0)
				throw InvalidParameterException("PH pseudocomponent (index 1) must be non-binding (NBOUND = 0)");
		}

		for (int i = 0; i < _nComp; ++i)
		{
			if (_nBoundStates[i] > 1)
				throw InvalidParameterException("Binding model supports at most one bound state per component");
		}

		return res;
	}

	template <typename StateType, typename CpStateType, typename ResidualType, typename ParamType>
	int fluxImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* y,
				 CpStateType const* yCp, ResidualType* res, LinearBufferAllocator workSpace) const
	{
		using CpStateParamType = typename DoubleActivePromoter<CpStateType, ParamType>::type;
		using StateParamType = typename DoubleActivePromoter<StateType, ParamType>::type;

		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		ResidualType qSum = 0.0;
		int bndIdx = 0;
		for (int i = _startIdx; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			qSum += y[bndIdx];

			// Next bound component
			++bndIdx;
		}

		const bool phEnabled = (_startIdx == 2);
		const CpStateType& cpSalt = yCp[0];

		if (qSum <= p->linThreshold)
		{
			bndIdx = 0;
			for (int i = _startIdx; i < _nComp; ++i)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[i] == 0)
					continue;

				StateParamType logKeq = pow(cpSalt, -static_cast<ParamType>(p->kEqSaltPowerExp[i])) *
											static_cast<ParamType>(p->kEqSaltPowerFact[i]) +
										exp(cpSalt * static_cast<ParamType>(p->kEqSaltExpExp[i])) *
											static_cast<ParamType>(p->kEqSaltExpFact[i]);

				if (phEnabled)
				{
					const CpStateType& pH = yCp[1];
					logKeq *= pow(pH, static_cast<ParamType>(p->kEqPhExp[i]));
				}

				// Residual
				res[bndIdx] = static_cast<ParamType>(p->kKin[i]) * (y[bndIdx] * exp(-logKeq) - yCp[i]);

				// Next bound component
				++bndIdx;
			}
		}
		else
		{
			const CpStateParamType kappa =
				1e9 / (pow(cpSalt, static_cast<ParamType>(p->kappaExp)) * static_cast<ParamType>(p->kappaFact) +
					   static_cast<ParamType>(p->kappaConst));
			const StateParamType R = sqrt(_rFactor * static_cast<ParamType>(p->phi) / qSum);
			const StateParamType Sfactor = static_cast<ParamType>(p->cordNum) * 0.125 / qSum * (3.0 / R + kappa);

			bndIdx = 0;
			for (int i = _startIdx; i < _nComp; ++i)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[i] == 0)
					continue;

				StateParamType logKeq = pow(cpSalt, -static_cast<ParamType>(p->kEqSaltPowerExp[i])) *
											static_cast<ParamType>(p->kEqSaltPowerFact[i]) +
										exp(cpSalt * static_cast<ParamType>(p->kEqSaltExpExp[i])) *
											static_cast<ParamType>(p->kEqSaltExpFact[i]);
				StateParamType bpp_i = pow(cpSalt, static_cast<ParamType>(p->bppSaltPowerExp[i])) *
										   static_cast<ParamType>(p->bppSaltPowerFact[i]) +
									   exp(cpSalt * static_cast<ParamType>(p->bppSaltExpExp[i])) *
										   static_cast<ParamType>(p->bppSaltExpFact[i]);

				if (phEnabled)
				{
					const CpStateType& pH = yCp[1];

					logKeq *= pow(pH, static_cast<ParamType>(p->kEqPhExp[i]));
					bpp_i *= pow(pH, static_cast<ParamType>(p->bppPhExp[i]));
				}

				StateParamType S_i = 0.0;
				int bndIdx2 = 0;
				for (int j = _startIdx; j < _nComp; ++j)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[j] == 0)
						continue;

					StateParamType bpp_j = pow(cpSalt, static_cast<ParamType>(p->bppSaltPowerExp[j])) *
											   static_cast<ParamType>(p->bppSaltPowerFact[j]) +
										   exp(cpSalt * static_cast<ParamType>(p->bppSaltExpExp[j])) *
											   static_cast<ParamType>(p->bppSaltExpFact[j]);
					if (phEnabled)
					{
						const CpStateType& pH = yCp[1];
						bpp_j *= pow(pH, static_cast<ParamType>(p->bppPhExp[j]));
					}

					const ParamType radSum =
						static_cast<ParamType>(p->radius[i]) + static_cast<ParamType>(p->radius[j]);
					S_i += y[bndIdx2] * sqrt(bpp_i * bpp_j) * radSum * exp(-kappa * (R - radSum));

					// Next bound component
					++bndIdx2;
				}

				// Residual
				res[bndIdx] = static_cast<ParamType>(p->kKin[i]) * (y[bndIdx] * exp(S_i * Sfactor - logKeq) - yCp[i]);

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

		double qSum = 0.0;
		int bndIdx = 0;
		for (int i = _startIdx; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nBoundStates[i] == 0)
				continue;

			qSum += y[bndIdx];

			// Next bound component
			++bndIdx;
		}

		const bool phEnabled = (_startIdx == 2);
		const double cpSalt = yCp[0];

		if (qSum <= p->linThreshold)
		{
			bndIdx = 0;
			for (int i = _startIdx; i < _nComp; ++i)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[i] == 0)
					continue;

				double kKinKeq =
					pow(cpSalt, -static_cast<double>(p->kEqSaltPowerExp[i])) *
						static_cast<double>(p->kEqSaltPowerFact[i]) +
					exp(cpSalt * static_cast<double>(p->kEqSaltExpExp[i])) * static_cast<double>(p->kEqSaltExpFact[i]);
				double logKeq_dSalt = -static_cast<double>(p->kEqSaltPowerExp[i]) *
										  pow(cpSalt, -static_cast<double>(p->kEqSaltPowerExp[i]) - 1.0) *
										  static_cast<double>(p->kEqSaltPowerFact[i]) +
									  exp(cpSalt * static_cast<double>(p->kEqSaltExpExp[i])) *
										  static_cast<double>(p->kEqSaltExpFact[i]) *
										  static_cast<double>(p->kEqSaltExpExp[i]);

				if (phEnabled)
				{
					const double pH = yCp[1];
					const double pHfactor = pow(pH, static_cast<double>(p->kEqPhExp[i]));
					const double logKeq_dPh =
						pow(pH, static_cast<double>(p->kEqPhExp[i]) - 1.0) * static_cast<double>(p->kEqPhExp[i]);
					kKinKeq *= pHfactor;
					logKeq_dSalt *= pHfactor;

					kKinKeq = exp(-kKinKeq) * static_cast<double>(p->kKin[i]);

					// dres_i / dc_{p,1} (pH)
					jac[1 - bndIdx - offsetCp] = kKinKeq * (-logKeq_dPh) * y[bndIdx];
				}
				else
					kKinKeq = exp(-kKinKeq) * static_cast<double>(p->kKin[i]);

				// dres_i / dc_{p,0} (salt)
				jac[0 - bndIdx - offsetCp] = kKinKeq * (-logKeq_dSalt) * y[bndIdx];

				// dres_i / dc_{p,i}
				jac[i - bndIdx - offsetCp] = -static_cast<double>(p->kKin[i]);

				// dres_i / dq_i
				jac[0] = kKinKeq;

				// Advance to next flux and Jacobian row
				++bndIdx;
				++jac;
			}
		}
		else
		{
			const double kappa_denom =
				pow(cpSalt, static_cast<double>(p->kappaExp)) * static_cast<double>(p->kappaFact) +
				static_cast<double>(p->kappaConst);
			const double kappa = 1e9 / kappa_denom;
			const double kappa_dSalt = -1e9 / sqr(kappa_denom) * pow(cpSalt, static_cast<double>(p->kappaExp) - 1.0) *
									   static_cast<double>(p->kappaExp) * static_cast<double>(p->kappaFact);
			const double R = sqrt(_rFactor * static_cast<double>(p->phi) / qSum);
			const double R_dq = -0.5 * R / qSum;
			const double Sfactor = static_cast<double>(p->cordNum) * 0.125 / qSum * (3.0 / R + kappa);
			const double Sfactor_dSalt = static_cast<double>(p->cordNum) * 0.125 / qSum * kappa_dSalt;
			const double Sfactor_dq =
				-static_cast<double>(p->cordNum) * 0.125 * ((3.0 / R + kappa) / qSum + 3.0 / sqr(R) * R_dq) / qSum;

			bndIdx = 0;
			for (int i = _startIdx; i < _nComp; ++i)
			{
				// Skip components without bound states (bound state index bndIdx is not advanced)
				if (_nBoundStates[i] == 0)
					continue;

				// dres_i / dc_{p,i}
				jac[i - bndIdx - offsetCp] = -static_cast<double>(p->kKin[i]);

				double logKeq =
					pow(cpSalt, -static_cast<double>(p->kEqSaltPowerExp[i])) *
						static_cast<double>(p->kEqSaltPowerFact[i]) +
					exp(cpSalt * static_cast<double>(p->kEqSaltExpExp[i])) * static_cast<double>(p->kEqSaltExpFact[i]);
				double logKeq_dSalt = -static_cast<double>(p->kEqSaltPowerExp[i]) *
										  pow(cpSalt, -static_cast<double>(p->kEqSaltPowerExp[i]) - 1.0) *
										  static_cast<double>(p->kEqSaltPowerFact[i]) +
									  exp(cpSalt * static_cast<double>(p->kEqSaltExpExp[i])) *
										  static_cast<double>(p->kEqSaltExpFact[i]) *
										  static_cast<double>(p->kEqSaltExpExp[i]);
				double logKeq_dPh = 0.0;

				if (phEnabled)
				{
					const double pH = yCp[1];
					const double pHfactor = pow(pH, static_cast<double>(p->kEqPhExp[i]));
					logKeq *= pHfactor;
					logKeq_dSalt *= pHfactor;
					logKeq_dPh = pow(pH, static_cast<double>(p->kEqPhExp[i]) - 1.0) *
								 static_cast<double>(p->kEqPhExp[i]) * logKeq;
				}

				double bpp_i =
					pow(cpSalt, static_cast<double>(p->bppSaltPowerExp[i])) *
						static_cast<double>(p->bppSaltPowerFact[i]) +
					exp(cpSalt * static_cast<double>(p->bppSaltExpExp[i])) * static_cast<double>(p->bppSaltExpFact[i]);
				double bpp_i_dSalt =
					pow(cpSalt, static_cast<double>(p->bppSaltPowerExp[i]) - 1.0) *
						static_cast<double>(p->bppSaltPowerExp[i]) * static_cast<double>(p->bppSaltPowerFact[i]) +
					exp(cpSalt * static_cast<double>(p->bppSaltExpExp[i])) * static_cast<double>(p->bppSaltExpFact[i]) *
						static_cast<double>(p->bppSaltExpExp[i]);
				double bpp_i_dPh = bpp_i;
				if (phEnabled)
				{
					const double pH = yCp[1];
					const double phFactor = pow(pH, static_cast<double>(p->bppPhExp[i]));
					bpp_i *= phFactor;
					bpp_i_dSalt *= phFactor;
					bpp_i_dPh *=
						pow(pH, static_cast<double>(p->bppPhExp[i]) - 1.0) * static_cast<double>(p->bppPhExp[i]);
				}

				double S_i_sum = 0.0;
				double S_i_dPh = 0.0;
				double S_i_dSalt = 0.0;
				int bndIdx2 = 0;
				for (int j = _startIdx; j < _nComp; ++j)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[j] == 0)
						continue;

					double bpp_j = pow(cpSalt, static_cast<double>(p->bppSaltPowerExp[j])) *
									   static_cast<double>(p->bppSaltPowerFact[j]) +
								   exp(cpSalt * static_cast<double>(p->bppSaltExpExp[j])) *
									   static_cast<double>(p->bppSaltExpFact[j]);
					double bpp_j_dSalt =
						pow(cpSalt, static_cast<double>(p->bppSaltPowerExp[j]) - 1.0) *
							static_cast<double>(p->bppSaltPowerExp[j]) * static_cast<double>(p->bppSaltPowerFact[j]) +
						exp(cpSalt * static_cast<double>(p->bppSaltExpExp[j])) *
							static_cast<double>(p->bppSaltExpFact[j]) * static_cast<double>(p->bppSaltExpExp[j]);
					double bpp_j_dPh = bpp_j;
					if (phEnabled)
					{
						const double pH = yCp[1];
						const double phFactor = pow(pH, static_cast<double>(p->bppPhExp[j]));
						bpp_j *= phFactor;
						bpp_j_dSalt *= phFactor;
						bpp_j_dPh *=
							pow(pH, static_cast<double>(p->bppPhExp[j]) - 1.0) * static_cast<double>(p->bppPhExp[j]);
					}

					const double sqrtBpp_ij = sqrt(bpp_i * bpp_j);
					const double radSum = static_cast<double>(p->radius[i]) + static_cast<double>(p->radius[j]);
					const double S_temp = y[bndIdx2] * radSum * exp(-kappa * (R - radSum));
					const double S_summand = S_temp * sqrtBpp_ij;
					S_i_sum += S_summand;
					S_i_dSalt += S_temp * 0.5 / sqrtBpp_ij * (bpp_i * bpp_j_dSalt + bpp_i_dSalt * bpp_j);
					S_i_dSalt -= S_summand * R * kappa_dSalt;

					if (phEnabled)
						S_i_dPh += S_temp * 0.5 / sqrtBpp_ij * (bpp_i * bpp_j_dPh + bpp_i_dPh * bpp_j);

					++bndIdx2;
				}

				S_i_dSalt *= Sfactor;
				S_i_dSalt += Sfactor_dSalt * S_i_sum;
				S_i_dPh *= Sfactor;

				const double S_i = Sfactor * S_i_sum;
				const double commonFactor = static_cast<double>(p->kKin[i]) * y[bndIdx] * exp(S_i - logKeq);

				// dres_i / dc_{p,0} (salt)
				jac[0 - bndIdx - offsetCp] = commonFactor * (S_i_dSalt - logKeq_dSalt);

				// dres_i / dc_{p,1} (pH)
				if (phEnabled)
					jac[1 - bndIdx - offsetCp] = commonFactor * (S_i_dPh - logKeq_dPh);

				bndIdx2 = 0;
				for (int j = _startIdx; j < _nComp; ++j)
				{
					// Skip components without bound states (bound state index bndIdx is not advanced)
					if (_nBoundStates[j] == 0)
						continue;

					double bpp_j = pow(cpSalt, static_cast<double>(p->bppSaltPowerExp[j])) *
									   static_cast<double>(p->bppSaltPowerFact[j]) +
								   exp(cpSalt * static_cast<double>(p->bppSaltExpExp[j])) *
									   static_cast<double>(p->bppSaltExpFact[j]);
					if (phEnabled)
					{
						const double pH = yCp[1];
						const double phFactor = pow(pH, static_cast<double>(p->bppPhExp[j]));
						bpp_j *= phFactor;
					}

					const double sqrtBpp_ij = sqrt(bpp_i * bpp_j);
					const double radSum = static_cast<double>(p->radius[i]) + static_cast<double>(p->radius[j]);
					const double S_temp = radSum * exp(-kappa * (R - radSum));
					const double S_summand = S_temp * sqrtBpp_ij;
					const double S_temp_dq = y[bndIdx2] * S_summand * kappa * R_dq;

					// dres_i / dq_j
					jac[bndIdx2 - bndIdx] = commonFactor * (Sfactor_dq * S_i_sum + Sfactor * (S_summand - S_temp_dq));

					++bndIdx2;
				}

				// dres_i / dq_i
				jac[0] += static_cast<double>(p->kKin[i]) * exp(S_i - logKeq);

				// Advance to next flux and Jacobian row
				++bndIdx;
				++jac;
			}
		}
	}
};

typedef ColloidalBindingBase<ColloidalParamHandler> ColloidalBinding;
typedef ColloidalBindingBase<ExtColloidalParamHandler> ExternalColloidalBinding;

namespace binding
{
void registerColloidalModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
{
	bindings[ColloidalBinding::identifier()] = []() { return new ColloidalBinding(); };
	bindings[ExternalColloidalBinding::identifier()] = []() { return new ExternalColloidalBinding(); };
}
} // namespace binding

} // namespace model

} // namespace cadet
