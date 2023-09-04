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

#include "model/binding/SimplifiedMultiStateStericMassActionBinding.hpp"
#include "model/binding/RefConcentrationSupport.hpp"
#include "model/ModelUtils.hpp"
#include "cadet/Exceptions.hpp"
#include "nonlin/Solver.hpp"
#include "ParamReaderHelper.hpp"
#include "ConfigurationHelper.hpp"
#include "SimulationTypes.hpp"

#include "AdUtils.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

namespace
{
inline double getExtremeValueOfOrthoParametrization(double sMin, double sMax, double sQuad, unsigned int nStates)
{
	return (sMax - sMin) * 0.5 / (sQuad * static_cast<double>(nStates - 1)) + 0.5 * static_cast<double>(nStates - 1);
}
} // namespace

namespace cadet
{

namespace model
{

SimplifiedMultiStateStericMassActionBinding::SimplifiedMultiStateStericMassActionBinding()
{
}
SimplifiedMultiStateStericMassActionBinding::~SimplifiedMultiStateStericMassActionBinding() CADET_NOEXCEPT
{
}

bool SimplifiedMultiStateStericMassActionBinding::configureModelDiscretization(IParameterProvider& paramProvider,
																			   unsigned int nComp,
																			   unsigned int const* nBound,
																			   unsigned int const* boundOffset)
{
	const bool res = BindingModelBase::configureModelDiscretization(paramProvider, nComp, nBound, boundOffset);

	if (nBound[0] != 1)
		throw InvalidParameterException(
			"Simplified Multi-State SMA binding model requires exactly one bound state for salt");

	const unsigned int totalBoundStates = numBoundStates(nBound, nComp);

	// Allocate space for parameters
	_kA.reserve(totalBoundStates, nComp);
	_kD.reserve(totalBoundStates, nComp);
	_nuMin.reserve(nComp);
	_nuMax.reserve(nComp);
	_nuQuad.reserve(nComp);

	_sigmaMin.reserve(nComp);
	_sigmaMax.reserve(nComp);
	_sigmaQuad.reserve(nComp);

	_kSW.reserve(nComp);
	_kSW_lin.reserve(nComp);
	_kSW_quad.reserve(nComp);
	_kWS.reserve(nComp);
	_kWS_lin.reserve(nComp);
	_kWS_quad.reserve(nComp);

	// First flux is salt, which is always quasi-stationary
	_reactionQuasistationarity[0] = true;

	return res;
}

bool SimplifiedMultiStateStericMassActionBinding::configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx,
																ParticleTypeIdx parTypeIdx)
{
	const unsigned int totalBoundStates = numBoundStates(_nBoundStates, _nComp);

	// Read parameters
	_lambda = paramProvider.getDouble("SMSSMA_LAMBDA");
	readBoundStateDependentParameter<util::SlicedVector<active>, active>(_kA, paramProvider, "SMSSMA_KA", _nComp,
																		 _nBoundStates);
	readBoundStateDependentParameter<util::SlicedVector<active>, active>(_kD, paramProvider, "SMSSMA_KD", _nComp,
																		 _nBoundStates);
	readParameterMatrix(_nuMin, paramProvider, "SMSSMA_NU_MIN", _nComp, 1);
	readParameterMatrix(_nuMax, paramProvider, "SMSSMA_NU_MAX", _nComp, 1);
	readParameterMatrix(_nuQuad, paramProvider, "SMSSMA_NU_QUAD", _nComp, 1);
	readParameterMatrix(_sigmaMin, paramProvider, "SMSSMA_SIGMA_MIN", _nComp, 1);
	readParameterMatrix(_sigmaMax, paramProvider, "SMSSMA_SIGMA_MAX", _nComp, 1);
	readParameterMatrix(_sigmaQuad, paramProvider, "SMSSMA_SIGMA_QUAD", _nComp, 1);
	readParameterMatrix(_kSW, paramProvider, "SMSSMA_KSW", _nComp, 1);
	readParameterMatrix(_kSW_lin, paramProvider, "SMSSMA_KSW_LIN", _nComp, 1);
	readParameterMatrix(_kSW_quad, paramProvider, "SMSSMA_KSW_QUAD", _nComp, 1);
	readParameterMatrix(_kWS, paramProvider, "SMSSMA_KWS", _nComp, 1);
	readParameterMatrix(_kWS_lin, paramProvider, "SMSSMA_KWS_LIN", _nComp, 1);
	readParameterMatrix(_kWS_quad, paramProvider, "SMSSMA_KWS_QUAD", _nComp, 1);
	readReferenceConcentrations(paramProvider, "SMSSMA_", _refC0, _refQ);

	if (paramProvider.exists("SMSSMA_NU_SALT"))
		_nu0 = paramProvider.getDouble("SMSSMA_NU_SALT");
	else
		_nu0 = 1.0;

	// Check parameters
	if ((_kA.size() != _kD.size()) || (_kA.size() != totalBoundStates))
		throw InvalidParameterException("SMSSMA_KA and SMSSMA_KD have to have the same size");
	if ((_nuMin.size() != _nuMax.size()) || (_nuMin.size() != _nuQuad.size()) ||
		(static_cast<int>(_nuMin.size()) != _nComp))
		throw InvalidParameterException("SMSSMA_NU_MIN, SMSSMA_NU_MAX and SMSSMA_NU_QUAD require " +
										std::to_string(_nComp) + " elements");
	if ((_sigmaMin.size() != _sigmaMax.size()) || (_sigmaMin.size() != _sigmaQuad.size()) ||
		(static_cast<int>(_sigmaMin.size()) != _nComp))
		throw InvalidParameterException("SMSSMA_SIGMA_MIN, SMSSMA_SIGMA_MAX and SMSSMA_SIGMA_QUAD require " +
										std::to_string(_nComp) + " elements");
	if ((_kSW.size() != _kSW_lin.size()) || (_kSW.size() != _kSW_quad.size()) ||
		(static_cast<int>(_kSW.size()) != _nComp))
		throw InvalidParameterException("SMSSMA_KSW, SMSSMA_KSW_LIN and SMSSMA_KSW_QUAD require " +
										std::to_string(_nComp) + " elements");
	if ((_kWS.size() != _kWS_lin.size()) || (_kWS.size() != _kWS_quad.size()) ||
		(static_cast<int>(_kWS.size()) != _nComp))
		throw InvalidParameterException("SMSSMA_KWS, SMSSMA_KWS_LIN and SMSSMA_KWS_QUAD require " +
										std::to_string(_nComp) + " elements");

	// Check for negative values
	for (int i = 0; i < _nComp; ++i)
	{
		if (_nBoundStates[i] == 0)
			continue;

		// ---- Check sigma
		// Endpoints have to be non-negative
		if ((_sigmaMin[i] < 0.0) || (_sigmaMax[i] < 0.0))
			throw InvalidParameterException("SMSSMA_SIGMA_MIN, SMSSMA_SIGMA_MAX and SMSSMA_SIGMA_QUAD of component " +
											std::to_string(i) +
											" have to be set such that all computed sigma are non-negative");

		if (_sigmaQuad[i] != 0.0)
		{
			// Position of global extremum
			const double extStateSigma = getExtremeValueOfOrthoParametrization(
				static_cast<double>(_sigmaMin[i]), static_cast<double>(_sigmaMax[i]),
				static_cast<double>(_sigmaQuad[i]), _nBoundStates[i]);
			if ((extStateSigma >= 0.0) && (extStateSigma <= static_cast<double>(_nBoundStates[i] - 1)))
			{
				// Extremum is inside interval, so we have to check all three values
				const double extVal = sigma<double>(i, extStateSigma);
				if (extVal < 0.0)
					throw InvalidParameterException(
						"SMSSMA_SIGMA_MIN, SMSSMA_SIGMA_MAX and SMSSMA_SIGMA_QUAD of component " + std::to_string(i) +
						" have to be set such that all computed sigma are non-negative");
			}
		}

		// ---- Check nu
		// Endpoints have to be non-negative
		if ((_nuMin[i] < 0.0) || (_nuMax[i] < 0.0))
			throw InvalidParameterException("SMSSMA_NU_MIN, SMSSMA_NU_MAX and SMSSMA_NU_QUAD of component " +
											std::to_string(i) +
											" have to be set such that all computed nu are non-negative");

		if (_nuQuad[i] != 0.0)
		{
			// Position of global extremum
			const double extStateNu =
				getExtremeValueOfOrthoParametrization(static_cast<double>(_nuMin[i]), static_cast<double>(_nuMax[i]),
													  static_cast<double>(_nuQuad[i]), _nBoundStates[i]);
			if ((extStateNu >= 0.0) && (extStateNu <= static_cast<double>(_nBoundStates[i] - 1)))
			{
				// Extremum is inside interval, so we have to check all three values
				const double extVal = nu<double>(i, extStateNu);
				if (extVal < 0.0)
					throw InvalidParameterException("SMSSMA_NU_MIN, SMSSMA_NU_MAX and SMSSMA_NU_QUAD of component " +
													std::to_string(i) +
													" have to be set such that all computed nu are non-negative");
			}
		}

		// ---- Check kSW
		// Endpoints have to be non-negative
		if ((_kSW[i] < 0.0) || (k_sw<double>(i, _nBoundStates[i] - 2) < 0.0))
			throw InvalidParameterException("SMSSMA_KSW, SMSSMA_KSW_LIN and SMSSMA_KSW_QUAD of component " +
											std::to_string(i) +
											" have to be set such that all computed rates are non-negative");

		if (_kSW_quad[i] != 0.0)
		{
			// Position of global extremum
			const double extStateSW = static_cast<double>(_kSW_lin[i]) * 0.5 / static_cast<double>(_kSW_quad[i]) +
									  static_cast<double>(_nBoundStates[i]) * 0.5 - 1.0;
			if ((extStateSW >= 0.0) && (extStateSW <= static_cast<double>(_nBoundStates[i] - 2)))
			{
				// Extremum is inside interval, so we have to check all three values
				const double extVal = k_sw<double>(i, extStateSW);
				if (extVal < 0.0)
					throw InvalidParameterException("SMSSMA_KSW, SMSSMA_KSW_LIN and SMSSMA_KSW_QUAD of component " +
													std::to_string(i) +
													" have to be set such that all computed rates are non-negative");
			}
		}

		// ---- Check kWS
		// Endpoints have to be non-negative
		if ((_kWS[i] < 0.0) || (k_ws<double>(i, _nBoundStates[i] - 2) < 0.0))
			throw InvalidParameterException("SMSSMA_KWS, SMSSMA_KWS_LIN and SMSSMA_KWS_QUAD of component " +
											std::to_string(i) +
											" have to be set such that all computed rates are non-negative");

		if (_kWS_quad[i] != 0.0)
		{
			// Position of global extremum
			const double extStateWS = static_cast<double>(_kWS_lin[i]) * 0.5 / static_cast<double>(_kWS_quad[i]) +
									  static_cast<double>(_nBoundStates[i]) * 0.5 - 1.0;
			if ((extStateWS >= 0.0) && (extStateWS <= static_cast<double>(_nBoundStates[i] - 2)))
			{
				// Extremum is inside interval, so we have to check all three values
				const double extVal = k_ws<double>(i, extStateWS);
				if (extVal < 0.0)
					throw InvalidParameterException("SMSSMA_KWS, SMSSMA_KWS_LIN and SMSSMA_KWS_QUAD of component " +
													std::to_string(i) +
													" have to be set such that all computed rates are non-negative");
			}
		}
	}

	// Register parameters
	_parameters[makeParamId(hashString("SMSSMA_LAMBDA"), unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep,
							ReactionIndep, SectionIndep)] = &_lambda;
	registerComponentBoundStateDependentParamCompMajor(hashString("SMSSMA_KA"), _parameters, _kA, unitOpIdx,
													   parTypeIdx);
	registerComponentBoundStateDependentParamCompMajor(hashString("SMSSMA_KD"), _parameters, _kD, unitOpIdx,
													   parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_NU_MIN"), _parameters, _nuMin, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_NU_MAX"), _parameters, _nuMax, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_NU_QUAD"), _parameters, _nuQuad, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_SIGMA_MIN"), _parameters, _sigmaMin, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_SIGMA_MAX"), _parameters, _sigmaMax, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_SIGMA_QUAD"), _parameters, _sigmaQuad, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_KSW"), _parameters, _kSW, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_KSW_LIN"), _parameters, _kSW_lin, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_KSW_QUAD"), _parameters, _kSW_quad, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_KWS"), _parameters, _kWS, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_KWS_LIN"), _parameters, _kWS_lin, unitOpIdx, parTypeIdx);
	registerComponentDependentParam(hashString("SMSSMA_KWS_QUAD"), _parameters, _kWS_quad, unitOpIdx, parTypeIdx);
	_parameters[makeParamId(hashString("SMSSMA_NU_SALT"), unitOpIdx, 0, parTypeIdx, BoundStateIndep, ReactionIndep,
							SectionIndep)] = &_nu0;
	_parameters[makeParamId(hashString("SMSSMA_REFC0"), unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep,
							ReactionIndep, SectionIndep)] = &_refC0;
	_parameters[makeParamId(hashString("SMSSMA_REFQ"), unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep, ReactionIndep,
							SectionIndep)] = &_refQ;

	return true;
}

unsigned int SimplifiedMultiStateStericMassActionBinding::workspaceSize(
	unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nBoundStates) const CADET_NOEXCEPT
{
	return 0;
}

template <typename ParamType>
inline ParamType SimplifiedMultiStateStericMassActionBinding::sigma(int comp, double state) const
{
	if (_nBoundStates[comp] > 1)
		return static_cast<ParamType>(_sigmaMin[comp]) +
			   state * (static_cast<ParamType>(_sigmaMax[comp]) - static_cast<ParamType>(_sigmaMin[comp])) /
				   (_nBoundStates[comp] - 1) -
			   static_cast<ParamType>(_sigmaQuad[comp]) * state * (1 - static_cast<int>(_nBoundStates[comp]) + state);
	else
		return static_cast<ParamType>(_sigmaMin[comp]);
}

template <typename ParamType>
inline ParamType SimplifiedMultiStateStericMassActionBinding::nu(int comp, double state) const
{
	if (_nBoundStates[comp] > 1)
		return static_cast<ParamType>(_nuMin[comp]) +
			   state * (static_cast<ParamType>(_nuMax[comp]) - static_cast<ParamType>(_nuMin[comp])) /
				   (_nBoundStates[comp] - 1) -
			   static_cast<ParamType>(_nuQuad[comp]) * state * (1 - static_cast<int>(_nBoundStates[comp]) + state);
	else
		return static_cast<ParamType>(_nuMin[comp]);
}

template <typename ParamType>
inline ParamType SimplifiedMultiStateStericMassActionBinding::k_sw(int comp, double state) const
{
	return static_cast<ParamType>(_kSW[comp]) + state * static_cast<ParamType>(_kSW_lin[comp]) -
		   static_cast<ParamType>(_kSW_quad[comp]) * state * (2 - static_cast<int>(_nBoundStates[comp]) + state);
}

template <typename ParamType>
inline ParamType SimplifiedMultiStateStericMassActionBinding::k_ws(int comp, double state) const
{
	return static_cast<ParamType>(_kWS[comp]) + state * static_cast<ParamType>(_kWS_lin[comp]) -
		   static_cast<ParamType>(_kWS_quad[comp]) * state * (2 - static_cast<int>(_nBoundStates[comp]) + state);
}

template <typename StateType, typename CpStateType, typename ResidualType, typename ParamType>
int SimplifiedMultiStateStericMassActionBinding::fluxImpl(double t, unsigned int secIdx, const ColumnPosition& colPos,
														  StateType const* y, CpStateType const* yCp, ResidualType* res,
														  LinearBufferAllocator workSpace) const
{
	using CpStateParamType = typename DoubleActivePromoter<CpStateType, ParamType>::type;
	using StateParamType = typename DoubleActivePromoter<StateType, ParamType>::type;

	// Salt equation: q_0 - Lambda + Sum[Sum[nu_i^j * q_i^j, j], i] == 0
	//           <=>  q_0 == Lambda - Sum[Sum[nu_i^j * q_i^j, j], i]
	// Also compute \bar{q}_0 = q_0 - Sum[Sum[sigma_i^j * q_i^j, j], i]
	res[0] = static_cast<ParamType>(_nu0) * y[0] - static_cast<ParamType>(_lambda);
	StateParamType q0_bar = static_cast<ParamType>(_nu0) * y[0];

	unsigned int bndIdx = 1;

	// Loop over all components i
	for (int i = 1; i < _nComp; ++i)
	{
		// Loop over bound states j of component i
		for (unsigned int j = 0; j < _nBoundStates[i]; ++j)
		{
			res[0] += nu<ParamType>(i, j) * y[bndIdx];
			q0_bar -= sigma<ParamType>(i, j) * y[bndIdx];

			// Next bound component
			++bndIdx;
		}
	}

	const StateParamType q0_bar_divRef = q0_bar / static_cast<ParamType>(_refQ);
	const CpStateParamType yCp0_divRef = yCp[0] / static_cast<ParamType>(_refC0);

	// Protein equations

	// Loop over all components i
	bndIdx = 1;
	for (int i = 1; i < _nComp; ++i)
	{
		active const* const curKa = _kA[i];
		active const* const curKd = _kD[i];

		// Loop over bound states j of component i
		for (unsigned int j = 0; j < _nBoundStates[i]; ++j)
		{
			const ParamType curNu = nu<ParamType>(i, j) / static_cast<ParamType>(_nu0);
			const CpStateParamType c0_pow_nu = pow(yCp0_divRef, curNu);
			const StateParamType q0_bar_pow_nu = pow(q0_bar_divRef, curNu);

			// Calculate residual
			// Adsorption and desorption
			res[bndIdx] = static_cast<ParamType>(curKd[j]) * y[bndIdx] * c0_pow_nu -
						  static_cast<ParamType>(curKa[j]) * yCp[i] * q0_bar_pow_nu;

			// Conversion to and from weaker state j - 1
			if (j > 0)
			{
				const ParamType curSW = k_sw<ParamType>(i, j - 1);
				const ParamType curWS = k_ws<ParamType>(i, j - 1);

				const ParamType nuDiff = curNu - nu<ParamType>(i, j - 1) / static_cast<ParamType>(_nu0);
				// Conversion to weaker state
				res[bndIdx] += curSW * y[bndIdx] * pow(yCp0_divRef, nuDiff);
				// Conversion from weaker state
				res[bndIdx] -= curWS * y[bndIdx - 1] * pow(q0_bar_divRef, nuDiff);
			}

			// Conversion to and from stronger state j + 1
			if (j < _nBoundStates[i] - 1)
			{
				const ParamType curSW = k_sw<ParamType>(i, j);
				const ParamType curWS = k_ws<ParamType>(i, j);

				const ParamType nuDiff = nu<ParamType>(i, j + 1) / static_cast<ParamType>(_nu0) - curNu;
				// Conversion to stronger state
				res[bndIdx] += curWS * y[bndIdx] * pow(q0_bar_divRef, nuDiff);
				// Conversion from stronger state
				res[bndIdx] -= curSW * y[bndIdx + 1] * pow(yCp0_divRef, nuDiff);
			}

			// Next bound component
			++bndIdx;
		}
	}
	return 0;
}

template <typename RowIterator>
void SimplifiedMultiStateStericMassActionBinding::jacobianImpl(double t, unsigned int secIdx,
															   const ColumnPosition& colPos, double const* y,
															   double const* yCp, int offsetCp, RowIterator jac,
															   LinearBufferAllocator workSpace) const
{
	double q0_bar = static_cast<double>(_nu0) * y[0];

	// Salt equation: q_0 - Lambda + Sum[Sum[nu_j * q_i^j, j], i] == 0
	jac[0] = static_cast<double>(_nu0);
	int bndIdx = 1;
	for (int i = 1; i < _nComp; ++i)
	{
		for (unsigned int j = 0; j < _nBoundStates[i]; ++j)
		{
			jac[bndIdx] = nu<double>(i, j);

			// Calculate \bar{q}_0 = q_0 - Sum[Sum[sigma_j * q_i^j, j], i]
			q0_bar -= sigma<double>(i, j) * y[bndIdx];

			// Next bound component
			++bndIdx;
		}
	}

	// Advance to protein equations
	++jac;

	const double refC0 = static_cast<double>(_refC0);
	const double refQ = static_cast<double>(_refQ);
	const double yCp0_divRef = yCp[0] / refC0;
	const double q0_bar_divRef = q0_bar / refQ;

	// Protein equations
	// We have already computed \bar{q}_0 in the loop above
	bndIdx = 1;
	for (int i = 1; i < _nComp; ++i)
	{
		active const* const curKa = _kA[i];
		active const* const curKd = _kD[i];

		for (int j = 0; j < static_cast<int>(_nBoundStates[i]); ++j)
		{
			// Getting to c_{p,0}: -bndIdx takes us to q_0, another -offsetCp to c_{p,0}. This means jac[-bndIdx -
			// offsetCp] corresponds to c_{p,0}. Getting to c_{p,i}: -bndIdx takes us to q_0, another -offsetCp to
			// c_{p,0} and a +i to c_{p,i}.
			//                     This means jac[i - bndIdx - offsetCp] corresponds to c_{p,i}.

			const double ka = static_cast<double>(curKa[j]);
			const double kd = static_cast<double>(curKd[j]);
			const double curNu = nu<double>(i, j) / static_cast<double>(_nu0);

			const double c0_pow_nu = pow(yCp0_divRef, curNu);
			const double q0_bar_pow_nu = pow(q0_bar_divRef, curNu);
			const double c0_pow_nu_m1_divRef = pow(yCp0_divRef, curNu - 1.0) / refC0;
			const double q0_bar_pow_nu_m1_divRef = curNu * pow(q0_bar_divRef, curNu - 1.0) / refQ;

			// dres_i / dc_{p,0}
			jac[-bndIdx - offsetCp] = kd * y[bndIdx] * curNu * c0_pow_nu_m1_divRef;
			// dres_i / dc_{p,i}
			jac[i - bndIdx - offsetCp] = -ka * q0_bar_pow_nu;
			// dres_i / dq_0
			jac[-bndIdx] = -ka * yCp[i] * q0_bar_pow_nu_m1_divRef * static_cast<double>(_nu0);

			// Fill dres_i / dq_i^j (no flux terms, just handle \bar{q}_0^{nu_i} term)
			int bndIdx2 = 1;
			for (int i2 = 1; i2 < _nComp; ++i2)
			{
				for (unsigned int j2 = 0; j2 < _nBoundStates[i2]; ++j2)
				{
					// dres_i / dq_{i2}^{j2}
					jac[bndIdx2 - bndIdx] = ka * yCp[i] * q0_bar_pow_nu_m1_divRef * sigma<double>(i2, j2);
					// Getting to q_{i2}^{j2}: -bndIdx takes us to q_1^0, another +bndIdx2 to q_{i2}^{j2}.
					// This means jac[bndIdx2 - bndIdx] corresponds to q_{i2}^{j2}.

					++bndIdx2;
				}
			}

			// Add to dres_i / dq_i
			jac[0] += kd * c0_pow_nu;

			// Handle flux terms

			// Conversion to and from weaker states
			if (j > 0)
			{
				const double curSW = k_sw<double>(i, j - 1);
				const double curWS = k_ws<double>(i, j - 1);

				const double nuDiff = curNu - nu<double>(i, j - 1) / static_cast<double>(_nu0);
				const double q0_bar_pow_nudiff_deriv =
					curWS * y[bndIdx - 1] * nuDiff * pow(q0_bar_divRef, nuDiff - 1.0) / refQ;

				// dres_i / dc_{p,0}
				jac[-bndIdx - offsetCp] += curSW * y[bndIdx] * nuDiff * pow(yCp0_divRef, nuDiff - 1.0) / refC0;
				// dres_i / dq_0
				jac[-bndIdx] -= q0_bar_pow_nudiff_deriv * static_cast<double>(_nu0);
				// dres_i / dq_i^j (current outer loop element)
				jac[0] += curSW * pow(yCp0_divRef, nuDiff);
				// dres_i / dq_i^l (without dq0_bar / dq_i^l)
				jac[-1] -= curWS * pow(q0_bar_divRef, nuDiff);
				// dres_i / dq_{i2}^{j2} (accounts for all dq0_bar / dq_{i2}^{j2} including missing dq0_bar / dq_i^l
				// from above)
				bndIdx2 = 1;
				for (int i2 = 1; i2 < _nComp; ++i2)
				{
					for (int j2 = 0; j2 < static_cast<int>(_nBoundStates[i2]); ++j2, ++bndIdx2)
						jac[bndIdx2 - bndIdx] += q0_bar_pow_nudiff_deriv * sigma<double>(i2, j2);
				}
			}

			// Conversion to and from stronger states
			if (j < static_cast<int>(_nBoundStates[i]) - 1)
			{
				const double curSW = k_sw<double>(i, j);
				const double curWS = k_ws<double>(i, j);

				const double nuDiff = nu<double>(i, j + 1) / static_cast<double>(_nu0) - curNu;
				const double q0_bar_pow_nudiff_deriv =
					curWS * y[bndIdx] * nuDiff * pow(q0_bar_divRef, nuDiff - 1.0) / refQ;

				// dres_i / dc_{p,0}
				jac[-bndIdx - offsetCp] -= curSW * y[bndIdx + 1] * nuDiff * pow(yCp0_divRef, nuDiff - 1.0) / refC0;
				// dres_i / dq_0
				jac[-bndIdx] += q0_bar_pow_nudiff_deriv * static_cast<double>(_nu0);
				// dres_i / dq_i^j (current outer loop element)
				jac[0] += curWS * pow(q0_bar_divRef, nuDiff);
				// dres_i / dq_i^l
				jac[1] -= curSW * pow(yCp0_divRef, nuDiff);
				// dres_i / dq_{i2}^{j2} (accounts for all dq0_bar / dq_{i2}^{j2})
				bndIdx2 = 1;
				for (int i2 = 1; i2 < _nComp; ++i2)
				{
					for (int j2 = 0; j2 < static_cast<int>(_nBoundStates[i2]); ++j2, ++bndIdx2)
						jac[bndIdx2 - bndIdx] -= q0_bar_pow_nudiff_deriv * sigma<double>(i2, j2);
				}
			}

			// Advance to next equation and Jacobian row
			++bndIdx;
			++jac;
		}
	}
}

bool SimplifiedMultiStateStericMassActionBinding::preConsistentInitialState(double t, unsigned int secIdx,
																			const ColumnPosition& colPos, double* y,
																			double const* yCp,
																			LinearBufferAllocator workSpace) const
{
	// Compute salt component from given bound states q_i^j

	// Salt equation: nu_0 * q_0 - Lambda + Sum[Sum[nu_i^j * q_i^j, j], i] == 0
	//           <=>  q_0 == (Lambda - Sum[Sum[nu_i^j * q_i^j, j], i]) / nu_0
	y[0] = static_cast<double>(_lambda);

	// Loop over all components i
	unsigned int bndIdx = 1;
	for (int i = 1; i < _nComp; ++i)
	{
		// Loop over all bound states j
		for (unsigned int j = 0; j < _nBoundStates[i]; ++j)
		{
			y[0] -= nu<double>(i, j) * y[bndIdx];

			// Next bound component
			++bndIdx;
		}
	}

	y[0] /= static_cast<double>(_nu0);

	return true;
}

void SimplifiedMultiStateStericMassActionBinding::postConsistentInitialState(double t, unsigned int secIdx,
																			 const ColumnPosition& colPos, double* y,
																			 double const* yCp,
																			 LinearBufferAllocator workSpace) const
{
	preConsistentInitialState(t, secIdx, colPos, y, yCp, workSpace);
}

} // namespace model

} // namespace cadet
