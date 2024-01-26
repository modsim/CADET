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

/**
 * @file
 * Defines helper functions used by all ModelSystemImpl-xyz.cpp files.
 */

namespace
{
/**
 * @brief Computes a total return code from a list of separate return codes
 * @details A negative return code indicates a non-recoverable error. Positive
 *          values indicate recoverable errors and a value of @c 0 indicates
 *          no error.
 * @param [in] err List of error codes to be fused into one
 * @return Total error code summarizing all codes in the list
 */
inline int totalErrorIndicatorFromLocal(const std::vector<int>& err)
{
	int totalError = 0;
	for (std::size_t i = 0; i < err.size(); ++i)
	{
		// Negative values are non-recoverable errors
		if (err[i] < 0)
			return err[i];

		// 0 = okay, positive values = recoverable error
		totalError = std::max(totalError, err[i]);
	}
	return totalError;
}

/**
 * @brief Fuses two error codes into one
 * @details A negative return code indicates a non-recoverable error. Positive
 *          values indicate recoverable errors and a value of @c 0 indicates
 *          no error.
 * @param [in] curCode Current error code
 * @param [in] nextCode Next error code that is fused into the current one
 * @return Fused error code summarizing both inputs
 */
inline int updateErrorIndicator(int curCode, int nextCode)
{
	if ((curCode < 0) || (nextCode < 0))
		return std::min(curCode, nextCode);
	return std::max(curCode, nextCode);
}

template <typename state_t> inline state_t applyOffset(const state_t& state, unsigned int offset)
{
	return state_t{state.vecStateY + offset, (state.vecStateYdot) ? (state.vecStateYdot + offset) : nullptr};
}

template <> inline cadet::AdJacobianParams applyOffset(const cadet::AdJacobianParams& adJac, unsigned int offset)
{
	return cadet::AdJacobianParams{(adJac.adRes) ? (adJac.adRes + offset) : nullptr,
								   (adJac.adY) ? (adJac.adY + offset) : nullptr, adJac.adDirOffset};
}

template <typename res_t>
inline res_t cubicPoly(const cadet::active& constCoeff, const cadet::active& linCoeff, const cadet::active& quadCoeff,
					   const cadet::active& cubCoeff, double v)
{
	return static_cast<res_t>(constCoeff) +
		   v * (static_cast<res_t>(linCoeff) + v * (static_cast<res_t>(quadCoeff) + v * static_cast<res_t>(cubCoeff)));
}

template <typename res_t>
inline res_t cubicPoly(cadet::active const* constCoeff, cadet::active const* linCoeff, cadet::active const* quadCoeff,
					   cadet::active const* cubCoeff, unsigned int idx, double v)
{
	return cubicPoly<res_t>(constCoeff[idx], linCoeff[idx], quadCoeff[idx], cubCoeff[idx], v);
}

template <typename res_t>
inline res_t cubicPolyDeriv(const cadet::active& linCoeff, const cadet::active& quadCoeff,
							const cadet::active& cubCoeff, double v)
{
	return static_cast<res_t>(linCoeff) +
		   v * (2.0 * static_cast<res_t>(quadCoeff) + 3.0 * v * static_cast<res_t>(cubCoeff));
}

template <typename res_t>
inline res_t cubicPolyDeriv(cadet::active const* linCoeff, cadet::active const* quadCoeff,
							cadet::active const* cubCoeff, unsigned int idx, double v)
{
	return cubicPolyDeriv<res_t>(linCoeff[idx], quadCoeff[idx], cubCoeff[idx], v);
}

inline double cubicPoly(const cadet::active& constCoeff, const cadet::active& linCoeff, const cadet::active& quadCoeff,
						const cadet::active& cubCoeff, double v, unsigned int adDir)
{
	return constCoeff.getADValue(adDir) +
		   v * (linCoeff.getADValue(adDir) + v * (quadCoeff.getADValue(adDir) + v * cubCoeff.getADValue(adDir)));
}

inline double cubicPoly(cadet::active const* constCoeff, cadet::active const* linCoeff, cadet::active const* quadCoeff,
						cadet::active const* cubCoeff, unsigned int idx, double v, unsigned int adDir)
{
	return cubicPoly(constCoeff[idx], linCoeff[idx], quadCoeff[idx], cubCoeff[idx], v, adDir);
}

inline double cubicPolyDeriv(const cadet::active& linCoeff, const cadet::active& quadCoeff,
							 const cadet::active& cubCoeff, double v, unsigned int adDir)
{
	return linCoeff.getADValue(adDir) + v * (2.0 * quadCoeff.getADValue(adDir) + 3.0 * v * cubCoeff.getADValue(adDir));
}

inline double cubicPolyDeriv(cadet::active const* linCoeff, cadet::active const* quadCoeff,
							 cadet::active const* cubCoeff, unsigned int idx, double v, unsigned int adDir)
{
	return cubicPolyDeriv(linCoeff[idx], quadCoeff[idx], cubCoeff[idx], v, adDir);
}
} // namespace
