// =============================================================================
//  CADET
//  
//  Copyright © 2008-2021: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#ifndef LIBCADET_SPLINE_HPP_
#define LIBCADET_SPLINE_HPP_

#include <algorithm>
#include <tuple>

#include "cadet/cadetCompilerInfo.hpp"
#include "AutoDiff.hpp"

namespace cadet 
{

	/**
	 * @brief      Evaluates a piecewise cubic polynomial
	 * @details    If the evaluation point @p x is outside the domain of the piecewise polynomial,
	 *             constant extrapolation from the first and last value of the polynomial on its
	 *             domain is applied.
	 * @param[in]  x           Evaluation position
	 * @param[in]  breaks      Array with @c nPieces+1 elements (strictly increasing) that defines the domain pieces
	 * @param[in]  constCoeff  Array with constant coefficients of size @p nPieces
	 * @param[in]  linCoeff    Array with linear coefficients of size @p nPieces
	 * @param[in]  quadCoeff   Array with quadratic coefficients of size @p nPieces
	 * @param[in]  cubeCoeff   Array with cubic coefficients of size @p nPieces
	 * @param[in]  nPieces     Number of pieces (at least 1)
	 * @tparam     value_t     Type of the evaluation point
	 * @tparam     result_t    Type of the result
	 * @return     Value of the piecewise cubic polynomial at the given point @p x
	 */
	template <typename value_t, typename result_t>
	result_t evaluateCubicPiecewisePolynomial(const value_t& x, active const* breaks, active const* constCoeff,
		active const* linCoeff, active const* quadCoeff, active const* cubeCoeff, int nPieces) CADET_NOEXCEPT
	{
		// Test if outside of domain, apply constant extrapolation
		if (x < breaks[0])
		{
			return constCoeff[0];
		}

		if (x >= breaks[nPieces])
		{
			// Return the value at the right of the domain
			const result_t y = static_cast<result_t>(breaks[nPieces]) - static_cast<result_t>(breaks[nPieces - 1]);
			const int idx = nPieces - 1;
			return static_cast<result_t>(constCoeff[idx]) + y * (
				static_cast<result_t>(linCoeff[idx]) + y * (
					static_cast<result_t>(quadCoeff[idx]) + y * static_cast<result_t>(cubeCoeff[idx]) 
					)
				);
		}

		// Find correct piece
		int idx = 0;
		for (; idx < nPieces; ++idx)
		{
			if (breaks[idx] >= x)
				break;
		}
		
		// Evaluate polynomial
		const typename DoubleActivePromoter<value_t, result_t>::type y = x - static_cast<result_t>(breaks[idx]);
		return static_cast<result_t>(constCoeff[idx]) + y * (
			static_cast<result_t>(linCoeff[idx]) + y * (
				static_cast<result_t>(quadCoeff[idx]) + y * static_cast<result_t>(cubeCoeff[idx]) 
				)
			);
	}

	/**
	 * @brief      Evaluates a piecewise cubic polynomial and returns base and dependent values
	 * @details    If the evaluation point @p x is outside the domain of the piecewise polynomial,
	 *             constant extrapolation from the first and last value of the polynomial on its
	 *             domain is applied.
	 *             
	 *             The function returns the constant base value and the variable part depending on @p x.
	 * @param[in]  x           Evaluation position
	 * @param[in]  breaks      Array with @c nPieces+1 elements (strictly increasing) that defines the domain pieces
	 * @param[in]  constCoeff  Array with constant coefficients of size @p nPieces
	 * @param[in]  linCoeff    Array with linear coefficients of size @p nPieces
	 * @param[in]  quadCoeff   Array with quadratic coefficients of size @p nPieces
	 * @param[in]  cubeCoeff   Array with cubic coefficients of size @p nPieces
	 * @param[in]  nPieces     Number of pieces (at least 1)
	 * @tparam     value_t     Type of the evaluation point
	 * @tparam     result_t    Type of the result
	 * @return     Constant and dynamic value of the piecewise cubic polynomial at the given point @p x
	 */
	template <typename value_t, typename result_t>
	std::tuple<result_t, result_t> evaluateCubicPiecewisePolynomialSplit(const value_t& x, active const* breaks, active const* constCoeff,
		active const* linCoeff, active const* quadCoeff, active const* cubeCoeff, int nPieces) CADET_NOEXCEPT
	{
		// Test if outside of domain, apply constant extrapolation
		if (x < breaks[0])
		{
			return {static_cast<result_t>(constCoeff[0]), result_t(0.0)};
		}

		if (x >= breaks[nPieces])
		{
			// Return the value at the right of the domain
			const result_t y = static_cast<result_t>(breaks[nPieces]) - static_cast<result_t>(breaks[nPieces - 1]);
			const int idx = nPieces - 1;
			return {static_cast<result_t>(constCoeff[idx]) + y * (
				static_cast<result_t>(linCoeff[idx]) + y * (
					static_cast<result_t>(quadCoeff[idx]) + y * static_cast<result_t>(cubeCoeff[idx]) 
					)
				), result_t(0.0)};
		}

		// Find correct piece
		int idx = 0;
		for (; idx < nPieces; ++idx)
		{
			if (breaks[idx] >= x)
				break;
		}
		
		// Evaluate polynomial
		const typename DoubleActivePromoter<value_t, result_t>::type y = x - static_cast<result_t>(breaks[idx]);
		return {static_cast<result_t>(constCoeff[idx]), y * (
			static_cast<result_t>(linCoeff[idx]) + y * (
				static_cast<result_t>(quadCoeff[idx]) + y * static_cast<result_t>(cubeCoeff[idx]) 
				)
			)};
	}

	/**
	 * @brief      Evaluates the derivative of a piecewise cubic polynomial
	 * @details    If the evaluation point @p x is outside the domain of the piecewise polynomial,
	 *             constant extrapolation from the first and last value of the polynomial on its
	 *             domain is applied.
	 * @param[in]  x           Evaluation position
	 * @param[in]  breaks      Array with @c nPieces+1 elements (strictly increasing) that defines the domain pieces
	 * @param[in]  linCoeff    Array with linear coefficients of size @p nPieces
	 * @param[in]  quadCoeff   Array with quadratic coefficients of size @p nPieces
	 * @param[in]  cubeCoeff   Array with cubic coefficients of size @p nPieces
	 * @param[in]  nPieces     Number of pieces (at least 1)
	 * @return     Derivative of the piecewise cubic polynomial at the given point @p x
	 */
	inline double evaluateCubicPiecewisePolynomialDerivative(double x, active const* breaks, active const* linCoeff,
		active const* quadCoeff, active const* cubeCoeff, int nPieces) CADET_NOEXCEPT
	{
		// Test if outside of domain, apply constant extrapolation
		if (x < breaks[0])
			return 0.0;

		if (x >= breaks[nPieces])
			return 0.0;

		// Find correct piece
		int idx = 0;
		for (; idx < nPieces; ++idx)
		{
			if (breaks[idx] >= x)
				break;
		}
		
		// Evaluate polynomial
		const double y = x - static_cast<double>(breaks[idx]);
		return static_cast<double>(linCoeff[idx]) + y * (2.0 * static_cast<double>(quadCoeff[idx]) + 3.0 * y * static_cast<double>(cubeCoeff[idx]));
	}

} // namespace cadet

#endif  // LIBCADET_SPLINE_HPP_
