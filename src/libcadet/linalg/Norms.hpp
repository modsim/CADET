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

#ifndef LIBCADET_NORMS_HPP_
#define LIBCADET_NORMS_HPP_

#include <cmath>
#include <limits>
#include <algorithm>
#include "MathUtil.hpp"
#include "common/CompilerSpecific.hpp"

namespace cadet
{

namespace linalg
{
/**
 * @brief Computes the (discrete) @f$\ell^1@f$-norm of the given vector
 * @details The (discrete) @f$\ell^1@f$-norm is given by @f$ \lVert v \rVert_{\ell^1} = \sum_{i=1}^N \abs{v_i} @f$.
 * @param [in] x Pointer to vector whose norm is to be evaluated
 * @param [in] size Number of elements in the vector
 * @return The @f$\ell^1@f$-norm of the vector
 */
inline double l1Norm(double const* const x, int size)
{
	double res = 0.0;
	for (int i = 0; i < size; ++i)
		res += std::abs(x[i]);
	return res;
}

/**
 * @brief Computes the squared (discrete) @f$\ell^2@f$-norm of the given vector
 * @details The squared (discrete) @f$\ell^2@f$-norm is given by @f$ \lVert v \rVert_{\ell^2}^2 = \sum_{i=1}^N v_i^2
 * @f$.
 * @param [in] x Pointer to vector whose norm is to be evaluated
 * @param [in] size Number of elements in the vector
 * @return The squared @f$\ell^2@f$-norm of the vector
 */
inline double l2NormSquared(double const* const x, int size)
{
	double res = 0.0;
	for (int i = 0; i < size; ++i)
		res += sqr(x[i]);
	return res;
}

/**
 * @brief Computes the (discrete) @f$\ell^2@f$-norm of the given vector
 * @details The (discrete) @f$\ell^2@f$-norm is given by @f$ \lVert v \rVert_{\ell^2} = \sqrt{ \sum_{i=1}^N v_i^2 } @f$.
 * @param [in] x Pointer to vector whose norm is to be evaluated
 * @param [in] size Number of elements in the vector
 * @return The @f$\ell^2@f$-norm of the vector
 */
inline double l2Norm(double const* const x, int size)
{
	return std::sqrt(l2NormSquared(x, size));
}

/**
 * @brief Computes the (discrete) @f$\ell^\infty@f$-norm of the given vector
 * @details The (discrete) @f$\ell^\infty@f$-norm is given by @f$ \lVert v \rVert_{\ell^\infty} = \max \{ \abs{v_i} : i
 * = 1, \dots, N \} @f$.
 * @param [in] x Pointer to vector whose norm is to be evaluated
 * @param [in] size Number of elements in the vector
 * @return The @f$\ell^\infty@f$-norm of the vector
 */
inline double linfNorm(double const* const x, int size)
{
	double res = 0.0;
	for (int i = 0; i < size; ++i)
	{
#ifdef CADET_DEBUG
		if (cadet_unlikely(std::isnan(x[i])))
			return std::numeric_limits<double>::quiet_NaN();
#endif
		res = std::max(std::abs(x[i]), res);
	}
	return res;
}

/**
 * @brief Computes the weighted (discrete) @f$\lVert D^{-1} \cdot \rVert_{\ell^2}@f$-norm of the given vector
 * @details The (discrete) weighted @f$\ell^2(w)@f$-norm is given by @f$ \lVert v \rVert_{\ell^2(w)} = \sqrt{
 * \sum_{i=1}^N \left(\frac{v_i}{w_i}\right)^2 } @f$, where @f$ w @f$ is a vector with weights and @f$ D =
 * \operatorname{diag}(w) @f$.
 * @param [in] x Pointer to vector whose norm is to be evaluated
 * @param [in] weight Pointer to weight vector @f$ w @f$
 * @param [in] size Number of elements in the vector
 * @return The weighted @f$\ell^2(w)@f$-norm of the vector
 */
inline double l2normWeighted(double const* const x, double const* const weight, int size)
{
	double res = 0.0;
	for (int i = 0; i < size; ++i)
		res += sqr(x[i] / weight[i]);
	return std::sqrt(res);
}

/**
 * @brief Computes the (discrete) @f$\ell^1@f$-norm of the difference of the given vectors
 * @details The (discrete) @f$\ell^1@f$-norm is given by @f$ \lVert x - y \rVert_{\ell^1} = \sum_{i=1}^N \abs{x_i - y_i}
 * @f$.
 * @param [in] x Pointer to vector @f$ x @f$
 * @param [in] y Pointer to vector @f$ y @f$
 * @param [in] size Number of elements in the vector
 * @return The @f$\ell^1@f$-norm of the difference
 */
inline double l1NormDiff(double const* const x, double const* const y, int size)
{
	double res = 0.0;
	for (int i = 0; i < size; ++i)
		res += std::abs(x[i] - y[i]);
	return res;
}

/**
 * @brief Computes the squared (discrete) @f$\ell^2@f$-norm of the difference of the given vectors
 * @details The squared (discrete) @f$\ell^2@f$-norm is given by @f$ \lVert x - y \rVert_{\ell^2}^2 = \sum_{i=1}^N
 * \left(x_i - y_i\right)^2 @f$.
 * @param [in] x Pointer to vector @f$ x @f$
 * @param [in] y Pointer to vector @f$ y @f$
 * @param [in] size Number of elements in the vector
 * @return The squared @f$\ell^2@f$-norm of the difference
 */
inline double l2NormSquaredDiff(double const* const x, double const* const y, int size)
{
	double res = 0.0;
	for (int i = 0; i < size; ++i)
		res += sqr(x[i] - y[i]);
	return res;
}

/**
 * @brief Computes the (discrete) @f$\ell^2@f$-norm of the difference of the given vectors
 * @details The (discrete) @f$\ell^2@f$-norm is given by @f$ \lVert x - y \rVert_{\ell^2} = \sqrt{ \sum_{i=1}^N
 * \left(x_i - y_i\right)^2 } @f$.
 * @param [in] x Pointer to vector @f$ x @f$
 * @param [in] y Pointer to vector @f$ y @f$
 * @param [in] size Number of elements in the vector
 * @return The @f$\ell^2@f$-norm of the difference
 */
inline double l2NormDiff(double const* const x, double const* const y, int size)
{
	return std::sqrt(l2NormSquaredDiff(x, y, size));
}

/**
 * @brief Computes the (discrete) @f$\ell^\infty@f$-norm of the difference of the given vectors
 * @details The (discrete) @f$\ell^\infty@f$-norm is given by @f$ \lVert x - y \rVert_{\ell^\infty} = \max \{ \abs{x_i -
 * y_i} : i = 1, \dots, N \} @f$.
 * @param [in] x Pointer to vector @f$ x @f$
 * @param [in] y Pointer to vector @f$ y @f$
 * @param [in] size Number of elements in the vector
 * @return The @f$\ell^\infty@f$-norm of the difference
 */
inline double linfNormDiff(double const* const x, double const* const y, int size)
{
	double res = 0.0;
	for (int i = 0; i < size; ++i)
	{
		const double diff = x[i] - y[i];
#ifdef CADET_DEBUG
		if (cadet_unlikely(std::isnan(diff)))
			return std::numeric_limits<double>::quiet_NaN();
#endif
		res = std::max(std::abs(diff), res);
	}
	return res;
}

} // namespace linalg

} // namespace cadet

#endif // LIBCADET_NORMS_HPP_
