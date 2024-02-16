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
 * Implements the High Resolution Koren method
 */

#ifndef LIBCADET_HIGHRESKOREN_HPP_
#define LIBCADET_HIGHRESKOREN_HPP_

#include "AutoDiff.hpp"
#include "MathUtil.hpp"
#include "Memory.hpp"
#include "common/CompilerSpecific.hpp"
#include "cadet/Exceptions.hpp"

#include <algorithm>

namespace cadet
{

/**
 * @brief Implements the High Resolution Koren scheme for convection
 * @details This scheme is based on upwind stencils and provides WENO methods 1-1, 2-3, and 3-5.
 *          In general, WENO achieves order \f$ r \f$ using a stencil with \f$ (2r-1) \f$ points
 *          that is subdivided into \f$ r \f$ smaller stencils having \f$ r \f$ points each.
 *          WENO combines all substencils with an estimate of the smoothness of the solution (also obtained from the
 *          substencils) in order to achieve a non-oscillatory high order reconstruction of the face values given
 *          volume averages (cell boundary fluxes in finite volume schemes).
 *          For details see \cite Liu1994 and \cite Jiang1996.
 */
class HighResolutionKoren
{
public:

	/**
	 * @brief Creates the HighResolutionKoren scheme
	 * @details The order is 2. Remember to add these parameters in ConvectionDispersionOperator.cpp.
	 */
	HighResolutionKoren(): _epsilon(1e-10), _order(2) { }

	/**
    * @brief Returns the maximum order \f$ r \f$ of the implemented schemes
    * @return Maximum WENO order \f$ r \f$
    */
	CADET_CONSTEXPR static inline unsigned int maxOrder() CADET_NOEXCEPT { return 2; }

	/**
	 * @brief Returns the maximum stencil size for the implemented schemes
	 * @return Maximum stencil size
	 */
	CADET_CONSTEXPR static inline unsigned int maxStencilSize() CADET_NOEXCEPT { return 2 * maxOrder() - 1; }

	/**
	 * @brief Reconstructs a cell face value from volume averages
	 * @param [in] cellIdx Index of the current cell
	 * @param [in] numCells Number of cells
	 * @param [in] w Stencil that contains 3 volume averages from which the cell face values are reconstructed centered at the 
	 *               current cell (i.e., index 0 is the current cell, -2 the next to previous cell, 2 the next but one cell)
	 * @param [out] result Reconstructed cell face value
	 * @param [out] Dvm Gradient of the reconstructed cell face value. The array needs to be of size 3, which are the derivatives of right face flux wrt n_{i-1}, n_{i}, n_{i+1}. 
	 * @tparam StateType Type of the state variables
	 * @tparam StencilType Type of the stencil (can be a dedicated class with overloaded operator[] or a simple pointer)
	 * @return Order of the scheme that was used in the computation: 2 or 1
	 */
	template <typename StateType, typename StencilType>
	int reconstruct(unsigned int cellIdx, unsigned int numCells, const StencilType& w, StateType& result, double* const Dvm)
	{
		return reconstruct<StateType, StencilType, true>(cellIdx, numCells, w, result, Dvm);
	}

	/**
	 * @brief Reconstructs a cell face value from volume averages
	 * @param [in] cellIdx Index of the current cell
	 * @param [in] numCells Number of cells
	 * @param [in] w Stencil that contains 3 volume averages from which the cell face values are reconstructed centered at the 
	 *               current cell (i.e., index 0 is the current cell, -2 the next to previous cell, 2 the next but one cell)
	 * @param [out] result Reconstructed cell face value
	 * @tparam StateType Type of the state variables
	 * @tparam StencilType Type of the stencil (can be a dedicated class with overloaded operator[] or a simple pointer)
	 * @return Order of the scheme that was used in the computation: 2 or 1
	 */
	template <typename StateType, typename StencilType>
	int reconstruct(unsigned int cellIdx, unsigned int numCells, const StencilType& w, StateType& result)
	{
		return reconstruct<StateType, StencilType, false>(cellIdx, numCells, w, result, nullptr);
	}

	/**
	 * @brief Returns the \f$ \varepsilon \f$ of the HR Koren emthod (prevents division by zero in the weights)
	 * @return HR Koren \f$ \varepsilon \f$
	 */
	inline double epsilon() const CADET_NOEXCEPT { return _epsilon; }

	/**
	 * @brief Sets the order
	 * @param [in] order Order of the HR Koren method
	 */
	inline void order(int order)
	{
		cadet_assert(order <= 2); // what is cadet_assert?  Anyway, max order is set to 2.
		cadet_assert(order > 0);
		_order = order;
	}

	/**
	 * @brief Returns the HR Koren order
	 * @return Order of the HR Koren method
	 */
	inline int order() const CADET_NOEXCEPT { return _order; }

	/**
	 * @brief Returns the number of upper diagonals required in the Jacobian
	 * @return Number of required Jacobian upper diagonals
	 */
	inline unsigned int upperBandwidth() const CADET_NOEXCEPT { return _order - 1; }

	/**
	 * @brief Returns the number of lower diagonals required in the Jacobian
	 * @return Number of required Jacobian lower diagonals
	 */
	inline unsigned int lowerBandwidth() const CADET_NOEXCEPT { return _order - 1; }

	/**
	 * @brief Returns the size of the stencil (i.e., the number of required elements)
	 * @return Size of the stencil. Either 3 or 1. 
	 */
	inline unsigned int stencilSize() const CADET_NOEXCEPT { return 2 * _order - 1; }

private:

	 /**
      * @brief Reconstructs a cell face value from volume averages
      * @param [in] cellIdx Index of the current cell
      * @param [in] numCells Number of cells
      * @param [in] w Stencil that contains 3 volume averages from which the cell face values are reconstructed centered at the
      *               current cell (i.e., index 0 is the current cell, -2 the next to previous cell, 2 the next but one cell)
      * @param [out] result Reconstructed cell face value
      * @param [out] Dvm Gradient of the reconstructed cell face value. The array needs to be of size 3, which are the derivatives of right face flux wrt c_{i-1}, c_{i}, c_{i+1}.
      * @tparam StateType Type of the state variables
      * @tparam StencilType Type of the stencil (can be a dedicated class with overloaded operator[] or a simple pointer)
      * @tparam wantJac Determines if the gradient is computed (@c true) or not (@c false)
      * @return Order of the scheme that was used in the computation: 2 or 1
      */

	template <typename StateType, typename StencilType, bool wantJac>
	int reconstruct(unsigned int cellIdx, unsigned int numCells, const StencilType& w, StateType& result, double* const Dvm)
	{
#if defined(ACTIVE_SETFAD) || defined(ACTIVE_SFAD)
		using cadet::sqr;
		using sfad::sqr;
#endif

		// This very statement selects the max. order for the current column cell
		// order = min(maxOrderleft, maxOrderright). It can be either 1 or 2 depending on the cellIdx. 
		const int order = std::min(std::min(static_cast<int>(cellIdx) + 1, _order), std::min(static_cast<int>(numCells - cellIdx), _order));

		// upwind scheme when order=1
		if (order == 1)
		{
			result = w[0];
			if (wantJac)
				*Dvm = 1.0;
			return order;
		}

		// calculate the smoothness indicators
		StateType const r = (w[0] - w[-1] + _epsilon) / (w[1] - w[0] + _epsilon);

		// apply the van Leer flux limiter. Other flux limiters may be applied here.
		StateType const vLf = (cadet_likely(r > 0)) ? r / (1 + r) : 0.0;

		// calculate the flux (0.5 * 2.0 are merged)
		result = w[0] + vLf * (w[1] - w[0]);

		// define some jacobian related local variables
		StateType const r_1_1 = 1.0 / (1.0 + r);
		StateType const ci_complex = 1.0 - _epsilon / (w[1] - w[0] + _epsilon);

		if (wantJac)
		{ 
			// Dvm start with index 0 (drivative wrt c_{i-1})
			if (cadet_likely(r > 0))
			{
				Dvm[0] = static_cast<double>(r_1_1) * (static_cast<double>(vLf) - 1.0) * static_cast<double>(ci_complex);
				Dvm[1] = 1.0 - static_cast<double>(ci_complex) * static_cast<double>(vLf) * static_cast<double>(vLf) - static_cast<double>(ci_complex) * static_cast<double>(r_1_1) * static_cast<double>(vLf) + static_cast<double>(ci_complex) * static_cast<double>(vLf) - static_cast<double>(vLf) + static_cast<double>(r_1_1) * static_cast<double>(ci_complex);
				Dvm[2] = static_cast<double>(vLf) * static_cast<double>(vLf) * static_cast<double>(ci_complex) - static_cast<double>(vLf) * static_cast<double>(ci_complex) + static_cast<double>(vLf);
			}
			else
			{
				Dvm[0] = 0.0;
				Dvm[1] = 1.0; // becomes upwind when r<=0
				Dvm[2] = 0.0;
			}
		}

		return order;
	}

	int _order; //!< Selected order: 2 or 1 for HR Koren
	double _epsilon; //!< Small number preventing divsion by zero

};

} // namespace cadet

#endif  // LIBCADET_HIGHRESKOREN_HPP_
