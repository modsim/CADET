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
 * Implements the WENO method
 */

#ifndef LIBCADET_WENO_HPP_
#define LIBCADET_WENO_HPP_

#include "AutoDiff.hpp"
#include "MathUtil.hpp"
#include "Memory.hpp"
#include "common/CompilerSpecific.hpp"
#include "cadet/Exceptions.hpp"

#include <algorithm>

namespace cadet
{

/**
 * @brief Implements the WENO scheme for convection
 * @details This scheme is based on upwind stencils and provides WENO methods 1-1, 2-3, and 3-5.
 *          In general, WENO achieves order \f$ r \f$ using a stencil with \f$ (2r-1) \f$ points
 *          that is subdivided into \f$ r \f$ smaller stencils having \f$ r \f$ points each.
 *          WENO combines all substencils with an estimate of the smoothness of the solution (also obtained from the
 *          substencils) in order to achieve a non-oscillatory high order reconstruction of the face values given
 *          volume averages (cell boundary fluxes in finite volume schemes).
 *          For details see \cite Liu1994 and \cite Jiang1996.
 */
class Weno
{
public:

	/**
	 * @brief Boundary treatment method determines how the reconstruction handles insufficient available elements (i.e., less elements available than stencil size)
	 */
	enum class BoundaryTreatment : int
	{
		ReduceOrder = 0, //!< Reduce the order of the WENO method such that the stencil is small enough
		ZeroWeights = 1,  //!< Set weights of WENO method to 0 for unavailable elements
		ZeroWeightsForPnotZero = 2, //!< Set weights of WENO method to 0 for unavailable elements, except for the first cell (order is reduced to 1)
		LargeGhostNodes = 3,
	};

	/**
	 * @brief Creates the WENO scheme
	 */
	Weno() : _epsilon(1e-10), _order(maxOrder()), _boundaryTreatment(BoundaryTreatment::ReduceOrder), _intermediateValues(3 * maxOrder() * sizeof(active)) { }

	/**
	 * @brief Returns the maximum order \f$ r \f$ of the implemented schemes
	 * @return Maximum WENO order \f$ r \f$
	 */
	CADET_CONSTEXPR static inline unsigned int maxOrder() CADET_NOEXCEPT { return 3; }

	/**
	 * @brief Returns the maximum stencil size for the implemented schemes
	 * @return Maximum stencil size
	 */
	CADET_CONSTEXPR static inline unsigned int maxStencilSize() CADET_NOEXCEPT { return 2 * maxOrder() - 1; }

	/**
	 * @brief Reconstructs a cell face value from volume averages
	 * @param [in] cellIdx Index of the current cell
	 * @param [in] numCells Number of cells
	 * @param [in] w Stencil that contains the \f$ 2r-1 \f$ volume averages from which the cell face values are reconstructed centered at the 
	 *               current cell (i.e., index 0 is the current cell, -2 the next to previous cell, 2 the next but one cell)
	 * @param [out] result Reconstructed cell face value
	 * @param [out] Dvm Gradient of the reconstructed cell face value (array has to be of size \f$ 2r-1\f$ where \f$ r \f$ is the WENO order)
	 * @tparam StateType Type of the state variables
	 * @tparam StencilType Type of the stencil (can be a dedicated class with overloaded operator[] or a simple pointer)
	 * @return Order of the WENO scheme that was used in the computation
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
	 * @param [in] w Stencil that contains the \f$ 2r-1 \f$ volume averages from which the cell face values are reconstructed centered at the 
	 *               current cell (i.e., index 0 is the current cell, -2 the next to previous cell, 2 the next but one cell)
	 * @param [out] result Reconstructed cell face value
	 * @tparam StateType Type of the state variables
	 * @tparam StencilType Type of the stencil (can be a dedicated class with overloaded operator[] or a simple pointer)
	 * @return Order of the WENO scheme that was used in the computation
	 */
	template <typename StateType, typename StencilType>
	int reconstruct(unsigned int cellIdx, unsigned int numCells, const StencilType& w, StateType& result)
	{
		return reconstruct<StateType, StencilType, false>(cellIdx, numCells, w, result, nullptr);
	}

	/**
	 * @brief Sets the \f$ \varepsilon \f$ of the WENO emthod (prevents division by zero in the weights)
	 * @param [in] eps Boundary treatment method
	 */
	inline void epsilon(double eps) { _epsilon = eps; }

	/**
	 * @brief Returns the \f$ \varepsilon \f$ of the WENO emthod (prevents division by zero in the weights)
	 * @return WENO \f$ \varepsilon \f$
	 */
	inline double epsilon() const CADET_NOEXCEPT { return _epsilon; }

	/**
	 * @brief Sets the WENO order
	 * @param [in] order Order of the WENO method
	 */
	inline void order(int order)
	{
		cadet_assert(order <= static_cast<int>(maxOrder()));
		cadet_assert(order > 0);
		_order = order;
	}
	
	/**
	 * @brief Returns the WENO order
	 * @return Order of the WENO method
	 */
	inline int order() const CADET_NOEXCEPT { return _order; }

	/**
	 * @brief Sets the boundary treatment method
	 * @param [in] bndTreatment Boundary treatment method
	 */
	inline void boundaryTreatment(BoundaryTreatment bndTreatment) { _boundaryTreatment = bndTreatment; }

	/**
	 * @brief Returns the boundary treatment method
	 * @return Boundary treatment method
	 */
	inline BoundaryTreatment boundaryTreatment() const CADET_NOEXCEPT { return _boundaryTreatment; }

	/**
	 * @brief Sets the boundary treatment method
	 * @param [in] bndTreatment Boundary treatment method, is converted to the BoundaryTreatment enum
	 */
	inline void boundaryTreatment(int bndTreatment)
	{
		switch(bndTreatment)
		{
			case static_cast<typename std::underlying_type<BoundaryTreatment>::type>(BoundaryTreatment::ReduceOrder):
				_boundaryTreatment = BoundaryTreatment::ReduceOrder;
				return;
			case static_cast<typename std::underlying_type<BoundaryTreatment>::type>(BoundaryTreatment::ZeroWeights):
				_boundaryTreatment = BoundaryTreatment::ZeroWeights;
				return;
			case static_cast<typename std::underlying_type<BoundaryTreatment>::type>(BoundaryTreatment::ZeroWeightsForPnotZero):
				_boundaryTreatment = BoundaryTreatment::ZeroWeightsForPnotZero;
				return;
			case static_cast<typename std::underlying_type<BoundaryTreatment>::type>(BoundaryTreatment::LargeGhostNodes):
				_boundaryTreatment = BoundaryTreatment::LargeGhostNodes;
				return;
		}
		throw InvalidParameterException("Unknown boundary treatment type");
	}

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
	 * @return Size of the stencil
	 */
	inline unsigned int stencilSize() const CADET_NOEXCEPT { return 2 * _order - 1; }

private:

	/**
	 * @brief Reconstructs a cell face value from volume averages
	 * @param [in] cellIdx Index of the current cell
	 * @param [in] numCells Number of cells
	 * @param [in] w Stencil that contains the \f$ 2r-1 \f$ volume averages from which the cell face values are reconstructed centered at the 
	 *               current cell (i.e., index 0 is the current cell, -2 the next to previous cell, 2 the next but one cell)
	 * @param [out] result Reconstructed cell face value
	 * @param [out] Dvm Gradient of the reconstructed cell face value (array has to be of size \f$ 2r-1\f$ where \f$ r \f$ is the WENO order)
	 * @tparam StateType Type of the state variables
	 * @tparam StencilType Type of the stencil (can be a dedicated class with overloaded operator[] or a simple pointer)
	 * @tparam wantJac Determines if the gradient is computed (@c true) or not (@c false)
	 * @return Order of the WENO scheme that was used in the computation
	 */
	template <typename StateType, typename StencilType, bool wantJac>
	int reconstruct(unsigned int cellIdx, unsigned int numCells, const StencilType& w, StateType& result, double* const Dvm)
	{
#if defined(ACTIVE_SETFAD) || defined(ACTIVE_SFAD)
		using cadet::sqr;
		using sfad::sqr;
#endif

		// Local order of the scheme that is actually used (may be changed by treatment of boundaries)
		int order = _order;

		// Boundaries
		int bnd = 0;
		switch (_boundaryTreatment)
		{
		default:
		case BoundaryTreatment::ReduceOrder:
			// Lower WENO order such that maximum order is used at all points
			// This very statement selects the max. weno order for the current column cell
			// order = min(maxOrderleft, maxOrderright)
			order = std::min(std::min(static_cast<int>(cellIdx) + 1, _order), std::min(static_cast<int>(numCells - cellIdx), _order));
			break;

		case BoundaryTreatment::ZeroWeights:
			// Zero weights for cell averages outside the domain
			if (cellIdx < static_cast<unsigned int>(order - 1))
				bnd = -(order - 1 - cellIdx);
			else if (cellIdx > numCells - order)
				bnd = numCells - cellIdx;
			break;

		case BoundaryTreatment::ZeroWeightsForPnotZero:
			// Zero weights for p != 0
			if (cellIdx == 0)
				order = 1;
			else
			{
				if (cellIdx < static_cast<unsigned int>(order - 1))
					bnd = -(order - 1 - cellIdx);
				else if (cellIdx > numCells - order)
					bnd = numCells - cellIdx;
			}
			break;
/*
		case BoundaryTreatment::LargeGhostNodes:
			// Large ghost points
			if (cellIdx == 0)
			{
				w[-1] = 1e20;
				w[-2] = 1e50;
			}
			else if (cellIdx == numCells - 2)
				w[2] = 1e20;
			else if (cellIdx == numCells - 1)
				w[2] = 1e50;
			break;
*/
		}		

		// Total stencil size
		const int sl = 2 * order - 1;

		// Simple upwind scheme
		if (order == 1)
		{
			result = w[0];
			if (wantJac)
				*Dvm = 1.0;
			return order;
		}

		// Allocate memory for intermediate values: beta, alpha (= omega), and vr
		StateType* const work = _intermediateValues.create<StateType>(3 * order);
		StateType* const beta  = work;
		StateType* const alpha = work + order;
		StateType* const omega = work + order;
		StateType* const vr    = work + 2*order; // Reconstructed values

		const double* d = nullptr;
		const double* c = nullptr;
		const double* Jbvv = nullptr;

		// Calculate smoothness measures
		switch (order)
		{
			case 2:
				beta[0] = sqr(w[1] - w[0]);
				beta[1] = sqr(w[0] - w[-1]);
				d = _wenoD2;
				c = _wenoC2;
				Jbvv = _wenoJbvv2;
				break;
			case 3:
				beta[0] = 13.0/12.0 * sqr(w[ 0] - 2.0 * w[ 1] + w[2]) + 0.25 * sqr(3.0 * w[ 0] - 4.0 * w[ 1] +       w[2]);
				beta[1] = 13.0/12.0 * sqr(w[-1] - 2.0 * w[ 0] + w[1]) + 0.25 * sqr(      w[-1] -       w[ 1]             );
				beta[2] = 13.0/12.0 * sqr(w[-2] - 2.0 * w[-1] + w[0]) + 0.25 * sqr(      w[-2] - 4.0 * w[-1] + 3.0 * w[0]);
				d = _wenoD3;
				c = _wenoC3;
				Jbvv = _wenoJbvv3;
				break;
		}

		// Add eps to avoid divide-by-zeros
		for (int r = 0; r < order; ++r)
			beta[r] += _epsilon;

		// Calculate weights
		for (int r = 0; r < order; ++r)
			alpha[r] = d[r] / sqr(beta[r]);

		// Avoid boundaries
		if (cadet_unlikely(bnd != 0))
		{
			if (bnd < 0)
				// Beginning of interval
				for (int r = 0; r < -bnd; ++r)
					alpha[order - 1 - r] = 0.0;
			else
				// End of interval
				for (int r = 0; r < bnd; ++r)
					alpha[r] = 0.0;
		}

		// Normalize weights
		StateType alpha_sum = alpha[0];
		for (int r = 1; r < order; ++r)
			alpha_sum += alpha[r];
		for (int r = 0; r < order; ++r)
			omega[r] /= alpha_sum;

		// Calculate reconstructed values
		for (int r = 0; r < order; ++r)
		{
			vr[r] = 0.0;
			for (int j = 0; j < order; ++j)
				vr[r] += c[r + order * j] * w[-r+j];
		}

		// Weighted sum
		result = 0;
		for (int r = 0; r < order; ++r)
			result += vr[r] * omega[r];

		// Jacobian
		if (wantJac)
		{
			// Dependencies
			// 1. Constant vr in (*)

			// Start with "d(result)/d(omega)" = vr and
			// multiply with "d(omega)/d(alpha)" to get "d(result)/d(alpha)"
			double dot = 0.0;
			for (int r = 0; r < order; ++r)
				dot += static_cast<double>(vr[r]) * static_cast<double>(omega[r]); //StateType(vr[r] * omega[r]);
			for (int r = 0; r < order; ++r)
				vr[r] = (vr[r] - dot) / alpha_sum;

			// Multiply with "d(alpha)/d(beta)" to get "d(result)/d(beta)"
			for (int r = 0; r < order; ++r)
				vr[r] *= -2.0 * d[r] / pow(beta[r], 3.0);

			// Multiply with "d(beta)/d(v)" to get Dvm = "d(result)/d(v)"
			for (int j = 0; j < sl; ++j)
			{
				Dvm[j] = 0.0;
				for (int r = 0; r < order; ++r)
				{
					dot = 0.0;
					for (int i = 0; i < sl; ++i)
						dot += static_cast<double>(Jbvv[r + order * j + order * sl * i]) * static_cast<double>(w[i - order + 1]);
					// To do: re-arange Jbvv to reduce cache misses !
					Dvm[j] += static_cast<double>(vr[r]) * dot; // StateType(vr[r] * dot);
				}
			}

			// 2. Constant omega[r] in (*)
			for (int r = 0; r < order; ++r)
				for (int j = 0; j < order; ++j)
					Dvm[order - 1 + j - r] += static_cast<double>(omega[r]) * c[r + order * j];
		}

		_intermediateValues.destroy<StateType>();
		return order;
	}

	double _epsilon; //!< Small number preventing divsion by zero
	int _order; //!< Selected WENO order
	BoundaryTreatment _boundaryTreatment; //!< Controls how to treat boundary cells
	ArrayPool _intermediateValues; //!< Buffer for intermediate and temporary values

	static const double _wenoD2[2];
	static const double _wenoC2[2*2];
	static const double _wenoJbvv2[2*3*3];

	static const double _wenoD3[3];
	static const double _wenoC3[3*3];
	static const double _wenoJbvv3[3*5*5];

};

} // namespace cadet

#endif  // LIBCADET_WENO_HPP_
