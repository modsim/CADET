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

#ifndef LIBCADET_WENODG_HPP_
#define LIBCADET_WENODG_HPP_

#include "AutoDiff.hpp"
#include "MathUtil.hpp"
#include "Memory.hpp"
#include "common/CompilerSpecific.hpp"
#include "cadet/Exceptions.hpp"

#include <algorithm>

namespace cadet
{

	//template<typename StateType> // todo
	class WENOLimiter {
	public:
		virtual ~WENOLimiter() {}
		virtual double call(double u0, double u1, double u2) = 0;
		virtual double call(active u0, active u1, active u2) = 0;
	};

	class FullLimiter : public WENOLimiter {

	public:
		FullLimiter() {}
		double call(double u0, double u1, double u2) override {
			return 0.0;
		}
		double call(active u0, active u1, active u2) override {
			return 0.0;
		}
	};

	class MinmodWENO : public WENOLimiter {

	public:
		MinmodWENO() {}
		double call(double u0, double u1, double u2) override {
			if (std::signbit(u0) == std::signbit(u1) && std::signbit(u0) == std::signbit(u2))
				return std::copysign(std::min(std::abs(u0), std::min(std::abs(u1), std::abs(u2))), u0);
			else
				return 0.0;
		}
		double call(active u0, active u1, active u2) override {
			if (std::signbit(u0.getValue()) == std::signbit(u1.getValue()) && std::signbit(u0.getValue()) == std::signbit(u2.getValue())) {
				if (std::signbit(u0.getValue())) // i.e. negative sign
					return -static_cast<double>(u0);
				else
					return static_cast<double>(u0);
			}
			else
				return 0.0;
		}
	};

	class TVBMinmodWENO : public WENOLimiter {

	public:
		TVBMinmodWENO() {}
		double call(double u0, double u1, double u2) override {
			if (std::abs(u0) <= M * h * h)
				return u0;
			else
				return minmod.call(u0, u1, u2);
		}
		double call(active u0, active u1, active u2) override {
			if (std::abs(u0.getValue()) <= M * h * h)
				return static_cast<double>(u0);
			else
				return minmod.call(u0, u1, u2);
		}
	private:
		MinmodWENO minmod;
		active h;
		active M;
	};

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
	class WenoDG
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
		//WenoDG() : _order(maxOrder()), _boundaryTreatment(BoundaryTreatment::ReduceOrder), _intermediateValues(3 * maxOrder() * sizeof(active)) { }

		~WenoDG() CADET_NOEXCEPT
		{
			delete[] _weno_gamma;
			delete[] _w;
			delete[] _wHat;
			delete[] _beta;
			delete[] _troubled_cells;
		}

		void init(const double eps, const double r, const double gamma, const unsigned int nCells, const unsigned int nComp, const bool trackTroubledCells = false) {

			_weno_gamma = new double[3]{ (1 - gamma) / 2.0, gamma, (1 - gamma) / 2.0 };
			_weno_r = r;
			_weno_eps = eps;
			if (trackTroubledCells)
				_troubled_cells = new double[nCells * nComp];
			else
				_troubled_cells = nullptr;
		}

		inline double* troubledCells() const CADET_NOEXCEPT { return &_troubled_cells[0]; }

		//private:

		// indicator for troubled cells
		std::unique_ptr<WENOLimiter> weno_limiter;

		// WENO parameters
		double* _weno_gamma;
		double _weno_r = 2.0;
		double _weno_eps = 1e-6;
		// weights, smoothness indicator, integral values.
		active* _wHat = new active[3]{ 0.0, 0.0, 0.0 };
		active* _w = new active[3]{ 0.0, 0.0, 0.0 };
		active _wSum = 0.0;
		active* _beta = new active[3]{ 0.0, 0.0, 0.0 };
		active _pAvg0 = 0.0;
		active _pAvg1 = 0.0;
		active _pAvg2 = 0.0;
		// limiter inputs
		active _avgUdeltaPlus = 0.0;
		active _avgUdeltaMinus = 0.0;
		active _uTilde = 0.0;
		active _u2Tilde = 0.0;
		// mark troubled cells (smoothness indicator)
		double* _troubled_cells;

	};

} // namespace cadet

#endif  // LIBCADET_WENODG_HPP_
