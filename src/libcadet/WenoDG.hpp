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
			if (std::signbit(u0.getValue()) == std::signbit(u1.getValue()) && std::signbit(u0.getValue()) == std::signbit(u2.getValue()))
				return std::copysign(std::min(std::abs(u0.getValue()), std::min(std::abs(u1.getValue()), std::abs(u2.getValue()))), u0.getValue());
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
		 * @brief Creates the WENO scheme
		 */
		WenoDG() : _weno_gamma(nullptr), _w(nullptr), _wHat(nullptr), _beta(nullptr), weno_limiter(nullptr) { }

		~WenoDG() CADET_NOEXCEPT
		{
			delete[] _weno_gamma;
			delete[] _w;
			delete[] _wHat;
			delete[] _beta;
		}

		void init(std::string limiter, const double eps, const double r, const double gamma, double* invWeights, Eigen::MatrixXd polyDerM, const unsigned int nCells, const unsigned int nNodes, const unsigned int nComp) {

			_nComp = nComp;
			_nCells = nCells;
			_nNodes = nNodes;

			_polyDerM = polyDerM;
			_LGLweights.resize(nNodes);
			for (int node = 0; node < nNodes; node++)
				_LGLweights[node] = 1.0 / invWeights[node];

			_weno_gamma = new double[3]{ (1.0 - gamma) / 2.0, gamma, (1.0 - gamma) / 2.0 };
			_weno_r = r;
			_weno_eps = eps;

			if (limiter == "MINMOD")
				weno_limiter = std::make_unique<MinmodWENO>();
			else if (limiter == "TVB_MINMOD")
				weno_limiter = std::make_unique<TVBMinmodWENO>();
		}

		void setDeltaZ(active deltaZ) { _deltaZ = deltaZ; }

		template<typename StateType>
		void calcSmoothness(const StateType* const localC, const int strideNode, const int comp, const int cell, double* troubled_cells)
		{
			troubled_cells[comp + cell * _nComp] = 2.0;

			Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>> _C(localC, _nNodes, InnerStride<Dynamic>(strideNode));

			if (cell > 0 && cell < _nCells - 1) // todo boundary treatment
			{
				// todo store mass matrix and LGL weights additionally to respective inverse
				// todo overwrite values instead of recalculation // todo deltaZ ParamType
				_pAvg0 = 1.0 / static_cast<double>(_deltaZ) * (_LGLweights.array() * _C.segment((cell - 1) * _nNodes, _nNodes).template cast<StateType>().array()).sum();
				_pAvg1 = 1.0 / static_cast<double>(_deltaZ) * (_LGLweights.array() * _C.segment(cell * _nNodes, _nNodes).template cast<StateType>().array()).sum();
				_pAvg2 = 1.0 / static_cast<double>(_deltaZ) * (_LGLweights.array() * _C.segment((cell + 1) * _nNodes, _nNodes).template cast<StateType>().array()).sum();

				_uTilde = _C[cell * _nNodes + _nNodes - 1] - _pAvg1; // average minus inner interface value on right face
				_u2Tilde = _pAvg1 - _C[cell * _nNodes]; // average minus inner interface value on left face

				double trigger1 = weno_limiter->call(_uTilde, _pAvg2 - _pAvg1, _pAvg1 - _pAvg0);
				double trigger2 = weno_limiter->call(_u2Tilde, _pAvg2 - _pAvg1, _pAvg1 - _pAvg0);

				double M2 = (_polyDerM * _polyDerM * _C.segment(cell * _nNodes, _nNodes).template cast<double>()).maxCoeff();
				//double u_x = (_polyDerM * _C.segment(cell * _nNodes, _nNodes)).maxCoeff();
				double M_ = 2.0 / 3.0 * M2;
				//double hmpf = 2.0 / 9.0 * (3.0 + 10.0 * M2) * M2 * _deltaZ / (_deltaZ + 2.0 * u_x * _deltaZ);

				// reconstruct if cell is troubled, i.e. potential oscillations
				if (abs(trigger1 - _uTilde) > 1e-8 || abs(trigger2 - _u2Tilde) > 1e-8)
				{
					if (abs(_uTilde) > M_ * _deltaZ * _deltaZ || abs(_u2Tilde) > M_ * _deltaZ * _deltaZ)
						// mark troubled cell
						troubled_cells[comp + cell * _nComp] = 1.0;
				}
			}
		}

		//private:

		// indicator for troubled cells
		std::unique_ptr<WENOLimiter> weno_limiter;

		int _nNodes;
		int _nComp;
		int _nCells;

		// WENO parameters
		double* _weno_gamma;
		double _weno_r = 2.0;
		double _weno_eps = 1e-6;

		// weights, smoothness indicator, integral values.
		Eigen::VectorXd _LGLweights; //!< DG method LGL weights, i.e. subcell sizes
		Eigen::MatrixXd _polyDerM; //!< DG method LGL weights, i.e. subcell sizes
		active _deltaZ;
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

	};

} // namespace cadet

#endif  // LIBCADET_WENODG_HPP_
