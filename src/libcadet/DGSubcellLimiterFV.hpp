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
 * Implements the subcell FV limiting method for collocation DG
 */

#ifndef LIBCADET_DGSubcellLimiterFV_HPP_
#define LIBCADET_DGSubcellLimiterFV_HPP_

#include "AutoDiff.hpp"
#include "MathUtil.hpp"
#include "Memory.hpp"
#include "common/CompilerSpecific.hpp"
#include "cadet/Exceptions.hpp"

#include <algorithm>

namespace cadet
{
	/**
	 * @brief Implements the subcell FV slope limited reconstruction
	 * @detail Defines the 2nd order (linear) subcell FV reconstruction by implementing its limiter.
	 */
	// todo active types required in reconstruction?
	class SlopeLimiterFV {
	public:
		enum class SlopeLimiterID : int
		{
			NoLimiter1stOrder = 0, //!< 1st order (constant) reconstruction, i.e. no reconstruction
			Minmod = 1, //!< Minmod slope limiter
			Superbee = 2, //!< Superbee slope limiter
			vanLeer = 3, //!< van Leer slope limiter
			vanAlbada = 4, //!< van Alaba slope limiter
			//Koren = 5, //!< Koren slope limiter // todo: can we use a 3rd order approximation, i.e. can we accomodate the MUSCL scheme/k-interpolation(vanLeer) into our setting?
			RelaxedVanAlbada = 6, //!< relaxed van Alaba slope limiter
			//MonotonizedCentral = 7, //!< Monotonized central slope limiter // interface change required to compute central slope
			NoLimiterForward = -1, //!< unlimited forward FD reconstruction
			NoLimiterBackward = -2, //!< unlimited backward FD reconstruction
		};

		virtual ~SlopeLimiterFV() {}
		virtual double call(double fwdSlope, double bwdSlope) = 0;
		virtual active call(active fwdSlope, active bwdSlope) = 0;
		virtual SlopeLimiterID getID() = 0;
	};

	/**
	 * @brief 1st order (constant) reconstruction
	 */
	class NoLimiter1stOrder : public SlopeLimiterFV {

	public:
		NoLimiter1stOrder() {}
		double call(double fwdSlope, double bwdSlope) override {
			return 0.0;
		}
		active call(active fwdSlope, active bwdSlope) override {
			return 0.0;
		}
		SlopeLimiterID getID() override { return SlopeLimiterID::NoLimiter1stOrder; }
	};

	/**
	 * @brief MINMOD reconstruction slope limiter
	 */
	class Minmod : public SlopeLimiterFV {

	public:
		Minmod() {}
		double call(double fwdSlope, double bwdSlope) override {
			if (std::signbit(fwdSlope) == std::signbit(bwdSlope))
				return std::copysign(std::min(std::abs(fwdSlope), std::abs(bwdSlope)), fwdSlope);
			else
				return 0.0;
		}
		active call(active fwdSlope, active bwdSlope) override {

			if (std::signbit(static_cast<double>(fwdSlope)) == std::signbit(static_cast<double>(bwdSlope)))
				return abs(static_cast<double>(fwdSlope)) < abs(static_cast<double>(bwdSlope)) ? fwdSlope : bwdSlope;
			else
				return active(0.0);
		}
		SlopeLimiterID getID() override { return SlopeLimiterID::Minmod; }
	};

	/**
	 * @brief Superbee slope limiter for reconstruction
	 */
	class Superbee : public SlopeLimiterFV {

	public:
		Superbee() {}

		double maxmod(double fwdSlope, double bwdSlope) {
			if (std::signbit(fwdSlope) == std::signbit(bwdSlope))
				return std::copysign(std::max(std::abs(fwdSlope), std::abs(bwdSlope)), fwdSlope);
			else
				return 0.0;
		}
		active maxmod(active fwdSlope, active bwdSlope) {

			if (std::signbit(static_cast<double>(fwdSlope)) == std::signbit(static_cast<double>(bwdSlope)))
				return abs(static_cast<double>(fwdSlope)) < abs(static_cast<double>(bwdSlope)) ? bwdSlope : fwdSlope;
			else
				return active(0.0);
		}

		double call(double fwdSlope, double bwdSlope) override {
			return maxmod(minmod.call(2 * fwdSlope, bwdSlope), minmod.call(fwdSlope, 2 * bwdSlope));
		}
		active call(active fwdSlope, active bwdSlope) override {
			return maxmod(minmod.call(2 * fwdSlope, bwdSlope), minmod.call(fwdSlope, 2 * bwdSlope));
		}
		SlopeLimiterID getID() override { return SlopeLimiterID::Superbee; }

	private:
		Minmod minmod;
	};

	/**
	 * @brief van Leer slope limiter for reconstruction
	 */
	class vanLeer : public SlopeLimiterFV {

	public:
		vanLeer() {}

		double call(double fwdSlope, double bwdSlope) override {
			if (std::signbit(fwdSlope) == std::signbit(bwdSlope))
			{
				if (std::abs(fwdSlope) < eps && std::abs(bwdSlope) < eps) // catch zero-division problem
					return 0.0;
				else
					return (2.0 * fwdSlope * bwdSlope) / (fwdSlope + bwdSlope);
			}
			else
				return 0.0;
		}
		active call(active fwdSlope, active bwdSlope) override {
			if (std::signbit(static_cast<double>(fwdSlope)) == std::signbit(static_cast<double>(bwdSlope)))
			{
				if (abs(fwdSlope) < eps && abs(bwdSlope) < eps) // catch zero-division problem
					return 0.0;
				else
					return (2.0 * fwdSlope * bwdSlope) / (fwdSlope + bwdSlope);
			}
			else
				return 0.0;
		}
		SlopeLimiterID getID() override { return SlopeLimiterID::vanLeer; }

	private:
		const double eps = 1e-8;
	};

	/**
	 * @brief van Albada slope limiter for reconstruction
	 */
	class vanAlbada : public SlopeLimiterFV {

	public:
		vanAlbada() {}

		double call(double fwdSlope, double bwdSlope) override {
			if (std::signbit(fwdSlope) == std::signbit(bwdSlope))
				return ((fwdSlope * fwdSlope + eps) * bwdSlope + ((bwdSlope * bwdSlope + eps) * fwdSlope)) / (fwdSlope * fwdSlope + bwdSlope * bwdSlope + 2.0 * eps);
			else
				return 0.0;
		}
		active call(active fwdSlope, active bwdSlope) override {
			if (std::signbit(static_cast<double>(fwdSlope)) == std::signbit(static_cast<double>(bwdSlope)))
				return ((fwdSlope * fwdSlope + eps) * bwdSlope + ((bwdSlope * bwdSlope + eps) * fwdSlope)) / (fwdSlope * fwdSlope + bwdSlope * bwdSlope + 2.0 * eps);
			else
				return 0.0;
		}
		SlopeLimiterID getID() override { return SlopeLimiterID::vanAlbada; }

	private:
		const double eps = 1e-6; // constant chosen according to https://doi.org/10.2514/3.9465
	};

	/**
	 * @brief relaxed van Albada slope limiter for reconstruction
	 * @detail relaxed w.r.t. required arithmetic operations, according to https://doi.org/10.1007/978-981-15-9011-5_5
	 */
	class RelaxedVanAlbada : public SlopeLimiterFV {

	public:
		RelaxedVanAlbada() {}

		double call(double fwdSlope, double bwdSlope) override {
			if (std::signbit(fwdSlope) == std::signbit(bwdSlope))
				return fwdSlope * (2.0 * fwdSlope * bwdSlope + eps) / (fwdSlope * fwdSlope + bwdSlope * bwdSlope + eps);
			else
				return 0.0;
		}
		active call(active fwdSlope, active bwdSlope) override {
			if (std::signbit(static_cast<double>(fwdSlope)) == std::signbit(static_cast<double>(bwdSlope)))
				return fwdSlope * (2.0 * fwdSlope * bwdSlope + eps) / (fwdSlope * fwdSlope + bwdSlope * bwdSlope + eps);
			else
				return 0.0;
		}
		SlopeLimiterID getID() override { return SlopeLimiterID::RelaxedVanAlbada; }

	private:
		const double eps = 1e-6; // constant chosen according to https://doi.org/10.2514/3.9465
	};

	///**
	//* @brief Monotonized central slope limiter for reconstruction
	//* @todo change interface to compute central slope?
	//*/
	//class MonotonizedCentral : public SlopeLimiterFV {

	//public:
	//	MonotonizedCentral() {}

	//	double minmod3(double fwdSlope, double cSlope, double bwdSlope) {
	//		if (std::signbit(fwdSlope) == std::signbit(cSlope) && std::signbit(fwdSlope) == std::signbit(bwdSlope))
	//			return std::copysign(std::max(std::abs(fwdSlope), std::abs(cSlope), std::abs(bwdSlope)), fwdSlope);
	//		else
	//			return 0.0;
	//	}
	//	active minmod3(active fwdSlope, active cSlope, active bwdSlope) {

	//		if (std::signbit(static_cast<double>(fwdSlope)) == std::signbit(static_cast<double>(cSlope)) && std::signbit(static_cast<double>(fwdSlope)) == std::signbit(static_cast<double>(bwdSlope)))
	//		{
	//			if (abs(static_cast<double>(fwdSlope)) < abs(static_cast<double>(bwdSlope)))
	//			{
	//				if (abs(static_cast<double>(fwdSlope)) < abs(static_cast<double>(cSlope)))
	//				{
	//					return fwdSlope;
	//				}
	//				else return cSlope;
	//			}
	//			else
	//			{
	//				if (abs(static_cast<double>(bwdSlope)) < abs(static_cast<double>(cSlope)))
	//				{
	//					return bwdSlope;
	//				}
	//				else return cSlope;
	//			}
	//		}
	//		else
	//			return active(0.0);
	//	}

	//	double call(double fwdSlope, double bwdSlope) override {
	//		return minmod3(2.0 * fwdSlope, cSlope, 2.0 * bwdSlope);
	//	}
	//	active call(active fwdSlope, active bwdSlope) override {
	//		return minmod3(2.0 * fwdSlope, cSlope, 2.0 * bwdSlope);
	//	}

	//	SlopeLimiterID getID() override { return SlopeLimiterID::MonotonizedCentral; }
	//};

	/**
	 * @brief Implements unlimited forward slope for reconstruction
	 */
	class NoLimiterForward : public SlopeLimiterFV {

	public:
		NoLimiterForward() {}
		double call(double fwdSlope, double bwdSlope) override {
			return bwdSlope;
		}
		active call(active fwdSlope, active bwdSlope) override {
			return bwdSlope;
		}
		SlopeLimiterID getID() override { return SlopeLimiterID::NoLimiterForward; }
	};

	/**
	 * @brief Implements unlimited backward slope for reconstruction
	 */
	class NoLimiterBackward : public SlopeLimiterFV {

	public:
		NoLimiterBackward() {}
		double call(double fwdSlope, double bwdSlope) override {
			return fwdSlope;
		}
		active call(active fwdSlope, active bwdSlope) override {
			return fwdSlope;
		}
		SlopeLimiterID getID() override { return SlopeLimiterID::NoLimiterBackward; }
	};

	///**
	// * @brief Implements unlimited central slope reconstruction
	// */
	//class NoLimiterCentral : public SlopeLimiterFV {

	//public:
	//	NoLimiterCentral() {}
	//	double call(double fwdSlope, double bwdSlope) override {
	//		return fwdSlope;
	//	}
	//	active call(active fwdSlope, active bwdSlope) override {
	//		return fwdSlope;
	//	}
	// SlopeLimiterID getID() override { return SlopeLimiterID::NoLimiterCentral; }
	//};

	// todo, note: if more slope limiters are added, note that they have to be symmetrical in the current implementation (i.e. in reconstruction function reconstructedUpwindValue)

	/**
	 * @brief Implements the subcell FV limiting scheme for convection
	 * @details //@TODO.
	 */
	class DGSubcellLimiterFV
	{
	public:

		/**
		 * @brief Boundary treatment method determines how the reconstruction handles DG element boundaries.
		 */
		enum class BoundaryTreatment : int
		{
			LimiterSlope = 0, //!< Slope limiter reconstruction using neighbour interface value.
			CentralSlope = 1, //!< Central slope reconstruction with interior information.
			Constant = 2 //!< Constant reconstruction of boundary subcells.
		};

		/**
		 * @brief Creates the subcell Finite Volume scheme
		 */
		DGSubcellLimiterFV() : _LGLweights(nullptr), _LGLnodes(nullptr), _subcellGrid(nullptr) { }

		~DGSubcellLimiterFV() CADET_NOEXCEPT
		{
			delete[] _LGLweights;
			delete[] _LGLnodes;
			delete[] _subcellGrid;
		}

		void init(std::string limiter, const int FVorder, const int  boundaryTreatment, const unsigned int nNodes, double* LGLnodes, double* invWeights, const unsigned int nCells, const unsigned int nComp) {

			_nNodes = nNodes;
			_polyDeg = nNodes - 1;
			_nComp = nComp;

			_LGLweights = new double[nNodes];
			for (int node = 0; node < nNodes; node++)
				_LGLweights[node] = 1.0 / invWeights[node];

			_LGLnodes = new double[nNodes];
			std::copy(LGLnodes, LGLnodes + nNodes, _LGLnodes);

			_subcellGrid = new double[nNodes + 1];
			_subcellGrid[0] = -1.0;
			for (int subcell = 1; subcell < nNodes + 1; subcell++)
				_subcellGrid[subcell] = _subcellGrid[subcell - 1] + _LGLweights[subcell - 1];

			_FVorder = FVorder;
			_slope_limiter = std::make_unique<NoLimiter1stOrder>();

			if (_FVorder == 2) {
				if (limiter == "MINMOD")
					_slope_limiter = std::make_unique<Minmod>();
				else if (limiter == "UNLIMITED_FORWARD_RECONSTRUCTION")
					_slope_limiter = std::make_unique<NoLimiterForward>();
				else if (limiter == "SUPERBEE")
					_slope_limiter = std::make_unique<Superbee>();
				else if (limiter == "VANLEER")
					_slope_limiter = std::make_unique<vanLeer>();
				else if (limiter == "VANALBADA")
					_slope_limiter = std::make_unique<vanAlbada>();
				else if (limiter == "RELAXED_VANALBADA")
					_slope_limiter = std::make_unique<RelaxedVanAlbada>();
				else if (limiter != "NONE")
					throw InvalidParameterException("Subcell FV slope limiter " + limiter + " unknown.");

				switch (boundaryTreatment)
				{
				case static_cast<typename std::underlying_type<BoundaryTreatment>::type>(BoundaryTreatment::LimiterSlope):
					_FVboundaryTreatment = BoundaryTreatment::LimiterSlope;
					break;
				case static_cast<typename std::underlying_type<BoundaryTreatment>::type>(BoundaryTreatment::CentralSlope):
					_FVboundaryTreatment = BoundaryTreatment::CentralSlope;
					break;
				case static_cast<typename std::underlying_type<BoundaryTreatment>::type>(BoundaryTreatment::Constant):
					_FVboundaryTreatment = BoundaryTreatment::Constant;
					break;
				default:
					throw InvalidParameterException("Unknown subcell FV boundary treatment.");
				}
			}
			else if (_FVorder != 1)
				throw InvalidParameterException("Subcell FV order must be 1 or 2, but was specified as " + std::to_string(_FVorder));
		}

		void notifyDiscontinuousSectionTransition(int direction) {

			if (_slope_limiter) { // only when subcell limiter is initialized, i.e. used
				// Forward FD slope reconstruction depends on flow direction // todo: Koren limiter which is also non-symmetrical
				if (_slope_limiter->getID() == SlopeLimiterFV::SlopeLimiterID::NoLimiterForward && direction == -1)
					_slope_limiter = std::make_unique<NoLimiterBackward>();

				else if (_slope_limiter->getID() == SlopeLimiterFV::SlopeLimiterID::NoLimiterBackward && direction == 1)
					_slope_limiter = std::make_unique<NoLimiterForward>();
			}
		}

		/**
		 * @brief Implements FV (limited) reconstruction
		 * @details TODO
		 * @param [in] leftState left neighbour state
		 * @param [in] centerState current state
		 * @param [in] rightState right neighbour state
		 * @param [in] subcellIdx current (center state) subcell index
		 * @param [in] rightInterface specifies whther right or left interface reconstruction value is returned
		 * @return @c reconstructed FV subcell value at left or right interface
		 */
		// forward backward flow unterscheidung. Check slope computation.
		template<typename StateType>
		StateType reconstructedInterfaceValue(const StateType leftState, const StateType centerState, const StateType rightState, const int subcellIdx, bool rightInterface) {

			if (_FVorder == 1) // No reconstruction
				return centerState;
			else // Order = 2 -> reconstruction
			{
				// TODO what happens here with non-symmetrical slope limiters?
				StateType slope;

				// Boundary cell reconstruction
				if (subcellIdx == 0)
				{
					switch (_FVboundaryTreatment)
					{
					default:
					case BoundaryTreatment::LimiterSlope:
						slope = _slope_limiter->call((rightState - centerState) / (_LGLnodes[1] - _LGLnodes[0]), (rightState - leftState) / (_LGLnodes[1] - _LGLnodes[0]));
						break;

					case BoundaryTreatment::CentralSlope:
						slope = (rightState - centerState) / (_LGLnodes[1] - _LGLnodes[0]);
						break;

					case BoundaryTreatment::Constant:
						return centerState;
					}
				}
				else if (subcellIdx == _nNodes - 1)
				{
					switch (_FVboundaryTreatment)
					{
					default:
					case BoundaryTreatment::LimiterSlope:
						slope = _slope_limiter->call((centerState - leftState) / (_LGLnodes[_nNodes - 1] - _LGLnodes[_nNodes - 2]), (rightState - leftState) / (_LGLnodes[_nNodes - 1] - _LGLnodes[_nNodes - 2]));
						break;

					case BoundaryTreatment::CentralSlope:
						slope = (centerState - leftState) / (_LGLnodes[_nNodes - 1] - _LGLnodes[_nNodes - 2]);
						break;

					case BoundaryTreatment::Constant:
						return centerState;
					}
				}
				else // Inner subcells
					slope = _slope_limiter->call((rightState - centerState) / (_LGLnodes[subcellIdx + 1] - _LGLnodes[subcellIdx]), (centerState - leftState) / (_LGLnodes[subcellIdx] - _LGLnodes[subcellIdx - 1]));

				if (rightInterface == 1)
					return centerState + slope * (_subcellGrid[subcellIdx + 1] - _LGLnodes[subcellIdx]); // Return state at right interface, i.e. forward flow upwind state
				else
					return centerState + slope * (_subcellGrid[subcellIdx] - _LGLnodes[subcellIdx]); // Return state at left interface, i.e. backward flow upwind state
			}
		}

		inline double LGLweights(int idx) { return _LGLweights[idx]; }

		private:

			std::unique_ptr<SlopeLimiterFV> _slope_limiter; //!< Slope limiter for second order FV reconstruction
			unsigned int _FVorder; //!< FV subcell method order
			BoundaryTreatment _FVboundaryTreatment; //!< boundary treatment of FV subcell method
			double* _LGLweights; //!< DG method LGL weights, i.e. subcell sizes
			double* _LGLnodes; //!< DG method LGL weights, i.e. subcell sizes
			int _nComp; //!< Number of liquid phase components
			int _nNodes; //!< Number of DG nodes per element, i.e. FV subcells
			int _polyDeg; //!< Polynomial degree of DG Ansatz
			double* _subcellGrid; //!< FV subcell grid, i.e. subcell borders
	};

} // namespace cadet

#endif  // LIBCADET_DGSubcellLimiterFV_HPP_
