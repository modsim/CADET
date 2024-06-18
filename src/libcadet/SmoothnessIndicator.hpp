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
 * Implements smooothness indicators used in oscillation suppression for DG
 */


#ifndef LIBCADET_SMOOTHNESSINDICATOR_HPP_
#define LIBCADET_SMOOTHNESSINDICATOR_HPP_

#include "AutoDiff.hpp"
#include <Eigen/Dense>
#include "ParamReaderHelper.hpp"

namespace cadet
{
	/**
	 * @brief indicator to detect and assess DG elements troubled by oscillations and strong gradients
	 */
	class SmoothnessIndicator {
	public:
		virtual ~SmoothnessIndicator() {}
		// todo: no templates for virtual functions. Is there another way to overload?

		/**
		 * @brief smoothness detector for DG element
		 * @param [in] localC pointer to first node of current DG element
		 * @param [in] strideNode node stride in state vector
		 * @param [in] strideLiquid liquid phase stride in state vector
		 * @param [in] cellIdx index of DG element
		 */
		virtual double calcSmoothness(const double* const localC, const int strideNode, const int strideLiquid, const int cellIdx) = 0;
		/**
		 * @brief smoothness detector for DG element
		 * @param [in] localC pointer to first node of current DG element
		 * @param [in] strideNode node stride in state vector
		 * @param [in] strideLiquid liquid phase stride in state vector
		 * @param [in] cellIdx index of DG element
		 */
		virtual double calcSmoothness(const active* const localC, const int strideNode, const int strideLiquid, const int cellIdx) = 0;
		/**
		 * @brief tells if and how much time relaxation is applied, i.e. change in smoothness indication is limited over time
		 */
		inline virtual double timeRelaxation() = 0;

		virtual int configure(IParameterProvider& parameterProvider) = 0;
	};

	/**
	 * @brief dummy indicator that identifies every element as troubled with max weight 1.0
	 */
	class AllElementsIndicator : public SmoothnessIndicator {

	public:

		AllElementsIndicator() {}

		double calcSmoothness(const double* const localC, const int strideNode, const int strideLiquid, const int cellIdx) override { return 1.0; }
		double calcSmoothness(const active* const localC, const int strideNode, const int strideLiquid, const int cellIdx) override { return 1.0; }
		inline double timeRelaxation() override { return false; }
		int configure(IParameterProvider& parameterProvider) override { return 1; };

	};

	/**
	 * @brief indicates troubled elements based on the relative modal energy of the two highes modes
	 */
	class ModalEnergyIndicator : public SmoothnessIndicator {

	public:

		ModalEnergyIndicator() {}
		ModalEnergyIndicator(int polyDeg, const Eigen::MatrixXd& modalVanInv, int numModes, double nodalCoefThreshold, double nodalCoefShift, bool timeRelax)
			: _polyDeg(polyDeg), _numModes(numModes), _modalVanInv(modalVanInv), _nodalCoefThreshold(nodalCoefThreshold), _nodalCoefShift(nodalCoefShift), _timeRelaxation(timeRelax)
		{
			_nNodes = _polyDeg + 1;
			_modalCoeff.resize(_nNodes);
			_modalCoeff.setZero();
		}

		inline double timeRelaxation() override { return _timeRelaxation; }
		int configure(IParameterProvider& parameterProvider) override { return 1; };

		double calcSmoothness(const double* const localC, const int strideNode, const int strideLiquid, const int cellIdx) override
		{
			// TODO: default constants, store them.
			const double _s = 9.2102;

			Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<Eigen::Dynamic>> _C(localC, _nNodes, Eigen::InnerStride<Eigen::Dynamic>(strideNode));

			if (_C.cwiseAbs().maxCoeff() < _nodalCoefThreshold)
				return 0.0;
			else
				_modalCoeff = _modalVanInv * (_C + _nodalCoefShift * Eigen::VectorXd::Ones(_nNodes));

			double modalEnergySquareSum = _modalCoeff.cwiseAbs2().sum();
			double modeEnergy = (modalEnergySquareSum == 0.0) ? 0.0 : std::pow(_modalCoeff[_polyDeg], 2.0) / modalEnergySquareSum;

			if (_numModes == 2) {
				modalEnergySquareSum = _modalCoeff.segment(0, _polyDeg).cwiseAbs2().sum();
				modeEnergy = std::max(modeEnergy, (modalEnergySquareSum == 0.0) ? 0.0 : std::pow(_modalCoeff[_polyDeg - 1], 2.0) / modalEnergySquareSum);
			}

			const double _T = 0.5 * std::pow(10.0, -1.8 * std::pow(static_cast<double>(_nNodes), 0.25)); // todo store this constant
			return 1.0 / (1.0 + exp(-_s / _T * (modeEnergy - _T)));
		}
		double calcSmoothness(const active* const localC, const int strideNode, const int strideLiquid, const int cellIdx) override
		{
			// TODO: default constants, store them.
			const double _s = 9.2102;

			Eigen::Map<const Eigen::Vector<active, Eigen::Dynamic>, 0, Eigen::InnerStride<Eigen::Dynamic>> _C(localC, _nNodes, Eigen::InnerStride<Eigen::Dynamic>(strideNode));
			
			if (_C.template cast<double>().cwiseAbs().maxCoeff() < _nodalCoefThreshold)
				return 0.0;
			else
				_modalCoeff = _modalVanInv * (_C.template cast<double>() + _nodalCoefShift * Eigen::VectorXd::Ones(_nNodes));

			double modalEnergySquareSum = _modalCoeff.cwiseAbs2().sum();
			double modeEnergy = (modalEnergySquareSum == 0.0) ? 0.0 : std::pow(_modalCoeff[_polyDeg], 2.0) / modalEnergySquareSum;

			if (_numModes == 2) {
				modalEnergySquareSum = _modalCoeff.segment(0, _polyDeg).cwiseAbs2().sum();
				modeEnergy = std::max(modeEnergy, (modalEnergySquareSum == 0.0) ? 0.0 : std::pow(_modalCoeff[_polyDeg - 1], 2.0) / modalEnergySquareSum);
			}

			const double _T = 0.5 * std::pow(10.0, -1.8 * std::pow(static_cast<double>(_nNodes), 0.25)); // 10-1.8(N + 1)0.2; // todo store this constant
			return 1.0 / (1.0 + exp(-_s / _T * (modeEnergy - _T)));
		}

	private: 

		int _nNodes; //!< Number of polynomial interpolation nodes of DG ansatz
		int _polyDeg; //!< Polynomial degree of DG ansatz

		double _nodalCoefThreshold; //!< Threshold for nodal polynomial coefficients
		double _nodalCoefShift; //!< Shift for nodal polynomial coefficients (To get c(z) >= eps > 0 at constant 0 concentration values)
		Eigen::VectorXd _modalCoeff; //!< Modal polynomial coefficient memory buffer
		Eigen::MatrixXd _modalVanInv; //!< Inverse Vandermonde matrix of modal polynomial basis
		int _numModes; //!< Number of modes to be considered in modal energy comparison
		double _timeRelaxation; //!< Time relaxation of smoothness indicator, i.e. limit sudden changes by _timeRelaxation * old_value
	};

	/**
	 * @brief indicates troubled elements based on the relative modal energy of the two highes modes but gives a (binary) hard switch instead of a weight
	 */
	class ModalEnergyIndicatorHardSwitch : public SmoothnessIndicator {

	public:

		ModalEnergyIndicatorHardSwitch(int polyDeg, const Eigen::MatrixXd& modalVanInv, int numModes, double nodalCoefThreshold, double nodalCoefShift, bool timeRelax, double FVswitchAt)
			: _FVswitchAt(FVswitchAt)
		{
			_ModalEnergyIndicator = std::make_unique<ModalEnergyIndicator>(polyDeg, modalVanInv, numModes, nodalCoefThreshold, nodalCoefShift, timeRelax);
		}

		inline double timeRelaxation() { return _ModalEnergyIndicator->timeRelaxation(); }
		int configure(IParameterProvider& parameterProvider) override { return 1; };

		double calcSmoothness(const double* const localC, const int strideNode, const int strideLiquid, const int cellIdx) override
		{
			return _ModalEnergyIndicator->calcSmoothness(localC, strideNode, strideLiquid, cellIdx) >= _FVswitchAt ? 1.0 : 0.0;
		}
		double calcSmoothness(const active* const localC, const int strideNode, const int strideLiquid, const int cellIdx) override
		{
			return _ModalEnergyIndicator->calcSmoothness(localC, strideNode, strideLiquid, cellIdx) >= _FVswitchAt ? 1.0 : 0.0;
		}

	private:

		std::unique_ptr<SmoothnessIndicator> _ModalEnergyIndicator;
		double _FVswitchAt;
	};

	/**
	 * @brief indicates troubled elements using slope limiters based on element-averaged and interface slopes
	 */
	class SlopeIndicator : public SmoothnessIndicator {

		/**
		 * @brief slope limiters for three (consecutive) slopes
		 */
		class IndicatorLimiter {
		public:
			virtual ~IndicatorLimiter() {}
			virtual double call(double u0, double u1, double u2) = 0;
			virtual active call(active u0, active u1, active u2) = 0;
		};

		/**
		 * @brief MINMOD slope limiter
		 */
		class MinmodIndicator : public IndicatorLimiter {

		public:
			MinmodIndicator() {}
			double call(double u0, double u1, double u2) override {
				bool sgn = std::signbit(u0);
				if (sgn == std::signbit(u1) && sgn == std::signbit(u2))
					return std::copysign(std::min(std::min(std::abs(u0), std::abs(u1)), std::abs(u1)), u0);
				else
					return 0.0;
			}
			active call(active u0, active u1, active u2) override {
				bool sgn = std::signbit(static_cast<double>(u0));
				if (sgn == std::signbit(static_cast<double>(u1)) && sgn == std::signbit(static_cast<double>(u2)))
				{
					if (abs(static_cast<double>(u0)) < abs(static_cast<double>(u1)))
					{
						if (abs(static_cast<double>(u0)) < abs(static_cast<double>(u2)))
							return (sgn) ? -u0 : u0;
						else
							return (sgn) ? -u2 : u2;
					}
					else if (abs(static_cast<double>(u1)) < abs(static_cast<double>(u2)))
						return (sgn) ? -u1 : u1;
					else
						return (sgn) ? -u2 : u2;
				}
				else
					return active(0.0);
			}
		};

		class TVBMinmodIndicator : public IndicatorLimiter {

		public:
			TVBMinmodIndicator() {}
			double call(double u0, double u1, double u2) override {
				if (std::abs(u0) <= M * h * h)
					return u0;
				else
					return minmod.call(u0, u1, u2);
			}
			active call(active u0, active u1, active u2) override {
				if (std::abs(static_cast<double>(u0)) <= M * h * h)
					return u0;
				else
					return minmod.call(u0, u1, u2);
			}
		private:
			MinmodIndicator minmod;
			active h;
			active M;
		};

	public:

		SlopeIndicator(int polyDeg, int nCells, double deltaZ, Eigen::VectorXd LGLweights, Eigen::MatrixXd polyDerM, std::string limiter, bool timeRelax)
			: _polyDeg(polyDeg), _nCells(nCells), _deltaZ(deltaZ), _LGLweights(LGLweights), _polyDerM(polyDerM), _timeRelaxation(timeRelax)
		{
			_nNodes = _polyDeg + 1;
		}

		inline double timeRelaxation() override { return _timeRelaxation; } // todo time relaxation required in this indicator which is either 1 or 0?
		int configure(IParameterProvider& paramProvider) override
		{
			_deltaZ = paramProvider.getDouble("COL_LENGTH") / _nCells;
			return 1;
		};

		double calcSmoothness(const double* const localC, const int strideNode, const int strideLiquid, const int cellIdx) override
		{
			Eigen::Map<const Eigen::Vector<double, Eigen::Dynamic>, 0, Eigen::InnerStride<Eigen::Dynamic>> _C(localC, _nNodes, Eigen::InnerStride<Eigen::Dynamic>(strideNode));

			if (cellIdx > 0 && cellIdx < _nCells - 1) // todo boundary treatment
			{
				// todo store mass matrix and LGL weights additionally to respective inverse
				// todo overwrite values instead of recalculation // todo deltaZ ParamType
				_pAvg0 = 1.0 / static_cast<double>(_deltaZ) * (_LGLweights.array() * _C.segment((cellIdx - 1) * _nNodes, _nNodes).array()).sum();
				_pAvg1 = 1.0 / static_cast<double>(_deltaZ) * (_LGLweights.array() * _C.segment(cellIdx * _nNodes, _nNodes).array()).sum();
				_pAvg2 = 1.0 / static_cast<double>(_deltaZ) * (_LGLweights.array() * _C.segment((cellIdx + 1) * _nNodes, _nNodes).array()).sum();

				_uTilde = _C[cellIdx * _nNodes + _nNodes - 1] - _pAvg1; // average minus inner interface value on right face
				_u2Tilde = _pAvg1 - _C[cellIdx * _nNodes]; // average minus inner interface value on left face

				double trigger1 = indicatorLimiter->call(_uTilde, _pAvg2 - _pAvg1, _pAvg1 - _pAvg0);
				double trigger2 = indicatorLimiter->call(_u2Tilde, _pAvg2 - _pAvg1, _pAvg1 - _pAvg0);

				double M2 = (_polyDerM * _polyDerM * _C.segment(cellIdx * _nNodes, _nNodes)).maxCoeff();
				//double u_x = (_polyDerM * _C.segment(cell * _nNodes, _nNodes)).maxCoeff();
				double M_ = 2.0 / 3.0 * M2;
				//double hmpf = 2.0 / 9.0 * (3.0 + 10.0 * M2) * M2 * _deltaZ / (_deltaZ + 2.0 * u_x * _deltaZ);

				// reconstruct if cell is troubled, i.e. potential oscillations
				if (abs(trigger1 - _uTilde) > 1e-8 || abs(trigger2 - _u2Tilde) > 1e-8)
				{
					if (abs(_uTilde) > M_ * _deltaZ * _deltaZ || abs(_u2Tilde) > M_ * _deltaZ * _deltaZ)
						return 1.0;
				}
			}
			return 0.0;
		}
		double calcSmoothness(const active* const localC, const int strideNode, const int strideLiquid, const int cellIdx) override
		{
			Eigen::Map<const Eigen::Vector<active, Eigen::Dynamic>, 0, Eigen::InnerStride<Eigen::Dynamic>> _C(localC, _nNodes, Eigen::InnerStride<Eigen::Dynamic>(strideNode));

			if (cellIdx > 0 && cellIdx < _nCells - 1) // todo boundary treatment
			{
				// todo store mass matrix and LGL weights additionally to respective inverse
				// todo overwrite values instead of recalculation // todo deltaZ ParamType
				_pAvg0 = 1.0 / static_cast<double>(_deltaZ) * (_LGLweights.array() * _C.segment((cellIdx - 1) * _nNodes, _nNodes).template cast<double>().array()).sum();
				_pAvg1 = 1.0 / static_cast<double>(_deltaZ) * (_LGLweights.array() * _C.segment(cellIdx * _nNodes, _nNodes).template cast<double>().array()).sum();
				_pAvg2 = 1.0 / static_cast<double>(_deltaZ) * (_LGLweights.array() * _C.segment((cellIdx + 1) * _nNodes, _nNodes).template cast<double>().array()).sum();

				_uTilde = static_cast<double>(_C[cellIdx * _nNodes + _nNodes - 1]) - _pAvg1; // average minus inner interface value on right face
				_u2Tilde = _pAvg1 - static_cast<double>(_C[cellIdx * _nNodes]); // average minus inner interface value on left face

				double trigger1 = indicatorLimiter->call(_uTilde, _pAvg2 - _pAvg1, _pAvg1 - _pAvg0);
				double trigger2 = indicatorLimiter->call(_u2Tilde, _pAvg2 - _pAvg1, _pAvg1 - _pAvg0);

				double M2 = (_polyDerM * _polyDerM * _C.segment(cellIdx * _nNodes, _nNodes).template cast<double>()).maxCoeff();
				//double u_x = (_polyDerM * _C.segment(cell * _nNodes, _nNodes)).maxCoeff();
				double M_ = 2.0 / 3.0 * M2;
				//double hmpf = 2.0 / 9.0 * (3.0 + 10.0 * M2) * M2 * _deltaZ / (_deltaZ + 2.0 * u_x * _deltaZ);

				// reconstruct if cell is troubled, i.e. potential oscillations
				if (abs(trigger1 - _uTilde) > 1e-8 || abs(trigger2 - _u2Tilde) > 1e-8)
				{
					if (abs(_uTilde) > M_ * _deltaZ * _deltaZ || abs(_u2Tilde) > M_ * _deltaZ * _deltaZ)
						return 1.0;
				}
			}
			return 0.0;
		}

		double getCellAverage(int cellIdx) {
			switch (cellIdx)
			{
			case 0: return _pAvg0; case 1: return _pAvg1; case 2: return _pAvg0; default: return 0.0;
			}
		}

	private:

		int _nNodes; //!< Number of polynomial interpolation nodes of DG ansatz
		int _polyDeg; //!< Polynomial degree of DG ansatz
		int _nCells; //!< Number of DG cells/elements
		double _deltaZ;
		Eigen::VectorXd _LGLweights;
		Eigen::MatrixXd _polyDerM;

		std::unique_ptr<IndicatorLimiter> indicatorLimiter;

		double _pAvg0;
		double _pAvg1;
		double _pAvg2;
		double _uTilde;
		double _u2Tilde;

		double _timeRelaxation;
	};
}

#endif // LIBCADET_SMOOTHNESSINDICATOR_HPP_
