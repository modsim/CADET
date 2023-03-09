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

/**
 * @file
 * Defines the multi-channel transport operator.
 */

#ifndef LIBCADET_MULTICHANNELTRANSPORTOPERATOR_HPP_
#define LIBCADET_MULTICHANNELTRANSPORTOPERATOR_HPP_

#include "ParamIdUtil.hpp"
#include "AutoDiff.hpp"
#include "linalg/CompressedSparseMatrix.hpp"
#include "Memory.hpp"
#include "Weno.hpp"
#include "model/ParameterMultiplexing.hpp"
#include "SimulationTypes.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cadet
{

class IParameterProvider;

namespace model
{

namespace parts
{

/**
 * @brief Multi-channel transport operator
 * @details Implements the equation
 *
 * @f[\begin{align}
	\frac{\partial c_i}{\partial t} &= - u \frac{\partial c_i}{\partial z} + D_{\text{ax},i}(\rho) \frac{\partial^2 c_i}{\partial z^2} \\
\end{align} @f]
 * with Danckwerts boundary conditions on the axial boundary (see @cite Danckwerts1953)
@f[ \begin{align}
u c_{\text{in},i}(t) &= u c_i(t,0,\rho) - D_{\text{ax},i}(\rho) \frac{\partial c_i}{\partial z}(t,0,\rho) \\
\frac{\partial c_i}{\partial z}(t,L,\rho) &= 0
\end{align} @f]
 * Methods are described in @cite VonLieres2010a (WENO)
 */
class MultiChannelConvectionDispersionOperator
{
public:

	MultiChannelConvectionDispersionOperator();
	~MultiChannelConvectionDispersionOperator() CADET_NOEXCEPT;

	void setFlowRates(int compartment, const active& in, const active& out) CADET_NOEXCEPT;
	void setFlowRates(active const* in, active const* out) CADET_NOEXCEPT;

	bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int nCol, unsigned int nRad, bool dynamicReactions);
	bool configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters);
	bool notifyDiscontinuousSectionTransition(double t, unsigned int secIdx);

	int residual(double t, unsigned int secIdx, double const* y, double const* yDot, double* res, bool wantJac, WithoutParamSensitivity);
	int residual(double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithoutParamSensitivity);
	int residual(double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity);
	int residual(double t, unsigned int secIdx, double const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity);

	bool solveTimeDerivativeSystem(const SimulationTime& simTime, double* const rhs);
	void multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const;

	bool assembleAndFactorizeDiscretizedJacobian(double alpha);
	bool solveDiscretizedJacobian(double* rhs, double const* weight, double const* init, double outerTol) const;

	bool setParameter(const ParameterId& pId, double value);
	bool setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue);
	bool setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& id, double value);

	inline const active& columnLength() const CADET_NOEXCEPT { return _colLength; }
	inline const active& currentVelocity(int idx) const CADET_NOEXCEPT { return _curVelocity[idx]; }
	inline const active& crossSection(int idx) const CADET_NOEXCEPT { return _crossSections[idx]; }
	inline active const* crossSections() const CADET_NOEXCEPT { return _crossSections.data(); }
	inline bool isCurrentFlowForward(int idx) const CADET_NOEXCEPT { return _curVelocity[idx] >= 0.0; }
	const active& axialDispersion(unsigned int idxSec, int idxRad, int idxComp) const CADET_NOEXCEPT;
	const active& radialDispersion(unsigned int idxSec, int idxRad, int idxComp) const CADET_NOEXCEPT;
	double inletFactor(unsigned int idxSec, int idxRad) const CADET_NOEXCEPT;

	inline linalg::CompressedSparseMatrix& jacobian() CADET_NOEXCEPT { return _jacC; }
	inline const linalg::CompressedSparseMatrix& jacobian() const CADET_NOEXCEPT { return _jacC; }

protected:

	class LinearSolver;
	class GmresSolver;
	template <typename sparse_t> class SparseDirectSolver;
	class DenseDirectSolver;

	friend int matrixMultiplierMultiChannelCDO(void* userData, double const* x, double* z);

	void assembleDiscretizedJacobian(double alpha);

	template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	int residualImpl(double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res);

	void setSparsityPattern();

	void setEquidistantRadialDisc();
	void setEquivolumeRadialDisc();
	void setUserdefinedRadialDisc();
	void updateRadialDisc();

	enum class RadialDiscretizationMode : int
	{
		/**
		 * Equidistant distribution of tubular shell edges
		 */
		Equidistant,

		/**
		 * Volumes of tubular shells are uniform
		 */
		Equivolume,

		/**
		 * Shell edges specified by user
		 */
		UserDefined
	};

	unsigned int _nComp; //!< Number of components
	unsigned int _nCol; //!< Number of axial cells
	unsigned int _nChannel; //!< Number of channels
	bool _hasDynamicReactions; //!< Determines whether the model has dynamic reactions (only relevant for sparsity pattern)

	active _colLength; //!< Column length \f$ L \f$
	std::vector<active> _crossSections; //!< Cross section area of each compartment

	std::vector<active> _axialDispersion; //!< Axial dispersion coefficient \f$ D_{\text{ax}} \f$
	MultiplexMode _axialDispersionMode; //!< Multiplex mode of the axial dispersion
	std::vector<active> _radialDispersion; //!< Radial dispersion coefficient \f$ D_{\rho} \f$
	MultiplexMode _radialDispersionMode; //!< Multiplex mode of the radial dispersion
	std::vector<active> _velocity; //!< Interstitial velocity parameter
	std::vector<active> _curVelocity; //!< Current interstitial velocity \f$ u \f$
	std::vector<int> _dir; //!< Current flow direction 
	bool _singleVelocity; //!< Determines whether only one velocity for all compartments is given

    std::vector<active> _exchangeMatrix; //!< Matrix of exchange coeffs for the linear inter-channel transport

	ArrayPool _stencilMemory; //!< Provides memory for the stencil
	double* _wenoDerivatives; //!< Holds derivatives of the WENO scheme
	Weno _weno; //!< The WENO scheme implementation
	double _wenoEpsilon; //!< The @f$ \varepsilon @f$ of the WENO scheme (prevents division by zero)

	linalg::CompressedSparseMatrix _jacC; //!< Jacobian
	LinearSolver* _linearSolver; //!< Solves linear system with time discretized Jacobian
};

} // namespace parts
} // namespace model
} // namespace cadet

#endif  // LIBCADET_MULTICHANNELTRANSPORTOPERATOR_HPP_
