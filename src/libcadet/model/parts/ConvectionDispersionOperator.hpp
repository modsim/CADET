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
 * Defines the convection dispersion transport operator.
 */

#ifndef LIBCADET_CONVECTIONDISPERSIONOPERATOR_HPP_
#define LIBCADET_CONVECTIONDISPERSIONOPERATOR_HPP_

#include "ParamIdUtil.hpp"
#include "AutoDiff.hpp"
#include "linalg/BandMatrix.hpp"
#include "Memory.hpp"
#include "SimulationTypes.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cadet
{

class IParameterProvider;
struct AdJacobianParams;
struct SimulationTime;

class Weno;
class HighResolutionKoren;

namespace model
{

namespace parts
{

/**
 * @brief Convection dispersion transport operator
 * @details Implements the equation
 * 
 * @f[\begin{align}
	\frac{\partial c_i}{\partial t} &= - u \frac{\partial c_i}{\partial z} + D_{\text{ax},i} \frac{\partial^2 c_i}{\partial z^2} \\
\end{align} @f]
 * with Danckwerts boundary conditions (see @cite Danckwerts1953)
@f[ \begin{align}
u c_{\text{in},i}(t) &= u c_i(t,0) - D_{\text{ax},i} \frac{\partial c_i}{\partial z}(t,0) \\
\frac{\partial c_i}{\partial z}(t,L) &= 0
\end{align} @f]
 * Methods are described in @cite VonLieres2010a (WENO, linear solver), and @cite Puttmann2013, @cite Puttmann2016 (forward sensitivities, AD, band compression)
 * 
 * This class does not store the Jacobian. It only fills existing matrices given to its residual() functions.
 * It assumes that there is no offset to the inlet in the local state vector and that the firsts cell is placed
 * directly after the inlet DOFs.
 */
class ConvectionDispersionOperatorBase
{
public:

	ConvectionDispersionOperatorBase();
	~ConvectionDispersionOperatorBase() CADET_NOEXCEPT;

	void setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT;

	bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int nCol, unsigned int strideCell);
	bool configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters);
	bool notifyDiscontinuousSectionTransition(double t, unsigned int secIdx);

	int residual(double t, unsigned int secIdx, double const* y, double const* yDot, double* res, linalg::BandMatrix& jac);
	int residual(double t, unsigned int secIdx, double const* y, double const* yDot, active* res, linalg::BandMatrix& jac);
	int residual(double t, unsigned int secIdx, double const* y, double const* yDot, double* res, WithoutParamSensitivity);
	int residual(double t, unsigned int secIdx, double const* y, double const* yDot, active* res, WithParamSensitivity);
	int residual(double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithParamSensitivity);
	int residual(double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithoutParamSensitivity);

	void multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const;
	void addTimeDerivativeToJacobian(double alpha, linalg::FactorizableBandMatrix& jacDisc);

	inline const active& columnLength() const CADET_NOEXCEPT { return _colLength; }
	inline const active& crossSectionArea() const CADET_NOEXCEPT { return _crossSection; }
	inline const active& currentVelocity() const CADET_NOEXCEPT { return _curVelocity; }

	inline unsigned int nComp() const CADET_NOEXCEPT { return _nComp; }
	inline unsigned int nCol() const CADET_NOEXCEPT { return _nCol; }
	inline Weno const* weno() const CADET_NOEXCEPT { return _weno; }
	inline HighResolutionKoren const* koren() const CADET_NOEXCEPT { return _koren; }

	unsigned int jacobianLowerBandwidth() const CADET_NOEXCEPT;
	unsigned int jacobianUpperBandwidth() const CADET_NOEXCEPT;
	unsigned int jacobianDiscretizedBandwidth() const CADET_NOEXCEPT;

	bool setParameter(const ParameterId& pId, double value);
	bool setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue);
	bool setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& id, double value);

protected:

	template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
	int residualImpl(double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin);

	template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
	int residualForwardsFlow(double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin);

	template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
	int residualBackwardsFlow(double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin);

	unsigned int _nComp; //!< Number of components
	unsigned int _nCol; //!< Number of axial cells
	unsigned int _strideCell; //!< Number of elements between the same item in two adjacent cells

	active _colLength; //!< Column length \f$ L \f$
	active _crossSection; //!< Cross section area 

	// Section dependent parameters
	std::vector<active> _colDispersion; //!< Column dispersion (may be section dependent) \f$ D_{\text{ax}} \f$
	std::vector<active> _velocity; //!< Interstitial velocity (may be section dependent) \f$ u \f$
	active _curVelocity; //!< Current interstitial velocity \f$ u \f$ in this time section
	int _dir; //!< Current flow direction in this time section

	ArrayPool _stencilMemory; //!< Provides memory for the stencil
	double* _reconstrDerivatives; //!< Holds derivatives of the reconstruction scheme
	Weno* _weno; //!< The WENO scheme implementation
	HighResolutionKoren* _koren; //!< The High Resolution Koren scheme implementation

	bool _dispersionCompIndep; //!< Determines whether dispersion is component independent

	// Indexer functionality

	// Strides
	inline int strideColCell() const CADET_NOEXCEPT { return static_cast<int>(_strideCell); }
	inline int strideColComp() const CADET_NOEXCEPT { return 1; }

	// Offsets
	inline int offsetC() const CADET_NOEXCEPT { return _nComp; }
};


/**
 * @brief Convection dispersion transport operator
 * @details Implements the equation
 * 
 * @f[\begin{align}
	\frac{\partial c_i}{\partial t} &= - u \frac{\partial c_i}{\partial z} + D_{\text{ax},i} \frac{\partial^2 c_i}{\partial z^2} \\
\end{align} @f]
 * with Danckwerts boundary conditions (see @cite Danckwerts1953)
@f[ \begin{align}
u c_{\text{in},i}(t) &= u c_i(t,0) - D_{\text{ax},i} \frac{\partial c_i}{\partial z}(t,0) \\
\frac{\partial c_i}{\partial z}(t,L) &= 0
\end{align} @f]
 * Methods are described in @cite VonLieres2010a (WENO, linear solver), and @cite Puttmann2013, @cite Puttmann2016 (forward sensitivities, AD, band compression)
 * 
 * This class wraps ConvectionDispersionOperatorBase and provides all the functionality it does. In addition,
 * the Jacobian is stored and corresponding functions are provided (assembly, factorization, solution, retrieval).
 * 
 * This class assumes that the first cell is offset by the number of components (inlet DOFs) in the global state vector.
 */
class ConvectionDispersionOperator
{
public:

	ConvectionDispersionOperator();
	~ConvectionDispersionOperator() CADET_NOEXCEPT;

	int requiredADdirs() const CADET_NOEXCEPT;

	void setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT;

	bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int nCol);
	bool configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters);
	bool notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const AdJacobianParams& adJac);

	int residual(double t, unsigned int secIdx, double const* y, double const* yDot, double* res, bool wantJac, WithoutParamSensitivity);
	int residual(double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity);
	int residual(double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithoutParamSensitivity);
	int residual(double t, unsigned int secIdx, double const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity);

	void prepareADvectors(const AdJacobianParams& adJac) const;
	void extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset);

	bool solveTimeDerivativeSystem(const SimulationTime& simTime, double* const rhs);
	void multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const;

	bool assembleAndFactorizeDiscretizedJacobian(double alpha);
	bool solveDiscretizedJacobian(double* rhs) const;

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
	double checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const;
#endif

	inline const active& columnLength() const CADET_NOEXCEPT { return _baseOp.columnLength(); }
	inline const active& crossSectionArea() const CADET_NOEXCEPT { return _baseOp.crossSectionArea(); }
	inline const active& currentVelocity() const CADET_NOEXCEPT { return _baseOp.currentVelocity(); }

	inline linalg::BandMatrix& jacobian() CADET_NOEXCEPT { return _jacC; }
	inline const linalg::BandMatrix& jacobian() const CADET_NOEXCEPT { return _jacC; }

	inline linalg::FactorizableBandMatrix& jacobianDisc() CADET_NOEXCEPT { return _jacCdisc; }
	inline const linalg::FactorizableBandMatrix& jacobianDisc() const CADET_NOEXCEPT { return _jacCdisc; }

	inline bool setParameter(const ParameterId& pId, double value)
	{
		return _baseOp.setParameter(pId, value);
	}
	inline bool setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue)
	{
		return _baseOp.setSensitiveParameter(sensParams, pId, adDirection, adValue);
	}
	inline bool setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& id, double value)
	{
		return _baseOp.setSensitiveParameterValue(sensParams, id, value);
	}

protected:

	void addTimeDerivativeToJacobian(double alpha);
	void assembleDiscretizedJacobian(double alpha);

	ConvectionDispersionOperatorBase _baseOp;

	linalg::BandMatrix _jacC; //!< Jacobian
	linalg::FactorizableBandMatrix _jacCdisc; //!< Jacobian with time derivatives from BDF method

	// Indexer functionality

	// Offsets
	inline int offsetC() const CADET_NOEXCEPT { return _baseOp.nComp(); }
};

} // namespace parts
} // namespace model
} // namespace cadet

#endif  // LIBCADET_CONVECTIONDISPERSIONOPERATOR_HPP_
