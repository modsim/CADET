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
#include "Weno.hpp"
#include "SimulationTypes.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cadet
{

class IParameterProvider;
class IConfigHelper;
struct AdJacobianParams;
struct SimulationTime;
class IModel;

namespace model
{

class IParameterParameterDependence;

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
class AxialConvectionDispersionOperatorBase
{
public:

	AxialConvectionDispersionOperatorBase();
	~AxialConvectionDispersionOperatorBase() CADET_NOEXCEPT;

	void setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT;

	bool configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, unsigned int nCol, unsigned int strideCell);
	bool configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters);
	bool notifyDiscontinuousSectionTransition(double t, unsigned int secIdx);

	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, linalg::BandMatrix& jac);
	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, linalg::BandMatrix& jac);
	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, WithoutParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, WithParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithoutParamSensitivity);

	void multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const;
	void addTimeDerivativeToJacobian(double alpha, linalg::FactorizableBandMatrix& jacDisc);

	inline const active& columnLength() const CADET_NOEXCEPT { return _colLength; }
	inline const active& currentVelocity(double pos) const CADET_NOEXCEPT { return _curVelocity; }
	inline bool forwardFlow() const CADET_NOEXCEPT { return _curVelocity >= 0.0; }

	inline double cellCenter(unsigned int idx) const CADET_NOEXCEPT { return static_cast<double>(_colLength) / _nCol * (idx + 0.5); }
	inline double relativeCoordinate(unsigned int idx) const CADET_NOEXCEPT { return (0.5 + idx) / _nCol; }

	inline unsigned int nComp() const CADET_NOEXCEPT { return _nComp; }
	inline unsigned int nCol() const CADET_NOEXCEPT { return _nCol; }
	inline const Weno& weno() const CADET_NOEXCEPT { return _weno; }

	unsigned int jacobianLowerBandwidth() const CADET_NOEXCEPT;
	unsigned int jacobianUpperBandwidth() const CADET_NOEXCEPT;
	unsigned int jacobianDiscretizedBandwidth() const CADET_NOEXCEPT;
	double inletJacobianFactor() const CADET_NOEXCEPT;

	bool setParameter(const ParameterId& pId, double value);
	bool setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue);
	bool setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& id, double value);

protected:

	template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
	int residualImpl(const IModel& model, double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin);

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
	double* _wenoDerivatives; //!< Holds derivatives of the WENO scheme
	Weno _weno; //!< The WENO scheme implementation
	double _wenoEpsilon; //!< The @f$ \varepsilon @f$ of the WENO scheme (prevents division by zero)

	bool _dispersionCompIndep; //!< Determines whether dispersion is component independent

	IParameterParameterDependence* _dispersionDep;

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
 * This class does not store the Jacobian. It only fills existing matrices given to its residual() functions.
 * It assumes that there is no offset to the inlet in the local state vector and that the firsts cell is placed
 * directly after the inlet DOFs.
 */
class RadialConvectionDispersionOperatorBase
{
public:

	RadialConvectionDispersionOperatorBase();
	~RadialConvectionDispersionOperatorBase() CADET_NOEXCEPT;

	void setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT;

	bool configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, unsigned int nCol, unsigned int strideCell);
	bool configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters);
	bool notifyDiscontinuousSectionTransition(double t, unsigned int secIdx);

	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, linalg::BandMatrix& jac);
	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, linalg::BandMatrix& jac);
	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, WithoutParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, WithParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithoutParamSensitivity);

	void multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const;
	void addTimeDerivativeToJacobian(double alpha, linalg::FactorizableBandMatrix& jacDisc);

	inline const active& columnLength() const CADET_NOEXCEPT { return _colLength; }
	inline const active& innerRadius() const CADET_NOEXCEPT { return _innerRadius; }
	inline const active& outerRadius() const CADET_NOEXCEPT { return _outerRadius; }
	active currentVelocity(double pos) const CADET_NOEXCEPT;
	inline bool forwardFlow() const CADET_NOEXCEPT { return _curVelocity >= 0.0; }

	inline double cellCenter(unsigned int idx) const CADET_NOEXCEPT { return static_cast<double>(_cellCenters[idx]); }
	inline double relativeCoordinate(unsigned int idx) const CADET_NOEXCEPT { return (0.5 + idx) / _nCol; }

	inline unsigned int nComp() const CADET_NOEXCEPT { return _nComp; }
	inline unsigned int nCol() const CADET_NOEXCEPT { return _nCol; }
//	inline const Weno& weno() const CADET_NOEXCEPT { return _weno; }

	unsigned int jacobianLowerBandwidth() const CADET_NOEXCEPT;
	unsigned int jacobianUpperBandwidth() const CADET_NOEXCEPT;
	unsigned int jacobianDiscretizedBandwidth() const CADET_NOEXCEPT;
	double inletJacobianFactor() const CADET_NOEXCEPT;

	bool setParameter(const ParameterId& pId, double value);
	bool setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue);
	bool setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& id, double value);

protected:

	template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
	int residualImpl(const IModel& model, double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin);

	void equidistantCells();

	unsigned int _nComp; //!< Number of components
	unsigned int _nCol; //!< Number of axial cells
	unsigned int _strideCell; //!< Number of elements between the same item in two adjacent cells

	active _colLength; //!< Column length \f$ L \f$
	active _innerRadius; //!< Inner radius
	active _outerRadius; //!< Outer radius

	// Section dependent parameters
	std::vector<active> _colDispersion; //!< Column dispersion (may be section dependent) \f$ D_{\text{rad}} \f$
	std::vector<active> _velocity; //!< Radial velocity (may be section dependent) \f$ v \f$
	active _curVelocity; //!< Current interstitial velocity \f$ u \f$ in this time section
	int _dir; //!< Current flow direction in this time section

	ArrayPool _stencilMemory; //!< Provides memory for the stencil
//	double* _wenoDerivatives; //!< Holds derivatives of the WENO scheme
//	Weno _weno; //!< The WENO scheme implementation
//	double _wenoEpsilon; //!< The @f$ \varepsilon @f$ of the WENO scheme (prevents division by zero)

	bool _dispersionCompIndep; //!< Determines whether dispersion is component independent

	IParameterParameterDependence* _dispersionDep;

	// Grid info
	std::vector<active> _cellCenters;
	std::vector<active> _cellSizes;
	std::vector<active> _cellBounds;

	// Indexer functionality

	// Strides
	inline int strideColCell() const CADET_NOEXCEPT { return static_cast<int>(_strideCell); }
	inline int strideColComp() const CADET_NOEXCEPT { return 1; }

	// Offsets
	inline int offsetC() const CADET_NOEXCEPT { return _nComp; }
};


/**
 * @brief Convection dispersion transport operator
 * @details This class wraps AxialConvectionDispersionOperatorBase or RadialConvectionDispersionOperatorBase
 * and provides all the functionality it does. In addition, the Jacobian is stored and corresponding functions
 * are provided (assembly, factorization, solution, retrieval).
 * 
 * This class assumes that the first cell is offset by the number of components (inlet DOFs) in the global state vector.
 */
template <typename BaseOperator>
class ConvectionDispersionOperator
{
public:

	ConvectionDispersionOperator();
	~ConvectionDispersionOperator() CADET_NOEXCEPT;

	int requiredADdirs() const CADET_NOEXCEPT;

	void setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT;

	bool configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, unsigned int nCol);
	bool configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters);
	bool notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const AdJacobianParams& adJac);

	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, bool wantJac, WithoutParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithoutParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity);

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
	inline active currentVelocity(double pos) const CADET_NOEXCEPT { return _baseOp.currentVelocity(pos); }
	inline bool forwardFlow() const CADET_NOEXCEPT { return _baseOp.forwardFlow(); }
	inline double inletJacobianFactor() const CADET_NOEXCEPT { return _baseOp.inletJacobianFactor(); }

	inline double cellCenter(unsigned int idx) const CADET_NOEXCEPT { return _baseOp.cellCenter(idx); }
	inline double relativeCoordinate(unsigned int idx) const CADET_NOEXCEPT { return _baseOp.relativeCoordinate(idx); }

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

	BaseOperator _baseOp;

	linalg::BandMatrix _jacC; //!< Jacobian
	linalg::FactorizableBandMatrix _jacCdisc; //!< Jacobian with time derivatives from BDF method

	// Indexer functionality

	// Offsets
	inline int offsetC() const CADET_NOEXCEPT { return _baseOp.nComp(); }
};

extern template class ConvectionDispersionOperator<AxialConvectionDispersionOperatorBase>;
extern template class ConvectionDispersionOperator<RadialConvectionDispersionOperatorBase>;

typedef ConvectionDispersionOperator<AxialConvectionDispersionOperatorBase> AxialConvectionDispersionOperator;
typedef ConvectionDispersionOperator<RadialConvectionDispersionOperatorBase> RadialConvectionDispersionOperator;

} // namespace parts
} // namespace model
} // namespace cadet

#endif  // LIBCADET_CONVECTIONDISPERSIONOPERATOR_HPP_
