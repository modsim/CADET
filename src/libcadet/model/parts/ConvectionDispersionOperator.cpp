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

#include "model/parts/ConvectionDispersionOperator.hpp"
#include "cadet/Exceptions.hpp"

#include "Stencil.hpp"
#include "ParamReaderHelper.hpp"
#include "AdUtils.hpp"
#include "SimulationTypes.hpp"
#include "model/parts/AxialConvectionDispersionKernel.hpp"
#include "model/parts/RadialConvectionDispersionKernel.hpp"
#include "model/ParameterDependence.hpp"
#include "SensParamUtil.hpp"
#include "ConfigurationHelper.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include <algorithm>
#include <cmath>

namespace cadet
{

namespace model
{

namespace parts
{

/**
 * @brief Creates an AxialConvectionDispersionOperatorBase
 */
AxialConvectionDispersionOperatorBase::AxialConvectionDispersionOperatorBase() : _stencilMemory(sizeof(active) * Weno::maxStencilSize()), 
	_wenoDerivatives(new double[Weno::maxStencilSize()]), _weno(), _dispersionDep(nullptr)
{
}

AxialConvectionDispersionOperatorBase::~AxialConvectionDispersionOperatorBase() CADET_NOEXCEPT
{
	if (_dispersionDep)
		delete _dispersionDep;

	delete[] _wenoDerivatives;
}

/**
 * @brief Reads parameters and allocates memory
 * @details Has to be called once before the operator is used.
 * @param [in] paramProvider Parameter provider for reading parameters
 * @param [in] nComp Number of components
 * @param [in] nCol Number of axial cells
 * @return @c true if configuration went fine, @c false otherwise
 */
bool AxialConvectionDispersionOperatorBase::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, unsigned int nCol, unsigned int strideCell)
{
	_nComp = nComp;
	_nCol = nCol;
	_strideCell = strideCell;

	if (paramProvider.exists("COL_DISPERSION_DEP"))
	{
		const std::string paramDepName = paramProvider.getString("COL_DISPERSION_DEP");
		_dispersionDep = helper.createParameterParameterDependence(paramDepName);
		if (!_dispersionDep)
			throw InvalidParameterException("Unknown parameter dependence " + paramDepName + " in COL_DISPERSION_DEP");

		_dispersionDep->configureModelDiscretization(paramProvider);
	}
	else
		_dispersionDep = helper.createParameterParameterDependence("IDENTITY");

	paramProvider.pushScope("discretization");

	// Read WENO settings and apply them
	paramProvider.pushScope("weno");
	_weno.order(paramProvider.getInt("WENO_ORDER"));
	_weno.boundaryTreatment(paramProvider.getInt("BOUNDARY_MODEL"));
	_wenoEpsilon = paramProvider.getDouble("WENO_EPS");
	paramProvider.popScope();

	paramProvider.popScope();

	return true;
}

/**
 * @brief Reads model parameters
 * @details Only reads parameters that do not affect model structure (e.g., discretization).
 * @param [in] unitOpIdx Unit operation id of the owning unit operation model
 * @param [in] paramProvider Parameter provider for reading parameters
 * @param [out] parameters Map in which local parameters are inserted
 * @return @c true if configuration went fine, @c false otherwise
 */
bool AxialConvectionDispersionOperatorBase::configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters)
{
	// Read geometry parameters
	_colLength = paramProvider.getDouble("COL_LENGTH");

	// Read cross section area or set to -1
	_crossSection = -1.0;
	if (paramProvider.exists("CROSS_SECTION_AREA"))
	{
		_crossSection = paramProvider.getDouble("CROSS_SECTION_AREA");
	}

	// Read section dependent parameters (transport)

	// Read VELOCITY
	_velocity.clear();
	if (paramProvider.exists("VELOCITY"))
	{
		readScalarParameterOrArray(_velocity, paramProvider, "VELOCITY", 1);
	}
	_dir = 1;

	readScalarParameterOrArray(_colDispersion, paramProvider, "COL_DISPERSION", 1);
	if (paramProvider.exists("COL_DISPERSION_MULTIPLEX"))
	{
		const int mode = paramProvider.getInt("COL_DISPERSION_MULTIPLEX");
		if (mode == 0)
			// Comp-indep, sec-indep
			_dispersionCompIndep = true;
		else if (mode == 1)
			// Comp-dep, sec-indep
			_dispersionCompIndep = false;
		else if (mode == 2)
			// Comp-indep, sec-dep
			_dispersionCompIndep = true;
		else if (mode == 3)
			// Comp-dep, sec-dep
			_dispersionCompIndep = false;

		if (!_dispersionCompIndep && (_colDispersion.size() % _nComp != 0))
			throw InvalidParameterException("Number of elements in field COL_DISPERSION is not a positive multiple of NCOMP (" + std::to_string(_nComp) + ")");
		if ((mode == 0) && (_colDispersion.size() != 1))
			throw InvalidParameterException("Number of elements in field COL_DISPERSION inconsistent with COL_DISPERSION_MULTIPLEX (should be 1)");
		if ((mode == 1) && (_colDispersion.size() != _nComp))
			throw InvalidParameterException("Number of elements in field COL_DISPERSION inconsistent with COL_DISPERSION_MULTIPLEX (should be " + std::to_string(_nComp) + ")");
	}
	else
	{
		// Infer component dependence of COL_DISPERSION:
		//   size not divisible by NCOMP -> component independent
		_dispersionCompIndep = ((_colDispersion.size() % _nComp) != 0);
	}

	// Expand _colDispersion to make it component dependent
	if (_dispersionCompIndep)
	{
		std::vector<active> expanded(_colDispersion.size() * _nComp);
		for (std::size_t i = 0; i < _colDispersion.size(); ++i)
			std::fill(expanded.begin() + i * _nComp, expanded.begin() + (i + 1) * _nComp, _colDispersion[i]);

		_colDispersion = std::move(expanded);
	}

	if (_dispersionDep)
	{
		if (!_dispersionDep->configure(paramProvider, unitOpIdx, ParTypeIndep, BoundStateIndep, "COL_DISPERSION_DEP"))
			throw InvalidParameterException("Failed to configure dispersion parameter dependency (COL_DISPERSION_DEP)");
	}

	if (_velocity.empty() && (_crossSection <= 0.0))
	{
		throw InvalidParameterException("At least one of CROSS_SECTION_AREA and VELOCITY has to be set");
	}

	// Add parameters to map
	if (_dispersionCompIndep)
	{
		if (_colDispersion.size() > _nComp)
		{
			// Register only the first item in each section
			for (std::size_t i = 0; i < _colDispersion.size() / _nComp; ++i)
				parameters[makeParamId(hashString("COL_DISPERSION"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, i)] = &_colDispersion[i * _nComp];
		}
		else
		{
			// We have only one parameter
			parameters[makeParamId(hashString("COL_DISPERSION"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_colDispersion[0];
		}
	}
	else
		registerParam2DArray(parameters, _colDispersion, [=](bool multi, unsigned int sec, unsigned int comp) { return makeParamId(hashString("COL_DISPERSION"), unitOpIdx, comp, ParTypeIndep, BoundStateIndep, ReactionIndep, multi ? sec : SectionIndep); }, _nComp);

	registerScalarSectionDependentParam(hashString("VELOCITY"), parameters, _velocity, unitOpIdx, ParTypeIndep);
	parameters[makeParamId(hashString("COL_LENGTH"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_colLength;
	parameters[makeParamId(hashString("CROSS_SECTION_AREA"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_crossSection;

	return true;
}

/**
 * @brief Notifies the operator that a discontinuous section transition is in progress
 * @details In addition to changing flow direction internally, if necessary, the function returns whether
 *          the flow direction has changed.
 * @param [in] t Current time point
 * @param [in] secIdx Index of the new section that is about to be integrated
 * @return @c true if flow direction has changed, otherwise @c false
 */
bool AxialConvectionDispersionOperatorBase::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx)
{
	// setFlowRates() was called before, so _curVelocity has direction dirOld
	const int dirOld = _dir;

	if (_crossSection <= 0.0)
	{
		// Use the provided _velocity (direction is also set), only update _dir
		_curVelocity = getSectionDependentScalar(_velocity, secIdx);
		_dir = (_curVelocity >= 0.0) ? 1 : -1;
	}
	else if (!_velocity.empty())
	{
		// Use network flow rate but take direction from _velocity
		_dir = (getSectionDependentScalar(_velocity, secIdx) >= 0.0) ? 1 : -1;

		// _curVelocity has correct magnitude but previous direction, so flip it if necessary
		if (dirOld * _dir < 0)
			_curVelocity *= -1.0;
	}

	// Remaining case: _velocity is empty and _crossSection <= 0.0
	// _curVelocity is goverend by network flow rate provided in setFlowRates().
	// Direction never changes (always forward, that is, _dir = 1)-
	// No action required.

	// Detect change in flow direction
	return (dirOld * _dir < 0);
}

/**
 * @brief Sets the flow rates for the current time section
 * @details The flow rates may change due to valve switches.
 * @param [in] in Total volumetric inlet flow rate
 * @param [in] out Total volumetric outlet flow rate
 * @param [in] colPorosity Porosity used for computing interstitial velocity from volumetric flow rate
 */
void AxialConvectionDispersionOperatorBase::setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT
{
	// If we have cross section area, interstitial velocity is given by network flow rates
	if (_crossSection > 0.0)
		_curVelocity = _dir * in / (_crossSection * colPorosity);
}

/**
 * @brief Computes the residual of the transport equations
 * @param [in] model Model that owns the operator
 * @param [in] t Current time point
 * @param [in] secIdx Index of the current section
 * @param [in] y Pointer to unit operation's state vector
 * @param [in] yDot Pointer to unit operation's time derivative state vector
 * @param [out] res Pointer to unit operation's residual vector
 * @param [in] jac Matrix that holds the Jacobian
 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
 */
int AxialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, linalg::BandMatrix& jac)
{
	// Reset Jacobian
	jac.setAll(0.0);

	return residualImpl<double, double, double, linalg::BandMatrix::RowIterator, true>(model, t, secIdx, y, yDot, res, jac.row(0));
}

int AxialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, WithoutParamSensitivity)
{
	return residualImpl<double, double, double, linalg::BandMatrix::RowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandMatrix::RowIterator());
}

int AxialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithoutParamSensitivity)
{
	return residualImpl<active, active, double, linalg::BandMatrix::RowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandMatrix::RowIterator());
}

int AxialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, linalg::BandMatrix& jac)
{
	// Reset Jacobian
	jac.setAll(0.0);

	return residualImpl<double, active, active, linalg::BandMatrix::RowIterator, true>(model, t, secIdx, y, yDot, res, jac.row(0));
}

int AxialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, WithParamSensitivity)
{
	return residualImpl<double, active, active, linalg::BandMatrix::RowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandMatrix::RowIterator());
}

int AxialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithParamSensitivity)
{
	return residualImpl<active, active, active, linalg::BandMatrix::RowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandMatrix::RowIterator());
}

template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
int AxialConvectionDispersionOperatorBase::residualImpl(const IModel& model, double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin)
{
	const ParamType u = static_cast<ParamType>(_curVelocity);
	active const* const d_c = getSectionDependentSlice(_colDispersion, _nComp, secIdx);
	const ParamType h = static_cast<ParamType>(_colLength) / static_cast<double>(_nCol);
//	const int strideCell = strideColCell();

	convdisp::AxialFlowParameters<ParamType> fp{
		u,
		d_c,
		h,
		_wenoDerivatives,
		&_weno,
		&_stencilMemory,
		_wenoEpsilon,
		strideColCell(),
		_nComp,
		_nCol,
		0u,
		_nComp,
		_dispersionDep,
		model
	};

	return convdisp::residualKernelAxial<StateType, ResidualType, ParamType, RowIteratorType, wantJac>(SimulationTime{t, secIdx}, y, yDot, res, jacBegin, fp);
}

/**
 * @brief Multiplies the time derivative Jacobian @f$ \frac{\partial F}{\partial \dot{y}}\left(t, y, \dot{y}\right) @f$ with a given vector
 * @details The operation @f$ z = \frac{\partial F}{\partial \dot{y}} x @f$ is performed.
 *          The matrix-vector multiplication is performed matrix-free (i.e., no matrix is explicitly formed).
 *          
 *          Note that this function only performs multiplication with the Jacobian of the (axial) transport equations.
 *          The input vectors are assumed to point to the beginning (including inlet DOFs) of the respective unit operation's arrays.
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in] sDot Vector @f$ x @f$ that is transformed by the Jacobian @f$ \frac{\partial F}{\partial \dot{y}} @f$
 * @param [out] ret Vector @f$ z @f$ which stores the result of the operation
 */
void AxialConvectionDispersionOperatorBase::multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const
{
	double* localRet = ret + offsetC();
	double const* localSdot = sDot + offsetC();
	const int gapCell = strideColCell() - static_cast<int>(_nComp) * strideColComp();

	for (unsigned int i = 0; i < _nCol; ++i, localRet += gapCell, localSdot += gapCell)
	{
		for (unsigned int j = 0; j < _nComp; ++j, ++localRet, ++localSdot)
		{
			*localRet = (*localSdot);
		}
	}
}

/**
 * @brief Adds the derivatives with respect to @f$ \dot{y} @f$ of @f$ F(t, y, \dot{y}) @f$ to the Jacobian
 * @details This functions computes 
 *          @f[ \begin{align*} \text{_jacCdisc} = \text{_jacCdisc} + \alpha \frac{\partial F}{\partial \dot{y}}. \end{align*} @f]
 *          The factor @f$ \alpha @f$ is useful when constructing the linear system in the time integration process.
 * @param [in] alpha Factor in front of @f$ \frac{\partial F}{\partial \dot{y}} @f$
 */
void AxialConvectionDispersionOperatorBase::addTimeDerivativeToJacobian(double alpha, linalg::FactorizableBandMatrix& jacDisc)
{
	const int gapCell = strideColCell() - static_cast<int>(_nComp) * strideColComp();
	linalg::FactorizableBandMatrix::RowIterator jac = jacDisc.row(0);
	for (unsigned int i = 0; i < _nCol; ++i, jac += gapCell)
	{
		for (unsigned int j = 0; j < _nComp; ++j, ++jac)
		{
			// Add time derivative to main diagonal
			jac[0] += alpha;
		}
	}
}

unsigned int AxialConvectionDispersionOperatorBase::jacobianLowerBandwidth() const CADET_NOEXCEPT
{
	// Note that we have to increase the lower bandwidth by 1 because the WENO stencil is applied to the
	// right cell face (lower + 1 + upper) and to the left cell face (shift the stencil by -1 because influx of cell i
	// is outflux of cell i-1)
	// We also have to make sure that there's at least one sub and super diagonal for the dispersion term
	return std::max(_weno.lowerBandwidth() + 1u, 1u) * strideColCell();
}

unsigned int AxialConvectionDispersionOperatorBase::jacobianUpperBandwidth() const CADET_NOEXCEPT
{
	// We have to make sure that there's at least one sub and super diagonal for the dispersion term
	return std::max(_weno.upperBandwidth(), 1u) * strideColCell();
}

unsigned int AxialConvectionDispersionOperatorBase::jacobianDiscretizedBandwidth() const CADET_NOEXCEPT
{
	// When flow direction is changed, the bandwidths of the Jacobian swap.
	// Hence, we have to reserve memory such that the swapped Jacobian can fit into the matrix.
	return std::max(jacobianLowerBandwidth(), jacobianUpperBandwidth());
}

double AxialConvectionDispersionOperatorBase::inletJacobianFactor() const CADET_NOEXCEPT
{
	const double h = static_cast<double>(_colLength) / static_cast<double>(_nCol);
	const double u = static_cast<double>(_curVelocity);
	return u / h;
}

bool AxialConvectionDispersionOperatorBase::setParameter(const ParameterId& pId, double value)
{
	// Check if parameter is in parameter dependence of column dispersion coefficient
	if (_dispersionDep)
	{
		if (_dispersionDep->hasParameter(pId))
		{
			_dispersionDep->setParameter(pId, value);
			return true;
		}
	}

	// We only need to do something if COL_DISPERSION is component independent
	if (!_dispersionCompIndep)
		return false;

	if ((pId.name != hashString("COL_DISPERSION")) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep) || (pId.particleType != ParTypeIndep))
		return false;

	if (_colDispersion.size() > _nComp)
	{
		// Section dependent
		if (pId.section == SectionIndep)
			return false;

		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[pId.section * _nComp + i].setValue(value);
	}
	else
	{
		// Section independent
		if (pId.section != SectionIndep)
			return false;

		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[i].setValue(value);
	}

	return true;
}

bool AxialConvectionDispersionOperatorBase::setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& pId, double value)
{
	// Check if parameter is in parameter dependence of column dispersion coefficient
	if (_dispersionDep)
	{
		active* const param = _dispersionDep->getParameter(pId);
		if (param)
		{
			param->setValue(value);
			return true;
		}
	}

	// We only need to do something if COL_DISPERSION is component independent
	if (!_dispersionCompIndep)
		return false;

	if ((pId.name != hashString("COL_DISPERSION")) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep) || (pId.particleType != ParTypeIndep))
		return false;

	if (_colDispersion.size() > _nComp)
	{
		// Section dependent
		if (pId.section == SectionIndep)
			return false;

		if (!contains(sensParams, &_colDispersion[pId.section * _nComp]))
			return false;

		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[pId.section * _nComp + i].setValue(value);
	}
	else
	{
		// Section independent
		if (pId.section != SectionIndep)
			return false;

		if (!contains(sensParams, &_colDispersion[0]))
			return false;

		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[i].setValue(value);
	}

	return true;
}

bool AxialConvectionDispersionOperatorBase::setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue)
{
	// Check if parameter is in parameter dependence of column dispersion coefficient
	if (_dispersionDep)
	{
		active* const param = _dispersionDep->getParameter(pId);
		if (param)
		{
			param->setADValue(adDirection, adValue);
			return true;
		}
	}

	// We only need to do something if COL_DISPERSION is component independent
	if (!_dispersionCompIndep)
		return false;

	if ((pId.name != hashString("COL_DISPERSION")) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep) || (pId.particleType != ParTypeIndep))
		return false;

	if (_colDispersion.size() > _nComp)
	{
		// Section dependent
		if (pId.section == SectionIndep)
			return false;

		sensParams.insert(&_colDispersion[pId.section * _nComp]);
		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[pId.section * _nComp + i].setADValue(adDirection, adValue);
	}
	else
	{
		// Section independent
		if (pId.section != SectionIndep)
			return false;

		sensParams.insert(&_colDispersion[0]);
		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[i].setADValue(adDirection, adValue);
	}

	return true;
}




/**
 * @brief Creates a RadialConvectionDispersionOperatorBase
 */
RadialConvectionDispersionOperatorBase::RadialConvectionDispersionOperatorBase() : _stencilMemory(sizeof(active) * Weno::maxStencilSize()), _dispersionDep(nullptr)
{
}

RadialConvectionDispersionOperatorBase::~RadialConvectionDispersionOperatorBase() CADET_NOEXCEPT
{
	if (_dispersionDep)
		delete _dispersionDep;
}

/**
 * @brief Reads parameters and allocates memory
 * @details Has to be called once before the operator is used.
 * @param [in] paramProvider Parameter provider for reading parameters
 * @param [in] nComp Number of components
 * @param [in] nCol Number of axial cells
 * @return @c true if configuration went fine, @c false otherwise
 */
bool RadialConvectionDispersionOperatorBase::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, unsigned int nCol, unsigned int strideCell)
{
	_nComp = nComp;
	_nCol = nCol;
	_strideCell = strideCell;

	if (paramProvider.exists("COL_DISPERSION_DEP"))
	{
		const std::string paramDepName = paramProvider.getString("COL_DISPERSION_DEP");
		_dispersionDep = helper.createParameterParameterDependence(paramDepName);
		if (!_dispersionDep)
			throw InvalidParameterException("Unknown parameter dependence " + paramDepName + " in COL_DISPERSION_DEP");

		_dispersionDep->configureModelDiscretization(paramProvider);
	}
	else
		_dispersionDep = helper.createParameterParameterDependence("IDENTITY");

	paramProvider.pushScope("discretization");

	// Read WENO settings and apply them
/*
	paramProvider.pushScope("weno");
	_weno.order(paramProvider.getInt("WENO_ORDER"));
	_weno.boundaryTreatment(paramProvider.getInt("BOUNDARY_MODEL"));
	_wenoEpsilon = paramProvider.getDouble("WENO_EPS");
	paramProvider.popScope();
*/

	paramProvider.popScope();

	return true;
}

/**
 * @brief Reads model parameters
 * @details Only reads parameters that do not affect model structure (e.g., discretization).
 * @param [in] unitOpIdx Unit operation id of the owning unit operation model
 * @param [in] paramProvider Parameter provider for reading parameters
 * @param [out] parameters Map in which local parameters are inserted
 * @return @c true if configuration went fine, @c false otherwise
 */
bool RadialConvectionDispersionOperatorBase::configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters)
{
	// Read geometry parameters
	_innerRadius = paramProvider.getDouble("COL_RADIUS_INNER");
	_outerRadius = paramProvider.getDouble("COL_RADIUS_OUTER");

	// Read length or set to -1
	_colLength = -1.0;
	if (paramProvider.exists("COL_LENGTH"))
	{
		_colLength = paramProvider.getDouble("COL_LENGTH");
	}

	// Read section dependent parameters (transport)

	// Read VELOCITY
	_velocity.clear();
	if (paramProvider.exists("VELOCITY"))
	{
		readScalarParameterOrArray(_velocity, paramProvider, "VELOCITY", 1);
	}
	_dir = 1;

	readScalarParameterOrArray(_colDispersion, paramProvider, "COL_DISPERSION", 1);
	if (paramProvider.exists("COL_DISPERSION_MULTIPLEX"))
	{
		const int mode = paramProvider.getInt("COL_DISPERSION_MULTIPLEX");
		if (mode == 0)
			// Comp-indep, sec-indep
			_dispersionCompIndep = true;
		else if (mode == 1)
			// Comp-dep, sec-indep
			_dispersionCompIndep = false;
		else if (mode == 2)
			// Comp-indep, sec-dep
			_dispersionCompIndep = true;
		else if (mode == 3)
			// Comp-dep, sec-dep
			_dispersionCompIndep = false;

		if (!_dispersionCompIndep && (_colDispersion.size() % _nComp != 0))
			throw InvalidParameterException("Number of elements in field COL_DISPERSION is not a positive multiple of NCOMP (" + std::to_string(_nComp) + ")");
		if ((mode == 0) && (_colDispersion.size() != 1))
			throw InvalidParameterException("Number of elements in field COL_DISPERSION inconsistent with COL_DISPERSION_MULTIPLEX (should be 1)");
		if ((mode == 1) && (_colDispersion.size() != _nComp))
			throw InvalidParameterException("Number of elements in field COL_DISPERSION inconsistent with COL_DISPERSION_MULTIPLEX (should be " + std::to_string(_nComp) + ")");
	}
	else
	{
		// Infer component dependence of COL_DISPERSION:
		//   size not divisible by NCOMP -> component independent
		_dispersionCompIndep = ((_colDispersion.size() % _nComp) != 0);
	}

	// Expand _colDispersion to make it component dependent
	if (_dispersionCompIndep)
	{
		std::vector<active> expanded(_colDispersion.size() * _nComp);
		for (std::size_t i = 0; i < _colDispersion.size(); ++i)
			std::fill(expanded.begin() + i * _nComp, expanded.begin() + (i + 1) * _nComp, _colDispersion[i]);

		_colDispersion = std::move(expanded);
	}

	if (_dispersionDep)
	{
		if (!_dispersionDep->configure(paramProvider, unitOpIdx, ParTypeIndep, BoundStateIndep, "COL_DISPERSION_DEP"))
			throw InvalidParameterException("Failed to configure dispersion parameter dependency (COL_DISPERSION_DEP)");
	}

	if (_velocity.empty() && (_colLength <= 0.0))
	{
		throw InvalidParameterException("At least one of COL_LENGTH and VELOCITY has to be set");
	}

	// Add parameters to map
	if (_dispersionCompIndep)
	{
		if (_colDispersion.size() > _nComp)
		{
			// Register only the first item in each section
			for (std::size_t i = 0; i < _colDispersion.size() / _nComp; ++i)
				parameters[makeParamId(hashString("COL_DISPERSION"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, i)] = &_colDispersion[i * _nComp];
		}
		else
		{
			// We have only one parameter
			parameters[makeParamId(hashString("COL_DISPERSION"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_colDispersion[0];
		}
	}
	else
		registerParam2DArray(parameters, _colDispersion, [=](bool multi, unsigned int sec, unsigned int comp) { return makeParamId(hashString("COL_DISPERSION"), unitOpIdx, comp, ParTypeIndep, BoundStateIndep, ReactionIndep, multi ? sec : SectionIndep); }, _nComp);

	registerScalarSectionDependentParam(hashString("VELOCITY"), parameters, _velocity, unitOpIdx, ParTypeIndep);
	parameters[makeParamId(hashString("COL_LENGTH"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_colLength;
	parameters[makeParamId(hashString("COL_RADIUS_INNER"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_innerRadius;
	parameters[makeParamId(hashString("COL_RADIUS_OUTER"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_outerRadius;

	equidistantCells();

	return true;
}

/**
 * @brief Notifies the operator that a discontinuous section transition is in progress
 * @details In addition to changing flow direction internally, if necessary, the function returns whether
 *          the flow direction has changed.
 * @param [in] t Current time point
 * @param [in] secIdx Index of the new section that is about to be integrated
 * @return @c true if flow direction has changed, otherwise @c false
 */
bool RadialConvectionDispersionOperatorBase::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx)
{
	// setFlowRates() was called before, so _curVelocity has direction dirOld
	const int dirOld = _dir;

	if (_colLength <= 0.0)
	{
		// Use the provided _velocity (direction is also set), only update _dir
		_curVelocity = getSectionDependentScalar(_velocity, secIdx);
		_dir = (_curVelocity >= 0.0) ? 1 : -1;
	}
	else if (!_velocity.empty())
	{
		// Use network flow rate but take direction from _velocity
		_dir = (getSectionDependentScalar(_velocity, secIdx) >= 0.0) ? 1 : -1;

		// _curVelocity has correct magnitude but previous direction, so flip it if necessary
		if (dirOld * _dir < 0)
			_curVelocity *= -1.0;
	}

	// Remaining case: _velocity is empty and _crossSection <= 0.0
	// _curVelocity is goverend by network flow rate provided in setFlowRates().
	// Direction never changes (always forward, that is, _dir = 1)-
	// No action required.

	// Detect change in flow direction
	return (dirOld * _dir < 0);
}

/**
 * @brief Sets the flow rates for the current time section
 * @details The flow rates may change due to valve switches.
 * @param [in] in Total volumetric inlet flow rate
 * @param [in] out Total volumetric outlet flow rate
 * @param [in] colPorosity Porosity used for computing interstitial velocity from volumetric flow rate
 */
void RadialConvectionDispersionOperatorBase::setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT
{
	const double pi = 3.1415926535897932384626434;

	// If we have cross section area, interstitial velocity is given by network flow rates
	if (_colLength > 0.0)
		_curVelocity = _dir * in / (2.0 * pi * _colLength * colPorosity);
}

active RadialConvectionDispersionOperatorBase::currentVelocity(double pos) const CADET_NOEXCEPT
{
	const active radius = pos * (_outerRadius - _innerRadius) + _innerRadius;
	return _curVelocity / radius;
}

/**
 * @brief Computes the residual of the transport equations
 * @param [in] model Model that owns the operator
 * @param [in] t Current time point
 * @param [in] secIdx Index of the current section
 * @param [in] y Pointer to unit operation's state vector
 * @param [in] yDot Pointer to unit operation's time derivative state vector
 * @param [out] res Pointer to unit operation's residual vector
 * @param [in] jac Matrix that holds the Jacobian
 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
 */
int RadialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, linalg::BandMatrix& jac)
{
	// Reset Jacobian
	jac.setAll(0.0);

	return residualImpl<double, double, double, linalg::BandMatrix::RowIterator, true>(model, t, secIdx, y, yDot, res, jac.row(0));
}

int RadialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, WithoutParamSensitivity)
{
	return residualImpl<double, double, double, linalg::BandMatrix::RowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandMatrix::RowIterator());
}

int RadialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithoutParamSensitivity)
{
	return residualImpl<active, active, double, linalg::BandMatrix::RowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandMatrix::RowIterator());
}

int RadialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, linalg::BandMatrix& jac)
{
	// Reset Jacobian
	jac.setAll(0.0);

	return residualImpl<double, active, active, linalg::BandMatrix::RowIterator, true>(model, t, secIdx, y, yDot, res, jac.row(0));
}

int RadialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, WithParamSensitivity)
{
	return residualImpl<double, active, active, linalg::BandMatrix::RowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandMatrix::RowIterator());
}

int RadialConvectionDispersionOperatorBase::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithParamSensitivity)
{
	return residualImpl<active, active, active, linalg::BandMatrix::RowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandMatrix::RowIterator());
}

template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
int RadialConvectionDispersionOperatorBase::residualImpl(const IModel& model, double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin)
{
	const ParamType u = static_cast<ParamType>(_curVelocity);
	active const* const d_rad = getSectionDependentSlice(_colDispersion, _nComp, secIdx);

	convdisp::RadialFlowParameters<ParamType> fp{
		u,
		d_rad,
		_cellCenters.data(),
		_cellSizes.data(),
		_cellBounds.data(),
		&_stencilMemory,
		strideColCell(),
		_nComp,
		_nCol,
		0u,
		_nComp,
		_dispersionDep,
		model
	};

	return convdisp::residualKernelRadial<StateType, ResidualType, ParamType, RowIteratorType, wantJac>(SimulationTime{t, secIdx}, y, yDot, res, jacBegin, fp);
}

/**
 * @brief Multiplies the time derivative Jacobian @f$ \frac{\partial F}{\partial \dot{y}}\left(t, y, \dot{y}\right) @f$ with a given vector
 * @details The operation @f$ z = \frac{\partial F}{\partial \dot{y}} x @f$ is performed.
 *          The matrix-vector multiplication is performed matrix-free (i.e., no matrix is explicitly formed).
 *          
 *          Note that this function only performs multiplication with the Jacobian of the (axial) transport equations.
 *          The input vectors are assumed to point to the beginning (including inlet DOFs) of the respective unit operation's arrays.
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in] sDot Vector @f$ x @f$ that is transformed by the Jacobian @f$ \frac{\partial F}{\partial \dot{y}} @f$
 * @param [out] ret Vector @f$ z @f$ which stores the result of the operation
 */
void RadialConvectionDispersionOperatorBase::multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const
{
	double* localRet = ret + offsetC();
	double const* localSdot = sDot + offsetC();
	const int gapCell = strideColCell() - static_cast<int>(_nComp) * strideColComp();

	for (unsigned int i = 0; i < _nCol; ++i, localRet += gapCell, localSdot += gapCell)
	{
		for (unsigned int j = 0; j < _nComp; ++j, ++localRet, ++localSdot)
		{
			*localRet = (*localSdot);
		}
	}
}

/**
 * @brief Adds the derivatives with respect to @f$ \dot{y} @f$ of @f$ F(t, y, \dot{y}) @f$ to the Jacobian
 * @details This functions computes 
 *          @f[ \begin{align*} \text{_jacCdisc} = \text{_jacCdisc} + \alpha \frac{\partial F}{\partial \dot{y}}. \end{align*} @f]
 *          The factor @f$ \alpha @f$ is useful when constructing the linear system in the time integration process.
 * @param [in] alpha Factor in front of @f$ \frac{\partial F}{\partial \dot{y}} @f$
 */
void RadialConvectionDispersionOperatorBase::addTimeDerivativeToJacobian(double alpha, linalg::FactorizableBandMatrix& jacDisc)
{
	const int gapCell = strideColCell() - static_cast<int>(_nComp) * strideColComp();
	linalg::FactorizableBandMatrix::RowIterator jac = jacDisc.row(0);
	for (unsigned int i = 0; i < _nCol; ++i, jac += gapCell)
	{
		for (unsigned int j = 0; j < _nComp; ++j, ++jac)
		{
			// Add time derivative to main diagonal
			jac[0] += alpha;
		}
	}
}

unsigned int RadialConvectionDispersionOperatorBase::jacobianLowerBandwidth() const CADET_NOEXCEPT
{
	// Note that we have to increase the lower bandwidth by 1 because the WENO stencil is applied to the
	// right cell face (lower + 1 + upper) and to the left cell face (shift the stencil by -1 because influx of cell i
	// is outflux of cell i-1)
	// We also have to make sure that there's at least one sub and super diagonal for the dispersion term
//	return std::max(_weno.lowerBandwidth() + 1u, 1u) * strideColCell();
	return strideColCell();
}

unsigned int RadialConvectionDispersionOperatorBase::jacobianUpperBandwidth() const CADET_NOEXCEPT
{
	// We have to make sure that there's at least one sub and super diagonal for the dispersion term
//	return std::max(_weno.upperBandwidth(), 1u) * strideColCell();
	return strideColCell();
}

unsigned int RadialConvectionDispersionOperatorBase::jacobianDiscretizedBandwidth() const CADET_NOEXCEPT
{
	// When flow direction is changed, the bandwidths of the Jacobian swap.
	// Hence, we have to reserve memory such that the swapped Jacobian can fit into the matrix.
	return std::max(jacobianLowerBandwidth(), jacobianUpperBandwidth());
}

double RadialConvectionDispersionOperatorBase::inletJacobianFactor() const CADET_NOEXCEPT
{
	const double denom = static_cast<double>(_cellCenters[0]) * static_cast<double>(_cellSizes[0]);
	const double u = static_cast<double>(_curVelocity);
	return u / denom;
}

bool RadialConvectionDispersionOperatorBase::setParameter(const ParameterId& pId, double value)
{
	// Check if parameter is in parameter dependence of column dispersion coefficient
	if (_dispersionDep)
	{
		if (_dispersionDep->hasParameter(pId))
		{
			_dispersionDep->setParameter(pId, value);
			return true;
		}
	}

	// Check whether column radius has changed and update discretization if necessary
	if (pId.name == hashString("COL_RADIUS_INNER"))
	{
		_innerRadius.setValue(value);
		equidistantCells();
		return true;
	}

	if (pId.name == hashString("COL_RADIUS_OUTER"))
	{
		_outerRadius.setValue(value);
		equidistantCells();
		return true;
	}

	// We only need to do something if COL_DISPERSION is component independent
	if (!_dispersionCompIndep)
		return false;

	if ((pId.name != hashString("COL_DISPERSION")) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep) || (pId.particleType != ParTypeIndep))
		return false;

	if (_colDispersion.size() > _nComp)
	{
		// Section dependent
		if (pId.section == SectionIndep)
			return false;

		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[pId.section * _nComp + i].setValue(value);
	}
	else
	{
		// Section independent
		if (pId.section != SectionIndep)
			return false;

		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[i].setValue(value);
	}

	return true;
}

bool RadialConvectionDispersionOperatorBase::setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& pId, double value)
{
	// Check if parameter is in parameter dependence of column dispersion coefficient
	if (_dispersionDep)
	{
		active* const param = _dispersionDep->getParameter(pId);
		if (param)
		{
			param->setValue(value);
			return true;
		}
	}

	// Check whether column radius has changed and update discretization if necessary
	if (pId.name == hashString("COL_RADIUS_INNER"))
	{
		_innerRadius.setValue(value);
		equidistantCells();
		return true;
	}

	if (pId.name == hashString("COL_RADIUS_OUTER"))
	{
		_outerRadius.setValue(value);
		equidistantCells();
		return true;
	}

	// We only need to do something if COL_DISPERSION is component independent
	if (!_dispersionCompIndep)
		return false;

	if ((pId.name != hashString("COL_DISPERSION")) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep) || (pId.particleType != ParTypeIndep))
		return false;

	if (_colDispersion.size() > _nComp)
	{
		// Section dependent
		if (pId.section == SectionIndep)
			return false;

		if (!contains(sensParams, &_colDispersion[pId.section * _nComp]))
			return false;

		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[pId.section * _nComp + i].setValue(value);
	}
	else
	{
		// Section independent
		if (pId.section != SectionIndep)
			return false;

		if (!contains(sensParams, &_colDispersion[0]))
			return false;

		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[i].setValue(value);
	}

	return true;
}

bool RadialConvectionDispersionOperatorBase::setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue)
{
	// Check if parameter is in parameter dependence of column dispersion coefficient
	if (_dispersionDep)
	{
		active* const param = _dispersionDep->getParameter(pId);
		if (param)
		{
			param->setADValue(adDirection, adValue);
			return true;
		}
	}

	// Check whether column radius has changed and update discretization if necessary
	if (pId.name == hashString("COL_RADIUS_INNER"))
	{
		_innerRadius.setADValue(adDirection, adValue);
		equidistantCells();
		return true;
	}

	if (pId.name == hashString("COL_RADIUS_OUTER"))
	{
		_outerRadius.setADValue(adDirection, adValue);
		equidistantCells();
		return true;
	}

	// We only need to do something if COL_DISPERSION is component independent
	if (!_dispersionCompIndep)
		return false;

	if ((pId.name != hashString("COL_DISPERSION")) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) || (pId.reaction != ReactionIndep) || (pId.particleType != ParTypeIndep))
		return false;

	if (_colDispersion.size() > _nComp)
	{
		// Section dependent
		if (pId.section == SectionIndep)
			return false;

		sensParams.insert(&_colDispersion[pId.section * _nComp]);
		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[pId.section * _nComp + i].setADValue(adDirection, adValue);
	}
	else
	{
		// Section independent
		if (pId.section != SectionIndep)
			return false;

		sensParams.insert(&_colDispersion[0]);
		for (unsigned int i = 0; i < _nComp; ++i)
			_colDispersion[i].setADValue(adDirection, adValue);
	}

	return true;
}

void RadialConvectionDispersionOperatorBase::equidistantCells()
{
	const active dr = (_outerRadius - _innerRadius) / _nCol;
	std::vector<active> centers(_nCol, 0.0);
	_cellSizes = std::vector<active>(_nCol, dr);
	std::vector<active> bounds(_nCol + 1, 0.0);

	for (unsigned int i = 0; i < _nCol; ++i)
	{
		centers[i] = (i + 0.5) * dr + _innerRadius;
		bounds[i] = i * dr + _innerRadius;
	}
	bounds[_nCol] = _outerRadius;

	_cellCenters = std::move(centers);
	_cellBounds = std::move(bounds);
}


/**
 * @brief Creates an ConvectionDispersionOperator
 */
template <typename Operator>
ConvectionDispersionOperator<Operator>::ConvectionDispersionOperator()
{
}

template <typename Operator>
ConvectionDispersionOperator<Operator>::~ConvectionDispersionOperator() CADET_NOEXCEPT
{
}

/**
 * @brief Returns the number of AD directions required for computing the Jacobian
 * @details Band compression is used to minimize the amount of AD directions.
 * @return Number of required AD directions
 */
template <typename Operator>
int ConvectionDispersionOperator<Operator>::requiredADdirs() const CADET_NOEXCEPT
{
	return _jacC.stride();
}

/**
 * @brief Reads parameters and allocates memory
 * @details Has to be called once before the operator is used.
 * @param [in] unitOpIdx Unit operation id of the owning unit operation model
 * @param [in] paramProvider Parameter provider for reading parameters
 * @param [out] parameters Map in which local parameters are inserted
 * @param [in] nComp Number of components
 * @param [in] nCol Number of axial cells
 * @return @c true if configuration went fine, @c false otherwise
 */
template <typename Operator>
bool ConvectionDispersionOperator<Operator>::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, unsigned int nCol)
{
	const bool retVal = _baseOp.configureModelDiscretization(paramProvider, helper, nComp, nCol, nComp);

	// Allocate memory
	const unsigned int lb = _baseOp.jacobianLowerBandwidth();
	const unsigned int ub = _baseOp.jacobianUpperBandwidth();
	const unsigned int mb = _baseOp.jacobianDiscretizedBandwidth();

	// Allocate matrices such that bandwidths can be switched (backwards flow support)
	_jacC.resize(nCol * nComp, lb, ub);

	_jacCdisc.resize(nCol * nComp, mb, mb);
	_jacCdisc.repartition(lb, ub);
	return retVal;
}

/**
 * @brief Reads model parameters
 * @details Only reads parameters that do not affect model structure (e.g., discretization).
 * @param [in] unitOpIdx Unit operation id of the owning unit operation model
 * @param [in] paramProvider Parameter provider for reading parameters
 * @param [out] parameters Map in which local parameters are inserted
 * @return @c true if configuration went fine, @c false otherwise
 */
template <typename Operator>
bool ConvectionDispersionOperator<Operator>::configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters)
{
	return _baseOp.configure(unitOpIdx, paramProvider, parameters);
}

/**
 * @brief Notifies the operator that a discontinuous section transition is in progress
 * @details In addition to changing flow direction internally, if necessary, the function returns whether
 *          the flow direction has changed.
 * @param [in] t Current time point
 * @param [in] secIdx Index of the new section that is about to be integrated
 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
 * @return @c true if flow direction has changed, otherwise @c false
 */
template <typename Operator>
bool ConvectionDispersionOperator<Operator>::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const AdJacobianParams& adJac)
{
	const bool hasChanged = _baseOp.notifyDiscontinuousSectionTransition(t, secIdx);

	// Check whether flow direction has changed and we need to update AD vectors
	// Exit if we do not need to setup (secIdx == 0) or change (prevU and u differ in sign) matrices
	if ((secIdx != 0) && !hasChanged)
		return false;

	const unsigned int lb = _baseOp.jacobianLowerBandwidth();
	const unsigned int ub = _baseOp.jacobianUpperBandwidth();
	if (_baseOp.forwardFlow())
	{
		// Forwards flow

		// Repartition column bulk Jacobians
		_jacC.repartition(lb, ub);
		_jacCdisc.repartition(lb, ub);
	}
	else
	{
		// Backwards flow

		// Repartition column bulk Jacobians
		_jacC.repartition(ub, lb);
		_jacCdisc.repartition(ub, lb);
	}

	// Update AD seed vectors since Jacobian structure has changed (bulk block bandwidths)
	prepareADvectors(adJac);

	return true;
}

/**
 * @brief Sets the AD seed vectors for the bulk transport variables
 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
 */
template <typename Operator>
void ConvectionDispersionOperator<Operator>::prepareADvectors(const AdJacobianParams& adJac) const
{
	// Early out if AD is disabled
	if (!adJac.adY)
		return;

	// Get bandwidths of blocks
	const unsigned int lowerColBandwidth = _jacC.lowerBandwidth();
	const unsigned int upperColBandwidth = _jacC.upperBandwidth();

	// Column block
	ad::prepareAdVectorSeedsForBandMatrix(adJac.adY + offsetC(), adJac.adDirOffset, _baseOp.nComp() * _baseOp.nCol(), lowerColBandwidth, upperColBandwidth, lowerColBandwidth);
}

/**
 * @brief Sets the flow rates for the current time section
 * @details The flow rates may change due to valve switches.
 * @param [in] in Total volumetric inlet flow rate
 * @param [in] out Total volumetric outlet flow rate
 * @param [in] colPorosity Porosity used for computing interstitial velocity from volumetric flow rate
 */
template <typename Operator>
void ConvectionDispersionOperator<Operator>::setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT
{
	_baseOp.setFlowRates(in, out, colPorosity);
}

/**
 * @brief Computes the residual of the transport equations
 * @param [in] model Model that owns the operator
 * @param [in] t Current time point
 * @param [in] secIdx Index of the current section
 * @param [in] y Pointer to unit operation's state vector
 * @param [in] yDot Pointer to unit operation's time derivative state vector
 * @param [out] res Pointer to unit operation's residual vector
 * @param [in] wantJac Determines whether analytic Jacobian is computed
 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
 */
template <typename Operator>
int ConvectionDispersionOperator<Operator>::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, bool wantJac, WithoutParamSensitivity)
{
	if (wantJac)
		return _baseOp.residual(model, t, secIdx, y, yDot, res, _jacC);
	else
		return _baseOp.residual(model, t, secIdx, y, yDot, res, WithoutParamSensitivity());
}

template <typename Operator>
int ConvectionDispersionOperator<Operator>::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithoutParamSensitivity)
{
	return _baseOp.residual(model, t, secIdx, y, yDot, res, WithoutParamSensitivity());
}

template <typename Operator>
int ConvectionDispersionOperator<Operator>::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity)
{
	return _baseOp.residual(model, t, secIdx, y, yDot, res, WithParamSensitivity());
}

template <typename Operator>
int ConvectionDispersionOperator<Operator>::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity)
{
	if (wantJac)
		return _baseOp.residual(model, t, secIdx, y, yDot, res, _jacC);
	else
		return _baseOp.residual(model, t, secIdx, y, yDot, res, WithParamSensitivity());
}

/**
 * @brief Multiplies the time derivative Jacobian @f$ \frac{\partial F}{\partial \dot{y}}\left(t, y, \dot{y}\right) @f$ with a given vector
 * @details The operation @f$ z = \frac{\partial F}{\partial \dot{y}} x @f$ is performed.
 *          The matrix-vector multiplication is performed matrix-free (i.e., no matrix is explicitly formed).
 *          
 *          Note that this function only performs multiplication with the Jacobian of the (axial) transport equations.
 *          The input vectors are assumed to point to the beginning (including inlet DOFs) of the respective unit operation's arrays.
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in] sDot Vector @f$ x @f$ that is transformed by the Jacobian @f$ \frac{\partial F}{\partial \dot{y}} @f$
 * @param [out] ret Vector @f$ z @f$ which stores the result of the operation
 */
template <typename Operator>
void ConvectionDispersionOperator<Operator>::multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const
{
	_baseOp.multiplyWithDerivativeJacobian(simTime, sDot, ret);
}

/**
 * @brief Extracts the system Jacobian from band compressed AD seed vectors
 * @details The input vectors are assumed to point to the beginning (including inlet DOFs) of the respective unit operation's arrays.
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 */
template <typename Operator>
void ConvectionDispersionOperator<Operator>::extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset)
{
	ad::extractBandedJacobianFromAd(adRes + offsetC(), adDirOffset, _jacC.lowerBandwidth(), _jacC);
}

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN

/**
 * @brief Compares the analytical Jacobian with a Jacobian derived by AD
 * @details The analytical Jacobian is assumed to be stored in the corresponding band matrices.
 *          The input vectors are assumed to point to the beginning (including inlet DOFs) of the respective unit operation's arrays.
 * @param [in] adRes Residual vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Number of AD directions used for non-Jacobian purposes (e.g., parameter sensitivities)
 * @return Maximum elementwise absolute difference between analytic and AD Jacobian
 */
template <typename Operator>
double ConvectionDispersionOperator<Operator>::checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const
{
	// Column
	const double maxDiffCol = ad::compareBandedJacobianWithAd(adRes + offsetC(), adDirOffset, _jacC.lowerBandwidth(), _jacC);
	LOG(Debug) << "-> Col block diff: " << maxDiffCol;

	return maxDiffCol;
}

#endif

/**
 * @brief Assembles the axial transport Jacobian @f$ J_0 @f$ of the time-discretized equations
 * @details The system \f[ \left( \frac{\partial F}{\partial y} + \alpha \frac{\partial F}{\partial \dot{y}} \right) x = b \f]
 *          has to be solved. The system Jacobian of the original equations,
 *          \f[ \frac{\partial F}{\partial y}, \f]
 *          is already computed (by AD or manually in residualImpl() with @c wantJac = true). This function is responsible
 *          for adding
 *          \f[ \alpha \frac{\partial F}{\partial \dot{y}} \f]
 *          to the system Jacobian, which yields the Jacobian of the time-discretized equations
 *          \f[ F\left(t, y_0, \sum_{k=0}^N \alpha_k y_k \right) = 0 \f]
 *          when a BDF method is used. The time integrator needs to solve this equation for @f$ y_0 @f$, which requires
 *          the solution of the linear system mentioned above (@f$ \alpha_0 = \alpha @f$ given in @p alpha).
 *
 * @param [in] alpha Value of \f$ \alpha \f$ (arises from BDF time discretization)
 */
template <typename Operator>
void ConvectionDispersionOperator<Operator>::assembleDiscretizedJacobian(double alpha)
{
	// Copy normal matrix over to factorizable matrix
	_jacCdisc.copyOver(_jacC);

	// Add time derivatives
	addTimeDerivativeToJacobian(alpha);
}


/**
 * @brief Adds the derivatives with respect to @f$ \dot{y} @f$ of @f$ F(t, y, \dot{y}) @f$ to the Jacobian
 * @details This functions computes 
 *          @f[ \begin{align*} \text{_jacCdisc} = \text{_jacCdisc} + \alpha \frac{\partial F}{\partial \dot{y}}. \end{align*} @f]
 *          The factor @f$ \alpha @f$ is useful when constructing the linear system in the time integration process.
 * @param [in] alpha Factor in front of @f$ \frac{\partial F}{\partial \dot{y}} @f$
 */
template <typename Operator>
void ConvectionDispersionOperator<Operator>::addTimeDerivativeToJacobian(double alpha)
{
	_baseOp.addTimeDerivativeToJacobian(alpha, _jacCdisc);
}

/**
 * @brief Assembles and factorizes the time discretized Jacobian
 * @details See assembleDiscretizedJacobian() for assembly of the time discretized Jacobian.
 * @param [in] alpha Factor in front of @f$ \frac{\partial F}{\partial \dot{y}} @f$
 * @return @c true if factorization went fine, otherwise @c false
 */
template <typename Operator>
bool ConvectionDispersionOperator<Operator>::assembleAndFactorizeDiscretizedJacobian(double alpha)
{
	assembleDiscretizedJacobian(alpha);
	return _jacCdisc.factorize();
}

/**
 * @brief Solves a (previously factorized) equation system
 * @details The (time discretized) Jacobian matrix has to be factorized before calling this function.
 *          Note that the given right hand side vector @p rhs is not shifted by the inlet DOFs. That
 *          is, it is assumed to point directly to the first axial DOF.
 * 
 * @param [in,out] rhs On entry, right hand side of the equation system. On exit, solution of the system.
 * @return @c true if the system was solved correctly, otherwise @c false
 */
template <typename Operator>
bool ConvectionDispersionOperator<Operator>::solveDiscretizedJacobian(double* rhs) const
{
	return _jacCdisc.solve(rhs);
}

/**
 * @brief Solves a system with the time derivative Jacobian and given right hand side
 * @details Note that the given right hand side vector @p rhs is not shifted by the inlet DOFs. That
 *          is, it is assumed to point directly to the first axial DOF.
 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
 * @param [in,out] rhs On entry, right hand side. On exit, solution of the system.
 * @return @c true if the system was solved correctly, @c false otherwise
 */
template <typename Operator>
bool ConvectionDispersionOperator<Operator>::solveTimeDerivativeSystem(const SimulationTime& simTime, double* const rhs)
{
	// Assemble
	_jacCdisc.setAll(0.0);
	addTimeDerivativeToJacobian(1.0);

	// Factorize
	const bool result = _jacCdisc.factorize();
	if (!result)
	{
		LOG(Error) << "Factorize() failed for bulk block";
		return false;
	}

	// Solve
	const bool result2 = _jacCdisc.solve(rhs);
	if (!result2)
	{
		LOG(Error) << "Solve() failed for bulk block";
		return false;
	}

	return true;
}

// Template instantiations
template class ConvectionDispersionOperator<AxialConvectionDispersionOperatorBase>;
template class ConvectionDispersionOperator<RadialConvectionDispersionOperatorBase>;

}  // namespace parts

}  // namespace model

}  // namespace cadet
