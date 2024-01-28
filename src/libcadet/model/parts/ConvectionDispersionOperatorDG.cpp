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

#include "model/parts/ConvectionDispersionOperatorDG.hpp"
#include "cadet/Exceptions.hpp"

#include "Stencil.hpp"
#include "ParamReaderHelper.hpp"
#include "AdUtils.hpp"
#include "SimulationTypes.hpp"
//#include "model/parts/AxialConvectionDispersionKernelDG.hpp" // todo radial flow DG and outsource residual implementation to kernel
//#include "model/parts/RadialConvectionDispersionKernelDG.hpp"
#include "model/ParameterDependence.hpp"
#include "SensParamUtil.hpp"
#include "ConfigurationHelper.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>

namespace cadet
{

namespace model
{

namespace parts
{

/**
 * @brief Creates an AxialConvectionDispersionOperatorBaseDG
 */
AxialConvectionDispersionOperatorBaseDG::AxialConvectionDispersionOperatorBaseDG() :
	_dispersionDep(nullptr), _DGjacAxDispBlocks(nullptr)
{
}

AxialConvectionDispersionOperatorBaseDG::~AxialConvectionDispersionOperatorBaseDG() CADET_NOEXCEPT
{
	if (_dispersionDep)
		delete _dispersionDep;

	delete[] _DGjacAxDispBlocks;
	delete[] _auxState;
	delete[] _subsState;
}

/**
 * @brief Reads parameters and allocates memory
 * @details Has to be called once before the operator is used.
 * @param [in] paramProvider Parameter provider for reading parameters
 * @param [in] helper Configures parameter dependencies
 * @param [in] nComp Number of components
 * @param [in] exact_integration DG volume integral computation
 * @param [in] nCells Number of axial cells
 * @param [in] polyDeg Polynomial degree of DG approach
 * @param [in] strideNode node stride in state vector
 * @return @c true if configuration went fine, @c false otherwise
 */
bool AxialConvectionDispersionOperatorBaseDG::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, int polynomial_integration_mode, unsigned int nCells, unsigned int polyDeg, unsigned int strideNode)
{

	_nComp = nComp;
	_exactInt = static_cast<bool>(polynomial_integration_mode); // only integration mode 0 applies the inexact collocated diagonal LGL mass matrix
	_nCells = nCells;
	_polyDeg = polyDeg;
	_nNodes = _polyDeg + 1u;
	_nPoints = _nNodes * _nCells;
	_strideNode = strideNode;
	_strideCell = _nNodes * strideNode;

	/* Allocate space for DG discretization operations */
	_nodes.resize(_nNodes);
	_nodes.setZero();
	_invWeights.resize(_nNodes);
	_invWeights.setZero();
	_polyDerM.resize(_nNodes, _nNodes);
	_polyDerM.setZero();
	_invMM.resize(_nNodes, _nNodes);
	_invMM.setZero();

	_auxState = new active[_nPoints];
	_subsState = new active[_nPoints];
	for (int i = 0; i < _nPoints; i++) {
		_auxState[i] = 0.0;
		_subsState[i] = 0.0;
	}
	_boundary.setZero();
	_surfaceFlux.resize(_nCells + 1u);
	_surfaceFlux.setZero();

	_newStaticJac = true;

	dgtoolbox::lglNodesWeights(_polyDeg, _nodes, _invWeights, true);
	_invMM = dgtoolbox::invMMatrix(_polyDeg, _nodes);
	_polyDerM = dgtoolbox::derivativeMatrix(_polyDeg, _nodes);

	if(polynomial_integration_mode == 2) // use Gauss quadrature for exact integration
		_invMM = dgtoolbox::gaussQuadratureMMatrix(_nodes, _nNodes).inverse();

	if (paramProvider.exists("COL_DISPERSION_DEP"))
	{
		const std::string paramDepName = paramProvider.getString("COL_DISPERSION_DEP");
		_dispersionDep = helper.createParameterParameterDependence(paramDepName);
		if (!_dispersionDep)
			throw InvalidParameterException("Unknown parameter dependence " + paramDepName + " in COL_DISPERSION_DEP");

		_dispersionDep->configureModelDiscretization(paramProvider);
	}
	else
		_dispersionDep = helper.createParameterParameterDependence("CONSTANT_ONE");

	paramProvider.pushScope("discretization");
	_OSmode = 0;
	// Read oscillation suppression settings and apply them
	if (paramProvider.exists("oscillation_suppression")) {

		paramProvider.pushScope("oscillation_suppression");

		 std::string osMethod = paramProvider.getString("METHOD");

		 if (osMethod == "WENO")
		 {
			 LOG(Debug) << "WENO used as oscillation suppression mechanism.";

			 _OSmode = 1;
			 double eps = paramProvider.exists("WENO_EPS") ? paramProvider.getDouble("WENO_EPS") : 1e-6;
			 double r = paramProvider.exists("WENO_R") ? paramProvider.getDouble("WENO_R") : 2.0;
			 double gamma = paramProvider.exists("WENO_GAMMA") ? paramProvider.getDouble("WENO_GAMMA") : 0.998;
			 bool write_smoothness_indicator = paramProvider.exists("RETURN_SMOOTHNESS_INDICATOR") ? static_cast<bool>(paramProvider.getInt("RETURN_SMOOTHNESS_INDICATOR")) : false;

			 _weno.init(eps, r, gamma, _nCells, _nComp, write_smoothness_indicator);
		 }
		 else if (osMethod == "SUBCELL_LIMITING")
		 {
			 LOG(Debug) << "Subcell limiting used as oscillation suppression mechanism.";

			 _OSmode = 2;
			 int order = paramProvider.getInt("FV_ORDER");
			 //  @TODO Subcell limiting
			 throw InvalidParameterException("Oscillation suppression mechanism " + osMethod + " not implemented yet");

		 }
		 else if (osMethod != "NONE")
			throw InvalidParameterException("Unknown oscillation suppression mechanism " + osMethod + " in oscillation_suppression_mode");

		paramProvider.popScope();
	}
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
bool AxialConvectionDispersionOperatorBaseDG::configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters)
{
	// Read geometry parameters
	_colLength = paramProvider.getDouble("COL_LENGTH");
	_deltaZ = _colLength / _nCells;

	/* compute dispersion jacobian blocks(without parameters except element spacing, i.e. static entries) */
	// we only need unique dispersion blocks, which are given by cells 1, 2, nCol for inexact integration DG and by cells 1, 2, 3, nCol-1, nCol for eaxct integration DG
	_DGjacAxDispBlocks = new Eigen::MatrixXd[(_exactInt ? std::min(_nCells, 5u) : std::min(_nCells, 3u))];
	_DGjacAxDispBlocks[0] = DGjacobianDispBlock(1);
	if (_nCells > 1)
		_DGjacAxDispBlocks[1] = DGjacobianDispBlock(2);
	if (_nCells > 2 && _exactInt)
		_DGjacAxDispBlocks[2] = DGjacobianDispBlock(3);
	else if (_nCells > 2 && !_exactInt)
		_DGjacAxDispBlocks[2] = DGjacobianDispBlock(_nCells);
	if (_exactInt && _nCells > 3)
		_DGjacAxDispBlocks[3] = DGjacobianDispBlock(std::max(4u, _nCells - 1u));
	if (_exactInt && _nCells > 4)
		_DGjacAxDispBlocks[4] = DGjacobianDispBlock(_nCells);

	// note that convection jacobian block is computet in notifyDiscontinuousSectionTransition() since this block needs to be recomputed when flow direction changes

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
bool AxialConvectionDispersionOperatorBaseDG::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, MatrixXd& jacInlet)
{

	_curSection = secIdx;
	_newStaticJac = true;

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

	 // recompute convection jacobian block, which depends on flow direction
	_DGjacAxConvBlock = DGjacobianConvBlock();

	if (_curVelocity >= 0.0) { // forward flow upwind convection
		if (_exactInt)
			jacInlet = static_cast<double>(_curVelocity) * _DGjacAxConvBlock.col(0); // only first cell depends on inlet concentration
		else
			jacInlet(0, 0) = static_cast<double>(_curVelocity) * _DGjacAxConvBlock(0, 0); // only first node depends on inlet concentration
	}
	else {  // backward flow upwind convection
		if (_exactInt)
			jacInlet = static_cast<double>(_curVelocity) * _DGjacAxConvBlock.col(_DGjacAxConvBlock.cols() - 1); // only last cell depends on inlet concentration
		else
			jacInlet(0, 0) = static_cast<double>(_curVelocity) * _DGjacAxConvBlock(_DGjacAxConvBlock.rows() - 1, _DGjacAxConvBlock.cols() - 1); // only last node depends on inlet concentration
	}
	
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
void AxialConvectionDispersionOperatorBaseDG::setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT
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
int AxialConvectionDispersionOperatorBaseDG::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, Eigen::SparseMatrix<double, Eigen::RowMajor>& jac)
{
	//// Reset Jacobian but keep pattern
	//double* val = jac.valuePtr();
	//for (unsigned int entry = 0; entry < jac.nonZeros(); val++)
	//	val[0] = 0.0;

	linalg::BandedEigenSparseRowIterator jacIt(jac, 0);

	return residualImpl<double, double, double, linalg::BandedEigenSparseRowIterator, true>(model, t, secIdx, y, yDot, res, jacIt);
}

int AxialConvectionDispersionOperatorBaseDG::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, WithoutParamSensitivity)
{
	return residualImpl<double, double, double, linalg::BandedEigenSparseRowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandedEigenSparseRowIterator());
}

int AxialConvectionDispersionOperatorBaseDG::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithoutParamSensitivity)
{
	return residualImpl<active, active, double, linalg::BandedEigenSparseRowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandedEigenSparseRowIterator());
}

int AxialConvectionDispersionOperatorBaseDG::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, Eigen::SparseMatrix<double, Eigen::RowMajor>& jac)
{
	// todo include jacobian in convDisp operator
	//// Reset Jacobian but keep pattern
	//double* val = jac.valuePtr();
	//for (unsigned int entry = 0; entry < jac.nonZeros(); val++)
	//	val[0] = 0.0;

	linalg::BandedEigenSparseRowIterator jacIt(jac, 0);

	return residualImpl<double, active, active, linalg::BandedEigenSparseRowIterator, true>(model, t, secIdx, y, yDot, res, jacIt);
}

int AxialConvectionDispersionOperatorBaseDG::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, WithParamSensitivity)
{
	return residualImpl<double, active, active, linalg::BandedEigenSparseRowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandedEigenSparseRowIterator());
}

int AxialConvectionDispersionOperatorBaseDG::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithParamSensitivity)
{
	return residualImpl<active, active, active, linalg::BandedEigenSparseRowIterator, false>(model, t, secIdx, y, yDot, res, linalg::BandedEigenSparseRowIterator());
}

template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
int AxialConvectionDispersionOperatorBaseDG::residualImpl(const IModel& model, double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin)
{
	for (unsigned int comp = 0; comp < _nComp; comp++) {

		// create Eigen objects
		Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>> _C(y + offsetC() + comp, _nPoints, InnerStride<Dynamic>(_strideNode));
		Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>> _resC(res + offsetC() + comp, _nPoints, InnerStride<Dynamic>(_strideNode));
		Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<>> _h(reinterpret_cast<ResidualType*>(_subsState), _nPoints, InnerStride<>(1));
		Eigen::Map<Vector<StateType, Dynamic>, 0, InnerStride<>> _g(reinterpret_cast<StateType*>(_auxState), _nPoints, InnerStride<>(1));

		////ResidualType _pAvg0 = reinterpret_cast<ResidualType>(_weno._pAvg0);
		////ResidualType _pAvg1 = reinterpret_cast<ResidualType>(_weno._pAvg1);
		////ResidualType _pAvg2 = reinterpret_cast<ResidualType>(_weno._pAvg2);

		//// calculate smoothness indicator, if oscillation suppression is active
		//if (_OSmode != 0)
		//{
		//	if (_OSmode == 1) // WENO
		//	{
		//		for (unsigned int cell = 0; cell < _nCells; cell++)
		//		{
		//			if (cell > 0 && cell < _nCells - 1) // todo boundary treatment
		//			{
		//				// todo store weights additionally to inverse
		//				// todo overwrite values instead of recalculation
		//				_weno._pAvg0 = 1.0 / static_cast<ParamType>(_deltaZ) * (_invWeights.cwiseInverse().cast<StateType>().array() * _C.segment((cell - 1) * _nNodes, _nNodes).array()).sum();
		//				_weno._pAvg1 = 1.0 / static_cast<ParamType>(_deltaZ) * (_invWeights.cwiseInverse().cast<StateType>().array() * _C.segment(cell       * _nNodes, _nNodes).array()).sum();
		//				_weno._pAvg2 = 1.0 / static_cast<ParamType>(_deltaZ) * (_invWeights.cwiseInverse().cast<StateType>().array() * _C.segment((cell + 1) * _nNodes, _nNodes).array()).sum();


		//				// mark troubled cell
		//				if (true)
		//				{
		//					int hm = 0;
		//				}
		//			}
		//		}
		//	}
		//}

		// Add time derivative to bulk residual
		if (yDot)
		{
			Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>> _Cdot(yDot + offsetC() + comp, _nPoints, InnerStride<Dynamic>(_strideNode));
			_resC = _Cdot.template cast<ResidualType>();
		}
		else
			_resC.setZero();

		const ParamType u = static_cast<ParamType>(_curVelocity);
		const ParamType d_ax = static_cast<ParamType>(getSectionDependentSlice(_colDispersion, _nComp, secIdx)[comp]);

		// ===================================//
		// reset cache                        //
		// ===================================//

		_h.setZero();
		_g.setZero();
		_boundary[0] = y[comp]; // copy inlet DOFs to ghost node

		// ======================================//
		// solve auxiliary system g = d c / d x  //
		// ======================================//

		 // DG volume integral in strong form
		volumeIntegral<StateType, StateType>(_C, _g);

		// calculate numerical flux values c*
		InterfaceFluxAuxiliary<StateType>(y + offsetC() + comp, _strideNode, _strideCell);

		// DG surface integral in strong form
		surfaceIntegral<StateType, StateType>(y + offsetC() + comp, &_g[0], _strideNode, _strideCell, 1u, _nNodes);

		// ======================================//
		// solve main equation RHS  d h / d x    //
		// ======================================//

		// calculate the substitute h(g(c), c) and apply inverse mapping jacobian (reference space)
		_h = 2.0 / static_cast<ParamType>(_deltaZ) * (-u * _C + d_ax * (-2.0 / static_cast<ParamType>(_deltaZ)) * _g).template cast<ResidualType>();

		// DG volume integral in strong form
		volumeIntegral<ResidualType, ResidualType>(_h, _resC);

		// update boundary values for auxiliary variable g (solid wall)
		calcBoundaryValues<StateType>();

		// calculate numerical flux values h*
		InterfaceFlux<StateType, ParamType>(y + offsetC() + comp, d_ax);

		// DG surface integral in strong form
		surfaceIntegral<ResidualType, ResidualType>(&_h[0], res + offsetC() + comp,
			1u, _nNodes, _strideNode, _strideCell);
	}

	return 0;
}
/**
* @brief analytically calculates the (static) state jacobian
* @return 1 if jacobain estimation fits the predefined pattern of the jacobian, 0 if not.
*/
int AxialConvectionDispersionOperatorBaseDG::calcStaticAnaJacobian(Eigen::SparseMatrix<double, RowMajor>& jacobian, Eigen::MatrixXd& jacInlet, const int bulkOffset) {

	// DG convection dispersion Jacobian
	if (_exactInt)
		calcConvDispDGSEMJacobian(jacobian, jacInlet, bulkOffset);
	else
		calcConvDispCollocationDGSEMJacobian(jacobian, jacInlet, bulkOffset);

	if (!jacobian.isCompressed()) // if matrix lost its compressed storage, the pattern did not fit.
		return 0;

	return 1;
}
/**
 * @brief calculates the number of entris for the DG convection dispersion jacobian
 * @note only dispersion entries are relevant for jacobian NNZ as the convection entries are a subset of these
 */
unsigned int AxialConvectionDispersionOperatorBaseDG::nConvDispEntries(bool pureNNZ) {

	if (_exactInt) {
		if (pureNNZ) {
			return _nComp * ((3u * _nCells - 2u) * _nNodes * _nNodes + (2u * _nCells - 3u) * _nNodes); // dispersion entries
		}
		return _nComp * _nNodes * _nNodes + _nNodes // convection entries
			+ _nComp * ((3u * _nCells - 2u) * _nNodes * _nNodes + (2u * _nCells - 3u) * _nNodes); // dispersion entries
	}
	else {
		if (pureNNZ) {
			return _nComp * (_nCells * _nNodes * _nNodes + 8u * _nNodes); // dispersion entries
		}
		return _nComp * _nNodes * _nNodes + 1u // convection entries
			+ _nComp * (_nCells * _nNodes * _nNodes + 8u * _nNodes); // dispersion entries
	}
}
void model::parts::AxialConvectionDispersionOperatorBaseDG::convDispJacPattern(std::vector<T>& tripletList, const int bulkOffset)
{
	if (_exactInt)
		ConvDispModalPattern(tripletList, bulkOffset);
	else
		ConvDispNodalPattern(tripletList, bulkOffset);
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
void AxialConvectionDispersionOperatorBaseDG::multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const
{
	double* localRet = ret + offsetC();
	double const* localSdot = sDot + offsetC();
	const int gapCell = strideColNode() - static_cast<int>(_nComp) * strideColComp();

	for (unsigned int i = 0; i < _nPoints; ++i, localRet += gapCell, localSdot += gapCell)
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
void AxialConvectionDispersionOperatorBaseDG::addTimeDerivativeToJacobian(double alpha, Eigen::SparseMatrix<double, Eigen::RowMajor>& jacDisc, unsigned int blockOffset)
{
	const int gapCell = strideColNode() - static_cast<int>(_nComp) * strideColComp();
	linalg::BandedEigenSparseRowIterator jac(jacDisc, blockOffset);

	for (unsigned int point = 0; point < _nPoints; ++point, jac+=gapCell) {
		for (unsigned int comp = 0; comp < _nComp; ++comp, ++jac) {
			// dc_b / dt in transport equation
			jac[0] += alpha;
		}
	}
}

unsigned int AxialConvectionDispersionOperatorBaseDG::jacobianLowerBandwidth() const CADET_NOEXCEPT
{
	// @todo use more efficient seed vectors. currently, we treat the jacobian as banded, but the pattern is actually more sparse when multiple components are considered
	// (note that active type directions are limited)
	// We have different jacobian structure for exact integration and collocation DG scheme, i.e. we need different seed vectors
	// collocation DG: 2 * N_n * (N_c + N_q) + 1 = total bandwidth (main diagonal entries maximally depend on the next and last N_n liquid phase entries of same component)
	//    ex. int. DG: 4 * N_n * (N_c + N_q) + 1 = total bandwidth (main diagonal entries maximally depend on the next and last 2*N_n liquid phase entries of same component)

	return (_exactInt) ? 2 * _nNodes * strideColNode() : _nNodes * strideColNode();
}

unsigned int AxialConvectionDispersionOperatorBaseDG::jacobianUpperBandwidth() const CADET_NOEXCEPT
{
	return jacobianLowerBandwidth();
}

bool AxialConvectionDispersionOperatorBaseDG::setParameter(const ParameterId& pId, double value)
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

bool AxialConvectionDispersionOperatorBaseDG::setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& pId, double value)
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

bool AxialConvectionDispersionOperatorBaseDG::setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue)
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

// @todo: radial flow DG

/**
 * @brief Creates an ConvectionDispersionOperatorDG
 */
template <typename Operator>
ConvectionDispersionOperatorDG<Operator>::ConvectionDispersionOperatorDG()
{
}

template <typename Operator>
ConvectionDispersionOperatorDG<Operator>::~ConvectionDispersionOperatorDG() CADET_NOEXCEPT
{
}

/**
 * @brief Returns the number of AD directions required for computing the Jacobian
 * @details Band compression is used to minimize the amount of AD directions.
 * @return Number of required AD directions
 */
template <typename Operator>
int ConvectionDispersionOperatorDG<Operator>::requiredADdirs() const CADET_NOEXCEPT
{
	return _baseOp.requiredADdirs();
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
bool ConvectionDispersionOperatorDG<Operator>::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, int polynomial_integration_mode, unsigned int nCells, unsigned int polyDeg, unsigned int strideNode)
{
	const bool retVal = _baseOp.configureModelDiscretization(paramProvider, helper, nComp, polynomial_integration_mode, nCells, polyDeg, strideNode);

	// todo: manage jacobians in convDispOp instead of unitOp ?
	//// Allocate memory
	//if (_disc.exactInt)
	//	_jacInlet.resize(_disc.nNodes, 1); // first cell depends on inlet concentration (same for every component)
	//else
	//	_jacInlet.resize(1, 1); // first cell depends on inlet concentration (same for every component)
	//_jac.resize((_disc.nComp + _disc.strideBound) * _disc.nPoints, (_disc.nComp + _disc.strideBound) * _disc.nPoints);
	//_jacDisc.resize((_disc.nComp + _disc.strideBound) * _disc.nPoints, (_disc.nComp + _disc.strideBound) * _disc.nPoints);

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
bool ConvectionDispersionOperatorDG<Operator>::configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters)
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
bool ConvectionDispersionOperatorDG<Operator>::notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const AdJacobianParams& adJac)
{
	if (_baseOp.exactInt())
		_jacInlet.resize(_baseOp.nNodes(), 1); // first cell depends on inlet concentration (same for every component)
	else
		_jacInlet.resize(1, 1); // first cell depends on inlet concentration (same for every component)

	const bool hasChanged = _baseOp.notifyDiscontinuousSectionTransition(t, secIdx, _jacInlet);

	// Check whether flow direction has changed and we need to update AD vectors
	// Exit if we do not need to setup (secIdx == 0) or change (prevU and u differ in sign) matrices
	if ((secIdx != 0) && !hasChanged)
		return false;

	// todo: manage jacobians in convDispOp instead of unitOp ?
	//const unsigned int lb = _baseOp.jacobianLowerBandwidth();
	//const unsigned int ub = _baseOp.jacobianUpperBandwidth();
	//if (_baseOp.forwardFlow())
	//{
	//	// Forwards flow

	//	// Repartition column bulk Jacobians
	//	_jacC.repartition(lb, ub);
	//	_jacCdisc.repartition(lb, ub);
	//}
	//else
	//{
	//	// Backwards flow

	//	// Repartition column bulk Jacobians
	//	_jacC.repartition(ub, lb);
	//	_jacCdisc.repartition(ub, lb);
	//}

	// Update AD seed vectors since Jacobian structure has changed (bulk block bandwidths)
	prepareADvectors(adJac);

	return true;
}

/**
 * @brief Sets the AD seed vectors for the bulk transport variables
 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
 */
template <typename Operator>
void ConvectionDispersionOperatorDG<Operator>::prepareADvectors(const AdJacobianParams& adJac) const
{
	// Early out if AD is disabled
	if (!adJac.adY)
		return;

	// Get bandwidths of blocks
	const unsigned int lowerColBandwidth = _baseOp.exactInt() ? 2 * _baseOp.nNodes() * _baseOp.strideColNode() : _baseOp.nNodes() * _baseOp.strideColNode();
	const unsigned int upperColBandwidth = lowerColBandwidth;

	// Column block
	ad::prepareAdVectorSeedsForBandMatrix(adJac.adY + _baseOp.nComp(), adJac.adDirOffset, _baseOp.nComp() * _baseOp.nPoints(), lowerColBandwidth, upperColBandwidth, lowerColBandwidth);
}

/**
 * @brief Sets the flow rates for the current time section
 * @details The flow rates may change due to valve switches.
 * @param [in] in Total volumetric inlet flow rate
 * @param [in] out Total volumetric outlet flow rate
 * @param [in] colPorosity Porosity used for computing interstitial velocity from volumetric flow rate
 */
template <typename Operator>
void ConvectionDispersionOperatorDG<Operator>::setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT
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
int ConvectionDispersionOperatorDG<Operator>::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, bool wantJac, WithoutParamSensitivity)
{
	if (wantJac)
		return _baseOp.residual(model, t, secIdx, y, yDot, res, _jacC);
	else
		return _baseOp.residual(model, t, secIdx, y, yDot, res, WithoutParamSensitivity());
}

template <typename Operator>
int ConvectionDispersionOperatorDG<Operator>::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithoutParamSensitivity)
{
	return _baseOp.residual(model, t, secIdx, y, yDot, res, WithoutParamSensitivity());
}

template <typename Operator>
int ConvectionDispersionOperatorDG<Operator>::residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity)
{
	return _baseOp.residual(model, t, secIdx, y, yDot, res, WithParamSensitivity());
}

template <typename Operator>
int ConvectionDispersionOperatorDG<Operator>::residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity)
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
void ConvectionDispersionOperatorDG<Operator>::multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const
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
void ConvectionDispersionOperatorDG<Operator>::extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset)
{
	const active* const adVec = adRes + offsetC();

	const int lowerBandwidth = _baseOp.jacobianLowerBandwidth();
	const int upperBandwidth = lowerBandwidth;
	const int stride = lowerBandwidth + 1 + upperBandwidth;

	int diagDir = lowerBandwidth;
	for (int eq = 0; eq < _jacC.rows(); ++eq)
	{
		// Start with lowest subdiagonal and stay in the range of the columns:
		// diagDir might be too big for the matrix and, hence, dir ranges between
		// diagDir - lowerBandwidth <= dir <= diagDir + upperBandwidth
		int dir = diagDir - lowerBandwidth + eq % stride;

		// Loop over diagonals
		for (int diag = 0; diag < stride; ++diag)
		{
			if (eq - lowerBandwidth + diag >= 0 && // left boundary
				eq - lowerBandwidth + diag < _jacC.cols() && // right boundary
				adVec[eq].getADValue(adDirOffset + dir) != 0.0 // keep pattern
				)
				_jacC.coeffRef(eq, eq - lowerBandwidth + diag) = adVec[eq].getADValue(adDirOffset + dir);

			// Wrap around at end of row and jump to lowest subdiagonal
			if (dir == diagDir + upperBandwidth)
				dir = diagDir - lowerBandwidth;
			else
				++dir;
		}
	}
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
double ConvectionDispersionOperatorDG<Operator>::checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const
{
	// todo implement this function
	//// Column
	//const double maxDiffCol = ad::compareBandedJacobianWithAd(adRes + offsetC(), adDirOffset, _jacC.lowerBandwidth(), _jacC);
	//LOG(Debug) << "-> Col block diff: " << maxDiffCol;

	//return maxDiffCol;
	
	LOG(Debug) << "checkAnalyticJacobianAgainstAd not implemented in ConvectionDispersionOperatorDG";
	return 1;
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
void ConvectionDispersionOperatorDG<Operator>::assembleDiscretizedJacobian(double alpha)
{
	// set to static jacobian entries
	_jacCdisc = _jacC;

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
void ConvectionDispersionOperatorDG<Operator>::addTimeDerivativeToJacobian(double alpha)
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
bool ConvectionDispersionOperatorDG<Operator>::assembleAndFactorizeDiscretizedJacobian(double alpha)
{
	// todo: this functionality is currently not needed for DG since we assemble and factorize a global jacobian, not blocks (as in FV)
	//assembleDiscretizedJacobian(alpha);
	//return _jacCdisc.factorize();
	return true;
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
bool ConvectionDispersionOperatorDG<Operator>::solveDiscretizedJacobian(double* rhs) const
{
	// todo: this functionality is currently not needed for DG since we assemble and factorize a global jacobian, not blocks (as in FV)
	return true;// _jacCdisc.solve(rhs);
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
bool ConvectionDispersionOperatorDG<Operator>::solveTimeDerivativeSystem(const SimulationTime& simTime, double* const rhs)
{
	// Assemble
	double* vals = _jacCdisc.valuePtr();
	for (int entry = 0; entry < _jacCdisc.nonZeros(); entry++)
		vals[entry] = 0.0;

	addTimeDerivativeToJacobian(1.0);

	Eigen::SparseLU<Eigen::SparseMatrix<double>> _linSolver;
	_linSolver.analyzePattern(_jacCdisc);
	_linSolver.factorize(_jacCdisc);

	if (_linSolver.info() != Success) {
		LOG(Error) << "factorization failed in sensitivity initialization";
	}

	Eigen::Map<Eigen::VectorXd> ret_vec(rhs, _jacCdisc.rows());
	ret_vec = _linSolver.solve(ret_vec);

	// Use the factors to solve the linear system 
	if (_linSolver.info() != Success) {
		LOG(Error) << "solve failed in sensitivity initialization";
	}

	return true;
}

// Template instantiations
template class ConvectionDispersionOperatorDG<AxialConvectionDispersionOperatorBaseDG>;
//template class ConvectionDispersionOperatorDG<RadialConvectionDispersionOperatorBaseDG>; // todo

}  // namespace parts

}  // namespace model

}  // namespace cadet
