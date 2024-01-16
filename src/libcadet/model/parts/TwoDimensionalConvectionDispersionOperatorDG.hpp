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
 * Defines the 2D convection dispersion transport operator for a spatial Discontinuous Galerkin method.
 */

#ifndef LIBCADET_2DCONVECTIONDISPERSIONOPERATORDG_HPP_
#define LIBCADET_2DCONVECTIONDISPERSIONOPERATORDG_HPP_

#include "ParamIdUtil.hpp"
#include "AutoDiff.hpp"
#include "linalg/CompressedSparseMatrix.hpp" // todo not needed in the future
#include "Memory.hpp"
#include "model/ParameterMultiplexing.hpp"
#include "SimulationTypes.hpp"

#include <Eigen/Dense>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cadet
{

class IParameterProvider;
class IModel;
class IConfigHelper;

namespace model
{

class IParameterParameterDependence;

namespace parts
{
/**
 * @brief 2D Convection dispersion transport operator
 * @details Implements the equation
 * 
 * @f[\begin{align}
	\frac{\partial c_i}{\partial t} &= - u \frac{\partial c_i}{\partial z} + D_{\text{ax},i}(\rho) \frac{\partial^2 c_i}{\partial z^2} + \frac{1}{\rho} \frac{\partial}{\partial \rho} \left( \rho D_{\rho} \frac{\partial c_i}{\partial \rho} \right) \\
\end{align} @f]
 * with Danckwerts boundary conditions on the axial boundary (see @cite Danckwerts1953)
@f[ \begin{align}
u c_{\text{in},i}(t) &= u c_i(t,0,\rho) - D_{\text{ax},i}(\rho) \frac{\partial c_i}{\partial z}(t,0,\rho) \\
\frac{\partial c_i}{\partial z}(t,L,\rho) &= 0
\end{align} @f]
 * and Neumann boundary conditions on the radial boundary
@f[ \begin{align}
\frac{\partial c_i}{\partial \rho}(t,z,0) &= 0 \\
\frac{\partial c_i}{\partial z}(t,z,R) &= 0
\end{align} @f]
 * Methods are described in @cite VonLieres2010a (WENO, linear solver), and @cite Puttmann2013, @cite Puttmann2016 (forward sensitivities, AD, band compression)
 */
class TwoDimensionalConvectionDispersionOperatorDG
{
public:

	TwoDimensionalConvectionDispersionOperatorDG();
	~TwoDimensionalConvectionDispersionOperatorDG() CADET_NOEXCEPT;

	void setFlowRates(int compartment, const active& in, const active& out) CADET_NOEXCEPT;
	void setFlowRates(active const* in, active const* out) CADET_NOEXCEPT;

	Eigen::MatrixXd TwoDimensionalConvectionDispersionOperatorDG::tildeMr(const unsigned int elemIdx);
	Eigen::MatrixXd TwoDimensionalConvectionDispersionOperatorDG::tildeMrDash(const unsigned int elemIdx);
	Eigen::MatrixXd TwoDimensionalConvectionDispersionOperatorDG::tildeSrDash(const unsigned int elemIdx);

	bool configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, unsigned int axNodeStride, unsigned int radNodeStride, bool dynamicReactions);
	bool configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters);
	bool notifyDiscontinuousSectionTransition(double t, unsigned int secIdx);

	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, bool wantJac, WithoutParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithoutParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity);
	int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, bool wantJac, WithParamSensitivity);

	bool solveTimeDerivativeSystem(const SimulationTime& simTime, double* const rhs);
	void multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const;

	bool assembleAndFactorizeDiscretizedJacobian(double alpha);
	bool solveDiscretizedJacobian(double* rhs, double const* weight, double const* init, double outerTol) const;

	bool setParameter(const ParameterId& pId, double value);
	bool setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue);
	bool setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& id, double value);

	inline const active& columnLength() const CADET_NOEXCEPT { return _colLength; }
	inline const active& columnRadius() const CADET_NOEXCEPT { return _colRadius; }
	inline const active& currentVelocity(int idx) const CADET_NOEXCEPT { return _curVelocity[idx]; }
	inline const active& columnPorosity(int idx) const CADET_NOEXCEPT { return _colPorosities[idx]; }
	inline const active& crossSection(int idx) const CADET_NOEXCEPT { return _elemCrossSections[idx]; }
	inline active const* crossSections() const CADET_NOEXCEPT { return _crossSections.data(); }
	inline active const* radialCenters() const CADET_NOEXCEPT { return _radialCenters.data(); }
//	inline active const* radialCentroids() const CADET_NOEXCEPT { return _radialCentroids.data(); }
	inline active const* radialEdges() const CADET_NOEXCEPT { return _radialEdges.data(); }
	inline bool isCurrentFlowForward(int idx) const CADET_NOEXCEPT { return _curVelocity[idx] >= 0.0; }
	const active& axialDispersion(unsigned int idxSec, int idxRad, int idxComp) const CADET_NOEXCEPT;
	const active& radialDispersion(unsigned int idxSec, int idxRad, int idxComp) const CADET_NOEXCEPT;
	double inletFactor(unsigned int idxSec, int idxRad) const CADET_NOEXCEPT;

	inline linalg::CompressedSparseMatrix& jacobian() CADET_NOEXCEPT { return _jacC; }
	inline const linalg::CompressedSparseMatrix& jacobian() const CADET_NOEXCEPT { return _jacC; }

protected:

	typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> MatrixMap;

	class LinearSolver;
	class GmresSolver;
	template <typename sparse_t> class SparseDirectSolver;
	class DenseDirectSolver;

	void assembleDiscretizedJacobian(double alpha);

	template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	int residualImpl(const IModel& model, double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res);

	void setSparsityPattern();

	void setEquidistantRadialDisc();
	void setEquivolumeRadialDisc();
	void setUserdefinedRadialDisc();
	void updateRadialDisc();
	void initializeDG();

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
	unsigned int _axNPoints; //!< Number of axial discrete points
	unsigned int _axNElem; //!< Number of axial elements
	unsigned int _axPolyDeg; //!< Polynomial degree of axial discretization
	unsigned int _axNNodes; //!< Number of axial discrete points
	unsigned int _radNPoints; //!< Number of radial discrete points
	unsigned int _radNElem; //!< Number of radial elements
	unsigned int _radPolyDeg; //!< Polynomial degree of radial discretization
	unsigned int _radNNodes; //!< Number of radial discrete points
	unsigned int _quadratureRule; //!< Numerical quadrature rule
	unsigned int _quadratureOrder; //!< Order of the numerical quadrature
	unsigned int _qNNodes; //!< Number of quadrature nodes

	unsigned int _axNodeStride; //!< Stride to next axial point in state vector
	unsigned int _axElemStride; //!< Stride to next axial element in state vector
	unsigned int _radNodeStride; //!< Stride to next radial point in state vector
	unsigned int _radElemStride; //!< Stride to next radial element in state vector

	Eigen::VectorXd _axNodes; //!< axial LGL nodes on the reference element
	Eigen::VectorXd _axInvWeights; //!< axial LGL inverted weights on the reference element
	Eigen::VectorXd _radNodes; //!< radial LGL nodes on the reference element
	Eigen::VectorXd _radInvWeights; //!< radial LGL inverted weights on the reference element
	Eigen::VectorXd _qNodes; //!< quadrature nodes on the reference element
	Eigen::VectorXd _qWeights; //!< quadrature inverted weights on the reference element
	// operators
	Eigen::MatrixXd _axStiffM; //!< Axial polynomial stiffness matrix
	Eigen::MatrixXd _radStiffM; //!< Radial polynomial stiffness matrix
	Eigen::MatrixXd _axInvMM; //!< Axial mass matrix
	Eigen::MatrixXd _radInvMM; //!< Radial mass matrix
	//Eigen::MatrixXd _axPolyDerM; //!< Axial polynomial derivative matrix
	//Eigen::MatrixXd _radPolyDerM; //!< Radial polynomial derivative matrix
	Eigen::MatrixXd _axLiftM; //!< Axial lifting matrix
	Eigen::MatrixXd _radLiftM; //!< Radial lifting matrix
	// main eq. operators
	Eigen::MatrixXd _interpolationM;  //!< Polynomial interpolation matrix from LGL nodes to quadrature nodes
	Eigen::MatrixXd* _tildeMr; //!< Main eq. mass matrix adjusted for cylindrical metrics and dispersion
	Eigen::MatrixXd* _tildeMrDash; //!< Main eq. mass matrix on quadrature nodes adjusted for cylindrical metrics and dispersion
	Eigen::MatrixXd* _tildeSrDash; //!< Main eq. stiffness matrix on quadrature nodes adjusted for cylindrical metrics and dispersion
	
	// todo add active operators. double types serve as cache for the base values, then parameters (e.g. diffusion) need to be added in every residual call
	//Eigen::Matrix<active, Eigen::Dynamic, Eigen::Dynamic>* _AtildeMr; //!< Active type main eq. mass matrix adjusted for cylindrical metrics and dispersion

	// DG cache
	Eigen::Vector<active, Eigen::Dynamic> _fStarAux1; //!< 
	Eigen::Vector<active, Eigen::Dynamic> _fStarAux2; //!< 
	Eigen::Vector<active, Eigen::Dynamic> _fStarConv; //!< 
	Eigen::Vector<active, Eigen::Dynamic> _gZStarDisp; //!< 
	Eigen::Vector<active, Eigen::Dynamic> _gRStarDisp; //!< 
	Eigen::Matrix<active, Eigen::Dynamic, Eigen::Dynamic> _gZMat; //!< 
	Eigen::Matrix<active, Eigen::Dynamic, Eigen::Dynamic> _gRMat; //!< 

	bool _hasDynamicReactions; //!< Determines whether the model has dynamic reactions (only relevant for sparsity pattern)
	active _colLength; //!< Column length \f$ L \f$
	active _colRadius; //!< Column radius \f$ r_c \f$
	std::vector<active> _radialCoordinates; //!< Coordinates of the radial discrete points
	std::vector<active> _radialElemInterfaces; //!< Coordinates of the element interfaces
	std::vector<active> _radDelta; //!< Radial element spacing
	std::vector<active> _nodalCrossSections; //!< cross section area for each node
	// todo needed?
	//std::vector<active> _elementCrossSections; //!< cross section area for each element
	RadialDiscretizationMode _radialDiscretizationMode;

	std::vector<active> _colPorosities; //!< Bulk porosity for each compartment
	bool _singlePorosity; //!< Determines whether only one porosity for all compartments is given

	std::vector<active> _axialDispersion; //!< Axial dispersion coefficient \f$ D_{\text{ax}} \f$
	MultiplexMode _axialDispersionMode; //!< Multiplex mode of the axial dispersion
	std::vector<active> _radialDispersion; //!< Radial dispersion coefficient \f$ D_{\rho} \f$ at interpolation nodes
	std::vector<active> _curRadialDispersionTilde; //!< Radial dispersion coefficient \f$ D_{\rho} \f$ at quadrature ndoes
	MultiplexMode _radialDispersionMode; //!< Multiplex mode of the radial dispersion
	std::vector<active> _velocity; //!< Interstitial velocity parameter
	std::vector<active> _curVelocity; //!< Current interstitial velocity \f$ u \f$
	std::vector<int> _dir; //!< Current flow direction 
	bool _singleVelocity; //!< Determines whether only one velocity for all compartments is given

	ArrayPool _stencilMemory; //!< Provides memory for the stencil

	linalg::CompressedSparseMatrix _jacC; //!< Jacobian
	LinearSolver* _linearSolver; //!< Solves linear system with time discretized Jacobian

	IParameterParameterDependence* _dispersionDep;
};

} // namespace parts
} // namespace model
} // namespace cadet

#endif  // LIBCADET_2DCONVECTIONDISPERSIONOPERATORDG_HPP_
