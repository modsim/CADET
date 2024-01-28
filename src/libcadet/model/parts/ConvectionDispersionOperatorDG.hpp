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
 * Defines the convection dispersion transport operator according to the discontinuous Galerkin discretization.
 */

#ifndef LIBCADET_CONVECTIONDISPERSIONOPERATORDG_HPP_
#define LIBCADET_CONVECTIONDISPERSIONOPERATORDG_HPP_

#include "ParamIdUtil.hpp"
#include "AutoDiff.hpp"
#include "Memory.hpp"
#include "Weno_DG.hpp"
#include "SimulationTypes.hpp"
#include <ParamReaderHelper.hpp>
#include "linalg/BandedEigenSparseRowIterator.hpp"
#include "model/parts/DGToolbox.hpp"

#include <unordered_map>
#include <unordered_set>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

using namespace Eigen;

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
			 * Methods are described in @cite todo, and @cite Puttmann2013, @cite Puttmann2016 (forward sensitivities, AD, band compression)
			 *
			 * This class does not store the Jacobian. It only fills existing matrices given to its residual() functions.
			 * It assumes that there is no offset to the inlet in the local state vector and that the firsts cell is placed
			 * directly after the inlet DOFs.
			 */
			class AxialConvectionDispersionOperatorBaseDG
			{
			public:

				AxialConvectionDispersionOperatorBaseDG();
				~AxialConvectionDispersionOperatorBaseDG() CADET_NOEXCEPT;

				void setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT;

				bool configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, int polynomial_integration_mode, unsigned int nCells, unsigned int polyDeg, unsigned int strideNode);
				bool configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters);
				bool notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, Eigen::MatrixXd& jacInlet);

				int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, Eigen::SparseMatrix<double, Eigen::RowMajor>& jac);
				int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, Eigen::SparseMatrix<double, Eigen::RowMajor>& jac);
				int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, WithoutParamSensitivity);
				int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, WithParamSensitivity);
				int residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithParamSensitivity);
				int residual(const IModel& model, double t, unsigned int secIdx, active const* y, double const* yDot, active* res, WithoutParamSensitivity);

				int calcStaticAnaJacobian(Eigen::SparseMatrix<double, Eigen::RowMajor>& jacobian, Eigen::MatrixXd& jacInlet, const int bulkOffset = 0);
				typedef Eigen::Triplet<double> T;
				void convDispJacPattern(std::vector<T>& tripletList, const int bulkOffset = 0);
				unsigned int nConvDispEntries(bool pureNNZ = false);
				void multiplyWithDerivativeJacobian(const SimulationTime& simTime, double const* sDot, double* ret) const;
				void addTimeDerivativeToJacobian(double alpha, Eigen::SparseMatrix<double, Eigen::RowMajor>& jacDisc, unsigned int blockOffset = 0);

				inline const active& columnLength() const CADET_NOEXCEPT { return _colLength; }
				inline const active& currentVelocity(double pos) const CADET_NOEXCEPT { return _curVelocity; }
				inline bool forwardFlow() const CADET_NOEXCEPT { return _curVelocity >= 0.0; }

				inline double cellLeftBound(unsigned int idx) const CADET_NOEXCEPT { return idx * static_cast<double>(_deltaZ); }
				double relativeCoordinate(unsigned int idx) const CADET_NOEXCEPT
				{
					//const unsigned int cell = floor(idx / _nNodes);
					//const unsigned int node = idx % _nNodes;
					// divide by column length to get relative position
					return (floor(idx / _nNodes) * static_cast<double>(_deltaZ) + 0.5 * static_cast<double>(_deltaZ) * (1.0 + _nodes[idx % _nNodes])) / static_cast<double>(_colLength);
				}

				inline const double* LGLnodes() const CADET_NOEXCEPT { return &_nodes[0]; }
				inline const active& currentVelocity() const CADET_NOEXCEPT { return _curVelocity; }
				inline const active* currentDispersion(const int secIdx) const CADET_NOEXCEPT { return getSectionDependentSlice(_colDispersion, _nComp, secIdx); }
				inline const bool dispersionCompIndep() const CADET_NOEXCEPT { return _dispersionCompIndep; }

				inline unsigned int nComp() const CADET_NOEXCEPT { return _nComp; }
				inline unsigned int nCells() const CADET_NOEXCEPT { return _nCells; }
				inline unsigned int nNodes() const CADET_NOEXCEPT { return _nNodes; }
				inline unsigned int nPoints() const CADET_NOEXCEPT { return _nPoints; }
				inline bool exactInt() const CADET_NOEXCEPT { return _exactInt; }
				inline bool hasSmoothnessIndicator() const CADET_NOEXCEPT { return static_cast<bool>(_OSmode); } // only zero if no oscillation suppression
				inline double* smoothnessIndicator() const CADET_NOEXCEPT
				{
					if (_OSmode == 1)
						return _weno.troubledCells();
					//else if (_OSmode == 2) // todo subcell limiting
					//	return _subcell.troubledCells();
					else
						return nullptr;
				}

				// Indexer functionality:
				// Strides
				inline int strideColCell() const CADET_NOEXCEPT { return static_cast<int>(_strideCell); }
				inline int strideColNode() const CADET_NOEXCEPT { return static_cast<int>(_strideNode); }
				inline int strideColComp() const CADET_NOEXCEPT { return 1; }
				// Offsets
				inline int offsetC() const CADET_NOEXCEPT { return _nComp; }

				unsigned int jacobianLowerBandwidth() const CADET_NOEXCEPT;
				unsigned int jacobianUpperBandwidth() const CADET_NOEXCEPT;
				double inletJacobianFactor() const CADET_NOEXCEPT;

				// @todo use more efficient seed vectors. currently, we treat the jacobian as banded, but the pattern is actually more sparse when multiple components are considered
				// (note that active type directions are limited)
				// We have different jacobian structure for exact integration and collocation DG scheme, i.e. we need different seed vectors
				// collocation DG: 2 * N_n * (strideNode) + 1 = total bandwidth (main diagonal entries maximally depend on the next and last N_n liquid phase entries of same component)
				//    ex. int. DG: 4 * N_n * (strideNode) + 1 = total bandwidth (main diagonal entries maximally depend on the next and last 2*N_n liquid phase entries of same component)
				int requiredADdirs() const CADET_NOEXCEPT { return (_exactInt) ? 4 * _nNodes * strideColNode() + 1 : 2 * _nNodes * strideColNode() + 1; }


				bool setParameter(const ParameterId& pId, double value);
				bool setSensitiveParameter(std::unordered_set<active*>& sensParams, const ParameterId& pId, unsigned int adDirection, double adValue);
				bool setSensitiveParameterValue(const std::unordered_set<active*>& sensParams, const ParameterId& id, double value);

			protected:

				template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
				int residualImpl(const IModel& model, double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin);

				// discretization parameters
				unsigned int _nComp; //!< Number of components
				bool _exactInt;	//!< specifies whether integrals are calculated exactly or approximated with LGL quadrature
				unsigned int _polyDeg; //!< DG discretization polynomial degree
				unsigned int _nCells; //!< Number of axial cells
				unsigned int _nNodes; //!< Number of nodes per cell
				unsigned int _nPoints; //!< Number of axial cells

				unsigned int _strideNode; //!< Number of values between the same item in two adjacent nodes
				unsigned int _strideCell; //!< Number of values between the same item in two adjacent cells

				// discretization toolbox and memory buffers
				active _deltaZ; //!< Cell spacing
				Eigen::VectorXd _nodes; //!< LGL nodes in [-1, 1]
				Eigen::MatrixXd _polyDerM; //!< Polynomial derivative Matrix
				Eigen::VectorXd _invWeights; //!< Inverse LGL quadrature weights -> diagonal (lumped) LGL mass matrix
				Eigen::MatrixXd _invMM; //!< Inverse (exact) mass matrix
				Eigen::MatrixXd* _DGjacAxDispBlocks; //!< Unique Jacobian blocks for axial dispersion
				Eigen::MatrixXd _DGjacAxConvBlock; //!< Unique Jacobian blocks for axial convection

				active* _auxState; //!< auxiliary variable
				active* _subsState; //!< auxiliary substitute
				Eigen::Vector<active, Eigen::Dynamic> _surfaceFlux; //!< stores the surface flux values
				Eigen::Vector<active, 4> _boundary; //!< stores the boundary values from Danckwert boundary conditions

				// non-linear oscillation prevention mechanism
				int _OSmode; //!< oscillation suppression mode; 0 : none, 1 : WENO, 2 : Subcell limiting
				WenoDG _weno; //!< WENO operator
				//SucellLimiter _subcellLimiter; // todo

				// Simulation parameters
				active _colLength; //!< Column length \f$ L \f$
				active _crossSection; //!< Cross section area 

				// Section dependent parameters
				std::vector<active> _colDispersion; //!< Column dispersion (may be section dependent) \f$ D_{\text{ax}} \f$
				std::vector<active> _velocity; //!< Interstitial velocity (may be section dependent) \f$ u \f$
				active _curVelocity; //!< Current interstitial velocity \f$ u \f$ in this time section
				int _dir; //!< Current flow direction in this time section

				// needed?
				int _curSection; //!< current section index
				bool _newStaticJac; //!< determines wether static analytical jacobian needs to be computed (every section)

				// todo weno
				//ArrayPool _stencilMemory; //!< Provides memory for the stencil
				//double* _wenoDerivatives; //!< Holds derivatives of the WENO scheme
				//Weno _weno; //!< The WENO scheme implementation
				//double _wenoEpsilon; //!< The @f$ \varepsilon @f$ of the WENO scheme (prevents division by zero)

				bool _dispersionCompIndep; //!< Determines whether dispersion is component independent

				IParameterParameterDependence* _dispersionDep;

				/* ===================================================================================
				 *  Functions to calculate Jacobian blocks
				 * =================================================================================== */

				/**
				 * @brief calculates the convection part of the DG jacobian
				 */
				Eigen::MatrixXd DGjacobianConvBlock() {

					// Convection block [ d RHS_conv / d c ], additionally depends on upwind flux part from corresponding neighbour cell
					Eigen::MatrixXd convBlock = Eigen::MatrixXd::Zero(_nNodes, _nNodes + 1);

					if (_curVelocity >= 0.0) { // forward flow -> Convection block additionally depends on last entry of previous cell
						convBlock.block(0, 1, _nNodes, _nNodes) -= _polyDerM;

						if (_exactInt) {
							convBlock.block(0, 0, _nNodes, 1) += _invMM.block(0, 0, _nNodes, 1);
							convBlock.block(0, 1, _nNodes, 1) -= _invMM.block(0, 0, _nNodes, 1);
						}
						else {
							convBlock(0, 0) += _invWeights[0];
							convBlock(0, 1) -= _invWeights[0];
						}
					}
					else { // backward flow -> Convection block additionally depends on first entry of subsequent cell
						convBlock.block(0, 0, _nNodes, _nNodes) -= _polyDerM;

						if (_exactInt) {
							convBlock.block(0, _nNodes - 1, _nNodes, 1) += _invMM.block(0, _nNodes - 1, _nNodes, 1);
							convBlock.block(0, _nNodes, _nNodes, 1) -= _invMM.block(0, _nNodes - 1, _nNodes, 1);
						}
						else {
							convBlock(_nNodes - 1, _nNodes - 1) += _invWeights[_nNodes - 1];
							convBlock(_nNodes - 1, _nNodes) -= _invWeights[_nNodes - 1];
						}
					}
					convBlock *= 2 / static_cast<double>(_deltaZ);

					return -convBlock; // *-1 for residual
				}

				/**
				 * @brief calculates the DG Jacobian auxiliary block
				 * @param [in] exInt true if exact integration DG scheme
				 * @param [in] cellIdx cell index
				 */
				Eigen::MatrixXd getGBlock(unsigned int cellIdx) {

					// Auxiliary Block [ d g(c) / d c ], additionally depends on boundary entries of neighbouring cells
					Eigen::MatrixXd gBlock = Eigen::MatrixXd::Zero(_nNodes, _nNodes + 2);
					gBlock.block(0, 1, _nNodes, _nNodes) = _polyDerM;
					if (_exactInt) {
						if (cellIdx != 1 && cellIdx != _nCells) {
							gBlock.block(0, 0, _nNodes, 1) -= 0.5 * _invMM.block(0, 0, _nNodes, 1);
							gBlock.block(0, 1, _nNodes, 1) += 0.5 * _invMM.block(0, 0, _nNodes, 1);
							gBlock.block(0, _nNodes, _nNodes, 1) -= 0.5 * _invMM.block(0, _nNodes - 1, _nNodes, 1);
							gBlock.block(0, _nNodes + 1, _nNodes, 1) += 0.5 * _invMM.block(0, _nNodes - 1, _nNodes, 1);
						}
						else if (cellIdx == 1) { // left
							if (cellIdx == _nCells)
								return gBlock * 2 / static_cast<double>(_deltaZ);
							;
							gBlock.block(0, _nNodes, _nNodes, 1) -= 0.5 * _invMM.block(0, _nNodes - 1, _nNodes, 1);
							gBlock.block(0, _nNodes + 1, _nNodes, 1) += 0.5 * _invMM.block(0, _nNodes - 1, _nNodes, 1);
						}
						else if (cellIdx == _nCells) { // right
							gBlock.block(0, 0, _nNodes, 1) -= 0.5 * _invMM.block(0, 0, _nNodes, 1);
							gBlock.block(0, 1, _nNodes, 1) += 0.5 * _invMM.block(0, 0, _nNodes, 1);
						}
						else if (cellIdx == 0 || cellIdx == _nCells + 1) {
							gBlock.setZero();
						}
						gBlock *= 2 / static_cast<double>(_deltaZ);
					}
					else {
						if (cellIdx == 0 || cellIdx == _nCells + 1)
							return Eigen::MatrixXd::Zero(_nNodes, _nNodes + 2);

						gBlock(0, 0) -= 0.5 * _invWeights[0];
						gBlock(0, 1) += 0.5 * _invWeights[0];
						gBlock(_nNodes - 1, _nNodes) -= 0.5 * _invWeights[_nNodes - 1];
						gBlock(_nNodes - 1, _nNodes + 1) += 0.5 * _invWeights[_nNodes - 1];
						gBlock *= 2 / static_cast<double>(_deltaZ);

						if (cellIdx == 1) {
							// adjust auxiliary Block [ d g(c) / d c ] for left boundary cell
							gBlock(0, 1) -= 0.5 * _invWeights[0] * 2 / static_cast<double>(_deltaZ);
							if (cellIdx == _nCells) { // adjust for special case one cell
								gBlock(0, 0) += 0.5 * _invWeights[0] * 2 / static_cast<double>(_deltaZ);
								gBlock(_nNodes - 1, _nNodes + 1) -= 0.5 * _invWeights[_nNodes - 1] * 2 / static_cast<double>(_deltaZ);
								gBlock(_nNodes - 1, _nNodes) += 0.5 * _invWeights[_polyDeg] * 2 / static_cast<double>(_deltaZ);
							}
						}
						else if (cellIdx == _nCells) {
							// adjust auxiliary Block [ d g(c) / d c ] for right boundary cell
							gBlock(_nNodes - 1, _nNodes) += 0.5 * _invWeights[_polyDeg] * 2 / static_cast<double>(_deltaZ);
						}
					}

					return gBlock;
				}
				/**
				 * @brief calculates the num. flux part of a dispersion DG Jacobian block
				 * @param [in] cellIdx cell index
				 * @param [in] leftG left neighbour auxiliary block
				 * @param [in] middleG neighbour auxiliary block
				 * @param [in] rightG neighbour auxiliary block
				 */
				Eigen::MatrixXd auxBlockGstar(unsigned int cellIdx, Eigen::MatrixXd leftG, Eigen::MatrixXd middleG, Eigen::MatrixXd rightG) {

					// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
					Eigen::MatrixXd gStarDC = Eigen::MatrixXd::Zero(_nNodes, 3 * _nNodes + 2);
					// NOTE: N = polyDeg
					// indices  gStarDC    :     0   ,   1   , ..., _nNodes; _nNodes+1, ..., 2 * _nNodes;	2*_nNodes+1, ..., 3 * _nNodes; 3*_nNodes+1
					// derivative index j  : -(N+1)-1, -(N+1),... ,  -1   ;   0     , ...,		N	 ;	  N + 1	  , ..., 2N + 2    ; 2(N+1) +1
					// auxiliary block [d g^* / d c]
					if (cellIdx != 1) {
						gStarDC.block(0, _nNodes, 1, _nNodes + 2) += middleG.block(0, 0, 1, _nNodes + 2);
						gStarDC.block(0, 0, 1, _nNodes + 2) += leftG.block(_nNodes - 1, 0, 1, _nNodes + 2);
					}
					if (cellIdx != _nCells) {
						gStarDC.block(_nNodes - 1, _nNodes, 1, _nNodes + 2) += middleG.block(_nNodes - 1, 0, 1, _nNodes + 2);
						gStarDC.block(_nNodes - 1, 2 * _nNodes, 1, _nNodes + 2) += rightG.block(0, 0, 1, _nNodes + 2);
					}
					gStarDC *= 0.5;

					return gStarDC;
				}

				Eigen::MatrixXd getBMatrix() {

					Eigen::MatrixXd B = Eigen::MatrixXd::Zero(_nNodes, _nNodes);
					B(0, 0) = -1.0;
					B(_nNodes - 1, _nNodes - 1) = 1.0;

					return B;
				}

				/**
				 * @brief calculates the dispersion part of the DG jacobian
				 * @param [in] exInt true if exact integration DG scheme
				 * @param [in] cellIdx cell index
				 */
				Eigen::MatrixXd DGjacobianDispBlock(unsigned int cellIdx) {

					int offC = 0; // inlet DOFs not included in Jacobian

					Eigen::MatrixXd dispBlock;

					if (_exactInt) {

						// Inner dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
						dispBlock = Eigen::MatrixXd::Zero(_nNodes, 3 * _nNodes + 2);

						Eigen::MatrixXd B = getBMatrix(); // "Lifting" matrix
						Eigen::MatrixXd gBlock = getGBlock(cellIdx); // current cell auxiliary block matrix
						Eigen::MatrixXd gStarDC = auxBlockGstar(cellIdx, getGBlock(cellIdx - 1), gBlock, getGBlock(cellIdx + 1)); // Numerical flux block

						//  indices  dispBlock :   0	 ,   1   , ..., _nNodes;	_nNodes+1, ..., 2 * _nNodes;	2*_nNodes+1, ..., 3 * _nNodes; 3*_nNodes+1
						//	derivative index j  : -(N+1)-1, -(N+1),...,	 -1	  ;   0     , ...,		N	 ;	  N + 1	  , ..., 2N + 2    ; 2(N+1) +1
						dispBlock.block(0, _nNodes, _nNodes, _nNodes + 2) += _polyDerM * gBlock - _invMM * B * gBlock;
						dispBlock += _invMM * B * gStarDC;
						dispBlock *= 2 / static_cast<double>(_deltaZ);
					}
					else { // inexact integration collocation DGSEM

						dispBlock = Eigen::MatrixXd::Zero(_nNodes, 3 * _nNodes);
						Eigen::MatrixXd GBlockLeft = getGBlock(cellIdx - 1);
						Eigen::MatrixXd GBlock = getGBlock(cellIdx);
						Eigen::MatrixXd GBlockRight = getGBlock(cellIdx + 1);

						// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell
						// NOTE: N = polyDeg
						// cell indices :  0  , ..., _nNodes - 1;	_nNodes, ..., 2 * _nNodes - 1;	2 * _nNodes, ..., 3 * _nNodes - 1
						//			 j  : -N-1, ..., -1		  ; 0     , ..., N			   ;	N + 1, ..., 2N + 1
						dispBlock.block(0, _nNodes - 1, _nNodes, _nNodes + 2) = _polyDerM * GBlock;

						if (cellIdx > 1) {
							dispBlock(0, _nNodes - 1) += -_invWeights[0] * (-0.5 * GBlock(0, 0) + 0.5 * GBlockLeft(_nNodes - 1, _nNodes)); // G_N,N		i=0, j=-1
							dispBlock(0, _nNodes) += -_invWeights[0] * (-0.5 * GBlock(0, 1) + 0.5 * GBlockLeft(_nNodes - 1, _nNodes + 1)); // G_N,N+1	i=0, j=0
							dispBlock.block(0, _nNodes + 1, 1, _nNodes) += -_invWeights[0] * (-0.5 * GBlock.block(0, 2, 1, _nNodes)); // G_i,j		i=0, j=1,...,N+1
							dispBlock.block(0, 0, 1, _nNodes - 1) += -_invWeights[0] * (0.5 * GBlockLeft.block(_nNodes - 1, 1, 1, _nNodes - 1)); // G_N,j+N+1		i=0, j=-N-1,...,-2
						}
						else if (cellIdx == 1) { // left boundary cell
							dispBlock.block(0, _nNodes - 1, 1, _nNodes + 2) += -_invWeights[0] * (-GBlock.block(0, 0, 1, _nNodes + 2)); // G_N,N		i=0, j=-1,...,N+1
						}
						if (cellIdx < _nCells) {
							dispBlock.block(_nNodes - 1, _nNodes - 1, 1, _nNodes) += _invWeights[_nNodes - 1] * (-0.5 * GBlock.block(_nNodes - 1, 0, 1, _nNodes)); // G_i,j+N+1		i=N, j=-1,...,N-1
							dispBlock(_nNodes - 1, 2 * _nNodes - 1) += _invWeights[_nNodes - 1] * (-0.5 * GBlock(_nNodes - 1, _nNodes) + 0.5 * GBlockRight(0, 0)); // G_i,j		i=N, j=N
							dispBlock(_nNodes - 1, 2 * _nNodes) += _invWeights[_nNodes - 1] * (-0.5 * GBlock(_nNodes - 1, _nNodes + 1) + 0.5 * GBlockRight(0, 1)); // G_i,j		i=N, j=N+1
							dispBlock.block(_nNodes - 1, 2 * _nNodes + 1, 1, _nNodes - 1) += _invWeights[_nNodes - 1] * (0.5 * GBlockRight.block(0, 2, 1, _nNodes - 1)); // G_0,j-N-1		i=N, j=N+2,...,2N+1
						}
						else if (cellIdx == _nCells) { // right boundary cell
							dispBlock.block(_nNodes - 1, _nNodes - 1, 1, _nNodes + 2) += _invWeights[_nNodes - 1] * (-GBlock.block(_nNodes - 1, 0, 1, _nNodes + 2)); // G_i,j+N+1		i=N, j=--1,...,N+1
						}

						dispBlock *= 2 / static_cast<double>(_deltaZ);
					}

					return -dispBlock; // *-1 for residual
				}

				/* ===================================================================================
				*   Residual functions
				* =================================================================================== */
				/**
				* @brief calculates the volume Integral of the auxiliary equation
				* @param [in] state state vector
				* @param [in, out] stateDer state derivative vector
				* @detail performs matrix-vector multiplication optimized depending on state types and stores the result in stateDer.
				*/
				template<typename StateType, typename ResidualType>
				void volumeIntegral(Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>>& state, Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>>& stateDer) {

					for (unsigned int Cell = 0; Cell < _nCells; Cell++) {

						// exploit Eigen3 performance if no mixed scalar types
						if constexpr (std::is_same_v<StateType, double>) {
							if constexpr (std::is_same_v<ResidualType, double>) {
								stateDer.segment(Cell * _nNodes, _nNodes) -= _polyDerM * state.segment(Cell * _nNodes, _nNodes);
							}
							else {
								stateDer.segment(Cell * _nNodes, _nNodes) -= (_polyDerM * state.segment(Cell * _nNodes, _nNodes)).template cast<ResidualType>();
							}
						}
						else { // both active types
							// todo use custom (mixed scalar-type) matrix vector multiplication?
							stateDer.segment(Cell * _nNodes, _nNodes) -= _polyDerM.template cast<active>() * state.segment(Cell * _nNodes, _nNodes);
						}
					}
				}
				template<typename StateType, typename ResidualType>
				void volumeIntegral(Eigen::Map<Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>>& state, Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>>& stateDer) {
					Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>> state_const(state.data(), state.size(), InnerStride<Dynamic>(1));
					volumeIntegral(state_const, stateDer);
				}
				/*
				 * @brief calculates the interface fluxes h* of Convection Dispersion equation
				 */
				template<typename StateType, typename ParamType>
				void InterfaceFlux(const StateType* C, ParamType _dispersion) {

					StateType* g = reinterpret_cast<StateType*>(_auxState);

					// component-wise strides
					unsigned int strideNode = _strideNode;
					unsigned int strideCell = _nNodes * strideNode;
					unsigned int strideNode_g = 1u;
					unsigned int strideCell_g = _nNodes * strideNode_g;

					// Conv.Disp. flux: h* = h*_conv + h*_disp = v c_up + 0.5 sqrt(D_ax) (S_l + S_r)

					if (_curVelocity >= 0.0) { // forward flow (upwind num. flux)
						// calculate inner interface fluxes
						for (unsigned int Cell = 1; Cell < _nCells; Cell++) {
							// h* = h*_conv + h*_disp
							_surfaceFlux[Cell] // inner interfaces
								= _curVelocity * (C[Cell * strideCell - strideNode]) // left cell (i.e. forward flow upwind)
								- 0.5 * (-2.0 / static_cast<ParamType>(_deltaZ)) * _dispersion *
								(g[Cell * strideCell_g - strideNode_g] // left cell
									+ g[Cell * strideCell_g]); // right cell
						}

						// boundary fluxes
						// inlet (left) boundary interface
						_surfaceFlux[0]
							= _curVelocity * _boundary[0];

						// outlet (right) boundary interface
						_surfaceFlux[_nCells]
							= _curVelocity * (C[_nCells * strideCell - strideNode])
							- 0.5 * (-2.0 / static_cast<ParamType>(_deltaZ)) * _dispersion *
							(g[_nCells * strideCell_g - strideNode_g] // last cell last node
								+ _boundary[3]); // right boundary value g
					}
					else { // backward flow (upwind num. flux)
						// calculate inner interface fluxes
						for (unsigned int Cell = 1; Cell < _nCells; Cell++) {
							// h* = h*_conv + h*_disp
							_surfaceFlux[Cell] // inner interfaces
								= _curVelocity * (C[Cell * strideCell]) // right cell (i.e. backward flow upwind)
								- 0.5 * (-2.0 / static_cast<ParamType>(_deltaZ)) * _dispersion *
								(g[Cell * strideCell_g - strideNode_g] // left cell
									+ g[Cell * strideCell_g]); // right cell
						}

						// boundary fluxes
						// inlet boundary interface
						_surfaceFlux[_nCells]
							= _curVelocity * _boundary[0];

						// outlet boundary interface
						_surfaceFlux[0]
							= _curVelocity * (C[0])
							- 0.5 * (-2.0 / static_cast<ParamType>(_deltaZ)) * _dispersion *
							(g[0] // first cell first node
								+ _boundary[2]); // left boundary value g
					}
					// apply inverse mapping jacobian (reference space)
					_surfaceFlux *= -2.0 / static_cast<ParamType>(_deltaZ);
				}
				/**
				 * @brief calculates and fills the surface flux values for auxiliary equation
				 * @param [in] C bulk liquid phase
				 * @param [in] strideNode node stride w.r.t. C
				 * @param [in] strideCell cell stride w.r.t. C
				 */
				template<typename StateType>
				void InterfaceFluxAuxiliary(const StateType* C, unsigned int strideNode, unsigned int strideCell) {

					// Auxiliary flux: c* = 0.5 (c_l + c_r)

					// calculate inner interface fluxes
					for (unsigned int Cell = 1; Cell < _nCells; Cell++) {
						_surfaceFlux[Cell] // left interfaces
							= 0.5 * (C[Cell * strideCell - strideNode] + // left node
								C[Cell * strideCell]); // right node
					}
					// calculate boundary interface fluxes

					_surfaceFlux[0] // left boundary interface
						= 0.5 * (C[0] + // boundary value
							C[0]); // first cell first node

					_surfaceFlux[(_nCells)] // right boundary interface
						= 0.5 * (C[_nCells * strideCell - strideNode] + // last cell last node
							C[_nCells * strideCell - strideNode]);// // boundary value
				}
				/**
				 * @brief calculates the string form surface Integral
				 * @param [in] state relevant state vector
				 * @param [in] stateDer state derivative vector the solution is added to
				 * @param [in] strideNode_state node stride w.r.t. state
				 * @param [in] strideCell_state cell stride w.r.t. state
				 * @param [in] strideNode_stateDer node stride w.r.t. stateDer
				 * @param [in] strideCell_stateDer cell stride w.r.t. stateDer
				 * @detail calculates stateDer = M^-1 * B * (state - state^*) and exploits LGL sparsity if applied
				 */
				template<typename StateType, typename ResidualType>
				void surfaceIntegral(const StateType* state, ResidualType* stateDer,
					unsigned int strideNode_state, unsigned int strideCell_state,
					unsigned int strideNode_stateDer, unsigned int strideCell_stateDer) {

					if (_exactInt) { // non-collocated integration -> dense mass matrix
						for (unsigned int Cell = 0; Cell < _nCells; Cell++) {
							// strong surface integral -> M^-1 B [state - state*]
							for (unsigned int Node = 0; Node < _nNodes; Node++) {
								stateDer[Cell * strideCell_stateDer + Node * strideNode_stateDer]
									-= static_cast<ResidualType>(_invMM(Node, 0) * (state[Cell * strideCell_state] - _surfaceFlux[Cell])
										- _invMM(Node, _polyDeg) * (state[Cell * strideCell_state + _polyDeg * strideNode_state] - _surfaceFlux[(Cell + 1)]));
							}
						}
					}
					else { // collocated numerical integration -> diagonal mass matrix
						for (unsigned int Cell = 0; Cell < _nCells; Cell++) {
							// strong surface integral -> M^-1 B [state - state*]
							stateDer[Cell * strideCell_stateDer] // first cell, node
								-= static_cast<ResidualType>(_invWeights[0]
									* (state[Cell * strideCell_state] - _surfaceFlux(Cell)));

							stateDer[Cell * strideCell_stateDer + _polyDeg * strideNode_stateDer] // last cell, node
								+= static_cast<ResidualType>(_invWeights[_polyDeg]
									* (state[Cell * strideCell_state + _polyDeg * strideNode_state] - _surfaceFlux(Cell + 1)));
						}
					}
				}
				/**
				 * @brief computes ghost nodes to implement boundary conditions
				 * @detail to implement Danckwert boundary conditions, we only need to set the solid wall BC values for auxiliary variable
				 */
				template<typename StateType>
				void calcBoundaryValues() {
					//cache.boundary[0] = c_in -> inlet DOF already set
					//_boundary[1] = (_velocity >= 0.0) ? C[_nPoints - 1] : C[0]; // c_r outlet not required in Danckwerts BC
					_boundary[2] = -reinterpret_cast<StateType*>(_auxState)[0]; // g_l left boundary (inlet/outlet for forward/backward flow)
					_boundary[3] = -reinterpret_cast<StateType*>(_auxState)[_nPoints - 1]; // g_r right boundary (outlet/inlet for forward/backward flow)
				}

				// ==========================================================================================================================================================  //
				// ========================================						DG Jacobian							=========================================================  //
				// ==========================================================================================================================================================  //

				/**
				 * @brief sets the sparsity pattern of the convection dispersion Jacobian for the nodal DG scheme
				 */
				int ConvDispNodalPattern(std::vector<T>& tripletList, const int offC = 0) {

					/*======================================================*/
					/*			Define Convection Jacobian Block			*/
					/*======================================================*/

					// Convection block [ d RHS_conv / d c ], also depends on upwind entry

					if (_curVelocity >= 0.0) { // forward flow upwind entry -> last node of previous cell
					// special inlet DOF treatment for inlet boundary cell (first cell)
						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								//tripletList.push_back(T(offC + comp * sComp + i * sNode, comp * sComp, 0.0)); // inlet DOFs not included in Jacobian
								for (unsigned int j = 1; j < _nNodes + 1; j++) {
									tripletList.push_back(T(offC + comp * strideColComp() + i * strideColNode(),
										offC + comp * strideColComp() + (j - 1) * strideColNode(),
										0.0));
								}
							}
						}
						for (unsigned int cell = 1; cell < _nCells; cell++) {
							for (unsigned int comp = 0; comp < _nComp; comp++) {
								for (unsigned int i = 0; i < _nNodes; i++) {
									for (unsigned int j = 0; j < _nNodes + 1; j++) {
										// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
										// col: jump over inlet DOFs and previous cells, go back one node, add component offset and go node strides from there for each convection block entry
										tripletList.push_back(T(offC + cell * strideColCell() + comp * strideColComp() + i * strideColNode(),
											offC + cell * strideColCell() - strideColNode() + comp * strideColComp() + j * strideColNode(),
											0.0));
									}
								}
							}
						}
					}
					else { // backward flow upwind entry -> first node of subsequent cell
						// special inlet DOF treatment for inlet boundary cell (last cell)
						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								// inlet DOFs not included in Jacobian
								for (unsigned int j = 0; j < _nNodes; j++) {
									tripletList.push_back(T(offC + (_nCells - 1) * strideColCell() + comp * strideColComp() + i * strideColNode(),
										offC + (_nCells - 1) * strideColCell() + comp * strideColComp() + j * strideColNode(),
										0.0));
								}
							}
						}
						for (unsigned int cell = 0; cell < _nCells - 1u; cell++) {
							for (unsigned int comp = 0; comp < _nComp; comp++) {
								for (unsigned int i = 0; i < _nNodes; i++) {
									for (unsigned int j = 0; j < _nNodes + 1; j++) {
										// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
										// col: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
										tripletList.push_back(T(offC + cell * strideColCell() + comp * strideColComp() + i * strideColNode(),
											offC + cell * strideColCell() + comp * strideColComp() + j * strideColNode(),
											0.0));
									}
								}
							}
						}
					}

					/*======================================================*/
					/*			Define Dispersion Jacobian Block			*/
					/*======================================================*/

					/*		Inner cell dispersion blocks		*/

					// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell

					// insert Blocks to Jacobian inner cells (only for _nCells >= 3)
					if (_nCells >= 3u) {
						for (unsigned int cell = 1; cell < _nCells - 1; cell++) {
							for (unsigned int comp = 0; comp < _nComp; comp++) {
								for (unsigned int i = 0; i < _nNodes; i++) {
									for (unsigned int j = 0; j < 3 * _nNodes; j++) {
										// pattern is more sparse than a nNodes x 3*nNodes block.
										if ((j >= _nNodes - 1 && j <= 2 * _nNodes) ||
											(i == 0 && j <= 2 * _nNodes) ||
											(i == _nNodes - 1 && j >= _nNodes - 1))
											// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each entry
											// col: jump over inlet DOFs and previous cells, go back one cell, add component offset and go node strides from there for each entry
											tripletList.push_back(T(offC + cell * strideColCell() + comp * strideColComp() + i * strideColNode(),
												offC + (cell - 1) * strideColCell() + comp * strideColComp() + j * strideColNode(),
												0.0));
									}
								}
							}
						}
					}

					/*				Boundary cell Dispersion blocks			*/

					if (_nCells != 1) { // Note: special case _nCells = 1 already set by advection block
						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								for (unsigned int j = _nNodes; j < 3 * _nNodes; j++) {
									// pattern is more sparse than a nNodes x 2*nNodes block.
									if ((j >= _nNodes - 1 && j <= 2 * _nNodes) ||
										(i == 0 && j <= 2 * _nNodes) ||
										(i == _nNodes - 1 && j >= _nNodes - 1))
										tripletList.push_back(T(offC + comp * strideColComp() + i * strideColNode(),
											offC + comp * strideColComp() + (j - _nNodes) * strideColNode(),
											0.0));
								}
							}
						}

						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								for (unsigned int j = 0; j < 2 * _nNodes; j++) {
									// pattern is more sparse than a nNodes x 2*nNodes block.
									if ((j >= _nNodes - 1 && j <= 2 * _nNodes) ||
										(i == 0 && j <= 2 * _nNodes) ||
										(i == _nNodes - 1 && j >= _nNodes - 1))
										tripletList.push_back(T(offC + (_nCells - 1) * strideColCell() + comp * strideColComp() + i * strideColNode(),
											offC + (_nCells - 1 - 1) * strideColCell() + comp * strideColComp() + j * strideColNode(),
											0.0));
								}
							}
						}
					}

					return 0;
				}

				/**
				* @brief sets the sparsity pattern of the convection dispersion Jacobian for the exact integration (her: modal) DG scheme
				*/
				int ConvDispModalPattern(std::vector<T>& tripletList, const int offC = 0) {

					/*======================================================*/
					/*			Define Convection Jacobian Block			*/
					/*======================================================*/

					// Convection block [ d RHS_conv / d c ], also depends on upwind entry

					if (_curVelocity >= 0.0) { // forward flow upwind entry -> last node of previous cell
						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								//tripletList.push_back(T(offC + comp * strideColComp() + i * strideColNode(), comp * strideColComp(), 0.0)); // inlet DOFs not included in Jacobian
								for (unsigned int j = 1; j < _nNodes + 1; j++) {
									tripletList.push_back(T(offC + comp * strideColComp() + i * strideColNode(),
										offC + comp * strideColComp() + (j - 1) * strideColNode(),
										0.0));
								}
							}
						}
						for (unsigned int cell = 1; cell < _nCells; cell++) {
							for (unsigned int comp = 0; comp < _nComp; comp++) {
								for (unsigned int i = 0; i < _nNodes; i++) {
									for (unsigned int j = 0; j < _nNodes + 1; j++) {
										// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
										// col: jump over inlet DOFs and previous cells, go back one node, add component offset and go node strides from there for each convection block entry
										tripletList.push_back(T(offC + cell * strideColCell() + comp * strideColComp() + i * strideColNode(),
											offC + cell * strideColCell() - strideColNode() + comp * strideColComp() + j * strideColNode(),
											0.0));
									}
								}
							}
						}
					}
					else { // backward flow upwind entry -> first node of subsequent cell
						// special inlet DOF treatment for inlet boundary cell (last cell)
						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								// inlet DOFs not included in Jacobian
								for (unsigned int j = 0; j < _nNodes; j++) {
									tripletList.push_back(T(offC + (_nCells - 1) * strideColCell() + comp * strideColComp() + i * strideColNode(),
										offC + (_nCells - 1) * strideColCell() + comp * strideColComp() + j * strideColNode(),
										0.0));
								}
							}
						}
						for (unsigned int cell = 0; cell < _nCells - 1u; cell++) {
							for (unsigned int comp = 0; comp < _nComp; comp++) {
								for (unsigned int i = 0; i < _nNodes; i++) {
									for (unsigned int j = 0; j < _nNodes + 1; j++) {
										// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
										// col: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
										tripletList.push_back(T(offC + cell * strideColCell() + comp * strideColComp() + i * strideColNode(),
											offC + cell * strideColCell() + comp * strideColComp() + j * strideColNode(),
											0.0));
									}
								}
							}
						}
					}

					/*======================================================*/
					/*			Define Dispersion Jacobian Block			*/
					/*======================================================*/

					/* Inner cells */
					if (_nCells >= 5u) {
						// Inner dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
						for (unsigned int cell = 2; cell < _nCells - 2; cell++) {
							for (unsigned int comp = 0; comp < _nComp; comp++) {
								for (unsigned int i = 0; i < _nNodes; i++) {
									for (unsigned int j = 0; j < 3 * _nNodes + 2; j++) {
										// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
										// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry
										tripletList.push_back(T(offC + cell * strideColCell() + comp * strideColComp() + i * strideColNode(),
											offC + cell * strideColCell() - (_nNodes + 1) * strideColNode() + comp * strideColComp() + j * strideColNode(),
											0.0));
									}
								}
							}
						}
					}

					/*		boundary cell neighbours		*/

					// left boundary cell neighbour
					if (_nCells >= 4u) {
						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								for (unsigned int j = 1; j < 3 * _nNodes + 2; j++) {
									// row: jump over inlet DOFs and previous cell, add component offset and go node strides from there for each dispersion block entry
									// col: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry. Also adjust for iterator j (-1)
									tripletList.push_back(T(offC + _nNodes * strideColNode() + comp * strideColComp() + i * strideColNode(),
										offC + comp * strideColComp() + (j - 1) * strideColNode(),
										0.0));
								}
							}
						}
					}
					else if (_nCells == 3u) { // special case: only depends on the two neighbouring cells
						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								for (unsigned int j = 1; j < 3 * _nNodes + 1; j++) {
									// row: jump over inlet DOFs and previous cell, add component offset and go node strides from there for each dispersion block entry
									// col: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry. Also adjust for iterator j (-1)
									tripletList.push_back(T(offC + _nNodes * strideColNode() + comp * strideColComp() + i * strideColNode(),
										offC + comp * strideColComp() + (j - 1) * strideColNode(),
										0.0));
								}
							}
						}
					}
					// right boundary cell neighbour
					if (_nCells >= 4u) {
						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								for (unsigned int j = 0; j < 3 * _nNodes + 2 - 1; j++) {
									// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
									// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry.
									tripletList.push_back(T(offC + (_nCells - 2) * strideColCell() + comp * strideColComp() + i * strideColNode(),
										offC + (_nCells - 2) * strideColCell() - (_nNodes + 1) * strideColNode() + comp * strideColComp() + j * strideColNode(),
										0.0));
								}
							}
						}
					}
					/*			boundary cells			*/

					// left boundary cell
					unsigned int end = 3u * _nNodes + 2u;
					if (_nCells == 1u) end = 2u * _nNodes + 1u;
					else if (_nCells == 2u) end = 3u * _nNodes + 1u;
					for (unsigned int comp = 0; comp < _nComp; comp++) {
						for (unsigned int i = 0; i < _nNodes; i++) {
							for (unsigned int j = _nNodes + 1; j < end; j++) {
								// row: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry
								// col: jump over inlet DOFs, add component offset, adjust for iterator j (-Nnodes-1) and go node strides from there for each dispersion block entry.
								tripletList.push_back(T(offC + comp * strideColComp() + i * strideColNode(),
									offC + comp * strideColComp() + (j - (_nNodes + 1)) * strideColNode(),
									0.0));
							}
						}
					}
					// right boundary cell
					if (_nCells >= 3u) {
						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								for (unsigned int j = 0; j < 2 * _nNodes + 1; j++) {
									// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
									// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry.
									tripletList.push_back(T(offC + (_nCells - 1) * strideColCell() + comp * strideColComp() + i * strideColNode(),
										offC + (_nCells - 1) * strideColCell() - (_nNodes + 1) * strideColNode() + comp * strideColComp() + j * strideColNode(),
										0.0));
								}
							}
						}
					}
					else if (_nCells == 2u) { // special case for _nCells == 2: depends only on left cell
						for (unsigned int comp = 0; comp < _nComp; comp++) {
							for (unsigned int i = 0; i < _nNodes; i++) {
								for (unsigned int j = 0; j < 2 * _nNodes; j++) {
									// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
									// col: jump over inlet DOFs and previous cells, go back one cell, add component offset and go node strides from there for each dispersion block entry.
									tripletList.push_back(T(offC + (_nCells - 1) * strideColCell() + comp * strideColComp() + i * strideColNode(),
										offC + (_nCells - 1) * strideColCell() - _nNodes * strideColNode() + comp * strideColComp() + j * strideColNode(),
										0.0));
								}
							}
						}
					}

					return 0;
				}
				/**
				* @brief analytically calculates the convection dispersion jacobian for the nodal DG scheme
				*/
				int calcConvDispCollocationDGSEMJacobian(Eigen::SparseMatrix<double, Eigen::RowMajor>& jacobian, Eigen::MatrixXd& jacInlet, const int offC = 0) {

					const int strideColBound = strideColNode() - _nComp;

					/*======================================================*/
					/*			Compute Dispersion Jacobian Block			*/
					/*======================================================*/

					/*		Inner cell dispersion blocks		*/

					if (_nCells >= 3u) {
						MatrixXd dispBlock = _DGjacAxDispBlocks[1];
						linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC + strideColCell()); // row iterator starting at second cell and component

						for (unsigned int cell = 1; cell < _nCells - 1; cell++) {
							for (unsigned int i = 0; i < dispBlock.rows(); i++, jacIt += strideColBound) {
								for (unsigned int comp = 0; comp < _nComp; comp++, ++jacIt) {
									for (unsigned int j = 0; j < dispBlock.cols(); j++) {
										// pattern is more sparse than a nNodes x 3*nNodes block.
										if ((j >= _nNodes - 1 && j <= 2 * _nNodes) ||
											(i == 0 && j <= 2 * _nNodes) ||
											(i == _nNodes - 1 && j >= _nNodes - 1))
											// row: iterator is at current node i and current component comp
											// col: start at previous cell and jump to node j
											jacIt[-strideColCell() + (j - i) * strideColNode()] = dispBlock(i, j) * static_cast<double>(currentDispersion(_curSection)[comp]);
									}
								}
							}
						}
					}

					/*				Boundary cell Dispersion blocks			*/

					/* left cell */
					MatrixXd dispBlock = _DGjacAxDispBlocks[0];

					if (_nCells != 1u) { // "standard" case
						linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC); // row iterator starting at first cell and component

						for (unsigned int i = 0; i < dispBlock.rows(); i++, jacIt += strideColBound) {
							for (unsigned int comp = 0; comp < _nComp; comp++, ++jacIt) {
								for (unsigned int j = _nNodes; j < dispBlock.cols(); j++) {
									// pattern is more sparse than a nNodes x 2*nNodes block.
									if ((j >= _nNodes - 1 && j <= 2 * _nNodes) ||
										(i == 0 && j <= 2 * _nNodes) ||
										(i == _nNodes - 1 && j >= _nNodes - 1))
										// row: iterator is at current node i and current component comp
										// col: jump to node j
										jacIt[((j - _nNodes) - i) * strideColNode()] = dispBlock(i, j) * static_cast<double>(currentDispersion(_curSection)[comp]);
								}
							}
						}
					}
					else { // special case
						linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC); // row iterator starting at first cell and component
						for (unsigned int i = 0; i < dispBlock.rows(); i++, jacIt += strideColBound) {
							for (unsigned int comp = 0; comp < _nComp; comp++, ++jacIt) {
								for (unsigned int j = _nNodes; j < _nNodes * 2u; j++) {
									// row: iterator is at current node i and current component comp
									// col: jump to node j
									jacIt[((j - _nNodes) - i) * strideColNode()] = dispBlock(i, j) * static_cast<double>(currentDispersion(_curSection)[comp]);
								}
							}
						}
					}

					/* right cell */
					if (_nCells != 1u) { // "standard" case
						dispBlock = _DGjacAxDispBlocks[std::min(_nCells, 3u) - 1];
						linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC + (_nCells - 1) * strideColCell()); // row iterator starting at last cell

						for (unsigned int i = 0; i < dispBlock.rows(); i++, jacIt += strideColBound) {
							for (unsigned int comp = 0; comp < _nComp; comp++, ++jacIt) {
								for (unsigned int j = 0; j < 2 * _nNodes; j++) {
									// pattern is more sparse than a nNodes x 2*nNodes block.
									if ((j >= _nNodes - 1 && j <= 2 * _nNodes) ||
										(i == 0 && j <= 2 * _nNodes) ||
										(i == _nNodes - 1 && j >= _nNodes - 1))
										// row: iterator is at current node i and current component comp
										// col: start at previous cell and jump to node j
										jacIt[-strideColCell() + (j - i) * strideColNode()] = dispBlock(i, j) * static_cast<double>(currentDispersion(_curSection)[comp]);
								}
							}
						}
					}

					/*======================================================*/
					/*			Compute Convection Jacobian Block			*/
					/*======================================================*/

					// Convection block [ d RHS_conv / d c ], also depends on first entry of previous cell
					MatrixXd convBlock = _DGjacAxConvBlock;
					linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC); // row iterator starting at first cell and component

					if (_curVelocity >= 0.0) { // forward flow upwind convection
						// special inlet DOF treatment for first cell (inlet boundary cell)
						jacInlet(0, 0) = static_cast<double>(_curVelocity) * convBlock(0, 0); // only first node depends on inlet concentration
						for (unsigned int i = 0; i < convBlock.rows(); i++, jacIt += strideColBound) {
							for (unsigned int comp = 0; comp < _nComp; comp++, ++jacIt) {
								//jacIt[0] = -convBlock(i, 0); // dependency on inlet DOFs is handled in _jacInlet
								for (unsigned int j = 1; j < convBlock.cols(); j++) {
									jacIt[((j - 1) - i) * strideColNode()] += static_cast<double>(_curVelocity) * convBlock(i, j);
								}
							}
						}
						// remaining cells
						for (unsigned int cell = 1; cell < _nCells; cell++) {
							for (unsigned int i = 0; i < convBlock.rows(); i++, jacIt += strideColBound) {
								for (unsigned int comp = 0; comp < _nComp; comp++, ++jacIt) {
									for (unsigned int j = 0; j < convBlock.cols(); j++) {
										// row: iterator is at current cell and component
										// col: start at previous cells last node and go to node j.
										jacIt[-strideColNode() + (j - i) * strideColNode()] += static_cast<double>(_curVelocity) * convBlock(i, j);
									}
								}
							}
						}
					}
					else { // backward flow upwind convection
						// non-inlet cells
						for (unsigned int cell = 0; cell < _nCells - 1u; cell++) {
							for (unsigned int i = 0; i < convBlock.rows(); i++, jacIt += strideColBound) {
								for (unsigned int comp = 0; comp < _nComp; comp++, ++jacIt) {
									for (unsigned int j = 0; j < convBlock.cols(); j++) {
										// row: iterator is at current cell and component
										// col: start at current cells first node and go to node j.
										jacIt[(j - i) * strideColNode()] += static_cast<double>(_curVelocity) * convBlock(i, j);
									}
								}
							}
						}
						// special inlet DOF treatment for last cell (inlet boundary cell)
						jacInlet(0, 0) = static_cast<double>(_curVelocity) * convBlock(convBlock.rows() - 1, convBlock.cols() - 1); // only last node depends on inlet concentration
						for (unsigned int i = 0; i < convBlock.rows(); i++, jacIt += strideColBound) {
							for (unsigned int comp = 0; comp < _nComp; comp++, ++jacIt) {
								for (unsigned int j = 0; j < convBlock.cols() - 1; j++) {
									jacIt[(j - i) * strideColNode()] += static_cast<double>(_curVelocity) * convBlock(i, j);
								}
							}
						}
					}

					return 0;
				}
				/**
				 * @brief inserts a liquid state block with different factors for components into the system jacobian
				 * @param [in] block (sub)block to be added
				 * @param [in] jac row iterator at first (i.e. upper) entry
				 * @param [in] offCol column to row offset (i.e. start at upper left corner of block)
				 * @param [in] nCells determines how often the block is added (diagonally)
				 * @param [in] Compfactor component dependend factors
				 */
				void insertCompDepLiquidJacBlock(Eigen::MatrixXd block, linalg::BandedEigenSparseRowIterator& jac, int offCol, unsigned int nCells, const active* Compfactor) {

					const int strideColBound = strideColNode() - _nComp;

					for (unsigned int cell = 0; cell < nCells; cell++) {
						for (unsigned int i = 0; i < block.rows(); i++, jac += strideColBound) {
							for (unsigned int comp = 0; comp < _nComp; comp++, ++jac) {
								for (unsigned int j = 0; j < block.cols(); j++) {
									// row: at current node component
									// col: jump to node j
									jac[(j - i) * strideColNode() + offCol] = block(i, j) * static_cast<double>(Compfactor[comp]);
								}
							}
						}
					}
				}
				/**
				 * @brief adds liquid state blocks for all components to the system jacobian
				 * @param [in] block to be added
				 * @param [in] jac row iterator at first (i.e. upper left) entry
				 * @param [in] column to row offset (i.e. start at upper left corner of block)
				 * @param [in] nCells determines how often the block is added (diagonally)
				 */
				void addLiquidJacBlock(Eigen::MatrixXd block, linalg::BandedEigenSparseRowIterator& jac, int offCol, unsigned int nCells) {

					unsigned int strideColBound = strideColNode() - _nComp;

					for (unsigned int cell = 0; cell < nCells; cell++) {
						for (unsigned int i = 0; i < block.rows(); i++, jac += strideColBound) {
							for (unsigned int comp = 0; comp < _nComp; comp++, ++jac) {
								for (unsigned int j = 0; j < block.cols(); j++) {
									// row: at current node component
									// col: jump to node j
									jac[(j - i) * strideColNode() + offCol] += block(i, j);
								}
							}
						}
					}
				}
				/**
				 * @brief analytically calculates the convection dispersion jacobian for the exact integration (here: modal) DG scheme
				 */
				int calcConvDispDGSEMJacobian(Eigen::SparseMatrix<double, Eigen::RowMajor>& jacobian, Eigen::MatrixXd& jacInlet, const int offC = 0) {

					/*======================================================*/
					/*			Compute Dispersion Jacobian Block			*/
					/*======================================================*/

					/* Inner cells (exist only if nCells >= 5) */
					if (_nCells >= 5) {
						linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC + strideColCell() * 2); // row iterator starting at third cell, first component
						// insert all (nCol - 4) inner cell blocks
						insertCompDepLiquidJacBlock(_DGjacAxDispBlocks[2], jacIt, -(strideColCell() + strideColNode()), _nCells - 4u, currentDispersion(_curSection));
					}

					/*	boundary cell neighbours (exist only if nCells >= 4)	*/
					if (_nCells >= 4) {
						linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC + strideColCell()); // row iterator starting at second cell, first component

						insertCompDepLiquidJacBlock(_DGjacAxDispBlocks[1].block(0, 1, _nNodes, 3 * _nNodes + 1), jacIt, -strideColCell(), 1u, currentDispersion(_curSection));

						jacIt += (_nCells - 4) * strideColCell(); // move iterator to preultimate cell (already at third cell)
						insertCompDepLiquidJacBlock(_DGjacAxDispBlocks[_nCells > 4 ? 3 : 2].block(0, 0, _nNodes, 3 * _nNodes + 1), jacIt, -(strideColCell() + strideColNode()), 1u, currentDispersion(_curSection));
					}

					/*			boundary cells (exist only if nCells >= 3)			*/
					if (_nCells >= 3) {

						linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC); // row iterator starting at first cell, first component

						insertCompDepLiquidJacBlock(_DGjacAxDispBlocks[0].block(0, _nNodes + 1, _nNodes, 2 * _nNodes + 1), jacIt, 0, 1u, currentDispersion(_curSection));

						jacIt += (_nCells - 2) * strideColCell(); // move iterator to last cell (already at second cell)
						insertCompDepLiquidJacBlock(_DGjacAxDispBlocks[std::min(_nCells, 5u) - 1u].block(0, 0, _nNodes, 2 * _nNodes + 1), jacIt, -(strideColCell() + strideColNode()), 1u, currentDispersion(_curSection));
					}

					/* For special cases nCells = 1, 2, 3, some cells still have to be treated separately*/

					if (_nCells == 1) {
						linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC); // row iterator starting at first cell, first component
						// insert the only block
						insertCompDepLiquidJacBlock(_DGjacAxDispBlocks[0].block(0, _nNodes + 1, _nNodes, _nNodes), jacIt, 0, 1u, currentDispersion(_curSection));
					}
					else if (_nCells == 2) {
						linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC); // row iterator starting at first cell, first component
						// left Bacobian block
						insertCompDepLiquidJacBlock(_DGjacAxDispBlocks[0].block(0, _nNodes + 1, _nNodes, 2 * _nNodes), jacIt, 0, 1u, currentDispersion(_curSection));
						// right Bacobian block, iterator is already moved to second cell
						insertCompDepLiquidJacBlock(_DGjacAxDispBlocks[1].block(0, 1, _nNodes, 2 * _nNodes), jacIt, -strideColCell(), 1u, currentDispersion(_curSection));
					}
					else if (_nCells == 3) {
						linalg::BandedEigenSparseRowIterator jacIt(jacobian, offC + strideColCell()); // row iterator starting at first cell, first component
						insertCompDepLiquidJacBlock(_DGjacAxDispBlocks[1].block(0, 1, _nNodes, 3 * _nNodes), jacIt, -strideColCell(), 1u, currentDispersion(_curSection));
					}

					/*======================================================*/
					/*			Compute Convection Jacobian Block			*/
					/*======================================================*/

					linalg::BandedEigenSparseRowIterator jac(jacobian, offC);

					if (_curVelocity >= 0.0) { // Forward flow
						// special inlet DOF treatment for inlet (first) cell
						jacInlet = static_cast<double>(_curVelocity) * _DGjacAxConvBlock.col(0); // only first cell depends on inlet concentration
						addLiquidJacBlock(static_cast<double>(_curVelocity) * _DGjacAxConvBlock.block(0, 1, _nNodes, _nNodes), jac, 0, 1);
						if (_nCells > 1) // iterator already moved to second cell
							addLiquidJacBlock(static_cast<double>(_curVelocity) * _DGjacAxConvBlock, jac, -strideColNode(), _nCells - 1);
					}
					else { // Backward flow
						// non-inlet cells first
						if (_nCells > 1)
							addLiquidJacBlock(static_cast<double>(_curVelocity) * _DGjacAxConvBlock, jac, 0, _nCells - 1);
						// special inlet DOF treatment for inlet (last) cell. Iterator already moved to last cell
						jacInlet = static_cast<double>(_curVelocity) * _DGjacAxConvBlock.col(_DGjacAxConvBlock.cols() - 1); // only last cell depends on inlet concentration
						addLiquidJacBlock(static_cast<double>(_curVelocity) * _DGjacAxConvBlock.block(0, 0, _nNodes, _nNodes), jac, 0, 1);
					}

					return 0;
				}
				// todo time derivative pattern
				///**
				//* @brief adds time derivative to the jacobian
				//*/
				//void addTimederJacobian(Eigen::SparseMatrix<double, Eigen::RowMajor>& jacobian, double alpha) {

				//	unsigned int offC = 0; // inlet DOFs not included in Jacobian

				//	// =================================================================================================== //
				//	//	 Time derivative Jacobian: d Residual / d y_t   												   //
				//	// =================================================================================================== //

				//	linalg::BandedEigenSparseRowIterator jac(jacobian, offC);

				//	double Beta = (1.0 - static_cast<double>(_totalPorosity)) / static_cast<double>(_totalPorosity);

				//	for (unsigned int point = 0; point < _disc.nPoints; point++) {
				//		for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				//			// d convDispRHS / d c_t
				//			jac[0] += alpha;

				//			if (_disc.nBound[comp]) { // either one or null; no loop necessary
				//				// d convDispRHS / d q_t
				//				jac[idxr.strideColLiquid() - comp + idxr.offsetBoundComp(comp)] += alpha * Beta;
				//			}
				//			++jac;
				//		}
				//		for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				//			if (_disc.nBound[comp]) { // either one or null; no loop over bound states necessary
				//				if (!_binding[0]->hasQuasiStationaryReactions()) {
				//					// d isotherm / d q_t
				//					jac[0] += alpha;
				//				}
				//				++jac;
				//			}
				//		}
				//	}
				//}
			};


			//// todo radial flow DG
			//class RadialConvectionDispersionOperatorBaseDG { };


			/**
			 * @brief Convection dispersion transport operator
			 * @details This class wraps AxialConvectionDispersionOperatorBaseDG and provides all the functionality it does.
			 * In addition, the Jacobian is stored and corresponding functions are provided
			 * (assembly, factorization, solution, retrieval).
			 *
			 * This class assumes that the first cell is offset by the number of components (inlet DOFs) in the global state vector.
			 */
			template <typename BaseOperator>
			class ConvectionDispersionOperatorDG
			{
			public:

				ConvectionDispersionOperatorDG();
				~ConvectionDispersionOperatorDG() CADET_NOEXCEPT;

				int requiredADdirs() const CADET_NOEXCEPT;

				void setFlowRates(const active& in, const active& out, const active& colPorosity) CADET_NOEXCEPT;

				bool configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper, unsigned int nComp, int polynomial_integration_mode, unsigned int nCells, unsigned int polyDeg, unsigned int strideNode);
				bool configure(UnitOpIdx unitOpIdx, IParameterProvider& paramProvider, std::unordered_map<ParameterId, active*>& parameters);
				bool notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const AdJacobianParams& adJac);

				int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, double* res, Eigen::SparseMatrix<double, Eigen::RowMajor>& jac);
				int residual(const IModel& model, double t, unsigned int secIdx, double const* y, double const* yDot, active* res, Eigen::SparseMatrix<double, Eigen::RowMajor>& jac);
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

				inline double cellLeftBound(unsigned int idx) const CADET_NOEXCEPT { return _baseOp.cellLeftBound(idx); }
				inline double relativeCoordinate(unsigned int idx) const CADET_NOEXCEPT { return _baseOp.relativeCoordinate(idx); }

				inline Eigen::SparseMatrix<double, Eigen::RowMajor>& jacobian() CADET_NOEXCEPT { return _jacC; }
				inline const Eigen::SparseMatrix<double, Eigen::RowMajor>& jacobian() const CADET_NOEXCEPT { return _jacC; }

				inline Eigen::SparseMatrix<double, Eigen::RowMajor>& jacobianDisc() CADET_NOEXCEPT { return _jacCdisc; }
				inline const Eigen::SparseMatrix<double, Eigen::RowMajor>& jacobianDisc() const CADET_NOEXCEPT { return _jacCdisc; }

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

				Eigen::MatrixXd _jacInlet; //!< inlet Jacobian
				Eigen::SparseMatrix<double, Eigen::RowMajor> _jacC; //!< Jacobian
				Eigen::SparseMatrix<double, Eigen::RowMajor> _jacCdisc; //!< Jacobian with time derivatives from BDF method

				// Indexer functionality
				// Offsets
				inline int offsetC() const CADET_NOEXCEPT { return _baseOp.nComp(); }

			};

			extern template class ConvectionDispersionOperatorDG<AxialConvectionDispersionOperatorBaseDG>;
			//extern template class ConvectionDispersionOperatorDG<RadialConvectionDispersionOperatorBaseDG>; // todo radial flow DG

			typedef ConvectionDispersionOperatorDG<AxialConvectionDispersionOperatorBaseDG> AxialConvectionDispersionOperatorDG;
			//typedef ConvectionDispersionOperatorDG<RadialConvectionDispersionOperatorBaseDG> RadialConvectionDispersionOperatorDG; // todo radial flow DG

		} // namespace parts
	} // namespace model
} // namespace cadet

#endif  // LIBCADET_CONVECTIONDISPERSIONOPERATORDG_HPP_