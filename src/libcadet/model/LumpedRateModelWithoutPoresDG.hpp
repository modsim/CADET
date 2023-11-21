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
 * Defines the lumped rate model without pores (LRM) using a Discontinous Galerkin (DG) scheme.
 */

#ifndef LIBCADET_LUMPEDRATEMODELWITHOUTPORESDG_HPP_
#define LIBCADET_LUMPEDRATEMODELWITHOUTPORESDG_HPP_

#include "BindingModel.hpp"
#include "ParallelSupport.hpp"

#include "UnitOperationBase.hpp"
#include "cadet/SolutionExporter.hpp"
#include "model/parts/ConvectionDispersionOperator.hpp"
#include "AutoDiff.hpp"
#include "linalg/SparseMatrix.hpp"
#include "linalg/BandMatrix.hpp"
#include "linalg/BandedEigenSparseRowIterator.hpp"
#include "linalg/Gmres.hpp"
#include "Memory.hpp"
#include "model/ModelUtils.hpp"

#include <array>
#include <vector>

#include "Benchmark.hpp"
#include <Eigen/Dense> // use LA lib Eigen for Matrix operations
#include <Eigen/Sparse>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>
#endif

using namespace Eigen;

namespace cadet
{

	namespace model
	{

		/**
		* @brief Lumped rate model of liquid column chromatography without pores
		* @details See @cite Guiochon2006, @cite Gu1995, @cite Felinger2004
		*
		* @f[\begin{align}
				\frac{\partial c_i}{\partial t} + \frac{1 - \varepsilon_t}{\varepsilon_t} \frac{\partial q_{i}}{\partial t} &= - u \frac{\partial c_i}{\partial z} + D_{\text{ax},i} \frac{\partial^2 c_i}{\partial z^2} \\
				a \frac{\partial q_i}{\partial t} &= f_{\text{iso}}(c, q)
			\end{align} @f]
		* Danckwerts boundary conditions (see @cite Danckwerts1953)
			@f[ \begin{align}
				u c_{\text{in},i}(t) &= u c_i(t,0) - D_{\text{ax},i} \frac{\partial c_i}{\partial z}(t,0) \\
				\frac{\partial c_i}{\partial z}(t,L) &= 0
			\end{align} @f]
		* Methods are described in @cite Breuer2023 (DGSEM discretization), @cite Puttmann2013 @cite Puttmann2016 (forward sensitivities, AD, band compression)
		*/
		class LumpedRateModelWithoutPoresDG : public UnitOperationBase
		{
		public:

			LumpedRateModelWithoutPoresDG(UnitOpIdx unitOpIdx);
			virtual ~LumpedRateModelWithoutPoresDG() CADET_NOEXCEPT;

			virtual unsigned int numDofs() const CADET_NOEXCEPT;
			virtual unsigned int numPureDofs() const CADET_NOEXCEPT;
			virtual bool usesAD() const CADET_NOEXCEPT;
			virtual unsigned int requiredADdirs() const CADET_NOEXCEPT;

			virtual UnitOpIdx unitOperationId() const CADET_NOEXCEPT { return _unitOpIdx; }
			virtual unsigned int numComponents() const CADET_NOEXCEPT { return _disc.nComp; }
			virtual void setFlowRates(active const* in, active const* out) CADET_NOEXCEPT;
			virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
			virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
			virtual bool canAccumulate() const CADET_NOEXCEPT { return false; }

			static const char* identifier() { return "LUMPED_RATE_MODEL_WITHOUT_PORES_DG"; }
			virtual const char* unitOperationName() const CADET_NOEXCEPT { return identifier(); }

			virtual bool configureModelDiscretization(IParameterProvider& paramProvider, IConfigHelper& helper);
			virtual bool configure(IParameterProvider& paramProvider);
			virtual void notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac);

			virtual void useAnalyticJacobian(const bool analyticJac);

			virtual void reportSolution(ISolutionRecorder& recorder, double const* const solution) const;
			virtual void reportSolutionStructure(ISolutionRecorder& recorder) const;

			virtual int residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem);

			virtual int residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem);
			virtual int residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem);
			virtual int residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem);

			virtual int residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState,
				const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS, active const* adRes,
				double* const tmp1, double* const tmp2, double* const tmp3);

			virtual int linearSolve(double t, double alpha, double tol, double* const rhs, double const* const weight,
				const ConstSimulationState& simState);

			virtual void prepareADvectors(const AdJacobianParams& adJac) const;

			virtual void applyInitialCondition(const SimulationState& simState) const;
			virtual void readInitialCondition(IParameterProvider& paramProvider);

			virtual void consistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem);
			virtual void consistentInitialTimeDerivative(const SimulationTime& simTime, double const* vecStateY, double* const vecStateYdot, util::ThreadLocalStorage& threadLocalMem);

			virtual void initializeSensitivityStates(const std::vector<double*>& vecSensY) const;
			virtual void consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
				std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem);

			virtual void leanConsistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem);
			virtual void leanConsistentInitialTimeDerivative(double t, double const* const vecStateY, double* const vecStateYdot, double* const res, util::ThreadLocalStorage& threadLocalMem);

			virtual void leanConsistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
				std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem);

			virtual bool hasInlet() const CADET_NOEXCEPT { return true; }
			virtual bool hasOutlet() const CADET_NOEXCEPT { return true; }

			virtual unsigned int localOutletComponentIndex(unsigned int port) const CADET_NOEXCEPT;
			virtual unsigned int localOutletComponentStride(unsigned int port) const CADET_NOEXCEPT;
			virtual unsigned int localInletComponentIndex(unsigned int port) const CADET_NOEXCEPT;
			virtual unsigned int localInletComponentStride(unsigned int port) const CADET_NOEXCEPT;

			virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size);
			virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections) { }

			virtual void expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut);

			virtual void multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret);
			virtual void multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret);

			inline void multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double* ret)
			{
				multiplyWithJacobian(simTime, simState, yS, 1.0, 0.0, ret);
			}

			virtual bool setParameter(const ParameterId& pId, double value);
			virtual bool setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue);
			virtual void setSensitiveParameterValue(const ParameterId& id, double value);

			virtual unsigned int threadLocalMemorySize() const CADET_NOEXCEPT;


#ifdef CADET_BENCHMARK_MODE
			virtual std::vector<double> benchmarkTimings() const
			{
				return std::vector<double>({
					static_cast<double>(numDofs()),
					_timerResidual.totalElapsedTime(),
					_timerResidualPar.totalElapsedTime(),
					_timerResidualSens.totalElapsedTime(),
					_timerResidualSensPar.totalElapsedTime(),
					_timerConsistentInit.totalElapsedTime(),
					_timerConsistentInitPar.totalElapsedTime(),
					_timerLinearSolve.totalElapsedTime()
					});
			}

			virtual char const* const* benchmarkDescriptions() const
			{
				static const char* const desc[] = {
					"DOFs",
					"Residual",
					"ResidualPar",
					"ResidualSens",
					"ResidualSensPar",
					"ConsistentInit",
					"ConsistentInitPar",
					"LinearSolve"
				};
				return desc;
			}
#endif

		protected:

			class Indexer;

			int residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem, bool updateJacobian, bool paramSensitivity);

			template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
			int residualImpl(double t, unsigned int secIdx, StateType const* const y, double const* const yDot, ResidualType* const res, util::ThreadLocalStorage& threadLocalMem);

			void extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset);

			void assembleDiscretizedJacobian(double alpha, const Indexer& idxr);
			void addTimeDerivativeToJacobianNode(linalg::BandedEigenSparseRowIterator& jac, const Indexer& idxr, double alpha, double invBetaP) const;

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
			void checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const;
#endif

			class Discretization
			{
			public:
				unsigned int nComp; //!< Number of components
				unsigned int nCol; //!< Number of column cells
				unsigned int polyDeg; //!< polynomial degree
				unsigned int nNodes; //!< Number of nodes per cell
				unsigned int nPoints; //!< Number of discrete Points
				double deltaZ; //!< cell spacing
				bool exactInt;	//!< 1 for exact integration, 0 for LGL quadrature
				Eigen::VectorXd nodes; //!< Array with positions of nodes in reference element
				Eigen::MatrixXd polyDerM; //!< Array with polynomial derivative Matrix
				Eigen::VectorXd invWeights; //!< Array with weights for numerical quadrature of size nNodes
				Eigen::MatrixXd invMM; //!< dense inverse mass matrix for exact integration

				Eigen::MatrixXd* DGjacAxDispBlocks; //!< axial dispersion blocks of DG jacobian (unique blocks only)
				Eigen::MatrixXd DGjacAxConvBlock; //!< axial convection block of DG jacobian

				unsigned int* nBound; //!< Array with number of bound states for each component
				unsigned int* boundOffset; //!< Array with offset to the first bound state of each component in the solid phase
				unsigned int strideBound; //!< Total number of bound states

				// vgl. convDispOperator
				Eigen::VectorXd dispersion; //!< Column dispersion (may be section and component dependent)
				bool _dispersionCompIndep; //!< Determines whether dispersion is component independent
				double velocity; //!< Interstitial velocity (may be section dependent) \f$ u \f$
				int curSection; //!< current section index

				double length_;
				double porosity;

				Eigen::VectorXd g; //!< auxiliary variable
				Eigen::VectorXd h; //!< auxiliary substitute
				Eigen::VectorXd surfaceFlux; //!< stores the surface flux values
				Eigen::Vector4d boundary; //!< stores the boundary values from Danckwert boundary conditions

				bool newStaticJac; //!< determines wether static analytical jacobian needs to be computed (every section)

				/**
				* @brief computes LGL nodes, integration weights, polynomial derivative matrix
				*/
				void initializeDG() {

					nNodes = polyDeg + 1;
					nPoints = nNodes * nCol;
					// Allocate space for DG discretization
					nodes.resize(nNodes);
					nodes.setZero();
					invWeights.resize(nNodes);
					invWeights.setZero();
					polyDerM.resize(nNodes, nNodes);
					polyDerM.setZero();
					invMM.resize(nNodes, nNodes);
					invMM.setZero();

					g.resize(nPoints);
					g.setZero();
					h.resize(nPoints);
					h.setZero();
					boundary.setZero();
					surfaceFlux.resize(nCol + 1);
					surfaceFlux.setZero();

					newStaticJac = true;

					lglNodesWeights();
					invMMatrix();
					derivativeMatrix();
				}

				void initializeDGjac() {
					DGjacAxDispBlocks = new MatrixXd[(exactInt ? std::min(nCol, 5u) : std::min(nCol, 3u))];
					// we only need unique dispersion blocks, which are given by cells 1, 2, nCol for inexact integration DG and by cells 1, 2, 3, nCol-1, nCol for eaxct integration DG
					DGjacAxDispBlocks[0] = DGjacobianDispBlock(1);
					if (nCol > 1)
						DGjacAxDispBlocks[1] = DGjacobianDispBlock(2);
					if (nCol > 2 && exactInt)
						DGjacAxDispBlocks[2] = DGjacobianDispBlock(3);
					else if (nCol > 2 && !exactInt)
						DGjacAxDispBlocks[2] = DGjacobianDispBlock(nCol);
					if (exactInt && nCol > 3)
						DGjacAxDispBlocks[3] = DGjacobianDispBlock(std::max(4u, nCol - 1u));
					if (exactInt && nCol > 4)
						DGjacAxDispBlocks[4] = DGjacobianDispBlock(nCol);

					DGjacAxConvBlock = DGjacobianConvBlock();
				}

			private:

				/* ===================================================================================
				*   Polynomial Basis operators and auxiliary functions
				* =================================================================================== */

				/**
				* @brief computes the Legendre polynomial L_N and q = L_N+1 - L_N-2 and q' at point x
				* @param [in] polyDeg polynomial degree of spatial Discretization
				* @param [in] x evaluation point
				* @param [in] L <- L(x)
				* @param [in] q <- q(x) = L_N+1 (x) - L_N-2(x)
				* @param [in] qder <- q'(x) = [L_N+1 (x) - L_N-2(x)]'
				*/
				void qAndL(const double x, double& L, double& q, double& qder) {
					// auxiliary variables (Legendre polynomials)
					double L_2 = 1.0;
					double L_1 = x;
					double Lder_2 = 0.0;
					double Lder_1 = 1.0;
					double Lder = 0.0;
					for (double k = 2; k <= polyDeg; k++) { // note that this function is only called for polyDeg >= 2.
						L = ((2 * k - 1) * x * L_1 - (k - 1) * L_2) / k;
						Lder = Lder_2 + (2 * k - 1) * L_1;
						L_2 = L_1;
						L_1 = L;
						Lder_2 = Lder_1;
						Lder_1 = Lder;
					}
					q = ((2.0 * polyDeg + 1) * x * L - polyDeg * L_2) / (polyDeg + 1.0) - L_2;
					qder = Lder_1 + (2.0 * polyDeg + 1) * L_1 - Lder_2;
				}

				/**
				 * @brief computes the Legendre-Gauss-Lobatto nodes and (inverse) quadrature weights
				 * @detail inexact LGL-quadrature leads to a diagonal mass matrix (mass lumping), defined by the quadrature weights
				 */
				void lglNodesWeights() {
					// tolerance and max #iterations for Newton iteration
					int nIterations = 10;
					double tolerance = 1e-15;
					// Legendre polynomial and derivative
					double L = 0;
					double q = 0;
					double qder = 0;
					switch (polyDeg) {
					case 0:
						throw std::invalid_argument("Polynomial degree must be at least 1 !");
						break;
					case 1:
						nodes[0] = -1;
						invWeights[0] = 1;
						nodes[1] = 1;
						invWeights[1] = 1;
						break;
					default:
						nodes[0] = -1;
						nodes[polyDeg] = 1;
						invWeights[0] = 2.0 / (polyDeg * (polyDeg + 1.0));
						invWeights[polyDeg] = invWeights[0];
						// use symmetrie, only compute half of points and weights
						for (unsigned int j = 1; j <= floor((polyDeg + 1) / 2) - 1; j++) {
							//  first guess for Newton iteration
							nodes[j] = -cos(M_PI * (j + 0.25) / polyDeg - 3 / (8.0 * polyDeg * M_PI * (j + 0.25)));
							// Newton iteration to find roots of Legendre Polynomial
							for (unsigned int k = 0; k <= nIterations; k++) {
								qAndL(nodes[j], L, q, qder);
								nodes[j] = nodes[j] - q / qder;
								if (abs(q / qder) <= tolerance * abs(nodes[j])) {
									break;
								}
							}
							// calculate weights
							qAndL(nodes[j], L, q, qder);
							invWeights[j] = 2.0 / (polyDeg * (polyDeg + 1.0) * pow(L, 2.0));
							nodes[polyDeg - j] = -nodes[j]; // copy to second half of points and weights
							invWeights[polyDeg - j] = invWeights[j];
						}
					}
					if (polyDeg % 2 == 0) { // for even polyDeg we have an odd number of points which include 0.0
						qAndL(0.0, L, q, qder);
						nodes[polyDeg / 2] = 0;
						invWeights[polyDeg / 2] = 2.0 / (polyDeg * (polyDeg + 1.0) * pow(L, 2.0));
					}
					// inverse the weights
					invWeights = invWeights.cwiseInverse();
				}

				/**
				 * @brief computation of barycentric weights for fast polynomial evaluation
				 * @param [in] baryWeights vector to store barycentric weights. Must already be initialized with ones!
				 */
				void barycentricWeights(Eigen::VectorXd& baryWeights) {
					for (unsigned int j = 1; j <= polyDeg; j++) {
						for (unsigned int k = 0; k <= j - 1; k++) {
							baryWeights[k] = baryWeights[k] * (nodes[k] - nodes[j]) * 1.0;
							baryWeights[j] = baryWeights[j] * (nodes[j] - nodes[k]) * 1.0;
						}
					}
					for (unsigned int j = 0; j <= polyDeg; j++) {
						baryWeights[j] = 1 / baryWeights[j];
					}
				}

				/**
				 * @brief computation of nodal (lagrange) polynomial derivative matrix
				 */
				void derivativeMatrix() {
					Eigen::VectorXd baryWeights = Eigen::VectorXd::Ones(polyDeg + 1u);
					barycentricWeights(baryWeights);
					for (unsigned int i = 0; i <= polyDeg; i++) {
						for (unsigned int j = 0; j <= polyDeg; j++) {
							if (i != j) {
								polyDerM(i, j) = baryWeights[j] / (baryWeights[i] * (nodes[i] - nodes[j]));
								polyDerM(i, i) += -polyDerM(i, j);
							}
						}
					}
				}

				/**
				 * @brief factor to normalize legendre polynomials
				 */
				double orthonFactor(int polyDeg) {

					double n = static_cast<double> (polyDeg);
					// alpha = beta = 0 to get legendre polynomials as special case from jacobi polynomials.
					double a = 0.0;
					double b = 0.0;
					return std::sqrt(((2.0 * n + a + b + 1.0) * std::tgamma(n + 1.0) * std::tgamma(n + a + b + 1.0))
						/ (std::pow(2.0, a + b + 1.0) * std::tgamma(n + a + 1.0) * std::tgamma(n + b + 1.0)));
				}

				/**
				 * @brief calculates the Vandermonde matrix of the normalized legendre polynomials
				 */
				Eigen::MatrixXd getVandermonde_LEGENDRE() {

					Eigen::MatrixXd V(nodes.size(), nodes.size());

					double alpha = 0.0;
					double beta = 0.0;

					// degree 0
					V.block(0, 0, nNodes, 1) = VectorXd::Ones(nNodes) * orthonFactor(0);
					// degree 1
					for (int node = 0; node < static_cast<int>(nNodes); node++) {
						V(node, 1) = nodes[node] * orthonFactor(1);
					}

					for (int deg = 2; deg <= static_cast<int>(polyDeg); deg++) {

						for (int node = 0; node < static_cast<int>(nNodes); node++) {

							double orthn_1 = orthonFactor(deg) / orthonFactor(deg - 1);
							double orthn_2 = orthonFactor(deg) / orthonFactor(deg - 2);

							double fac_1 = ((2.0 * deg - 1.0) * 2.0 * deg * (2.0 * deg - 2.0) * nodes[node]) / (2.0 * deg * deg * (2.0 * deg - 2.0));
							double fac_2 = (2.0 * (deg - 1.0) * (deg - 1.0) * 2.0 * deg) / (2.0 * deg * deg * (2.0 * deg - 2.0));

							V(node, deg) = orthn_1 * fac_1 * V(node, deg - 1) - orthn_2 * fac_2 * V(node, deg - 2);

						}

					}

					return V;
				}
				/**
				* @brief calculates mass matrix for exact polynomial integration
				* @detail exact polynomial integration leads to a full mass matrix
				*/
				void invMMatrix() {
					invMM = (getVandermonde_LEGENDRE() * (getVandermonde_LEGENDRE().transpose()));
				}
				/**
				 * @brief calculates the convection part of the DG jacobian
				 */
				MatrixXd DGjacobianConvBlock() {

					// Convection block [ d RHS_conv / d c ], additionally depends on upwind flux part from corresponding neighbour cell
					MatrixXd convBlock = MatrixXd::Zero(nNodes, nNodes + 1);

					if (velocity >= 0.0) { // forward flow -> Convection block additionally depends on last entry of previous cell
						convBlock.block(0, 1, nNodes, nNodes) -= polyDerM;

						if (exactInt) {
							convBlock.block(0, 0, nNodes, 1) += invMM.block(0, 0, nNodes, 1);
							convBlock.block(0, 1, nNodes, 1) -= invMM.block(0, 0, nNodes, 1);
						}
						else {
							convBlock(0, 0) += invWeights[0];
							convBlock(0, 1) -= invWeights[0];
						}
					}
					else { // backward flow -> Convection block additionally depends on first entry of subsequent cell
						convBlock.block(0, 0, nNodes, nNodes) -= polyDerM;

						if (exactInt) {
							convBlock.block(0, nNodes - 1, nNodes, 1) += invMM.block(0, nNodes - 1, nNodes, 1);
							convBlock.block(0, nNodes, nNodes, 1) -= invMM.block(0, nNodes - 1, nNodes, 1);
						}
						else {
							convBlock(nNodes - 1, nNodes - 1) += invWeights[nNodes - 1];
							convBlock(nNodes - 1, nNodes)     -= invWeights[nNodes - 1];
						}
					}
					convBlock *= 2 / deltaZ;

					return -convBlock; // *-1 for residual
				}
				/**
				 * @brief calculates the DG Jacobian auxiliary block
				 * @param [in] exInt true if exact integration DG scheme
				 * @param [in] cellIdx cell index
				 */
				MatrixXd getGBlock(unsigned int cellIdx) {

					// Auxiliary Block [ d g(c) / d c ], additionally depends on boundary entries of neighbouring cells
					MatrixXd gBlock = MatrixXd::Zero(nNodes, nNodes + 2);
					gBlock.block(0, 1, nNodes, nNodes) = polyDerM;
					if (exactInt) {
						if (cellIdx != 1 && cellIdx != nCol) {
							gBlock.block(0, 0, nNodes, 1) -= 0.5 * invMM.block(0, 0, nNodes, 1);
							gBlock.block(0, 1, nNodes, 1) += 0.5 * invMM.block(0, 0, nNodes, 1);
							gBlock.block(0, nNodes, nNodes, 1) -= 0.5 * invMM.block(0, nNodes - 1, nNodes, 1);
							gBlock.block(0, nNodes + 1, nNodes, 1) += 0.5 * invMM.block(0, nNodes - 1, nNodes, 1);
						}
						else if (cellIdx == 1) { // left
							if (cellIdx == nCol)
								return gBlock * 2 / deltaZ;
							;
							gBlock.block(0, nNodes, nNodes, 1) -= 0.5 * invMM.block(0, nNodes - 1, nNodes, 1);
							gBlock.block(0, nNodes + 1, nNodes, 1) += 0.5 * invMM.block(0, nNodes - 1, nNodes, 1);
						}
						else if (cellIdx == nCol) { // right
							gBlock.block(0, 0, nNodes, 1) -= 0.5 * invMM.block(0, 0, nNodes, 1);
							gBlock.block(0, 1, nNodes, 1) += 0.5 * invMM.block(0, 0, nNodes, 1);
						}
						else if (cellIdx == 0 || cellIdx == nCol + 1) {
							gBlock.setZero();
						}
						gBlock *= 2 / deltaZ;
					}
					else {
						if (cellIdx == 0 || cellIdx == nCol + 1)
							return MatrixXd::Zero(nNodes, nNodes + 2);

						gBlock(0, 0) -= 0.5 * invWeights[0];
						gBlock(0, 1) += 0.5 * invWeights[0];
						gBlock(nNodes - 1, nNodes) -= 0.5 * invWeights[nNodes - 1];
						gBlock(nNodes - 1, nNodes + 1) += 0.5 * invWeights[nNodes - 1];
						gBlock *= 2 / deltaZ;

						if (cellIdx == 1) {
							// adjust auxiliary Block [ d g(c) / d c ] for left boundary cell
							gBlock(0, 1) -= 0.5 * invWeights[0] * 2 / deltaZ;
							if (cellIdx == nCol) { // adjust for special case one cell
								gBlock(0, 0) += 0.5 * invWeights[0] * 2 / deltaZ;
								gBlock(nNodes - 1, nNodes + 1) -= 0.5 * invWeights[nNodes - 1] * 2 / deltaZ;
								gBlock(nNodes - 1, nNodes) += 0.5 * invWeights[polyDeg] * 2 / deltaZ;
							}
						}
						else if (cellIdx == nCol) {
							// adjust auxiliary Block [ d g(c) / d c ] for right boundary cell
							gBlock(nNodes - 1, nNodes) += 0.5 * invWeights[polyDeg] * 2 / deltaZ;
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
				Eigen::MatrixXd auxBlockGstar(unsigned int cellIdx, MatrixXd leftG, MatrixXd middleG, MatrixXd rightG) {

					// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
					MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes + 2);
					// NOTE: N = polyDeg
					// indices  gStarDC    :     0   ,   1   , ..., nNodes; nNodes+1, ..., 2 * nNodes;	2*nNodes+1, ..., 3 * nNodes; 3*nNodes+1
					// derivative index j  : -(N+1)-1, -(N+1),... ,  -1   ;   0     , ...,		N	 ;	  N + 1	  , ..., 2N + 2    ; 2(N+1) +1
					// auxiliary block [d g^* / d c]
					if (cellIdx != 1) {
						gStarDC.block(0, nNodes, 1, nNodes + 2) += middleG.block(0, 0, 1, nNodes + 2);
						gStarDC.block(0, 0, 1, nNodes + 2) += leftG.block(nNodes - 1, 0, 1, nNodes + 2);
					}
					if (cellIdx != nCol) {
						gStarDC.block(nNodes - 1, nNodes, 1, nNodes + 2) += middleG.block(nNodes - 1, 0, 1, nNodes + 2);
						gStarDC.block(nNodes - 1, 2 * nNodes, 1, nNodes + 2) += rightG.block(0, 0, 1, nNodes + 2);
					}
					gStarDC *= 0.5;

					return gStarDC;
				}

				Eigen::MatrixXd getBMatrix() {

					MatrixXd B = MatrixXd::Zero(nNodes, nNodes);
					B(0, 0) = -1.0;
					B(nNodes - 1, nNodes - 1) = 1.0;

					return B;
				}

				/**
				 * @brief calculates the dispersion part of the DG jacobian
				 * @param [in] exInt true if exact integration DG scheme
				 * @param [in] cellIdx cell index
				 */
				MatrixXd DGjacobianDispBlock(unsigned int cellIdx) {

					int offC = 0; // inlet DOFs not included in Jacobian

					MatrixXd dispBlock;

					if (exactInt) {

						// Inner dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
						dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2);

						MatrixXd B = getBMatrix(); // "Lifting" matrix
						MatrixXd gBlock = getGBlock(cellIdx); // current cell auxiliary block matrix
						MatrixXd gStarDC = auxBlockGstar(cellIdx, getGBlock(cellIdx - 1), gBlock, getGBlock(cellIdx + 1)); // Numerical flux block

						//  indices  dispBlock :   0	 ,   1   , ..., nNodes;	nNodes+1, ..., 2 * nNodes;	2*nNodes+1, ..., 3 * nNodes; 3*nNodes+1
						//	derivative index j  : -(N+1)-1, -(N+1),...,	 -1	  ;   0     , ...,		N	 ;	  N + 1	  , ..., 2N + 2    ; 2(N+1) +1
						dispBlock.block(0, nNodes, nNodes, nNodes + 2) += polyDerM * gBlock - invMM * B * gBlock;
						dispBlock += invMM * B * gStarDC;
						dispBlock *= 2 / deltaZ;
					}
					else { // inexact integration collocation DGSEM

						dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes);
						MatrixXd GBlockLeft = getGBlock(cellIdx - 1);
						MatrixXd GBlock = getGBlock(cellIdx);
						MatrixXd GBlockRight = getGBlock(cellIdx + 1);

						// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell
						// NOTE: N = polyDeg
						// cell indices :  0  , ..., nNodes - 1;	nNodes, ..., 2 * nNodes - 1;	2 * nNodes, ..., 3 * nNodes - 1
						//			 j  : -N-1, ..., -1		  ; 0     , ..., N			   ;	N + 1, ..., 2N + 1
						dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) = polyDerM * GBlock;

						if (cellIdx > 1) {
							dispBlock(0, nNodes - 1) += -invWeights[0] * (-0.5 * GBlock(0, 0) + 0.5 * GBlockLeft(nNodes - 1, nNodes)); // G_N,N		i=0, j=-1
							dispBlock(0, nNodes) += -invWeights[0] * (-0.5 * GBlock(0, 1) + 0.5 * GBlockLeft(nNodes - 1, nNodes + 1)); // G_N,N+1	i=0, j=0
							dispBlock.block(0, nNodes + 1, 1, nNodes) += -invWeights[0] * (-0.5 * GBlock.block(0, 2, 1, nNodes)); // G_i,j		i=0, j=1,...,N+1
							dispBlock.block(0, 0, 1, nNodes - 1) += -invWeights[0] * (0.5 * GBlockLeft.block(nNodes - 1, 1, 1, nNodes - 1)); // G_N,j+N+1		i=0, j=-N-1,...,-2
						}
						else if (cellIdx == 1) { // left boundary cell
							dispBlock.block(0, nNodes - 1, 1, nNodes + 2) += -invWeights[0] * (-GBlock.block(0, 0, 1, nNodes + 2)); // G_N,N		i=0, j=-1,...,N+1
						}
						if (cellIdx < nCol) {
							dispBlock.block(nNodes - 1, nNodes - 1, 1, nNodes) += invWeights[nNodes - 1] * (-0.5 * GBlock.block(nNodes - 1, 0, 1, nNodes)); // G_i,j+N+1		i=N, j=-1,...,N-1
							dispBlock(nNodes - 1, 2 * nNodes - 1) += invWeights[nNodes - 1] * (-0.5 * GBlock(nNodes - 1, nNodes) + 0.5 * GBlockRight(0, 0)); // G_i,j		i=N, j=N
							dispBlock(nNodes - 1, 2 * nNodes) += invWeights[nNodes - 1] * (-0.5 * GBlock(nNodes - 1, nNodes + 1) + 0.5 * GBlockRight(0, 1)); // G_i,j		i=N, j=N+1
							dispBlock.block(nNodes - 1, 2 * nNodes + 1, 1, nNodes - 1) += invWeights[nNodes - 1] * (0.5 * GBlockRight.block(0, 2, 1, nNodes - 1)); // G_0,j-N-1		i=N, j=N+2,...,2N+1
						}
						else if (cellIdx == nCol) { // right boundary cell
							dispBlock.block(nNodes - 1, nNodes - 1, 1, nNodes + 2) += invWeights[nNodes - 1] * (-GBlock.block(nNodes - 1, 0, 1, nNodes + 2)); // G_i,j+N+1		i=N, j=--1,...,N+1
						}

						dispBlock *= 2 / deltaZ;
					}

					return -dispBlock; // *-1 for residual
				}
			};

			Discretization _disc; //!< Discretization info

		//	IExternalFunction* _extFun; //!< External function (owned by library user)

			// used as auxiliary supplier 
			parts::ConvectionDispersionOperatorBase _convDispOp; //!< Convection dispersion operator for interstitial volume transport

			// linear solver (Eigen lib)
			Eigen::SparseLU<Eigen::SparseMatrix<double>> _linSolver;
			//Eigen::BiCGSTAB<Eigen::SparseMatrix<double, RowMajor>, Eigen::DiagonalPreconditioner<double>> solver; (needs _tempState, cant solve inplace)

			Eigen::SparseMatrix<double, RowMajor> _jac; //!< Jacobian
			Eigen::SparseMatrix<double, RowMajor> _jacDisc; //!< Jacobian with time derivatives from BDF method
			Eigen::MatrixXd _jacInlet; //!< Jacobian inlet DOF block matrix connects inlet DOFs to first bulk cells
			//MatrixXd FDjac; // test purpose FD Jacobian

			active _totalPorosity; //!< Total porosity \f$ \varepsilon_t \f$

			bool _analyticJac; //!< Determines whether AD or analytic Jacobians are used
			unsigned int _jacobianAdDirs; //!< Number of AD seed vectors required for Jacobian computation
			bool _factorizeJacobian; //!< Determines whether the Jacobian needs to be factorized
			double* _tempState; //!< Temporary storage with the size of the state vector or larger if binding models require it

			std::vector<active> _initC; //!< Liquid phase initial conditions
			std::vector<active> _initQ; //!< Solid phase initial conditions
			std::vector<double> _initState; //!< Initial conditions for state vector if given
			std::vector<double> _initStateDot; //!< Initial conditions for time derivative

			BENCH_TIMER(_timerResidual)
				BENCH_TIMER(_timerResidualPar)
				BENCH_TIMER(_timerResidualSens)
				BENCH_TIMER(_timerResidualSensPar)
				BENCH_TIMER(_timerConsistentInit)
				BENCH_TIMER(_timerConsistentInitPar)
				BENCH_TIMER(_timerLinearSolve)

				class Indexer
			{
			public:
				Indexer(const Discretization& disc) : _disc(disc) { }

				// Strides
				inline int strideColNode() const CADET_NOEXCEPT { return static_cast<int>(_disc.nComp + _disc.strideBound); }
				inline int strideColCell() const CADET_NOEXCEPT { return static_cast<int>(_disc.nNodes * strideColNode()); }
				inline int strideColComp() const CADET_NOEXCEPT { return 1; }

				inline int strideColLiquid() const CADET_NOEXCEPT { return static_cast<int>(_disc.nComp); }
				inline int strideColBound() const CADET_NOEXCEPT { return static_cast<int>(_disc.strideBound); }

				// Offsets
				inline int offsetC() const CADET_NOEXCEPT { return _disc.nComp; }
				inline int offsetBoundComp(unsigned int comp) const CADET_NOEXCEPT { return _disc.boundOffset[comp]; }

				// Return pointer to first element of state variable in state vector
				template <typename real_t> inline real_t* c(real_t* const data) const { return data + offsetC(); }
				template <typename real_t> inline real_t const* c(real_t const* const data) const { return data + offsetC(); }

				template <typename real_t> inline real_t* q(real_t* const data) const { return data + offsetC() + strideColLiquid(); }
				template <typename real_t> inline real_t const* q(real_t const* const data) const { return data + offsetC() + strideColLiquid(); }

				// Return specific variable in state vector
				template <typename real_t> inline real_t& c(real_t* const data, unsigned int node, unsigned int comp) const { return data[offsetC() + comp * strideColComp() + node * strideColNode()]; }
				template <typename real_t> inline const real_t& c(real_t const* const data, unsigned int node, unsigned int comp) const { return data[offsetC() + comp * strideColComp() + node * strideColNode()]; }

			protected:
				const Discretization& _disc;
			};

			class Exporter : public ISolutionExporter
			{
			public:

				Exporter(const Discretization& disc, const LumpedRateModelWithoutPoresDG& model, double const* data) : _disc(disc), _idx(disc), _model(model), _data(data) { }
				Exporter(const Discretization&& disc, const LumpedRateModelWithoutPoresDG& model, double const* data) = delete;

				virtual bool hasParticleFlux() const CADET_NOEXCEPT { return false; }
				virtual bool hasParticleMobilePhase() const CADET_NOEXCEPT { return false; }
				virtual bool hasSolidPhase() const CADET_NOEXCEPT { return _disc.strideBound > 0; }
				virtual bool hasVolume() const CADET_NOEXCEPT { return false; }
				virtual bool isParticleLumped() const CADET_NOEXCEPT { return false; }
				virtual bool hasPrimaryExtent() const CADET_NOEXCEPT { return true; }

				virtual unsigned int numComponents() const CADET_NOEXCEPT { return _disc.nComp; }
				virtual unsigned int numPrimaryCoordinates() const CADET_NOEXCEPT { return _disc.nPoints; }
				virtual unsigned int numSecondaryCoordinates() const CADET_NOEXCEPT { return 0; }
				virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
				virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
				virtual unsigned int numParticleTypes() const CADET_NOEXCEPT { return 1; }
				virtual unsigned int numParticleShells(unsigned int parType) const CADET_NOEXCEPT { return 0; }
				virtual unsigned int numBoundStates(unsigned int parType) const CADET_NOEXCEPT { return _disc.strideBound; }
				virtual unsigned int numMobilePhaseDofs() const CADET_NOEXCEPT { return _disc.nComp * _disc.nPoints; }
				virtual unsigned int numParticleMobilePhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return 0; }
				virtual unsigned int numParticleMobilePhaseDofs() const CADET_NOEXCEPT { return 0; }
				virtual unsigned int numSolidPhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return _disc.strideBound * _disc.nPoints; }
				virtual unsigned int numSolidPhaseDofs() const CADET_NOEXCEPT { return _disc.strideBound * _disc.nPoints; }
				virtual unsigned int numParticleFluxDofs() const CADET_NOEXCEPT { return 0u; }
				virtual unsigned int numVolumeDofs() const CADET_NOEXCEPT { return 0; }

				virtual int writeMobilePhase(double* buffer) const;
				virtual int writeSolidPhase(double* buffer) const;
				virtual int writeParticleMobilePhase(double* buffer) const { return 0; }
				virtual int writeSolidPhase(unsigned int parType, double* buffer) const;
				virtual int writeParticleMobilePhase(unsigned int parType, double* buffer) const { return 0; }
				virtual int writeParticleFlux(double* buffer) const { return 0; }
				virtual int writeParticleFlux(unsigned int parType, double* buffer) const { return 0; }
				virtual int writeVolume(double* buffer) const { return 0; }
				virtual int writeInlet(unsigned int port, double* buffer) const;
				virtual int writeInlet(double* buffer) const;
				virtual int writeOutlet(unsigned int port, double* buffer) const;
				virtual int writeOutlet(double* buffer) const;
				/**
				 * @brief calculates the physical node coordinates of the DG discretization with double! interface nodes
				 */
				virtual int writePrimaryCoordinates(double* coords) const
				{
					VectorXd x_l = VectorXd::LinSpaced(static_cast<int>(_disc.nCol + 1u), 0.0, static_cast<double>(_disc.length_));
					for (unsigned int i = 0; i < _disc.nCol; i++) {
						for (unsigned int j = 0; j < _disc.nNodes; j++) {
							// mapping 
							coords[i * _disc.nNodes + j] = x_l[i] + 0.5 * (_disc.length_ / static_cast<double>(_disc.nCol)) * (1.0 + _disc.nodes[j]);
						}
					}
					return _disc.nPoints;
				}

				virtual int writeSecondaryCoordinates(double* coords) const { return 0; }
				virtual int writeParticleCoordinates(unsigned int parType, double* coords) const { return 0; }

			protected:
				const Discretization& _disc;
				const Indexer _idx;
				const LumpedRateModelWithoutPoresDG& _model;
				double const* const _data;
			};

			/**
			* @brief sets the current section index and section dependend velocity, dispersion
			*/
			void updateSection(int secIdx) {

				if (_disc.curSection != secIdx) {

					_disc.curSection = secIdx;
					_disc.newStaticJac = true;

					// update velocity and dispersion
					_disc.velocity = static_cast<double>(_convDispOp.currentVelocity());
					if (_convDispOp.dispersionCompIndep())
						for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
							_disc.dispersion[comp] = static_cast<double>(_convDispOp.currentDispersion(secIdx)[0]);
						}
					else {
						for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
							_disc.dispersion[comp] = static_cast<double>(_convDispOp.currentDispersion(secIdx)[comp]);
						}
					}

				}
			}

			// ==========================================================================================================================================================  //
			// ========================================						DG Residual, RHS						======================================================  //
			// ==========================================================================================================================================================  //

			/**
			* @brief calculates the volume Integral of the auxiliary equation
			* @param [in] current state vector
			* @param [in] stateDer vector to be changed
			* @param [in] aux true if auxiliary, else main equation
			*/
			void volumeIntegral(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state, Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& stateDer) {
				// comp-cell-node state vector: use of Eigen lib performance
				for (unsigned int Cell = 0; Cell < _disc.nCol; Cell++) {
					stateDer.segment(Cell * _disc.nNodes, _disc.nNodes)
						-= _disc.polyDerM * state.segment(Cell * _disc.nNodes, _disc.nNodes);
				}
			}

			/*
			* @brief calculates the interface fluxes h* of Convection Dispersion equation
			*/
			void InterfaceFlux(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C, const VectorXd& g, unsigned int comp) {

				// component-wise strides
				unsigned int strideCell = _disc.nNodes;
				unsigned int strideNode = 1u;

				// Conv.Disp. flux: h* = h*_conv + h*_disp = numFlux(v c_l, v c_r) + 0.5 sqrt(D_ax) (S_l + S_r)

				if (_disc.velocity >= 0.0) { // forward flow (upwind num. flux)
					// calculate inner interface fluxes
					for (unsigned int Cell = 1; Cell < _disc.nCol; Cell++) {
						// h* = h*_conv + h*_disp
						_disc.surfaceFlux[Cell] // inner interfaces
							= _disc.velocity * (C[Cell * strideCell - strideNode]) // left cell (i.e. forward flow upwind)
							- 0.5 * std::sqrt(_disc.dispersion[comp]) * (g[Cell * strideCell - strideNode] // left cell
								+ g[Cell * strideCell]); // right cell
					}

					// boundary fluxes
					// inlet (left) boundary interface
					_disc.surfaceFlux[0]
						= _disc.velocity * _disc.boundary[0];

					// outlet (right) boundary interface
					_disc.surfaceFlux[_disc.nCol]
						= _disc.velocity * (C[_disc.nCol * strideCell - strideNode])
						- std::sqrt(_disc.dispersion[comp]) * 0.5 * (g[_disc.nCol * strideCell - strideNode] // last cell last node
							+ _disc.boundary[3]); // right boundary value S
				}
				else { // backward flow (upwind num. flux)
					// calculate inner interface fluxes
					for (unsigned int Cell = 1; Cell < _disc.nCol; Cell++) {
						// h* = h*_conv + h*_disp
						_disc.surfaceFlux[Cell] // inner interfaces
							= _disc.velocity * (C[Cell * strideCell]) // right cell (i.e. backward flow upwind)
							- 0.5 * std::sqrt(_disc.dispersion[comp]) * (g[Cell * strideCell - strideNode] // left cell
								+ g[Cell * strideCell]); // right cell
					}

					// boundary fluxes
					// inlet boundary interface
					_disc.surfaceFlux[_disc.nCol]
						= _disc.velocity * _disc.boundary[0];

					// outlet boundary interface
					_disc.surfaceFlux[0]
						= _disc.velocity * (C[0])
						- std::sqrt(_disc.dispersion[comp]) * 0.5 * (g[0] // first cell first node
							+ _disc.boundary[2]); // left boundary value g
				}
			}
			/**
			* @brief calculates and fills the surface flux values for auxiliary equation
			*/
			void InterfaceFluxAuxiliary(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C) {

				// component-wise strides
				unsigned int strideCell = _disc.nNodes;
				unsigned int strideNode = 1u;

				// Auxiliary flux: c* = 0.5 (c_l + c_r)

				// calculate inner interface fluxes
				for (unsigned int Cell = 1; Cell < _disc.nCol; Cell++) {
					_disc.surfaceFlux[Cell] // left interfaces
						= 0.5 * (C[Cell * strideCell - strideNode] + // left node
							C[Cell * strideCell]); // right node
				}
				// calculate boundary interface fluxes

				_disc.surfaceFlux[0] // left boundary interface
					= 0.5 * (C[0] + // boundary value
						C[0]); // first cell first node

				_disc.surfaceFlux[(_disc.nCol)] // right boundary interface
					= 0.5 * (C[_disc.nCol * strideCell - strideNode] + // last cell last node
						C[_disc.nCol * strideCell - strideNode]);// // boundary value
			}

			/**
			* @brief calculates the surface Integral, depending on the approach (exact/inexact integration)
			* @param [in] state relevant state vector
			* @param [in] stateDer state derivative vector the solution is added to
			* @param [in] aux true for auxiliary equation, false for main equation
				surfaceIntegral(cPtr, &(disc.g[0]), disc,&(disc.h[0]), resPtrC, 0, secIdx);
			*/
			void surfaceIntegral(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C, Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state, Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& stateDer, bool aux, unsigned int Comp) {

				// component-wise strides
				unsigned int strideCell = _disc.nNodes;
				unsigned int strideNode = 1u;

				// calc numerical flux values c* or h* depending on auxiliary equation switch aux
				(aux == 1) ? InterfaceFluxAuxiliary(C) : InterfaceFlux(C, _disc.g, Comp);
				if (_disc.exactInt) { // modal approach -> dense mass matrix
					for (unsigned int Cell = 0; Cell < _disc.nCol; Cell++) {
						// strong surface integral -> M^-1 B [state - state*]
						for (unsigned int Node = 0; Node < _disc.nNodes; Node++) {
							stateDer[Cell * strideCell + Node * strideNode]
								-= _disc.invMM(Node, 0) * (state[Cell * strideCell]
									- _disc.surfaceFlux[Cell])
								- _disc.invMM(Node, _disc.polyDeg) * (state[Cell * strideCell + _disc.polyDeg * strideNode]
									- _disc.surfaceFlux[(Cell + 1)]);
						}
					}
				}
				else { // nodal approach -> diagonal mass matrix
					for (unsigned int Cell = 0; Cell < _disc.nCol; Cell++) {
						// strong surface integral -> M^-1 B [state - state*]
						stateDer[Cell * strideCell] // first cell node
							-= _disc.invWeights[0] * (state[Cell * strideCell] // first node
								- _disc.surfaceFlux(Cell));
						stateDer[Cell * strideCell + _disc.polyDeg * strideNode] // last cell node
							+= _disc.invWeights[_disc.polyDeg] * (state[Cell * strideCell + _disc.polyDeg * strideNode]
								- _disc.surfaceFlux(Cell + 1));
					}
				}
			}

			/**
			* @brief calculates the substitute h = vc - sqrt(D_ax) g(c)
			*/
			void calcH(Eigen::Map<const VectorXd, 0, InnerStride<>>& C, unsigned int Comp) {
				_disc.h = _disc.velocity * C - std::sqrt(_disc.dispersion[Comp]) * _disc.g;
			}

			/**
			* @brief calculates the isotherm right hand side
			*/
			void calcRHSq_DG(double t, unsigned int secIdx, const double* yPtr, double* const resPtr, util::ThreadLocalStorage& threadLocalMem) {

				Indexer idx(_disc);

				const double* localC = yPtr + idx.offsetC();
				const double* localQ = yPtr + idx.offsetC() + idx.strideColLiquid() + idx.offsetBoundComp(0);
				double* localQRes = resPtr + idx.offsetC() + idx.strideColLiquid() + idx.offsetBoundComp(0);

				for (unsigned int point = 0; point < _disc.nPoints; point++) {

					double z = _disc.deltaZ * std::floor(point / _disc.nNodes)
						+ 0.5 * _disc.deltaZ * (1 + _disc.nodes[point % _disc.nNodes]);

					_binding[0]->flux(t, secIdx, ColumnPosition{ z, 0.0, 0.0 }, localQ, localC, localQRes, threadLocalMem.get());

					localC += idx.strideColNode(); // next solid concentration
					localQ += idx.strideColNode(); // next liquid concentration
					localQRes += idx.strideColNode(); // next liquid concentration

				}
			}

			/**
			* @brief applies the inverse Jacobian of the mapping
			*/
			void applyMapping(Eigen::Map<VectorXd, 0, InnerStride<>>& state) {
				state *= (2.0 / _disc.deltaZ);
			}
			/**
			* @brief applies the inverse Jacobian of the mapping and auxiliary factor -1
			*/
			void applyMapping_Aux(Eigen::Map<VectorXd, 0, InnerStride<>>& state, unsigned int Comp) {
				state *= (-2.0 / _disc.deltaZ) * ((_disc.dispersion[Comp] == 0.0) ? 1.0 : std::sqrt(_disc.dispersion[Comp]));
			}

			void ConvDisp_DG(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C, Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& resC, double t, unsigned int Comp) {

				// ===================================//
				// reset cache                        //
				// ===================================//

				resC.setZero();
				_disc.h.setZero();
				_disc.g.setZero();
				_disc.surfaceFlux.setZero();
				// get Map objects of auxiliary variable memory
				Eigen::Map<VectorXd, 0, InnerStride<>> g(&_disc.g[0], _disc.nPoints, InnerStride<>(1));
				Eigen::Map<const VectorXd, 0, InnerStride<>> h(&_disc.h[0], _disc.nPoints, InnerStride<>(1));

				// ======================================//
				// solve auxiliary system g = d c / d x  //
				// ======================================//

				volumeIntegral(C, g); // DG volumne integral in strong form

				surfaceIntegral(C, C, g, 1, Comp); // surface integral in strong form

				applyMapping_Aux(g, Comp); // inverse mapping from reference space and auxiliary factor

				_disc.surfaceFlux.setZero(); // reset surface flux storage as it is used twice

				// ======================================//
				// solve main equation w_t = d h / d x   //
				// ======================================//

				calcH(C, Comp); // calculate the substitute h(S(c), c) = - (sqrt(D_ax) g(c) - v c)

				volumeIntegral(h, resC); // DG volumne integral in strong form

				calcBoundaryValues(C);// update boundary values including auxiliary variable g

				surfaceIntegral(C, h, resC, 0, Comp); // DG surface integral in strong form

				applyMapping(resC); // inverse mapping to reference space

			}
			/**
			* @brief computes ghost nodes used to implement Danckwerts boundary conditions
			*/
			void calcBoundaryValues(Eigen::Map<const VectorXd, 0, InnerStride<>>& C) {

				//cache.boundary[0] = c_in -> inlet DOF idas suggestion
				//_disc.boundary[1] = (_disc.velocity >= 0.0) ? C[_disc.nPoints - 1] : C[0]; // c_r outlet not required
				_disc.boundary[2] = -_disc.g[0]; // g_l left boundary (inlet/outlet for forward/backward flow)
				_disc.boundary[3] = -_disc.g[_disc.nPoints - 1]; // g_r right boundary (outlet/inlet for forward/backward flow)
			}

			// ==========================================================================================================================================================  //
			// ========================================						DG Jacobian							=========================================================  //
			// ==========================================================================================================================================================  //

			typedef Eigen::Triplet<double> T;

			/**
			* @brief sets the sparsity pattern of the static Jacobian
			* @detail DG ConvDisp pattern and isotherm pattern. Independent of the isotherm,
				all liquid and solid entries at a discrete point are set.
			* @param [in] stateDer bool if state derivative pattern should be added (for _jacDisc)
			*/
			void setPattern(Eigen::SparseMatrix<double, RowMajor>& mat, bool stateDer, bool has_reaction) {

				std::vector<T> tripletList;

				unsigned int isotherm_entries = _disc.nPoints * _disc.strideBound * (_disc.strideBound + _disc.nComp);
				unsigned int reaction_entries = has_reaction ? _disc.nPoints * _disc.nComp * (_disc.strideBound + _disc.nComp) : 0;

				tripletList.reserve(nConvDispEntries(false) + isotherm_entries + reaction_entries);

				if (_disc.exactInt)
					ConvDispModalPattern(tripletList);
				else
					ConvDispNodalPattern(tripletList);

				bindingAndReactionPattern(tripletList, has_reaction);

				if (stateDer)
					stateDerPattern(tripletList); // only adds [ d convDisp / d q_t ] because main diagonal is already included !

				mat.setFromTriplets(tripletList.begin(), tripletList.end());

			}

			/**
			* @brief computes the convection dispersion part of the state derivative pattern of the jacobian.
			* @detail the main also diagonal belongs to the state derivative pattern but is not set here, so it should be set previously.
			*/
			void stateDerPattern(std::vector<T>& tripletList) {

				Indexer idxr(_disc);
				unsigned int offC = 0; // inlet DOFs not included in Jacobian

				for (unsigned int point = 0; point < _disc.nPoints; point++) {
					for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
						if (_disc.nBound[comp]) // either one or null
							// row:    jump over inlet, go node and comp strides to find all liquid states
							// column: jump over inlet and liquid states, add component offset and go node strides to find corresponding solid states
							tripletList.push_back(T(offC + point * idxr.strideColNode() + comp * idxr.strideColComp(),
								offC + idxr.strideColLiquid() + idxr.offsetBoundComp(comp) + point * idxr.strideColNode(),
								0.0));
					}
				}

			}

			/**
			* @brief sets the sparsity pattern of the convection dispersion Jacobian for the nodal DG scheme
			*/
			int ConvDispNodalPattern(std::vector<T>& tripletList) {

				Indexer idxr(_disc);

				int sNode = idxr.strideColNode();
				int sCell = idxr.strideColCell();
				int sComp = idxr.strideColComp();
				int offC = 0; // inlet DOFs not included in Jacobian

				unsigned int nNodes = _disc.nNodes;
				unsigned int nCells = _disc.nCol;
				unsigned int nComp = _disc.nComp;

				/*======================================================*/
				/*			Define Convection Jacobian Block			*/
				/*======================================================*/

				// Convection block [ d RHS_conv / d c ], also depends on upwind entry

				if (_disc.velocity >= 0.0) { // forward flow upwind entry -> last node of previous cell
				// special inlet DOF treatment for inlet boundary cell (first cell)
					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							//tripletList.push_back(T(offC + comp * sComp + i * sNode, comp * sComp, 0.0)); // inlet DOFs not included in Jacobian
							for (unsigned int j = 1; j < nNodes + 1; j++) {
								tripletList.push_back(T(offC + comp * sComp + i * sNode,
									offC + comp * sComp + (j - 1) * sNode,
									0.0));
							}
						}
					}
					for (unsigned int cell = 1; cell < nCells; cell++) {
						for (unsigned int comp = 0; comp < nComp; comp++) {
							for (unsigned int i = 0; i < nNodes; i++) {
								for (unsigned int j = 0; j < nNodes + 1; j++) {
									// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
									// col: jump over inlet DOFs and previous cells, go back one node, add component offset and go node strides from there for each convection block entry
									tripletList.push_back(T(offC + cell * sCell + comp * sComp + i * sNode,
										offC + cell * sCell - sNode + comp * sComp + j * sNode,
										0.0));
								}
							}
						}
					}
				}
				else { // backward flow upwind entry -> first node of subsequent cell
					// special inlet DOF treatment for inlet boundary cell (last cell)
					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							// inlet DOFs not included in Jacobian
							for (unsigned int j = 0; j < nNodes; j++) {
								tripletList.push_back(T(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
									offC + (nCells - 1) * sCell + comp * sComp + j * sNode,
									0.0));
							}
						}
					}
					for (unsigned int cell = 0; cell < nCells - 1u; cell++) {
						for (unsigned int comp = 0; comp < nComp; comp++) {
							for (unsigned int i = 0; i < nNodes; i++) {
								for (unsigned int j = 0; j < nNodes + 1; j++) {
									// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
									// col: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
									tripletList.push_back(T(offC + cell * sCell + comp * sComp + i * sNode,
										offC + cell * sCell + comp * sComp + j * sNode,
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

				// insert Blocks to Jacobian inner cells (only for nCells >= 3)
				if (nCells >= 3u) {
					for (unsigned int cell = 1; cell < nCells - 1; cell++) {
						for (unsigned int comp = 0; comp < nComp; comp++) {
							for (unsigned int i = 0; i < nNodes; i++) {
								for (unsigned int j = 0; j < 3 * nNodes; j++) {
									// pattern is more sparse than a nNodes x 3*nNodes block.
									if ((j >= nNodes - 1 && j <= 2 * nNodes) ||
										(i == 0 && j <= 2 * nNodes) ||
										(i == nNodes - 1 && j >= nNodes - 1))
										// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each entry
										// col: jump over inlet DOFs and previous cells, go back one cell, add component offset and go node strides from there for each entry
										tripletList.push_back(T(offC + cell * sCell + comp * sComp + i * sNode,
											offC + (cell - 1) * sCell + comp * sComp + j * sNode,
											0.0));
								}
							}
						}
					}
				}

				/*				Boundary cell Dispersion blocks			*/

				if (nCells != 1) { // Note: special case nCells = 1 already set by advection block
					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = nNodes; j < 3 * nNodes; j++) {
								// pattern is more sparse than a nNodes x 2*nNodes block.
								if ((j >= nNodes - 1 && j <= 2 * nNodes) ||
									(i == 0 && j <= 2 * nNodes) ||
									(i == nNodes - 1 && j >= nNodes - 1))
									tripletList.push_back(T(offC + comp * sComp + i * sNode,
										offC + comp * sComp + (j - nNodes) * sNode,
										0.0));
							}
						}
					}

					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = 0; j < 2 * nNodes; j++) {
								// pattern is more sparse than a nNodes x 2*nNodes block.
								if ((j >= nNodes - 1 && j <= 2 * nNodes) ||
									(i == 0 && j <= 2 * nNodes) ||
									(i == nNodes - 1 && j >= nNodes - 1))
									tripletList.push_back(T(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
										offC + (nCells - 1 - 1) * sCell + comp * sComp + j * sNode,
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
			int ConvDispModalPattern(std::vector<T>& tripletList) {

				Indexer idxr(_disc);

				int sNode = idxr.strideColNode();
				int sCell = idxr.strideColCell();
				int sComp = idxr.strideColComp();
				int offC = 0; // inlet DOFs not included in Jacobian

				unsigned int nNodes = _disc.nNodes;
				unsigned int nCells = _disc.nCol;
				unsigned int nComp = _disc.nComp;

				/*======================================================*/
				/*			Define Convection Jacobian Block			*/
				/*======================================================*/

				// Convection block [ d RHS_conv / d c ], also depends on upwind entry

				if (_disc.velocity >= 0.0) { // forward flow upwind entry -> last node of previous cell
					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							//tripletList.push_back(T(offC + comp * sComp + i * sNode, comp * sComp, 0.0)); // inlet DOFs not included in Jacobian
							for (unsigned int j = 1; j < nNodes + 1; j++) {
								tripletList.push_back(T(offC + comp * sComp + i * sNode,
									offC + comp * sComp + (j - 1) * sNode,
									0.0));
							}
						}
					}
					for (unsigned int cell = 1; cell < nCells; cell++) {
						for (unsigned int comp = 0; comp < nComp; comp++) {
							for (unsigned int i = 0; i < nNodes; i++) {
								for (unsigned int j = 0; j < nNodes + 1; j++) {
									// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
									// col: jump over inlet DOFs and previous cells, go back one node, add component offset and go node strides from there for each convection block entry
									tripletList.push_back(T(offC + cell * sCell + comp * sComp + i * sNode,
										offC + cell * sCell - sNode + comp * sComp + j * sNode,
										0.0));
								}
							}
						}
					}
				}
				else { // backward flow upwind entry -> first node of subsequent cell
					// special inlet DOF treatment for inlet boundary cell (last cell)
					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							// inlet DOFs not included in Jacobian
							for (unsigned int j = 0; j < nNodes; j++) {
								tripletList.push_back(T(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
									offC + (nCells - 1) * sCell + comp * sComp + j * sNode,
									0.0));
							}
						}
					}
					for (unsigned int cell = 0; cell < nCells - 1u; cell++) {
						for (unsigned int comp = 0; comp < nComp; comp++) {
							for (unsigned int i = 0; i < nNodes; i++) {
								for (unsigned int j = 0; j < nNodes + 1; j++) {
									// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
									// col: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
									tripletList.push_back(T(offC + cell * sCell + comp * sComp + i * sNode,
										offC + cell * sCell + comp * sComp + j * sNode,
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
				if (nCells >= 5u) {
					// Inner dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
					for (unsigned int cell = 2; cell < nCells - 2; cell++) {
						for (unsigned int comp = 0; comp < nComp; comp++) {
							for (unsigned int i = 0; i < nNodes; i++) {
								for (unsigned int j = 0; j < 3 * nNodes + 2; j++) {
									// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
									// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry
									tripletList.push_back(T(offC + cell * sCell + comp * sComp + i * sNode,
										offC + cell * sCell - (nNodes + 1) * sNode + comp * sComp + j * sNode,
										0.0));
								}
							}
						}
					}
				}

				/*		boundary cell neighbours		*/

				// left boundary cell neighbour
				if (nCells >= 4u) {
					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = 1; j < 3 * nNodes + 2; j++) {
								// row: jump over inlet DOFs and previous cell, add component offset and go node strides from there for each dispersion block entry
								// col: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry. Also adjust for iterator j (-1)
								tripletList.push_back(T(offC + nNodes * sNode + comp * sComp + i * sNode,
									offC + comp * sComp + (j - 1) * sNode,
									0.0));
							}
						}
					}
				}
				else if (nCells == 3u) { // special case: only depends on the two neighbouring cells
					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = 1; j < 3 * nNodes + 1; j++) {
								// row: jump over inlet DOFs and previous cell, add component offset and go node strides from there for each dispersion block entry
								// col: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry. Also adjust for iterator j (-1)
								tripletList.push_back(T(offC + nNodes * sNode + comp * sComp + i * sNode,
									offC + comp * sComp + (j - 1) * sNode,
									0.0));
							}
						}
					}
				}
				// right boundary cell neighbour
				if (nCells >= 4u) {
					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = 0; j < 3 * nNodes + 2 - 1; j++) {
								// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
								// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry.
								tripletList.push_back(T(offC + (nCells - 2) * sCell + comp * sComp + i * sNode,
									offC + (nCells - 2) * sCell - (nNodes + 1) * sNode + comp * sComp + j * sNode,
									0.0));
							}
						}
					}
				}
				/*			boundary cells			*/

				// left boundary cell
				unsigned int end = 3u * nNodes + 2u;
				if (nCells == 1u) end = 2u * nNodes + 1u;
				else if (nCells == 2u) end = 3u * nNodes + 1u;
				for (unsigned int comp = 0; comp < nComp; comp++) {
					for (unsigned int i = 0; i < nNodes; i++) {
						for (unsigned int j = nNodes + 1; j < end; j++) {
							// row: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry
							// col: jump over inlet DOFs, add component offset, adjust for iterator j (-Nnodes-1) and go node strides from there for each dispersion block entry.
							tripletList.push_back(T(offC + comp * sComp + i * sNode,
								offC + comp * sComp + (j - (nNodes + 1)) * sNode,
								0.0));
						}
					}
				}
				// right boundary cell
				if (nCells >= 3u) {
					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = 0; j < 2 * nNodes + 1; j++) {
								// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
								// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry.
								tripletList.push_back(T(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
									offC + (nCells - 1) * sCell - (nNodes + 1) * sNode + comp * sComp + j * sNode,
									0.0));
							}
						}
					}
				}
				else if (nCells == 2u) { // special case for nCells == 2: depends only on left cell
					for (unsigned int comp = 0; comp < nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = 0; j < 2 * nNodes; j++) {
								// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
								// col: jump over inlet DOFs and previous cells, go back one cell, add component offset and go node strides from there for each dispersion block entry.
								tripletList.push_back(T(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
									offC + (nCells - 1) * sCell - (nNodes)*sNode + comp * sComp + j * sNode,
									0.0));
							}
						}
					}
				}

				return 0;
			}

			/**
			* @brief sets the sparsity pattern of the isotherm and reaction Jacobian
			* @detail Independent of the isotherm, all liquid and solid entries (so all entries, the isotherm could theoretically depend on) at a discrete point are set.
			*/
			void bindingAndReactionPattern(std::vector<T>& tripletList, bool has_reaction) {

				Indexer idxr(_disc);
				int offC = 0; // inlet DOFs not included in Jacobian

				// isotherm pattern: loop over all discrete points and solid states and add all liquid plus solid entries at that solid state at that discrete point
				for (unsigned int point = 0; point < _disc.nPoints; point++) {
					for (unsigned int solid = 0; solid < _disc.strideBound; solid++) {
						for (unsigned int conc = 0; conc < _disc.nComp + _disc.strideBound; conc++) {
							// row:		  jump over inlet and previous discrete points, liquid concentration and add the offset of the current bound state
							// column:    jump over inlet and previous discrete points. add entries for all liquid and bound concentrations (conc)
							tripletList.push_back(T(offC + idxr.strideColNode() * point + idxr.strideColLiquid() + solid,
								offC + idxr.strideColNode() * point + conc,
								0.0));
						}
					}
				}

				// reaction pattern: loop over all discrete points and liquid states and add all liquid plus solid entries at that liquid state at that discrete point
				if (has_reaction) {
					for (unsigned int point = 0; point < _disc.nPoints; point++) {
						for (unsigned int liquid = 0; liquid < _disc.nComp; liquid++) {
							for (unsigned int conc = 0; conc < _disc.nComp + _disc.strideBound; conc++) {
								// row:		  jump over inlet and previous discrete points, liquid concentration and add the offset of the current bound state
								// column:    jump over inlet and previous discrete points. add entries for all liquid and bound concentrations (conc)
								tripletList.push_back(T(offC + idxr.strideColNode() * point + liquid,
									offC + idxr.strideColNode() * point + conc,
									0.0));
							}
						}
					}
				}
			}

			/**
			* @brief analytically calculates the (static) state jacobian
			* @return 1 if jacobain estimation fits the predefined pattern of the jacobian, 0 if not.
			*/
			int calcStaticAnaJacobian(double t, unsigned int secIdx, const double* const y, util::ThreadLocalStorage& threadLocalMem) {

				// DG convection dispersion Jacobian
				if (_disc.exactInt)
					calcConvDispDGSEMJacobian();
				else
					calcConvDispCollocationDGSEMJacobian();

				if (!_jac.isCompressed()) // if matrix lost its compressed storage, the pattern did not fit.
					return 0;

				return 1;
			}

			/**
			 * @brief calculates the number of entris for the DG convection dispersion jacobian
			 * @note only dispersion entries are relevant for jacobian NNZ as the convection entries are a subset of these
			 */
			unsigned int nConvDispEntries(bool pureNNZ = false) {

				if (_disc.exactInt) {
					if (pureNNZ) {
						return _disc.nComp * ((3u * _disc.nCol - 2u) * _disc.nNodes * _disc.nNodes + (2u * _disc.nCol - 3u) * _disc.nNodes); // dispersion entries
					}
					return _disc.nComp * _disc.nNodes * _disc.nNodes + _disc.nNodes // convection entries
						+ _disc.nComp * ((3u * _disc.nCol - 2u) * _disc.nNodes * _disc.nNodes + (2u * _disc.nCol - 3u) * _disc.nNodes); // dispersion entries
				}
				else {
					if (pureNNZ) {
						return _disc.nComp * (_disc.nCol * _disc.nNodes * _disc.nNodes + 8u * _disc.nNodes); // dispersion entries
					}
					return _disc.nComp * _disc.nNodes * _disc.nNodes + 1u // convection entries
						+ _disc.nComp * (_disc.nCol * _disc.nNodes * _disc.nNodes + 8u * _disc.nNodes); // dispersion entries
				}
			}

			/**
			* @brief analytically calculates the convection dispersion jacobian for the nodal DG scheme
			*/
			int calcConvDispCollocationDGSEMJacobian() {

				Indexer idxr(_disc);

				int offC = 0; // inlet DOFs not included in Jacobian

				unsigned int nNodes = _disc.nNodes;
				unsigned int nCells = _disc.nCol;
				unsigned int nComp = _disc.nComp;

				/*======================================================*/
				/*			Compute Dispersion Jacobian Block			*/
				/*======================================================*/

				/*		Inner cell dispersion blocks		*/

				if (nCells >= 3u) {
					MatrixXd dispBlock = _disc.DGjacAxDispBlocks[1];
					linalg::BandedEigenSparseRowIterator jacIt(_jac, offC + idxr.strideColCell()); // row iterator starting at second cell and component

					for (unsigned int cell = 1; cell < nCells - 1; cell++) {
						for (unsigned int i = 0; i < dispBlock.rows(); i++, jacIt += idxr.strideColBound()) {
							for (unsigned int comp = 0; comp < nComp; comp++, ++jacIt) {
								for (unsigned int j = 0; j < dispBlock.cols(); j++) {
									// pattern is more sparse than a nNodes x 3*nNodes block.
									if ((j >= nNodes - 1 && j <= 2 * nNodes) ||
										(i == 0 && j <= 2 * nNodes) ||
										(i == nNodes - 1 && j >= nNodes - 1))
										// row: iterator is at current node i and current component comp
										// col: start at previous cell and jump to node j
										jacIt[-idxr.strideColCell() + (j - i) * idxr.strideColNode()] = dispBlock(i, j) * _disc.dispersion[comp];
								}
							}
						}
					}
				}

				/*				Boundary cell Dispersion blocks			*/

				/* left cell */
				MatrixXd dispBlock = _disc.DGjacAxDispBlocks[0];

				if (nCells != 1u) { // "standard" case
					linalg::BandedEigenSparseRowIterator jacIt(_jac, offC); // row iterator starting at first cell and component

					for (unsigned int i = 0; i < dispBlock.rows(); i++, jacIt += idxr.strideColBound()) {
						for (unsigned int comp = 0; comp < nComp; comp++, ++jacIt) {
							for (unsigned int j = nNodes; j < dispBlock.cols(); j++) {
								// pattern is more sparse than a nNodes x 2*nNodes block.
								if ((j >= nNodes - 1 && j <= 2 * nNodes) ||
									(i == 0 && j <= 2 * nNodes) ||
									(i == nNodes - 1 && j >= nNodes - 1))
									// row: iterator is at current node i and current component comp
									// col: jump to node j
									jacIt[((j - nNodes) - i) * idxr.strideColNode()] = dispBlock(i, j) * _disc.dispersion[comp];
							}
						}
					}
				}
				else { // special case
					linalg::BandedEigenSparseRowIterator jacIt(_jac, offC); // row iterator starting at first cell and component
					for (unsigned int i = 0; i < dispBlock.rows(); i++, jacIt += idxr.strideColBound()) {
						for (unsigned int comp = 0; comp < nComp; comp++, ++jacIt) {
							for (unsigned int j = nNodes; j < nNodes * 2u; j++) {
								// row: iterator is at current node i and current component comp
								// col: jump to node j
								jacIt[((j - nNodes) - i) * idxr.strideColNode()] = dispBlock(i, j) * _disc.dispersion[comp];
							}
						}
					}
				}

				/* right cell */
				if (nCells != 1u) { // "standard" case
					dispBlock = _disc.DGjacAxDispBlocks[std::min(nCells, 3u) - 1];
					linalg::BandedEigenSparseRowIterator jacIt(_jac, offC + (nCells - 1) * idxr.strideColCell()); // row iterator starting at last cell

					for (unsigned int i = 0; i < dispBlock.rows(); i++, jacIt += idxr.strideColBound()) {
						for (unsigned int comp = 0; comp < nComp; comp++, ++jacIt) {
							for (unsigned int j = 0; j < 2 * nNodes; j++) {
								// pattern is more sparse than a nNodes x 2*nNodes block.
								if ((j >= nNodes - 1 && j <= 2 * nNodes) ||
									(i == 0 && j <= 2 * nNodes) ||
									(i == nNodes - 1 && j >= nNodes - 1))
									// row: iterator is at current node i and current component comp
									// col: start at previous cell and jump to node j
									jacIt[-idxr.strideColCell() + (j - i) * idxr.strideColNode()] = dispBlock(i, j) * _disc.dispersion[comp];
							}
						}
					}
				}

				/*======================================================*/
				/*			Compute Convection Jacobian Block			*/
				/*======================================================*/

				// Convection block [ d RHS_conv / d c ], also depends on first entry of previous cell
				MatrixXd convBlock = _disc.DGjacAxConvBlock;
				linalg::BandedEigenSparseRowIterator jacIt(_jac, offC); // row iterator starting at first cell and component

				if (_disc.velocity >= 0.0) { // forward flow upwind convection
					// special inlet DOF treatment for first cell (inlet boundary cell)
					_jacInlet(0, 0) = _disc.velocity * convBlock(0, 0); // only first node depends on inlet concentration
					for (unsigned int i = 0; i < convBlock.rows(); i++, jacIt += idxr.strideColBound()) {
						for (unsigned int comp = 0; comp < nComp; comp++, ++jacIt) {
							//jacIt[0] = -convBlock(i, 0); // dependency on inlet DOFs is handled in _jacInlet
							for (unsigned int j = 1; j < convBlock.cols(); j++) {
								jacIt[((j - 1) - i) * idxr.strideColNode()] += _disc.velocity * convBlock(i, j);
							}
						}
					}
					// remaining cells
					for (unsigned int cell = 1; cell < nCells; cell++) {
						for (unsigned int i = 0; i < convBlock.rows(); i++, jacIt += idxr.strideColBound()) {
							for (unsigned int comp = 0; comp < nComp; comp++, ++jacIt) {
								for (unsigned int j = 0; j < convBlock.cols(); j++) {
									// row: iterator is at current cell and component
									// col: start at previous cells last node and go to node j.
									jacIt[-idxr.strideColNode() + (j - i) * idxr.strideColNode()] += _disc.velocity * convBlock(i, j);
								}
							}
						}
					}
				}
				else { // backward flow upwind convection
					// non-inlet cells
					for (unsigned int cell = 0; cell < nCells - 1u; cell++) {
						for (unsigned int i = 0; i < convBlock.rows(); i++, jacIt += idxr.strideColBound()) {
							for (unsigned int comp = 0; comp < nComp; comp++, ++jacIt) {
								for (unsigned int j = 0; j < convBlock.cols(); j++) {
									// row: iterator is at current cell and component
									// col: start at current cells first node and go to node j.
									jacIt[(j - i) * idxr.strideColNode()] += _disc.velocity * convBlock(i, j);
								}
							}
						}
					}
					// special inlet DOF treatment for last cell (inlet boundary cell)
					_jacInlet(0, 0) = _disc.velocity * convBlock(convBlock.rows() - 1, convBlock.cols() - 1); // only last node depends on inlet concentration
					for (unsigned int i = 0; i < convBlock.rows(); i++, jacIt += idxr.strideColBound()) {
						for (unsigned int comp = 0; comp < nComp; comp++, ++jacIt) {
							for (unsigned int j = 0; j < convBlock.cols() - 1; j++) {
								jacIt[(j - i) * idxr.strideColNode()] += _disc.velocity * convBlock(i, j);
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
			 * @param [in] idxr Indexer
			 * @param [in] nCells determines how often the block is added (diagonally)
			 * @param [in] Compfactor component dependend factors
			 */
			void insertCompDepLiquidJacBlock(Eigen::MatrixXd block, linalg::BandedEigenSparseRowIterator& jac, int offCol, Indexer& idxr, unsigned int nCells, double* Compfactor) {

				for (unsigned int cell = 0; cell < nCells; cell++) {
					for (unsigned int i = 0; i < block.rows(); i++, jac += idxr.strideColBound()) {
						for (unsigned int comp = 0; comp < _disc.nComp; comp++, ++jac) {
							for (unsigned int j = 0; j < block.cols(); j++) {
								// row: at current node component
								// col: jump to node j
								jac[(j - i) * idxr.strideColNode() + offCol] = block(i, j) * Compfactor[comp];
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
			 * @param [in] idxr Indexer
			 * @param [in] nCells determines how often the block is added (diagonally)
			 */
			void addLiquidJacBlock(Eigen::MatrixXd block, linalg::BandedEigenSparseRowIterator& jac, int offCol, Indexer& idxr, unsigned int nCells) {

				for (unsigned int cell = 0; cell < nCells; cell++) {
					for (unsigned int i = 0; i < block.rows(); i++, jac += idxr.strideColBound()) {
						for (unsigned int comp = 0; comp < _disc.nComp; comp++, ++jac) {
							for (unsigned int j = 0; j < block.cols(); j++) {
								// row: at current node component
								// col: jump to node j
								jac[(j - i) * idxr.strideColNode() + offCol] += block(i, j);
							}
						}
					}
				}
			}
			/**
			 * @brief analytically calculates the convection dispersion jacobian for the exact integration (here: modal) DG scheme
			 */
			int calcConvDispDGSEMJacobian() {

				Indexer idxr(_disc);

				int offC = 0; // inlet DOFs not included in Jacobian

				unsigned int nNodes = _disc.nNodes;
				unsigned int nCells = _disc.nCol;
				unsigned int nComp = _disc.nComp;

				/*======================================================*/
				/*			Compute Dispersion Jacobian Block			*/
				/*======================================================*/

				/* Inner cells (exist only if nCells >= 5) */
				if (nCells >= 5) {
					linalg::BandedEigenSparseRowIterator jacIt(_jac, offC + idxr.strideColCell() * 2); // row iterator starting at third cell, first component
					// insert all (nCol - 4) inner cell blocks
					insertCompDepLiquidJacBlock(_disc.DGjacAxDispBlocks[2], jacIt, -(idxr.strideColCell() + idxr.strideColNode()), idxr, _disc.nCol - 4u, &(_disc.dispersion[0]));
				}

				/*	boundary cell neighbours (exist only if nCells >= 4)	*/
				if (nCells >= 4) {
					linalg::BandedEigenSparseRowIterator jacIt(_jac, offC + idxr.strideColCell()); // row iterator starting at second cell, first component

					insertCompDepLiquidJacBlock(_disc.DGjacAxDispBlocks[1].block(0, 1, nNodes, 3 * nNodes + 1), jacIt, -idxr.strideColCell(), idxr, 1u, &(_disc.dispersion[0]));

					jacIt += (_disc.nCol - 4) * idxr.strideColCell(); // move iterator to preultimate cell (already at third cell)
					insertCompDepLiquidJacBlock(_disc.DGjacAxDispBlocks[nCells > 4 ? 3 : 2].block(0, 0, nNodes, 3 * nNodes + 1), jacIt, -(idxr.strideColCell() + idxr.strideColNode()), idxr, 1u, &(_disc.dispersion[0]));
				}

				/*			boundary cells (exist only if nCells >= 3)			*/
				if (nCells >= 3) {

					linalg::BandedEigenSparseRowIterator jacIt(_jac, offC); // row iterator starting at first cell, first component

					insertCompDepLiquidJacBlock(_disc.DGjacAxDispBlocks[0].block(0, nNodes + 1, nNodes, 2 * nNodes + 1), jacIt, 0, idxr, 1u, &(_disc.dispersion[0]));

					jacIt += (_disc.nCol - 2) * idxr.strideColCell(); // move iterator to last cell (already at second cell)
					insertCompDepLiquidJacBlock(_disc.DGjacAxDispBlocks[std::min(nCells, 5u) - 1u].block(0, 0, nNodes, 2 * nNodes + 1), jacIt, -(idxr.strideColCell() + idxr.strideColNode()), idxr, 1u, &(_disc.dispersion[0]));
				}

				/* For special cases nCells = 1, 2, 3, some cells still have to be treated separately*/

				if (nCells == 1) {
					linalg::BandedEigenSparseRowIterator jacIt(_jac, offC); // row iterator starting at first cell, first component
					// insert the only block
					insertCompDepLiquidJacBlock(_disc.DGjacAxDispBlocks[0].block(0, nNodes + 1, nNodes, nNodes), jacIt, 0, idxr, 1u, &(_disc.dispersion[0]));
				}
				else if (nCells == 2) {
					linalg::BandedEigenSparseRowIterator jacIt(_jac, offC); // row iterator starting at first cell, first component
					// left Bacobian block
					insertCompDepLiquidJacBlock(_disc.DGjacAxDispBlocks[0].block(0, nNodes + 1, nNodes, 2 * nNodes), jacIt, 0, idxr, 1u, &(_disc.dispersion[0]));
					// right Bacobian block, iterator is already moved to second cell
					insertCompDepLiquidJacBlock(_disc.DGjacAxDispBlocks[1].block(0, 1, nNodes, 2 * nNodes), jacIt, -idxr.strideColCell(), idxr, 1u, &(_disc.dispersion[0]));
				}
				else if (nCells == 3) {
					linalg::BandedEigenSparseRowIterator jacIt(_jac, offC + idxr.strideColCell()); // row iterator starting at first cell, first component
					insertCompDepLiquidJacBlock(_disc.DGjacAxDispBlocks[1].block(0, 1, nNodes, 3 * nNodes), jacIt, -idxr.strideColCell(), idxr, 1u, &(_disc.dispersion[0]));
				}

				/*======================================================*/
				/*			Compute Convection Jacobian Block			*/
				/*======================================================*/

				int sComp = idxr.strideColComp();
				int sNode = idxr.strideColNode();
				int sCell = idxr.strideColCell();

				linalg::BandedEigenSparseRowIterator jac(_jac, offC);

				if (_disc.velocity >= 0.0) { // Forward flow
					// special inlet DOF treatment for inlet (first) cell
					_jacInlet = _disc.velocity * _disc.DGjacAxConvBlock.col(0); // only first cell depends on inlet concentration
					addLiquidJacBlock(_disc.velocity * _disc.DGjacAxConvBlock.block(0, 1, nNodes, nNodes), jac, 0, idxr, 1);
					if (_disc.nCol > 1) // iterator already moved to second cell
						addLiquidJacBlock(_disc.velocity * _disc.DGjacAxConvBlock, jac, -idxr.strideColNode(), idxr, _disc.nCol - 1);
				}
				else { // Backward flow
					// non-inlet cells first
					if (_disc.nCol > 1)
						addLiquidJacBlock(_disc.velocity * _disc.DGjacAxConvBlock, jac, 0, idxr, _disc.nCol - 1);
					// special inlet DOF treatment for inlet (last) cell. Iterator already moved to last cell
					_jacInlet = _disc.velocity * _disc.DGjacAxConvBlock.col(_disc.DGjacAxConvBlock.cols() - 1); // only last cell depends on inlet concentration
					addLiquidJacBlock(_disc.velocity * _disc.DGjacAxConvBlock.block(0, 0, nNodes, nNodes), jac, 0, idxr, 1);
				}

				return 0;
			}
			/**
			* @brief adds time derivative to the jacobian
			*/
			void addTimederJacobian(double alpha) {

				Indexer idxr(_disc);

				int sNode = idxr.strideColNode();
				unsigned int offC = 0; // inlet DOFs not included in Jacobian

				// =================================================================================================== //
				//	 Time derivative Jacobian: d Residual / d y_t   												   //
				// =================================================================================================== //

				linalg::BandedEigenSparseRowIterator jac(_jacDisc, offC);

				double Beta = (1.0 - _disc.porosity) / _disc.porosity;

				for (unsigned int point = 0; point < _disc.nPoints; point++) {
					for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
						// d convDispRHS / d c_t
						jac[0] += alpha;

						if (_disc.nBound[comp]) { // either one or null; no loop necessary
							// d convDispRHS / d q_t
							jac[idxr.strideColLiquid() - comp + idxr.offsetBoundComp(comp)] += alpha * Beta;
						}
						++jac;
					}
					for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
						if (_disc.nBound[comp]) { // either one or null; no loop over bound states necessary
							if (!_binding[0]->hasQuasiStationaryReactions()) {
								// d isotherm / d q_t
								jac[0] += alpha;
							}
							++jac;
						}
					}
				}

			}

			/**
			* @brief computes the jacobian via finite differences (testing purpose)
			*/
			MatrixXd calcFDJacobian(const double* y_, const double* yDot_, const SimulationTime simTime, util::ThreadLocalStorage& threadLocalMem, double alpha) {

				////// create solution vectors
				Eigen::Map<const VectorXd> hmpf(y_, numDofs());
				VectorXd y = hmpf;
				VectorXd yDot;
				if (yDot_) {
					Eigen::Map<const VectorXd> hmpf2(yDot_, numDofs());
					yDot = hmpf2;

					//// set to LWE initial conditions
					//Indexer idxr(_disc);
					//for(unsigned int blk=0;blk<_disc.nPoints;blk++){
					//	y[idxr.offsetC() + blk * idxr.strideColNode()] = 50.0;
					//	y[idxr.offsetC() + blk * idxr.strideColNode() + idxr.strideColLiquid()] = 1200.0;
					//}
					//yDot.setZero();
				}
				else {
					return MatrixXd::Zero(numDofs(), numDofs());
				}
				//VectorXd y = VectorXd::Zero(numDofs());
				//VectorXd yDot = VectorXd::Zero(numDofs());

				VectorXd res = VectorXd::Zero(numDofs());
				const double* yPtr = &y[0];
				const double* yDotPtr = &yDot[0];
				double* resPtr = &res[0];
				// create FD jacobian
				MatrixXd Jacobian = MatrixXd::Zero(numDofs(), numDofs());
				// set FD step
				double epsilon = 0.01;

				residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, yPtr, yDotPtr, resPtr, threadLocalMem);

				for (int col = 0; col < Jacobian.cols(); col++) {
					Jacobian.col(col) = -(1.0 + alpha) * res;
				}
				/*	 Residual(y+h)	*/
				// state DOFs
				for (int dof = 0; dof < Jacobian.cols(); dof++) {
					y[dof] += epsilon;
					residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, yPtr, yDotPtr, resPtr, threadLocalMem);
					y[dof] -= epsilon;
					Jacobian.col(dof) += res;
				}

				// state derivative Jacobian
				for (int dof = 0; dof < Jacobian.cols(); dof++) {
					yDot[dof] += epsilon;
					residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, yPtr, yDotPtr, resPtr, threadLocalMem);
					yDot[dof] -= epsilon;
					Jacobian.col(dof) += alpha * res;
				}

				////*	exterminate numerical noise	and divide by epsilon*/
				for (int i = 0; i < Jacobian.rows(); i++) {
					for (int j = 0; j < Jacobian.cols(); j++) {
						if (std::abs(Jacobian(i, j)) < 1e-10) Jacobian(i, j) = 0.0;
					}
				}
				Jacobian /= epsilon;

				return Jacobian;
			}

		};

	} // namespace model
} // namespace cadet

#endif  // LIBCADET_LUMPEDRATEMODELWITHOUTPORESDG_HPP_
