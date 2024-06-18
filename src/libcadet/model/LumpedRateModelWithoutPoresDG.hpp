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
#include "model/parts/ConvectionDispersionOperatorDG.hpp"
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
#include <Eigen/Dense>
#include <Eigen/Sparse>

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

			virtual bool configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper);
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
				bool exactInt;	//!< 1 for exact integration, 0 for LGL quadrature

				unsigned int* nBound; //!< Array with number of bound states for each component
				unsigned int* boundOffset; //!< Array with offset to the first bound state of each component in the solid phase
				unsigned int strideBound; //!< Total number of bound states

				int curSection; //!< current section index
				bool newStaticJac; //!< determines wether static analytical jacobian needs to be computed (every section)
			};

			Discretization _disc; //!< Discretization info

		//	IExternalFunction* _extFun; //!< External function (owned by library user)

			// used as auxiliary supplier 
			parts::AxialConvectionDispersionOperatorBaseDG _convDispOp; //!< Convection dispersion operator for interstitial volume transport

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
				virtual bool hasSmoothnessIndicator() const CADET_NOEXCEPT { return _model._convDispOp.hasSmoothnessIndicator(); }

				virtual unsigned int primaryPolynomialDegree() const CADET_NOEXCEPT { return _disc.polyDeg; }
				virtual unsigned int secondaryPolynomialDegree() const CADET_NOEXCEPT { return 0; }
				virtual unsigned int particlePolynomialDegree(unsigned int parType) const CADET_NOEXCEPT { return 0; }
				virtual unsigned int numComponents() const CADET_NOEXCEPT { return _disc.nComp; }
				virtual unsigned int numPrimaryCoordinates() const CADET_NOEXCEPT { return _disc.nPoints; }
				virtual unsigned int numSecondaryCoordinates() const CADET_NOEXCEPT { return 0; }
				virtual unsigned int numPrimaryPolynomialDegree() const CADET_NOEXCEPT { return _disc.polyDeg; }
				virtual unsigned int numSecondaryPolynomialDegree() const CADET_NOEXCEPT { return 0; }
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

				virtual int writeSmoothnessIndicator(double* indicator) const;
				
				/**
				 * @brief calculates the physical node coordinates of the DG discretization with double! interface nodes
				 */
				virtual int writePrimaryCoordinates(double* coords) const
				{
					for (unsigned int i = 0; i < _disc.nCol; i++) {
						for (unsigned int j = 0; j < _disc.nNodes; j++) {
							// mapping 
							coords[i * _disc.nNodes + j] = _model._convDispOp.cellLeftBound(i) + 0.5 * (static_cast<double>(_model._convDispOp.columnLength()) / static_cast<double>(_disc.nCol)) * (1.0 + _model._convDispOp.LGLnodes()[j]);
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
				}
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

				tripletList.reserve(_convDispOp.nConvDispEntries(false) + isotherm_entries + reaction_entries);

				_convDispOp.convDispJacPattern(tripletList);

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
