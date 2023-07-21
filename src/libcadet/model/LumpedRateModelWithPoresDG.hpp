// =============================================================================
//  CADET
//  
//  Copyright � 2008-2021: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

/**
 * @file
 * Defines the lumped rate model with pores (LRMP).
 */

#ifndef LIBCADET_LUMPEDRATEMODELWITHPORESDG_HPP_
#define LIBCADET_LUMPEDRATEMODELWITHPORESDG_HPP_

#include "BindingModel.hpp"
#include "ParallelSupport.hpp"

#include "UnitOperationBase.hpp"
#include "cadet/StrongTypes.hpp"
#include "cadet/SolutionExporter.hpp"
#include "model/parts/ConvectionDispersionOperatorDG.hpp"
#include "AutoDiff.hpp"
#include "linalg/BandedEigenSparseRowIterator.hpp"
#include "Memory.hpp"
#include "model/ModelUtils.hpp"
#include "ParameterMultiplexing.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <vector>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "Benchmark.hpp"

using namespace Eigen;

namespace cadet
{

namespace model
{

class IDynamicReactionModel;

/**
 * @brief Lumped rate model of liquid column chromatography with pores
 * @details See @cite Guiochon2006, @cite Gu1995, @cite Felinger2004
 *
 * @f[\begin{align}
	\frac{\partial c_i}{\partial t} &= - u \frac{\partial c_i}{\partial z} + D_{\text{ax},i} \frac{\partial^2 c_i}{\partial z^2} - \frac{1 - \varepsilon_c}{\varepsilon_c} \frac{3 k_{f,i}}{r_p} j_{f,i} \\
	\frac{\partial c_{p,i}}{\partial t} + \frac{1 - \varepsilon_p}{\varepsilon_p} \frac{\partial q_{i}}{\partial t} &= \frac{3 k_{f,i}}{\varepsilon_p r_p} j_{f,i} \\
	a \frac{\partial q_i}{\partial t} &= f_{\text{iso}}(c_p, q)
\end{align} @f]
@f[ \begin{align}
	j_{f,i} = c_i - c_{p,i}
\end{align} @f]
 * Danckwerts boundary conditions (see @cite Danckwerts1953)
@f[ \begin{align}
u c_{\text{in},i}(t) &= u c_i(t,0) - D_{\text{ax},i} \frac{\partial c_i}{\partial z}(t,0) \\
\frac{\partial c_i}{\partial z}(t,L) &= 0
\end{align} @f]
 * Methods are described in @cite todo, @cite Puttmann2013 @cite Puttmann2016 (forward sensitivities, AD, band compression)
 */
class LumpedRateModelWithPoresDG : public UnitOperationBase
{
public:

	LumpedRateModelWithPoresDG(UnitOpIdx unitOpIdx);
	virtual ~LumpedRateModelWithPoresDG() CADET_NOEXCEPT;

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

	static const char* identifier() { return "LUMPED_RATE_MODEL_WITH_PORES_DG"; }
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

	void solveBulkTimeDerivativeSystem(const SimulationTime& simTime, double* const rhs);
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
			_timerJacobianPar.totalElapsedTime(),
			_timerConsistentInit.totalElapsedTime(),
			_timerConsistentInitPar.totalElapsedTime(),
			_timerLinearSolve.totalElapsedTime(),
			_timerFactorize.totalElapsedTime(),
			_timerFactorizePar.totalElapsedTime(),
			_timerMatVec.totalElapsedTime(),
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
			"JacobianPar",
			"ConsistentInit",
			"ConsistentInitPar",
			"LinearSolve",
			"Factorize",
			"FactorizePar",
			"MatVec",
		};
		return desc;
	}
#endif

protected:

	class Indexer;

	int residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem, bool updateJacobian, bool paramSensitivity);

	template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	int residualImpl(double t, unsigned int secIdx, StateType const* const y, double const* const yDot, ResidualType* const res, util::ThreadLocalStorage& threadLocalMem);

	template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	int residualBulk(double t, unsigned int secIdx, StateType const* yBase, double const* yDotBase, ResidualType* resBase, util::ThreadLocalStorage& threadLocalMem);

	template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	int residualParticle(double t, unsigned int parType, unsigned int colCell, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, util::ThreadLocalStorage& threadLocalMem);

	template <typename StateType, typename ResidualType, typename ParamType>
	int residualFlux(double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res);

	void assembleFluxJacobian(unsigned int secIdx);
	void extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset);

	void assembleDiscretizedGlobalJacobian(double alpha, Indexer idxr);

	void addTimeDerivativeToJacobianParticleBlock(linalg::BandedEigenSparseRowIterator& jac, const Indexer& idxr, double alpha, unsigned int parType);

	unsigned int numAdDirsForJacobian() const CADET_NOEXCEPT;

	int multiplexInitialConditions(const cadet::ParameterId& pId, unsigned int adDirection, double adValue);
	int multiplexInitialConditions(const cadet::ParameterId& pId, double val, bool checkSens);

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
	void checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const;
#endif

	class Discretization
	{
	public:
		unsigned int nParType; //!< Number of particle types
		unsigned int* parTypeOffset; //!< Array with offsets (in particle block) to particle type, additional last element contains total number of particle DOFs
		unsigned int* nBound; //!< Array with number of bound states for each component and particle type (particle type major ordering)
		unsigned int* boundOffset; //!< Array with offset to the first bound state of each component in the solid phase (particle type major ordering)
		unsigned int* strideBound; //!< Total number of bound states for each particle type, additional last element contains total number of bound states for all types
		unsigned int* nBoundBeforeType; //!< Array with number of bound states before a particle type (cumulative sum of strideBound)
		unsigned int nComp; //!< Number of components

		bool exactInt;	//!< 1 for exact integration, 0 for LGL quadrature
		unsigned int nCol; //!< Number of column cells
		unsigned int polyDeg; //!< polynomial degree
		unsigned int nNodes; //!< Number of nodes per cell
		unsigned int nPoints; //!< Number of discrete Points

		bool curSection; //!< current section index

		bool newStaticJac;
	};

	Discretization _disc; //!< Discretization info
//	IExternalFunction* _extFun; //!< External function (owned by library user)

	parts::AxialConvectionDispersionOperatorBaseDG _convDispOp; //!< Convection dispersion operator for interstitial volume transport
	IDynamicReactionModel* _dynReactionBulk; //!< Dynamic reactions in the bulk volume

	Eigen::SparseLU<Eigen::SparseMatrix<double>> _globalSolver; //!< linear solver for the bulk concentration
	//Eigen::BiCGSTAB<Eigen::SparseMatrix<double, RowMajor>, Eigen::DiagonalPreconditioner<double>> _globalSolver;

	Eigen::MatrixXd _jacInlet; //!< Jacobian inlet DOF block matrix connects inlet DOFs to first bulk cells

	// for FV the bulk jacobians are defined in the ConvDisp operator.
	Eigen::SparseMatrix<double, RowMajor> _globalJac; //!< global Jacobian
	Eigen::SparseMatrix<double, RowMajor> _globalJacDisc; //!< global Jacobian with time derivatove from BDF method
	//Eigen::MatrixXd _FDjac; //!< test purpose FD jacobian // todo delete

	active _colPorosity; //!< Column porosity (external porosity) \f$ \varepsilon_c \f$
	std::vector<double> _parGeomSurfToVol; //!< Particle surface to volume ratio factor (i.e., 3.0 for spherical, 2.0 for cylindrical, 1.0 for hexahedral)
	std::vector<active> _parRadius; //!< Particle radius \f$ r_p \f$
	bool _singleParRadius;
	std::vector<active> _parPorosity; //!< Particle porosity (internal porosity) \f$ \varepsilon_p \f$
	bool _singleParPorosity;
	std::vector<active> _parTypeVolFrac; //!< Volume fraction of each particle type

	// Vectorial parameters
	std::vector<active> _filmDiffusion; //!< Film diffusion coefficient \f$ k_f \f$
	MultiplexMode _filmDiffusionMode;
	std::vector<active> _poreAccessFactor; //!< Pore accessibility factor \f$ F_{\text{acc}} \f$
	MultiplexMode _poreAccessFactorMode;

	bool _axiallyConstantParTypeVolFrac; //!< Determines whether particle type volume fraction is homogeneous across axial coordinate
	bool _analyticJac; //!< Determines whether AD or analytic Jacobians are used
	unsigned int _jacobianAdDirs; //!< Number of AD seed vectors required for Jacobian computation

	bool _factorizeJacobian; //!< Determines whether the Jacobian needs to be factorized
	double* _tempState; //!< Temporary storage with the size of the state vector or larger if binding models require it

	std::vector<active> _initC; //!< Liquid bulk phase initial conditions
	std::vector<active> _initCp; //!< Liquid particle phase initial conditions
	std::vector<active> _initQ; //!< Solid phase initial conditions
	std::vector<double> _initState; //!< Initial conditions for state vector if given
	std::vector<double> _initStateDot; //!< Initial conditions for time derivative

	BENCH_TIMER(_timerResidual)
		BENCH_TIMER(_timerResidualPar)
		BENCH_TIMER(_timerResidualSens)
		BENCH_TIMER(_timerResidualSensPar)
		BENCH_TIMER(_timerJacobianPar)
		BENCH_TIMER(_timerConsistentInit)
		BENCH_TIMER(_timerConsistentInitPar)
		BENCH_TIMER(_timerLinearSolve)
		BENCH_TIMER(_timerFactorize)
		BENCH_TIMER(_timerFactorizePar)
		BENCH_TIMER(_timerMatVec)

	class Indexer
	{
	public:
		Indexer(const Discretization& disc) : _disc(disc) { }

		// Strides
		inline int strideColNode() const CADET_NOEXCEPT { return static_cast<int>(_disc.nComp); }
		inline int strideColCell() const CADET_NOEXCEPT { return static_cast<int>(_disc.nNodes * strideColNode()); }
		inline int strideColComp() const CADET_NOEXCEPT { return 1; }

		inline int strideParComp() const CADET_NOEXCEPT { return 1; }
		inline int strideParLiquid() const CADET_NOEXCEPT { return static_cast<int>(_disc.nComp); }
		inline int strideParBound(int parType) const CADET_NOEXCEPT { return static_cast<int>(_disc.strideBound[parType]); }
		inline int strideParBlock(int parType) const CADET_NOEXCEPT { return strideParLiquid() + strideParBound(parType); }

		// Offsets
		inline int offsetC() const CADET_NOEXCEPT { return _disc.nComp; }
		inline int offsetCp() const CADET_NOEXCEPT { return _disc.nComp * _disc.nPoints + offsetC(); }
		inline int offsetCp(ParticleTypeIndex pti) const CADET_NOEXCEPT { return offsetCp() + _disc.parTypeOffset[pti.value]; }
		inline int offsetCp(ParticleTypeIndex pti, ParticleIndex pi) const CADET_NOEXCEPT { return offsetCp(pti) + strideParBlock(pti.value) * pi.value; }
		inline int offsetBoundComp(ParticleTypeIndex pti, ComponentIndex comp) const CADET_NOEXCEPT { return _disc.boundOffset[pti.value * _disc.nComp + comp.value]; }

		// Return pointer to first element of state variable in state vector
		template <typename real_t> inline real_t* c(real_t* const data) const { return data + offsetC(); }
		template <typename real_t> inline real_t const* c(real_t const* const data) const { return data + offsetC(); }

		template <typename real_t> inline real_t* cp(real_t* const data) const { return data + offsetCp(); }
		template <typename real_t> inline real_t const* cp(real_t const* const data) const { return data + offsetCp(); }

		template <typename real_t> inline real_t* q(real_t* const data) const { return data + offsetCp() + strideParLiquid(); }
		template <typename real_t> inline real_t const* q(real_t const* const data) const { return data + offsetCp() + strideParLiquid(); }

		// Return specific variable in state vector
		template <typename real_t> inline real_t& c(real_t* const data, unsigned int point, unsigned int comp) const { return data[offsetC() + comp + point * strideColNode()]; }
		template <typename real_t> inline const real_t& c(real_t const* const data, unsigned int point, unsigned int comp) const { return data[offsetC() + comp + point * strideColNode()]; }

	protected:
		const Discretization& _disc;
	};

	class Exporter : public ISolutionExporter
	{
	public:

		Exporter(const Discretization& disc, const LumpedRateModelWithPoresDG& model, double const* data) : _disc(disc), _idx(disc), _model(model), _data(data) { }
		Exporter(const Discretization&& disc, const LumpedRateModelWithPoresDG& model, double const* data) = delete;

		virtual bool hasParticleFlux() const CADET_NOEXCEPT { return false; }
		virtual bool hasParticleMobilePhase() const CADET_NOEXCEPT { return true; }
		virtual bool hasSolidPhase() const CADET_NOEXCEPT { return _disc.strideBound[_disc.nParType] > 0; }
		virtual bool hasVolume() const CADET_NOEXCEPT { return false; }
		virtual bool hasSmoothnessIndicator() const CADET_NOEXCEPT { return _model._convDispOp.hasSmoothnessIndicator(); }

		virtual unsigned int primaryPolynomialDegree() const CADET_NOEXCEPT { return _disc.polyDeg; }
		virtual unsigned int secondaryPolynomialDegree() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int particlePolynomialDegree(unsigned int parType) const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numComponents() const CADET_NOEXCEPT { return _disc.nComp; }
		virtual unsigned int numPrimaryCoordinates() const CADET_NOEXCEPT { return _disc.nPoints; }
		virtual unsigned int numSecondaryCoordinates() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numParticleTypes() const CADET_NOEXCEPT { return _disc.nParType; }
		virtual unsigned int numParticleShells(unsigned int parType) const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numBoundStates(unsigned int parType) const CADET_NOEXCEPT { return _disc.strideBound[parType]; }
		virtual unsigned int numMobilePhaseDofs() const CADET_NOEXCEPT { return _disc.nComp * _disc.nPoints; }
		virtual unsigned int numParticleMobilePhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return _disc.nComp * _disc.nPoints; }
		virtual unsigned int numParticleMobilePhaseDofs() const CADET_NOEXCEPT { return _disc.nComp * _disc.nPoints * _disc.nParType; }
		virtual unsigned int numSolidPhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return _disc.strideBound[parType] * _disc.nPoints; }
		virtual unsigned int numSolidPhaseDofs() const CADET_NOEXCEPT {
			unsigned int nDofPerParType = 0;
			for (unsigned int i = 0; i < _disc.nParType; ++i)
				nDofPerParType += _disc.strideBound[i];
			return _disc.nPoints * nDofPerParType;
		}
		virtual unsigned int numParticleFluxDofs() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numVolumeDofs() const CADET_NOEXCEPT { return 0; }

		virtual int writeMobilePhase(double* buffer) const;
		virtual int writeSolidPhase(double* buffer) const;
		virtual int writeParticleMobilePhase(double* buffer) const;
		virtual int writeSolidPhase(unsigned int parType, double* buffer) const;
		virtual int writeParticleMobilePhase(unsigned int parType, double* buffer) const;
		virtual int writeParticleFlux(double* buffer) const { return 0; }
		virtual int writeParticleFlux(unsigned int parType, double* buffer) const { return 0; }
		virtual int writeVolume(double* buffer) const { return 0; }
		virtual int writeInlet(unsigned int port, double* buffer) const;
		virtual int writeInlet(double* buffer) const;
		virtual int writeOutlet(unsigned int port, double* buffer) const;
		virtual int writeOutlet(double* buffer) const;

		virtual int writeSmoothnessIndicator(double* indicator) const { return 0; }
		/**
		* @brief calculates and writes the physical node coordinates of the DG discretization with double! interface nodes
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
		virtual int writeParticleCoordinates(unsigned int parType, double* coords) const
		{
			coords[0] = static_cast<double>(_model._parRadius[parType]) * 0.5;
			return 1;
		}

	protected:
		const Discretization& _disc;
		const Indexer _idx;
		const LumpedRateModelWithPoresDG& _model;
		double const* const _data;
	};

	// ==========================================================================================================================================================  //
	// ========================================						DG Jacobian							=========================================================  //
	// ==========================================================================================================================================================  //

	typedef Eigen::Triplet<double> T;

	/**
	* @brief sets the sparsity pattern of the convection dispersion Jacobian
	*/
	void setGlobalJacPattern(Eigen::SparseMatrix<double, RowMajor>& mat, const bool hasBulkReaction) {

		std::vector<T> tripletList;

		tripletList.reserve(nJacEntries(hasBulkReaction));

		_convDispOp.convDispJacPattern(tripletList);

		if (hasBulkReaction)
			bulkReactionPattern(tripletList);

		particlePattern(tripletList);

		fluxPattern(tripletList);

		mat.setFromTriplets(tripletList.begin(), tripletList.end());

	}
	/**
	 * @brief sets the sparsity pattern of the bulkreaction  Jacobian pattern
	 */
	int bulkReactionPattern(std::vector<T>& tripletList) {

		Indexer idxr(_disc);

		for (unsigned int blk = 0; blk < _disc.nPoints; blk++) {
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int toComp = 0; toComp < _disc.nComp; toComp++) {
					tripletList.push_back(T(idxr.offsetC() + blk * idxr.strideColNode() + comp * idxr.strideColComp(),
						idxr.offsetC() + blk * idxr.strideColNode() + toComp * idxr.strideColComp(),
						0.0));
				}
			}
		}
		return 1;
	}
	/**
	 * @brief sets the sparsity pattern of the particle Jacobian pattern (isotherm, reaction pattern)
	 */
	int particlePattern(std::vector<T>& tripletList) {

		Indexer idxr(_disc);

		for (unsigned int parType = 0; parType < _disc.nParType; parType++) {
			for (unsigned int nCol = 0; nCol < _disc.nPoints; nCol++) {

				int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ nCol }) - idxr.offsetC(); // inlet DOFs not included in Jacobian

				// add dense nComp * nBound blocks, since all solid and liquid entries can be coupled through binding.
				for (unsigned int parState = 0; parState < _disc.nComp + _disc.strideBound[parType]; parState++) {
					for (unsigned int toParState = 0; toParState < _disc.nComp + _disc.strideBound[parType]; toParState++) {
						tripletList.push_back(T(offset + parState, offset + toParState, 0.0));
					}
				}
			}
		}
		return 1;
	}

	/**
	 * @brief sets the sparsity pattern of the flux Jacobian pattern
	 */
	int fluxPattern(std::vector<T>& tripletList) {
		
		Indexer idxr(_disc);

		for (unsigned int parType = 0; parType < _disc.nParType; parType++) {
			
			int offC = 0; // inlet DOFs not included in Jacobian
			int offP = idxr.offsetCp(ParticleTypeIndex{ parType }) - idxr.offsetC(); // inlet DOFs not included in Jacobian

			// add dependency of c^b, c^p and flux on another
			for (unsigned int nCol = 0; nCol < _disc.nPoints; nCol++) {
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					// c^b on c^b entry already set
					tripletList.push_back(T(offC + nCol * _disc.nComp + comp, offP + nCol * idxr.strideParBlock(parType) + comp, 0.0)); // c^b on c^p
					// c^p on c^p entry already set
					tripletList.push_back(T(offP + nCol * idxr.strideParBlock(parType) + comp, offC + nCol * _disc.nComp + comp, 0.0)); // c^p on c^b
				}
			}
		}
		return 1;
	}
	/**
	 * @brief analytically calculates the static (per section) bulk jacobian (inlet DOFs included!)
	 * @return 1 if jacobain estimation fits the predefined pattern of the jacobian, 0 if not.
	 */
	int calcStaticAnaGlobalJacobian(unsigned int secIdx) {
		
		bool success = _convDispOp.calcStaticAnaJacobian(_globalJac, _jacInlet);
		success = success && calcFluxJacobians(secIdx);

		return success;
	}

	int calcFluxJacobians(unsigned int secIdx, bool ADjac = false) {

		Indexer idxr(_disc);

		const double invBetaC = 1.0 / static_cast<double>(_colPorosity) - 1.0;

		for (unsigned int type = 0; type < _disc.nParType; type++) {

			const double epsP = static_cast<double>(_parPorosity[type]);
			const double radius = static_cast<double>(_parRadius[type]);
			const double jacCF_val = invBetaC * _parGeomSurfToVol[type] / radius;
			const double jacPF_val = -_parGeomSurfToVol[type] / (radius * epsP);

			// Ordering of diffusion:
			// sec0type0comp0, sec0type0comp1, sec0type0comp2, sec0type1comp0, sec0type1comp1, sec0type1comp2,
			// sec1type0comp0, sec1type0comp1, sec1type0comp2, sec1type1comp0, sec1type1comp1, sec1type1comp2, ...
			active const* const filmDiff = getSectionDependentSlice(_filmDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;
			active const* const poreAccFactor = _poreAccessFactor.data() + type * _disc.nComp;

			linalg::BandedEigenSparseRowIterator jacC(_globalJac, 0);
			linalg::BandedEigenSparseRowIterator jacP(_globalJac, idxr.offsetCp(ParticleTypeIndex{ type }) - idxr.offsetC());

			for (unsigned int colNode = 0; colNode < _disc.nPoints; colNode++, jacP += _disc.strideBound[type])
			{
				for (unsigned int comp = 0; comp < _disc.nComp; comp++, ++jacC, ++jacP) {

					// add Cl on Cl entries (added since already set in bulk jacobian)
					// row: already at bulk phase. already at current node and component.
					// col: already at bulk phase. already at current node and component.
					if (!ADjac) // entry already filled for AD
						jacC[0] += jacCF_val * static_cast<double>(filmDiff[comp]) * static_cast<double>(_parTypeVolFrac[type + _disc.nParType * colNode]);

					// add Cl on Cp entries
					// row: already at bulk phase. already at current node and component.
					// col: already at bulk phase. already at current node and component.
					jacC[jacP.row() - jacC.row()] = -jacCF_val * static_cast<double>(filmDiff[comp]) * static_cast<double>(_parTypeVolFrac[type + _disc.nParType * colNode]);

					// add Cp on Cp entries
					// row: already at particle. already at current node and liquid state.
					// col: go to flux of current parType and adjust for offsetC. jump over previous colNodes and add component offset
					if (!ADjac) // entry already filled for AD
						jacP[0] += -jacPF_val / static_cast<double>(poreAccFactor[comp]) * static_cast<double>(filmDiff[comp]);

					// add Cp on Cl entries
					// row: already at particle. already at current node and liquid state.
					// col: go to flux of current parType and adjust for offsetC. jump over previous colNodes and add component offset
					jacP[jacC.row() - jacP.row()]
						= jacPF_val / static_cast<double>(poreAccFactor[comp]) * static_cast<double>(filmDiff[comp]);
				}
			}
		}
		return 1;
	}
	/**
	 * @brief calculates the number of entris for the DG convection dispersion jacobian
	 * @note only dispersion entries are relevant for jacobian NNZ as the convection entries are a subset of these
	 */
	unsigned int nJacEntries(const bool hasBulkReaction, const bool pureNNZ = false) {

		unsigned int nEntries = 0;
		// Convection dispersion
		nEntries = _convDispOp.nConvDispEntries(false);

		// Bulk reaction entries
		if (hasBulkReaction)
			nEntries += _disc.nPoints * _disc.nComp * _disc.nComp; // add nComp entries for every component at each discrete bulk point

		// Particle binding and reaction entries
		for (unsigned int type = 0; type < _disc.nParType; type++)
			nEntries += _disc.nComp + _disc.nBoundBeforeType[type];

		return nEntries;
	}

	// testing purpose
	MatrixXd calcFDJacobian(const SimulationTime simTime, util::ThreadLocalStorage& threadLocalMem, double alpha) {

		// create solution vectors
		VectorXd y = VectorXd::Zero(numDofs());
		VectorXd yDot = VectorXd::Zero(numDofs());
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

		///*	exterminate numerical noise	and divide by epsilon*/
		//for (int i = 0; i < Jacobian.rows(); i++) {
		//	for (int j = 0; j < Jacobian.cols(); j++) {
		//		if (std::abs(Jacobian(i, j)) < 1e-10) Jacobian(i, j) = 0.0;
		//	}
		//}
		Jacobian /= epsilon;

		return Jacobian;
	}

};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_LUMPEDRATEMODELWITHPORESDG_HPP_