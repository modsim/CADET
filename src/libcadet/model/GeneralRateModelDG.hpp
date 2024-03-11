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
 * Defines the general rate model (GRM).
 */

#ifndef LIBCADET_GENERALRATEMODELDG_HPP_
#define LIBCADET_GENERALRATEMODELDG_HPP_

#include "model/UnitOperationBase.hpp"
#include "model/BindingModel.hpp"
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

namespace parts
{
	namespace cell
	{
		struct CellParameters;
	}
}

class IDynamicReactionModel;
class IParameterStateDependence;

/**
 * @brief General rate model of liquid column chromatography
 * @details See @cite Guiochon2006, @cite Gu1995, @cite Felinger2004
 * 
 * @f[\begin{align}
	\frac{\partial c_i}{\partial t} &= - u \frac{\partial c_i}{\partial z} + D_{\text{ax},i} \frac{\partial^2 c_i}{\partial z^2} - \frac{1 - \varepsilon_c}{\varepsilon_c} \frac{3}{r_p} j_{f,i} \\
	\frac{\partial c_{p,i}}{\partial t} + \frac{1 - \varepsilon_p}{\varepsilon_p} \frac{\partial q_{i}}{\partial t} &= D_{p,i} \left( \frac{\partial^2 c_{p,i}}{\partial r^2} + \frac{2}{r} \frac{\partial c_{p,i}}{\partial r} \right) + D_{s,i} \frac{1 - \varepsilon_p}{\varepsilon_p} \left( \frac{\partial^2 q_{i}}{\partial r^2} + \frac{2}{r} \frac{\partial q_{i}}{\partial r} \right) \\
	a \frac{\partial q_i}{\partial t} &= f_{\text{iso}}(c_p, q)
\end{align} @f]
@f[ \begin{align}
	j_{f,i} = k_{f,i} \left( c_i - c_{p,i} \left(\cdot, \cdot, r_p\right)\right)
\end{align} @f]
 * Danckwerts boundary conditions (see @cite Danckwerts1953)
@f[ \begin{align}
u c_{\text{in},i}(t) &= u c_i(t,0) - D_{\text{ax},i} \frac{\partial c_i}{\partial z}(t,0) \\
\frac{\partial c_i}{\partial z}(t,L) &= 0 \\
\varepsilon_p D_{p,i} \frac{\partial c_{p,i}}{\partial r}(\cdot, \cdot, r_p) + (1-\varepsilon_p) D_{s,i} \frac{\partial q_{i}}{\partial r}(\cdot, \cdot, r_p) &= j_{f,i} \\
\frac{\partial c_{p,i}}{\partial r}(\cdot, \cdot, 0) &= 0
\end{align} @f]
 * Methods are described in @cite Breuer2023 (DGSEM discretization), @cite Puttmann2013 @cite Puttmann2016 (forward sensitivities, AD, band compression)
 */
class GeneralRateModelDG : public UnitOperationBase
{
public:

	GeneralRateModelDG(UnitOpIdx unitOpIdx);
	virtual ~GeneralRateModelDG() CADET_NOEXCEPT;

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

	static const char* identifier() { return "GENERAL_RATE_MODEL_DG"; }
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
	virtual bool setParameter(const ParameterId& pId, int value);
	virtual bool setParameter(const ParameterId& pId, bool value);
	virtual bool setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue);
	virtual void setSensitiveParameterValue(const ParameterId& id, double value);

	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const;
	virtual double getParameterDouble(const ParameterId& pId) const;
	virtual bool hasParameter(const ParameterId& pId) const;

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
			_timerGmres.totalElapsedTime(),
			static_cast<double>(_gmres.numIterations())
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
			"Gmres",
			"NumGMRESIter"
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
	int residualBulk(double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, util::ThreadLocalStorage& threadLocalMem);

	template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	int residualParticle(double t, unsigned int parType, unsigned int colCell, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, util::ThreadLocalStorage& threadLocalMem);

	template <typename StateType, typename ResidualType, typename ParamType>
	int residualFlux(double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res);

	void extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset);

	void assembleDiscretizedGlobalJacobian(double alpha, Indexer idxr);

	void setEquidistantRadialDisc(unsigned int parType);
	void setEquivolumeRadialDisc(unsigned int parType);
	void setUserdefinedRadialDisc(unsigned int parType);
	void updateRadialDisc();

	void addTimeDerivativeToJacobianParticleShell(linalg::BandedEigenSparseRowIterator& jac, const Indexer& idxr, double alpha, unsigned int parType);

	unsigned int numAdDirsForJacobian() const CADET_NOEXCEPT;

	int multiplexInitialConditions(const cadet::ParameterId& pId, unsigned int adDirection, double adValue);
	int multiplexInitialConditions(const cadet::ParameterId& pId, double val, bool checkSens);

	void clearParDepSurfDiffusion();

	parts::cell::CellParameters makeCellResidualParams(unsigned int parType, int const* qsReaction) const;

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
	void checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const;
#endif

	struct Discretization
	{
		unsigned int nComp; //!< Number of components
		unsigned int nCol; //!< Number of column cells
		unsigned int polyDeg; //!< polynomial degree of column elements
		unsigned int nNodes; //!< Number of nodes per column cell
		unsigned int nPoints; //!< Number of discrete column Points
		bool exactInt;	//!< 1 for exact integration, 0 for inexact LGL quadrature
		unsigned int nParType; //!< Number of particle types
		unsigned int* nParCell; //!< Array with number of radial cells in each particle type
		unsigned int* nParPointsBeforeType; //!< Array with total number of radial points before a particle type (cumulative sum of nParPoints), additional last element contains total number of particle shells
		unsigned int* parPolyDeg; //!< polynomial degree of particle elements
		unsigned int* nParNode; //!< Array with number of radial nodes per cell in each particle type
		unsigned int* nParPoints; //!< Array with number of radial nodes per cell in each particle type
		bool* parExactInt; //!< 1 for exact integration, 0 for inexact LGL quadrature for each particle type
		bool* parGSM; //!< specifies whether (single element) Galerkin spectral method should be used in particles
		unsigned int* parTypeOffset; //!< Array with offsets (in particle block) to particle type, additional last element contains total number of particle DOFs
		unsigned int* nBound; //!< Array with number of bound states for each component and particle type (particle type major ordering)
		unsigned int* boundOffset; //!< Array with offset to the first bound state of each component in the solid phase (particle type major ordering)
		unsigned int* strideBound; //!< Total number of bound states for each particle type, additional last element contains total number of bound states for all types
		unsigned int* nBoundBeforeType; //!< Array with number of bound states before a particle type (cumulative sum of strideBound)

		bool newStaticJac; //!< determines wether static analytical jacobian is to be computed

		// parameter
		unsigned int* offsetSurfDiff; //!< particle surface diffusion (may be section and component dependent)
		int curSection; //!< current time section index

		const double SurfVolRatioSlab = 1.0; //!< Surface to volume ratio for a slab-shaped particle
		const double SurfVolRatioCylinder = 2.0; //!< Surface to volume ratio for a cylindrical particle
		const double SurfVolRatioSphere = 3.0; //!< Surface to volume ratio for a spherical particle

		/*					DG specifics				*/
		
		active* deltaR; //!< equidistant particle element spacing for each particle type
		Eigen::VectorXd* parNodes; //!< Array with positions of nodes in radial reference element for each particle
		Eigen::MatrixXd* parPolyDerM; //!< Array with polynomial derivative Matrix for each particle
		Eigen::MatrixXd* minus_InvMM_ST; //!< equals minus inverse mass matrix times transposed stiffness matrix. Required solely for exact integration DG discretization of particle equation
		Eigen::VectorXd* parInvWeights; //!< Array with weights for LGL quadrature of size nNodes for each particle
		Eigen::MatrixXd* parInvMM; //!< dense inverse mass matrix for exact integration of integrals with metrics, for each particle
		Eigen::MatrixXd* parInvMM_Leg; //!< dense inverse mass matrix (Legendre) for exact integration of integral without metric, for each particle
		Eigen::MatrixXd* secondOrderStiffnessM; //!< specific second order stiffness matrix
		Eigen::MatrixXd* minus_parInvMM_Ar; //!< inverse mass matrix times specific second order stiffness matrix
		Eigen::Vector<active, Dynamic>* Ir; //!< metric part for each particle type and cell, particle type major ordering
		Eigen::MatrixXd* Dr; //!< derivative matrices including metrics for each particle type and cell, particle type major ordering
		Eigen::VectorXi offsetMetric; //!< offset required to access metric dependent DG operator storage of Ir, Dr -> summed up nCells of all previous parTypes

		Eigen::MatrixXd* DGjacParDispBlocks; //!< particle dispersion blocks of DG jacobian

		Eigen::Vector<active, Dynamic>* g_p; //!< auxiliary variable g = dc_p / dr
		Eigen::Vector<active, Dynamic>* g_pSum; //!< auxiliary variable g = sum_{k \in p, s_i} dc_k / dr
		Eigen::Vector<active, Dynamic>* surfaceFluxParticle; //!< stores the surface flux values for each particle
		active* localFlux; //!< stores the local (at respective particle) film diffusion flux

		/**
		* @brief allocates memory for DG operators and computes those that are metric independent. Also allocates required containers needed for the DG discretization.
		*/
		void initializeDG() {

			/* Allocate space for DG operators and containers */
			
			newStaticJac = true;

			// particles
			nParNode = new unsigned int [nParType];
			nParPoints = new unsigned int [nParType];
			g_p = new Vector<active, Dynamic>[nParType];
			g_pSum = new Vector<active, Dynamic>[nParType];
			surfaceFluxParticle = new Vector<active, Dynamic>[nParType];
			parNodes = new VectorXd [nParType];
			parInvWeights = new VectorXd [nParType];
			parInvMM_Leg = new MatrixXd [nParType];
			parPolyDerM = new MatrixXd[nParType];
			localFlux = new active[nComp];

			for (int parType = 0; parType < nParType; parType++) 
			{
				nParNode[parType] = parPolyDeg[parType] + 1u;
				nParPoints[parType] = nParNode[parType] * nParCell[parType];
				g_p[parType].resize(nParPoints[parType]);
				g_p[parType].setZero();
				g_pSum[parType].resize(nParPoints[parType]);
				g_pSum[parType].setZero();
				surfaceFluxParticle[parType].resize(nParCell[parType] + 1);
				surfaceFluxParticle[parType].setZero();
				parNodes[parType].resize(nParNode[parType]);
				parNodes[parType].setZero();
				parInvWeights[parType].resize(nParNode[parType]);
				parInvWeights[parType].setZero();
				parPolyDerM[parType].resize(nParNode[parType], nParNode[parType]);
				parPolyDerM[parType].setZero();
				parInvMM_Leg[parType].resize(nParNode[parType], nParNode[parType]);
				parInvMM_Leg[parType].setZero();
			}

			offsetMetric = VectorXi::Zero(nParType + 1);
			for (int parType = 1; parType <= nParType; parType++) {
				offsetMetric[parType] += nParCell[parType - 1];
			}
			Dr = new MatrixXd[offsetMetric[nParType]];
			Ir = new Vector<active, Dynamic>[offsetMetric[nParType]];
			minus_InvMM_ST = new MatrixXd[offsetMetric[nParType]];
			parInvMM = new MatrixXd[offsetMetric[nParType]];
			secondOrderStiffnessM = new MatrixXd[nParType];
			minus_parInvMM_Ar = new MatrixXd[nParType];
			
			/* compute metric independent DG operators for bulk and particles. Note that metric dependent DG operators are computet in updateRadialDisc(). */

			for (int parType = 0; parType < nParType; parType++)
			{
				parts::dgtoolbox::lglNodesWeights(parPolyDeg[parType], parNodes[parType], parInvWeights[parType], true);
				parPolyDerM[parType] = parts::dgtoolbox::derivativeMatrix(parPolyDeg[parType], parNodes[parType]);
				parInvMM_Leg[parType] = parts::dgtoolbox::invMMatrix(parPolyDeg[parType], parNodes[parType], 0.0, 0.0);
			}
		}

		void initializeDGjac(std::vector<double> parGeomSurfToVol) {

			// particle jacobian blocks (each is unique)
			DGjacParDispBlocks = new MatrixXd[std::accumulate(nParCell, nParCell + nParType, 0)];

			for (unsigned int type = 0; type < nParType; type++)
			{
				for (unsigned int block = 0; block < nParCell[type]; block++)
				{
					if (parGSM[type])
						DGjacParDispBlocks[offsetMetric[type] + block] = GSMjacobianParDispBlock(type, parGeomSurfToVol[type]);
					else
						DGjacParDispBlocks[offsetMetric[type] + block] = DGjacobianParDispBlock(block + 1u, type, parGeomSurfToVol[type]);
				}
			}
		}

	private:

		/**
		 * @brief calculates the DG Jacobian auxiliary block
		 * @param [in] exInt true if exact integration DG scheme
		 * @param [in] cellIdx cell index
		 */
		MatrixXd getParGBlock(unsigned int cellIdx, unsigned int parType) {

			// Auxiliary Block [ d g(c) / d c ], additionally depends on boundary entries of neighbouring cells
			MatrixXd gBlock = MatrixXd::Zero(nParNode[parType], nParNode[parType] + 2);
			gBlock.block(0, 1, nParNode[parType], nParNode[parType]) = parPolyDerM[parType];
			if (parExactInt[parType]) {
				if (cellIdx == 0 || cellIdx == nParCell[parType] + 1) { // cellIdx out of bounds
					return MatrixXd::Zero(nParNode[parType], nParNode[parType] + 2);
				}
				if (cellIdx != 1 && cellIdx != nParCell[parType]) { // inner cell
					gBlock.block(0, 0, nParNode[parType], 1) -= 0.5 * parInvMM_Leg[parType].block(0, 0, nParNode[parType], 1);
					gBlock.block(0, 1, nParNode[parType], 1) += 0.5 * parInvMM_Leg[parType].block(0, 0, nParNode[parType], 1);
					gBlock.block(0, nParNode[parType], nParNode[parType], 1) -= 0.5 * parInvMM_Leg[parType].block(0, nParNode[parType] - 1, nParNode[parType], 1);
					gBlock.block(0, nParNode[parType] + 1, nParNode[parType], 1) += 0.5 * parInvMM_Leg[parType].block(0, nParNode[parType] - 1, nParNode[parType], 1);
				}
				else if (cellIdx == 1u) { // left boundary cell
					if (cellIdx == nParCell[parType]) // special case one cell
						return gBlock * 2.0 / static_cast<double>(deltaR[offsetMetric[parType] + (cellIdx - 1)]);
					gBlock.block(0, nParNode[parType], nParNode[parType], 1) -= 0.5 * parInvMM_Leg[parType].block(0, nParNode[parType] - 1, nParNode[parType], 1);
					gBlock.block(0, nParNode[parType] + 1, nParNode[parType], 1) += 0.5 * parInvMM_Leg[parType].block(0, nParNode[parType] - 1, nParNode[parType], 1);
				}
				else if (cellIdx == nParCell[parType]) { // right boundary cell
					gBlock.block(0, 0, nParNode[parType], 1) -= 0.5 * parInvMM_Leg[parType].block(0, 0, nParNode[parType], 1);
					gBlock.block(0, 1, nParNode[parType], 1) += 0.5 * parInvMM_Leg[parType].block(0, 0, nParNode[parType], 1);
				}
				gBlock *= 2.0 / static_cast<double>(deltaR[offsetMetric[parType] + (cellIdx - 1)]);
			}
			else {
				// inexact integration not maintained due to inferior performance. Code is part of calcParticleCollocationDGSEMJacobian()
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
		Eigen::MatrixXd parAuxBlockGstar(unsigned int cellIdx, unsigned int parType, MatrixXd leftG, MatrixXd middleG, MatrixXd rightG) {

			// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
			MatrixXd gStarDC = MatrixXd::Zero(nParNode[parType], 3 * nParNode[parType] + 2);
			// NOTE: N = polyDeg
			// indices  gStarDC    :     0   ,   1   , ..., nNodes; nNodes+1, ..., 2 * nNodes;	2*nNodes+1, ..., 3 * nNodes; 3*nNodes+1
			// derivative index j  : -(N+1)-1, -(N+1),... ,  -1   ;   0     , ...,		N	 ;	  N + 1	  , ..., 2N + 2    ; 2(N+1) +1
			// auxiliary block [d g^* / d c]
			if (cellIdx != 1) {
				gStarDC.block(0, nParNode[parType], 1, nParNode[parType] + 2) += middleG.block(0, 0, 1, nParNode[parType] + 2);
				gStarDC.block(0, 0, 1, nParNode[parType] + 2) += leftG.block(nParNode[parType] - 1, 0, 1, nParNode[parType] + 2);
			}
			if (cellIdx != nParCell[parType]) {
				gStarDC.block(nParNode[parType] - 1, nParNode[parType], 1, nParNode[parType] + 2) += middleG.block(nParNode[parType] - 1, 0, 1, nParNode[parType] + 2);
				gStarDC.block(nParNode[parType] - 1, 2 * nParNode[parType], 1, nParNode[parType] + 2) += rightG.block(0, 0, 1, nParNode[parType] + 2);
			}
			gStarDC *= 0.5;

			return gStarDC;
		}

		Eigen::MatrixXd getParBMatrix(int parType, int cell, double parGeomSurfToVol) {
			// also known as "lifting" matrix and includes metric dependent terms for particle discretization
			MatrixXd B = MatrixXd::Zero(nParNode[parType], nParNode[parType]);
			if (parGeomSurfToVol == SurfVolRatioSlab) {
				B(0, 0) = -1.0;
				B(nParNode[parType] - 1, nParNode[parType] - 1) = 1.0;
			}
			else {
				B(0, 0) = -static_cast<double>(Ir[offsetMetric[parType] + (cell - 1)][0]);
				B(nParNode[parType] - 1, nParNode[parType] - 1) = static_cast<double>(Ir[offsetMetric[parType] + (cell - 1)][nParNode[parType] - 1]);
			}

			return B;
		}
		/**
		 * @brief calculates the dispersion part of the DG jacobian
		 * @param [in] parType particle type index
		 * @param [in] parGeomSurfToVol particle geometry
		 */
		Eigen::MatrixXd GSMjacobianParDispBlock(unsigned int parType, double parGeomSurfToVol) {

			MatrixXd dispBlock;

			// We have to match the DGSEM interface, where the dispersion block [ d RHS_disp / d c ] depends on whole previous and subsequent cell plus first entries of subsubsequent cells
			dispBlock = MatrixXd::Zero(nParNode[parType], 3 * nParNode[parType] + 2);

			dispBlock.block(0, nParNode[parType] + 1, nParNode[parType], nParNode[parType]) = minus_parInvMM_Ar[parType];
			dispBlock *= 2.0 / static_cast<double>(deltaR[offsetMetric[parType]]) * 2.0 / static_cast<double>(deltaR[offsetMetric[parType]]);

			return -dispBlock; // *-1 for residual
		}
		/**
		 * @brief calculates the dispersion part of the DG jacobian
		 * @param [in] cellIdx cell index
		 * @param [in] parType particle type index
		 * @param [in] parGeomSurfToVol particle geometry
		 */
		Eigen::MatrixXd DGjacobianParDispBlock(unsigned int cellIdx, unsigned int parType, double parGeomSurfToVol) {

			MatrixXd dispBlock;
			// Inner dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
			dispBlock = MatrixXd::Zero(nParNode[parType], 3 * nParNode[parType] + 2);

			if (parExactInt[parType])
			{
				MatrixXd B = getParBMatrix(parType, cellIdx, parGeomSurfToVol); // "Lifting" matrix
				MatrixXd gBlock = getParGBlock(cellIdx, parType); // current cell auxiliary block matrix
				MatrixXd gStarDC = parAuxBlockGstar(cellIdx, parType, getParGBlock(cellIdx - 1, parType), gBlock, getParGBlock(cellIdx + 1, parType)); // Numerical flux block

				if (parGeomSurfToVol != SurfVolRatioSlab) // weak form DGSEM required
					dispBlock.block(0, nParNode[parType], nParNode[parType], nParNode[parType] + 2) = minus_InvMM_ST[offsetMetric[parType] + (cellIdx - 1)] * gBlock;
				else // strong form DGSEM
					dispBlock.block(0, nParNode[parType], nParNode[parType], nParNode[parType] + 2) = (parPolyDerM[parType] - parInvMM[offsetMetric[parType] + (cellIdx - 1)] * B) * gBlock;
				
				dispBlock += parInvMM[offsetMetric[parType] + (cellIdx - 1)] * B * gStarDC;
				dispBlock *= 2.0 / static_cast<double>(deltaR[offsetMetric[parType] + (cellIdx - 1)]);
			}
			else
			{
				// inexact integration is not maintained due to inferior performance. Code is in calcParticleCollocationDGSEMJacobian
			}

			return -dispBlock; // *-1 for residual
		}
	};

	enum class ParticleDiscretizationMode : int
	{
		/**
		 * Equidistant distribution of shell edges
		 */
		Equidistant,

		/**
		 * Volumes of shells are uniform
		 */
		Equivolume,

		/**
		 * Shell edges specified by user
		 */
		UserDefined
	};

	Discretization _disc; //!< Discretization info
	std::vector<bool> _hasSurfaceDiffusion; //!< Determines whether surface diffusion is present in each particle type
//	IExternalFunction* _extFun; //!< External function (owned by library user)

	parts::AxialConvectionDispersionOperatorBaseDG _convDispOp; //!< Convection dispersion operator base for interstitial volume transport
	IDynamicReactionModel* _dynReactionBulk; //!< Dynamic reactions in the bulk volume

	Eigen::SparseLU<Eigen::SparseMatrix<double>> _globalSolver; //!< linear solver
	//Eigen::BiCGSTAB<Eigen::SparseMatrix<double, RowMajor>, Eigen::DiagonalPreconditioner<double>> _globalSolver;

	Eigen::SparseMatrix<double, RowMajor> _globalJac; //!< static part of global Jacobian
	Eigen::SparseMatrix<double, RowMajor> _globalJacDisc; //!< global Jacobian with time derivative from BDF method
	//MatrixXd FDJac; // test purpose FD Jacobian

	Eigen::MatrixXd _jacInlet; //!< Jacobian inlet DOF block matrix connects inlet DOFs to first bulk cells

	active _colPorosity; //!< Column porosity (external porosity) \f$ \varepsilon_c \f$
	std::vector<active> _parRadius; //!< Particle radius \f$ r_p \f$
	bool _singleParRadius;
	std::vector<active> _parCoreRadius; //!< Particle core radius \f$ r_c \f$
	bool _singleParCoreRadius;
	std::vector<active> _parPorosity; //!< Particle porosity (internal porosity) \f$ \varepsilon_p \f$
	bool _singleParPorosity;
	std::vector<active> _parTypeVolFrac; //!< Volume fraction of each particle type
	std::vector<ParticleDiscretizationMode> _parDiscType; //!< Particle discretization mode
	std::vector<double> _parDiscVector; //!< Particle discretization shell edges
	std::vector<double> _parGeomSurfToVol; //!< Particle surface to volume ratio factor (i.e., 3.0 for spherical, 2.0 for cylindrical, 1.0 for hexahedral)

	// Vectorial parameters
	std::vector<active> _filmDiffusion; //!< Film diffusion coefficient \f$ k_f \f$
	MultiplexMode _filmDiffusionMode;
	std::vector<active> _parDiffusion; //!< Particle diffusion coefficient \f$ D_p \f$
	MultiplexMode _parDiffusionMode;
	std::vector<active> _parSurfDiffusion; //!< Particle surface diffusion coefficient \f$ D_s \f$
	MultiplexMode _parSurfDiffusionMode;
	std::vector<active> _poreAccessFactor; //!< Pore accessibility factor \f$ F_{\text{acc}} \f$
	MultiplexMode _poreAccessFactorMode;
	std::vector<IParameterStateDependence*> _parDepSurfDiffusion; //!< Parameter dependencies for particle surface diffusion
	bool _singleParDepSurfDiffusion; //!< Determines whether a single parameter dependence for particle surface diffusion is used
	bool _hasParDepSurfDiffusion; //!< Determines whether particle surface diffusion parameter dependencies are present

	bool _axiallyConstantParTypeVolFrac; //!< Determines whether particle type volume fraction is homogeneous across axial coordinate
	bool _analyticJac; //!< Determines whether AD or analytic Jacobians are used
	unsigned int _jacobianAdDirs; //!< Number of AD seed vectors required for Jacobian computation

	std::vector<active> _parCellSize; //!< Particle shell size
	std::vector<active> _parCenterRadius; //!< Particle node-centered position for each particle node
	std::vector<active> _parOuterSurfAreaPerVolume; //!< Particle shell outer sphere surface to volume ratio
	std::vector<active> _parInnerSurfAreaPerVolume; //!< Particle shell inner sphere surface to volume ratio

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
	BENCH_TIMER(_timerGmres)

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
		inline int strideParNode(int parType) const CADET_NOEXCEPT { return strideParLiquid() + strideParBound(parType); }
		inline int strideParShell(int parType) const CADET_NOEXCEPT { return strideParNode(parType) * _disc.nParNode[parType]; }
		inline int strideParBlock(int parType) const CADET_NOEXCEPT { return static_cast<int>(_disc.nParPoints[parType]) * strideParNode(parType); }

		// Offsets
		inline int offsetC() const CADET_NOEXCEPT { return _disc.nComp; }
		inline int offsetCp() const CADET_NOEXCEPT { return _disc.nComp * _disc.nPoints + offsetC(); }
		inline int offsetCp(ParticleTypeIndex pti) const CADET_NOEXCEPT { return offsetCp() + _disc.parTypeOffset[pti.value]; }
		inline int offsetCp(ParticleTypeIndex pti, ParticleIndex pi) const CADET_NOEXCEPT { return offsetCp() + _disc.parTypeOffset[pti.value] + strideParBlock(pti.value) * pi.value; }
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

		Exporter(const Discretization& disc, const GeneralRateModelDG& model, double const* data) : _disc(disc), _idx(disc), _model(model), _data(data) { }
		Exporter(const Discretization&& disc, const GeneralRateModelDG& model, double const* data) = delete;

		virtual bool hasParticleFlux() const CADET_NOEXCEPT { return false; }
		virtual bool hasParticleMobilePhase() const CADET_NOEXCEPT { return true; }
		virtual bool hasSolidPhase() const CADET_NOEXCEPT { return _disc.strideBound[_disc.nParType] > 0; }
		virtual bool hasVolume() const CADET_NOEXCEPT { return false; }
		virtual bool isParticleLumped() const CADET_NOEXCEPT { return false; }
		virtual bool hasPrimaryExtent() const CADET_NOEXCEPT { return true; }

		virtual unsigned int numComponents() const CADET_NOEXCEPT { return _disc.nComp; }
		virtual unsigned int numPrimaryCoordinates() const CADET_NOEXCEPT { return _disc.nPoints; }
		virtual unsigned int numSecondaryCoordinates() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numParticleTypes() const CADET_NOEXCEPT { return _disc.nParType; }
		virtual unsigned int numParticleShells(unsigned int parType) const CADET_NOEXCEPT { return _disc.nParPoints[parType]; }
		virtual unsigned int numBoundStates(unsigned int parType) const CADET_NOEXCEPT { return _disc.strideBound[parType]; }
		virtual unsigned int numMobilePhaseDofs() const CADET_NOEXCEPT { return _disc.nComp * _disc.nPoints; }
		virtual unsigned int numParticleMobilePhaseDofs() const CADET_NOEXCEPT
		{
			unsigned int nDofPerParType = 0;
			for (unsigned int i = 0; i < _disc.nParType; ++i)
				nDofPerParType += _disc.nParPoints[i];
			return _disc.nPoints * nDofPerParType * _disc.nComp;
		}
		virtual unsigned int numParticleMobilePhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return _disc.nPoints * _disc.nParPoints[parType] * _disc.nComp; }
		virtual unsigned int numSolidPhaseDofs() const CADET_NOEXCEPT
		{
			unsigned int nDofPerParType = 0;
			for (unsigned int i = 0; i < _disc.nParType; ++i)
				nDofPerParType += _disc.nParPoints[i] * _disc.strideBound[i];
			return _disc.nPoints * nDofPerParType;
		}
		virtual unsigned int numSolidPhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return _disc.nPoints * _disc.nParPoints[parType] * _disc.strideBound[parType]; }
		virtual unsigned int numParticleFluxDofs() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numVolumeDofs() const CADET_NOEXCEPT { return 0; }

		virtual int writeMobilePhase(double* buffer) const;
		virtual int writeSolidPhase(double* buffer) const;
		virtual int writeParticleMobilePhase(double* buffer) const;
		virtual int writeSolidPhase(unsigned int parType, double* buffer) const;
		virtual int writeParticleMobilePhase(unsigned int parType, double* buffer) const;
		virtual int writeParticleFlux(double* buffer) const;
		virtual int writeParticleFlux(unsigned int parType, double* buffer) const;
		virtual int writeVolume(double* buffer) const { return 0; }
		virtual int writeInlet(unsigned int port, double* buffer) const;
		virtual int writeInlet(double* buffer) const;
		virtual int writeOutlet(unsigned int port, double* buffer) const;
		virtual int writeOutlet(double* buffer) const;
		/**
		* @brief calculates, writes the physical axial/column coordinates of the DG discretization with double! interface nodes
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
		/**
		* @brief calculates, writes the physical radial/particle coordinates of the DG discretization with double! interface nodes
		*/
		virtual int writeParticleCoordinates(unsigned int parType, double* coords) const
		{
			active const* const pcr = _model._parCenterRadius.data() + _disc.offsetMetric[parType];

			// Note that the DG particle shells are oppositely ordered compared to the FV particle shells
			for (unsigned int par = 0; par < _disc.nParPoints[parType]; par++) {

				unsigned int cell = std::floor(par / _disc.nParNode[parType]);

				double r_L = static_cast<double>(pcr[cell]) - 0.5 * static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType] + cell]);
				coords[par] = r_L + 0.5 * static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType] + cell]) * (1.0 + _disc.parNodes[parType][par % _disc.nParNode[parType]]);
			}

			return _disc.nParPoints[parType];
		}

	protected:
		const Discretization& _disc;
		const Indexer _idx;
		const GeneralRateModelDG& _model;
		double const* const _data;
	};

	/**
	* @brief sets the current section index and loads section dependend velocity, dispersion
	*/
	void updateSection(int secIdx) {

		if (cadet_unlikely(_disc.curSection != secIdx)) {
			_disc.curSection = secIdx;
			_disc.newStaticJac = true;
		}
	}

// ===========================================================================================================================================================  //
// ========================================			DG functions to compute particle discretization			==================================================  //
// ===========================================================================================================================================================  //

	/**
	 * @brief promotes doubles to actives
	 * @detail promotes consecutive doubles to consecutive actives (with zero gradients) based on input double pointer
	 */
	void vectorPromoter(double* state, const unsigned int nVals) {

		const int nDirs = ad::getDirections();
		const int stride = (1 + nDirs);
		const int ADsize = stride * nVals;
		double buff = 0.0;

		for (int val = 1; val <= nVals; val++) // start with last entry to avoid overwriting
		{
			buff = state[nVals - val];
			std::fill(state + ADsize - val * stride, state + ADsize - (val - 1) * stride, 0.0);
			state[ADsize - val * stride] = buff;
		}
	}

	template<typename ResidualType, typename ParamType>
	void applyParInvMap(Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<>>& state, unsigned int parType) {
		for (int cell = 0; cell < _disc.nParCell[parType]; cell++) {
			state.segment(cell * _disc.nParNode[parType], _disc.nParNode[parType]) *= 2.0 / static_cast<ParamType>(_disc.deltaR[_disc.offsetMetric[parType] + cell]) * 2.0 / static_cast<ParamType>(_disc.deltaR[_disc.offsetMetric[parType] + cell]);
		}
	}

	template<typename StateType, typename ResidualType>
	void parGSMVolumeIntegral(const int parType, Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>>& state, Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>>& stateDer) {

		int nNodes = _disc.nParNode[parType];

		stateDer.segment(0, nNodes)
			-= (_disc.minus_parInvMM_Ar[parType].template cast<StateType>() * state.segment(0, nNodes)).template cast<ResidualType>();
	}

	template<typename StateType, typename ResidualType>
	void parVolumeIntegral(const int parType, const bool aux, Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>>& state, Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>>& stateDer) {

		int nNodes = _disc.nParNode[parType];

		/* no additional metric term for auxiliary equation or particle equation with exact integration scheme
		   -> res = - D * (d_p * c^p + invBeta_p sum_mi d_s c^s) */
		if (aux || (_disc.parExactInt[parType] && _parGeomSurfToVol[parType] == _disc.SurfVolRatioSlab)) {
			// comp-cell-node state vector: use of Eigen lib performance
			for (unsigned int Cell = 0; Cell < _disc.nParCell[parType]; Cell++) {
				stateDer.segment(Cell * nNodes, nNodes)
					-= (_disc.parPolyDerM[parType].template cast<StateType>() * state.segment(Cell * nNodes, nNodes)).template cast<ResidualType>();
			}
		}
		else if (_disc.parExactInt[parType] && _parGeomSurfToVol[parType] != _disc.SurfVolRatioSlab) {
			// comp-cell-node state vector: use of Eigen lib performance
			for (unsigned int Cell = 0; Cell < _disc.nParCell[parType]; Cell++) {
				stateDer.segment(Cell * nNodes, nNodes)
					-= (_disc.minus_InvMM_ST[_disc.offsetMetric[parType] + Cell].template cast<StateType>() * state.segment(Cell * nNodes, nNodes)).template cast<ResidualType>();
			}
		}
		/* include metrics for main particle equation -> res = - D * (d_p * c^p + invBeta_p sum_mi d_s c^s) */
		else { // inexact integration, main equation

			int Cell0 = 0; // auxiliary variable to distinguish special case

			// special case for non slab-shaped particles without core => r(xi_0) = 0
			if (_parGeomSurfToVol[parType] != _disc.SurfVolRatioSlab && _parCoreRadius[parType] == 0.0) {
				Cell0 = 1;

				// compute volume integral except for boundary node
				stateDer.segment(1, nNodes - 1) -= (_disc.Dr[_disc.offsetMetric[parType]].block(1, 1, nNodes - 1, nNodes - 1).template cast<StateType>() * state.segment(1, nNodes - 1)).template cast<ResidualType>();
				// estimate volume integral for boundary node: sum_{j=1}^N state_j * w_j * D_{j,0} * r_j
				stateDer[0] += static_cast<ResidualType>(
					(state.segment(1, nNodes - 1).array()
					* _disc.parInvWeights[parType].segment(1, nNodes - 1).array().cwiseInverse().template cast<StateType>()
					* _disc.parPolyDerM[parType].block(1, 0, nNodes - 1, 1).array().template cast<StateType>()
					* _disc.Ir[_disc.offsetMetric[parType]].segment(1, nNodes - 1).array().template cast<StateType>()
					).sum()
					);
			}

			// "standard" computation for remaining cells
			for (int cell = Cell0; cell < _disc.nParCell[parType]; cell++) {
				stateDer.segment(cell * nNodes, nNodes) -= (_disc.Dr[_disc.offsetMetric[parType] + cell].template cast<StateType>() * state.segment(cell * nNodes, nNodes)).template cast<ResidualType>();
			}
		}
	}
	/*
	 * @brief calculates the interface fluxes g* of particle mass balance equation and implements the respective boundary conditions
	 * @param [in] aux bool if interface flux for auxiliary equation
	 * @param [in] addParDisc bool if interface flux for additional particle DG-discretized equation
	*/
	template<typename StateType>
	void InterfaceFluxParticle(int parType, Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>>& state,
		const unsigned int strideCell, const unsigned int strideNode, const bool aux, const int comp, const bool addParDisc = false) {

		Eigen::Map<Vector<StateType, Dynamic>, 0, InnerStride<>> _surfFluxPar(reinterpret_cast<StateType*>(&_disc.surfaceFluxParticle[parType][0]), _disc.nParCell[parType] + 1, InnerStride<>(1));

		// reset surface flux storage as it is used multiple times
		_surfFluxPar.setZero();

		// numerical flux: state* = 0.5 (state^+ + state^-)

		// calculate inner interface fluxes
		for (unsigned int Cell = 1u; Cell < _disc.nParCell[parType]; Cell++) {
			_surfFluxPar[Cell] // left interfaces
				= 0.5 * (state[Cell * strideCell - strideNode] + // outer/left node
					state[Cell * strideCell]); // inner/right node
		}

		// calculate boundary interface fluxes.
		if (aux) { // ghost nodes given by state^- := state^+ for auxiliary equation
			_surfFluxPar[0] = state[0];

			_surfFluxPar[_disc.nParCell[parType]] = state[_disc.nParCell[parType] * strideCell - strideNode];
		}
		else if (addParDisc) {
			_surfFluxPar[0] = 0.0;

			_surfFluxPar[_disc.nParCell[parType]] = 0.0;
		}
		else {
			
			// film diffusion BC
			_surfFluxPar[_disc.nParCell[parType]] = static_cast<StateType>(_disc.localFlux[comp])
					/ (static_cast<double>(_parPorosity[parType]) * static_cast<double>(_poreAccessFactor[parType * _disc.nComp + comp]))
				* (2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType]])); // inverse squared mapping was also applied, so we apply Map * invMap^2 = invMap

			// inner particle BC
			_surfFluxPar[0] = 0.0;

		}
	}
	/**
	 * @brief calculates the particle surface Integral (type- and component-wise)
	 * @param [in] parType current particle type
	 * @param [in] state relevant state vector
	 * @param [in] stateDer state derivative vector the solution is added to
	 * @param [in] aux true for auxiliary equation, false for main equation
	 * @param [in] strideCell component-wise cell stride
	 * @param [in] strideNodecomponent-wise node stride
	 * @param [in] comp current component
	*/
	template<typename StateType, typename ResidualType>
	void parSurfaceIntegral(int parType, Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<Dynamic>>& state,
		Eigen::Map<Vector<ResidualType, Dynamic>, 0, InnerStride<Dynamic>>& stateDer, unsigned const int strideCell, unsigned const int strideNode,
		const bool aux, const int comp = 0, const bool addParDisc = false) {

		Eigen::Map<Vector<StateType, Dynamic>, 0, InnerStride<>> _surfFluxPar(reinterpret_cast<StateType*>(&_disc.surfaceFluxParticle[parType][0]), _disc.nParCell[parType] + 1, InnerStride<>(1));

		// calc numerical flux values
		InterfaceFluxParticle<StateType>(parType, state, strideCell, strideNode, aux, comp, addParDisc);

		// strong surface integral -> M^-1 B [state - state*]
		if (!_disc.parExactInt[parType]) { // inexact integration approach -> diagonal mass matrix
			int Cell0 = 0; // auxiliary variable to distinguish special case
			// special case for sphere and cylinder if particle core = 0.0 -> leave out inner particle boundary flux
			if (_parGeomSurfToVol[parType] != _disc.SurfVolRatioSlab && _parCoreRadius[parType] == 0.0) {

				Cell0 = 1;

				stateDer[_disc.parPolyDeg[parType] * strideNode] // last cell node
					+= _disc.parInvWeights[parType][_disc.parPolyDeg[parType]] * (state[_disc.parPolyDeg[parType] * strideNode] - _surfFluxPar[1]);
			}

			for (unsigned int Cell = Cell0; Cell < _disc.nParCell[parType]; Cell++) {

				stateDer[Cell * strideCell] // first cell node
					-= _disc.parInvWeights[parType][0] * (state[Cell * strideCell] - _surfFluxPar[Cell]);

				stateDer[Cell * strideCell + _disc.parPolyDeg[parType] * strideNode] // last cell node
					+= _disc.parInvWeights[parType][_disc.parPolyDeg[parType]] * (state[Cell * strideCell + _disc.parPolyDeg[parType] * strideNode] - _surfFluxPar[Cell + 1u]);
			}
		}
		else { // exact integration approach -> dense mass matrix
			for (unsigned int Cell = 0; Cell < _disc.nParCell[parType]; Cell++) {

				for (unsigned int Node = 0; Node < _disc.nParNode[parType]; Node++) {
					if (aux) { // strong surface integral -> M^-1 B [state - state*] and B has two non-zero entries, -1 and 1
						stateDer[Cell * strideCell + Node * strideNode]
							-= _disc.parInvMM_Leg[parType](Node, 0) * (state[Cell * strideCell]- _surfFluxPar[Cell])
							- _disc.parInvMM_Leg[parType](Node, _disc.parPolyDeg[parType]) * (state[Cell * strideCell + _disc.parPolyDeg[parType] * strideNode] - _surfFluxPar[Cell + 1u]);
					}
					else {
						if (_parGeomSurfToVol[parType] == _disc.SurfVolRatioSlab) { // strong surface integral -> M^-1 B [state - state*] and B has two non-zero entries, -1 and 1
							stateDer[Cell * strideCell + Node * strideNode]
								-= static_cast<ResidualType>(
									_disc.parInvMM[parType](Node, 0) * (state[Cell * strideCell] - _surfFluxPar[Cell])
									- _disc.parInvMM[parType](Node, _disc.parPolyDeg[parType]) * (state[Cell * strideCell + _disc.parPolyDeg[parType] * strideNode] - _surfFluxPar[Cell + 1u])
									);
						}
						else if (_parGeomSurfToVol[parType] == _disc.SurfVolRatioCylinder) { // weak surface integral -> M^-1 B [- state*] and B has two non-zero entries, which depend on metrics
							stateDer[Cell * strideCell + Node * strideNode]
								-= static_cast<ResidualType>(
									_disc.Ir[_disc.offsetMetric[parType] + Cell][0] * _disc.parInvMM[_disc.offsetMetric[parType] + Cell](Node, 0) * (-_surfFluxPar[Cell])
									+ _disc.Ir[_disc.offsetMetric[parType] + Cell][_disc.nParNode[parType] - 1] * _disc.parInvMM[_disc.offsetMetric[parType] + Cell](Node, _disc.parPolyDeg[parType]) * _surfFluxPar[Cell + 1u]
									);
						}
						else if (_parGeomSurfToVol[parType] == _disc.SurfVolRatioSphere) { // weak surface integral -> M^-1 B [- state*] and B has two non-zero entries, which depend on metrics
							stateDer[Cell * strideCell + Node * strideNode]
								-= static_cast<ResidualType>(
									_disc.Ir[_disc.offsetMetric[parType] + Cell][0] * _disc.parInvMM[_disc.offsetMetric[parType] + Cell](Node, 0) * (-_surfFluxPar[Cell])
									+ _disc.Ir[_disc.offsetMetric[parType] + Cell][_disc.nParNode[parType] - 1] * _disc.parInvMM[_disc.offsetMetric[parType] + Cell](Node, _disc.parPolyDeg[parType]) * _surfFluxPar[Cell + 1u]
									);
						}
					}
				}
			}
		}
	}
	/**
	 * @brief solves the auxiliary system g = d c / d xi
	 * @detail computes g = Dc - M^-1 B [c - c^*] and stores this in _disc.g_p
	*/
	template<typename StateType>
	void solve_auxiliary_DG(int parType, Eigen::Map<const Vector<StateType, Dynamic>, 0, InnerStride<>>& conc, unsigned int strideCell, unsigned int strideNode, int comp) {

		Eigen::Map<Vector<StateType, Dynamic>, 0, InnerStride<>> g_p(reinterpret_cast<StateType*>(&_disc.g_p[parType][0]), _disc.nParPoints[parType], InnerStride<>(1));
		Eigen::Map<Vector<StateType, Dynamic>, 0, InnerStride<>> _surfFluxPar(reinterpret_cast<StateType*>(&_disc.surfaceFluxParticle[parType][0]), _disc.nParCell[parType] + 1, InnerStride<>(1));
		_surfFluxPar.setZero(); // reset surface flux storage as it is used multiple times
		g_p.setZero(); // reset auxiliary variable g

		// ========================================================================================//
		// solve auxiliary systems g = d c / d xi	 =>		g_p = Dc - M^-1 B [c - c^*]			   //
		// ========================================================================================//
		
		parVolumeIntegral<StateType, StateType>(parType, true, conc, g_p); // volumne integral in strong DG form: - D c
		
		parSurfaceIntegral<StateType>(parType, conc, g_p, strideCell, strideNode, true, comp); // surface integral in strong DG form: M^-1 B [c - c^*]
		
		g_p *= -1.0; // auxiliary factor -1
	}

	// ==========================================================================================================================================================  //
	// ========================================						DG particle Jacobian							=============================================  //
	// ==========================================================================================================================================================  //

	/**
	* @brief computes the jacobian via finite differences (testing purpose)
	*/
	MatrixXd calcFDJacobian(const double* y_, const double* yDot_, const SimulationTime simTime, util::ThreadLocalStorage& threadLocalMem, double alpha) {

		// create solution vectors
		Eigen::Map<const VectorXd> hmpf(y_, numDofs());
		VectorXd y = hmpf;
		VectorXd yDot;
		if (yDot_) {
			Eigen::Map<const VectorXd> hmpf2(yDot_, numDofs());
			yDot = hmpf2;
		}
		else {
			return MatrixXd::Zero(numDofs(), numDofs());
		}
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

		///*	exterminate numerical noise	 */
		//for (int i = 0; i < Jacobian.rows(); i++) {
		//	for (int j = 0; j < Jacobian.cols(); j++) {
		//		if (std::abs(Jacobian(i, j)) < 1e-10) Jacobian(i, j) = 0.0;
		//	}
		//}
		Jacobian /= epsilon;

		return Jacobian;
	}

	typedef Eigen::Triplet<double> T;
	/**
	 * @brief calculates the particle dispersion jacobian Pattern of the exact/inexact integration DG scheme for the given particle type and bead
	*/
	void calcParticleJacobianPattern(std::vector<T>& tripletList, unsigned int parType, unsigned int colNode, unsigned int secIdx) {

		// Ordering of particle surface diffusion:
		// bnd0comp0, bnd0comp1, bnd0comp2, bnd1comp0, bnd1comp1, bnd1comp2
		active const* const _parSurfDiff = getSectionDependentSlice(_parSurfDiffusion, _disc.strideBound[_disc.nParType], secIdx) + _disc.nBoundBeforeType[parType];

		Indexer idxr(_disc);

		// (global) strides
		unsigned int sCell = _disc.nParNode[parType] * idxr.strideParNode(parType);
		unsigned int sNode = idxr.strideParNode(parType);
		unsigned int sComp = 1u;
		unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });
		unsigned int nNodes = _disc.nParNode[parType];

		// case: one cell  -> diffBlock \in R^(nParNodes x nParNodes), GBlock = parPolyDerM
		if (_disc.nParCell[parType] == 1) {

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 0; j < nNodes; j++) {
						// handle liquid state
						// row: add component offset and go node strides from there for each dispersion block entry
						// col: add component offset and go node strides from there for each dispersion block entry
						tripletList.push_back(T(offset + comp * sComp + i * sNode,
												offset + comp * sComp + j * sNode, 0.0));

						// handle surface diffusion of bound states.
						if (_hasSurfaceDiffusion[parType]) {

							int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								if (_parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) {
									// row: add current component offset and go node strides from there for each dispersion block entry
									// col: jump oover liquid states, add current bound state offset and go node strides from there for each dispersion block entry
									tripletList.push_back(T(offset + comp * sComp + i * sNode,
															offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode, 0.0));

									/* add surface diffusion dispersion block to solid */
									if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
										// row: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
										// col: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
										tripletList.push_back(T(offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode, 0.0));

									}
								}
							}
						}
					}
				}
			}
		}
		else {

			if (!_disc.parExactInt[parType]) {

				/*			 left boundary cell				*/

				// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					for (unsigned int i = 0; i < nNodes; i++) {
						for (unsigned int j = nNodes; j < 3 * nNodes; j++) {
							// pattern is more sparse than a nNodes x 2*nNodes block.
							if ((j >= nNodes - 1 && j <= 2 * nNodes) ||
								(i == 0 && j <= 2 * nNodes) ||
								(i == nNodes - 1 && j >= nNodes - 1)) {
								// handle liquid state
								// row: add component offset and go node strides from there for each dispersion block entry
								// col: add component offset and go node strides from there for each dispersion block entry
								tripletList.push_back(T(offset + comp * sComp + i * sNode,
														offset + comp * sComp + (j - nNodes) * sNode,
														0.0));

								// handle surface diffusion of bound states. binding is handled in residualKernel().
								if (_hasSurfaceDiffusion[parType]) {

									int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

									for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
										if (_parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) {
											// row: add current component offset and go node strides from there for each dispersion block entry
											// col: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
											tripletList.push_back(T(offset + comp * sComp + i * sNode,
																	offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (j - nNodes) * sNode,
																	0.0));

											/* add surface diffusion dispersion block to solid */
											if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
												// row: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
												// col: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
												tripletList.push_back(T(offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																		offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (j - nNodes) * sNode,
																		0.0));

											}
										}
									}
								}
							}
						}
					}
				}

				/*			 right boundary cell				*/

				// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					for (unsigned int i = 0; i < nNodes; i++) {
						for (unsigned int j = 0; j < 2 * nNodes; j++) {
							// pattern is more sparse than a nNodes x 2*nNodes block.
							if ((j >= nNodes - 1 && j <= 2 * nNodes) ||
								(i == 0 && j <= 2 * nNodes) ||
								(i == nNodes - 1 && j >= nNodes - 1)) {
								// handle liquid state
								// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
								// col: add component offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
								tripletList.push_back(T(offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
														offset + comp * sComp + (_disc.nParCell[parType] - 2) * sCell + j * sNode,
														0.0));
								// handle surface diffusion of bound states. binding is handled in residualKernel().
								if (_hasSurfaceDiffusion[parType]) {

									int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

									for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
										if (_parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) {
											// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
											// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
											tripletList.push_back(T(offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
																	offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (_disc.nParCell[parType] - 2) * sCell + j * sNode,
																	0.0));

											/* add surface diffusion dispersion block to solid */
											if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
												// row: jump over previous cells and liquid states, add current bound state offset and go node strides from there for each dispersion block entry
												// col: jump over previous cells and liquid states, go back one cell, add current bound state offset and go node strides from there for each dispersion block entry
												tripletList.push_back(T(offset + (_disc.nParCell[parType] - 1) * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																		offset + (_disc.nParCell[parType] - 2) * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode,
																		0.0));

											}
										}
									}
								}
							}
						}
					}
				}

				/*				inner cells				*/

				for (int cell = 1; cell < _disc.nParCell[parType] - 1; cell++) {

					// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
					for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = 0; j < 3 * nNodes; j++) {
								// pattern is more sparse than a nNodes x 3*nNodes block.
								if ((j >= nNodes - 1 && j <= 2 * nNodes) ||
									(i == 0 && j <= 2 * nNodes) ||
									(i == nNodes - 1 && j >= nNodes - 1)) {
									// handle liquid state
									// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
									// col: add component offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
									tripletList.push_back(T(offset + comp * sComp + cell * sCell + i * sNode,
															offset + comp * sComp + (cell - 1) * sCell + j * sNode, 0.0));
									// handle surface diffusion of bound states. binding is handled in residualKernel().
									if (_hasSurfaceDiffusion[parType]) {

										int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

										for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
											if (_parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) {
												// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
												// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
												tripletList.push_back(T(offset + comp * sComp + cell * sCell + i * sNode,
																		offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (cell - 1) * sCell + j * sNode,
																		0.0));

												/* add surface diffusion dispersion block to solid */
												if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
													// row: jump over previous cells and liquid states, add current bound state offset and go node strides from there for each dispersion block entry
													// col: jump over previous cells and liquid states, go back one cell, add current bound state offset and go node strides from there for each dispersion block entry
													tripletList.push_back(T(offset + cell * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																			offset + (cell - 1) * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode,
																			0.0));

												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
			else { //exact integration

			/*			boundary cells			*/

			/*			 left boundary cell				*/

				unsigned int special = 0u; if (_disc.nParCell[parType] < 3u) special = 1u; // limits the iterator for special case nCells = 3 (dependence on additional entry)
				// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					for (unsigned int i = 0; i < nNodes; i++) {
						for (unsigned int j = nNodes + 1; j < 3 * nNodes + 2 - special; j++) {
							// handle liquid state
							// row: add component offset and go node strides from there for each dispersion block entry
							// col: add component offset and go node strides from there for each dispersion block entry. adjust for j start
							tripletList.push_back(T(offset + comp * sComp + i * sNode,
													offset + comp * sComp + j * sNode - (nNodes + 1) * sNode,
													0.0));

							// handle surface diffusion of bound states. binding is handled in residualKernel().
							if (_hasSurfaceDiffusion[parType]) {

								int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

								for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
									if (_parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) {
										// row: add current component offset and go node strides from there for each dispersion block entry
										// col: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry. adjust for j start
										tripletList.push_back(T(offset + comp * sComp + i * sNode,
																offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode - (nNodes + 1) * sNode,
																0.0));

										/* add surface diffusion dispersion block to solid */
										if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
											// row: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
											// col: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry. adjust for j start
											tripletList.push_back(T(offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																	offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode - (nNodes + 1) * sNode,
																	0.0));

										}
									}
								}
							}
						}
					}
				}

				/*			 right boundary cell				*/

				// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					for (unsigned int i = 0; i < nNodes; i++) {

						for (unsigned int j = special; j < 2 * nNodes + 1; j++) {
							// handle liquid state
							// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
							// col: add component offset and jump over previous cells. Go back one cell (and node or adjust for start) and go node strides from there for each dispersion block entry.
							tripletList.push_back(T(offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
													offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell - sCell - sNode + j * sNode,
													0.0));

							// handle surface diffusion of bound states. binding is handled in residualKernel().
							if (_hasSurfaceDiffusion[parType]) {

								int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

								for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
									if (_parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) {
										// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
										// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell (and node or adjust for start) and go node strides from there for each dispersion block entry.
										tripletList.push_back(T(offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
																offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (_disc.nParCell[parType] - 2) * sCell - sNode + j * sNode,
																0.0));

										/* add surface diffusion dispersion block to solid */
										if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
											// row: jump over previous cells and over liquid states, add current bound state offset. go node strides from there for each dispersion block entry
											// col: jump over previous cells and over liquid states, add current bound state offset. go back one cell (and node or adjust for start) and go node strides from there for each dispersion block entry.
											tripletList.push_back(T(offset + (_disc.nParCell[parType] - 1) * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																	offset + (_disc.nParCell[parType] - 2) * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) - sNode + bnd + j * sNode,
																	0.0));
										}
									}
								}
							}
						}
					}
				}
				if (_disc.nParCell[parType] == 3) {
					for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = 1; j < 3 * nNodes + 2 - 1; j++) {
								// handle liquid state
								// row: add component offset and jump over previous cell. Go node strides from there for each dispersion block entry
								// col: add component offset. Go node strides from there for each dispersion block entry. adjust for j start
								tripletList.push_back(T(offset + comp * sComp + sCell + i * sNode,
														offset + comp * sComp + j * sNode - sNode,
														0.0));

								// handle surface diffusion of bound states. binding is handled in residualKernel().
								if (_hasSurfaceDiffusion[parType]) {

									int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

									for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
										if (_parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) {
											// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
											// col: jump over liquid states, add current bound state offset and jump over previous cell. go back one cell and go node strides from there for each dispersion block entry. adjust for j start
											tripletList.push_back(T(offset + comp * sComp + sCell + i * sNode,
																	offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode - sNode,
																	0.0));

											/* add surface diffusion dispersion block to solid */
											if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
												// row: jump over previous cells and over liquid states, add current bound state offset. go node strides from there for each dispersion block entry
												// col: jump over liquid states, add current bound state offset and jump over previous cell. go node strides from there for each dispersion block entry. adjust for j start
												tripletList.push_back(T(offset + sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																		offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode - sNode,
																		0.0));

											}
										}
									}
								}

							}
						}
					}
				}// special case nCells == 3
				/*	boundary cell neighbours (exist only if nCells >= 4)	*/
				if (_disc.nParCell[parType] >= 4) {

					for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = 1; j < 3 * nNodes + 2; j++) {
								// handle liquid state
								// row: add component offset and jump over previous cell. Go node strides from there for each dispersion block entry
								// col: add component offset. Go node strides from there for each dispersion block entry. adjust for j start
								tripletList.push_back(T(offset + comp * sComp + sCell + i * sNode,
														offset + comp * sComp + j * sNode - sNode,
														0.0));

								// handle surface diffusion of bound states. binding is handled in residualKernel().
								if (_hasSurfaceDiffusion[parType]) {

									int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

									for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
										if (_parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) {
											// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
											// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry. adjust for j start
											tripletList.push_back(T(offset + comp * sComp + sCell + i * sNode,
																	offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode - sNode,
																	0.0));

											/* add surface diffusion dispersion block to solid */
											if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
												// row: jump over previous cells and over liquid states, add current bound state offset. go node strides from there for each dispersion block entry
												// col: jump over previous cells and over liquid states, add current bound state offset. go back one cell and go node strides from there for each dispersion block entry. adjust for j start
												tripletList.push_back(T(offset + sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																		offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode - sNode,
																		0.0));

											}
										}
									}
								}

							}
						}
					}

					for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
						for (unsigned int i = 0; i < nNodes; i++) {
							for (unsigned int j = 0; j < 3 * nNodes + 2 - 1; j++) {
								// handle liquid state
								// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
								// col: add component offset and jump over previous cells. Go back one cell and node. Go node strides from there for each dispersion block entry.
								tripletList.push_back(T(offset + comp * sComp + (_disc.nParCell[parType] - 2) * sCell + i * sNode,
														offset + comp * sComp + (_disc.nParCell[parType] - 2) * sCell - sCell - sNode + j * sNode,
														0.0));

								// handle surface diffusion of bound states. binding is handled in residualKernel().
								if (_hasSurfaceDiffusion[parType]) {

									int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

									for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
										if (_parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) {
											// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
											// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and node and go node strides from there for each dispersion block entry
											tripletList.push_back(T(offset + comp * sComp + (_disc.nParCell[parType] - 2) * sCell + i * sNode,
																	offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (_disc.nParCell[parType] - 2) * sCell - sCell - sNode + j * sNode,
																	0.0));

											/* add surface diffusion dispersion block to solid */
											if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
												// row: jump over previous cells and over liquid states, add current bound state offset. go node strides from there for each dispersion block entry
												// col: jump over previous cells and over liquid states, add current bound state offset. go back one cell and node and go node strides from there for each dispersion block entry
												tripletList.push_back(T(offset + (_disc.nParCell[parType] - 2) * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																		offset + (_disc.nParCell[parType] - 2) * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) - sCell - sNode + bnd + j * sNode,
																		0.0));

											}
										}
									}
								}
							}
						}
					}
				}

				/* Inner cells (exist only if nCells >= 5) */

				if (_disc.nParCell[parType] >= 5) {

					for (unsigned int cell = 2; cell < _disc.nParCell[parType] - 2; cell++) {

						for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
							for (unsigned int i = 0; i < nNodes; i++) {
								for (unsigned int j = 0; j < 3 * nNodes + 2; j++) {
									// handle liquid state
									// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
									// col: add component offset and jump over previous cells. Go back one cell and node. Go node strides from there for each dispersion block entry.
									tripletList.push_back(T(offset + comp * sComp + cell * sCell + i * sNode,
															offset + comp * sComp + cell * sCell - sCell - sNode + j * sNode,
															0.0));

									// handle surface diffusion of bound states. binding is handled in residualKernel().
									if (_hasSurfaceDiffusion[parType]) {

										int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

										for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
											if (_parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)] != 0.0) {
												// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
												// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and node and go node strides from there for each dispersion block entry
												tripletList.push_back(T(offset + comp * sComp + cell * sCell + i * sNode,
																		offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + cell * sCell - sCell - sNode + j * sNode,
																		0.0));

												/* add surface diffusion dispersion block to solid */
												if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
													// row: jump over previous cells and over liquid states, add current bound state offset. go node strides from there for each dispersion block entry
													// col: jump over previous cells and over liquid states, add current bound state offset. go back one cell and node and go node strides from there for each dispersion block entry
													tripletList.push_back(T(offset + cell * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																			offset + cell * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd - sCell - sNode + j * sNode,
																			0.0));

												}
											}
										}
									}
								}
							}
						}
					}

				}

			} // parExactInt
		} // if nCells > 1
	}
	unsigned int calcParDispNNZ(int parType) {

		if (_disc.parExactInt[parType]) {
			return _disc.nComp * ((3u * _disc.nParCell[parType] - 2u) * _disc.nParNode[parType] * _disc.nParNode[parType] + (2u * _disc.nParCell[parType] - 3u) * _disc.nParNode[parType]);
		}
		else {
			return _disc.nComp * (_disc.nParCell[parType] * _disc.nParNode[parType] * _disc.nParNode[parType] + 8u * _disc.nParNode[parType]);
		}
	}
	/**
	 * @brief sets the sparsity pattern of the binding Jacobian
	 */
	void parBindingPattern_GRM(std::vector<T>& tripletList, unsigned int parType, unsigned int colNode) {

		Indexer idxr(_disc);

		int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });

		// every bound state might depend on every bound and liquid state
		for (int parNode = 0; parNode < _disc.nParPoints[parType]; parNode++) {
			for (int bnd = 0; bnd < _disc.strideBound[parType]; bnd++) {
				for (int conc = 0; conc < idxr.strideParNode(parType); conc++) {
					// row: jump over previous nodes and liquid states and add current bound state offset
					// col: jump over previous nodes and add current concentration offset (liquid and bound)
					tripletList.push_back(T(offset + parNode * idxr.strideParNode(parType) + idxr.strideParLiquid() + bnd,
						offset + parNode * idxr.strideParNode(parType) + conc, 0.0));
				}
			}
		}
	}
	/**
	 *@brief adds the time derivative entries from particle equations
	 *@detail since the main diagonal entries are already set, we actually only set the solid phase time derivative entries for the discretized particle mass balance equations
	 */
	void parTimeDerJacPattern_GRM(std::vector<T>& tripletList, unsigned int parType, unsigned int colNode, unsigned int secIdx) {

		Indexer idxr(_disc);

		active const* const _parSurfDiff = getSectionDependentSlice(_parSurfDiffusion, _disc.strideBound[_disc.nParType], secIdx) + _disc.nBoundBeforeType[parType];
		unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });

		for (unsigned int parNode = 0; parNode < _disc.nParPoints[parType]; parNode++) {

			// discretization special case: we get an algebraic equation at inner particle boundary
			 if (!_disc.parExactInt[parType]  && parNode == 0u && _parGeomSurfToVol[parType] != _disc.SurfVolRatioSlab && _parCoreRadius[parType] == 0.0)
				continue;

			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {

				for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
					// row: jump over previous nodes add current component offset
					// col: jump over previous nodes, liquid phase and previous bound states
					tripletList.push_back(T(offset + parNode * idxr.strideParNode(parType) + comp,
											offset + parNode * idxr.strideParNode(parType) + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd,
											0.0));
				}
			}
		}
	}
	/**
	 * @brief sets the sparsity pattern of the global Jacobian
	 */
	void setParJacPattern(std::vector<T>& tripletList, unsigned int parType, unsigned int colNode, unsigned int secIdx) {

		calcParticleJacobianPattern(tripletList, parType, colNode, secIdx);

		parTimeDerJacPattern_GRM(tripletList, parType, colNode, secIdx);

		parBindingPattern_GRM(tripletList, parType, colNode);
	}
	/**
	 * @brief returns the offset between state order and parameter storage (bnd0comp0, bnd0comp1, bnd0comp2, bnd1comp0, bnd1comp1, bnd1comp2, ...) for one components bound state for a certain particle type
	 * @todo code review, is there a different more elegant/easy way?
	 */
	unsigned int getOffsetSurfDiff(unsigned int parType, unsigned int comp, unsigned int bnd) {
		
		unsigned int offNextBound = 0;
		
		// we need to estimate the offset to the next parameter of current components next bound state
		// Ordering of particle surface diffusion: bnd0comp0, bnd0comp1, bnd0comp2, bnd1comp0, bnd1comp1, bnd1comp2
			for (unsigned int _comp = 0; _comp < _disc.nComp; _comp++) {
				if(_comp < comp) // if its a component that occurs before comp, add all bound states of that component up to bnd + 1 (note that bound index starts at 0 -> +1).
					offNextBound += std::min(bnd + 1u, _disc.nBound[parType * _disc.nComp + _comp]);
				else // Otherwise, only add all previous (i.e. up to bnd) bound states of that component. This includes the current component itself (comp == _comp).
					offNextBound += std::min(bnd, _disc.nBound[parType * _disc.nComp + _comp]);
			}

		return offNextBound;
	}
	/**
	 * @brief calculate offsets between surface diffusion parameter storage and state ordering
	 */
	void orderSurfDiff() {

		Indexer idxr(_disc);

		for (unsigned int type = 0; type < _disc.nParType; type++) {
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int bnd = 0; bnd < _disc.nBound[type * _disc.nComp + comp]; bnd++) {
					_disc.offsetSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{ type }, ComponentIndex{ comp }) + bnd] = getOffsetSurfDiff(type, comp, bnd);
				}
			}
		}
	}
	/**
	 * @brief analytically calculates the particle dispersion jacobian of the DGSEM (exact integration) for a single particle type and bead
	 */
	int calcParticleDGSEMJacobian(unsigned int parType, unsigned int colNode, const active* const parDiff, const active* const parSurfDiff, const double* const invBetaP) {

		Indexer idxr(_disc);

		// (global) strides
		unsigned int sCell = _disc.nParNode[parType] * idxr.strideParNode(parType);
		unsigned int sNode = idxr.strideParNode(parType);
		unsigned int sComp = 1u;
		unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });
		unsigned int nNodes = _disc.nParNode[parType];

		/* Special case */
		if (_disc.nParCell[parType] == 1) {

			linalg::BandedEigenSparseRowIterator jacIt(_globalJac, idxr.offsetCp(ParticleTypeIndex{parType}, ParticleIndex{colNode})); // row iterator starting at first cell, first component
			
			insertParJacBlock(_disc.DGjacParDispBlocks[_disc.offsetMetric[parType] + 0].block(0, nNodes + 1, nNodes, nNodes), jacIt, idxr, parDiff, parSurfDiff, invBetaP, _binding[parType]->reactionQuasiStationarity(), parType, 1u, 0);
			return 1;
		}

		/* Special case */
		if (_disc.nParCell[parType] == 2) {

			linalg::BandedEigenSparseRowIterator jacIt(_globalJac, idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode })); // row iterator starting at first cell, first component
			
			insertParJacBlock(_disc.DGjacParDispBlocks[_disc.offsetMetric[parType] + 0].block(0, nNodes + 1, nNodes, 2 * nNodes), jacIt, idxr, parDiff, parSurfDiff, invBetaP, _binding[parType]->reactionQuasiStationarity(), parType, 1u, 0);
			// right Bacobian block, iterator is already moved to second cell
			insertParJacBlock(_disc.DGjacParDispBlocks[_disc.offsetMetric[parType] + 1].block(0, 1, nNodes, 2 * nNodes), jacIt, idxr, parDiff, parSurfDiff, invBetaP, _binding[parType]->reactionQuasiStationarity(), parType, 1u, -idxr.strideParShell(parType));
			return 1;
		}

		/* Special case */
		if (_disc.nParCell[parType] == 3) {

			linalg::BandedEigenSparseRowIterator jacIt(_globalJac, idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode }) + idxr.strideParShell(parType)); // row iterator starting at first cell, first component
			
			insertParJacBlock(_disc.DGjacParDispBlocks[_disc.offsetMetric[parType] + 1].block(0, 1, nNodes, 3 * nNodes), jacIt, idxr, parDiff, parSurfDiff, invBetaP, _binding[parType]->reactionQuasiStationarity(), parType, 1u, -idxr.strideParShell(parType));
		}

		/* Inner cells (exist only if nCells >= 5) */
		if (_disc.nParCell[parType] >= 5) {

			linalg::BandedEigenSparseRowIterator jacIt(_globalJac, idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode }) + idxr.strideParShell(parType) * 2); // row iterator starting at third cell, first component
			
																																											  // insert all (nCol - 4) inner cell blocks
			for (unsigned int cell = 2; cell < _disc.nParCell[parType] - 2; cell++)
				insertParJacBlock(_disc.DGjacParDispBlocks[_disc.offsetMetric[parType] + cell], jacIt, idxr, parDiff, parSurfDiff, invBetaP, _binding[parType]->reactionQuasiStationarity(), parType, 1u, -(idxr.strideParShell(parType) + idxr.strideParNode(parType)));
		}

		/*	boundary cell neighbours (exist only if nCells >= 4)	*/
		if (_disc.nParCell[parType] >= 4) {

			linalg::BandedEigenSparseRowIterator jacIt(_globalJac, idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode }) + idxr.strideParShell(parType)); // row iterator starting at second cell, first component

			insertParJacBlock(_disc.DGjacParDispBlocks[_disc.offsetMetric[parType] + 1].block(0, 1, nNodes, 3 * nNodes + 1), jacIt, idxr, parDiff, parSurfDiff, invBetaP, _binding[parType]->reactionQuasiStationarity(), parType, 1u, -idxr.strideParShell(parType));

			jacIt += (_disc.nParCell[parType] - 4) * idxr.strideParShell(parType); // move iterator to preultimate cell (already at third cell)
			insertParJacBlock(_disc.DGjacParDispBlocks[_disc.offsetMetric[parType] + _disc.nParCell[parType] - 2u].block(0, 0, nNodes, 3 * nNodes + 1), jacIt, idxr, parDiff, parSurfDiff, invBetaP, _binding[parType]->reactionQuasiStationarity(), parType, 1u, -(idxr.strideParShell(parType) + idxr.strideParNode(parType)));
		}

		/*			boundary cells (exist only if nCells >= 3)			*/
		if (_disc.nParCell[parType] >= 3) {

			linalg::BandedEigenSparseRowIterator jacIt(_globalJac, idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode })); // row iterator starting at first cell, first component

			insertParJacBlock(_disc.DGjacParDispBlocks[_disc.offsetMetric[parType] + 0].block(0, nNodes + 1, nNodes, 2 * nNodes + 1), jacIt, idxr, parDiff, parSurfDiff, invBetaP, _binding[parType]->reactionQuasiStationarity(), parType, 1u, 0);

			jacIt += (_disc.nParCell[parType] - 2) * idxr.strideParShell(parType); // move iterator to last cell (already at second cell)
			insertParJacBlock(_disc.DGjacParDispBlocks[_disc.offsetMetric[parType] + _disc.nParCell[parType] - 1u].block(0, 0, nNodes, 2 * nNodes + 1), jacIt, idxr, parDiff, parSurfDiff, invBetaP, _binding[parType]->reactionQuasiStationarity(), parType, 1u, -(idxr.strideParShell(parType) + idxr.strideParNode(parType)));
		}

		return 1;
	}
	/**
	 * @brief analytically calculates the particle dispersion jacobian of the collocation DGSEM (inexact integration) for one particle type and bead
	 * @note deprecated, not further development
	 */
	int calcParticleCollocationDGSEMJacobian(unsigned int parType, unsigned int colNode, const active* const parDiff, const active* const parSurfDiff, const double* const invBetaP) {

		Indexer idxr(_disc);

		// (global) strides
		unsigned int sCell = _disc.nParNode[parType] * idxr.strideParNode(parType);
		unsigned int sNode = idxr.strideParNode(parType);
		unsigned int sComp = 1u;
		unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });
		unsigned int nNodes = _disc.nParNode[parType];

		// blocks to compute jacobian
		Eigen::MatrixXd dispBlock;
		Eigen::MatrixXd B = MatrixXd::Zero(nNodes, nNodes);
		B(0, 0) = -1.0; B(nNodes - 1, nNodes - 1) = 1.0;

		// special case: one cell -> diffBlock \in R^(nParNodes x nParNodes), GBlock = parPolyDerM
		if (_disc.nParCell[parType] == 1) {

			double invMap = (2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType]]));

			if (_parGeomSurfToVol[parType] == _disc.SurfVolRatioSlab || _parCoreRadius[parType] != 0.0)
				dispBlock = invMap * invMap * (_disc.Dr[parType] - _disc.parInvWeights[parType].asDiagonal() * B) * _disc.parPolyDerM[parType];

			else { // special treatment of inner boundary node for spherical and cylindrical particles without particle core

				dispBlock = MatrixXd::Zero(nNodes, nNodes);

				// reduced system
				dispBlock.block(1, 0, nNodes - 1, nNodes)
					= (_disc.Dr[parType].block(1, 1, nNodes - 1, nNodes - 1)
						- _disc.parInvWeights[parType].segment(1, nNodes - 1).asDiagonal() * B.block(1, 1, nNodes - 1, nNodes - 1))
					* _disc.parPolyDerM[parType].block(1, 0, nNodes - 1, nNodes);

				// inner boundary node
				dispBlock.block(0, 0, 1, nNodes)
					= -(_disc.Ir[parType].segment(1, nNodes - 1).template cast<double>().cwiseProduct(
						_disc.parInvWeights[parType].segment(1, nNodes - 1).cwiseInverse()).cwiseProduct(
							_disc.parPolyDerM[parType].block(1, 0, nNodes - 1, 1))).transpose()
					* _disc.parPolyDerM[parType].block(1, 0, nNodes - 1, nNodes);

				dispBlock *= invMap * invMap;
			}

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < dispBlock.cols(); j++) {
						// handle liquid state
						// row: add component offset and go node strides from there for each dispersion block entry
						// col: add component offset and go node strides from there for each dispersion block entry
						_globalJac.coeffRef(offset + comp * sComp + i * sNode,
											offset + comp * sComp + j * sNode)
							= -(static_cast<double>(parDiff[comp])) * dispBlock(i, j); // - D_p * (Delta r / 2)^2 * (D_r D - M^-1 B D)

						// handle surface diffusion of bound states. binding is handled in residualKernel().
						if (_hasSurfaceDiffusion[parType]) {

							int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								if (static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) != 0.0) {
									/* add surface diffusion dispersion block to liquid */
									// row: add current component offset and go node strides from there for each dispersion block entry
									// col: jump oover liquid states, add current bound state offset and go node strides from there for each dispersion block entry
									_globalJac.coeffRef(offset + comp * sComp + i * sNode,
														offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode)
										= -(static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) * invBetaP[comp]) * dispBlock(i, j); // -  D_s * (1 / Beta_p) * (Delta r / 2)^2 * (D_r D - M^-1 B D)

									/* add surface diffusion dispersion block to solid */
									if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
										// row: jump oover liquid states, add current bound state offset and go node strides from there for each dispersion block entry
										// col: jump oover liquid states, add current bound state offset and go node strides from there for each dispersion block entry
										_globalJac.coeffRef(offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
															offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode)
											= -(static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)])) * dispBlock(i, j); // -  D_s * (Delta r / 2)^2 * (D_r D - M^-1 B D)

									}
								}
							}
						}
					}
				}
			}
		}
		else {

			/*			boundary cells			*/

			// initialize dispersion and metric block matrices
			MatrixXd bnd_dispBlock = MatrixXd::Zero(nNodes, 2 * nNodes); // boundary cell specific
			dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes);

			// compute blocks used for inexact integration scheme
			// auxiliary block [ d g(c) / d c ] for left boundary cell
			MatrixXd GBlock_l = MatrixXd::Zero(nNodes, nNodes + 1);
			GBlock_l.block(0, 0, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock_l(nNodes - 1, nNodes - 1) -= 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			GBlock_l(nNodes - 1, nNodes) += 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			// auxiliary block [ d g(c) / d c ] for right boundary cell
			MatrixXd GBlock_r = MatrixXd::Zero(nNodes, nNodes + 1);
			GBlock_r.block(0, 1, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock_r(0, 0) -= 0.5 * _disc.parInvWeights[parType][0];
			GBlock_r(0, 1) += 0.5 * _disc.parInvWeights[parType][0];
			// numerical flux contribution for right interface of left boundary cell -> d f^*_N / d cp
			MatrixXd bnd_gStarDC = MatrixXd::Zero(nNodes, 2 * nNodes);
			bnd_gStarDC.block(nNodes - 1, 0, 1, nNodes + 1) = GBlock_l.block(nNodes - 1, 0, 1, nNodes + 1);
			bnd_gStarDC.block(nNodes - 1, nNodes - 1, 1, nNodes + 1) += GBlock_r.block(0, 0, 1, nNodes + 1);
			bnd_gStarDC *= 0.5;

			/*			 left boundary cell				*/
			double invMap = (2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType]]));

			// "standard" computation for slab-shaped particles and spherical, cylindrical particles without core
			if (_parGeomSurfToVol[parType] == _disc.SurfVolRatioSlab || _parCoreRadius[parType] != 0.0) {
				// dispBlock <- invMap^2 * ( D * G_l - M^-1 * B * [G_l - g^*] )
				bnd_dispBlock.block(0, 0, nNodes, nNodes + 1) = (_disc.Dr[_disc.offsetMetric[parType]] - _disc.parInvWeights[parType].asDiagonal() * B) * GBlock_l;
				bnd_dispBlock.block(0, 0, nNodes, 2 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * bnd_gStarDC;
				bnd_dispBlock *= invMap * invMap;
			}
			else { // special treatment of inner boundary node for spherical and cylindrical particles without particle core

				// inner boundary node
				bnd_dispBlock.block(0, 0, 1, nNodes + 1)
					= -(_disc.Ir[_disc.offsetMetric[parType]].template cast<double>().segment(1, nNodes - 1).cwiseProduct(
						_disc.parInvWeights[parType].segment(1, nNodes - 1).cwiseInverse()).cwiseProduct(
							_disc.parPolyDerM[parType].block(1, 0, nNodes - 1, 1))).transpose()
					* GBlock_l.block(1, 0, nNodes - 1, nNodes + 1);

				// reduced system for remaining nodes
				bnd_dispBlock.block(1, 0, nNodes - 1, nNodes + 1)
					= (_disc.Dr[_disc.offsetMetric[parType]].block(1, 1, nNodes - 1, nNodes - 1)
						- _disc.parInvWeights[parType].segment(1, nNodes - 1).asDiagonal() * B.block(1, 1, nNodes - 1, nNodes - 1)
						) * GBlock_l.block(1, 0, nNodes - 1, nNodes + 1);

				bnd_dispBlock.block(1, 0, nNodes - 1, 2 * nNodes)
					+= _disc.parInvWeights[parType].segment(1, nNodes - 1).asDiagonal() * B.block(1, 1, nNodes - 1, nNodes - 1) * bnd_gStarDC.block(1, 0, nNodes - 1, 2 * nNodes);

				// mapping
				bnd_dispBlock *= invMap * invMap;
			}
			
			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < bnd_dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < bnd_dispBlock.cols(); j++) {
						// handle liquid state
						// inexact integration pattern is more sparse than a nNodes x 2*nNodes block.
						if ((j <= nNodes) || (i == nNodes - 1)) {
							// row: add component offset and go node strides from there for each dispersion block entry
							// col: add component offset and go node strides from there for each dispersion block entry
							_globalJac.coeffRef(offset + comp * sComp + i * sNode,
												offset + comp * sComp + j * sNode)
								= -static_cast<double>(parDiff[comp]) * bnd_dispBlock(i, j); // dispBlock <- D_p * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

							// handle surface diffusion of bound states. binding is handled in residualKernel().
							if (_hasSurfaceDiffusion[parType]) {

								int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

								for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
									if (static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) != 0.0) {
										// row: add current component offset and go node strides from there for each dispersion block entry
										// col: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
										_globalJac.coeffRef(offset + comp * sComp + i * sNode,
															offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode)
											= -static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) * invBetaP[comp] * bnd_dispBlock(i, j); // dispBlock <- D_s * invBeta * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

										/* add surface diffusion dispersion block to solid */
										if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
											// row: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
											// col: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
											_globalJac.coeffRef(offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode)
												= -(static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)])) * bnd_dispBlock(i, j); // -  D_s * (Delta r / 2)^2 * (D_r D - M^-1 B D)

										}
									}
								}
							}
						}
					}
				}
			}

			/*			 right boundary cell				*/
			invMap = (2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType] + _disc.nParCell[parType] - 1]));

			// numerical flux contribution for left interface of right boundary cell -> d f^*_0 / d cp
			bnd_gStarDC.setZero();
			bnd_gStarDC.block(0, nNodes - 1, 1, nNodes + 1) = GBlock_r.block(0, 0, 1, nNodes + 1);
			bnd_gStarDC.block(0, 0, 1, nNodes + 1) += GBlock_l.block(nNodes - 1, 0, 1, nNodes + 1);
			bnd_gStarDC *= 0.5;
			// dispBlock <- invMap * ( D_r * G_r - M^-1 * B * [G_r - g^*] )
			bnd_dispBlock.setZero();
			bnd_dispBlock.block(0, nNodes - 1, nNodes, nNodes + 1) = (_disc.Dr[_disc.offsetMetric[parType] + _disc.nParCell[parType] - 1] - _disc.parInvWeights[parType].asDiagonal() * B) * GBlock_r;
			bnd_dispBlock.block(0, 0, nNodes, 2 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * bnd_gStarDC;
			bnd_dispBlock *= invMap * invMap;

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < bnd_dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < bnd_dispBlock.cols(); j++) {
						// handle liquid state
						// inexact integration pattern is more sparse than a nNodes x 2*nNodes block.
						if ((j <= nNodes) || (i == nNodes - 1)) {
							// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
							// col: add component offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
							_globalJac.coeffRef(offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
												offset + comp * sComp + (_disc.nParCell[parType] - 2) * sCell + j * sNode)
								= -static_cast<double>(parDiff[comp]) * bnd_dispBlock(i, j); // dispBlock <- D_p * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

							// handle surface diffusion of bound states. binding is handled in residualKernel().
							if (_hasSurfaceDiffusion[parType]) {

								int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

								for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
									if (static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) != 0.0) {
										// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
										// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
										_globalJac.coeffRef(offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
															offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (_disc.nParCell[parType] - 2) * sCell + j * sNode)
											= -static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) * invBetaP[comp] * bnd_dispBlock(i, j); // dispBlock <- D_s * invBeta * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

										/* add surface diffusion dispersion block to solid */
										if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
											// row: jump over previous cells and over liquid states, add current bound state offset. go node strides from there for each dispersion block entry
											// col: jump over previous cells and over liquid states, add current bound state offset. go back one cell and go node strides from there for each dispersion block entry
											_globalJac.coeffRef(offset + (_disc.nParCell[parType] - 1) * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																offset + (_disc.nParCell[parType] - 2) * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode)
												= -(static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)])) * bnd_dispBlock(i, j); // -  D_s * (Delta r / 2)^2 * (D_r D - M^-1 B D)

										}
									}
								}
							}
						}
					}
				}
			}

			/*				inner cells				*/

			// auxiliary block [ d g(c) / d c ] for inner cells
			MatrixXd GBlock = MatrixXd::Zero(nNodes, nNodes + 2);
			GBlock.block(0, 1, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock(0, 0) -= 0.5 * _disc.parInvWeights[parType][0];
			GBlock(0, 1) += 0.5 * _disc.parInvWeights[parType][0];
			GBlock(nNodes - 1, nNodes) -= 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			GBlock(nNodes - 1, nNodes + 1) += 0.5 * _disc.parInvWeights[parType][nNodes - 1];

			// numerical flux contribution
			MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes);
			gStarDC.block(0, nNodes - 1, 1, nNodes + 2) = GBlock.block(0, 0, 1, nNodes + 2);
			gStarDC.block(0, 0, 1, nNodes + 1) += GBlock.block(nNodes - 1, 1, 1, nNodes + 1);
			gStarDC.block(nNodes - 1, nNodes - 1, 1, nNodes + 2) += GBlock.block(nNodes - 1, 0, 1, nNodes + 2);
			gStarDC.block(nNodes - 1, 2 * nNodes - 1, 1, nNodes + 1) += GBlock.block(0, 0, 1, nNodes + 1);
			gStarDC *= 0.5;

			dispBlock.setZero();
			// dispersion block part without metrics
			dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) = -1.0 * _disc.parInvWeights[parType].asDiagonal() * B * GBlock;
			dispBlock.block(0, 0, nNodes, 3 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * gStarDC;
			dispBlock *= invMap * invMap;

			for (int cell = 1; cell < _disc.nParCell[parType] - 1; cell++) {
				double invMap = (2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType] + cell]));

				// add metric part, dependent on current cell
				dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) += _disc.Dr[_disc.offsetMetric[parType] + cell] * GBlock * invMap * invMap;

				// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					for (unsigned int i = 0; i < dispBlock.rows(); i++) {
						for (unsigned int j = 0; j < dispBlock.cols(); j++) {
							// handle liquid state
							// pattern is more sparse than a nNodes x 3*nNodes block.
							if ((j >= nNodes - 1 && j <= 2 * nNodes) ||
								(i == 0 && j <= 2 * nNodes) ||
								(i == nNodes - 1 && j >= nNodes - 1)) {
								// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
								// col: add component offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
								_globalJac.coeffRef(offset + comp * sComp + cell * sCell + i * sNode,
													offset + comp * sComp + (cell - 1) * sCell + j * sNode)
									= -static_cast<double>(parDiff[comp]) * dispBlock(i, j); // dispBlock <- D_p * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

								// handle surface diffusion of bound states. binding is handled in residualKernel().
								if (_hasSurfaceDiffusion[parType]) {

									int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();

									for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
										if (static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) != 0.0) {
											// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
											// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
											_globalJac.coeffRef(offset + comp * sComp + cell * sCell + i * sNode,
																offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (cell - 1) * sCell + j * sNode)
												= -static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) * invBetaP[comp] * dispBlock(i, j); // dispBlock <- D_s * invBeta * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

											/* add surface diffusion dispersion block to solid */
											if (!qsReaction[idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd]) {
												// row: jump over previous cells and liquid states, add current bound state offset and go node strides from there for each dispersion block entry
												// col: jump over previous cells and liquid states, go back one cell and add current bound state offset and go node strides from there for each dispersion block entry
												_globalJac.coeffRef(offset + cell * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + i * sNode,
																	offset + (cell - 1) * sCell + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode)
													= -(static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)])) * dispBlock(i, j); // -  D_s * (Delta r / 2)^2 * (D_r D - M^-1 B D)

											}
										}
									}
								}
							}
						}
					}
				}
				// substract metric part in preparation of next iteration
				dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) -= _disc.Dr[_disc.offsetMetric[parType] + cell] * GBlock * invMap * invMap;
			}

		} // if nCells > 1

		return 1;
	}
	/**
	 * @brief adds jacobian entries which have been overwritten by the binding kernel (only use for surface diffusion combined with kinetic binding)
	 * @detail only adds the entries d RHS_i / d c^s_i, which lie on the diagonal
	 * @parType[in] current particle type
	 * @parSurfDiff[in] pointer to particle surface diffusion at current section and particle type
	 */
	int addSolidDGentries(unsigned int parType, const active* const parSurfDiff) {

		if (!_disc.parExactInt[parType])
			return addSolidDGentries_inexInt(parType, parSurfDiff);

		Indexer idxr(_disc);

		for (unsigned int col = 0; col < _disc.nPoints; col++) {
			// Get jacobian iterator at first solid entry of first particle of current type
			linalg::BandedEigenSparseRowIterator jac(_globalJac, idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ col }) + idxr.strideParLiquid());

			for (unsigned int cell = 0; cell < _disc.nParCell[parType]; cell++)
				addDiagonalSolidJacobianEntries(_disc.DGjacParDispBlocks[_disc.offsetMetric[parType] + cell].block(0, _disc.nParNode[parType] + 1, _disc.nParNode[parType], _disc.nParNode[parType]),
					jac, idxr, parSurfDiff, _binding[parType]->reactionQuasiStationarity(), parType);
		}

		return 1;
	}
	/**
	 * @brief adds jacobian entries which have been overwritten by the binding kernel (only use for surface diffusion combined with kinetic binding)
	 * @detail only adds the entries d RHS_i / d c^s_i, which lie on the diagonal
	 * @parType[in] current particle type
	 * @parSurfDiff[in] pointer to particle surface diffusion at current section and particle type
	 */
	int addSolidDGentries_inexInt(unsigned int parType, const active* const parSurfDiff) {

		Indexer idxr(_disc);

		// (global) strides
		unsigned int sCell = _disc.nParNode[parType] * idxr.strideParNode(parType);
		unsigned int sNode = idxr.strideParNode(parType);
		unsigned int sComp = 1u;
		unsigned int nNodes = _disc.nParNode[parType];

		// blocks to compute jacobian
		Eigen::MatrixXd dispBlock;
		Eigen::MatrixXd B = MatrixXd::Zero(nNodes, nNodes);
		B(0, 0) = -1.0; B(nNodes - 1, nNodes - 1) = 1.0;

		// special case: one cell -> diffBlock \in R^(nParNodes x nParNodes), GBlock = parPolyDerM
		if (_disc.nParCell[parType] == 1) {

			double invMap = (2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType]]));

			if (_parGeomSurfToVol[parType] == _disc.SurfVolRatioSlab || _parCoreRadius[parType] != 0.0)
				dispBlock = invMap * invMap * (_disc.Dr[parType] - _disc.parInvWeights[parType].asDiagonal() * B) * _disc.parPolyDerM[parType];

			else { // special treatment of inner boundary node for spherical and cylindrical particles without particle core

				dispBlock = MatrixXd::Zero(nNodes, nNodes);

				// reduced system
				dispBlock.block(1, 0, nNodes - 1, nNodes)
					= (_disc.Dr[parType].block(1, 1, nNodes - 1, nNodes - 1)
						- _disc.parInvWeights[parType].segment(1, nNodes - 1).asDiagonal() * B.block(1, 1, nNodes - 1, nNodes - 1))
					* _disc.parPolyDerM[parType].block(1, 0, nNodes - 1, nNodes);

				// inner boundary node
				dispBlock.block(0, 0, 1, nNodes)
					= -(_disc.Ir[parType].segment(1, nNodes - 1).template cast<double>().cwiseProduct(
						_disc.parInvWeights[parType].segment(1, nNodes - 1).cwiseInverse()).cwiseProduct(
							_disc.parPolyDerM[parType].block(1, 0, nNodes - 1, 1))).transpose()
					* _disc.parPolyDerM[parType].block(1, 0, nNodes - 1, nNodes);

				dispBlock *= invMap * invMap;
			}

			for (unsigned int colNode = 0; colNode < _disc.nPoints; colNode++) {

				unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });
				// start at first solid entry
				linalg::BandedEigenSparseRowIterator jac(_globalJac, offset + idxr.strideParLiquid());

				for (unsigned int node = 0; node < _disc.nParNode[parType]; node++, jac += idxr.strideParLiquid()) {

					// @TODO test if we use correct diffusion parameter entry for more complicated cases, i.e. multiple comp, bnd states
					for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
						for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++, ++jac) {
							if (static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) != 0.0) {
								jac[0] += -(static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)])) * dispBlock(node, node);
							}
						}
					}
				}
			}
		}
		else {

			/*			boundary cells			*/
			// initialize dispersion and metric block matrices
			MatrixXd bnd_dispBlock = MatrixXd::Zero(nNodes, 2 * nNodes); // boundary cell specific
			dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes);

			// auxiliary block [ d g(c) / d c ] for left boundary cell
			MatrixXd GBlock_l = MatrixXd::Zero(nNodes, nNodes + 1);
			GBlock_l.block(0, 0, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock_l(nNodes - 1, nNodes - 1) -= 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			GBlock_l(nNodes - 1, nNodes) += 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			// auxiliary block [ d g(c) / d c ] for right boundary cell
			MatrixXd GBlock_r = MatrixXd::Zero(nNodes, nNodes + 1);
			GBlock_r.block(0, 1, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock_r(0, 0) -= 0.5 * _disc.parInvWeights[parType][0];
			GBlock_r(0, 1) += 0.5 * _disc.parInvWeights[parType][0];

			/*			 left boundary cell				*/
			int _cell = 0;
			double invMap = (2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType] + _cell]));

			// numerical flux contribution for right interface of left boundary cell -> d f^*_N / d cp
			MatrixXd bnd_gStarDC = MatrixXd::Zero(nNodes, 2 * nNodes);
			bnd_gStarDC.block(nNodes - 1, 0, 1, nNodes + 1) = GBlock_l.block(nNodes - 1, 0, 1, nNodes + 1);
			bnd_gStarDC.block(nNodes - 1, nNodes - 1, 1, nNodes + 1) += GBlock_r.block(0, 0, 1, nNodes + 1);
			bnd_gStarDC *= 0.5;

			// "standard" computation for slab-shaped particles and spherical, cylindrical particles without core
			if (_parGeomSurfToVol[parType] == _disc.SurfVolRatioSlab || _parCoreRadius[parType] != 0.0) {
				// dispBlock <- invMap^2 * ( D * G_l - M^-1 * B * [G_l - g^*] )
				bnd_dispBlock.block(0, 0, nNodes, nNodes + 1) = (_disc.Dr[_disc.offsetMetric[parType]] - _disc.parInvWeights[parType].asDiagonal() * B) * GBlock_l;
				bnd_dispBlock.block(0, 0, nNodes, 2 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * bnd_gStarDC;
				bnd_dispBlock *= invMap * invMap;
			}
			else { // special treatment of inner boundary node for spherical and cylindrical particles without particle core

				// inner boundary node
				bnd_dispBlock.block(0, 0, 1, nNodes + 1)
					= -(_disc.Ir[_disc.offsetMetric[parType]].template cast<double>().segment(1, nNodes - 1).cwiseProduct(
						_disc.parInvWeights[parType].segment(1, nNodes - 1).cwiseInverse()).cwiseProduct(
							_disc.parPolyDerM[parType].block(1, 0, nNodes - 1, 1))).transpose()
					* GBlock_l.block(1, 0, nNodes - 1, nNodes + 1);

				// reduced system for remaining nodes
				bnd_dispBlock.block(1, 0, nNodes - 1, nNodes + 1)
					= (_disc.Dr[_disc.offsetMetric[parType]].block(1, 1, nNodes - 1, nNodes - 1)
						- _disc.parInvWeights[parType].segment(1, nNodes - 1).asDiagonal() * B.block(1, 1, nNodes - 1, nNodes - 1)
						) * GBlock_l.block(1, 0, nNodes - 1, nNodes + 1);

				bnd_dispBlock.block(1, 0, nNodes - 1, 2 * nNodes)
					+= _disc.parInvWeights[parType].segment(1, nNodes - 1).asDiagonal() * B.block(1, 1, nNodes - 1, nNodes - 1) * bnd_gStarDC.block(1, 0, nNodes - 1, 2 * nNodes);

				// mapping
				bnd_dispBlock *= invMap * invMap;
			}

			for (unsigned int colNode = 0; colNode < _disc.nPoints; colNode++) {

				unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });
				// start at first solid entry of first cell
				linalg::BandedEigenSparseRowIterator jac_left(_globalJac, offset + idxr.strideParLiquid());

				for (unsigned int node = 0; node < _disc.nParNode[parType]; node++, jac_left += idxr.strideParLiquid()) {
					for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
						for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++, ++jac_left) {
							if (static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) != 0.0) {
								jac_left[0] += -(static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)])) * bnd_dispBlock(node, node);
							}
						}
					}
				}
			}

			/*			 right boundary cell				*/
			_cell = _disc.nParCell[parType] - 1;
			invMap = (2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType] + _cell]));

			bnd_gStarDC = MatrixXd::Zero(nNodes, 2 * nNodes);
			// numerical flux contribution for left interface of right boundary cell -> d f^*_0 / d cp
			bnd_gStarDC.setZero();
			bnd_gStarDC.block(0, nNodes - 1, 1, nNodes + 1) = GBlock_r.block(0, 0, 1, nNodes + 1);
			bnd_gStarDC.block(0, 0, 1, nNodes + 1) += GBlock_l.block(nNodes - 1, 0, 1, nNodes + 1);
			bnd_gStarDC *= 0.5;
			// dispBlock <- invMap * ( D_r * G_r - M^-1 * B * [G_r - g^*] )
			bnd_dispBlock.setZero();
			bnd_dispBlock.block(0, nNodes - 1, nNodes, nNodes + 1) = (_disc.Dr[_disc.offsetMetric[parType] + _cell] - _disc.parInvWeights[parType].asDiagonal() * B) * GBlock_r;
			bnd_dispBlock.block(0, 0, nNodes, 2 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * bnd_gStarDC;
			bnd_dispBlock *= invMap * invMap;

			for (unsigned int colNode = 0; colNode < _disc.nPoints; colNode++) {

				unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });
				// start at first solid entry of last cell
				linalg::BandedEigenSparseRowIterator jac_right(_globalJac, offset + (_disc.nParCell[parType] - 1) * sCell + idxr.strideParLiquid());

				for (unsigned int node = 0; node < _disc.nParNode[parType]; node++, jac_right += idxr.strideParLiquid()) {
					for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
						for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++, ++jac_right) {
							if (static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) != 0.0) {
								jac_right[0] += -(static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)])) * bnd_dispBlock(node, _disc.nParNode[parType] + node);
							}
						}
					}
				}
			}

			/*				inner cells				*/

				// auxiliary block [ d g(c) / d c ] for inner cells
			MatrixXd GBlock = MatrixXd::Zero(nNodes, nNodes + 2);
			GBlock.block(0, 1, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock(0, 0) -= 0.5 * _disc.parInvWeights[parType][0];
			GBlock(0, 1) += 0.5 * _disc.parInvWeights[parType][0];
			GBlock(nNodes - 1, nNodes) -= 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			GBlock(nNodes - 1, nNodes + 1) += 0.5 * _disc.parInvWeights[parType][nNodes - 1];

			// numerical flux contribution
			MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes);
			gStarDC.block(0, nNodes - 1, 1, nNodes + 2) = GBlock.block(0, 0, 1, nNodes + 2);
			gStarDC.block(0, 0, 1, nNodes + 1) += GBlock.block(nNodes - 1, 1, 1, nNodes + 1);
			gStarDC.block(nNodes - 1, nNodes - 1, 1, nNodes + 2) += GBlock.block(nNodes - 1, 0, 1, nNodes + 2);
			gStarDC.block(nNodes - 1, 2 * nNodes - 1, 1, nNodes + 1) += GBlock.block(0, 0, 1, nNodes + 1);
			gStarDC *= 0.5;

			dispBlock.setZero();
			// dispersion block part without metrics
			dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) = -1.0 * _disc.parInvWeights[parType].asDiagonal() * B * GBlock;
			dispBlock.block(0, 0, nNodes, 3 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * gStarDC;

			for (int cell = 1; cell < _disc.nParCell[parType] - 1; cell++) {

				// add metric part, dependent on current cell
				dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) += _disc.Dr[_disc.offsetMetric[parType] + cell] * GBlock;
				invMap = (2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[parType] + cell]));
				dispBlock *= invMap * invMap;

				for (unsigned int colNode = 0; colNode < _disc.nPoints; colNode++) {

					unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });
					// start at first solid entry of current inner cell
					linalg::BandedEigenSparseRowIterator jac_inner(_globalJac, offset + cell * sCell + idxr.strideParLiquid());

					for (unsigned int node = 0; node < _disc.nParNode[parType]; node++, jac_inner += idxr.strideParLiquid()) {
						for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++, ++jac_inner) {
								if (static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)]) != 0.0) {
									jac_inner[0] += -(static_cast<double>(parSurfDiff[getOffsetSurfDiff(parType, comp, bnd)])) * dispBlock(node, _disc.nParNode[parType] + node);
								}
							}
						}
					}
				}

				// substract metric part in preparation of next iteration
				dispBlock /= invMap * invMap;
				dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) -= _disc.Dr[_disc.offsetMetric[parType] + cell] * GBlock;
			}
		} // if nCells > 1

		return 1;
	}
	/**
	 * @brief adds a state block into the system jacobian.
	 * @param [in] block (sub)block to be added
	 * @param [in] jac row iterator at first (i.e. upper) entry
	 * @param [in] offRowToCol column to row offset (i.e. start at upper left corner of block)
	 * @param [in] idxr Indexer
	 * @param [in] nCells determines how often the block is added (diagonally)
	 * @param [in] stateFactor state dependend factors
	 * @param [in] strideDead how many (dead) states to be jumped over after each state block
	 * @param [in] nStates how many states are concerned, defaults to nComp
	 */
	void addJacBlock(Eigen::MatrixXd block, linalg::BandedEigenSparseRowIterator& jac, int offRowToCol, Indexer& idxr, unsigned int nCells, unsigned int strideNode, unsigned int nStates, unsigned int strideDead = 0) {

		for (unsigned int cell = 0; cell < nCells; cell++) {
			for (unsigned int i = 0; i < block.rows(); i++, jac += strideDead) {
				for (unsigned int state = 0; state < nStates; state++, ++jac) {
					for (unsigned int j = 0; j < block.cols(); j++) {
						// row: at current node component
						// col: jump to node j
						jac[(j - i) * strideNode + offRowToCol] += block(i, j);
					}
				}
			}
		}
	}
	/**
	 * @brief adds a state block into the system jacobian.
	 * @param [in] block (sub)block whose diagonal entries are to be added
	 * @param [in] jac row iterator at first (i.e. upper) entry
	 * @param [in] idxr Indexer
	 * @param [in] surfDiff pointer to surfaceDiffusion storage
	 * @param [in] nonKinetic pointer to binding kinetics
	 * @param [in] type particle type
	 */
	template<typename ParamType>
	void addDiagonalSolidJacobianEntries(Eigen::MatrixXd block, linalg::BandedEigenSparseRowIterator& jac, Indexer& idxr, ParamType* surfDiff, const int* nonKinetic, unsigned int type) {

		for (unsigned int i = 0; i < block.rows(); i++, jac += idxr.strideParLiquid()) {
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int bnd = 0; bnd < _disc.nBound[type * _disc.nComp + comp]; bnd++, ++jac) {
					if (static_cast<double>(surfDiff[_disc.offsetSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{ type }, ComponentIndex{ comp }) + bnd]]) != 0.0
						&& !nonKinetic[idxr.offsetBoundComp(ParticleTypeIndex{ type }, ComponentIndex{ comp }) + bnd]) {
						// row, col: at current node and bound state
						jac[0] += block(i, i)
							* static_cast<double>(surfDiff[_disc.offsetSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{ type }, ComponentIndex{ comp }) + bnd]]);
					}
				}
			}
		}
	}
	/**
	 * @brief adds a state block into the particle jacobian.
	 * @param [in] block (sub)block to be added
	 * @param [in] jac row iterator at first (i.e. upper) entry
	 * @param [in] idxr Indexer
	 * @param [in] idxr parDiff pointer to particle diffusion parameters
	 * @param [in] idxr surfDiff pointer to particle surface diffusion parameters
	 * @param [in] idxr beta_p pointer to particle porosity parameters
	 * @param [in] idxr nonKinetic pointer to binding kinetics parameters
	 * @param [in] type particle type
	 * @param [in] nBlocks number of blocks, i.e. cells/elements, to be inserted
	 * @param [in] offRowToCol column to row offset (i.e. start at upper left corner of block)
	 */
	void insertParJacBlock(Eigen::MatrixXd block, linalg::BandedEigenSparseRowIterator& jac, Indexer& idxr, const active* const parDiff, const active* const surfDiff, const double* const beta_p, const int* nonKinetic, unsigned int type, unsigned int nBlocks, int offRowToCol) {

		for (unsigned int cell = 0; cell < nBlocks; cell++) {
			for (unsigned int i = 0; i < block.rows(); i++) {
				for (unsigned int comp = 0; comp < _disc.nComp; comp++, ++jac) {
					for (unsigned int j = 0; j < block.cols(); j++) {
						/* liquid on liquid blocks */
						// row: at current node and component; col: jump to node j
						jac[(j - i) * idxr.strideParNode(type) + offRowToCol] = block(i, j) * static_cast<double>(parDiff[comp]);
					}
					/* liquid on solid blocks */
					for (unsigned int bnd = 0; bnd < _disc.nBound[type * _disc.nComp + comp]; bnd++) {
						if (static_cast<double>(surfDiff[_disc.offsetSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{ type }, ComponentIndex{ comp }) + bnd]]) != 0.0) {
							for (unsigned int j = 0; j < block.cols(); j++) {
								// row: at current node and component; col: jump to node j and to current bound state
								jac[(j - i) * idxr.strideParNode(type) + offRowToCol + idxr.strideParLiquid() - comp
									+ idxr.offsetBoundComp(ParticleTypeIndex{ type }, ComponentIndex{ comp }) + bnd
								]
									= block(i, j) * static_cast<double>(beta_p[comp])
									* static_cast<double>(surfDiff[_disc.offsetSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{ type }, ComponentIndex{ comp }) + bnd]]);
							}
						}
					}
				}
				/* solid on solid blocks */
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					for (unsigned int bnd = 0; bnd < _disc.nBound[type * _disc.nComp + comp]; bnd++, ++jac) {
						if (static_cast<double>(surfDiff[_disc.offsetSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{ type }, ComponentIndex{ comp }) + bnd]]) != 0.0
							&& !nonKinetic[idxr.offsetBoundComp(ParticleTypeIndex{ type }, ComponentIndex{ comp }) + bnd]) {
							for (unsigned int j = 0; j < block.cols(); j++) {
								// row: at current node and bound state; col: jump to node j
								jac[(j - i) * idxr.strideParNode(type) + offRowToCol + bnd]
									= block(i, j)
									* static_cast<double>(surfDiff[_disc.offsetSurfDiff[idxr.offsetBoundComp(ParticleTypeIndex{ type }, ComponentIndex{ comp }) + bnd]]);
							}
						}
					}
				}
			}
		}
	}
	/**
	 * @brief analytically calculates the static (per section) particle jacobian
	 * @return 1 if jacobain calculation fits the predefined pattern of the jacobian, 0 if not.
	 */
	int calcStaticAnaParticleDispJacobian(unsigned int parType, unsigned int colNode, const active* const parDiff, const active* const parSurfDiff, const double* const invBetaP) {

		// DG particle dispersion Jacobian
		if(_disc.parExactInt[parType])
			calcParticleDGSEMJacobian(parType, colNode, parDiff, parSurfDiff, invBetaP);
		else // deprecated
			calcParticleCollocationDGSEMJacobian(parType, colNode, parDiff, parSurfDiff, invBetaP);

		return _globalJac.isCompressed(); // if matrix lost its compressed storage, the calculation did not fit the pre-defined pattern.
	}


	void setJacobianPattern_GRM(SparseMatrix<double, RowMajor>& globalJ, unsigned int secIdx, bool hasBulkReaction) {

		Indexer idxr(_disc);

		std::vector<T> tripletList;
		// reserve space for all entries
		int bulkEntries = _convDispOp.nConvDispEntries(false);
		if (hasBulkReaction)
			bulkEntries += _disc.nPoints * _disc.nComp * _disc.nComp; // add nComp entries for every component at each discrete bulk point

		// particle
		int addTimeDer = 0; // additional time derivative entries: bound states in particle dispersion equation
		int isothermNNZ = 0;
		int particleEntries = 0;
		for (int type = 0; type < _disc.nParType; type++) {
			isothermNNZ = (idxr.strideParNode(type)) * _disc.nParPoints[type] * _disc.strideBound[type]; // every bound satte might depend on every bound and liquid state
			addTimeDer = _disc.nParPoints[type] * _disc.strideBound[type];
			particleEntries += calcParDispNNZ(type) + addTimeDer + isothermNNZ;
		}

		int fluxEntries = 4 * _disc.nParType * _disc.nPoints * _disc.nComp;

		tripletList.reserve(fluxEntries + bulkEntries + particleEntries);

		// NOTE: inlet and jacF flux jacobian are set in calc jacobian function (identity matrices)
		// Note: flux jacobian (identity matrix) is handled in calc jacobian function
		unsigned int bulkOffset = idxr.offsetC();
		_convDispOp.convDispJacPattern(tripletList, bulkOffset);

		// bulk reaction jacobian
		if (hasBulkReaction) {
			for (unsigned int colNode = 0; colNode < _disc.nPoints; colNode++) {
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					for (unsigned int toComp = 0; toComp < _disc.nComp; toComp++) {
						tripletList.push_back(T(idxr.offsetC() + colNode * idxr.strideColNode() + comp * idxr.strideColComp(),
							idxr.offsetC() + colNode * idxr.strideColNode() + toComp * idxr.strideColComp(),
							0.0));
					}
				}
			}
		}

		// particle jacobian (including isotherm and time derivative)
		for (int colNode = 0; colNode < _disc.nPoints; colNode++) {
			for (int type = 0; type < _disc.nParType; type++) {
				setParJacPattern(tripletList, type, colNode, secIdx);
			}
		}

		// flux jacobians
		for (unsigned int type = 0; type < _disc.nParType; type++) {
			for (unsigned int colNode = 0; colNode < _disc.nPoints; colNode++) {
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					// add Cl on Cp entries
					// row: add bulk offset, jump over previous nodes and components
					// col: add flux offset to current parType, jump over previous nodes and components
					tripletList.push_back(T(idxr.offsetC() + colNode * idxr.strideColNode() + comp * idxr.strideColComp(),
											idxr.offsetCp(ParticleTypeIndex{ type }, ParticleIndex{ colNode }) + (_disc.nParPoints[type] - 1) * idxr.strideParNode(type) + comp * idxr.strideParComp(), 0.0));

					// add Cp on Cl entries
					if(!_disc.parExactInt[type])
						// row: add particle offset to current parType and particle, go to last node and add component offset
						// col: add flux offset to current component, jump over previous nodes and components
						tripletList.push_back(T(idxr.offsetCp(ParticleTypeIndex{ type }, ParticleIndex{ colNode })
												+ (_disc.nParPoints[type] - 1) * idxr.strideParNode(type) + comp * idxr.strideParComp(),
												idxr.offsetC() + colNode * idxr.strideColNode() + comp, 0.0));
					else {
						for (unsigned int node = 0; node < _disc.nParNode[type]; node++) {
							// row: add particle offset to current parType and particle, go to last cell and current node and add component offset
							// col: add flux offset to current component, jump over previous nodes and components
							tripletList.push_back(T(idxr.offsetCp(ParticleTypeIndex{ type }, ParticleIndex{ colNode }) + (_disc.nParCell[type] - 1) * _disc.nParNode[type] * idxr.strideParNode(type)
													+ node * idxr.strideParNode(type) + comp * idxr.strideParComp(),
													idxr.offsetC() + colNode * idxr.strideColNode() + comp, 0.0));
						}
					}
				}
			}
		}

		globalJ.setFromTriplets(tripletList.begin(), tripletList.end());
	}

	int calcFluxJacobians(unsigned int secIdx, bool outliersOnly = false) {

		Indexer idxr(_disc);

		for (unsigned int type = 0; type < _disc.nParType; type++) {

			// lifting matrix entry for exact integration scheme depends on metrics for sphere and cylinder
			double exIntLiftContribution = static_cast<double>(_disc.Ir[_disc.offsetMetric[type] + _disc.nParCell[type] - 1][_disc.nParNode[type] - 1]);
			if (_parGeomSurfToVol[type] == _disc.SurfVolRatioSlab)
				exIntLiftContribution = 1.0;

			// Ordering of diffusion:
			// sec0type0comp0, sec0type0comp1, sec0type0comp2, sec0type1comp0, sec0type1comp1, sec0type1comp2,
			// sec1type0comp0, sec1type0comp1, sec1type0comp2, sec1type1comp0, sec1type1comp1, sec1type1comp2, ...
			active const* const filmDiff = getSectionDependentSlice(_filmDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;

			linalg::BandedEigenSparseRowIterator jacCl(_globalJac, idxr.offsetC());
			linalg::BandedEigenSparseRowIterator jacCp(_globalJac, idxr.offsetCp(ParticleTypeIndex{ type }) + (_disc.nParPoints[type] - 1) * idxr.strideParNode(type));

			for (unsigned int colNode = 0; colNode < _disc.nPoints; colNode++)
			{
				for (unsigned int comp = 0; comp < _disc.nComp; comp++, ++jacCp, ++jacCl) {
					// add Cl on Cl entries (added since these entries are also touched by bulk jacobian)
					// row: already at bulk phase. already at current node and component.
					// col: already at bulk phase. already at current node and component.
					if(!outliersOnly)
						jacCl[0] += static_cast<double>(filmDiff[comp]) * (1.0 - static_cast<double>(_colPorosity)) / static_cast<double>(_colPorosity)
									* _parGeomSurfToVol[type] / static_cast<double>(_parRadius[type])
									* _parTypeVolFrac[type + colNode * _disc.nParType].getValue();
					// add Cl on Cp entries (added since these entries are also touched by bulk jacobian)
					// row: already at bulk phase. already at current node and component.
					// col: go to current particle phase entry.
					jacCl[jacCp.row() - jacCl.row()] = -static_cast<double>(filmDiff[comp]) * (1.0 - static_cast<double>(_colPorosity)) / static_cast<double>(_colPorosity)
														* _parGeomSurfToVol[type] / static_cast<double>(_parRadius[type])
														* _parTypeVolFrac[type + colNode * _disc.nParType].getValue();

					// add Cp on Flux entries
					if (!_disc.parExactInt[type]) {
						// row: already at particle. already at current node and liquid state.
						// col: already at particle. already at current node and liquid state.
						if (!outliersOnly) // Cp on Cb
							jacCp[0] += static_cast<double>(filmDiff[comp]) * 2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[type]]) * _disc.parInvWeights[type][0] / static_cast<double>(_parPorosity[type]) / static_cast<double>(_poreAccessFactor[type * _disc.nComp + comp]);
						// row: already at particle. already at current node and liquid state.
						// col: go to current bulk phase.
						jacCp[jacCl.row() - jacCp.row()] = -static_cast<double>(filmDiff[comp]) * 2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[type]]) * _disc.parInvWeights[type][0] / static_cast<double>(_parPorosity[type]) / static_cast<double>(_poreAccessFactor[type * _disc.nComp + comp]);
					}
					else {
						unsigned int entry = jacCp.row();
						for (int node = _disc.parPolyDeg[type]; node >= 0; node--, jacCp -= idxr.strideParNode(type)) {
							// row: already at particle. Already at current node and liquid state.
							// col: original entry at outer node.
							if (!outliersOnly) // Cp on Cb
								jacCp[entry - jacCp.row()]
								+= static_cast<double>(filmDiff[comp]) * 2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[type]]) * _disc.parInvMM[_disc.offsetMetric[type] + _disc.nParCell[type] - 1](node, _disc.nParNode[type] - 1) * exIntLiftContribution / static_cast<double>(_parPorosity[type]) / static_cast<double>(_poreAccessFactor[type * _disc.nComp + comp]);
							// row: already at particle. Already at current node and liquid state.
							// col: go to current bulk phase.
							jacCp[jacCl.row() - jacCp.row()]
								= -static_cast<double>(filmDiff[comp]) * 2.0 / static_cast<double>(_disc.deltaR[_disc.offsetMetric[type]]) * _disc.parInvMM[_disc.offsetMetric[type] + _disc.nParCell[type] - 1](node, _disc.nParNode[type] - 1) * exIntLiftContribution / static_cast<double>(_parPorosity[type]) / static_cast<double>(_poreAccessFactor[type * _disc.nComp + comp]);
						}
						// set back iterator to first node as required by component loop
						jacCp += _disc.nParNode[type] * idxr.strideParNode(type);
					}
				}
				if (colNode < _disc.nPoints - 1) // execute iteration statement only when condition is true in next loop.
					jacCp += _disc.strideBound[type] + (_disc.nParPoints[type] - 1) * idxr.strideParNode(type);
			}
		}

		return 1;
	}

	int calcStaticAnaJacobian_GRM(unsigned int secIdx) {

		Indexer idxr(_disc);
		// inlet and bulk jacobian
		_convDispOp.calcStaticAnaJacobian(_globalJac, _jacInlet, idxr.offsetC());

		// particle jacobian (without isotherm, which is handled in residualKernel)
		for (int colNode = 0; colNode < _disc.nPoints; colNode++) {
			for (int type = 0; type < _disc.nParType; type++) {

				// Prepare parameters
				const active* const parDiff = getSectionDependentSlice(_parDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;

				// Ordering of particle surface diffusion:
				// bnd0comp0, bnd0comp1, bnd0comp2, bnd1comp0, bnd1comp1, bnd1comp2
				const active* const  parSurfDiff = getSectionDependentSlice(_parSurfDiffusion, _disc.strideBound[_disc.nParType], secIdx) + _disc.nBoundBeforeType[type];

				double* invBetaP = new double[_disc.nComp];
				for (int comp = 0; comp < _disc.nComp; comp++) {
					invBetaP[comp] = (1.0 - static_cast<double>(_parPorosity[type])) / (static_cast<double>(_poreAccessFactor[_disc.nComp * type + comp]) * static_cast<double>(_parPorosity[type]));
				}

				calcStaticAnaParticleDispJacobian(type, colNode, parDiff, parSurfDiff, invBetaP);
			}
		}

		calcFluxJacobians(secIdx);

		return _globalJac.isCompressed(); // check if the jacobian estimation fits the pattern
	}

};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_GENERALRATEMODELDG_HPP_
