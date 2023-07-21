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
* Defines the CSTR model as a unit operation.
*/

#ifndef LIBCADET_CSTR_HPP_
#define LIBCADET_CSTR_HPP_

#include "model/UnitOperationBase.hpp"
#include "cadet/SolutionExporter.hpp"
#include "AutoDiff.hpp"
#include "linalg/DenseMatrix.hpp"
#include "model/ModelUtils.hpp"
#include "Memory.hpp"

#include <array>
#include <vector>

namespace cadet
{

namespace model
{

/**
 * @brief Continuous stirred tank (reactor) model
 * @details This is a simple CSTR model with variable volume using the ``well mixed assumption''.
 * @f[\begin{align}
	\frac{\mathrm{d}}{\mathrm{d} t}\left( V \left[ c_i + \frac{1}{\beta} \sum_{j=1}^{N_{\text{bnd},i}} q_{i,j} \right] \right) &= F_{\text{in}} c_{\text{in},i} - F_{\text{out}} c_i \\
	a \frac{\partial q_{i,j}}{\partial t} &= f_{\text{iso},i,j}(c, q) \\
	\frac{\partial V}{\partial t} &= F_{\text{in}} - F_{\text{out}} - F_{\text{filter}}
\end{align} @f]
 * The model can be used as a plain stir tank without any binding states.
 */
class CSTRModel : public UnitOperationBase
{
public:

	CSTRModel(UnitOpIdx unitOpIdx);
	virtual ~CSTRModel() CADET_NOEXCEPT;

	virtual unsigned int numDofs() const CADET_NOEXCEPT;
	virtual unsigned int numPureDofs() const CADET_NOEXCEPT;
	virtual bool usesAD() const CADET_NOEXCEPT;
	virtual unsigned int requiredADdirs() const CADET_NOEXCEPT;

	virtual UnitOpIdx unitOperationId() const CADET_NOEXCEPT { return _unitOpIdx; }
	virtual unsigned int numComponents() const CADET_NOEXCEPT { return _nComp; }
	virtual void setFlowRates(active const* in, active const* out) CADET_NOEXCEPT;
	virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
	virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
	virtual bool canAccumulate() const CADET_NOEXCEPT { return true; }

	static const char* identifier() { return "CSTR"; }
	virtual const char* unitOperationName() const CADET_NOEXCEPT { return identifier(); }

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper);
	virtual bool configure(IParameterProvider& paramProvider);
	virtual void notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac);

	virtual void useAnalyticJacobian(const bool analyticJac);

	virtual void reportSolution(ISolutionRecorder& recorder, double const* const solution) const;
	virtual void reportSolutionStructure(ISolutionRecorder& recorder) const;

	virtual int residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem);
	virtual int residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem, bool updateJacobian, bool paramSensitivity);

	virtual int residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem);
	virtual int residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem);
	virtual int residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem);

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

	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size) { }

	virtual void multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret);
	virtual void multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret);

	virtual bool hasInlet() const CADET_NOEXCEPT { return true; }
	virtual bool hasOutlet() const CADET_NOEXCEPT { return true; }

	virtual unsigned int localOutletComponentIndex(unsigned int port) const CADET_NOEXCEPT { return _nComp; }
	virtual unsigned int localOutletComponentStride(unsigned int port) const CADET_NOEXCEPT { return 1; }
	virtual unsigned int localInletComponentIndex(unsigned int port) const CADET_NOEXCEPT { return 0; }
	virtual unsigned int localInletComponentStride(unsigned int port) const CADET_NOEXCEPT { return 1; }

	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections);

	virtual void expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut) { }

	virtual unsigned int threadLocalMemorySize() const CADET_NOEXCEPT;

#ifdef CADET_BENCHMARK_MODE
	virtual std::vector<double> benchmarkTimings() const { return std::vector<double>(0); }
	virtual char const* const* benchmarkDescriptions() const { return nullptr; }
#endif

	inline const std::vector<active>& flowRateFilter() const { return _flowRateFilter; }
	inline std::vector<active>& flowRateFilter() { return _flowRateFilter; }
	inline void flowRateFilter(const std::vector<active>& frf) { _flowRateFilter = frf; }
	inline void setFlowRates(double in, double out) CADET_NOEXCEPT { _flowRateIn = in; _flowRateOut = out; }

protected:

	template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	int residualImpl(double t, unsigned int secIdx, StateType const* const y, double const* const yDot, ResidualType* const res, LinearBufferAllocator threadLocalMem);

	template <typename MatrixType>
	void addTimeDerivativeJacobian(double t, double alpha, const ConstSimulationState& simState, MatrixType& mat);

	void extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset);
#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
	void checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const;
#endif

	unsigned int _nComp; //!< Number of components
	unsigned int _nParType; //!< Number of particle types
	unsigned int* _nBound; //!< Array with number of bound states for each component in each particle type
	unsigned int* _boundOffset; //!< Array with offset to the first bound state of each component in the solid phase for each particle type
	unsigned int* _strideBound; //!< Total number of bound states per particle type
	unsigned int* _offsetParType; //!< Offset to bound states of a particle type in solid phase
	unsigned int _totalBound; //!< Total number of all bound states in all particle types

	active _porosity; //!< Porosity \f$ \varepsilon \f$
	active _flowRateIn; //!< Volumetric flow rate of incoming stream
	active _flowRateOut; //!< Volumetric flow rate of drawn outgoing stream
	active _curFlowRateFilter; //!< Current volumetric flow rate of liquid outtake stream for this section
	std::vector<active> _flowRateFilter; //!< Volumetric flow rate of liquid outtake stream
	std::vector<active> _parTypeVolFrac; //!< Volume fraction of each particle type

	bool _analyticJac; //!< Flag that determines whether analytic or AD Jacobian is used
	linalg::DenseMatrix _jac; //!< Jacobian
	linalg::DenseMatrix _jacFact; //!< Factorized Jacobian
	bool _factorizeJac; //!< Flag that tracks whether the Jacobian needs to be factorized

	std::vector<active> _initConditions; //!< Initial conditions, ordering: Liquid phase concentration, solid phase concentration, volume
	std::vector<double> _initConditionsDot; //!< Initial conditions for time derivative

	IDynamicReactionModel* _dynReactionBulk; //!< Dynamic reactions in the bulk volume

	class Exporter : public ISolutionExporter
	{
	public:

		Exporter(unsigned int nComp, unsigned int nParType, unsigned int const* nBound, unsigned int const* strideBound, unsigned int const* boundOffset, unsigned int totalBound, double const* data)
			: _data(data), _nComp(nComp), _nParType(nParType), _nBound(nBound), _strideBound(strideBound), _boundOffset(boundOffset), _totalBound(totalBound) { }

		virtual bool hasParticleFlux() const CADET_NOEXCEPT { return false; }
		virtual bool hasParticleMobilePhase() const CADET_NOEXCEPT { return false; }
		virtual bool hasSolidPhase() const CADET_NOEXCEPT { return _totalBound > 0; }
		virtual bool hasVolume() const CADET_NOEXCEPT { return true; }

		virtual unsigned int numComponents() const CADET_NOEXCEPT { return _nComp; }
		virtual unsigned int numPrimaryCoordinates() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numSecondaryCoordinates() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numParticleTypes() const CADET_NOEXCEPT { return _nParType; }
		virtual unsigned int numParticleShells(unsigned int parType) const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numBoundStates(unsigned int parType) const CADET_NOEXCEPT { return _strideBound[parType]; }
		virtual unsigned int numMobilePhaseDofs() const CADET_NOEXCEPT { return _nComp; }
		virtual unsigned int numParticleMobilePhaseDofs() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numParticleMobilePhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numSolidPhaseDofs() const CADET_NOEXCEPT;
		virtual unsigned int numSolidPhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return _strideBound[parType]; }
		virtual unsigned int numParticleFluxDofs() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numVolumeDofs() const CADET_NOEXCEPT { return 1; }

		virtual int writeMobilePhase(double* buffer) const;
		virtual int writeSolidPhase(double* buffer) const;
		virtual int writeSolidPhase(unsigned int parType, double* buffer) const;
		virtual int writeParticleMobilePhase(double* buffer) const { return 0; }
		virtual int writeParticleMobilePhase(unsigned int parType, double* buffer) const { return 0; }
		virtual int writeParticleFlux(double* buffer) const { return 0; }
		virtual int writeParticleFlux(unsigned int parType, double* buffer) const { return 0; }
		virtual int writeVolume(double* buffer) const;
		virtual int writeInlet(unsigned int port, double* buffer) const;
		virtual int writeInlet(double* buffer) const;
		virtual int writeOutlet(unsigned int port, double* buffer) const;
		virtual int writeOutlet(double* buffer) const;

		virtual double const* solidPhase(unsigned int parType) const { return _data + 2 * _nComp; }

		virtual int writePrimaryCoordinates(double* coords) const { return 0; }
		virtual int writeSecondaryCoordinates(double* coords) const { return 0; }
		virtual int writeParticleCoordinates(unsigned int parType, double* coords) const { return 0; }

	protected:
		double const* const _data;
		unsigned int _nComp;
		unsigned int _nParType;
		unsigned int const* _nBound;
		unsigned int const* _strideBound;
		unsigned int const* _boundOffset;
		unsigned int _totalBound;
	};
};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_CSTR_HPP_
