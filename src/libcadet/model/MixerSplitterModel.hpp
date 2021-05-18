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
 * Defines the outlet model.
 */

#ifndef LIBCADET_MIXERSPLITTERMODEL_HPP_
#define LIBCADET_MIXERSPLITTERMODEL_HPP_

#include "cadet/SolutionExporter.hpp"

#include "model/UnitOperation.hpp"
#include "AutoDiff.hpp"
#include "ParamIdUtil.hpp"
#include "model/ModelUtils.hpp"

#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cadet
{

namespace model
{

/**
 * @brief MixerSplitter model
 * @details This unit operation is for recombination of streams
 */
class MixerSplitterModel : public IUnitOperation
{
public:

	MixerSplitterModel(UnitOpIdx unitOpIdx);
	virtual ~MixerSplitterModel() CADET_NOEXCEPT;

	virtual unsigned int numDofs() const CADET_NOEXCEPT;
	virtual unsigned int numPureDofs() const CADET_NOEXCEPT;
	virtual bool usesAD() const CADET_NOEXCEPT;
	virtual unsigned int requiredADdirs() const CADET_NOEXCEPT;

	virtual UnitOpIdx unitOperationId() const CADET_NOEXCEPT { return _unitOpIdx; }
	virtual unsigned int numComponents() const CADET_NOEXCEPT { return _nComp; }
	virtual void setFlowRates(active const* in, active const* out) CADET_NOEXCEPT { }
	virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
	virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
	virtual bool canAccumulate() const CADET_NOEXCEPT { return false; }

	static const char* identifier() { return "MIXER_SPLITTER"; }
	virtual const char* unitOperationName() const CADET_NOEXCEPT { return "MIXER_SPLITTER"; }

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, IConfigHelper& helper);
	virtual bool configure(IParameterProvider& paramProvider);
	virtual void notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac);
	
	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const;
	virtual bool hasParameter(const ParameterId& pId) const;
	virtual double getParameterDouble(const ParameterId& pId) const;

	virtual bool setParameter(const ParameterId& pId, int value);
	virtual bool setParameter(const ParameterId& pId, double value);
	virtual bool setParameter(const ParameterId& pId, bool value);

	virtual bool setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue);
	virtual void setSensitiveParameterValue(const ParameterId& id, double value);

	virtual void clearSensParams();
	virtual unsigned int numSensParams() const;

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


	// linearSolve and assembleAndPrepareDAEJacobian are null operations since there are only inlet DOFs, which are treated by ModelSystem
	virtual int linearSolve(double t, double alpha, double tol, double* const rhs, double const* const weight,
		const ConstSimulationState& simState);

	virtual void prepareADvectors(const AdJacobianParams& adJac) const;

	virtual void applyInitialCondition(const SimulationState& simState) const;
	virtual void readInitialCondition(IParameterProvider& paramProvider);

	virtual void consistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem) { }
	virtual void consistentInitialTimeDerivative(const SimulationTime& simTime, double const* vecStateY, double* const vecStateYdot, util::ThreadLocalStorage& threadLocalMem) { }

	virtual void consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
		std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem);
	virtual void initializeSensitivityStates(const std::vector<double*>& vecSensY) const;

	virtual void leanConsistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem) { }
	virtual void leanConsistentInitialTimeDerivative(double t, double const* const vecStateY, double* const vecStateYdot, double* const res, util::ThreadLocalStorage& threadLocalMem) { }

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

	virtual unsigned int threadLocalMemorySize() const CADET_NOEXCEPT { return 0; }

#ifdef CADET_BENCHMARK_MODE
	virtual std::vector<double> benchmarkTimings() const { return std::vector<double>(0); }
	virtual char const* const* benchmarkDescriptions() const { return nullptr; }
#endif

protected:

	UnitOpIdx _unitOpIdx; //!< Unit operation index
	unsigned int _nComp; //!< Number of components

	class Exporter : public ISolutionExporter
	{
	public:

		Exporter(unsigned int nComp, double const* data) : _data(data), _nComp(nComp) { }

		virtual bool hasParticleFlux() const CADET_NOEXCEPT { return false; }
		virtual bool hasParticleMobilePhase() const CADET_NOEXCEPT { return false; }
		virtual bool hasSolidPhase() const CADET_NOEXCEPT { return false; }
		virtual bool hasVolume() const CADET_NOEXCEPT { return false; }

		virtual unsigned int numComponents() const CADET_NOEXCEPT { return _nComp; }
		virtual unsigned int numAxialCells() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numRadialCells() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numParticleTypes() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numParticleShells(unsigned int parType) const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numBoundStates(unsigned int parType) const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numBulkDofs() const CADET_NOEXCEPT { return _nComp; }
		virtual unsigned int numParticleMobilePhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numSolidPhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numFluxDofs() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numVolumeDofs() const CADET_NOEXCEPT { return 0; }

		virtual double const* concentration() const { return _data; }
		virtual double const* flux() const { return nullptr; }
		virtual double const* particleMobilePhase(unsigned int parType) const { return nullptr; }
		virtual double const* solidPhase(unsigned int parType) const { return nullptr; }
		virtual double const* volume() const { return nullptr; }
		virtual double const* inlet(unsigned int port, unsigned int& stride) const
		{
			stride = 1;
			return _data;
		}
		virtual double const* outlet(unsigned int port, unsigned int& stride) const
		{
			stride = 1;
			return _data + _nComp;
		}

		virtual StateOrdering const* concentrationOrdering(unsigned int& len) const
		{
			len = _concentrationOrdering.size();
			return _concentrationOrdering.data();
		}

		virtual StateOrdering const* fluxOrdering(unsigned int& len) const
		{
			len = 0;
			return nullptr;
		}

		virtual StateOrdering const* mobilePhaseOrdering(unsigned int& len) const
		{
			len = 0;
			return nullptr;
		}

		virtual StateOrdering const* solidPhaseOrdering(unsigned int& len) const
		{
			len = 0;
			return nullptr;
		}

		virtual unsigned int bulkMobilePhaseStride() const { return _nComp; }
		virtual unsigned int particleMobilePhaseStride(unsigned int parType) const { return 0; }
		virtual unsigned int solidPhaseStride(unsigned int parType) const { return 0; }

		virtual void axialCoordinates(double* coords) const { }
		virtual void radialCoordinates(double* coords) const { }
		virtual void particleCoordinates(unsigned int parType, double* coords) const { }

	protected:
		double const* const _data;
		unsigned int _nComp;

		const std::array<StateOrdering, 1> _concentrationOrdering = { { StateOrdering::Component } };
	};
};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_MIXERSPLITTERMODEL_HPP_
