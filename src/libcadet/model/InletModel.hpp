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
 * Defines the inlet model which encapsulates an inlet profile as unit operation.
 */

#ifndef LIBCADET_INLETMODEL_HPP_
#define LIBCADET_INLETMODEL_HPP_

#include "cadet/SolutionExporter.hpp"

#include "model/UnitOperation.hpp"
#include "AutoDiff.hpp"
#include "ParamIdUtil.hpp"
#include "model/ModelUtils.hpp"

#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>

namespace cadet
{

class IInletProfile;

namespace model
{

/**
 * @brief Inlet model which encapsulates an IInletProfile as unit operation
 * @details This unit operation model does not possess DOFs. Its values (inlet profile)
 *          is directly injected into other models' DOFs.
 */
class InletModel : public IUnitOperation
{
public:

	InletModel(UnitOpIdx unitOpIdx);
	virtual ~InletModel() CADET_NOEXCEPT;

	virtual unsigned int numDofs() const CADET_NOEXCEPT;
	virtual unsigned int numPureDofs() const CADET_NOEXCEPT;
	virtual bool usesAD() const CADET_NOEXCEPT;
	virtual unsigned int requiredADdirs() const CADET_NOEXCEPT;

	virtual UnitOpIdx unitOperationId() const CADET_NOEXCEPT { return _unitOpIdx; }
	virtual unsigned int numComponents() const CADET_NOEXCEPT { return _nComp; }
	virtual void setFlowRates(active const* in, active const* out) CADET_NOEXCEPT { }
	virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 0; }
	virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
	virtual bool canAccumulate() const CADET_NOEXCEPT { return true; }

	static const char* identifier() { return "INLET"; }
	virtual const char* unitOperationName() const CADET_NOEXCEPT { return identifier(); }

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper);
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

	// linearSolve is a null operation (the result is I^-1 *rhs -> rhs) since the Jacobian is an identity matrix
	virtual int linearSolve(double t, double alpha, double tol, double* const rhs, double const* const weight,
		const ConstSimulationState& simState) { return 0; }

	virtual void prepareADvectors(const AdJacobianParams& adJac) const;

	virtual void applyInitialCondition(const SimulationState& simState) const;
	virtual void readInitialCondition(IParameterProvider& paramProvider);

	virtual void consistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem);
	virtual void consistentInitialTimeDerivative(const SimulationTime& simTime, double const* vecStateY, double* const vecStateYdot, util::ThreadLocalStorage& threadLocalMem);

	virtual void consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
		std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem);
	virtual void initializeSensitivityStates(const std::vector<double*>& vecSensY) const;

	virtual void leanConsistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem);
	virtual void leanConsistentInitialTimeDerivative(double t, double const* const vecStateY, double* const vecStateYdot, double* const res, util::ThreadLocalStorage& threadLocalMem);

	virtual void leanConsistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
		std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem);

	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size) { }

	virtual void multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret);
	virtual void multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret);

	virtual bool hasInlet() const CADET_NOEXCEPT { return false; }
	virtual bool hasOutlet() const CADET_NOEXCEPT { return true; }

	virtual unsigned int localOutletComponentIndex(unsigned int port) const CADET_NOEXCEPT { return 0; }
	virtual unsigned int localOutletComponentStride(unsigned int port) const CADET_NOEXCEPT { return 1; }
	virtual unsigned int localInletComponentIndex(unsigned int port) const CADET_NOEXCEPT { return 0; }
	virtual unsigned int localInletComponentStride(unsigned int port) const CADET_NOEXCEPT { return 0; }

	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections);

	virtual void expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut) { }

	virtual unsigned int threadLocalMemorySize() const CADET_NOEXCEPT { return 0; }

#ifdef CADET_BENCHMARK_MODE
	virtual std::vector<double> benchmarkTimings() const { return std::vector<double>(0); }
	virtual char const* const* benchmarkDescriptions() const { return nullptr; }
#endif

protected:

	template <typename T> T const* moveInletValues(double const* const rawValues, double t, unsigned int secIdx) const;

	template <typename T> T const* getData() const;

	template <typename ResidualType, typename ParamType>
	int residualImpl(double t, unsigned int secIdx, const ConstSimulationState& simState, ResidualType* const res, util::ThreadLocalStorage& threadLocalMem);

	UnitOpIdx _unitOpIdx; //!< Unit operation index
	unsigned int _nComp; //!< Number of components
	std::vector<double> _tempState; // Temporary storage for the state vector

	IInletProfile* _inlet; //!< Inlet profile (owned by library user)

	double* _inletConcentrationsRaw; //!< Provides memory for getting the inlet concentrations from the inlet profile
	double* _inletDerivatives; //!< Points to memory used for retrieving the derivatives of the inlet profile (do not delete since the memory is owned by _inletConcentrationsRaw)
	active* _inletConcentrations; //!< Provides memory for the inlet concentrations

	std::unordered_map<ParameterId, std::tuple<unsigned int, double>> _sensParamsInlet; //!< Maps an inlet parameter to its AD direction and derivative value

	class Exporter : public ISolutionExporter
	{
	public:

		Exporter(unsigned int nComp, double const* data) : _data(data), _nComp(nComp) { }

		virtual bool hasParticleFlux() const CADET_NOEXCEPT { return false; }
		virtual bool hasParticleMobilePhase() const CADET_NOEXCEPT { return false; }
		virtual bool hasSolidPhase() const CADET_NOEXCEPT { return false; }
		virtual bool hasVolume() const CADET_NOEXCEPT { return false; }

		virtual unsigned int numComponents() const CADET_NOEXCEPT { return _nComp; }
		virtual unsigned int numPrimaryCoordinates() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numSecondaryCoordinates() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numParticleTypes() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numParticleShells(unsigned int parType) const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numBoundStates(unsigned int parType) const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numMobilePhaseDofs() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numParticleMobilePhaseDofs() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numParticleMobilePhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numSolidPhaseDofs() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numSolidPhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numParticleFluxDofs() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numVolumeDofs() const CADET_NOEXCEPT { return 0; }

		virtual int writeMobilePhase(double* buffer) const { return 0; }
		virtual int writeSolidPhase(double* buffer) const { return 0; }
		virtual int writeParticleMobilePhase(double* buffer) const { return 0; }
		virtual int writeSolidPhase(unsigned int parType, double* buffer) const { return 0; }
		virtual int writeParticleMobilePhase(unsigned int parType, double* buffer) const { return 0; }
		virtual int writeParticleFlux(double* buffer) const { return 0; }
		virtual int writeParticleFlux(unsigned int parType, double* buffer) const { return 0; }
		virtual int writeVolume(double* buffer) const { return 0; }
		virtual int writeInlet(unsigned int port, double* buffer) const;
		virtual int writeInlet(double* buffer) const;
		virtual int writeOutlet(unsigned int port, double* buffer) const;
		virtual int writeOutlet(double* buffer) const;

		virtual int writePrimaryCoordinates(double* coords) const { return 0; }
		virtual int writeSecondaryCoordinates(double* coords) const { return 0; }
		virtual int writeParticleCoordinates(unsigned int parType, double* coords) const { return 0; }

	protected:
		double const* const _data;
		unsigned int _nComp;
	};
};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_INLETMODEL_HPP_
