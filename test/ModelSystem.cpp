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

#include <catch.hpp>
#include "cadet/cadet.hpp"

#define CADET_LOGGING_DISABLE
#include "Logging.hpp"

#include "ModelBuilderImpl.hpp"
#include "model/ModelSystemImpl.hpp"
#include "SimulationTypes.hpp"
#include "ParallelSupport.hpp"
#include "JsonTestModels.hpp"
#include "JacobianHelper.hpp"
#include "ColumnTests.hpp"
#include "Utils.hpp"
#include "Dummies.hpp"
#include "model/UnitOperation.hpp"

#include <limits>
#include <vector>
#include <set>

namespace
{

	class DummyUnitOperation : public cadet::IUnitOperation
	{
	public:
		DummyUnitOperation(cadet::UnitOpIdx unitOpIdx) : _unitOpIdx(unitOpIdx), _inFlow(0), _outFlow(0) { }
		DummyUnitOperation(cadet::UnitOpIdx unitOpIdx, unsigned int nComp, unsigned int nInletPorts, unsigned int nOutletPorts, bool canAccumulate)
			: _unitOpIdx(unitOpIdx), _nComp(nComp), _nInletPorts(nInletPorts), _nOutletPorts(nOutletPorts), _canAccumulate(canAccumulate),
			_inFlow(nInletPorts, 0.0), _outFlow(nOutletPorts, 0.0)
		{ }

		// Default copy and move mechanisms
		DummyUnitOperation(const DummyUnitOperation& cpy) = default;
		DummyUnitOperation(DummyUnitOperation&& cpy) CADET_NOEXCEPT = default;

		inline DummyUnitOperation& operator=(const DummyUnitOperation& cpy) = default;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
		inline DummyUnitOperation& operator=(DummyUnitOperation&& cpy) CADET_NOEXCEPT = default;
#else
		inline DummyUnitOperation& operator=(DummyUnitOperation&& cpy) = default;
#endif

		inline void setup(cadet::UnitOpIdx unitOpIdx, unsigned int nComp, unsigned int nInletPorts, unsigned int nOutletPorts, bool canAccumulate)
		{
			_unitOpIdx = unitOpIdx;
			_nComp = nComp; 
			_nInletPorts = nInletPorts;
			_nOutletPorts = nOutletPorts;
			_canAccumulate = canAccumulate;

			_inFlow.resize(nInletPorts, 0.0); 
			_outFlow.resize(nOutletPorts, 0.0);
		}

		virtual cadet::UnitOpIdx unitOperationId() const CADET_NOEXCEPT { return _unitOpIdx; }
		virtual const char* unitOperationName() const CADET_NOEXCEPT { return "DUMMY"; }

		virtual bool setParameter(const cadet::ParameterId& pId, int value) { return false; }
		virtual bool setParameter(const cadet::ParameterId& pId, double value) { return false; }
		virtual bool setParameter(const cadet::ParameterId& pId, bool value) { return false; }
		virtual bool hasParameter(const cadet::ParameterId& pId) const { return false; }
		virtual std::unordered_map<cadet::ParameterId, double> getAllParameterValues() const { return std::unordered_map<cadet::ParameterId, double>(); }
		virtual double getParameterDouble(const cadet::ParameterId& pId) const { return 0.0; }
		virtual void useAnalyticJacobian(const bool analyticJac) { }

#ifdef CADET_BENCHMARK_MODE
		virtual std::vector<double> benchmarkTimings() const { return std::vector<double>(); }
		virtual char const* const* benchmarkDescriptions() const { return nullptr; }
#endif

		virtual unsigned int numDofs() const CADET_NOEXCEPT { return (_nInletPorts + _nOutletPorts) * _nComp; }
		virtual unsigned int numPureDofs() const CADET_NOEXCEPT { return _nOutletPorts * _nComp; }

		virtual bool usesAD() const CADET_NOEXCEPT { return false; }
		virtual unsigned int requiredADdirs() const CADET_NOEXCEPT { return 0; }

		virtual bool configureModelDiscretization(cadet::IParameterProvider& paramProvider, const cadet::IConfigHelper& helper) { return true; }
		virtual bool configure(cadet::IParameterProvider& paramProvider) { return true; }

		virtual void reportSolution(cadet::ISolutionRecorder& reporter, double const* const solution) const { }
		virtual void reportSolutionStructure(cadet::ISolutionRecorder& reporter) const { }

		virtual bool setSensitiveParameter(const cadet::ParameterId& pId, unsigned int adDirection, double adValue) { return false; }
		virtual void setSensitiveParameterValue(const cadet::ParameterId& id, double value) { }
		virtual void clearSensParams() { }
		virtual unsigned int numSensParams() const { return 0; }

		virtual void notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const cadet::ConstSimulationState& simState, const cadet::AdJacobianParams& adJac) { }
		virtual void applyInitialCondition(const cadet::SimulationState& simState) const { }
		virtual void readInitialCondition(cadet::IParameterProvider& paramProvider) { }

		virtual int residual(const cadet::SimulationTime& simTime, const cadet::ConstSimulationState& simState, double* const res, cadet::util::ThreadLocalStorage& tls)
		{
			std::copy(simState.vecStateY, simState.vecStateY + numDofs(), res);
			return 0;
		}

		virtual int residualWithJacobian(const cadet::SimulationTime& simTime, const cadet::ConstSimulationState& simState, double* const res,
			const cadet::AdJacobianParams& adJac, cadet::util::ThreadLocalStorage& tls)
		{
			std::copy(simState.vecStateY, simState.vecStateY + numDofs(), res);
			return 0;
		}

		virtual int linearSolve(double t, double alpha, double tol, double* const rhs, double const* const weight,
			const cadet::ConstSimulationState& simState)
		{
			return 0;
		}

		virtual void prepareADvectors(const cadet::AdJacobianParams& adJac) const { }
		virtual void initializeSensitivityStates(const std::vector<double*>& vecSensY) const { }

		virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections) { }
		virtual void expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut) { }

		virtual unsigned int numComponents() const CADET_NOEXCEPT { return _nComp; }
		virtual bool hasInlet() const CADET_NOEXCEPT { return _nInletPorts > 0; }
		virtual bool hasOutlet() const CADET_NOEXCEPT { return _nOutletPorts > 0; }
		virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return _nInletPorts; }
		virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return _nOutletPorts; }

		virtual unsigned int localOutletComponentIndex(unsigned int port) const CADET_NOEXCEPT { return (_nInletPorts + port) * _nComp; }
		virtual unsigned int localOutletComponentStride(unsigned int port) const CADET_NOEXCEPT { return 1u; }
		virtual unsigned int localInletComponentIndex(unsigned int port) const CADET_NOEXCEPT { return port * _nComp; }
		virtual unsigned int localInletComponentStride(unsigned int port) const CADET_NOEXCEPT { return 1u; }

		virtual int residualSensFwdAdOnly(const cadet::SimulationTime& simTime, const cadet::ConstSimulationState& simState, cadet::active* const adRes, cadet::util::ThreadLocalStorage& tls) { return 0; }

		virtual int residualSensFwdCombine(const cadet::SimulationTime& simTime, const cadet::ConstSimulationState& simState, 
			const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS, cadet::active const* adRes, 
			double* const tmp1, double* const tmp2, double* const tmp3) { return 0; }

		virtual int residualSensFwdWithJacobian(const cadet::SimulationTime& simTime, const cadet::ConstSimulationState& simState, const cadet::AdJacobianParams& adJac, cadet::util::ThreadLocalStorage& tls) { return 0; }

		virtual void consistentInitialState(const cadet::SimulationTime& simTime, double* const vecStateY, const cadet::AdJacobianParams& adJac, double errorTol, cadet::util::ThreadLocalStorage& tls)
		{
			std::fill_n(vecStateY + _nInletPorts * _nComp, _nOutletPorts * _nComp, 0.0);
		}

		virtual void consistentInitialTimeDerivative(const cadet::SimulationTime& simTime, double const* vecStateY, double* const vecStateYdot, cadet::util::ThreadLocalStorage& tls)
		{
			std::fill_n(vecStateYdot + _nInletPorts * _nComp, _nOutletPorts * _nComp, 0.0);
		}

		virtual void leanConsistentInitialState(const cadet::SimulationTime& simTime, double* const vecStateY, const cadet::AdJacobianParams& adJac, double errorTol, cadet::util::ThreadLocalStorage& tls) { }
		virtual void leanConsistentInitialTimeDerivative(double t, double const* const vecStateY, double* const vecStateYdot, double* const res, cadet::util::ThreadLocalStorage& tls) { }

		virtual void consistentInitialSensitivity(const cadet::SimulationTime& simTime, const cadet::ConstSimulationState& simState, std::vector<double*>& vecSensY,
			std::vector<double*>& vecSensYdot, cadet::active const* const adRes, cadet::util::ThreadLocalStorage& tls) { }
		virtual void leanConsistentInitialSensitivity(const cadet::SimulationTime& simTime, const cadet::ConstSimulationState& simState, std::vector<double*>& vecSensY,
			std::vector<double*>& vecSensYdot, cadet::active const* const adRes, cadet::util::ThreadLocalStorage& tls) { }

		virtual void setExternalFunctions(cadet::IExternalFunction** extFuns, unsigned int size) { }

		virtual void setFlowRates(cadet::active const* in, cadet::active const* out) CADET_NOEXCEPT
		{
			std::copy(in, in + _nInletPorts, _inFlow.begin());
			std::copy(out, out + _nOutletPorts, _outFlow.begin());
		}

		virtual bool canAccumulate() const CADET_NOEXCEPT { return _canAccumulate; }

		virtual void multiplyWithJacobian(const cadet::SimulationTime& simTime, const cadet::ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret)
		{
			// dF / dy = I (identity matrix)
			for (unsigned int i = 0; i < numDofs(); ++i)
			{
				ret[i] = alpha * yS[i] + beta * ret[i];
			}
		}

		virtual void multiplyWithDerivativeJacobian(const cadet::SimulationTime& simTime, const cadet::ConstSimulationState& simState, double const* sDot, double* ret)
		{
			std::fill_n(ret, numDofs(), 0.0);
		}

		virtual unsigned int threadLocalMemorySize() const CADET_NOEXCEPT { return 0; }

		inline const std::vector<cadet::active>& inFlow() const CADET_NOEXCEPT { return _inFlow; }
		inline const std::vector<cadet::active>& outFlow() const CADET_NOEXCEPT { return _outFlow; }

	protected:
		cadet::UnitOpIdx _unitOpIdx;
		unsigned int _nComp;
		unsigned int _nInletPorts;
		unsigned int _nOutletPorts;
		bool _canAccumulate;
		std::vector<cadet::active> _inFlow;
		std::vector<cadet::active> _outFlow;
	};

	cadet::JsonParameterProvider createSystemConfig(const std::vector<double>& connections)
	{
		cadet::JsonParameterProvider jpp(R"json({
			"solver":
			{
				"MAX_KRYLOV": 0,
				"GS_TYPE": 1,
				"MAX_RESTARTS": 10,
				"SCHUR_SAFETY": 1e-8
			},
			"connections":
			{
				"NSWITCHES": 1,
				"CONNECTIONS_INCLUDE_PORTS": true,
				"switch_000":
				{
					"SECTION": 0
				}
			}
		})json");

		jpp.pushScope("connections");
		jpp.pushScope("switch_000");

		jpp.set("CONNECTIONS", connections);

		jpp.popScope();
		jpp.popScope();

		return jpp;
	}


	std::vector<unsigned int> calculateDofOffsets(const cadet::model::ModelSystem& sys)
	{
		std::vector<unsigned int> dofOffsets(sys.numModels() + 1, 0u);
		for (unsigned int i = 0; i < sys.numModels(); ++i)
			dofOffsets[i+1] = dofOffsets[i] + sys.getUnitOperationModel(i)->numDofs();

		return dofOffsets;
	}

	std::vector<double> calculateJacobian(cadet::model::ModelSystem& sys)
	{
		const unsigned int nDof = sys.numDofs();
		std::vector<double> jac(nDof * nDof, 0.0);
		std::vector<double> y(nDof, 0.0);

		const cadet::SimulationTime simTime{0.0, 0u};
		const cadet::ConstSimulationState simState{y.data(), y.data()};
		for (unsigned int i = 0; i < nDof; ++i)
		{
			y[i] = 1.0;
			sys.multiplyWithJacobian(simTime, simState, y.data(), 1.0, 0.0, jac.data() + i * nDof);
			y[i] = 0.0;
		}

		return jac;
	}

	void checkDedicatedInletDofConnections(cadet::model::ModelSystem& sys, const std::vector<unsigned int>& dofOffsets, const std::vector<double>& jac)
	{
		const unsigned int nDof = sys.numDofs();

		// Check that the inlet DOFs are taken in with a 1.0 in each unit operation
		// That is, the Jacobian of the unit operation should contain the identity matrix at the inlet DOFs
		for (unsigned int i = 0; i < sys.numModels(); ++i)
		{
			cadet::IUnitOperation const* const m = sys.getUnitOperationModel(i);

			// Jacobian is in column-major format (columns of Jacobian are concatenated to a big array)
			// Move to first column of unit operation and inside this column to the first DOF of the unit operation
			double const* local = jac.data() + dofOffsets[i] * nDof + dofOffsets[i];

			for (unsigned int port = 0; port < m->numInletPorts(); ++port)
			{
				const unsigned int inletOffset = m->localInletComponentIndex(port);
				const unsigned int inletStride = m->localInletComponentStride(port);

				for (unsigned int comp = 0; comp < m->numComponents(); ++comp, local += nDof)
				{
					CAPTURE(i);
					CAPTURE(port);
					CAPTURE(comp);
					CAPTURE(inletOffset + comp * inletStride);
					CAPTURE(dofOffsets[i]);
					CHECK(local[inletOffset + comp * inletStride] == 1.0);

					// This loop moves the Jacobian pointer to the next column
				}
			}
		}

		// Check that the inlet DOFs are fed with a -1.0 by the model system
		// That is, in the same row of the identity matrix (see above),
		// the system Jacobian has a negative identity matrix in the right macro column
		unsigned int nInletDofsUpToNow = 0;
		for (unsigned int i = 0; i < sys.numModels(); ++i)
		{
			cadet::IUnitOperation const* const m = sys.getUnitOperationModel(i);

			// Jacobian is in column-major format (columns of Jacobian are concatenated to a big array)
			// Move to first column behind all unit operations, from there to the column of the current
			// unit operation and inside this column to the first DOF of the current unit operation
			double const* local = jac.data() + (dofOffsets.back() + nInletDofsUpToNow) * nDof + dofOffsets[i];

			for (unsigned int port = 0; port < m->numInletPorts(); ++port)
			{
				const unsigned int inletOffset = m->localInletComponentIndex(port);
				const unsigned int inletStride = m->localInletComponentStride(port);

				for (unsigned int comp = 0; comp < m->numComponents(); ++comp, local += nDof, ++nInletDofsUpToNow)
				{
					CAPTURE(i);
					CAPTURE(port);
					CAPTURE(comp);
					CAPTURE(inletOffset + comp * inletStride);
					CAPTURE(dofOffsets[i]);
					CHECK(local[inletOffset + comp * inletStride] == -1.0);

					// This loop moves the Jacobian pointer to the next column
				}
			}
		}
	}

	void checkRightMacroColumn(cadet::model::ModelSystem& sys, const std::vector<unsigned int>& dofOffsets, const std::vector<double>& jac)
	{
		// The right macro column should only have two entries per column (-1 in the unit operation row, +1 in the coupling DOF row)

		const unsigned int nDof = sys.numDofs();
		double const* local = jac.data() + nDof * dofOffsets.back();
		for (unsigned int c = dofOffsets.back(); c < nDof; ++c, local += nDof)
		{
			unsigned int count = 0;
			for (unsigned int r = 0; r < nDof; ++r)
			{
				if (local[r] != 0.0)
					++count;
			}

			CAPTURE(c);
			CHECK(count == 2);
		}
	}

	bool connectedPreviously(const std::vector<double>& connections, unsigned int curPos, unsigned int uoSource, int portSource, unsigned int uoDest, unsigned int portDest)
	{
		double const* conRow = connections.data();
		for (unsigned int i = 0; i < curPos; ++i, conRow += 7)
		{
			if (static_cast<unsigned int>(conRow[0]) != uoSource)
				continue;

			if (static_cast<unsigned int>(conRow[1]) != uoDest)
				continue;

			if (((conRow[2] < 0.0) || (static_cast<int>(conRow[2]) == portSource)) && ((conRow[3] < 0.0) || (static_cast<unsigned int>(conRow[3]) == portDest)))
				return true;
		}

		return false;
	}

	void checkCouplingRow(cadet::model::ModelSystem& sys, const std::vector<unsigned int>& dofOffsets, const std::vector<double>& jac,
		const std::vector<double>& connections, unsigned int row, unsigned int uoDest, unsigned int portDest, unsigned int compDest,
		std::set<std::pair<unsigned int, unsigned int>>& nonZeros)
	{
		// Find total influx
		double totalInFlow = 0.0;

		double const* conRow = connections.data();
		for (std::size_t i = 0; i < connections.size() / 7; ++i, conRow += 7)
		{
			if (static_cast<unsigned int>(conRow[1]) != uoDest)
				continue;

			if ((conRow[3] < 0.0) || (static_cast<unsigned int>(conRow[3]) == portDest))
			{
				if (!connectedPreviously(connections, i, static_cast<unsigned int>(conRow[0]), static_cast<int>(conRow[2]), uoDest, portDest))
					totalInFlow += conRow[6];
			}
		}

		// Check connection in Jacobian
		const unsigned int nDof = sys.numDofs();
		conRow = connections.data();
		for (std::size_t i = 0; i < connections.size() / 7; ++i, conRow += 7)
		{
			if (static_cast<unsigned int>(conRow[1]) != uoDest)
				continue;

			const unsigned int uoSource = static_cast<unsigned int>(conRow[0]);
			cadet::IUnitOperation const* const m = sys.getUnitOperationModel(uoSource);
			const double jacVal = -conRow[6] / totalInFlow;

			if (conRow[3] < 0.0)
			{
				if (conRow[5] < 0.0)
				{
					// Connect all components of all ports
					const unsigned int pos = m->localOutletComponentIndex(portDest) + m->localOutletComponentStride(portDest) * compDest;
					CAPTURE(uoSource);
					CAPTURE(uoDest);
					CAPTURE(portDest);
					CAPTURE(compDest);
					CHECK(jac[nDof * (dofOffsets[uoSource] + pos) + dofOffsets.back() + row] == jacVal);

					nonZeros.insert(std::make_pair(row, dofOffsets[uoSource] + pos));
				}
				else if (static_cast<unsigned int>(conRow[5]) == compDest)
				{
					// Connect specific component of all ports
					const unsigned int compSource = static_cast<unsigned int>(conRow[4]);
					const unsigned int pos = m->localOutletComponentIndex(portDest) + m->localOutletComponentStride(portDest) * compSource;
					CAPTURE(uoSource);
					CAPTURE(uoDest);
					CAPTURE(portDest);
					CAPTURE(compSource);
					CAPTURE(compDest);
					CHECK(jac[nDof * (dofOffsets[uoSource] + pos) + dofOffsets.back() + row] == jacVal);

					nonZeros.insert(std::make_pair(row, dofOffsets[uoSource] + pos));
				}
			}
			else if (static_cast<unsigned int>(conRow[3]) == portDest)
			{
				const unsigned int portSource = static_cast<unsigned int>(conRow[2]);
				if (conRow[5] < 0.0)
				{
					// Connect all components of specific port
					const unsigned int pos = m->localOutletComponentIndex(portSource) + m->localOutletComponentStride(portSource) * compDest;
					CAPTURE(uoSource);
					CAPTURE(uoDest);
					CAPTURE(portSource);
					CAPTURE(portDest);
					CAPTURE(compDest);
					CHECK(jac[nDof * (dofOffsets[uoSource] + pos) + dofOffsets.back() + row] == jacVal);

					nonZeros.insert(std::make_pair(row, dofOffsets[uoSource] + pos));
				}
				else if (static_cast<unsigned int>(conRow[5]) == compDest)
				{
					// Connect specific component of specific port
					const unsigned int compSource = static_cast<unsigned int>(conRow[4]);
					const unsigned int pos = m->localOutletComponentIndex(portSource) + m->localOutletComponentStride(portSource) * compSource;
					CAPTURE(uoSource);
					CAPTURE(uoDest);
					CAPTURE(portSource);
					CAPTURE(portDest);
					CAPTURE(compSource);
					CAPTURE(compDest);
					CHECK(jac[nDof * (dofOffsets[uoSource] + pos) + dofOffsets.back() + row] == jacVal);

					nonZeros.insert(std::make_pair(row, dofOffsets[uoSource] + pos));
				}
			}
		}
	}

	void checkCouplingDofs(cadet::model::ModelSystem& sys, const std::vector<unsigned int>& dofOffsets, const std::vector<double>& jac, const std::vector<double>& connections)
	{
		const unsigned int nDof = sys.numDofs();

		// Check identity matrix of coupling DOFs in bottom macro row

		// Move to beginning of identitiy matrix
		double const* local = jac.data() + dofOffsets.back() * nDof + dofOffsets.back();
		for (unsigned int i = 0; i < nDof - dofOffsets.back(); ++i, local += nDof + 1)
		{
			CHECK(local[0] == 1.0);

			// This loop advances the Jacobian pointer to the next column and by one additional row
		}		

		// Check remaining entries in bottom macro row (corresponding to outlet DOFs)
		std::set<std::pair<unsigned int, unsigned int>> nonZeros;

		// Iterate over coupling dofs (rows in bottom macro row)
		unsigned int idxCoupling = 0;
		for (unsigned int uoDest = 0; uoDest < sys.numModels(); ++uoDest)
		{
			cadet::IUnitOperation const* const m = sys.getUnitOperationModel(uoDest);

			if (!m->hasInlet())
				continue;

			for (unsigned int portDest = 0; portDest < m->numInletPorts(); ++portDest)
			{
				for (unsigned int compDest = 0; compDest < m->numComponents(); ++compDest, ++idxCoupling)
				{
					checkCouplingRow(sys, dofOffsets, jac, connections, idxCoupling, uoDest, portDest, compDest, nonZeros);
				}
			}
		}

		// Make sure that the remaining entries are all zero
		for (unsigned int col = 0; col < dofOffsets.back(); ++col)
		{
			for (unsigned int row = 0; row < nDof - dofOffsets.back(); ++row)
			{
				if (nonZeros.find(std::make_pair(row, col)) != nonZeros.end())
					continue;

				CHECK(jac[nDof * col + dofOffsets.back() + row] == 0.0);
			}
		}
	}

	void checkCouplingJacobian(const std::vector<unsigned int> sysDescription, const std::vector<double>& connections, const std::vector<double>& inFlow, const std::vector<double>& outFlow)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		cadet::IModelSystem* const cadSys = mb->createSystem();
		REQUIRE(cadSys);
		cadet::model::ModelSystem* const sys = reinterpret_cast<cadet::model::ModelSystem*>(cadSys);

		const std::size_t numUnits = sysDescription.size() / 4;
		unsigned int const* cd = sysDescription.data();
		for (std::size_t i = 0; i < numUnits; ++i, cd += 4)
			sys->addModel(new DummyUnitOperation(i, cd[0], cd[1], cd[2], cd[3]));

		DummyConfigHelper dch;
		cadet::JsonParameterProvider jpp = createSystemConfig(connections);
		REQUIRE(sys->configureModelDiscretization(jpp, dch));
		REQUIRE(sys->configure(jpp));

		// Setup matrices
		const cadet::AdJacobianParams noParams{nullptr, nullptr, 0u};
		sys->notifyDiscontinuousSectionTransition(0.0, 0u, {nullptr, nullptr}, noParams);

		const std::vector<unsigned int> dofOffsets = calculateDofOffsets(*sys);
		const std::vector<double> jac = calculateJacobian(*sys);

		checkDedicatedInletDofConnections(*sys, dofOffsets, jac);
		checkRightMacroColumn(*sys, dofOffsets, jac);
		checkCouplingDofs(*sys, dofOffsets, jac, connections);

		// Check values of setFlowRate() received by unit operation models
		double const* refInFlow = inFlow.data();
		double const* refOutFlow = outFlow.data();
		for (unsigned int model = 0; model < sys->numModels(); ++model)
		{
			DummyUnitOperation const* const m = static_cast<DummyUnitOperation*>(sys->getUnitOperationModel(model));

			const std::vector<cadet::active>& unitInFlow = m->inFlow();
			for (unsigned int port = 0; port < m->numInletPorts(); ++port, ++refInFlow)
			{
				CAPTURE(model);
				CAPTURE(port);
				CHECK(*refInFlow == static_cast<double>(unitInFlow[port]));
			}

			const std::vector<cadet::active>& unitOutFlow = m->outFlow();
			for (unsigned int port = 0; port < m->numOutletPorts(); ++port, ++refOutFlow)
			{
				CAPTURE(model);
				CAPTURE(port);
				CHECK(*refOutFlow == static_cast<double>(unitOutFlow[port]));
			}
		}

		destroyModelBuilder(mb);
	}

}

TEST_CASE("ModelSystem Jacobian AD vs analytic", "[ModelSystem],[Jacobian],[AD]")
{
	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	for (int bindMode = 0; bindMode < 2; ++bindMode)
	{
		const bool isKinetic = bindMode;
		SECTION(isKinetic ? "Kinetic binding" : "Quasi-stationary binding")
		{
			// Use some test case parameters
			cadet::JsonParameterProvider jpp = createLinearBenchmark(isKinetic, false, "GENERAL_RATE_MODEL");

			// Extract section times
			jpp.pushScope("solver");
			jpp.pushScope("sections");

			const std::vector<double> secTimes = jpp.getDoubleArray("SECTION_TIMES");
			std::vector<bool> secCont(secTimes.size() - 2, false);
			if (jpp.exists("SECTION_CONTINUITY")) 
				secCont = jpp.getBoolArray("SECTION_CONTINUITY");

			jpp.popScope();
			jpp.popScope();
			
			// Create and configure ModelSystem
			jpp.pushScope("model");
			cadet::test::column::setNumAxialCells(jpp, 10);
			cadet::IModelSystem* const cadSysAna = mb->createSystem(jpp);
			REQUIRE(cadSysAna);
			cadet::model::ModelSystem* const sysAna = reinterpret_cast<cadet::model::ModelSystem*>(cadSysAna);
			sysAna->setupParallelization(cadet::util::getMaxThreads());

			cadet::IModelSystem* const cadSysAD = mb->createSystem(jpp);
			REQUIRE(cadSysAD);
			cadet::model::ModelSystem* const sysAD = reinterpret_cast<cadet::model::ModelSystem*>(cadSysAD);
			sysAD->setupParallelization(cadet::util::getMaxThreads());

			bool* const secContArray = new bool[secCont.size()];
			std::copy(secCont.begin(), secCont.end(), secContArray);
			sysAna->setSectionTimes(secTimes.data(), secContArray, secTimes.size() - 1);
			sysAD->setSectionTimes(secTimes.data(), secContArray, secTimes.size() - 1);
			delete[] secContArray;

			// Enable AD
			cadet::ad::setDirections(cadet::ad::getMaxDirections());
			for (unsigned int i = 0; i < sysAD->numModels(); ++i)
				sysAD->getUnitOperationModel(i)->useAnalyticJacobian(false);

			cadet::active* adRes = new cadet::active[sysAD->numDofs()];
			cadet::active* adY = new cadet::active[sysAD->numDofs()];

			const cadet::AdJacobianParams adParams{adRes, adY, 0u};
			const cadet::AdJacobianParams noParams{nullptr, nullptr, 0u};

			sysAD->prepareADvectors(adParams);

			// Obtain memory for state, Jacobian multiply direction, Jacobian column
			const unsigned int nDof = sysAD->numDofs();
			std::vector<double> y(nDof, 0.0);
			std::vector<double> yDot(nDof, 0.0);
			std::vector<double> jacDir(nDof, 0.0);
			std::vector<double> jacCol1(nDof, 0.0);
			std::vector<double> jacCol2(nDof, 0.0);

			// Fill state vector with some values
			cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
			cadet::test::util::populate(yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);

			// Setup matrices
			sysAna->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, noParams);
			sysAD->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, adParams);

			// Compute state Jacobian
			const cadet::SimulationTime simTime{0.0, 0u};
			sysAna->residualWithJacobian(simTime, cadet::ConstSimulationState{y.data(), yDot.data()}, jacDir.data(), noParams);
			sysAD->residualWithJacobian(simTime, cadet::ConstSimulationState{y.data(), yDot.data()}, jacDir.data(), adParams);
			std::fill(jacDir.begin(), jacDir.end(), 0.0);

			// Compare Jacobians
			cadet::test::checkJacobianPatternFD(
				[sysAna, &yDot](double const* lDir, double* res) -> void { sysAna->residual(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{lDir, yDot.data()}, res); },
				[sysAD, &y, &yDot](double const* lDir, double* res) -> void { sysAD->multiplyWithJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, lDir, 1.0, 0.0, res); },
				y.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), nDof);

			cadet::test::checkJacobianPatternFD(
				[sysAna, &yDot](double const* lDir, double* res) -> void { sysAna->residual(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{lDir, yDot.data()}, res); },
				[sysAna, &y, &yDot](double const* lDir, double* res) -> void { sysAna->multiplyWithJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, lDir, 1.0, 0.0, res); },
				y.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), nDof);

			cadet::test::compareJacobian(
				[sysAna, &y, &yDot](double const* lDir, double* res) -> void { sysAna->multiplyWithJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, lDir, 1.0, 0.0, res); },
				[sysAD, &y, &yDot](double const* lDir, double* res) -> void { sysAD->multiplyWithJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, lDir, 1.0, 0.0, res); },
				jacDir.data(), jacCol1.data(), jacCol2.data(), nDof);

			delete[] adRes;
			delete[] adY;
		}
	}
	destroyModelBuilder(mb);
}

TEST_CASE("ModelSystem time derivative Jacobian FD vs analytic", "[ModelSystem],[Jacobian],[AD]")
{
	const double h = 5e-4;
	const double absTol = 0.0;
	const double relTol = std::numeric_limits<float>::epsilon() * 100.0;

	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	for (int bindMode = 0; bindMode < 2; ++bindMode)
	{
		const bool isKinetic = bindMode;
		SECTION(isKinetic ? "Kinetic binding" : "Quasi-stationary binding")
		{
			// Use some test case parameters
			cadet::JsonParameterProvider jpp = createLinearBenchmark(isKinetic, false, "GENERAL_RATE_MODEL");

			// Extract section times
			jpp.pushScope("solver");
			jpp.pushScope("sections");

			const std::vector<double> secTimes = jpp.getDoubleArray("SECTION_TIMES");
			std::vector<bool> secCont(secTimes.size() - 2, false);
			if (jpp.exists("SECTION_CONTINUITY")) 
				secCont = jpp.getBoolArray("SECTION_CONTINUITY");

			jpp.popScope();
			jpp.popScope();
			
			// Create and configure ModelSystem
			jpp.pushScope("model");
			cadet::test::column::setNumAxialCells(jpp, 10);
			cadet::IModelSystem* const cadSys = mb->createSystem(jpp);
			REQUIRE(cadSys);
			cadet::model::ModelSystem* const sys = reinterpret_cast<cadet::model::ModelSystem*>(cadSys);
			sys->setupParallelization(cadet::util::getMaxThreads());

			bool* const secContArray = new bool[secCont.size()];
			std::copy(secCont.begin(), secCont.end(), secContArray);
			sys->setSectionTimes(secTimes.data(), secContArray, secTimes.size() - 1);
			delete[] secContArray;

			// Obtain memory for state, Jacobian multiply direction, Jacobian column
			const unsigned int nDof = sys->numDofs();
			std::vector<double> y(nDof, 0.0);
			std::vector<double> yDot(nDof, 0.0);
			std::vector<double> jacDir(nDof, 0.0);
			std::vector<double> jacCol1(nDof, 0.0);
			std::vector<double> jacCol2(nDof, 0.0);

			// Fill state vector with some values
			cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
			cadet::test::util::populate(yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);

			// Setup matrices
			sys->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, cadet::AdJacobianParams{nullptr, nullptr, 0u});

			// Compute state Jacobian
			sys->residualWithJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, jacDir.data(), cadet::AdJacobianParams{nullptr, nullptr, 0u});
			std::fill(jacDir.begin(), jacDir.end(), 0.0);

			// Compare Jacobians
			cadet::test::compareJacobianFD(
				[sys, &y](double const* lDir, double* res) -> void { sys->residual(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), lDir}, res); },
				[sys, &y, &yDot](double const* lDir, double* res) -> void { sys->multiplyWithDerivativeJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, lDir, res); },
				yDot.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), nDof, h, absTol, relTol);
		}
	}
	destroyModelBuilder(mb);
}

TEST_CASE("ModelSystem sensitivity Jacobians", "[ModelSystem],[Sensitivity]")
{
	const double h = 5e-5;
	const double absTol = 5e-8;
	const double relTol = 5e-6; // std::numeric_limits<float>::epsilon() * 100.0;

	cadet::IModelBuilder* const mb = cadet::createModelBuilder();
	REQUIRE(nullptr != mb);

	for (int bindMode = 0; bindMode < 2; ++bindMode)
	{
		const bool isKinetic = bindMode;
		SECTION(isKinetic ? "Kinetic binding" : "Quasi-stationary binding")
		{
			// Use some test case parameters
			cadet::JsonParameterProvider jpp = createLinearBenchmark(isKinetic, false, "GENERAL_RATE_MODEL");

			// Extract section times
			jpp.pushScope("solver");
			jpp.pushScope("sections");

			const std::vector<double> secTimes = jpp.getDoubleArray("SECTION_TIMES");
			std::vector<bool> secCont(secTimes.size() - 2, false);
			if (jpp.exists("SECTION_CONTINUITY")) 
				secCont = jpp.getBoolArray("SECTION_CONTINUITY");

			jpp.popScope();
			jpp.popScope();
			
			// Create and configure ModelSystem
			jpp.pushScope("model");
			cadet::test::column::setNumAxialCells(jpp, 10);
			cadet::IModelSystem* const cadSys = mb->createSystem(jpp);
			REQUIRE(cadSys);
			cadet::model::ModelSystem* const sys = reinterpret_cast<cadet::model::ModelSystem*>(cadSys);
			sys->setupParallelization(cadet::util::getMaxThreads());

			bool* const secContArray = new bool[secCont.size()];
			std::copy(secCont.begin(), secCont.end(), secContArray);
			sys->setSectionTimes(secTimes.data(), secContArray, secTimes.size() - 1);
			delete[] secContArray;

			// Enable AD
			cadet::ad::setDirections(cadet::ad::getMaxDirections());
			cadet::active* adRes = new cadet::active[sys->numDofs()];
			sys->prepareADvectors(cadet::AdJacobianParams{adRes, nullptr, 0});

			// Add dispersion parameter sensitivity
			REQUIRE(sys->setSensitiveParameter(cadet::makeParamId(cadet::hashString("COL_DISPERSION"), 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 0, 1.0));

			// Obtain memory for state, Jacobian multiply direction, Jacobian column
			const unsigned int nDof = sys->numDofs();
			const std::vector<double> zeros(nDof, 0.0);
			const std::vector<double> ones(nDof, 1.0);
			std::vector<double> y(nDof, 0.0);
			std::vector<double> yDot(nDof, 0.0);
			std::vector<double> jacDir(nDof, 0.0);
			std::vector<double> jacCol1(nDof, 0.0);
			std::vector<double> jacCol2(nDof, 0.0);
			std::vector<double> temp1(nDof, 0.0);
			std::vector<double> temp2(nDof, 0.0);
			std::vector<double> temp3(nDof, 0.0);

			std::vector<const double*> yS(1, zeros.data());
			std::vector<const double*> ySdot(1, zeros.data());
			std::vector<double*> resS(1, nullptr);

			// Fill state vector with some values
			cadet::test::util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
			cadet::test::util::populate(yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);

			// Setup matrices
			sys->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, cadet::AdJacobianParams{adRes, nullptr, 0u});

			// Calculate Jacobian
			sys->residualWithJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, jacDir.data(), cadet::AdJacobianParams{adRes, nullptr, 0u});

			// Check state Jacobian
			cadet::test::compareJacobianFD(
				[&](double const* lDir, double* res) -> void {
					yS[0] = lDir;
					resS[0] = res;
					sys->residualSensFwd(1, cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, nullptr, yS, ySdot, resS, adRes, temp1.data(), temp2.data(), temp3.data());
				}, 
				[&](double const* lDir, double* res) -> void { sys->multiplyWithJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, lDir, 1.0, 0.0, res); }, 
				zeros.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), nDof, h, absTol, relTol);

			// Reset evaluation point
			yS[0] = zeros.data();
			ySdot[0] = zeros.data();

			// Check time derivative Jacobian
			cadet::test::compareJacobianFD(
				[&](double const* lDir, double* res) -> void {
					ySdot[0] = lDir;
					resS[0] = res;
					sys->residualSensFwd(1, cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, nullptr, yS, ySdot, resS, adRes, temp1.data(), temp2.data(), temp3.data());
				}, 
				[&](double const* lDir, double* res) -> void { sys->multiplyWithDerivativeJacobian(cadet::SimulationTime{0.0, 0u}, cadet::ConstSimulationState{y.data(), yDot.data()}, lDir, res); }, 
				zeros.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), nDof, h, absTol, relTol);

			delete[] adRes;
		}
	}

	destroyModelBuilder(mb);
}

TEST_CASE("ModelSystem coupling Jacobian linear chain single port (all) comp all", "[ModelSystem],[Jacobian],[Inlet]")
{
	const std::vector<unsigned int> sysDescription = {
		3, 0, 1, 0,
		3, 1, 1, 0,
		3, 1, 1, 0,
		3, 1, 0, 0
	};

	const std::vector<double> connections = {
		0, 1, -1, -1, -1, -1, 1.0,
		1, 2, -1, -1, -1, -1, 1.0,
		2, 3, -1, -1, -1, -1, 1.0
	};

	const std::vector<double> inFlow = {
		1.0,
		1.0,
		1.0
	};

	const std::vector<double> outFlow = {
		1.0,
		1.0,
		1.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}

TEST_CASE("ModelSystem coupling Jacobian linear chain single port (specific) comp all", "[ModelSystem],[Jacobian],[Inlet]")
{
	const std::vector<unsigned int> sysDescription = {
		3, 0, 1, 0,
		3, 1, 1, 0,
		3, 1, 1, 0,
		3, 1, 0, 0
	};

	const std::vector<double> connections = {
		0, 1, 0, 0, -1, -1, 1.0,
		1, 2, 0, 0, -1, -1, 1.0,
		2, 3, 0, 0, -1, -1, 1.0
	};

	const std::vector<double> inFlow = {
		1.0,
		1.0,
		1.0
	};

	const std::vector<double> outFlow = {
		1.0,
		1.0,
		1.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}

TEST_CASE("ModelSystem coupling Jacobian linear chain single port (all) comp specific", "[ModelSystem],[Jacobian],[Inlet]")
{
	const std::vector<unsigned int> sysDescription = {
		3, 0, 1, 0,
		3, 1, 1, 0,
		3, 1, 1, 0,
		3, 1, 0, 0
	};

	const std::vector<double> connections = {
		0, 1, -1, -1, 0, 0, 1.0,
		0, 1, -1, -1, 1, 1, 1.0,
		0, 1, -1, -1, 2, 2, 1.0,
		1, 2, -1, -1, 0, 0, 1.0,
		1, 2, -1, -1, 1, 1, 1.0,
		1, 2, -1, -1, 2, 2, 1.0,
		2, 3, -1, -1, 0, 0, 1.0,
		2, 3, -1, -1, 1, 1, 1.0,
		2, 3, -1, -1, 2, 2, 1.0
	};

	const std::vector<double> inFlow = {
		1.0,
		1.0,
		1.0
	};

	const std::vector<double> outFlow = {
		1.0,
		1.0,
		1.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}

TEST_CASE("ModelSystem coupling Jacobian linear chain single port (specific) comp specific", "[ModelSystem],[Jacobian],[Inlet]")
{
	const std::vector<unsigned int> sysDescription = {
		3, 0, 1, 0,
		3, 1, 1, 0,
		3, 1, 1, 0,
		3, 1, 0, 0
	};

	const std::vector<double> connections = {
		0, 1, 0, 0, 0, 0, 1.0,
		0, 1, 0, 0, 1, 1, 1.0,
		0, 1, 0, 0, 2, 2, 1.0,
		1, 2, 0, 0, 0, 0, 1.0,
		1, 2, 0, 0, 1, 1, 1.0,
		1, 2, 0, 0, 2, 2, 1.0,
		2, 3, 0, 0, 0, 0, 1.0,
		2, 3, 0, 0, 1, 1, 1.0,
		2, 3, 0, 0, 2, 2, 1.0
	};

	const std::vector<double> inFlow = {
		1.0,
		1.0,
		1.0
	};

	const std::vector<double> outFlow = {
		1.0,
		1.0,
		1.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}

TEST_CASE("ModelSystem coupling Jacobian linear chain multi port (specific) comp all", "[ModelSystem],[Jacobian],[Inlet]")
{
	/*
	         1     1     1
	      /-->--O-->--O-->--\
	2 -->O                   O--> 2
	      \-->--O-->--O-->--/
	         1     1     1
	*/

	const std::vector<unsigned int> sysDescription = {
		3, 0, 1, 0,
		3, 2, 2, 0,
		3, 2, 2, 0,
		3, 1, 0, 0
	};

	const std::vector<double> connections = {
		0, 1, 0, 0, -1, -1, 1.0,
		0, 1, 0, 1, -1, -1, 1.0,
		1, 2, 0, 1, -1, -1, 1.0,
		1, 2, 1, 0, -1, -1, 1.0,
		2, 3, 0, 0, -1, -1, 1.0,
		2, 3, 1, 0, -1, -1, 1.0
	};

	const std::vector<double> inFlow = {
		1.0, 1.0,
		1.0, 1.0,
		2.0
	};

	const std::vector<double> outFlow = {
		2.0,
		1.0, 1.0,
		1.0, 1.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}

TEST_CASE("ModelSystem coupling Jacobian linear chain multi port (specific) comp specific", "[ModelSystem],[Jacobian],[Inlet]")
{
	/*
	         1     1     1
	      /-->--O-->--O-->--\
	2 -->O                   O--> 2
	      \-->--O-->--O-->--/
	         1     1     1
	*/

	const std::vector<unsigned int> sysDescription = {
		3, 0, 1, 0,
		3, 2, 2, 0,
		3, 2, 2, 0,
		3, 1, 0, 0
	};

	const std::vector<double> connections = {
		0, 1, 0, 0, 0, 0, 1.0,
		0, 1, 0, 0, 1, 1, 1.0,
		0, 1, 0, 0, 2, 2, 1.0,
		0, 1, 0, 1, 0, 0, 1.0,
		0, 1, 0, 1, 1, 1, 1.0,
		0, 1, 0, 1, 2, 2, 1.0,
		1, 2, 0, 1, 0, 0, 1.0,
		1, 2, 0, 1, 1, 1, 1.0,
		1, 2, 0, 1, 2, 2, 1.0,
		1, 2, 1, 0, 0, 0, 1.0,
		1, 2, 1, 0, 1, 1, 1.0,
		1, 2, 1, 0, 2, 2, 1.0,
		2, 3, 0, 0, 0, 0, 1.0,
		2, 3, 0, 0, 1, 1, 1.0,
		2, 3, 0, 0, 2, 2, 1.0,
		2, 3, 1, 0, 0, 0, 1.0,
		2, 3, 1, 0, 1, 1, 1.0,
		2, 3, 1, 0, 2, 2, 1.0
	};

	const std::vector<double> inFlow = {
		1.0, 1.0,
		1.0, 1.0,
		2.0
	};

	const std::vector<double> outFlow = {
		2.0,
		1.0, 1.0,
		1.0, 1.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}

TEST_CASE("ModelSystem coupling Jacobian X single port (all) comp all", "[ModelSystem],[Jacobian],[Inlet]")
{
	/*
	    O--\     /--O
	        --O--
	    O--/     \--O
	*/

	const std::vector<unsigned int> sysDescription = {
		2, 0, 1, 0,
		2, 0, 1, 0,
		2, 1, 1, 0,
		2, 1, 0, 0,
		2, 1, 0, 0
	};

	const std::vector<double> connections = {
		0, 2, 0, 0, -1, -1, 1.0,
		1, 2, 0, 0, -1, -1, 1.0,
		2, 3, 0, 0, -1, -1, 1.0,
		2, 4, 0, 0, -1, -1, 1.0
	};

	const std::vector<double> inFlow = {
		2.0,
		1.0,
		1.0
	};

	const std::vector<double> outFlow = {
		1.0,
		1.0,
		2.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}

TEST_CASE("ModelSystem coupling Jacobian X multi port comp all", "[ModelSystem],[Jacobian],[Inlet]")
{
	/*
	    O--\     /--O
	        --O--
	    O--/     \--O
	*/

	const std::vector<unsigned int> sysDescription = {
		2, 0, 1, 0,
		2, 0, 1, 0,
		2, 2, 2, 0,
		2, 1, 0, 0,
		2, 1, 0, 0
	};

	const std::vector<double> connections = {
		0, 2, 0, 0, -1, -1, 1.0,
		1, 2, 0, 1, -1, -1, 1.0,
		2, 3, 1, 0, -1, -1, 1.0,
		2, 4, 0, 0, -1, -1, 1.0
	};

	const std::vector<double> inFlow = {
		1.0, 1.0,
		1.0,
		1.0
	};

	const std::vector<double> outFlow = {
		1.0,
		1.0,
		1.0, 1.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}

TEST_CASE("ModelSystem coupling Jacobian linear chain single port (all) comp twisted", "[ModelSystem],[Jacobian],[Inlet]")
{
	const std::vector<unsigned int> sysDescription = {
		3, 0, 1, 0,
		3, 1, 1, 0,
		3, 1, 1, 0,
		3, 1, 0, 0
	};

	const std::vector<double> connections = {
		0, 1, 0, 0, 0, 0, 1.0,
		0, 1, 0, 0, 1, 1, 1.0,
		0, 1, 0, 0, 2, 2, 1.0,
		1, 2, 0, 0, 0, 2, 1.0,
		1, 2, 0, 0, 1, 1, 1.0,
		1, 2, 0, 0, 2, 0, 1.0,
		2, 3, 0, 0, 0, 0, 1.0,
		2, 3, 0, 0, 1, 1, 1.0,
		2, 3, 0, 0, 2, 2, 1.0
	};

	const std::vector<double> inFlow = {
		1.0,
		1.0,
		1.0
	};

	const std::vector<double> outFlow = {
		1.0,
		1.0,
		1.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}


TEST_CASE("ModelSystem coupling Jacobian circular", "[ModelSystem],[Jacobian],[Inlet]")
{
	/*
	    ___________________
	    |                 |
	0---2--\     /--5---  |
	        --4--      |  |
	1---3--/     \--6-----
	    |______________|
	*/

	const std::vector<unsigned int> sysDescription = {
		2, 0, 1, 0,
		2, 0, 1, 0,
		2, 2, 1, 0,
		2, 2, 1, 0,
		2, 1, 1, 1,
		2, 1, 1, 0,
		2, 1, 1, 0
	};

	const std::vector<double> connections = {
		0, 2,  0,  0, -1, -1, 1.0,
		1, 3,  0,  0, -1, -1, 1.0,
		2, 4, -1, -1, -1, -1, 2.0,
		3, 4, -1, -1, -1, -1, 2.0,
		4, 5,  0,  0, -1, -1, 1.0,
		4, 6, -1, -1, -1, -1, 1.0,
		5, 3,  0,  1, -1, -1, 1.0,
		6, 2,  0,  1, -1, -1, 1.0
	};

	const std::vector<double> inFlow = {
		1.0, 1.0,
		1.0, 1.0,
		4.0,
		1.0,
		1.0
	};

	const std::vector<double> outFlow = {
		1.0,
		1.0,
		2.0,
		2.0,
		2.0,
		1.0,
		1.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}

TEST_CASE("ModelSystem coupling Jacobian component Y", "[ModelSystem],[Jacobian],[Inlet]")
{
	/*
	    O--\     
	        --O--O
	    O--/     
	*/

	const std::vector<unsigned int> sysDescription = {
		2, 0, 1, 0,
		2, 0, 1, 0,
		4, 1, 1, 0,
		4, 1, 0, 0
	};

	const std::vector<double> connections = {
		0, 2, 0, 0,  0,  0, 1.0,
		0, 2, 0, 0,  1,  1, 1.0,
		1, 2, 0, 0,  0,  2, 1.0,
		1, 2, 0, 0,  0,  3, 1.0,
		2, 3, 0, 0, -1, -1, 2.0
	};

	const std::vector<double> inFlow = {
		2.0,
		2.0
	};

	const std::vector<double> outFlow = {
		1.0,
		1.0,
		2.0
	};

	checkCouplingJacobian(sysDescription, connections, inFlow, outFlow);
}
