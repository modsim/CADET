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

#include "model/ModelSystemImpl.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/Exceptions.hpp"
#include "cadet/ExternalFunction.hpp"

#include "ConfigurationHelper.hpp"
#include "graph/GraphAlgos.hpp"
#include "SensParamUtil.hpp"
#include "SimulationTypes.hpp"

#include <sstream>
#include <iomanip>
#include <iterator>

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include "model/ModelSystemImpl-Helper.hpp"

namespace
{
	/**
	 * @brief Find array index of unit operation with given id
	 * @details Unit operation index does not need to match index of unit operation in an array.
	 * @param [in] models List of models
	 * @param [in] unitOpIdx Unit operation index to look for
	 * @return Array index of the unit operation identified by unit operation index or invalid array index if the unit operation has not been found
	 */
	inline unsigned int indexOfUnitOp(const std::vector<cadet::IUnitOperation*>& models, unsigned int unitOpIdx)
	{
		for (std::size_t i = 0; i < models.size(); ++i)
		{
			if (models[i]->unitOperationId() == unitOpIdx)
				return i;
		}
		return models.size();
	}

	/**
	 * @brief Returns whether a given unit operation is a terminal node in the network
	 * @param [in] conn Connection list
	 * @param [in] size Number of rows in the connection list
	 * @param [in] unitOpIdx Index of the unit operation to check
	 * @return @c true if the unit operation is a terminal node in the network, @c false otherwise
	 */
	inline bool isTerminal(int const* const conn, unsigned int size, int unitOpIdx)
	{
		for (unsigned int i = 0; i < size; ++i)
		{
			if (conn[6*i] == unitOpIdx)
				return false;
		}
		return true;
	}

	/**
	 * @brief Returns whether one or more unit operations have multiple ports
	 * @param [in] models List of unit operation models
	 * @return @c true if the list contains at least one unit operation model with multiple ports, @c false otherwise
	 */
	inline bool hasMultiPortUnits(const std::vector<cadet::IUnitOperation*>& models)
	{
		for (cadet::IUnitOperation const* m : models)
		{
			if ((m->numInletPorts() > 1) || (m->numOutletPorts() > 1))
				return true;
		}
		return false;
	}

	template <typename T>
	std::string toSciString(const T val, const int prec = 6)
	{
		std::ostringstream out;
		out << std::scientific << std::setprecision(prec) << val;
		return out.str();
	}
}

namespace cadet
{

namespace model
{

ModelSystem::ModelSystem() : _jacNF(nullptr), _jacFN(nullptr), _jacActiveFN(nullptr), _curSwitchIndex(0), _tempState(nullptr), _initState(0, 0.0), _initStateDot(0, 0.0)
{
}

ModelSystem::~ModelSystem() CADET_NOEXCEPT
{
	for (IUnitOperation* model : _models)
		delete model;

	for (IExternalFunction* extFun : _extFunctions)
		delete extFun;

	delete[] _tempState;
	delete[] _jacNF;
	delete[] _jacFN;
	delete[] _jacActiveFN;
}

void ModelSystem::addModel(IModel* unitOp)
{
	// Check for unique unit operation id
	if (indexOfUnitOp(_models, unitOp->unitOperationId()) < _models.size())
		throw InvalidParameterException("Cannot add model because of already existing unit operation id " + std::to_string(unitOp->unitOperationId()));

	IUnitOperation* const uo = static_cast<IUnitOperation*>(unitOp);
	_models.push_back(uo);

	if (uo->hasInlet() && uo->hasOutlet())
		_inOutModels.push_back(_models.size() - 1);

	// Propagate external functions to submodel
	uo->setExternalFunctions(_extFunctions.data(), _extFunctions.size());
}

IModel* ModelSystem::getModel(unsigned int index)
{
	if (index < _models.size())
		return _models[index];
	else
		return nullptr;
}

IModel const* ModelSystem::getModel(unsigned int index) const
{
	if (index < _models.size())
		return _models[index];
	else
		return nullptr;
}

IUnitOperation* ModelSystem::getUnitOperationModel(unsigned int unitOpIdx)
{
	for (IUnitOperation* m : _models)
	{
		if (m->unitOperationId() == unitOpIdx)
			return m;
	}
	return nullptr;
}

IUnitOperation const* ModelSystem::getUnitOperationModel(unsigned int unitOpIdx) const
{
	for (IUnitOperation* m : _models)
	{
		if (m->unitOperationId() == unitOpIdx)
			return m;
	}
	return nullptr;
}

unsigned int ModelSystem::numModels() const CADET_NOEXCEPT
{
	return _models.size();
}

void ModelSystem::removeModel(IModel const* unitOp)
{
	std::vector<IUnitOperation*>::iterator it = std::find(_models.begin(), _models.end(), static_cast<IUnitOperation const*>(unitOp));
	if (it != _models.end())
		_models.erase(it);
}

IModel* ModelSystem::removeModel(UnitOpIdx unitOp)
{
	for (std::vector<IUnitOperation*>::iterator it = _models.begin(); it != _models.end(); ++it)
	{
		if ((*it)->unitOperationId() == unitOp)
		{
			IModel* const mod = static_cast<IModel*>(*it);
			_models.erase(it);
			return mod;
		}
	}
	return nullptr;
}

UnitOpIdx ModelSystem::maxUnitOperationId() const CADET_NOEXCEPT
{
	if (_models.empty())
		return UnitOpIndep;

	UnitOpIdx id = 0;
	for (IUnitOperation* m : _models)
		id = std::max(id, m->unitOperationId());
	return id;
}

unsigned int ModelSystem::totalNumInletPorts() const CADET_NOEXCEPT
{
	unsigned int nPorts = 0;
	for (IUnitOperation const* m : _models)
		nPorts += m->numInletPorts();

	return nPorts;
}

unsigned int ModelSystem::totalNumOutletPorts() const CADET_NOEXCEPT
{
	unsigned int nPorts = 0;
	for (IUnitOperation const* m : _models)
		nPorts += m->numOutletPorts();

	return nPorts;
}

unsigned int ModelSystem::addExternalFunction(IExternalFunction& extFun)
{
	_extFunctions.push_back(&extFun);

	// Propagate external functions to submodels
	for (IUnitOperation* m : _models)
		m->setExternalFunctions(_extFunctions.data(), _extFunctions.size());

	return _extFunctions.size() - 1;
}

IExternalFunction* ModelSystem::getExternalFunction(unsigned int index)
{
	if (index < _extFunctions.size())
		return _extFunctions[index];
	else
		return nullptr;
}

IExternalFunction const* ModelSystem::getExternalFunction(unsigned int index) const
{
	if (index < _extFunctions.size())
		return _extFunctions[index];
	else
		return nullptr;
}

unsigned int ModelSystem::numExternalFunctions() const CADET_NOEXCEPT
{
	return _extFunctions.size();
}

void ModelSystem::removeExternalFunction(IExternalFunction const* extFun)
{
	for (std::vector<IExternalFunction*>::iterator it = _extFunctions.begin(); it != _extFunctions.end(); ++it)
	{
		if (*it == extFun)
		{
			_extFunctions.erase(it);
			break;
		}
	}

	// Update external functions in submodels
	for (IUnitOperation* m : _models)
		m->setExternalFunctions(_extFunctions.data(), _extFunctions.size());
}

unsigned int ModelSystem::numDofs() const CADET_NOEXCEPT
{
	return _dofOffset.back() + numCouplingDOF();
}

unsigned int ModelSystem::numPureDofs() const CADET_NOEXCEPT
{
	unsigned int dofs = 0;
	for (IUnitOperation* m : _models)
		dofs += m->numPureDofs();
	return dofs;
}

std::tuple<unsigned int, unsigned int> ModelSystem::getModelStateOffsets(UnitOpIdx unitOp) const CADET_NOEXCEPT
{
	for (int i = 0; i < _models.size(); ++i)
	{
		if (_models[i]->unitOperationId() == unitOp)
			return std::make_tuple(_dofOffset[i], _dofOffset[i+1]);
	}

	return std::make_tuple(-1u, -1u);
}

bool ModelSystem::usesAD() const CADET_NOEXCEPT
{
	for (IUnitOperation* m : _models)
	{
		if (m->usesAD())
			return true;
	}
	return false;
}

/**
* @brief Create data structures to keep track of entries and locations in the state vector
* @details Three data structures are created. One keeps track of the offsets to each unit operation.
*          The second keeps track of the size of each unit operation. The third is mapping from
*          unit operation and component index to unique inlet DOF index.
*/
void ModelSystem::rebuildInternalDataStructures()
{
	// Calculate array with DOF offsets
	_dofOffset.clear();
	_dofs.clear();
	_conDofOffset.clear();

	// The additional entry holds the offset for the superstructure
	_dofOffset.reserve(_models.size()+1);
	_dofs.reserve(_models.size() + 1);
	_conDofOffset.reserve(_models.size() + 1);

	// Process DOF from models
	unsigned int totalDof = 0;
	unsigned int conTotalDof = 0;
	for (IUnitOperation const* m : _models)
	{
		_dofOffset.push_back(totalDof);
		_conDofOffset.push_back(conTotalDof);
		totalDof += m->numDofs();
		conTotalDof += m->numInletPorts() * m->numComponents();
		_dofs.push_back(m->numDofs());
	}

	// Process DOF from superstructure
	_dofOffset.push_back(totalDof);
	_conDofOffset.push_back(conTotalDof);

	/*
		A mapping is required that turns a local model, port index, and component index into the location of the inlet DOF in
		the global state vector. Some unit operations do not have inlet DOFs (e.g., inlet unit operation). Hence,
		a map is constructed which converts local indices into global ones.
	*/

	// Build a mapping (unitOpIdx, PortIdx, CompIdx) -> local coupling DOF index
	_couplingIdxMap.clear();
	unsigned int counter = 0;
	for (unsigned int i = 0; i < numModels(); ++i)
	{
		IUnitOperation const* const model = _models[i];
		
		//Only unit operations with an inlet have dedicated inlet DOFs
		if (model->hasInlet())
		{
			for (unsigned int port = 0; port < model->numInletPorts(); ++port)
			{
				for (unsigned int comp = 0; comp < model->numComponents(); ++comp)
				{
					_couplingIdxMap.insert({ std::make_tuple(i, port, comp), counter });
					++counter;
				}
			}
		}
	}

	_dofs.push_back(numCouplingDOF());

	// Allocate error indicator vector
	_errorIndicator.resize(_models.size(), 0);

	LOG(Debug) << "DOF offsets: " << _dofOffset;
}

/**
* @brief Allocates memory for the superstructure matrices
* @details How many connections each unit has determines how much memory has to be allocated for the coupling matrices.
*          This function walks the connections over the entire simulation in order to determine the maximum number of 
*          connections over the whole simulation which governs the number of entries in the sparse superstructure coupling
*          matrices. Finally, the required memory is allocated in the matrices.
*/
void ModelSystem::allocateSuperStructMatrices()
{
	// Step 1: Calculate number of connections per unit operation per valve switch
	// We record the number of outgoing connections (i.e., components) which act as 
	// sources in the bottom macro-row of the superstructure
	const unsigned int nModels = numModels();
	std::vector<unsigned int> sourcesPerUnitOpPerConfig(nModels * _switchSectionIndex.size(), 0u);

	for (std::size_t idx = 0; idx < _switchSectionIndex.size(); ++idx)
	{
		int const* ptrConn = _connections[idx];

		for (unsigned int i = 0; i < _connections.sliceSize(idx) / 6; ++i, ptrConn += 6)
		{
			// Extract current connection
			const int uoSource = ptrConn[0];
			const int portSource = ptrConn[2];
			const int compSource = ptrConn[4];

			IUnitOperation const* const model = _models[uoSource];
			if (portSource == -1)
			{
				if (compSource == -1)
					sourcesPerUnitOpPerConfig[nModels * idx + uoSource] += model->numOutletPorts() * model->numComponents();
				else
					sourcesPerUnitOpPerConfig[nModels * idx + uoSource] += model->numOutletPorts();
			}
			else
			{
				if (compSource == -1)
					sourcesPerUnitOpPerConfig[nModels * idx + uoSource] += model->numComponents();
				else
					sourcesPerUnitOpPerConfig[nModels * idx + uoSource] += 1;
			}
		}
	}

	// Step 2: Take maximum over valve switches to obtain maximum number of matrix entries per unit operation
	std::vector<unsigned int> numOutgoing(nModels, 0u);

	// Assign the first row of sourcesPerUnitOpPerConfig to the maximum number of outputs
	std::copy(sourcesPerUnitOpPerConfig.begin(), sourcesPerUnitOpPerConfig.begin() + nModels, numOutgoing.begin());

	// Loop over remaining lines and take maximum per element
	for (std::size_t sectionIdx = 1; sectionIdx < _switchSectionIndex.size(); ++sectionIdx)
	{
		for (unsigned int i = 0; i < nModels; ++i)
		{
			numOutgoing[i] = std::max(numOutgoing[i], sourcesPerUnitOpPerConfig[nModels * sectionIdx + i]);
		}
	}

	// Step 3: Allocate memory based on maximum number of connections for each unit operation
	for (unsigned int i = 0; i < numModels(); ++i)
	{
		// Bottom macro-row
		_jacActiveFN[i].resize(numOutgoing[i]);

		// Right macro-column
		// Each unit operation has inlets equal to numComponents * numInletPorts so long as the unit operation has an inlet
		IUnitOperation const* const model = _models[i];
		if (model->hasInlet())
			_jacNF[i].resize(model->numComponents() * model->numInletPorts());
	}
}

bool ModelSystem::configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper)
{
	// Discretizations of unit operation models are already configured
	rebuildInternalDataStructures();

	_parameters.clear();

	paramProvider.pushScope("solver");
	readLinearSolutionMode(paramProvider);
	paramProvider.popScope();

	configureSwitches(paramProvider);
	_curSwitchIndex = 0;

	// Allocate memory to coupling matrices
	_jacActiveFN = new linalg::SparseMatrix<active>[numModels()];
	_jacNF = new linalg::SparseMatrix<double>[numModels()];
	_jacFN = new linalg::SparseMatrix<double>[numModels()];

	// Calculate the sizes that need to be allocated for all the inlets and outlets
	allocateSuperStructMatrices();

	// Create and configure all external functions
	bool success = true;
	if (paramProvider.exists("external"))
	{
		paramProvider.pushScope("external");

		unsigned int i = 0;
		std::ostringstream oss;
		oss << "source_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
		while (paramProvider.exists(oss.str()))
		{
			// Create and configure unit operation
			paramProvider.pushScope(oss.str());

			const std::string extType = paramProvider.getString("EXTFUN_TYPE");
			IExternalFunction* const func = helper.createExternalFunction(extType);
			if (func)
			{
				const bool confSuccess = func->configure(&paramProvider);

				if (confSuccess)
					_extFunctions.push_back(func);
				else
				{
					// Ignore the external function and delete this instance
					_extFunctions.push_back(nullptr);
					delete func;
					success = false;

					LOG(Error) << "Failed to configure external source " << i << " (" << extType << "), source is ignored";
				}
			}
			else
			{
				// Unknown type of external function
				_extFunctions.push_back(nullptr);
				success = false;

				LOG(Error) << "Failed to create external source " << i << " as type " << extType << " is unknown, source is ignored";
			}

			paramProvider.popScope();

			// Next group in file format
			++i;
			oss.str("");
			oss << "source_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
		}

		paramProvider.popScope();
	}

	// Propagate external functions to submodels
	for (IUnitOperation* m : _models)
		m->setExternalFunctions(_extFunctions.data(), _extFunctions.size());

	// Read solver settings
	paramProvider.pushScope("solver");
	const int maxKrylov = paramProvider.getInt("MAX_KRYLOV");
	const int gsType = paramProvider.getInt("GS_TYPE");
	const int maxRestarts = paramProvider.getInt("MAX_RESTARTS");
	_schurSafety = paramProvider.getDouble("SCHUR_SAFETY");
	paramProvider.popScope();

	// Initialize and configure GMRES for solving the Schur-complement
    _gmres.initialize(numCouplingDOF(), maxKrylov, linalg::toOrthogonalization(gsType), maxRestarts);

	// Allocate tempState vector
	delete[] _tempState;
	_tempState = new double[numDofs()];

//	_tempSchur = new double[*std::max_element(_dofs.begin(), _dofs.end())];
	_flowRateIn.reserve(totalNumInletPorts(), _models.size());
	_flowRateOut.reserve(totalNumOutletPorts(), _models.size());

	_totalInletFlow.reserve(totalNumInletPorts(), _models.size());
	_totalInletFlowLin.reserve(totalNumInletPorts(), _models.size());
	_totalInletFlowQuad.reserve(totalNumInletPorts(), _models.size());
	_totalInletFlowCub.reserve(totalNumInletPorts(), _models.size());

	_totalOutletFlow.reserve(totalNumOutletPorts(), _models.size());
	_totalOutletFlowLin.reserve(totalNumOutletPorts(), _models.size());
	_totalOutletFlowQuad.reserve(totalNumOutletPorts(), _models.size());
	_totalOutletFlowCub.reserve(totalNumOutletPorts(), _models.size());

	for (IUnitOperation const* m : _models)
	{
		_flowRateIn.pushBackSlice(m->numInletPorts());
		_totalInletFlow.pushBackSlice(m->numInletPorts());
		_totalInletFlowLin.pushBackSlice(m->numInletPorts());
		_totalInletFlowQuad.pushBackSlice(m->numInletPorts());
		_totalInletFlowCub.pushBackSlice(m->numInletPorts());

		_flowRateOut.pushBackSlice(m->numOutletPorts());
		_totalOutletFlow.pushBackSlice(m->numOutletPorts());
		_totalOutletFlowLin.pushBackSlice(m->numOutletPorts());
		_totalOutletFlowQuad.pushBackSlice(m->numOutletPorts());
		_totalOutletFlowCub.pushBackSlice(m->numOutletPorts());
	}

	assembleRightMacroColumn();

#ifdef CADET_DEBUG

	LOG(Debug) << "Num units " << _models.size() << " max ID " << maxUnitOperationId() << " total DOF " << numDofs();
	LOG(Debug) << "Unit op state vector size: uoDOFs = " << log::VectorPtr<unsigned int>(_dofs.data(), _dofs.size());
	LOG(Debug) << "Unit op state vector offsets: uoOffset = " << log::VectorPtr<unsigned int>(_dofOffset.data(), _dofOffset.size());

#endif

	return success;
}

bool ModelSystem::configure(IParameterProvider& paramProvider)
{
	_parameters.clear();

	// Reconfigure all external functions
	bool success = true;
	if (paramProvider.exists("external"))
	{
		paramProvider.pushScope("external");

		std::ostringstream oss;
		for (std::size_t i = 0; i < _extFunctions.size(); ++i)
		{
			IExternalFunction* const func = _extFunctions[i];
			if (!func)
				continue;

			oss.str("");
			oss << "source_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
			if (!paramProvider.exists(oss.str()))
				continue;

			paramProvider.pushScope(oss.str());
			const bool localSuccess = func->configure(&paramProvider);
			paramProvider.popScope();

			if (!localSuccess)
			{
				LOG(Error) << "Failed to reconfigure external source " << i;
			}

			success = localSuccess && success;
		}

		paramProvider.popScope();
	}

	// Reconfigure solver settings
	paramProvider.pushScope("solver");

	const int gsType = paramProvider.getInt("GS_TYPE");
	const int maxRestarts = paramProvider.getInt("MAX_RESTARTS");
	_schurSafety = paramProvider.getDouble("SCHUR_SAFETY");
	readLinearSolutionMode(paramProvider);

	paramProvider.popScope();

	_gmres.orthoMethod(linalg::toOrthogonalization(gsType));
	_gmres.maxRestarts(maxRestarts);

	// Reconfigure switches
	configureSwitches(paramProvider);

	return success;
}

/**
 * @brief Reads valve switches from the given parameter provider
 * @param [in] paramProvider Parameter provider
 */
void ModelSystem::configureSwitches(IParameterProvider& paramProvider)
{
	// Read connections of unit operations
	paramProvider.pushScope("connections");

	const unsigned int numSwitches = paramProvider.getInt("NSWITCHES");
	
	// TODO Improve those very conservative size estimates
	_switchSectionIndex.clear();
	_switchSectionIndex.reserve(numSwitches);
	_connections.clear();
	_connections.reserve(numSwitches * 6 * _models.size() * _models.size(), numSwitches);
	_flowRates.clear();
	_flowRates.reserve(numSwitches * _models.size() * _models.size(), numSwitches);
	_linearModelOrdering.reserve(numSwitches * _models.size(), numSwitches);
	_linearModelOrdering.clear();

#if CADET_COMPILER_CXX_CONSTEXPR
	constexpr StringHash flowHash = hashString("CONNECTION");
	constexpr StringHash flowHashLin = hashString("CONNECTION_LIN");
	constexpr StringHash flowHashQuad = hashString("CONNECTION_QUAD");
	constexpr StringHash flowHashCub = hashString("CONNECTION_CUBE");
#else
	const StringHash flowHash = hashString("CONNECTION");
	const StringHash flowHashLin = hashString("CONNECTION_LIN");
	const StringHash flowHashQuad = hashString("CONNECTION_QUAD");
	const StringHash flowHashCub = hashString("CONNECTION_CUBE");
#endif

	// Default: ports are not given in connection list
	bool conListHasPorts = false;

	// Override default by user option
	if (paramProvider.exists("CONNECTIONS_INCLUDE_PORTS"))
		conListHasPorts = paramProvider.getBool("CONNECTIONS_INCLUDE_PORTS");

	// If we have unit operations with multiple ports, we require ports in connection list
	if (hasMultiPortUnits(_models))
		conListHasPorts = true;

	// Default: no dynamic flow rates
	_hasDynamicFlowRates = false;

	// Override default by user option
	if (paramProvider.exists("CONNECTIONS_INCLUDE_DYNAMIC_FLOW"))
		_hasDynamicFlowRates = paramProvider.getBool("CONNECTIONS_INCLUDE_DYNAMIC_FLOW");

	std::ostringstream oss;
	for (unsigned int i = 0; i < numSwitches; ++i)
	{
		oss.str("");
		oss << "switch_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;

		paramProvider.pushScope(oss.str());

		_switchSectionIndex.push_back(paramProvider.getInt("SECTION"));

		if ((i > 0) && (_switchSectionIndex.back() <= _switchSectionIndex[_switchSectionIndex.size() - 2]))
			throw InvalidParameterException("SECTION index has to be monotonically increasing (switch " + std::to_string(i) + ")");

		std::vector<double> connFlow = paramProvider.getDoubleArray("CONNECTIONS");
		if (!conListHasPorts)
		{
			if (_hasDynamicFlowRates)
			{
				if ((connFlow.size() % 8) != 0)
					throw InvalidParameterException("CONNECTIONS matrix has to have 8 columns if CONNECTIONS_INCLUDE_PORTS is disabled and CONNECTIONS_INCLUDE_DYNAMIC_FLOW is enabled");
			}
			else
			{
				if ((connFlow.size() % 5) != 0)
					throw InvalidParameterException("CONNECTIONS matrix has to have 5 columns if CONNECTIONS_INCLUDE_PORTS is disabled and CONNECTIONS_INCLUDE_DYNAMIC_FLOW is disabled");
			}

			addDefaultPortsToConnectionList(connFlow);
		}

		if (!_hasDynamicFlowRates)
		{
			if ((connFlow.size() % 7) != 0)
				throw InvalidParameterException("CONNECTIONS matrix has to have 5 or 7 columns if CONNECTIONS_INCLUDE_DYNAMIC_FLOW is disabled");

			addDefaultDynamicFlowRatesToConnectionList(connFlow);
		}

		if ((connFlow.size() % 10) != 0)
			throw InvalidParameterException("CONNECTIONS matrix has to have 10 columns");

		std::vector<int> conn(connFlow.size() / 10 * 6, 0);
		std::vector<double> fr(connFlow.size() / 10, 0.0);
		std::vector<double> frLin(connFlow.size() / 10, 0.0);
		std::vector<double> frQuad(connFlow.size() / 10, 0.0);
		std::vector<double> frCub(connFlow.size() / 10, 0.0);

		checkConnectionList(connFlow, conn, fr, frLin, frQuad, frCub, i);

		_connections.pushBackSlice(conn);

		// Convert double to active while pushing into the SlicedVector
		// also register parameter to enable sensitivities
		if (fr.size() > 0)
		{
			_flowRates.pushBack(fr[0]);
			_flowRatesLin.pushBack(frLin[0]);
			_flowRatesQuad.pushBack(frQuad[0]);
			_flowRatesCub.pushBack(frCub[0]);
			_parameters[makeParamId(flowHash, UnitOpIndep, conn[2], conn[3], conn[0], conn[1], _switchSectionIndex.back())] = _flowRates.back();
			_parameters[makeParamId(flowHashLin, UnitOpIndep, conn[2], conn[3], conn[0], conn[1], _switchSectionIndex.back())] = _flowRatesLin.back();
			_parameters[makeParamId(flowHashQuad, UnitOpIndep, conn[2], conn[3], conn[0], conn[1], _switchSectionIndex.back())] = _flowRatesQuad.back();
			_parameters[makeParamId(flowHashCub, UnitOpIndep, conn[2], conn[3], conn[0], conn[1], _switchSectionIndex.back())] = _flowRatesCub.back();
			for (std::size_t j = 1; j < fr.size(); ++j)
			{
				_flowRates.pushBackInLastSlice(fr[j]);
				_flowRatesLin.pushBackInLastSlice(frLin[j]);
				_flowRatesQuad.pushBackInLastSlice(frQuad[j]);
				_flowRatesCub.pushBackInLastSlice(frCub[j]);

				// Check if a previous identical connection (except for component indices) exists
				bool found = false;
				for (unsigned int k = 0; k < j; ++k)
				{
					if ((conn[6*k] == conn[6*j]) && (conn[6*k+1] == conn[6*j+1]) && (conn[6*k+2] == conn[6*j+2]) && (conn[6*k+3] == conn[6*j+3]))
					{
						found = true;
						break;
					}
				}

				// Only register the first occurrence of a flow parameter
				if (!found)
				{
					_parameters[makeParamId(flowHash, UnitOpIndep, conn[6*j+2], conn[6*j+3], conn[6*j], conn[6*j+1], _switchSectionIndex.back())] = (_flowRates.back() + j);
					_parameters[makeParamId(flowHashLin, UnitOpIndep, conn[6*j+2], conn[6*j+3], conn[6*j], conn[6*j+1], _switchSectionIndex.back())] = (_flowRatesLin.back() + j);
					_parameters[makeParamId(flowHashQuad, UnitOpIndep, conn[6*j+2], conn[6*j+3], conn[6*j], conn[6*j+1], _switchSectionIndex.back())] = (_flowRatesQuad.back() + j);
					_parameters[makeParamId(flowHashCub, UnitOpIndep, conn[6*j+2], conn[6*j+3], conn[6*j], conn[6*j+1], _switchSectionIndex.back())] = (_flowRatesCub.back() + j);
				}
			}
		}
		else
		{
			// Add empty slice
			_flowRates.pushBackSlice(nullptr, 0);
			_flowRatesLin.pushBackSlice(nullptr, 0);
			_flowRatesQuad.pushBackSlice(nullptr, 0);
			_flowRatesCub.pushBackSlice(nullptr, 0);
		}

		if (_linearSolutionMode == 1)
		{
			// Parallel solution method
			_linearModelOrdering.pushBackSlice(0);
			LOG(Debug) << "Select parallel solution method for switch " << i;
		}
		else if (_linearSolutionMode == 2)
		{
			// Sequential solution method
			
			const util::SlicedVector<int> adjList = graph::adjacencyListFromConnectionList(conn.data(), _models.size(), conn.size() / 6);
			std::vector<int> topoOrder;
			const bool hasCycles = graph::topologicalSort(adjList, topoOrder);

			if (hasCycles)
			{
				LOG(Warning) << "Detected cycle in connections of switch " << i << ", reverting to parallel solution method";
				_linearModelOrdering.pushBackSlice(0);
			}
			else
			{
				_linearModelOrdering.pushBackSlice(topoOrder);
				LOG(Debug) << "Select sequential solution method for switch " << i;
				LOG(Debug) << "Reversed ordering: " << topoOrder;
			}
		}
		else
		{
			// Auto detect solution method

			// TODO: Add heuristic for number and complexity of unit operations
			
			// Simple heuristic: At least 25 models (regardless of complexity) => Parallelize
			if (_models.size() >= 25)
			{
				_linearModelOrdering.pushBackSlice(0);
				LOG(Debug) << "Select parallel solution method for switch " << i << " (at least 25 models)";
			}
			else
			{
				// Less than 25 models => Select depending on existence of cycles
				const util::SlicedVector<int> adjList = graph::adjacencyListFromConnectionList(conn.data(), _models.size(), conn.size() / 6);
				std::vector<int> topoOrder;
				const bool hasCycles = graph::topologicalSort(adjList, topoOrder);

				if (hasCycles)
				{
					_linearModelOrdering.pushBackSlice(0);
					LOG(Debug) << "Select parallel solution method for switch " << i << " (cycles found)";
				}
				else
				{
					_linearModelOrdering.pushBackSlice(topoOrder);
					LOG(Debug) << "Select sequential solution method for switch " << i << " (no cycles found, less than 6 models)";
					LOG(Debug) << "Reversed ordering: " << topoOrder;
				}
			}
		}

		paramProvider.popScope();
	}

	paramProvider.popScope();

	if (_switchSectionIndex[0] != 0)
		throw InvalidParameterException("First element of SECTION in connections group has to be 0");
}

/**
 * @brief Add default ports to connection list
 * @details Adds source and destination ports of @c -1 to the connection list. The list is
 *          overwritten, that is, all pointers and iterators are invalidated.
 * @param [in,out] conList On entry, list of connections without ports;
 *                         on exit, list of connections including default ports
 */
void ModelSystem::addDefaultPortsToConnectionList(std::vector<double>& conList) const
{
	const int strideSrc = _hasDynamicFlowRates ? 8 : 5;
	const int strideDest = _hasDynamicFlowRates ? 10 : 7;
	const int numRows = conList.size() / strideSrc;
	std::vector<double> newList(numRows * strideDest, -1.0);

	double const* src = conList.data();
	double* dest = newList.data();
	for (int i = 0; i < numRows; ++i, src += strideSrc, dest += strideDest)
	{
		// Always copy unit operation indices
		dest[0] = src[0];
		dest[1] = src[1];

		// Skip ports (default filled with -1)

		// Copy the rest of the line
		std::copy_n(src + 2, strideDest - 4, dest + 4);
	}

	conList = std::move(newList);
}


/**
 * @brief Add default dynamic flow rates to connection list
 * @details Adds linear, quadratic, and cubic flow rate coefficients to the list
 *          The list is overwritten, that is, all pointers and iterators are invalidated.
 * @param [in,out] conList On entry, list of connections without dynamic flow rates;
 *                         on exit, list of connections including dynamic flow rates
 */
void ModelSystem::addDefaultDynamicFlowRatesToConnectionList(std::vector<double>& conList) const
{
	const int numRows = conList.size() / 7;
	std::vector<double> newList(numRows * 10, 0.0);

	double const* src = conList.data();
	double* dest = newList.data();
	for (int i = 0; i < numRows; ++i, src += 7, dest += 10)
	{
		std::copy_n(src, 7, dest);
	}

	conList = std::move(newList);
}

bool ModelSystem::configureModel(IParameterProvider& paramProvider, unsigned int unitOpIdx)
{
	IUnitOperation* const model = getUnitOperationModel(unitOpIdx);
	if (!model)
		return false;

	return model->configure(paramProvider);
}

void ModelSystem::readLinearSolutionMode(IParameterProvider& paramProvider)
{
	// Default: automatic
	_linearSolutionMode = 0;

	// Override default by user option
	if (paramProvider.exists("LINEAR_SOLUTION_MODE"))
		_linearSolutionMode = paramProvider.getInt("LINEAR_SOLUTION_MODE");
}

/**
 * @brief Checks the given unit operation connection list and reformats it
 * @details Throws an exception if something is incorrect. Reformats the connection list by
 *          substituting unit operation IDs with local indices.
 * @param [in] conn Matrix with 7 columns holding all connections. The matrix is expected
 *             to be in row-major storage format.
 * @param [out] connOnly Matrix with 6 columns holding all connections. While @p conn contains
 *              connection indices and flow rate, this matrix only holds connection indices.
 *              It is assumed to be pre-allocated (same number of rows as @p conn). The unit
 *              operation IDs are substituted by the corresponding indices of the unit operations
 *              in the local _models vector.
 * @param [out] flowRates Vector with flow rates for each connection in the list. It is assumed
 *              to be pre-allocated (same number of rows as @p conn).
 * @param [out] flowRatesLin Vector with linear flow rate coefficients for each connection in the list. It is assumed
 *              to be pre-allocated (same number of rows as @p conn).
 * @param [out] flowRatesQuad Vector with quadratic flow rate coefficients for each connection in the list. It is assumed
 *              to be pre-allocated (same number of rows as @p conn).
 * @param [out] flowRatesCub Vector with cubic flow rate coefficients for each connection in the list. It is assumed
 *              to be pre-allocated (same number of rows as @p conn).
 * @param [in] idxSwitch Index of the valve switch that corresponds to this connection list
 */
void ModelSystem::checkConnectionList(const std::vector<double>& conn, std::vector<int>& connOnly, std::vector<double>& flowRates, std::vector<double>& flowRatesLin, std::vector<double>& flowRatesQuad, std::vector<double>& flowRatesCub, unsigned int idxSwitch) const
{
	std::vector<double> totalInflow(_models.size(), 0.0);
	std::vector<double> totalInflowLin(_models.size(), 0.0);
	std::vector<double> totalInflowQuad(_models.size(), 0.0);
	std::vector<double> totalInflowCub(_models.size(), 0.0);
	std::vector<double> totalOutflow(_models.size(), 0.0);
	std::vector<double> totalOutflowLin(_models.size(), 0.0);
	std::vector<double> totalOutflowQuad(_models.size(), 0.0);
	std::vector<double> totalOutflowCub(_models.size(), 0.0);

	for (std::size_t i = 0; i < conn.size() / 10; ++i)
	{
		// Extract current connection
		int uoSource = static_cast<int>(conn[10*i]);
		int uoDest = static_cast<int>(conn[10*i+1]);
		const int portSource = static_cast<int>(conn[10*i+2]);
		const int portDest = static_cast<int>(conn[10*i+3]);
		const int compSource = static_cast<int>(conn[10*i+4]);
		const int compDest = static_cast<int>(conn[10*i+5]);
		double fr = conn[10*i + 6];
		double frLin = conn[10*i + 7];
		double frQuad = conn[10*i + 8];
		double frCub = conn[10*i + 9];

		if (uoSource < 0)
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Source unit operation id has to be at least 0 in connection");
		if (uoDest < 0)
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Destination unit operation id has to be at least 0 in connection");

		// Convert to index
		uoSource = indexOfUnitOp(_models, uoSource);
		uoDest = indexOfUnitOp(_models, uoDest);

		if (static_cast<unsigned int>(uoSource) >= _models.size())
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Source unit operation id " + std::to_string(uoSource) + " not found in connection");
		if (static_cast<unsigned int>(uoDest) >= _models.size())
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Destination unit operation id " + std::to_string(uoDest) + " not found in connection");

		// Check if unit operations have inlets and outlets
		if (!_models[uoSource]->hasOutlet())
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Source unit operation " + std::to_string(_models[uoSource]->unitOperationId()) + " does not have an outlet");
		if (!_models[uoDest]->hasInlet())
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Source unit operation " + std::to_string(_models[uoDest]->unitOperationId()) + " does not have an inlet");

		// Check port indices
		if ((portSource >= 0) && (static_cast<unsigned int>(portSource) >= _models[uoSource]->numOutletPorts()))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Source port index (" + std::to_string(portSource) + ") exceeds number of outlet ports (" + std::to_string(_models[uoSource]->numOutletPorts()) + ")");
		if ((portDest >= 0) && (static_cast<unsigned int>(portDest) >= _models[uoDest]->numInletPorts()))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Destination port index (" + std::to_string(portDest) + ") exceeds number of inlet ports (" + std::to_string(_models[uoDest]->numInletPorts()) + ")");

		if (((portSource < 0) && (portDest >= 0)) || ((portSource >= 0) && (portDest < 0)))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Only source or destination (not both) are set to connect all ports in connection from unit " 
				+ std::to_string(_models[uoSource]->unitOperationId()) + " to " + std::to_string(_models[uoDest]->unitOperationId()));

		if ((portSource < 0) && (portDest < 0) && (_models[uoSource]->numOutletPorts() != _models[uoDest]->numInletPorts()))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Number of ports not equal when connecting all ports from unit " 
				+ std::to_string(_models[uoSource]->unitOperationId()) + " (" + std::to_string(_models[uoSource]->numOutletPorts()) + " ports) to " + std::to_string(_models[uoDest]->unitOperationId()) + " (" + std::to_string(_models[uoDest]->numInletPorts()) + " ports)");

		// Check component indices
		if ((compSource >= 0) && (static_cast<unsigned int>(compSource) >= _models[uoSource]->numComponents()))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Source component index (" + std::to_string(compSource) + ") exceeds number of components (" + std::to_string(_models[uoSource]->numComponents()) + ")");
		if ((compDest >= 0) && (static_cast<unsigned int>(compDest) >= _models[uoDest]->numComponents()))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Destination component index (" + std::to_string(compDest) + ") exceeds number of components (" + std::to_string(_models[uoDest]->numComponents()) + ")");

		if (((compSource < 0) && (compDest >= 0)) || ((compSource >= 0) && (compDest < 0)))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Only source or destination (not both) are set to connect all components in connection from unit " 
				+ std::to_string(_models[uoSource]->unitOperationId()) + " to " + std::to_string(_models[uoDest]->unitOperationId()));

		if ((compSource < 0) && (compDest < 0) && (_models[uoSource]->numComponents() != _models[uoDest]->numComponents()))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + " row " + std::to_string(i) + "): Number of components not equal when connecting all components from unit " 
				+ std::to_string(_models[uoSource]->unitOperationId()) + " to " + std::to_string(_models[uoDest]->unitOperationId()));
	
		// Add connection to index matrix
		connOnly[6*i] = uoSource;
		connOnly[6*i+1] = uoDest;
		connOnly[6*i+2] = portSource;
		connOnly[6*i+3] = portDest;
		connOnly[6*i+4] = compSource;
		connOnly[6*i+5] = compDest;

		// Add flow rate of connection to balance

		// Check if such a connection has occurred before (for a different component)
		bool found = false;
		for (unsigned int j = 0; j < i; ++j)
		{
			if ((conn[10*j] == uoSource) && (uoDest == conn[10*j+1]) && (conn[10*j+2] == portSource) && (portDest == conn[10*j+3]))
			{
				// Take flow rate that appears first
				fr = conn[10*j+6];
				frLin = conn[10*j+7];
				frQuad = conn[10*j+8];
				frCub = conn[10*j+9];
				found = true;
				break;
			}
		}

		// Total flow rates: Only add flow rate once (not for each component)
		if (!found)
		{
			// Add flow rates to balance
			totalInflow[uoDest] += fr;
			totalOutflow[uoSource] += fr;
			totalInflowLin[uoDest] += frLin;
			totalOutflowLin[uoSource] += frLin;
			totalInflowQuad[uoDest] += frQuad;
			totalOutflowQuad[uoSource] += frQuad;
			totalInflowCub[uoDest] += frCub;
			totalOutflowCub[uoSource] += frCub;
		}

		// Add flow rate to list
		flowRates[i] = fr;
		flowRatesLin[i] = frLin;
		flowRatesQuad[i] = frQuad;
		flowRatesCub[i] = frCub;
	}

	// Check flow rate balance
	for (std::size_t i = 0; i < _models.size(); ++i)
	{
		// Unit operations with only one port (inlet or outlet) do not need to balance their flows
		if ((totalInflow[i] >= 0.0) && (totalOutflow[i] == 0.0) && _models[i]->hasInlet() && !_models[i]->hasOutlet())
			continue;
		if ((totalInflow[i] == 0.0) && (totalOutflow[i] >= 0.0) && !_models[i]->hasInlet() && _models[i]->hasOutlet())
			continue;

		// Terminal unit operations do not need to balance their flows
		if ((totalOutflow[i] >= 0.0) && isTerminal(connOnly.data(), connOnly.size() / 6, i))
			continue;

		// Unit operations that can accumulate cannot be checked
		if (_models[i]->canAccumulate())
			continue;

		// Check balance
		const double diff = std::abs(totalInflow[i] - totalOutflow[i]);
		if ((diff >= 1e-15) || (diff > 1e-15 * std::abs(totalOutflow[i])))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + "): Flow rate balance is not closed for unit operation " + std::to_string(i) + ", imbalanced by " + toSciString(diff));

		const double diffLin = std::abs(totalInflowLin[i] - totalOutflowLin[i]);
		if ((diffLin >= 1e-15) || (diffLin > 1e-15 * std::abs(totalOutflowLin[i])))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + "): Linear flow rate coefficient balance is not closed for unit operation " + std::to_string(i) + ", imbalanced by " + toSciString(diffLin));

		const double diffQuad = std::abs(totalInflowQuad[i] - totalOutflowQuad[i]);
		if ((diffQuad >= 1e-15) || (diffQuad > 1e-15 * std::abs(totalOutflowQuad[i])))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + "): Quadratic flow rate coefficient balance is not closed for unit operation " + std::to_string(i) + ", imbalanced by " + toSciString(diffQuad));

		const double diffCub = std::abs(totalInflowCub[i] - totalOutflowCub[i]);
		if ((diffCub >= 1e-15) || (diffCub > 1e-15 * std::abs(totalOutflowCub[i])))
			throw InvalidParameterException("In CONNECTIONS matrix (switch " + std::to_string(idxSwitch) + "): Cubic flow rate coefficient balance is not closed for unit operation " + std::to_string(i) + ", imbalanced by " + toSciString(diffCub));
	}

	// TODO: Check for conflicting entries
	// TODO: Plausibility check of total connections
}

std::unordered_map<ParameterId, double> ModelSystem::getAllParameterValues() const
{
	std::unordered_map<ParameterId, double> data;
	std::transform(_parameters.begin(), _parameters.end(), std::inserter(data, data.end()),
	               [](const std::pair<const ParameterId, active*>& p) { return std::make_pair(p.first, static_cast<double>(*p.second)); });

	for (IUnitOperation* m : _models)
	{
		const std::unordered_map<ParameterId, double> localData = m->getAllParameterValues();
		for (const std::pair<const ParameterId, double>& val : localData)
			data[val.first] = val.second;
	}

	return data;
}

bool ModelSystem::hasParameter(const ParameterId& pId) const
{
	for (IUnitOperation* m : _models)
	{
		if ((m->unitOperationId() == pId.unitOperation) || (pId.unitOperation == UnitOpIndep))
		{
			if (m->hasParameter(pId))
				return true;
		}
	}
	return _parameters.find(pId) != _parameters.end();
}

double ModelSystem::getParameterDouble(const ParameterId& pId) const
{
	for (IUnitOperation* m : _models)
	{
		if ((m->unitOperationId() == pId.unitOperation) || (pId.unitOperation == UnitOpIndep))
		{
			if (m->hasParameter(pId))
			{
				const double val = m->getParameterDouble(pId);
				if (!std::isnan(val))
					return val;
			}
		}
	}
	const std::unordered_map<ParameterId, active*>::const_iterator it = _parameters.find(pId);
	if (it == _parameters.end())
		return std::numeric_limits<double>::quiet_NaN();

	return static_cast<double>(*it->second);
}

template <typename ParamType>
bool ModelSystem::setParameterImpl(const ParameterId& pId, const ParamType value)
{
	// Filter by unit operation ID
	for (IUnitOperation* m : _models)
	{
		if ((m->unitOperationId() == pId.unitOperation) || (pId.unitOperation == UnitOpIndep))
			return m->setParameter(pId, value);
	}
	return false;
}

bool ModelSystem::setParameter(const ParameterId& pId, int value)
{
	return setParameterImpl(pId, value);
}

bool ModelSystem::setParameter(const ParameterId& pId, double value)
{
	bool found = false;
	auto paramHandle = _parameters.find(pId);
	if (paramHandle != _parameters.end())
	{
		paramHandle->second->setValue(value);
		found = true;

		// Multiplex flow rate parameters
#if CADET_COMPILER_CXX_CONSTEXPR
		constexpr StringHash flowHash = hashString("CONNECTION");
#else
		const StringHash flowHash = hashString("CONNECTION");
#endif
		if (flowHash == pId.name)
		{
			// Find the index of the valve switch
			const auto it = std::find(_switchSectionIndex.begin(), _switchSectionIndex.end(), pId.section);
			if (it != _switchSectionIndex.end())
			{
				const unsigned int idxSwitch = std::distance(_switchSectionIndex.begin(), it);
				int const* ptrConn = _connections[idxSwitch];
				active* conRates = _flowRates[idxSwitch];

				// Set value for all flow rates of the same connection (except for components)
				for (unsigned int i = 0; i < _connections.sliceSize(idxSwitch) / 6; ++i, ptrConn += 6, ++conRates)
				{
					if ((ptrConn[2] != pId.component) || (ptrConn[3] != pId.particleType) || (ptrConn[0] != pId.boundState) || (ptrConn[1] != pId.reaction))
						continue;

					conRates->setValue(value);
				}
			}
		}
	}

	return setParameterImpl(pId, value) || found;
}

bool ModelSystem::setParameter(const ParameterId& pId, bool value)
{
	return setParameterImpl(pId, value);
}

void ModelSystem::setSensitiveParameterValue(const ParameterId& pId, double value)
{
	if (pId.unitOperation == UnitOpIndep)
	{
		// Handle flow rates
		auto paramHandle = _parameters.find(pId);
		if ((paramHandle != _parameters.end()) && contains(_sensParams, paramHandle->second))
		{
			paramHandle->second->setValue(value);

			// Multiplex flow rate parameters
#if CADET_COMPILER_CXX_CONSTEXPR
			constexpr StringHash flowHash = hashString("CONNECTION");
#else
			const StringHash flowHash = hashString("CONNECTION");
#endif
			if (flowHash == pId.name)
			{
				// Find the index of the valve switch
				const auto it = std::find(_switchSectionIndex.begin(), _switchSectionIndex.end(), pId.section);
				if (it != _switchSectionIndex.end())
				{
					const unsigned int idxSwitch = std::distance(_switchSectionIndex.begin(), it);
					int const* ptrConn = _connections[idxSwitch];
					active* conRates = _flowRates[idxSwitch];

					// Set value for all flow rates of the same connection (except for components)
					for (unsigned int i = 0; i < _connections.sliceSize(idxSwitch) / 6; ++i, ptrConn += 6, ++conRates)
					{
						if ((ptrConn[2] != pId.component) || (ptrConn[3] != pId.particleType) || (ptrConn[0] != pId.boundState) || (ptrConn[1] != pId.reaction))
							continue;

						conRates->setValue(value);
					}
				}
			}
		}
	}

	// Filter by unit operation ID
	for (IUnitOperation* m : _models)
	{
		if ((m->unitOperationId() == pId.unitOperation) || (pId.unitOperation == UnitOpIndep))
		{
			m->setSensitiveParameterValue(pId, value);
		}
	}
}

bool ModelSystem::setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue)
{
	bool found = false;

	// Check own parameters
	if (pId.unitOperation == UnitOpIndep)
	{
		auto paramHandle = _parameters.find(pId);
		if (paramHandle != _parameters.end())
		{
			LOG(Debug) << "Found parameter " << pId << " in ModelSystem: Dir " << adDirection << " is set to " << adValue;

			// Register parameter and set AD seed / direction
			_sensParams.insert(paramHandle->second);
			paramHandle->second->setADValue(adDirection, adValue);

			// Multiplex flow rate parameters
#if CADET_COMPILER_CXX_CONSTEXPR
			constexpr StringHash flowHash = hashString("CONNECTION");
#else
			const StringHash flowHash = hashString("CONNECTION");
#endif
			if (flowHash == pId.name)
			{
				// Find the index of the valve switch
				const auto it = std::find(_switchSectionIndex.begin(), _switchSectionIndex.end(), pId.section);
				if (it != _switchSectionIndex.end())
				{
					const unsigned int idxSwitch = std::distance(_switchSectionIndex.begin(), it);
					int const* ptrConn = _connections[idxSwitch];
					active* conRates = _flowRates[idxSwitch];

					// Set AD direction for all flow rates of the same connection (except for components)
					for (unsigned int i = 0; i < _connections.sliceSize(idxSwitch) / 6; ++i, ptrConn += 6, ++conRates)
					{
						if ((ptrConn[2] != pId.component) || (ptrConn[3] != pId.particleType) || (ptrConn[0] != pId.boundState) || (ptrConn[1] != pId.reaction))
							continue;

						conRates->setADValue(adDirection, adValue);
					}
				}
			}

			found = true;
		}
	}

	// Filter by unit operation ID
	for (IUnitOperation* m : _models)
	{
		if ((m->unitOperationId() == pId.unitOperation) || (pId.unitOperation == UnitOpIndep))
		{
			found = m->setSensitiveParameter(pId, adDirection, adValue) || found;
		}
	}
	return found;
}

void ModelSystem::clearSensParams()
{
	// Remove AD directions from parameters
	for (auto sp : _sensParams)
		sp->setADValue(0.0);

	_sensParams.clear();

	// Propagate call to models
	for (IUnitOperation* m : _models)
		m->clearSensParams();
}

void ModelSystem::reportSolution(ISolutionRecorder& recorder, double const* const solution) const
{
	for (std::size_t i = 0; i < _models.size(); ++i)
	{
		IUnitOperation* const m = _models[i];
		m->reportSolution(recorder, solution + _dofOffset[i]);
	}
}

void ModelSystem::reportSolutionStructure(ISolutionRecorder& recorder) const
{
	for (IUnitOperation* m : _models)
		m->reportSolutionStructure(recorder);
}

unsigned int ModelSystem::requiredADdirs() const CADET_NOEXCEPT
{
	// Take maximum of required AD directions since each unit operation is (locally) independent from the rest
	unsigned int dirs = 0;
	for (IUnitOperation* m : _models)
		dirs = std::max(dirs, m->requiredADdirs());
	return dirs;
}

void ModelSystem::prepareADvectors(const AdJacobianParams& adJac) const
{
	// Early out if AD is disabled
	if (!adJac.adY)
		return;

	for (std::size_t i = 0; i < _models.size(); ++i)
	{
		if (_models[i]->usesAD())
		{
			const unsigned int offset = _dofOffset[i];
			_models[i]->prepareADvectors(applyOffset(adJac, offset));
		}
	}
}

void ModelSystem::setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections)
{
	for (IUnitOperation* m : _models)
		m->setSectionTimes(secTimes, secContinuity, nSections);	

	for (IExternalFunction* extFun : _extFunctions)
		extFun->setSectionTimes(secTimes, secContinuity, nSections);
}

/**
 * @brief Calculates error tolerances for additional coupling DOFs
 * @details ModelSystem uses additional DOFs to decouple a system of unit operations for parallelization.
 *          These additional DOFs don't get an error tolerance from the user because he shouldn't be
 *          aware of those (implementation detail). This function is responsible for calculating error
 *          tolerances for these additional coupling DOFs.
 * 
 * @param [in] errorTol Pointer to array of error tolerances for system without coupling DOFs
 * @param [in] errorTolLength Length of @p errorTol array
 * @return Vector with error tolerances for additional coupling DOFs
 */
std::vector<double> ModelSystem::calculateErrorTolsForAdditionalDofs(double const* errorTol, unsigned int errorTolLength)
{
	// Return empty vector since we don't have coupling DOFs, yet
	return std::vector<double>(0, 0.0);
}

void ModelSystem::expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut)
{
	// TODO: Adjust indexing / offset of vectors
	// TODO: Handle connection of unit operations
	for (IUnitOperation* m : _models)
	{
		m->expandErrorTol(errorSpec, errorSpecSize, expandOut);
	}
}

void ModelSystem::setupParallelization(unsigned int numThreads)
{
	unsigned int tlsSize = 0;
	for (IUnitOperation const* m : _models)
		tlsSize = std::max(tlsSize, m->threadLocalMemorySize());

	_threadLocalStorage.resize(numThreads, tlsSize);
}

}  // namespace model

}  // namespace cadet
