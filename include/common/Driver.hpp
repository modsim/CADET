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
 * Provides a driver for configuring and running a CADET simulation
 */

#ifndef CADET_DRIVER_HPP_
#define CADET_DRIVER_HPP_

#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

#include "cadet/cadet.hpp"

#include "common/SolutionRecorderImpl.hpp"

namespace cadet
{

namespace detail
{

template <class ParamProvider_t, class StorageConfig_t>
void readDataOutputConfig(ParamProvider_t& pp, StorageConfig_t& cfg, const std::string& dataType)
{
	// Configure data output
	if (pp.exists("WRITE_" + dataType + "_BULK"))
		cfg.storeBulk = pp.getBool("WRITE_" + dataType + "_BULK");
	else
	{
		if (pp.exists("WRITE_" + dataType + "_COLUMN"))
			cfg.storeBulk = pp.getBool("WRITE_" + dataType + "_COLUMN");
		else
			cfg.storeBulk = false;
	}

	if (pp.exists("WRITE_" + dataType + "_PARTICLE"))
		cfg.storeParticle = pp.getBool("WRITE_" + dataType + "_PARTICLE");
	else
		cfg.storeParticle = false;

	if (pp.exists("WRITE_" + dataType + "_SOLID"))
		cfg.storeSolid = pp.getBool("WRITE_" + dataType + "_SOLID");
	else
		cfg.storeSolid = false;

	if (pp.exists("WRITE_" + dataType + "_FLUX"))
		cfg.storeFlux = pp.getBool("WRITE_" + dataType + "_FLUX");
	else
		cfg.storeFlux = false;

/*
	if (pp.exists("WRITE_" + dataType + "_ALL"))
	{
		cfg.storeColumn = pp.getBool("WRITE_" + dataType + "_ALL");
		cfg.storeParticle = cfg.storeColumn;
		cfg.storeFlux = cfg.storeColumn;
	}
*/

	if (pp.exists("WRITE_" + dataType + "_INLET"))
		cfg.storeInlet = pp.getBool("WRITE_" + dataType + "_INLET");
	else
	{
		if (pp.exists("WRITE_" + dataType + "_COLUMN_INLET"))
			cfg.storeInlet = pp.getBool("WRITE_" + dataType + "_COLUMN_INLET");
		else
			cfg.storeInlet = false;
	}

	if (pp.exists("WRITE_" + dataType + "_OUTLET"))
		cfg.storeOutlet = pp.getBool("WRITE_" + dataType + "_OUTLET");
	else
	{
		if (pp.exists("WRITE_" + dataType + "_COLUMN_OUTLET"))
			cfg.storeOutlet = pp.getBool("WRITE_" + dataType + "_COLUMN_OUTLET");
		else
			cfg.storeOutlet = false;		
	}

	if (pp.exists("WRITE_" + dataType + "_VOLUME"))
		cfg.storeVolume = pp.getBool("WRITE_" + dataType + "_VOLUME");
	else
		cfg.storeVolume = false;

	if (pp.exists("WRITE_" + dataType + "_SMOOTHNESS_INDICATOR"))
		cfg.storeSmoothnessIndicator = pp.getBool("WRITE_" + dataType + "_SMOOTHNESS_INDICATOR");
	else
		cfg.storeSmoothnessIndicator = false;
}

template <class ParamProvider_t>
void configureSystemRecorder(cadet::InternalStorageSystemRecorder& recorder, ParamProvider_t& pp, unsigned int maxUnitOperationId)
{

#ifdef MATLAB_MEX_FILE

	// Do not split components into multiple datasets when using MEX interface
	const bool splitComponents = false;
	// Do not split ports into multiple datasets when using MEX interface
	const bool splitPorts = false;
	// Treat single port unit operations as multi port unit operations when using MEX interface
	const bool singleAsMultiPort = true;

#else

	bool splitComponents = true;
	if (pp.exists("SPLIT_COMPONENTS_DATA"))
		splitComponents = pp.getBool("SPLIT_COMPONENTS_DATA");

	bool splitPorts = true;
	if (pp.exists("SPLIT_PORTS_DATA"))
		splitPorts = pp.getBool("SPLIT_PORTS_DATA");

	bool singleAsMultiPort = false;
	if (pp.exists("SINGLE_AS_MULTI_PORT"))
		singleAsMultiPort = pp.getBool("SINGLE_AS_MULTI_PORT");

#endif

	recorder.deleteRecorders();

	cadet::InternalStorageUnitOpRecorder::StorageConfig cfg;
	std::ostringstream oss;
	for (unsigned int i = 0; i <= maxUnitOperationId; ++i)
	{
		oss.str("");
		oss << "unit_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
		if (!pp.exists(oss.str()))
			continue;

		// Create and configure unit operation
		pp.pushScope(oss.str());

		cadet::InternalStorageUnitOpRecorder* const subRec = new cadet::InternalStorageUnitOpRecorder(i);
		
		readDataOutputConfig(pp, cfg, "SOLUTION");
		subRec->solutionConfig(cfg);

		readDataOutputConfig(pp, cfg, "SOLDOT");
		subRec->solutionDotConfig(cfg);

		readDataOutputConfig(pp, cfg, "SENS");
		subRec->sensitivityConfig(cfg);

		readDataOutputConfig(pp, cfg, "SENSDOT");
		subRec->sensitivityDotConfig(cfg);

		if (pp.exists("WRITE_COORDINATES"))
			subRec->storeCoordinates(pp.getBool("WRITE_COORDINATES"));

		subRec->splitComponents(splitComponents);
		subRec->splitPorts(splitPorts);
		subRec->treatSingleAsMultiPortUnitOps(singleAsMultiPort);
		pp.popScope();

		recorder.addRecorder(subRec);
	}
}

template <class ParamProvider_t>
void readSensitivityInitialState(ParamProvider_t& pp, const char* prefix, std::vector<double const*>& out, std::vector<std::vector<double>>& data)
{
	unsigned int i = 0;
	std::ostringstream oss;
	oss << prefix << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
	while(pp.exists(oss.str()))
	{
		data.push_back(pp.getDoubleArray(oss.str()));
		out.push_back(data.back().data());

		++i;
		oss.str("");
		oss << prefix << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
	}
}

} // namespace detail

/**
 * @brief Driver that provides building blocks for working with a cadet::ISimulator
 * @details Simplifies and summarizes steps necessary to construct and configure models,
 *          run simulations, and return the results.
 */
class Driver
{
public:
	Driver() : _sim(nullptr), _builder(nullptr), _storage(nullptr), _writeLastState(false), _writeLastStateSens(false)
	{
		_builder = cadetCreateModelBuilder();
	}

	~Driver() CADET_NOEXCEPT
	{
		delete _storage;

		if (_sim)
			cadetDestroySimulator(_sim);
		
		cadetDestroyModelBuilder(_builder);
	}

	Driver(Driver&&) = default;

	/**
	 * @brief Clears all objects from memory and starts fresh as if just created
	 * @details Deletes the data storage, destroys the simulator, and obtains a new
	 *          model builder.
	 */
	void clear()
	{
		delete _storage;
		_storage = nullptr;

		if (_sim)
		{
			cadetDestroySimulator(_sim);
			_sim = nullptr;
		}
		
		cadetDestroyModelBuilder(_builder);
		_builder = cadetCreateModelBuilder();
	}

	/**
	 * @brief Builds and configures a simulator and a model
	 * @details Creates a new simulator (destroying any already existing ones) and
	 *          creates a new model. Both are configured and on exit ready for time
	 *          integration. All stored results are wiped out.
	 * @param [in] pp Implementation of cadet::IParameterProvider used as input
	 * @tparam ParamProvider_t Type of the parameter provider
	 */
	template <typename ParamProvider_t>
	void configure(ParamProvider_t& pp)
	{
		// Create storage
		delete _storage;
		_storage = new cadet::InternalStorageSystemRecorder();

		// Create simulator
		if (_sim)
			cadetDestroySimulator(_sim);

		_sim = cadetCreateSimulator();

		// Configure main solver parameters
		pp.pushScope("solver");
		_sim->configure(pp);
		
		// Configure section times
		std::vector<double> secTimes;
		std::vector<bool> secCont;
		extractSectionTimes(pp, secTimes, secCont);

		pp.popScope(); // solver scope

		pp.pushScope("model");

		// Create and configure model
		cadet::IModelSystem* model = _builder->createSystem(pp);
		if (!model)
			throw InvalidParameterException("Invalid model");

		// Hand model over to simulator
		_sim->initializeModel(*model);
		_sim->setSectionTimes(secTimes, secCont);

		// Specify initial values
		if (pp.exists("INIT_STATE_Y"))
		{
			const std::vector<double> initY = pp.getDoubleArray("INIT_STATE_Y");
			if (initY.size() != _sim->numDofs())
			{
				throw InvalidParameterException("Length of INIT_STATE_Y should be equal to NDOF");
			}

			if (pp.exists("INIT_STATE_YDOT"))
			{
				const std::vector<double> initYdot = pp.getDoubleArray("INIT_STATE_YDOT");
				if (initYdot.size() != _sim->numDofs())
				{
					throw InvalidParameterException("Length of INIT_STATE_YDOT should be equal to NDOF");
				}
				_sim->applyInitialCondition(initY.data(), initYdot.data());
			}
			else
			{
				_sim->applyInitialCondition(initY.data());
			}
		}
		else
		{
			_sim->setInitialCondition(pp);
			_sim->applyInitialCondition();
		}

		// Read initial values of sensitivities
		std::vector<double const*> initSensY;
		std::vector<double const*> initSensYdot;
		std::vector<std::vector<double>> initDataSensY;
		initDataSensY.reserve(10);
		std::vector<std::vector<double>> initDataSensYdot;
		initDataSensYdot.reserve(10);
		detail::readSensitivityInitialState(pp, "INIT_STATE_SENSY_", initSensY, initDataSensY);
		detail::readSensitivityInitialState(pp, "INIT_STATE_SENSYDOT_", initSensYdot, initDataSensYdot);

		pp.popScope(); // scope model

		// Configure data output (wait for sensitivities before sending storage to simulator)
		setReturnConfiguration(pp, false);

		// Model should be fully configured and ready to run at this point

		// Read and configure parameters
		unsigned int numSens = 0;
		if (pp.exists("sensitivity"))
		{
			pp.pushScope("sensitivity");

			numSens = static_cast<unsigned int>(pp.getInt("NSENS"));
			const std::string sensMethod = pp.getString("SENS_METHOD");

			std::ostringstream oss;
			for (unsigned int i = 0; i < numSens; ++i)
			{
				oss.str("");
				oss << "param_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;

				pp.pushScope(oss.str());

				const std::vector<std::string> sensName = pp.getStringArray("SENS_NAME");
				const std::vector<int> sensUnit = pp.getIntArray("SENS_UNIT");
				const std::vector<int> sensComp = pp.getIntArray("SENS_COMP");
				const std::vector<int> sensReaction = pp.getIntArray("SENS_REACTION");
				const std::vector<int> sensSection = pp.getIntArray("SENS_SECTION");
				const std::vector<int> sensBoundState = pp.getIntArray("SENS_BOUNDPHASE");
				const std::vector<int> sensParType = pp.getIntArray("SENS_PARTYPE");

				// Convert to ParameterIds
				std::vector<cadet::ParameterId> sensParams;
				sensParams.reserve(sensName.size());
				for (std::size_t i = 0; i < sensName.size(); ++i)
					sensParams.push_back(cadet::makeParamId(sensName[i], sensUnit[i], sensComp[i], sensParType[i], sensBoundState[i], sensReaction[i], sensSection[i]));

				double sensTol = 1e-05;
				if (pp.exists("SENS_ABSTOL"))
					sensTol = pp.getDouble("SENS_ABSTOL");

				// Read factors, but default to 1.0 if none are given
				std::vector<double> sensFactor;
				if (pp.exists("SENS_FACTOR"))
					sensFactor = pp.getDoubleArray("SENS_FACTOR");
				else
					sensFactor.resize(sensParams.size(), 1.0);

				_sim->setSensitiveParameter(sensParams.data(), sensFactor.data(), sensParams.size(), sensTol);
				pp.popScope();
			}

			pp.popScope(); // scope sensitivity

			if (numSens > 0)
			{
				if ((initSensY.size() >= numSens) && (initSensYdot.size() >= numSens))
					_sim->initializeFwdSensitivities(initSensY.data(), initSensYdot.data());
				else
					_sim->initializeFwdSensitivities();
			}
		}

		// Set storage for solution
		_sim->setSolutionRecorder(_storage);
	}

	/**
	 * @brief Sets initial conditions from the given parameter provider
	 * @details Assumes that the simulator is already configured
	 * @param [in] pp Implementation of cadet::IParameterProvider used as input
	 * @tparam ParamProvider_t Type of the parameter provider
	 */
	template <typename ParamProvider_t>
	void setInitialCondition(ParamProvider_t& pp)
	{
		if (pp.exists("INIT_STATE_Y") && pp.exists("INIT_STATE_YDOT"))
		{
			const std::vector<double> initY = pp.getDoubleArray("INIT_STATE_Y");
			const std::vector<double> initYdot = pp.getDoubleArray("INIT_STATE_YDOT");
			if (initY.size() >= _sim->numDofs())
			{
				if (initYdot.size() >= _sim->numDofs())
					_sim->applyInitialCondition(initY.data(), initYdot.data());
				else
					_sim->applyInitialCondition(initY.data());
			}
		}
		else
		{
			_sim->setInitialCondition(pp);
			_sim->applyInitialCondition();
		}

		if (_sim->numSensParams() > 0)
		{
			// Read initial values of sensitivities
			std::vector<double const*> initSensY;
			std::vector<double const*> initSensYdot;
			std::vector<std::vector<double>> initDataSensY;
			initDataSensY.reserve(10);
			std::vector<std::vector<double>> initDataSensYdot;
			initDataSensYdot.reserve(10);
			detail::readSensitivityInitialState(pp, "INIT_STATE_SENSY_", initSensY, initDataSensY);
			detail::readSensitivityInitialState(pp, "INIT_STATE_SENSYDOT_", initSensYdot, initDataSensYdot);

			if ((initSensY.size() >= _sim->numSensParams()) && (initSensYdot.size() >= _sim->numSensParams()))
				_sim->initializeFwdSensitivities(initSensY.data(), initSensYdot.data());
			else
				_sim->initializeFwdSensitivities();
		}
	}

	/**
	 * @brief Sets section times and section continuity from the given parameter provider
	 * @details Assumes that the simulator is already configured
	 * @param [in] pp Implementation of cadet::IParameterProvider used as input
	 * @tparam ParamProvider_t Type of the parameter provider
	 */
	template <typename ParamProvider_t>
	void setSectionTimes(ParamProvider_t& pp)
	{
		std::vector<double> secTimes;
		std::vector<bool> secCont;
		extractSectionTimes(pp, secTimes, secCont);
		_sim->setSectionTimes(secTimes, secCont);
	}

	/**
	 * @brief Reads the return configuration of the ModelSystem and its UnitOperation models from the given parameter provider
	 * @details Does nothing if the simulator or model have not been configured yet.
	 * @param [in] pp Implementation of cadet::IParameterProvider used as input
	 * @param [in] applyInSimulator Determines whether the storage is set in the simulator (@c true), or not (@c false)
	 * @tparam ParamProvider_t Type of the parameter provider
	 */
	template <typename ParamProvider_t>
	void setReturnConfiguration(ParamProvider_t& pp, bool applyInSimulator)
	{
		if (!_storage || !_sim || !_sim->model())
			return;

		// Configure data output
		pp.pushScope("return");

		detail::configureSystemRecorder(*_storage, pp, _sim->model()->maxUnitOperationId());

		if (pp.exists("WRITE_SOLUTION_TIMES"))
			_storage->storeTime(pp.getBool("WRITE_SOLUTION_TIMES"));
		else
			_storage->storeTime(true);
		
		if (pp.exists("WRITE_SOLUTION_LAST"))
			_writeLastState = pp.getBool("WRITE_SOLUTION_LAST");
		else
			_writeLastState = false;

		// Adding a new output variable
		std::ostringstream oss;
		for (int i = 0; i <= _sim->model()->maxUnitOperationId(); ++i)
		{
			oss.str("");
			oss << "unit_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
			if (pp.exists(oss.str()))
			{
				pp.pushScope(oss.str());

				if (pp.exists("WRITE_SOLUTION_LAST_UNIT"))
				{
					const bool writeLastStateUnit = pp.getBool("WRITE_SOLUTION_LAST_UNIT");
					if (writeLastStateUnit)
						_writeLastStateUnitId.push_back(i);
				}

				pp.popScope();
			}
		}

		if (pp.exists("WRITE_SENS_LAST"))
			_writeLastStateSens = pp.getBool("WRITE_SENS_LAST");
		else
			_writeLastStateSens = false;

		pp.popScope(); // scope return

		if (applyInSimulator)
			_sim->setSolutionRecorder(_storage);
	}

	/**
	 * @brief Performs time integration
	 * @details The simulator has to be setup and configured for time integration.
	 */
	void run()
	{
		// Run simulation
		_sim->integrate();
	}

	/**
	 * @brief Writes the current results to the given writer
	 * @param [in] writer Writer to write to
	 * @tparam Writer_t Type of the writer
	 */
	template <typename Writer_t>
	void write(Writer_t& writer)
	{
		if (!_sim || !_storage)
			return;

		LOG(Debug) << "Writing " << _storage->numDataPoints() << " data points to file";

		writer.unlinkGroup("output");
		
		writer.extendibleFields(false);
		writer.compressFields(true);

		writer.pushGroup("output");

		if (_storage->anyUnitStoresCoordinates())
		{
			writer.pushGroup("coordinates");
			_storage->writeCoordinates(writer);
			writer.popGroup();
		}

		writer.pushGroup("solution");
		_storage->writeSolution(writer);
		writer.popGroup();

		if (_sim->numSensParams() > 0)
		{
			writer.pushGroup("sensitivity");
			_storage->writeSensitivity(writer);
			writer.popGroup();
		}

		if (_writeLastState)
		{
			unsigned int len = 0;
			double const* const lastY = _sim->getLastSolution(len);
			double const* const lastYdot = _sim->getLastSolutionDerivative(len);

			writer.vector("LAST_STATE_Y", len, lastY);
			writer.vector("LAST_STATE_YDOT", len, lastYdot);
		}

		std::ostringstream oss;
		for (int i = 0; i < _writeLastStateUnitId.size(); ++i)
		{
			unsigned int sliceStart;
			unsigned int sliceEnd;
			std::tie(sliceStart, sliceEnd) = _sim->model()->getModelStateOffsets(_writeLastStateUnitId[i]);

			oss.str("");
			oss << "unit_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << _writeLastStateUnitId[i];
			writer.pushGroup("solution");
			writer.pushGroup(oss.str());

			unsigned int len = 0;
			double const* const lastY = _sim->getLastSolution(len);
			double const* const lastYdot = _sim->getLastSolutionDerivative(len);

			writer.template vector<double>("LAST_STATE_Y", sliceEnd - sliceStart, lastY + sliceStart);
			writer.template vector<double>("LAST_STATE_YDOT", sliceEnd - sliceStart, lastYdot + sliceStart);

			writer.popGroup();
			writer.popGroup();
		}

		if (_writeLastStateSens)
		{

			unsigned int len = 0;
			const std::vector<double const*> lastY = _sim->getLastSensitivities(len);
			const std::vector<double const*> lastYdot = _sim->getLastSensitivityDerivatives(len);

			std::ostringstream oss;
			for (std::size_t i = 0; i < lastY.size(); ++i)
			{
				oss.str("");
				oss << "LAST_STATE_SENSY_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
				writer.vector(oss.str(), len, lastY[i]);

				oss.str("");
				oss << "LAST_STATE_SENSYDOT_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;
				writer.vector(oss.str(), len, lastYdot[i]);
			}
		}

		writer.popGroup();

		if (writer.exists("meta"))
		{
			writer.pushGroup("meta");
			if (writer.exists("CADET_VERSION"))
				writer.unlinkDataset("CADET_VERSION");
			if (writer.exists("CADET_COMMIT"))
				writer.unlinkDataset("CADET_COMMIT");
			if (writer.exists("CADET_BRANCH"))
				writer.unlinkDataset("CADET_BRANCH");
			if (writer.exists("TIME_SIM"))
				writer.unlinkDataset("TIME_SIM");
		}
		else
			writer.pushGroup("meta");

		// Make sure to write first (otherwise the /meta group will not be created)
		writer.scalar("CADET_VERSION", std::string(cadet::getLibraryVersion()));
		writer.scalar("CADET_COMMIT", std::string(cadet::getLibraryCommitHash()));
		writer.scalar("CADET_BRANCH", std::string(cadet::getLibraryBranchRefspec()));
		writer.scalar("TIME_SIM", _sim->lastSimulationDuration());

		if (!writer.exists("FILE_FORMAT"))
			writer.scalar("FILE_FORMAT", 40000);

		writer.popGroup();
	}

	/**
	 * @brief Removes all stored results
	 */
	inline void clearResults()
	{
		if (_storage)
			_storage->clear();
	}

	inline cadet::ISimulator* simulator() const CADET_NOEXCEPT { return _sim; }
	inline cadet::IModelBuilder* modelBuilder() const CADET_NOEXCEPT { return _builder; }
	inline cadet::IModelSystem* model() const { return _sim->model(); }

	inline void setWriteLastStateOfUnit(UnitOpIdx uid, bool writeLastStateUnit)
	{
		const std::vector<UnitOpIdx>::iterator it = std::find(_writeLastStateUnitId.begin(), _writeLastStateUnitId.end(), uid);
		if (writeLastStateUnit)
		{
			if (it == _writeLastStateUnitId.end())
			{
				// Unit not in list, so add it
				_writeLastStateUnitId.push_back(uid);
			}
		}
		else
		{
			if (it != _writeLastStateUnitId.end())
			{
				// Unit is listed, so remove it
				_writeLastStateUnitId.erase(it);
			}
		}
	}

	inline void setWriteLastState(bool writeLastState) CADET_NOEXCEPT { _writeLastState = writeLastState; }
	inline void setWriteLastStateSens(bool writeLastState) CADET_NOEXCEPT { _writeLastStateSens = writeLastState; }
	inline void setWriteSolutionTimes(bool solTimes) CADET_NOEXCEPT
	{
		if (_storage)
			_storage->storeTime(solTimes);
	}

	inline cadet::InternalStorageSystemRecorder* solution() CADET_NOEXCEPT { return _storage; }
	inline cadet::InternalStorageSystemRecorder const* solution() const CADET_NOEXCEPT { return _storage; }

protected:
	cadet::ISimulator* _sim; //!< Simulator owned by this driver
	cadet::IModelBuilder* _builder; //!< Model builder owned by this driver
	cadet::InternalStorageSystemRecorder* _storage; //!< Storage for results

	bool _writeLastState;
	std::vector<UnitOpIdx> _writeLastStateUnitId;
	bool _writeLastStateSens;

	/**
	 * @brief Sets section times and section continuity from the given parameter provider
	 * @details Assumes that the simulator is already configured
	 * @param [in] pp Implementation of cadet::IParameterProvider used as input
	 * @param [out] secTimes Vector with section times
	 * @param [out] secCont Vector indicating continuity of section transitions
	 * @tparam ParamProvider_t Type of the parameter provider
	 */
	template <typename ParamProvider_t>
	void extractSectionTimes(ParamProvider_t& pp, std::vector<double>& secTimes, std::vector<bool>& secCont)
	{
		pp.pushScope("sections");

		secTimes = pp.getDoubleArray("SECTION_TIMES");
		if (pp.exists("SECTION_CONTINUITY")) 
			secCont = pp.getBoolArray("SECTION_CONTINUITY");
		else
			secCont = std::vector<bool>(secTimes.size() - 2, false);

		pp.popScope(); // sections scope
	}

private:
	Driver(const Driver&) = delete;
};

} // namespace cadet

#endif  // CADET_DRIVER_HPP_
