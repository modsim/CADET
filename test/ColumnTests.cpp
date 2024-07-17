// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include <catch.hpp>
#include "Approx.hpp"
#include "cadet/cadet.hpp"

#define CADET_LOGGING_DISABLE
#include "Logging.hpp"

#include "ColumnTests.hpp"
#include "Utils.hpp"
#include "SimHelper.hpp"
#include "ModelBuilderImpl.hpp"
#include "common/Driver.hpp"
#include "Weno.hpp"
#include "linalg/Norms.hpp"
#include "SimulationTypes.hpp"
#include "ParallelSupport.hpp"

#include "JsonTestModels.hpp"
#include "JacobianHelper.hpp"
#include "UnitOperationTests.hpp"
#include "LoggingUtils.hpp"
#include "../include/io/hdf5/HDF5Reader.hpp"
#include "common/ParameterProviderImpl.hpp"

#include <cmath>
#include <functional>
#include <cstdint>

/**
 * @brief Returns the absolute path to the test/ folder of the project
 * @details Absolute path to the test/ folder of the project without trailing slash
 * @return Absolute path to the test/ folder
 */
const char* getTestDirectory();

namespace
{
	/**
	 * @brief Creates a runnable column model
	 * @details Creates a column model and configures it using the given IParameterProvider @p jpp.
	 * @param [in] mb ModelBuilder
	 * @param [in] jpp Configuration of the model
	 * @return Runnable column model
	 */
	inline cadet::IUnitOperation* createAndConfigureUnit(cadet::IModelBuilder& mb, cadet::JsonParameterProvider& jpp)
	{
		// Create a unit
		cadet::IModel* const iUnit = mb.createUnitOperation(jpp, 0);
		REQUIRE(nullptr != iUnit);

		cadet::IUnitOperation* const unit = reinterpret_cast<cadet::IUnitOperation*>(iUnit);

		// Configure
		cadet::ModelBuilder& temp = *reinterpret_cast<cadet::ModelBuilder*>(&mb);
		REQUIRE(unit->configureModelDiscretization(jpp, temp));
		REQUIRE(unit->configure(jpp));

		// Do some checks
		const unsigned int nComp = jpp.getInt("NCOMP");
		REQUIRE(unit->numComponents() == nComp);

		return unit;
	}
	/**
	 * @brief Creates a runnable column model with given discretization parameters
	 * @details Creates a column model and configures it using the given IParameterProvider @p jpp.
	 * @param [in] mb ModelBuilder
	 * @param [in] jpp Configuration of the model
	 * @param [in] disc discretization parameters
	 * @return Runnable column model
	 */
	inline cadet::IUnitOperation* createAndConfigureUnit(cadet::IModelBuilder& mb, cadet::JsonParameterProvider& jpp, cadet::test::column::DiscParams& disc)
	{
		// Set discretization parameters
		disc.setDisc(jpp);
		return createAndConfigureUnit(mb, jpp);
	}

	class FluxOffsetExtractionRecorder : public cadet::ISolutionRecorder
	{
	public:
		FluxOffsetExtractionRecorder() : _fluxOffset(0) { }
		virtual void clear() { }
		virtual void prepare(unsigned int numDofs, unsigned int numSens, unsigned int numTimesteps) { }
		virtual void notifyIntegrationStart(unsigned int numDofs, unsigned int numSens, unsigned int numTimesteps) { }
		virtual void beginTimestep(double t) { }
		virtual void beginUnitOperation(cadet::UnitOpIdx idx, const cadet::IModel& model, const cadet::ISolutionExporter& exporter) { }
		virtual void endUnitOperation() { }
		virtual void endTimestep() { }
		virtual void beginSolution() { }
		virtual void endSolution() { }
		virtual void beginSolutionDerivative() { }
		virtual void endSolutionDerivative() { }
		virtual void beginSensitivity(const cadet::ParameterId& pId, unsigned int sensIdx) { }
		virtual void endSensitivity(const cadet::ParameterId& pId, unsigned int sensIdx) { }
		virtual void beginSensitivityDerivative(const cadet::ParameterId& pId, unsigned int sensIdx) { }
		virtual void endSensitivityDerivative(const cadet::ParameterId& pId, unsigned int sensIdx) { }

		virtual void unitOperationStructure(cadet::UnitOpIdx idx, const cadet::IModel& model, const cadet::ISolutionExporter& exporter)
		{
			_fluxOffset = exporter.numComponents() * exporter.numInletPorts() + exporter.numMobilePhaseDofs();

			const unsigned int nParType = exporter.numParticleTypes();
			for (unsigned int i = 0; i < nParType; ++i)
				_fluxOffset += exporter.numParticleMobilePhaseDofs(i) + exporter.numSolidPhaseDofs(i);

			// Make sure this is correct
			const unsigned int nFluxes = exporter.numComponents() * exporter.numPrimaryCoordinates() * nParType * std::max(exporter.numSecondaryCoordinates(), 1u);
			REQUIRE(reinterpret_cast<cadet::IUnitOperation const*>(&model)->numDofs() - nFluxes == _fluxOffset);
		}

		inline unsigned int fluxOffset() const CADET_NOEXCEPT { return _fluxOffset; }
	protected:
		unsigned int _fluxOffset;
	};
}

namespace cadet
{

namespace test
{

namespace column
{

	void FVparams::setDisc(JsonParameterProvider& jpp, const std::string unitID) const {

		int level = 0;

		if (jpp.exists("model"))
		{
			jpp.pushScope("model");
			++level;
		}
		if (jpp.exists("unit_" + unitID))
		{
			jpp.pushScope("unit_" + unitID);
			++level;
		}

		std::string unitType = jpp.getString("UNIT_TYPE");

		jpp.pushScope("discretization");
		jpp.set("SPATIAL_METHOD", "FV");

		if (nAxCells)
			jpp.set("NCOL", nAxCells);

		if (nParCells)
			jpp.set("NPAR", nParCells);

		if (nRadCells)
		{
			if (unitType == "MULTI_CHANNEL_TRANSPORT")
			{
				jpp.popScope();
				jpp.set("NCHANNEL", nRadCells);
				jpp.pushScope("discretization");
			}
			else
				jpp.set("NRAD", nRadCells);
		}

		if (jpp.exists("weno"))
		{
			jpp.pushScope("weno");
			if (wenoOrder)
				jpp.set("WENO_ORDER", wenoOrder);

			jpp.popScope();
		}
		jpp.popScope();

		for (int l = 0; l < level; ++l)
			jpp.popScope();
	}

	void DGparams::setDisc(JsonParameterProvider& jpp, const std::string unitID) const {

		int level = 0;

		if (jpp.exists("model"))
		{
			jpp.pushScope("model");
			++level;
		}
		if (jpp.exists("unit_" + unitID))
		{
			jpp.pushScope("unit_" + unitID);
			++level;
		}

		jpp.pushScope("discretization");
		jpp.set("SPATIAL_METHOD", "DG");

		if (exactIntegration > -1)
			jpp.set("EXACT_INTEGRATION", exactIntegration);
		if (polyDeg)
			jpp.set("POLYDEG", polyDeg);
		if (nElem)
			jpp.set("NELEM", nElem);
		if (parNelem)
			jpp.set("PAR_NELEM", parNelem);
		if (parPolyDeg)
			jpp.set("PAR_POLYDEG", parPolyDeg);

		jpp.popScope();

		for (int l = 0; l < level; ++l)
			jpp.popScope();
	}

	/**
	 * @brief Reads reference chromatograms from a test data file
	 * @details The file format is as follows:
	 *          Number of data points (uint32)
	 *          Time points (array of doubles)
	 *          Chromatogram for dynamic binding (array of doubles)
	 *          Chromatogram for quasi-stationary binding (array of doubles)
	 */
	class ReferenceDataReader
	{
	public:
		ReferenceDataReader(const char* fileName) : _f(nullptr)
		{
			_f = std::fopen(fileName, "rb");
			std::fread(&_numElements, 4, 1, _f);
		}

		~ReferenceDataReader()
		{
			std::fclose(_f);
		}

		std::vector<double> time()
		{
			std::vector<double> v(_numElements, 0.0);
			std::fseek(_f, 4, SEEK_SET);
			std::fread(v.data(), 8, _numElements, _f);
			return v;
		}

		std::vector<double> analyticDynamic()
		{
			std::vector<double> v(_numElements, 0.0);
			std::fseek(_f, 4 + _numElements * 8, SEEK_SET);
			std::fread(v.data(), 8, _numElements, _f);
			return v;
		}

		std::vector<double> analyticQuasiStationary()
		{
			std::vector<double> v(_numElements, 0.0);
			std::fseek(_f, 4 + 2 * _numElements * 8, SEEK_SET);
			std::fread(v.data(), 8, _numElements, _f);
			return v;
		}

	private:
		std::FILE* _f;
		uint32_t _numElements;
	};

	void setNumAxialCells(cadet::JsonParameterProvider& jpp, unsigned int nCol, std::string unitID)
	{
		int level = 0;

		if (jpp.exists("model"))
		{
			jpp.pushScope("model");
			++level;
		}
		if (jpp.exists("unit_"+ unitID))
		{
			jpp.pushScope("unit_" + unitID);
			++level;
		}

		jpp.pushScope("discretization");

		jpp.set("NCOL", static_cast<int>(nCol));

		jpp.popScope();
	
		for (int l = 0; l < level; ++l)
			jpp.popScope();
	}

	void setDG(cadet::JsonParameterProvider& jpp, std::string basis, unsigned int polyDeg, unsigned int nCol)
	{
		int level = 0;

		if (jpp.exists("model"))
		{
			jpp.pushScope("model");
			++level;
		}
		if (jpp.exists("unit_000"))
		{
			jpp.pushScope("unit_000");
			++level;
		}

		jpp.pushScope("discretization");

		// Set discretization parameters
		jpp.set("NCOL", static_cast<int>(nCol));
		jpp.set("NNODES", static_cast<int>(polyDeg + 1));
		jpp.set("POLYNOMIAL_BASIS", basis);

		jpp.popScope();

		for (int l = 0; l < level; ++l)
			jpp.popScope();
	}

	void setNumParCells(cadet::JsonParameterProvider& jpp, unsigned int nPar, std::string unitID)
	{
		int level = 0;

		if (jpp.exists("model"))
		{
			jpp.pushScope("model");
			++level;
		}
		if (jpp.exists("unit_" + unitID))
		{
			jpp.pushScope("unit_" + unitID);
			++level;
		}

		jpp.pushScope("discretization");

		jpp.set("NPAR", static_cast<int>(nPar));

		jpp.popScope();

		for (int l = 0; l < level; ++l)
			jpp.popScope();
	}

	void setNumRadCells(cadet::JsonParameterProvider& jpp, unsigned int nRad, std::string unitID, const bool mctModel = false)
	{
		int level = 0;

		if (jpp.exists("model"))
		{
			jpp.pushScope("model");
			++level;
		}
		if (jpp.exists("unit_" + unitID))
		{
			jpp.pushScope("unit_" + unitID);
			++level;
		}

		jpp.pushScope("discretization");

		if (mctModel)
		{
			jpp.popScope();
			jpp.set("NCHANNEL", static_cast<int>(nRad));
			jpp.pushScope("discretization");
		}
		else
			jpp.set("NRAD", static_cast<int>(nRad));

		jpp.popScope();

		for (int l = 0; l < level; ++l)
			jpp.popScope();
	}

	void setWenoOrder(cadet::JsonParameterProvider& jpp, int order)
	{
		int level = 0;

		if (jpp.exists("model"))
		{
			jpp.pushScope("model");
			++level;
		}
		if (jpp.exists("unit_000"))
		{
			jpp.pushScope("unit_000");
			++level;
		}

		jpp.pushScope("discretization");
		jpp.pushScope("weno");

		jpp.set("WENO_ORDER", order);

		jpp.popScope();
		jpp.popScope();

		for (int l = 0; l < level; ++l)
			jpp.popScope();
	}

	// todo make copy optional for all variables
	void setNumericalMethod(cadet::IParameterProvider& pp, nlohmann::json& setupJson, const std::string unitID, const bool copy = false)
	{
		// copy over numerical methods from reference file. Note that we leave out spatial and time step resolution parameters
		pp.pushScope("model");
		pp.pushScope("solver");
		nlohmann::json solver = setupJson["model"]["solver"];
		solver["GS_TYPE"] = pp.getInt("GS_TYPE");
		solver["MAX_KRYLOV"] = pp.getInt("MAX_KRYLOV");
		solver["MAX_RESTARTS"] = pp.getInt("MAX_RESTARTS");
		solver["SCHUR_SAFETY"] = pp.getDouble("SCHUR_SAFETY");
		setupJson["model"]["solver"] = solver;
		pp.popScope();

		pp.pushScope("unit_" + unitID);

		if (pp.getString("UNIT_TYPE") != "CSTR") // check for units that dont have a spatial discretization
		{
			pp.pushScope("discretization");
			nlohmann::json discretization = setupJson["model"]["unit_" + unitID]["discretization"];
			if (pp.exists("NBOUND"))
				discretization["NBOUND"] = pp.getIntArray("NBOUND"); // note: in the future this might be included somewhere else in the setup as its part of the model
			if (pp.exists("RECONSTRUCTION"))
				discretization["RECONSTRUCTION"] = pp.getString("RECONSTRUCTION");
			if (pp.exists("USE_ANALYTIC_JACOBIAN"))
				discretization["USE_ANALYTIC_JACOBIAN"] = pp.getInt("USE_ANALYTIC_JACOBIAN");
			if (pp.exists("GS_TYPE"))
				discretization["GS_TYPE"] = pp.getInt("GS_TYPE");
			if (pp.exists("MAX_KRYLOV"))
				discretization["MAX_KRYLOV"] = pp.getInt("MAX_KRYLOV");
			if (pp.exists("MAX_RESTARTS"))
				discretization["MAX_RESTARTS"] = pp.getInt("MAX_RESTARTS");
			if (pp.exists("SCHUR_SAFETY"))
				discretization["SCHUR_SAFETY"] = pp.getDouble("SCHUR_SAFETY");
			if (pp.exists("PAR_DISC_TYPE"))
				discretization["PAR_DISC_TYPE"] = pp.getStringArray("PAR_DISC_TYPE");
			if (pp.exists("PAR_GEOM")) // note: in the future this might be included somewhere else in the setup as its part of the model
				discretization["PAR_GEOM"] = pp.getStringArray("PAR_GEOM");
			if (pp.exists("weno"))
			{
				pp.pushScope("weno");
				nlohmann::json weno;
				weno["WENO_ORDER"] = pp.getInt("WENO_ORDER");
				weno["WENO_EPS"] = pp.getDouble("WENO_EPS");
				weno["BOUNDARY_MODEL"] = pp.getInt("BOUNDARY_MODEL");
				discretization["weno"] = weno;
				pp.popScope();
			}
			setupJson["model"]["unit_" + unitID]["discretization"] = discretization;
			pp.popScope();
		}
		pp.popScope();
		pp.popScope();

		pp.pushScope("solver");
		if (pp.exists("CONSISTENT_INIT_MODE"))
			setupJson["solver"]["CONSISTENT_INIT_MODE"] = pp.getInt("CONSISTENT_INIT_MODE");
		if (pp.exists("CONSISTENT_INIT_MODE_SENS"))
			setupJson["solver"]["CONSISTENT_INIT_MODE_SENS"] = pp.getInt("CONSISTENT_INIT_MODE_SENS");
		setupJson["solver"]["NTHREADS"] = pp.getInt("NTHREADS");
		nlohmann::json timeIntegrator;
		pp.pushScope("time_integrator");
		timeIntegrator["ABSTOL"] = copy ? pp.getDouble("ABSTOL") : 1e-8;
		timeIntegrator["ALGTOL"] = copy ? pp.getDouble("ALGTOL") : 1e-8;
		timeIntegrator["RELTOL"] = copy ? pp.getDouble("RELTOL") : 1e-6;
		timeIntegrator["INIT_STEP_SIZE"] = copy ? pp.getDouble("INIT_STEP_SIZE") : 1e-10;
		timeIntegrator["MAX_STEPS"] = copy ? pp.getInt("MAX_STEPS") : 1000000;
		pp.popScope();
		setupJson["solver"]["time_integrator"] = timeIntegrator;
		pp.popScope();
	}

	void copySensitivities(cadet::IParameterProvider& pp, nlohmann::json& setupJson, const std::string unitID)
	{
		// copy over sensitivity settings
		if (!pp.exists("sensitivity"))
			return;

		pp.pushScope("sensitivity");
		nlohmann::json sens;
		sens["NSENS"] = pp.getInt("NSENS");
		sens["SENS_METHOD"] = pp.getString("SENS_METHOD");

		for (int sensID = 0; true; sensID++)
		{
			std::string sensParam = std::to_string(sensID);
			sensParam = "param_" + std::string(3 - sensParam.length(), '0') + sensParam;
			if (!pp.exists(sensParam))
				break;
			pp.pushScope(sensParam);
			nlohmann::json sens_param;
			sens_param["SENS_NAME"] = pp.getString("SENS_NAME");
			sens_param["SENS_COMP"] = pp.getInt("SENS_COMP");
			sens_param["SENS_BOUNDPHASE"] = pp.getInt("SENS_BOUNDPHASE");
			sens_param["SENS_PARTYPE"] = pp.getInt("SENS_PARTYPE");
			sens_param["SENS_REACTION"] = pp.getInt("SENS_REACTION");
			sens_param["SENS_SECTION"] = pp.getInt("SENS_SECTION");
			sens_param["SENS_UNIT"] = pp.getInt("SENS_UNIT");
			sens[sensParam] = sens_param;
			pp.popScope();
		}

		pp.popScope();
		setupJson["sensitivity"] = sens;
	}

	void copyMultiplexData(cadet::IParameterProvider& pp, nlohmann::json& setupJson, const std::string unitID)
	{
		pp.pushScope("model");
		pp.pushScope("unit_"+unitID);

		if (pp.exists("FILM_DIFFUSION_MULTIPLEX"))
			setupJson["model"]["unit_" + unitID]["FILM_DIFFUSION_MULTIPLEX"] = pp.getInt("FILM_DIFFUSION_MULTIPLEX");

		if (pp.exists("ADSORPTION_MODEL_MULTIPLEX"))
			setupJson["model"]["unit_" + unitID]["ADSORPTION_MODEL_MULTIPLEX"] = pp.getInt("ADSORPTION_MODEL_MULTIPLEX");

		if (pp.exists("COL_DISPERSION_MULTIPLEX"))
			setupJson["model"]["unit_" + unitID]["COL_DISPERSION_MULTIPLEX"] = pp.getInt("COL_DISPERSION_MULTIPLEX");

		if (pp.exists("PAR_DIFFUSION_MULTIPLEX"))
			setupJson["model"]["unit_" + unitID]["PAR_DIFFUSION_MULTIPLEX"] = pp.getInt("PAR_DIFFUSION_MULTIPLEX");

		if (pp.exists("PAR_SURFDIFFUSION_MULTIPLEX"))
			setupJson["model"]["unit_" + unitID]["PAR_SURFDIFFUSION_MULTIPLEX"] = pp.getInt("PAR_SURFDIFFUSION_MULTIPLEX");

		if (pp.exists("PORE_ACCESSIBILITY_MULTIPLEX"))
			setupJson["model"]["unit_" + unitID]["PORE_ACCESSIBILITY_MULTIPLEX"] = pp.getInt("PORE_ACCESSIBILITY_MULTIPLEX");

		if (pp.exists("REACTION_MODEL_PARTICLES_MULTIPLEX"))
			setupJson["model"]["unit_" + unitID]["REACTION_MODEL_PARTICLES_MULTIPLEX"] = pp.getInt("REACTION_MODEL_PARTICLES_MULTIPLEX");

		pp.popScope();
		pp.popScope();
	}

	void copyReturnData(cadet::IParameterProvider& pp, nlohmann::json& setupJson, const std::string unitID)
	{
		// copy over return settings
		pp.pushScope("return");

		nlohmann::json ret;
		{
			pp.pushScope("unit_" + unitID);
			nlohmann::json ret_unit;

			if (pp.exists("WRITE_COORDINATES"))
				ret_unit["WRITE_COORDINATES"] = pp.getInt("WRITE_COORDINATES");

			if (pp.exists("WRITE_SOLUTION_BULK"))
				ret_unit["WRITE_SOLUTION_BULK"] = pp.getInt("WRITE_SOLUTION_BULK");

			if (pp.exists("WRITE_SOLUTION_OUTLET"))
				ret_unit["WRITE_SOLUTION_OUTLET"] = pp.getInt("WRITE_SOLUTION_OUTLET");

			if (pp.exists("WRITE_SOLUTION_LAST"))
				ret_unit["WRITE_SOLUTION_LAST"] = pp.getInt("WRITE_SOLUTION_LAST");

			if (pp.exists("WRITE_SENS_BULK"))
				ret_unit["WRITE_SENS_BULK"] = pp.getInt("WRITE_SENS_BULK");

			if (pp.exists("WRITE_SENS_OUTLET"))
				ret_unit["WRITE_SENS_OUTLET"] = pp.getInt("WRITE_SENS_OUTLET");

			if (pp.exists("WRITE_SENS_LAST"))
				ret_unit["WRITE_SENS_LAST"] = pp.getInt("WRITE_SENS_LAST");

			if (pp.exists("WRITE_SOLUTION_FLUX"))
				ret_unit["WRITE_SOLUTION_FLUX"] = pp.getInt("WRITE_SOLUTION_FLUX");

			if (pp.exists("WRITE_SOLUTION_INLET"))
				ret_unit["WRITE_SOLUTION_INLET"] = pp.getInt("WRITE_SOLUTION_INLET");

			if (pp.exists("WRITE_SOLUTION_PARTICLE"))
				ret_unit["WRITE_SOLUTION_PARTICLE"] = pp.getInt("WRITE_SOLUTION_PARTICLE");

			if (pp.exists("WRITE_SOLUTION_SOLID"))
				ret_unit["WRITE_SOLUTION_SOLID"] = pp.getInt("WRITE_SOLUTION_SOLID");

			if (pp.exists("WRITE_SOLUTION_VOLUME"))
				ret_unit["WRITE_SOLUTION_VOLUME"] = pp.getInt("WRITE_SOLUTION_VOLUME");

			ret["unit_" + unitID] = ret_unit;
			pp.popScope();
		}

		if (pp.exists("SPLIT_COMPONENTS_DATA"))
			ret["SPLIT_COMPONENTS_DATA"] = pp.getInt("SPLIT_COMPONENTS_DATA");
		if (pp.exists("SPLIT_PORTS_DATA"))
			ret["SPLIT_PORTS_DATA"] = pp.getInt("SPLIT_PORTS_DATA");
		if (pp.exists("WRITE_SOLUTION_TIMES"))
			ret["WRITE_SOLUTION_TIMES"] = pp.getInt("WRITE_SOLUTION_TIMES");
		pp.popScope();
		setupJson["return"] = ret;
	}

	void reverseFlow(cadet::JsonParameterProvider& jpp)
	{
		int level = 0;

		if (jpp.exists("model"))
		{
			jpp.pushScope("model");
			++level;
		}
		if (jpp.exists("unit_000"))
		{
			jpp.pushScope("unit_000");
			++level;
		}

		jpp.set("VELOCITY", -jpp.getDouble("VELOCITY"));

		for (int l = 0; l < level; ++l)
			jpp.popScope();
	}

	void setCrossSectionArea(cadet::JsonParameterProvider& jpp, bool useTotalPorosity, int dir)
	{
		int level = 0;

		if (jpp.exists("model"))
		{
			jpp.pushScope("model");
			++level;
		}
		if (jpp.exists("unit_000"))
		{
			jpp.pushScope("unit_000");
			++level;
		}

		const double vel = jpp.getDouble("VELOCITY");
		double por = 0.0;
		if (useTotalPorosity && (jpp.exists("TOTAL_POROSITY")))
			por = jpp.getDouble("TOTAL_POROSITY");
		else
			por = jpp.getDouble("COL_POROSITY");

		// Assume a volumetric flow rate of 1.0 m^3/s
		jpp.set("CROSS_SECTION_AREA", 1.0 / (vel * por));

		if (dir == 0)
			jpp.remove("VELOCITY");
		else
		{
			if (dir > 0)
				jpp.set("VELOCITY", 1.0);
			else
				jpp.set("VELOCITY", -1.0);
		}

		for (int l = 0; l < level; ++l)
			jpp.popScope();
	}

	unsigned int fluxOffsetOfColumnUnitOp(cadet::IUnitOperation* unit)
	{
		// Obtain offset to fluxes
		FluxOffsetExtractionRecorder foer;
		unit->reportSolutionStructure(foer);
		return foer.fluxOffset();
	}

	void testForwardBackward(cadet::JsonParameterProvider jpp, double absTol, double relTol)
	{
		// Forward flow
		cadet::Driver drvFwd;
		drvFwd.configure(jpp);
		drvFwd.run();

		// Backward flow
		reverseFlow(jpp);
		cadet::Driver drvBwd;
		drvBwd.configure(jpp);
		drvBwd.run();

		cadet::InternalStorageUnitOpRecorder const* const fwdData = drvFwd.solution()->unitOperation(0);
		cadet::InternalStorageUnitOpRecorder const* const bwdData = drvBwd.solution()->unitOperation(0);

		double const* fwdInlet = fwdData->inlet();
		double const* fwdOutlet = fwdData->outlet();
		double const* bwdInlet = bwdData->inlet();
		double const* bwdOutlet = bwdData->outlet();

		const unsigned int nComp = fwdData->numComponents();
		for (unsigned int i = 0; i < fwdData->numDataPoints() * fwdData->numInletPorts() * nComp; ++i, ++fwdInlet, ++fwdOutlet, ++bwdInlet, ++bwdOutlet)
		{
			// Forward flow inlet = backward flow outlet
			CAPTURE(i);
			CHECK((*fwdInlet) == makeApprox(*bwdInlet, relTol, absTol));

			// Forward flow outlet = backward flow inlet
			CAPTURE(i);
			CHECK((*fwdOutlet) == makeApprox(*bwdOutlet, relTol, absTol));
		}
	}

	void testForwardBackward(const char* uoType, FVparams disc, double absTol, double relTol)
	{
		SECTION("Forward vs backward flow (WENO=" + std::to_string(disc.getWenoOrder()) + ")")
		{
			// Use Load-Wash-Elution test case
			cadet::JsonParameterProvider jpp = createLWE(uoType, "FV");
			disc.setDisc(jpp);

			testForwardBackward(jpp, absTol, relTol);
		}
	}

	void testForwardBackward(const char* uoType, DGparams disc, double absTol, double relTol)
	{
		SECTION("Forward vs backward flow (DG integration mode " + std::to_string(disc.getIntegrationMode()) + ")")
		{
			// Use Load-Wash-Elution test case
			cadet::JsonParameterProvider jpp = createLWE(uoType, "DG");
			disc.setDisc(jpp);

			testForwardBackward(jpp, absTol, relTol);
		}
	}

	void testAnalyticBenchmark(cadet::JsonParameterProvider jpp, const char* refFileRelPath, bool forwardFlow, bool dynamicBinding, double absTol, double relTol)
	{
		if (!forwardFlow)
			reverseFlow(jpp);

		// Run simulation
		cadet::Driver drv;
		drv.configure(jpp);
		drv.run();

		// Read reference data from test file
		const std::string refFile = std::string(getTestDirectory()) + std::string(refFileRelPath);
		ReferenceDataReader rd(refFile.c_str());
		const std::vector<double> time = rd.time();
		const std::vector<double> ref = (dynamicBinding ? rd.analyticDynamic() : rd.analyticQuasiStationary());

		// Get data from simulation
		cadet::InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(0);
		double const* outlet = simData->outlet();

		// Compare
		for (unsigned int i = 0; i < simData->numDataPoints() * simData->numComponents() * simData->numInletPorts(); ++i, ++outlet)
		{
			// Note that the simulation only saves the chromatogram at multiples of 2 (i.e., 0s, 2s, 4s, ...)
			// whereas the reference solution is given at every second (0s, 1s, 2s, 3s, ...)
			// Thus, we only take the even indices of the reference array
			CAPTURE(time[2 * i]);
			CHECK((*outlet) == makeApprox(ref[2 * i], relTol, absTol));
		}
	}

	void testAnalyticBenchmark(const char* uoType, const char* refFileRelPath, bool forwardFlow, bool dynamicBinding, DiscParams& disc, const std::string method, double absTol, double relTol)
	{
		const std::string fwdStr = (forwardFlow ? "forward" : "backward");
		SECTION(method + ": Analytic " + fwdStr + " flow with " + (dynamicBinding ? "dynamic" : "quasi-stationary") + " binding")
		{
			// Setup simulation
			cadet::JsonParameterProvider jpp = createLinearBenchmark(dynamicBinding, false, uoType, method);
			disc.setDisc(jpp);

			testAnalyticBenchmark(jpp, refFileRelPath, forwardFlow, dynamicBinding, absTol, relTol);
		}
	}

	void testAnalyticBenchmark(const char* uoType, const char* refFileRelPath, bool forwardFlow, bool dynamicBinding, FVparams& disc, double absTol, double relTol)
	{
		testAnalyticBenchmark(uoType, refFileRelPath, forwardFlow, dynamicBinding, disc, "FV", absTol, relTol);
	}

	void testAnalyticBenchmark(const char* uoType, const char* refFileRelPath, bool forwardFlow, bool dynamicBinding, DGparams& disc, double absTol, double relTol)
	{
		testAnalyticBenchmark(uoType, refFileRelPath, forwardFlow, dynamicBinding, disc, "DG", absTol, relTol);
	}

	void testAnalyticNonBindingBenchmark(const char* uoType, const char* refFileRelPath, bool forwardFlow, DiscParams& disc, const std::string method, double absTol, double relTol)
	{
		const std::string fwdStr = (forwardFlow ? "forward" : "backward");
		SECTION(method + ": Analytic " + fwdStr + " flow")
		{
			// Setup simulation
			cadet::JsonParameterProvider jpp = createLinearBenchmark(true, true, uoType, method);
			disc.setDisc(jpp);

			if (!forwardFlow)
				reverseFlow(jpp);

			// Run simulation
			cadet::Driver drv;
			drv.configure(jpp);
			drv.run();

			// Read reference data from test file
			const std::string refFile = std::string(getTestDirectory()) + std::string(refFileRelPath);
			ReferenceDataReader rd(refFile.c_str());
			const std::vector<double> time = rd.time();
			const std::vector<double> ref = rd.analyticDynamic();

			// Get data from simulation
			cadet::InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(0);
			double const* outlet = simData->outlet();

			// Compare
			for (unsigned int i = 0; i < simData->numDataPoints() * simData->numComponents() * simData->numInletPorts(); ++i, ++outlet)
			{
				CAPTURE(time[i]);
				CHECK((*outlet) == makeApprox(ref[i], relTol, absTol));
			}
		}
	}

	void testAnalyticNonBindingBenchmark(const char* uoType, const char* refFileRelPath, bool forwardFlow, FVparams& disc, double absTol, double relTol)
	{
		testAnalyticNonBindingBenchmark(uoType, refFileRelPath, forwardFlow, disc, "FV", absTol, relTol);
	}

	void testAnalyticNonBindingBenchmark(const char* uoType, const char* refFileRelPath, bool forwardFlow, DGparams& disc, double absTol, double relTol)
	{
		testAnalyticNonBindingBenchmark(uoType, refFileRelPath, forwardFlow, disc, "DG", absTol, relTol);
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void testJacobianAD(cadet::JsonParameterProvider& jpp, const double absTolFDpattern)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		cadet::IUnitOperation* const unitAna = unitoperation::createAndConfigureUnit(jpp, *mb);
		cadet::IUnitOperation* const unitAD = unitoperation::createAndConfigureUnit(jpp, *mb);

		// Enable AD
		cadet::ad::setDirections(cadet::ad::getMaxDirections());
		unitAD->useAnalyticJacobian(false);

		cadet::active* adRes = new cadet::active[unitAD->numDofs()];
		cadet::active* adY = new cadet::active[unitAD->numDofs()];

		// Obtain memory for state, Jacobian multiply direction, Jacobian column
		std::vector<double> y(unitAD->numDofs(), 0.0);
		std::vector<double> jacDir(unitAD->numDofs(), 0.0);
		std::vector<double> jacCol1(unitAD->numDofs(), 0.0);
		std::vector<double> jacCol2(unitAD->numDofs(), 0.0);

		// Fill state vector with some values
		util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, unitAna->numDofs());
//			util::populate(y.data(), [](unsigned int idx) { return 1.0; }, unitAna->numDofs());

		// Setup matrices
		const AdJacobianParams noAdParams{nullptr, nullptr, 0u};
		const AdJacobianParams adParams{adRes, adY, 0u};
		unitAD->prepareADvectors(adParams);

		const ConstSimulationState simState{y.data(), nullptr};
		unitAna->notifyDiscontinuousSectionTransition(0.0, 0u, simState, noAdParams);
		unitAD->notifyDiscontinuousSectionTransition(0.0, 0u, simState, adParams);

		// Compute state Jacobian
		const SimulationTime simTime{0.0, 0u};
		cadet::util::ThreadLocalStorage tls;
		tls.resize(unitAna->threadLocalMemorySize());

		unitAna->residualWithJacobian(simTime, simState, jacDir.data(), noAdParams, tls);
		unitAD->residualWithJacobian(simTime, simState, jacDir.data(), adParams, tls);
		std::fill(jacDir.begin(), jacDir.end(), 0.0);

		// Compare Jacobians
		cadet::test::checkJacobianPatternFD(unitAna, unitAD, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data(), tls, absTolFDpattern);
		cadet::test::checkJacobianPatternFD(unitAna, unitAna, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data(), tls, absTolFDpattern);
		cadet::test::compareJacobian(unitAna, unitAD, nullptr, nullptr, jacDir.data(), jacCol1.data(), jacCol2.data());
//				cadet::test::compareJacobianFD(unitAna, unitAD, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data());

		delete[] adRes;
		delete[] adY;
		mb->destroyUnitOperation(unitAna);
		mb->destroyUnitOperation(unitAD);
		destroyModelBuilder(mb);
	}

	void testJacobianForwardBackward(cadet::JsonParameterProvider& jpp, const double absTolFDpattern)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		// Use some test case parameters
		const unsigned int nComp = jpp.getInt("NCOMP");

		cadet::IUnitOperation* const unitAna = unitoperation::createAndConfigureUnit(jpp, *mb);
		cadet::IUnitOperation* const unitAD = unitoperation::createAndConfigureUnit(jpp, *mb);

		// Enable AD
		cadet::ad::setDirections(cadet::ad::getMaxDirections());
		unitAD->useAnalyticJacobian(false);

		cadet::active* adRes = new cadet::active[unitAD->numDofs()];
		cadet::active* adY = new cadet::active[unitAD->numDofs()];

		// Obtain memory for state, Jacobian multiply direction, Jacobian column
		std::vector<double> y(unitAD->numDofs(), 0.0);
		std::vector<double> jacDir(unitAD->numDofs(), 0.0);
		std::vector<double> jacCol1(unitAD->numDofs(), 0.0);
		std::vector<double> jacCol2(unitAD->numDofs(), 0.0);

		// Fill state vector with some values
		util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, unitAna->numDofs());
//			util::populate(y.data(), [](unsigned int idx) { return 1.0; }, unitAna->numDofs());

		// Setup matrices
		const AdJacobianParams noAdParams{nullptr, nullptr, 0u};
		const AdJacobianParams adParams{adRes, adY, 0u};
		unitAD->prepareADvectors(adParams);

		unitAna->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, noAdParams);
		unitAD->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, adParams);

		SECTION("Forward then backward flow (nonzero state)")
		{
			// Compute state Jacobian
			const SimulationTime simTime{0.0, 0u};
			const ConstSimulationState simState{y.data(), nullptr};
			cadet::util::ThreadLocalStorage tls;
			tls.resize(unitAna->threadLocalMemorySize());

			unitAna->residualWithJacobian(simTime, simState, jacDir.data(), noAdParams, tls);
			unitAD->residualWithJacobian(simTime, simState, jacDir.data(), adParams, tls);
			std::fill(jacDir.begin(), jacDir.end(), 0.0);

			// Compare Jacobians
			cadet::test::checkJacobianPatternFD(unitAna, unitAD, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data(), tls, absTolFDpattern);
			cadet::test::checkJacobianPatternFD(unitAna, unitAna, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data(), tls, absTolFDpattern);
			cadet::test::compareJacobian(unitAna, unitAD, nullptr, nullptr, jacDir.data(), jacCol1.data(), jacCol2.data());
//				cadet::test::compareJacobianFD(unitAna, unitAD, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data());

			if (jpp.getString("UNIT_TYPE").substr(0, 6) == "RADIAL")
				{
					// Reverse flow
					const bool paramSet = unitAna->setParameter(cadet::makeParamId(cadet::hashString("VELOCITY_COEFF"), 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), -jpp.getDouble("VELOCITY_COEFF"));
					REQUIRE(paramSet);
					// Reverse flow
					const bool paramSet2 = unitAD->setParameter(cadet::makeParamId(cadet::hashString("VELOCITY_COEFF"), 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), -jpp.getDouble("VELOCITY_COEFF"));
					REQUIRE(paramSet2);
				}
			else
			{
				// Reverse flow
				const bool paramSet = unitAna->setParameter(cadet::makeParamId(cadet::hashString("VELOCITY"), 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), -jpp.getDouble("VELOCITY"));
				REQUIRE(paramSet);
				// Reverse flow
				const bool paramSet2 = unitAD->setParameter(cadet::makeParamId(cadet::hashString("VELOCITY"), 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), -jpp.getDouble("VELOCITY"));
				REQUIRE(paramSet2);
			}

			// Setup
			unitAna->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, noAdParams);
			unitAD->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, adParams);

			// Compute state Jacobian
			unitAna->residualWithJacobian(simTime, simState, jacDir.data(), noAdParams, tls);
			unitAD->residualWithJacobian(simTime, simState, jacDir.data(), adParams, tls);
			std::fill(jacDir.begin(), jacDir.end(), 0.0);

			// Compare Jacobians
			cadet::test::checkJacobianPatternFD(unitAna, unitAD, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data(), tls, absTolFDpattern);
			cadet::test::checkJacobianPatternFD(unitAna, unitAna, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data(), tls, absTolFDpattern);
//				cadet::test::compareJacobianFD(unitAD, unitAna, y.data(), jacDir.data(), nullptr, jacCol1.data(), jacCol2.data());
//				cadet::test::compareJacobianFD(unitAna, unitAD, y.data(), jacDir.data(), nullptr, jacCol1.data(), jacCol2.data());
			cadet::test::compareJacobian(unitAna, unitAD, nullptr, nullptr, jacDir.data(), jacCol1.data(), jacCol2.data());
		}

		delete[] adRes;
		delete[] adY;
		mb->destroyUnitOperation(unitAna);
		mb->destroyUnitOperation(unitAD);
		
		destroyModelBuilder(mb);
	}

	void testJacobianForwardBackward(const char* uoType, FVparams disc, const double absTolFDpattern)
	{
		SECTION("Forward vs backward flow Jacobian (WENO=" + std::to_string(disc.getWenoOrder()) + ")")
		{
			// Use Load-Wash-Elution test case
			cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType, "FV");
			disc.setDisc(jpp);

			testJacobianForwardBackward(jpp, absTolFDpattern);
		}
	}

	void testJacobianForwardBackward(const char* uoType, DGparams disc, const double absTolFDpattern)
	{
		SECTION("Forward vs backward flow Jacobian (DG integration mode " + std::to_string(disc.getIntegrationMode()) + ")")
		{
			// Use Load-Wash-Elution test case
			cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType, "DG");
			disc.setDisc(jpp);

			testJacobianForwardBackward(jpp, absTolFDpattern);
		}
	}

	void testJacobianWenoForwardBackwardFD(const std::string& uoType, const std::string& spatialMethod, int wenoOrder, double h, double absTol, double relTol)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		SECTION("Forward vs backward flow Jacobian (WENO=" + std::to_string(wenoOrder) + ")")
		{
			// Use some test case parameters
			cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType, spatialMethod);
			const unsigned int nComp = jpp.getInt("NCOMP");

			FVparams disc;
			disc.setWenoOrder(wenoOrder);
			cadet::IUnitOperation* const unit = createAndConfigureUnit(*mb, jpp, disc);

			// Obtain memory for state, Jacobian multiply direction, Jacobian column
			std::vector<double> y(unit->numDofs(), 0.0);
			std::vector<double> jacDir(unit->numDofs(), 0.0);
			std::vector<double> jacCol1(unit->numDofs(), 0.0);
			std::vector<double> jacCol2(unit->numDofs(), 0.0);

			// Fill state vector with some values
			util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, unit->numDofs());
//			util::populate(y.data(), [](unsigned int idx) { return 1.0; }, unit->numDofs());

			// Setup matrices
			const AdJacobianParams noAdParams{nullptr, nullptr, 0u};
			unit->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, noAdParams);

			SECTION("Forward then backward flow (nonzero state)")
			{
				// Compute state Jacobian
				const SimulationTime simTime{0.0, 0u};
				const ConstSimulationState simState{y.data(), nullptr};
				cadet::util::ThreadLocalStorage tls;
				tls.resize(unit->threadLocalMemorySize());

				unit->residualWithJacobian(simTime, simState, jacDir.data(), noAdParams, tls);
				std::fill(jacDir.begin(), jacDir.end(), 0.0);

				// Compare Jacobians
				cadet::test::checkJacobianPatternFD(unit, unit, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data(), tls);
				cadet::test::compareJacobianFD(unit, unit, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data(), tls, h, absTol, relTol);

				// Reverse flow
				const bool paramSet = unit->setParameter(cadet::makeParamId(cadet::hashString("VELOCITY"), 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), -jpp.getDouble("VELOCITY"));
				REQUIRE(paramSet);

				// Setup
				unit->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, noAdParams);

				// Compute state Jacobian
				unit->residualWithJacobian(simTime, simState, jacDir.data(), noAdParams, tls);
				std::fill(jacDir.begin(), jacDir.end(), 0.0);

				// Compare Jacobians
				cadet::test::checkJacobianPatternFD(unit, unit, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data(), tls);
				cadet::test::compareJacobianFD(unit, unit, y.data(), nullptr, jacDir.data(), jacCol1.data(), jacCol2.data(), tls, h, absTol, relTol);
			}

			mb->destroyUnitOperation(unit);
		}
		destroyModelBuilder(mb);
	}

	void testTimeDerivativeJacobianFD(const std::string& uoType, const std::string& spatialMethod, double h, double absTol, double relTol)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		// Use some test case parameters
		cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType, spatialMethod);

		for (int bindMode = 0; bindMode < 2; ++bindMode)
		{
			const bool isKinetic = bindMode;
			SECTION(isKinetic ? "Kinetic binding" : "Quasi-stationary binding")
			{
				cadet::test::setBindingMode(jpp, isKinetic);

				cadet::IUnitOperation* const unit = createAndConfigureUnit(*mb, jpp);

				cadet::util::ThreadLocalStorage tls;
				tls.resize(unit->threadLocalMemorySize());

				// Obtain memory for state, Jacobian multiply direction, Jacobian column
				const unsigned int nDof = unit->numDofs();
				std::vector<double> y(nDof, 0.0);
				std::vector<double> yDot(nDof, 0.0);
				std::vector<double> jacDir(nDof, 0.0);
				std::vector<double> jacCol1(nDof, 0.0);
				std::vector<double> jacCol2(nDof, 0.0);

				// Fill state vectors with some values
				util::populate(y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
				util::populate(yDot.data(), [=](unsigned int idx) {
					return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4;
				}, nDof);

				// Setup matrices
				unit->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, AdJacobianParams{nullptr, nullptr, 0u});

				// Compare Jacobians
				cadet::test::compareTimeDerivativeJacobianFD(unit, unit, y.data(), yDot.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), tls, h, absTol, relTol);

				mb->destroyUnitOperation(unit);
			}
		}

		destroyModelBuilder(mb);
	}

	void testArrowHeadJacobianFD(const std::string& uoType, double h, double absTol, double relTol)
	{
		cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType, "FV");
		testArrowHeadJacobianFD(jpp, h, absTol, relTol);
	}

	void testArrowHeadJacobianFD(const std::string& uoType, bool dynamicBinding, double h, double absTol, double relTol)
	{
		cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType, "FV");
		setBindingMode(jpp, dynamicBinding);
		testArrowHeadJacobianFD(jpp, h, absTol, relTol);
	}

	void testArrowHeadJacobianFDVariableParSurfDiff(const std::string& uoType, double h, double absTol, double relTol)
	{
		cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType, "FV");
		setBindingMode(jpp, false);
		{
			auto ms = util::makeOptionalGroupScope(jpp, "model");
			auto us = util::makeOptionalGroupScope(jpp, "unit_000");

			jpp.set("PAR_SURFDIFFUSION_DEP", "LIQUID_SALT_EXPONENTIAL");
			jpp.set("PAR_SURFDIFFUSION_EXPFACTOR", std::vector<double>{ 0.8, 1.6 });
			jpp.set("PAR_SURFDIFFUSION_EXPARGMULT", std::vector<double>{ 1.3, 2.1 });
		}

		testArrowHeadJacobianFD(jpp, h, absTol, relTol);
	}

	void testJacobianADVariableParSurfDiff(const std::string& uoType, const std::string& spatialMethod, bool dynamicBinding)
	{
		cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType, spatialMethod);
		setBindingMode(jpp, dynamicBinding);
		{
			auto ms = util::makeOptionalGroupScope(jpp, "model");
			auto us = util::makeOptionalGroupScope(jpp, "unit_000");

			jpp.set("PAR_SURFDIFFUSION_DEP", "LIQUID_SALT_EXPONENTIAL");
			jpp.set("PAR_SURFDIFFUSION_EXPFACTOR", std::vector<double>{ 0.8, 1.6 });
			jpp.set("PAR_SURFDIFFUSION_EXPARGMULT", std::vector<double>{ 1.3, 2.1 });
		}

		testJacobianAD(jpp);
	}

	void testArrowHeadJacobianFD(cadet::JsonParameterProvider& jpp, double h, double absTol, double relTol)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		cadet::IUnitOperation* const unitAna = unitoperation::createAndConfigureUnit(jpp, *mb);
		cadet::IUnitOperation* const unitFD = unitoperation::createAndConfigureUnit(jpp, *mb);

		cadet::util::ThreadLocalStorage tls;
		tls.resize(unitAna->threadLocalMemorySize());

		// Obtain offset to fluxes
		const unsigned int fluxOffset = fluxOffsetOfColumnUnitOp(unitFD);

		// Obtain memory for state, Jacobian multiply direction, Jacobian column
		std::vector<double> y(unitFD->numDofs(), 0.0);
		std::vector<double> jacDir(unitFD->numDofs(), 0.0);
		std::vector<double> jacCol1(unitFD->numDofs(), 0.0);
		std::vector<double> jacCol2(unitFD->numDofs(), 0.0);

		// Fill state vector with some values
		util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, unitAna->numDofs());
//		util::populate(y.data(), [](unsigned int idx) { return 1.0; }, unitAna->numDofs());

		// Setup matrices
		const AdJacobianParams noParams{nullptr, nullptr, 0u};
		unitAna->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, noParams);
		unitFD->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), nullptr}, noParams);

		// Compute state Jacobian
		unitAna->residualWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), nullptr}, jacDir.data(), noParams, tls);
		unitFD->residualWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), nullptr}, jacDir.data(), noParams, tls);
		std::fill(jacDir.begin(), jacDir.end(), 0.0);

		// Compare Jacobians
		cadet::test::compareJacobianArrowHeadFD(
			[=, &tls](double const* lDir, double* res) -> void { unitFD->residual(SimulationTime{0.0, 0u}, ConstSimulationState{lDir, nullptr}, res, tls); }, 
			[&](double const* lDir, double* res) -> void { unitAna->multiplyWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), nullptr}, lDir, 1.0, 0.0, res); }, 
			y.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), unitFD->numDofs(), fluxOffset, h, absTol, relTol);

		mb->destroyUnitOperation(unitAna);
		mb->destroyUnitOperation(unitFD);
		destroyModelBuilder(mb);
	}

	void testFwdSensJacobians(cadet::JsonParameterProvider jpp, double h, double absTol, double relTol, const bool hasBinding)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		for (int bindMode = 0; bindMode < static_cast<int>(hasBinding) + 1; ++bindMode)
		{
			const bool isKinetic = bindMode;
			SECTION(hasBinding ? "No binding" : (isKinetic ? "Kinetic binding" : "Quasi-stationary binding"))
			{
				if (hasBinding)
					cadet::test::setBindingMode(jpp, isKinetic);
				cadet::IUnitOperation* const unit = createAndConfigureUnit(*mb, jpp);

				// Enable AD
				cadet::ad::setDirections(cadet::ad::getMaxDirections());
				cadet::active* adRes = new cadet::active[unit->numDofs()];
				const AdJacobianParams adParams{adRes, nullptr, 0};
				unit->prepareADvectors(adParams);

				// Add dispersion parameter sensitivity
				REQUIRE(unit->setSensitiveParameter(makeParamId(hashString("COL_DISPERSION"), 0, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep), 0, 1.0));

				// Obtain memory for state, Jacobian multiply direction, Jacobian column
				const unsigned int nDof = unit->numDofs();
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

				cadet::util::ThreadLocalStorage tls;
				tls.resize(unit->threadLocalMemorySize());

				// Fill state vector with some values
				util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
				util::populate(yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);

				// Setup matrices
				unit->notifyDiscontinuousSectionTransition(0.0, 0u, {y.data(), yDot.data()}, adParams);

				// Calculate Jacobian
				unit->residualWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), yDot.data()}, jacDir.data(), adParams, tls);

				// Calculate parameter derivative
				unit->residualSensFwdAdOnly(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), yDot.data()}, adRes, tls);

				// Check state Jacobian
				cadet::test::compareJacobianFD(
					[&](double const* lDir, double* res) -> void {
						yS[0] = lDir;
						resS[0] = res;
						unit->residualSensFwdCombine(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), yDot.data()}, yS, ySdot, resS, adRes, temp1.data(), temp2.data(), temp3.data());
					}, 
					[&](double const* lDir, double* res) -> void { unit->multiplyWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), yDot.data()}, lDir, 1.0, 0.0, res); }, 
					zeros.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), nDof, h, absTol, relTol);

				// Reset evaluation point
				yS[0] = zeros.data();
				ySdot[0] = zeros.data();

				// Check time derivative Jacobian
				cadet::test::compareJacobianFD(
					[&](double const* lDir, double* res) -> void {
						ySdot[0] = lDir;
						resS[0] = res;
						unit->residualSensFwdCombine(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), yDot.data()}, yS, ySdot, resS, adRes, temp1.data(), temp2.data(), temp3.data());
					}, 
					[&](double const* lDir, double* res) -> void { unit->multiplyWithDerivativeJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y.data(), yDot.data()}, lDir, res); }, 
					zeros.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), nDof, h, absTol, relTol);

				delete[] adRes;
				mb->destroyUnitOperation(unit);
			}
		}

		destroyModelBuilder(mb);
	}

	void testFwdSensSolutionFD(const std::string& uoType, const std::string& spatialMethod, bool disableSensErrorTest, double const* fdStepSize, double const* absTols, double const* relTols, double const* passRates)
	{
		const std::vector<cadet::ParameterId> params = {
			cadet::makeParamId("COL_DISPERSION", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep),
			cadet::makeParamId("CONST_COEFF", 1, 0, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, 0),
			cadet::makeParamId("SMA_KA", 0, 1, cadet::ParTypeIndep, 0, cadet::ReactionIndep, cadet::SectionIndep),
			cadet::makeParamId("CONNECTION", cadet::UnitOpIndep, cadet::CompIndep, cadet::ParTypeIndep, 1, 0, 0),
		};
		const std::vector<const char*> paramNames = {"COL_DISPERSION", "CONST_COEFF", "SMA_KA", "CONNECTION"};
		const double absTolSens[] = {1e-12, 1e-6, 1e-6, 1e-6};

		for (int bindMode = 0; bindMode < 2; ++bindMode)
		{
			const bool isKinetic = bindMode;
			for (std::size_t n = 0; n < params.size(); ++n)
			{
				SECTION("Parameter " + std::string(paramNames[n]) + (isKinetic ? " Kinetic binding" : " Quasi-stationary binding"))
				{
					const double absTol = absTols[n];
					const double relTol = relTols[n];
					const double passRate = passRates[n];
					const cadet::ParameterId& curParam = params[n];
					const double h = fdStepSize[n];

					// Setup simulation including forward sensitivities
					cadet::JsonParameterProvider jppAna = createLWE(uoType, spatialMethod);
					cadet::test::setBindingMode(jppAna, isKinetic);
					cadet::test::column::setCrossSectionArea(jppAna, uoType == "LUMPED_RATE_MODEL_WITHOUT_PORES", 0);
					cadet::test::addSensitivity(jppAna, paramNames[n], curParam, absTolSens[n]);
					cadet::test::returnSensitivities(jppAna, 0);
					if (disableSensErrorTest)
						cadet::test::disableSensitivityErrorTest(jppAna);

					// Run simulation
					cadet::Driver drv;
					drv.configure(jppAna);
					REQUIRE(drv.simulator()->numSensParams() == 1);
					drv.run();

					// Setup FD simulation
					cadet::JsonParameterProvider jppFD = createLWE(uoType, spatialMethod);
					cadet::test::setBindingMode(jppFD, isKinetic);
					cadet::test::column::setCrossSectionArea(jppFD, uoType == "LUMPED_RATE_MODEL_WITHOUT_PORES", 0);

					// Configure FD simulation
					cadet::Driver drvLeft;
					drvLeft.configure(jppFD);
					
					// Extract parameter values
					const double baseVal = drvLeft.simulator()->model()->getParameterDouble(curParam);
					REQUIRE(!std::isnan(baseVal));

					// Run left FD point
					drvLeft.simulator()->setParameterValue(curParam, baseVal * (1.0 - h));
					drvLeft.run();

					// Configure and run right FD simulation
					cadet::Driver drvRight;
					drvRight.configure(jppFD);
					drvRight.simulator()->setParameterValue(curParam, baseVal * (1.0 + h));
					drvRight.run();

					// Get data from simulation
					cadet::InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(0);
					const unsigned int nComp = simData->numComponents();
					const unsigned int nPorts = simData->numInletPorts();
					const unsigned int nDataPoints = simData->numDataPoints() * nComp * nPorts;
					double const* const time = drv.solution()->time();
					double const* outlet = simData->sensOutlet(0);
					double const* outletL = drvLeft.solution()->unitOperation(0)->outlet();
					double const* outletR = drvRight.solution()->unitOperation(0)->outlet();

					// Compare
					const double actStepSize = 2.0 * h * baseVal;
					unsigned int numPassed = 0;
					for (unsigned int i = 0; i < nDataPoints; ++i, ++outlet, ++outletL, ++outletR)
					{
						const double cmpVal = *outlet;
						const double fdVal = ((*outletR) - (*outletL)) / actStepSize;
						const unsigned int comp = (i % (nComp * nPorts)) % nComp;
						const unsigned int port = (i % (nComp * nPorts)) / nComp;
						const unsigned int timeIdx = i / (nComp * nPorts);

						INFO("Time " << time[timeIdx] << " Port " << port << " Component " << comp << " time point idx " << timeIdx);
						CHECK(fdVal == makeApprox(*outlet, relTol, absTol));

						const bool relativeOK = std::abs(*outlet - fdVal) <= relTol * std::abs(*outlet);
						if (relativeOK)
							++numPassed;
					}

					const double ratio = static_cast<double>(numPassed) / static_cast<double>(nDataPoints);
					CAPTURE(ratio);
					CHECK(ratio >= passRate);
				}
			}
		}
	}

	void testFwdSensSolutionForwardBackward(const std::string& uoType, const std::string& spatialMethod, double const* absTols, double const* relTols, double const* passRates)
	{
		const std::vector<cadet::ParameterId> params = {
			cadet::makeParamId("COL_DISPERSION", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep),
			cadet::makeParamId("CONST_COEFF", 1, 0, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, 0),
			cadet::makeParamId("SMA_KA", 0, 1, cadet::ParTypeIndep, 0, cadet::ReactionIndep, cadet::SectionIndep),
			cadet::makeParamId("CONNECTION", cadet::UnitOpIndep, cadet::CompIndep, cadet::ParTypeIndep, 1, 0, 0),
		};
		const std::vector<const char*> paramNames = {"COL_DISPERSION", "CONST_COEFF", "SMA_KA", "CONNECTION"};
		const double absTolSens[] = {1e-12, 1e-6, 1e-6, 1e-6};

		for (int bindMode = 0; bindMode < 2; ++bindMode)
		{
			const bool isKinetic = bindMode;
			for (std::size_t n = 0; n < params.size(); ++n)
			{
				SECTION("Parameter " + std::string(paramNames[n]) + (isKinetic ? " Kinetic binding" : " Quasi-stationary binding"))
				{
					const double absTol = absTols[n];
					const double relTol = relTols[n];
					const double passRate = passRates[n];
					const cadet::ParameterId& curParam = params[n];

					// Setup simulation including forward sensitivities
					cadet::JsonParameterProvider jpp = createLWE(uoType, spatialMethod);
					cadet::test::setBindingMode(jpp, isKinetic);
					cadet::test::column::setCrossSectionArea(jpp, uoType == "LUMPED_RATE_MODEL_WITHOUT_PORES", 1);
					cadet::test::addSensitivity(jpp, paramNames[n], curParam, absTolSens[n]);
					cadet::test::returnSensitivities(jpp, 0, true);
					cadet::test::disableSensitivityErrorTest(jpp);

					// Run simulation
					cadet::Driver drvFwd;
					drvFwd.configure(jpp);
					drvFwd.run();

					// Run simulation with reversed flow
					reverseFlow(jpp);
					cadet::Driver drvBwd;
					drvBwd.configure(jpp);
					drvBwd.run();

					// Get data from simulation
					cadet::InternalStorageUnitOpRecorder const* const fwdData = drvFwd.solution()->unitOperation(0);
					cadet::InternalStorageUnitOpRecorder const* const bwdData = drvBwd.solution()->unitOperation(0);

					const unsigned int nComp = fwdData->numComponents();
					const unsigned int nPorts = fwdData->numInletPorts();
					const unsigned int nDataPoints = fwdData->numDataPoints() * nComp * nPorts;
					double const* const time = drvFwd.solution()->time();
					double const* fwdOutlet = fwdData->sensOutlet(0);
					double const* bwdOutlet = bwdData->sensOutlet(0);

					// Compare
					unsigned int numPassed = 0;
					for (unsigned int i = 0; i < nDataPoints; ++i, ++fwdOutlet, ++bwdOutlet)
					{
						const unsigned int comp = (i % (nComp * nPorts)) % nComp;
						const unsigned int port = (i % (nComp * nPorts)) / nComp;
						const unsigned int timeIdx = i / (nComp * nPorts);

						INFO("Time " << time[timeIdx] << " Port " << port << " Component " << comp << " time point idx " << timeIdx);
						CHECK(*bwdOutlet == makeApprox(*fwdOutlet, relTol, absTol));

						const bool relativeOK = std::abs(*bwdOutlet - *fwdOutlet) <= relTol * std::abs(*fwdOutlet);
						if (relativeOK)
							++numPassed;
					}

					const double ratio = static_cast<double>(numPassed) / static_cast<double>(nDataPoints);
					CAPTURE(ratio);
					CHECK(ratio >= passRate);
				}
			}
		}
	}

	void testConsistentInitializationLinearBinding(const std::string& uoType, const std::string& spatialMethod, double consTol, double absTol, const int reqBnd, const int useAD)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		for (int bindingMode = 0; bindingMode < 2; ++bindingMode)
		{
			if ((bindingMode == 0 && reqBnd == 1) || (bindingMode == 1 && reqBnd == 0))
				continue;
			const bool isKinetic = (bindingMode == 0);
			for (int adMode = 0; adMode < 2; ++adMode)
			{
				if ((adMode == 0 && useAD == 1) || (adMode == 1 && useAD == 0))
					continue;
				const bool adEnabled = (adMode > 0);
				SECTION(std::string(isKinetic ? " Kinetic binding" : " Quasi-stationary binding") + " with AD " + (adEnabled ? "enabled" : "disabled"))
				{
					// Use some test case parameters
					cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType, spatialMethod);
					cadet::test::setBindingMode(jpp, isKinetic);
					cadet::IUnitOperation* const unit = createAndConfigureUnit(*mb, jpp);

					// Fill state vector with given initial values
					std::vector<double> y(unit->numDofs(), 0.0);
					util::populate(y.data(), [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, unit->numDofs());

					unitoperation::testConsistentInitialization(unit, adEnabled, y.data(), consTol, absTol);

					mb->destroyUnitOperation(unit);
				}
			}
		}
		destroyModelBuilder(mb);
	}

	void testConsistentInitializationSMABinding(const std::string& uoType, const std::string& spatialMethod, double const* const initState, double consTol, double absTol, const int reqBnd, const int useAD)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		for (int bindingMode = 0; bindingMode < 2; ++bindingMode)
		{
			if ((bindingMode == 0 && reqBnd == 1) || (bindingMode == 1 && reqBnd == 0))
				continue;
			const bool isKinetic = (bindingMode == 0);
			for (int adMode = 0; adMode < 2; ++adMode)
			{
				if ((adMode == 0 && useAD == 1) || (adMode == 1 && useAD == 0))
					continue;
				const bool adEnabled = (adMode > 0);
				SECTION(std::string(isKinetic ? " Kinetic binding" : " Quasi-stationary binding") + " with AD " + (adEnabled ? "enabled" : "disabled"))
				{
					// Use some test case parameters
					cadet::JsonParameterProvider jpp = createColumnWithSMA(uoType, spatialMethod);
					cadet::test::setBindingMode(jpp, isKinetic);
					cadet::IUnitOperation* const unit = createAndConfigureUnit(*mb, jpp);

					// Fill state vector with given initial values
					std::vector<double> y(initState, initState + unit->numDofs());

					unitoperation::testConsistentInitialization(unit, adEnabled, y.data(), consTol, absTol);

					mb->destroyUnitOperation(unit);
				}
			}
		}
		destroyModelBuilder(mb);
	}

	void testConsistentInitializationSensitivity(const std::string& uoType, const std::string& spatialMethod, double const* const y, double const* const yDot, bool linearBinding, double absTol, const int reqBnd, const int useAD)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		for (int bindingMode = 0; bindingMode < 1; ++bindingMode)
		{
			if ((bindingMode == 0 && reqBnd == 1) || (bindingMode == 1 && reqBnd == 0))
				continue;
			const bool isKinetic = (bindingMode == 0);
			for (int adMode = 0; adMode < 2; ++adMode)
			{
				if ((adMode == 0 && useAD == 1) || (adMode == 1 && useAD == 0))
					continue;
				const bool adEnabled = (adMode > 0);
				SECTION(std::string(isKinetic ? " Kinetic binding" : " Quasi-stationary binding") + " with AD " + (adEnabled ? "enabled" : "disabled"))
				{
					// Use some test case parameters
					cadet::JsonParameterProvider jpp = linearBinding ? createColumnWithTwoCompLinearBinding(uoType, spatialMethod) : createColumnWithSMA(uoType, spatialMethod);
					cadet::test::setBindingMode(jpp, isKinetic);
					cadet::IUnitOperation* const unit = createAndConfigureUnit(*mb, jpp);

					unit->setSensitiveParameter(cadet::makeParamId("INIT_C", 0, 0, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 0, 1.0);
					if (linearBinding)
						unit->setSensitiveParameter(cadet::makeParamId("LIN_KA", 0, 0, cadet::ParTypeIndep, 0, cadet::ReactionIndep, cadet::SectionIndep), 1, 1.0);
					else
						unit->setSensitiveParameter(cadet::makeParamId("SMA_NU", 0, 1, cadet::ParTypeIndep, 0, cadet::ReactionIndep, cadet::SectionIndep), 1, 1.0);

					unit->setSensitiveParameter(cadet::makeParamId("COL_LENGTH", 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep), 2, 1.0);

					REQUIRE(unit->numSensParams() == 3);
					unitoperation::testConsistentInitializationSensitivity(unit, adEnabled, y, yDot, absTol);

					mb->destroyUnitOperation(unit);
				}
			}
		}
		destroyModelBuilder(mb);
	}

	void testInletDofJacobian(const std::string& uoType, const std::string& spatialMethod)
	{
		cadet::IModelBuilder* const mb = cadet::createModelBuilder();
		REQUIRE(nullptr != mb);

		for (int adMode = 0; adMode < 2; ++adMode)
		{
			const bool adEnabled = (adMode > 0);
			SECTION(std::string("AD ") + (adEnabled ? "enabled" : "disabled"))
			{
				// Use some test case parameters
				cadet::JsonParameterProvider jpp = createColumnWithSMA(uoType, spatialMethod);
				cadet::IUnitOperation* const unit = createAndConfigureUnit(*mb, jpp);

				unitoperation::testInletDofJacobian(unit, adEnabled);

				mb->destroyUnitOperation(unit);
			}
		}
		destroyModelBuilder(mb);
	}

	void testReferenceBenchmark(const std::string& modelFileRelPath, const std::string& refFileRelPath, const std::string& unitID, const std::vector<double> absTol, const std::vector<double> relTol, const DiscParams& disc, const bool compare_sens, const int simDataStride)
	{
		const int unitOpID = std::stoi(unitID);

		// read json model setup file
		const std::string setupFile = std::string(getTestDirectory()) + modelFileRelPath;
		JsonParameterProvider pp_setup(JsonParameterProvider::fromFile(setupFile));

		// adjust numerical parameters
		nlohmann::json* setupJson = pp_setup.data();

		// copy over some numerical parameters from reference file, e.g. consistent initialization
		cadet::io::HDF5Reader rd;
		const std::string refFile = std::string(getTestDirectory()) + refFileRelPath;
		rd.openFile(refFile, "r");
		ParameterProviderImpl<cadet::io::HDF5Reader> pp_ref(rd);
		setNumericalMethod(pp_ref, *setupJson, unitID, true);
		pp_ref.popScope();

		// copy solution times
		pp_ref.pushScope("input");
		pp_ref.pushScope("solver");
		setupJson[0]["solver"]["USER_SOLUTION_TIMES"] = pp_ref.getDoubleArray("USER_SOLUTION_TIMES");
		pp_ref.popScope();

		// copy multiplex data
		copyMultiplexData(pp_ref, *setupJson, unitID);

		// copy return data
		copyReturnData(pp_ref, *setupJson, unitID);

		// copy sensitivity setup
		if (pp_ref.exists("sensitivity") && compare_sens)
			copySensitivities(pp_ref, *setupJson, unitID);
		
		pp_ref.popScope();
		rd.closeFile();

		// set remaining spatial numerical parameters
		disc.setDisc(pp_setup, unitID);

		// run simulation
		Driver drv;
		drv.configure(pp_setup);
		drv.run();

		// get simulation result
		InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(unitOpID);
		double const* sim_outlet = simData->outlet();

		// read h5 reference data
		rd.openFile(refFile, "r");

		// get outlet and sensitivity reference
		pp_ref.pushScope("output");
		pp_ref.pushScope("solution");
		pp_ref.pushScope("unit_" + unitID);
		std::vector<double> ref_outlet;
		if (pp_ref.exists("SOLUTION_OUTLET"))
			ref_outlet = pp_ref.getDoubleArray("SOLUTION_OUTLET");
		else
			ref_outlet = pp_ref.getDoubleArray("SOLUTION_OUTLET_PORT_000");
		pp_ref.popScope();
		pp_ref.popScope();

		// compare the simulation results with the reference data
		for (unsigned int i = 0; i < ref_outlet.size(); ++i)
			CHECK((sim_outlet[i * simDataStride]) == cadet::test::makeApprox(ref_outlet[i], relTol[0], absTol[0]));

		if (pp_ref.exists("sensitivity") && compare_sens)
		{
			pp_ref.pushScope("sensitivity");

			unsigned int sensID = 0;
			std::string sensParam = std::to_string(sensID);
			sensParam = "param_" + std::string(3 - sensParam.length(), '0') + sensParam;
			CAPTURE(sensParam);

			while (pp_ref.exists(sensParam))
			{
				pp_ref.pushScope(sensParam);
				pp_ref.pushScope("unit_" + unitID);
				const std::vector<double> ref_sens = pp_ref.getDoubleArray("SENS_OUTLET");
				pp_ref.popScope();
				pp_ref.popScope();

				double const* sim_sens = simData->sensOutlet(sensID);

				for (unsigned int i = 0; i < ref_sens.size(); ++i)
					CHECK((sim_sens[i]) == cadet::test::makeApprox(ref_sens[i], relTol[sensID + 1], absTol[sensID + 1]));

				sensID++;
				sensParam = std::to_string(sensID);
				sensParam = "param_" + std::string(3 - sensParam.length(), '0') + sensParam;
			}
		}
		rd.closeFile();
	}

	// todo ? include L1 errors or parameterize error choice ?
	void testEOCReferenceBenchmark(const std::string& modelFileRelPath, const std::string& refFileRelPath, const std::string& convFileRelPath, const std::string& unitID, const std::vector<double> absTol, const std::vector<double> relTol, const unsigned int nDisc, const DiscParams& startDisc, const bool compare_sens)
	{
		const int unitOpID = std::stoi(unitID);

		// read model setup file
		const std::string setupFile = std::string(getTestDirectory()) + modelFileRelPath;
		JsonParameterProvider pp_setup(JsonParameterProvider::fromFile(setupFile));

		// adjust numerical parameters
		nlohmann::json* setupJson = pp_setup.data();

		// copy over some numerical parameters from reference file, e.g. consistent initialization
		cadet::io::HDF5Reader rd;
		const std::string refFile = std::string(getTestDirectory()) + refFileRelPath;
		rd.openFile(refFile, "r");
		ParameterProviderImpl<cadet::io::HDF5Reader> pp_ref(rd);
		setNumericalMethod(pp_ref, *setupJson, unitID);
		pp_ref.popScope();

		// copy solution times
		pp_ref.pushScope("input");
		pp_ref.pushScope("solver");
		setupJson[0]["solver"]["USER_SOLUTION_TIMES"] = pp_ref.getDoubleArray("USER_SOLUTION_TIMES");
		pp_ref.popScope();

		// copy multiplex data
		copyMultiplexData(pp_ref, *setupJson, unitID);

		// copy sensitivity setup
		int nSens = 0;
		if (pp_ref.exists("sensitivity"))
		{
			pp_ref.pushScope("sensitivity");
			nSens = pp_ref.getInt("NSENS");
			pp_ref.popScope();
		}

		// copy return data
		copyReturnData(pp_ref, *setupJson, unitID);

		// configure sensitivities
		if (nSens && compare_sens)
			copySensitivities(pp_ref, *setupJson, unitID);

		pp_ref.popScope();

		// read h5 reference data
		pp_ref.pushScope("output");
		pp_ref.pushScope("solution");
		pp_ref.pushScope("unit_" + unitID);
		const std::vector<double> ref_outlet = pp_ref.getDoubleArray("SOLUTION_OUTLET");
		pp_ref.popScope();
		pp_ref.popScope();
		pp_ref.popScope();

		// read convergence file
		const std::string convFile = std::string(getTestDirectory()) + convFileRelPath;
		JsonParameterProvider pp_conv(JsonParameterProvider::fromFile(convFile));

		pp_conv.pushScope("convergence");
		pp_conv.pushScope("outlet");
		std::vector<double> discZ = pp_conv.getDoubleArray("$N_e^z$");
		std::vector<double> discP;
		const int startNCol = startDisc.getNAxCells();
		const int startNPar = startDisc.getNParCells();
		if (startNPar > 0)
			discP = pp_conv.getDoubleArray("$N_e^p$");
		pp_conv.popScope();

		int discIdx = 0;
		auto it = std::find(discZ.begin(), discZ.end(), static_cast<double>(startNCol));
		if (it != discZ.end())
			discIdx = std::distance(discZ.begin(), it);
		else
			throw std::out_of_range("discretization N_e^z = " + std::to_string(startNCol) + " not found in convergence reference data");
		if (startNPar > 0 && discP[discIdx] != static_cast<double>(startNPar))
			throw std::out_of_range("discretization N_e^p = " + std::to_string(startNPar) + " not found in convergence reference data at the same index as N_e^z");

		// run the simulations and compute EOC
		std::vector<double> absErrors(ref_outlet.size());
		std::vector<std::vector<double>> L1Errors(1 + nSens, std::vector<double>(nDisc, 0.0));
		std::vector<std::vector<double>> LinfErrors(1 + nSens, std::vector<double>(nDisc, 0.0));
		std::vector<std::vector<double>> L1EOC(1 + nSens, std::vector<double>(nDisc, 0.0));
		std::vector<std::vector<double>> LinfEOC(1 + nSens, std::vector<double>(nDisc, 0.0));

		for (int disc = 1; disc <= nDisc; disc++)
		{
			const int nAxCells = startNCol * std::pow(2, disc - 1);
			const int nParCells = startNPar * std::pow(2, disc - 1);
			setNumAxialCells(pp_setup, nAxCells, unitID);
			if (startNPar > 0)
				setNumParCells(pp_setup, nParCells, unitID);

			// run simulation
			Driver drv;
			drv.configure(pp_setup);
			drv.run();

			// compute errors and EOC
			InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(unitOpID);
			double const* approximation = simData->outlet();

			std::transform(approximation, approximation + ref_outlet.size(), ref_outlet.begin(), absErrors.begin(),
				[](double a, double b) { return std::abs(a - b); });

			L1Errors[0][disc - 1] = std::accumulate(absErrors.begin(), absErrors.end(), 0.0) / ref_outlet.size();
			LinfErrors[0][disc - 1] = *std::max_element(absErrors.begin(), absErrors.end());

			pp_conv.pushScope("outlet");
			std::vector<double> LinfErrors_ref = pp_conv.getDoubleArray("Max. error");
			std::vector<double> LinfEOC_ref = pp_conv.getDoubleArray("Max. EOC");
			pp_conv.popScope();

			CAPTURE(nAxCells);
			CAPTURE(LinfErrors[0][disc - 1]);
			CAPTURE(LinfErrors_ref[discIdx + disc - 1]);
			CHECK((LinfErrors[0][disc - 1]) == cadet::test::makeApprox(LinfErrors_ref[discIdx + disc - 1], relTol[0], absTol[0]));

			if (disc > 1)
			{
				L1EOC[0][disc - 1] = std::log(L1Errors[0][disc - 1] / L1Errors[0][disc - 2]) / std::log(std::pow(2, disc - 1) / std::pow(2, disc));
				LinfEOC[0][disc - 1] = std::log(LinfErrors[0][disc - 1] / LinfErrors[0][disc - 2]) / std::log(std::pow(2, disc - 1) / std::pow(2, disc));
				CAPTURE(LinfEOC[0][disc - 1]);
				CAPTURE(LinfEOC_ref[discIdx + disc - 1]);
				CHECK((LinfEOC[0][disc - 1]) == cadet::test::makeApprox(LinfEOC_ref[discIdx + disc - 1], relTol[0], absTol[0]));
			}

			// compare sensitivity EOC
			if (nSens && compare_sens)
			{
				for (int sensID = 0; sensID < nSens; sensID++)
				{
					// get reference errors and EOC (to this end get sensitivity name to push convergence scope to reference solution)
					pp_ref.pushScope("input");
					pp_ref.pushScope("sensitivity");
					std::string sensParam = std::to_string(sensID);
					sensParam = "param_" + std::string(3 - sensParam.length(), '0') + sensParam;
					pp_ref.pushScope(sensParam);
					std::string sens_name = pp_ref.getString("SENS_NAME");
					pp_conv.pushScope("sens_" + sens_name);
					LinfErrors_ref = pp_conv.getDoubleArray("Max. error");
					LinfEOC_ref = pp_conv.getDoubleArray("Max. EOC");
					pp_conv.popScope();
					pp_ref.popScope();
					pp_ref.popScope();
					pp_ref.popScope();

					// compute error and EOC
					pp_ref.pushScope("output");
					pp_ref.pushScope("sensitivity");
					pp_ref.pushScope(sensParam);
					pp_ref.pushScope("unit_" + unitID);
					const std::vector<double> ref_sens = pp_ref.getDoubleArray("SENS_OUTLET");

					approximation = simData->sensOutlet(sensID);

					std::transform(approximation, approximation + ref_sens.size(), ref_sens.begin(), absErrors.begin(),
						[](double a, double b) { return std::abs(a - b); });
					L1Errors[sensID+1][disc - 1] = std::accumulate(absErrors.begin(), absErrors.end(), 0.0) / ref_sens.size();
					LinfErrors[sensID + 1][disc - 1] = *std::max_element(absErrors.begin(), absErrors.end());

					CAPTURE(sensID, sens_name);
					CAPTURE(LinfErrors[sensID + 1][disc - 1]);
					CAPTURE(LinfErrors_ref[discIdx + disc - 1]);
					CHECK((LinfErrors[sensID + 1][disc - 1]) == cadet::test::makeApprox(LinfErrors_ref[discIdx + disc - 1], relTol[1 + sensID], absTol[1 + sensID]));
					
					if (disc > 1)
					{
						L1EOC[sensID + 1][disc - 1] = std::log(L1Errors[sensID + 1][disc - 1] / L1Errors[sensID + 1][disc - 2]) / std::log(std::pow(2, disc - 1) / std::pow(2, disc));
						LinfEOC[sensID + 1][disc - 1] = std::log(LinfErrors[sensID + 1][disc - 1] / LinfErrors[sensID + 1][disc - 2]) / std::log(std::pow(2, disc - 1) / std::pow(2, disc));
						CAPTURE(LinfEOC[sensID + 1][disc - 1]);
						CAPTURE(LinfEOC_ref[discIdx + disc - 1]);
						CHECK((LinfEOC[sensID + 1][disc - 1]) == cadet::test::makeApprox(LinfEOC_ref[discIdx + disc - 1], relTol[1 + sensID], absTol[1 + sensID]));
					}
					pp_ref.popScope();
					pp_ref.popScope();
					pp_ref.popScope();
					pp_ref.popScope();
				}
			}
		}
		rd.closeFile();
	}

	void testForeignReferenceBenchmark(const std::string& configFileRelPath, const std::string& refFileRelPath, const std::string& unitID, const double absTol, const double relTol, const int compIdx)
	{
		const int unitOpID = std::stoi(unitID);

		// read json model setup file
		const std::string setupFile = std::string(getTestDirectory()) + configFileRelPath;
		JsonParameterProvider pp_setup(JsonParameterProvider::fromFile(setupFile));

		// adjust numerical parameters
		nlohmann::json* setupJson = pp_setup.data();

		// run simulation
		Driver drv;
		drv.configure(pp_setup);
		drv.run();

		// get simulation result
		InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(unitOpID);
		double const* sim_outlet = simData->outlet();

		// read h5 reference data
		cadet::io::HDF5Reader rd;
		const std::string refFile = std::string(getTestDirectory()) + refFileRelPath;
		rd.openFile(refFile, "r");
		ParameterProviderImpl<cadet::io::HDF5Reader> pp_ref(rd);
		pp_ref.popScope();
		pp_ref.pushScope("output");
		pp_ref.pushScope("solution");
		const std::vector<double> ref_outlet = pp_ref.getDoubleArray("SOLUTION_OUTLET");

		// check if only one specific component is considered in the reference solution and set stride and offset accordingly
		const unsigned int compStride = (compIdx == -1) ? 1 : simData->numComponents();
		sim_outlet += (compIdx == -1) ? 0 : compIdx;

		// compare the simulation results with the reference data
		for (unsigned int i = 0, j = 0; j < ref_outlet.size(); i += compStride, j += 1)
			CHECK((sim_outlet[i]) == cadet::test::makeApprox(ref_outlet[j], relTol, absTol));

		rd.closeFile();
	}

} // namespace column
} // namespace test
} // namespace cadet
