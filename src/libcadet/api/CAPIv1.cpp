// =============================================================================
//  CADET - The Chromatography Analysis and Design Toolkit
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include "cadet/cadet.h"

#include "common/CompilerSpecific.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/Exceptions.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include "common/Driver.hpp"


#define CADET_XSTR(a) #a
#define CADET_STR(a) CADET_XSTR(a)

#define CADET_COMMA ,


extern "C"
{
	struct cdtDriver
	{
		cadet::Driver* driver;
	};
}

namespace cadet
{

namespace api
{

namespace v1
{

	cdtDriver* createDriver()
	{
		return new cdtDriver{ new cadet::Driver() };
	}

	void deleteDriver(cdtDriver* drv)
	{
		if (!drv)
			return;
		if (!drv->driver)
			return;

		delete drv->driver;
		drv->driver = nullptr;
	}

	/**
	 * @brief ParameterProvider implementation using C callback functions
	 */
	class CallbackParameterProvider : public IParameterProvider
	{
	public:

		CallbackParameterProvider(const cdtParameterProvider& pp) : _pp(pp)
		{
			if (!pp.getDouble)
				throw InvalidParameterException("ParameterProvider does not implement getDouble()");
			if (!pp.getInt)
				throw InvalidParameterException("ParameterProvider does not implement getInt()");
			if (!pp.getBool)
				throw InvalidParameterException("ParameterProvider does not implement getBool()");
			if (!pp.getString)
				throw InvalidParameterException("ParameterProvider does not implement getString()");
			if (!pp.getDoubleArray && !pp.getDoubleArrayItem)
				throw InvalidParameterException("ParameterProvider does neither implement getDoubleArray() nor getDoubleArrayItem()");
			if (!pp.getIntArray && !pp.getIntArrayItem)
				throw InvalidParameterException("ParameterProvider does neither implement getIntArray() nor getIntArrayItem()");
			if (!pp.getBoolArray && !pp.getBoolArrayItem)
				throw InvalidParameterException("ParameterProvider does neither implement getBoolArray() nor getBoolArrayItem()");
			if (!pp.getStringArray && !pp.getStringArrayItem)
				throw InvalidParameterException("ParameterProvider does neither implement getStringArray() nor getStringArrayItem()");
			if (!pp.exists)
				throw InvalidParameterException("ParameterProvider does not implement exists()");
			if (!pp.isArray)
				throw InvalidParameterException("ParameterProvider does not implement isArray()");
			if (!pp.numElements)
				throw InvalidParameterException("ParameterProvider does not implement numElements()");
			if (!pp.pushScope)
				throw InvalidParameterException("ParameterProvider does not implement pushScope()");
			if (!pp.popScope)
				throw InvalidParameterException("ParameterProvider does not implement popScope()");
		}
		virtual ~CallbackParameterProvider() CADET_NOEXCEPT { }

		virtual double getDouble(const std::string& paramName)
		{
			double v = 0.0;
			if (CADET_ERR(_pp.getDouble(_pp.userData, paramName.c_str(), &v)))
				throw InvalidParameterException("Retrieving double parameter " + paramName + " failed");

			LOG(Debug) << "GET scalar [double] " << paramName << ": " << v;
			return v;
		}

		virtual int getInt(const std::string& paramName)
		{
			int v = 0;
			if (CADET_ERR(_pp.getInt(_pp.userData, paramName.c_str(), &v)))
				throw InvalidParameterException("Retrieving int parameter " + paramName + " failed");

			LOG(Debug) << "GET scalar [int] " << paramName << ": " << v;
			return v;
		}

		virtual uint64_t getUint64(const std::string& paramName)
		{
			throw std::logic_error("getUint64 not implemented");
		}

		virtual bool getBool(const std::string& paramName)
		{
			uint8_t v = 0;
			if (CADET_ERR(_pp.getBool(_pp.userData, paramName.c_str(), &v)))
				throw InvalidParameterException("Retrieving bool parameter " + paramName + " failed");

			LOG(Debug) << "GET scalar [bool] " << paramName << ": " << bool(v);
			return v;
		}

		virtual std::string getString(const std::string& paramName)
		{
			char const* v = nullptr;
			if (CADET_ERR(_pp.getString(_pp.userData, paramName.c_str(), &v)))
				throw InvalidParameterException("Retrieving string parameter " + paramName + " failed");

			if (!v)
				throw InvalidParameterException("Retrieving string parameter " + paramName + " failed (received nullptr)");

			LOG(Debug) << "GET scalar [string] " << paramName << ": " << v << " (mem: " << static_cast<void*>(&v) << " " << static_cast<void const*>(v) << ")";
			return std::string(v);
		}

		virtual std::vector<double> getDoubleArray(const std::string& paramName)
		{
			if (_pp.getDoubleArray)
			{
				int num = 0;
				double* v = nullptr;
				if (CADET_ERR(_pp.getDoubleArray(_pp.userData, paramName.c_str(), &num, &v)))
				{
					if (_pp.getDoubleArrayItem)
						return getDoubleArrayElementwise(paramName);
					else
						throw InvalidParameterException("Retrieving double parameter array " + paramName + " failed");
				}

				LOG(Debug) << "GET array [double] " << paramName << ": " << log::VectorPtr<double>(v, num);
				return std::vector<double>(v, v + num);
			}

			return getDoubleArrayElementwise(paramName);
		}

		virtual std::vector<int> getIntArray(const std::string& paramName)
		{
			if (_pp.getIntArray)
			{
				int num = 0;
				int* v = nullptr;
				if (CADET_ERR(_pp.getIntArray(_pp.userData, paramName.c_str(), &num, &v)))
				{
					if (_pp.getIntArrayItem)
						return getIntArrayElementwise(paramName);
					else
						throw InvalidParameterException("Retrieving int parameter array " + paramName + " failed");
				}

				LOG(Debug) << "GET array [int] " << paramName << ": " << log::VectorPtr<int>(v, num);
				return std::vector<int>(v, v + num);
			}

			return getIntArrayElementwise(paramName);
		}

		virtual std::vector<uint64_t> getUint64Array(const std::string& paramName)
		{
			throw std::logic_error("getUint64Array not implemented");
		}

		virtual std::vector<bool> getBoolArray(const std::string& paramName)
		{
			if (_pp.getBoolArray)
			{
				int num = 0;
				uint8_t* v = nullptr;
				if (CADET_ERR(_pp.getBoolArray(_pp.userData, paramName.c_str(), &num, &v)))
				{
					if (_pp.getBoolArrayItem)
						return getBoolArrayElementwise(paramName);
					else
						throw InvalidParameterException("Retrieving bool parameter array " + paramName + " failed");
				}

				std::vector<bool> vc(num);
				for (int i = 0; i < num; ++i)
					vc[i] = v[i];

				LOG(Debug) << "GET array [bool] " << paramName << ": " << vc;
				return vc;
			}

			return getBoolArrayElementwise(paramName);
		}

		virtual std::vector<std::string> getStringArray(const std::string& paramName)
		{
			if (_pp.getStringArray)
			{
				int num = 0;
				char const** v = nullptr;
				if (CADET_ERR(_pp.getStringArray(_pp.userData, paramName.c_str(), &num, &v)))
				{
					if (_pp.getStringArrayItem)
						return getStringArrayElementwise(paramName);
					else
						throw InvalidParameterException("Retrieving string parameter array " + paramName + " failed");
				}

				std::vector<std::string> vc(num);
				for (int i = 0; i < num; ++i)
					vc[i] = v[i];

				LOG(Debug) << "GET array [string] " << paramName << ": " << vc;
				return vc;
			}

			return getStringArrayElementwise(paramName);
		}

		virtual bool exists(const std::string& paramName)
		{
			const bool r = _pp.exists(_pp.userData, paramName.c_str()) != 0;
			LOG(Debug) << "EXISTS " << paramName << ": " << r;
			return r;
		}

		virtual bool isArray(const std::string& paramName)
		{
			uint8_t res = 0;
			if (CADET_ERR(_pp.isArray(_pp.userData, paramName.c_str(), &res)))
				throw InvalidParameterException("Checking whether parameter " + paramName + " is array failed");

			LOG(Debug) << "ISARRAY " << paramName << ": " << (res != 0);
			return res != 0;
		}

		virtual std::size_t numElements(const std::string& paramName)
		{
			const int num = _pp.numElements(_pp.userData, paramName.c_str());
			if (num < 0)
				throw InvalidParameterException("Retrieving string parameter array " + paramName + " failed (does not exist)");

			LOG(Debug) << "NUMELEMENTS " << paramName << ": " << num;
			return num;
		}

		virtual void pushScope(const std::string& scope)
		{
			if (CADET_ERR(_pp.pushScope(_pp.userData, scope.c_str())))
				throw InvalidParameterException("Failed to enter scope " + scope);

			LOG(Debug) << "PUSHSCOPE " << scope;
		}

		virtual void popScope()
		{
			if (CADET_ERR(_pp.popScope(_pp.userData)))
				throw InvalidParameterException("Failed to exit current scope");

			LOG(Debug) << "POPSCOPE";
		}

	private:
		const cdtParameterProvider& _pp;

		std::vector<double> getDoubleArrayElementwise(const std::string& paramName)
		{
			const int num = _pp.numElements(_pp.userData, paramName.c_str());
			if (num < 0)
				throw InvalidParameterException("Retrieving double parameter array " + paramName + " failed (does not exist)");

			if (num == 0)
				return std::vector<double>(0);

			std::vector<double> v(num);
			for (int i = 0; i < num; ++i)
			{
				if (CADET_ERR(_pp.getDoubleArrayItem(_pp.userData, paramName.c_str(), i, v.data() + i)))
					throw InvalidParameterException("Retrieving double parameter " + paramName + " failed");

				LOG(Debug) << "GET array (" << i << ") [double] " << paramName << ": " << v[i];
			}

			return v;
		}

		std::vector<int> getIntArrayElementwise(const std::string& paramName)
		{
			const int num = _pp.numElements(_pp.userData, paramName.c_str());
			if (num < 0)
				throw InvalidParameterException("Retrieving int parameter array " + paramName + " failed (does not exist)");

			if (num == 0)
				return std::vector<int>(0);

			std::vector<int> v(num);
			for (int i = 0; i < num; ++i)
			{
				if (CADET_ERR(_pp.getIntArrayItem(_pp.userData, paramName.c_str(), i, v.data() + i)))
					throw InvalidParameterException("Retrieving int parameter " + paramName + " failed");

				LOG(Debug) << "GET array (" << i << ") [int] " << paramName << ": " << v[i];
			}

			return v;
		}

		std::vector<bool> getBoolArrayElementwise(const std::string& paramName)
		{
			const int num = _pp.numElements(_pp.userData, paramName.c_str());
			if (num < 0)
				throw InvalidParameterException("Retrieving bool parameter array " + paramName + " failed (does not exist)");

			if (num == 0)
				return std::vector<bool>(0);

			std::vector<bool> v(num);
			for (int i = 0; i < num; ++i)
			{
				uint8_t temp = 0;
				if (CADET_ERR(_pp.getBoolArrayItem(_pp.userData, paramName.c_str(), i, &temp)))
					throw InvalidParameterException("Retrieving bool parameter " + paramName + " failed");

				v[i] = temp;
				LOG(Debug) << "GET array (" << i << ") [bool] " << paramName << ": " << static_cast<bool>(v[i]);
			}

			return v;
		}

		std::vector<std::string> getStringArrayElementwise(const std::string& paramName)
		{
			const int num = _pp.numElements(_pp.userData, paramName.c_str());
			if (num < 0)
				throw InvalidParameterException("Retrieving string parameter array " + paramName + " failed (does not exist)");

			if (num == 0)
				return std::vector<std::string>(0);

			std::vector<std::string> v(num);
			for (int i = 0; i < num; ++i)
			{
				char const* temp = nullptr;
				if (CADET_ERR(_pp.getStringArrayItem(_pp.userData, paramName.c_str(), i, &temp)))
					throw InvalidParameterException("Retrieving string parameter " + paramName + " failed");

				if (!temp)
					throw InvalidParameterException("Retrieving string parameter " + paramName + " failed (received nullptr)");

				v[i] = temp;
				LOG(Debug) << "GET array (" << i << ") [string] " << paramName << ": " << v[i];
			}

			return v;
		}
	};


	cdtResult runSimulation(cdtDriver* drv, cdtParameterProvider const* paramProvider)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
			return cdtErrorInvalidInputs;
		if (!paramProvider)
			return cdtErrorInvalidInputs;

		try
		{
			if (!realDrv->simulator())
			{
				CallbackParameterProvider cpp(*paramProvider);
				realDrv->configure(cpp);
			}
			else
				realDrv->clearResults();

			realDrv->run();

		}
		catch(const std::exception& e)
		{
			LOG(Error) << "Simulation failed: " << e.what();
			return cdtError;
		}

		return cdtOK;
	}

	cdtResult getNumParTypes(cdtDriver* drv, int unitOpId, int* nParTypes)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
			return cdtErrorInvalidInputs;

		InternalStorageSystemRecorder* const sysRec = realDrv->solution();
		if (!sysRec)
		{
			LOG(Error) << "System solution recorder not available";
			return cdtError;
		}

		InternalStorageUnitOpRecorder* const unitRec = sysRec->unitOperation(unitOpId);
		if (!unitRec)
		{
			LOG(Error) << "Solution recorder for unit ID " << unitOpId << " not found";
			return cdtErrorInvalidInputs;
		}

		if (nParTypes)
			*nParTypes = unitRec->numParticleTypes();

		return cdtOK;
	}


	InternalStorageUnitOpRecorder* getUnitRecorder(cdtDriver* drv, int unitOpId, double const** time, int* nTime, cdtResult& retCode)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
		{
			retCode = cdtErrorInvalidInputs;
			return nullptr;
		}

		InternalStorageSystemRecorder* const sysRec = realDrv->solution();
		if (!sysRec)
		{
			LOG(Error) << "System solution recorder not available";
			retCode = cdtError;
			return nullptr;
		}

		if (nTime)
			*nTime = sysRec->numDataPoints();
		if (time)
			*time = sysRec->time();

		InternalStorageUnitOpRecorder* const unitRec = sysRec->unitOperation(unitOpId);
		if (!unitRec)
		{
			LOG(Error) << "Solution recorder for unit ID " << unitOpId << " not found";
			retCode = cdtErrorInvalidInputs;
			return nullptr;
		}
		return unitRec;
	}


	#define CADET_API_GET_INLET_OUTLET(TYPE, NAME, REC_CONFIG, REC_QUERY, SENS_IDX_SIG, SENS_IDX) \
		cdtResult get##NAME##TYPE(cdtDriver* drv, int unitOpId SENS_IDX_SIG, double const** time, double const** data, int* nTime, int* nPort, int* nComp) \
		{ \
			cdtResult retCode = cdtOK; \
			InternalStorageUnitOpRecorder* const unitRec = getUnitRecorder(drv, unitOpId, time, nTime, retCode); \
			if (!unitRec) \
				return retCode; \
	\
			if (!unitRec->REC_CONFIG().store##TYPE) \
			{ \
				LOG(Error) << CADET_STR(TYPE)" of unit " << unitOpId << " not recorded"; \
				return cdtDataNotStored; \
			} \
	\
			if (nPort) \
				*nPort = unitRec->num##TYPE##Ports(); \
			if (nComp) \
				*nComp = unitRec->numComponents(); \
			if (data) \
				*data = unitRec->REC_QUERY( SENS_IDX ); \
	\
			return cdtOK; \
		}


	CADET_API_GET_INLET_OUTLET(Inlet, Solution, solutionConfig, inlet,,)
	CADET_API_GET_INLET_OUTLET(Inlet, Sensitivity, sensitivityConfig, sensInlet, CADET_COMMA int sensIdx, sensIdx )
	CADET_API_GET_INLET_OUTLET(Outlet, Solution, solutionConfig, outlet,,)
	CADET_API_GET_INLET_OUTLET(Outlet, Sensitivity, sensitivityConfig, sensOutlet, CADET_COMMA int sensIdx, sensIdx )

	CADET_API_GET_INLET_OUTLET(Inlet, SolutionDerivative, solutionDotConfig, inletDot,,)
	CADET_API_GET_INLET_OUTLET(Inlet, SensitivityDerivative, sensitivityDotConfig, sensInletDot, CADET_COMMA int sensIdx, sensIdx )
	CADET_API_GET_INLET_OUTLET(Outlet, SolutionDerivative, solutionDotConfig, outletDot,,)
	CADET_API_GET_INLET_OUTLET(Outlet, SensitivityDerivative, sensitivityDotConfig, sensOutletDot, CADET_COMMA int sensIdx, sensIdx )


	#define CADET_API_GET_BULK(NAME, REC_CONFIG, REC_QUERY, SENS_IDX_SIG, SENS_IDX) \
		cdtResult get##NAME##Bulk(cdtDriver* drv, int unitOpId SENS_IDX_SIG, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nComp) \
		{ \
			cdtResult retCode = cdtOK; \
			InternalStorageUnitOpRecorder* const unitRec = getUnitRecorder(drv, unitOpId, time, nTime, retCode); \
			if (!unitRec) \
				return retCode; \
		\
			if (!unitRec->REC_CONFIG().storeBulk) \
			{ \
				LOG(Error) << "Bulk of unit " << unitOpId << " not recorded"; \
				return cdtDataNotStored; \
			} \
		\
			if (nAxialCells) \
				*nAxialCells = unitRec->numAxialCells(); \
			if (nRadialCells) \
				*nRadialCells = unitRec->numRadialCells(); \
			if (nComp) \
				*nComp = unitRec->numComponents(); \
			if (data) \
				*data = unitRec->REC_QUERY( SENS_IDX ); \
		\
			return cdtOK; \
		}

	CADET_API_GET_BULK(Solution, solutionConfig, bulk,,)
	CADET_API_GET_BULK(Sensitivity, sensitivityConfig, sensBulk, CADET_COMMA int sensIdx, sensIdx )
	CADET_API_GET_BULK(SolutionDerivative, solutionDotConfig, bulkDot,,)
	CADET_API_GET_BULK(SensitivityDerivative, sensitivityDotConfig, sensBulkDot, CADET_COMMA int sensIdx, sensIdx )


	#define CADET_API_GET_PARTICLE(NAME, REC_CONFIG, REC_QUERY, SENS_IDX_SIG, SENS_IDX) \
		cdtResult get##NAME##Particle(cdtDriver* drv, int unitOpId SENS_IDX_SIG, int parType, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParShells, int* nComp) \
		{ \
			cdtResult retCode = cdtOK; \
			InternalStorageUnitOpRecorder* const unitRec = getUnitRecorder(drv, unitOpId, time, nTime, retCode); \
			if (!unitRec) \
				return retCode; \
		\
			if (!unitRec->REC_CONFIG().storeParticle) \
			{ \
				LOG(Error) << "Particle of unit " << unitOpId << " not recorded"; \
				return cdtDataNotStored; \
			} \
		\
			if (nAxialCells) \
				*nAxialCells = unitRec->numAxialCells(); \
			if (nRadialCells) \
				*nRadialCells = unitRec->numRadialCells(); \
			if (nParShells) \
				*nParShells = unitRec->numParticleShells(parType); \
			if (nComp) \
				*nComp = unitRec->numComponents(); \
			if (data) \
				*data = unitRec->REC_QUERY( SENS_IDX parType); \
		\
			return cdtOK; \
		}

	CADET_API_GET_PARTICLE(Solution, solutionConfig, particle,,)
	CADET_API_GET_PARTICLE(Sensitivity, sensitivityConfig, sensParticle, CADET_COMMA int sensIdx, sensIdx CADET_COMMA )
	CADET_API_GET_PARTICLE(SolutionDerivative, solutionDotConfig, particleDot,,)
	CADET_API_GET_PARTICLE(SensitivityDerivative, sensitivityDotConfig, sensParticleDot, CADET_COMMA int sensIdx, sensIdx CADET_COMMA )


	#define CADET_API_GET_SOLID(NAME, REC_CONFIG, REC_QUERY, SENS_IDX_SIG, SENS_IDX) \
		cdtResult get##NAME##Solid(cdtDriver* drv, int unitOpId SENS_IDX_SIG, int parType, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParShells, int* nBound) \
		{ \
			cdtResult retCode = cdtOK; \
			InternalStorageUnitOpRecorder* const unitRec = getUnitRecorder(drv, unitOpId, time, nTime, retCode); \
			if (!unitRec) \
				return retCode; \
		\
			if (!unitRec->REC_CONFIG().storeOutlet) \
			{ \
				LOG(Error) << "Solid of unit " << unitOpId << " not recorded"; \
				return cdtDataNotStored; \
			} \
		\
			if (nAxialCells) \
				*nAxialCells = unitRec->numAxialCells(); \
			if (nRadialCells) \
				*nRadialCells = unitRec->numRadialCells(); \
			if (nParShells) \
				*nParShells = unitRec->numParticleShells(parType); \
			if (nBound) \
				*nBound = unitRec->numBoundStates(); \
			if (data) \
				*data = unitRec->REC_QUERY( SENS_IDX parType ); \
		\
			return cdtOK; \
		}


	CADET_API_GET_SOLID(Solution, solutionConfig, solid,,)
	CADET_API_GET_SOLID(Sensitivity, sensitivityConfig, sensSolid, CADET_COMMA int sensIdx, sensIdx CADET_COMMA )
	CADET_API_GET_SOLID(SolutionDerivative, solutionDotConfig, solidDot,,)
	CADET_API_GET_SOLID(SensitivityDerivative, sensitivityDotConfig, sensSolidDot, CADET_COMMA int sensIdx, sensIdx CADET_COMMA )


	#define CADET_API_GET_FLUX(NAME, REC_CONFIG, REC_QUERY, SENS_IDX_SIG, SENS_IDX) \
		cdtResult get##NAME##Flux(cdtDriver* drv, int unitOpId SENS_IDX_SIG, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParType, int* nComp) \
		{ \
			cdtResult retCode = cdtOK; \
			InternalStorageUnitOpRecorder* const unitRec = getUnitRecorder(drv, unitOpId, time, nTime, retCode); \
			if (!unitRec) \
				return retCode; \
		\
			if (!unitRec->REC_CONFIG().storeFlux) \
			{ \
				LOG(Error) << "Flux of unit " << unitOpId << " not recorded"; \
				return cdtDataNotStored; \
			} \
		\
			if (nAxialCells) \
				*nAxialCells = unitRec->numAxialCells(); \
			if (nRadialCells) \
				*nRadialCells = unitRec->numRadialCells(); \
			if (nParType) \
				*nParType = unitRec->numParticleTypes(); \
			if (nComp) \
				*nComp = unitRec->numComponents(); \
			if (data) \
				*data = unitRec->REC_QUERY( SENS_IDX ); \
		\
			return cdtOK; \
		}

	CADET_API_GET_FLUX(Solution, solutionConfig, flux,,)
	CADET_API_GET_FLUX(Sensitivity, sensitivityConfig, sensFlux, CADET_COMMA int sensIdx, sensIdx)
	CADET_API_GET_FLUX(SolutionDerivative, solutionDotConfig, fluxDot,,)
	CADET_API_GET_FLUX(SensitivityDerivative, sensitivityDotConfig, sensFluxDot, CADET_COMMA int sensIdx, sensIdx)



	#define CADET_API_GET_VOLUME(NAME, REC_CONFIG, REC_QUERY, SENS_IDX_SIG, SENS_IDX) \
		cdtResult get##NAME##Volume(cdtDriver* drv, int unitOpId SENS_IDX_SIG, double const** time, double const** data, int* nTime) \
		{ \
			cdtResult retCode = cdtOK; \
			InternalStorageUnitOpRecorder* const unitRec = getUnitRecorder(drv, unitOpId, time, nTime, retCode); \
			if (!unitRec) \
				return retCode; \
		\
			if (!unitRec->REC_CONFIG().storeVolume) \
			{ \
				LOG(Error) << "Volume of unit " << unitOpId << " not recorded"; \
				return cdtDataNotStored; \
			} \
		\
			if (data) \
				*data = unitRec->REC_QUERY( SENS_IDX ); \
		\
			return cdtOK; \
		}

	CADET_API_GET_VOLUME(Solution, solutionConfig, volume,,)
	CADET_API_GET_VOLUME(Sensitivity, sensitivityConfig, sensVolume, CADET_COMMA int sensIdx, sensIdx)
	CADET_API_GET_VOLUME(SolutionDerivative, solutionDotConfig, volumeDot,,)
	CADET_API_GET_VOLUME(SensitivityDerivative, sensitivityDotConfig, sensVolumeDot, CADET_COMMA int sensIdx, sensIdx)


	cdtResult getLastState(cdtDriver* drv, double const** state, int* nStates)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
			return cdtErrorInvalidInputs;

		cadet::ISimulator* const sim = realDrv->simulator();
		unsigned int len = 0;

		if (state)
			*state = sim->getLastSolution(len);

		if (nStates)
			*nStates = len;

		return cdtOK;
	}

	cdtResult getLastStateTimeDerivative(cdtDriver* drv, double const** state, int* nStates)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
			return cdtErrorInvalidInputs;

		cadet::ISimulator* const sim = realDrv->simulator();
		unsigned int len = 0;

		if (state)
			*state = sim->getLastSolutionDerivative(len);

		if (nStates)
			*nStates = len;

		return cdtOK;
	}

	cdtResult getLastUnitState(cdtDriver* drv, int unitOpId, double const** state, int* nStates)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
			return cdtErrorInvalidInputs;

		cadet::ISimulator* const sim = realDrv->simulator();
		unsigned int len = 0;

		unsigned int sliceStart;
		unsigned int sliceEnd;
		std::tie(sliceStart, sliceEnd) = sim->model()->getModelStateOffsets(unitOpId);

		if (state)
			*state = sim->getLastSolution(len) + sliceStart;

		if (nStates)
			*nStates = sliceEnd - sliceStart;

		return cdtOK;
	}

	cdtResult getLastUnitStateTimeDerivative(cdtDriver* drv, int unitOpId, double const** state, int* nStates)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
			return cdtErrorInvalidInputs;

		cadet::ISimulator* const sim = realDrv->simulator();
		unsigned int len = 0;

		unsigned int sliceStart;
		unsigned int sliceEnd;
		std::tie(sliceStart, sliceEnd) = sim->model()->getModelStateOffsets(unitOpId);

		if (state)
			*state = sim->getLastSolutionDerivative(len) + sliceStart;

		if (nStates)
			*nStates = sliceEnd - sliceStart;

		return cdtOK;
	}


	cdtResult getLastSensitivityState(cdtDriver* drv, int sensIdx, double const** state, int* nStates)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
			return cdtErrorInvalidInputs;

		cadet::ISimulator* const sim = realDrv->simulator();
		unsigned int len = 0;

		const std::vector<double const*> lastY = sim->getLastSensitivities(len);
		if ((lastY.size() <= sensIdx) || (sensIdx < 0))
			return cdtErrorInvalidInputs;

		if (state)
			*state = lastY[sensIdx];

		if (nStates)
			*nStates = len;

		return cdtOK;
	}

	cdtResult getLastSensitivityStateTimeDerivative(cdtDriver* drv, int sensIdx, double const** state, int* nStates)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
			return cdtErrorInvalidInputs;

		cadet::ISimulator* const sim = realDrv->simulator();
		unsigned int len = 0;

		const std::vector<double const*> lastY = sim->getLastSensitivityDerivatives(len);
		if ((lastY.size() <= sensIdx) || (sensIdx < 0))
			return cdtErrorInvalidInputs;

		if (state)
			*state = lastY[sensIdx];

		if (nStates)
			*nStates = len;

		return cdtOK;
	}

	cdtResult getLastSensitivityUnitState(cdtDriver* drv, int sensIdx, int unitOpId, double const** state, int* nStates)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
			return cdtErrorInvalidInputs;

		cadet::ISimulator* const sim = realDrv->simulator();
		unsigned int len = 0;

		unsigned int sliceStart;
		unsigned int sliceEnd;
		std::tie(sliceStart, sliceEnd) = sim->model()->getModelStateOffsets(unitOpId);

		const std::vector<double const*> lastY = sim->getLastSensitivities(len);
		if ((lastY.size() <= sensIdx) || (sensIdx < 0))
			return cdtErrorInvalidInputs;

		if (state)
			*state = lastY[sensIdx] + sliceStart;

		if (nStates)
			*nStates = sliceEnd - sliceStart;

		return cdtOK;
	}

	cdtResult getLastSensitivityUnitStateTimeDerivative(cdtDriver* drv, int sensIdx, int unitOpId, double const** state, int* nStates)
	{
		Driver* const realDrv = drv->driver;
		if (!realDrv)
			return cdtErrorInvalidInputs;

		cadet::ISimulator* const sim = realDrv->simulator();
		unsigned int len = 0;

		unsigned int sliceStart;
		unsigned int sliceEnd;
		std::tie(sliceStart, sliceEnd) = sim->model()->getModelStateOffsets(unitOpId);

		const std::vector<double const*> lastY = sim->getLastSensitivityDerivatives(len);
		if ((lastY.size() <= sensIdx) || (sensIdx < 0))
			return cdtErrorInvalidInputs;

		if (state)
			*state = lastY[sensIdx] + sliceStart;

		if (nStates)
			*nStates = sliceEnd - sliceStart;

		return cdtOK;
	}

}  // namespace v1

}  // namespace api

}  // namespace cadet


extern "C"
{

	CADET_API cdtResult cdtGetAPIv010000(cdtAPIv010000* ptr)
	{
		if (!ptr)
			return cdtErrorInvalidInputs;

		ptr->createDriver = &cadet::api::v1::createDriver;
		ptr->deleteDriver = &cadet::api::v1::deleteDriver;
		ptr->runSimulation = &cadet::api::v1::runSimulation;
		ptr->getNumParTypes = &cadet::api::v1::getNumParTypes;
		ptr->getSolutionInlet = &cadet::api::v1::getSolutionInlet;
		ptr->getSolutionOutlet = &cadet::api::v1::getSolutionOutlet;
		ptr->getSolutionBulk = &cadet::api::v1::getSolutionBulk;
		ptr->getSolutionParticle = &cadet::api::v1::getSolutionParticle;
		ptr->getSolutionSolid = &cadet::api::v1::getSolutionSolid;
		ptr->getSolutionFlux = &cadet::api::v1::getSolutionFlux;
		ptr->getSolutionVolume = &cadet::api::v1::getSolutionVolume;
		ptr->getSolutionDerivativeInlet = &cadet::api::v1::getSolutionDerivativeInlet;
		ptr->getSolutionDerivativeOutlet = &cadet::api::v1::getSolutionDerivativeOutlet;
		ptr->getSolutionDerivativeBulk = &cadet::api::v1::getSolutionDerivativeBulk;
		ptr->getSolutionDerivativeParticle = &cadet::api::v1::getSolutionDerivativeParticle;
		ptr->getSolutionDerivativeSolid = &cadet::api::v1::getSolutionDerivativeSolid;
		ptr->getSolutionDerivativeFlux = &cadet::api::v1::getSolutionDerivativeFlux;
		ptr->getSolutionDerivativeVolume = &cadet::api::v1::getSolutionDerivativeVolume;
		ptr->getSensitivityInlet = &cadet::api::v1::getSensitivityInlet;
		ptr->getSensitivityOutlet = &cadet::api::v1::getSensitivityOutlet;
		ptr->getSensitivityBulk = &cadet::api::v1::getSensitivityBulk;
		ptr->getSensitivityParticle = &cadet::api::v1::getSensitivityParticle;
		ptr->getSensitivitySolid = &cadet::api::v1::getSensitivitySolid;
		ptr->getSensitivityFlux = &cadet::api::v1::getSensitivityFlux;
		ptr->getSensitivityVolume = &cadet::api::v1::getSensitivityVolume;
		ptr->getSensitivityDerivativeInlet = &cadet::api::v1::getSensitivityDerivativeInlet;
		ptr->getSensitivityDerivativeOutlet = &cadet::api::v1::getSensitivityDerivativeOutlet;
		ptr->getSensitivityDerivativeBulk = &cadet::api::v1::getSensitivityDerivativeBulk;
		ptr->getSensitivityDerivativeParticle = &cadet::api::v1::getSensitivityDerivativeParticle;
		ptr->getSensitivityDerivativeSolid = &cadet::api::v1::getSensitivityDerivativeSolid;
		ptr->getSensitivityDerivativeFlux = &cadet::api::v1::getSensitivityDerivativeFlux;
		ptr->getSensitivityDerivativeVolume = &cadet::api::v1::getSensitivityDerivativeVolume;
		ptr->getLastState = &cadet::api::v1::getLastState;
		ptr->getLastStateTimeDerivative = &cadet::api::v1::getLastStateTimeDerivative;
		ptr->getLastUnitState = &cadet::api::v1::getLastUnitState;
		ptr->getLastUnitStateTimeDerivative = &cadet::api::v1::getLastUnitStateTimeDerivative;
		ptr->getLastSensitivityState = &cadet::api::v1::getLastSensitivityState;
		ptr->getLastSensitivityStateTimeDerivative = &cadet::api::v1::getLastSensitivityStateTimeDerivative;
		ptr->getLastSensitivityUnitState = &cadet::api::v1::getLastSensitivityUnitState;
		ptr->getLastSensitivityUnitStateTimeDerivative = &cadet::api::v1::getLastSensitivityUnitStateTimeDerivative;

		return cdtOK;
	}

}

