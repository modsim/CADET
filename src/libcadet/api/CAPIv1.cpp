// =============================================================================
//  CADET - The Chromatography Analysis and Design Toolkit
//  
//  Copyright © 2008-2020: The CADET Authors
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
#include "common/Driver.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

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
				LOG(Debug) << "GET array (" << i << ") [bool] " << paramName << ": " << v[i];
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

//			cadet::mex::MatlabReaderWriter writer(&output);
//			drv.write(writer);
		}
		catch(const std::exception& e)
		{
			LOG(Error) << "Simulation failed: " << e.what();
			return cdtError;
		}

		return cdtOK;
	}

	cdtResult getSolutionOutlet(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime, int* nPort, int* nComp)
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

		if (nTime)
			*nTime = sysRec->numDataPoints();
		if (time)
			*time = sysRec->time();

		InternalStorageUnitOpRecorder* const unitRec = sysRec->unitOperation(unitOpId);
		if (!unitRec)
		{
			LOG(Error) << "Solution recorder for unit ID " << unitOpId << " not found";
			return cdtErrorInvalidInputs;
		}

		if (!unitRec->solutionConfig().storeOutlet)
		{
			LOG(Error) << "Outlet of unit " << unitOpId << " not recorded";
			return cdtError;
		}

		if (nPort)
			*nPort = unitRec->numOutletPorts();
		if (nComp)
			*nComp = unitRec->numComponents();
		if (data)
			*data = unitRec->outlet();

		return cdtOK;
	}

	cdtResult getSolutionInlet(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime, int* nPort, int* nComp)
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

		if (nTime)
			*nTime = sysRec->numDataPoints();
		if (time)
			*time = sysRec->time();

		InternalStorageUnitOpRecorder* const unitRec = sysRec->unitOperation(unitOpId);
		if (!unitRec)
		{
			LOG(Error) << "Solution recorder for unit ID " << unitOpId << " not found";
			return cdtErrorInvalidInputs;
		}

		if (!unitRec->solutionConfig().storeInlet)
		{
			LOG(Error) << "Inlet of unit " << unitOpId << " not recorded";
			return cdtError;
		}

		if (nPort)
			*nPort = unitRec->numInletPorts();
		if (nComp)
			*nComp = unitRec->numComponents();
		if (data)
			*data = unitRec->inlet();

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
		ptr->getSolutionOutlet = &cadet::api::v1::getSolutionOutlet;
		ptr->getSolutionInlet = &cadet::api::v1::getSolutionInlet;
		return cdtOK;
	}

}
