#include <stdexcept>
#include <stdint.h>
#include <windows.h>
#include <iostream>
#include <memory>
#include <functional>
#include <vector>
#include <stack>
#include <string>

#include <json.hpp>
using json = nlohmann::json;

#include "cadet/cadet.h"

const char* getLibBinaryPath();

typedef cdtResult (*cdtGetAPIv010000_t)(cdtAPIv010000* ptr);

typedef void (*cdtSetLogReceiver_t)(cdtLogHandler recv);
typedef void (*cdtSetLogLevel_t)(int lvl);

void logHandler(const char* file, const char* func, const unsigned int line, int lvl, const char* lvlStr,
				const char* message)
{
	std::cout << lvlStr << " [" << func << ":" << line << "] " << message << std::endl;
}

class LibLoader
{
public:
	LibLoader(char const* path) : _hdLib(0)
	{
		_hdLib = LoadLibrary(path);
		if (_hdLib)
			std::cout << "Loaded library " << path << std::endl;
	}

	~LibLoader()
	{
		if (_hdLib)
		{
			FreeLibrary(_hdLib);
			std::cout << "Deleted library" << std::endl;
		}
	}

	bool isValid()
	{
		return _hdLib;
	}

	template <typename T> T load(char const* func)
	{
		return reinterpret_cast<T>(GetProcAddress(_hdLib, func));
	}

private:
	HINSTANCE _hdLib;
};

json createColumnWithSMAJson(const std::string& uoType)
{
	json config;
	config["UNIT_TYPE"] = uoType;
	config["NCOMP"] = 4;
	config["VELOCITY"] = 5.75e-4;
	config["COL_DISPERSION"] = 5.75e-8;
	config["COL_DISPERSION_RADIAL"] = 1e-6;
	config["FILM_DIFFUSION"] = {6.9e-6, 6.9e-6, 6.9e-6, 6.9e-6};
	config["PAR_DIFFUSION"] = {7e-10, 6.07e-11, 6.07e-11, 6.07e-11};
	config["PAR_SURFDIFFUSION"] = {0.0, 0.0, 0.0, 0.0};

	// Geometry
	config["COL_LENGTH"] = 0.014;
	config["COL_RADIUS"] = 0.01;
	config["PAR_RADIUS"] = 4.5e-5;
	config["COL_POROSITY"] = 0.37;
	config["PAR_POROSITY"] = 0.75;
	config["TOTAL_POROSITY"] = 0.37 + (1.0 - 0.37) * 0.75;

	// Initial conditions
	config["INIT_C"] = {50.0, 0.0, 0.0, 0.0};
	config["INIT_Q"] = {1.2e3, 0.0, 0.0, 0.0};

	// Adsorption
	config["ADSORPTION_MODEL"] = std::string("STERIC_MASS_ACTION");
	{
		json ads;
		ads["IS_KINETIC"] = 1;
		ads["SMA_LAMBDA"] = 1.2e3;
		ads["SMA_KA"] = {0.0, 35.5, 1.59, 7.7};
		ads["SMA_KD"] = {0.0, 1000.0, 1000.0, 1000.0};
		ads["SMA_NU"] = {0.0, 4.7, 5.29, 3.7};
		ads["SMA_SIGMA"] = {0.0, 11.83, 10.6, 10.0};
		config["adsorption"] = ads;
	}

	// Discretization
	{
		json disc;

		disc["NCOL"] = 16;
		disc["NPAR"] = 4;
		disc["NBOUND"] = {1, 1, 1, 1};

		if (uoType == "GENERAL_RATE_MODEL_2D")
		{
			disc["NCOL"] = 8;
			disc["NRAD"] = 3;
			disc["NPAR"] = 3;
			disc["RADIAL_DISC_TYPE"] = "EQUIDISTANT";
		}

		disc["PAR_DISC_TYPE"] = std::string("EQUIDISTANT_PAR");

		disc["USE_ANALYTIC_JACOBIAN"] = true;
		disc["MAX_KRYLOV"] = 0;
		disc["GS_TYPE"] = 1;
		disc["MAX_RESTARTS"] = 10;
		disc["SCHUR_SAFETY"] = 1e-8;

		// WENO
		{
			json weno;

			weno["WENO_ORDER"] = 3;
			weno["BOUNDARY_MODEL"] = 0;
			weno["WENO_EPS"] = 1e-10;
			disc["weno"] = weno;
		}
		config["discretization"] = disc;
	}

	return config;
}

json createLWEJson(const std::string& uoType)
{
	json config;
	// Model
	{
		json model;
		model["NUNITS"] = 2;
		model["unit_000"] = createColumnWithSMAJson(uoType);

		// Inlet - unit 001
		{
			json inlet;

			inlet["UNIT_TYPE"] = std::string("INLET");
			inlet["INLET_TYPE"] = std::string("PIECEWISE_CUBIC_POLY");
			inlet["NCOMP"] = 4;

			{
				json sec;

				sec["CONST_COEFF"] = {50.0, 1.0, 1.0, 1.0};
				sec["LIN_COEFF"] = {0.0, 0.0, 0.0, 0.0};
				sec["QUAD_COEFF"] = {0.0, 0.0, 0.0, 0.0};
				sec["CUBE_COEFF"] = {0.0, 0.0, 0.0, 0.0};

				inlet["sec_000"] = sec;
			}

			{
				json sec;

				sec["CONST_COEFF"] = {50.0, 0.0, 0.0, 0.0};
				sec["LIN_COEFF"] = {0.0, 0.0, 0.0, 0.0};
				sec["QUAD_COEFF"] = {0.0, 0.0, 0.0, 0.0};
				sec["CUBE_COEFF"] = {0.0, 0.0, 0.0, 0.0};

				inlet["sec_001"] = sec;
			}

			{
				json sec;

				sec["CONST_COEFF"] = {100.0, 0.0, 0.0, 0.0};
				sec["LIN_COEFF"] = {0.2, 0.0, 0.0, 0.0};
				sec["QUAD_COEFF"] = {0.0, 0.0, 0.0, 0.0};
				sec["CUBE_COEFF"] = {0.0, 0.0, 0.0, 0.0};

				inlet["sec_002"] = sec;
			}

			model["unit_001"] = inlet;
		}

		// Valve switches
		{
			json con;
			con["NSWITCHES"] = 1;
			con["CONNECTIONS_INCLUDE_PORTS"] = true;

			{
				json sw;

				// This switch occurs at beginning of section 0 (initial configuration)
				sw["SECTION"] = 0;

				if (uoType == "GENERAL_RATE_MODEL_2D")
				{
					// Connection list is 3x7 since we have 1 connection between
					// the two unit operations with 3 ports (and we need to have 7 columns)
					sw["CONNECTIONS"] = {1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 7.42637597e-09,
										 1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 2.22791279e-08,
										 1.0, 0.0, 0.0, 2.0, -1.0, -1.0, 3.71318798e-08};
					// Connections: From unit operation 1 port 0
					//              to unit operation 0 port 0,
					//              connect component -1 (i.e., all components)
					//              to component -1 (i.e., all components) with
					//              volumetric flow rate 7.42637597e-09 m^3/s
				}
				else
				{
					// Connection list is 1x7 since we have 1 connection between
					// the two unit operations (and we need to have 7 columns)
					sw["CONNECTIONS"] = {1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 1.0};
					// Connections: From unit operation 1 port -1 (i.e., all ports)
					//              to unit operation 0 port -1 (i.e., all ports),
					//              connect component -1 (i.e., all components)
					//              to component -1 (i.e., all components) with
					//              volumetric flow rate 1.0 m^3/s
				}

				con["switch_000"] = sw;
			}
			model["connections"] = con;
		}

		// Solver settings
		{
			json solver;

			solver["MAX_KRYLOV"] = 0;
			solver["GS_TYPE"] = 1;
			solver["MAX_RESTARTS"] = 10;
			solver["SCHUR_SAFETY"] = 1e-8;
			model["solver"] = solver;
		}

		config["model"] = model;
	}

	// Return
	{
		json ret;
		ret["WRITE_SOLUTION_TIMES"] = true;

		json grm;
		grm["WRITE_SOLUTION_BULK"] = false;
		grm["WRITE_SOLUTION_PARTICLE"] = false;
		grm["WRITE_SOLUTION_FLUX"] = false;
		grm["WRITE_SOLUTION_INLET"] = true;
		grm["WRITE_SOLUTION_OUTLET"] = true;

		ret["unit_000"] = grm;
		config["return"] = ret;
	}

	// Solver
	{
		json solver;

		{
			std::vector<double> solTimes;

			if (uoType == "LUMPED_RATE_MODEL_WITHOUT_PORES")
			{
				// Lumped rate model without pores has less rate limiting
				// Thus, a shorter simulation time suffices
				solTimes.reserve(1101);
				for (double t = 0.0; t <= 1100.0; t += 1.0)
					solTimes.push_back(t);
			}
			else
			{
				solTimes.reserve(1501);
				for (double t = 0.0; t <= 1500.0; t += 1.0)
					solTimes.push_back(t);
			}

			solver["USER_SOLUTION_TIMES"] = solTimes;
		}

		solver["NTHREADS"] = 1;

		// Sections
		{
			json sec;

			sec["NSEC"] = 3;
			if (uoType == "LUMPED_RATE_MODEL_WITHOUT_PORES")
				sec["SECTION_TIMES"] = {0.0, 10.0, 90.0, 1100.0};
			else
				sec["SECTION_TIMES"] = {0.0, 10.0, 90.0, 1500.0};
			sec["SECTION_CONTINUITY"] = {false, false};

			solver["sections"] = sec;
		}

		// Time integrator
		{
			json ti;

			ti["ABSTOL"] = 1e-8;
			ti["RELTOL"] = 1e-6;
			ti["ALGTOL"] = 1e-12;
			ti["INIT_STEP_SIZE"] = 1e-6;
			ti["MAX_STEPS"] = 10000;
			ti["MAX_STEP_SIZE"] = 0.0;
			ti["RELTOL_SENS"] = 1e-6;
			ti["ERRORTEST_SENS"] = true;
			ti["MAX_NEWTON_ITER"] = 3;
			ti["MAX_ERRTEST_FAIL"] = 7;
			ti["MAX_CONVTEST_FAIL"] = 10;
			ti["MAX_NEWTON_ITER_SENS"] = 3;
			ti["CONSISTENT_INIT_MODE"] = 1;
			ti["CONSISTENT_INIT_MODE_SENS"] = 1;

			solver["time_integrator"] = ti;
		}

		config["solver"] = solver;
	}
	return config;
}

class JsonNavigator
{
public:
	JsonNavigator(const json& root) : _root(root), _scopePath("/")
	{
		_opened.push(&_root);
	}
	~JsonNavigator()
	{
	}

	void pushScope(const std::string& scope)
	{
		std::cout << "[PP] SCOPE " << scope << "\n";
		_opened.push(&_opened.top()->at(scope));
		_scopePath += "/" + scope;
	}

	void popScope()
	{
		std::cout << "[PP] SCOPE POP\n";
		_opened.pop();

		std::size_t lastIdx = std::string::npos;
		if (_scopePath.back() == '/')
			lastIdx = _scopePath.length() - 2;

		const std::size_t idx = _scopePath.find_last_of('/', lastIdx);
		_scopePath.erase(idx);
	}

	const json& current() const
	{
		return *_opened.top();
	}

private:
	const json& _root;
	std::stack<json const*> _opened;
	std::string _scopePath;
};

cdtResult getDouble(void* userData, const char* paramName, double* val)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		const json& p = jn.current().at(paramName);
		const json& pp = (p.is_array() && (p.size() == 1)) ? p[0] : p;

		std::cout << "[PP] GET scalar [double] " << paramName << " = " << pp.get<double>() << "\n";
		*val = pp.get<double>();
		return cdtOK;
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] GET scalar [double] ERROR: " << e.what() << std::endl;
		return cdtError;
	}
}

cdtResult getInt(void* userData, const char* paramName, int* val)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		const json& p = jn.current().at(paramName);
		const json& pp = (p.is_array() && (p.size() == 1)) ? p[0] : p;

		if (pp.is_boolean())
		{
			std::cout << "[PP] GET scalar [int] " << paramName << " = " << static_cast<int>(pp.get<bool>()) << "\n";
			*val = pp.get<bool>();
		}
		else
		{
			std::cout << "[PP] GET scalar [int] " << paramName << " = " << pp.get<int>() << "\n";
			*val = pp.get<int>();
		}
		return cdtOK;
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] GET scalar [int] ERROR: " << e.what() << std::endl;
		return cdtError;
	}
}

cdtResult getBool(void* userData, const char* paramName, uint8_t* val)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		const json& p = jn.current().at(paramName);
		const json& pp = (p.is_array() && (p.size() == 1)) ? p[0] : p;

		if (pp.is_number_integer())
		{
			std::cout << "[PP] GET scalar [bool] " << paramName << " = " << static_cast<bool>(pp.get<int>()) << "\n";
			*val = static_cast<bool>(pp.get<int>());
		}
		else
		{
			std::cout << "[PP] GET scalar [bool] " << paramName << " = " << pp.get<bool>() << "\n";
			*val = pp.get<bool>();
		}
		return cdtOK;
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] GET scalar [bool] ERROR: " << e.what() << std::endl;
		return cdtError;
	}
}

cdtResult getString(void* userData, const char* paramName, char const** val)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		const json& p = jn.current().at(paramName);
		const json& pp = (p.is_array() && (p.size() == 1)) ? p[0] : p;

		std::cout << "[PP] GET scalar [string] " << paramName << " = " << pp.get_ref<const std::string&>() << "\n";
		*val = pp.get_ptr<const std::string*>()->c_str();
		return cdtOK;
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] GET scalar [string] ERROR: " << e.what() << std::endl;
		return cdtError;
	}
}

cdtResult getDoubleArrayItem(void* userData, const char* paramName, int idx, double* val)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		const json& p = jn.current().at(paramName);

		if ((idx == 0) && !p.is_array())
		{
			std::cout << "[PP] GET array (" << idx << ") [double] " << paramName << " = " << p.get<double>() << "\n";
			*val = p.get<double>();
			return cdtOK;
		}
		if ((idx > 0) && !p.is_array())
		{
			std::cout << "[PP] GET array (" << idx << ") [double] ERROR: Item is scalar instead of array" << std::endl;
			return cdtError;
		}
		if ((idx > 0) && (idx >= p.size()))
		{
			std::cout << "[PP] GET array (" << idx << ") [double] ERROR: Index out of bounds (size is " << p.size()
					  << ")" << std::endl;
			return cdtError;
		}

		const json& pp = p[idx];

		std::cout << "[PP] GET array (" << idx << ") [double] " << paramName << " = " << pp.get<double>() << "\n";
		*val = pp.get<double>();
		return cdtOK;
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] GET array (" << idx << ") [double] ERROR: " << e.what() << std::endl;
		return cdtError;
	}
}

cdtResult getIntArrayItem(void* userData, const char* paramName, int idx, int* val)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		const json& p = jn.current().at(paramName);

		if ((idx == 0) && !p.is_array())
		{
			std::cout << "[PP] GET array (" << idx << ") [int] " << paramName << " = " << p.get<int>() << "\n";
			*val = p.get<int>();
			return cdtOK;
		}
		if ((idx > 0) && !p.is_array())
		{
			std::cout << "[PP] GET array (" << idx << ") [int] ERROR: Item is scalar instead of array" << std::endl;
			return cdtError;
		}
		if ((idx > 0) && (idx >= p.size()))
		{
			std::cout << "[PP] GET array (" << idx << ") [int] ERROR: Index out of bounds (size is " << p.size() << ")"
					  << std::endl;
			return cdtError;
		}

		const json& pp = p[idx];

		if (pp.is_boolean())
		{
			std::cout << "[PP] GET array (" << idx << ") [int] " << paramName << " = "
					  << static_cast<int>(pp.get<bool>()) << "\n";
			*val = pp.get<bool>();
		}
		else
		{
			std::cout << "[PP] GET array (" << idx << ") [int] " << paramName << " = " << pp.get<int>() << "\n";
			*val = pp.get<int>();
		}
		return cdtOK;
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] GET array (" << idx << ") [int] ERROR: " << e.what() << std::endl;
		return cdtError;
	}
}

cdtResult getBoolArrayItem(void* userData, const char* paramName, int idx, uint8_t* val)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		const json& p = jn.current().at(paramName);

		if ((idx == 0) && !p.is_array())
		{
			std::cout << "[PP] GET array (" << idx << ") [bool] " << paramName << " = " << p.get<bool>() << "\n";
			*val = p.get<bool>();
			return cdtOK;
		}
		if ((idx > 0) && !p.is_array())
		{
			std::cout << "[PP] GET array (" << idx << ") [bool] ERROR: Item is scalar instead of array" << std::endl;
			return cdtError;
		}
		if ((idx > 0) && (idx >= p.size()))
		{
			std::cout << "[PP] GET array (" << idx << ") [bool] ERROR: Index out of bounds (size is " << p.size() << ")"
					  << std::endl;
			return cdtError;
		}

		const json& pp = p[idx];

		if (pp.is_number_integer())
		{
			std::cout << "[PP] GET array (" << idx << ") [bool] " << paramName << " = "
					  << static_cast<bool>(pp.get<int>()) << "\n";
			*val = static_cast<bool>(pp.get<int>());
		}
		else
		{
			std::cout << "[PP] GET array (" << idx << ") [bool] " << paramName << " = " << pp.get<bool>() << "\n";
			*val = pp.get<bool>();
		}
		return cdtOK;
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] GET array (" << idx << ") [bool] ERROR: " << e.what() << std::endl;
		return cdtError;
	}
}

cdtResult getStringArrayItem(void* userData, const char* paramName, int idx, char const** val)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		const json& p = jn.current().at(paramName);

		if ((idx == 0) && !p.is_array())
		{
			std::cout << "[PP] GET array (" << idx << ") [string] " << paramName << " = "
					  << p.get_ref<const std::string&>() << "\n";
			*val = p.get_ptr<const std::string*>()->c_str();
			return cdtOK;
		}
		if ((idx > 0) && !p.is_array())
		{
			std::cout << "[PP] GET array (" << idx << ") [string] ERROR: Item is scalar instead of array" << std::endl;
			return cdtError;
		}
		if ((idx > 0) && (idx >= p.size()))
		{
			std::cout << "[PP] GET array (" << idx << ") [string] ERROR: Index out of bounds (size is " << p.size()
					  << ")" << std::endl;
			return cdtError;
		}

		const json& pp = p[idx];

		std::cout << "[PP] GET array (" << idx << ") [string] " << paramName << " = "
				  << pp.get_ref<const std::string&>() << "\n";
		*val = pp.get_ptr<const std::string*>()->c_str();
		return cdtOK;
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] GET array (" << idx << ") [string] ERROR: " << e.what() << std::endl;
		return cdtError;
	}
}

int exists(void* userData, const char* elemName)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	const json& p = jn.current();
	const bool found = p.find(elemName) != p.end();

	std::cout << "[PP] EXISTS " << elemName << " = " << (found ? "yes" : "no") << "\n";
	return found;
}

cdtResult isArray(void* userData, const char* paramName, uint8_t* res)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		const json& p = jn.current().at(paramName);

		if (p.is_array())
			*res = 1;
		else
			*res = 0;

		std::cout << "[PP] ISARRAY " << paramName << " = " << p.is_array() << "\n";
		return cdtOK;
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] ISARRAY ERROR: " << e.what() << std::endl;
		return cdtError;
	}
}

int numElements(void* userData, const char* paramName)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		const json& p = jn.current().at(paramName);
		std::cout << "[PP] NUMELEMENTS " << paramName << " = " << p.size() << "\n";
		return p.size();
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] NUMELEMENTS ERROR: " << e.what() << std::endl;
		return -1;
	}
}

cdtResult pushScope(void* userData, const char* scope)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	try
	{
		jn.pushScope(scope);
		return cdtOK;
	}
	catch (const std::exception& e)
	{
		std::cout << "[PP] PUSHSCOPE ERROR: " << e.what() << std::endl;
		return cdtError;
	}
}

cdtResult popScope(void* userData)
{
	JsonNavigator& jn = *static_cast<JsonNavigator*>(userData);
	jn.popScope();
	return cdtOK;
}

int main(int argc, char** argv)
{
	LibLoader loader(getLibBinaryPath());

	if (!loader.isValid())
	{
		std::cout << "Failed to load DLL" << std::endl;
		return 1;
	}

	const cdtSetLogReceiver_t setLogReceiver = loader.load<cdtSetLogReceiver_t>("cdtSetLogReceiver");
	if (!setLogReceiver)
	{
		std::cout << "Failed to load cdtSetLogReceiver() function" << std::endl;
		return 1;
	}
	setLogReceiver(&logHandler);

	const cdtSetLogLevel_t setLogLevel = loader.load<cdtSetLogLevel_t>("cdtSetLogLevel");
	if (!setLogLevel)
	{
		std::cout << "Failed to load cdtSetLogLevel() function" << std::endl;
		return 1;
	}
	setLogLevel(cdtLogLevelTrace);

	const cdtGetAPIv010000_t apiLoader = loader.load<cdtGetAPIv010000_t>("cdtGetAPIv010000");
	if (!apiLoader)
	{
		std::cout << "Failed to load cdtGetAPIv010000() function" << std::endl;
		return 1;
	}

	cdtAPIv010000 api;
	const cdtResult resApiLoad = apiLoader(&api);

	std::cout << "cdtGetAPIv010000() = " << resApiLoad << " api.createDriver = " << api.createDriver << std::endl;
	if (CADET_ERR(resApiLoad))
	{
		std::cout << "Error obtaining API loader" << std::endl;
		return 1;
	}

	std::unique_ptr<cdtDriver, std::function<void(cdtDriver*)>> drv(api.createDriver(), [&api](cdtDriver* ptr) {
		api.deleteDriver(ptr);
		std::cout << "Delete driver" << std::endl;
	});

	const json simSpec = createLWEJson("GENERAL_RATE_MODEL");
	JsonNavigator jn(simSpec);

	cdtParameterProvider pp{&jn,
							&getDouble,
							&getInt,
							&getBool,
							&getString,
							nullptr,
							nullptr,
							nullptr,
							nullptr,
							&getDoubleArrayItem,
							&getIntArrayItem,
							&getBoolArrayItem,
							&getStringArrayItem,
							&exists,
							&isArray,
							&numElements,
							&pushScope,
							&popScope};

	const cdtResult resSim = api.runSimulation(drv.get(), &pp);
	std::cout << "runSimulation() = " << resSim << std::endl;

	if (CADET_ERR(resSim))
	{
		std::cout << "Simulation failed" << std::endl;
		return 1;
	}

	double const* time = nullptr;
	double const* outlet = nullptr;
	int nTime = 0;
	int nPort = 0;
	int nComp = 0;
	const cdtResult resSol = api.getSolutionOutlet(drv.get(), 0, &time, &outlet, &nTime, &nPort, &nComp);

	std::cout << "getSolutionOutlet() = " << resSol << " nTime = " << nTime << " nPort = " << nPort
			  << " nComp = " << nComp << std::endl;

	return 0;
}
