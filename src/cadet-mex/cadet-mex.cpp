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

#ifndef MATLAB_MEX_FILE
#define MATLAB_MEX_FILE
#endif

#include <mex.h>

#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>

// Everything that includes mex.h should go here
#include "MatlabException.hpp"
#include "MatlabHandle.hpp"
#include "MatlabIO.hpp"
#include "MatlabCommands.hpp"

// Take care of namespace pollution / macros
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include "Logging.hpp"

#include "cadet/cadet.hpp"

#include "common/CompilerSpecific.hpp"
#include "common/ParameterProviderImpl.hpp"
#include "common/Driver.hpp"

#ifndef CADET_LOGGING_DISABLE
template <>
cadet::LogLevel cadet::log::RuntimeFilteringLogger<cadet::log::GlobalLogger>::_minLvl = cadet::LogLevel::Trace;

#ifdef __clang__
// Silence -Wundefined-var-template warning
template class cadet::log::RuntimeFilteringLogger<cadet::log::GlobalLogger>;
#endif
#endif

namespace
{
inline void setMexLogLevel(cadet::LogLevel newLL)
{
#ifndef CADET_LOGGING_DISABLE
	cadet::log::RuntimeFilteringLogger<cadet::log::GlobalLogger>::level(newLL);
#endif
}
} // namespace

/**
 * @brief Forwards log messages to Matlab using mexPrintf() or mexWarnMsg()
 */
class LogReceiver : public cadet::ILogReceiver
{
public:
	LogReceiver()
	{
	}

	virtual void message(const char* file, const char* func, const unsigned int line, cadet::LogLevel lvl,
						 const char* lvlStr, const char* message)
	{
		// @todo: Use red colors for error LogLevels
		if (lvl == cadet::LogLevel::Warning)
			mexWarnMsgIdAndTxt("CADET:mexWarning", "In function %s on line %u: %s", func, line, message);
		else
			mexPrintf("[%s: %s::%u] %s", lvlStr, func, line, message);
	}
};

/**
 * @brief Scope class that subscribes and unsubscribes a LogReceiver from the logging mechanism of libcadet
 */
class LogReceiverScope
{
public:
	LogReceiverScope() : _lr()
	{
		cadetSetLogReceiver(&_lr);
	}
	~LogReceiverScope() CADET_NOEXCEPT
	{
		cadetSetLogReceiver(nullptr);
	}

private:
	LogReceiver _lr;
};

namespace cadet
{
namespace mex
{
void registerMatlabExtFun(IModelBuilder& builder);

void prepareDriver(cadet::Driver& drv)
{
	cadet::mex::registerMatlabExtFun(*drv.modelBuilder());
}
} // namespace mex
} // namespace cadet

MEXFUNCTION_LINKAGE void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	LogReceiverScope lrs;

	if (nrhs == 0)
	{
		// Print version
		if (nlhs == 0)
			mexPrintf("This is CADET version %s built from commit %s on branch %s\n", cadet::getLibraryVersion(),
					  cadet::getLibraryCommitHash(), cadet::getLibraryBranchRefspec());

		// Return version
		if (nlhs >= 1)
		{
			plhs[0] = mxCreateString(cadet::getLibraryVersion());
		}
		if (nlhs >= 2)
		{
			plhs[1] = mxCreateString(cadet::getLibraryCommitHash());
		}
		if (nlhs >= 3)
		{
			plhs[2] = mxCreateString(cadet::getLibraryBranchRefspec());
		}
		return;
	}

	try
	{
		if ((nrhs == 1) && (nlhs == 1) && mxIsStruct(prhs[0]))
		{
			// Compatibility old style call: res = CadetMex(task);

			// Run and exit
			cadet::Driver drv;
			cadet::mex::prepareDriver(drv);

			cadet::mex::runFullSimulation(drv, prhs[0], plhs[0]);
			return;
		}

		// New style call: CadetMex('action')
		//                 CadetMex('action', driverHandle)
		//                 CadetMex('action', driverHandle, args, ...)

		if (!mxIsChar(prhs[0]))
			mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Expecting first input argument of type 'char' / string.\n");

		// Extract action / command
		const char* const strData = mxArrayToString(prhs[0]);
		std::string command = strData;
		mxFree(const_cast<char*>(strData));

		cadet::mex::CommandMap commands = cadet::mex::registeredCommands();

#ifdef CADET_DEBUG
		{
			const auto itCreate = commands.find("create");
			cadet_assert(itCreate == commands.end());

			const auto itDestroy = commands.find("destroy");
			cadet_assert(itDestroy == commands.end());

			const auto itLogLevel = commands.find("loglevel");
			cadet_assert(itLogLevel == commands.end());
		}
#endif

		// Dispatch command
		// All commands that do not require a cadet::Driver object have to be handled here
		// Other commands are registered in the map in registerCommands()
		if (command == "create")
		{
			if (nlhs != 1)
				mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'create' requires exactly one output.\n");

			cadet::Driver* const drv = new cadet::Driver();
			cadet::mex::prepareDriver(*drv);

			plhs[0] = cadet::mex::convertPtr2Mat(drv);
		}
		else if (command == "destroy")
		{
			if (nrhs != 2)
				mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'destroy' requires a handle to operate on.\n");
			cadet::mex::destroyObject<cadet::Driver>(prhs[1]);
		}
		else if (command == "loglevel")
		{
			// Check if just the current log level is requested
			if (nrhs == 1)
			{
				if (nlhs != 1)
					mexErrMsgIdAndTxt("CADET:mexError",
									  "CadetMex: Command 'loglevel' expects either an input or an output argument.\n");

				plhs[0] = cadet::mex::io::scalar<double, double>(static_cast<double>(cadetGetLogLevel()));
				return;
			}

			// From now on we want to set the log level
			if (nrhs != 2)
				mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'loglevel' requires a log level as argument.\n");

			if (cadet::mex::io::isEmpty(prhs[1]) ||
				(!cadet::mex::io::isType<double>(prhs[1]) && !cadet::mex::io::isType<std::string>(prhs[1])))
				mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'loglevel' requires a scalar argument of type "
													"'double' or 'string' as log level.\n");

			// Additionally return current LogLevel if requested
			if (nlhs == 1)
				plhs[0] = cadet::mex::io::scalar<double, double>(static_cast<double>(cadetGetLogLevel()));

			// Set new LogLevel in libcadet as well as in this particular frontend
			if (cadet::mex::io::isType<double>(prhs[1]))
			{
				const unsigned int newLL = static_cast<unsigned int>(cadet::mex::io::scalar<double>(prhs[1]));
				cadetSetLogLevel(newLL);
				setMexLogLevel(static_cast<cadet::LogLevel>(newLL));
			}
			else if (cadet::mex::io::isType<int32_t>(prhs[1]))
			{
				const unsigned int newLL = cadet::mex::io::scalar<int32_t>(prhs[1]);
				cadetSetLogLevel(newLL);
				setMexLogLevel(static_cast<cadet::LogLevel>(newLL));
			}
			else
			{
				const char* const strLogLevel = mxArrayToString(prhs[1]);
				std::string ll = strLogLevel;
				mxFree(const_cast<char*>(strLogLevel));

				const cadet::LogLevel newLL = cadet::to_loglevel(ll);
				cadetSetLogLevel(static_cast<typename std::underlying_type<cadet::LogLevel>::type>(newLL));
				setMexLogLevel(newLL);
			}
		}
		else
		{
			// Look for command in map
			auto it = commands.find(command);
			if (it == commands.end())
				mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Unkown command '%s'.\n", command.c_str());

			if (nrhs < 2)
				mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command '%s' requires a handle to operate on.\n",
								  command.c_str());

			cadet::Driver* const drv = cadet::mex::convertMat2Ptr<cadet::Driver>(prhs[1]);
			it->second(*drv, nlhs, plhs, nrhs, prhs);
		}
	}
	catch (const cadet::mex::MatlabException& e)
	{
		mexErrMsgIdAndTxt("CADET:mexError", "Error in Matlab communication: %s\n", e.what());
	}
	catch (const std::exception& e)
	{
		mexErrMsgIdAndTxt("CADET:mexError", "Error in simulation: %s\n", e.what());
	}
}
