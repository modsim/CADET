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

#include "cadet/cadet.hpp"
#include "io/hdf5/HDF5Reader.hpp"
#include "io/hdf5/HDF5Writer.hpp"
#include "io/xml/XMLReader.hpp"
#include "io/xml/XMLWriter.hpp"
#include "common/JsonParameterProvider.hpp"

#include <tclap/CmdLine.h>
#include "common/TclapUtils.hpp"
#include "ProgressBar.hpp"
#include "SignalHandler.hpp"

#include "Logging.hpp"

#include "common/CompilerSpecific.hpp"
#include "common/ParameterProviderImpl.hpp"
#include "common/Driver.hpp"

#ifdef CADET_BENCHMARK_MODE
#include "common/Timer.hpp"
#endif

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cctype>

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
inline void setLocalLogLevel(cadet::LogLevel newLL)
{
#ifndef CADET_LOGGING_DISABLE
	cadet::log::RuntimeFilteringLogger<cadet::log::GlobalLogger>::level(newLL);
#endif
}
} // namespace

class LogReceiver : public cadet::ILogReceiver
{
public:
	LogReceiver()
	{
	}

	virtual void message(const char* file, const char* func, const unsigned int line, cadet::LogLevel lvl,
						 const char* lvlStr, const char* message)
	{
		std::cout << '[' << lvlStr << ": " << func << "::" << line << "] " << message << std::flush;
	}
};

#ifdef CADET_BENCHMARK_MODE
/**
 * @brief Scope class that starts a timer on construction and stops it on destruction
 */
class BenchScope
{
public:
	BenchScope() : _timer()
	{
		_timer.start();
	}
	~BenchScope() CADET_NOEXCEPT
	{
		const double t = _timer.stop();
		std::cout << "Total elapsed time: " << t << " sec" << std::endl;
	}

private:
	cadet::Timer _timer;
};
#endif

// Command line parsing support for cadet::LogLevel type
namespace TCLAP
{
template <> struct ArgTraits<cadet::LogLevel>
{
	typedef StringLike ValueCategory;
};

template <> void SetString<cadet::LogLevel>(cadet::LogLevel& v, const std::string& s)
{
	if (std::isdigit(s[0]))
	{
		const unsigned int lvl = std::stoul(s);
		if (lvl > static_cast<typename std::underlying_type<cadet::LogLevel>::type>(cadet::LogLevel::Trace))
			throw TCLAP::ArgParseException("Couldn't convert '" + s + "' to a valid log level");

		v = static_cast<cadet::LogLevel>(lvl);
	}
	else
	{
		const cadet::LogLevel lvl = cadet::to_loglevel(s);
		if (cadet::to_string(lvl) != s)
			throw TCLAP::ArgParseException("Couldn't convert '" + s + "' to a valid log level");

		v = lvl;
	}
}
} // namespace TCLAP

class ProgressBarNotifier : public cadet::INotificationCallback
{
public:
	ProgressBarNotifier() : _progBar(), _secStrBuffer(13)
	{
	}
	virtual ~ProgressBarNotifier() CADET_NOEXCEPT
	{
	}

	virtual void timeIntegrationStart()
	{
		_progBar.begin();
	}

	virtual void timeIntegrationEnd()
	{
		_progBar.finish("Done");
	}

	virtual void timeIntegrationError(char const* message, unsigned int section, double time, double progress)
	{
	}

	virtual bool timeIntegrationSection(unsigned int section, double time, double const* state, double const* stateDot,
										double progress)
	{
		snprintf(_secStrBuffer.data(), _secStrBuffer.size(), "Init Sec %3u", section);
		_progBar.update(progress, _secStrBuffer.data());
		return !cadet::stopExecutionRequested();
	}

	virtual bool timeIntegrationStep(unsigned int section, double time, double const* state, double const* stateDot,
									 double progress)
	{
		snprintf(_secStrBuffer.data(), _secStrBuffer.size(), "Section %3u", section);
		_progBar.update(progress, _secStrBuffer.data());
		return !cadet::stopExecutionRequested();
	}

protected:
	cadet::ProgressBar _progBar;
	std::vector<char> _secStrBuffer;
};

class SignalHandlingNotifier : public cadet::INotificationCallback
{
public:
	SignalHandlingNotifier()
	{
	}
	virtual ~SignalHandlingNotifier() CADET_NOEXCEPT
	{
	}

	virtual void timeIntegrationStart()
	{
	}
	virtual void timeIntegrationEnd()
	{
	}
	virtual void timeIntegrationError(char const* message, unsigned int section, double time, double progress)
	{
	}

	virtual bool timeIntegrationSection(unsigned int section, double time, double const* state, double const* stateDot,
										double progress)
	{
		return !cadet::stopExecutionRequested();
	}

	virtual bool timeIntegrationStep(unsigned int section, double time, double const* state, double const* stateDot,
									 double progress)
	{
		return !cadet::stopExecutionRequested();
	}
};

template <class Reader_t> class FileReaderDriverConfigurator
{
public:
	FileReaderDriverConfigurator()
	{
	}

	void configure(cadet::Driver& drv, const std::string& inFileName)
	{
		Reader_t rd;
		rd.openFile(inFileName, "r");

		cadet::ParameterProviderImpl<Reader_t> pp(rd);
		drv.configure(pp);

		rd.closeFile();
	}
};

class JsonDriverConfigurator
{
public:
	JsonDriverConfigurator()
	{
	}

	void configure(cadet::Driver& drv, const std::string& inFileName)
	{
		cadet::JsonParameterProvider pp = cadet::JsonParameterProvider::fromFile(inFileName);

		// Skip input scope if it exists
		if (pp.exists("input"))
			pp.pushScope("input");

		drv.configure(pp);
	}
};

template <class DriverConfigurator_t, class Writer_t>
int run(const std::string& inFileName, const std::string& outFileName, bool showProgressBar)
{
	int returnCode = 0;
	cadet::Driver drv;

	{
		DriverConfigurator_t dc;
		dc.configure(drv, inFileName);
	}

	std::unique_ptr<SignalHandlingNotifier> shn = nullptr;

#ifndef CADET_BENCHMARK_MODE
	// Select between progress bar or signal handling only

	std::unique_ptr<ProgressBarNotifier> pb = nullptr;

	if (showProgressBar)
	{
		pb = std::make_unique<ProgressBarNotifier>();
		drv.simulator()->setNotificationCallback(pb.get());
	}
	else
	{
		shn = std::make_unique<SignalHandlingNotifier>();
		drv.simulator()->setNotificationCallback(shn.get());
	}
#else
	// Always handle signals in benchmark mode (no progress bar overhead)
	shn = std::make_unique<SignalHandlingNotifier>();
	drv.simulator()->setNotificationCallback(shn.get());
#endif

	try
	{
		drv.run();
	}
	catch (const cadet::IntegrationException& e)
	{
		std::cerr << "SOLVER ERROR: " << e.what() << std::endl;
		returnCode = 3;
	}

	Writer_t writer;
	if (inFileName == outFileName)
		writer.openFile(outFileName, "rw");
	else
		writer.openFile(outFileName, "co");

	drv.write(writer);
	writer.closeFile();

#ifdef CADET_BENCHMARK_MODE
	// Write timings in JSON format

	// First, timings of the ModelSystem
	std::cout << "{\n\"TotalTimeIntegration\": " << drv.simulator()->totalSimulationDuration() << ",\n";
	std::cout << "\"ModelSystem\":\n\t{\n";
	const std::vector<double> sysTiming = drv.model()->benchmarkTimings();
	char const* const* sysDesc = drv.model()->benchmarkDescriptions();

	for (std::size_t i = 0; i < sysTiming.size() - 1; ++i)
		std::cout << "\t\t\"" << sysDesc[i] << "\": " << sysTiming[i] << ",\n";

	std::cout << "\t\t\"" << sysDesc[sysTiming.size() - 1] << "\": " << sysTiming[sysTiming.size() - 1] << "\n\t}";

	// Then, timings for all unit operations
	for (unsigned int j = 0; j < drv.model()->numModels(); ++j)
	{
		// Skip unit operations that do not provide timings
		cadet::IModel const* const m = drv.model()->getModel(j);
		if (!m->benchmarkDescriptions())
			continue;

		std::cout << ",\n\"" << m->unitOperationName() << static_cast<int>(m->unitOperationId()) << "\":\n\t{\n";
		const std::vector<double> grmTiming = m->benchmarkTimings();
		char const* const* grmDesc = m->benchmarkDescriptions();

		for (std::size_t i = 0; i < grmTiming.size() - 1; ++i)
			std::cout << "\t\t\"" << grmDesc[i] << "\": " << grmTiming[i] << ",\n";

		std::cout << "\t\t\"" << grmDesc[grmTiming.size() - 1] << "\": " << grmTiming[grmTiming.size() - 1] << "\n\t}";
	}
	std::cout << "\n}" << std::endl;
#endif

	return returnCode;
}

int main(int argc, char** argv)
{
#ifdef CADET_BENCHMARK_MODE
	// Benchmark the whole program from start to finish
	BenchScope bsTotalTime;
#endif

	// Install signal handler
	cadet::installSignalHandler();

	// Program options
	std::string inFileName = "";
	std::string outFileName = "";
	cadet::LogLevel logLevel = cadet::LogLevel::Trace;
	bool showProgressBar = false;

	try
	{
		TCLAP::CustomOutput customOut("cadet-cli");
		TCLAP::CmdLine cmd("Simulates a chromatography setup using CADET", ' ', "1.0");
		cmd.setOutput(&customOut);

		cmd >> (new TCLAP::SwitchArg("", "progress", "Show a progress bar"))->storeIn(&showProgressBar);
		cmd >> (new TCLAP::ValueArg<cadet::LogLevel>("L", "loglevel", "Set the log level", false,
													 cadet::LogLevel::Trace, "LogLevel"))
				   ->storeIn(&logLevel);
		cmd >>
			(new TCLAP::UnlabeledValueArg<std::string>("input", "Input file", true, "", "File"))->storeIn(&inFileName);
		cmd >> (new TCLAP::UnlabeledValueArg<std::string>("output", "Output file (defaults to input file)", false, "",
														  "File"))
				   ->storeIn(&outFileName);

		cmd.parse(argc, argv);
	}
	catch (const TCLAP::ArgException& e)
	{
		std::cerr << "ERROR: " << e.error() << " for argument " << e.argId() << std::endl;
		return 1;
	}

	// If no dedicated output filename was given, assume output = input file
	if (outFileName.empty())
		outFileName = inFileName;

	std::cout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1);

	// Set LogLevel in library and locally
	LogReceiver lr;
	cadet::setLogReceiver(&lr);
	cadet::setLogLevel(logLevel);
	setLocalLogLevel(logLevel);

	// Obtain file extensions for selecting corresponding reader and writer
	const std::size_t dotPosIn = inFileName.find_last_of('.');
	if (dotPosIn == std::string::npos)
	{
		std::cerr << "Could not deduce input filetype due to missing extension: " << inFileName << std::endl;
		return 2;
	}

	const std::size_t dotPosOut = outFileName.find_last_of('.');
	if (dotPosOut == std::string::npos)
	{
		std::cerr << "Could not deduce output filetype due to missing extension: " << outFileName << std::endl;
		return 2;
	}

	const std::string fileExtIn = inFileName.substr(dotPosIn + 1);
	const std::string fileExtOut = outFileName.substr(dotPosOut + 1);
	int returnCode = 0;

	try
	{
		if (cadet::util::caseInsensitiveEquals(fileExtIn, "h5"))
		{
			if (cadet::util::caseInsensitiveEquals(fileExtOut, "h5"))
			{
				returnCode = run<FileReaderDriverConfigurator<cadet::io::HDF5Reader>, cadet::io::HDF5Writer>(
					inFileName, outFileName, showProgressBar);
			}
			else if (cadet::util::caseInsensitiveEquals(fileExtOut, "xml"))
			{
				returnCode = run<FileReaderDriverConfigurator<cadet::io::HDF5Reader>, cadet::io::XMLWriter>(
					inFileName, outFileName, showProgressBar);
			}
			else
			{
				std::cerr << "Output file format ('." << fileExtOut << "') not supported" << std::endl;
				return 2;
			}
		}
		else if (cadet::util::caseInsensitiveEquals(fileExtIn, "xml"))
		{
			if (cadet::util::caseInsensitiveEquals(fileExtOut, "xml"))
			{
				returnCode = run<FileReaderDriverConfigurator<cadet::io::XMLReader>, cadet::io::XMLWriter>(
					inFileName, outFileName, showProgressBar);
			}
			else if (cadet::util::caseInsensitiveEquals(fileExtOut, "h5"))
			{
				returnCode = run<FileReaderDriverConfigurator<cadet::io::XMLReader>, cadet::io::HDF5Writer>(
					inFileName, outFileName, showProgressBar);
			}
			else
			{
				std::cerr << "Output file format ('." << fileExtOut << "') not supported" << std::endl;
				return 2;
			}
		}
		else if (cadet::util::caseInsensitiveEquals(fileExtIn, "json"))
		{
			if (cadet::util::caseInsensitiveEquals(fileExtOut, "xml"))
			{
				returnCode =
					run<JsonDriverConfigurator, cadet::io::XMLWriter>(inFileName, outFileName, showProgressBar);
			}
			else if (cadet::util::caseInsensitiveEquals(fileExtOut, "h5"))
			{
				returnCode =
					run<JsonDriverConfigurator, cadet::io::HDF5Writer>(inFileName, outFileName, showProgressBar);
			}
			else
			{
				std::cerr << "Output file format ('." << fileExtOut << "') not supported" << std::endl;
				return 2;
			}
		}
		else
		{
			std::cerr << "Input file format ('." << fileExtIn << "') not supported" << std::endl;
			return 2;
		}
	}
	catch (const cadet::io::IOException& e)
	{
		std::cerr << "IO ERROR: " << e.what() << std::endl;
		return 2;
	}
	catch (const cadet::IntegrationException& e)
	{
		std::cerr << "SOLVER ERROR: " << e.what() << std::endl;
		return 3;
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << std::endl;
		return 1;
	}

	return returnCode;
}
