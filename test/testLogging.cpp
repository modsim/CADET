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

#include "cadet/Logging.hpp"
#include "common/Logger.hpp"

typedef cadet::log::NonFilteringLogger<cadet::log::StandardFormattingPolicy, cadet::log::StdOutWritePolicy>
	GlobalLogger;
typedef cadet::log::Logger<GlobalLogger, cadet::LogLevel::Trace> NonFilteringLogger;
typedef cadet::log::Logger<GlobalLogger, cadet::LogLevel::Warning> FilterLogger;

typedef cadet::log::Logger<cadet::log::RuntimeFilteringLogger<GlobalLogger>, cadet::LogLevel::Normal> RtFilterLogger;

template <> cadet::LogLevel cadet::log::RuntimeFilteringLogger<GlobalLogger>::_minLvl = cadet::LogLevel::Trace;

#define LOG_GLOBAL(lvl) LOG_BASE(NonFilteringLogger, lvl)
#define LOG_FILTER(lvl) LOG_BASE(FilterLogger, lvl)
#define LOG_RT_FILTER(lvl) LOG_BASE(RtFilterLogger, lvl)

int main(int argc, char** argv)
{

	LOG_GLOBAL(None) << "This is a test " << 1;
	LOG_GLOBAL(Fatal) << "This is a test " << 1;
	LOG_GLOBAL(Error) << "This is a test " << 1;
	LOG_GLOBAL(Warning) << "This is a test " << 1;
	LOG_GLOBAL(Normal) << "This is a test " << 1;
	LOG_GLOBAL(Info) << "This is a test " << 1;
	LOG_GLOBAL(Debug) << "This is a test " << 1;
	LOG_GLOBAL(Trace) << "This is a test " << 1;

	std::cout << " =================================== " << std::endl;
	LOG_FILTER(None) << "This is a test " << 1;
	LOG_FILTER(Fatal) << "This is a test " << 1;
	LOG_FILTER(Error) << "This is a test " << 1;
	LOG_FILTER(Warning) << "This is a test " << 1;
	LOG_FILTER(Normal) << "This is a test " << 1;
	LOG_FILTER(Info) << "This is a test " << 1;
	LOG_FILTER(Debug) << "This is a test " << 1;
	LOG_FILTER(Trace) << "This is a test " << 1;

	std::cout << " =================================== " << std::endl;
	cadet::log::RuntimeFilteringLogger<GlobalLogger>::level(cadet::LogLevel::Warning);
	LOG_RT_FILTER(None) << "This is a test " << 1;
	LOG_RT_FILTER(Fatal) << "This is a test " << 1;
	LOG_RT_FILTER(Error) << "This is a test " << 1;
	LOG_RT_FILTER(Warning) << "This is a test " << 1;
	LOG_RT_FILTER(Normal) << "This is a test " << 1;
	LOG_RT_FILTER(Info) << "This is a test " << 1;
	LOG_RT_FILTER(Debug) << "This is a test " << 1;
	LOG_RT_FILTER(Trace) << "This is a test " << 1;

	std::cout << " =================================== " << std::endl;
	cadet::log::RuntimeFilteringLogger<GlobalLogger>::level(cadet::LogLevel::Error);
	LOG_RT_FILTER(None) << "This is a test " << 1;
	LOG_RT_FILTER(Fatal) << "This is a test " << 1;
	LOG_RT_FILTER(Error) << "This is a test " << 1;
	LOG_RT_FILTER(Warning) << "This is a test " << 1;
	LOG_RT_FILTER(Normal) << "This is a test " << 1;
	LOG_RT_FILTER(Info) << "This is a test " << 1;
	LOG_RT_FILTER(Debug) << "This is a test " << 1;
	LOG_RT_FILTER(Trace) << "This is a test " << 1;

	return 0;
}
