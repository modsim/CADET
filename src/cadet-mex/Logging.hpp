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
 * Logging configuration for cadet-cli.
 */

#ifndef CADETMEX_LOGGING_IMPL_HPP_
#define CADETMEX_LOGGING_IMPL_HPP_

#include "cadet/Logging.hpp"
#include "common/LoggerBase.hpp"

#include <mex.h>

namespace cadet
{
namespace log
{

/**
 * @brief Implements a standard formatting policy
 */
class CadetMexFormattingPolicy : public FormattingPolicyBase<CadetMexFormattingPolicy>
{
public:
	template <class writePolicy_t, class receiver_t, class paramList_t>
	static inline void format(receiver_t& recv, const char* fileName, const char* funcName, unsigned int line,
							  LogLevel lvl, const paramList_t& p)
	{
		writeParams<writePolicy_t>(recv, lvl, p);
	}
};

/**
 * @brief Sends all messages to Matlab
 */
class MatlabWritePolicy : public NonBufferedWritePolicyBase<MatlabWritePolicy>
{
public:
	static inline void writeLine(const char* fileName, const char* funcName, unsigned int line, LogLevel lvl,
								 const std::string& msg)
	{
		if (lvl == LogLevel::Warning)
			mexWarnMsgIdAndTxt("CADET:mexWarning", "In function %s on line %u: %s", funcName, line, msg.c_str());
		else
			mexPrintf("[%s: %s::%u] %s", to_string(lvl), funcName, line, msg.c_str());
	}
};

typedef NonFilteringLogger<CadetMexFormattingPolicy, MatlabWritePolicy> GlobalLogger;

#ifndef CADET_LOGGING_DISABLE
typedef Logger<RuntimeFilteringLogger<GlobalLogger>, LogLevel::CADET_LOGLEVEL_MIN> DoubleFilterLogger;

#ifdef __clang__
// Silence -Wundefined-var-template warning by indicating an
// explicit instantiation of this template in another translation unit
template <> LogLevel RuntimeFilteringLogger<GlobalLogger>::_minLvl;
extern template class RuntimeFilteringLogger<GlobalLogger>;
#endif
#else
typedef Logger<GlobalLogger, LogLevel::None> DiscardingLogger;
#endif

} // namespace log
} // namespace cadet

#ifndef CADET_LOGGING_DISABLE

/**
 * @brief Base for logging macros
 * @details Note that because of the usage pattern
 *          <pre>LOG(Info) << "My log line " << arg1;</pre>
 *          no semicolon is appended.
 */
#define LOG(lvl)                                                                                                       \
	cadet::log::DoubleFilterLogger::statement(__FILE__, __func__, __LINE__) =                                          \
		cadet::log::DoubleFilterLogger::template createMessage<cadet::LogLevel::lvl>()

#else

/**
 * @brief Base for logging macros
 * @details Note that because of the usage pattern
 *          <pre>LOG(Info) << "My log line " << arg1;</pre>
 *          no semicolon is appended.
 */
#define LOG(lvl)                                                                                                       \
	cadet::log::DiscardingLogger::statement(__FILE__, __func__, __LINE__) =                                            \
		cadet::log::DiscardingLogger::template createMessage<cadet::LogLevel::lvl>()

#endif

#endif // CADETMEX_LOGGING_IMPL_HPP_
