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

#ifndef CADETCLI_LOGGING_IMPL_HPP_
#define CADETCLI_LOGGING_IMPL_HPP_

#include "common/Logger.hpp"

namespace cadet
{
namespace log
{

/**
 * @brief Implements a standard formatting policy
 */
class CadetCliFormattingPolicy : public FormattingPolicyBase<CadetCliFormattingPolicy>
{
public:
	template <class writePolicy_t, class receiver_t, class paramList_t>
	static inline void format(receiver_t& recv, const char* fileName, const char* funcName, unsigned int line,
							  LogLevel lvl, const paramList_t& p)
	{
		writeObj<writePolicy_t>(recv, lvl, '[');
		writeObj<writePolicy_t>(recv, lvl, to_string(lvl));
		writeObj<writePolicy_t>(recv, lvl, ": ");
		//			writeObj<writePolicy_t>(recv, lvl, fileName);
		//			writeObj<writePolicy_t>(recv, lvl, "::");
		writeObj<writePolicy_t>(recv, lvl, funcName);
		writeObj<writePolicy_t>(recv, lvl, "::");
		writeObj<writePolicy_t>(recv, lvl, line);
		writeObj<writePolicy_t>(recv, lvl, "] ");
		writeParams<writePolicy_t>(recv, lvl, p);
	}
};

typedef NonFilteringLogger<CadetCliFormattingPolicy, StdOutWritePolicy> GlobalLogger;

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

#endif // CADETCLI_LOGGING_IMPL_HPP_
