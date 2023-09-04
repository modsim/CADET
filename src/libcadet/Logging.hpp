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
 * Adapter for transmitting the log messages to a receiver.
 */

#ifndef LIBCADET_LOGGING_IMPL_HPP_
#define LIBCADET_LOGGING_IMPL_HPP_

#include "cadet/Logging.hpp"
#include "common/LoggerBase.hpp"

namespace cadet
{
namespace log
{

/**
 * @brief Dispatches a log message to a receiver
 * @param [in] file Filename in which the log message was raised
 * @param [in] func Name of the function (implementation defined @c __func__ variable)
 * @param [in] line Number of the line in which the log message was raised
 * @param [in] lvl LogLevel representing the severity of the message
 * @param [in] message Message string
 */
void emitLog(const char* file, const char* func, const unsigned int line, LogLevel lvl, const char* message);

/**
 * @brief Implements a standard formatting policy
 */
class LibCadetFormattingPolicy : public FormattingPolicyBase<LibCadetFormattingPolicy>
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
 * @brief Sends all messages to std::cout
 */
class EmitterWritePolicy : public NonBufferedWritePolicyBase<EmitterWritePolicy>
{
public:
	static inline void writeLine(const char* fileName, const char* funcName, unsigned int line, LogLevel lvl,
								 const std::string& msg)
	{
		emitLog(fileName, funcName, line, lvl, msg.c_str());
	}
};

typedef NonFilteringLogger<LibCadetFormattingPolicy, EmitterWritePolicy> GlobalLogger;

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

#endif // LIBCADET_LOGGING_IMPL_HPP_
