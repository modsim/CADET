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
 * Provides logging functionality.
 */

#ifndef LIBCADET_LOGGING_HPP_
#define LIBCADET_LOGGING_HPP_

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"

#include <string>

namespace cadet
{
/**
 * @brief LogLevel represents the severity of log messages
 * @details The levels are nested, such that the highest level (Trace) includes all lower levels.
 */
enum class LogLevel : unsigned int
{
	/**
	 * @brief Nothing is logged
	 */
	None = 0,
	/**
	 * @brief Non-recoverable (fatal) error (terminates program / simulation)
	 */
	Fatal = 1,
	/**
	 * @brief Error, which may be recoverable
	 */
	Error = 2,
	/**
	 * @brief Warning
	 */
	Warning = 3,
	/**
	 * @brief Normal output (informative)
	 */
	Normal = 4,
	/**
	 * @brief Additional info output
	 */
	Info = 5,
	/**
	 * @brief Debug messages and values of (intermediate) variables or calculations
	 */
	Debug = 6,
	/**
	 * @brief Full trace including entering / leaving functions
	 */
	Trace = 7,
};

/**
 * @brief Converts a LogLevel to a string
 * @param [in] lvl LogLevel to be converted
 * @return String representation of the LogLevel
 */
inline const char* to_string(LogLevel lvl) CADET_NOEXCEPT
{
	switch (lvl)
	{
	case LogLevel::None:
		return "None";
	case LogLevel::Fatal:
		return "Fatal";
	case LogLevel::Error:
		return "Error";
	case LogLevel::Warning:
		return "Warning";
	case LogLevel::Normal:
		return "Normal";
	case LogLevel::Info:
		return "Info";
	case LogLevel::Debug:
		return "Debug";
	case LogLevel::Trace:
		return "Trace";
	}
	return "Unkown";
}

/**
 * @brief Converts a string to a LogLevel
 * @param [in] ll LogLevel as string
 * @return LogLevel corresponding to the given string
 */
inline LogLevel to_loglevel(const std::string& ll) CADET_NOEXCEPT
{
	if (ll == "None")
		return LogLevel::None;
	else if (ll == "Fatal")
		return LogLevel::Fatal;
	else if (ll == "Error")
		return LogLevel::Error;
	else if (ll == "Warning")
		return LogLevel::Warning;
	else if (ll == "Normal")
		return LogLevel::Normal;
	else if (ll == "Info")
		return LogLevel::Info;
	else if (ll == "Debug")
		return LogLevel::Debug;
	else if (ll == "Trace")
		return LogLevel::Trace;

	return LogLevel::None;
}

/**
 * @brief Interface for receiving log messages
 */
class CADET_API ILogReceiver
{
public:
	virtual ~ILogReceiver() CADET_NOEXCEPT
	{
	}

	/**
	 * @brief Receives a log message
	 * @param [in] file Filename in which the log message was raised
	 * @param [in] func Name of the function (implementation defined @c __func__ variable)
	 * @param [in] line Number of the line in which the log message was raised
	 * @param [in] lvl LogLevel representing the severity of the message
	 * @param [in] lvlStr String representation of the log level
	 * @param [in] message Message string
	 * @sa LogLevel
	 */
	virtual void message(const char* file, const char* func, const unsigned int line, LogLevel lvl, const char* lvlStr,
						 const char* message) = 0;
};

/**
 * @brief Sets the log receiver replacing any previously set receiver
 * @param [in] recv Pointer to ILogReceiver implementation or @c nullptr
 * @sa cadetSetLogReceiver()
 */
CADET_API void setLogReceiver(ILogReceiver* const recv);

/**
 * @brief Sets the log level
 * @details All messages on a lower log level (i.e., higher severity or information content)
 *          are filtered out.
 * @param [in] lvl New log level
 */
CADET_API void setLogLevel(LogLevel lvl);

/**
 * @brief Returns the current log level
 * @return Current log level
 */
CADET_API LogLevel getLogLevel();

} // namespace cadet

#endif // LIBCADET_LOGGING_HPP_
