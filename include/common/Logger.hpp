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
 * Implementation of a logging mechanism that can filter messages at compile- and runtime.
 *
 * This implementation is based on templog, a logging library created by Hendrik Schober
 * distributed under the Boost Software License, Version 1.0 (see http://www.boost.org/LICENSE_1_0.txt).
 * Templog can be found at http://templog.sourceforge.net/
 */

#ifndef CADET_LOGGER_HPP_
#define CADET_LOGGER_HPP_

#include "common/LoggerBase.hpp"

#include <iostream>

namespace cadet
{
namespace log
{
/**
 * @brief Sends all messages to std::cout
 */
class StdOutWritePolicy : public BufferedWritePolicyBase<StdOutWritePolicy>
{
public:
	static inline void begin(const char* fileName, const char* funcName, unsigned int line, LogLevel lvl)
	{
	}

	static inline void end(LogLevel lvl)
	{
		std::cout << std::endl;
	}

	template <class T> static inline void writeObj(LogLevel lvl, const T& obj)
	{
		std::cout << obj;
	}
};

/**
 * @brief Sends all messages to std::cerr
 */
class StdErrWritePolicy : public BufferedWritePolicyBase<StdErrWritePolicy>
{
public:
	static inline void begin(const char* fileName, const char* funcName, unsigned int line, LogLevel lvl)
	{
	}

	static inline void end(LogLevel lvl)
	{
		std::cerr << std::endl;
	}

	template <class T> static inline void writeObj(LogLevel lvl, const T& obj)
	{
		std::cerr << obj;
	}
};

/**
 * @brief Sends messages up to a specified log level to std::cerr and the rest to std::cout
 * @tparam switchLvl Defines the first log level that is sent to std::cout
 */
template <LogLevel switchLvl>
class SelectiveStdWritePolicy : public BufferedWritePolicyBase<SelectiveStdWritePolicy<switchLvl>>
{
public:
	static inline void begin(const char* fileName, const char* funcName, unsigned int line, LogLevel lvl)
	{
	}

	static inline void end(LogLevel lvl)
	{
		if (lvl < switchLvl)
			std::cerr << std::endl;
		else
			std::cout << std::endl;
	}

	template <class T> static inline void writeObj(LogLevel lvl, const T& obj)
	{
		if (lvl < switchLvl)
			std::cerr << std::endl;
		else
			std::cout << std::endl;
	}
};

} // namespace log
} // namespace cadet

#endif // CADET_LOGGER_HPP_
