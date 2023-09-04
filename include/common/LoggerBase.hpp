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

#ifndef CADET_LOGGERBASE_HPP_
#define CADET_LOGGERBASE_HPP_

#include "common/CompilerSpecific.hpp"
#include "cadet/cadetCompilerInfo.hpp"

#include <vector>
#include <string>
#include <sstream>
#include <ostream>
#include <utility>
#include <type_traits>

namespace cadet
{

enum class LogLevel : unsigned int;
inline const char* to_string(LogLevel lvl) CADET_NOEXCEPT;

namespace log
{

namespace detail
{
/**
 * @brief Empty type used as @c nullptr pendant
 */
struct NullType
{
};

template <class T1, class T2> struct NestedList
{
	typedef T1 left_t;
	typedef T2 right_t;
	T1 left;
	T2 right;

	NestedList(const T1& l, const T2& r) CADET_NOEXCEPT : left(l), right(r)
	{
	}
};

/**
 * @brief Stores a log message including all parameters / variables
 * @details If the log statement is not passed on, it will be removed. Otherwise, the
 *          statement and the stored variables are assembled to a string and passed on
 *          to a receiver.
 * @tparam lvl LogLevel of this message
 * @tparam passOn Determines whether the message is passed on or filtered out
 * @tparam params_t Parameters of the statement (type list)
 */
template <LogLevel lvl, bool passOn, class params_t> struct LogMessage
{
	params_t params;

	LogMessage(const params_t& p = params_t()) CADET_NOEXCEPT : params(p)
	{
	}
};

/**
 * @brief Adds a parameter to a log statement
 * @details In this version, the statement is discarded. Thus, no parameters are saved.
 */
template <LogLevel lvl, class paramList_t, class param_t>
inline LogMessage<lvl, false, NullType> operator<<(const LogMessage<lvl, false, paramList_t>&,
												   const param_t&) CADET_NOEXCEPT
{
	return LogMessage<lvl, false, NullType>();
}

/**
 * @brief Adds a parameter to a log statement
 * @details In this version, the statement is passed on. Thus, the parameter is saved in a (nested) pair.
 */
template <LogLevel lvl, class paramList_t, class param_t>
inline LogMessage<lvl, true, NestedList<paramList_t, const param_t*>> operator<<(
	const LogMessage<lvl, true, paramList_t>& lm, const param_t& p) CADET_NOEXCEPT
{
	return LogMessage<lvl, true, NestedList<paramList_t, const param_t*>>(
		NestedList<paramList_t, const param_t*>(lm.params, &p));
}

/**
 * @brief Compiles a full log statement consisting of positional information and a log mesage that is sent to the @p
 * logger_t
 * @details LogStatement holds positional information about a log statement (filename, function name, line number)
 *          and, upon assignment of a LogMessage, forwards the entire statement (including the message) to the
 *          message() function of the logger_t type.
 * @tparam logger_t Logger to forward messages to
 */
template <class logger_t> struct LogStatement
{
	const char* fileName;
	const char* funcName;
	unsigned int line;

	LogStatement(const char* fin, const char* fun, unsigned int ln) CADET_NOEXCEPT : fileName(fin),
																					 funcName(fun),
																					 line(ln)
	{
	}

	template <LogLevel lvl, bool passOn, class params_t>
	inline void operator=(const LogMessage<lvl, passOn, params_t>& lm)
	{
		logger_t::forward(fileName, funcName, line, lm);
	}

	inline LogStatement& operator=(const LogStatement& cpy) CADET_NOEXCEPT
	{
		fileName = cpy.fileName;
		funcName = cpy.funcName;
		line = cpy.line;
	}
};

} // namespace detail

template <class T> inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	os << "[";
	if (!v.empty())
	{
		for (std::size_t i = 0; i < v.size() - 1; ++i)
			os << v[i] << ",";
		os << v.back();
	}
	os << "]";
	return os;
}

/**
 * @brief Logger class that filters log messages at compile- or runtime
 * @details A hierarchy of loggers is used to filter out messages. Only the logger at the very
 *          top of the chain actually writes messages to its destination. This logger should
 *          always be the bottom one in the chain (i.e., the first to see a LogMessage) in order
 *          for compile-time filtering to work. The loggers messages are passed on to by this
 *          logger has to provide two functions with the following signatures:
 *          <pre>
 *              template <LogLevel stmtLevel, class params_t>
 *              static inline void forward(const char* fileName, const char* funcName, unsigned int line, const
 * LogMessage<stmtLevel, true, params_t>& lm)
 *
 *              template <LogLevel stmtLevel, class params_t>
 *              static inline void forward(const char* fileName, const char* funcName, unsigned int line, const
 * LogMessage<stmtLevel, false, params_t>& lm)
 *          </pre>
 *          These functions receive log messages that have not been filtered out or have been filtered out
 * (respectively) at compile time.
 */
template <class nextLogger_t, LogLevel lvl> class Logger
{
private:
	/**
	 * @brief Compile-time template construct for determining whether a message will be discarded
	 */
	template <LogLevel queryLvl> struct ForwardMessage
	{
		enum
		{
			result = (lvl >= queryLvl)
		};
	};

public:
	/**
	 * @brief This logger type
	 */
	typedef Logger<nextLogger_t, lvl> this_logger_t;

	/**
	 * @brief Type of logger the messages are passed on to
	 */
	typedef nextLogger_t forward_logger_t;

	/**
	 * @brief Creates a LogStatement with positional information
	 */
	static inline detail::LogStatement<this_logger_t> statement(const char* fileName, const char* funcName,
																unsigned int line)
	{
		return detail::LogStatement<this_logger_t>(fileName, funcName, line);
	}

	/**
	 * @brief Creates a LogMessage which is subsequently constructed and then assigned to a LogStatement
	 * @details Here, it is decided whether the LogStatement / LogMessage is filtered out at compile time.
	 *          If <tt>ForwardMessage<stmtLevel>::result</tt> if @c true, then the message is passed on.
	 *          Otherwise it is filtered out by not passing it on to the next logger.
	 */
	template <LogLevel stmtLevel>
	static inline detail::LogMessage<stmtLevel, ForwardMessage<stmtLevel>::result, detail::NullType> createMessage()
	{
		return detail::LogMessage<stmtLevel, ForwardMessage<stmtLevel>::result, detail::NullType>();
	}

	/**
	 * @brief Passes a given LogMessage on to the next logger. Overload for discarded messages (no forwarding).
	 */
	template <LogLevel stmtLevel, class params_t>
	static inline void forward(const char*, const char*, unsigned int,
							   const detail::LogMessage<stmtLevel, false, params_t>&)
	{
	}

	/**
	 * @brief Passes a given LogMessage on to the next logger. Overload for passed on messages (forwarding).
	 */
	template <LogLevel stmtLevel, class params_t>
	static inline void forward(const char* fileName, const char* funcName, unsigned int line,
							   const detail::LogMessage<stmtLevel, true, params_t>& lm)
	{
		forward_logger_t::forward(fileName, funcName, line, lm);
	}
};

/**
 * @brief Provides base for formatting of log messages
 * @details Formats a log message and uses a write policy to send the message to a receiver.
 *          The formatting policy collaborates closely with the write policy in order to
 *          achieve a decent performance via compile time optimization.
 *
 *          Implementations have to implement the function
 *          <pre>
 *             template <class writePolicy_t, class receiver_t, LogLevel lvl, class paramList_t>
 *             static inline void format(receiver_t& recv, const char* fileName, const char* funcName, unsigned int
 * line, const paramList_t& p)
 *          </pre>
 *          which uses the methods writeObj() and writeParams() of this class to format the given
 *          log message.
 *
 * @sa StandardFormattingPolicy
 * @tparam subFormattingPolicy_t Actual formatting policy implementation (CRTP)
 */
template <class subFormattingPolicy_t> class FormattingPolicyBase
{
public:
	/**
	 * @brief Formats and writes a log message
	 * @details This function is called by the write policy and serves as an entry point to
	 *          the formatting policy mechanism.
	 */
	template <class writePolicy_t, class paramList_t>
	static inline void formatMessage(const char* fileName, const char* funcName, unsigned int line, LogLevel lvl,
									 const paramList_t& p)
	{
		formatImpl<writePolicy_t>(fileName, funcName, line, lvl, p, typename writePolicy_t::hasBuffer());
	}

protected:
	/**
	 * @brief Writes an object (e.g., string) using the given write policy
	 */
	template <class writePolicy_t, class receiver_t, class T>
	static inline void writeObj(receiver_t& recv, LogLevel lvl, const T& obj)
	{
		writeObjImpl<writePolicy_t>(recv, lvl, obj);
	}

	/**
	 * @brief Writes the parameters of the log message (i.e., the message itself) using the given write policy
	 */
	template <class writePolicy_t, class receiver_t, class paramList_t, class T>
	static inline void writeParams(receiver_t& recv, LogLevel lvl, const detail::NestedList<paramList_t, T>& p)
	{
		writeParamsImpl<writePolicy_t>(recv, lvl, p);
	}

private:
	/**
	 * @brief Boiler plate code for the implementation in the @c subFormattingPolicy_t
	 * @details Buffered write policy version.
	 */
	template <class writePolicy_t, class paramList_t>
	static inline void formatImpl(const char* fileName, const char* funcName, unsigned int line, LogLevel lvl,
								  const paramList_t& p, std::true_type inc)
	{
		writePolicy_t::begin(fileName, funcName, line, lvl);

		detail::NullType dummy;
		subFormattingPolicy_t::template format<writePolicy_t, detail::NullType>(dummy, fileName, funcName, line, lvl,
																				p);

		writePolicy_t::end(lvl);
	}

	/**
	 * @brief Boiler plate code for the implementation in the @c subFormattingPolicy_t
	 * @details Non-buffered write policy version. Buffers everything in a stringstream
	 *          which is then handed over at once to the write policy.
	 */
	template <class writePolicy_t, class paramList_t>
	static inline void formatImpl(const char* fileName, const char* funcName, unsigned int line, LogLevel lvl,
								  const paramList_t& p, std::false_type inc)
	{
		std::ostringstream oss;
		subFormattingPolicy_t::template format<writePolicy_t, std::ostringstream>(oss, fileName, funcName, line, lvl,
																				  p);
		oss << "\n";
		writePolicy_t::writeLine(fileName, funcName, line, lvl, oss.str());
	}

	/**
	 * @brief Writes an object using the given write policy
	 * @details Non-buffered write policy version.
	 */
	template <class writePolicy_t, class T>
	static inline void writeObjImpl(detail::NullType& dummy, LogLevel lvl, const T& obj)
	{
		writePolicy_t::writeObj(lvl, obj);
	}

	/**
	 * @brief Writes an object using the given write policy
	 * @details Buffered write policy version.
	 */
	template <class writePolicy_t, class T>
	static inline void writeObjImpl(std::ostream& os, LogLevel lvl, const T& obj)
	{
		os << obj;
	}

	/**
	 * @brief Writes all parameters (i.e., the actual log message)
	 * @details The parameters are saved in a nested chain of std::pair. Recursion is
	 *          applied to write all parameters in the correct order. Given an std::pair,
	 *          the parameter is placed in the second argument and the rest of the chain in
	 *          the first. Thus, the first parameter handed to the LogMessage object is the
	 *          most left leaf in the tree.
	 */
	template <class writePolicy_t, class receiver_t, class paramList_t, class T>
	static inline void writeParamsImpl(receiver_t& recv, LogLevel lvl, const detail::NestedList<paramList_t, T>& p)
	{
		writeParamsImpl<writePolicy_t>(recv, lvl, p.left);
		writeParamsImpl<writePolicy_t>(recv, lvl, p.right);
	}

	/**
	 * @brief Writes all parameters (i.e., the actual log message)
	 * @details Terminates the recursion.
	 */
	template <class writePolicy_t, class receiver_t>
	static inline void writeParamsImpl(receiver_t& recv, LogLevel lvl, detail::NullType p)
	{
	}

	/**
	 * @brief Writes all parameters (i.e., the actual log message)
	 * @details Terminates the recursion.
	 */
	template <class writePolicy_t, class receiver_t>
	static inline void writeParamsImpl(receiver_t& recv, LogLevel lvl, const detail::NullType* p)
	{
	}

	/**
	 * @brief Writes all parameters (i.e., the actual log message)
	 * @details Writes an object to a receiver using the given write policy. Considering
	 *          the chain of std::pairs as a tree, this function handles all the right
	 *          leaves containing parameters.
	 */
	template <class writePolicy_t, class receiver_t, class T>
	static inline void writeParamsImpl(receiver_t& recv, LogLevel lvl, const T* p)
	{
		writeObj<writePolicy_t>(recv, lvl, *p);
	}
};

/**
 * @brief Implements a standard formatting policy
 */
class StandardFormattingPolicy : public FormattingPolicyBase<StandardFormattingPolicy>
{
public:
	template <class writePolicy_t, class receiver_t, class paramList_t>
	static inline void format(receiver_t& recv, const char* fileName, const char* funcName, unsigned int line,
							  LogLevel lvl, const paramList_t& p)
	{
		writeObj<writePolicy_t>(recv, lvl, '[');
		writeObj<writePolicy_t>(recv, lvl, to_string(lvl));
		writeObj<writePolicy_t>(recv, lvl, ": ");
		writeObj<writePolicy_t>(recv, lvl, fileName);
		writeObj<writePolicy_t>(recv, lvl, "::");
		writeObj<writePolicy_t>(recv, lvl, funcName);
		writeObj<writePolicy_t>(recv, lvl, "::");
		writeObj<writePolicy_t>(recv, lvl, line);
		writeObj<writePolicy_t>(recv, lvl, "] ");
		writeParams<writePolicy_t>(recv, lvl, p);
	}
};

/**
 * @brief Base class for write policies
 * @details Write policies can either use an internal buffer that allows for incremental writing
 *          like streams (think of std::cout, for example), or use direct writing without an internal
 *          buffer (e.g., printf()).
 *
 *          The write policy calls the formatting policy which in turn then uses the write policy to
 *          send the formatted message to the receiver. Thus, write and formatting policies collaborate
 *          closely.
 *
 *          Derive your implementation from BufferedWritePolicyBase or NonBufferedWritePolicyBase
 *          for convenience.
 *
 * @sa BufferedWritePolicyBase
 * @sa NonBufferedWritePolicyBase
 * @tparam subWritePolicy_t Actual write policy implementation (CRTP)
 * @tparam incWrite @c true for buffered and @c false for direct writing
 */
template <class subWritePolicy_t, bool incWrite> class WritePolicyBase
{
public:
	typedef typename std::integral_constant<bool, incWrite> hasBuffer;

	template <class formattingPolicy_t, class paramList_t>
	static inline void write(const char* fileName, const char* funcName, unsigned int line, LogLevel lvl,
							 const paramList_t& p)
	{
		writeImpl<formattingPolicy_t, paramList_t>(fileName, funcName, line, lvl, p);
	}

private:
	template <class formattingPolicy_t, class paramList_t>
	static inline void writeImpl(const char* fileName, const char* funcName, unsigned int line, LogLevel lvl,
								 const paramList_t& p)
	{
		formattingPolicy_t::template formatMessage<subWritePolicy_t>(fileName, funcName, line, lvl, p);
	}
};

/**
 * @brief Buffered write policies have an internal buffer
 * @details Derive from this class to implement your own write policy that has an internal buffer
 *          and an interface such as std::ostream. You need to implement the following functions:
 *          <pre>
 *              static inline void beginWrite(const char* fileName, const char* funcName, unsigned int line, LogLevel
 * lvl); static inline void endWrite(LogLevel lvl); template <class T> static inline void writeObj(LogLevel lvl, const
 * T& obj);
 *          </pre>
 *          The functions are called at the beginning of a log statement, its end, and for every
 *          object that is written. The endWrite() function should append a newline '\n' character.
 * @sa WritePolicyBase
 */
template <class writePolicy_t> class BufferedWritePolicyBase : public WritePolicyBase<writePolicy_t, true>
{
};

/**
 * @brief Non-buffered write policies do not have an internal buffer
 * @details Derive from this class to implement your own write policy that does not have an internal
 *          buffer. Instead, buffering is done by the logging library and full strings / lines are handed
 *          to the writer. You need to implement the following functions:
 *          <pre>
 *              static inline void writeLine(const char* fileName, const char* funcName, unsigned int line, LogLevel
 * lvl, const std::string& msg);
 *          </pre>
 *          The lines send to write() already contain a final newline '\n' character.
 * @sa WritePolicyBase
 */
template <class writePolicy_t> class NonBufferedWritePolicyBase : public WritePolicyBase<writePolicy_t, false>
{
};

template <class forward_logger_t> class RuntimeFilteringLogger
{
public:
	template <LogLevel lvl, class params_t>
	static inline void forward(const char* fileName, const char* funcName, unsigned int line,
							   const detail::LogMessage<lvl, true, params_t>& lm)
	{
		if (lvl <= _minLvl)
			forward_logger_t::forward(fileName, funcName, line, lm);
	}

	static inline LogLevel level() CADET_NOEXCEPT
	{
		return _minLvl;
	}
	static inline void level(LogLevel newLvl) CADET_NOEXCEPT
	{
		_minLvl = newLvl;
	}

private:
	static LogLevel _minLvl;
};

template <class formattingPolicy_t, class writePolicy_t> class NonFilteringLogger
{
public:
	template <LogLevel lvl, class params_t>
	static inline void forward(const char* fileName, const char* funcName, unsigned int line,
							   const detail::LogMessage<lvl, false, params_t>& lm)
	{
	}

	template <LogLevel lvl, class params_t>
	static inline void forward(const char* fileName, const char* funcName, unsigned int line,
							   const detail::LogMessage<lvl, true, params_t>& lm)
	{
		writePolicy_t::template write<formattingPolicy_t>(fileName, funcName, line, lvl, lm.params);
	}
};

} // namespace log
} // namespace cadet

/**
 * @brief Base for logging macros
 * @details Note that because of the usage pattern
 *          <pre>LOG_BASE(myLogger, Info) << "My log line " << arg1;</pre>
 *          no semicolon is appended.
 */
#define LOG_BASE(logger_t, lvl)                                                                                        \
	logger_t::statement(__FILE__, __func__, __LINE__) = logger_t::template createMessage<cadet::LogLevel::lvl>()

#endif // CADET_LOGGERBASE_HPP_
