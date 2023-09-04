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

#include <json.hpp>

#include <sstream>
#include <fstream>
#include <iomanip>

#include "common/CompilerSpecific.hpp"

#include "common/JsonParameterProvider.hpp"

// Uncomment next line to enable logging in JsonParameterProvider
// #define CADET_JSON_LOGGING_ENABLE

#ifdef CADET_JSON_LOGGING_ENABLE

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
#else
typedef Logger<GlobalLogger, LogLevel::None> DiscardingLogger;
#endif

} // namespace log
} // namespace cadet

/**
 * @brief Base for logging macros
 * @details Note that because of the usage pattern
 *          <pre>LOG(Info) << "My log line " << arg1;</pre>
 *          no semicolon is appended.
 */
#define LOG(lvl)                                                                                                       \
	cadet::log::DoubleFilterLogger::statement(__FILE__, __func__, __LINE__) =                                          \
		cadet::log::DoubleFilterLogger::template createMessage<cadet::LogLevel::lvl>()

#endif

using json = nlohmann::json;

namespace cadet
{

JsonParameterProvider::JsonParameterProvider() : _root(nullptr)
{
#ifdef CADET_DEBUG
	_scopePath = "/";
#endif
}

JsonParameterProvider::JsonParameterProvider(const char* data) : _root(new json(json::parse(data)))
{
	_opened.push(_root);
#ifdef CADET_DEBUG
	_scopePath = "/";
#endif
}

JsonParameterProvider::JsonParameterProvider(const std::string& data) : _root(new json(json::parse(data)))
{
	_opened.push(_root);
#ifdef CADET_DEBUG
	_scopePath = "/";
#endif
}

JsonParameterProvider::JsonParameterProvider(const json& data) : _root(new json(data))
{
	_opened.push(_root);
#ifdef CADET_DEBUG
	_scopePath = "/";
#endif
}

JsonParameterProvider::JsonParameterProvider(json* data) : _root(data)
{
	_opened.push(_root);
#ifdef CADET_DEBUG
	_scopePath = "/";
#endif
}

JsonParameterProvider::JsonParameterProvider(const JsonParameterProvider& cpy)
{
	_root = new json(*cpy._root);
	_opened = cpy._opened;
#ifdef CADET_DEBUG
	_scopePath = cpy._scopePath;
#endif
}

JsonParameterProvider::JsonParameterProvider(JsonParameterProvider&& cpy) CADET_NOEXCEPT
	: _root(cpy._root),
	  _opened(std::move(cpy._opened))
{
	cpy._root = nullptr;
	cpy._opened = std::stack<json*>();
#ifdef CADET_DEBUG
	_scopePath = std::move(cpy._scopePath);
#endif
}

JsonParameterProvider::~JsonParameterProvider() CADET_NOEXCEPT
{
	delete _root;
}

JsonParameterProvider& JsonParameterProvider::operator=(const JsonParameterProvider& cpy)
{
	delete _root;

	_root = new json(*cpy._root);
	_opened = cpy._opened;

#ifdef CADET_DEBUG
	_scopePath = cpy._scopePath;
#endif

	return *this;
}

JsonParameterProvider& JsonParameterProvider::operator=(JsonParameterProvider&& cpy) CADET_NOEXCEPT
{
	delete _root;
	_opened = std::stack<nlohmann::json*>();
	_root = cpy._root;
	cpy._root = nullptr;
	cpy._opened = std::stack<nlohmann::json*>();

	_opened.push(_root);

#ifdef CADET_DEBUG
	_scopePath = std::move(cpy._scopePath);
#endif

	return *this;
}

double JsonParameterProvider::getDouble(const std::string& paramName)
{
	json& p = _opened.top()->at(paramName);
	if (p.is_array() && (p.size() == 1))
		p = p[0];

#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "GET scalar [double] " << paramName << " = " << p.get<double>();
#endif
	return p.get<double>();
}

int JsonParameterProvider::getInt(const std::string& paramName)
{
	json& p = _opened.top()->at(paramName);
	if (p.is_array() && (p.size() == 1))
		p = p[0];

#ifdef CADET_JSON_LOGGING_ENABLE
	if (p.is_boolean())
		LOG(Debug) << "GET scalar [int] " << paramName << " = " << static_cast<int>(p.get<bool>());
	else
		LOG(Debug) << "GET scalar [int] " << paramName << " = " << p.get<int>();
#endif
	if (p.is_boolean())
		return p.get<bool>();
	else
		return p.get<int>();
}

uint64_t JsonParameterProvider::getUint64(const std::string& paramName)
{
	json& p = _opened.top()->at(paramName);
	if (p.is_array() && (p.size() == 1))
		p = p[0];

#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "GET scalar [uint64_t] " << paramName << " = " << p.get<uint64_t>();
#endif
	return p.get<uint64_t>();
}

bool JsonParameterProvider::getBool(const std::string& paramName)
{
	json& p = _opened.top()->at(paramName);
	if (p.is_array() && (p.size() == 1))
		p = p[0];

#ifdef CADET_JSON_LOGGING_ENABLE
	if (p.is_number_integer())
		LOG(Debug) << "GET scalar [bool] " << paramName << " = " << static_cast<bool>(p.get<int>());
	else
		LOG(Debug) << "GET scalar [bool] " << paramName << " = " << p.get<bool>();
#endif
	if (p.is_number_integer())
		return p.get<int>();
	else
		return p.get<bool>();
}

std::string JsonParameterProvider::getString(const std::string& paramName)
{
	json& p = _opened.top()->at(paramName);
	if (p.is_array() && (p.size() == 1))
		p = p[0];

#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "GET scalar [string] " << paramName << " = " << p.get<std::string>();
#endif
	return p.get<std::string>();
}

std::vector<double> JsonParameterProvider::getDoubleArray(const std::string& paramName)
{
	const json& p = _opened.top()->at(paramName);
	if (!p.is_array())
	{
#ifdef CADET_JSON_LOGGING_ENABLE
		LOG(Debug) << "GET vector [double] " << paramName << " = " << p.get<double>();
#endif
		return std::vector<double>(1, p.get<double>());
	}

#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "GET vector [double] " << paramName << " = " << p.get<std::vector<double>>();
#endif
	return p.get<std::vector<double>>();
}

std::vector<int> JsonParameterProvider::getIntArray(const std::string& paramName)
{
	const json& p = _opened.top()->at(paramName);
	if (p.is_array())
	{
		if (p.size() == 0)
			return std::vector<int>(0);

		if (p[0].is_boolean())
		{
			const std::vector<bool> d = p.template get<std::vector<bool>>();
			std::vector<int> bd(d.size());
			for (std::size_t i = 0; i < d.size(); ++i)
				bd[i] = d[i];

#ifdef CADET_JSON_LOGGING_ENABLE
			LOG(Debug) << "GET vector [int] " << paramName << " = " << bd;
#endif
			return bd;
		}

#ifdef CADET_JSON_LOGGING_ENABLE
		LOG(Debug) << "GET vector [int] " << paramName << " = " << p.template get<std::vector<int>>();
#endif
		return p.template get<std::vector<int>>();
	}
	else
	{
		if (p.is_boolean())
		{
#ifdef CADET_JSON_LOGGING_ENABLE
			LOG(Debug) << "GET vector [int] " << paramName << " = " << static_cast<int>(p.template get<bool>());
#endif
			return std::vector<int>(1, p.template get<bool>());
		}

#ifdef CADET_JSON_LOGGING_ENABLE
		LOG(Debug) << "GET vector [int] " << paramName << " = " << p.template get<int>();
#endif

		return std::vector<int>(1, p.template get<int>());
	}
}

std::vector<uint64_t> JsonParameterProvider::getUint64Array(const std::string& paramName)
{
	const json& p = _opened.top()->at(paramName);
	if (!p.is_array())
	{
#ifdef CADET_JSON_LOGGING_ENABLE
		LOG(Debug) << "GET vector [uint64_t] " << paramName << " = " << p.get<uint64_t>();
#endif
		return std::vector<uint64_t>(1, p.get<uint64_t>());
	}
#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "GET vector [uint64_t] " << paramName << " = " << p.get<std::vector<uint64_t>>();
#endif
	return p.get<std::vector<uint64_t>>();
}

std::vector<bool> JsonParameterProvider::getBoolArray(const std::string& paramName)
{
	const json& p = _opened.top()->at(paramName);
	if (p.is_array())
	{
		if (p.size() == 0)
			return std::vector<bool>(0);

		if (p[0].is_number_integer())
		{
			const std::vector<int> d = p.template get<std::vector<int>>();
			std::vector<bool> bd(d.size());
			for (std::size_t i = 0; i < d.size(); ++i)
				bd[i] = d[i];

#ifdef CADET_JSON_LOGGING_ENABLE
			LOG(Debug) << "GET vector [bool] " << paramName << " = " << bd;
#endif
			return bd;
		}

#ifdef CADET_JSON_LOGGING_ENABLE
		LOG(Debug) << "GET vector [bool] " << paramName << " = " << p.template get<std::vector<bool>>();
#endif
		return p.template get<std::vector<bool>>();
	}
	else
	{
		if (p.is_number_integer())
		{
#ifdef CADET_JSON_LOGGING_ENABLE
			LOG(Debug) << "GET vector [bool] " << paramName << " = " << static_cast<bool>(p.template get<int>());
#endif
			return std::vector<bool>(1, p.template get<int>());
		}

#ifdef CADET_JSON_LOGGING_ENABLE
		LOG(Debug) << "GET vector [bool] " << paramName << " = " << p.template get<bool>();
#endif
		return std::vector<bool>(1, p.template get<bool>());
	}
}

std::vector<std::string> JsonParameterProvider::getStringArray(const std::string& paramName)
{
	const json& p = _opened.top()->at(paramName);
	if (!p.is_array())
	{
#ifdef CADET_JSON_LOGGING_ENABLE
		LOG(Debug) << "GET vector [string] " << paramName << " = " << p.get<std::string>();
#endif
		return std::vector<std::string>(1, p.get<std::string>());
	}
#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "GET vector [string] " << paramName << " = " << p.get<std::vector<std::string>>();
#endif
	return p.get<std::vector<std::string>>();
}

bool JsonParameterProvider::exists(const std::string& paramName)
{
#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "EXISTS " << paramName << " = "
			   << ((_opened.top()->find(paramName) != _opened.top()->end()) ? "yes" : "no");
#endif
	return _opened.top()->find(paramName) != _opened.top()->end();
}

bool JsonParameterProvider::isArray(const std::string& paramName)
{
#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "ISARRAY " << paramName << " = " << (_opened.top()->at(paramName).is_array() ? "yes" : "no");
#endif
	return _opened.top()->at(paramName).is_array();
}

std::size_t JsonParameterProvider::numElements(const std::string& paramName)
{
#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "NUMELEMENTS " << paramName << " = " << (_opened.top()->at(paramName).size());
#endif
	return _opened.top()->at(paramName).size();
}

void JsonParameterProvider::pushScope(const std::string& scope)
{
#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "SCOPE " << scope;
#endif
	_opened.push(&_opened.top()->at(scope));

#ifdef CADET_DEBUG
	_scopePath += "/" + scope;
#endif
}

void JsonParameterProvider::popScope()
{
#ifdef CADET_JSON_LOGGING_ENABLE
	LOG(Debug) << "SCOPE POP";
#endif
	_opened.pop();

#ifdef CADET_DEBUG
	std::size_t lastIdx = std::string::npos;
	if (_scopePath.back() == '/')
		lastIdx = _scopePath.length() - 2;

	const std::size_t idx = _scopePath.find_last_of('/', lastIdx);
	_scopePath.erase(idx);
#endif
}

void JsonParameterProvider::addScope(const std::string& scope)
{
	if (!exists(scope))
	{
		json j;
		j["blubber"] = 0.0;
		j.erase("blubber");
		(*_opened.top())[scope] = j;
	}
}

void JsonParameterProvider::set(const std::string& paramName, double val)
{
	(*_opened.top())[paramName] = val;
}

void JsonParameterProvider::set(const std::string& paramName, int val)
{
	(*_opened.top())[paramName] = val;
}

void JsonParameterProvider::set(const std::string& paramName, uint64_t val)
{
	(*_opened.top())[paramName] = val;
}

void JsonParameterProvider::set(const std::string& paramName, bool val)
{
	(*_opened.top())[paramName] = val;
}

void JsonParameterProvider::set(const std::string& paramName, char const* val)
{
	(*_opened.top())[paramName] = std::string(val);
}

void JsonParameterProvider::set(const std::string& paramName, const std::string& val)
{
	(*_opened.top())[paramName] = val;
}

void JsonParameterProvider::set(const std::string& paramName, const std::vector<double>& val)
{
	(*_opened.top())[paramName] = val;
}

void JsonParameterProvider::set(const std::string& paramName, const std::vector<int>& val)
{
	(*_opened.top())[paramName] = val;
}

void JsonParameterProvider::set(const std::string& paramName, const std::vector<uint64_t>& val)
{
	(*_opened.top())[paramName] = val;
}

void JsonParameterProvider::set(const std::string& paramName, const std::vector<std::string>& val)
{
	(*_opened.top())[paramName] = val;
}

void JsonParameterProvider::remove(const std::string& name)
{
	(*_opened.top()).erase(name);
}

void JsonParameterProvider::copy(const std::string& src, const std::string& dest)
{
	const json j = (*_opened.top())[src];
	(*_opened.top())[dest] = j;
}

void JsonParameterProvider::toFile(const std::string& fileName) const
{
	std::ofstream ofs(fileName, std::ios::out | std::ios::trunc);
	ofs << _root->dump(4);
}

JsonParameterProvider JsonParameterProvider::fromFile(const std::string& fileName)
{
	std::ifstream ifs(fileName);
	json* root = new json();
	ifs >> (*root);

	return JsonParameterProvider(root);
}

std::ostream& operator<<(std::ostream& out, const JsonParameterProvider& jpp)
{
	out << jpp.data()->dump(4);
	return out;
}

} // namespace cadet
