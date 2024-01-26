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
 * Defines a ParameterProvider that uses JSON.
 */

#ifndef CADET_JSONPARAMETERPROVIDER_HPP_
#define CADET_JSONPARAMETERPROVIDER_HPP_

#include "cadet/ParameterProvider.hpp"
#include "common/CompilerSpecific.hpp"

#include <string>
#include <stack>
#include <ostream>
#include <json.hpp>

namespace cadet
{

class JsonParameterProvider : public cadet::IParameterProvider
{
public:
	JsonParameterProvider(const char* data);
	JsonParameterProvider(const std::string& data);
	JsonParameterProvider(const nlohmann::json& data);
	JsonParameterProvider(const JsonParameterProvider& cpy);
	JsonParameterProvider(JsonParameterProvider&& cpy) CADET_NOEXCEPT;

	virtual ~JsonParameterProvider() CADET_NOEXCEPT;

	JsonParameterProvider& operator=(const JsonParameterProvider& cpy);
	JsonParameterProvider& operator=(JsonParameterProvider&& cpy) CADET_NOEXCEPT;

	virtual double getDouble(const std::string& paramName);
	virtual int getInt(const std::string& paramName);
	virtual uint64_t getUint64(const std::string& paramName);
	virtual bool getBool(const std::string& paramName);
	virtual std::string getString(const std::string& paramName);
	virtual std::vector<double> getDoubleArray(const std::string& paramName);
	virtual std::vector<int> getIntArray(const std::string& paramName);
	virtual std::vector<uint64_t> getUint64Array(const std::string& paramName);
	virtual std::vector<bool> getBoolArray(const std::string& paramName);
	virtual std::vector<std::string> getStringArray(const std::string& paramName);
	virtual bool exists(const std::string& paramName);
	virtual bool isArray(const std::string& paramName);
	virtual std::size_t numElements(const std::string& paramName);
	virtual void pushScope(const std::string& scope);
	virtual void popScope();

	virtual void addScope(const std::string& scope);

	void set(const std::string& paramName, double val);
	void set(const std::string& paramName, int val);
	void set(const std::string& paramName, uint64_t val);
	void set(const std::string& paramName, bool val);
	void set(const std::string& paramName, char const* val);
	void set(const std::string& paramName, const std::string& val);
	void set(const std::string& paramName, const std::vector<double>& val);
	void set(const std::string& paramName, const std::vector<int>& val);
	void set(const std::string& paramName, const std::vector<uint64_t>& val);
	void set(const std::string& paramName, const std::vector<std::string>& val);

	void remove(const std::string& name);

	void copy(const std::string& src, const std::string& dest);

	inline nlohmann::json* data()
	{
		return _root;
	}
	inline nlohmann::json const* data() const
	{
		return _root;
	}

	void toFile(const std::string& fileName) const;
	static JsonParameterProvider fromFile(const std::string& fileName);

private:
	JsonParameterProvider();
	JsonParameterProvider(nlohmann::json* data);

	nlohmann::json* _root;
	std::stack<nlohmann::json*> _opened;

#ifdef CADET_DEBUG
	std::string _scopePath;
#endif
};

std::ostream& operator<<(std::ostream& out, const JsonParameterProvider& jpp);
} // namespace cadet

#endif // CADET_JSONPARAMETERPROVIDER_HPP_
