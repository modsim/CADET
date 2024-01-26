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
 * Provides an implementation of the cadet::IParameterProvider interface
 */

#ifndef CADET_PARAMPROVIDER_HPP_
#define CADET_PARAMPROVIDER_HPP_

#include <string>

#include "cadet/ParameterProvider.hpp"

namespace cadet
{

template <class Reader_t> class ParameterProviderImpl : public cadet::IParameterProvider
{
public:
	ParameterProviderImpl(Reader_t& reader) : ParameterProviderImpl(reader, true)
	{
	}

	ParameterProviderImpl(Reader_t& reader, bool inputPrefix) : _reader(reader)
	{
		if (inputPrefix)
			_reader.setGroup("input");
	}

	virtual ~ParameterProviderImpl() CADET_NOEXCEPT
	{
	}

	virtual double getDouble(const std::string& paramName)
	{
		const double val = _reader.template scalar<double>(paramName);
		LOG(Debug) << "GET scalar [double] " << paramName << " = " << val;
		return val;
	}

	virtual int getInt(const std::string& paramName)
	{
		const int val = _reader.template scalar<int>(paramName);
		LOG(Debug) << "GET scalar [int] " << paramName << " = " << val;
		return val;
	}

	virtual uint64_t getUint64(const std::string& paramName)
	{
		const uint64_t val = _reader.template scalar<uint64_t>(paramName);
		LOG(Debug) << "GET scalar [uint64_t] " << paramName << " = " << val;
		return val;
	}

	virtual bool getBool(const std::string& paramName)
	{
		const bool val = _reader.template scalar<int>(paramName);
		LOG(Debug) << "GET scalar [bool] " << paramName << " = " << val;
		return val;
	}

	virtual std::string getString(const std::string& paramName)
	{
		const std::string val = _reader.template scalar<std::string>(paramName);
		LOG(Debug) << "GET scalar [string] " << paramName << " = " << val;
		return val;
	}

	virtual std::vector<double> getDoubleArray(const std::string& paramName)
	{
		const std::vector<double> res = _reader.template vector<double>(paramName);
		LOG(Debug) << "GET vector [double] " << paramName << " = " << res;
		return res;
	}

	virtual std::vector<int> getIntArray(const std::string& paramName)
	{
		const std::vector<int> res = _reader.template vector<int>(paramName);
		LOG(Debug) << "GET vector [int] " << paramName << " = " << res;
		return res;
	}

	virtual std::vector<uint64_t> getUint64Array(const std::string& paramName)
	{
		const std::vector<uint64_t> res = _reader.template vector<uint64_t>(paramName);
		LOG(Debug) << "GET vector [uint64_t] " << paramName << " = " << res;
		return res;
	}

	virtual std::vector<bool> getBoolArray(const std::string& paramName)
	{
		const std::vector<int> data = _reader.template vector<int>(paramName);
		std::vector<bool> bd(data.size());
		for (std::size_t i = 0; i < data.size(); ++i)
			bd[i] = data[i];

		LOG(Debug) << "GET vector [bool] " << paramName << " = " << bd;
		return bd;
	}

	virtual std::vector<std::string> getStringArray(const std::string& paramName)
	{
		const std::vector<std::string> res = _reader.template vector<std::string>(paramName);
		LOG(Debug) << "GET vector [string] " << paramName << " = " << res;
		return res;
	}

	virtual bool exists(const std::string& paramName)
	{
		const bool val = _reader.exists(paramName);
		LOG(Debug) << "EXISTS " << paramName << " = " << (val ? "yes" : "no");
		return val;
	}

	virtual bool isArray(const std::string& paramName)
	{
		const bool val = _reader.isVector(paramName);
		LOG(Debug) << "ISARRAY " << paramName << " = " << (val ? "yes" : "no");
		return val;
	}

	virtual std::size_t numElements(const std::string& paramName)
	{
		const std::size_t val = _reader.arraySize(paramName);
		LOG(Debug) << "NUMELEMENTS " << paramName << " = " << val;
		return val;
	}

	virtual void pushScope(const std::string& scope)
	{
		LOG(Debug) << "SCOPE " << scope;
		_reader.pushGroup(scope);
	}

	virtual void popScope()
	{
		LOG(Debug) << "SCOPE POP";
		_reader.popGroup();
	}

private:
	Reader_t& _reader;
};

} // namespace cadet

#endif // CADET_PARAMPROVIDER_HPP_
