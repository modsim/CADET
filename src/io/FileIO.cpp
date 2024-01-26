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

#include <sstream>
#include <iomanip>
#include <stack>
#include <iterator>
#include <fstream>

#include "io/FileIO.hpp"
#include "io/hdf5/HDF5Reader.hpp"
#include "io/hdf5/HDF5Writer.hpp"
#include "io/xml/XMLReader.hpp"
#include "io/xml/XMLWriter.hpp"
#include "cadet/StringUtil.hpp"

#include <json.hpp>

using json = nlohmann::json;

namespace
{

/**
 * @brief Implements reading capabilities from a generic file reader class
 * @details Wraps a generic file reading class providing the IFileReader interface.
 * @tparam reader_t Generic file reader class
 */
template <typename reader_t> class FileReader : public cadet::io::IFileReader
{
public:
	FileReader()
	{
	}
	virtual ~FileReader() CADET_NOEXCEPT
	{
	}

	virtual double getDouble(const std::string& paramName)
	{
		return _io.template scalar<double>(paramName);
	}

	virtual int getInt(const std::string& paramName)
	{
		return _io.template scalar<int>(paramName);
	}

	virtual bool getBool(const std::string& paramName)
	{
		return _io.template scalar<int>(paramName);
	}

	virtual std::string getString(const std::string& paramName)
	{
		return _io.template scalar<std::string>(paramName);
	}

	virtual std::vector<double> getDoubleArray(const std::string& paramName)
	{
		return _io.template vector<double>(paramName);
	}

	virtual std::vector<int> getIntArray(const std::string& paramName)
	{
		return _io.template vector<int>(paramName);
	}

	virtual std::vector<bool> getBoolArray(const std::string& paramName)
	{
		const std::vector<int> data = _io.template vector<int>(paramName);
		std::vector<bool> bd(data.size());
		for (std::size_t i = 0; i < data.size(); ++i)
			bd[i] = data[i];

		return bd;
	}

	virtual std::vector<std::string> getStringArray(const std::string& paramName)
	{
		return _io.template vector<std::string>(paramName);
	}

	virtual bool isArray(const std::string& paramName)
	{
		return _io.isVector(paramName);
	}

	virtual bool isString(const std::string& elementName)
	{
		return _io.isString(elementName);
	}

	virtual bool isInt(const std::string& elementName)
	{
		return _io.isInt(elementName);
	}

	virtual bool isDouble(const std::string& elementName)
	{
		return _io.isDouble(elementName);
	}

	virtual std::vector<std::size_t> tensorDimensions(const std::string& elementName)
	{
		return _io.tensorDimensions(elementName);
	}

	virtual std::size_t numElements(const std::string& elementName)
	{
		return _io.arraySize(elementName);
	}

protected:
	reader_t _io;
};

/**
 * @brief Implements writing capabilities from a generic file writer class
 * @details Wraps a generic file writing class providing the IFileWriter interface.
 * @tparam writer_t Generic file writer class
 */
template <typename writer_t> class FileWriter : public cadet::io::IFileWriter
{
public:
	FileWriter()
	{
	}
	virtual ~FileWriter() CADET_NOEXCEPT
	{
	}

	virtual void writeTensorDouble(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								   const double* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		_io.template tensor<double>(dataSetName, rank, dims, buffer, stride, blockSize);
	}

	virtual void writeTensorInt(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								const int* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		_io.template tensor<int>(dataSetName, rank, dims, buffer, stride, blockSize);
	}

	virtual void writeTensorBool(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								 const bool* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		// Compute buffer size
		std::size_t bufSize = 1;
		for (std::size_t i = 0; i < rank; ++i)
		{
			bufSize *= dims[i];
		}

		// Convert buffer to int
		std::vector<int> bd(bufSize);
		std::size_t counter = 0;
		for (std::size_t i = 0; i < bufSize / blockSize; ++i)
			for (std::size_t j = 0; j < blockSize; ++j, ++counter)
				bd[counter] = (buffer[i * stride + j] ? 1 : 0);

		_io.template tensor<int>(dataSetName, rank, dims, bd.data(), 1, 1);
	}

	virtual void writeTensorString(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								   const std::string* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		_io.template tensor<std::string>(dataSetName, rank, dims, buffer, stride, blockSize);
	}

	virtual void writeMatrixDouble(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								   const double* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		_io.template matrix<double>(dataSetName, rows, cols, buffer, stride, blockSize);
	}

	virtual void writeMatrixInt(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								const int* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		_io.template matrix<int>(dataSetName, rows, cols, buffer, stride, blockSize);
	}

	virtual void writeMatrixBool(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								 const bool* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		// Compute buffer size
		const std::size_t bufSize = rows * cols;

		// Convert buffer to int
		std::vector<int> bd(bufSize);
		std::size_t counter = 0;
		for (std::size_t i = 0; i < bufSize / blockSize; ++i)
			for (std::size_t j = 0; j < blockSize; ++j, ++counter)
				bd[counter] = (buffer[i * stride + j] ? 1 : 0);

		_io.template matrix<int>(dataSetName, rows, cols, bd.data(), 1, 1);
	}

	virtual void writeMatrixString(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								   const std::string* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		_io.template matrix<std::string>(dataSetName, rows, cols, buffer, stride, blockSize);
	}

	virtual void writeVectorDouble(const std::string& dataSetName, const std::size_t length, const double* buffer,
								   const std::size_t stride, const std::size_t blockSize)
	{
		_io.template vector<double>(dataSetName, length, buffer, stride, blockSize);
	}

	virtual void writeVectorInt(const std::string& dataSetName, const std::size_t length, const int* buffer,
								const std::size_t stride, const std::size_t blockSize)
	{
		_io.template vector<int>(dataSetName, length, buffer, stride, blockSize);
	}

	virtual void writeVectorBool(const std::string& dataSetName, const std::size_t length, const bool* buffer,
								 const std::size_t stride, const std::size_t blockSize)
	{
		// Convert buffer to int
		std::vector<int> bd(length);
		std::size_t counter = 0;
		for (std::size_t i = 0; i < length / blockSize; ++i)
			for (std::size_t j = 0; j < blockSize; ++j, ++counter)
				bd[counter] = (buffer[i * stride + j] ? 1 : 0);

		_io.template vector<int>(dataSetName, length, bd.data(), 1, 1);
	}

	virtual void writeVectorString(const std::string& dataSetName, const std::size_t length, const std::string* buffer,
								   const std::size_t stride, const std::size_t blockSize)
	{
		_io.template vector<std::string>(dataSetName, length, buffer, stride, blockSize);
	}

	virtual void writeDouble(const std::string& dataSetName, const double buffer)
	{
		_io.template scalar<double>(dataSetName, buffer);
	}

	virtual void writeInt(const std::string& dataSetName, const int buffer)
	{
		_io.template scalar<int>(dataSetName, buffer);
	}

	virtual void writeBool(const std::string& dataSetName, const bool buffer)
	{
		_io.template scalar<int>(dataSetName, buffer);
	}

	virtual void writeString(const std::string& dataSetName, const std::string& buffer)
	{
		_io.template scalar<std::string>(dataSetName, buffer);
	}

	virtual void deleteGroup(const std::string& groupName)
	{
		_io.unlinkGroup(groupName);
	}

	virtual void deleteDataset(const std::string& dsName)
	{
		_io.unlinkDataset(dsName);
	}

protected:
	writer_t _io;
};

/**
 * @brief Implements file IO and navigation functions on top of FileReader or FileWriter
 * @tparam base_t FileReader or FileWriter class
 */
template <typename base_t> class BaseIOWrapper : public base_t
{
public:
	BaseIOWrapper()
	{
	}

	virtual void openFile(const std::string& fileName, const char* mode)
	{
		base_t::_io.openFile(fileName, mode);
	}

	virtual void closeFile()
	{
		base_t::_io.closeFile();
	}

	virtual void pushGroup(const std::string& scope)
	{
		base_t::_io.pushGroup(scope);
	}

	virtual void popGroup()
	{
		base_t::_io.popGroup();
	}

	virtual void setGroup(const std::string& scope)
	{
		base_t::_io.setGroup(scope);
	}

	virtual bool exists(const std::string& paramName)
	{
		return base_t::_io.exists(paramName);
	}

	virtual bool isGroup(const std::string& elementName)
	{
		return base_t::_io.isGroup(elementName);
	}

	virtual int numItems()
	{
		return base_t::_io.numItems();
	}

	virtual std::string itemName(int n)
	{
		return base_t::_io.itemName(n);
	}

	virtual std::vector<std::string> itemNames()
	{
		return base_t::_io.itemNames();
	}
};

/**
 * @brief Implements JSON IO functions on top of some other class
 * @tparam base_t Base class that is augmented with JSON IO functions
 */
template <typename base_t> class JSONBaseIOWrapper : public base_t
{
public:
	JSONBaseIOWrapper()
	{
	}

	virtual void pushGroup(const std::string& scope)
	{
		if (exists(scope))
			base_t::_opened.push(&base_t::_opened.top()->at(scope));
		else
		{
			json j;
			j["blubber"] = 0.0;
			j.erase("blubber");
			(*base_t::_opened.top())[scope] = j;
			base_t::_opened.push(&base_t::_opened.top()->at(scope));
		}
	}

	virtual void popGroup()
	{
		base_t::_opened.pop();
	}

	virtual void setGroup(const std::string& scope)
	{
		std::size_t start = 0;
		std::size_t end = 0;
		const std::string delimiter("/");

		// Quick return when called with empty group name
		if (scope.empty() || (scope == "/"))
		{
			base_t::_opened = std::stack<nlohmann::json*>();
			base_t::_opened.push(base_t::_root);
			return;
		}

		// Don't care for a preceding delimiter
		if (scope[0] == delimiter[0])
			++start;

		while (end != std::string::npos)
		{
			end = scope.find(delimiter, start);

			// If at end, use length = maxLength.  Else use length = end - start.
			pushGroup(scope.substr(start, (end == std::string::npos) ? std::string::npos : end - start));

			// If at end, use start = maxSize.  Else use start = end + delimiter.
			start = ((end > (std::string::npos - delimiter.size())) ? std::string::npos : end + delimiter.size());
		}
	}

	virtual bool exists(const std::string& paramName)
	{
		return base_t::_opened.top()->find(paramName) != base_t::_opened.top()->end();
	}

	virtual double getDouble(const std::string& paramName)
	{
		json& p = base_t::_opened.top()->at(paramName);
		if (p.is_array() && (p.size() == 1))
			p = p[0];

		return p.template get<double>();
	}

	virtual int getInt(const std::string& paramName)
	{
		json& p = base_t::_opened.top()->at(paramName);
		if (p.is_array() && (p.size() == 1))
			p = p[0];

		if (p.is_boolean())
			return p.template get<bool>();
		else
			return p.template get<int>();
	}

	virtual bool getBool(const std::string& paramName)
	{
		json& p = base_t::_opened.top()->at(paramName);
		if (p.is_array() && (p.size() == 1))
			p = p[0];

		if (p.is_number_integer())
			return p.template get<int>();
		else
			return p.template get<bool>();
	}

	virtual std::string getString(const std::string& paramName)
	{
		json& p = base_t::_opened.top()->at(paramName);
		if (p.is_array() && (p.size() == 1))
			p = p[0];

		return p.template get<std::string>();
	}

	virtual std::vector<double> getDoubleArray(const std::string& paramName)
	{
		const json& p = base_t::_opened.top()->at(paramName);
		if (!p.is_array())
			return std::vector<double>(1, p.template get<double>());
		else
			return p.template get<std::vector<double>>();
	}

	virtual std::vector<int> getIntArray(const std::string& paramName)
	{
		const json& p = base_t::_opened.top()->at(paramName);
		if (p.is_array())
		{
			if (p[0].is_boolean())
			{
				const std::vector<bool> d = p.template get<std::vector<bool>>();
				std::vector<int> bd(d.size());
				for (std::size_t i = 0; i < d.size(); ++i)
					bd[i] = d[i];

				return bd;
			}

			return p.template get<std::vector<int>>();
		}
		else
		{
			if (p.is_boolean())
				return std::vector<int>(1, p.template get<bool>());

			return std::vector<int>(1, p.template get<int>());
		}
	}

	virtual std::vector<bool> getBoolArray(const std::string& paramName)
	{
		const json& p = base_t::_opened.top()->at(paramName);
		if (p.is_array())
		{
			if (p[0].is_number_integer())
			{
				const std::vector<int> d = p.template get<std::vector<int>>();
				std::vector<bool> bd(d.size());
				for (std::size_t i = 0; i < d.size(); ++i)
					bd[i] = d[i];

				return bd;
			}

			return p.template get<std::vector<bool>>();
		}
		else
		{
			if (p.is_number_integer())
				return std::vector<bool>(1, p.template get<int>());

			return std::vector<bool>(1, p.template get<bool>());
		}
	}

	virtual std::vector<std::string> getStringArray(const std::string& paramName)
	{
		const json& p = base_t::_opened.top()->at(paramName);
		if (!p.is_array())
			return std::vector<std::string>(1, p.template get<std::string>());
		else
			return p.template get<std::vector<std::string>>();
	}

	virtual bool isArray(const std::string& elementName)
	{
		return base_t::_opened.top()->at(elementName).is_array();
	}

	virtual bool isString(const std::string& elementName)
	{
		const json& p = base_t::_opened.top()->at(elementName);
		if (p.is_array())
			return p[0].is_string();
		return p.is_string();
	}

	virtual bool isInt(const std::string& elementName)
	{
		const json& p = base_t::_opened.top()->at(elementName);
		if (p.is_array())
			return p[0].is_number_integer();
		return p.is_number_integer();
	}

	virtual bool isDouble(const std::string& elementName)
	{
		const json& p = base_t::_opened.top()->at(elementName);
		if (p.is_array())
			return p[0].is_number_float();
		return p.is_number_float();
	}

	virtual bool isGroup(const std::string& elementName)
	{
		return base_t::_opened.top()->at(elementName).is_object();
	}

	virtual int numItems()
	{
		return base_t::_opened.top()->size();
	}

	virtual std::string itemName(int n)
	{
		json::iterator it = base_t::_opened.top()->begin();
		std::advance(it, n);
		return it.key();
	}

	virtual std::vector<std::string> itemNames()
	{
		std::vector<std::string> names;
		names.reserve(base_t::_opened.top()->size());
		for (json::iterator it = base_t::_opened.top()->begin(); it != base_t::_opened.top()->end(); ++it)
			names.push_back(it.key());
		return names;
	}

	virtual std::vector<std::size_t> tensorDimensions(const std::string& elementName)
	{
		if (base_t::_opened.top()->find(elementName) == base_t::_opened.top()->end())
			return std::vector<std::size_t>();

		if (exists(elementName + "_rank") && exists(elementName + "_dims"))
		{
			return base_t::_opened.top()->at(elementName + "_dims").template get<std::vector<std::size_t>>();
		}
		else if (exists(elementName + "_cols") && exists(elementName + "_rows"))
		{
			const std::size_t cols = base_t::_opened.top()->at(elementName + "_cols").template get<std::size_t>();
			const std::size_t rows = base_t::_opened.top()->at(elementName + "_rows").template get<std::size_t>();
			return std::vector<std::size_t>({cols, rows});
		}

		const json p = base_t::_opened.top()->at(elementName);
		if (p.is_array())
		{
			if (p.is_number_float())
				return std::vector<std::size_t>({p.template get<std::vector<double>>().size()});
			else if (p.is_number_integer())
				return std::vector<std::size_t>({p.template get<std::vector<int>>().size()});
			else if (p.is_string())
				return std::vector<std::size_t>({p.template get<std::string>().size()});
			else if (p.is_boolean())
				return std::vector<std::size_t>({p.template get<std::vector<bool>>().size()});

			return std::vector<std::size_t>({0});
		}
		else
			return std::vector<std::size_t>();
	}

	virtual std::size_t numElements(const std::string& elementName)
	{
		if (base_t::_opened.top()->find(elementName) == base_t::_opened.top()->end())
			return 0;

		return base_t::_opened.top()->at(elementName).size();
	}

	virtual void writeTensorDouble(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								   const double* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		writeTensor<double, double>(dataSetName, rank, dims, buffer, stride, blockSize);
	}

	virtual void writeTensorInt(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								const int* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		writeTensor<int, int>(dataSetName, rank, dims, buffer, stride, blockSize);
	}

	virtual void writeTensorBool(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								 const bool* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		writeTensor<bool, int>(dataSetName, rank, dims, buffer, stride, blockSize);
	}

	virtual void writeTensorString(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								   const std::string* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		writeTensor<std::string, std::string>(dataSetName, rank, dims, buffer, stride, blockSize);
	}

	virtual void writeMatrixDouble(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								   const double* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		writeMatrix<double, double>(dataSetName, rows, cols, buffer, stride, blockSize);
	}

	virtual void writeMatrixInt(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								const int* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		writeMatrix<int, int>(dataSetName, rows, cols, buffer, stride, blockSize);
	}

	virtual void writeMatrixBool(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								 const bool* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		writeMatrix<bool, int>(dataSetName, rows, cols, buffer, stride, blockSize);
	}

	virtual void writeMatrixString(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								   const std::string* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		writeMatrix<std::string, std::string>(dataSetName, rows, cols, buffer, stride, blockSize);
	}

	virtual void writeVectorDouble(const std::string& dataSetName, const std::size_t length, const double* buffer,
								   const std::size_t stride, const std::size_t blockSize)
	{
		writeVector<double, double>(dataSetName, length, buffer, stride, blockSize);
	}

	virtual void writeVectorInt(const std::string& dataSetName, const std::size_t length, const int* buffer,
								const std::size_t stride, const std::size_t blockSize)
	{
		writeVector<int, int>(dataSetName, length, buffer, stride, blockSize);
	}

	virtual void writeVectorBool(const std::string& dataSetName, const std::size_t length, const bool* buffer,
								 const std::size_t stride, const std::size_t blockSize)
	{
		writeVector<bool, int>(dataSetName, length, buffer, stride, blockSize);
	}

	virtual void writeVectorString(const std::string& dataSetName, const std::size_t length, const std::string* buffer,
								   const std::size_t stride, const std::size_t blockSize)
	{
		writeVector<std::string, std::string>(dataSetName, length, buffer, stride, blockSize);
	}

	virtual void writeDouble(const std::string& dataSetName, const double buffer)
	{
		(*base_t::_opened.top())[dataSetName] = buffer;
	}

	virtual void writeInt(const std::string& dataSetName, const int buffer)
	{
		(*base_t::_opened.top())[dataSetName] = buffer;
	}

	virtual void writeBool(const std::string& dataSetName, const bool buffer)
	{
		(*base_t::_opened.top())[dataSetName] = buffer;
	}

	virtual void writeString(const std::string& dataSetName, const std::string& buffer)
	{
		(*base_t::_opened.top())[dataSetName] = buffer;
	}

	virtual void deleteGroup(const std::string& groupName)
	{
		base_t::_opened.top()->erase(groupName);
	}

	virtual void deleteDataset(const std::string& dsName)
	{
		base_t::_opened.top()->erase(dsName);
	}

	inline nlohmann::json* data()
	{
		return base_t::_root;
	}
	inline nlohmann::json const* data() const
	{
		return base_t::_root;
	}

protected:
	template <typename source_t, typename target_t>
	void writeVector(const std::string& dataSetName, const std::size_t length, const source_t* buffer,
					 const std::size_t stride, const std::size_t blockSize)
	{
		std::vector<target_t> d(length);
		for (std::size_t i = 0; i < length / blockSize; ++i)
		{
			for (std::size_t j = 0; j < blockSize; ++j)
				d[i * blockSize + j] = static_cast<target_t>(buffer[i * stride + j]);
		}

		(*base_t::_opened.top())[dataSetName] = d;
	}

	template <typename source_t, typename target_t>
	void writeMatrix(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
					 const source_t* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		std::vector<target_t> d(rows * cols);
		for (std::size_t i = 0; i < d.size() / blockSize; ++i)
		{
			for (std::size_t j = 0; j < blockSize; ++j)
				d[i * blockSize + j] = static_cast<target_t>(buffer[i * stride + j]);
		}

		(*base_t::_opened.top())[dataSetName] = d;
		(*base_t::_opened.top())[dataSetName + "_rows"] = rows;
		(*base_t::_opened.top())[dataSetName + "_cols"] = cols;
	}

	template <typename source_t, typename target_t>
	void writeTensor(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
					 const source_t* buffer, const std::size_t stride, const std::size_t blockSize)
	{
		// Handle scalars
		if (rank == 0)
		{
			(*base_t::_opened.top())[dataSetName] = static_cast<target_t>(buffer[0]);
			return;
		}

		// Handle vectors
		if (rank == 1)
		{
			std::vector<target_t> d(dims[0]);

			for (std::size_t i = 0; i < d.size() / blockSize; ++i)
			{
				for (std::size_t j = 0; j < blockSize; ++j)
					d[i * blockSize + j] = static_cast<target_t>(buffer[i * stride + j]);
			}
			(*base_t::_opened.top())[dataSetName] = d;
			return;
		}

		// Handle tensors starting from rank 2 (matrices)
		std::vector<std::size_t> dimsV(rank);
		std::size_t bufSize = 1;
		for (std::size_t i = 0; i < rank; ++i)
		{
			dimsV[i] = dims[i];
			bufSize *= dims[i];
		}

		std::vector<target_t> d(bufSize);
		for (std::size_t i = 0; i < d.size() / blockSize; ++i)
		{
			for (std::size_t j = 0; j < blockSize; ++j)
				d[i * blockSize + j] = static_cast<target_t>(buffer[i * stride + j]);
		}

		(*base_t::_opened.top())[dataSetName] = d;
		(*base_t::_opened.top())[dataSetName + "_rank"] = rank;
		(*base_t::_opened.top())[dataSetName + "_dims"] = dimsV;
	}
};

/**
 * @brief Root class for implementing IMemoryIO using JSON
 * @details Serves as root class for JSONBaseIOWrapper.
 */
class JSONMemoryReaderWriterImpl : public cadet::io::IMemoryIO
{
public:
	JSONMemoryReaderWriterImpl() : _root(new json())
	{
		_opened.push(_root);
	}

	virtual ~JSONMemoryReaderWriterImpl() CADET_NOEXCEPT
	{
		delete _root;
	}

protected:
	nlohmann::json* _root;
	std::stack<nlohmann::json*> _opened;
};

/**
 * @brief Root class for JSON FileReader and FileWriter implementations
 * @details Serves as root class for JSONBaseIOWrapper.
 * @tparam iface_t Interface to implement, either IFileReader or IFileWriter
 */
template <class iface_t> class JSONFileIOProxy : public iface_t
{
public:
	JSONFileIOProxy() : _writeBack(false), _root(new json())
	{
		_opened.push(_root);
	}

	virtual ~JSONFileIOProxy() CADET_NOEXCEPT
	{
		delete _root;
	}

	virtual void openFile(const std::string& fileName, const char* mode)
	{
		const std::string sm(mode);
		if (sm == "r")
		{
			_writeBack = false;

			_fs.open(fileName.c_str(), std::ios_base::in);
			_fs >> (*_root);
		}
		else if (sm == "rw")
		{
			_writeBack = true;

			_fs.open(fileName.c_str(), std::ios_base::in);
			_fs >> (*_root);
			_fs.close();

			_fs.open(fileName.c_str(), std::ios_base::out | std::ios_base::trunc);
		}
		else if ((sm == "c") || (sm == "co"))
		{
			_writeBack = true;
			_fs.open(fileName.c_str(), std::ios_base::trunc | std::ios_base::out);
		}
	}

	virtual void closeFile()
	{
		if (_writeBack)
			_fs << std::setw(4) << (*_root);

		_fs.close();
	}

protected:
	bool _writeBack;
	std::fstream _fs;
	nlohmann::json* _root;
	std::stack<nlohmann::json*> _opened;
};

} // namespace

namespace cadet
{
namespace io
{

template <typename io_t> using FileReaderImpl = BaseIOWrapper<FileReader<io_t>>;
template <typename io_t> using FileWriterImpl = BaseIOWrapper<FileWriter<io_t>>;

typedef JSONBaseIOWrapper<JSONMemoryReaderWriterImpl> JSONMemoryReaderWriter;
typedef JSONBaseIOWrapper<JSONFileIOProxy<cadet::io::IFileReader>> JSONFileReader;
typedef JSONBaseIOWrapper<JSONFileIOProxy<cadet::io::IFileWriter>> JSONFileWriter;

IMemoryIO* createMemoryReaderWriter()
{
	return new JSONMemoryReaderWriter();
}

IFileReader* createReader(const std::string& fileExt)
{
	if (cadet::util::caseInsensitiveEquals(fileExt, "h5"))
	{
		return new FileReaderImpl<cadet::io::HDF5Reader>();
	}
	else if (cadet::util::caseInsensitiveEquals(fileExt, "xml"))
	{
		return new FileReaderImpl<cadet::io::XMLReader>();
	}
	else if (cadet::util::caseInsensitiveEquals(fileExt, "json"))
	{
		return new JSONFileReader();
	}
	return nullptr;
}

IFileWriter* createWriter(const std::string& fileExt)
{
	if (cadet::util::caseInsensitiveEquals(fileExt, "h5"))
	{
		return new FileWriterImpl<cadet::io::HDF5Writer>();
	}
	else if (cadet::util::caseInsensitiveEquals(fileExt, "xml"))
	{
		return new FileWriterImpl<cadet::io::XMLWriter>();
	}
	else if (cadet::util::caseInsensitiveEquals(fileExt, "json"))
	{
		return new JSONFileWriter();
	}
	return nullptr;
}

} // namespace io
} // namespace cadet
