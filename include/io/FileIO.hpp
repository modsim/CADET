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

#ifndef LIBCADET_FILEIO_HPP_
#define LIBCADET_FILEIO_HPP_

#include <string>
#include <vector>

#include "cadet/cadetCompilerInfo.hpp"

namespace cadet
{

namespace io
{

class IReaderCapabilities
{
public:
	IReaderCapabilities()
	{
	}
	virtual ~IReaderCapabilities() CADET_NOEXCEPT
	{
	}

	virtual double getDouble(const std::string& paramName) = 0;
	virtual int getInt(const std::string& paramName) = 0;
	virtual bool getBool(const std::string& paramName) = 0;
	virtual std::string getString(const std::string& paramName) = 0;

	virtual std::vector<double> getDoubleArray(const std::string& paramName) = 0;
	virtual std::vector<int> getIntArray(const std::string& paramName) = 0;
	virtual std::vector<bool> getBoolArray(const std::string& paramName) = 0;
	virtual std::vector<std::string> getStringArray(const std::string& paramName) = 0;

	virtual std::vector<std::size_t> tensorDimensions(const std::string& elementName) = 0;
	virtual std::size_t numElements(const std::string& elementName) = 0;

	virtual bool isArray(const std::string& paramName) = 0;
	virtual bool isString(const std::string& elementName) = 0;
	virtual bool isInt(const std::string& elementName) = 0;
	virtual bool isDouble(const std::string& elementName) = 0;
};

class IWriterCapabilities
{
public:
	IWriterCapabilities()
	{
	}
	virtual ~IWriterCapabilities() CADET_NOEXCEPT
	{
	}

	virtual void writeTensorDouble(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								   const double* buffer, const std::size_t stride, const std::size_t blockSize) = 0;
	virtual void writeTensorInt(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								const int* buffer, const std::size_t stride, const std::size_t blockSize) = 0;
	virtual void writeTensorBool(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								 const bool* buffer, const std::size_t stride, const std::size_t blockSize) = 0;
	virtual void writeTensorString(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								   const std::string* buffer, const std::size_t stride,
								   const std::size_t blockSize) = 0;

	virtual void writeMatrixDouble(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								   const double* buffer, const std::size_t stride, const std::size_t blockSize) = 0;
	virtual void writeMatrixInt(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								const int* buffer, const std::size_t stride, const std::size_t blockSize) = 0;
	virtual void writeMatrixBool(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								 const bool* buffer, const std::size_t stride, const std::size_t blockSize) = 0;
	virtual void writeMatrixString(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
								   const std::string* buffer, const std::size_t stride,
								   const std::size_t blockSize) = 0;

	virtual void writeVectorDouble(const std::string& dataSetName, const std::size_t length, const double* buffer,
								   const std::size_t stride, const std::size_t blockSize) = 0;
	virtual void writeVectorInt(const std::string& dataSetName, const std::size_t length, const int* buffer,
								const std::size_t stride, const std::size_t blockSize) = 0;
	virtual void writeVectorBool(const std::string& dataSetName, const std::size_t length, const bool* buffer,
								 const std::size_t stride, const std::size_t blockSize) = 0;
	virtual void writeVectorString(const std::string& dataSetName, const std::size_t length, const std::string* buffer,
								   const std::size_t stride, const std::size_t blockSize) = 0;

	virtual void writeDouble(const std::string& dataSetName, const double buffer) = 0;
	virtual void writeInt(const std::string& dataSetName, const int buffer) = 0;
	virtual void writeBool(const std::string& dataSetName, const bool buffer) = 0;
	virtual void writeString(const std::string& dataSetName, const std::string& buffer) = 0;

	virtual void deleteGroup(const std::string& groupName) = 0;
	virtual void deleteDataset(const std::string& dsName) = 0;
};

class INavigationCapabilities
{
public:
	INavigationCapabilities()
	{
	}
	virtual ~INavigationCapabilities() CADET_NOEXCEPT
	{
	}

	virtual void pushGroup(const std::string& scope) = 0;
	virtual void popGroup() = 0;
	virtual void setGroup(const std::string& scope) = 0;
	virtual bool exists(const std::string& paramName) = 0;

	virtual bool isGroup(const std::string& elementName) = 0;
	virtual int numItems() = 0;
	virtual std::string itemName(int n) = 0;
	virtual std::vector<std::string> itemNames() = 0;
};

class IMemoryIO : public INavigationCapabilities, public IReaderCapabilities, public IWriterCapabilities
{
};

class IFileIO
{
public:
	IFileIO()
	{
	}
	virtual ~IFileIO() CADET_NOEXCEPT
	{
	}

	virtual void openFile(const std::string& fileName, const char* mode) = 0;
	virtual void closeFile() = 0;
};

class IReader : public INavigationCapabilities, public IReaderCapabilities
{
};
class IWriter : public INavigationCapabilities, public IWriterCapabilities
{
};
class IFileReader : public IFileIO, public IReader
{
};
class IFileWriter : public IFileIO, public IWriter
{
};

IMemoryIO* createMemoryReaderWriter();
IFileReader* createReader(const std::string& fileExt);
IFileWriter* createWriter(const std::string& fileExt);

} // namespace io
} // namespace cadet

#endif /* LIBCADET_FILEIO_HPP_ */
