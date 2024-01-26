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

#ifndef XMLWRITER_HPP_
#define XMLWRITER_HPP_

#include <iomanip>

#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"
#include "XMLBase.hpp"

namespace cadet
{

namespace io
{

using namespace pugi;

class XMLWriter : public XMLBase
{
public:
	/// \brief Constructor
	XMLWriter();

	/// \brief Destructor
	~XMLWriter() CADET_NOEXCEPT;

	/// \brief Write data from C-array to a dataset
	template <typename T>
	void write(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims, const T* buffer);

	/// \brief Write data from C-array to a dataset
	template <typename T>
	void write(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims, const T* buffer,
			   const std::size_t stride, const std::size_t blockSize);

	/// \brief Convenience wrapper for writing tensors from C-array
	template <typename T>
	void tensor(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims, const T* buffer,
				const std::size_t stride = 1, const std::size_t blockSize = 1);

	/// \brief Convenience wrapper for writing tensors from std::vector
	template <typename T>
	void tensor(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
				const std::vector<T>& buffer, const std::size_t stride = 1, const std::size_t blockSize = 1);

	/// \brief Convenience wrapper for writing matrices from C-array
	template <typename T>
	void matrix(const std::string& dataSetName, const std::size_t rows, const std::size_t cols, const T* buffer,
				const std::size_t stride = 1, const std::size_t blockSize = 1);

	/// \brief Convenience wrapper for writing matrices from std::vector
	template <typename T>
	void matrix(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
				const std::vector<T>& buffer, const std::size_t stride = 1, const std::size_t blockSize = 1);

	/// \brief Convenience wrapper for writing vectors from C-array
	template <typename T>
	void vector(const std::string& dataSetName, const std::size_t length, const T* buffer, const std::size_t stride = 1,
				const std::size_t blockSize = 1);

	/// \brief Convenience wrapper for writing vectors from std::vector
	template <typename T>
	void vector(const std::string& dataSetName, const std::vector<T>& buffer, const std::size_t stride = 1,
				const std::size_t blockSize = 1);

	/// \brief Convenience wrapper for writing scalars
	template <typename T> void scalar(const std::string& dataSetName, const T buffer);

	/// \brief This functionality is not supported by XML - this is a stub.
	///        Removes an existing group from the file.
	inline void unlinkGroup(const std::string& groupName)
	{
	}

	/// \brief This functionality is not supported by XML - this is a stub.
	///        Removes an existing dataset from the current group.
	inline void unlinkDataset(const std::string& dsName)
	{
	}

	/// \brief This functionality is not supported by XML - this is a stub.
	///        Set the level of compression used for tensors of 2nd order and above
	inline void compressFields(bool setCompression)
	{
	}

	/// \brief This functionality is not supported by XML - this is a stub.
	///        Tensors of 2nd order (vectors) and above are written as extendible fields
	///        (maxsize = unlimited, chunked layout), when set to true.
	inline void extendibleFields(bool setExtendible)
	{
	}

private:
	std::string _typeName; //!< Name of the type to be written

	template <typename T>
	void writeWork(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims, const T* buffer,
				   const std::size_t stride, const std::size_t blockSize);
};

XMLWriter::XMLWriter()
{
}

XMLWriter::~XMLWriter() CADET_NOEXCEPT
{
}

// ============================================================================================================
//   Template specializations of member function write() for diffenet data types
// ============================================================================================================

template <>
void XMLWriter::write<double>(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
							  const double* buffer, const std::size_t stride, const std::size_t blockSize)
{
	_typeName = _typeDouble;
	writeWork<double>(dataSetName, rank, dims, buffer, stride, blockSize);
}

template <>
void XMLWriter::write<int>(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
						   const int* buffer, const std::size_t stride, const std::size_t blockSize)
{
	_typeName = _typeUint64;
	writeWork<int>(dataSetName, rank, dims, buffer, stride, blockSize);
}

template <>
void XMLWriter::write<uint64_t>(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								const uint64_t* buffer, const std::size_t stride, const std::size_t blockSize)
{
	_typeName = _typeInt;
	writeWork<uint64_t>(dataSetName, rank, dims, buffer, stride, blockSize);
}

// template <>
// void XMLWriter::write<bool>(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims, const
// bool* buffer)
//{
//     _typeName = _typeBool;
//     writeWork<bool>(dataSetName, rank, dims, buffer);
// }

template <>
void XMLWriter::write<std::string>(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
								   const std::string* buffer, const std::size_t stride, const std::size_t blockSize)
{
	// Compute bufSize and create charBuf
	std::size_t bufSize = 1;
	for (std::size_t i = 0; i < rank; ++i)
		bufSize *= dims[i];

	// Create a contiguous array of pointers
	char const** charBuf = new char const*[bufSize];

	// Loop over blocks
	std::size_t counter = 0;
	for (std::size_t i = 0; i < bufSize / blockSize; ++i)
	{
		for (std::size_t j = 0; j < blockSize; ++j, ++counter)
			charBuf[counter] = buffer[i * stride + j].c_str();
	}

	_typeName = _typeChar;
	writeWork<char_t const*>(dataSetName, rank, dims, charBuf, 1, 1);

	delete[] charBuf;
}

// Template that matches on every unsupported type and throws an exception
template <typename T>
void XMLWriter::write(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims, const T* buffer)
{
	throw IOException("You may not try to write an unsupported type");
}

template <typename T>
void XMLWriter::write(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims, const T* buffer,
					  const std::size_t stride, const std::size_t blockSize)
{
	throw IOException("You may not try to write an unsupported type");
}
// ============================================================================================================

// ============================================================================================================
//   Convenience wrappers
// ============================================================================================================
template <typename T>
void XMLWriter::tensor(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
					   const std::vector<T>& buffer, const std::size_t stride, const std::size_t blockSize)
{
#ifdef CADET_DEBUG
	std::size_t bufSize = 1;
	for (std::size_t i = 0; i < rank; ++i)
		bufSize *= dims[i];
	cadet_assert(bufSize <= buffer.size());
#endif
	write<T>(dataSetName, rank, dims, buffer.data(), stride, blockSize);
}

template <typename T>
void XMLWriter::tensor(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims, const T* buffer,
					   const std::size_t stride, const std::size_t blockSize)
{
	write<T>(dataSetName, rank, dims, buffer, stride, blockSize);
}

template <typename T>
void XMLWriter::matrix(const std::string& dataSetName, const std::size_t rows, const std::size_t cols, const T* buffer,
					   const std::size_t stride, const std::size_t blockSize)
{
	std::size_t dims[2] = {rows, cols};
	write<T>(dataSetName, 2, dims, buffer, stride, blockSize);
}

template <typename T>
void XMLWriter::matrix(const std::string& dataSetName, const std::size_t rows, const std::size_t cols,
					   const std::vector<T>& buffer, const std::size_t stride, const std::size_t blockSize)
{
	cadet_assert(rows * cols <= buffer.size());
	std::size_t dims[2] = {rows, cols};
	write<T>(dataSetName, 2, dims, buffer.data(), stride, blockSize);
}

template <typename T>
void XMLWriter::vector(const std::string& dataSetName, const std::size_t length, const T* buffer,
					   const std::size_t stride, const std::size_t blockSize)
{
	write<T>(dataSetName, 1, &length, buffer, stride, blockSize);
}

template <typename T>
void XMLWriter::vector(const std::string& dataSetName, const std::vector<T>& buffer, const std::size_t stride,
					   const std::size_t blockSize)
{
	std::size_t length = buffer.size() / stride;
	write<T>(dataSetName, 1, &length, buffer.data(), stride, blockSize);
}

template <typename T> void XMLWriter::scalar(const std::string& dataSetName, const T buffer)
{
	vector<T>(dataSetName, 1, &buffer);
}
// ============================================================================================================

// ============================================================================================================
//   Private members
// ============================================================================================================

// This method can only work on template parameters of type: const char_t*, int, unsigned int, double and bool!
template <typename T>
void XMLWriter::writeWork(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
						  const T* buffer, const std::size_t stride, const std::size_t blockSize)
{
	openGroup(true);

	// Detect scalars
	const bool isScalar = (rank == 0);

	// Create the dims-string
	std::ostringstream dims_str;
	std::size_t bufSize = 1;

	if (isScalar)
	{
		dims_str << 0;
	}
	else
	{
		for (std::size_t i = 0; i < rank - 1; ++i)
		{
			dims_str << dims[i] << _dimsSeparator;
			bufSize *= dims[i];
		}
		dims_str << dims[rank - 1];
		bufSize *= dims[rank - 1];
	}

	// Create the text-string
	std::ostringstream text_str;
	text_str << std::setprecision(16);
	for (std::size_t i = 0; i < (bufSize - 1) / blockSize; ++i)
		for (std::size_t j = 0; j < blockSize; ++j)
			text_str << buffer[i * stride + j] << _textSeparator;

	for (std::size_t j = 0; j < blockSize - 1; ++j)
		text_str << buffer[(bufSize - 1) * stride + j] << _textSeparator;
	text_str << buffer[(bufSize - 1) * stride + blockSize - 1];

	xml_node dataset =
		_groupOpened.node().find_child_by_attribute(_nodeDset.c_str(), _attrName.c_str(), dataSetName.c_str());
	if (dataset)
	{
		dataset.attribute(_attrType.c_str()) = _typeName.c_str();
		dataset.attribute(_attrRank.c_str()) = unsigned(rank);
		dataset.attribute(_attrDims.c_str()) = dims_str.str().c_str();
		dataset.text() = text_str.str().c_str();
	}
	else
	{
		dataset = _groupOpened.node().append_child(_nodeDset.c_str());

		dataset.append_attribute(_attrName.c_str()) = dataSetName.c_str();
		dataset.append_attribute(_attrType.c_str()) = _typeName.c_str();
		dataset.append_attribute(_attrRank.c_str()) = unsigned(rank);
		dataset.append_attribute(_attrDims.c_str()) = dims_str.str().c_str();
		dataset.text() = text_str.str().c_str();
	}

	closeGroup();
}

} // namespace io

} // namespace cadet

#endif /* XMLWRITER_HPP_ */
