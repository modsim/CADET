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
 * Provides implementation of a Matlab reader / writer operating on Matlab structs
 */

#ifndef CADET_MEX_MATLABREADERWRITER_HPP_
#define CADET_MEX_MATLABREADERWRITER_HPP_

#include <mex.h>

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"

#include "MatlabException.hpp"

namespace cadet
{

namespace mex
{

/**
 * @brief Provides means to read from and write to Matlab structs
 * @details The cadet::io::HDF5Reader / cadet::io::HDF5Writer counterpart for Matlab structs.
 */
class MatlabReaderWriter
{
public:
	/// \brief Constructor
	MatlabReaderWriter(mxArray** data) : _groupName(""), _root(data), _group(0), _dataSet(0)
	{
	}
	MatlabReaderWriter() : _groupName(""), _root(0), _group(0), _dataSet(0)
	{
	}

	MatlabReaderWriter(const MatlabReaderWriter& cpy)
		: _groupName(""), _root(cpy._root), _group(cpy._group), _dataSet(cpy._dataSet)
	{
	}

	/// \brief Destructor
	~MatlabReaderWriter() CADET_NOEXCEPT
	{
	}

	/// \brief Open an HDF5 file
	inline void openFile(const std::string& fileName, const std::string& mode = "r")
	{
	}
	inline void openFile(const char* fileName, const std::string& mode = "r")
	{
	}

	/// \brief Close the currently opened file
	inline void closeFile()
	{
	}

	/// \brief Set a group to be [read from/written to] in all subsequent calls to [read/write] methods
	inline void setGroup(const std::string& groupName)
	{
		_groupName = groupName;
	}

	/// \brief Checks if the given dataset or group exists in the file
	inline bool exists(const std::string& elementName)
	{
		return exists(elementName.c_str());
	}
	inline bool exists(const char* elementName);

	/// \brief Checks if the given dataset is a vector (i.e., has more than one value)
	inline bool isVector(const std::string& elementName)
	{
		return isVector(elementName.c_str());
	}
	inline bool isVector(const char* elementName);

	/// \brief Returns the number of elements in the array identified by name
	inline std::size_t arraySize(const std::string& elementName)
	{
		return arraySize(elementName.c_str());
	}
	inline std::size_t arraySize(const char* elementName);

	/// \brief Convenience wrapper for reading vectors
	template <typename T> std::vector<T> vector(const std::string& dataSetName);

	/// \brief Convenience wrapper for reading scalars
	template <typename T> T scalar(const std::string& dataSetName, std::size_t position = 0);

	/// \brief Write data from C-array to a dataset
	template <typename T>
	void write(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims, const T* buffer,
			   const std::size_t stride = 1, const std::size_t blockSize = 1);

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

	/// \brief Removes an existing group from the file
	inline void unlinkGroup(const std::string& groupName);

	/// \brief Removes an existing dataset from the current group
	inline void unlinkDataset(const std::string& dsName);

	/// \brief Enable/disable compression for tensors of 2nd order and above
	inline void compressFields(bool setCompression)
	{
	}

	/// \brief Tensors of 2nd order (vectors) and above are written as extendible fields
	///        (maxsize = unlimited, chunked layout), when set to true.
	inline void extendibleFields(bool setExtendible)
	{
	}

	inline void pushGroup(const std::string& groupName);
	inline void popGroup();

protected:
	std::string _groupName;
	mxArray** _root;
	mxArray* _group;
	mxArray* _dataSet;

	template <typename T> std::vector<T> read();

	mxArray* createStructField(const std::string& dataSetName, const std::size_t rank, const std::size_t* dims,
							   const void* buffer, const std::size_t stride, const mxClassID type);

	void openGroup(bool create = false);
	void openAndCreateGroup();

	bool checkForVector() const
	{
		// Validate the current item (expect column or row vector)
		const mwSize numDims = mxGetNumberOfDimensions(_dataSet);
		if (numDims > 2)
		{
			// Error
			std::ostringstream str;
			str << "CadetMex: Trying to read vector from field '" << _groupName << "', but got tensor rank " << numDims
				<< ".\n";
			throw MatlabException(str.str());
		}

		const mwSize* dims = mxGetDimensions(_dataSet);
		if ((dims[0] > 1) && (dims[1] > 1))
		{
			// Error
			std::ostringstream str;
			str << "CadetMex: Trying to read vector from field '" << _groupName << "', but got " << dims[0] << " x "
				<< dims[1] << " matrix.\n";
			throw MatlabException(str.str());
		}
		return true;
	}
};

bool MatlabReaderWriter::exists(const char* elementName)
{
	openGroup();
	return mxGetField(_group, 0, elementName);
}

bool MatlabReaderWriter::isVector(const char* elementName)
{
	openGroup();
	return mxGetNumberOfElements(mxGetField(_group, 0, elementName)) > 1;
}

std::size_t MatlabReaderWriter::arraySize(const char* elementName)
{
	openGroup();
	return mxGetNumberOfElements(mxGetField(_group, 0, elementName));
}

void MatlabReaderWriter::pushGroup(const std::string& groupName)
{
	if (_groupName.empty())
	{
		_groupName = "/" + groupName;
		return;
	}

	if (_groupName.back() == '/')
		_groupName += groupName;
	else
		_groupName += "/" + groupName;
}

void MatlabReaderWriter::popGroup()
{
	if (_groupName.empty() || (_groupName == "/"))
		return;

	std::size_t lastIdx = std::string::npos;
	if (_groupName.back() == '/')
		lastIdx = _groupName.length() - 2;

	const std::size_t idx = _groupName.find_last_of('/', lastIdx);
	_groupName.erase(idx);
}

void MatlabReaderWriter::openGroup(bool create)
{
	_group = *_root;
	if (create)
		return openAndCreateGroup();

	// Early out if empty group or root
	if (_groupName.empty() || (_groupName == "/"))
		return;

	// Ignore root slash if present
	std::size_t start = 0;
	if (_groupName[0] == '/')
		start = 1;

	std::size_t end = 0;
	std::string currentPath;
	while (end != std::string::npos)
	{
		end = _groupName.find("/", start);

		// If at end, use length = maxLength.  Else use length = end - start.
		std::string itemName = _groupName.substr(start, (end == std::string::npos) ? std::string::npos : end - start);

		// If at end, use start = maxSize.  Else use start = end + delimiter.
		start = ((end > (std::string::npos - 1)) ? std::string::npos : end + 1);

		// Check if current item is a struct
		if (!mxIsStruct(_group))
		{
			// Error
			std::ostringstream str;
			str << "CadetMex: The element '" << currentPath << "' is not a struct (requested field '" << _groupName
				<< "').\n";
			throw MatlabException(str.str());
		}

		_group = mxGetField(_group, 0, itemName.c_str());

		// Check if field exists
		if (!_group)
		{
			// Error
			std::ostringstream str;
			str << "CadetMex: The struct '" << currentPath << "' does not contain the field '" << itemName
				<< "' (requested field '" << _groupName << "').\n";
			throw MatlabException(str.str());
		}

		if (!currentPath.empty())
			currentPath += "." + itemName;
		else
			currentPath += itemName;
	}
}

void MatlabReaderWriter::openAndCreateGroup()
{
	_group = *_root;

	// Check if root is a struct
	if (!_group || !mxIsStruct(_group))
	{
		_group = mxCreateStructMatrix(1, 1, 0, NULL);
		*_root = _group;
	}

	// Early out if empty group
	if (_groupName.empty())
		return;

	// Ignore root slash if present
	std::size_t start = 0;
	if (_groupName[0] == '/')
		start = 1;

	std::size_t end = 0;
	std::string currentPath;
	while (end != std::string::npos)
	{
		end = _groupName.find("/", start);

		// If at end, use length = maxLength.  Else use length = end - start.
		std::string itemName = _groupName.substr(start, (end == std::string::npos) ? std::string::npos : end - start);

		// If at end, use start = maxSize.  Else use start = end + delimiter.
		start = ((end > (std::string::npos - 1)) ? std::string::npos : end + 1);

		mxArray* nextGroup = mxGetField(_group, 0, itemName.c_str());

		// Check if field exists
		if (!nextGroup)
		{
			// Create the field and insert a struct
			mwIndex index = mxAddField(_group, itemName.c_str());
			mxSetFieldByNumber(_group, 0, index, mxCreateStructMatrix(1, 1, 0, NULL));
			nextGroup = mxGetFieldByNumber(_group, 0, index);
		}

		_group = nextGroup;

		if (!currentPath.empty())
			currentPath += "." + itemName;
		else
			currentPath += itemName;
	}
}
// ============================================================================================================
// Reading
// ============================================================================================================

// Double specialization of vector()
template <> std::vector<double> MatlabReaderWriter::vector<double>(const std::string& dataSetName)
{
	openGroup();

	// Validate data type
	_dataSet = mxGetField(_group, 0, dataSetName.c_str());
	if (!_dataSet)
	{
		// Error
		std::ostringstream str;
		str << "CadetMex: Trying to read vector from non-existent field '" << _groupName << "/" << dataSetName
			<< "'.\n";
		throw MatlabException(str.str());
	}
	if (!mxIsDouble(_dataSet))
	{
		// Error
		std::ostringstream str;
		str << "CadetMex: Trying to read vector of doubles from field '" << _groupName << "/" << dataSetName
			<< "', but got non-double values.\n";
		throw MatlabException(str.str());
	}

	return read<double>();
}

// Integer specialization of vector()
template <> std::vector<int> MatlabReaderWriter::vector<int>(const std::string& dataSetName)
{
	openGroup();

	// Validate data type
	_dataSet = mxGetField(_group, 0, dataSetName.c_str());
	if (!_dataSet)
	{
		// Error
		std::ostringstream str;
		str << "CadetMex: Trying to read vector from non-existent field '" << _groupName << "/" << dataSetName
			<< "'.\n";
		throw MatlabException(str.str());
	}
	if (!mxIsInt32(_dataSet))
	{
		// Error
		std::ostringstream str;
		str << "CadetMex: Trying to read vector of int32 from field '" << _groupName << "/" << dataSetName
			<< "', but got non-int32 values.\n";
		throw MatlabException(str.str());
	}

	return read<int>();
}

// std::string specialization of vector()
template <> std::vector<std::string> MatlabReaderWriter::vector<std::string>(const std::string& dataSetName)
{
	openGroup();

	// Validate
	_dataSet = mxGetField(_group, 0, dataSetName.c_str());
	if (!_dataSet)
	{
		// Error
		std::ostringstream str;
		str << "CadetMex: Trying to read cell array from non-existent field '" << _groupName << "/" << dataSetName
			<< "'.\n";
		throw MatlabException(str.str());
	}
	if (!mxIsCell(_dataSet) && !mxIsChar(_dataSet))
	{
		// Error
		std::ostringstream str;
		str << "CadetMex: Trying to read cell array of strings from field '" << _groupName << "/" << dataSetName
			<< "', but got non-cell array.\n";
		throw MatlabException(str.str());
	}

	// Check for single string
	if (mxIsChar(_dataSet))
	{
		std::vector<std::string> data(1);

		const char* const strData = mxArrayToString(_dataSet);
		data[0] = strData;
		mxFree(const_cast<char*>(strData));

		return data;
	}

	// We have a cell array
	const mwSize numElems = mxGetNumberOfElements(_dataSet);
	std::vector<std::string> stringVector(numElems);

	for (mwIndex i = 0; i < numElems; ++i)
	{
		mxArray* curCell = mxGetCell(_dataSet, i);
		if (!mxIsChar(curCell))
		{
			// Error
			std::ostringstream str;
			str << "CadetMex: Expected string in element " << (i + 1) << " of cell array " << _groupName << "/"
				<< dataSetName << "', but got non-string data.\n";
			throw MatlabException(str.str());
		}

		const char* const strData = mxArrayToString(curCell);
		stringVector[i] = strData;
		mxFree(const_cast<char*>(strData));
	}

	return stringVector;
}

// Template that matches on every unsupported type
template <typename T> std::vector<T> MatlabReaderWriter::vector(const std::string& dataSetName)
{
	std::ostringstream str;
	str << "CadetMex: Trying to read vector of unsupported type from field '" << _groupName << "/" << dataSetName
		<< "'.\n";
	throw MatlabException(str.str());
}
// ============================================================================================================

template <typename T> T MatlabReaderWriter::scalar(const std::string& dataSetName, size_t position)
{
	return this->template vector<T>(dataSetName).at(position);
}

template <typename T> std::vector<T> MatlabReaderWriter::read()
{
	checkForVector();

	const mwSize numElems = mxGetNumberOfElements(_dataSet);

	// Copy data
	if (mxIsDouble(_dataSet))
	{
		double const* const src = mxGetPr(_dataSet);
		return std::vector<T>(src, src + numElems);
	}
	else if (mxIsInt32(_dataSet))
	{
		int const* const src = static_cast<int const* const>(mxGetData(_dataSet));
		return std::vector<T>(src, src + numElems);
	}
	else
	{
		// Unsupported data type
		std::ostringstream str;
		str << "CadetMex: Trying to read vector of unsupported type from struct '" << _groupName << "'.\n";
		throw MatlabException(str.str());
	}
}

// ============================================================================================================
// Writing
// ============================================================================================================

mxArray* MatlabReaderWriter::createStructField(const std::string& dataSetName, const size_t rank, const size_t* dims,
											   const void* buffer, const size_t stride, const mxClassID type)
{
	openAndCreateGroup();
	//	mexPrintf("Writing dataset %s.%s of rank %u\n", _groupName.c_str(), dataSetName.c_str(), rank);

	// Create Matlab array
	mwSize* dimsMatlab = static_cast<mwSize*>(mxMalloc(rank * sizeof(mwIndex)));
	for (size_t i = 0; i < rank; ++i)
		dimsMatlab[i] = dims[i];

	mxArray* const data = mxCreateUninitNumericArray(rank, dimsMatlab, type, mxREAL);

	mxFree(dimsMatlab);

	// Assign it to the field
	mxSetFieldByNumber(_group, 0, mxAddField(_group, dataSetName.c_str()), data);

	return data;
}

template <>
void MatlabReaderWriter::write<double>(const std::string& dataSetName, const size_t rank, const size_t* dims,
									   const double* buffer, const size_t stride, const size_t blockSize)
{
	mxArray* matData = createStructField(dataSetName, rank, dims, buffer, stride, mxDOUBLE_CLASS);

	size_t numEl = mxGetNumberOfElements(matData);
	double* data = mxGetPr(matData);

	// Write data to matlab
	for (size_t i = 0; i < numEl / blockSize; ++i)
	{
		for (size_t j = 0; j < blockSize; ++j, ++data)
			*data = buffer[i * stride + j];
	}
}

template <>
void MatlabReaderWriter::write<int>(const std::string& dataSetName, const size_t rank, const size_t* dims,
									const int* buffer, const size_t stride, const size_t blockSize)
{
	mxArray* matData = createStructField(dataSetName, rank, dims, buffer, stride, mxINT32_CLASS);

	size_t numEl = mxGetNumberOfElements(matData);
	int* data = static_cast<int*>(mxGetData(matData));

	// Write data to matlab
	for (size_t i = 0; i < numEl / blockSize; ++i)
	{
		for (size_t j = 0; j < blockSize; ++j, ++data)
			*data = buffer[i * stride + j];
	}
}

template <>
void MatlabReaderWriter::write<std::string>(const std::string& dataSetName, const size_t rank, const size_t* dims,
											const std::string* buffer, const size_t stride, const size_t blockSize)
{
	openAndCreateGroup();

	// Create Matlab array
	mwSize* dimsMatlab = static_cast<mwSize*>(mxMalloc(rank * sizeof(mwIndex)));
	for (size_t i = 0; i < rank; ++i)
		dimsMatlab[i] = dims[i];

	mxArray* cell_array_ptr = mxCreateCellArray(rank, dimsMatlab);
	mxFree(dimsMatlab);

	size_t numEl = mxGetNumberOfElements(cell_array_ptr);
	for (mwIndex i = 0; i < numEl / blockSize; ++i)
	{
		for (mwIndex j = 0; j < blockSize; ++j)
			mxSetCell(cell_array_ptr, i * blockSize + j, mxCreateString(buffer[i * stride + j].c_str()));
	}

	mxSetFieldByNumber(_group, 0, mxAddField(_group, dataSetName.c_str()), cell_array_ptr);
}

// Template that matches on every unsupported type
template <typename T>
void MatlabReaderWriter::write(const std::string& dataSetName, const size_t rank, const size_t* dims, const T* buffer,
							   const size_t stride, const size_t blockSize)
{
	std::ostringstream str;
	str << "CadetMex: Trying to write vector of unsupported type to struct '" << _groupName << "/" << dataSetName
		<< "'.\n";
	throw MatlabException(str.str());
}
// ============================================================================================================

// ============================================================================================================
//   Convenience wrappers
// ============================================================================================================

// HDF5: Row Major
// Matlab: Column Major

template <typename T>
void MatlabReaderWriter::tensor(const std::string& dataSetName, const size_t rank, const size_t* dims,
								const std::vector<T>& buffer, const size_t stride, const size_t blockSize)
{
	// Check size
	size_t bufSize = 1;
	for (size_t i = 0; i < rank; ++i)
		bufSize *= dims[i];

	if (bufSize > buffer.size())
	{
		std::ostringstream str;
		str << "CadetMex: Trying to write tensor of size ";
		for (size_t i = 0; i < rank - 1; ++i)
			str << dims[i] << " x ";
		str << dims[rank - 1] << " but got " << buffer.size() << " elements.\n";

		throw MatlabException(str.str());
	}
	write<T>(dataSetName, rank, dims, buffer.data(), stride, blockSize);
}

template <typename T>
void MatlabReaderWriter::tensor(const std::string& dataSetName, const size_t rank, const size_t* dims, const T* buffer,
								const size_t stride, const size_t blockSize)
{
	write<T>(dataSetName, rank, dims, buffer, stride, blockSize);
}

template <typename T>
void MatlabReaderWriter::matrix(const std::string& dataSetName, const size_t rows, const size_t cols, const T* buffer,
								const size_t stride, const size_t blockSize)
{
	size_t dims[2] = {rows, cols};
	write<T>(dataSetName, 2, dims, buffer, stride, blockSize);
}

template <typename T>
void MatlabReaderWriter::matrix(const std::string& dataSetName, const size_t rows, const size_t cols,
								const std::vector<T>& buffer, const size_t stride, const size_t blockSize)
{
#ifdef CADET_DEBUG
	if (rows * cols >= buffer.size())
	{
		std::ostringstream str;
		str << "CadetMex: Trying to write matrix of size " << rows << "x" << cols << " but got " << buffer.size()
			<< " elements.\n";
		throw MatlabException(str.str());
	}
#endif
	size_t dims[2] = {rows, cols};
	write<T>(dataSetName, 2, dims, buffer.data(), stride, blockSize);
}

template <typename T>
void MatlabReaderWriter::vector(const std::string& dataSetName, const size_t length, const T* buffer,
								const size_t stride, const size_t blockSize)
{
	write<T>(dataSetName, 1, &length, buffer, stride, blockSize);
}

template <typename T>
void MatlabReaderWriter::vector(const std::string& dataSetName, const std::vector<T>& buffer, const size_t stride,
								const size_t blockSize)
{
	size_t length = buffer.size() / stride;
	write<T>(dataSetName, 1, &length, buffer.data(), stride, blockSize);
}

template <> void MatlabReaderWriter::scalar<std::string>(const std::string& dataSetName, const std::string buffer)
{
	openAndCreateGroup();
	mxSetFieldByNumber(_group, 0, mxAddField(_group, dataSetName.c_str()), mxCreateString(buffer.c_str()));
}

template <> void MatlabReaderWriter::scalar<const char*>(const std::string& dataSetName, const char* buffer)
{
	openAndCreateGroup();
	mxSetFieldByNumber(_group, 0, mxAddField(_group, dataSetName.c_str()), mxCreateString(buffer));
}

template <typename T> void MatlabReaderWriter::scalar(const std::string& dataSetName, const T buffer)
{
	vector<T>(dataSetName, 1, &buffer);
}
// ============================================================================================================

void MatlabReaderWriter::unlinkGroup(const std::string& groupName)
{
}

void MatlabReaderWriter::unlinkDataset(const std::string& dsName)
{
}

} // namespace mex
} // namespace cadet

#endif // CADET_MEX_MATLABREADERWRITER_HPP_
