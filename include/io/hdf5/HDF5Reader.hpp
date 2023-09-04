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

#ifndef HDF5READER_HPP_
#define HDF5READER_HPP_

#include <vector>
#include <string>

#include "cadet/cadetCompilerInfo.hpp"

#include "HDF5Base.hpp"

namespace cadet
{

namespace io
{

class HDF5Reader : public HDF5Base
{
public:
	/// \brief Constructor
	HDF5Reader();

	/// \brief Destructor
	~HDF5Reader() CADET_NOEXCEPT;

	/// \brief Convenience wrapper for reading vectors
	template <typename T> std::vector<T> vector(const std::string& dataSetName);

	/// \brief Convenience wrapper for reading scalars
	template <typename T> T scalar(const std::string& dataSetName, std::size_t position = 0);

private:
	template <typename T> std::vector<T> read(const std::string& dataSetName, hid_t dataType);
};

HDF5Reader::HDF5Reader()
{
}

HDF5Reader::~HDF5Reader() CADET_NOEXCEPT
{
}

// ============================================================================================================
//   Template specializations of member functions for diffenet data types
// ============================================================================================================
// Double specialization of vector()
template <> std::vector<double> HDF5Reader::vector<double>(const std::string& dataSetName)
{
	return read<double>(dataSetName, H5T_NATIVE_DOUBLE);
}

// Integer specializations of vector()
template <> std::vector<int> HDF5Reader::vector<int>(const std::string& dataSetName)
{
	return read<int>(dataSetName, H5T_NATIVE_INT);
}

template <> std::vector<uint64_t> HDF5Reader::vector<uint64_t>(const std::string& dataSetName)
{
	return read<uint64_t>(dataSetName, H5T_NATIVE_UINT64);
}

// std::string specialization of vector()
template <> std::vector<std::string> HDF5Reader::vector<std::string>(const std::string& dataSetName)
{
	// Get the dataset we want to read from
	openGroup();
	const hid_t dataSet = H5Dopen2(_groupsOpened.top(), dataSetName.c_str(), H5P_DEFAULT);
	closeGroup();

	if (dataSet < 0)
		throw IOException("Field \"" + dataSetName + "\" does not exist in group " + getFullGroupName());

	// Determine the datatype and allocate buffer
	const hid_t dataType = H5Dget_type(dataSet);
	const hid_t dataSpace = H5Dget_space(dataSet);
	const std::size_t bufSize = H5Sget_simple_extent_npoints(dataSpace);

	std::vector<std::string> stringVector;

	if (bufSize == 0)
	{
		H5Tclose(dataType);
		H5Sclose(dataSpace);
		H5Dclose(dataSet);
		return stringVector;
	}

	if (H5Tis_variable_str(dataType))
	{
		char** buffer = new char*[bufSize];

		const hid_t memType = H5Tcopy(H5T_C_S1);
		H5Tset_size(memType, H5T_VARIABLE);

		// Read data from file and write it to buffer
		H5Dread(dataSet, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

		// Copy read c-strings to a vector of std::strings
		for (std::size_t i = 0; i < bufSize; ++i)
			stringVector.push_back(std::string(buffer[i]));

		// Free memory alloc'd by the variable length read mechanism
		H5Dvlen_reclaim(dataType, dataSpace, H5P_DEFAULT, buffer);
		H5Tclose(memType);
		delete[] buffer;
	}
	else
	{
		const std::size_t strLen = H5Tget_size(dataType) + 1; // Add 1 for null terminator (maybe unnecessary)
		char* buffer = new char[strLen * bufSize];

		const hid_t memType = H5Tcopy(H5T_C_S1);
		H5Tset_size(memType, strLen);

		// Read data from file, write it to buffer and copy the items to vector
		H5Dread(dataSet, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
		for (std::size_t i = 0; i < bufSize; ++i)
			stringVector.push_back(std::string(buffer + i * strLen));

		delete[] buffer;

		H5Tclose(memType);
	}

	H5Tclose(dataType);
	H5Sclose(dataSpace);
	H5Dclose(dataSet);

	return stringVector;
}

// Template that matches on every unsupported type and throws an exception
template <typename T> std::vector<T> HDF5Reader::vector(const std::string& dataSetName)
{
	throw IOException("You may not try to read an unsupported type");
}
// ============================================================================================================

template <typename T> T HDF5Reader::scalar(const std::string& dataSetName, std::size_t position)
{
	return vector<T>(dataSetName).at(position);
}

template <typename T> std::vector<T> HDF5Reader::read(const std::string& dataSetName, hid_t memType)
{
	// Get the dataset we want to read from
	openGroup();
	const hid_t dataSet = H5Dopen2(_groupsOpened.top(), dataSetName.c_str(), H5P_DEFAULT);
	closeGroup();

	if (dataSet < 0)
		throw IOException("Field \"" + dataSetName + "\" does not exist in group " + getFullGroupName());

	// Determine the datatype and allocate buffer
	const hid_t dataType = H5Dget_type(dataSet);
	const hid_t dataSpace = H5Dget_space(dataSet);
	const std::size_t bufSize = H5Sget_simple_extent_npoints(dataSpace);

	if (bufSize == 0)
	{
		H5Tclose(dataType);
		H5Sclose(dataSpace);
		H5Dclose(dataSet);
		return std::vector<T>();
	}

	std::vector<T> buffer(bufSize);

	// Read data from file and write it to buffer
	H5Dread(dataSet, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());

	H5Tclose(dataType);
	H5Sclose(dataSpace);
	H5Dclose(dataSet);

	return buffer;
}

} // namespace io

} // namespace cadet

#endif /* HDF5READER_HPP_ */
