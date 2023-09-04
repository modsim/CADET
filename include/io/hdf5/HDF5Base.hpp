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

#ifndef HDF5BASE_HPP_
#define HDF5BASE_HPP_

#include <vector>
#include <string>
#include <sstream>
#include <stack>
#include <algorithm>

#include "cadet/cadetCompilerInfo.hpp"

#include "io/IOException.hpp"
#include <hdf5.h>

namespace cadet
{

namespace io
{

class HDF5Base
{
public:
	/// \brief Constructor
	HDF5Base();

	/// \brief Destructor
	~HDF5Base() CADET_NOEXCEPT;

	/// \brief Open an HDF5 file
	inline void openFile(const std::string& fileName, const std::string& mode = "r");
	inline void openFile(const char* fileName, const std::string& mode = "r")
	{
		openFile(std::string(fileName), mode);
	}

	/// \brief Close the currently opened file
	inline void closeFile();

	/// \brief Set a group to be [read from/written to] in all subsequent calls to [read/write] methods
	inline void setGroup(const std::string& groupName);

	/// \brief Open the subgroup with the given name
	inline void pushGroup(const std::string& groupName);
	/// \brief Close the currently open subgroup
	inline void popGroup();

	/// \brief Checks if the given dataset or group exists in the file
	inline bool exists(const std::string& elementName);
	inline bool exists(const char* elementName)
	{
		return exists(std::string(elementName));
	}

	/// \brief Checks if the given dataset is a vector (i.e., has more than one value)
	inline bool isVector(const std::string& elementName);
	inline bool isVector(const char* elementName)
	{
		return isVector(std::string(elementName));
	}

	/// \brief Checks if the given dataset is a string
	inline bool isString(const std::string& elementName);
	inline bool isString(const char* elementName)
	{
		return isString(std::string(elementName));
	}

	/// \brief Checks if the given dataset is a signed int
	inline bool isInt(const std::string& elementName)
	{
		return isDataType(elementName, H5T_NATIVE_INT);
	}
	inline bool isInt(const char* elementName)
	{
		return isInt(std::string(elementName));
	}

	/// \brief Checks if the given dataset is a double
	inline bool isDouble(const std::string& elementName)
	{
		return isDataType(elementName, H5T_NATIVE_DOUBLE);
	}
	inline bool isDouble(const char* elementName)
	{
		return isDouble(std::string(elementName));
	}

	/// \brief Checks whether the given element is a group
	inline bool isGroup(const std::string& elementName);
	inline bool isGroup(const char* elementName)
	{
		return isGroup(std::string(elementName));
	}

	/// \brief Returns the dimensions of the tensor identified by name
	inline std::vector<std::size_t> tensorDimensions(const std::string& elementName);
	inline std::vector<std::size_t> tensorDimensions(const char* elementName)
	{
		return tensorDimensions(std::string(elementName));
	}

	/// \brief Returns the number of elements in the array identified by name
	inline std::size_t arraySize(const std::string& elementName);
	inline std::size_t arraySize(const char* elementName)
	{
		return arraySize(std::string(elementName));
	}

	/// \brief Returns the number of items in the group
	inline int numItems();

	/// \brief Returns the name of the n-th item in the group
	inline std::string itemName(int n);

	/// \brief Returns the names of all items in the group
	inline std::vector<std::string> itemNames();

protected:
	hid_t _file;

	std::stack<hid_t> _groupsOpened;
	std::vector<std::string> _groupNames;

	void openGroup(bool forceCreation = false);
	void closeGroup();
	std::string getFullGroupName();

	bool isDataType(const std::string& elementName, hid_t refType);
};

HDF5Base::HDF5Base()
{
	H5Eset_auto(H5E_DEFAULT, NULL, NULL);
	_groupNames.push_back("/");
}

HDF5Base::~HDF5Base() CADET_NOEXCEPT
{
}

void HDF5Base::openFile(const std::string& fileName, const std::string& mode)
{
	if (mode == "r")
		_file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT); // open in read mode
	else if (mode == "rw")
		_file = H5Fopen(fileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT); // open in read / write mode
	else if (mode == "c")
		_file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); // create new file
	else if (mode == "co")
		_file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); // create / overwrite new file
	else
		throw IOException("Wrong file open mode");

	if (_file < 0)
		throw IOException("Failed to open or create HDF5 file \"" + fileName + "\" in mode " + mode);
}

void HDF5Base::closeFile()
{
	closeGroup();
	H5Fclose(_file);
}

bool HDF5Base::exists(const std::string& elementName)
{
	openGroup();

	// Try to open elementName as group
	hid_t grp = H5Gopen2(_groupsOpened.top(), elementName.c_str(), H5P_DEFAULT);
	if (grp >= 0)
	{
		// Found the group
		H5Gclose(grp);
		closeGroup();
		return true;
	}

	// Try to open elementName as dataset
	hid_t dset = H5Dopen2(_groupsOpened.top(), elementName.c_str(), H5P_DEFAULT);
	if (dset >= 0)
	{
		// Found the dataset
		H5Dclose(dset);
		closeGroup();
		return true;
	}

	// Not found
	closeGroup();
	return false;
}

bool HDF5Base::isVector(const std::string& elementName)
{
	openGroup();

	hid_t dset = H5Dopen2(_groupsOpened.top(), elementName.c_str(), H5P_DEFAULT);
	if (dset < 0)
	{
		closeGroup();
		return false;
	}

	hid_t dspace = H5Dget_space(dset);
	if (dspace < 0)
	{
		H5Dclose(dset);
		closeGroup();
		return false;
	}

	const bool isVector = H5Sget_simple_extent_npoints(dspace) > 1;

	H5Sclose(dspace);
	H5Dclose(dset);
	closeGroup();

	return isVector;
}

bool HDF5Base::isString(const std::string& elementName)
{
	// Get the dataset we want to read from
	openGroup();
	const hid_t obj = H5Oopen(_groupsOpened.top(), elementName.c_str(), H5P_DEFAULT);

	if (obj < 0)
	{
		closeGroup();
		throw IOException("Field \"" + elementName + "\" does not exist in group " + getFullGroupName());
	}

	const H5I_type_t oType = H5Iget_type(obj);
	H5Oclose(obj);

	if (oType != H5I_DATASET)
	{
		closeGroup();
		throw IOException("Field \"" + elementName + "\" in group " + getFullGroupName() + " is not a dataset");
	}

	const hid_t dataSet = H5Dopen2(_groupsOpened.top(), elementName.c_str(), H5P_DEFAULT);
	closeGroup();

	if (dataSet < 0)
		throw IOException("Field \"" + elementName + "\" does not exist in group " + getFullGroupName());

	// Determine the datatype
	const hid_t dataType = H5Dget_type(dataSet);
	const H5T_class_t typeClass = H5Tget_class(dataType);
	const bool result = (H5Tis_variable_str(dataType) > 0) || (typeClass == H5T_STRING);

	H5Tclose(dataType);
	H5Dclose(dataSet);

	return result;
}

bool HDF5Base::isDataType(const std::string& elementName, hid_t refType)
{
	// Get the dataset we want to read from
	openGroup();
	const hid_t obj = H5Oopen(_groupsOpened.top(), elementName.c_str(), H5P_DEFAULT);

	if (obj < 0)
	{
		closeGroup();
		throw IOException("Field \"" + elementName + "\" does not exist in group " + getFullGroupName());
	}

	const H5I_type_t oType = H5Iget_type(obj);
	H5Oclose(obj);

	if (oType != H5I_DATASET)
	{
		closeGroup();
		throw IOException("Field \"" + elementName + "\" in group " + getFullGroupName() + " is not a dataset");
	}

	const hid_t dataSet = H5Dopen2(_groupsOpened.top(), elementName.c_str(), H5P_DEFAULT);
	closeGroup();

	if (dataSet < 0)
		throw IOException("Field \"" + elementName + "\" does not exist in group " + getFullGroupName());

	// Determine the datatype
	const hid_t dataType = H5Dget_type(dataSet);
	const hid_t nativeType = H5Tget_native_type(dataType, H5T_DIR_ASCEND);

	const bool result = H5Tequal(nativeType, refType) > 0;

	H5Tclose(nativeType);
	H5Tclose(dataType);
	H5Dclose(dataSet);

	return result;
}

bool HDF5Base::isGroup(const std::string& elementName)
{
	// Get the dataset we want to read from
	openGroup();
	const hid_t obj = H5Oopen(_groupsOpened.top(), elementName.c_str(), H5P_DEFAULT);

	if (obj < 0)
	{
		closeGroup();
		throw IOException("Field \"" + elementName + "\" does not exist in group " + getFullGroupName());
	}

	const H5I_type_t oType = H5Iget_type(obj);
	H5Oclose(obj);
	closeGroup();

	return oType == H5I_GROUP;
}

std::vector<std::size_t> HDF5Base::tensorDimensions(const std::string& elementName)
{
	// Get the dataset we want to read from
	openGroup();
	const hid_t dataSet = H5Dopen2(_groupsOpened.top(), elementName.c_str(), H5P_DEFAULT);
	closeGroup();

	if (dataSet < 0)
		throw IOException("Field \"" + elementName + "\" does not exist in group " + getFullGroupName());

	// Determine the datatype and allocate buffer
	const hid_t dataSpace = H5Dget_space(dataSet);
	const int rank = H5Sget_simple_extent_ndims(dataSpace);

	std::vector<std::size_t> dims(rank);
	if (rank == 0)
		return dims;

	std::vector<hsize_t> buffer(rank);
	H5Sget_simple_extent_dims(dataSpace, buffer.data(), nullptr);

	H5Sclose(dataSpace);
	H5Dclose(dataSet);

	for (int i = 0; i < rank; ++i)
		dims[i] = buffer[i];

	return dims;
}

std::size_t HDF5Base::arraySize(const std::string& elementName)
{
	// Get the dataset we want to read from
	openGroup();
	const hid_t dataSet = H5Dopen2(_groupsOpened.top(), elementName.c_str(), H5P_DEFAULT);
	closeGroup();

	if (dataSet < 0)
		throw IOException("Field \"" + elementName + "\" does not exist in group " + getFullGroupName());

	const hid_t dataSpace = H5Dget_space(dataSet);
	const hsize_t n = H5Sget_simple_extent_npoints(dataSpace);

	H5Sclose(dataSpace);
	H5Dclose(dataSet);

	return n;
}

int HDF5Base::numItems()
{
	H5G_info_t info;
	openGroup();
	H5Gget_info(_groupsOpened.top(), &info);
	closeGroup();

	return info.nlinks;
}

std::string HDF5Base::itemName(int n)
{
	openGroup();

	const std::size_t len =
		H5Lget_name_by_idx(_groupsOpened.top(), ".", H5_INDEX_NAME, H5_ITER_NATIVE, n, nullptr, 0, H5P_DEFAULT) + 1;
	char* const data = new char[len];
	data[len - 1] = 0;

	H5Lget_name_by_idx(_groupsOpened.top(), ".", H5_INDEX_NAME, H5_ITER_NATIVE, n, data, len, H5P_DEFAULT);
	const std::string name(data);
	delete[] data;

	closeGroup();

	return name;
}

std::vector<std::string> HDF5Base::itemNames()
{
	H5G_info_t info;
	openGroup();

	// Get number of elements
	H5Gget_info(_groupsOpened.top(), &info);
	std::vector<std::string> names;
	names.reserve(info.nlinks);

	// Get required buffer size
	std::size_t maxBufSize = 0;
	for (std::size_t i = 0; i < info.nlinks; ++i)
	{
		const std::size_t len =
			H5Lget_name_by_idx(_groupsOpened.top(), ".", H5_INDEX_NAME, H5_ITER_NATIVE, i, nullptr, 0, H5P_DEFAULT);
		maxBufSize = std::max(maxBufSize, len + 1);
	}

	// Read names
	char* const buffer = new char[maxBufSize];
	for (std::size_t i = 0; i < info.nlinks; ++i)
	{
		std::fill(buffer, buffer + maxBufSize, 0);

		H5Lget_name_by_idx(_groupsOpened.top(), ".", H5_INDEX_NAME, H5_ITER_NATIVE, i, buffer, maxBufSize, H5P_DEFAULT);
		names.push_back(std::string(buffer));
	}

	delete[] buffer;
	closeGroup();

	return names;
}

void HDF5Base::setGroup(const std::string& groupName)
{
	_groupNames.clear();

	std::size_t start = 0;
	std::size_t end = 0;
	std::string delimiter("/");

	// Quick return when called with empty group name
	if (groupName.empty())
	{
		_groupNames.push_back("/");
		return;
	}

	// Don't care for a preceding delimiter
	if (groupName[0] == delimiter[0])
		++start;

	while (end != std::string::npos)
	{
		end = groupName.find(delimiter, start);

		// If at end, use length = maxLength.  Else use length = end - start.
		_groupNames.push_back(delimiter +
							  groupName.substr(start, (end == std::string::npos) ? std::string::npos : end - start));

		// If at end, use start = maxSize.  Else use start = end + delimiter.
		start = ((end > (std::string::npos - delimiter.size())) ? std::string::npos : end + delimiter.size());
	}
}

void HDF5Base::pushGroup(const std::string& groupName)
{
	_groupNames.push_back("/" + groupName);
}

void HDF5Base::popGroup()
{
	_groupNames.pop_back();
}

void HDF5Base::openGroup(bool forceCreation)
{
	std::string dynName;
	for (std::vector<std::string>::const_iterator it = _groupNames.begin(); it < _groupNames.end(); ++it)
	{
		dynName += *it;
		hid_t grp = H5Gopen2(_file, dynName.c_str(), H5P_DEFAULT);
		if (grp >= 0)
			_groupsOpened.push(grp);
		else if (forceCreation)
			_groupsOpened.push(H5Gcreate2(_file, dynName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
		else
			throw IOException("Group '" + dynName + "' doesn't exist in file");
	}
}

void HDF5Base::closeGroup()
{
	while (_groupsOpened.size() >= 1)
	{
		H5Gclose(_groupsOpened.top());
		_groupsOpened.pop();
	}
}

std::string HDF5Base::getFullGroupName()
{
	std::ostringstream oss;
	for (std::vector<std::string>::const_iterator it = _groupNames.begin(); it < _groupNames.end(); ++it)
	{
		oss << *it;
	}
	return oss.str();
}

} // namespace io

} // namespace cadet

#endif /* HDF5BASE_HPP_ */
