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

#ifndef XMLBASE_HPP_
#define XMLBASE_HPP_

#include <string>
#include <sstream>
#include <vector>
#include <stack>

#include "cadet/cadetCompilerInfo.hpp"

#include <pugixml.hpp>
#include "io/IOException.hpp"

namespace cadet
{

namespace io
{

using namespace pugi;

class XMLBase
{
public:
	/// \brief Constructor
	XMLBase();

	/// \brief Destructor
	~XMLBase() CADET_NOEXCEPT;

	/// \brief Open an XML file
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

	/// \brief Checks if the given dataset is a string
	inline bool isString(const std::string& elementName)
	{
		return isDataType<std::string>(elementName, _typeChar);
	}
	inline bool isString(const char* elementName)
	{
		return isString(std::string(elementName));
	}

	/// \brief Checks if the given dataset is a signed int
	inline bool isInt(const std::string& elementName)
	{
		return isDataType<int>(elementName, _typeInt);
	}
	inline bool isInt(const char* elementName)
	{
		return isInt(std::string(elementName));
	}

	/// \brief Checks if the given dataset is a double
	inline bool isDouble(const std::string& elementName)
	{
		return isDataType<double>(elementName, _typeDouble);
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
	xml_document _doc;     //!< XML document
	xml_node _root;        //!< XML root node
	bool _enable_write;    //!< Specifies write permission for currently opened XML file
	std::string _fileName; //!< Name of the currently opened XML file

	static const std::string _textSeparator; //!< Character sequence for separation of text entries
	static const std::string _dimsSeparator; //!< Character sequence for separation of dimensions

	static const std::string _nodeRoot; //!< Name used for the root node
	static const std::string _nodeGrp;  //!< Name used for creation of 'group' nodes
	static const std::string _nodeDset; //!< Name used for creation of 'dataset' nodes

	static const std::string _attrName;  //!< Name used for creation of 'name' attributes
	static const std::string _attrRank;  //!< Name used for creation of 'rank' attributes
	static const std::string _attrDims;  //!< Name used for creation of 'dims' attributes
	static const std::string _attrType;  //!< Name used for creation of 'type' attributes
	static const std::string _attrValue; //!< Name used for creation of 'value' attributes

	static const std::string _typeChar;   //!< Name used for 'type' attributes of type 'char'
	static const std::string _typeInt;    //!< Name used for 'type' attributes of type 'int'
	static const std::string _typeUint64; //!< Name used for 'type' attributes of type 'uint64_t'
	static const std::string _typeDouble; //!< Name used for 'type' attributes of type 'double'
	static const std::string _typeBool;   //!< Name used for 'type' attributes of type 'bool'

	xpath_node _groupOpened;              //!< Holds the group that is currently opened
	std::stack<std::string> _groupExists; //!< Stack holding the last successfully opened group
	std::vector<std::string> _groupNames; //!< Vector of group names for the currently selected group

	void openGroup(bool forceCreation = false);
	void closeGroup();

	void findOrCreateRootNode();

	template <typename T> bool isDataType(const std::string& elementName, const std::string& typeName);

	// Some helper methods
	std::vector<std::string>& split(const std::string& s, const char* delim, std::vector<std::string>& elems);
	std::vector<std::string> split(const std::string& s, const char* delim);
};

// Setting constant values
const std::string XMLBase::_textSeparator = ", ";
const std::string XMLBase::_dimsSeparator = "x";

const std::string XMLBase::_nodeRoot = "cadet";
const std::string XMLBase::_nodeGrp = "group";
const std::string XMLBase::_nodeDset = "dataset";

const std::string XMLBase::_attrName = "name";
const std::string XMLBase::_attrRank = "rank";
const std::string XMLBase::_attrDims = "dims";
const std::string XMLBase::_attrType = "type";
const std::string XMLBase::_attrValue = "value";

const std::string XMLBase::_typeChar = "char";
const std::string XMLBase::_typeInt = "int";
const std::string XMLBase::_typeUint64 = "uint64_t";
const std::string XMLBase::_typeDouble = "double";
const std::string XMLBase::_typeBool = "bool";

XMLBase::XMLBase() : _enable_write(false)
{
}

XMLBase::~XMLBase() CADET_NOEXCEPT
{
	closeFile();
}

void XMLBase::openFile(const std::string& fileName, const std::string& mode)
{
	xml_parse_result flag;

	if (mode == "r") // open in read mode
	{
		if (!_doc.load_file(fileName.c_str()))
			throw IOException("XML file does not exist!");
		_enable_write = false;
	}
	else if (mode == "rw") // open in read / write mode
	{
		if (!_doc.load_file(fileName.c_str()))
			throw IOException("XML file does not exist!");
		_enable_write = true;
	}
	else if (mode == "c") // create new file
	{
		if (_doc.load_file(fileName.c_str()))
			throw IOException("XML file already exists");
		_enable_write = true;
	}
	else if (mode == "co") // create / overwrite new file
		_enable_write = true;
	else
		throw IOException("Wrong file open mode");

	findOrCreateRootNode();
	_fileName = fileName;
}

void XMLBase::closeFile()
{
	if (_enable_write)
		_doc.save_file(_fileName.c_str());
}

bool XMLBase::exists(const char* elementName)
{
	openGroup();
	bool exists = _groupOpened.node().child(elementName);
	closeGroup();
	return exists;
}

bool XMLBase::isVector(const char* elementName)
{
	openGroup();

	// Open dataset and throw if it does not exist
	xml_node dataset = _groupOpened.node().find_child_by_attribute(_nodeDset.c_str(), _attrName.c_str(), elementName);
	if (!dataset)
	{
		std::ostringstream oss;
		oss << "Dataset '" << elementName << "' does not exist!";
		throw IOException(oss.str());
	}

	// Read text and attributes
	std::size_t rank = dataset.attribute(_attrRank.c_str()).as_int();
	std::string dims_str = dataset.attribute(_attrDims.c_str()).value();

	// Get dims and compute buffer size
	std::vector<std::string> dims_vec = split(dims_str, _dimsSeparator.c_str());
	int items = 1;
	for (std::size_t i = 0; i < rank; ++i)
	{
		items *= std::stoi(dims_vec[i]);
	}

	closeGroup();
	return items > 1;
}

template <typename T> bool XMLBase::isDataType(const std::string& elementName, const std::string& typeName)
{
	openGroup();

	// Open dataset and throw if it does not exist
	xml_node dataset =
		_groupOpened.node().find_child_by_attribute(_nodeDset.c_str(), _attrName.c_str(), elementName.c_str());
	if (!dataset)
	{
		std::ostringstream oss;
		oss << "Dataset '" << elementName << "' does not exist!";
		throw IOException(oss.str());
	}

	// Read text and attributes
	const xml_attribute typeAttrib = dataset.attribute(_attrType.c_str());
	bool result = false;
	if (typeAttrib.empty() || (typeAttrib.value() == std::string("")))
	{
		// Infer type from data
		std::stringstream convert(dataset.text().get());
		T temp;
		convert >> temp;
		result = convert.fail();
	}
	else
	{
		// Use attribute
		result = typeAttrib.value() == typeName;
	}

	closeGroup();
	return result;
}

bool XMLBase::isGroup(const std::string& elementName)
{
	openGroup();
	xml_node dataset =
		_groupOpened.node().find_child_by_attribute(_nodeGrp.c_str(), _attrName.c_str(), elementName.c_str());
	closeGroup();

	return !!dataset;
}

std::vector<std::size_t> XMLBase::tensorDimensions(const std::string& elementName)
{
	// Open dataset and throw if it does not exist
	xml_node dataset =
		_groupOpened.node().find_child_by_attribute(_nodeDset.c_str(), _attrName.c_str(), elementName.c_str());
	if (!dataset)
	{
		std::ostringstream oss;
		oss << "Dataset '" << elementName << "' does not exist!";
		throw IOException(oss.str());
	}

	const std::string dims_str = dataset.attribute(_attrDims.c_str()).value();

	// Get dims and compute buffer size
	const std::vector<std::string> dims_vec = split(dims_str, _dimsSeparator.c_str());
	std::vector<std::size_t> dims(dims_vec.size());
	for (std::size_t i = 0; i < dims_vec.size(); ++i)
	{
		dims[i] = std::stoi(dims_vec[i]);
	}

	closeGroup();
	return dims;
}

std::size_t XMLBase::arraySize(const std::string& elementName)
{
	// Open dataset and throw if it does not exist
	xml_node dataset =
		_groupOpened.node().find_child_by_attribute(_nodeDset.c_str(), _attrName.c_str(), elementName.c_str());
	if (!dataset)
	{
		std::ostringstream oss;
		oss << "Dataset '" << elementName << "' does not exist!";
		throw IOException(oss.str());
	}

	const std::string dims_str = dataset.attribute(_attrDims.c_str()).value();

	// Get dims and compute number of elements
	const std::vector<std::string> dims_vec = split(dims_str, _dimsSeparator.c_str());
	if (dims_vec.empty())
		return 0;

	std::size_t n = 1;
	for (std::size_t i = 0; i < dims_vec.size(); ++i)
	{
		n *= std::stoi(dims_vec[i]);
	}

	closeGroup();
	return n;
}

int XMLBase::numItems()
{
	openGroup();

	int n = 0;
	for (xml_node::iterator it = _groupOpened.node().begin(); it != _groupOpened.node().end(); ++it)
		++n;

	closeGroup();
	return n;
}

std::string XMLBase::itemName(int n)
{
	openGroup();

	std::string name = "";
	int i = 0;
	for (xml_node::iterator it = _groupOpened.node().begin(); it != _groupOpened.node().end(); ++it)
	{
		if (i == n)
		{
			const xml_attribute nameAttrib = it->attribute(_attrName.c_str());
			name = nameAttrib.value();
			break;
		}
		++i;
	}

	closeGroup();
	return name;
}

std::vector<std::string> XMLBase::itemNames()
{
	openGroup();

	std::vector<std::string> names;
	for (xml_node::iterator it = _groupOpened.node().begin(); it != _groupOpened.node().end(); ++it)
	{
		const xml_attribute nameAttrib = it->attribute(_attrName.c_str());
		names.push_back(nameAttrib.value());
	}

	closeGroup();
	return names;
}

void XMLBase::setGroup(const std::string& groupName)
{
	_groupNames.clear();

	std::size_t start = 0;
	std::size_t end = 0;
	std::string delimiter("/");

	// Quick return when called with empty group name
	if (groupName.empty() || (groupName == "/"))
		return;

	// Don't care for a preceding delimiter
	if (groupName[0] == delimiter[0])
		++start;

	while (end != std::string::npos)
	{
		end = groupName.find(delimiter, start);

		// If at end, use length = maxLength.  Else use length = end - start.
		_groupNames.push_back(groupName.substr(start, (end == std::string::npos) ? std::string::npos : end - start));

		// If at end, use start = maxSize.  Else use start = end + delimiter.
		start = ((end > (std::string::npos - delimiter.size())) ? std::string::npos : end + delimiter.size());
	}
}

void XMLBase::pushGroup(const std::string& groupName)
{
	_groupNames.push_back(groupName);
}

void XMLBase::popGroup()
{
	_groupNames.pop_back();
}

void XMLBase::openGroup(bool forceCreation)
{
	// Clear existent group cache
	while (_groupExists.size() > 0)
		_groupExists.pop();

	// Generate xpath query string
	std::ostringstream query(std::ostringstream::ate);
	for (std::vector<std::string>::const_iterator it = _groupNames.begin(); it < _groupNames.end(); ++it)
	{
		query << "/" << _nodeGrp << "[@" << _attrName << "='" << *it << "']";
		// Store query string of parent groups for potential later usage
		if (_root.select_single_node(("/" + _nodeRoot + query.str()).c_str()))
		{
			_groupExists.push(query.str());
		}
	}

	if (_groupNames.empty())
	{
		// Select root node
		_groupOpened = _root.select_single_node(("/" + _nodeRoot).c_str());
		return;
	}

	// Try to open the selected group
	_groupOpened = _root.select_single_node(("/" + _nodeRoot + query.str()).c_str());

	// Create new group if not existent, creation is forced and write is permitted
	if (!_groupOpened)
	{
		if (forceCreation && _enable_write)
		{
			xml_node newNode;
			std::size_t ngrpexist = _groupExists.size();
			// Check for any parent group to be existent
			for (std::vector<std::string>::const_iterator it = _groupNames.begin() + ngrpexist; it < _groupNames.end();
				 ++it)
			{
				query.str("");
				// Select the least group
				if (_groupExists.size() > 0)
				{
					_groupOpened = _root.select_single_node(("/" + _nodeRoot + _groupExists.top()).c_str());
					// Create a new group
					newNode = _groupOpened.node().append_child(_nodeGrp.c_str());
					query << _groupExists.top();
				}
				else
					newNode = _root.append_child(_nodeGrp.c_str());

				// Set attribute name
				newNode.append_attribute(_attrName.c_str()) = it->c_str();

				// Store the new group in existent groups
				query << "/" << _nodeGrp << "[@" << _attrName << "='" << *it << "']";
				_groupExists.push(query.str());
			}
			// Open newly created group
			_groupOpened = _root.select_single_node(("/" + _nodeRoot + query.str()).c_str());
		}
		else
			throw IOException("Group was not opened/created! Either not existent, creation not forced or file not "
							  "opened in write mode");
	}
}

void XMLBase::closeGroup()
{
	while (_groupExists.size() > 0)
		_groupExists.pop();
}

void XMLBase::findOrCreateRootNode()
{
	xpath_node rootNode = _doc.select_single_node(("/" + _nodeRoot).c_str());
	if (!rootNode && _enable_write)
	{
		// Root node not found and we can write -> create one
		_root = _doc.append_child(_nodeRoot.c_str());
	}
	else if (rootNode)
	{
		// Root node found
		_root = rootNode.node();
	}
}

std::vector<std::string>& XMLBase::split(const std::string& s, const char* delim, std::vector<std::string>& elems)
{
	std::stringstream ss(s);
	std::string item;

	while (std::getline(ss, item, *delim))
		elems.push_back(item);

	return elems;
}

std::vector<std::string> XMLBase::split(const std::string& s, const char* delim)
{
	std::vector<std::string> elems;
	return split(s, delim, elems);
}

} // namespace io

} // namespace cadet

#endif /* XMLBASE_HPP_ */
