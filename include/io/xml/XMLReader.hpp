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

#ifndef XMLREADER_HPP_
#define XMLREADER_HPP_

#include "cadet/cadetCompilerInfo.hpp"
#include "XMLBase.hpp"

namespace cadet
{

namespace io
{

using namespace pugi;

class XMLReader : public XMLBase
{
public:
	/// \brief Constructor
	XMLReader();

	/// \brief Destructor
	~XMLReader() CADET_NOEXCEPT;

	/// \brief Convenience wrapper for reading vectors
	template <typename T> std::vector<T> vector(const std::string& dataSetName);

	/// \brief Convenience wrapper for reading scalars
	template <typename T> T scalar(const std::string& dataSetName, std::size_t position = 0);

private:
	template <typename T> std::vector<T> read(const std::string& dataSetName);
};

XMLReader::XMLReader()
{
}

XMLReader::~XMLReader() CADET_NOEXCEPT
{
}

// ============================================================================================================
//   Template specializations of member functions for diffenet data types
// ============================================================================================================
// Double specialization of vector()
template <> std::vector<double> XMLReader::vector<double>(const std::string& dataSetName)
{
	return read<double>(dataSetName);
}

// Integer specializations of vector()
template <> std::vector<int> XMLReader::vector<int>(const std::string& dataSetName)
{
	return read<int>(dataSetName);
}

template <> std::vector<uint64_t> XMLReader::vector<uint64_t>(const std::string& dataSetName)
{
	return read<uint64_t>(dataSetName);
}

// std::string specialization of vector()
template <> std::vector<std::string> XMLReader::vector<std::string>(const std::string& dataSetName)
{
	return read<std::string>(dataSetName);
}

// Template that matches on every unsupported type and throws an exception
template <typename T> std::vector<T> XMLReader::vector(const std::string& dataSetName)
{
	throw IOException("You may not try to read an unsupported type");
}
// ============================================================================================================

template <typename T> T XMLReader::scalar(const std::string& dataSetName, std::size_t position)
{
	return vector<T>(dataSetName).at(position);
}

template <typename T> std::vector<T> XMLReader::read(const std::string& dataSetName)
{
	openGroup();

	// Open dataset and throw if it does not exist
	xml_node dataset =
		_groupOpened.node().find_child_by_attribute(_nodeDset.c_str(), _attrName.c_str(), dataSetName.c_str());
	if (!dataset)
	{
		std::ostringstream oss;
		oss << "Dataset '" << dataSetName << "' does not exist!";
		throw IOException(oss.str());
	}

	// Read text and attributes
	std::string data_str = dataset.text().get();
	std::size_t rank = dataset.attribute(_attrRank.c_str()).as_int();
	std::string dims_str = dataset.attribute(_attrDims.c_str()).value();

	// Get dims and compute buffer size
	std::vector<std::string> dims_vec = split(dims_str, _dimsSeparator.c_str());
	std::size_t* dims = new std::size_t[rank];
	std::size_t bufSize = 1;
	for (std::size_t i = 0; i < rank; ++i)
	{
		std::stringstream ss(dims_vec[i]);
		ss >> dims[i];
		bufSize *= dims[i];
	}
	delete[] dims;

	// Split data and convert to right type
	std::vector<std::string> data_vec = split(data_str, _textSeparator.c_str());
	if (bufSize != data_vec.size())
	{
		std::ostringstream oss;
		oss << "XML file is inconsistent: Possibly wrong no. of entrys in dataset '"
			<< dataset.attribute(_attrName.c_str()).value() << "'";
		throw IOException(oss.str());
	}

	std::vector<T> data(bufSize);
	for (std::size_t i = 0; i < bufSize; ++i)
	{
		std::stringstream ss(data_vec[i]);
		ss >> data[i];
	}

	closeGroup();
	return data;
}

} // namespace io

} // namespace cadet

#endif /* XMLREADER_HPP_ */
