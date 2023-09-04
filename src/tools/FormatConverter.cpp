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

#include "FormatConverter.hpp"
#include "io/FileIO.hpp"

#include <functional>

template <class reader_t, class writer_t> class DualScope
{
public:
	DualScope(reader_t& reader, writer_t& writer) : _reader(reader), _writer(writer)
	{
	}
	DualScope(reader_t& reader, writer_t& writer, const std::string& scope) : _reader(reader), _writer(writer)
	{
		_reader.pushGroup(scope);
		_writer.pushGroup(scope);
	}
	~DualScope()
	{
		_reader.popGroup();
		_writer.popGroup();
	}

protected:
	reader_t& _reader;
	writer_t& _writer;
};

template <typename reader_t, typename writer_t>
void copyGroup(reader_t& rd, writer_t& wr, const std::string& path,
			   const std::function<bool(const std::string&, const std::string&)>& filter)
{
	const std::vector<std::string> names = rd.itemNames();

	for (std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it)
	{
		if (!filter(path, *it))
			continue;

		if (rd.isGroup(*it))
		{
			// Recurse into subgroup
			DualScope<reader_t, writer_t> s(rd, wr, *it);
			copyGroup(rd, wr, path + "/" + (*it), filter);
		}
		else
		{
			// Copy dataset
			const std::vector<std::size_t> dims = rd.tensorDimensions(*it);
			if (dims.size() == 0)
			{
				// Handle scalars
				if (rd.isInt(*it))
				{
					const int d = rd.getInt(*it);
					wr.writeInt(*it, d);
				}
				else if (rd.isDouble(*it))
				{
					const double d = rd.getDouble(*it);
					wr.writeDouble(*it, d);
				}
				else if (rd.isString(*it))
				{
					const std::string d = rd.getString(*it);
					wr.writeString(*it, d);
				}
				else
				{
					const bool d = rd.getBool(*it);
					wr.writeBool(*it, d);
				}
			}
			else if (dims.size() == 1)
			{
				// Handle vectors
				if (rd.isInt(*it))
				{
					const std::vector<int> data = rd.getIntArray(*it);
					wr.writeVectorInt(*it, data.size(), data.data(), 1, 1);
				}
				else if (rd.isDouble(*it))
				{
					const std::vector<double> data = rd.getDoubleArray(*it);
					wr.writeVectorDouble(*it, data.size(), data.data(), 1, 1);
				}
				else if (rd.isString(*it))
				{
					const std::vector<std::string> data = rd.getStringArray(*it);
					wr.writeVectorString(*it, data.size(), data.data(), 1, 1);
				}
				else
				{
					const std::vector<bool> data = rd.getBoolArray(*it);
					bool* dp = new bool[data.size()];
					std::copy(data.begin(), data.end(), dp);
					wr.writeVectorBool(*it, data.size(), dp, 1, 1);
					delete[] dp;
				}
			}
			else if (dims.size() == 2)
			{
				// Handle matrices
				if (rd.isInt(*it))
				{
					const std::vector<int> data = rd.getIntArray(*it);
					wr.writeMatrixInt(*it, dims[0], dims[1], data.data(), 1, 1);
				}
				else if (rd.isDouble(*it))
				{
					const std::vector<double> data = rd.getDoubleArray(*it);
					wr.writeMatrixDouble(*it, dims[0], dims[1], data.data(), 1, 1);
				}
				else if (rd.isString(*it))
				{
					const std::vector<std::string> data = rd.getStringArray(*it);
					wr.writeMatrixString(*it, dims[0], dims[1], data.data(), 1, 1);
				}
				else
				{
					const std::vector<bool> data = rd.getBoolArray(*it);
					bool* dp = new bool[data.size()];
					std::copy(data.begin(), data.end(), dp);
					wr.writeMatrixBool(*it, dims[0], dims[1], dp, 1, 1);
					delete[] dp;
				}
			}
			else if (dims.size() > 2)
			{
				// Handle tensors
				if (rd.isInt(*it))
				{
					const std::vector<int> data = rd.getIntArray(*it);
					wr.writeTensorInt(*it, dims.size(), dims.data(), data.data(), 1, 1);
				}
				else if (rd.isDouble(*it))
				{
					const std::vector<double> data = rd.getDoubleArray(*it);
					wr.writeTensorDouble(*it, dims.size(), dims.data(), data.data(), 1, 1);
				}
				else if (rd.isString(*it))
				{
					const std::vector<std::string> data = rd.getStringArray(*it);
					wr.writeTensorString(*it, dims.size(), dims.data(), data.data(), 1, 1);
				}
				else
				{
					const std::vector<bool> data = rd.getBoolArray(*it);
					bool* dp = new bool[data.size()];
					std::copy(data.begin(), data.end(), dp);
					wr.writeTensorBool(*it, dims.size(), dims.data(), dp, 1, 1);
					delete[] dp;
				}
			}
		}
	}
}

template <typename reader_t, typename writer_t> void copyGroup(reader_t& rd, writer_t& wr, const std::string& path)
{
	copyGroup<reader_t, writer_t>(rd, wr, path,
								  [](const std::string& scope, const std::string& item) -> bool { return true; });
}

void copyGroup(cadet::io::IFileReader& rd, cadet::io::IFileWriter& wr, const std::string& path)
{
	copyGroup<cadet::io::IFileReader, cadet::io::IFileWriter>(
		rd, wr, path, [](const std::string& scope, const std::string& item) -> bool { return true; });
}
