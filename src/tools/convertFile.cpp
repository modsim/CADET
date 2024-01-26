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

#include <iostream>
#include <string>

#include "io/FileIO.hpp"
#include "io/IOException.hpp"

#include <tclap/CmdLine.h>
#include "common/TclapUtils.hpp"
#include "FormatConverter.hpp"

template <class Writer_t> class FileCloser
{
public:
	FileCloser(Writer_t& writer) : _writer(writer)
	{
	}
	~FileCloser()
	{
		_writer.closeFile();
	}

protected:
	Writer_t& _writer;
};

template <class class_t> class DeleterScope
{
public:
	DeleterScope(class_t* elem) : _elem(elem)
	{
	}
	~DeleterScope()
	{
		delete _elem;
	}

protected:
	class_t* const _elem;
};

void translateFormat(cadet::io::IFileReader* rd, cadet::io::IFileWriter* wr)
{
	copyGroup(*rd, *wr, "/");
}

bool run(const std::string& inFileName, const std::string& inFileExt, const std::string& outFileName,
		 const std::string& outFileExt, bool translate)
{
	cadet::io::IFileReader* rd = cadet::io::createReader(inFileExt);
	if (!rd)
	{
		std::cerr << "Input file format ('." << inFileExt << "') not supported" << std::endl;
		return false;
	}
	DeleterScope<cadet::io::IFileReader> dsR(rd);

	cadet::io::IFileWriter* wr = cadet::io::createWriter(outFileExt);
	if (!wr)
	{
		std::cerr << "Output file format ('." << outFileExt << "') not supported" << std::endl;
		return false;
	}
	DeleterScope<cadet::io::IFileWriter> dsW(wr);

	rd->openFile(inFileName, "r");
	FileCloser<cadet::io::IFileReader> fcR(*rd);

	// Only translate between file format types
	if (translate)
	{
		std::cout << "Translating file format from " << inFileExt << " to " << outFileExt << std::endl;

		wr->openFile(outFileName, "co");
		FileCloser<cadet::io::IFileWriter> fcW(*wr);
		translateFormat(rd, wr);
		return true;
	}

	std::cout << "Conversion between file format versions is not yet implemented" << std::endl;
	return false;
}

int main(int argc, char** argv)
{
	std::string inFile;
	std::string outFile;
	bool translateFormat = false;
	try
	{
		TCLAP::CustomOutputWithoutVersion customOut("convertFile");
		TCLAP::CmdLine cmd("Converts between CADET file format versions", ' ', "1.0");
		cmd.setOutput(&customOut);

		cmd >> (new TCLAP::SwitchArg("t", "translate", "Translate file format type keeping its version"))
				   ->storeIn(&translateFormat);
		cmd >> (new TCLAP::UnlabeledValueArg<std::string>("input", "Input file", true, "", "File"))->storeIn(&inFile);
		cmd >>
			(new TCLAP::UnlabeledValueArg<std::string>("output", "Output file", false, "", "File"))->storeIn(&outFile);

		cmd.parse(argc, argv);
	}
	catch (const TCLAP::ArgException& e)
	{
		std::cerr << "ERROR: " << e.error() << " for argument " << e.argId() << std::endl;
		return 1;
	}

	if (outFile == inFile)
	{
		std::cerr << "Input and output file must be distinct" << std::endl;
		return 2;
	}

	// Obtain file extensions for selecting corresponding reader and writer
	const std::size_t dotPosIn = inFile.find_last_of('.');
	if (dotPosIn == std::string::npos)
	{
		std::cerr << "Could not deduce input filetype due to missing extension: " << inFile << std::endl;
		return 2;
	}
	const std::string fileExtIn = inFile.substr(dotPosIn + 1);

	// Set output file name
	if (outFile.empty())
	{
		outFile = inFile.substr(0, dotPosIn) + "-converted." + fileExtIn;
	}

	const std::size_t dotPosOut = outFile.find_last_of('.');
	if (dotPosOut == std::string::npos)
	{
		std::cerr << "Could not deduce output filetype due to missing extension: " << outFile << std::endl;
		return 2;
	}
	const std::string fileExtOut = outFile.substr(dotPosOut + 1);

	try
	{
		if (run(inFile, fileExtIn, outFile, fileExtOut, translateFormat))
			return 0;
	}
	catch (const cadet::io::IOException& e)
	{
		std::cerr << "IO ERROR: " << e.what() << std::endl;
		return 2;
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
