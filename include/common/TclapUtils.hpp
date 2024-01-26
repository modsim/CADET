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

#ifndef LIBCADET_TCLAPUTILS_HPP_
#define LIBCADET_TCLAPUTILS_HPP_

#include <tclap/StdOutput.h>
#include <string>

#include "cadet/LibVersionInfo.hpp"

namespace TCLAP
{

/**
 * @brief Modifies the standard behavior of TCLAP to output a better version notice
 * @details The version notice includes the version of CADET (tag, branch, and commit hash)
 *          as well hints to CADET-Web and the GitHub project.
 */
class CustomOutput : public StdOutput
{
public:
	CustomOutput(const std::string& progName) : _progName(progName)
	{
	}

	virtual void version(CmdLineInterface& c)
	{
		std::cout << "This is " << _progName << " version " << cadet::getLibraryVersion() << " ("
				  << cadet::getLibraryBranchRefspec() << " branch)\n";
		std::cout << "Built from commit " << cadet::getLibraryCommitHash() << "\n";
		std::cout << "CADET homepage: <http://www.cadet-web.de>\n";
		std::cout << "Fork CADET on GitHub: <https://github.com/modsim/CADET>\n";
		std::cout << "Report bugs to the issue tracker on GitHub or <cadet@fz-juelich.de>\n";
		std::cout << "See the accompanying LICENSE.txt, AUTHORS, and CONTRIBUTORS files" << std::endl;
	}

protected:
	std::string _progName;
};

/**
 * @brief Modifies the standard behavior of TCLAP to output a better version notice (without library version)
 * @details The version notice includes hints to CADET-Web and the GitHub project.
 */
class CustomOutputWithoutVersion : public StdOutput
{
public:
	CustomOutputWithoutVersion(const std::string& progName) : _progName(progName)
	{
	}

	virtual void version(CmdLineInterface& c)
	{
		std::cout << "This is " << _progName << "\n";
		std::cout << "CADET homepage: <http://www.cadet-web.de>\n";
		std::cout << "Fork CADET on GitHub: <https://github.com/modsim/CADET>\n";
		std::cout << "Report bugs to the issue tracker on GitHub or <cadet@fz-juelich.de>\n";
		std::cout << "See the accompanying LICENSE.txt, AUTHORS, and CONTRIBUTORS files" << std::endl;
	}

protected:
	std::string _progName;
};

} // namespace TCLAP

#endif // LIBCADET_TCLAPUTILS_HPP_
