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
 * Provides helper functions for tools apps
 */

#ifndef CADETTOOLS_TOOLSHELPER_HPP_
#define CADETTOOLS_TOOLSHELPER_HPP_

#include <string>
#include <vector>
#include <sstream>
#include <iostream>

#include <tclap/CmdLine.h>

template <class Writer_t>
class Scope
{
public:
	Scope(Writer_t& writer) : _writer(writer) { }
	Scope(Writer_t& writer, const std::string& scope) : _writer(writer) { _writer.pushGroup(scope); }
	~Scope() { _writer.popGroup(); }
protected:
	Writer_t& _writer;
};


inline void split(const std::string& s, const char delim, std::vector<std::string>& elems)
{
	std::stringstream ss(s);
	std::string item;

	while(std::getline(ss, item, delim))
		elems.push_back(item);
}

template <class ProgramOptions_t>
inline void addMiscToCmdLine(TCLAP::CmdLine& cmd, ProgramOptions_t& opts)
{
	cmd >> (new TCLAP::ValueArg<int>("", "par", "Number of particle cells (default: 4)", false, 4, "Value"))->storeIn(&opts.nPar);
	cmd >> (new TCLAP::ValueArg<int>("", "col", "Number of axial cells (default: 10)", false, 10, "Value"))->storeIn(&opts.nCol);

	cmd >> (new TCLAP::SwitchArg("", "solverTimes", "Save all solver timesteps"))->storeIn(&opts.solverTimes);
	cmd >> (new TCLAP::SwitchArg("k", "kinetic", "Kinetic adsorption model used (default: quasi-stationary)"))->storeIn(&opts.isKinetic);
	cmd >> (new TCLAP::ValueArg<int>("j", "threads", "Number of threads (default: 1)", false, 1, "Value"))->storeIn(&opts.nThreads);
	cmd >> (new TCLAP::SwitchArg("", "ad", "Calculate Jacobian using AD (default: analytic)"))->storeIn(&opts.adJacobian);
}

inline void addUnitTypeToCmdLine(TCLAP::CmdLine& cmd, std::string& unitType)
{
	cmd >> (new TCLAP::ValueArg<std::string>("u", "unit", "Unit operation (default: GRM)", false, "GRM", "Unit"))->storeIn(&unitType);
}

inline void parseUnitType(std::string& unitType)
{
	if ((unitType == "GRM") || (unitType == "grm"))
		unitType = "GENERAL_RATE_MODEL";
	else if ((unitType == "LRMP") || (unitType == "lrmp"))
		unitType = "LUMPED_RATE_MODEL_WITH_PORES";
	else if ((unitType == "LRM") || (unitType == "lrm"))
		unitType = "LUMPED_RATE_MODEL_WITHOUT_PORES";
	else if ((unitType == "GRM2D") || (unitType == "grm2d"))
		unitType = "GENERAL_RATE_MODEL_2D";
}

inline void parseUnitType(std::string& unitType, bool radialFlow)
{
	parseUnitType(unitType);
	
	if (radialFlow)
		unitType = "RADIAL_" + unitType;
}

inline void addSensitivitiyParserToCmdLine(TCLAP::CmdLine& cmd, std::vector<std::string>& sensitivities)
{
	cmd >> (new TCLAP::MultiArg<std::string>("S", "sens", 
			"Add parameter sensitivity (use -1 for component, reaction, section, bound phase, or unit operation independent parameters)", 
			false, "ParamName/Comp/Reaction/Section/ParType/BoundPhase[/Factor/UnitOp][+ParamName/...]")
		)->storeIn(&sensitivities);	
}

template <class Writer_t>
inline void parseAndWriteSensitivitiesFromCmdLine(Writer_t& writer, const std::vector<std::string>& sensitivities)
{
	if (sensitivities.size() == 0)
		return;
	// Sensitivities
	Scope<Writer_t> s(writer, "sensitivity");

	writer.template scalar<std::string>("SENS_METHOD", "ad1");

	unsigned int correctParams = 0;
	std::ostringstream oss;
	for (std::size_t i = 0; i < sensitivities.size(); ++i)
	{
		std::vector<std::string> fusedParams;
		split(sensitivities[i], '+', fusedParams);

		std::vector<std::string> paramNames;
		std::vector<int> paramUnit;
		std::vector<int> paramComp;
		std::vector<int> paramReaction;
		std::vector<int> paramSection;
		std::vector<int> paramParType;
		std::vector<int> paramBoundState;
		std::vector<double> paramFactor;

		paramNames.reserve(fusedParams.size());
		paramUnit.reserve(fusedParams.size());
		paramComp.reserve(fusedParams.size());
		paramReaction.reserve(fusedParams.size());
		paramSection.reserve(fusedParams.size());
		paramParType.reserve(fusedParams.size());
		paramBoundState.reserve(fusedParams.size());
		paramFactor.reserve(fusedParams.size());

		for (std::size_t j = 0; j < fusedParams.size(); ++j)
		{
			std::vector<std::string> tokens;
			split(fusedParams[j], '/', tokens);

			if (tokens.size() < 6)
			{
				std::cout << "Warning: Invalid parameter no " << (i+1) << "." << (j+1) << " (" << fusedParams[j] << ") was ignored" << std::endl; 
				continue;
			}

			paramNames.push_back(tokens[0]);
			paramComp.push_back(std::stoi(tokens[1]));
			paramReaction.push_back(std::stoi(tokens[2]));
			paramSection.push_back(std::stoi(tokens[3]));
			paramParType.push_back(std::stoi(tokens[4]));
			paramBoundState.push_back(std::stoi(tokens[5]));

			if (tokens.size() >= 6)
				paramFactor.push_back(std::stod(tokens[6]));
			else
				paramFactor.push_back(1.0);

			if (tokens.size() >= 7)
				paramUnit.push_back(std::stoi(tokens[7]));
			else
				paramUnit.push_back(0);
		}

		if (paramNames.size() > 0)
		{
			oss.str("");
			oss << "param_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;

			Scope<Writer_t> s2(writer, oss.str());

			writer.template vector<std::string>("SENS_NAME", paramNames.size(), paramNames.data());
			writer.template vector<int>("SENS_UNIT", paramUnit.size(), paramUnit.data());
			writer.template vector<int>("SENS_COMP", paramComp.size(), paramComp.data());
			writer.template vector<int>("SENS_REACTION", paramReaction.size(), paramReaction.data());
			writer.template vector<int>("SENS_SECTION", paramSection.size(), paramSection.data());
			writer.template vector<int>("SENS_PARTYPE", paramParType.size(), paramParType.data());
			writer.template vector<int>("SENS_BOUNDPHASE", paramBoundState.size(), paramBoundState.data());
			writer.template vector<double>("SENS_FACTOR", paramFactor.size(), paramFactor.data());

			// Other fields: SENS_ABSTOL, SENS_FD_DELTA

			++correctParams;
		}
		else
			std::cout << "Warning: Invalid parameter " << (i+1) << " (" << sensitivities[i] << ") was ignored" << std::endl;
	}

	writer.template scalar<int>("NSENS", correctParams);
}

inline void addOutputParserToCmdLine(TCLAP::CmdLine& cmd, std::string& outSol)
{
	cmd >> (new TCLAP::ValueArg<std::string>("", "outSol",
			"Solution output format ([I]nlet,[O]utlet,[B]ulk,[P]article,[F]luxes, default: O)",
			false, "O", "IOBPF")
		)->storeIn(&outSol);	
}

inline void addOutputParserToCmdLine(TCLAP::CmdLine& cmd, std::string& outSol, std::string& outSens)
{
	addOutputParserToCmdLine(cmd, outSol);
	cmd >> (new TCLAP::ValueArg<std::string>("", "outSens",
			"Sensitivity output format ([I]nlet,[O]utlet,[B]ulk,[P]article,[F]luxes, default: O)",
			false, "O", "IOBPF")
		)->storeIn(&outSens);	
}

template <class Writer_t>
inline void parseAndWriteOutputFormatInternal(Writer_t& writer, const std::string& prefix, const std::string& outVal)
{
	bool inlet = false;
	bool outlet = false;
	bool bulk = false;
	bool particle = false;
	bool flux = false;

	for (unsigned int i = 0; i < outVal.length(); ++i)
	{
		switch (outVal[i])
		{
			case 'I':
				inlet = true;
				break;
			case 'O':
				outlet = true;
				break;
			case 'B':
				bulk = true;
				break;
			case 'P':
				particle = true;
				break;
			case 'F':
				flux = true;
				break;
		}
	}
	writer.template scalar<int>("WRITE_" + prefix + "_BULK", bulk);
	writer.template scalar<int>("WRITE_" + prefix + "_PARTICLE", particle);
	writer.template scalar<int>("WRITE_" + prefix + "_FLUX", flux);
	writer.template scalar<int>("WRITE_" + prefix + "_INLET", inlet);
	writer.template scalar<int>("WRITE_" + prefix + "_OUTLET", outlet);
}

template <class Writer_t>
inline void parseAndWriteOutputFormatsFromCmdLine(Writer_t& writer, const std::string& outSol, const std::string& outSens)
{
	parseAndWriteOutputFormatInternal(writer, "SOLUTION", outSol);
	parseAndWriteOutputFormatInternal(writer, "SENS", outSens);
}

template <class Writer_t>
inline void parseAndWriteOutputFormatsFromCmdLine(Writer_t& writer, const std::string& outSol)
{
	parseAndWriteOutputFormatInternal(writer, "SOLUTION", outSol);
	writer.template scalar<int>("WRITE_SOLUTION_TIMES", true);
}

#endif  // CADETTOOLS_TOOLSHELPER_HPP_
