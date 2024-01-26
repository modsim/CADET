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

#ifndef MATLAB_MEX_FILE
#define MATLAB_MEX_FILE
#endif

#include <mex.h>

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <string>

// Everything that includes mex.h should go here
#include "Logging.hpp"
#include "MatlabIO.hpp"
#include "MatlabReaderWriter.hpp"
#include "MatlabCommands.hpp"

// Take care of namespace pollution / macros
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include "cadet/cadet.hpp"

#include "common/CompilerSpecific.hpp"
#include "common/ParameterProviderImpl.hpp"
#include "common/Driver.hpp"

namespace std
{
template <> struct hash<cadet::ParameterId>
{
	inline std::size_t operator()(const cadet::ParameterId& pId) const CADET_NOEXCEPT
	{
		return cadet::hashParameter(pId);
	}
};
} // namespace std

namespace
{
/**
 * @brief Automatically converts Matlab data to a given target type
 * @details Checks whether the Matlab data is of type double or a given native Matlab type
 *          and performs automatic type conversion. Supports iterating via the braces [] operator.
 *
 * @tparam Target_t Target type that is used for processing
 * @tparam Matlab_t Accepted native Matlab type
 */
template <typename Target_t, typename Matlab_t> class MatlabAutoConverter
{
public:
	/**
	 * @brief Creates a MatlabAutoConverter on the given Matlab data
	 * @details If the data is neither of double nor of the specified native type, an exception is thrown.
	 *
	 * @param [in] data Matlab data handle
	 * @param [in] errMsg Error message thrown if the data is neither of type double, nor of the accepted native type
	 */
	MatlabAutoConverter(mxArray const* const data, const char* errMsg)
		: _data(data), _doubleData(nullptr), _nativeData(nullptr)
	{
		if (cadet::mex::io::isType<double>(data))
			_doubleData = cadet::mex::io::data<double>(data);
		else if (cadet::mex::io::isType<Matlab_t>(data))
			_nativeData = cadet::mex::io::data<Matlab_t>(data);
		else
			mexErrMsgIdAndTxt("CADET:mexError", errMsg);
	}

	Target_t operator[](int idx) const
	{
		if (_doubleData)
			return static_cast<Target_t>(_doubleData[idx]);
		else
			return _nativeData[idx];
	}

protected:
	mxArray const* const _data;
	double const* _doubleData;
	Matlab_t const* _nativeData;
};

inline void checkInputArgs(int nrhs, int required, const char* cmdName)
{
	if (nrhs != required)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command '%s' requires %d input arguments.\n", cmdName, required);
}

inline void checkOutputArgs(int nlhs, int required, const char* cmdName)
{
	if (nlhs != required)
	{
		if (required == 0)
			mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command '%s' does not return anything.\n", cmdName);
		else
			mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command '%s' requires exactly %d output arguments.\n",
							  cmdName, required);
	}
}

inline void requireConfiguredSimulator(cadet::ISimulator* sim, const char* funcName)
{
	if (!sim)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command '%s' requires a configured simulator.\n", funcName);
}

inline void requireConfiguredSimulatorAndModel(cadet::ISimulator* sim, const char* funcName)
{
	requireConfiguredSimulator(sim, funcName);
	if (!sim->model())
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command '%s' requires a configured model.\n", funcName);
}

inline void printParam(const cadet::ParameterId& pId)
{
	std::stringstream out;
	out << "{" << cadet::hashParameter(pId) << " = " << pId.name << ", Unit "
		<< static_cast<unsigned int>(pId.unitOperation) << " Comp " << static_cast<unsigned int>(pId.component)
		<< " BoundState " << static_cast<unsigned int>(pId.boundState) << " Reaction "
		<< static_cast<unsigned int>(pId.reaction) << " Section " << static_cast<unsigned int>(pId.section) << "}";
	mexPrintf("%s\n", out.str().c_str());
}

inline void printParam(const cadet::ParameterId& pId, double value)
{
	std::stringstream out;
	out << "{" << cadet::hashParameter(pId) << " = " << pId.name << ", Unit "
		<< static_cast<unsigned int>(pId.unitOperation) << " Comp " << static_cast<unsigned int>(pId.component)
		<< " BoundState " << static_cast<unsigned int>(pId.boundState) << " Reaction "
		<< static_cast<unsigned int>(pId.reaction) << " Section " << static_cast<unsigned int>(pId.section) << "}";
	mexPrintf("%s = %g\n", out.str().c_str(), value);
}

inline mxArray* createParamStructArray(unsigned int nElements)
{
	const char* fieldNames[] = {"NAMEHASH", "UNIT", "COMP", "REACTION", "BOUNDPHASE", "PARTYPE", "SECTION"};
	return mxCreateStructMatrix(nElements, 1, 7, fieldNames);
}

inline void writeParameterToMatlab(mxArray* structArray, unsigned int idx, const cadet::ParameterId& pId)
{
	mxSetFieldByNumber(structArray, idx, 0, cadet::mex::io::scalar<cadet::StringHash, uint64_t>(pId.name));
	mxSetFieldByNumber(structArray, idx, 1,
					   cadet::mex::io::scalar<cadet::UnitOpIdx, int32_t>(pId.unitOperation, cadet::UnitOpIndep));
	mxSetFieldByNumber(structArray, idx, 2,
					   cadet::mex::io::scalar<cadet::ComponentIdx, int32_t>(pId.component, cadet::CompIndep));
	mxSetFieldByNumber(structArray, idx, 3,
					   cadet::mex::io::scalar<cadet::ReactionIdx, int32_t>(pId.reaction, cadet::ReactionIndep));
	mxSetFieldByNumber(structArray, idx, 4,
					   cadet::mex::io::scalar<cadet::BoundStateIdx, int32_t>(pId.boundState, cadet::BoundStateIndep));
	mxSetFieldByNumber(structArray, idx, 5,
					   cadet::mex::io::scalar<cadet::ParticleTypeIdx, int32_t>(pId.particleType, cadet::ParTypeIndep));
	mxSetFieldByNumber(structArray, idx, 6,
					   cadet::mex::io::scalar<cadet::SectionIdx, int32_t>(pId.section, cadet::SectionIndep));
}

} // namespace

namespace cadet
{

namespace mex
{

/**
 * @brief Runs a full CADET simulation either configuring a new simulator if none exists, or clearing all previous
 * results
 * @param [in] drv Driver
 * @param [in] input Matlab struct with input data
 * @param [out] output Matlab struct to write the results to
 */
void runFullSimulation(cadet::Driver& drv, mxArray const*& input, mxArray*& output)
{
	if (!drv.simulator())
	{
		mxArray** temp = const_cast<mxArray**>(&input);
		cadet::mex::MatlabReaderWriter reader(temp);
		cadet::ParameterProviderImpl<cadet::mex::MatlabReaderWriter> pp(reader);
		drv.configure(pp);
	}
	else
		drv.clearResults();

	drv.run();

	cadet::mex::MatlabReaderWriter writer(&output);
	drv.write(writer);
}

namespace command
{

/**
 * @brief Clears the simulator, all results, and all models from memory
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void clearSimulator(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nlhs != 0)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'clearsim' does not return anything.\n");

	drv.clear();
}

/**
 * @brief Determines whether the simulator and the models are configured
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void isConfigured(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs != 2)
		mexWarnMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'isconf' ignores additional arguments.\n");
	if (nlhs != 1)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'isconf' requires exactly one output.\n");

	plhs[0] = mxCreateLogicalScalar(drv.simulator());
}

/**
 * @brief Configures the simulator and creates and configures model and submodels
 * @details Builds and configures a new model without running it.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void configure(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs != 3)
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'conf' requires a handle and a task (struct) to operate on.\n");
	if (nlhs != 0)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'conf' does not return anything.\n");

	if (!drv.simulator())
	{
		cadet::mex::MatlabReaderWriter reader(const_cast<mxArray**>(&prhs[2]));
		cadet::ParameterProviderImpl<cadet::mex::MatlabReaderWriter> pp(reader);
		drv.configure(pp);
	}
	else
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'conf' cannot configure an already existing model (use 'reconf').\n");
}

/**
 * @brief Runs a full CADET simulation cycle
 * @details Builds and configures a new model, runs it, and writes the results back.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void run(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs != 3)
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'run' requires a handle and a task (struct) to operate on.\n");
	if (nlhs != 1)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'run' requires exactly one output.\n");
	if (drv.simulator())
		mexErrMsgIdAndTxt(
			"CADET:mexError",
			"CadetMex: Command 'run' cannot configure an already existing model (use 'rerun' and 'reconf').\n");

	runFullSimulation(drv, prhs[2], plhs[0]);
}

/**
 * @brief Sets all sensitive parameters to given values
 * @details Requires that the model and its sensitivities are already configured.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setSensitiveParameterValues(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs != 3)
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'setsensparval' requires a vector with parameter values.\n");
	if (nlhs != 0)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'setsensparval' does not return anything.\n");
	requireConfiguredSimulatorAndModel(drv.simulator(), "setsensparval");

	const mxArray* const matlabVals = prhs[2];

	if (!io::isType<double>(matlabVals))
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'setsensparval' requires a vector of 'double' parameter values.\n");

	const double* const parVals = io::data<double>(matlabVals);
	for (unsigned int i = 0; i < io::numElements(matlabVals); ++i)
		drv.simulator()->setSensitiveParameterValue(i, parVals[i]);
}

/**
 * @brief Sets the linear factors of a sensitive parameter to given values
 * @details Requires that the model and its sensitivities are already configured.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setSensitiveParameterFactors(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs != 4)
		mexErrMsgIdAndTxt(
			"CADET:mexError",
			"CadetMex: Command 'setsensparfactor' requires a sensitivity index and a vector with parameter values.\n");
	if (nlhs != 0)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'setsensparfactor' does not return anything.\n");
	requireConfiguredSimulatorAndModel(drv.simulator(), "setsensparfactor");

	const MatlabAutoConverter<int32_t, int32_t> idxSens(
		prhs[2], "CadetMex: Command 'setsensparfactor' requires sensitivity index of type 'int32'.\n");
	const mxArray* const matlabVals = prhs[3];

	if (!io::isType<double>(matlabVals))
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'setsensparfactor' requires a vector of 'double' parameter values.\n");

	const double* const parVals = io::data<double>(matlabVals);
	drv.simulator()->setSensitiveParameterFactors(idxSens[0], parVals);
}

/**
 * @brief Sets all sensitive error tolerances to given values
 * @details Requires that the model and its sensitivities are already configured.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setSensitivityErrorTolerance(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs != 4)
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'setsenserror' requires a vector with parameter values.\n");
	if (nlhs != 0)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'setsenserror' does not return anything.\n");
	requireConfiguredSimulatorAndModel(drv.simulator(), "setsenserror");

	if (drv.simulator()->numSensParams() == 0)
		return;

	if (!io::isType<double>(prhs[2]))
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'setsenserror' requires a scalar of 'double' as relative tolerance.\n");

	if (!io::isType<double>(prhs[3]))
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'setsenserror' requires a vector of 'double' for absolute tolerances.\n");

	if (io::numElements(prhs[3]) < drv.simulator()->numSensParams())
		mexErrMsgIdAndTxt(
			"CADET:mexError",
			"CadetMex: Command 'setsenserror' requires a vector of 'double' for absolute tolerances with %u entries.\n",
			drv.simulator()->numSensParams());

	const double relTol = io::scalar<double>(prhs[2]);
	const double* const absTol = io::data<double>(prhs[3]);
	drv.simulator()->setSensitivityErrorTolerance(relTol, absTol);
}

/**
 * @brief Sets consistent initialization mode
 * @details Requires that the simulator is configured.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setConsistentInitializationMode(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nlhs > 0)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'setconsinitmode' does not return anything.\n");
	if ((nrhs == 2) || (nrhs > 4))
		mexWarnMsgIdAndTxt("CADET:mexWarn", "CadetMex: Command 'setconsinitmode' requires 3 or 4 arguments.\n");
	requireConfiguredSimulator(drv.simulator(), "setconsinitmode");

	if (!io::isEmpty(prhs[2]))
	{
		if (io::isType<std::string>(prhs[2]))
		{
			const char* const strConsInit = mxArrayToString(prhs[2]);
			drv.simulator()->setConsistentInitialization(cadet::to_consistentinitialization(strConsInit));
			mxFree(const_cast<char*>(strConsInit));
		}
		else
		{
			const MatlabAutoConverter<int32_t, int32_t> ci(
				prhs[2],
				"CadetMex: Command 'setconsinitmode' requires consistent initialization mode of type 'int32'.\n");
			const int32_t ciVal = ci[0];
			if (cadet::isValidConsistentInitialization(ciVal))
				drv.simulator()->setConsistentInitialization(cadet::toConsistentInitialization(ciVal));
			else
				mexErrMsgIdAndTxt("CADET:mexError",
								  "CadetMex: Command 'setconsinitmode' received invalid parameter %d.\n", ciVal);
		}
	}

	if ((nrhs == 4) && !io::isEmpty(prhs[3]))
	{
		if (io::isType<std::string>(prhs[3]))
		{
			const char* const strConsInit = mxArrayToString(prhs[3]);
			drv.simulator()->setConsistentInitializationSens(cadet::to_consistentinitialization(strConsInit));
			mxFree(const_cast<char*>(strConsInit));
		}
		else
		{
			const MatlabAutoConverter<int32_t, int32_t> ci(prhs[3],
														   "CadetMex: Command 'setconsinitmode' requires consistent "
														   "sensitivity initialization mode of type 'int32'.\n");
			const int32_t ciVal = ci[0];
			if (cadet::isValidConsistentInitialization(ciVal))
				drv.simulator()->setConsistentInitializationSens(cadet::toConsistentInitialization(ciVal));
			else
				mexErrMsgIdAndTxt("CADET:mexError",
								  "CadetMex: Command 'setconsinitmode' received invalid parameter %d.\n", ciVal);
		}
	}
}

/**
 * @brief Returns all available parameters that can be sensitive with their current values
 * @details Requires that the model is already configured.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void availableParameters(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs > 2)
		mexWarnMsgIdAndTxt("CADET:mexWarn",
						   "CadetMex: Command 'getallpar' ignores all additional arguments (requires only 2).\n");
	if (nrhs == 0)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'getallpar' requires at least 1 output.\n");
	if (nrhs > 2)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'getallpar' processes only up to 2 outputs.\n");
	requireConfiguredSimulatorAndModel(drv.simulator(), "getallpar");

	const std::unordered_map<cadet::ParameterId, double> data = drv.simulator()->getAllParameterValues();
	plhs[0] = createParamStructArray(data.size());

	double* ptrVal = nullptr;
	if (nlhs == 2)
	{
		plhs[1] = cadet::mex::io::createArray<double>(data.size(), 1);
		ptrVal = mxGetPr(plhs[1]);
	}

	unsigned int i = 0;
	for (const std::pair<cadet::ParameterId, double>& val : data)
	{
		writeParameterToMatlab(plhs[0], i, val.first);
		if (nlhs == 2)
			ptrVal[i] = val.second;

		++i;
	}
}

/**
 * @brief Sets given parameters to given values
 * @details Requires that the model is already configured.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setParameters(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	checkInputArgs(nrhs, 10, "setparval");
	checkOutputArgs(nlhs, 0, "setparval");
	requireConfiguredSimulatorAndModel(drv.simulator(), "setparval");

	// Check lengths of vectors and cell arrays
	const unsigned int nPar = io::numElements(prhs[2]);
	for (unsigned int i = 3; i < 10; ++i)
	{
		if (io::numElements(prhs[i]) != nPar)
			mexErrMsgIdAndTxt("CADET:mexError",
							  "CadetMex: Command 'setparval' requires all additional arguments to be of the same size "
							  "(failed for argument %d).\n",
							  i + 1);
	}

	const mxArray* const mNames = prhs[2];
	const MatlabAutoConverter<UnitOpIdx, int32_t> unitOps(
		prhs[3], "CadetMex: Command 'setparval' requires unit operation ids of type 'int32'.\n");
	const MatlabAutoConverter<ComponentIdx, int32_t> comps(
		prhs[4], "CadetMex: Command 'setparval' requires component ids of type 'int32'.\n");
	const MatlabAutoConverter<ParticleTypeIdx, int32_t> parType(
		prhs[5], "CadetMex: Command 'setparval' requires particle type ids of type 'int32'.\n");
	const MatlabAutoConverter<BoundStateIdx, int32_t> boundStates(
		prhs[6], "CadetMex: Command 'setparval' requires bound state ids of type 'int32'.\n");
	const MatlabAutoConverter<ReactionIdx, int32_t> reactIdx(
		prhs[7], "CadetMex: Command 'setparval' requires reaction ids of type 'int32'.\n");
	const MatlabAutoConverter<SectionIdx, int32_t> secIdx(
		prhs[8], "CadetMex: Command 'setparval' requires section ids of type 'int32'.\n");

	if (!cadet::mex::io::isType<double>(prhs[9]))
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'setparval' requires parameter values of type 'double'.\n");

	if (!cadet::mex::io::isCell(mNames) && !cadet::mex::io::isType<uint64_t>(mNames))
		mexErrMsgIdAndTxt(
			"CADET:mexError",
			"CadetMex: Command 'setparval' requires parameter names as cell array of strings or hashes (uint64).\n");

	uint64_t const* nameHash = nullptr;
	if (cadet::mex::io::isType<uint64_t>(mNames))
		nameHash = cadet::mex::io::data<uint64_t>(mNames);

	double const* const parVals = cadet::mex::io::data<double>(prhs[9]);

	// Execute parameter changes
	for (unsigned int i = 0; i < nPar; ++i)
	{
		cadet::StringHash paramName;

		if (!nameHash)
		{
			mxArray* const curCellStr = mxGetCell(mNames, i);
			if (!mxIsChar(curCellStr))
				mexErrMsgIdAndTxt("CADET:mexError",
								  "CadetMex: Command 'setparval' requires all parameter name cells to contain strings "
								  "(failed for element %d).\n",
								  i + 1);

			const char* const strName = mxArrayToString(curCellStr);
			paramName = cadet::hashStringRuntime(strName);
			mxFree(const_cast<char*>(strName));
		}
		else
			paramName = nameHash[i];

		// Construct parameter ID
		const ParameterId pId =
			makeParamId(paramName, unitOps[i], comps[i], parType[i], boundStates[i], reactIdx[i], secIdx[i]);

		// Set the parameter
		drv.simulator()->setParameterValue(pId, parVals[i]);
	}
}

/**
 * @brief Checks whether the given parameters exist
 * @details Requires that the model is already configured.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void checkParameters(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	checkInputArgs(nrhs, 9, "checkpar");
	checkOutputArgs(nlhs, 1, "checkpar");
	requireConfiguredSimulatorAndModel(drv.simulator(), "checkpar");

	// Check lengths of vectors and cell arrays
	const unsigned int nPar = io::numElements(prhs[2]);
	for (unsigned int i = 3; i < 9; ++i)
	{
		if (io::numElements(prhs[i]) != nPar)
			mexErrMsgIdAndTxt("CADET:mexError",
							  "CadetMex: Command 'checkpar' requires all additional arguments to be of the same size "
							  "(failed for argument %d).\n",
							  i + 1);
	}

	const mxArray* const mNames = prhs[2];
	const MatlabAutoConverter<UnitOpIdx, int32_t> unitOps(
		prhs[3], "CadetMex: Command 'checkpar' requires unit operation ids of type 'int32'.\n");
	const MatlabAutoConverter<ComponentIdx, int32_t> comps(
		prhs[4], "CadetMex: Command 'checkpar' requires component ids of type 'int32'.\n");
	const MatlabAutoConverter<ParticleTypeIdx, int32_t> parType(
		prhs[5], "CadetMex: Command 'checkpar' requires particle type ids of type 'int32'.\n");
	const MatlabAutoConverter<BoundStateIdx, int32_t> boundStates(
		prhs[6], "CadetMex: Command 'checkpar' requires bound state ids of type 'int32'.\n");
	const MatlabAutoConverter<ReactionIdx, int32_t> reactIdx(
		prhs[7], "CadetMex: Command 'checkpar' requires reaction ids of type 'int32'.\n");
	const MatlabAutoConverter<SectionIdx, int32_t> secIdx(
		prhs[8], "CadetMex: Command 'checkpar' requires section ids of type 'int32'.\n");

	if (!cadet::mex::io::isCell(mNames) && !cadet::mex::io::isType<uint64_t>(mNames))
		mexErrMsgIdAndTxt(
			"CADET:mexError",
			"CadetMex: Command 'checkpar' requires parameter names as cell array of strings or hashes (uint64).\n");

	uint64_t const* nameHash = nullptr;
	if (cadet::mex::io::isType<uint64_t>(mNames))
		nameHash = cadet::mex::io::data<uint64_t>(mNames);

	plhs[0] = mxCreateLogicalMatrix(nPar, 1);
	mxLogical* const out = mxGetLogicals(plhs[0]);

	// Check parameters
	for (unsigned int i = 0; i < nPar; ++i)
	{
		cadet::StringHash paramName;

		if (!nameHash)
		{
			mxArray* const curCellStr = mxGetCell(mNames, i);
			if (!mxIsChar(curCellStr))
				mexErrMsgIdAndTxt("CADET:mexError",
								  "CadetMex: Command 'checkpar' requires all parameter name cells to contain strings "
								  "(failed for element %d).\n",
								  i + 1);

			const char* const strName = mxArrayToString(curCellStr);
			paramName = cadet::hashStringRuntime(strName);
			mxFree(const_cast<char*>(strName));
		}
		else
			paramName = nameHash[i];

		// Construct parameter ID
		const ParameterId pId =
			makeParamId(paramName, unitOps[i], comps[i], parType[i], boundStates[i], reactIdx[i], secIdx[i]);
		//		printParam(pId);

		// Check the parameter
		out[i] = drv.simulator()->hasParameter(pId);
	}
}

/**
 * @brief Sets given parameters to given values
 * @details Requires that the model is already configured.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void makeParameterSensitive(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	checkInputArgs(nrhs, 11, "setsenspar");
	checkOutputArgs(nlhs, 0, "setsenspar");
	requireConfiguredSimulatorAndModel(drv.simulator(), "setsenspar");

	// Check lengths of vectors and cell arrays
	const unsigned int nPar = io::numElements(prhs[2]);
	for (unsigned int i = 3; i < 10; ++i)
	{
		if (io::numElements(prhs[i]) != nPar)
			mexErrMsgIdAndTxt("CADET:mexError",
							  "CadetMex: Command 'setsenspar' requires all additional arguments to be of the same size "
							  "(failed for argument %d).\n",
							  i + 1);
	}

	const mxArray* const mNames = prhs[2];
	const MatlabAutoConverter<UnitOpIdx, int32_t> unitOps(
		prhs[3], "CadetMex: Command 'setsenspar' requires unit operation ids of type 'int32'.\n");
	const MatlabAutoConverter<ComponentIdx, int32_t> comps(
		prhs[4], "CadetMex: Command 'setsenspar' requires component ids of type 'int32'.\n");
	const MatlabAutoConverter<ParticleTypeIdx, int32_t> parType(
		prhs[5], "CadetMex: Command 'setsenspar' requires particle type ids of type 'int32'.\n");
	const MatlabAutoConverter<BoundStateIdx, int32_t> boundStates(
		prhs[6], "CadetMex: Command 'setsenspar' requires bound state ids of type 'int32'.\n");
	const MatlabAutoConverter<ReactionIdx, int32_t> reactIdx(
		prhs[7], "CadetMex: Command 'setsenspar' requires reaction ids of type 'int32'.\n");
	const MatlabAutoConverter<SectionIdx, int32_t> secIdx(
		prhs[8], "CadetMex: Command 'setsenspar' requires section ids of type 'int32'.\n");

	if (!cadet::mex::io::isType<double>(prhs[9]))
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'setsenspar' requires linear factors of type 'double'.\n");

	if (!cadet::mex::io::isType<double>(prhs[10]))
		mexErrMsgIdAndTxt("CADET:mexError",
						  "CadetMex: Command 'setsenspar' requires absolute tolerance of type 'double'.\n");

	if (!cadet::mex::io::isCell(mNames) && !cadet::mex::io::isType<uint64_t>(mNames))
		mexErrMsgIdAndTxt(
			"CADET:mexError",
			"CadetMex: Command 'setsenspar' requires parameter names as cell array of strings or hashes (uint64).\n");

	uint64_t const* nameHash = nullptr;
	if (cadet::mex::io::isType<uint64_t>(mNames))
		nameHash = cadet::mex::io::data<uint64_t>(mNames);

	std::vector<cadet::ParameterId> sensParams;
	sensParams.reserve(nPar);

	const double sensTol = cadet::mex::io::scalar<double>(prhs[10]);

	// Execute parameter changes
	for (unsigned int i = 0; i < nPar; ++i)
	{
		cadet::StringHash paramName;

		if (!nameHash)
		{
			mxArray* const curCellStr = mxGetCell(mNames, i);
			if (!mxIsChar(curCellStr))
				mexErrMsgIdAndTxt("CADET:mexError",
								  "CadetMex: Command 'setparval' requires all parameter name cells to contain strings "
								  "(failed for element %d).\n",
								  i + 1);

			const char* const strName = mxArrayToString(curCellStr);
			paramName = cadet::hashStringRuntime(strName);
			mxFree(const_cast<char*>(strName));
		}
		else
			paramName = nameHash[i];

		// Construct parameter ID
		sensParams.push_back(
			makeParamId(paramName, unitOps[i], comps[i], parType[i], boundStates[i], reactIdx[i], secIdx[i]));
	}

	drv.simulator()->setSensitiveParameter(sensParams.data(), cadet::mex::io::data<double>(prhs[9]), nPar, sensTol);
}

/**
 * @brief Runs an already configured CADET simulation again
 * @details Requires an already configured model and clears all existing results from memory.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void reRun(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs > 3)
		mexWarnMsgIdAndTxt(
			"CADET:mexWarn",
			"CadetMex: Command 'rerun' ignores all additional arguments (requires only 2, uses at most 3).\n");
	if (nlhs != 1)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'rerun' requires exactly one output.\n");

	requireConfiguredSimulatorAndModel(drv.simulator(), "rerun");

	drv.clearResults();

	if (nrhs == 3)
	{
		if (mxIsLogical(prhs[2]))
		{
			// Third parameter is a logical that controls whether consistent initialization is skipped
			mxLogical const* const skipConsistInit = mxGetLogicals(prhs[2]);
			if (*skipConsistInit)
			{
				drv.simulator()->skipConsistentInitialization();
			}
		}
		else
		{
			// Set initial conditions
			cadet::mex::MatlabReaderWriter reader(const_cast<mxArray**>(&prhs[2]));
			cadet::ParameterProviderImpl<cadet::mex::MatlabReaderWriter> pp(reader, false);
			drv.setInitialCondition(pp);
		}
	}
	else
	{
		// Resume simulation from last state, skip consistent initialization
		drv.simulator()->skipConsistentInitialization();
	}

	drv.run();

	cadet::mex::MatlabReaderWriter writer(&plhs[0]);
	drv.write(writer);
}

/**
 * @brief Reconfigures a given unit operation model, the model system itself, or the time integrator
 * @details Requires an already configured model. The entity that is configured depends on the
 *          number of input arguments.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void reconfigureModelOrSimulator(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs < 3)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'reconf' requires at least 3 input arguments.\n");
	if (nrhs > 4)
		mexWarnMsgIdAndTxt("CADET:mexWarn",
						   "CadetMex: Command 'reconf' ignores all additional arguments (takes at most 4).\n");
	checkOutputArgs(nlhs, 0, "reconf");
	requireConfiguredSimulatorAndModel(drv.simulator(), "reconf");

	cadet::mex::MatlabReaderWriter reader(const_cast<mxArray**>(&prhs[2]));
	cadet::ParameterProviderImpl<cadet::mex::MatlabReaderWriter> pp(reader, false);

	if (nrhs == 3)
	{
		// Configuring the time integrator / simulator
		drv.simulator()->reconfigure(pp);

		// Section times and continuity are handled separately
		drv.setSectionTimes(pp);
	}
	else
	{
		// Configure the model
		const MatlabAutoConverter<UnitOpIdx, int32_t> unitOps(
			prhs[3], "CadetMex: Command 'reconf' requires unit operation ids of type 'int32'.\n");
		const UnitOpIdx uid = unitOps[0];
		if (uid == UnitOpIndep)
		{
			// Configure the model system
			drv.simulator()->reconfigureModel(pp);
		}
		else
		{
			// Configure a specific unit operation model
			drv.simulator()->reconfigureModel(pp, uid);
		}
	}
}

/**
 * @brief Sets the section times of the time integrator
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setReturnConfiguration(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	checkInputArgs(nrhs, 3, "setreturnconf");
	checkOutputArgs(nlhs, 0, "setreturnconf");

	if (!drv.simulator() || !drv.simulator()->model())
		return;

	cadet::mex::MatlabReaderWriter reader(const_cast<mxArray**>(&prhs[2]));
	cadet::ParameterProviderImpl<cadet::mex::MatlabReaderWriter> pp(reader);
	drv.setReturnConfiguration(pp, true);
}

/**
 * @brief Sets time integrator options and settings (e.g., error tolerances)
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setTimeIntegratorOptions(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	checkInputArgs(nrhs, 9, "settimeintopts");
	checkOutputArgs(nlhs, 0, "settimeintopts");

	if (!drv.simulator())
		return;

	// Relative error tolerance
	if (!io::isEmpty(prhs[2]))
	{
		const double val = cadet::mex::io::scalar<double>(prhs[2]);
		drv.simulator()->setRelativeErrorTolerance(val);
	}

	// Absolute error tolerance (scalar or vector)
	if (!io::isEmpty(prhs[3]))
	{
		const unsigned int numElm = cadet::mex::io::numElements(prhs[3]);
		if (numElm > 1)
		{
			double const* const data = cadet::mex::io::data<double>(prhs[3]);
			const std::vector<double> val(data, data + numElm);
			drv.simulator()->setAbsoluteErrorTolerance(val);
		}
		else
		{
			const double val = cadet::mex::io::scalar<double>(prhs[3]);
			drv.simulator()->setAbsoluteErrorTolerance(val);
		}
	}

	// Algebraic error tolerance
	if (!io::isEmpty(prhs[4]))
	{
		const double val = cadet::mex::io::scalar<double>(prhs[4]);
		drv.simulator()->setAlgebraicErrorTolerance(val);
	}

	// Initial step size (scalar or vector)
	if (!io::isEmpty(prhs[5]))
	{
		const unsigned int numElm = cadet::mex::io::numElements(prhs[5]);
		if (numElm > 1)
		{
			double const* const data = cadet::mex::io::data<double>(prhs[5]);
			const std::vector<double> val(data, data + numElm);
			drv.simulator()->setInitialStepSize(val);
		}
		else
		{
			const double val = cadet::mex::io::scalar<double>(prhs[5]);
			drv.simulator()->setInitialStepSize(val);
		}
	}

	// Maximum number of time steps
	if (!io::isEmpty(prhs[6]))
	{
		const unsigned int val = cadet::mex::io::scalar<int32_t>(prhs[6]);
		drv.simulator()->setMaximumSteps(val);
	}

	// Maximum time step size
	if (!io::isEmpty(prhs[7]))
	{
		const unsigned int val = cadet::mex::io::scalar<int32_t>(prhs[7]);
		drv.simulator()->setMaximumStepSize(val);
	}

	// Relative error tolerance for sensitivities
	if (!io::isEmpty(prhs[8]))
	{
		const double val = cadet::mex::io::scalar<double>(prhs[8]);
		drv.simulator()->setRelativeErrorToleranceSens(val);
	}
}

/**
 * @brief Sets time integrator solver options and settings (e.g., number of Newton iterations)
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setTimeIntegratorSolverOptions(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	checkInputArgs(nrhs, 7, "settimeintsolveropts");
	checkOutputArgs(nlhs, 0, "settimeintsolveropts");

	if (!drv.simulator())
		return;

	// Forward sensitivity error test participation
	if (!io::isEmpty(prhs[2]))
	{
		const int32_t val = cadet::mex::io::scalar<int32_t>(prhs[2]);
		drv.simulator()->setSensitivityErrorControl(val);
	}

	// Maximum number of Newton iterations
	if (!io::isEmpty(prhs[3]))
	{
		const unsigned int val = cadet::mex::io::scalar<int32_t>(prhs[3]);
		drv.simulator()->setMaxNewtonIteration(val);
	}

	// Maximum number of error test failures
	if (!io::isEmpty(prhs[4]))
	{
		const unsigned int val = cadet::mex::io::scalar<int32_t>(prhs[4]);
		drv.simulator()->setMaxErrorTestFails(val);
	}

	// Maximum number of convergence test failures
	if (!io::isEmpty(prhs[5]))
	{
		const unsigned int val = cadet::mex::io::scalar<int32_t>(prhs[5]);
		drv.simulator()->setMaxConvergenceFails(val);
	}

	// Maximum number of sensitivity Newton iterations
	if (!io::isEmpty(prhs[6]))
	{
		const unsigned int val = cadet::mex::io::scalar<int32_t>(prhs[6]);
		drv.simulator()->setMaxSensNewtonIteration(val);
	}
}

/**
 * @brief Sets the section times of the time integrator
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setSectionTimes(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs < 3)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'setsectimes' requires at least 3 input arguments.\n");
	if (nrhs > 4)
		mexWarnMsgIdAndTxt("CADET:mexWarn",
						   "CadetMex: Command 'setsectimes' ignores all additional arguments (takes at most 4).\n");
	checkOutputArgs(nlhs, 0, "setsectimes");

	if (!drv.simulator())
		return;

	const unsigned int nSections = cadet::mex::io::numElements(prhs[2]);
	if (nSections <= 1)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'setsectimes' requires at least 2 section times.\n");

	double const* const secTimesData = cadet::mex::io::data<double>(prhs[2]);
	std::vector<double> secTimes(secTimesData, secTimesData + nSections);
	if ((nrhs == 3) || ((nrhs == 4) && cadet::mex::io::isEmpty(prhs[3])))
		drv.simulator()->setSectionTimes(secTimes);
	else
	{
		const unsigned int nCont = cadet::mex::io::numElements(prhs[3]);
		if (nCont < nSections - 2)
			mexErrMsgIdAndTxt("CADET:mexError",
							  "CadetMex: Command 'setsectimes' requires at least %u section continuity entries.\n",
							  nSections - 2);

		std::vector<bool> secCont(nSections - 2);
		const MatlabAutoConverter<bool, int32_t> secContSource(
			prhs[3], "CadetMex: Command 'setsectimes' requires section continuity of type 'int32'.\n");

		for (std::size_t i = 0; i < secCont.size(); ++i)
			secCont[i] = secContSource[i];

		drv.simulator()->setSectionTimes(secTimes, secCont);
	}
}

/**
 * @brief Sets the solution times of the time integrator
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setSolutionTimes(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	checkInputArgs(nrhs, 3, "setsoltimes");
	checkOutputArgs(nlhs, 0, "setsoltimes");

	if (!drv.simulator())
		return;

	double const* const solTimesData = cadet::mex::io::data<double>(prhs[2]);
	std::vector<double> solTimes(solTimesData, solTimesData + cadet::mex::io::numElements(prhs[2]));
	drv.simulator()->setSolutionTimes(solTimes);
}

/**
 * @brief Sets the number of threads used by the time integrator
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setNumThreads(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	checkInputArgs(nrhs, 3, "setnumthreads");
	checkOutputArgs(nlhs, 0, "setnumthreads");

	if (!drv.simulator())
		return;

	const unsigned int nThreads = cadet::mex::io::scalar<int32_t>(prhs[2]);
	drv.simulator()->setNumThreads(nThreads);
}

/**
 * @brief Sets whether the time points of the solution and the last state is returned or not
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void setWriteTimeAndLastState(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	checkInputArgs(nrhs, 5, "setwritetimeandlaststate");
	checkOutputArgs(nlhs, 0, "setwritetimeandlaststate");

	if (!cadet::mex::io::isEmpty(prhs[2]))
		drv.setWriteSolutionTimes(cadet::mex::io::scalar<int32_t>(prhs[2]));

	if (!cadet::mex::io::isEmpty(prhs[3]))
		drv.setWriteLastState(cadet::mex::io::scalar<int32_t>(prhs[3]));

	if (!cadet::mex::io::isEmpty(prhs[4]))
		drv.setWriteLastStateSens(cadet::mex::io::scalar<int32_t>(prhs[4]));
}

/**
 * @brief Clears all sensitivity settings from the simulator
 * @details Requires an already configured model and simulator.
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void clearSensitivities(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nlhs != 0)
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'clearsens' does not return anything.\n");

	if (drv.simulator())
		drv.simulator()->clearSensParams();
}

/**
 * @brief Returns the last and total accumulated simulation time
 * @param [in] drv Driver
 * @param [in] nlhs Number of left hand side (output) arguments
 * @param [out] plhs List with output arguments
 * @param [in] nrhs Number of right hand side (input) arguments
 * @param [in] prhs List with input arguments
 */
void getSimulationTime(cadet::Driver& drv, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
	if (nrhs != 2)
		mexWarnMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'getsimtime' ignores additional arguments.\n");
	if ((nlhs == 0) || (nlhs > 2))
		mexErrMsgIdAndTxt("CADET:mexError", "CadetMex: Command 'getsimtime' requires one or two outputs.\n");

	cadet::ISimulator* const sim = drv.simulator();
	if (sim)
	{
		plhs[0] = mxCreateDoubleScalar(sim->lastSimulationDuration());
		if (nlhs == 2)
			plhs[1] = mxCreateDoubleScalar(sim->totalSimulationDuration());
	}
	else
	{
		plhs[0] = mxCreateDoubleScalar(0.0);
		if (nlhs == 2)
			plhs[1] = mxCreateDoubleScalar(0.0);
	}
}

} // namespace command

CommandMap registeredCommands()
{
	CommandMap map;

	map["clearsim"] = &command::clearSimulator;
	map["isconf"] = &command::isConfigured;
	map["conf"] = &command::configure;
	map["run"] = &command::run;
	map["getallpar"] = &command::availableParameters;
	map["checkpar"] = &command::checkParameters;
	map["setparval"] = &command::setParameters;
	map["setsensparval"] = &command::setSensitiveParameterValues;
	map["setsensparfactor"] = &command::setSensitiveParameterFactors;
	map["setconsinitmode"] = &command::setConsistentInitializationMode;
	map["rerun"] = &command::reRun;
	map["reconf"] = &command::reconfigureModelOrSimulator;
	map["setreturnconf"] = &command::setReturnConfiguration;
	map["settimeintopts"] = &command::setTimeIntegratorOptions;
	map["settimeintsolveropts"] = &command::setTimeIntegratorSolverOptions;
	map["clearsens"] = &command::clearSensitivities;
	map["setsenspar"] = &command::makeParameterSensitive;
	map["setsenserror"] = &command::setSensitivityErrorTolerance;
	map["setsectimes"] = &command::setSectionTimes;
	map["setsoltimes"] = &command::setSolutionTimes;
	map["setnumthreads"] = &command::setNumThreads;
	map["setwritetimeandlaststate"] = &command::setWriteTimeAndLastState;
	map["getsimtime"] = &command::getSimulationTime;

	return map;
}

} // namespace mex
} // namespace cadet
