// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include <catch.hpp>
#include "Approx.hpp"

#include "ColumnTests.hpp"

#include "Utils.hpp"
#include "JsonTestModels.hpp"
#include <common/Driver.hpp>

#include "../src/cadet-cli/Logging.hpp" // todo: this is only needed because LOGs are written in the parameterProviderImpl. Should probably already be included through cmake?
#include "../include/io/hdf5/HDF5Reader.hpp"
#include "common/ParameterProviderImpl.hpp"

//#include "LoggingUtils.hpp"
//#include "Logging.hpp"

// TODO
// todo: doesnt build in debug mode, unresolved external symbol RunTimeFilteringLogger (defined in include/common/LoggerBase.hpp)
// todo: why include looging, when its not required elsewhere when ParameterProviderImpl is being used in the code
// todo: include HDF::HDF in cmake is not elegant/repetitive?


// Linear interpolation function, naive approach
inline double lerp(double a, double b, double t) {
	return a + t * (b - a);
}
 
// Function to interpolate a 1D vector
inline std::vector<double> interpolate(const std::vector<double>& inputVector, size_t outputSize) {
	std::vector<double> outputVector(outputSize);

	double inputStep = 1.0 / (inputVector.size() - 1);
	double outputStep = 1.0 / (outputSize - 1);

	for (size_t i = 0; i < outputSize; ++i) {
		double t = i * outputStep;

		// Find the two adjacent points in the input vector
		size_t index1 = static_cast<size_t>(t / inputStep);
		size_t index2 = std::min(index1 + 1, inputVector.size() - 1);

		// Perform linear interpolation
		outputVector[i] = lerp(inputVector[index1], inputVector[index2], t / inputStep);
	}

	return outputVector;
}

const char* getTestDirectory(); // test file dir
cadet::io::HDF5Reader rd;  // h5 reader

/**
 * @brief Prepare the simulation results
 * @return The simulation results
 */
std::vector<double> getSim(const std::string& simFilePath, const int& time_resolution) {
	rd.openFile(simFilePath, "r");

	cadet::ParameterProviderImpl<cadet::io::HDF5Reader> pp(rd);

	pp.popScope();
	pp.pushScope("input");
	pp.pushScope("model");
	pp.pushScope("unit_001");

	// number of component
	const int Ncomp_sim = pp.getInt("NCOMP");
	pp.popScope();
	pp.popScope();

	// use h5 input to setup and run simulation
	cadet::Driver drv;

	drv.configure(pp);
	drv.run();

	// pointer to sim solution
	cadet::InternalStorageUnitOpRecorder const* const simData = drv.solution()->unitOperation(1);
	double const* sim_n_full_pr = simData->outlet();

	rd.closeFile();

	// construct the sim solution at the last time without ceq and c, assumes identical time_res
	std::vector<double> sim_n;
	for (unsigned int i = 0; i < Ncomp_sim * time_resolution; ++sim_n_full_pr, ++i)
	{
		if ((i > Ncomp_sim * (time_resolution - 1)) && (i < Ncomp_sim * time_resolution - 1)) {
			sim_n.emplace_back(*sim_n_full_pr);
		}
		else {
			continue;
		}
	}

	return sim_n;
}

/**
 * @brief Prepare the reference results. All reference solutions in CSTRs are obtained using 1000 cells, WENO23. In DPFR, 100 X 200(N_x X Ncol) WENO23-WENO23 is used. 
 * @return The full reference solution
 */
const std::vector<double> getRef(const std::string& refFilePath, int& time_res) {
	rd.openFile(refFilePath, "r");

	cadet::ParameterProviderImpl<cadet::io::HDF5Reader> ppr(rd);

	ppr.popScope();
	ppr.pushScope("input");
	ppr.pushScope("model");
	ppr.pushScope("unit_001");

	// change the time resolution
	time_res = ppr.getInt("TIME_RES");

	ppr.popScope();
	ppr.popScope();
	ppr.popScope();

	ppr.pushScope("output");
	ppr.pushScope("solution");
	ppr.pushScope("unit_001");

	// full ref solution 
	const std::vector<double> ref_n = ppr.getDoubleArray("SOLUTION_OUTLET");

	rd.closeFile();

	return ref_n;
}


/**
 * @brief Test growth and primary nucleation in a CSTR
 */
TEST_CASE("Crystillization test CSTR, primary nucleation and growth", "[CRYSTALLIZATION],[Simulation],[primaryNucAndGrowthCSTR]")
{
	int time_res;

	// reference solution
	const std::string refFile = std::string(getTestDirectory()) + std::string("/data/cry_PNG_CSTR_ref.h5");
	const std::vector<double> ref_n_full = getRef(refFile, time_res);

	// simulation solution
	const std::string simFile = std::string(getTestDirectory()) + std::string("/data/cry_PNG_CSTR_sim.h5");
	std::vector<double> sim_n_full = getSim(simFile, time_res);

	// interpolate the ref solution
	std::vector<double> n_ref_interp = interpolate(ref_n_full, sim_n_full.size());

	// compare the results
	for (unsigned int i = 0; i < sim_n_full.size(); ++i)
	{
		CHECK((sim_n_full[i]) == cadet::test::makeApprox(n_ref_interp[i], 1e-3, 1e16));  // reltol, abstol
	}
}

/**
 * @brief Test growth and primary and secondary nucleation in a CSTR
 */
TEST_CASE("Crystillization test CSTR, primary and secondary nucleation and growth", "[CRYSTALLIZATION],[Simulation],[priSecNucAndGrowthCSTR]")
{
	int time_res;

	// reference solution
	const std::string refFile = std::string(getTestDirectory()) + std::string("/data/cry_PSNG_CSTR_ref.h5");
	const std::vector<double> ref_n_full = getRef(refFile, time_res);

	// simulation solution
	const std::string simFile = std::string(getTestDirectory()) + std::string("/data/cry_PSNG_CSTR_sim.h5");
	std::vector<double> sim_n_full = getSim(simFile, time_res);

	// interpolate the ref solution
	std::vector<double> n_ref_interp = interpolate(ref_n_full, sim_n_full.size());

	// compare the results
	for (unsigned int i = 0; i < sim_n_full.size(); ++i)
	{
		CHECK((sim_n_full[i]) == cadet::test::makeApprox(n_ref_interp[i], 1e-3, 1e16));  // reltol, abstol
	}
}

/**
 * @brief Test growth, growth rate dispersion and primary nucleation in a CSTR
 */
TEST_CASE("Crystillization test CSTR, primary nucleation and growth and growth rate dispersion", "[CRYSTALLIZATION],[Simulation],[pricNucGrowthAndDispersionCSTR]")
{
	int time_res;

	// reference solution
	const std::string refFile = std::string(getTestDirectory()) + std::string("/data/cry_PNGGD_CSTR_ref.h5");
	const std::vector<double> ref_n_full = getRef(refFile, time_res);

	// simulation solution
	const std::string simFile = std::string(getTestDirectory()) + std::string("/data/cry_PNGGD_CSTR_sim.h5");
	std::vector<double> sim_n_full = getSim(simFile, time_res);

	// interpolate the ref solution
	std::vector<double> n_ref_interp = interpolate(ref_n_full, sim_n_full.size());

	// compare the results
	for (unsigned int i = 0; i < sim_n_full.size(); ++i)
	{
		CHECK((sim_n_full[i]) == cadet::test::makeApprox(n_ref_interp[i], 1e-3, 1e15));  // reltol, abstol
	}
}

/**
 * @brief Test growth, growth rate dispersion, primary and secondary nucleation in a DPFR. This test does not work now. 
 */
TEST_CASE("Crystillization test DPFR with axial dispersion, primary secondary nucleation and growth and growth rate dispersion", "[CRYSTALLIZATION],[Simulation],[priSecNucGrowthAndDispersionDPFR]")
{
	int time_res;

	// reference solution
	const std::string refFile = std::string(getTestDirectory()) + std::string("/data/practice1_ref.h5");
	const std::vector<double> ref_n_full = getRef(refFile, time_res);

	// simulation solution
	const std::string simFile = std::string(getTestDirectory()) + std::string("/data/practice1_sim.h5");
	std::vector<double> sim_n_full = getSim(simFile, time_res);

	// interpolate the ref solution
	std::vector<double> n_ref_interp = interpolate(ref_n_full, sim_n_full.size());

	// compare the results
	for (unsigned int i = 0; i < sim_n_full.size(); ++i)
	{
		CHECK((sim_n_full[i]) == cadet::test::makeApprox(n_ref_interp[i], 1e-3, 1e15));  // reltol, abstol
	}
}


// TODO: add Jacobian tests
