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
// 
// tests
// todo: add benchmark tests for all crystallization submodels/features
// cleanup: FLAG FlynnsTest is just to launch it, dont put it on github

/**
 * @brief Returns the absolute path to the test/ folder of the project
 * @details Absolute path to the test/ folder of the project without trailing slash
 * @return Absolute path to the test/ folder
 */

// Linear interpolation function, naive approach
double lerp(double a, double b, double t) {
	return a + t * (b - a);
}
 
// Function to interpolate a 1D vector
std::vector<double> interpolate(const std::vector<double>& inputVector, size_t outputSize) {
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

const char* getTestDirectory();

TEST_CASE("Crystillization test CSTR, nucleation and growth", "[CRYSTALLIZATION],[Simulation],[firstTest]")
{
	cadet::io::HDF5Reader rd;

	// configure the simulations
	const std::string simFile = std::string(getTestDirectory()) + std::string("/data/cry_NGD_CSTR_sim.h5");

	rd.openFile(simFile, "r");

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

	// read reference solutions
	const std::string refFile = std::string(getTestDirectory()) + std::string("/data/cry_NGD_CSTR_ref.h5");

	rd.openFile(refFile, "r");

	cadet::ParameterProviderImpl<cadet::io::HDF5Reader> ppr(rd);

	ppr.popScope();
	ppr.pushScope("input");
	ppr.pushScope("model");
	ppr.pushScope("unit_001");

	// number of component and time resolution
	const int time_res = ppr.getInt("TIME_RES");

	ppr.popScope();
	ppr.popScope();
	ppr.popScope();

	ppr.pushScope("output");
	ppr.pushScope("solution");
	ppr.pushScope("unit_001");

	// full ref solution 
	const std::vector<double> ref_n = ppr.getDoubleArray("SOLUTION_OUTLET");

	// prepare the sim solution at the last time without ceq and c, assumes identical time_res
	std::vector<double> sim_n;
	for (unsigned int i = 0; i < Ncomp_sim * time_res; ++sim_n_full_pr, ++i) 
	{
		if ((i > Ncomp_sim * (time_res - 1)) && (i < Ncomp_sim * time_res - 1)) {
			sim_n.emplace_back(*sim_n_full_pr);
		}
		else {
			continue;
		}
	}

	// std::cout << "SIZE sim_n: " << sim_n.size() << std::endl;
    // std::cout << "the first ref_n is " << ref_n[0] << std::endl;
	// std::cout << "the first sim_n is " << sim_n[0] << std::endl;

	rd.closeFile();

	// interpolate the ref solution
	size_t outputSize = sim_n.size();
	std::vector<double> n_ref_interp = interpolate(ref_n, outputSize);
	
	// Compare
	for (unsigned int i = 0; i < sim_n.size(); ++i)
	{
		CHECK((sim_n[i]) == cadet::test::makeApprox(n_ref_interp[i], 9.9999e-1, 1e15));
	}
}


//TEST_CASE("Crystillization test DPFR", "[CRYSTALLIZATION],[Simulation]")
//{
//
//}
//
//TEST_CASE("Crystillization test critical nucleation", "[CRYSTALLIZATION],[Simulation]")
//{
//
//}
//
//TEST_CASE("Crystillization test propability density function nucleation", "[CRYSTALLIZATION],[Simulation]")
//{
//
//}

// add Jacobian tests
