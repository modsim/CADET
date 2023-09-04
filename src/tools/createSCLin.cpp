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

#include <sstream>
#include <iostream>
#include <string>
#include <vector>

#include <tclap/CmdLine.h>
#include "common/TclapUtils.hpp"
#include "io/hdf5/HDF5Writer.hpp"
#include "ToolsHelper.hpp"

struct ProgramOptions
{
	std::string fileName;
	bool isKinetic;
	std::vector<std::string> sensitivities;
	std::string outSol;
	std::string outSens;
};

int main(int argc, char** argv)
{
	ProgramOptions opts;

	try
	{
		TCLAP::CustomOutputWithoutVersion customOut("createSCL");
		TCLAP::CmdLine cmd("Create an HDF5 input file for a single component linear benchmark case", ' ', "1.0");
		cmd.setOutput(&customOut);

		cmd >> (new TCLAP::ValueArg<std::string>("o", "out", "Write output to file (default: SCLin.h5)", false,
												 "SCLin.h5", "File"))
				   ->storeIn(&opts.fileName);
		cmd >> (new TCLAP::SwitchArg("k", "kinetic", "Kinetic adsorption model used (default: quasi-stationary)"))
				   ->storeIn(&opts.isKinetic);
		addSensitivitiyParserToCmdLine(cmd, opts.sensitivities);
		addOutputParserToCmdLine(cmd, opts.outSol, opts.outSens);

		cmd.parse(argc, argv);
	}
	catch (const TCLAP::ArgException& e)
	{
		std::cerr << "ERROR: " << e.error() << " for argument " << e.argId() << std::endl;
		return 1;
	}

	cadet::io::HDF5Writer writer;
	writer.openFile(opts.fileName, "co");
	writer.pushGroup("input");

	// Model
	{
		Scope<cadet::io::HDF5Writer> s(writer, "model");
		writer.scalar<int>("NUNITS", 2);

		// Column - unit 000
		{
			Scope<cadet::io::HDF5Writer> su(writer, "unit_000");

			writer.scalar("UNIT_TYPE", std::string("GENERAL_RATE_MODEL"));
			writer.scalar<int>("NCOMP", 1);

			// Transport
			writer.scalar<double>("VELOCITY", 5.75e-4);
			writer.scalar<double>("COL_DISPERSION", 5.75e-8);

			const double filmDiff[] = {6.9e-6};
			const double parDiff[] = {7e-10};
			const double parSurfDiff[] = {0.0};

			writer.vector<double>("FILM_DIFFUSION", 1, filmDiff);
			writer.vector<double>("PAR_DIFFUSION", 1, parDiff);
			writer.vector<double>("PAR_SURFDIFFUSION", 1, parSurfDiff);

			// Geometry
			writer.scalar<double>("COL_LENGTH", 0.014);
			writer.scalar<double>("PAR_RADIUS", 4.5e-5);
			writer.scalar<double>("PAR_CORERADIUS", 0.0);
			writer.scalar<double>("COL_POROSITY", 0.37);
			writer.scalar<double>("PAR_POROSITY", 0.75);

			// Initial conditions
			const double initC[] = {0.0};
			const double initQ[] = {0.0};
			writer.vector<double>("INIT_C", 1, initC);
			writer.vector<double>("INIT_Q", 1, initQ);

			// Adsorption
			writer.scalar("ADSORPTION_MODEL", std::string("LINEAR"));
			{
				Scope<cadet::io::HDF5Writer> s2(writer, "adsorption");

				if (opts.isKinetic)
					writer.scalar<int>("IS_KINETIC", 1);
				else
					writer.scalar<int>("IS_KINETIC", 0);

				const double kA[] = {35.5};
				const double kD[] = {1000.0};
				writer.vector<double>("LIN_KA", 1, kA);
				writer.vector<double>("LIN_KD", 1, kD);
			}

			// Discretization
			{
				Scope<cadet::io::HDF5Writer> s2(writer, "discretization");

				writer.scalar<int>("NCOL", 10); // 64
				writer.scalar<int>("NPAR", 4);  // 16
				const int nBound[] = {1};
				writer.vector<int>("NBOUND", 1, nBound);

				writer.scalar("PAR_DISC_TYPE", std::string("EQUIDISTANT_PAR"));

				writer.scalar<int>("USE_ANALYTIC_JACOBIAN", 1);
				writer.scalar<int>("MAX_KRYLOV", 0);
				writer.scalar<int>("GS_TYPE", 1);
				writer.scalar<int>("MAX_RESTARTS", 10);
				writer.scalar<double>("SCHUR_SAFETY", 1e-8);

				// WENO
				{
					Scope<cadet::io::HDF5Writer> s3(writer, "weno");

					writer.scalar<int>("WENO_ORDER", 3);
					writer.scalar<int>("BOUNDARY_MODEL", 0);
					writer.scalar<double>("WENO_EPS", 1e-12);
				}
			}
		}

		// Inlet - unit 001
		{
			Scope<cadet::io::HDF5Writer> su(writer, "unit_001");

			writer.scalar("UNIT_TYPE", std::string("INLET"));
			writer.scalar("INLET_TYPE", std::string("PIECEWISE_CUBIC_POLY"));
			writer.scalar<int>("NCOMP", 1);

			{
				Scope<cadet::io::HDF5Writer> s3(writer, "sec_000");

				const double constCoeff[] = {0.0};
				const double linCoeff[] = {1.0};
				const double quadCoeff[] = {0.0};

				writer.vector<double>("CONST_COEFF", 1, constCoeff);
				writer.vector<double>("LIN_COEFF", 1, linCoeff);
				writer.vector<double>("QUAD_COEFF", 1, quadCoeff);
				writer.vector<double>("CUBE_COEFF", 1, quadCoeff);
			}

			{
				Scope<cadet::io::HDF5Writer> s3(writer, "sec_001");

				const double constCoeff[] = {10.0};
				const double linCoeff[] = {0.0};

				writer.vector<double>("CONST_COEFF", 1, constCoeff);
				writer.vector<double>("LIN_COEFF", 1, linCoeff);
				writer.vector<double>("QUAD_COEFF", 1, linCoeff);
				writer.vector<double>("CUBE_COEFF", 1, linCoeff);
			}

			{
				Scope<cadet::io::HDF5Writer> s3(writer, "sec_002");

				const double constCoeff[] = {10.0};
				const double linCoeff[] = {-0.00709219858};
				const double quadCoeff[] = {0.0};

				writer.vector<double>("CONST_COEFF", 1, constCoeff);
				writer.vector<double>("LIN_COEFF", 1, linCoeff);
				writer.vector<double>("QUAD_COEFF", 1, quadCoeff);
				writer.vector<double>("CUBE_COEFF", 1, quadCoeff);
			}
		}

		// Valve switches
		{
			Scope<cadet::io::HDF5Writer> su(writer, "connections");
			writer.scalar<int>("NSWITCHES", 1);
			writer.scalar<int>("CONNECTIONS_INCLUDE_PORTS", 1);

			{
				Scope<cadet::io::HDF5Writer> s1(writer, "switch_000");

				// Connection list is 1x7 since we have 1 connection between
				// the two unit operations (and we need to have 7 columns)
				const double connMatrix[] = {1, 0, -1, -1, -1, -1, 1.0};
				// Connections: From unit operation 1 port -1 (i.e., all ports)
				//              to unit operation 0 port -1 (i.e., all ports),
				//              connect component -1 (i.e., all components)
				//              to component -1 (i.e., all components) with
				//              a flow rate of 1.0

				// This switch occurs at beginning of section 0 (initial configuration)
				writer.scalar<int>("SECTION", 0);
				writer.vector<double>("CONNECTIONS", 7, connMatrix);
			}
		}

		// Solver settings
		{
			Scope<cadet::io::HDF5Writer> su(writer, "solver");

			writer.scalar<int>("MAX_KRYLOV", 0);
			writer.scalar<int>("GS_TYPE", 1);
			writer.scalar<int>("MAX_RESTARTS", 10);
			writer.scalar<double>("SCHUR_SAFETY", 1e-8);
		}
	}

	// Return
	{
		Scope<cadet::io::HDF5Writer> s(writer, "return");
		writer.template scalar<int>("WRITE_SOLUTION_TIMES", true);

		Scope<cadet::io::HDF5Writer> s2(writer, "unit_000");
		parseAndWriteOutputFormatsFromCmdLine(writer, opts.outSol, opts.outSens);
	}

	// Solver
	{
		Scope<cadet::io::HDF5Writer> s(writer, "solver");

		std::vector<double> solTimes;
		solTimes.reserve(1501);
		for (int t = 0; t <= 1500; t += 1)
			//		solTimes.reserve(101);
			//		for (int t = 0; t <= 100; t += 1)
			//		for (double t = 0; t <= 0.01; t += 0.001)
			solTimes.push_back(t);

		writer.vector<double>("USER_SOLUTION_TIMES", solTimes.size(), solTimes.data());

		writer.scalar<int>("NTHREADS", 1);

		// Sections
		{
			Scope<cadet::io::HDF5Writer> s2(writer, "sections");
			writer.scalar<int>("NSEC", 3);

			const double secTimes[] = {0.0, 10.0, 90.0, 1500.0};
			writer.vector<double>("SECTION_TIMES", 4, secTimes);

			const int secCont[] = {1, 1};
			writer.vector<int>("SECTION_CONTINUITY", 2, secCont);
		}

		// Time integrator
		{
			Scope<cadet::io::HDF5Writer> s2(writer, "time_integrator");

			writer.scalar<double>("ABSTOL", 1e-8);
			writer.scalar<double>("RELTOL", 1e-5);
			writer.scalar<double>("ALGTOL", 1e-12);
			writer.scalar<double>("INIT_STEP_SIZE", 1e-6);
			writer.scalar<int>("MAX_STEPS", 10000);
		}
	}

	parseAndWriteSensitivitiesFromCmdLine(writer, opts.sensitivities);

	writer.closeFile();
	return 0;
}
