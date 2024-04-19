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
#include <random>
#include <chrono>
#include <limits>
#include <cmath>
#include <stdexcept>

#include <tclap/CmdLine.h>
#include "common/TclapUtils.hpp"
#include "io/hdf5/HDF5Writer.hpp"
#include "ToolsHelper.hpp"


struct ProgramOptions
{
	std::string fileName;
	bool isKinetic;
	bool solverTimes;
	double startTime;
	double endTime;
	double constAlg;
	double stddevAlg;
	bool reverseFlow;
	bool adJacobian;
	int nPar;
	int nCol;
	int nRad;
	int nParType;
	int nThreads;
	std::vector<std::string> sensitivities;
	std::string outSol;
	std::string outSens;
	std::string unitType;
};

void configureDiscretization(cadet::io::HDF5Writer& writer, int nCol, int nParType, int nPar, int nRad, bool adJacobian)
{
    Scope<cadet::io::HDF5Writer> s2(writer, "discretization");

    writer.scalar<int>("NCOL", nCol); // 64
    if (nParType > 1)
    {
        const std::vector<int> nPar_vec(nParType, nPar);
        writer.vector<int>("NPAR", nPar_vec.size(), nPar_vec.data()); // 16
    }
    else
        writer.scalar<int>("NPAR", nPar);

    if (nRad > 1)
        writer.scalar<int>("NRAD", nRad);
        writer.scalar("RADIAL_DISC_TYPE", std::string("EQUIDISTANT"));

    if (nParType > 1)
    {
        std::vector<std::string> parDiscType(nParType, std::string("EQUIDISTANT_PAR"));
        writer.vector<std::string>("PAR_DISC_TYPE", parDiscType.size(), parDiscType.data());
    }
    else
        writer.scalar("PAR_DISC_TYPE", std::string("EQUIDISTANT_PAR"));

    writer.scalar<int>("USE_ANALYTIC_JACOBIAN", !adJacobian);
    writer.scalar<int>("MAX_KRYLOV", 0);
    writer.scalar<int>("GS_TYPE", 1);
    writer.scalar<int>("MAX_RESTARTS", 10);
    writer.scalar<double>("SCHUR_SAFETY", 1e-8);

    // WENO
    {
        Scope<cadet::io::HDF5Writer> s3(writer, "weno");

        writer.scalar<int>("WENO_ORDER", 3);
        writer.scalar<int>("BOUNDARY_MODEL", 0);
        writer.scalar<double>("WENO_EPS", 1e-10);
    }
}

void configureParticles(cadet::io::HDF5Writer& writer, int nParType)
{
    const double par_radius = 4.5e-5;
    const double par_coreradius = 0.0;
    const double par_porosity = 0.75;

    writer.scalar<int>("NPARTYPE", nParType);

    if (nParType > 1)
    {
        std::vector<std::string> par_geom;
        std::vector<double> par_radii;
        std::vector<double> par_coreradii;
        std::vector<double> par_porosities;
        std::vector<double> par_volfrac;
        for (int i = 0; i < nParType; ++i) {
            par_radii.push_back(par_radius);
            par_geom.push_back("SPHERE");
            par_coreradii.push_back(par_coreradius);
            par_porosities.push_back(par_porosity);
            par_volfrac.push_back(1.0 / static_cast<double>(nParType));
        }

        writer.vector<std::string>("PAR_GEOM", par_geom.size(), par_geom.data());
        writer.vector<double>("PAR_RADIUS", par_radii.size(), par_radii.data());
        writer.vector<double>("PAR_CORERADIUS", par_coreradii.size(), par_coreradii.data());
        writer.vector<double>("PAR_POROSITY", par_porosities.size(), par_porosities.data());
        writer.vector<double>("PAR_TYPE_VOLFRAC", nParType, par_volfrac.data());
    }
    else
    {
        writer.scalar("PAR_GEOM", std::string("SPHERE"));
        writer.scalar<double>("PAR_RADIUS", par_radius);
        writer.scalar<double>("PAR_CORERADIUS", par_coreradius);
        writer.scalar<double>("PAR_POROSITY", par_porosity);
    }
}

void configureAdsorption(cadet::io::HDF5Writer& writer, int nParType, bool isKinetic)
{
    const std::vector<int> nBound(4 * nParType, 1);
    writer.vector<int>("NBOUND", nBound.size(), nBound.data());

    if (nParType > 1)
    {
        std::vector<std::string> adsorption_models;
        for (int i = 0; i < nParType; ++i) {
            adsorption_models.push_back("STERIC_MASS_ACTION");
        }
        writer.vector<std::string>("ADSORPTION_MODEL", adsorption_models.size(), adsorption_models.data());
        writer.scalar<int>("ADSORPTION_MODEL_MULTIPLEX", 0);
    }
    else
        writer.scalar("ADSORPTION_MODEL", std::string("STERIC_MASS_ACTION"));

    const double kA[] = { 0.0, 35.5, 1.59, 7.7 };
    const double kD[] = { 0.0, 1000.0, 1000.0, 1000.0 };
    const double smaLambda = 1.2e3;
    const double nu[] = { 0.0, 4.7, 5.29, 3.7 };
    const double sigma[] = { 0.0, 11.83, 10.6, 10.0 };


    for (int i = 0; i < nParType; ++i)
    {
        if (nParType > 1)
        {
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(3) << i;
            std::string parIdx = ss.str();
            Scope<cadet::io::HDF5Writer> s2(writer, "adsorption_" + parIdx);

            writer.scalar<int>("IS_KINETIC", isKinetic);
            writer.vector<double>("SMA_KA", 4, kA);
            writer.vector<double>("SMA_KD", 4, kD);
            writer.scalar<double>("SMA_LAMBDA", smaLambda);
            writer.vector<double>("SMA_NU", 4, nu);
            writer.vector<double>("SMA_SIGMA", 4, sigma);
        }
        else
        {
            Scope<cadet::io::HDF5Writer> s2(writer, "adsorption");

            writer.scalar<int>("IS_KINETIC", isKinetic);
            writer.vector<double>("SMA_KA", 4, kA);
            writer.vector<double>("SMA_KD", 4, kD);
            writer.scalar<double>("SMA_LAMBDA", smaLambda);
            writer.vector<double>("SMA_NU", 4, nu);
            writer.vector<double>("SMA_SIGMA", 4, sigma);
        }
    }
}

void configureFilmDiffusion(cadet::io::HDF5Writer& writer, int nComp, int nParType)
{
    const double filmDiff[] = { 6.9e-6, 6.9e-6, 6.9e-6, 6.9e-6 };
    if (nParType > 1)
    {
        writer.scalar<int>("FILM_DIFFUSION_MULTIPLEX", 2); // component and particle type dependent, section independent

        std::vector<double> filmDiffMultiplex;

        for (int i = 0; i < nParType; ++i) {
            filmDiffMultiplex.insert(filmDiffMultiplex.end(), filmDiff, filmDiff + nComp);
        }

        writer.vector<double>("FILM_DIFFUSION", filmDiffMultiplex.size(), filmDiffMultiplex.data());
    }
    else
    {
        writer.scalar<int>("FILM_DIFFUSION_MULTIPLEX", 0); // componenent dependent, particle type and section independent
        writer.vector<double>("FILM_DIFFUSION", 4, filmDiff);
    }

}

void configurePoreDiffusion(cadet::io::HDF5Writer& writer, int nComp, int nParType)
{
    const double parDiff[] = { 7e-10, 6.07e-11, 6.07e-11, 6.07e-11 };

    if (nParType > 1)
    {
        std::vector<double> parDiffMultiplex;

        for (int i = 0; i < nParType; ++i) {
            parDiffMultiplex.insert(parDiffMultiplex.end(), parDiff, parDiff + nComp);
        }

        writer.vector<double>("PAR_DIFFUSION", parDiffMultiplex.size(), parDiffMultiplex.data());
    }
    else
    {
        writer.vector<double>("PAR_DIFFUSION", 4, parDiff);
    }

}

void configureSurfaceDiffusion(cadet::io::HDF5Writer& writer, int nComp, int nParType)
{
    const double parSurfDiff[] = { 0.0, 0.0, 0.0, 0.0 };

    if (nParType > 1)
    {
        std::vector<double> parSurfDiffMultiplex;

        for (int i = 0; i < nParType; ++i) {
            parSurfDiffMultiplex.insert(parSurfDiffMultiplex.end(), parSurfDiff, parSurfDiff + nComp);
        }

        writer.vector<double>("PAR_SURFDIFFUSION", parSurfDiffMultiplex.size(), parSurfDiffMultiplex.data());
    }
    else
    {
        writer.vector<double>("PAR_SURFDIFFUSION", 4, parSurfDiff);
    }

}

void configureFlowDirection(cadet::io::HDF5Writer& writer, bool reverseFlow)
{
    if (!reverseFlow)
        writer.scalar<double>("VELOCITY", 1);
    else
        writer.scalar<double>("VELOCITY", -1);
}

void configureUnitSolver(cadet::io::HDF5Writer& writer)
{
    Scope<cadet::io::HDF5Writer> su(writer, "solver");

    writer.scalar<int>("MAX_KRYLOV", 0);
    writer.scalar<int>("GS_TYPE", 1);
    writer.scalar<int>("MAX_RESTARTS", 10);
    writer.scalar<double>("SCHUR_SAFETY", 1e-8);
}

void configureCstr(cadet::io::HDF5Writer& writer, ProgramOptions& opts, int nComp)
{
    writer.scalar<double>("POROSITY", 0.37 + (1.0 - 0.37) * 0.75);

    configureAdsorption(writer, opts.nParType, opts.isKinetic);
    writer.scalar<double>("INIT_VOLUME", 1e-3);
}

void configureLRM(cadet::io::HDF5Writer& writer, ProgramOptions& opts, int nComp)
{
    writer.scalar<double>("COL_LENGTH", 0.014);
    writer.scalar<double>("COL_RADIUS", 0.01);

    writer.scalar<double>("TOTAL_POROSITY", 0.37 + (1.0 - 0.37) * 0.75);

    writer.scalar<double>("COL_DISPERSION", 5.75e-8);

    configureDiscretization(writer, opts.nCol, opts.nParType, opts.nPar, opts.nRad, opts.adJacobian);

    configureAdsorption(writer, opts.nParType, opts.isKinetic);

    configureFlowDirection(writer, opts.reverseFlow);
}

void configureLRMP(cadet::io::HDF5Writer& writer, ProgramOptions& opts, int nComp)
{
    writer.scalar<double>("COL_LENGTH", 0.014);
    writer.scalar<double>("COL_RADIUS", 0.01);
    writer.scalar<double>("COL_POROSITY", 0.37);

    writer.scalar<double>("COL_DISPERSION", 5.75e-8);

    configureDiscretization(writer, opts.nCol, opts.nParType, opts.nPar, opts.nRad, opts.adJacobian);

    configureParticles(writer, opts.nParType);
    configureAdsorption(writer, opts.nParType, opts.isKinetic);

    configureFilmDiffusion(writer, nComp, opts.nParType);

    configureFlowDirection(writer, opts.reverseFlow);
}

void configureGRM(cadet::io::HDF5Writer& writer, ProgramOptions& opts, int nComp)
{
    writer.scalar<double>("COL_LENGTH", 0.014);
    writer.scalar<double>("COL_RADIUS", 0.01);
    writer.scalar<double>("COL_POROSITY", 0.37);

    writer.scalar<double>("COL_DISPERSION", 5.75e-8);

    configureDiscretization(writer, opts.nCol, opts.nParType, opts.nPar, opts.nRad, opts.adJacobian);

    configureParticles(writer, opts.nParType);
    configureAdsorption(writer, opts.nParType, opts.isKinetic);

    configureFilmDiffusion(writer, nComp, opts.nParType);
    configurePoreDiffusion(writer, nComp, opts.nParType);
    configureSurfaceDiffusion(writer, nComp, opts.nParType);

    configureFlowDirection(writer, opts.reverseFlow);
}

void configure2DGRM(cadet::io::HDF5Writer& writer, ProgramOptions& opts, int nComp)
{
    writer.scalar<double>("COL_LENGTH", 0.014);
    writer.scalar<double>("COL_RADIUS", 0.01);
    writer.scalar<double>("COL_POROSITY", 0.37);

    writer.scalar<double>("COL_DISPERSION", 5.75e-8);
    writer.scalar<double>("COL_DISPERSION_RADIAL", 1e-6);

    configureDiscretization(writer, opts.nCol, opts.nParType, opts.nPar, opts.nRad, opts.adJacobian);

    configureParticles(writer, opts.nParType);
    configureAdsorption(writer, opts.nParType, opts.isKinetic);

    configureFilmDiffusion(writer, nComp, opts.nParType);
    configurePoreDiffusion(writer, nComp, opts.nParType);
    configureSurfaceDiffusion(writer, nComp, opts.nParType);

    configureFlowDirection(writer, opts.reverseFlow);
}

void configureInitialConditions(cadet::io::HDF5Writer& writer, int nParType)
{
    const double initC[] = {50.0, 0.0, 0.0, 0.0};
    const double initQ[] = {1.2e3, 0.0, 0.0, 0.0};
    writer.vector<double>("INIT_C", 4, initC);

    if (nParType > 1)
    {
        //std::vector<double> init_cps;
        std::vector<double> init_qs;

        for (int i = 0; i < nParType; ++i)
        {
            //init_cps.insert(init_cps.end(), initC, initC + 4);
            init_qs.insert(init_qs.end(), initQ, initQ + 4);
        }

        //writer.vector<double>("INIT_CP", init_cps.size(), init_qs.data());
        writer.vector<double>("INIT_Q", init_qs.size(), init_qs.data());
    }
    else
        writer.vector<double>("INIT_Q", 4, initQ);
}

void configureInlet(cadet::io::HDF5Writer& writer, double startTime)
{
    Scope<cadet::io::HDF5Writer> su(writer, "unit_001");

    writer.scalar("UNIT_TYPE", std::string("INLET"));
    writer.scalar("INLET_TYPE", std::string("PIECEWISE_CUBIC_POLY"));
    writer.scalar<int>("NCOMP", 4);

    if (startTime < 10.0)
    {
        {
            Scope<cadet::io::HDF5Writer> s3(writer, "sec_000");

            const double constCoeff[] = {50.0, 1.0, 1.0, 1.0};
            const double linCoeff[] = {0.0, 0.0, 0.0, 0.0};

            writer.vector<double>("CONST_COEFF", 4, constCoeff);
            writer.vector<double>("LIN_COEFF", 4, linCoeff);
            writer.vector<double>("QUAD_COEFF", 4, linCoeff);
            writer.vector<double>("CUBE_COEFF", 4, linCoeff);
        }

        {
            Scope<cadet::io::HDF5Writer> s3(writer, "sec_001");

            const double constCoeff[] = {50.0, 0.0, 0.0, 0.0};
            const double linCoeff[] = {0.0, 0.0, 0.0, 0.0};

            writer.vector<double>("CONST_COEFF", 4, constCoeff);
            writer.vector<double>("LIN_COEFF", 4, linCoeff);
            writer.vector<double>("QUAD_COEFF", 4, linCoeff);
            writer.vector<double>("CUBE_COEFF", 4, linCoeff);
        }

        {
            Scope<cadet::io::HDF5Writer> s3(writer, "sec_002");

            const double constCoeff[] = {100.0, 0.0, 0.0, 0.0};
            const double linCoeff[] = {0.2, 0.0, 0.0, 0.0};
            const double quadCoeff[] = {0.0, 0.0, 0.0, 0.0};

            writer.vector<double>("CONST_COEFF", 4, constCoeff);
            writer.vector<double>("LIN_COEFF", 4, linCoeff);
            writer.vector<double>("QUAD_COEFF", 4, quadCoeff);
            writer.vector<double>("CUBE_COEFF", 4, quadCoeff);
        }
    }
    else if (startTime < 90.0)
    {
        {
            Scope<cadet::io::HDF5Writer> s3(writer, "sec_000");

            const double constCoeff[] = {50.0, 0.0, 0.0, 0.0};
            const double linCoeff[] = {0.0, 0.0, 0.0, 0.0};

            writer.vector<double>("CONST_COEFF", 4, constCoeff);
            writer.vector<double>("LIN_COEFF", 4, linCoeff);
            writer.vector<double>("QUAD_COEFF", 4, linCoeff);
            writer.vector<double>("CUBE_COEFF", 4, linCoeff);
        }

        {
            Scope<cadet::io::HDF5Writer> s3(writer, "sec_001");

            const double constCoeff[] = {100.0, 0.0, 0.0, 0.0};
            const double linCoeff[] = {0.2, 0.0, 0.0, 0.0};
            const double quadCoeff[] = {0.0, 0.0, 0.0, 0.0};

            writer.vector<double>("CONST_COEFF", 4, constCoeff);
            writer.vector<double>("LIN_COEFF", 4, linCoeff);
            writer.vector<double>("QUAD_COEFF", 4, quadCoeff);
            writer.vector<double>("CUBE_COEFF", 4, quadCoeff);
        }
    }
    else if (startTime < 1500.0)
    {
        {
            Scope<cadet::io::HDF5Writer> s3(writer, "sec_000");

            const double constCoeff[] = {100.0 + (startTime - 90.0) * 0.2, 0.0, 0.0, 0.0};
            const double linCoeff[] = {0.2, 0.0, 0.0, 0.0};
            const double quadCoeff[] = {0.0, 0.0, 0.0, 0.0};

            writer.vector<double>("CONST_COEFF", 4, constCoeff);
            writer.vector<double>("LIN_COEFF", 4, linCoeff);
            writer.vector<double>("QUAD_COEFF", 4, quadCoeff);
            writer.vector<double>("CUBE_COEFF", 4, quadCoeff);
        }
    }
}

void configureValveSwitches(cadet::io::HDF5Writer& writer, bool hasPorts)
{
    Scope<cadet::io::HDF5Writer> su(writer, "connections");
    writer.scalar<int>("NSWITCHES", 1);
    writer.scalar<int>("CONNECTIONS_INCLUDE_PORTS", 1);

    {
        Scope<cadet::io::HDF5Writer> s1(writer, "switch_000");

        if (!hasPorts)
        {
            // Connection list is 1x7 since we have 1 connection between
            // the two unit operations (and we need to have 7 columns)
            const double connMatrix[] = {1, 0, -1, -1, -1, -1, 1e-8};
            // Connections: From unit operation 1 port -1 (i.e., all ports)
            //              to unit operation 0 port -1 (i.e., all ports),
            //              connect component -1 (i.e., all components)
            //              to component -1 (i.e., all components) with
            //              a flow rate of 1e-8 m^3/s

            writer.vector<double>("CONNECTIONS", 7, connMatrix);
        }
        else
        {
            // Connection list is 3x7 since we have 1 connection between
            // the two unit operations with 3 ports (and we need to have 7 columns)
            const double connMatrix[] = {1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 7.42637597e-09,
                                         1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 2.22791279e-08,
                                         1.0, 0.0, 0.0, 2.0, -1.0, -1.0, 3.71318798e-08};
            // Connections: From unit operation 1 port 0
            //              to unit operation 0 port 0,
            //              connect component -1 (i.e., all components)
            //              to component -1 (i.e., all components) with
            //              volumetric flow rate 7.42637597e-09 m^3/s

            writer.vector<double>("CONNECTIONS", 21, connMatrix);
        }

        // This switch occurs at beginning of section 0 (initial configuration)
        writer.scalar<int>("SECTION", 0);
    }
}

void configureReturn(cadet::io::HDF5Writer& writer, ProgramOptions& opts)
{
    Scope<cadet::io::HDF5Writer> s(writer, "return");
    writer.template scalar<int>("WRITE_SOLUTION_TIMES", true);

    Scope<cadet::io::HDF5Writer> s2(writer, "unit_000");
    parseAndWriteOutputFormatsFromCmdLine(writer, opts.outSol, opts.outSens);
}

void configureSolver(cadet::io::HDF5Writer& writer, ProgramOptions& opts)
{
    Scope<cadet::io::HDF5Writer> s(writer, "solver");

    if (!opts.solverTimes)
    {
        std::vector<double> solTimes;
        solTimes.reserve(1501);
        for (double t = 0.0; t <= opts.endTime - opts.startTime; t += 1.0)
            solTimes.push_back(t);

        writer.vector<double>("USER_SOLUTION_TIMES", solTimes.size(), solTimes.data());
    }

    writer.scalar<int>("NTHREADS", opts.nThreads);

    // Sections
    {
        Scope<cadet::io::HDF5Writer> s2(writer, "sections");

        if (opts.startTime < 10.0)
        {
            writer.scalar<int>("NSEC", 3);

            const double secTimes[] = {0.0, 10.0 - opts.startTime, 90.0 - opts.startTime, 1500.0 - opts.startTime};
            writer.vector<double>("SECTION_TIMES", 4, secTimes);

            const int secCont[] = {0, 0};
            writer.vector<int>("SECTION_CONTINUITY", 2, secCont);
        }
        else if (opts.startTime < 90.0)
        {
            writer.scalar<int>("NSEC", 2);

            const double secTimes[] = {0.0, 90.0 - opts.startTime, 1500.0 - opts.startTime};
            writer.vector<double>("SECTION_TIMES", 3, secTimes);

            const int secCont[] = {0, 0};
            writer.vector<int>("SECTION_CONTINUITY", 1, secCont);
        }
        else if (opts.startTime < 1500.0)
        {
            writer.scalar<int>("NSEC", 1);

            const double secTimes[] = {0.0, 1500.0 - opts.startTime};
            writer.vector<double>("SECTION_TIMES", 2, secTimes);
        }
    }

    // Time integrator
    {
        Scope<cadet::io::HDF5Writer> s2(writer, "time_integrator");

        writer.scalar<double>("ABSTOL", 1e-8);
        writer.scalar<double>("RELTOL", 1e-6);
        writer.scalar<double>("ALGTOL", 1e-12);
        writer.scalar<double>("INIT_STEP_SIZE", 1e-6);
        writer.scalar<int>("MAX_STEPS", 10000);
    }
}

void configureModel(cadet::io::HDF5Writer& writer, ProgramOptions& opts)
{
    Scope<cadet::io::HDF5Writer> s(writer, "model");

    writer.scalar<int>("NUNITS", 2);

    bool hasPorts = false;

    {
        Scope<cadet::io::HDF5Writer> su(writer, "unit_000");

        const int nComp = 4;
        writer.scalar<int>("NCOMP", nComp);

        parseUnitType(opts.unitType);
        writer.scalar("UNIT_TYPE", opts.unitType);

        // Unit operation parameters 
        if (opts.unitType == "CSTR")
            configureCstr(writer, opts, nComp);
        else if (opts.unitType == "LUMPED_RATE_MODEL_WITHOUT_PORES")
            configureLRM(writer, opts, nComp);
        else if (opts.unitType == "LUMPED_RATE_MODEL_WITH_PORES")
            configureLRMP(writer, opts, nComp);
        else if (opts.unitType == "GENERAL_RATE_MODEL")
            configureGRM(writer, opts, nComp);
        else if (opts.unitType == "GENERAL_RATE_MODEL_2D")
        {
            configure2DGRM(writer, opts, nComp);
            hasPorts = true;
        }
        else
            throw std::domain_error("Unknown unit operation type " + opts.unitType);

        configureInitialConditions(writer, opts.nParType);
    }

    configureInlet(writer, opts.startTime);
    configureValveSwitches(writer, hasPorts);
    configureUnitSolver(writer);
}

int main(int argc, char** argv)
{
	ProgramOptions opts;
	const double nanVal = std::numeric_limits<double>::quiet_NaN();

	try
	{
		TCLAP::CustomOutputWithoutVersion customOut("createLWE");
		TCLAP::CmdLine cmd("Create an HDF5 input file for load-wash-elution benchmark case", ' ', "1.0");
		cmd.setOutput(&customOut);

		cmd >> (new TCLAP::ValueArg<std::string>("o", "out", "Write output to file (default: LWE.h5)", false, "LWE.h5", "File"))->storeIn(&opts.fileName);
		cmd >> (new TCLAP::ValueArg<double>("t", "startTime", "Start time of simulation (default: 0sec)", false, 0.0, "Time"))->storeIn(&opts.startTime);
		cmd >> (new TCLAP::ValueArg<double>("T", "endTime", "End time of simulation (default: 1500sec)", false, 1500.0, "Time"))->storeIn(&opts.endTime);
		cmd >> (new TCLAP::ValueArg<double>("c", "constAlg", "Set all algebraic variables to constant value", false, nanVal, "Value"))->storeIn(&opts.constAlg);
		cmd >> (new TCLAP::ValueArg<double>("s", "stddevAlg", "Perturb algebraic variables with normal variates", false, nanVal, "Value"))->storeIn(&opts.stddevAlg);
		cmd >> (new TCLAP::SwitchArg("", "reverseFlow", "Reverse the flow for column"))->storeIn(&opts.reverseFlow);
		cmd >> (new TCLAP::ValueArg<int>("", "rad", "Number of radial cells (default: 3)", false, 3, "Value"))->storeIn(&opts.nRad);
		cmd >> (new TCLAP::ValueArg<int>("", "parTypes", "Number of particle types (default: 1)", false, 1, "Value"))->storeIn(&opts.nParType);
		addMiscToCmdLine(cmd, opts);
		addUnitTypeToCmdLine(cmd, opts.unitType);
		addSensitivitiyParserToCmdLine(cmd, opts.sensitivities);
		addOutputParserToCmdLine(cmd, opts.outSol, opts.outSens);

		cmd.parse(argc, argv);
	}
	catch (const TCLAP::ArgException &e)
	{
		std::cerr << "ERROR: " << e.error() << " for argument " << e.argId() << std::endl;
		return 1;
	}

	cadet::io::HDF5Writer writer;
	writer.openFile(opts.fileName, "co");
	writer.pushGroup("input");

    configureModel(writer, opts);
    configureReturn(writer, opts);
    configureSolver(writer, opts);

	parseAndWriteSensitivitiesFromCmdLine(writer, opts.sensitivities);

	writer.closeFile();
	return 0;
}

