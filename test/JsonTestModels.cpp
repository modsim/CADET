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

#include <json.hpp>

#include "common/JsonParameterProvider.hpp"

using json = nlohmann::json;

json createColumnWithSMAJson(const std::string& uoType)
{
	json config;
	config["UNIT_TYPE"] = uoType;
	config["NCOMP"] = 4;
	config["VELOCITY"] = 5.75e-4;
	config["COL_DISPERSION"] = 5.75e-8;
	config["COL_DISPERSION_RADIAL"] = 1e-6;
	config["FILM_DIFFUSION"] = {6.9e-6, 6.9e-6, 6.9e-6, 6.9e-6};
	config["PAR_DIFFUSION"] = {7e-10, 6.07e-11, 6.07e-11, 6.07e-11};
	config["PAR_SURFDIFFUSION"] = {0.0, 0.0, 0.0, 0.0};

	// Geometry
	config["COL_LENGTH"] = 0.014;
	config["COL_RADIUS"] = 0.01;
	config["COL_RADIUS_INNER"] = 0.001;
	config["COL_RADIUS_OUTER"] = 0.004;
	config["CROSS_SECTION_AREA"] = 0.0003141592653589793;
	config["PAR_RADIUS"] = 4.5e-5;
	config["COL_POROSITY"] = 0.37;
	config["PAR_POROSITY"] = 0.75;
	config["TOTAL_POROSITY"] = 0.37 + (1.0 - 0.37) * 0.75;

	// Initial conditions
	config["INIT_C"] = {50.0, 0.0, 0.0, 0.0};
	config["INIT_Q"] = {1.2e3, 0.0, 0.0, 0.0};

	// Adsorption
	config["ADSORPTION_MODEL"] = std::string("STERIC_MASS_ACTION");
	{
		json ads;
		ads["IS_KINETIC"] = 1;
		ads["SMA_LAMBDA"] = 1.2e3;
		ads["SMA_KA"] = {0.0, 35.5, 1.59, 7.7};
		ads["SMA_KD"] = {0.0, 1000.0, 1000.0, 1000.0};
		ads["SMA_NU"] = {0.0, 4.7, 5.29, 3.7};
		ads["SMA_SIGMA"] = {0.0, 11.83, 10.6, 10.0};
		config["adsorption"] = ads;
	}

	// Discretization
	{
		json disc;

		disc["NCOL"] = 16;
		disc["NPAR"] = 4;
		disc["NBOUND"] = {1, 1, 1, 1};

		if (uoType == "GENERAL_RATE_MODEL_2D")
		{
			disc["NCOL"] = 8;
			disc["NRAD"] = 3;
			disc["NPAR"] = 3;
			disc["RADIAL_DISC_TYPE"] = "EQUIDISTANT";
		}

		disc["PAR_DISC_TYPE"] = std::string("EQUIDISTANT_PAR");

		disc["USE_ANALYTIC_JACOBIAN"] = true;
		disc["MAX_KRYLOV"] = 0;
		disc["GS_TYPE"] = 1;
		disc["MAX_RESTARTS"] = 10;
		disc["SCHUR_SAFETY"] = 1e-8;

		// WENO
		{
			json weno;

			weno["WENO_ORDER"] = 3;
			weno["BOUNDARY_MODEL"] = 0;
			weno["WENO_EPS"] = 1e-10;
			disc["weno"] = weno;
		}
		config["discretization"] = disc;
	}

	return config;

/*
	return R"json({
	"UNIT_TYPE": "GENERAL_RATE_MODEL",
	"NCOMP": 4,
	"VELOCITY": 5.75e-4,
	"COL_DISPERSION": 5.75e-8,
	"FILM_DIFFUSION": [6.9e-6, 6.9e-6, 6.9e-6, 6.9e-6],
	"PAR_DIFFUSION": [7e-10, 6.07e-11, 6.07e-11, 6.07e-11],
	"PAR_SURFDIFFUSION": [0.0, 0.0, 0.0, 0.0],
	"COL_LENGTH": 0.014,
	"PAR_RADIUS": 4.5e-5,
	"COL_POROSITY": 0.37,
	"PAR_POROSITY": 0.75,
	"TOTAL_POROSITY": 0.8425,
	"INIT_C": [50.0, 0.0, 0.0, 0.0],
	"INIT_Q": [1.2e3, 0.0, 0.0, 0.0],
	"ADSORPTION_MODEL": "STERIC_MASS_ACTION",
	"adsorption":
	{
		"IS_KINETIC": 1,
		"SMA_LAMBDA": 1.2e3,
		"SMA_KA": [0.0, 35.5, 1.59, 7.7],
		"SMA_KD": [0.0, 1000.0, 1000.0, 1000.0],
		"SMA_NU": [0.0, 4.7, 5.29, 3.7],
		"SMA_SIGMA": [0.0, 11.83, 10.6, 10.0]
	},
	"discretization":
	{
		"NCOL": 16,
		"NPAR": 4,
		"NBOUND": [1, 1, 1, 1],
		"PAR_DISC_TYPE": "EQUIDISTANT_PAR",
		"USE_ANALYTIC_JACOBIAN": true,
		"MAX_KRYLOV": 0,
		"GS_TYPE": 1,
		"MAX_RESTARTS": 10,
		"SCHUR_SAFETY": 1e-8,
		"weno":
		{
			"WENO_ORDER": 3,
			"BOUNDARY_MODEL": 0,
			"WENO_EPS": 1e-10
		}
	}
	})json";
*/
}

cadet::JsonParameterProvider createColumnWithSMA(const std::string& uoType)
{
	return cadet::JsonParameterProvider(createColumnWithSMAJson(uoType));
}

json createColumnWithTwoCompLinearJson(const std::string& uoType)
{
	json config;
	config["UNIT_TYPE"] = uoType;
	config["NCOMP"] = 2;
	config["VELOCITY"] = 5.75e-4;
	config["COL_DISPERSION"] = 5.75e-8;
	config["COL_DISPERSION_RADIAL"] = 1e-6;
	config["FILM_DIFFUSION"] = {6.9e-6, 6.9e-6};
	config["PAR_DIFFUSION"] = {7e-10, 6.07e-11};
	config["PAR_SURFDIFFUSION"] = {1e-10, 1e-10};

	// Geometry
	config["COL_LENGTH"] = 0.014;
	config["COL_RADIUS"] = 0.01;
	config["COL_RADIUS_INNER"] = 0.001;
	config["COL_RADIUS_OUTER"] = 0.004;
	config["CROSS_SECTION_AREA"] = 0.0003141592653589793;
	config["PAR_RADIUS"] = 4.5e-5;
	config["COL_POROSITY"] = 0.37;
	config["PAR_POROSITY"] = 0.75;
	config["TOTAL_POROSITY"] = 0.37 + (1.0 - 0.37) * 0.75;

	// Initial conditions
	config["INIT_C"] = {1.0, 2.0, 3.0};
	config["INIT_Q"] = {5.0, 6.0, 7.0};

	// Adsorption
	config["ADSORPTION_MODEL"] = std::string("LINEAR");
	{
		json ads;
		ads["IS_KINETIC"] = 1;
		ads["LIN_KA"] = {12.3, 35.5, 1.59};
		ads["LIN_KD"] = {45.0, 20.0, 10.0};
		config["adsorption"] = ads;
	}

	// Discretization
	{
		json disc;

		disc["NCOL"] = 15;
		disc["NPAR"] = 5;
		disc["NBOUND"] = {1, 1};

		if (uoType == "GENERAL_RATE_MODEL_2D")
		{
			disc["NCOL"] = 8;
			disc["NRAD"] = 3;
			disc["NPAR"] = 3;
			disc["RADIAL_DISC_TYPE"] = "EQUIDISTANT";
		}

		disc["PAR_DISC_TYPE"] = std::string("EQUIDISTANT_PAR");

		disc["USE_ANALYTIC_JACOBIAN"] = true;
		disc["MAX_KRYLOV"] = 0;
		disc["GS_TYPE"] = 1;
		disc["MAX_RESTARTS"] = 10;
		disc["SCHUR_SAFETY"] = 1e-8;

		// WENO
		{
			json weno;

			weno["WENO_ORDER"] = 3;
			weno["BOUNDARY_MODEL"] = 0;
			weno["WENO_EPS"] = 1e-10;
			disc["weno"] = weno;
		}
		config["discretization"] = disc;
	}

	return config;
}

cadet::JsonParameterProvider createColumnWithTwoCompLinearBinding(const std::string& uoType)
{
	return cadet::JsonParameterProvider(createColumnWithTwoCompLinearJson(uoType));
}

json createLWEJson(const std::string& uoType)
{
	json config;
	// Model
	{
		json model;
		model["NUNITS"] = 2;
		model["unit_000"] = createColumnWithSMAJson(uoType);

		// Inlet - unit 001
		{
			json inlet;

			inlet["UNIT_TYPE"] = std::string("INLET");
			inlet["INLET_TYPE"] = std::string("PIECEWISE_CUBIC_POLY");
			inlet["NCOMP"] = 4;

			{
				json sec;

				sec["CONST_COEFF"] = {50.0, 1.0, 1.0, 1.0};
				sec["LIN_COEFF"] = {0.0, 0.0, 0.0, 0.0};
				sec["QUAD_COEFF"] = {0.0, 0.0, 0.0, 0.0};
				sec["CUBE_COEFF"] = {0.0, 0.0, 0.0, 0.0};

				inlet["sec_000"] = sec;
			}

			{
				json sec;

				sec["CONST_COEFF"] = {50.0, 0.0, 0.0, 0.0};
				sec["LIN_COEFF"] = {0.0, 0.0, 0.0, 0.0};
				sec["QUAD_COEFF"] = {0.0, 0.0, 0.0, 0.0};
				sec["CUBE_COEFF"] = {0.0, 0.0, 0.0, 0.0};

				inlet["sec_001"] = sec;
			}

			{
				json sec;

				sec["CONST_COEFF"] = {100.0, 0.0, 0.0, 0.0};
				sec["LIN_COEFF"] = {0.2, 0.0, 0.0, 0.0};
				sec["QUAD_COEFF"] = {0.0, 0.0, 0.0, 0.0};
				sec["CUBE_COEFF"] = {0.0, 0.0, 0.0, 0.0};

				inlet["sec_002"] = sec;
			}

			model["unit_001"] = inlet;
		}

		// Valve switches
		{
			json con;
			con["NSWITCHES"] = 1;
			con["CONNECTIONS_INCLUDE_PORTS"] = true;

			{
				json sw;

				// This switch occurs at beginning of section 0 (initial configuration)
				sw["SECTION"] = 0;

				if (uoType == "GENERAL_RATE_MODEL_2D")
				{
					// Connection list is 3x7 since we have 1 connection between
					// the two unit operations with 3 ports (and we need to have 7 columns)
					sw["CONNECTIONS"] = {1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 7.42637597e-09,
					                     1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 2.22791279e-08,
					                     1.0, 0.0, 0.0, 2.0, -1.0, -1.0, 3.71318798e-08};
					// Connections: From unit operation 1 port 0
					//              to unit operation 0 port 0,
					//              connect component -1 (i.e., all components)
					//              to component -1 (i.e., all components) with
					//              volumetric flow rate 7.42637597e-09 m^3/s
				}
				else
				{
					// Connection list is 1x7 since we have 1 connection between
					// the two unit operations (and we need to have 7 columns)
					sw["CONNECTIONS"] = {1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 6.683738370512285e-8};
					// Connections: From unit operation 1 port -1 (i.e., all ports)
					//              to unit operation 0 port -1 (i.e., all ports),
					//              connect component -1 (i.e., all components)
					//              to component -1 (i.e., all components) with
					//              volumetric flow rate 6.683738370512285e-8 m^3/s
				}

				con["switch_000"] = sw;
			}
			model["connections"] = con;
		}

		// Solver settings
		{
			json solver;

			solver["MAX_KRYLOV"] = 0;
			solver["GS_TYPE"] = 1;
			solver["MAX_RESTARTS"] = 10;
			solver["SCHUR_SAFETY"] = 1e-8;
			model["solver"] = solver;
		}

		config["model"] = model;
	}

	// Return
	{
		json ret;
		ret["WRITE_SOLUTION_TIMES"] = true;

		json grm;
		grm["WRITE_SOLUTION_BULK"] = false;
		grm["WRITE_SOLUTION_PARTICLE"] = false;
		grm["WRITE_SOLUTION_FLUX"] = false;
		grm["WRITE_SOLUTION_INLET"] = true;
		grm["WRITE_SOLUTION_OUTLET"] = true;

		ret["unit_000"] = grm;
		config["return"] = ret;
	}

	// Solver
	{
		json solver;

		{
			std::vector<double> solTimes;

			if (uoType == "LUMPED_RATE_MODEL_WITHOUT_PORES")
			{
				// Lumped rate model without pores has less rate limiting
				// Thus, a shorter simulation time suffices
				solTimes.reserve(1101);
				for (double t = 0.0; t <= 1100.0; t += 1.0)
					solTimes.push_back(t);
			}
			else
			{
				solTimes.reserve(1501);
				for (double t = 0.0; t <= 1500.0; t += 1.0)
					solTimes.push_back(t);
			}

			solver["USER_SOLUTION_TIMES"] = solTimes;
		}

		solver["NTHREADS"] = 1;

		// Sections
		{
			json sec;

			sec["NSEC"] = 3;
			if (uoType == "LUMPED_RATE_MODEL_WITHOUT_PORES")
				sec["SECTION_TIMES"] = {0.0, 10.0, 90.0, 1100.0};
			else
				sec["SECTION_TIMES"] = {0.0, 10.0, 90.0, 1500.0};
			sec["SECTION_CONTINUITY"] = {false, false};

			solver["sections"] = sec;
		}

		// Time integrator
		{
			json ti;

			ti["ABSTOL"] = 1e-8;
			ti["RELTOL"] = 1e-6;
			ti["ALGTOL"] = 1e-12;
			ti["INIT_STEP_SIZE"] = 1e-6;
			ti["MAX_STEPS"] = 10000;
			ti["MAX_STEP_SIZE"] = 0.0;
			ti["RELTOL_SENS"] = 1e-6;
			ti["ERRORTEST_SENS"] = true;
			ti["MAX_NEWTON_ITER"] = 3;
			ti["MAX_ERRTEST_FAIL"] = 7;
			ti["MAX_CONVTEST_FAIL"] = 10;
			ti["MAX_NEWTON_ITER_SENS"] = 3;
			ti["CONSISTENT_INIT_MODE"] = 1;
			ti["CONSISTENT_INIT_MODE_SENS"] = 1;

			solver["time_integrator"] = ti;
		}

		config["solver"] = solver;
	}
	return config;
}

cadet::JsonParameterProvider createLWE(const std::string& uoType)
{
	return cadet::JsonParameterProvider(createLWEJson(uoType));
}

cadet::JsonParameterProvider createPulseInjectionColumn(const std::string& uoType, bool dynamicBinding)
{
	json config;
	// Model
	{
		json model;
		model["NUNITS"] = 2;

		// GRM - unit 000
		{
			json grm;
			grm["UNIT_TYPE"] = uoType;
			grm["NCOMP"] = 1;
			grm["VELOCITY"] = 5.75e-4;
			grm["COL_DISPERSION"] = 5.75e-8;
			grm["COL_DISPERSION_MULTIPLEX"] = 0;
			grm["COL_DISPERSION_RADIAL"] = 1e-6;
			grm["FILM_DIFFUSION"] = {6.9e-6};
			grm["PAR_DIFFUSION"] = {7e-10};
			grm["PAR_SURFDIFFUSION"] = {0.0};

			// Geometry
			grm["COL_LENGTH"] = 0.014;
			grm["COL_RADIUS"] = 0.01;
			grm["COL_RADIUS_INNER"] = 0.001;
			grm["COL_RADIUS_OUTER"] = 0.004;
			grm["CROSS_SECTION_AREA"] = 0.0003141592653589793;
			grm["PAR_RADIUS"] = 4.5e-5;
			grm["PAR_CORERADIUS"] = 0.0;
			grm["COL_POROSITY"] = 0.37;
			grm["PAR_POROSITY"] = 0.75;
			grm["TOTAL_POROSITY"] = 0.37 + (1.0 - 0.37) * 0.75;

			// Initial conditions
			grm["INIT_C"] = {0.0};
			grm["INIT_Q"] = {0.0};

			// Adsorption
			{
				grm["ADSORPTION_MODEL"] = std::string("LINEAR");

				json ads;
				ads["IS_KINETIC"] = (dynamicBinding ? 1 : 0);
				ads["LIN_KA"] = {35.5};
				ads["LIN_KD"] = {1000.0};
				grm["adsorption"] = ads;
			}

			// Discretization
			{
				json disc;

				disc["NCOL"] = 10;
				disc["NPAR"] = 4;
				disc["NBOUND"] = {1};

				if (uoType == "GENERAL_RATE_MODEL_2D")
				{
					disc["NRAD"] = 3;
					disc["RADIAL_DISC_TYPE"] = "EQUIDISTANT";
				}

				disc["PAR_DISC_TYPE"] = std::string("EQUIDISTANT_PAR");

				disc["USE_ANALYTIC_JACOBIAN"] = true;
				disc["MAX_KRYLOV"] = 0;
				disc["GS_TYPE"] = 1;
				disc["MAX_RESTARTS"] = 10;
				disc["SCHUR_SAFETY"] = 1e-8;

				// WENO
				{
					json weno;

					weno["WENO_ORDER"] = 3;
					weno["BOUNDARY_MODEL"] = 0;
					weno["WENO_EPS"] = 1e-10;
					disc["weno"] = weno;
				}
				grm["discretization"] = disc;
			}

			model["unit_000"] = grm;
		}

		// Inlet - unit 001
		{
			json inlet;

			inlet["UNIT_TYPE"] = std::string("INLET");
			inlet["INLET_TYPE"] = std::string("PIECEWISE_CUBIC_POLY");
			inlet["NCOMP"] = 1;

			{
				json sec;

				sec["CONST_COEFF"] = {1.0};
				sec["LIN_COEFF"] = {0.0};
				sec["QUAD_COEFF"] = {0.0};
				sec["CUBE_COEFF"] = {0.0};

				inlet["sec_000"] = sec;
			}

			{
				json sec;

				sec["CONST_COEFF"] = {0.0};
				sec["LIN_COEFF"] = {0.0};
				sec["QUAD_COEFF"] = {0.0};
				sec["CUBE_COEFF"] = {0.0};

				inlet["sec_001"] = sec;
			}

			model["unit_001"] = inlet;
		}

		// Valve switches
		{
			json con;
			con["NSWITCHES"] = 1;
			con["CONNECTIONS_INCLUDE_PORTS"] = true;

			{
				json sw;

				// This switch occurs at beginning of section 0 (initial configuration)
				sw["SECTION"] = 0;

				if (uoType == "GENERAL_RATE_MODEL_2D")
				{
					// Connection list is 3x7 since we have 1 connection between
					// the two unit operations with 3 ports (and we need to have 7 columns)
					sw["CONNECTIONS"] = {1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 7.42637597e-09,
					                     1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 2.22791279e-08,
					                     1.0, 0.0, 0.0, 2.0, -1.0, -1.0, 3.71318798e-08};
					// Connections: From unit operation 1 port 0
					//              to unit operation 0 port 0,
					//              connect component -1 (i.e., all components)
					//              to component -1 (i.e., all components) with
					//              volumetric flow rate 7.42637597e-09 m^3/s
				}
				else
				{
					// Connection list is 1x7 since we have 1 connection between
					// the two unit operations (and we need to have 7 columns)
					sw["CONNECTIONS"] = {1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 1.0};
					// Connections: From unit operation 1 port -1 (i.e., all ports)
					//              to unit operation 0 port -1 (i.e., all ports),
					//              connect component -1 (i.e., all components)
					//              to component -1 (i.e., all components) with
					//              volumetric flow rate 1.0 m^3/s
				}

				con["switch_000"] = sw;
			}
			model["connections"] = con;
		}

		// Solver settings
		{
			json solver;

			solver["MAX_KRYLOV"] = 0;
			solver["GS_TYPE"] = 1;
			solver["MAX_RESTARTS"] = 10;
			solver["SCHUR_SAFETY"] = 1e-8;
			model["solver"] = solver;
		}

		config["model"] = model;
	}

	// Return
	{
		json ret;
		ret["WRITE_SOLUTION_TIMES"] = true;

		json grm;
		grm["WRITE_SOLUTION_BULK"] = false;
		grm["WRITE_SOLUTION_PARTICLE"] = false;
		grm["WRITE_SOLUTION_FLUX"] = false;
		grm["WRITE_SOLUTION_INLET"] = true;
		grm["WRITE_SOLUTION_OUTLET"] = true;

		ret["unit_000"] = grm;
		config["return"] = ret;
	}

	// Solver
	{
		json solver;

		{
			std::vector<double> solTimes;
			solTimes.reserve(151);
			for (double t = 0.0; t <= 150.0; t += 1.0)
				solTimes.push_back(t);

			solver["USER_SOLUTION_TIMES"] = solTimes;
		}

		solver["NTHREADS"] = 1;

		// Sections
		{
			json sec;

			sec["NSEC"] = 2;
			sec["SECTION_TIMES"] = {0.0, 10.0, 150.0};
			sec["SECTION_CONTINUITY"] = {false, false};

			solver["sections"] = sec;
		}

		// Time integrator
		{
			json ti;

			ti["ABSTOL"] = 1e-8;
			ti["RELTOL"] = 1e-6;
			ti["ALGTOL"] = 1e-12;
			ti["INIT_STEP_SIZE"] = 1e-6;
			ti["MAX_STEPS"] = 10000;
			ti["MAX_STEP_SIZE"] = 0.0;
			ti["RELTOL_SENS"] = 1e-6;
			ti["ERRORTEST_SENS"] = true;
			ti["MAX_NEWTON_ITER"] = 3;
			ti["MAX_ERRTEST_FAIL"] = 7;
			ti["MAX_CONVTEST_FAIL"] = 10;
			ti["MAX_NEWTON_ITER_SENS"] = 3;
			ti["CONSISTENT_INIT_MODE"] = 1;
			ti["CONSISTENT_INIT_MODE_SENS"] = 1;

			solver["time_integrator"] = ti;
		}

		config["solver"] = solver;
	}
	return cadet::JsonParameterProvider(config);
}

cadet::JsonParameterProvider createLinearBenchmark(bool dynamicBinding, bool nonBinding, const std::string& uoType)
{
	json config;
	// Model
	{
		json model;
		model["NUNITS"] = 2;

		// GRM - unit 000
		{
			json grm;
			grm["UNIT_TYPE"] = uoType;
			grm["NCOMP"] = 1;
			grm["VELOCITY"] = 0.5 / (100.0 * 60.0);
			grm["COL_DISPERSION"] = 0.002 / (100.0 * 100.0 * 60.0);
			grm["COL_DISPERSION_MULTIPLEX"] = 0;
			grm["COL_DISPERSION_RADIAL"] = 1e-6;
			grm["FILM_DIFFUSION"] = {0.01 / (100.0 * 60.0)};
			grm["PAR_DIFFUSION"] = {3.003e-6};
			grm["PAR_SURFDIFFUSION"] = {0.0};

			// Geometry
			grm["COL_LENGTH"] = 0.017;
			grm["COL_RADIUS"] = 0.01;
			grm["PAR_RADIUS"] = 4e-5;
			grm["COL_POROSITY"] = 0.4;
			grm["PAR_POROSITY"] = 0.333;
			grm["TOTAL_POROSITY"] = 0.4 + (1.0 - 0.4) * 0.333;

			// Initial conditions
			grm["INIT_C"] = {0.0};
			grm["INIT_Q"] = {0.0};

			// Adsorption
			if (nonBinding)
			{
				grm["ADSORPTION_MODEL"] = std::string("NONE");
			}
			else
			{
				grm["ADSORPTION_MODEL"] = std::string("LINEAR");

				json ads;
				ads["IS_KINETIC"] = (dynamicBinding ? 1 : 0);
				ads["LIN_KA"] = {2.5};
				ads["LIN_KD"] = {1.0};
				grm["adsorption"] = ads;
			}

			// Discretization
			{
				json disc;

				disc["NCOL"] = 512;
				disc["NPAR"] = 4;
				if (nonBinding)
					disc["NBOUND"] = {0};
				else
					disc["NBOUND"] = {1};

				if (uoType == "GENERAL_RATE_MODEL_2D")
				{
					disc["NRAD"] = 3;
					disc["RADIAL_DISC_TYPE"] = "EQUIDISTANT";
				}

				disc["PAR_DISC_TYPE"] = std::string("EQUIDISTANT_PAR");

				disc["USE_ANALYTIC_JACOBIAN"] = true;
				disc["MAX_KRYLOV"] = 0;
				disc["GS_TYPE"] = 1;
				disc["MAX_RESTARTS"] = 10;
				disc["SCHUR_SAFETY"] = 1e-8;

				// WENO
				{
					json weno;

					weno["WENO_ORDER"] = 3;
					weno["BOUNDARY_MODEL"] = 0;
					weno["WENO_EPS"] = 1e-10;
					disc["weno"] = weno;
				}
				grm["discretization"] = disc;
			}

			model["unit_000"] = grm;
		}

		// Inlet - unit 001
		{
			json inlet;

			inlet["UNIT_TYPE"] = std::string("INLET");
			inlet["INLET_TYPE"] = std::string("PIECEWISE_CUBIC_POLY");
			inlet["NCOMP"] = 1;

			{
				json sec;

				sec["CONST_COEFF"] = {1.0};
				sec["LIN_COEFF"] = {0.0};
				sec["QUAD_COEFF"] = {0.0};
				sec["CUBE_COEFF"] = {0.0};

				inlet["sec_000"] = sec;
			}

			{
				json sec;

				sec["CONST_COEFF"] = {0.0};
				sec["LIN_COEFF"] = {0.0};
				sec["QUAD_COEFF"] = {0.0};
				sec["CUBE_COEFF"] = {0.0};

				inlet["sec_001"] = sec;
			}

			model["unit_001"] = inlet;
		}

		// Valve switches
		{
			json con;
			con["NSWITCHES"] = 1;
			con["CONNECTIONS_INCLUDE_PORTS"] = true;

			{
				json sw;

				// This switch occurs at beginning of section 0 (initial configuration)
				sw["SECTION"] = 0;

				if (uoType == "GENERAL_RATE_MODEL_2D")
				{
					// Connection list is 3x7 since we have 1 connection between
					// the two unit operations with 3 ports (and we need to have 7 columns)
					sw["CONNECTIONS"] = {1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.16355283e-09,
					                     1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 3.49065850e-09,
					                     1.0, 0.0, 0.0, 2.0, -1.0, -1.0, 5.81776417e-09};
					// Connections: From unit operation 1 port 0
					//              to unit operation 0 port 0,
					//              connect component -1 (i.e., all components)
					//              to component -1 (i.e., all components) with
					//              volumetric flow rate 1.16355283e-09 m^3/s
				}
				else
				{
					// Connection list is 1x7 since we have 1 connection between
					// the two unit operations (and we need to have 7 columns)
					sw["CONNECTIONS"] = {1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 1.0};
					// Connections: From unit operation 1 port -1 (i.e., all ports)
					//              to unit operation 0 port -1 (i.e., all ports),
					//              connect component -1 (i.e., all components)
					//              to component -1 (i.e., all components) with
					//              volumetric flow rate 1.0 m^3/s
				}

				con["switch_000"] = sw;
			}
			model["connections"] = con;
		}

		// Solver settings
		{
			json solver;

			solver["MAX_KRYLOV"] = 0;
			solver["GS_TYPE"] = 1;
			solver["MAX_RESTARTS"] = 10;
			solver["SCHUR_SAFETY"] = 1e-8;
			model["solver"] = solver;
		}

		config["model"] = model;
	}

	// Return
	{
		json ret;
		ret["WRITE_SOLUTION_TIMES"] = true;

		json grm;
		grm["WRITE_SOLUTION_BULK"] = false;
		grm["WRITE_SOLUTION_PARTICLE"] = false;
		grm["WRITE_SOLUTION_FLUX"] = false;
		grm["WRITE_SOLUTION_INLET"] = true;
		grm["WRITE_SOLUTION_OUTLET"] = true;

		ret["unit_000"] = grm;
		config["return"] = ret;
	}

	// Solver
	{
		json solver;

		{
			std::vector<double> solTimes;
			if (nonBinding)
			{
				solTimes.reserve(2501);
				for (double t = 0.0; t <= 2500.0; t += 1.0)
					solTimes.push_back(t);
			}
			else
			{
				solTimes.reserve(3001);
				for (double t = 0.0; t <= 100.0 * 60.0; t += 2.0)
					solTimes.push_back(t);
			}

			solver["USER_SOLUTION_TIMES"] = solTimes;
		}

		solver["NTHREADS"] = 1;

		// Sections
		{
			json sec;

			sec["NSEC"] = 2;
			if (nonBinding)
				sec["SECTION_TIMES"] = {0.0, 20.0 * 60.0, 2500.0};
			else
				sec["SECTION_TIMES"] = {0.0, 20.0 * 60.0, 100.0 * 60.0};
			sec["SECTION_CONTINUITY"] = {false, false};

			solver["sections"] = sec;
		}

		// Time integrator
		{
			json ti;

			ti["ABSTOL"] = 1e-8;
			ti["RELTOL"] = 1e-6;
			ti["ALGTOL"] = 1e-12;
			ti["INIT_STEP_SIZE"] = 1e-6;
			ti["MAX_STEPS"] = 10000;
			ti["MAX_STEP_SIZE"] = 0.0;
			ti["RELTOL_SENS"] = 1e-6;
			ti["ERRORTEST_SENS"] = true;
			ti["MAX_NEWTON_ITER"] = 3;
			ti["MAX_ERRTEST_FAIL"] = 7;
			ti["MAX_CONVTEST_FAIL"] = 10;
			ti["MAX_NEWTON_ITER_SENS"] = 3;
			ti["CONSISTENT_INIT_MODE"] = 1;
			ti["CONSISTENT_INIT_MODE_SENS"] = 1;

			solver["time_integrator"] = ti;
		}

		config["solver"] = solver;
	}
	return cadet::JsonParameterProvider(config);
}

json createCSTRJson(unsigned int nComp)
{
	json config;
	config["UNIT_TYPE"] = std::string("CSTR");
	config["NCOMP"] = static_cast<int>(nComp);
	config["INIT_VOLUME"] = 1.0;
	config["INIT_C"] = std::vector<double>(nComp, 0.0);
	config["FLOWRATE_FILTER"] = {0.0};
	return config;
}

cadet::JsonParameterProvider createCSTR(unsigned int nComp)
{
	return cadet::JsonParameterProvider(createCSTRJson(nComp));
}

cadet::JsonParameterProvider createCSTRBenchmark(unsigned int nSec, double endTime, double interval)
{
	std::ostringstream ss;

	json config;
	// Model
	{
		json model;
		model["NUNITS"] = 3;

		// CSTR - unit 000
		model["unit_000"] = createCSTRJson(1);

		// Inlet - unit 001
		{
			json inlet;

			inlet["UNIT_TYPE"] = std::string("INLET");
			inlet["INLET_TYPE"] = std::string("PIECEWISE_CUBIC_POLY");
			inlet["NCOMP"] = 1;

			for (unsigned int i = 0; i < nSec; ++i)
			{
				json sec;

				sec["CONST_COEFF"] = {0.0};
				sec["LIN_COEFF"] = {0.0};
				sec["QUAD_COEFF"] = {0.0};
				sec["CUBE_COEFF"] = {0.0};

				ss << "sec_" << std::setfill('0') << std::setw(3) << i;
				inlet[ss.str()] = sec;
				ss.str("");
			}

			model["unit_001"] = inlet;
		}

		// Outlet - unit 002
		{
			json outlet;

			outlet["UNIT_TYPE"] = std::string("OUTLET");
			outlet["NCOMP"] = 1;

			model["unit_002"] = outlet;
		}
		// Valve switches
		{
			json con;
			con["NSWITCHES"] = nSec;
			con["CONNECTIONS_INCLUDE_PORTS"] = true;

			for (unsigned int i = 0; i < nSec; ++i)
			{
				json sw;

				// This switch occurs at beginning of section 0 (initial configuration)
				sw["SECTION"] = static_cast<int>(i);

				// Connection list is 2x7 since we have 2 connection between
				// the three unit operations (and we need to have 7 columns)
				sw["CONNECTIONS"] = {1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 1.0,
				                     0.0, 2.0, -1.0, -1.0, -1.0, -1.0, 1.0};
				// Connections: From unit operation 1 port -1 (i.e., all ports)
				//              to unit operation 0 port -1 (i.e., all ports),
				//              connect component -1 (i.e., all components)
				//              to component -1 (i.e., all components) with
				//              volumetric flow rate 1.0 m^3/s
				//              From unit operation 0 port -1 (i.e., all ports)
				//              to unit operation 2 port -1 (i.e., all ports),
				//              connect component -1 (i.e., all components)
				//              to component -1 (i.e., all components) with
				//              volumetric flow rate 1.0 m^3/s

				ss << "switch_" << std::setfill('0') << std::setw(3) << i;
				con[ss.str()] = sw;
				ss.str("");
			}
			model["connections"] = con;
		}

		// Solver settings
		{
			json solver;

			solver["MAX_KRYLOV"] = 0;
			solver["GS_TYPE"] = 1;
			solver["MAX_RESTARTS"] = 10;
			solver["SCHUR_SAFETY"] = 1e-8;
			model["solver"] = solver;
		}

		config["model"] = model;
	}

	// Return
	{
		json ret;
		ret["WRITE_SOLUTION_TIMES"] = true;

		json cstr;
		cstr["WRITE_SOLUTION_BULK"] = false;
		cstr["WRITE_SOLUTION_SOLID"] = true;
		cstr["WRITE_SOLUTION_FLUX"] = false;
		cstr["WRITE_SOLUTION_INLET"] = true;
		cstr["WRITE_SOLUTION_OUTLET"] = true;
		cstr["WRITE_SOLUTION_VOLUME"] = true;

		ret["unit_000"] = cstr;
		config["return"] = ret;
	}

	// Solver
	{
		json solver;

		{
			std::vector<double> solTimes;
			solTimes.reserve(static_cast<int>(endTime / interval) + 1);
			for (double t = 0.0; t <= endTime; t += interval)
				solTimes.push_back(t);

			solver["USER_SOLUTION_TIMES"] = solTimes;
		}

		solver["NTHREADS"] = 1;

		// Sections
		{
			json sec;

			sec["NSEC"] = nSec;
			sec["SECTION_TIMES"] = std::vector<double>(nSec + 1, 0.0);

			solver["sections"] = sec;
		}

		// Time integrator
		{
			json ti;

			ti["ABSTOL"] = 1e-8;
			ti["RELTOL"] = 1e-6;
			ti["ALGTOL"] = 1e-12;
			ti["INIT_STEP_SIZE"] = 1e-6;
			ti["MAX_STEPS"] = 10000;
			ti["MAX_STEP_SIZE"] = 0.0;
			ti["RELTOL_SENS"] = 1e-6;
			ti["ERRORTEST_SENS"] = true;
			ti["MAX_NEWTON_ITER"] = 3;
			ti["MAX_ERRTEST_FAIL"] = 7;
			ti["MAX_CONVTEST_FAIL"] = 10;
			ti["MAX_NEWTON_ITER_SENS"] = 3;
			ti["CONSISTENT_INIT_MODE"] = 1;
			ti["CONSISTENT_INIT_MODE_SENS"] = 1;

			solver["time_integrator"] = ti;
		}

		config["solver"] = solver;
	}
	return cadet::JsonParameterProvider(config);
}
