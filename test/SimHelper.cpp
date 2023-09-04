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

#include "SimHelper.hpp"
#include "Utils.hpp"

#include <iterator>
#include <algorithm>

namespace cadet
{

namespace test
{

void setNumberOfComponents(cadet::JsonParameterProvider& jpp, UnitOpIdx unit, int nComp)
{
	auto gs = util::makeModelGroupScope(jpp, unit);
	jpp.set("NCOMP", nComp);
}

void setInitialConditions(cadet::JsonParameterProvider& jpp, const std::vector<double>& c, const std::vector<double>& q,
						  double v)
{
	auto gs = util::makeModelGroupScope(jpp);

	jpp.set("INIT_C", c);
	jpp.set("INIT_VOLUME", v);
	if (!q.empty())
		jpp.set("INIT_Q", q);
}

void setInitialConditions(cadet::JsonParameterProvider& jpp, const std::vector<double>& c,
						  const std::vector<double>& cp, const std::vector<double>& q)
{
	auto gs = util::makeModelGroupScope(jpp);

	jpp.set("INIT_C", c);
	if (!cp.empty())
		jpp.set("INIT_CP", cp);
	if (!q.empty())
		jpp.set("INIT_Q", q);
}

void setFlowRates(cadet::JsonParameterProvider& jpp, unsigned int secIdx, double in, double out, double filter)
{
	setFlowRates(jpp, secIdx, filter);

	// Change to /model
	auto gs = util::makeOptionalGroupScope(jpp, "model");

	jpp.pushScope("connections");

	std::ostringstream ss;
	ss << "switch_" << std::setfill('0') << std::setw(3) << secIdx;
	jpp.pushScope(ss.str());

	std::vector<double> con = jpp.getDoubleArray("CONNECTIONS");
	con[6] = in;
	con[13] = out;
	jpp.set("CONNECTIONS", con);

	jpp.popScope();
	jpp.popScope();
}

void setFlowRates(cadet::JsonParameterProvider& jpp, unsigned int secIdx, double filter)
{
	// Change to /model/unit_000
	auto gs = util::makeModelGroupScope(jpp);

	std::vector<double> frf = jpp.getDoubleArray("FLOWRATE_FILTER");
	if (frf.size() <= secIdx)
		std::fill_n(std::back_inserter(frf), frf.size() - secIdx + 1, 0.0);

	frf[secIdx] = filter;
	jpp.set("FLOWRATE_FILTER", frf);
}

void setInletProfile(cadet::JsonParameterProvider& jpp, unsigned int secIdx, unsigned int comp, double con, double lin,
					 double quad, double cub)
{
	auto gs = util::makeModelGroupScope(jpp, 1);

	std::ostringstream ss;
	ss << "sec_" << std::setfill('0') << std::setw(3) << secIdx;
	jpp.pushScope(ss.str());

	std::vector<double> cCon = jpp.getDoubleArray("CONST_COEFF");
	if (cCon.size() <= comp)
		cCon.resize(comp + 1);
	cCon[comp] = con;
	jpp.set("CONST_COEFF", cCon);

	std::vector<double> cLin = jpp.getDoubleArray("LIN_COEFF");
	if (cLin.size() <= comp)
		cLin.resize(comp + 1);
	cLin[comp] = lin;
	jpp.set("LIN_COEFF", cLin);

	std::vector<double> cQuad = jpp.getDoubleArray("QUAD_COEFF");
	if (cQuad.size() <= comp)
		cQuad.resize(comp + 1);
	cQuad[comp] = quad;
	jpp.set("QUAD_COEFF", cQuad);

	std::vector<double> cCube = jpp.getDoubleArray("CUBE_COEFF");
	if (cCube.size() <= comp)
		cCube.resize(comp + 1);
	cCube[comp] = cub;
	jpp.set("CUBE_COEFF", cCube);

	jpp.popScope();
}

void setSectionTimes(cadet::JsonParameterProvider& jpp, const std::vector<double>& secTimes)
{
	jpp.pushScope("solver");
	jpp.pushScope("sections");

	jpp.set("SECTION_TIMES", secTimes);

	jpp.popScope();
	jpp.popScope();
}

void addBoundStates(cadet::JsonParameterProvider& jpp, const std::vector<int>& nBound, double porosity)
{
	auto gs = util::makeModelGroupScope(jpp);

	jpp.set("NBOUND", nBound);
	jpp.set("POROSITY", porosity);
}

void addDummyBindingModel(cadet::JsonParameterProvider& jpp)
{
	auto gs = util::makeModelGroupScope(jpp);
	jpp.set("ADSORPTION_MODEL", "NONE");
}

void addLinearBindingModel(cadet::JsonParameterProvider& jpp, bool kinetic, const std::vector<double>& kA,
						   const std::vector<double>& kD)
{
	auto gs = util::makeModelGroupScope(jpp);

	jpp.set("ADSORPTION_MODEL", "LINEAR");

	jpp.addScope("adsorption");
	jpp.pushScope("adsorption");

	jpp.set("IS_KINETIC", kinetic);
	jpp.set("LIN_KA", kA);
	jpp.set("LIN_KD", kD);

	jpp.popScope();
}

void addSMABindingModel(cadet::JsonParameterProvider& jpp, bool kinetic, double lambda, const std::vector<double>& kA,
						const std::vector<double>& kD, const std::vector<double>& nu, const std::vector<double>& sigma)
{
	auto gs = util::makeModelGroupScope(jpp);

	jpp.set("ADSORPTION_MODEL", "STERIC_MASS_ACTION");

	jpp.addScope("adsorption");
	jpp.pushScope("adsorption");

	jpp.set("IS_KINETIC", kinetic);
	jpp.set("SMA_LAMBDA", lambda);
	jpp.set("SMA_KA", kA);
	jpp.set("SMA_KD", kD);
	jpp.set("SMA_NU", nu);
	jpp.set("SMA_SIGMA", sigma);

	jpp.popScope();
}

void addLangmuirBindingModel(cadet::JsonParameterProvider& jpp, bool kinetic, const std::vector<double>& kA,
							 const std::vector<double>& kD, const std::vector<double>& qMax)
{
	auto gs = util::makeModelGroupScope(jpp);

	jpp.set("ADSORPTION_MODEL", "MULTI_COMPONENT_LANGMUIR");

	jpp.addScope("adsorption");
	jpp.pushScope("adsorption");

	jpp.set("IS_KINETIC", kinetic);
	jpp.set("MCL_KA", kA);
	jpp.set("MCL_KD", kD);
	jpp.set("MCL_QMAX", qMax);

	jpp.popScope();
}

void setBindingMode(cadet::JsonParameterProvider& jpp, bool isKinetic)
{
	auto ms = util::makeOptionalGroupScope(jpp, "model");
	auto us = util::makeOptionalGroupScope(jpp, "unit_000");

	jpp.pushScope("adsorption");
	jpp.set("IS_KINETIC", isKinetic);
	jpp.popScope();
}

void addSensitivity(cadet::JsonParameterProvider& jpp, const std::string& name, const ParameterId& id, double absTol)
{
	jpp.addScope("sensitivity");
	jpp.pushScope("sensitivity");

	jpp.set("SENS_METHOD", "ad1");
	int sensIdx = 0;
	if (jpp.exists("NSENS"))
	{
		sensIdx = jpp.getInt("NSENS");
	}
	jpp.set("NSENS", sensIdx + 1);

	std::ostringstream ss;
	ss << "param_" << std::setfill('0') << std::setw(3) << sensIdx;
	jpp.addScope(ss.str());
	jpp.pushScope(ss.str());

	jpp.set("SENS_UNIT", std::vector<int>(1, id.unitOperation));
	jpp.set("SENS_NAME", std::vector<std::string>(1, name));
	jpp.set("SENS_COMP", std::vector<int>(1, id.component));
	jpp.set("SENS_BOUNDPHASE", std::vector<int>(1, id.boundState));
	jpp.set("SENS_REACTION", std::vector<int>(1, id.reaction));
	jpp.set("SENS_SECTION", std::vector<int>(1, id.section));
	jpp.set("SENS_PARTYPE", std::vector<int>(1, id.particleType));
	jpp.set("SENS_FACTOR", std::vector<double>(1, 1.0));
	jpp.set("SENS_ABSTOL", absTol);

	jpp.popScope();
	jpp.popScope();
}

void returnSensitivities(cadet::JsonParameterProvider& jpp, UnitOpIdx unit, bool inlet)
{
	jpp.pushScope("return");

	std::ostringstream ss;
	ss << "unit_" << std::setfill('0') << std::setw(3) << unit;
	jpp.pushScope(ss.str());

	jpp.set("WRITE_SENS_INLET", inlet);
	jpp.set("WRITE_SENS_OUTLET", true);
	jpp.set("WRITE_SENS_VOLUME", true);
	jpp.popScope();
	jpp.popScope();
}

void disableSensitivityErrorTest(cadet::JsonParameterProvider& jpp, bool isDisabled)
{
	jpp.pushScope("solver");
	jpp.pushScope("time_integrator");
	jpp.set("ERRORTEST_SENS", !isDisabled);
	jpp.popScope();
	jpp.popScope();
}

void setMaxStepSize(cadet::JsonParameterProvider& jpp, double maxStepSize)
{
	jpp.pushScope("solver");
	jpp.pushScope("time_integrator");
	jpp.set("MAX_STEP_SIZE", std::max(maxStepSize, 0.0));
	jpp.popScope();
	jpp.popScope();
}

} // namespace test
} // namespace cadet
