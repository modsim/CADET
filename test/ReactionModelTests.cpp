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

#include <catch.hpp>
#include "Approx.hpp"
#include "cadet/ExternalFunction.hpp"
#include "cadet/ParameterProvider.hpp"

#define CADET_LOGGING_DISABLE
#include "Logging.hpp"

#include "JacobianHelper.hpp"
#include "ReactionModelTests.hpp"
#include "UnitOperationTests.hpp"
#include "JsonTestModels.hpp"
#include "Utils.hpp"

#include "common/JsonParameterProvider.hpp"

#include "ReactionModelFactory.hpp"
#include "model/ReactionModel.hpp"
#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"
#include "AdUtils.hpp"
#include "AutoDiff.hpp"
#include "Memory.hpp"

#include <algorithm>
#include <numeric>
#include <cstring>

namespace
{
inline cadet::model::IDynamicReactionModel* createDynamicReactionModel(const char* name)
{
	cadet::ReactionModelFactory rmf;
	cadet::model::IDynamicReactionModel* const rm = rmf.createDynamic(name);

	REQUIRE(nullptr != rm);
	return rm;
}

class ConstExternalFunction : public cadet::IExternalFunction
{
public:
	virtual bool configure(cadet::IParameterProvider* paramProvider)
	{
		return true;
	}
	virtual const char* name() const CADET_NOEXCEPT
	{
		return "CONSTFUN";
	}
	virtual double externalProfile(double t, double z, double rho, double r, unsigned int sec)
	{
		return 1.0;
	}
	virtual double timeDerivative(double t, double z, double rho, double r, unsigned int sec)
	{
		return 0.0;
	}
	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections)
	{
	}
};

class LinearExternalFunction : public cadet::IExternalFunction
{
public:
	virtual bool configure(cadet::IParameterProvider* paramProvider)
	{
		return true;
	}
	virtual const char* name() const CADET_NOEXCEPT
	{
		return "LINFUN";
	}
	virtual double externalProfile(double t, double z, double rho, double r, unsigned int sec)
	{
		return t;
	}
	virtual double timeDerivative(double t, double z, double rho, double r, unsigned int sec)
	{
		return 1.0;
	}
	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections)
	{
	}
};
} // namespace

namespace cadet
{

namespace test
{

namespace reaction
{

ConfiguredDynamicReactionModel::~ConfiguredDynamicReactionModel()
{
	::operator delete(_bufferMemory);

	delete[] _extFuns;
	delete[] _boundOffset;
	delete _reaction;
}

ConfiguredDynamicReactionModel ConfiguredDynamicReactionModel::create(const char* name, unsigned int nComp,
																	  unsigned int const* nBound, const char* config)
{
	cadet::model::IDynamicReactionModel* const rm = createDynamicReactionModel(name);

	// Calculate offset of bound states
	unsigned int* boundOffset = new unsigned int[nComp];
	boundOffset[0] = 0;
	for (unsigned int i = 1; i < nComp; ++i)
	{
		boundOffset[i] = boundOffset[i - 1] + nBound[i - 1];
	}
	const unsigned int totalBoundStates = boundOffset[nComp - 1] + nBound[nComp - 1];

	// Configure
	cadet::JsonParameterProvider jpp(config);
	rm->configureModelDiscretization(jpp, nComp, nBound, boundOffset);
	if (rm->requiresConfiguration())
	{
		jpp.set("EXTFUN", std::vector<int>(1, 0));
		REQUIRE(rm->configure(jpp, 0, 0));
	}

	// Assign external functions
	cadet::IExternalFunction* extFuns = new LinearExternalFunction[50];
	rm->setExternalFunctions(&extFuns, 50);

	// Allocate memory buffer
	unsigned int requiredMem = 0;
	if (rm->requiresWorkspace())
		requiredMem = rm->workspaceSize(nComp, totalBoundStates, boundOffset);

	void* buffer = nullptr;
	void* bufferEnd = nullptr;
	if (requiredMem > 0)
	{
		buffer = ::operator new(requiredMem);
		bufferEnd = static_cast<char*>(buffer) + requiredMem;
		std::memset(buffer, 0, requiredMem);
	}

	return ConfiguredDynamicReactionModel(rm, nComp, nBound, boundOffset, buffer, bufferEnd, extFuns);
}

void ConfiguredDynamicReactionModel::increaseBufferSize(int inc)
{
	const int bufSize = requiredBufferSize() + inc;

	::operator delete(_bufferMemory);
	_bufferMemory = ::operator new(bufSize);
	std::memset(_bufferMemory, 0, bufSize);

#ifdef CADET_DEBUG
	_buffer.setBuffer(_bufferMemory, static_cast<char*>(_bufferMemory) + bufSize);
#else
	_buffer.setBuffer(_bufferMemory);
#endif
}

int ConfiguredDynamicReactionModel::requiredBufferSize() CADET_NOEXCEPT
{
	if (_reaction->requiresWorkspace())
	{
		unsigned int totalBoundStates = 0;
		for (unsigned int i = 0; i < _nComp; ++i)
			totalBoundStates += _nBound[i];

		return _reaction->workspaceSize(_nComp, totalBoundStates, _boundOffset);
	}

	return 0;
}

void extendModelWithDynamicReactions(cadet::JsonParameterProvider& jpp, UnitOpIdx unit, bool bulk, bool particle,
									 bool particleModifiers)
{
	auto gs = util::makeModelGroupScope(jpp, unit);
	const int nComp = jpp.getInt("NCOMP");

	int nParType = 0;
	std::vector<int> nBound(0);
	{
		auto ds = cadet::test::util::makeOptionalGroupScope(jpp, "discretization");

		// Try to detect number of particle types
		if (jpp.exists("NPAR"))
			nParType = jpp.numElements("NPAR");

		if (jpp.exists("NBOUND"))
		{
			nBound = jpp.getIntArray("NBOUND");

			if (nParType == 0)
				nParType = nBound.size() / nComp;
		}
	}

	const std::string uoType = jpp.getString("UNIT_TYPE");
	const bool isLRMP = (uoType == "LUMPED_RATE_MODEL_WITHOUT_PORES");
	const int nTotalBound = std::accumulate(nBound.begin(), nBound.end(), 0);

	if (!isLRMP && bulk)
	{
		char const* const scopeName = "reaction_bulk";
		jpp.addScope(scopeName);
		auto gs2 = util::makeGroupScope(jpp, scopeName);

		const int nReactions = 4;

		std::vector<double> stoichMat(nReactions * nComp, 0.0);
		std::vector<double> expFwd(nReactions * nComp, 0.0);
		std::vector<double> expBwd(nReactions * nComp, 0.0);
		std::vector<double> rateFwd(nReactions, 0.0);
		std::vector<double> rateBwd(nReactions, 0.0);

		util::populate(
			stoichMat.data(), [](unsigned int idx) { return 2.0 * std::sin(0.5 + idx * 0.7); }, stoichMat.size());
		util::populate(
			expFwd.data(), [](unsigned int idx) { return std::sin(0.1 + idx * 0.7) + 1.4; }, expFwd.size());
		util::populate(
			expBwd.data(), [](unsigned int idx) { return std::sin(0.2 + idx * 0.3) + 1.1; }, expBwd.size());
		util::populate(
			rateFwd.data(), [](unsigned int idx) { return std::sin(0.3 + idx * 0.2) * 0.5 + 1.5; }, rateFwd.size());
		util::populate(
			rateBwd.data(), [](unsigned int idx) { return std::sin(0.4 + idx * 0.4) * 0.5 + 1.6; }, rateBwd.size());

		jpp.set("MAL_STOICHIOMETRY_BULK", stoichMat);
		jpp.set("MAL_EXPONENTS_BULK_FWD", expFwd);
		jpp.set("MAL_EXPONENTS_BULK_BWD", expBwd);
		jpp.set("MAL_KFWD_BULK", rateFwd);
		jpp.set("MAL_KBWD_BULK", rateBwd);
	}

	if ((!isLRMP && particle) || (isLRMP && bulk))
	{
		const int nReactions = 3;

		for (int i = 0; i < nParType; ++i)
		{
			std::ostringstream oss;
			if (isLRMP)
				oss << "reaction";
			else
				oss << "reaction_particle_" << std::setfill('0') << std::setw(3) << std::setprecision(0) << i;

			jpp.addScope(oss.str());
			auto gs2 = util::makeGroupScope(jpp, oss.str());

			std::vector<double> stoichMatLiquid(nReactions * nComp, 0.0);
			std::vector<double> expFwdLiquid(nReactions * nComp, 0.0);
			std::vector<double> expBwdLiquid(nReactions * nComp, 0.0);
			std::vector<double> rateFwdLiquid(nReactions, 0.0);
			std::vector<double> rateBwdLiquid(nReactions, 0.0);

			std::vector<double> stoichMatSolid(nReactions * nTotalBound, 0.0);
			std::vector<double> expFwdSolid(nReactions * nTotalBound, 0.0);
			std::vector<double> expBwdSolid(nReactions * nTotalBound, 0.0);
			std::vector<double> rateFwdSolid(nReactions, 0.0);
			std::vector<double> rateBwdSolid(nReactions, 0.0);

			util::populate(
				stoichMatLiquid.data(), [](unsigned int idx) { return 2.0 * std::sin(0.5 + idx * 0.7); },
				stoichMatLiquid.size());
			util::populate(
				expFwdLiquid.data(), [](unsigned int idx) { return std::sin(0.1 + idx * 0.7) + 1.4; },
				expFwdLiquid.size());
			util::populate(
				expBwdLiquid.data(), [](unsigned int idx) { return std::sin(0.2 + idx * 0.3) + 1.1; },
				expBwdLiquid.size());
			util::populate(
				rateFwdLiquid.data(), [](unsigned int idx) { return std::sin(0.3 + idx * 0.2) * 0.5 + 1.5; },
				rateFwdLiquid.size());
			util::populate(
				rateBwdLiquid.data(), [](unsigned int idx) { return std::sin(0.4 + idx * 0.4) * 0.5 + 1.6; },
				rateBwdLiquid.size());

			util::populate(
				stoichMatSolid.data(), [](unsigned int idx) { return 2.0 * std::sin(0.3 + idx * 0.9); },
				stoichMatSolid.size());
			util::populate(
				expFwdSolid.data(), [](unsigned int idx) { return std::sin(0.2 + idx * 0.6) + 1.3; },
				expFwdSolid.size());
			util::populate(
				expBwdSolid.data(), [](unsigned int idx) { return std::sin(0.1 + idx * 0.4) + 1.2; },
				expBwdSolid.size());
			util::populate(
				rateFwdSolid.data(), [](unsigned int idx) { return std::sin(0.5 + idx * 0.4) * 0.5 + 1.6; },
				rateFwdSolid.size());
			util::populate(
				rateBwdSolid.data(), [](unsigned int idx) { return std::sin(0.2 + idx * 0.3) * 0.5 + 1.5; },
				rateBwdSolid.size());

			jpp.set("MAL_STOICHIOMETRY_LIQUID", stoichMatLiquid);
			jpp.set("MAL_EXPONENTS_LIQUID_FWD", expFwdLiquid);
			jpp.set("MAL_EXPONENTS_LIQUID_BWD", expBwdLiquid);
			jpp.set("MAL_KFWD_LIQUID", rateFwdLiquid);
			jpp.set("MAL_KBWD_LIQUID", rateBwdLiquid);

			jpp.set("MAL_STOICHIOMETRY_SOLID", stoichMatSolid);
			jpp.set("MAL_EXPONENTS_SOLID_FWD", expFwdSolid);
			jpp.set("MAL_EXPONENTS_SOLID_BWD", expBwdSolid);
			jpp.set("MAL_KFWD_SOLID", rateFwdSolid);
			jpp.set("MAL_KBWD_SOLID", rateBwdSolid);

			if (particleModifiers)
			{
				std::vector<double> expFwdLiquidModSolid(nReactions * nTotalBound, 0.0);
				std::vector<double> expBwdLiquidModSolid(nReactions * nTotalBound, 0.0);
				std::vector<double> expFwdSolidModLiquid(nReactions * nComp, 0.0);
				std::vector<double> expBwdSolidModLiquid(nReactions * nComp, 0.0);

				util::populate(
					expFwdLiquidModSolid.data(), [](unsigned int idx) { return std::sin(idx * 0.7 + 0.1) * 0.5 + 1.6; },
					expFwdLiquidModSolid.size());
				util::populate(
					expBwdLiquidModSolid.data(), [](unsigned int idx) { return std::sin(idx * 0.7 + 0.2) * 0.5 + 1.6; },
					expBwdLiquidModSolid.size());
				util::populate(
					expFwdSolidModLiquid.data(), [](unsigned int idx) { return std::sin(idx * 0.7 + 0.3) * 0.5 + 1.6; },
					expFwdSolidModLiquid.size());
				util::populate(
					expBwdSolidModLiquid.data(), [](unsigned int idx) { return std::sin(idx * 0.7 + 0.4) * 0.5 + 1.6; },
					expBwdSolidModLiquid.size());

				jpp.set("MAL_EXPONENTS_LIQUID_FWD_MODSOLID", expFwdLiquidModSolid);
				jpp.set("MAL_EXPONENTS_LIQUID_BWD_MODSOLID", expBwdLiquidModSolid);
				jpp.set("MAL_EXPONENTS_SOLID_FWD_MODLIQUID", expFwdSolidModLiquid);
				jpp.set("MAL_EXPONENTS_SOLID_BWD_MODLIQUID", expBwdSolidModLiquid);
			}
		}
	}
}

void testDynamicJacobianAD(const char* modelName, unsigned int nComp, unsigned int const* nBound, const char* config,
						   double const* point, double absTol, double relTol)
{
	ConfiguredDynamicReactionModel crm = ConfiguredDynamicReactionModel::create(modelName, nComp, nBound, config);

	const unsigned int numDofs = crm.nComp() + crm.numBoundStates();
	std::vector<double> yState(numDofs, 0.0);
	std::copy_n(point, numDofs, yState.data());

	std::vector<double> dir(numDofs, 0.0);
	std::vector<double> colA(numDofs, 0.0);
	std::vector<double> colB(numDofs, 0.0);

	// Enable AD
	cadet::ad::setDirections(cadet::ad::getMaxDirections());
	cadet::active* adRes = new cadet::active[numDofs];
	cadet::active* adY = new cadet::active[numDofs];

	// Combined liquid and solid phase

	// Evaluate with AD
	ad::prepareAdVectorSeedsForDenseMatrix(adY, 0, numDofs);
	ad::copyToAd(yState.data(), adY, numDofs);
	ad::resetAd(adRes, numDofs);
	crm.model().residualCombinedAdd(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, adY, adY + crm.nComp(), adRes,
									adRes + crm.nComp(), 1.0, crm.buffer());

	// Extract Jacobian
	cadet::linalg::DenseMatrix jacAD;
	jacAD.resize(numDofs, numDofs);
	ad::extractDenseJacobianFromAd(adRes, 0, jacAD);

	// Calculate analytic Jacobian
	cadet::linalg::DenseMatrix jacAna;
	jacAna.resize(numDofs, numDofs);
	crm.model().analyticJacobianCombinedAdd(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yState.data(),
											yState.data() + crm.nComp(), 1.0, jacAna.row(0), jacAna.row(crm.nComp()),
											crm.buffer());

	cadet::test::checkJacobianPatternFD(
		[&](double const* lDir, double* res) -> void {
			std::fill_n(res, numDofs, 0.0);
			crm.model().residualCombinedAdd(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, lDir, lDir + crm.nComp(), res,
											res + crm.nComp(), 1.0, crm.buffer());
		},
		[&](double const* lDir, double* res) -> void { jacAna.multiplyVector(lDir, res); }, yState.data(), dir.data(),
		colA.data(), colB.data(), numDofs, numDofs);

	cadet::test::checkJacobianPatternFD(
		[&](double const* lDir, double* res) -> void {
			std::fill_n(res, numDofs, 0.0);
			crm.model().residualCombinedAdd(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, lDir, lDir + crm.nComp(), res,
											res + crm.nComp(), 1.0, crm.buffer());
		},
		[&](double const* lDir, double* res) -> void { jacAD.multiplyVector(lDir, res); }, yState.data(), dir.data(),
		colA.data(), colB.data(), numDofs, numDofs);

	// Check Jacobians against each other
	for (unsigned int row = 0; row < numDofs; ++row)
	{
		for (unsigned int col = 0; col < numDofs; ++col)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(jacAna.native(row, col) == makeApprox(jacAD.native(row, col), absTol, relTol));
		}
	}

	// Only liquid phase

	// Evaluate with AD
	ad::resetAd(adRes, numDofs);
	crm.model().residualLiquidAdd(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, adY, adRes, 1.0, crm.buffer());

	// Extract Jacobian
	jacAD.setAll(0.0);
	ad::extractDenseJacobianFromAd(adRes, 0, jacAD);

	// Calculate analytic Jacobian
	jacAna.setAll(0.0);
	crm.model().analyticJacobianLiquidAdd(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yState.data(), 1.0, jacAna.row(0),
										  crm.buffer());

	delete[] adY;
	delete[] adRes;

	cadet::test::checkJacobianPatternFD(
		[&](double const* lDir, double* res) -> void {
			std::fill_n(res, nComp, 0.0);
			crm.model().residualLiquidAdd(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, lDir, res, 1.0, crm.buffer());
		},
		[&](double const* lDir, double* res) -> void { jacAna.submatrixMultiplyVector(lDir, 0, 0, nComp, nComp, res); },
		yState.data(), dir.data(), colA.data(), colB.data(), nComp, nComp);

	cadet::test::checkJacobianPatternFD(
		[&](double const* lDir, double* res) -> void {
			std::fill_n(res, nComp, 0.0);
			crm.model().residualLiquidAdd(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, lDir, res, 1.0, crm.buffer());
		},
		[&](double const* lDir, double* res) -> void { jacAD.submatrixMultiplyVector(lDir, 0, 0, nComp, nComp, res); },
		yState.data(), dir.data(), colA.data(), colB.data(), nComp, nComp);

	// Check Jacobians against each other
	for (unsigned int row = 0; row < nComp; ++row)
	{
		for (unsigned int col = 0; col < nComp; ++col)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(jacAna.native(row, col) == makeApprox(jacAD.native(row, col), absTol, relTol));
		}
	}
}

void testUnitJacobianDynamicReactionsAD(cadet::JsonParameterProvider& jpp, bool bulk, bool particle,
										bool particleModifiers)
{
	extendModelWithDynamicReactions(jpp, 0, bulk, particle, particleModifiers);
	unitoperation::testJacobianAD(jpp);
}

void testUnitJacobianDynamicReactionsAD(const std::string& uoType, bool bulk, bool particle, bool particleModifiers)
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType);
	testUnitJacobianDynamicReactionsAD(jpp, bulk, particle, particleModifiers);
}

void testTimeDerivativeJacobianDynamicReactionsFD(cadet::JsonParameterProvider& jpp, bool bulk, bool particle,
												  bool particleModifiers, double h, double absTol, double relTol)
{
	extendModelWithDynamicReactions(jpp, 0, bulk, particle, particleModifiers);
	unitoperation::testTimeDerivativeJacobianFD(jpp, h, absTol, relTol);
}

void testTimeDerivativeJacobianDynamicReactionsFD(const std::string& uoType, bool bulk, bool particle,
												  bool particleModifiers, double h, double absTol, double relTol)
{
	cadet::JsonParameterProvider jpp = createColumnWithTwoCompLinearBinding(uoType);
	testTimeDerivativeJacobianDynamicReactionsFD(jpp, bulk, particle, particleModifiers, h, absTol, relTol);
}

} // namespace reaction
} // namespace test
} // namespace cadet
