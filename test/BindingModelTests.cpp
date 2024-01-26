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
#include "cadet/cadet.hpp"

#define CADET_LOGGING_DISABLE
#include "Logging.hpp"

#include "JacobianHelper.hpp"
#include "BindingModelTests.hpp"

#include "common/JsonParameterProvider.hpp"

#include "BindingModelFactory.hpp"
#include "model/BindingModel.hpp"
#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"
#include "AdUtils.hpp"
#include "AutoDiff.hpp"

#include <cstring>
#include <algorithm>

namespace
{
inline cadet::model::IBindingModel* createBindingModel(const char* name)
{
	cadet::BindingModelFactory bmf;
	cadet::model::IBindingModel* const bm = bmf.create(name);

	REQUIRE(nullptr != bm);
	return bm;
}
} // namespace

namespace cadet
{

namespace test
{

namespace binding
{

ConfiguredBindingModel::~ConfiguredBindingModel()
{
	delete _binding;
	delete[] _boundOffset;

	for (cadet::IExternalFunction*& ef : _extFuns)
		delete ef;

	::operator delete(_bufferMemory);
}

ConfiguredBindingModel ConfiguredBindingModel::create(const char* name, unsigned int nComp, unsigned int const* nBound,
													  bool isKinetic, const char* config)
{
	cadet::model::IBindingModel* const bm = createBindingModel(name);

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
	jpp.set("IS_KINETIC", isKinetic);
	bm->configureModelDiscretization(jpp, nComp, nBound, boundOffset);
	if (bm->requiresConfiguration())
	{
		jpp.set("EXTFUN", std::vector<int>(1, 0));
		REQUIRE(bm->configure(jpp, 0, 0));
	}

	// Assign external functions
	std::vector<cadet::IExternalFunction*> extFuns(50, nullptr);
	for (int i = 0; i < 50; ++i)
		extFuns[i] = new LinearExternalFunction();

	bm->setExternalFunctions(extFuns.data(), 50);

	// Allocate memory buffer
	unsigned int requiredMem = 0;
	if (bm->requiresWorkspace())
		requiredMem = bm->workspaceSize(nComp, totalBoundStates, boundOffset);

	void* buffer = nullptr;
	void* bufferEnd = nullptr;
	if (requiredMem > 0)
	{
		buffer = ::operator new(requiredMem);
		bufferEnd = static_cast<char*>(buffer) + requiredMem;
		std::memset(buffer, 0, requiredMem);
	}

	return ConfiguredBindingModel(bm, nComp, nBound, boundOffset, buffer, bufferEnd, std::move(extFuns));
}

ConfiguredBindingModel ConfiguredBindingModel::create(const char* name, unsigned int nComp, unsigned int const* nBound,
													  int const* isKinetic, const char* config)
{
	cadet::model::IBindingModel* const bm = createBindingModel(name);

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
	jpp.set("IS_KINETIC", std::vector<int>(isKinetic, isKinetic + nComp));
	bm->configureModelDiscretization(jpp, nComp, nBound, boundOffset);
	if (bm->requiresConfiguration())
	{
		jpp.set("EXTFUN", std::vector<int>(1, 0));
		REQUIRE(bm->configure(jpp, 0, 0));
	}

	// Assign external functions
	std::vector<cadet::IExternalFunction*> extFuns(50, nullptr);
	for (int i = 0; i < 50; ++i)
		extFuns[i] = new LinearExternalFunction();

	bm->setExternalFunctions(extFuns.data(), 50);

	// Allocate memory buffer
	unsigned int requiredMem = 0;
	if (bm->requiresWorkspace())
		requiredMem = bm->workspaceSize(nComp, totalBoundStates, boundOffset);

	void* buffer = nullptr;
	void* bufferEnd = nullptr;
	if (requiredMem > 0)
	{
		buffer = ::operator new(requiredMem);
		bufferEnd = static_cast<char*>(buffer) + requiredMem;
		std::memset(buffer, 0, requiredMem);
	}

	return ConfiguredBindingModel(bm, nComp, nBound, boundOffset, buffer, bufferEnd, std::move(extFuns));
}

void ConfiguredBindingModel::increaseBufferSize(int inc)
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

int ConfiguredBindingModel::requiredBufferSize() CADET_NOEXCEPT
{
	if (_binding->requiresWorkspace())
	{
		unsigned int totalBoundStates = 0;
		for (unsigned int i = 0; i < _nComp; ++i)
			totalBoundStates += _nBound[i];

		return _binding->workspaceSize(_nComp, totalBoundStates, _boundOffset);
	}

	return 0;
}

void testJacobianAD(const char* modelName, unsigned int nComp, unsigned int const* nBound, bool isKinetic,
					const char* config, double const* point, bool skipStructureTest, double absTol, double relTol)
{
	ConfiguredBindingModel cbm = ConfiguredBindingModel::create(modelName, nComp, nBound, isKinetic, config);

	const unsigned int numDofs = cbm.nComp() + cbm.numBoundStates();
	const unsigned int numEq = cbm.numBoundStates();
	std::vector<double> yState(numDofs, 0.0);
	double* const yBound = yState.data() + cbm.nComp();
	std::copy_n(point, numDofs, yState.data());

	std::vector<double> dir(numDofs, 0.0);
	std::vector<double> colA(numEq, 0.0);
	std::vector<double> colB(numEq, 0.0);

	// Calculate analytic Jacobian
	cadet::linalg::DenseMatrix jacAna;
	jacAna.resize(numDofs, numDofs);
	cbm.model().analyticJacobian(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yBound, cbm.nComp(), jacAna.row(cbm.nComp()),
								 cbm.buffer());

	// Enable AD
	cadet::ad::setDirections(cadet::ad::getMaxDirections());
	cadet::active* adRes = new cadet::active[numEq];
	cadet::active* adY = new cadet::active[numDofs];

	// Evaluate with AD
	ad::prepareAdVectorSeedsForDenseMatrix(adY, 0, numDofs);
	ad::copyToAd(yState.data(), adY, numDofs);
	cbm.model().flux(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, adY + cbm.nComp(), adY, adRes, cbm.buffer(),
					 cadet::WithoutParamSensitivity());

	// Extract Jacobian
	cadet::linalg::DenseMatrix jacAD;
	jacAD.resize(numEq, numDofs);
	ad::extractDenseJacobianFromAd(adRes, 0, jacAD);

	delete[] adY;
	delete[] adRes;

	if (!skipStructureTest)
	{
		cadet::test::checkJacobianPatternFD(
			[&](double const* lDir, double* res) -> void {
				cbm.model().flux(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, lDir + cbm.nComp(), lDir, res, cbm.buffer());
			},
			[&](double const* lDir, double* res) -> void {
				jacAna.submatrixMultiplyVector(lDir, cbm.nComp(), 0, numEq, numDofs, res);
			},
			yState.data(), dir.data(), colA.data(), colB.data(), numDofs, numEq);

		cadet::test::checkJacobianPatternFD(
			[&](double const* lDir, double* res) -> void {
				cbm.model().flux(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, lDir + cbm.nComp(), lDir, res, cbm.buffer());
			},
			[&](double const* lDir, double* res) -> void { jacAD.multiplyVector(lDir, res); }, yState.data(),
			dir.data(), colA.data(), colB.data(), numDofs, numEq);
	}

	// Check Jacobians against each other
	for (unsigned int row = 0; row < numEq; ++row)
	{
		for (unsigned int col = 0; col < numDofs; ++col)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(jacAna.native(row + cbm.nComp(), col) == makeApprox(jacAD.native(row, col), absTol, relTol));
		}
	}
}

void testNormalExternalConsistency(const char* modelName, const char* modelNameExt, unsigned int nComp,
								   unsigned int const* nBound, bool isKinetic, const char* config, double const* point)
{
	ConfiguredBindingModel cbm = ConfiguredBindingModel::create(modelName, nComp, nBound, isKinetic, config);
	ConfiguredBindingModel cbmExt = ConfiguredBindingModel::create(modelNameExt, nComp, nBound, isKinetic, config);

	const unsigned int numDofs = cbm.nComp() + cbm.numBoundStates();
	const unsigned int numEq = cbm.numBoundStates();

	std::vector<double> res(numEq, 0.0);
	std::vector<double> resExt(numEq, 0.0);
	std::vector<double> yState(numDofs, 0.0);
	double* const yBound = yState.data() + cbm.nComp();
	std::copy_n(point, numDofs, yState.data());

	// Evaluate residuals
	cbm.model().flux(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yBound, yState.data(), res.data(), cbm.buffer());
	cbmExt.model().flux(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yBound, yState.data(), resExt.data(), cbmExt.buffer());

	// Check residuals against each other
	for (unsigned int i = 0; i < numEq; ++i)
	{
		CAPTURE(i);
		CHECK(resExt[i] == RelApprox(res[i]));
	}

	// Calculate analytic Jacobians
	cadet::linalg::DenseMatrix jac;
	jac.resize(numDofs, numDofs);
	cbm.model().analyticJacobian(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yBound, cbm.nComp(), jac.row(cbm.nComp()),
								 cbm.buffer());

	cadet::linalg::DenseMatrix jacExt;
	jacExt.resize(numDofs, numDofs);
	cbmExt.model().analyticJacobian(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yBound, cbm.nComp(),
									jacExt.row(cbmExt.nComp()), cbmExt.buffer());

	// Check Jacobians against each other
	for (unsigned int r = cbm.nComp(); r < numDofs; ++r)
	{
		const unsigned int row = r - cbm.nComp();
		for (unsigned int col = 0; col < numDofs; ++col)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(jac.native(r, col) == RelApprox(jacExt.native(r, col)));
		}
	}
}

void testNonBindingConsistency(const char* modelName, unsigned int nComp, unsigned int const* nBound, bool isKinetic,
							   const char* config, bool useAD, double const* point)
{
	ConfiguredBindingModel cbm = ConfiguredBindingModel::create(modelName, nComp, nBound, isKinetic, config);

	const unsigned int numDofs = cbm.nComp() + cbm.numBoundStates();
	const unsigned int numEq = cbm.numBoundStates();

	std::vector<double> yState(numDofs, 0.0);
	double* const yBound = yState.data() + cbm.nComp();
	std::copy_n(point, numDofs, yState.data());

	// Calculate Jacobian
	cadet::linalg::DenseMatrix jac;
	if (useAD)
	{
		jac.resize(numEq, numDofs);

		// Enable AD
		cadet::ad::setDirections(cadet::ad::getMaxDirections());
		cadet::active* adRes = new cadet::active[numEq];
		cadet::active* adY = new cadet::active[numDofs];

		// Evaluate with AD
		ad::prepareAdVectorSeedsForDenseMatrix(adY, 0, numDofs);
		ad::copyToAd(yState.data(), adY, numDofs);
		cbm.model().flux(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, adY + cbm.nComp(), adY, adRes, cbm.buffer(),
						 cadet::WithoutParamSensitivity());

		// Extract Jacobian
		ad::extractDenseJacobianFromAd(adRes, 0, jac);

		delete[] adY;
		delete[] adRes;
	}
	else
	{
		jac.resize(numDofs, numDofs);
		cbm.model().analyticJacobian(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yBound, cbm.nComp(), jac.row(cbm.nComp()),
									 cbm.buffer());
	}

	// Check that columns of non-binding liquid phase components are all zero
	for (unsigned int col = 0; col < nComp; ++col)
	{
		if (nBound[col] > 0)
			continue;

		for (unsigned int row = 0; row < numEq; ++row)
		{
			if (useAD)
			{
				CAPTURE(row);
				CAPTURE(col);
				CHECK(jac.native(row, col) == 0.0);
			}
			else
			{
				CAPTURE(row);
				CAPTURE(col);
				CHECK(jac.native(row + cbm.nComp(), col) == 0.0);
			}
		}
	}
}

void testNonbindingBindingConsistency(const char* modelName, unsigned int nCompBnd, unsigned int nCompNonBnd,
									  unsigned int const* nBound, unsigned int const* nBoundNonBnd, bool isKinetic,
									  const char* configBnd, const char* configNonBnd, bool useAD,
									  double const* pointBnd, double const* pointNonBnd)
{
	ConfiguredBindingModel cbmBnd = ConfiguredBindingModel::create(modelName, nCompBnd, nBound, isKinetic, configBnd);
	ConfiguredBindingModel cbmNonBnd =
		ConfiguredBindingModel::create(modelName, nCompNonBnd, nBoundNonBnd, isKinetic, configNonBnd);

	// Setup all binding
	const unsigned int numDofsBnd = cbmBnd.nComp() + cbmBnd.numBoundStates();
	const unsigned int numEqBnd = cbmBnd.numBoundStates();

	std::vector<double> yStateBnd(numDofsBnd, 0.0);
	double* const yBoundBnd = yStateBnd.data() + cbmBnd.nComp();
	std::copy_n(pointBnd, numDofsBnd, yStateBnd.data());

	// Setup with nonbinding components
	const unsigned int numDofsNonBnd = cbmNonBnd.nComp() + cbmNonBnd.numBoundStates();
	const unsigned int numEqNonBnd = cbmNonBnd.numBoundStates();

	std::vector<double> yStateNonBnd(numDofsNonBnd, 0.0);
	double* const yBoundNonBnd = yStateNonBnd.data() + cbmNonBnd.nComp();
	std::copy_n(pointNonBnd, numDofsNonBnd, yStateNonBnd.data());

	// Number of equations has to be the same
	REQUIRE(numEqBnd == numEqNonBnd);

	// Check residual
	std::vector<double> res1(numEqBnd, 0.0);
	std::vector<double> res2(numEqBnd, 0.0);

	cbmBnd.model().flux(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yBoundBnd, yStateBnd.data(), res1.data(),
						cbmBnd.buffer());
	cbmNonBnd.model().flux(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yBoundNonBnd, yStateNonBnd.data(), res2.data(),
						   cbmNonBnd.buffer());

	for (unsigned int i = 0; i < numEqBnd; ++i)
	{
		CAPTURE(i);
		CHECK(res1[i] == res2[i]);
	}

	// Calculate Jacobians
	cadet::linalg::DenseMatrix jacBnd;
	cadet::linalg::DenseMatrix jacNonBnd;
	if (useAD)
	{
		jacBnd.resize(numEqBnd, numDofsBnd);
		jacNonBnd.resize(numEqBnd, numDofsNonBnd);

		// Enable AD
		cadet::ad::setDirections(cadet::ad::getMaxDirections());
		cadet::active* adRes = new cadet::active[numEqBnd];
		cadet::active* adY = new cadet::active[numDofsNonBnd];

		// Evaluate with AD, all binding
		ad::prepareAdVectorSeedsForDenseMatrix(adY, 0, numDofsBnd);
		ad::copyToAd(yStateBnd.data(), adY, numDofsBnd);
		cbmBnd.model().flux(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, adY + cbmBnd.nComp(), adY, adRes, cbmBnd.buffer(),
							cadet::WithoutParamSensitivity());

		// Extract Jacobian, all binding
		ad::extractDenseJacobianFromAd(adRes, 0, jacBnd);

		// Evaluate with AD, with nonbinding
		ad::resetAd(adRes, numEqBnd);
		ad::resetAd(adY, numDofsNonBnd);
		ad::prepareAdVectorSeedsForDenseMatrix(adY, 0, numDofsNonBnd);
		ad::copyToAd(yStateNonBnd.data(), adY, numDofsNonBnd);
		cbmNonBnd.model().flux(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, adY + cbmNonBnd.nComp(), adY, adRes,
							   cbmNonBnd.buffer(), cadet::WithoutParamSensitivity());

		// Extract Jacobian, with nonbinding
		ad::extractDenseJacobianFromAd(adRes, 0, jacNonBnd);

		delete[] adY;
		delete[] adRes;
	}
	else
	{
		jacBnd.resize(numDofsBnd, numDofsBnd);
		jacNonBnd.resize(numDofsNonBnd, numDofsNonBnd);
		cbmBnd.model().analyticJacobian(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yBoundBnd, cbmBnd.nComp(),
										jacBnd.row(cbmBnd.nComp()), cbmBnd.buffer());
		cbmNonBnd.model().analyticJacobian(1.0, 0u, ColumnPosition{0.0, 0.0, 0.0}, yBoundNonBnd, cbmNonBnd.nComp(),
										   jacNonBnd.row(cbmNonBnd.nComp()), cbmNonBnd.buffer());
	}

	// Compare Jacobians
	const unsigned int rowOffsetBnd = useAD ? 0 : cbmBnd.nComp();
	const unsigned int rowOffsetNonBnd = useAD ? 0 : cbmNonBnd.nComp();

	// Check liquid phase
	unsigned int c1 = 0;
	unsigned int c2 = 0;
	for (unsigned int col = 0; col < std::min(nCompBnd, nCompNonBnd); ++col, ++c1, ++c2)
	{
		if ((nBound[c1] == 0) && (nBoundNonBnd[c2] > 0))
			++c1;
		else if ((nBound[c1] > 0) && (nBoundNonBnd[c2] == 0))
			++c2;

		for (unsigned int row = 0; row < numEqBnd; ++row)
		{
			CAPTURE(col);
			CAPTURE(row);
			CHECK(jacBnd.native(row + rowOffsetBnd, c1) == jacNonBnd.native(row + rowOffsetNonBnd, c2));
		}
	}

	// Check all bound states
	for (unsigned int col = 0; col < numEqNonBnd; ++col)
	{
		for (unsigned int row = 0; row < numEqBnd; ++row)
		{
			CAPTURE(col);
			CAPTURE(row);
			CHECK(jacBnd.native(row + rowOffsetBnd, col + nCompBnd) ==
				  jacNonBnd.native(row + rowOffsetNonBnd, col + nCompNonBnd));
		}
	}
}

} // namespace binding
} // namespace test
} // namespace cadet
