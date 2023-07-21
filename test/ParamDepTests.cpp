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
#include "ParamDepTests.hpp"

#include "common/JsonParameterProvider.hpp"

#include "ParameterDependenceFactory.hpp"
#include "model/ParameterDependence.hpp"
#include "linalg/DenseMatrix.hpp"
#include "AdUtils.hpp"
#include "AutoDiff.hpp"

#include <cstring>
#include <algorithm>

namespace
{
	inline cadet::model::IParameterStateDependence* createParameterStateDependence(const char* name)
	{
		cadet::ParameterDependenceFactory pdf;
		cadet::model::IParameterStateDependence* const pd = pdf.createStateDependence(name);
		
		REQUIRE(nullptr != pd);
		return pd;
	}
}

namespace cadet
{

namespace test
{

namespace paramdep
{

ConfiguredParameterDependence::~ConfiguredParameterDependence()
{
	delete[] _boundOffset;
	delete _paramDep;
}

ConfiguredParameterDependence ConfiguredParameterDependence::create(const char* name, unsigned int nComp, unsigned int const* nBound, const char* config)
{
	cadet::model::IParameterStateDependence* const pd = createParameterStateDependence(name);

	// Calculate offset of bound states
	unsigned int* boundOffset = new unsigned int[nComp];
	boundOffset[0] = 0;
	for (unsigned int i = 1; i < nComp; ++i)
	{
		boundOffset[i] = boundOffset[i-1] + nBound[i-1];
	}
	const unsigned int totalBoundStates = boundOffset[nComp - 1] + nBound[nComp - 1];

	// Configure
	cadet::JsonParameterProvider jpp(config);
	pd->configureModelDiscretization(jpp, nComp, nBound, boundOffset);
	if (pd->requiresConfiguration())
	{
		REQUIRE(pd->configure(jpp, 0, cadet::ParTypeIndep, "PD"));
	}

	return ConfiguredParameterDependence(pd, nComp, nBound, boundOffset);
}

void testLiquidJacobianAD(const char* modelName, unsigned int nComp, unsigned int const* nBound, const char* config, double const* point, double absTol, double relTol)
{
	ConfiguredParameterDependence cpd = ConfiguredParameterDependence::create(modelName, nComp, nBound, config);
	const unsigned int numDofs = cpd.nComp();
	const double factor = 0.9;

	// Calculate analytic Jacobian
	cadet::linalg::DenseMatrix jacAna;
	jacAna.resize(numDofs, numDofs);

	for (unsigned int comp = 0; comp < nComp; ++comp)
		cpd.model().analyticJacobianLiquidAdd(ColumnPosition{0.0, 0.0, 0.0}, 2.1, point, comp, factor, 0, jacAna.row(comp));

	// Enable AD
	cadet::ad::setDirections(cadet::ad::getMaxDirections());
	cadet::active* adRes = new cadet::active[numDofs];
	cadet::active* adY = new cadet::active[numDofs];

	// Evaluate with AD
	ad::prepareAdVectorSeedsForDenseMatrix(adY, 0, numDofs);
	ad::copyToAd(point, adY, numDofs);

	for (unsigned int comp = 0; comp < nComp; ++comp)
		adRes[comp] = factor * cpd.model().liquidParameter(ColumnPosition{0.0, 0.0, 0.0}, 2.1, adY, comp);

	// Extract Jacobian
	cadet::linalg::DenseMatrix jacAD;
	jacAD.resize(numDofs, numDofs);
	ad::extractDenseJacobianFromAd(adRes, 0, jacAD);

	delete[] adY;
	delete[] adRes;

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

	// Check promised max elements per row
	for (unsigned int row = 0; row < numDofs; ++row)
	{
		int num = 0;
		for (unsigned int col = 0; col < numDofs; ++col)
		{
			if (jacAna.native(row, col) != 0.0)
				++num;
		}
		CHECK(num <= cpd.model().jacobianElementsPerRowLiquid());
	}
}

void testCombinedJacobianAD(const char* modelName, unsigned int nComp, unsigned int const* nBound, const char* config, double const* point, double absTol, double relTol)
{
	ConfiguredParameterDependence cpd = ConfiguredParameterDependence::create(modelName, nComp, nBound, config);
	const unsigned int numDofs = cpd.nComp() + cpd.numBoundStates();
	const double factor = 0.9;

	// Calculate analytic Jacobian
	cadet::linalg::DenseMatrix jacAna;
	jacAna.resize(numDofs, numDofs);

	for (unsigned int comp = 0; comp < nComp; ++comp)
		cpd.model().analyticJacobianCombinedAddLiquid(ColumnPosition{0.0, 0.0, 0.0}, 2.1, point, point + nComp, comp, factor, 0, jacAna.row(comp));
	for (unsigned int bnd = 0; bnd < cpd.numBoundStates(); ++bnd)
		cpd.model().analyticJacobianCombinedAddSolid(ColumnPosition{0.0, 0.0, 0.0}, 2.1, point, point + nComp, bnd, factor, 0, jacAna.row(nComp + bnd));

	// Enable AD
	cadet::ad::setDirections(cadet::ad::getMaxDirections());
	cadet::active* adRes = new cadet::active[numDofs];
	cadet::active* adY = new cadet::active[numDofs];

	// Evaluate with AD
	ad::prepareAdVectorSeedsForDenseMatrix(adY, 0, numDofs);
	ad::copyToAd(point, adY, numDofs);

	for (unsigned int comp = 0; comp < nComp; ++comp)
		adRes[comp] = factor * cpd.model().combinedParameterLiquid(ColumnPosition{0.0, 0.0, 0.0}, 2.1, adY, adY + nComp, comp);
	for (unsigned int bnd = 0; bnd < cpd.numBoundStates(); ++bnd)
		adRes[nComp + bnd] = factor * cpd.model().combinedParameterSolid(ColumnPosition{0.0, 0.0, 0.0}, 2.1, adY, adY + nComp, bnd);

	// Extract Jacobian
	cadet::linalg::DenseMatrix jacAD;
	jacAD.resize(numDofs, numDofs);
	ad::extractDenseJacobianFromAd(adRes, 0, jacAD);

	delete[] adY;
	delete[] adRes;

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

	// Check promised max elements per row
	for (unsigned int row = 0; row < numDofs; ++row)
	{
		int num = 0;
		for (unsigned int col = 0; col < numDofs; ++col)
		{
			if (jacAna.native(row, col) != 0.0)
				++num;
		}
		CHECK(num <= cpd.model().jacobianElementsPerRowCombined());
	}
}

} // namespace paramdep
} // namespace test
} // namespace cadet
