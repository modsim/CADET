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

#include "model/parts/BindingCellKernel.hpp"
#include "linalg/DenseMatrix.hpp"
#include "BindingModelTests.hpp"
#include "ReactionModelTests.hpp"
#include "ParallelSupport.hpp"
#include "JacobianHelper.hpp"
#include "Utils.hpp"
#include "AdUtils.hpp"
#include "SimulationTypes.hpp"

#include <vector>
#include <numeric>
#include <functional>
#include <string>

namespace
{
inline void recoverTimeDerivativeMatrixFromColumns(const std::function<void(double*, double*)>& multiplier,
												   cadet::linalg::DenseMatrix& mat, double* vec, double* rhs)
{
	std::fill_n(vec, mat.columns(), 0.0);
	for (int i = 0; i < mat.columns(); ++i)
	{
		vec[i] = 1.0;

		multiplier(vec, rhs);

		for (int j = 0; j < mat.rows(); ++j)
			mat.native(j, i) = rhs[j];

		vec[i] = 0.0;
	}
}

inline void compareMatrix(const cadet::linalg::DenseMatrix& matA, const cadet::linalg::DenseMatrix& matB, double relTol,
						  double absTol)
{
	for (int row = 0; row < matA.rows(); ++row)
	{
		for (int col = 0; col < matA.columns(); ++col)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(matA.native(row, col) == cadet::test::makeApprox(matB.native(row, col), relTol, absTol));
		}
	}
}
} // namespace

TEST_CASE("CellKernel time derivative Jacobian multiplier vs matrix", "[CellKernel],[Jacobian]")
{
	const double relTol = 1e-15;
	const double absTol = 1e-15;
	const unsigned int nComp = 5;
	const unsigned int nBound[] = {1, 0, 2, 1, 1};
	const unsigned int boundOffset[] = {0, 1, 1, 3, 4};
	const unsigned int nTotalBound = std::accumulate(nBound, nBound + nComp, 0u);

	std::vector<cadet::active> poreAccessFactor(nComp, 1.0);
	std::vector<double> y(nComp + nTotalBound, 0.0);
	std::vector<double> res(nComp + nTotalBound, 0.0);

	cadet::linalg::DenseMatrix matMult;
	matMult.resize(nComp + nTotalBound, nComp + nTotalBound);

	cadet::linalg::DenseMatrix matDirect;
	matDirect.resize(nComp + nTotalBound, nComp + nTotalBound);

	const std::function<void(int const*)> testFun = [=, &matMult, &matDirect, &poreAccessFactor, &y,
													 &res](int const* qsReaction) {
		recoverTimeDerivativeMatrixFromColumns(
			[=](double* a, double* b) {
				cadet::model::parts::cell::multiplyWithDerivativeJacobianKernel<true>(
					a, b, nComp, nBound, boundOffset, nTotalBound, qsReaction, 1.0, 3.0);
			},
			matMult, y.data(), res.data());

		cadet::linalg::DenseMatrix::RowIterator jac = matDirect.row(0);
		cadet::model::parts::cell::addTimeDerivativeToJacobianParticleShell<cadet::linalg::DenseMatrix::RowIterator,
																			true>(
			jac, 1.0, 0.25, nComp, nBound, poreAccessFactor.data(), nTotalBound, boundOffset, qsReaction);

		compareMatrix(matMult, matDirect, relTol, absTol);
	};

	SECTION("Multi bound state is quasi-stationary")
	{
		const int qsReaction[] = {0, 1, 1, 1, 0};
		testFun(qsReaction);
	}

	SECTION("Multi bound state is dynamic")
	{
		const int qsReaction[] = {1, 0, 0, 0, 1};
		testFun(qsReaction);
	}

	SECTION("Multi bound state is mixed")
	{
		const int qsReaction[] = {1, 0, 1, 0, 1};
		testFun(qsReaction);
	}

	SECTION("All dynamic")
	{
		const int qsReaction[] = {1, 1, 1, 1, 1};
		testFun(qsReaction);
	}

	SECTION("All quasi-stationary")
	{
		const int qsReaction[] = {0, 0, 0, 0, 0};
		testFun(qsReaction);
	}
}

TEST_CASE("CellKernel time derivative Jacobian analytic vs FD", "[CellKernel],[Jacobian]")
{
	const unsigned int nComp = 5;
	const unsigned int nBound[] = {1, 0, 2, 1, 1};
	const unsigned int nTotalBound = std::accumulate(nBound, nBound + nComp, 0u);

	std::vector<cadet::active> poreAccessFactor(nComp, 1.0);
	const cadet::active porosity = 0.25;

	const int qsReaction[] = {0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};

	for (int mode = 0; mode < 5; ++mode)
	{
		SECTION("Mode " + std::to_string(mode))
		{
			int const* const localQSreaction = qsReaction + nTotalBound * mode;

			cadet::test::binding::ConfiguredBindingModel cbm = cadet::test::binding::ConfiguredBindingModel::create(
				"MULTISTATE_STERIC_MASS_ACTION", nComp, nBound, localQSreaction,
				R"json({"MSSMA_KA": [0.0, 3.55, 1.59, 4.0, 5.0],
	                    "MSSMA_KD": [0.0, 10.0, 10.0, 10.0, 10.0],
	                    "MSSMA_NU": [0.0, 1.5, 2.0, 1.9, 2.3],
	                    "MSSMA_SIGMA": [0.0, 2.83, 3.6, 5.8, 6.5],
	                    "MSSMA_RATES": [0.0, 0.9, 0.8, 1.2, 1.1, 1.4, 1.3],
	                    "MSSMA_LAMBDA": 100.0})json");

			// Obtain memory for state, Jacobian multiply direction, Jacobian column
			const unsigned int nDof = nComp + nTotalBound;
			std::vector<double> y(nDof, 0.0);
			std::vector<double> yDot(nDof, 0.0);
			std::vector<double> jacDir(nDof, 0.0);
			std::vector<double> jacCol1(nDof, 0.0);
			std::vector<double> jacCol2(nDof, 0.0);

			// Fill state vectors with some values
			cadet::test::util::populate(
				y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
			cadet::test::util::populate(
				yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);
			y[nComp] = 90.0;

			const cadet::ColumnPosition colPos{0.0, 0.0, 0.0};
			const cadet::model::parts::cell::CellParameters params{
				nComp,           nBound,   cbm.boundOffset(),       nTotalBound,
				localQSreaction, porosity, poreAccessFactor.data(), &cbm.model(),
				nullptr};

			// Compare Jacobians
			cadet::test::compareJacobianFD(
				[=, &cbm, &colPos, &y, &params](double const* fx, double* fy) {
					cadet::linalg::DenseMatrix::RowIterator jac;
					cadet::model::parts::cell::residualKernel<double, double, double,
															  cadet::model::parts::cell::CellParameters,
															  cadet::linalg::DenseMatrix::RowIterator, false, true>(
						0.0, 0u, colPos, y.data(), fx, fy, jac, params, cbm.buffer());
				},
				[=, &cbm](double const* fx, double* fy) {
					cadet::model::parts::cell::multiplyWithDerivativeJacobianKernel<true>(
						fx, fy, nComp, nBound, cbm.boundOffset(), nTotalBound, localQSreaction, 1.0, 3.0);
				},
				yDot.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), nDof, 1e-7, 1e-14, 1e-4);
		}
	}
}

TEST_CASE("CellKernel Jacobian analytic vs AD", "[CellKernel],[Jacobian],[AD]")
{
	const double relTol = 1e-13;
	const double absTol = 1e-10;
	const unsigned int nComp = 5;
	const unsigned int nBound[] = {1, 0, 2, 1, 1};
	const unsigned int nTotalBound = std::accumulate(nBound, nBound + nComp, 0u);

	std::vector<cadet::active> poreAccessFactor(nComp, 1.0);
	const cadet::active porosity = 0.25;

	const int qsReaction[] = {0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};

	for (int mode = 0; mode < 5; ++mode)
	{
		SECTION("Mode " + std::to_string(mode))
		{
			int const* const localQSreaction = qsReaction + nTotalBound * mode;

			cadet::test::binding::ConfiguredBindingModel cbm = cadet::test::binding::ConfiguredBindingModel::create(
				"MULTISTATE_STERIC_MASS_ACTION", nComp, nBound, localQSreaction,
				R"json({"MSSMA_KA": [0.0, 3.55, 1.59, 4.0, 5.0],
	                    "MSSMA_KD": [0.0, 10.0, 10.0, 10.0, 10.0],
	                    "MSSMA_NU": [0.0, 1.5, 2.0, 1.9, 2.3],
	                    "MSSMA_SIGMA": [0.0, 2.83, 3.6, 5.8, 6.5],
	                    "MSSMA_RATES": [0.0, 0.9, 0.8, 1.2, 1.1, 1.4, 1.3],
	                    "MSSMA_LAMBDA": 100.0})json");

			// Obtain memory for state, Jacobian multiply direction, Jacobian column
			const unsigned int nDof = nComp + nTotalBound;
			std::vector<double> y(nDof, 0.0);
			std::vector<double> yDot(nDof, 0.0);

			// Fill state vectors with some values
			cadet::test::util::populate(
				y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
			cadet::test::util::populate(
				yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);
			y[nComp] = 90.0;

			// Enable AD
			cadet::ad::setDirections(cadet::ad::getMaxDirections());
			cadet::active* adRes = new cadet::active[nDof];
			cadet::active* adY = new cadet::active[nDof];

			cadet::ad::copyToAd(y.data(), adY, nDof);
			cadet::ad::prepareAdVectorSeedsForDenseMatrix(adY, 0u, nDof);

			const cadet::ColumnPosition colPos{0.0, 0.0, 0.0};
			const cadet::model::parts::cell::CellParameters params{
				nComp,           nBound,   cbm.boundOffset(),       nTotalBound,
				localQSreaction, porosity, poreAccessFactor.data(), &cbm.model(),
				nullptr};

			// Calculate Jacobian via AD
			cadet::linalg::DenseMatrix::RowIterator jac;
			cadet::model::parts::cell::residualKernel<cadet::active, cadet::active, double,
													  cadet::model::parts::cell::CellParameters,
													  cadet::linalg::DenseMatrix::RowIterator, false, true>(
				0.0, 0u, colPos, adY, yDot.data(), adRes, jac, params, cbm.buffer());
			cadet::linalg::DenseMatrix matAd;
			matAd.resize(nDof, nDof);
			cadet::ad::extractDenseJacobianFromAd(adRes, 0u, matAd);

			// Calculate analytic Jacobian
			std::vector<double> res(nDof, 0.0);
			cadet::linalg::DenseMatrix matAna;
			matAna.resize(nDof, nDof);
			jac = matAna.row(0);
			cadet::model::parts::cell::residualKernel<double, double, double, cadet::model::parts::cell::CellParameters,
													  cadet::linalg::DenseMatrix::RowIterator, true, true>(
				0.0, 0u, colPos, y.data(), yDot.data(), res.data(), jac, params, cbm.buffer());

			for (unsigned int i = 0; i < nDof; ++i)
				CHECK(res[i] == static_cast<double>(adRes[i]));

			delete[] adRes;
			delete[] adY;

			// Compare
			compareMatrix(matAna, matAd, relTol, absTol);
		}
	}
}

TEST_CASE("CellKernel time derivative Jacobian analytic vs FD with dummy reaction", "[CellKernel],[Jacobian]")
{
	const unsigned int nComp = 5;
	const unsigned int nBound[] = {1, 0, 2, 1, 1};
	const unsigned int nTotalBound = std::accumulate(nBound, nBound + nComp, 0u);

	std::vector<cadet::active> poreAccessFactor(nComp, 1.0);
	const cadet::active porosity = 0.25;

	const int qsReaction[] = {0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};

	for (int mode = 0; mode < 5; ++mode)
	{
		SECTION("Mode " + std::to_string(mode))
		{
			int const* const localQSreaction = qsReaction + nTotalBound * mode;

			cadet::test::binding::ConfiguredBindingModel cbm = cadet::test::binding::ConfiguredBindingModel::create(
				"MULTISTATE_STERIC_MASS_ACTION", nComp, nBound, localQSreaction,
				R"json({"MSSMA_KA": [0.0, 3.55, 1.59, 4.0, 5.0],
	                    "MSSMA_KD": [0.0, 10.0, 10.0, 10.0, 10.0],
	                    "MSSMA_NU": [0.0, 1.5, 2.0, 1.9, 2.3],
	                    "MSSMA_SIGMA": [0.0, 2.83, 3.6, 5.8, 6.5],
	                    "MSSMA_RATES": [0.0, 0.9, 0.8, 1.2, 1.1, 1.4, 1.3],
	                    "MSSMA_LAMBDA": 100.0})json");

			cadet::test::reaction::ConfiguredDynamicReactionModel cdrm =
				cadet::test::reaction::ConfiguredDynamicReactionModel::create("NONE", nComp, nBound, R"json({})json");

			cbm.increaseBufferSize(nTotalBound * (nTotalBound + nComp + 1) * sizeof(cadet::active) +
								   cdrm.requiredBufferSize());

			// Obtain memory for state, Jacobian multiply direction, Jacobian column
			const unsigned int nDof = nComp + nTotalBound;
			std::vector<double> y(nDof, 0.0);
			std::vector<double> yDot(nDof, 0.0);
			std::vector<double> jacDir(nDof, 0.0);
			std::vector<double> jacCol1(nDof, 0.0);
			std::vector<double> jacCol2(nDof, 0.0);

			// Fill state vectors with some values
			cadet::test::util::populate(
				y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
			cadet::test::util::populate(
				yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);
			y[nComp] = 90.0;

			const cadet::ColumnPosition colPos{0.0, 0.0, 0.0};
			const cadet::model::parts::cell::CellParameters params{
				nComp,           nBound,   cbm.boundOffset(),       nTotalBound,
				localQSreaction, porosity, poreAccessFactor.data(), &cbm.model(),
				&cdrm.model()};

			// Compare Jacobians
			cadet::test::compareJacobianFD(
				[=, &cbm, &colPos, &y, &params](double const* fx, double* fy) {
					cadet::linalg::DenseMatrix::RowIterator jac;
					cadet::model::parts::cell::residualKernel<double, double, double,
															  cadet::model::parts::cell::CellParameters,
															  cadet::linalg::DenseMatrix::RowIterator, false, true>(
						0.0, 0u, colPos, y.data(), fx, fy, jac, params, cbm.buffer());
				},
				[=, &cbm](double const* fx, double* fy) {
					cadet::model::parts::cell::multiplyWithDerivativeJacobianKernel<true>(
						fx, fy, nComp, nBound, cbm.boundOffset(), nTotalBound, localQSreaction, 1.0, 3.0);
				},
				yDot.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), nDof, 1e-7, 1e-14, 1e-4);
		}
	}
}

TEST_CASE("CellKernel Jacobian analytic vs AD with dummy reaction", "[CellKernel],[Jacobian],[AD]")
{
	const double relTol = 1e-13;
	const double absTol = 1e-10;
	const unsigned int nComp = 5;
	const unsigned int nBound[] = {1, 0, 2, 1, 1};
	const unsigned int nTotalBound = std::accumulate(nBound, nBound + nComp, 0u);

	std::vector<cadet::active> poreAccessFactor(nComp, 1.0);
	const cadet::active porosity = 0.25;

	const int qsReaction[] = {0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};

	for (int mode = 0; mode < 5; ++mode)
	{
		SECTION("Mode " + std::to_string(mode))
		{
			int const* const localQSreaction = qsReaction + nTotalBound * mode;

			cadet::test::binding::ConfiguredBindingModel cbm = cadet::test::binding::ConfiguredBindingModel::create(
				"MULTISTATE_STERIC_MASS_ACTION", nComp, nBound, localQSreaction,
				R"json({"MSSMA_KA": [0.0, 3.55, 1.59, 4.0, 5.0],
	                    "MSSMA_KD": [0.0, 10.0, 10.0, 10.0, 10.0],
	                    "MSSMA_NU": [0.0, 1.5, 2.0, 1.9, 2.3],
	                    "MSSMA_SIGMA": [0.0, 2.83, 3.6, 5.8, 6.5],
	                    "MSSMA_RATES": [0.0, 0.9, 0.8, 1.2, 1.1, 1.4, 1.3],
	                    "MSSMA_LAMBDA": 100.0})json");

			cadet::test::reaction::ConfiguredDynamicReactionModel cdrm =
				cadet::test::reaction::ConfiguredDynamicReactionModel::create("NONE", nComp, nBound, R"json({})json");

			cbm.increaseBufferSize(nTotalBound * (nTotalBound + nComp + 1) * sizeof(cadet::active) +
								   cdrm.requiredBufferSize());

			// Obtain memory for state, Jacobian multiply direction, Jacobian column
			const unsigned int nDof = nComp + nTotalBound;
			std::vector<double> y(nDof, 0.0);
			std::vector<double> yDot(nDof, 0.0);

			// Fill state vectors with some values
			cadet::test::util::populate(
				y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
			cadet::test::util::populate(
				yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);
			y[nComp] = 90.0;

			// Enable AD
			cadet::ad::setDirections(cadet::ad::getMaxDirections());
			cadet::active* adRes = new cadet::active[nDof];
			cadet::active* adY = new cadet::active[nDof];

			cadet::ad::copyToAd(y.data(), adY, nDof);
			cadet::ad::prepareAdVectorSeedsForDenseMatrix(adY, 0u, nDof);

			const cadet::ColumnPosition colPos{0.0, 0.0, 0.0};
			const cadet::model::parts::cell::CellParameters params{
				nComp,           nBound,   cbm.boundOffset(),       nTotalBound,
				localQSreaction, porosity, poreAccessFactor.data(), &cbm.model(),
				&cdrm.model()};

			// Calculate Jacobian via AD
			cadet::linalg::DenseMatrix::RowIterator jac;
			cadet::model::parts::cell::residualKernel<cadet::active, cadet::active, double,
													  cadet::model::parts::cell::CellParameters,
													  cadet::linalg::DenseMatrix::RowIterator, false, true>(
				0.0, 0u, colPos, adY, yDot.data(), adRes, jac, params, cbm.buffer());
			cadet::linalg::DenseMatrix matAd;
			matAd.resize(nDof, nDof);
			cadet::ad::extractDenseJacobianFromAd(adRes, 0u, matAd);

			// Calculate analytic Jacobian
			std::vector<double> res(nDof, 0.0);
			cadet::linalg::DenseMatrix matAna;
			matAna.resize(nDof, nDof);
			jac = matAna.row(0);
			cadet::model::parts::cell::residualKernel<double, double, double, cadet::model::parts::cell::CellParameters,
													  cadet::linalg::DenseMatrix::RowIterator, true, true>(
				0.0, 0u, colPos, y.data(), yDot.data(), res.data(), jac, params, cbm.buffer());

			for (unsigned int i = 0; i < nDof; ++i)
				CHECK(res[i] == static_cast<double>(adRes[i]));

			delete[] adRes;
			delete[] adY;

			// Compare
			compareMatrix(matAna, matAd, relTol, absTol);
		}
	}
}
