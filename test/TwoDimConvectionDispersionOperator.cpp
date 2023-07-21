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

#include "model/parts/TwoDimensionalConvectionDispersionOperator.hpp"
#include "Weno.hpp"
#include "ModelBuilderImpl.hpp"

#include "ColumnTests.hpp"
#include "Utils.hpp"
#include "Dummies.hpp"

#include "common/JsonParameterProvider.hpp"

namespace
{

	inline void createAndConfigureOperator(cadet::model::parts::TwoDimensionalConvectionDispersionOperator& convDispOp, int nComp, int nCol, int nRad, int wenoOrder)
	{
		// Obtain parameters from some test case
		cadet::JsonParameterProvider jpp(R"json({
				"COL_LENGTH": 10,
				"COL_RADIUS": 1,
				"COL_RADIUS_INNER": 0.001,
				"COL_RADIUS_OUTER": 0.004,
				"CROSS_SECTION_AREA": 0.0003141592653589793,
				"COL_POROSITY": 0.37,
				"COL_DISPERSION": 1e-6,
				"COL_DISPERSION_RADIAL": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
				"discretization":
				{
					"RADIAL_DISC_TYPE": "EQUIDISTANT",
					"LINEAR_SOLVER_BULK": "DENSE",
					"weno":
					{
						"WENO_ORDER": 3,
						"BOUNDARY_MODEL": 0,
						"WENO_EPS": 1e-10
					}					
				}
			})json");
		cadet::test::column::setWenoOrder(jpp, wenoOrder);

		std::vector<double> cd(nRad * nComp, 1e-6);
		jpp.set("COL_DISPERSION", cd);

		std::vector<double> cdr(nRad * nComp, 1e-4);
		jpp.set("COL_DISPERSION_RADIAL", cdr);

		std::vector<double> v(2*nRad, 1.0);
		for (int i = nRad; i < 2*nRad; ++i)
			v[i] = -1.0;
		jpp.set("VELOCITY", v);

		// Configure the operator
		typedef std::unordered_map<cadet::ParameterId, cadet::active*> ParameterMap;
		ParameterMap parameters;
		cadet::ModelBuilder builder;
		REQUIRE(convDispOp.configureModelDiscretization(jpp, builder, nComp, nCol, nRad, false));
		REQUIRE(convDispOp.configure(0, jpp, parameters));
	}

	inline void compareSparseJacobianAgainstFD(cadet::model::parts::TwoDimensionalConvectionDispersionOperator& convDispOp, int nInletDof, int nPureDof, double* y, double* jacCol1, double* jacCol2, double h, double relTol, double absTol)
	{
		for (int col = 0; col < nPureDof; ++col)
		{
			const double ref = y[nInletDof + col];

			// Central finite differences
			y[nInletDof + col] = ref * (1.0 + h);
			convDispOp.residual(DummyModel(), 0.0, 0u, y, nullptr, jacCol1, false, cadet::WithoutParamSensitivity());

			y[nInletDof + col] = ref * (1.0 - h);
			convDispOp.residual(DummyModel(), 0.0, 0u, y, nullptr, jacCol2, false, cadet::WithoutParamSensitivity());

			y[nInletDof + col] = ref;

			for (int row = 0; row < nPureDof; ++row)
			{
				const double fdVal = (jacCol1[row + nInletDof] - jacCol2[row + nInletDof]) / (2.0 * ref * h);
				const bool isFDnonZero = (fdVal != 0.0);

				if (isFDnonZero || convDispOp.jacobian().isNonZero(row, col))
				{
					CAPTURE(row);
					CAPTURE(col);
					CHECK(convDispOp.jacobian().native(row, col) == cadet::test::makeApprox(fdVal, relTol, absTol));
				}
			}
		}		
	}

} // namespace

void testBulk2DJacobianWenoForwardBackward(int wenoOrder)
{
	const int nComp = 3;
	const int nRad = 5;
	const int nCol = 19;

	const double h = 1e-6;
	const double relTol = 5e-5;
	const double absTol = 3e-8;

	SECTION("Forward vs backward flow Jacobian (WENO=" + std::to_string(wenoOrder) + ")")
	{
		cadet::model::parts::TwoDimensionalConvectionDispersionOperator convDispOp;
		createAndConfigureOperator(convDispOp, nComp, nCol, nRad, wenoOrder);

		// Obtain memory for state, Jacobian columns
		const int nInletDof = nComp * nRad;
		const int nPureDof = nComp * nCol * nRad;
		const int nDof = nInletDof + nPureDof;
		std::vector<double> y(nDof, 0.0);
		std::vector<double> jacCol1(nDof, 0.0);
		std::vector<double> jacCol2(nDof, 0.0);

		// Fill state vector with some values
		cadet::test::util::populate(y.data() + nInletDof, [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + std::abs(std::sin(idx * 0.3)) + 1e-4; }, nPureDof);

		// Compute Jacobian
		for (int i = 0; i < nRad; ++i)
			convDispOp.setFlowRates(i, 1e-2 * convDispOp.crossSection(i) * convDispOp.columnPorosity(i), 0.0);

		convDispOp.notifyDiscontinuousSectionTransition(0.0, 0u);
		convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, jacCol1.data(), true, cadet::WithoutParamSensitivity());

		// Compare Jacobian pattern against FD
		compareSparseJacobianAgainstFD(convDispOp, nInletDof, nPureDof, y.data(), jacCol1.data(), jacCol2.data(), h, relTol, absTol);

		// Reverse flow
		convDispOp.notifyDiscontinuousSectionTransition(0.0, 1u);
		convDispOp.residual(DummyModel(), 0.0, 1u, y.data(), nullptr, jacCol1.data(), true, cadet::WithoutParamSensitivity());

		// Compare Jacobian pattern against FD
		compareSparseJacobianAgainstFD(convDispOp, nInletDof, nPureDof, y.data(), jacCol1.data(), jacCol2.data(), h, relTol, absTol);
	}
}

void testBulk2DJacobianSparsityWeno(int wenoOrder, bool forwardFlow)
{
	const int nComp = 3;
	const int nRad = 5;
	const int nCol = 19;
	const double h = 1e-6;

	SECTION("WENO=" + std::to_string(wenoOrder))
	{
		cadet::model::parts::TwoDimensionalConvectionDispersionOperator convDispOp;
		createAndConfigureOperator(convDispOp, nComp, nCol, nRad, wenoOrder);

		// Obtain memory for state, Jacobian columns
		const int nInletDof = nComp * nRad;
		const int nPureDof = nComp * nCol * nRad;
		const int nDof = nInletDof + nPureDof;
		std::vector<double> y(nDof, 0.0);
		std::vector<double> jacCol1(nDof, 0.0);
		std::vector<double> jacCol2(nDof, 0.0);

		// Fill state vector with some values
		cadet::test::util::populate(y.data() + nInletDof, [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + std::abs(std::sin(idx * 0.3)) + 1e-4; }, nPureDof);

		// Compute Jacobian
		for (int i = 0; i < nRad; ++i)
			convDispOp.setFlowRates(i, 1e-2 * convDispOp.crossSection(i) * convDispOp.columnPorosity(i), 0.0);

		convDispOp.notifyDiscontinuousSectionTransition(0.0, 0u);
		convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, jacCol1.data(), true, cadet::WithoutParamSensitivity());

		// Compare Jacobian pattern with FD
		for (int col = 0; col < nPureDof; ++col)
		{
			const double ref = y[nInletDof + col];

			// Central finite differences
			y[nInletDof + col] = ref * (1.0 + h);
			convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, jacCol1.data(), false, cadet::WithoutParamSensitivity());

			y[nInletDof + col] = ref * (1.0 - h);
			convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, jacCol2.data(), false, cadet::WithoutParamSensitivity());

			y[nInletDof + col] = ref;

			for (int row = 0; row < nPureDof; ++row)
			{
				const double fd = (jacCol1[row + nInletDof] - jacCol2[row + nInletDof]) / (2.0 * ref * h);
				const bool isFDnonZero = (fd != 0.0);

				CAPTURE(row);
				CAPTURE(col);
				CAPTURE(isFDnonZero);
				CHECK(isFDnonZero == convDispOp.jacobian().isNonZero(row, col));
			}
		}
	}
}

TEST_CASE("TwoDimensionalConvectionDispersionOperator Jacobian forward vs backward flow", "[2D],[Operator],[Residual],[Jacobian]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		testBulk2DJacobianWenoForwardBackward(i);
}

TEST_CASE("TwoDimensionalConvectionDispersionOperator Jacobian sparsity pattern vs FD", "[2D],[Operator],[Residual],[Jacobian],[SparseMatrix]")
{
	SECTION("Forward flow")
	{
		// Test all WENO orders
		for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
			testBulk2DJacobianSparsityWeno(i, true);
	}
	SECTION("Backward flow")
	{
		// Test all WENO orders
		for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
			testBulk2DJacobianSparsityWeno(i, false);
	}
}
