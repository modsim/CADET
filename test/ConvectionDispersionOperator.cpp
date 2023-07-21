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

#include "model/parts/ConvectionDispersionOperator.hpp"
#include "model/parts/AxialConvectionDispersionKernel.hpp"
#include "model/parts/RadialConvectionDispersionKernel.hpp"
#include "Weno.hpp"
#include "AdUtils.hpp"
#include "SimulationTypes.hpp"
#include "ModelBuilderImpl.hpp"
#include "ParameterDependenceFactory.hpp"

#include "JsonTestModels.hpp"
#include "ColumnTests.hpp"
#include "JacobianHelper.hpp"
#include "Utils.hpp"
#include "Dummies.hpp"

#include <cmath>
#include <functional>
#include <memory>

namespace
{
	/**
	 * @brief Fills the bulk part of the state vector with a given function
	 * @details The function @p f uses the component index, the column cell index, and the 
	 *          current (forward flow) index to assign a value.
	 * @param [out] y Filled state vector
	 * @param [in] f Function for computing the content of the bulk part of the state vector
	 * @param [in] nComp Number of components
	 * @param [in] nCol Number of bulk cells per component
	 */
	inline void fillStateBulkFwd(double* y, std::function<double(unsigned int, unsigned int, unsigned int)> f, unsigned int nComp, unsigned int nCol)
	{
		for (unsigned int col = 0; col < nCol; ++col)
		{
			double* const r = y + nComp + nComp * col;
			for (unsigned int comp = 0; comp < nComp; ++comp)
			{
				r[comp] = f(comp, col, col * nComp + comp);
			}
		}
	}

	/**
	 * @brief Fills the bulk part of the state vector with a given function assuming backwards flow
	 * @details The function @p f uses the component index, the column cell index, and the 
	 *          current (forward flow) index to assign a value.
	 * @param [out] y Filled state vector
	 * @param [in] f Function for computing the content of the bulk part of the state vector
	 * @param [in] nComp Number of components
	 * @param [in] nCol Number of bulk cells per component
	 */
	inline void fillStateBulkBwd(double* y, std::function<double(unsigned int, unsigned int, unsigned int)> f, unsigned int nComp, unsigned int nCol)
	{
		for (unsigned int col = nCol-1; col < nCol; --col)
		{
			double* const r = y + nComp + nComp * col;
			for (unsigned int comp = 0; comp < nComp; ++comp)
			{
				r[comp] = f(comp, nCol - col - 1, (nCol - col - 1) * nComp + comp);
			}
		}
	}

	/**
	 * @brief Compares two residual bulk parts where one is created by forward and the other by backwards flow
	 * @param [in] r1 Residual of one mode (forward / backward)
	 * @param [in] r2 Residual of the other mode (backward / forward)
	 * @param [in] nComp Number of components
	 * @param [in] nCol Number of bulk cells per component
	 */
	inline void compareResidualBulkFwdBwd(double const* r1, double const* r2, int nComp, int nCol)
	{
		for (int col = 0; col < nCol; ++col)
		{
			double const* const a = r1 + nComp + nComp * col;
			double const* const b = r2 + nComp + nComp * (nCol - col - 1);
			for (int comp = 0; comp < nComp; ++comp)
			{
				CHECK(a[comp] == RelApprox(b[comp]));
			}
		}	
	}

	/**
	 * @brief Compares two residual bulk parts that have the same flow direction
	 * @param [in] r1 Residual
	 * @param [in] r2 Residual
	 * @param [in] nComp Number of components
	 * @param [in] nCol Number of bulk cells per component
	 */
	inline void compareResidualBulkFwdFwd(double const* r1, double const* r2, int nComp, int nCol)
	{
		for (int col = 0; col < nCol; ++col)
		{
			double const* const a = r1 + nComp + nComp * col;
			double const* const b = r2 + nComp + nComp * col;
			for (int comp = 0; comp < nComp; ++comp)
			{
				CHECK(a[comp] == RelApprox(b[comp]));
			}
		}	
	}

	template <typename OperatorType>
	inline cadet::active* createAndConfigureOperator(OperatorType& convDispOp, int& nComp, int& nCol, int wenoOrder)
	{
		// Obtain parameters from some test case
		cadet::JsonParameterProvider jpp = createColumnWithSMA("GENERAL_RATE_MODEL");
		cadet::test::column::setWenoOrder(jpp, wenoOrder);

		nComp = jpp.getInt("NCOMP");
		jpp.pushScope("discretization");
		nCol = jpp.getInt("NCOL");
		jpp.popScope();

		// Configure the operator
		typedef std::unordered_map<cadet::ParameterId, cadet::active*> ParameterMap;
		ParameterMap parameters;
		cadet::ModelBuilder builder;
		REQUIRE(convDispOp.configureModelDiscretization(jpp, builder, nComp, nCol));
		REQUIRE(convDispOp.configure(0, jpp, parameters));

		// Make sure that VELOCITY parameter is present
		const cadet::ParameterId paramVelocity = cadet::makeParamId(cadet::hashString("VELOCITY"), 0, cadet::CompIndep, cadet::ParTypeIndep, cadet::BoundStateIndep, cadet::ReactionIndep, cadet::SectionIndep);
		const typename ParameterMap::iterator itVelocity = parameters.find(paramVelocity);
		REQUIRE(itVelocity != parameters.end());

		return itVelocity->second;
	}

} // namespace

template <typename OperatorType>
void testResidualBulkWenoForwardBackward(int wenoOrder)
{
	SECTION("Forward vs backward flow residual bulk (WENO=" + std::to_string(wenoOrder) + ")")
	{
		int nComp = 0;
		int nCol = 0;
		OperatorType convDispOp;
		cadet::active* const velocity = createAndConfigureOperator(convDispOp, nComp, nCol, wenoOrder);
		const double origVelocity = velocity->getValue();

		// Setup matrices
		convDispOp.notifyDiscontinuousSectionTransition(0.0, 0u, cadet::AdJacobianParams{nullptr, nullptr, 0u});

		// Obtain memory for state and residual
		const int nDof = (nCol + 1) * nComp;
		std::vector<double> y(nDof, 0.0);
		std::vector<double> res(nDof, 0.0);

		// The first nComp DOFs are inlet DOFs
		for (int i = 0; i < nComp; ++i)
			y[i] = i + 1.0;

		// Check forward against backward residual with zero state
		SECTION("Forward flow yields backwards flow residual (zero state)")
		{
			// Forward flow residual
			convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, res.data(), false, cadet::WithoutParamSensitivity());

			// Reverse flow
			velocity->setValue(-origVelocity);
			convDispOp.notifyDiscontinuousSectionTransition(0.0, 0u, cadet::AdJacobianParams{nullptr, nullptr, 0u});
			std::vector<double> resRev(nDof, 0.0);

			// Backward flow residual
			convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, resRev.data(), false, cadet::WithoutParamSensitivity());

			// Compare
			compareResidualBulkFwdBwd(res.data(), resRev.data(), nComp, nCol);

			// Reverse flow again to go back to forward
			velocity->setValue(origVelocity);
			convDispOp.notifyDiscontinuousSectionTransition(0.0, 0u, cadet::AdJacobianParams{nullptr, nullptr, 0u});
			std::vector<double> resFwd2(nDof, 0.0);

			// Forward flow residual
			convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, resFwd2.data(), false, cadet::WithoutParamSensitivity());

			// Compare against first forward flow residual
			compareResidualBulkFwdFwd(res.data(), resFwd2.data(), nComp, nCol);
		}

		// Check forward against backward residual with some non-zero state
		SECTION("Forward flow yields backwards flow residual (nonzero state)")
		{
			// Fill state vector with some values
			fillStateBulkFwd(y.data(), [](unsigned int comp, unsigned int col, unsigned int idx) { return std::abs(std::sin(idx * 0.13)); }, nComp, nCol);

			convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, res.data(), false, cadet::WithoutParamSensitivity());

			// Reverse state for backwards flow
			fillStateBulkBwd(y.data(), [](unsigned int comp, unsigned int col, unsigned int idx) { return std::abs(std::sin(idx * 0.13)); }, nComp, nCol);

			// Reverse flow
			velocity->setValue(-origVelocity);
			convDispOp.notifyDiscontinuousSectionTransition(0.0, 0u, cadet::AdJacobianParams{nullptr, nullptr, 0u});
			std::vector<double> resRev(nDof, 0.0);

			// Backward flow residual
			convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, resRev.data(), false, cadet::WithoutParamSensitivity());

			// Compare
			compareResidualBulkFwdBwd(res.data(), resRev.data(), nComp, nCol);

			// Reverse flow again to go back to forward
			velocity->setValue(origVelocity);
			convDispOp.notifyDiscontinuousSectionTransition(0.0, 0u, cadet::AdJacobianParams{nullptr, nullptr, 0u});
			std::vector<double> resFwd2(nDof, 0.0);

			// Fill state vector with forward values again
			fillStateBulkFwd(y.data(), [](unsigned int comp, unsigned int col, unsigned int idx) { return std::abs(std::sin(idx * 0.13)); }, nComp, nCol);

			// Forward flow residual
			convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, resFwd2.data(), false, cadet::WithoutParamSensitivity());

			// Compare against first forward flow residual
			compareResidualBulkFwdFwd(res.data(), resFwd2.data(), nComp, nCol);
		}
	}
}

template <typename OperatorType>
void testTimeDerivativeBulkJacobianFD(double h, double absTol, double relTol)
{
	int nComp = 0;
	int nCol = 0;
	OperatorType convDispOp;
	createAndConfigureOperator(convDispOp, nComp, nCol, cadet::Weno::maxOrder());

	// Setup matrices
	convDispOp.notifyDiscontinuousSectionTransition(0.0, 0u, cadet::AdJacobianParams{nullptr, nullptr, 0u});

	// Obtain memory for state, Jacobian multiply direction, Jacobian column
	const int nDof = (nCol + 1) * nComp;
	std::vector<double> y(nDof, 0.0);
	std::vector<double> yDot(nDof, 0.0);
	std::vector<double> jacDir(nDof, 0.0);
	std::vector<double> jacCol1(nDof, 0.0);
	std::vector<double> jacCol2(nDof, 0.0);

	// Fill state vectors with some values
	cadet::test::util::populate(y.data(), [=](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof);
	cadet::test::util::populate(yDot.data(), [=](unsigned int idx) { return std::abs(std::sin((idx + nDof) * 0.13)) + 1e-4; }, nDof);

	// Compare Jacobians
	cadet::test::compareJacobianFD(
		[&](double const* dir, double* res) -> void { convDispOp.residual(DummyModel(), 0.0, 0u, y.data(), dir, res, false, cadet::WithoutParamSensitivity()); }, 
		[&](double const* dir, double* res) -> void { convDispOp.multiplyWithDerivativeJacobian(cadet::SimulationTime{0.0, 0u}, dir, res); }, 
		yDot.data(), jacDir.data(), jacCol1.data(), jacCol2.data(), nDof, h, absTol, relTol);
}

template <typename OperatorType>
void testBulkJacobianWenoForwardBackward(int wenoOrder)
{
	SECTION("Forward vs backward flow Jacobian (WENO=" + std::to_string(wenoOrder) + ")")
	{
		int nComp = 0;
		int nCol = 0;
		OperatorType opAna;
		OperatorType opAD;
		cadet::active* const anaVelocity = createAndConfigureOperator(opAna, nComp, nCol, wenoOrder);
		cadet::active* const adVelocity = createAndConfigureOperator(opAD, nComp, nCol, wenoOrder);

		// Enable AD
		const unsigned int nDof = nComp + nCol * nComp;
		cadet::ad::setDirections(cadet::ad::getMaxDirections());
		cadet::active* adRes = new cadet::active[nDof];
		cadet::active* adY = new cadet::active[nDof];

		opAD.prepareADvectors(cadet::AdJacobianParams{adRes, adY, 0u});

		// Setup matrices
		opAna.notifyDiscontinuousSectionTransition(0.0, 0u, cadet::AdJacobianParams{nullptr, nullptr, 0u});
		opAD.notifyDiscontinuousSectionTransition(0.0, 0u, cadet::AdJacobianParams{adRes, adY, 0u});

		// Obtain memory for state, Jacobian multiply direction, Jacobian column
		std::vector<double> y(nDof, 0.0);
		std::vector<double> jacDir(nDof, 0.0);
		std::vector<double> jacCol1(nDof, 0.0);
		std::vector<double> jacCol2(nDof, 0.0);

		// Fill state vector with some values
		cadet::test::util::populate(y.data() + nComp, [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof - nComp);

		SECTION("Forward then backward flow (nonzero state)")
		{
			// Compute state Jacobian
			opAna.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, jacDir.data(), true, cadet::WithoutParamSensitivity());
			std::fill(jacDir.begin(), jacDir.end(), 0.0);

			cadet::ad::copyToAd(y.data(), adY, nDof);
			cadet::ad::resetAd(adRes, nDof);
			opAD.residual(DummyModel(), 0.0, 0u, adY, nullptr, adRes, false, cadet::WithoutParamSensitivity());
			opAD.extractJacobianFromAD(adRes, 0);

			const std::function<void(double const*, double*)> anaResidual = [&](double const* lDir, double* res) -> void
				{
					opAna.residual(DummyModel(), 0.0, 0u, lDir - nComp, nullptr, res - nComp, false, cadet::WithoutParamSensitivity());
				};

			const std::function<void(double const*, double*)> anaMultJac = [&](double const* lDir, double* res) -> void
				{
					opAna.jacobian().multiplyVector(lDir, 1.0, 0.0, res);
				};

			const std::function<void(double const*, double*)> adMultJac = [&](double const* lDir, double* res) -> void
				{
					opAD.jacobian().multiplyVector(lDir, 1.0, 0.0, res);
				};

			// Compare Jacobians
			// Check AD Jacobian pattern
			cadet::test::checkJacobianPatternFD(anaResidual, adMultJac, y.data() + nComp, jacDir.data() + nComp, jacCol1.data() + nComp, jacCol2.data() + nComp, nDof - nComp);

			// Check analytic Jacobian pattern
			cadet::test::checkJacobianPatternFD(anaResidual, anaMultJac, y.data() + nComp, jacDir.data() + nComp, jacCol1.data() + nComp, jacCol2.data() + nComp, nDof - nComp);

			// Check analytic vs AD Jacobian
			cadet::test::compareJacobian(anaMultJac, adMultJac, jacDir.data(), jacCol1.data(), jacCol2.data(), nDof - nComp);

			// Reverse flow
			anaVelocity->setValue(-anaVelocity->getValue());
			adVelocity->setValue(-adVelocity->getValue());

			// Setup
			opAna.notifyDiscontinuousSectionTransition(0.0, 0u, cadet::AdJacobianParams{nullptr, nullptr, 0u});
			opAD.notifyDiscontinuousSectionTransition(0.0, 0u, cadet::AdJacobianParams{adRes, adY, 0u});

			// Compute state Jacobian
			opAna.residual(DummyModel(), 0.0, 0u, y.data(), nullptr, jacDir.data(), true, cadet::WithoutParamSensitivity());
			std::fill(jacDir.begin(), jacDir.end(), 0.0);

			cadet::ad::copyToAd(y.data(), adY, nDof);
			cadet::ad::resetAd(adRes, nDof);
			opAD.residual(DummyModel(), 0.0, 0u, adY, nullptr, adRes, false, cadet::WithoutParamSensitivity());
			opAD.extractJacobianFromAD(adRes, 0);

			// Compare Jacobians
			// Check AD Jacobian pattern
			cadet::test::checkJacobianPatternFD(anaResidual, adMultJac, y.data() + nComp, jacDir.data() + nComp, jacCol1.data() + nComp, jacCol2.data() + nComp, nDof - nComp);

			// Check analytic Jacobian pattern
			cadet::test::checkJacobianPatternFD(anaResidual, anaMultJac, y.data() + nComp, jacDir.data() + nComp, jacCol1.data() + nComp, jacCol2.data() + nComp, nDof - nComp);

			// Check analytic vs AD Jacobian
			cadet::test::compareJacobian(anaMultJac, adMultJac, jacDir.data(), jacCol1.data(), jacCol2.data(), nDof - nComp);
		}

		delete[] adRes;
		delete[] adY;
	}
}


struct AxialFlow
{
	typedef cadet::model::parts::convdisp::AxialFlowParameters<double> Params;

	static void sparsityPattern(cadet::linalg::SparsityPatternRowIterator itBegin, unsigned int nComp, unsigned int nCol, int strideCell, double u, cadet::Weno& weno)
	{
		cadet::model::parts::convdisp::sparsityPatternAxial(itBegin, nComp, nCol, strideCell, u, weno);
	}

	static void residual(double const* y, double const* yDot, double* res, const Params& fp)
	{
		cadet::model::parts::convdisp::residualKernelAxial<double, double, double, cadet::linalg::BandedSparseRowIterator, false>(cadet::SimulationTime{0.0, 0u}, y, yDot, res, cadet::linalg::BandedSparseRowIterator(), fp);
	}

	template <typename IteratorType>
	static void residualWithJacobian(double const* y, double const* yDot, double* res, IteratorType jacBegin, const Params& fp)
	{
		cadet::model::parts::convdisp::residualKernelAxial<double, double, double, IteratorType, true>(cadet::SimulationTime{0.0, 0u}, y, yDot, res, jacBegin, fp);
	}

	std::unique_ptr<cadet::model::IParameterParameterDependence> parDep;

	AxialFlow()
	{
		cadet::ParameterDependenceFactory paramDepFactory;
		parDep.reset(paramDepFactory.createParameterDependence("CONSTANT_ONE"));
	}

	Params makeParams(double u, cadet::active const* d_ax, double h, double* wenoDerivatives, cadet::Weno* weno, cadet::ArrayPool* stencilMemory, int strideCell, int nComp, int nCol)
	{
		return Params {
			u,
			d_ax,
			h,
			wenoDerivatives,
			weno,
			stencilMemory,
			1e-12,
			strideCell,
			static_cast<unsigned int>(nComp),
			static_cast<unsigned int>(nCol),
			0u,
			static_cast<unsigned int>(nComp),
			parDep.get(),
			DummyModel()
		};
	}
};

struct RadialFlow
{
	typedef cadet::model::parts::convdisp::RadialFlowParameters<double> Params;

	static void sparsityPattern(cadet::linalg::SparsityPatternRowIterator itBegin, unsigned int nComp, unsigned int nCol, int strideCell, double u, cadet::Weno& weno)
	{
		cadet::model::parts::convdisp::sparsityPatternRadial(itBegin, nComp, nCol, strideCell, u, weno);
	}

	static void residual(double const* y, double const* yDot, double* res, const Params& fp)
	{
		cadet::model::parts::convdisp::residualKernelRadial<double, double, double, cadet::linalg::BandedSparseRowIterator, false>(cadet::SimulationTime{0.0, 0u}, y, yDot, res, cadet::linalg::BandedSparseRowIterator(), fp);
	}

	template <typename IteratorType>
	static void residualWithJacobian(double const* y, double const* yDot, double* res, IteratorType jacBegin, const Params& fp)
	{
		cadet::model::parts::convdisp::residualKernelRadial<double, double, double, IteratorType, true>(cadet::SimulationTime{0.0, 0u}, y, yDot, res, jacBegin, fp);
	}

	std::unique_ptr<cadet::model::IParameterParameterDependence> parDep;
	std::vector<cadet::active> centers;
	std::vector<cadet::active> sizes;
	std::vector<cadet::active> bounds;

	RadialFlow()
	{
		cadet::ParameterDependenceFactory paramDepFactory;
		parDep.reset(paramDepFactory.createParameterDependence("CONSTANT_ONE"));
	}

	Params makeParams(double u, cadet::active const* d_rad, double h, double* wenoDerivatives, cadet::Weno* weno, cadet::ArrayPool* stencilMemory, int strideCell, int nComp, int nCol)
	{
		centers.resize(10);
		sizes.resize(10);
		bounds.resize(11);

		for (int i = 0; i < 10; ++i)
		{
			centers[i] = (i + 1.5) * h;
			sizes[i] = h;
			bounds[i] = (i + 1) * h;
		}
		bounds.back() = 11 * h;

		return Params {
			u,
			d_rad,
			centers.data(),
			sizes.data(),
			bounds.data(),
			stencilMemory,
			strideCell,
			static_cast<unsigned int>(nComp),
			static_cast<unsigned int>(nCol),
			0u,
			static_cast<unsigned int>(nComp),
			parDep.get(),
			DummyModel()
		};
	}
};


template <typename FlowType>
void testBulkJacobianSparsityWeno(int wenoOrder, bool forwardFlow)
{
	SECTION("WENO=" + std::to_string(wenoOrder))
	{
		int nComp = 3;
		int nCol = 20;

		const double u = (forwardFlow ? 1e-3 : -1e-3);
		const std::vector<cadet::active> d_c(nComp, 1e-6);
		const double h = 1e-3 / nCol;
		const int strideCell = nComp;

		cadet::ArrayPool stencilMemory(sizeof(double) * cadet::Weno::maxStencilSize());
		std::vector<double> wenoDerivatives(cadet::Weno::maxStencilSize(), 0.0);

		cadet::Weno weno;
		weno.order(wenoOrder);
		weno.boundaryTreatment(cadet::Weno::BoundaryTreatment::ReduceOrder);

		cadet::ParameterDependenceFactory paramDepFactory;
		std::unique_ptr<cadet::model::IParameterParameterDependence> parDep(paramDepFactory.createParameterDependence("CONSTANT_ONE"));

		FlowType ft;
		typename FlowType::Params fp = ft.makeParams(
			u,
			d_c.data(),
			h,
			wenoDerivatives.data(),
			&weno,
			&stencilMemory,
			strideCell,
			nComp,
			nCol
		);

		// Obtain sparsity pattern
		cadet::linalg::SparsityPattern pattern(nComp * nCol, std::max(weno.lowerBandwidth() + 1u, 1u) + 1u + std::max(weno.upperBandwidth(), 1u));
		FlowType::sparsityPattern(pattern.row(0), nComp, nCol, strideCell, u, weno);

		// Obtain memory for state, Jacobian columns
		const int nDof = nComp + nComp * nCol;
		std::vector<double> y(nDof, 0.0);
		std::vector<double> jacCol1(nDof, 0.0);
		std::vector<double> jacCol2(nDof, 0.0);

		// Fill state vector with some values
		cadet::test::util::populate(y.data() + nComp, [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof - nComp);

		for (int col = 0; col < pattern.rows(); ++col)
		{
			const double ref = y[nComp + col];

			// Central finite differences
			y[nComp + col] = ref * (1.0 + 1e-6);
			FlowType::residual(y.data(), nullptr, jacCol1.data(), fp);

			y[nComp + col] = ref * (1.0 - 1e-6);
			FlowType::residual(y.data(), nullptr, jacCol2.data(), fp);

			y[nComp + col] = ref;

			for (int row = 0; row < pattern.rows(); ++row)
			{
				const double fd = (jacCol1[row + nComp] - jacCol2[row + nComp]) / (2.0 * ref * 1e-6);
				const bool isFDnonZero = (fd != 0.0);

				CAPTURE(row);
				CAPTURE(col);
				CAPTURE(isFDnonZero);
				CHECK(isFDnonZero == pattern.isNonZero(row, col));
			}
		}
	}
}

template <typename FlowType>
void testBulkJacobianSparseBandedWeno(int wenoOrder, bool forwardFlow)
{
	SECTION("WENO=" + std::to_string(wenoOrder))
	{
		int nComp = 3;
		int nCol = 20;

		const double u = (forwardFlow ? 1e-3 : -1e-3);
		const std::vector<cadet::active> d_c(nComp, 1e-6);
		const double h = 1e-3 / nCol;
		const int strideCell = nComp;

		cadet::ArrayPool stencilMemory(sizeof(double) * cadet::Weno::maxStencilSize());
		std::vector<double> wenoDerivatives(cadet::Weno::maxStencilSize(), 0.0);

		cadet::Weno weno;
		weno.order(wenoOrder);
		weno.boundaryTreatment(cadet::Weno::BoundaryTreatment::ReduceOrder);

		cadet::ParameterDependenceFactory paramDepFactory;
		std::unique_ptr<cadet::model::IParameterParameterDependence> parDep(paramDepFactory.createParameterDependence("CONSTANT_ONE"));

		FlowType ft;
		typename FlowType::Params fp = ft.makeParams(
			u,
			d_c.data(),
			h,
			wenoDerivatives.data(),
			&weno,
			&stencilMemory,
			strideCell,
			nComp,
			nCol
		);

		// Obtain sparsity pattern
		const unsigned int lowerBandwidth = std::max(weno.lowerBandwidth() + 1u, 1u);
		const unsigned int upperBandwidth = std::max(weno.upperBandwidth(), 1u);
		cadet::linalg::SparsityPattern pattern(nComp * nCol, lowerBandwidth + 1u + upperBandwidth);
		FlowType::sparsityPattern(pattern.row(0), nComp, nCol, strideCell, u, weno);

		// Obtain memory for state
		const int nDof = nComp + nComp * nCol;
		std::vector<double> y(nDof, 0.0);
		std::vector<double> res(nDof, 0.0);

		// Fill state vector with some values
		cadet::test::util::populate(y.data() + nComp, [](unsigned int idx) { return std::abs(std::sin(idx * 0.13)) + 1e-4; }, nDof - nComp);

		// Populate sparse matrix
		cadet::linalg::CompressedSparseMatrix sparseMat(pattern);
		FlowType::template residualWithJacobian<cadet::linalg::BandedSparseRowIterator>(y.data(), nullptr, res.data(), sparseMat.row(0), fp);

		// Populate dense matrix
		cadet::linalg::BandMatrix bandMat;
		if (forwardFlow)
			bandMat.resize(nComp * nCol, lowerBandwidth * strideCell, upperBandwidth * strideCell);
		else
			bandMat.resize(nComp * nCol, upperBandwidth * strideCell, lowerBandwidth * strideCell);

		FlowType::template residualWithJacobian<cadet::linalg::BandMatrix::RowIterator>(y.data(), nullptr, res.data(), bandMat.row(0), fp);

		for (int col = 0; col < bandMat.rows(); ++col)
		{
			for (int row = 0; row < bandMat.rows(); ++row)
			{
				CAPTURE(row);
				CAPTURE(col);

				const int diagonal = col - row;
				if ((diagonal <= static_cast<int>(bandMat.upperBandwidth())) && (-diagonal <= static_cast<int>(bandMat.lowerBandwidth())))
					CHECK(bandMat(row, diagonal) == sparseMat(row, col));
				else
					CHECK(0.0 == sparseMat(row, col));
			}
		}
	}
}

TEST_CASE("AxialConvectionDispersionOperator residual forward vs backward flow", "[Operator],[AxialFlow],[Residual]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		testResidualBulkWenoForwardBackward<cadet::model::parts::AxialConvectionDispersionOperator>(i);
}

TEST_CASE("AxialConvectionDispersionOperator time derivative Jacobian vs FD", "[Operator],[AxialFlow],[Residual],[Jacobian]")
{
	testTimeDerivativeBulkJacobianFD<cadet::model::parts::AxialConvectionDispersionOperator>(1e-6, 0.0, 1e-5);
}

TEST_CASE("AxialConvectionDispersionOperator Jacobian forward vs backward flow", "[Operator],[AxialFlow],[Residual],[Jacobian],[AD]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
		testBulkJacobianWenoForwardBackward<cadet::model::parts::AxialConvectionDispersionOperator>(i);
}

TEST_CASE("AxialConvectionDispersionKernel Jacobian sparsity pattern vs FD", "[Operator],[AxialFlow],[Residual],[Jacobian],[SparseMatrix]")
{
	SECTION("Forward flow")
	{
		// Test all WENO orders
		for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
			testBulkJacobianSparsityWeno<AxialFlow>(i, true);
	}
	SECTION("Backward flow")
	{
		// Test all WENO orders
		for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
			testBulkJacobianSparsityWeno<AxialFlow>(i, false);
	}
}

TEST_CASE("AxialConvectionDispersionKernel Jacobian sparse vs banded", "[Operator],[AxialFlow],[Jacobian],[SparseMatrix]")
{
	SECTION("Forward flow")
	{
		// Test all WENO orders
		for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
			testBulkJacobianSparseBandedWeno<AxialFlow>(i, true);
	}
	SECTION("Backward flow")
	{
		// Test all WENO orders
		for (unsigned int i = 1; i <= cadet::Weno::maxOrder(); ++i)
			testBulkJacobianSparseBandedWeno<AxialFlow>(i, false);
	}
}


TEST_CASE("RadialConvectionDispersionOperator residual forward vs backward flow", "[Operator],[RadialFlow],[Residual]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= 1; ++i)
		testResidualBulkWenoForwardBackward<cadet::model::parts::RadialConvectionDispersionOperator>(i);
}

TEST_CASE("RadialConvectionDispersionOperator time derivative Jacobian vs FD", "[Operator],[RadialFlow],[Residual],[Jacobian]")
{
	testTimeDerivativeBulkJacobianFD<cadet::model::parts::RadialConvectionDispersionOperator>(1e-6, 0.0, 1e-5);
}

TEST_CASE("RadialConvectionDispersionOperator Jacobian forward vs backward flow", "[Operator],[RadialFlow],[Residual],[Jacobian],[AD]")
{
	// Test all WENO orders
	for (unsigned int i = 1; i <= 1; ++i)
		testBulkJacobianWenoForwardBackward<cadet::model::parts::RadialConvectionDispersionOperator>(i);
}

TEST_CASE("RadialConvectionDispersionKernel Jacobian sparsity pattern vs FD", "[Operator],[RadialFlow],[Residual],[Jacobian],[SparseMatrix]")
{
	SECTION("Forward flow")
	{
		// Test all WENO orders
		for (unsigned int i = 1; i <= 1; ++i)
			testBulkJacobianSparsityWeno<RadialFlow>(i, true);
	}
	SECTION("Backward flow")
	{
		// Test all WENO orders
		for (unsigned int i = 1; i <= 1; ++i)
			testBulkJacobianSparsityWeno<RadialFlow>(i, false);
	}
}

TEST_CASE("RadialConvectionDispersionKernel Jacobian sparse vs banded", "[Operator],[RadialFlow],[Jacobian],[SparseMatrix]")
{
	SECTION("Forward flow")
	{
		// Test all WENO orders
		for (unsigned int i = 1; i <= 1; ++i)
			testBulkJacobianSparseBandedWeno<RadialFlow>(i, true);
	}
	SECTION("Backward flow")
	{
		// Test all WENO orders
		for (unsigned int i = 1; i <= 1; ++i)
			testBulkJacobianSparseBandedWeno<RadialFlow>(i, false);
	}
}
