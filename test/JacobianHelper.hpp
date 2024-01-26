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

/**
 * @file
 * Defines helper functions for checking Jacobians.
 */

#ifndef CADETTEST_JACOBIANHELPER_HPP_
#define CADETTEST_JACOBIANHELPER_HPP_

#include "model/UnitOperation.hpp"
#include "SimulationTypes.hpp"

#include <catch.hpp>
#include <limits>
#include <functional>

#include "Approx.hpp"

namespace cadet
{

namespace util
{
class ThreadLocalStorage;
}

namespace test
{

/**
 * @brief Compares two given Jacobians column by column
 * @details A column is extracted by calling multiplyWithJacobian() on the model.
 * @param [in] multJacA Multiplies model A's Jacobian with a given vector
 * @param [in] multJacB Multiplies model B's Jacobian with a given vector
 * @param [in] y State vector
 * @param [in] yDot Time derivative of state vector
 * @param [in] dir Memory for extracting a column
 * @param [in] colA Memory for Jacobian column
 * @param [in] colB Memory for Jacobian column
 * @param [in] n Number of DOFs in the model
 * @param [in] m Number of equations in the model
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
inline void compareJacobian(const std::function<void(double const*, double*)> multJacA,
							const std::function<void(double const*, double*)> multJacB, double* dir, double* colA,
							double* colB, unsigned int n, unsigned int m, double absTol, double relTol)
{
	std::fill(dir, dir + n, 0.0);
	for (unsigned int col = 0; col < n; ++col)
	{
		dir[col] = 1.0;

		multJacA(dir, colA);
		multJacB(dir, colB);

		for (unsigned int row = 0; row < m; ++row)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(colA[row] == RelApprox(colB[row]).epsilon(relTol).margin(absTol));
		}

		dir[col] = 0.0;
	}
}

/**
 * @brief Compares two given Jacobians column by column
 * @details A column is extracted by calling multiplyWithJacobian() on the model.
 * @param [in] multJacA Multiplies model A's Jacobian with a given vector
 * @param [in] multJacB Multiplies model B's Jacobian with a given vector
 * @param [in] y State vector
 * @param [in] yDot Time derivative of state vector
 * @param [in] dir Memory for extracting a column
 * @param [in] colA Memory for Jacobian column
 * @param [in] colB Memory for Jacobian column
 * @param [in] n Number of DOFs in the model
 * @param [in] m Number of equations in the model
 */
inline void compareJacobian(const std::function<void(double const*, double*)> multJacA,
							const std::function<void(double const*, double*)> multJacB, double* dir, double* colA,
							double* colB, unsigned int n, unsigned int m)
{
	compareJacobian(multJacA, multJacB, dir, colA, colB, n, m, RelApprox::defaultEpsilon(), RelApprox::defaultMargin());
}

/**
 * @brief Compares two given Jacobians column by column
 * @details A column is extracted by calling multiplyWithJacobian() on the model.
 * @param [in] multJacA Multiplies model A's Jacobian with a given vector
 * @param [in] multJacB Multiplies model B's Jacobian with a given vector
 * @param [in] y State vector
 * @param [in] yDot Time derivative of state vector
 * @param [in] dir Memory for extracting a column
 * @param [in] colA Memory for Jacobian column
 * @param [in] colB Memory for Jacobian column
 * @param [in] n Number of DOFs in the model
 */
inline void compareJacobian(const std::function<void(double const*, double*)> multJacA,
							const std::function<void(double const*, double*)> multJacB, double* dir, double* colA,
							double* colB, unsigned int n)
{
	compareJacobian(multJacA, multJacB, dir, colA, colB, n, n);
}

/**
 * @brief Compares two given Jacobians column by column
 * @details A column is extracted by calling multiplyWithJacobian() on the model.
 * @param [in] modelA Model A
 * @param [in] modelB Model B
 * @param [in] y State vector
 * @param [in] yDot Time derivative of state vector
 * @param [in] dir Memory for extracting a column
 * @param [in] colA Memory for Jacobian column of @p modelA
 * @param [in] colB Memory for Jacobian column of @p modelB
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
inline void compareJacobian(cadet::IUnitOperation* modelA, cadet::IUnitOperation* modelB, double const* y,
							double const* yDot, double* dir, double* colA, double* colB, double absTol, double relTol)
{
	compareJacobian(
		[=](double const* lDir, double* res) -> void {
			modelA->multiplyWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y, yDot}, lDir, 1.0, 0.0, res);
		},
		[=](double const* lDir, double* res) -> void {
			modelB->multiplyWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y, yDot}, lDir, 1.0, 0.0, res);
		},
		dir, colA, colB, modelA->numDofs(), modelA->numDofs(), absTol, relTol);
}

/**
 * @brief Compares two given Jacobians column by column
 * @details A column is extracted by calling multiplyWithJacobian() on the model.
 * @param [in] modelA Model A
 * @param [in] modelB Model B
 * @param [in] y State vector
 * @param [in] yDot Time derivative of state vector
 * @param [in] dir Memory for extracting a column
 * @param [in] colA Memory for Jacobian column of @p modelA
 * @param [in] colB Memory for Jacobian column of @p modelB
 */
inline void compareJacobian(cadet::IUnitOperation* modelA, cadet::IUnitOperation* modelB, double const* y,
							double const* yDot, double* dir, double* colA, double* colB)
{
	compareJacobian(modelA, modelB, y, yDot, dir, colA, colB, RelApprox::defaultEpsilon(), RelApprox::defaultMargin());
}

/**
 * @brief Checks the bottom macro row and right macro column of the Jacobian against finite differences
 * @details Uses finite differences to determine the Jacobian. The two Jacobians
 *          are compared column by column. A column is extracted by calling multiplyJacobian().
 *          Only the bottom macro row and the right macro column of the arrow head are compared.
 * @param [in] residual Function that returns a residual vector
 * @param [in] multiplyJacobian Function that multiplies the (time derivative) Jacobian with a vector
 * @param [in] y State vector
 * @param [in] dir Memory for computing finite differences
 * @param [in] colA Memory for Jacobian column
 * @param [in] colB Memory for Jacobian column
 * @param [in] n Number of DOFs in the model
 * @param [in] offset Offset to first DOF of right macro column
 * @param [in] h Step size for centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
inline void compareJacobianArrowHeadFD(const std::function<void(double const*, double*)>& residual,
									   const std::function<void(double const*, double*)>& multiplyJacobian,
									   double const* y, double* dir, double* colA, double* colB, unsigned int n,
									   unsigned int offset, double h = 1e-6, double absTol = 0.0,
									   double relTol = std::numeric_limits<float>::epsilon() * 100.0)
{
	// Compare bottom macro row until we hit right macro column
	for (unsigned int col = 0; col < offset; ++col)
	{
		std::copy(y, y + n, dir);

		if (y[col] != 0.0)
			dir[col] = y[col] * (1.0 + h);
		else
			dir[col] = h;

		residual(dir, colA);

		if (y[col] != 0.0)
			dir[col] = y[col] * (1.0 - h);
		else
			dir[col] = -h;

		residual(dir, colB);

		for (unsigned int j = offset; j < n; ++j)
		{
			if (y[col] != 0.0)
				colA[j] = (colA[j] - colB[j]) / (y[col] * 2.0 * h);
			else
				colA[j] = (colA[j] - colB[j]) / (2.0 * h);
		}

		std::fill(dir, dir + n, 0.0);
		dir[col] = 1.0;
		multiplyJacobian(dir, colB);

		for (unsigned int row = offset; row < n; ++row)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(colA[row] == RelApprox(colB[row]).epsilon(relTol).margin(absTol));
		}
	}

	// Fully compare right macro column
	for (unsigned int col = offset; col < n; ++col)
	{
		std::copy(y, y + n, dir);

		if (y[col] != 0.0)
			dir[col] = y[col] * (1.0 + h);
		else
			dir[col] = h;

		residual(dir, colA);

		if (y[col] != 0.0)
			dir[col] = y[col] * (1.0 - h);
		else
			dir[col] = -h;

		residual(dir, colB);

		for (unsigned int j = 0; j < n; ++j)
		{
			if (y[col] != 0.0)
				colA[j] = (colA[j] - colB[j]) / (y[col] * 2.0 * h);
			else
				colA[j] = (colA[j] - colB[j]) / (2.0 * h);
		}

		std::fill(dir, dir + n, 0.0);
		dir[col] = 1.0;
		multiplyJacobian(dir, colB);

		for (unsigned int row = 0; row < n; ++row)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(colA[row] == RelApprox(colB[row]).epsilon(relTol).margin(absTol));
		}
	}
}

/**
 * @brief Checks a (time derivative) Jacobian against finite differences
 * @details Uses finite differences to determine the Jacobian. The two Jacobians
 *          are compared column by column. A column is extracted by calling multiplyJacobian().
 * @param [in] residual Function that returns a residual vector
 * @param [in] multiplyJacobian Function that multiplies the (time derivative) Jacobian with a vector
 * @param [in] y State vector
 * @param [in] dir Memory for computing finite differences
 * @param [in] colA Memory for Jacobian column
 * @param [in] colB Memory for Jacobian column
 * @param [in] n Number of DOFs in the model
 * @param [in] m Number of equations in the model
 * @param [in] h Step size for centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
inline void compareJacobianFD(const std::function<void(double const*, double*)>& residual,
							  const std::function<void(double const*, double*)>& multiplyJacobian, double const* y,
							  double* dir, double* colA, double* colB, unsigned int n, unsigned int m, double h = 1e-6,
							  double absTol = 0.0, double relTol = std::numeric_limits<float>::epsilon() * 100.0)
{
	for (unsigned int col = 0; col < n; ++col)
	{
		std::copy(y, y + n, dir);

		if (y[col] != 0.0)
			dir[col] = y[col] * (1.0 + h);
		else
			dir[col] = h;

		residual(dir, colA);

		if (y[col] != 0.0)
			dir[col] = y[col] * (1.0 - h);
		else
			dir[col] = -h;

		residual(dir, colB);

		for (unsigned int j = 0; j < m; ++j)
		{
			if (y[col] != 0.0)
				colA[j] = (colA[j] - colB[j]) / (y[col] * 2.0 * h);
			else
				colA[j] = (colA[j] - colB[j]) / (2.0 * h);
		}

		std::fill(dir, dir + n, 0.0);
		dir[col] = 1.0;
		multiplyJacobian(dir, colB);

		for (unsigned int row = 0; row < m; ++row)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(colA[row] == RelApprox(colB[row]).epsilon(relTol).margin(absTol));
		}
	}
}

/**
 * @brief Checks a (time derivative) Jacobian against finite differences
 * @details Uses finite differences to determine the Jacobian. The two Jacobians
 *          are compared column by column. A column is extracted by calling multiplyJacobian().
 * @param [in] residual Function that returns a residual vector
 * @param [in] multiplyJacobian Function that multiplies the (time derivative) Jacobian with a vector
 * @param [in] y State vector
 * @param [in] dir Memory for computing finite differences
 * @param [in] colA Memory for Jacobian column
 * @param [in] colB Memory for Jacobian column
 * @param [in] n Number of DOFs in the model
 * @param [in] h Step size for centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
inline void compareJacobianFD(const std::function<void(double const*, double*)>& residual,
							  const std::function<void(double const*, double*)>& multiplyJacobian, double const* y,
							  double* dir, double* colA, double* colB, unsigned int n, double h = 1e-6,
							  double absTol = 0.0, double relTol = std::numeric_limits<float>::epsilon() * 100.0)
{
	compareJacobianFD(residual, multiplyJacobian, y, dir, colA, colB, n, n, h, absTol, relTol);
}

/**
 * @brief Checks a state Jacobian against finite differences
 * @details Uses finite differences on @p modelA to determine the state
 *          Jacobian. The two Jacobians are compared column by column. A
 *          column is extracted by calling multiplyWithJacobian() on the
 *          model.
 * @param [in] modelA Model A
 * @param [in] modelB Model B
 * @param [in] y State vector
 * @param [in] yDot Time derivative of state vector
 * @param [in] dir Memory for computing finite differences
 * @param [in] colA Memory for Jacobian column of @p modelA
 * @param [in] colB Memory for Jacobian column of @p modelB
 * @param [in] tls Thread local storage for @p modelA
 * @param [in] h Step size for centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
inline void compareJacobianFD(cadet::IUnitOperation* modelA, cadet::IUnitOperation* modelB, double const* y,
							  double const* yDot, double* dir, double* colA, double* colB,
							  cadet::util::ThreadLocalStorage& tls, double h = 1e-6, double absTol = 0.0,
							  double relTol = std::numeric_limits<float>::epsilon() * 100.0)
{
	compareJacobianFD(
		[=, &tls](double const* lDir, double* res) -> void {
			modelA->residual(SimulationTime{0.0, 0u}, ConstSimulationState{lDir, yDot}, res, tls);
		},
		[=](double const* lDir, double* res) -> void {
			modelB->multiplyWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y, yDot}, lDir, 1.0, 0.0, res);
		},
		y, dir, colA, colB, modelA->numDofs(), modelA->numDofs(), h, absTol, relTol);
}

/**
 * @brief Checks the pattern of the two given Jacobians including sign of the entries
 * @details Uses finite differences to determine the Jacobian structure. The Jacobians
 *          are compared column by column. A column is extracted by calling multiplyWithJacobian().
 * @param [in] residual Function that returns a residual vector
 * @param [in] multiplyJacobian Function that multiplies the Jacobian with a vector
 * @param [in] y State vector
 * @param [in] dir Memory for computing finite differences
 * @param [in] colA Memory for Jacobian column
 * @param [in] colB Memory for Jacobian column
 * @param [in] n Number of DOFs in the model
 * @param [in] m Number of equations in the model
 */
inline void checkJacobianPatternFD(const std::function<void(double const*, double*)>& residual,
								   const std::function<void(double const*, double*)>& multiplyJacobian, double const* y,
								   double* dir, double* colA, double* colB, unsigned int n, unsigned int m)
{
	const double h = 1e-5;
	for (unsigned int col = 0; col < n; ++col)
	{
		std::copy(y, y + n, dir);
		if (y[col] != 0.0)
			dir[col] = y[col] * (1.0 + h);
		else
			dir[col] = h;

		residual(dir, colA);

		if (y[col] != 0.0)
			dir[col] = y[col] * (1.0 - h);
		else
			dir[col] = -h;

		residual(dir, colB);

		for (unsigned int j = 0; j < m; ++j)
		{
			if (y[col] != 0.0)
				colA[j] = (colA[j] - colB[j]) / (y[col] * 2.0 * h);
			else
				colA[j] = (colA[j] - colB[j]) / (2.0 * h);
		}

		std::fill(dir, dir + n, 0.0);
		dir[col] = 1.0;
		multiplyJacobian(dir, colB);

		// Check for pattern including sign
		for (unsigned int row = 0; row < m; ++row)
		{
			CAPTURE(row);
			CAPTURE(col);
			if (colA[row] == 0.0)
				CHECK(colB[row] == 0.0);
			else if (colA[row] > 0.0)
				CHECK(colB[row] > 0.0);
			else if (colA[row] < 0.0)
				CHECK(colB[row] < 0.0);
			else if (std::isnan(colA[row]))
				CHECK(std::isnan(colB[row]));
		}
	}
}

/**
 * @brief Checks the pattern of the two given Jacobians including sign of the entries
 * @details Uses finite differences to determine the Jacobian structure. The Jacobians
 *          are compared column by column. A column is extracted by calling multiplyWithJacobian().
 * @param [in] residual Function that returns a residual vector
 * @param [in] multiplyJacobian Function that multiplies the time derivative Jacobian with a vector
 * @param [in] y State vector
 * @param [in] dir Memory for computing finite differences
 * @param [in] colA Memory for Jacobian column
 * @param [in] colB Memory for Jacobian column
 * @param [in] n Number of DOFs in the model
 */
inline void checkJacobianPatternFD(const std::function<void(double const*, double*)>& residual,
								   const std::function<void(double const*, double*)>& multiplyJacobian, double const* y,
								   double* dir, double* colA, double* colB, unsigned int n)
{
	checkJacobianPatternFD(residual, multiplyJacobian, y, dir, colA, colB, n, n);
}

/**
 * @brief Checks the pattern of the two given Jacobians including sign of the entries
 * @details Uses finite differences on @p modelA to determine the Jacobian structure. The
 *          Jacobians are compared column by column. A column is extracted by calling
 *          multiplyWithJacobian() on the model.
 * @param [in] modelA Model A
 * @param [in] modelB Model B
 * @param [in] y State vector
 * @param [in] yDot Time derivative of state vector
 * @param [in] dir Memory for computing finite differences
 * @param [in] colA Memory for Jacobian column of @p modelA
 * @param [in] colB Memory for Jacobian column of @p modelB
 * @param [in] tls Thread local storage for @p modelA
 */
inline void checkJacobianPatternFD(cadet::IUnitOperation* modelA, cadet::IUnitOperation* modelB, double const* y,
								   double const* yDot, double* dir, double* colA, double* colB,
								   cadet::util::ThreadLocalStorage& tls)
{
	checkJacobianPatternFD(
		[=, &tls](double const* lDir, double* res) -> void {
			modelA->residual(SimulationTime{0.0, 0u}, ConstSimulationState{lDir, yDot}, res, tls);
		},
		[=](double const* lDir, double* res) -> void {
			modelB->multiplyWithJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y, yDot}, lDir, 1.0, 0.0, res);
		},
		y, dir, colA, colB, modelA->numDofs(), modelA->numDofs());
}

/**
 * @brief Compares two given time derivative Jacobians column by column
 * @details A column is extracted by calling multiplyWithDerivativeJacobian() on the model.
 * @param [in] modelA Model A
 * @param [in] modelB Model B
 * @param [in] y State vector
 * @param [in] yDot Time derivative of state vector
 * @param [in] dir Memory for extracting a column
 * @param [in] colA Memory for Jacobian column of @p modelA
 * @param [in] colB Memory for Jacobian column of @p modelB
 */
inline void compareTimeDerivativeJacobian(cadet::IUnitOperation* modelA, cadet::IUnitOperation* modelB, double const* y,
										  double const* yDot, double* dir, double* colA, double* colB)
{
	const unsigned int n = modelA->numDofs();
	std::fill(dir, dir + n, 0.0);
	for (unsigned int col = 0; col < n; ++col)
	{
		dir[col] = 1.0;

		modelA->multiplyWithDerivativeJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y, yDot}, dir, colA);
		modelB->multiplyWithDerivativeJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y, yDot}, dir, colB);

		for (unsigned int row = 0; row < n; ++row)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(colA[row] == RelApprox(colB[row]));
		}

		dir[col] = 0.0;
	}
}

/**
 * @brief Checks a time derivative Jacobian against finite differences
 * @details Uses finite differences on @p modelA to determine the time derivative
 *          Jacobian. The two Jacobians are compared column by column. A column is
 *          extracted by calling multiplyWithDerivativeJacobian() on the model.
 * @param [in] modelA Model A
 * @param [in] modelB Model B
 * @param [in] y State vector
 * @param [in] yDot Time derivative of state vector
 * @param [in] dir Memory for computing finite differences
 * @param [in] colA Memory for Jacobian column of @p modelA
 * @param [in] colB Memory for Jacobian column of @p modelB
 * @param [in] tls Thread local storage for @p modelA
 * @param [in] h Step size for centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
inline void compareTimeDerivativeJacobianFD(cadet::IUnitOperation* modelA, cadet::IUnitOperation* modelB,
											double const* y, double const* yDot, double* dir, double* colA,
											double* colB, cadet::util::ThreadLocalStorage& tls, double h = 1e-6,
											double absTol = 0.0,
											double relTol = std::numeric_limits<float>::epsilon() * 100.0)
{
	compareJacobianFD(
		[=, &tls](double const* lDir, double* res) -> void {
			modelA->residual(SimulationTime{0.0, 0u}, ConstSimulationState{y, lDir}, res, tls);
		},
		[=](double const* lDir, double* res) -> void {
			modelB->multiplyWithDerivativeJacobian(SimulationTime{0.0, 0u}, ConstSimulationState{y, yDot}, lDir, res);
		},
		yDot, dir, colA, colB, modelA->numDofs(), modelA->numDofs(), h, absTol, relTol);
}

} // namespace test
} // namespace cadet

#endif // CADETTEST_JACOBIANHELPER_HPP_
