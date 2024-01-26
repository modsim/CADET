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

#include <vector>
#include <limits>
#include <random>
#include <chrono>
#include <algorithm>

#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"
#include "linalg/Norms.hpp"

#include "MatrixHelper.hpp"

/**
 * @brief Creates a matrix that contains random values from the normal distribution
 * @param [in] numRows Number of rows
 * @param [in] numCols Number of columns
 * @return Random matrix
 */
inline cadet::linalg::DenseMatrix randomMatrix(unsigned int numRows, unsigned int numCols)
{
	// Initialize standard normal RNG
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1);

	cadet::linalg::DenseMatrix dm;
	dm.resize(numRows, numCols);

	for (int row = 0; row < dm.rows(); ++row)
	{
		for (int col = 0; col < dm.columns(); ++col)
			dm.native(row, col) = distribution(generator);
	}
	return dm;
}

/**
 * @brief Creates a random vector whose values are drawn from a normal distribution
 * @param [in] n Size of the vector
 * @return Random vector
 */
inline std::vector<double> randomVector(unsigned int n)
{
	// Initialize standard normal RNG
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1);

	std::vector<double> v(n, 0.0);
	for (unsigned int i = 0; i < n; ++i)
		v[i] = distribution(generator);

	return v;
}

/**
 * @brief Create a BandMatrix of given type and fill entries with their linear array index (1-based)
 * @param [in] Number of rows
 * @param [in] Lower bandwidth (excluding main diagonal)
 * @param [in] Upper bandwidth (excluding main diagonal)
 * @tparam Matrix_t Type of banded matrix to create
 * @return Banded matrix of given shape
 */
template <typename Matrix_t> Matrix_t createBandMatrix(unsigned int rows, unsigned int lower, unsigned int upper)
{
	Matrix_t bm;
	bm.resize(rows, lower, upper);

	double val = 1.0;
	for (int row = 0; row < bm.rows(); ++row)
	{
		const int lower = std::max(-static_cast<int>(bm.lowerBandwidth()), -static_cast<int>(row));
		const int upper = std::min(static_cast<int>(bm.upperBandwidth()), static_cast<int>(bm.rows() - row) - 1);
		for (int col = lower; col <= upper; ++col)
		{
			bm.centered(row, col) = val;
			val += 1.0;
		}
	}
	return bm;
}

/**
 * @details Combines copySubmatrixFromBanded() with checkMatrixAgainstLinearArray().
 * @param [in] bm BandMatrix to extract dense submatrix from
 * @param [in] startRow Index of the first row of the submatrix
 * @param [in] startDiag Diagonal of the first column of the submatrix in the bandmatrix
 * @param [in] numRows Number of rows to extract
 * @param [in] numCols Number of columns to extract
 * @param [in] ref Reference values (matrix in row-major format)
 * @tparam Matrix_t Type of banded matrix
 */
template <typename Matrix_t>
void testSubMatrixMultiply(const Matrix_t& bm, int startRow, int startDiag, int numRows, int numCols,
						   const std::vector<double>& ref)
{
	SECTION("From row " + std::to_string(startRow) + ", diagonal " + std::to_string(startDiag) + " extract " +
			std::to_string(numRows) + "x" + std::to_string(numCols) + " matrix")
	{
		cadet::linalg::DenseMatrix dm;
		dm.resize(numRows, numCols);
		dm.copySubmatrixFromBanded(bm, startRow, startDiag, numRows, numCols);
		cadet::test::checkMatrixAgainstLinearArray(dm.data(), ref);
	}
}

/**
 * @brief Calls testSubMatrixMultiply() with various sizes and shifts
 * @tparam Matrix_t Type of banded matrix
 */
template <typename Matrix_t> void testDenseSubmatrixFromBanded()
{
	const Matrix_t bm = createBandMatrix<Matrix_t>(8, 2, 3);

	testSubMatrixMultiply<Matrix_t>(bm, 0, 0, 4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 0, 16, 17, 18});
	testSubMatrixMultiply<Matrix_t>(bm, 4, 0, 4, 4, {24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 0, 37, 38, 39});
	testSubMatrixMultiply<Matrix_t>(bm, 2, 2, 4, 4, {14, 15, 0, 0, 19, 20, 21, 0, 24, 25, 26, 27, 29, 30, 31, 32});
	testSubMatrixMultiply<Matrix_t>(bm, 2, -1, 4, 4, {11, 12, 13, 14, 16, 17, 18, 19, 0, 22, 23, 24, 0, 0, 28, 29});
	testSubMatrixMultiply<Matrix_t>(bm, 1, 0, 4, 4, {6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 0, 22, 23, 24});
	testSubMatrixMultiply<Matrix_t>(bm, 0, 4, 4, 4, {0, 0, 0, 0, 9, 0, 0, 0, 14, 15, 0, 0, 19, 20, 21, 0});
	testSubMatrixMultiply<Matrix_t>(bm, 2, 1, 1, 2, {13, 14});
	testSubMatrixMultiply<Matrix_t>(bm, 2, -1, 1, 3, {11, 12, 13});
}

TEST_CASE("Dense submatrix from BandMatrix", "[DenseMatrix],[LinAlg]")
{
	testDenseSubmatrixFromBanded<cadet::linalg::BandMatrix>();
}

TEST_CASE("Dense submatrix from FactorizableBandMatrix", "[DenseMatrix],[LinAlg]")
{
	testDenseSubmatrixFromBanded<cadet::linalg::FactorizableBandMatrix>();
}

TEST_CASE("DenseMatrix LU solves", "[DenseMatrix],[LinAlg]")
{
	using cadet::linalg::DenseMatrix;

	// Probability of obtaining a non-invertible random matrix is 0
	const DenseMatrix dm = randomMatrix(8, 8);
	DenseMatrix fdm = dm;

	REQUIRE(fdm.factorize());

	// Prepare some right hand side
	std::vector<double> y = randomVector(dm.rows());

	// Solve
	std::vector<double> x = y;
	REQUIRE(fdm.solve(x.data()));

	// Calculate residual in y
	dm.multiplyVector(x.data(), 1.0, -1.0, y.data());
	REQUIRE(cadet::linalg::linfNorm(y.data(), y.size()) <= 1e-13);
}

TEST_CASE("DenseMatrix QR solves", "[DenseMatrix],[LinAlg]")
{
	using cadet::linalg::DenseMatrix;

	// Probability of obtaining a non-invertible random matrix is 0
	const DenseMatrix dm = randomMatrix(8, 8);
	DenseMatrix fdm = dm;

	std::vector<double> workingMemory(fdm.robustWorkspaceSize(), 0.0);
	REQUIRE(fdm.robustFactorize(workingMemory.data()));

	// Prepare some right hand side
	std::vector<double> y = randomVector(dm.rows());

	// Solve
	std::vector<double> x = y;
	REQUIRE(fdm.robustSolve(x.data(), workingMemory.data()));

	// Calculate residual in y
	dm.multiplyVector(x.data(), 1.0, -1.0, y.data());
	REQUIRE(cadet::linalg::linfNorm(y.data(), y.size()) <= 1e-13);
}

TEST_CASE("DenseMatrix LU vs QR factorization", "[DenseMatrix],[LinAlg]")
{
	using cadet::linalg::DenseMatrix;

	// Probability of obtaining a non-invertible random matrix is 0
	DenseMatrix dm = randomMatrix(4, 4);

	// Copy for QR
	DenseMatrix dm2 = dm;

	// Perform LU factorization
	REQUIRE(dm.factorize());

	// Perform QR factorization
	std::vector<double> workingMemory(dm2.robustWorkspaceSize(), 0.0);
	REQUIRE(dm2.robustFactorize(workingMemory.data()));

	// Compare inverse matrices
	std::vector<double> vecLU(dm.rows(), 0.0);
	std::vector<double> vecQR(dm2.rows(), 0.0);
	for (int i = 0; i < dm.rows(); ++i)
	{
		// Obtain i-th column of inverse matrix
		vecLU[i] = 1.0;
		vecQR[i] = 1.0;
		dm.solve(vecLU.data());
		dm2.robustSolve(vecQR.data(), workingMemory.data());

		for (int j = 0; j < dm.rows(); ++j)
		{
			CHECK(vecLU[j] == RelApprox(vecQR[j]));
			vecLU[j] = 0.0;
			vecQR[j] = 0.0;
		}
	}
}
