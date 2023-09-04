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

#include <vector>
#include <limits>
#include <algorithm>

#include "linalg/BandMatrix.hpp"
#include "linalg/Norms.hpp"

#include "MatrixHelper.hpp"

/**
 * @brief Converts a BandMatrix into a FactorizableBandMatrix
 * @details Just copies the values.
 * @param [in] bm Source matrix
 * @return FactorizableBandMatrix with the same content as @p bm
 */
inline cadet::linalg::FactorizableBandMatrix fromBandMatrix(const cadet::linalg::BandMatrix& bm)
{
	cadet::linalg::FactorizableBandMatrix fbm;
	fbm.resize(bm.rows(), bm.lowerBandwidth(), bm.upperBandwidth());
	fbm.copyOver(bm);
	return fbm;
}

/**
 * @brief Extracts a dense submatrix of a BandMatrix by calling submatrixMultiplyVector()
 * @details Extracts the dense submatrix column by column.
 * @param [in] mat Source matrix
 * @param [in] startRow Index of the first row of the submatrix
 * @param [in] startDiag Diagonal of the first column of the submatrix in the bandmatrix
 * @param [in] numRows Number of rows to extract
 * @param [in] numCols Number of columns to extract
 * @param [out] out Memory in which the dense submatrix is written in row-major format
 */
inline void extractDenseSubMatrix(const cadet::linalg::BandMatrix& mat, unsigned int startRow, int startDiag,
								  unsigned int numRows, unsigned int numCols, double* const out)
{
	std::vector<double> x(numCols, 0.0);
	std::vector<double> y(numRows, 0.0);

	// Iterate over columns
	for (unsigned int i = 0; i < numCols; ++i)
	{
		// Multiply with basis vectors
		x[i] = 1.0;
		mat.submatrixMultiplyVector(x.data(), startRow, startDiag, numRows, numCols, y.data());
		x[i] = 0.0;

		// Save column in dense matrix in row-major format
		for (unsigned int j = 0; j < numRows; ++j)
			out[i + j * numCols] = y[j];
	}
}

TEST_CASE("Copying BandMatrix to FactorizableBandMatrix", "[BandMatrix],[LinAlg]")
{
	using cadet::linalg::BandMatrix;
	using cadet::linalg::FactorizableBandMatrix;

	const BandMatrix bm = cadet::test::createBandMatrix<BandMatrix>(10, 2, 3);
	const FactorizableBandMatrix fbm = fromBandMatrix(bm);

	for (int row = 0; row < bm.rows(); ++row)
	{
		const int lower = std::max(-static_cast<int>(bm.lowerBandwidth()), -static_cast<int>(row));
		const int upper = std::min(static_cast<int>(bm.upperBandwidth()), static_cast<int>(bm.rows() - row) - 1);
		for (int col = lower; col <= upper; ++col)
		{
			CHECK(fbm.centered(row, col) == bm.centered(row, col));
		}
	}
}

TEST_CASE("FactorizableBandMatrix iterator read access", "[BandMatrix],[LinAlg]")
{
	using cadet::linalg::FactorizableBandMatrix;

	const FactorizableBandMatrix fbm = cadet::test::createBandMatrix<FactorizableBandMatrix>(10, 2, 3);
	FactorizableBandMatrix::ConstRowIterator it = fbm.row(0);
	for (int row = 0; row < fbm.rows(); ++row, ++it)
	{
		const int lower = std::max(-static_cast<int>(fbm.lowerBandwidth()), -static_cast<int>(row));
		const int upper = std::min(static_cast<int>(fbm.upperBandwidth()), static_cast<int>(fbm.rows() - row) - 1);
		for (int col = lower; col <= upper; ++col)
		{
			CHECK(fbm.centered(row, col) == it[col]);
		}
	}
}

TEST_CASE("FactorizableBandMatrix solves", "[BandMatrix],[LinAlg]")
{
	using cadet::linalg::BandMatrix;
	using cadet::linalg::FactorizableBandMatrix;

	const BandMatrix bm = cadet::test::createBandMatrix<BandMatrix>(10, 2, 3);
	FactorizableBandMatrix fbm = fromBandMatrix(bm);

	REQUIRE(fbm.factorize());

	// Prepare some right hand side
	std::vector<double> y(fbm.rows(), 0.0);
	for (int i = 0; i < fbm.rows(); ++i)
		y[i] = std::sin(6.283185307 * i / static_cast<double>(fbm.rows()));

	// Solve
	std::vector<double> x = y;
	REQUIRE(fbm.solve(x.data()));

	// Calculate residual in y
	bm.multiplyVector(x.data(), 1.0, -1.0, y.data());
	REQUIRE(cadet::linalg::linfNorm(y.data(), y.size()) <= 1e-10);
}

/**
 * @brief Tests the extraction of a dense submatrix via submatrixMultiplyVector()
 * @details Combines extractDenseSubMatrix() with checkMatrixAgainstLinearArray().
 * @param [in] bm BandMatrix to extract dense submatrix from
 * @param [in] startRow Index of the first row of the submatrix
 * @param [in] startDiag Diagonal of the first column of the submatrix in the bandmatrix
 * @param [in] numRows Number of rows to extract
 * @param [in] numCols Number of columns to extract
 * @param [in] ref Reference values (matrix in row-major format)
 */
void testSubMatrixMultiply(const cadet::linalg::BandMatrix& bm, int startRow, int startDiag, int numRows, int numCols,
						   const std::vector<double>& ref)
{
	SECTION("From row " + std::to_string(startRow) + ", diagonal " + std::to_string(startDiag) + " extract " +
			std::to_string(numRows) + "x" + std::to_string(numCols) + " matrix")
	{
		std::vector<double> out(bm.rows() * bm.stride(), 0.0);
		extractDenseSubMatrix(bm, startRow, startDiag, numRows, numCols, out.data());
		cadet::test::checkMatrixAgainstLinearArray(out.data(), ref);
	}
}

TEST_CASE("BandMatrix::submatrixMultiplyVector", "[BandMatrix],[LinAlg]")
{
	using cadet::linalg::BandMatrix;

	SECTION("Matrix size: 8 rows, 2+1+3 bandwidth")
	{
		const BandMatrix bm = cadet::test::createBandMatrix<BandMatrix>(8, 2, 3);

		testSubMatrixMultiply(bm, 0, 0, 5, 5,
							  {1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 16, 17, 18, 19, 0, 0, 22, 23, 24});
		testSubMatrixMultiply(bm, 2, 0, 5, 5, {12, 13, 14, 15, 0,  17, 18, 19, 20, 21, 22, 23, 24,
											   25, 26, 0,  28, 29, 30, 31, 0,  0,  33, 34, 35});
		testSubMatrixMultiply(bm, 2, -2, 5, 5, {10, 11, 12, 13, 14, 0,  16, 17, 18, 19, 0, 0, 22,
												23, 24, 0,  0,  0,  28, 29, 0,  0,  0,  0, 33});
		testSubMatrixMultiply(bm, 2, -1, 5, 5, {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0,  22, 23,
												24, 25, 0,  0,  28, 29, 30, 0,  0,  0,  33, 34});
		testSubMatrixMultiply(bm, 2, 1, 5, 5, {13, 14, 15, 0,  0,  18, 19, 20, 21, 0,  23, 24, 25,
											   26, 27, 28, 29, 30, 31, 32, 0,  33, 34, 35, 36});
		testSubMatrixMultiply(bm, 2, 3, 5, 5,
							  {15, 0, 0, 0, 0, 20, 21, 0, 0, 0, 25, 26, 27, 0, 0, 30, 31, 32, 0, 0, 34, 35, 36, 0, 0});
		testSubMatrixMultiply(bm, 2, 0, 5, 7, {12, 13, 14, 15, 0,  0,  0,  17, 18, 19, 20, 21, 0,  0,  22, 23, 24, 25,
											   26, 27, 0,  0,  28, 29, 30, 31, 32, 0,  0,  0,  33, 34, 35, 36, 0});
		testSubMatrixMultiply(bm, 2, -1, 5, 3, {11, 12, 13, 16, 17, 18, 0, 22, 23, 0, 0, 28, 0, 0, 0});
		testSubMatrixMultiply(bm, 2, 1, 5, 3, {13, 14, 15, 18, 19, 20, 23, 24, 25, 28, 29, 30, 0, 33, 34});
		testSubMatrixMultiply(bm, 2, 2, 5, 3, {14, 15, 0, 19, 20, 21, 24, 25, 26, 29, 30, 31, 33, 34, 35});
		testSubMatrixMultiply(bm, 2, 3, 5, 3, {15, 0, 0, 20, 21, 0, 25, 26, 27, 30, 31, 32, 34, 35, 36});
		testSubMatrixMultiply(bm, 2, -2, 3, 3, {10, 11, 12, 0, 16, 17, 0, 0, 22});
		testSubMatrixMultiply(bm, 2, 1, 3, 3, {13, 14, 15, 18, 19, 20, 23, 24, 25});
		testSubMatrixMultiply(bm, 2, 1, 3, 5, {13, 14, 15, 0, 0, 18, 19, 20, 21, 0, 23, 24, 25, 26, 27});
		testSubMatrixMultiply(bm, 3, 0, 1, 2, {18, 19});
		testSubMatrixMultiply(bm, 3, 1, 1, 2, {19, 20});
	}

	SECTION("Matrix size: 24 rows, 6+1+9 bandwidth")
	{
		const BandMatrix bm = cadet::test::createBandMatrix<BandMatrix>(24, 6, 9);

		testSubMatrixMultiply(bm, 3, 1, 1, 2, {38, 39});
		testSubMatrixMultiply(bm, 3, -1, 1, 3, {36, 37, 38});
	}
}
