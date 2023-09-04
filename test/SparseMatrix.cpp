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
#include <algorithm>

#include "linalg/CompressedSparseMatrix.hpp"
#include "linalg/DenseMatrix.hpp"

namespace
{

cadet::linalg::CompressedSparseMatrix createSmallMatrix()
{
	/*
	   Matrix pattern
	 . X . X . . . . . X
	 X . X . . . . . . .
	 . X . X . . . . . .
	 . . X . X . . . . .
	 . . . X . X . . . .
	 . . . . X . X . . .
	 . . . . . X . X . .
	 . . . . . . X . X .
	 . . . . . . . X . X
	 X . . . . . . . X .
	*/

	// Create pattern
	cadet::linalg::SparsityPattern pattern(10, 2);
	cadet::linalg::SparsityPatternRowIterator ri = pattern.row(0);
	for (int i = 0; i < pattern.rows(); ++i)
	{
		if (i > 0)
			ri[-1] = 1.0;
		if (i < pattern.rows() - 1)
			ri[1] = 1.0;
		++ri;
	}
	pattern.add(0, 3);
	pattern.add(0, 9);
	pattern.add(9, 0);

	REQUIRE(pattern.numNonZeros() == 21);

	// Create matrix with pattern
	cadet::linalg::CompressedSparseMatrix mat(pattern);
	REQUIRE(mat.numNonZeros() == pattern.numNonZeros());
	REQUIRE(mat.rows() == pattern.rows());

	// Assign some data
	double* const data = mat.data();
	for (int i = 0; i < pattern.numNonZeros(); ++i)
		data[i] = i + 1.0;

	return mat;
}

inline bool isNonZeroInSmallMatrix(int row, int diag)
{
	if ((diag + row < 0) || (diag + row >= 10))
		return false;

	if ((diag == -1) || (diag == 1))
		return true;
	if ((row == 0) && (diag == 9))
		return true;
	if ((row == 9) && (diag == -9))
		return true;
	if ((row == 0) && (diag == 3))
		return true;

	return false;
}

inline cadet::linalg::DenseMatrix createSmallMatrixDense()
{
	cadet::linalg::DenseMatrix dm;
	dm.resize(10, 10);
	dm.setAll(0.0);

	double curVal = 1.0;
	for (int row = 0; row < 10; ++row)
	{
		for (int diag = -row; diag < 10 - row; ++diag)
		{
			if (isNonZeroInSmallMatrix(row, diag))
			{
				dm.diagonalElement(row, diag) = curVal;
				curVal += 1.0;
			}
		}
	}
	return dm;
}
} // namespace

TEST_CASE("CompressedSparseMatrix assembly and iterator access", "[SparseMatrix],[LinAlg]")
{
	// Create matrix with pattern
	cadet::linalg::CompressedSparseMatrix mat = createSmallMatrix();

	// Check data via row iterator
	cadet::linalg::BandedSparseRowIterator bri = mat.row(0);
	double curVal = 1.0;
	for (int row = 0; row < mat.rows(); ++row)
	{
		for (int diag = -row; diag < static_cast<int>(mat.rows()) - row; ++diag)
		{
			CAPTURE(row);
			CAPTURE(diag);
			if (isNonZeroInSmallMatrix(row, diag))
			{
				CHECK(bri[diag] == curVal);
				++curVal;
			}
			else
			{
				CHECK(bri[diag] == 0.0);
			}
		}
		++bri;
	}
}

TEST_CASE("CompressedSparseMatrix assembly and direct access", "[SparseMatrix],[LinAlg]")
{
	// Create matrix with pattern
	cadet::linalg::CompressedSparseMatrix mat = createSmallMatrix();

	// Check data via direct access
	double curVal = 1.0;
	for (int row = 0; row < mat.rows(); ++row)
	{
		for (int diag = -row; diag < static_cast<int>(mat.rows()) - row; ++diag)
		{
			CAPTURE(row);
			CAPTURE(diag);
			if (isNonZeroInSmallMatrix(row, diag))
			{
				CHECK(mat.centered(row, diag) == curVal);
				++curVal;
			}
			else
			{
				CHECK(mat.centered(row, diag) == 0.0);
			}
		}
	}
}

TEST_CASE("CompressedSparseMatrix matrix-vector multiplication", "[SparseMatrix],[LinAlg]")
{
	// Create matrix with pattern
	cadet::linalg::CompressedSparseMatrix mat = createSmallMatrix();

	// Create equivalent dense matrix
	const cadet::linalg::DenseMatrix dm = createSmallMatrixDense();

	// Check matrix-vector multiplication against dense matrix
	std::vector<double> ys(mat.rows(), 0.0);
	std::vector<double> yd(mat.rows(), 0.0);
	std::vector<double> x(mat.rows(), 0.0);
	for (int col = 0; col < mat.rows(); ++col)
	{
		x[col] = 1.0;
		mat.multiplyVector(x.data(), ys.data());
		dm.multiplyVector(x.data(), yd.data());
		x[col] = 0.0;

		for (int row = 0; row < mat.rows(); ++row)
		{
			CAPTURE(col);
			CAPTURE(row);
			CHECK(ys[row] == cadet::test::makeApprox(yd[row], std::numeric_limits<double>::epsilon() * 100.0, 0.0));
		}
	}
}

TEST_CASE("CompressedSparseMatrix matrix-vector multiplication with factor", "[SparseMatrix],[LinAlg]")
{
	// Create matrix with pattern
	cadet::linalg::CompressedSparseMatrix mat = createSmallMatrix();

	// Create equivalent dense matrix
	const cadet::linalg::DenseMatrix dm = createSmallMatrixDense();

	// Prepare some vector
	std::vector<double> x(mat.rows(), 0.0);
	for (int i = 0; i < mat.rows(); ++i)
		x[i] = std::sin(6.283185307 * i / static_cast<double>(mat.rows()));

	// Check matrix-vector multiplication against dense matrix
	std::vector<double> ys(mat.rows(), 0.0);
	std::vector<double> yd(mat.rows(), 0.0);

	mat.multiplyVector(x.data(), 2.0, 1.0, ys.data());
	dm.multiplyVector(x.data(), 2.0, 1.0, yd.data());

	for (int row = 0; row < mat.rows(); ++row)
	{
		CAPTURE(row);
		CHECK(ys[row] == cadet::test::makeApprox(yd[row], std::numeric_limits<double>::epsilon() * 100.0, 0.0));
	}
}
