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

#ifdef SUPERLU_FOUND
#include "linalg/SuperLUSparseMatrix.hpp"
#endif
#ifdef UMFPACK_FOUND
#include "linalg/UMFPackSparseMatrix.hpp"
#endif

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

template <typename matrix_t>
inline void checkSparseAgainstDenseInverse(matrix_t& factMat, cadet::linalg::DenseMatrix& dm)
{
	const int numRows = static_cast<int>(factMat.rows());
	std::vector<double> rhs1(factMat.rows(), 0.0);
	std::vector<double> rhs2(factMat.rows(), 0.0);
	for (int col = 0; col < numRows; ++col)
	{
		std::fill(rhs1.begin(), rhs1.end(), 0.0);
		std::fill(rhs2.begin(), rhs2.end(), 0.0);
		rhs1[col] = 1.0;
		rhs2[col] = 1.0;

		factMat.solve(rhs1.data());
		dm.solve(rhs2.data());

		for (int row = 0; row < numRows; ++row)
		{
			CAPTURE(row);
			CAPTURE(col);
			CHECK(rhs1[row] == cadet::test::makeApprox(rhs2[row], std::numeric_limits<double>::epsilon() * 100.0, 0.0));
		}
	}
}

template <typename matrix_t> inline void sparseMatrixFactorize()
{
	// Create matrix with pattern
	cadet::linalg::CompressedSparseMatrix mat = createSmallMatrix();

	matrix_t factMat;
	factMat.assignPattern(mat);
	factMat.copyFromSamePattern(mat);

	factMat.prepare();
	REQUIRE(factMat.factorize());
}

template <typename matrix_t> inline void sparseMatrixInverse()
{
	// Create matrix with pattern
	cadet::linalg::CompressedSparseMatrix mat = createSmallMatrix();

	// Create equivalent dense matrix
	cadet::linalg::DenseMatrix dm = createSmallMatrixDense();

	matrix_t factMat;
	factMat.assignPattern(mat);
	factMat.copyFromSamePattern(mat);

	factMat.prepare();
	REQUIRE(factMat.factorize());

	REQUIRE(dm.factorize());

	checkSparseAgainstDenseInverse(factMat, dm);
}

template <typename matrix_t> inline void sparseMatrixRepeatedFactorization()
{
	// Create matrix with pattern
	cadet::linalg::CompressedSparseMatrix mat = createSmallMatrix();

	// Create equivalent dense matrix
	cadet::linalg::DenseMatrix dm = createSmallMatrixDense();

	matrix_t factMat;
	factMat.assignPattern(mat);
	factMat.copyFromSamePattern(mat);

	factMat.prepare();
	REQUIRE(factMat.factorize());

	REQUIRE(dm.factorize());

	checkSparseAgainstDenseInverse(factMat, dm);

	// Modify matrices
	double* const vs = factMat.valuesOfRow(0);
	for (int i = 0; i < factMat.numNonZeros(); ++i)
		vs[i] *= 2.0;

	dm = createSmallMatrixDense();
	double* const vd = dm.data();
	for (int i = 0; i < dm.rows() * dm.columns(); ++i)
		vd[i] *= 2.0;

	// Factorize again
	REQUIRE(factMat.factorize());
	REQUIRE(dm.factorize());

	checkSparseAgainstDenseInverse(factMat, dm);
}
} // namespace

#ifdef SUPERLU_FOUND

TEST_CASE("SuperLU sparse matrix factorize", "[SuperLU],[SparseMatrix],[LinAlg]")
{
	sparseMatrixFactorize<cadet::linalg::SuperLUSparseMatrix>();
}

TEST_CASE("SuperLU sparse matrix inverse", "[SuperLU],[SparseMatrix],[LinAlg]")
{
	sparseMatrixInverse<cadet::linalg::SuperLUSparseMatrix>();
}

TEST_CASE("SuperLU sparse matrix repeated factorization", "[SuperLU],[SparseMatrix],[LinAlg]")
{
	sparseMatrixRepeatedFactorization<cadet::linalg::SuperLUSparseMatrix>();
}

#endif

#ifdef UMFPACK_FOUND

TEST_CASE("UMFPACK sparse matrix factorize", "[UMFPACK],[SparseMatrix],[LinAlg]")
{
	sparseMatrixFactorize<cadet::linalg::UMFPackSparseMatrix>();
}

TEST_CASE("UMFPACK sparse matrix inverse", "[UMFPACK],[SparseMatrix],[LinAlg]")
{
	sparseMatrixInverse<cadet::linalg::UMFPackSparseMatrix>();
}

TEST_CASE("UMFPACK sparse matrix repeated factorization", "[UMFPACK],[SparseMatrix],[LinAlg]")
{
	sparseMatrixRepeatedFactorization<cadet::linalg::UMFPackSparseMatrix>();
}

#endif
