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

#include "linalg/DenseMatrix.hpp"
#include <cmath>
#include <algorithm>

namespace cadet
{

namespace linalg
{

namespace detail
{

void DenseMatrixBase::submatrixSetAll(double val, int startRow, int startCol, int numRows, int numCols)
{
	cadet_assert(_rows > startRow);
	cadet_assert(_cols > startCol);
	cadet_assert(_rows >= startRow + numRows);
	cadet_assert(_cols >= startCol + numCols);

	double* const ptrDest = _data + startRow * stride() + startCol;

	for (int i = 0; i < numRows; ++i)
	{
		std::fill(ptrDest + i * stride(), ptrDest + i * stride() + numCols, val);
	}
}

void DenseMatrixBase::submatrixAssign(const DenseMatrixBase& mat, int startRow, int startCol, int numRows, int numCols)
{
	cadet_assert(numRows == mat.rows());
	cadet_assert(numCols == mat.columns());
	cadet_assert(_rows > startRow);
	cadet_assert(_cols > startCol);
	cadet_assert(_rows >= startRow + numRows);
	cadet_assert(_cols >= startCol + numCols);

	double* const ptrDest = _data + startRow * stride() + startCol;
	double const* const ptrSrc = mat.data();

	for (int i = 0; i < numRows; ++i)
	{
		std::copy(ptrSrc + i * mat.stride(), ptrSrc + i * mat.stride() + numCols, ptrDest + i * stride());
	}
}

void DenseMatrixBase::multiplyVector(const double* const x, double alpha, double beta, double* const y) const
{
	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix. Thus,
	// rows and columns interchange.
	lapackInt_t m = _cols;
	lapackInt_t n = _rows;
	lapackInt_t lda = stride();
	lapackInt_t inc = 1; // Stride in vectors (here, elements are continuous without intermediate space)

	// For LAPACK the matrix looks like it's transposed. We, thus,
	// multiply with the transposed matrix, which in the end uses the original matrix.
	char trans[] = "T";

	// LAPACK computes y <- alpha * A * x + beta * y
	LapackMultiplyDense(trans, &m, &n, &alpha, const_cast<double*>(_data), &lda, const_cast<double*>(x), &inc, &beta,
						const_cast<double*>(y), &inc);
}

void DenseMatrixBase::submatrixMultiplyVector(const double* const x, int startRow, int startCol, int numRows,
											  int numCols, double alpha, double beta, double* const y) const
{
	cadet_assert(_rows > startRow);
	cadet_assert(_cols > startCol);
	cadet_assert(_rows >= startRow + numRows);
	cadet_assert(_cols >= startCol + numCols);

	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix. Thus,
	// rows and columns interchange.
	lapackInt_t m = numCols;
	lapackInt_t n = numRows;
	lapackInt_t lda = stride();
	lapackInt_t inc = 1; // Stride in vectors (here, elements are continuous without intermediate space)

	// For LAPACK the matrix looks like it's transposed. We, thus,
	// multiply with the transposed matrix, which in the end uses the original matrix.
	char trans[] = "T";

	// Pointer to first entry of submatrix
	double* const data = const_cast<double*>(_data) + startRow * lda + startCol;

	// LAPACK computes y <- alpha * A * x + beta * y
	LapackMultiplyDense(trans, &m, &n, &alpha, data, &lda, const_cast<double*>(x), &inc, &beta, const_cast<double*>(y),
						&inc);
}

void DenseMatrixBase::transposedMultiplyVector(const double* const x, double alpha, double beta, double* const y) const
{
	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix. Thus,
	// rows and columns do not interchange.
	lapackInt_t m = _rows;
	lapackInt_t n = _cols;
	lapackInt_t lda = stride();
	lapackInt_t inc = 1; // Stride in vectors (here, elements are continuous without intermediate space)

	char trans[] = "N";

	// LAPACK computes y <- alpha * A * x + beta * y
	LapackMultiplyDense(trans, &m, &n, &alpha, const_cast<double*>(_data), &lda, const_cast<double*>(x), &inc, &beta,
						const_cast<double*>(y), &inc);
}

void DenseMatrixBase::transposedSubmatrixMultiplyVector(const double* const x, int startRow, int startCol, int numRows,
														int numCols, double alpha, double beta, double* const y) const
{
	cadet_assert(_rows > startRow);
	cadet_assert(_cols > startCol);
	cadet_assert(_rows >= startRow + numRows);
	cadet_assert(_cols >= startCol + numCols);

	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix. Thus,
	// rows and columns do not interchange.
	lapackInt_t m = numRows;
	lapackInt_t n = numCols;
	lapackInt_t lda = stride();
	lapackInt_t inc = 1; // Stride in vectors (here, elements are continuous without intermediate space)

	char trans[] = "N";

	// Pointer to first entry of submatrix
	double* const data = const_cast<double*>(_data) + startRow * lda + startCol;

	// LAPACK computes y <- alpha * A * x + beta * y
	LapackMultiplyDense(trans, &m, &n, &alpha, data, &lda, const_cast<double*>(x), &inc, &beta, const_cast<double*>(y),
						&inc);
}

bool DenseMatrixBase::factorize()
{
	cadet_assert(_rows == _cols);

	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix. However,
	// this is irrelevant for factorization.
	lapackInt_t n = _rows;
	lapackInt_t lda = stride();
	lapackInt_t flag = 0;

	LapackFactorDense(&n, &n, _data, &lda, _pivot, &flag);

	// If the flag is -i (for i > 0), the ith argument is invalid
	// If the flag is +i (for i > 0), the ith main diagonal entry of U is 0 and, thus, the system is not solvable
	return flag == 0;
}

bool DenseMatrixBase::solve(double* rhs) const
{
	cadet_assert(_rows == _cols);

	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix.
	lapackInt_t n = _rows;
	lapackInt_t nrhs = 1;
	lapackInt_t lda = stride();
	lapackInt_t flag = 0;

	// For LAPACK the matrix looks like it's transposed. We, thus,
	// solve the transposed equation which uses the original matrix.
	char trans[] = "T";

	LapackSolveDense(trans, &n, &nrhs, const_cast<double*>(_data), &lda, const_cast<lapackInt_t*>(_pivot), rhs, &n,
					 &flag);

	// If the flag is -i (for i > 0), the ith argument is invalid
	return flag == 0;
}

bool DenseMatrixBase::solve(double const* scalingFactors, double* rhs) const
{
	for (int i = 0; i < _rows; ++i)
		rhs[i] /= scalingFactors[i];
	return solve(rhs);
}

int DenseMatrixBase::optimalLeastSquaresWorkspace() const
{
	cadet_assert(_rows >= _cols);

	// For LAPACK the matrix looks like it's transposed. We, thus,
	// solve the transposed equation which uses the original matrix.
	char trans[] = "T";

	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix.
	lapackInt_t n = _rows;
	lapackInt_t m = _cols;
	lapackInt_t nrhs = 1;
	lapackInt_t lda = stride();
	lapackInt_t lwork = -1;
	lapackInt_t flag = 0;
	double work = 0.0;

	LapackDenseLeastSquares(trans, &m, &n, &nrhs, const_cast<double*>(_data), &lda, nullptr, &n, &work, &lwork, &flag);

	if (flag != 0)
		return -1;

	return static_cast<int>(work);
}

bool DenseMatrixBase::leastSquaresSolve(double* rhs, double* workspace, int size)
{
	// @todo Replace with calls to underlying LAPACK methods dtrtrs, dgelqf, and dormlq
	cadet_assert(_rows >= _cols);
	cadet_assert(size >= std::min(_cols, _rows));

	// For LAPACK the matrix looks like it's transposed. We, thus,
	// solve the transposed equation which uses the original matrix.
	char trans[] = "T";

	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix.
	lapackInt_t n = _rows;
	lapackInt_t m = _cols;
	lapackInt_t nrhs = 1;
	lapackInt_t lda = stride();
	lapackInt_t lwork = size;
	lapackInt_t flag = 0;

	LapackDenseLeastSquares(trans, &m, &n, &nrhs, _data, &lda, rhs, &n, workspace, &lwork, &flag);

	return flag == 0;
}

bool DenseMatrixBase::robustFactorize(double* const workspace)
{
	cadet_assert(_rows == _cols);

	lapackInt_t n = _rows;
	lapackInt_t m = _cols;
	lapackInt_t lda = stride();
	lapackInt_t flag = 0;

	// Compute LQ factorization A = L * Q with orthogonal matrix Q and lower triagonal matrix L
	// Note that LAPACK sees the tranposed matrix A^T instead of A. The LQ factorization of A^T
	// is essentially the same as QR factorization of A.
	LapackFactorLQDense(&m, &n, _data, &lda, workspace, workspace + _rows, &n, &flag);

	return flag == 0;
}

bool DenseMatrixBase::robustSolve(double const* scalingFactors, double* rhs, double* const workspace) const
{
	for (int i = 0; i < _rows; ++i)
		rhs[i] /= scalingFactors[i];
	return robustSolve(rhs, workspace);
}

bool DenseMatrixBase::robustSolve(double* rhs, double* const workspace) const
{
	cadet_assert(_rows == _cols);

	// For LAPACK the matrix looks like it's transposed. We, thus,
	// work with the transposed (i.e., the original) matrix.
	// From LAPACK's point of view, we want to solve A^T * x = y with the factorization A = L * Q.
	// The solution is given by x = L^{-T} * Q * y.

	char side[] = "L";
	char transQ[] = "N";

	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix.
	lapackInt_t n = _rows;
	lapackInt_t m = _cols;
	lapackInt_t nrhs = 1;
	lapackInt_t lda = stride();
	lapackInt_t flag = 0;

	// Calculate z = Q * y
	LapackMultiplyFactorizedQ(side, transQ, &m, &nrhs, &m, _data, &lda, workspace, rhs, &n, workspace + _rows, &n,
							  &flag);
	if (flag != 0)
		return false;

	// Calculate x = L^{-T} * Q * y = L^{-T} * z
	char transL[] = "T";
	LapackSolveTriangular(side, transL, transQ, &m, &nrhs, _data, &lda, rhs, &n, &flag);
	return flag == 0;
}

void DenseMatrixBase::scaleRows(double const* scalingFactors, int numRows)
{
	const int ld = stride();
	for (int i = 0; i < numRows; ++i)
	{
		cadet_assert(scalingFactors[i] != 0.0);
		for (int j = 0; j < _cols; ++j)
			_data[i * ld + j] /= scalingFactors[i];
	}
}

void DenseMatrixBase::scaleColumns(double const* scalingFactors, int numCols)
{
	const int ld = stride();
	for (int i = 0; i < _rows; ++i)
	{
		for (int j = 0; j < numCols; ++j)
		{
			cadet_assert(scalingFactors[j] != 0.0);
			_data[i * ld + j] /= scalingFactors[j];
		}
	}
}

void DenseMatrixBase::rowScaleFactors(double* scalingFactors, int numRows) const
{
	const int ld = stride();
	for (int i = 0; i < numRows; ++i)
	{
		scalingFactors[i] = 0.0;
		for (int j = 0; j < _cols; ++j)
			scalingFactors[i] = std::max(scalingFactors[i], std::abs(_data[i * ld + j]));
	}
}

void DenseMatrixBase::columnScaleFactors(double* scalingFactors, int numCols) const
{
	const int ld = stride();
	for (int j = 0; j < numCols; ++j)
	{
		scalingFactors[j] = 0.0;
		for (int i = 0; i < _rows; ++i)
			scalingFactors[j] = std::max(scalingFactors[j], std::abs(_data[i * ld + j]));
	}
}

std::ostream& operator<<(std::ostream& out, const DenseMatrixBase& bm)
{
	out << "[";
	for (int i = 0; i < bm.rows(); ++i)
	{
		for (int j = 0; j < bm.columns(); ++j)
		{
			out << bm.native(i, j);
			if (j != bm.columns() - 1)
				out << ", ";
		}
		if (i != bm.rows() - 1)
			out << "; ";
	}
	out << "]";
	return out;
}

} // namespace detail

} // namespace linalg

} // namespace cadet
