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

#include "linalg/BandMatrix.hpp"

#include <sstream>
#include <ostream>
#include <cmath>
#include <algorithm>

namespace cadet
{

namespace linalg
{

// Anonymous namespace to hide implementation details in this translation unit
namespace
{

void bandMatrixVectorMultiplication(int rows, int upperBand, int lowerBand, int stride, double const* const data,
									double alpha, double beta, double const* const x, double* const y)
{
	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix. Thus,
	// upper and lower diagonals interchange.
	lapackInt_t n = rows;
	lapackInt_t kl = upperBand;
	lapackInt_t ku = lowerBand;
	lapackInt_t ldab = stride;
	lapackInt_t inc = 1; // Stride in vectors (here, elements are continuous without intermediate space)

	// For LAPACK the matrix looks like it's transposed. We, thus,
	// multiply with the transposed matrix, which in the end uses the original matrix.
	char trans[] = "T";

	// LAPACK computes y <- alpha * A * x + beta * y
	LapackMultiplyDenseBanded(trans, &n, &n, &kl, &ku, &alpha, const_cast<double*>(data), &ldab, const_cast<double*>(x),
							  &inc, &beta, const_cast<double*>(y), &inc);
}

template <class MatrixType> void bandMatrixToSparseString(std::ostream& out, const MatrixType& mt)
{
	std::ostringstream cols;
	std::ostringstream rows;
	std::ostringstream elems;

	cols.copyfmt(out);
	rows.copyfmt(out);
	elems.copyfmt(out);

	cols << "cols = [";
	rows << "rows = [";
	elems << "elems = [";

	for (int row = 0; row < mt.rows(); ++row)
	{
		const int lower = std::max(-static_cast<int>(mt.lowerBandwidth()), -static_cast<int>(row));
		const int upper = std::min(static_cast<int>(mt.upperBandwidth()), static_cast<int>(mt.rows() - row) - 1);
		for (int col = lower; col <= upper; ++col)
		{
			if ((row > 0) || (col > 0))
			{
				cols << ", ";
				rows << ", ";
				elems << ", ";
			}

			cols << col + static_cast<int>(row) + 1;
			rows << (row + 1);
			elems << mt.centered(row, col);
		}
	}

	cols << "];";
	rows << "];";
	elems << "];";

	out << cols.str() << "\n";
	out << rows.str() << "\n";
	out << elems.str() << "\n";
	out << "size: " << mt.rows() << " bandwidth: " << mt.lowerBandwidth() << " + 1 + " << mt.upperBandwidth();
}

void scaleRows(double* data, int elemPerRow, int stride, double const* scalingFactors, int numRows)
{
	for (int i = 0; i < numRows; ++i, data += stride)
	{
		cadet_assert(scalingFactors[i] != 0.0);
		for (int j = 0; j < elemPerRow; ++j)
			data[j] /= scalingFactors[i];
	}
}

void rowScaleFactors(double const* data, int elemPerRow, int stride, double* scalingFactors, int numRows)
{
	for (int i = 0; i < numRows; ++i, data += stride)
	{
		scalingFactors[i] = 0.0;
		for (int j = 0; j < elemPerRow; ++j)
			scalingFactors[i] = std::max(scalingFactors[i], std::abs(data[j]));
	}
}

} // namespace

void BandMatrix::multiplyVector(const double* const x, double alpha, double beta, double* const y) const
{
	bandMatrixVectorMultiplication(_rows, _upperBand, _lowerBand, stride(), _data, alpha, beta, x, y);
}

void BandMatrix::submatrixMultiplyVector(const double* const x, int startRow, int startDiag, int numRows, int numCols,
										 double alpha, double beta, double* const y) const
{
	cadet_assert(startDiag >= -static_cast<int>(_lowerBand));
	cadet_assert(startDiag <= static_cast<int>(_upperBand));
	cadet_assert(startRow < _rows);
	cadet_assert(startRow + numRows <= _rows);

	const int upperBand = static_cast<int>(_upperBand);
	const int lowerBand = static_cast<int>(_lowerBand);

	for (int r = 0; r < numRows; ++r)
	{
		double temp = 0.0;
		for (int c = 0; c < numCols; ++c)
		{
			// Compute diagonal index of current position and shift it by startDiag
			const int curDiag = c - r + startDiag;

			if (cadet_likely((curDiag >= -lowerBand) && (curDiag <= upperBand)))
				temp += centered(r + startRow, curDiag) * x[c];
		}
		y[r] = alpha * temp + beta * y[r];
	}

	// @todo Figure out why LAPACK implementation below does not work
	/*
		cadet_assert(startDiag >= -static_cast<int>(_lowerBand));
		cadet_assert(startDiag <= static_cast<int>(_upperBand));
		cadet_assert(startRow < _rows);
		cadet_assert(startRow + numRows <= _rows);

		// Since LAPACK uses column-major storage and we use row-major,
		// we actually have constructed the transposed matrix. Thus,
		// upper and lower diagonals interchange.
		lapackInt_t n = numRows;
		lapackInt_t m = numCols;
		lapackInt_t kl = _upperBand;
		lapackInt_t ku = _lowerBand;
		lapackInt_t ldab = static_cast<int>(stride());
		lapackInt_t inc = 1; // Stride in vectors (here, elements are continuous without intermediate space)

		// For LAPACK the matrix looks like it's transposed. We, thus,
		// multiply with the transposed matrix, which in the end uses the original matrix.
		char trans[] = "T";

		if (startDiag < 0)
		{
			ku = std::max(0, static_cast<int>(ku) + startDiag);
			kl = std::min(static_cast<lapackInt_t>(numCols), kl - static_cast<lapackInt_t>(startDiag));
		}
		else if (startDiag > 0)
		{
			kl = std::max(0, static_cast<int>(kl) - startDiag);
			ku = std::min(static_cast<lapackInt_t>(numCols), ku + static_cast<lapackInt_t>(startDiag));
		}

		int offset = 0;

		// LAPACK computes y <- alpha * A * x + beta * y
		LapackMultiplyDenseBanded(trans, &m, &n, &kl, &ku, &alpha, const_cast<double*>(_data) + startRow * stride() +
	   offset, &ldab, const_cast<double*>(x), &inc, &beta, const_cast<double*>(y), &inc);
	*/
}

void BandMatrix::scaleRows(double const* scalingFactors, int numRows)
{
	cadet_assert(numRows <= _rows);
	cadet::linalg::scaleRows(_data, stride(), stride(), scalingFactors, numRows);
}

void BandMatrix::rowScaleFactors(double* scalingFactors, int numRows) const
{
	cadet_assert(numRows <= _rows);
	cadet::linalg::rowScaleFactors(_data, stride(), stride(), scalingFactors, numRows);
}

void FactorizableBandMatrix::multiplyVector(const double* const x, double* const y) const
{
	bandMatrixVectorMultiplication(_rows, _upperBand, _lowerBand, stride(), _data + _upperBand, 1.0, 0.0, x, y);
}

void FactorizableBandMatrix::multiplyVector(const double* const x, double alpha, double beta, double* const y) const
{
	bandMatrixVectorMultiplication(_rows, _upperBand, _lowerBand, stride(), _data + _upperBand, alpha, beta, x, y);
}

bool FactorizableBandMatrix::factorize()
{
	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix. Thus,
	// upper and lower diagonals interchange.
	lapackInt_t n = _rows;
	lapackInt_t kl = _upperBand;
	lapackInt_t ku = _lowerBand;
	lapackInt_t ldab = stride();
	lapackInt_t flag = 0;

	LapackFactorDenseBanded(&n, &n, &kl, &ku, _data, &ldab, _pivot, &flag);

	// If the flag is -i (for i > 0), the ith argument is invalid
	// If the flag is +i (for i > 0), the ith main diagonal entry of U is 0 and, thus, the system is not solvable
	return flag == 0;
}

bool FactorizableBandMatrix::solve(double* rhs) const
{
	// Since LAPACK uses column-major storage and we use row-major,
	// we actually have constructed the transposed matrix. Thus,
	// upper and lower diagonals interchange.
	lapackInt_t n = _rows;
	lapackInt_t kl = _upperBand;
	lapackInt_t ku = _lowerBand;
	lapackInt_t nrhs = 1;
	lapackInt_t ldab = stride();
	lapackInt_t flag = 0;

	// For LAPACK the matrix looks like it's transposed. We, thus,
	// solve the transposed equation which uses the original matrix.
	char trans[] = "T";

	LapackSolveDenseBanded(trans, &n, &kl, &ku, &nrhs, const_cast<double*>(_data), &ldab,
						   const_cast<lapackInt_t*>(_pivot), rhs, &n, &flag);

	// If the flag is -i (for i > 0), the ith argument is invalid
	return flag == 0;
}

bool FactorizableBandMatrix::solve(double const* scalingFactors, double* rhs) const
{
	for (int i = 0; i < _rows; ++i)
		rhs[i] /= scalingFactors[i];
	return solve(rhs);
}

void FactorizableBandMatrix::scaleRows(double const* scalingFactors, int numRows)
{
	cadet_assert(numRows <= _rows);
	cadet::linalg::scaleRows(_data + _upperBand, apparentStride(), stride(), scalingFactors, numRows);
}

void FactorizableBandMatrix::rowScaleFactors(double* scalingFactors, int numRows) const
{
	cadet_assert(numRows <= _rows);
	cadet::linalg::rowScaleFactors(_data + _upperBand, apparentStride(), stride(), scalingFactors, numRows);
}

std::ostream& operator<<(std::ostream& out, const BandMatrix& bm)
{
	bandMatrixToSparseString(out, bm);
	return out;
}

std::ostream& operator<<(std::ostream& out, const FactorizableBandMatrix& fbm)
{
	bandMatrixToSparseString(out, fbm);
	return out;
}

} // namespace linalg

} // namespace cadet
