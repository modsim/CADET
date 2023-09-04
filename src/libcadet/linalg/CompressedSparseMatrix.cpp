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

#include "linalg/CompressedSparseMatrix.hpp"

#include <sstream>
#include <ostream>

namespace cadet
{

namespace linalg
{

void SparsityPattern::compressTo(sparse_int_t* colIdx, sparse_int_t* rowStart) const CADET_NOEXCEPT
{
	cadet_assert(colIdx != nullptr);
	cadet_assert(rowStart != nullptr);

	sparse_int_t* curColIdx = colIdx;
	int curRowStart = 0;
	for (int r = 0; r < static_cast<int>(_rows.size()); ++r)
	{
		// Mark beginning of row
		rowStart[r] = curRowStart;

		// Copy column indices
		const std::vector<sparse_int_t>& colIndicesInRow = _rows[r].columnIndices();
		std::copy(colIndicesInRow.begin(), colIndicesInRow.end(), curColIdx);

		// Advance pointers and counters
		curColIdx += colIndicesInRow.size();
		curRowStart += colIndicesInRow.size();
	}

	rowStart[_rows.size()] = curRowStart;
	cadet_assert(curRowStart == _numNonZero);
}

SparsityPatternRowIterator SparsityPattern::row(int idxRow) CADET_NOEXCEPT
{
	return SparsityPatternRowIterator(*this, idxRow);
}

std::ostream& operator<<(std::ostream& out, const CompressedSparseMatrix& sm)
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

	for (int row = 0; row < sm.rows(); ++row)
	{
		sparse_int_t const* const colIdx = sm.columnIndicesOfRow(row);
		double const* const localVal = sm.valuesOfRow(row);
		for (int col = 0; col < sm.numNonZerosInRow(row); ++col)
		{
			if ((row > 0) || (col > 0))
			{
				cols << ", ";
				rows << ", ";
				elems << ", ";
			}

			cols << (colIdx[col] + 1);
			rows << (row + 1);
			elems << localVal[col];
		}
	}

	cols << "];";
	rows << "];";
	elems << "];";

	out << cols.str() << "\n";
	out << rows.str() << "\n";
	out << elems.str() << "\n";
	out << "size: " << sm.rows();
	return out;
}

void CompressedSparseMatrix::copyFromSamePattern(const CompressedSparseMatrix& src)
{
	cadet_assert(src._rowStart.size() == _rowStart.size());
	cadet_assert(src._colIdx.size() == _colIdx.size());
	cadet_assert(src._values.size() == _values.size());
	std::copy(src._values.begin(), src._values.end(), _values.begin());
}

void CompressedSparseMatrix::resize(int numRows, int numNonZeros)
{
	_values.clear();
	_values.resize(numNonZeros, 0.0);

	_colIdx.clear();
	_colIdx.resize(numNonZeros, 0);

	_rowStart.clear();
	_rowStart.resize(numRows + 2, 0);
}

void CompressedSparseMatrix::assignPattern(const SparsityPattern& pattern)
{
	resize(pattern.rows(), pattern.numNonZeros());
	pattern.compressTo(_colIdx.data(), _rowStart.data());
}

void CompressedSparseMatrix::assignPattern(const CompressedSparseMatrix& pattern)
{
	_values.clear();
	_values.resize(pattern.numNonZeros(), 0.0);

	_colIdx = pattern._colIdx;
	_rowStart = pattern._rowStart;
}

void CompressedSparseMatrix::copyFromSubPattern(const CompressedSparseMatrix& mat)
{
	// Iterate over rows
	for (int row = 0; row < mat.rows(); ++row)
	{
		sparse_int_t curIdx = _rowStart[row];

		// Iterate over source elements in row
		for (sparse_int_t i = mat._rowStart[row]; i < mat._rowStart[row + 1]; ++i)
		{
			// Find destination column index mat._colIdx[i] in this row
			// Due to the column indices being sorted, this should be reasonably fast
			// TODO: Check whether to replace linear with binary search
			while (_colIdx[curIdx] != mat._colIdx[i])
			{
				++curIdx;
				cadet_assert(curIdx < _rowStart[row + 1]);
			}

			_values[curIdx] = mat._values[i];
		}
	}
}

void CompressedSparseMatrix::copyRowToRowSamePattern(const CompressedSparseMatrix& mat, int srcRow, int destRow)
{
	const sparse_int_t start = mat._rowStart[destRow];
	const sparse_int_t end = mat._rowStart[destRow + 1];
	std::copy(mat._values.begin() + start, mat._values.begin() + end, _values.begin() + _rowStart[destRow]);
}

void CompressedSparseMatrix::copyRowToRowSubPattern(const CompressedSparseMatrix& mat, int srcRow, int destRow)
{
	sparse_int_t curIdx = _rowStart[destRow];

	// Iterate over source elements in row
	for (sparse_int_t i = mat._rowStart[srcRow]; i < mat._rowStart[srcRow + 1]; ++i)
	{
		// Find destination column index mat._colIdx[i] in this row
		// Due to the column indices being sorted, this should be reasonably fast
		// TODO: Check whether to replace linear with binary search
		while (_colIdx[curIdx] != mat._colIdx[i])
		{
			++curIdx;
			cadet_assert(curIdx < _rowStart[destRow + 1]);
		}

		_values[curIdx] = mat._values[i];
	}
}

BandedSparseRowIterator CompressedSparseMatrix::row(int idx)
{
	cadet_assert((idx >= 0) && (idx < rows()));
	return BandedSparseRowIterator(*this, idx);
}

ConstBandedSparseRowIterator CompressedSparseMatrix::row(int idx) const
{
	cadet_assert((idx >= 0) && (idx < rows()));
	return ConstBandedSparseRowIterator(*this, idx);
}

void CompressedSparseMatrix::multiplyVector(double const* const x, double* y) const
{
	for (int r = 0; r < rows(); ++r)
	{
		y[r] = 0.0;

		const int nnzInRow = numNonZerosInRow(r);
		sparse_int_t const* const localColIdx = _colIdx.data() + _rowStart[r];
		double const* const localVal = _values.data() + _rowStart[r];

		for (int c = 0; c < nnzInRow; ++c)
			y[r] += localVal[c] * x[localColIdx[c]];
	}
}

void CompressedSparseMatrix::multiplyVector(double const* const x, double alpha, double beta, double* y) const
{
	for (int r = 0; r < rows(); ++r)
	{
		y[r] *= beta;

		const int nnzInRow = numNonZerosInRow(r);
		sparse_int_t const* const localColIdx = _colIdx.data() + _rowStart[r];
		double const* const localVal = _values.data() + _rowStart[r];

		for (int c = 0; c < nnzInRow; ++c)
			y[r] += alpha * localVal[c] * x[localColIdx[c]];
	}
}

} // namespace linalg

} // namespace cadet
