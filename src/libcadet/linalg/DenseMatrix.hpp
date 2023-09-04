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
 * Defines a rectangular dense matrix that can be factorized if it is square
 */

#ifndef LIBCADET_DENSEMATRIX_HPP_
#define LIBCADET_DENSEMATRIX_HPP_

#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"
#include "LapackInterface.hpp"

#include <ostream>
#include <algorithm>

namespace cadet
{

namespace linalg
{

namespace detail
{

/**
 * @brief Iterates over rows of a dense matrix providing a banded interface
 * @details The iterator always points to the main diagonal element in a row of
 *          a rectangular matrix. The user can then use the index of a sub-
 *          or superdiagonal to query and set matrix elements. Element queries
 *          outside the actual matrix are ignored and do not lead to failure.
 * @tparam MatrixType Type of the underlying dense matrix class
 */
template <class MatrixType> class DenseBandedRowIterator
{
public:
	/**
	 * @brief Creates an empty DenseBandedRowIterator pointing to nothing
	 * @param [in] mat Matrix this DenseBandedRowIterator accesses
	 */
	DenseBandedRowIterator() CADET_NOEXCEPT : _matrix(nullptr), _pos(nullptr), _rowIdx(0)
	{
	}

	/**
	 * @brief Creates a DenseBandedRowIterator for the given matrix
	 * @param [in] mat Matrix this DenseBandedRowIterator accesses
	 */
	DenseBandedRowIterator(MatrixType& mat) CADET_NOEXCEPT : _matrix(&mat), _pos(_matrix->_data), _rowIdx(0)
	{
	}

	/**
	 * @brief Creates a DenseBandedRowIterator for the given matrix
	 * @param [in] mat Matrix this DenseBandedRowIterator accesses
	 * @param [in] row Index of the row of the iterator points so
	 */
	DenseBandedRowIterator(MatrixType& mat, int row) CADET_NOEXCEPT
		: _matrix(&mat),
		  _pos(_matrix->_data + row * _matrix->stride() + row),
		  _rowIdx(row)
	{
	}

	/**
	 * @brief Creates a DenseBandedRowIterator for the given matrix
	 * @param [in] mat Matrix this DenseBandedRowIterator accesses
	 * @param [in] row Index of the row of the iterator points so
	 * @param [in] diagOffset Diagonal this iterator points to (offset)
	 */
	DenseBandedRowIterator(MatrixType& mat, int row, int diagOffset) CADET_NOEXCEPT
		: _matrix(&mat),
		  _pos(_matrix->_data + static_cast<int>(row * _matrix->stride() + row) + diagOffset),
		  _rowIdx(row)
	{
	}

	DenseBandedRowIterator(const DenseBandedRowIterator& cpy, int rowChange) CADET_NOEXCEPT
		: _matrix(cpy._matrix),
		  _pos(cpy._pos + rowChange * static_cast<int>(cpy._matrix->stride() + 1)),
		  _rowIdx(static_cast<int>(cpy._rowIdx) + rowChange)
	{
	}
	DenseBandedRowIterator(const DenseBandedRowIterator& cpy) CADET_NOEXCEPT : _matrix(cpy._matrix),
																			   _pos(cpy._pos),
																			   _rowIdx(cpy._rowIdx)
	{
	}
	DenseBandedRowIterator(DenseBandedRowIterator&& cpy) CADET_NOEXCEPT : _matrix(cpy._matrix),
																		  _pos(cpy._pos),
																		  _rowIdx(cpy._rowIdx)
	{
	}

	inline DenseBandedRowIterator& operator=(const DenseBandedRowIterator& cpy) CADET_NOEXCEPT
	{
		cadet_assert(&cpy != this);

		_matrix = cpy._matrix;
		_pos = cpy._pos;
		_rowIdx = cpy._rowIdx;
		return *this;
	}

	inline DenseBandedRowIterator& operator=(DenseBandedRowIterator&& cpy) CADET_NOEXCEPT
	{
		_matrix = cpy._matrix;
		_pos = cpy._pos;
		_rowIdx = cpy._rowIdx;
		return *this;
	}

	/**
	 * @brief Accesses an element in the current row where the main diagonal is centered (index @c 0)
	 * @details The @p diagonal determines the element in a row. A negative index indicates a lower diagonal,
	 *          while a positive index indicates an upper diagonal. If @p diagonal is @c 0, then the main
	 *          diagonal is retrieved.
	 *
	 * @param [in] diagonal Index of the diagonal
	 * @return Matrix element at the given position
	 */
	inline double& centered(int diagonal)
	{
		return (*this)(diagonal);
	}
	inline double centered(int diagonal) const
	{
		return (*this)(diagonal);
	}

	inline double& operator()(int diagonal)
	{
		// Check if out of scope
		if (cadet_unlikely(diagonal >= static_cast<int>(_matrix->columns()) - _rowIdx))
			return _dummy;
		if (cadet_unlikely(diagonal < -_rowIdx))
			return _dummy;

		return _pos[diagonal];
	}

	inline double operator()(int diagonal) const
	{
		// Check if out of scope
		if (cadet_unlikely(diagonal >= static_cast<int>(_matrix->columns()) - _rowIdx))
			return 0.0;
		if (cadet_unlikely(diagonal < -_rowIdx))
			return 0.0;

		return _pos[diagonal];
	}

	inline double& operator[](int diagonal)
	{
		return (*this)(diagonal);
	}
	inline double operator[](int diagonal) const
	{
		return (*this)(diagonal);
	}

	/**
	 * @brief Sets all row elements to the given value
	 * @param [in] val Value all row elements are set to
	 */
	inline void setAll(double val)
	{
		std::fill(_pos - _rowIdx, _pos + static_cast<int>(_matrix->columns()) - _rowIdx, val);
	}

	/**
	 * @brief Copies a row of another row iterator to the current row
	 * @param [in] ri Other row iterator
	 * @tparam OtherMatrix Underlying matrix type of other row iterator
	 */
	template <typename OtherMatrix> inline void copyRowFrom(const DenseBandedRowIterator<OtherMatrix>& ri)
	{
		cadet_assert(_matrix->columns() >= ri._matrix->columns());
		std::copy_n(ri._pos - ri._rowIdx, _matrix->columns(), _pos - _rowIdx);
	}

	/**
	 * @brief Adds a subset of this row @f$ x @f$ to another row @f$ y @f$ performing @f$ y = y + \alpha x @f$
	 * @details Although the subset is specified with a bandmatrix interface, the rows are
				added as in a normal matrix.
	 * @param [in] riDest Row iterator pointing to the destination row @f$ y @f$
	 * @param [in] factor Factor @f$ \alpha @f$
	 * @param [in] lowerBand Start point of the row subset as lower band index
	 * @param [in] upperBand End point of the row subset as upper band index
	 * @tparam OtherMatrix Underlying matrix type of other row iterator
	 */
	template <typename OtherMatrix>
	inline void addSubsetTo(const DenseBandedRowIterator<OtherMatrix>& riDest, double factor, int lowerBand,
							int upperBand) const
	{
		cadet_assert(_matrix->columns() >= riDest._matrix->columns());

		const int idxStart = _rowIdx + lowerBand;
		const int idxEnd = _rowIdx + upperBand;

		cadet_assert(idxStart >= 0);
		cadet_assert(idxEnd <= _matrix->columns());

		double* const dest = riDest._pos - riDest._rowIdx;
		double const* const src = _pos - _rowIdx;
		for (int i = idxStart; i < idxEnd; ++i)
			dest[i] += factor * src[i];
	}

	/**
	 * @brief Adds a subset of this row @f$ x @f$ to another row @f$ y @f$ performing @f$ y = y + \alpha x @f$
	 * @details Although the subset is specified with a bandmatrix interface, the rows are
				added as in a normal matrix.
	 * @param [in] riDest Row iterator pointing to the destination row @f$ y @f$
	 * @param [in] factor Factor @f$ \alpha @f$
	 * @param [in] lowerBand Start point of the row subset as lower band index
	 * @param [in] upperBand End point of the row subset as upper band index
	 * @param [in] shift Unused
	 * @tparam OtherMatrix Underlying matrix type of other row iterator
	 */
	template <typename OtherMatrix>
	inline void addSubsetTo(const DenseBandedRowIterator<OtherMatrix>& riDest, double factor, int lowerBand,
							int upperBand, int shift) const
	{
		addSubsetTo(riDest, factor, lowerBand, upperBand);
	}

	/**
	 * @brief Adds the given array to the current row
	 * @details Performs the operation @f$ y = y + \alpha x @f$, where @f$ x @f$ may only be a
	 *          subset of the current row the iterator points to. The start of the subset is
	 *          given by @p startDiag. The subset has to fully fit into the matrix row.
	 * @param [in] row Pointer to array @f$ x @f$ that is added to the given row @f$ y @f$
	 * @param [in] startDiag Index of the diagonal at which the row is added
	 * @param [in] length Length of the array
	 * @param [in] factor Factor @f$ \alpha @f$
	 */
	inline void addArray(double const* row, int startDiag, int length, double factor)
	{
		cadet_assert(startDiag + _rowIdx >= 0);
		cadet_assert(startDiag + length + _rowIdx <= static_cast<int>(_matrix->columns()));
		double* const dest = _pos + startDiag;
		for (int i = 0; i < length; ++i)
			dest[i] += factor * row[i];
	}

	inline DenseBandedRowIterator& operator++() CADET_NOEXCEPT
	{
		// Add one additional shift to stay on the main diagonal
		_pos += _matrix->stride() + 1;
		++_rowIdx;
		return *this;
	}

	inline DenseBandedRowIterator& operator--() CADET_NOEXCEPT
	{
		// Subtract one additional shift to stay on the main diagonal
		_pos -= _matrix->stride() + 1;
		--_rowIdx;
		return *this;
	}

	inline DenseBandedRowIterator& operator+=(int idx) CADET_NOEXCEPT
	{
		// Add additional shifts to stay on the main diagonal
		_pos += idx * _matrix->stride() + idx;
		_rowIdx += idx;
		return *this;
	}

	inline DenseBandedRowIterator& operator-=(int idx) CADET_NOEXCEPT
	{
		// Subtract additional shifts to stay on the main diagonal
		_pos -= idx * _matrix->stride() + idx;
		_rowIdx -= idx;
		return *this;
	}

	inline DenseBandedRowIterator operator+(int op) const CADET_NOEXCEPT
	{
		return DenseBandedRowIterator(*this, op);
	}

	inline friend DenseBandedRowIterator operator+(int op, const DenseBandedRowIterator& it) CADET_NOEXCEPT
	{
		return DenseBandedRowIterator(it, op);
	}

	inline DenseBandedRowIterator operator-(int op) const CADET_NOEXCEPT
	{
		return DenseBandedRowIterator(*this, -op);
	}

	inline friend DenseBandedRowIterator operator-(int op, const DenseBandedRowIterator& it) CADET_NOEXCEPT
	{
		return DenseBandedRowIterator(it, -op);
	}

	inline const MatrixType& matrix() const CADET_NOEXCEPT
	{
		return _matrix;
	}

private:
	MatrixType* _matrix;
	double* _pos;
	double _dummy;
	int _rowIdx;
};

/**
 * @brief Represents a dense matrix base class providing common functionality (e.g., factorization)
 * @details LAPACK uses column-major storage, whereas this class uses row-major.
 *          Thus, what we call a row here is actually a column for LAPACK.
 *          Concluding, we have to use the transposed LAPACK operations for
 *          solution and matrix-vector multiplication. The ordering is irrelevant
 *          for the factorization.
 *
 *          Because of the transposition induced by the differing ordering,
 *          the number of columns and rows switches (e.g., columns are transposed
 *          rows). The ordering of the elements inside one (original) row is maintained
 *          (i.e., the first element in a row becomes the first element in a column
 *          and the last element in a row transposes to the last element in a column).
 *
 *          LAPACK needs additional pivoting arrays for factorization, which are
 *          also stored in this class.
 */
class DenseMatrixBase
{
public:
	typedef DenseBandedRowIterator<DenseMatrixBase> RowIterator;
	friend class DenseBandedRowIterator<DenseMatrixBase>;

	~DenseMatrixBase() CADET_NOEXCEPT
	{
	}

	/**
	 * @brief Sets all matrix elements to the given value
	 * @param [in] val Value all matrix elements are set to
	 */
	inline void setAll(double val)
	{
		std::fill(_data, _data + stride() * _rows, val);
	}

	/**
	 * @brief Accesses an element in a diagonal of the matrix where the main diagonal has index @c 0
	 * @details The @p diagonal determines the element in a row. A negative index indicates a lower diagonal,
	 *          while a positive index indicates an upper diagonal. If @p diagonal is @c 0, then the main
	 *          diagonal of the matrix is retrieved.
	 *
	 * @param [in] row Index of the row
	 * @param [in] diagonal Index of the diagonal
	 *
	 * @return Matrix element at the given position
	 */
	inline double& diagonalElement(int row, int diagonal)
	{
		return (*this)(row, diagonal);
	}
	inline double diagonalElement(int row, int diagonal) const
	{
		return (*this)(row, diagonal);
	}

	/**
	 * @brief Accesses an element in the matrix
	 * @details In contrast to diagonalElement(), the indices do not refer to diagonals but to columns of the matrix.
	 *
	 * @param [in] row Index of the row
	 * @param [in] col Index of the column (from @c 0 to @c columns)
	 *
	 * @return Matrix element at the given position
	 */
	inline double& native(int row, int col)
	{
		cadet_assert(row < _rows);
		cadet_assert(col < _cols);
		return _data[row * stride() + col];
	}

	inline double native(int row, int col) const
	{
		cadet_assert(row < _rows);
		cadet_assert(col < _cols);
		return _data[row * stride() + col];
	}

	inline double& operator()(int row, int diagonal)
	{
		cadet_assert(row < _rows);
		cadet_assert(diagonal < static_cast<int>(_cols - row));
		cadet_assert(-diagonal <= static_cast<int>(row));
		return _data[static_cast<int>(row * stride() + row) + diagonal];
	}

	inline double operator()(int row, int diagonal) const
	{
		cadet_assert(row < _rows);
		cadet_assert(diagonal < static_cast<int>(_cols - row));
		cadet_assert(-diagonal <= static_cast<int>(row));
		return _data[static_cast<int>(row * stride() + row) + diagonal];
	}

	/**
	 * @brief Returns the number of elements in the matrix
	 * @return Number of elements in the matrix
	 */
	inline int elements() const CADET_NOEXCEPT
	{
		return _cols * _rows;
	}

	/**
	 * @brief Returns the number of columns
	 * @return Number of columns
	 */
	inline int columns() const CADET_NOEXCEPT
	{
		return _cols;
	}

	/**
	 * @brief Returns the number of rows
	 * @return Number of rows
	 */
	inline int rows() const CADET_NOEXCEPT
	{
		return _rows;
	}

	/**
	 * @brief Provides direct access to the underlying memory
	 * @return Pointer to the first element of the underlying array
	 */
	inline double* data() CADET_NOEXCEPT
	{
		return _data;
	}
	inline double const* data() const CADET_NOEXCEPT
	{
		return _data;
	}

	/**
	 * @brief Provides direct access to the underlying memory
	 * @return Pointer to the first element of the underlying array
	 */
	inline lapackInt_t* pivotData() CADET_NOEXCEPT
	{
		return _pivot;
	}
	inline lapackInt_t const* pivotData() const CADET_NOEXCEPT
	{
		return _pivot;
	}

	/**
	 * @brief Returns the number of elements in an array row
	 * @return Number of elements in a matrix row
	 */
	inline int stride() const CADET_NOEXCEPT
	{
		return _cols;
	}

	/**
	 * @brief Provides access to the underlying data in the given row
	 * @param [in] idx Index of the row
	 * @return Pointer to first element in the given row
	 */
	inline double* rowPtr(int idx)
	{
		cadet_assert(idx < _rows);
		return _data + stride() * idx;
	}

	/**
	 * @brief Creates a RowIterator pointing to the given row
	 * @param [in] idx Index of the row
	 * @return RowIterator pointing to the given row
	 */
	inline RowIterator row(int idx)
	{
		cadet_assert(idx < _rows);
		return RowIterator(*this, idx);
	}

	/**
	 * @brief Creates a RowIterator pointing to the given row
	 * @param [in] idx Index of the row
	 * @param [in] diag Index of the diagonal the iterator points to
	 * @return RowIterator pointing to the given row
	 */
	inline RowIterator row(int idx, int diag)
	{
		cadet_assert(idx < _rows);
		cadet_assert(static_cast<int>(idx) + diag < _cols);
		cadet_assert(static_cast<int>(idx) + diag >= 0);
		return RowIterator(*this, idx, diag);
	}

	/**
	 * @brief Copies a submatrix of a given banded matrix into this matrix
	 * @details Copies a rectangular submatrix from a given source in banded storage into this matrix.
	 *          The top left matrix element is given by @p startRow and @p startDiag. The size of the
	 *          submatrix is specified by @p numRows and @p numCols.
	 * @param [in] mat Source matrix in banded storage format
	 * @param [in] startRow Index of the first row of the submatrix in the source
	 * @param [in] startDiag Diagonal of the top left element in the source submatrix
	 * @param [in] numRows Number of rows to be copied
	 * @param [in] numCols Number of columns to be copied
	 * @tparam MatType_t Type of a matrix in banded storage
	 */
	template <typename MatType_t>
	inline void copySubmatrixFromBanded(const MatType_t& mat, const int startRow, const int startDiag,
										const int numRows, const int numCols)
	{
		cadet_assert(_rows >= numRows);
		cadet_assert(_cols >= numCols);
		cadet_assert(mat.rows() > startRow);
		cadet_assert(mat.rows() >= startRow + numRows);
		// @todo Add more asserts

		const int upperBand = static_cast<int>(mat.upperBandwidth());
		const int lowerBand = static_cast<int>(mat.lowerBandwidth());

		double* ptrDest = _data;
		for (int r = 0; r < numRows; ++r)
		{
			for (int c = 0; c < numCols; ++c, ++ptrDest)
			{
				*ptrDest = 0.0;

				// Compute diagonal index of current position and shift it by startDiag
				const int curDiag = c - r + startDiag;

				if (cadet_likely((curDiag >= -lowerBand) && (curDiag <= upperBand)))
					*ptrDest = mat.centered(r + startRow, curDiag);
			}
		}
	}

	/**
	 * @brief Copies a submatrix of a given dense matrix into this matrix
	 * @details Copies a rectangular submatrix from a given source in dense storage into this matrix.
	 *          The top left matrix element is given by @p startRow and @p startCol. The size of the
	 *          submatrix is specified by @p numRows and @p numCols.
	 *
	 *          The submatrix is copied into the top left corner of this matrix. The rest of the matrix
	 *          is left unchanged.
	 * @param [in] mat Source matrix in dense storage format
	 * @param [in] startRow Index of the first row of the submatrix in the source
	 * @param [in] startCol Index of the first column of the submatrix in the source
	 * @param [in] numRows Number of rows to be copied
	 * @param [in] numCols Number of columns to be copied
	 */
	inline void copySubmatrix(const DenseMatrixBase& mat, const int startRow, const int startCol, const int numRows,
							  const int numCols)
	{
		cadet_assert(_rows >= numRows);
		cadet_assert(_cols >= numCols);
		cadet_assert(mat._rows > startRow);
		cadet_assert(mat._rows >= startRow + numRows);
		cadet_assert(mat._cols > startCol);
		cadet_assert(mat._cols >= startCol + numCols);

		double* ptrDest = _data;
		double const* ptrSrc = mat._data + mat._cols * startRow + startCol;
		for (int r = 0; r < numRows; ++r, ptrDest += _cols, ptrSrc += mat._cols)
		{
			std::copy(ptrSrc, ptrSrc + numCols, ptrDest);
		}
	}

	/**
	 * @brief Adds a submatrix @f$ X @f$ of this matrix to a submatrix @f$ B @f$ of another matrix
	 * @details Performs @f$ Y = Y + \alpha X @f$.
	 * @param [in,out] dest Destination matrix containing @f$ Y @f$
	 * @param [in] factor Factor @f$ \alpha @f$
	 * @param [in] srcRow Index of the first row of the source submatrix
	 * @param [in] srcCol Index of the first column of the source submatrix
	 * @param [in] numRows Number of rows of the submatrix
	 * @param [in] numCols Number of columns of the submatrix
	 * @param [in] destRow Index of the first row of the destination submatrix
	 * @param [in] destCol Index of the first column of the destination submatrix
	 */
	inline void addSubmatrixTo(DenseMatrixBase& dest, const double factor, const int srcRow, const int srcCol,
							   const int numRows, const int numCols, const int destRow, const int destCol)
	{
		cadet_assert(_rows >= srcRow + numRows);
		cadet_assert(_cols >= srcCol + numCols);
		cadet_assert(_rows > srcRow);
		cadet_assert(_cols > srcCol);
		cadet_assert(dest._rows >= destRow + numRows);
		cadet_assert(dest._cols >= destCol + numCols);
		cadet_assert(dest._rows > destRow);
		cadet_assert(dest._cols > destCol);

		double* ptrDest = dest._data + destRow * dest.stride() + destCol;
		double const* ptrSrc = _data + srcRow * stride() + srcCol;
		for (int r = 0; r < numRows; ++r, ptrDest += dest.stride(), ptrSrc += stride())
		{
			for (int c = 0; c < numCols; ++c)
				ptrDest[c] += factor * ptrSrc[c];
		}
	}

	/**
	 * @brief Sets all elements of a submatrix of this matrix to the given value
	 * @details The submatrix is given by its first row and column and its number of rows and columns.
	 * @param [in] val Value the submatrix is set to
	 * @param [in] startRow Index of the first row of the submatrix
	 * @param [in] startCol Index of the first column of the submatrix
	 * @param [in] numRows Number of rows of the submatrix
	 * @param [in] numCols Number of columns of the submatrix
	 */
	void submatrixSetAll(double val, int startRow, int startCol, int numRows, int numCols);

	/**
	 * @brief Copies a given matrix into a submatrix of this matrix
	 * @details The submatrix is given by its first row and column and its number of rows and columns.
	 * @param [in] mat Source matrix to be copied into the submatrix
	 * @param [in] startRow Index of the first row of the submatrix
	 * @param [in] startCol Index of the first column of the submatrix
	 * @param [in] numRows Number of rows of the submatrix
	 * @param [in] numCols Number of columns of the submatrix
	 */
	void submatrixAssign(const DenseMatrixBase& mat, int startRow, int startCol, int numRows, int numCols);

	/**
	 * @brief Copies a row of a given dense matrix into this matrix
	 * @details Copies a row from a given source in dense storage into this matrix.
	 * @param [in] mat Source matrix in dense storage format
	 * @param [in] srcRow Index of the source row to be copied from @p mat
	 * @param [in] destRow Index of the target row
	 */
	inline void copyRowFrom(const DenseMatrixBase& mat, const int srcRow, int destRow)
	{
		cadet_assert(mat._rows > srcRow);
		cadet_assert(_cols >= mat._cols);
		std::copy_n(mat._data + mat._cols * srcRow, mat._cols, _data + _cols * destRow);
	}

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ using LAPACK
	 * @details Computes @f$ y = Ax @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	inline void multiplyVector(const double* const x, double* const y) const
	{
		multiplyVector(x, 1.0, 0.0, y);
	}

	/**
	 * @brief Multiplies a submatrix @f$ A @f$ with a given vector @f$ x @f$ using LAPACK
	 * @details Computes @f$ y = Ax @f$, where @f$ A @f$ is a submatrix of this matrix and @f$ x @f$ is given.
	 *          The submatrix is given by its first row and column and its number of rows and columns.
	 * @param [in] x Vector the submatrix is multiplied with
	 * @param [in] startRow Index of the first row of the submatrix
	 * @param [in] startCol Index of the first column of the submatrix
	 * @param [in] numRows Number of rows of the submatrix
	 * @param [in] numCols Number of columns of the submatrix
	 * @param [out] y Result of the submatrix-vector multiplication
	 */
	inline void submatrixMultiplyVector(const double* const x, int startRow, int startCol, int numRows, int numCols,
										double* const y) const
	{
		submatrixMultiplyVector(x, startRow, startCol, numRows, numCols, 1.0, 0.0, y);
	}

	/**
	 * @brief Multiplies the transpose of the matrix @f$ A @f$ with a given vector @f$ x @f$ using LAPACK
	 * @details Computes @f$ y = A^Tx @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	inline void transposedMultiplyVector(const double* const x, double* const y) const
	{
		transposedMultiplyVector(x, 1.0, 0.0, y);
	}

	/**
	 * @brief Multiplies the transpose of a submatrix @f$ A @f$ with a given vector @f$ x @f$ using LAPACK
	 * @details Computes @f$ y = A^Tx @f$, where @f$ A @f$ is a submatrix of this matrix and @f$ x @f$ is given.
	 *          The submatrix is given by its first row and column and its number of rows and columns.
	 * @param [in] x Vector the submatrix is multiplied with
	 * @param [in] startRow Index of the first row of the submatrix
	 * @param [in] startCol Index of the first column of the submatrix
	 * @param [in] numRows Number of rows of the submatrix
	 * @param [in] numCols Number of columns of the submatrix
	 * @param [out] y Result of the submatrix-vector multiplication
	 */
	inline void transposedSubmatrixMultiplyVector(const double* const x, int startRow, int startCol, int numRows,
												  int numCols, double* const y) const
	{
		transposedSubmatrixMultiplyVector(x, startRow, startCol, numRows, numCols, 1.0, 0.0, y);
	}

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another vector using LAPACK
	 * @details Computes @f$ y = \alpha Ax + \beta y @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	void multiplyVector(const double* const x, double alpha, double beta, double* const y) const;

	/**
	 * @brief Multiplies a submatrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another vector using LAPACK
	 * @details Computes @f$ y = \alpha Ax + \beta y @f$, where @f$ A @f$ is a submatrix of this matrix and @f$ x @f$ is
	 * given. The submatrix is given by its first row and column and its number of rows and columns.
	 * @param [in] x Vector the submatrix is multiplied with
	 * @param [in] startRow Index of the first row of the submatrix
	 * @param [in] startCol Index of the first column of the submatrix
	 * @param [in] numRows Number of rows of the submatrix
	 * @param [in] numCols Number of columns of the submatrix
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [out] y Result of the submatrix-vector multiplication
	 */
	void submatrixMultiplyVector(const double* const x, int startRow, int startCol, int numRows, int numCols,
								 double alpha, double beta, double* const y) const;

	/**
	 * @brief Multiplies the transpose of the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another
	 * vector using LAPACK
	 * @details Computes @f$ y = \alpha A^T x + \beta y @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ A^T x @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	void transposedMultiplyVector(const double* const x, double alpha, double beta, double* const y) const;

	/**
	 * @brief Multiplies the transpose of a submatrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another
	 * vector using LAPACK
	 * @details Computes @f$ y = \alpha A^T x + \beta y @f$, where @f$ A @f$ is a submatrix of this matrix and @f$ x @f$
	 * is given. The submatrix is given by its first row and column and its number of rows and columns.
	 * @param [in] x Vector the submatrix is multiplied with
	 * @param [in] startRow Index of the first row of the submatrix
	 * @param [in] startCol Index of the first column of the submatrix
	 * @param [in] numRows Number of rows of the submatrix
	 * @param [in] numCols Number of columns of the submatrix
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ A^T x @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [out] y Result of the submatrix-vector multiplication
	 */
	void transposedSubmatrixMultiplyVector(const double* const x, int startRow, int startCol, int numRows, int numCols,
										   double alpha, double beta, double* const y) const;

	/**
	 * @brief Factorizes the matrix using LAPACK (performs LU factorization)
	 * @details The original matrix is overwritten with the factorization and all data is lost.
	 * @return @c true if the factorization was successful, otherwise @c false
	 */
	bool factorize();

	/**
	 * @brief Uses the factorized matrix to solve the equation @f$ Ax = y @f$ with LAPACK
	 * @details Before the equation can be solved, the matrix has to be factorized first by calling factorize().
	 * @param [in,out] rhs On entry pointer to the right hand side vector @f$ y @f$ of the equation, on exit the
	 * solution @f$ x @f$
	 * @return @c true if the solution process was successful, otherwise @c false
	 */
	bool solve(double* rhs) const;

	/**
	 * @brief Uses the factorized matrix to solve the equation @f$ Ax = y @f$ with LAPACK
	 * @details Before the equation can be solved, the matrix has to be factorized first by calling factorize().
	 *          It is assumed that row scaling has been applied to the matrix before factorization.
	 *          In order to solve the equation system, the right hand side has to be scaled accordingly.
	 *          This is handled automatically by passing the required scaling factors.
	 * @param [in] scalingFactors Vector with scaling factor for each row
	 * @param [in,out] rhs On entry pointer to the right hand side vector @f$ y @f$ of the equation, on exit the
	 * solution @f$ x @f$
	 * @return @c true if the solution process was successful, otherwise @c false
	 */
	bool solve(double const* scalingFactors, double* rhs) const;

	/**
	 * @brief Returns the optimal working memory size for solving @f$ \text{min}_x \lVert Ax - y \rVert @f$ with LAPACK
	 * @details LAPACK requires at least @f$ 2 mn @f$ doubles, where @f$ m @f$ is the number of rows and @f$ n @f$ the
	 * number of columns.
	 * @return The optimal number of doubles in the working memory
	 */
	int optimalLeastSquaresWorkspace() const;

	/**
	 * @brief Solves the linear least squares problem @f$ \text{min}_x \lVert Ax - y \rVert @f$ using LAPACK (QR
	 * decomposition)
	 * @details The least squares system is solved using a QR decomposition performed by LAPACK. The matrix is
	 *          overwritten with the factorization and the original data is lost. It is assumed that @f$ A @f$
	 *          has full rank, otherwise the solution will fail. The optimal amount of working memory can be
	 *          computed by optimalLeastSquaresWorkspace().
	 * @param [in,out] rhs On entry pointer to the right hand side vector @f$ y @f$ of the equation,
	 *                 on exit the solution @f$ x @f$ in the first components
	 * @param [in,out] workspace Pointer to workspace required by LAPACK to perform the factorization
	 *                 (at least @f$ 2 \min\{m,n\} @f$ for an @f$ m \times n @f$ matrix)
	 * @param [in] size Size of the provided workspace
	 * @return @c true if the solution process was successful, otherwise @c false
	 */
	bool leastSquaresSolve(double* rhs, double* workspace, int size);

	/**
	 * @brief Performs a numerically robust (but slower and memory consuming) factorization of the matrix using LAPACK
	 * (QR factorization)
	 * @details The original matrix is overwritten with the factorization and all data is lost.
	 * @param [in,out] workspace Working memory which on exit contains the scalar factors of the Householder reflectors
	 * (size @f$ 2n @f$ for an @f$ n \times n@f$ matrix).
	 * @return @c true if the factorization was successful, otherwise @c false
	 */
	bool robustFactorize(double* const workspace);

	/**
	 * @brief Uses the factorized matrix to solve the equation @f$ Ax = y @f$ with LAPACK
	 * @details Before the equation can be solved, the matrix has to be factorized first by calling robustFactorize().
	 * @param [in,out] rhs On entry pointer to the right hand side vector @f$ y @f$ of the equation, on exit the
	 * solution @f$ x @f$
	 * @param [in,out] workspace Workspace used by robustFactorize() (i.e., it has to contain the scalar factors of the
	 * Householder reflectors). For an @f$ n \times n@f$ matrix, an array of size @f$ 2n @f$ is required. Note that if
	 * the equation system is singular, the least squares solution @f$ x @f$ satisfying @f$ \text{min}_x \lVert Ax - y
	 * \rVert @f$ is returned.
	 * @return @c true if the solution process was successful, otherwise @c false
	 */
	bool robustSolve(double* rhs, double* const workspace) const;

	/**
	 * @brief Uses the factorized matrix to solve the equation @f$ Ax = y @f$ with LAPACK
	 * @details Before the equation can be solved, the matrix has to be factorized first by calling robustFactorize().
	 *          It is assumed that row scaling has been applied to the matrix before factorization.
	 *          In order to solve the equation system, the right hand side has to be scaled accordingly.
	 *          This is handled automatically by passing the required scaling factors.
	 * @param [in] scalingFactors Vector with scaling factor for each row
	 * @param [in,out] rhs On entry pointer to the right hand side vector @f$ y @f$ of the equation, on exit the
	 * solution @f$ x @f$
	 * @param [in,out] workspace Workspace used by robustFactorize() (i.e., it has to contain the scalar factors of the
	 * Householder reflectors). For an @f$ n \times n@f$ matrix, an array of size @f$ 2n @f$ is required. Note that if
	 * the equation system is singular, the least squares solution @f$ x @f$ satisfying @f$ \text{min}_x \lVert Ax - y
	 * \rVert @f$ is returned.
	 * @return @c true if the solution process was successful, otherwise @c false
	 */
	bool robustSolve(double const* scalingFactors, double* rhs, double* const workspace) const;

	/**
	 * @brief Returns the size of the workspace required for calling robustFactorize()
	 * @details The workspace size is @f$ 2n @f$ for an @f$ n \times n@f$ matrix.
	 * @return Size of the workspace required for robustFactorize()
	 */
	inline int robustWorkspaceSize() const
	{
		return 2 * rows();
	}

	/**
	 * @brief Scales rows by dividing them with a given factor
	 * @details This corresponds to multiplying by a diagonal matrix from the left (i.e., @f$ D^{-1} A @f$).
	 * @param [in] scalingFactors Vector with scaling factor for each row
	 */
	inline void scaleRows(double const* scalingFactors)
	{
		scaleRows(scalingFactors, rows());
	}

	/**
	 * @brief Scales the first rows by dividing them with a given factor
	 * @details This corresponds to multiplying by a diagonal matrix from the left (i.e., @f$ D^{-1} A @f$).
	 *          Only the first @p numRows rows are scaled.
	 * @param [in] scalingFactors Vector with scaling factor for each row
	 * @param [in] numRows The number of rows which are to be scaled (from the top)
	 */
	void scaleRows(double const* scalingFactors, int numRows);

	/**
	 * @brief Scales columns by dividing them with a given factor
	 * @details This corresponds to multiplying by a diagonal matrix from the right (i.e., @f$ A D^{-1} @f$).
	 * @param [in] scalingFactors Vector with scaling factor for each column
	 */
	inline void scaleColumns(double const* scalingFactors)
	{
		scaleColumns(scalingFactors, columns());
	}

	/**
	 * @brief Scales the first columns by dividing them with a given factor
	 * @details This corresponds to multiplying by a diagonal matrix from the right (i.e., @f$ A D^{-1} @f$).
	 *          Only the first @p numCols columns are scaled.
	 * @param [in] scalingFactors Vector with scaling factor for each column
	 * @param [in] numCols The number of columns which are to be scaled (from the left)
	 */
	void scaleColumns(double const* scalingFactors, int numCols);

	/**
	 * @brief Computes row scaling factors such that the largest absolute value in each row is 1
	 * @details This can be used to equilibrate the matrix by calling scaleRows().
	 * @param [out] scalingFactors Vector in which the row scaling factors are written
	 */
	inline void rowScaleFactors(double* scalingFactors) const
	{
		rowScaleFactors(scalingFactors, rows());
	}

	/**
	 * @brief Computes row scaling factors for some rows such that the largest absolute value in each row is 1
	 * @details This can be used to equilibrate the matrix by calling scaleRows().
	 *          The scaling factors are computed for the first @p numRows rows from the top.
	 * @param [out] scalingFactors Vector in which the row scaling factors are written
	 * @param [in] numRows The number of rows to be scaled
	 */
	void rowScaleFactors(double* scalingFactors, int numRows) const;

	/**
	 * @brief Computes column scaling factors such that the largest absolute value in each column is 1
	 * @details This can be used to equilibrate the matrix by calling scaleColumns().
	 * @param [out] scalingFactors Vector in which the column scaling factors are written
	 */
	inline void columnScaleFactors(double* scalingFactors) const
	{
		columnScaleFactors(scalingFactors, columns());
	}

	/**
	 * @brief Computes column scaling factors for some columns such that the largest absolute value in each column is 1
	 * @details This can be used to equilibrate the matrix by calling scaleColumns().
	 *          The scaling factors are computed for the first @p numCols columns from the left.
	 * @param [out] scalingFactors Vector in which the column scaling factors are written
	 * @param [in] numCols The number of columns to be scaled
	 */
	void columnScaleFactors(double* scalingFactors, int numCols) const;

	/**
	 * @brief Resizes the matrix to the given size without deleting its content
	 * @details The matrix size is not checked and no memory is allocated or deleted.
	 *          The user has to take care that the memory is big enough to hold all
	 *          elements (beware of access violations).
	 * @param [in] rows The number of rows
	 * @param [in] cols The number of columns
	 */
	inline void shrinkOrExpandFast(int rows, int cols)
	{
		_cols = cols;
		_rows = rows;
	}

	/**
	 * @brief Resizes the matrix to the given size
	 * @details The matrix size is not checked and no memory is allocated or deleted.
	 *          The user has to take care that the memory is big enough to hold all
	 *          elements (beware of access violations).
	 *          All data is lost during this operation, the selected area is zeroed.
	 *
	 * @param [in] rows The number of rows
	 * @param [in] cols The number of columns
	 */
	inline void shrinkOrExpand(int rows, int cols)
	{
		shrinkOrExpandFast(rows, cols);
		setAll(0.0);
	}

protected:
	double* _data;       //!< Pointer to the array in which the matrix is stored
	int _rows;           //!< Number of rows
	int _cols;           //!< Number of columns
	lapackInt_t* _pivot; //!< Pointer to an array which is used for pivoting by factorization methods

	/**
	 * @brief Creates an empty, unitialized matrix
	 * @details No memory is allocated for the matrix. Users have to call resize() first.
	 */
	DenseMatrixBase() CADET_NOEXCEPT : _data(nullptr), _rows(0), _cols(0), _pivot(nullptr)
	{
	}

	/**
	 * @brief Initializes the matrix with the given memory of the given size
	 * @param [in] data Pointer to data array of size `rows * cols`
	 * @param [in] pivot Pointer to pivot array of size at least `min(rows, cols)`
	 * @param [in] rows Number of rows
	 * @param [in] cols Number of columns
	 */
	DenseMatrixBase(double* const data, lapackInt_t* const pivot, int rows, int cols) CADET_NOEXCEPT : _data(data),
																									   _rows(rows),
																									   _cols(cols),
																									   _pivot(pivot)
	{
	}

	/**
	 * @brief Copies all values from the source to the local array
	 * @param [in] src Data to be copied
	 */
	inline void copyValues(double const* const src)
	{
		std::copy(src, src + stride() * _rows, _data);
	}

	/**
	 * @brief Copies all pivots from the source to the local array
	 * @param [in] src Data to be copied
	 */
	inline void copyPivot(lapackInt_t const* const src)
	{
		std::copy(src, src + _cols, _pivot);
	}
};

std::ostream& operator<<(std::ostream& out, const DenseMatrixBase& bm);

} // namespace detail

typedef detail::DenseMatrixBase::RowIterator DenseBandedRowIterator;

/**
 * @brief Represents a dense matrix that is factorizable if it is square
 * @details LAPACK uses column-major storage, whereas this class uses row-major.
 *          Thus, what we call a row here is actually a column for LAPACK.
 *          Concluding, we have to use the transposed LAPACK operations for
 *          solution and matrix-vector multiplication. The ordering is irrelevant
 *          for the factorization.
 *
 *          Because of the transposition induced by the differing ordering,
 *          the number of columns and rows switches (e.g., columns are transposed
 *          rows). The ordering of the elements inside one (original) row is maintained
 *          (i.e., the first element in a row becomes the first element in a column
 *          and the last element in a row transposes to the last element in a column).
 *
 *          LAPACK needs additional pivoting arrays for factorization, which are
 *          also stored in this class.
 */
class DenseMatrix : public detail::DenseMatrixBase
{
public:
	/**
	 * @brief Creates an empty, unitialized matrix
	 * @details No memory is allocated for the matrix. Users have to call resize() first.
	 */
	DenseMatrix() CADET_NOEXCEPT
	{
	}
	~DenseMatrix() CADET_NOEXCEPT
	{
		delete[] _pivot;
		delete[] _data;
	}

	DenseMatrix(const DenseMatrix& cpy)
		: DenseMatrixBase(new double[cpy.stride() * cpy._rows], new lapackInt_t[std::min(cpy._rows, cpy._cols)],
						  cpy._rows, cpy._cols)
	{
		copyValues(cpy._data);
		copyPivot(cpy._pivot);
	}

	DenseMatrix(DenseMatrix&& cpy) CADET_NOEXCEPT : DenseMatrixBase(cpy._data, cpy._pivot, cpy._rows, cpy._cols)
	{
		cpy._data = nullptr;
		cpy._pivot = nullptr;
	}

	inline DenseMatrix& operator=(const DenseMatrix& cpy)
	{
		cadet_assert(&cpy != this);

		// Determine if reallocation is necessary
		const bool reAllocMat = (cpy.stride() * cpy._rows) > stride() * _rows;
		const bool reAllocPivot = std::min(cpy._rows, cpy._cols) > std::min(_rows, _cols);

		// Change size and copy all the stuff
		_cols = cpy._cols;
		_rows = cpy._rows;

		if (reAllocMat)
		{
			delete[] _data;
			_data = new double[cpy.stride() * cpy._rows];
		}
		copyValues(cpy._data);

		if (reAllocPivot)
		{
			delete[] _pivot;
			_pivot = new lapackInt_t[std::min(_rows, _cols)];
		}
		copyPivot(cpy._pivot);

		return *this;
	}

	inline DenseMatrix& operator=(DenseMatrix&& cpy) CADET_NOEXCEPT
	{
		_cols = cpy._cols;
		_rows = cpy._rows;

		delete[] _data;
		_data = cpy._data;
		cpy._data = nullptr;

		delete[] _pivot;
		_pivot = cpy._pivot;
		cpy._pivot = nullptr;

		return *this;
	}

	/**
	 * @brief Copies the given dense matrix @p cpy into this one
	 * @details It is assumed that the matrix has enough memory to hold the given copy @p cpy.
	 *          All current data is lost in this operation.
	 *          The matrix is resized to match the size of @p cpy.
	 *
	 * @param cpy Matrix to be copied
	 */
	inline void copyFrom(const DenseMatrixBase& cpy)
	{
		cadet_assert(&cpy != this);

		cadet_assert(stride() * _rows >= cpy.stride() * cpy.rows());
		cadet_assert(std::min(_rows, _cols) >= std::min(cpy.rows(), cpy.columns()));

		_rows = cpy.rows();
		_cols = cpy.columns();

		copyValues(cpy.data());
		copyPivot(cpy.pivotData());
	}

	/**
	 * @brief Resizes the matrix to the given size
	 * @details All data is lost in this operation.
	 *
	 * @param [in] rows The number of rows
	 * @param [in] cols The number of columns
	 */
	inline void resize(int rows, int cols)
	{
		_cols = cols;
		_rows = rows;

		delete[] _data;
		_data = new double[stride() * _rows];

		delete[] _pivot;
		_pivot = new lapackInt_t[std::min(_rows, _cols)];

		setAll(0.0);
	}
};

/**
 * @brief An interface to a general rectangular dense matrix that is factorizable (if square) with provided memory.
 * @details LAPACK uses column-major storage, whereas this class uses row-major.
 *          Thus, what we call a row here is actually a column for LAPACK.
 *          Concluding, we have to use the transposed LAPACK operations for
 *          solution and matrix-vector multiplication. The ordering is irrelevant
 *          for the factorization.
 *
 *          Because of the transposition induced by the differing ordering,
 *          the number of columns and rows switches (e.g., columns are transposed
 *          rows). The ordering of the elements inside one (original) row is maintained
 *          (i.e., the first element in a row becomes the first element in a column
 *          and the last element in a row transposes to the last element in a column).
 */
class DenseMatrixView : public detail::DenseMatrixBase
{
public:
	/**
	 * @brief Initializes the view with the given memory of the given size
	 * @param [in] data Pointer to data array of size `rows * cols`
	 * @param [in] pivot Pointer to pivot array of size at least `min(rows, cols)`
	 * @param [in] rows Number of rows
	 * @param [in] cols Number of columns
	 */
	DenseMatrixView(double* const data, lapackInt_t* const pivot, int rows, int cols) CADET_NOEXCEPT
		: DenseMatrixBase(data, pivot, rows, cols)
	{
	}

	/**
	 * @brief Initializes the view as a submatrix of the given DenseMatrix
	 * @param [in] mat Source matrix
	 * @param [in] startRow Index of the first row of this submatrix
	 * @param [in] rows Number of rows
	 */
	DenseMatrixView(detail::DenseMatrixBase& mat, int startRow, int rows) CADET_NOEXCEPT
		: DenseMatrixBase(mat.rowPtr(startRow), mat.pivotData(), rows, mat.columns())
	{
		cadet_assert(mat.rows() >= startRow + rows);
	}

	~DenseMatrixView() CADET_NOEXCEPT
	{
	}

	DenseMatrixView(const DenseMatrixView& cpy) CADET_NOEXCEPT
		: DenseMatrixBase(cpy._data, cpy._pivot, cpy._rows, cpy._cols)
	{
	}

	DenseMatrixView(DenseMatrixView&& cpy) CADET_NOEXCEPT : DenseMatrixBase(cpy._data, cpy._pivot, cpy._rows, cpy._cols)
	{
		cpy._data = nullptr;
		cpy._pivot = nullptr;
	}

	inline DenseMatrixView& operator=(const DenseMatrixView& cpy)
	{
		cadet_assert(&cpy != this);

		// We do not own the memory and, thus, can only decrease matrix size
		cadet_assert(_cols * _rows >= cpy._cols * cpy._rows);
		cadet_assert(std::min(_rows, _cols) >= std::min(cpy._rows, cpy._cols));

		_cols = cpy._cols;
		_rows = cpy._rows;
		copyValues(cpy._data);
		copyPivot(cpy._pivot);

		return *this;
	}

	inline DenseMatrixView& operator=(DenseMatrixView&& cpy) CADET_NOEXCEPT
	{
		_cols = cpy._cols;
		_rows = cpy._rows;

		_data = cpy._data;
		cpy._data = nullptr;

		_pivot = cpy._pivot;
		cpy._pivot = nullptr;

		return *this;
	}

	/**
	 * @brief Resizes the matrix to the given size without deleting its content
	 * @details The matrix size is not checked. The user has to take care that the memory
	 *          is big enough to hold all elements (beware of access violations).
	 * @param [in] rows The number of rows
	 * @param [in] cols The number of columns
	 */
	inline void resizeFast(int rows, int cols)
	{
		_cols = cols;
		_rows = rows;
	}

	/**
	 * @brief Resizes the matrix to the given size
	 * @details The matrix size is not checked. The user has to take care that the memory
	 *          is big enough to hold all elements (beware of access violations).
	 *          All data is lost during this operation.
	 *
	 * @param [in] rows The number of rows
	 * @param [in] cols The number of columns
	 */
	inline void resize(int rows, int cols)
	{
		resizeFast(rows, cols);
		setAll(0.0);
	}
};

} // namespace linalg

} // namespace cadet

#endif // LIBCADET_DENSEMATRIX_HPP_
