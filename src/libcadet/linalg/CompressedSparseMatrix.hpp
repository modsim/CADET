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
 * Defines a compressed sparse matrix
 */

#ifndef LIBCADET_COMPRESSEDSPARSEMATRIX_HPP_
#define LIBCADET_COMPRESSEDSPARSEMATRIX_HPP_

#include <vector>
#include <ostream>
#include <algorithm>

#include "SparseSolverInterface.hpp"
#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"

namespace cadet
{

namespace linalg
{

class SparsityPatternRowIterator;

/**
 * @brief Sparsity pattern that grows dynamically and can be turned into compressed row format
 */
class SparsityPattern
{
public:
	/**
	 * @brief Creates an empty SparsityPattern with preallocated memory
	 * @details Uses an estimate of the average number of non-zero entries per row to
	 *          preallocate memory.
	 * @param [in] nRows Number of rows in the square matrix
	 * @param [in] averageNumNonZeros Average number of non-zero entries per row
	 */
	SparsityPattern(int nRows, int averageNumNonZeros)
		: _rows(nRows, MatrixRowPattern(averageNumNonZeros)), _numNonZero(0)
	{
	}

	~SparsityPattern() CADET_NOEXCEPT
	{
	}

	// Default copy and assignment semantics
	SparsityPattern(const SparsityPattern& cpy) = default;
	SparsityPattern(SparsityPattern&& cpy) CADET_NOEXCEPT = default;

	SparsityPattern& operator=(const SparsityPattern& cpy) = default;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	SparsityPattern& operator=(SparsityPattern&& cpy) CADET_NOEXCEPT = default;
#else
	SparsityPattern& operator=(SparsityPattern&& cpy) = default;
#endif

	/**
	 * @brief Returns the number of non-zero entries in this pattern
	 * @return Number of non-zero entries in the matrix
	 */
	inline int numNonZeros() const CADET_NOEXCEPT
	{
		return _numNonZero;
	}

	/**
	 * @brief Returns the number of rows (or columns) of this square matrix
	 * @details It does not matter whether some rows are fully empty.
	 * @return Number of rows (or columns) of this square matrix
	 */
	inline int rows() const CADET_NOEXCEPT
	{
		return _rows.size();
	}

	/**
	 * @brief Adds an entry to the pattern
	 * @param row Index of the row
	 * @param col Index of the column
	 */
	inline void add(int row, int col)
	{
		cadet_assert((row >= 0) && (row < static_cast<int>(_rows.size())));
		cadet_assert((col >= 0) && (col < static_cast<int>(_rows.size())));

		if (_rows[row].add(col))
			++_numNonZero;
	}

	/**
	 * @brief Checks whether a given entry is non-zero in the pattern
	 * @param row Index of the row
	 * @param col Index of the column
	 * @return @c true if the entry exists, @c false otherwise
	 */
	inline bool isNonZero(int row, int col) const CADET_NOEXCEPT
	{
		cadet_assert((row >= 0) && (row < static_cast<int>(_rows.size())));
		cadet_assert((col >= 0) && (col < static_cast<int>(_rows.size())));

		return _rows[row].contains(col);
	}

	/**
	 * @brief Compresses the pattern into compressed row storage format
	 * @details Assumes that the memory for the new storage is already allocated.
	 * @param [out] colIdx Array with column indices of size numNonZeros()
	 * @param [out] rowStart Array with row start indices of size rows()+1
	 */
	void compressTo(sparse_int_t* colIdx, sparse_int_t* rowStart) const CADET_NOEXCEPT;

	/**
	 * @brief Returns a row iterator to the specified row
	 * @param [in] idxRow Index of the row
	 * @return SparsityPatternRowIterator pointing to the given row
	 */
	SparsityPatternRowIterator row(int idxRow) CADET_NOEXCEPT;

protected:
	/**
	 * @brief Stores column indices of a matrix row
	 * @details The column indices are always sorted ascendingly.
	 */
	class MatrixRowPattern
	{
	public:
		/**
		 * @brief Creates an empty MatrixRowPattern
		 */
		MatrixRowPattern() CADET_NOEXCEPT : _colIdx(0)
		{
		}

		/**
		 * @brief Creates an empty MatrixRowPattern with preallocated memory
		 * @details Preallocates memory.
		 * @param [in] averageNumNonZeros Average number of non-zero entries in this row
		 */
		MatrixRowPattern(std::size_t averageNumNonZeros) : _colIdx(0)
		{
			_colIdx.reserve(averageNumNonZeros);
		}

		~MatrixRowPattern() CADET_NOEXCEPT
		{
		}

		// Default copy and assignment semantics
		MatrixRowPattern(const MatrixRowPattern& cpy) = default;
		MatrixRowPattern(MatrixRowPattern&& cpy) CADET_NOEXCEPT = default;

		MatrixRowPattern& operator=(const MatrixRowPattern& cpy) = default;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
		MatrixRowPattern& operator=(MatrixRowPattern&& cpy) CADET_NOEXCEPT = default;
#else
		MatrixRowPattern& operator=(MatrixRowPattern&& cpy) = default;
#endif

		/**
		 * @brief Adds a column index to this row
		 * @details Maintains order of the stored column indices.
		 * @param [in] column Column index to store
		 * @return @c true if the entry is new, @c false if it already exists
		 */
		inline bool add(sparse_int_t column)
		{
			// Simply add if the matrix row is still empty
			if (cadet_unlikely(_colIdx.empty()))
			{
				_colIdx.push_back(column);
				return true;
			}

			// Find the position to insert via binary search
			const auto it = std::upper_bound(_colIdx.begin(), _colIdx.end(), column);

			// Check whether this index already exists
			if ((it != _colIdx.begin()) && (*(it - 1) == column))
				return false;

			_colIdx.insert(it, column);
			return true;
		}

		/**
		 * @brief Checkes whether a given column index exists in this row
		 * @param [in] column Column index to check
		 * @return @c true if the entry exists, @c false otherwise
		 */
		inline bool contains(sparse_int_t column) const CADET_NOEXCEPT
		{
			return std::binary_search(_colIdx.begin(), _colIdx.end(), column);
		}

		/**
		 * @brief Returns a sorted array of column indices
		 * @return Sorted array of column indices
		 */
		const std::vector<sparse_int_t>& columnIndices() const CADET_NOEXCEPT
		{
			return _colIdx;
		}

		/**
		 * @brief Returns the number of non-zero entries in this row
		 * @return Number of non-zero entries in this row
		 */
		int numNonZeros() const CADET_NOEXCEPT
		{
			return _colIdx.size();
		}

	protected:
		std::vector<sparse_int_t> _colIdx; //!< Sorted column indices of non-zero elements in this row
	};

	std::vector<MatrixRowPattern> _rows; //!< Rows with their non-zero indices
	int _numNonZero;                     //!< Number of non-zero entries in the matrix
};

/**
 * @brief Iterates over rows of the SparsityPattern
 * @details Inserts elements into the SparsityPattern whenever a write operation
 *          is encountered. Read operations do not add entries to the pattern.
 */
class SparsityPatternRowIterator
{
public:
	/**
	 * @brief Creates an empty SparsityPatternRowIterator pointing to nothing
	 */
	SparsityPatternRowIterator() CADET_NOEXCEPT : _pattern(nullptr), _row(-1)
	{
	}

	/**
	 * @brief Creates a SparsityPatternRowIterator for the given SparsityPattern
	 * @param [in] pattern SparsityPattern of the SparsityPatternRowIterator
	 */
	SparsityPatternRowIterator(SparsityPattern& pattern) CADET_NOEXCEPT : _pattern(&pattern), _row(0)
	{
	}

	/**
	 * @brief Creates a SparsityPatternRowIterator for the given SparsityPattern
	 * @param [in] pattern SparsityPattern of the SparsityPatternRowIterator
	 * @param [in] row Index of the row of the iterator points so
	 */
	SparsityPatternRowIterator(SparsityPattern& pattern, int row) CADET_NOEXCEPT : _pattern(&pattern), _row(row)
	{
	}

	~SparsityPatternRowIterator() CADET_NOEXCEPT
	{
	}

	// Default copy and assignment semantics
	SparsityPatternRowIterator(const SparsityPatternRowIterator& cpy) = default;
	SparsityPatternRowIterator(SparsityPatternRowIterator&& cpy) CADET_NOEXCEPT = default;

	SparsityPatternRowIterator& operator=(const SparsityPatternRowIterator& cpy) = default;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	SparsityPatternRowIterator& operator=(SparsityPatternRowIterator&& cpy) CADET_NOEXCEPT = default;
#else
	SparsityPatternRowIterator& operator=(SparsityPatternRowIterator&& cpy) = default;
#endif

	/**
	 * @brief Sets all matrix elements in the row to the given value
	 * @param [in] val Value all matrix elements in the row are set to
	 */
	inline void setAll(double val)
	{
		cadet_assert(_pattern);

		if (val == 0.0)
			return;

		for (int i = 0; i < _pattern->rows(); ++i)
			_pattern->add(_row, i);
	}

	/**
	 * @brief Accesses an element in the current row where the main diagonal is centered (index @c 0)
	 * @details The @p diagonal determines the element in a row. A negative index indicates a lower diagonal,
	 *          while a positive index indicates an upper diagonal. If @p diagonal is @c 0, then the main
	 *          diagonal is retrieved.
	 *
	 * @param [in] diagonal Index of the diagonal (between negative lower bandwidth and upper bandwidth)
	 *
	 * @return Dummy value or @c 0.0
	 */
	inline double& centered(int diagonal)
	{
		cadet_assert(_pattern);
		cadet_assert((_row + diagonal >= 0) && (_row + diagonal < _pattern->rows()));

		_pattern->add(_row, _row + diagonal);
		return _dummy;
	}
	inline double centered(int diagonal) const
	{
		return 0.0;
	}

	/**
	 * @brief Accesses an element in the current row where the lowest diagonal is indexed by @c 0
	 * @details In contrast to centered() the index of the column starts with the lowest diagonal (@c 0).
	 *          The main diagonal is, thus, retrieved for @c lowerBand and the highest upper diagonal
	 *          is returned for `lowerBand + upperBand`.
	 *
	 * @param [in] col Index of the column (from @c 0 to `lowerBand + upperBand`)
	 *
	 * @return Matrix element at the given position
	 */
	inline double& native(int col)
	{
		cadet_assert(_pattern);
		cadet_assert(col < _pattern->rows());

		_pattern->add(_row, col);
		return _dummy;
	}

	inline double native(int col) const
	{
		return 0.0;
	}

	inline double& operator()(int diagonal)
	{
		cadet_assert(_pattern);
		cadet_assert((_row + diagonal >= 0) && (_row + diagonal < _pattern->rows()));

		_pattern->add(_row, _row + diagonal);
		return _dummy;
	}

	inline double operator()(int diagonal) const
	{
		return 0.0;
	}

	inline double& operator[](int diagonal)
	{
		return (*this)(diagonal);
	}
	inline double operator[](int diagonal) const
	{
		return (*this)(diagonal);
	}

	inline SparsityPatternRowIterator& operator++() CADET_NOEXCEPT
	{
		++_row;
		return *this;
	}

	inline SparsityPatternRowIterator& operator--() CADET_NOEXCEPT
	{
		--_row;
		return *this;
	}

	inline SparsityPatternRowIterator& operator+=(int idx) CADET_NOEXCEPT
	{
		_row += idx;
		return *this;
	}

	inline SparsityPatternRowIterator& operator-=(int idx) CADET_NOEXCEPT
	{
		_row -= idx;
		return *this;
	}

	inline SparsityPatternRowIterator operator+(int op) const CADET_NOEXCEPT
	{
		return SparsityPatternRowIterator(*_pattern, _row + op);
	}

	inline friend SparsityPatternRowIterator operator+(int op, const SparsityPatternRowIterator& it) CADET_NOEXCEPT
	{
		return SparsityPatternRowIterator(*it._pattern, op + it._row);
	}

	inline SparsityPatternRowIterator operator-(int op) const CADET_NOEXCEPT
	{
		return SparsityPatternRowIterator(*_pattern, _row - op);
	}

	inline friend SparsityPatternRowIterator operator-(int op, const SparsityPatternRowIterator& it) CADET_NOEXCEPT
	{
		return SparsityPatternRowIterator(*it._pattern, op - it._row);
	}

	/**
	 * @brief Returns the underlying pattern this iterator is pointing into
	 * @return SparsityPattern this iterator is pointing into
	 */
	inline const SparsityPattern& pattern() const CADET_NOEXCEPT
	{
		return *_pattern;
	}

	/**
	 * @brief Returns the index of the current row
	 * @return Index of the current row
	 */
	inline int row() const CADET_NOEXCEPT
	{
		return _row;
	}

private:
	SparsityPattern* _pattern; //!< Underlying pattern
	int _row;                  //!< Index of the current row
	double _dummy;             //!< Dummy element for access by reference
};

class BandedSparseRowIterator;
class ConstBandedSparseRowIterator;

/**
 * @brief Sparse matrix with compressed row storage
 */
class CompressedSparseMatrix
{
public:
	/**
	 * @brief Creates an empty CompressedSparseMatrix with capacity @c 0
	 * @details Users have to call resize() prior to populating the matrix.
	 */
	CompressedSparseMatrix() CADET_NOEXCEPT : _values(0), _colIdx(0), _rowStart(0), _dummy(0.0)
	{
	}

	/**
	 * @brief Creates an empty CompressedSparseMatrix with the given capacity
	 * @param [in] numRows Matrix size (i.e., number of rows or columns)
	 * @param [in] numNonZeros Maximum number of non-zero elements
	 */
	CompressedSparseMatrix(int numRows, int numNonZeros) : _dummy(0.0)
	{
		resize(numRows, numNonZeros);
	}

	/**
	 * @brief Creates a CompressedSparseMatrix with the given pattern
	 * @param [in] pattern Sparsity pattern
	 */
	CompressedSparseMatrix(const SparsityPattern& pattern) : _values(0), _colIdx(0), _rowStart(0), _dummy(0.0)
	{
		assignPattern(pattern);
	}

	~CompressedSparseMatrix() CADET_NOEXCEPT
	{
	}

	// Default copy and assignment semantics
	CompressedSparseMatrix(const CompressedSparseMatrix& cpy) = default;
	CompressedSparseMatrix(CompressedSparseMatrix&& cpy) CADET_NOEXCEPT = default;

	CompressedSparseMatrix& operator=(const CompressedSparseMatrix& cpy) = default;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	CompressedSparseMatrix& operator=(CompressedSparseMatrix&& cpy) CADET_NOEXCEPT = default;
#else
	CompressedSparseMatrix& operator=(CompressedSparseMatrix&& cpy) = default;
#endif

	/**
	 * @brief Copies a CompressedSparseMatrix having the same pattern
	 * @param [in] src Source matrix to be copied
	 */
	void copyFromSamePattern(const CompressedSparseMatrix& src);

	/**
	 * @brief Copies the values of the given matrix into this matrix
	 * @details The sparsity pattern of the source matrix has to be a subset of the target's pattern.
	 * @param [in] mat Source matrix that is copied
	 */
	void copyFromSubPattern(const CompressedSparseMatrix& mat);

	/**
	 * @brief Copies the values of the given matrix row into a row of this matrix with the same pattern
	 * @details The sparsity pattern of the source matrix row has to match the target row's pattern.
	 * @param [in] mat Source matrix
	 * @param [in] srcRow Index of source row in @p mat
	 * @param [in] destRow Index of destination row in this matrix
	 */
	void copyRowToRowSamePattern(const CompressedSparseMatrix& mat, int srcRow, int destRow);

	/**
	 * @brief Copies the values of the given matrix row into a row of this matrix
	 * @details The sparsity pattern of the source matrix row has to be a subset of the target row's pattern.
	 * @param [in] mat Source matrix
	 * @param [in] srcRow Index of source row in @p mat
	 * @param [in] destRow Index of destination row in this matrix
	 */
	void copyRowToRowSubPattern(const CompressedSparseMatrix& mat, int srcRow, int destRow);

	/**
	 * @brief Allocates memory for the sparse matrix of given size
	 * @details The matrix is reset to an empty state. All previous content is lost.
	 *
	 * @param [in] numRows Matrix size (i.e., number of rows or columns)
	 * @param [in] numNonZeros Maximum number of non-zero elements
	 */
	void resize(int numRows, int numNonZeros);

	/**
	 * @brief Assigns the given sparsity pattern
	 * @details The previous pattern is lost. All matrix entries are lost.
	 * @param [in] pattern Sparsity pattern
	 */
	void assignPattern(const SparsityPattern& pattern);

	/**
	 * @brief Assigns the given sparsity pattern
	 * @details The previous pattern is lost. All matrix entries are lost.
	 * @param [in] pattern Sparsity pattern
	 */
	void assignPattern(const CompressedSparseMatrix& pattern);

	/**
	 * @brief Checks whether a given entry is structurally non-zero
	 * @param row Index of the row
	 * @param col Index of the column
	 * @return @c true if the entry is structurally non-zero, @c false otherwise
	 */
	inline bool isNonZero(sparse_int_t row, sparse_int_t col) const CADET_NOEXCEPT
	{
		cadet_assert((row >= 0) && (row < rows()));
		cadet_assert((col >= 0) && (col < rows()));

		// Try to find the element
		// TODO: Use binary search
		for (sparse_int_t i = _rowStart[row]; i < _rowStart[row + 1]; ++i)
		{
			if (_colIdx[i] == col)
				return true;
		}

		return false;
	}

	/**
	 * @brief Accesses an element at the given position
	 * @details If the element does not exist, @ç 0.0 is returned.
	 * @param [in] row Row index
	 * @param [in] col Column index
	 * @return Value of the element at the given position
	 */
	inline double& operator()(sparse_int_t row, sparse_int_t col)
	{
		cadet_assert((row >= 0) && (row < rows()));
		cadet_assert((col >= 0) && (col < rows()));

		// Try to find the element
		// TODO: Use binary search
		for (sparse_int_t i = _rowStart[row]; i < _rowStart[row + 1]; ++i)
		{
			if (_colIdx[i] == col)
				return _values[i];
		}

		// We don't have it
		_dummy = 0.0;
		return _dummy;
	}

	/**
	 * @brief Accesses an element at the given position
	 * @details If the element does not exist, @ç 0.0 is returned.
	 * @param [in] row Row index
	 * @param [in] col Column index
	 * @return Value of the element at the given position
	 */
	inline double operator()(sparse_int_t row, sparse_int_t col) const CADET_NOEXCEPT
	{
		cadet_assert((row >= 0) && (row < rows()));
		cadet_assert((col >= 0) && (col < rows()));

		// Try to find the element
		// TODO: Use binary search
		for (sparse_int_t i = _rowStart[row]; i < _rowStart[row + 1]; ++i)
		{
			if (_colIdx[i] == col)
				return _values[i];
		}

		// We don't have it
		return 0.0;
	}

	/**
	 * @brief Returns a vector with indices into columnIndices() that mark the beginning of a row
	 * @details The returned vector contains an additional last element that holds the number of non-zeros.
	 * @return Vector with row index pointers
	 */
	inline const std::vector<sparse_int_t>& rowStartIndices() const CADET_NOEXCEPT
	{
		return _rowStart;
	}

	/**
	 * @brief Returns a vector with column indices of the values
	 * @return Vector with column indices
	 */
	inline const std::vector<sparse_int_t>& columnIndices() const CADET_NOEXCEPT
	{
		return _colIdx;
	}

	/**
	 * @brief Returns a vector with element values
	 * @return Vector with element values
	 */
	inline const std::vector<double>& values() const CADET_NOEXCEPT
	{
		return _values;
	}

	/**
	 * @brief Returns an array with column indices of the values in the given row
	 * @param [in] row Index of the row
	 * @return Array with column indices
	 */
	inline sparse_int_t const* columnIndicesOfRow(int row) const CADET_NOEXCEPT
	{
		//		cadet_assert((row >= 0) && (row < rows()));
		return _colIdx.data() + _rowStart[row];
	}

	/**
	 * @brief Returns an array with element values of the given row
	 * @param [in] row Index of the row
	 * @return Array with element values
	 */
	inline double const* valuesOfRow(int row) const CADET_NOEXCEPT
	{
		//		cadet_assert((row >= 0) && (row < rows()));
		return _values.data() + _rowStart[row];
	}

	/**
	 * @brief Returns an array with element values of the given row
	 * @param [in] row Index of the row
	 * @return Array with element values
	 */
	inline double* valuesOfRow(int row) CADET_NOEXCEPT
	{
		//		cadet_assert((row >= 0) && (row < rows()));
		return _values.data() + _rowStart[row];
	}

	/**
	 * @brief Returns the number of (structurally) non-zero elements in the given matrix row
	 * @param [in] row Index of the row
	 * @return Number of (structurally) non-zero elements in the matrix row
	 */
	inline sparse_int_t numNonZerosInRow(int row) const CADET_NOEXCEPT
	{
		return _rowStart[row + 1] - _rowStart[row];
	}

	/**
	 * @brief Returns the number of (structurally) non-zero elements in the matrix
	 * @return Number of (structurally) non-zero elements in the matrix
	 */
	inline int numNonZeros() const CADET_NOEXCEPT
	{
		return _values.size();
	}

	/**
	 * @brief Returns the number of rows and columns (square matrix)
	 * @return Number of rows and columns
	 */
	inline int matrixSize() const CADET_NOEXCEPT
	{
		return _rowStart.size() - 2;
	}

	/**
	 * @brief Returns the number of rows
	 * @return Number of rows
	 */
	inline int rows() const CADET_NOEXCEPT
	{
		return _rowStart.size() - 2;
	}

	/**
	 * @brief Sets all matrix elements to the given value
	 * @param [in] val Value all matrix elements are set to
	 */
	inline void setAll(double val) CADET_NOEXCEPT
	{
		std::fill(_values.begin(), _values.end(), val);
	}

	/**
	 * @brief Sets all matrix elements of the given row to the given value
	 * @param [in] row Index of the row
	 * @param [in] val Value all matrix elements of the given row are set to
	 */
	inline void setAllInRow(int row, double val) CADET_NOEXCEPT
	{
		cadet_assert((row >= 0) && (row < rows()));
		std::fill(_values.begin() + _rowStart[row], _values.begin() + _rowStart[row + 1], val);
	}

	/**
	 * @brief Accesses an element in the matrix where the main diagonal is centered (index @c 0)
	 * @details The @p diagonal determines the element in a row. A negative index indicates a lower diagonal,
	 *          while a positive index indicates an upper diagonal. If @p diagonal is @c 0, then the main
	 *          diagonal is retrieved.
	 *
	 * @param [in] row Index of the row
	 * @param [in] diagonal Index of the diagonal (between negative lower bandwidth and upper bandwidth)
	 *
	 * @return Matrix element at the given position
	 */
	inline double& centered(int row, int diagonal)
	{
		return (*this)(row, row + diagonal);
	}
	inline double centered(int row, int diagonal) const
	{
		return (*this)(row, row + diagonal);
	}

	/**
	 * @brief Accesses an element in the matrix
	 *
	 * @param [in] row Index of the row
	 * @param [in] col Index of the column
	 *
	 * @return Matrix element at the given position
	 */
	inline double& native(int row, int col)
	{
		return (*this)(row, col);
	}
	inline double native(int row, int col) const
	{
		return (*this)(row, col);
	}

	/**
	 * @brief Provides direct access to the underlying memory
	 * @return Pointer to the first element of the underlying array
	 */
	inline double* data() CADET_NOEXCEPT
	{
		return _values.data();
	}
	inline double const* data() const CADET_NOEXCEPT
	{
		return _values.data();
	}

	/**
	 * @brief Creates a RowIterator pointing to the given row
	 * @param [in] idx Index of the row
	 * @return RowIterator pointing to the given row
	 */
	BandedSparseRowIterator row(int idx);
	ConstBandedSparseRowIterator row(int idx) const;

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ using LAPACK
	 * @details Computes @f$ y = Ax @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	void multiplyVector(double const* const x, double* y) const;

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another vector using LAPACK
	 * @details Computes @f$ y = \alpha Ax + \beta y@f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	void multiplyVector(double const* const x, double alpha, double beta, double* y) const;

protected:
	std::vector<double> _values;       //!< Values of matrix entries
	std::vector<sparse_int_t> _colIdx; //!< Column of the value in _values (size is _values.size() = numNonZeros)
	std::vector<sparse_int_t>
		_rowStart; //!< Index into _colIdx that marks the beginning of a row (two entries more than rows; second to last
				   //!< entry points beyond the last row, so that _rowStart[numRows] = _values.size() = numNonZeros)
	double _dummy; //!< Dummy for access by reference
};

class BandedSparseRowIterator
{
public:
	/**
	 * @brief Creates a ConstBandedSparseRowIterator pointing nowhere
	 */
	BandedSparseRowIterator()
		: _matrix(nullptr), _values(nullptr), _colIdx(nullptr), _row(-1), _numNonZero(0), _dummy(0.0)
	{
	}

	/**
	 * @brief Creates a BandedSparseRowIterator of the given matrix row
	 * @param [in] mat Matrix
	 * @param [in] row Index of the row
	 */
	BandedSparseRowIterator(CompressedSparseMatrix& mat, int row)
		: _matrix(&mat), _values(mat.valuesOfRow(row)), _colIdx(mat.columnIndicesOfRow(row)), _row(row),
		  _numNonZero(mat.numNonZerosInRow(row)), _dummy(0.0)
	{
	}

	~BandedSparseRowIterator() CADET_NOEXCEPT
	{
	}

	// Default copy and assignment semantics
	BandedSparseRowIterator(const BandedSparseRowIterator& cpy) = default;
	BandedSparseRowIterator& operator=(const BandedSparseRowIterator& cpy) = default;

	BandedSparseRowIterator(BandedSparseRowIterator&& cpy) CADET_NOEXCEPT = default;
#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	BandedSparseRowIterator& operator=(BandedSparseRowIterator&& cpy) CADET_NOEXCEPT = default;
#else
	BandedSparseRowIterator& operator=(BandedSparseRowIterator&& cpy) = default;
#endif

	/**
	 * @brief Sets all matrix elements in the row to the given value
	 * @param [in] val Value all matrix elements in the row are set to
	 */
	inline void setAll(double val)
	{
		std::fill(_values, _values + _numNonZero, val);
	}

	/**
	 * @brief Copies a row of another iterator to the row of this iterator
	 * @details Assumes the same sparsity pattern of source and destination row.
	 * @param [in] it Iterator pointing to a row of a matrix
	 */
	template <typename OtherIterator_t> inline void copyRowFrom(const OtherIterator_t& it)
	{
		cadet_assert(_numNonZero == it.numNonZeros());
		std::copy(it.data(), it.data() + _numNonZero, _values);
	}

	/**
	 * @brief Accesses an element in the current row where the main diagonal is centered (index @c 0)
	 * @details The @p diagonal determines the element in a row. A negative index indicates a lower diagonal,
	 *          while a positive index indicates an upper diagonal. If @p diagonal is @c 0, then the main
	 *          diagonal is retrieved.
	 *
	 * @param [in] diagonal Index of the diagonal
	 *
	 * @return Matrix element at the given position
	 */
	inline double& centered(int diagonal)
	{
		return native(_row + diagonal);
	}
	inline double centered(int diagonal) const
	{
		return native(_row + diagonal);
	}

	/**
	 * @brief Accesses an element in the current row where the lowest diagonal is indexed by @c 0
	 * @param [in] col Index of the column
	 * @return Matrix element at the given position
	 */
	inline double& native(sparse_int_t col)
	{
		cadet_assert((col >= 0) && (col < _matrix->rows()));

		// Try to find the element
		// TODO: Use binary search
		for (int i = 0; i < _numNonZero; ++i)
		{
			if (_colIdx[i] == col)
				return _values[i];
		}

		// We don't have it
		_dummy = 0.0;
		return _dummy;
	}

	inline double native(sparse_int_t col) const
	{
		cadet_assert((col >= 0) && (col < _matrix->rows()));

		// Try to find the element
		// TODO: Use binary search
		for (int i = 0; i < _numNonZero; ++i)
		{
			if (_colIdx[i] == col)
				return _values[i];
		}

		// We don't have it
		return 0.0;
	}

	inline double& operator()(int diagonal)
	{
		return centered(diagonal);
	}
	inline double operator()(int diagonal) const
	{
		return centered(diagonal);
	}

	inline double& operator[](int diagonal)
	{
		return centered(diagonal);
	}
	inline double operator[](int diagonal) const
	{
		return centered(diagonal);
	}

	inline BandedSparseRowIterator& operator++() CADET_NOEXCEPT
	{
		++_row;
		updateOnRowChange();
		return *this;
	}

	inline BandedSparseRowIterator& operator--() CADET_NOEXCEPT
	{
		--_row;
		updateOnRowChange();
		return *this;
	}

	inline BandedSparseRowIterator& operator+=(int idx) CADET_NOEXCEPT
	{
		_row += idx;
		updateOnRowChange();
		return *this;
	}

	inline BandedSparseRowIterator& operator-=(int idx) CADET_NOEXCEPT
	{
		_row -= idx;
		updateOnRowChange();
		return *this;
	}

	inline BandedSparseRowIterator operator+(int op) const CADET_NOEXCEPT
	{
		return BandedSparseRowIterator(*_matrix, _row + op);
	}

	inline friend BandedSparseRowIterator operator+(int op, const BandedSparseRowIterator& it) CADET_NOEXCEPT
	{
		return BandedSparseRowIterator(*it._matrix, op + it.row());
	}

	inline BandedSparseRowIterator operator-(int op) const CADET_NOEXCEPT
	{
		return BandedSparseRowIterator(*_matrix, _row - op);
	}

	inline friend BandedSparseRowIterator operator-(int op, const BandedSparseRowIterator& it) CADET_NOEXCEPT
	{
		return BandedSparseRowIterator(*it._matrix, op - it.row());
	}

	/**
	 * @brief Returns the underlying matrix this iterator is pointing into
	 * @return Matrix this iterator is pointing into
	 */
	inline const CompressedSparseMatrix& matrix() const CADET_NOEXCEPT
	{
		return *_matrix;
	}

	/**
	 * @brief Returns the index of the current row
	 * @return Index of the current row
	 */
	inline int row() const CADET_NOEXCEPT
	{
		return _row;
	}

	/**
	 * @brief Returns the number of non-zero entries in this row
	 * @return Number of non-zero entries in this row
	 */
	inline int numNonZeros() const CADET_NOEXCEPT
	{
		return _numNonZero;
	}

	/**
	 * @brief Returns an array of the matrix entries in this row
	 * @return Array with matrix entries in this row
	 */
	inline double* data() CADET_NOEXCEPT
	{
		return _values;
	}
	inline double const* data() const CADET_NOEXCEPT
	{
		return _values;
	}

protected:
	inline void updateOnRowChange() CADET_NOEXCEPT
	{
		_values = _matrix->valuesOfRow(_row);
		_colIdx = _matrix->columnIndicesOfRow(_row);
		_numNonZero = _matrix->numNonZerosInRow(_row);
	}

	CompressedSparseMatrix* _matrix;
	double* _values;
	sparse_int_t const* _colIdx;
	int _row;
	int _numNonZero;
	double _dummy;
};

class ConstBandedSparseRowIterator
{
public:
	/**
	 * @brief Creates a ConstBandedSparseRowIterator pointing nowhere
	 */
	ConstBandedSparseRowIterator() : _matrix(nullptr), _values(nullptr), _colIdx(nullptr), _row(-1), _numNonZero(0)
	{
	}

	/**
	 * @brief Creates a ConstBandedSparseRowIterator of the given matrix row
	 * @param [in] mat Matrix
	 * @param [in] row Index of the row
	 */
	ConstBandedSparseRowIterator(const CompressedSparseMatrix& mat, int row)
		: _matrix(&mat), _values(mat.valuesOfRow(row)), _colIdx(mat.columnIndicesOfRow(row)), _row(row),
		  _numNonZero(mat.numNonZerosInRow(row))
	{
	}

	~ConstBandedSparseRowIterator() CADET_NOEXCEPT
	{
	}

	// Default copy and assignment semantics
	ConstBandedSparseRowIterator(const ConstBandedSparseRowIterator& cpy) = default;
	ConstBandedSparseRowIterator& operator=(const ConstBandedSparseRowIterator& cpy) = default;

	ConstBandedSparseRowIterator(ConstBandedSparseRowIterator&& cpy) CADET_NOEXCEPT = default;
#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	ConstBandedSparseRowIterator& operator=(ConstBandedSparseRowIterator&& cpy) CADET_NOEXCEPT = default;
#else
	ConstBandedSparseRowIterator& operator=(ConstBandedSparseRowIterator&& cpy) = default;
#endif

	/**
	 * @brief Accesses an element in the current row where the main diagonal is centered (index @c 0)
	 * @details The @p diagonal determines the element in a row. A negative index indicates a lower diagonal,
	 *          while a positive index indicates an upper diagonal. If @p diagonal is @c 0, then the main
	 *          diagonal is retrieved.
	 *
	 * @param [in] diagonal Index of the diagonal
	 *
	 * @return Matrix element at the given position
	 */
	inline double centered(int diagonal) const
	{
		return native(_row + diagonal);
	}

	/**
	 * @brief Accesses an element in the current row where the lowest diagonal is indexed by @c 0
	 * @param [in] col Index of the column
	 * @return Matrix element at the given position
	 */
	inline double native(sparse_int_t col) const
	{
		cadet_assert((col >= 0) && (col < _matrix->rows()));

		// Try to find the element
		// TODO: Use binary search
		for (int i = 0; i < _numNonZero; ++i)
		{
			if (_colIdx[i] == col)
				return _values[i];
		}

		// We don't have it
		return 0.0;
	}

	inline double operator()(int diagonal) const
	{
		return centered(diagonal);
	}
	inline double operator[](int diagonal) const
	{
		return centered(diagonal);
	}

	inline ConstBandedSparseRowIterator& operator++() CADET_NOEXCEPT
	{
		++_row;
		updateOnRowChange();
		return *this;
	}

	inline ConstBandedSparseRowIterator& operator--() CADET_NOEXCEPT
	{
		--_row;
		updateOnRowChange();
		return *this;
	}

	inline ConstBandedSparseRowIterator& operator+=(int idx) CADET_NOEXCEPT
	{
		_row += idx;
		updateOnRowChange();
		return *this;
	}

	inline ConstBandedSparseRowIterator& operator-=(int idx) CADET_NOEXCEPT
	{
		_row -= idx;
		updateOnRowChange();
		return *this;
	}

	inline ConstBandedSparseRowIterator operator+(int op) const CADET_NOEXCEPT
	{
		return ConstBandedSparseRowIterator(*_matrix, _row + op);
	}

	inline friend ConstBandedSparseRowIterator operator+(int op, const ConstBandedSparseRowIterator& it) CADET_NOEXCEPT
	{
		return ConstBandedSparseRowIterator(*it._matrix, op + it._row);
	}

	inline ConstBandedSparseRowIterator operator-(int op) const CADET_NOEXCEPT
	{
		return ConstBandedSparseRowIterator(*_matrix, _row - op);
	}

	inline friend ConstBandedSparseRowIterator operator-(int op, const ConstBandedSparseRowIterator& it) CADET_NOEXCEPT
	{
		return ConstBandedSparseRowIterator(*it._matrix, op - it._row);
	}

	/**
	 * @brief Returns the underlying matrix this iterator is pointing into
	 * @return Matrix this iterator is pointing into
	 */
	inline const CompressedSparseMatrix& matrix() const CADET_NOEXCEPT
	{
		return *_matrix;
	}

	/**
	 * @brief Returns the index of the current row
	 * @return Index of the current row
	 */
	inline int row() const CADET_NOEXCEPT
	{
		return _row;
	}

	/**
	 * @brief Returns the number of non-zero entries in this row
	 * @return Number of non-zero entries in this row
	 */
	inline int numNonZeros() const CADET_NOEXCEPT
	{
		return _numNonZero;
	}

	/**
	 * @brief Returns an array of the matrix entries in this row
	 * @return Array with matrix entries in this row
	 */
	inline double const* data() const CADET_NOEXCEPT
	{
		return _values;
	}

protected:
	inline void updateOnRowChange() CADET_NOEXCEPT
	{
		_values = _matrix->valuesOfRow(_row);
		_colIdx = _matrix->columnIndicesOfRow(_row);
		_numNonZero = _matrix->numNonZerosInRow(_row);
	}

	CompressedSparseMatrix const* _matrix;
	double const* _values;
	sparse_int_t const* _colIdx;
	int _row;
	int _numNonZero;
};

std::ostream& operator<<(std::ostream& out, const CompressedSparseMatrix& sm);

} // namespace linalg

} // namespace cadet

#endif // LIBCADET_COMPRESSEDSPARSEMATRIX_HPP_
