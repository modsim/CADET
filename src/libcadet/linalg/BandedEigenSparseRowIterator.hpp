/**
 * @file
 * Defines RowIterator for Eigen lib compressed sparse matrix
 */

#ifndef LIBCADET_BANDEDEIGENSPARSEROWITERATOR_HPP_
#define LIBCADET_BANDEDEIGENSPARSEROWITERATOR_HPP_

#include <vector>
#include <ostream>
#include <algorithm>
#include <Eigen/Sparse>// @TODO: third party #include <Eigen>
#include <Eigen/Dense>

#include "SparseSolverInterface.hpp"
#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"

namespace cadet
{

namespace linalg
{


class BandedEigenSparseRowIterator
{
public:
	/**
	 * @brief Creates a ConstBandedSparseRowIterator pointing nowhere
	 */
	BandedEigenSparseRowIterator() : _matrix(nullptr), _values(nullptr), _colIdx(nullptr), _row(-1), _numNonZero(0), _dummy(0.0) { }

	/**
	 * @brief Creates a BandedSparseRowIterator of the given matrix row
	 * @param [in] mat Matrix
	 * @param [in] row Index of the row
	 */
	BandedEigenSparseRowIterator(Eigen::SparseMatrix<double, 0x1>& mat, int row)
		: _matrix(&mat), _values(valuesOfRow(mat, row)), _colIdx(columnIndicesOfRow(mat, row)), _row(row),
		_numNonZero(getInnerNumberOfNonZeros(mat, row)), _dummy(0.0)
	{
	}

	~BandedEigenSparseRowIterator() CADET_NOEXCEPT { }

	// Default copy and assignment semantics
	BandedEigenSparseRowIterator(const BandedEigenSparseRowIterator& cpy) = default;
	BandedEigenSparseRowIterator& operator=(const BandedEigenSparseRowIterator& cpy) = default;

	BandedEigenSparseRowIterator(BandedEigenSparseRowIterator&& cpy) CADET_NOEXCEPT = default;
#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	BandedSparseRowIterator& operator=(BandedSparseRowIterator&& cpy) CADET_NOEXCEPT = default;
#else
	BandedEigenSparseRowIterator& operator=(BandedEigenSparseRowIterator&& cpy) = default;
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
	template <typename OtherIterator_t>
	inline void copyRowFrom(const OtherIterator_t& it)
	{
		cadet_assert(_numNonZero == it.numNonZeros());
		std::copy(it.data(), it.data() + _numNonZero, _values);
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
		// shift to column index to start with
		int idx = startDiag - _row;

		for (int i = 0; i < length; i++) {
			_values[idx + i] = factor * row[i];
		}

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
	inline double& centered(int diagonal) { return native(_row + diagonal); }
	inline double centered(int diagonal) const { return native(_row + diagonal); }

	/**
	 * @brief Accesses an element in the current row where the lowest diagonal is indexed by @c 0
	 * @param [in] col Index of the column
	 * @return Matrix element at the given position
	 */
	inline double& native(int col)
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

	inline double& operator()(int diagonal) { return centered(diagonal); }
	inline double operator()(int diagonal) const { return centered(diagonal); }

	inline double& operator[](int diagonal) { return centered(diagonal); }
	inline double operator[](int diagonal) const { return centered(diagonal); }

	inline BandedEigenSparseRowIterator& operator++() CADET_NOEXCEPT
	{
		++_row;
		updateOnRowChange();
		return *this;
	}

	inline BandedEigenSparseRowIterator& operator--() CADET_NOEXCEPT
	{
		--_row;
		updateOnRowChange();
		return *this;
	}

	inline BandedEigenSparseRowIterator& operator+=(int idx) CADET_NOEXCEPT
	{
		_row += idx;
		updateOnRowChange();
		return *this;
	}

	inline BandedEigenSparseRowIterator& operator-=(int idx) CADET_NOEXCEPT
	{
		_row -= idx;
		updateOnRowChange();
		return *this;
	}

	inline BandedEigenSparseRowIterator operator+(int op) const CADET_NOEXCEPT
	{
		return BandedEigenSparseRowIterator(*_matrix, _row + op);
	}

	inline friend BandedEigenSparseRowIterator operator+(int op, const BandedEigenSparseRowIterator& it) CADET_NOEXCEPT
	{
		return BandedEigenSparseRowIterator(*it._matrix, op + it.row());
	}

	inline BandedEigenSparseRowIterator operator-(int op) const CADET_NOEXCEPT
	{
		return BandedEigenSparseRowIterator(*_matrix, _row - op);
	}

	inline friend BandedEigenSparseRowIterator operator-(int op, const BandedEigenSparseRowIterator& it) CADET_NOEXCEPT
	{
		return BandedEigenSparseRowIterator(*it._matrix, op - it.row());
	}

	/**
	 * @brief Returns the underlying matrix this iterator is pointing into
	 * @return Matrix this iterator is pointing into
	 */
	inline const Eigen::SparseMatrix<double, 0x1>& matrix() const CADET_NOEXCEPT { return *_matrix; }

	/**
	 * @brief Returns the index of the current row
	 * @return Index of the current row
	 */
	inline int row() const CADET_NOEXCEPT { return _row; }

	/**
	 * @brief Returns the number of non-zero entries in this row
	 * @return Number of non-zero entries in this row
	 */
	inline int numNonZeros() const { return _numNonZero; }

	/**
	 * @brief Returns an array of the matrix entries in this row
	 * @return Array with matrix entries in this row
	 */
	inline double* data() { return _values; }
	inline double const* data() const { return _values; }

protected:

	inline void updateOnRowChange()
	{
		_values = valuesOfRow(*_matrix, _row);
		_colIdx = columnIndicesOfRow(*_matrix, _row);
		_numNonZero = getInnerNumberOfNonZeros(*_matrix, _row);
	}

	inline double* valuesOfRow(Eigen::SparseMatrix<double, 0x1>& mat, int row) {
		return mat.valuePtr() + mat.outerIndexPtr()[row];
	}

	inline const int* columnIndicesOfRow(Eigen::SparseMatrix<double, 0x1>& mat, int row) {
		return mat.innerIndexPtr() + mat.outerIndexPtr()[row];
	}

	inline int getInnerNumberOfNonZeros(Eigen::SparseMatrix<double, 0x1>& mat, int row) {
		return mat.outerIndexPtr()[row + 1] - mat.outerIndexPtr()[row];
	}

	Eigen::SparseMatrix<double, 0x1>* _matrix;
	double* _values;
	int const* _colIdx;
	int _row;
	int _numNonZero;
	double _dummy;
};

} // namespace linalg

} // namespace cadet

#endif  // LIBCADET_BANDEDEIGENSPARSEROWITERATOR_HPP_