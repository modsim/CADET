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
 * Defines a band matrix
 */

#ifndef LIBCADET_BANDMATRIX_HPP_
#define LIBCADET_BANDMATRIX_HPP_

#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"
#include "LapackInterface.hpp"

#include <ostream>
#include <algorithm>

namespace cadet
{

namespace linalg
{

/**
 * @brief Iterates over rows of a banded matrix like BandMatrix or FactorizableBandMatrix
 * @tparam MatrixType Type of the matrix class, one of BandMatrix or FactorizableBandMatrix
 */
template <class MatrixType> class BandedRowIterator
{
public:
	/**
	 * @brief Creates an empty BandedRowIterator pointing to nothing
	 */
	BandedRowIterator() CADET_NOEXCEPT : _matrix(nullptr), _pos(nullptr)
	{
#ifdef CADET_DEBUG
		_row = -1;
#endif
	}

	/**
	 * @brief Creates a BandedRowIterator for the given MatrixType
	 * @param [in] mat MatrixType of the BandedRowIterator
	 */
	BandedRowIterator(MatrixType& mat) CADET_NOEXCEPT : _matrix(&mat), _pos(_matrix->_data)
	{
#ifdef CADET_DEBUG
		_row = 0;
#endif
	}

	/**
	 * @brief Creates a BandedRowIterator for the given MatrixType
	 * @param [in] mat MatrixType of the BandedRowIterator
	 * @param [in] ptr Pointer to the first row of the matrix
	 */
	BandedRowIterator(MatrixType& mat, double* const ptr) CADET_NOEXCEPT : _matrix(&mat), _pos(ptr)
	{
#ifdef CADET_DEBUG
		_row = 0;
#endif
	}

	/**
	 * @brief Creates a BandedRowIterator for the given MatrixType
	 * @param [in] mat MatrixType of the BandedRowIterator
	 * @param [in] row Index of the row of the iterator points so
	 */
	BandedRowIterator(MatrixType& mat, int row) CADET_NOEXCEPT : _matrix(&mat),
																 _pos(_matrix->_data + row * _matrix->stride())
	{
#ifdef CADET_DEBUG
		_row = row;
#endif
	}

	/**
	 * @brief Creates a BandedRowIterator for the given MatrixType
	 * @param [in] mat MatrixType of the BandedRowIterator
	 * @param [in] row Index of the row of the iterator points so
	 * @param [in] offset Additional  offset
	 */
	BandedRowIterator(MatrixType& mat, int row, int offset) CADET_NOEXCEPT
		: _matrix(&mat),
		  _pos(_matrix->_data + row * _matrix->stride() + offset)
	{
#ifdef CADET_DEBUG
		_row = row;
#endif
	}

	BandedRowIterator(const BandedRowIterator& cpy) CADET_NOEXCEPT : _matrix(cpy._matrix), _pos(cpy._pos)
	{
#ifdef CADET_DEBUG
		_row = cpy._row;
#endif
	}
	BandedRowIterator(const BandedRowIterator& cpy, int rowChange) CADET_NOEXCEPT
		: _matrix(cpy._matrix),
		  _pos(cpy._pos + rowChange * static_cast<int>(cpy._matrix->stride()))
	{
#ifdef CADET_DEBUG
		_row = cpy._row + rowChange;
#endif
	}
	BandedRowIterator(BandedRowIterator&& cpy) CADET_NOEXCEPT : _matrix(cpy._matrix), _pos(cpy._pos)
	{
#ifdef CADET_DEBUG
		_row = cpy._row;
#endif
	}

	inline BandedRowIterator& operator=(const BandedRowIterator& cpy) CADET_NOEXCEPT
	{
		cadet_assert(&cpy != this);

		_matrix = cpy._matrix;
		_pos = cpy._pos;
#ifdef CADET_DEBUG
		_row = cpy._row;
#endif
		return *this;
	}

	inline BandedRowIterator& operator=(BandedRowIterator&& cpy) CADET_NOEXCEPT
	{
		_matrix = cpy._matrix;
		_pos = cpy._pos;
#ifdef CADET_DEBUG
		_row = cpy._row;
#endif
		return *this;
	}

	/**
	 * @brief Sets all matrix elements in the row to the given value
	 * @param [in] val Value all matrix elements in the row are set to
	 */
	inline void setAll(double val)
	{
		for (int i = 0; i < _matrix->apparentStride(); ++i)
			_pos[i] = val;
	}

	/**
	 * @brief Copies a row of another iterator to the row of this iterator
	 * @param [in] it Iterator pointing to a row of a matrix
	 */
	template <typename OtherIterator_t> inline void copyRowFrom(const OtherIterator_t& it)
	{
		cadet_assert(it.nonZeroColumnsPerRow() <= nonZeroColumnsPerRow());
		cadet_assert(it.matrix().lowerBandwidth() == _matrix->lowerBandwidth());
		for (int i = 0; i < std::min(it.nonZeroColumnsPerRow(), nonZeroColumnsPerRow()); ++i)
			_pos[i] = it.native(i);
	}

	/**
	 * @brief Adds a subset of this row @f$ x @f$ to another row @f$ y @f$ performing @f$ y = y + \alpha x @f$
	 * @details The subset has to fully fit into the destination row (taking into account the @p shift).
	 * @param [in] riDest Row iterator pointing to the destination row @f$ y @f$
	 * @param [in] factor Factor @f$ \alpha @f$
	 * @param [in] lowerBand Start point of the row subset as lower band index
	 * @param [in] upperBand End point of the row subset as upper band index
	 * @param [in] shift Amount of items the subset is shifted to the right in the destination row
	 */
	template <typename OtherIterator_t>
	inline void addSubsetTo(const OtherIterator_t& riDest, double factor, int lowerBand, int upperBand, int shift) const
	{
		cadet_assert(static_cast<int>(riDest._matrix->upperBandwidth()) >= shift + upperBand);
		cadet_assert(static_cast<int>(riDest._matrix->lowerBandwidth()) >= -shift - lowerBand);
		double* const dest = riDest._pos + static_cast<int>(riDest._matrix->_lowerBand) + shift;
		double const* const src = _pos + _matrix->_lowerBand;
		for (int i = lowerBand; i < upperBand; ++i)
			dest[i] += factor * src[i];
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
		cadet_assert(-startDiag <= _matrix->lowerBandwidth());
		cadet_assert(startDiag + length <= _matrix->upperBandwidth() + 1);
		double* const dest = _pos + _matrix->_lowerBand + startDiag;
		for (int i = 0; i < length; ++i)
			dest[i] += factor * row[i];
	}

	/**
	 * @brief Accesses an element in the current row where the main diagonal is centered (index @c 0)
	 * @details The @p diagonal determines the element in a row. A negative index indicates a lower diagonal,
	 *          while a positive index indicates an upper diagonal. If @p diagonal is @c 0, then the main
	 *          diagonal is retrieved.
	 *
	 * @param [in] diagonal Index of the diagonal (between negative lower bandwidth and upper bandwidth)
	 *
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
		cadet_assert(col < _matrix->stride());
		cadet_assert(_row < _matrix->rows());
		return _pos[col];
	}

	inline double native(int col) const
	{
		cadet_assert(col < _matrix->stride());
		cadet_assert(_row < _matrix->rows());
		return _pos[col];
	}

	inline double& operator()(int diagonal)
	{
		cadet_assert(diagonal <= static_cast<int>(_matrix->_upperBand));
		cadet_assert(-diagonal <= static_cast<int>(_matrix->_lowerBand));
		cadet_assert(_row < _matrix->rows());
		return _pos[diagonal + _matrix->_lowerBand];
	}

	inline double operator()(int diagonal) const
	{
		cadet_assert(diagonal <= static_cast<int>(_matrix->_upperBand));
		cadet_assert(-diagonal <= static_cast<int>(_matrix->_lowerBand));
		cadet_assert(_row < _matrix->rows());
		return _pos[diagonal + _matrix->_lowerBand];
	}

	inline double& operator[](int diagonal)
	{
		return (*this)(diagonal);
	}
	inline double operator[](int diagonal) const
	{
		return (*this)(diagonal);
	}

	inline BandedRowIterator& operator++() CADET_NOEXCEPT
	{
		_pos += _matrix->stride();
#ifdef CADET_DEBUG
		++_row;
#endif
		return *this;
	}

	inline BandedRowIterator& operator--() CADET_NOEXCEPT
	{
		_pos -= _matrix->stride();
#ifdef CADET_DEBUG
		--_row;
#endif
		return *this;
	}

	inline BandedRowIterator& operator+=(int idx) CADET_NOEXCEPT
	{
		_pos += idx * _matrix->stride();
#ifdef CADET_DEBUG
		_row += idx;
#endif
		return *this;
	}

	inline BandedRowIterator& operator-=(int idx) CADET_NOEXCEPT
	{
		_pos -= idx * _matrix->stride();
#ifdef CADET_DEBUG
		_row -= idx;
#endif
		return *this;
	}

	inline BandedRowIterator operator+(int op) const CADET_NOEXCEPT
	{
		return BandedRowIterator(*this, op);
	}

	inline friend BandedRowIterator operator+(int op, const BandedRowIterator& it) CADET_NOEXCEPT
	{
		return BandedRowIterator(it, op);
	}

	inline BandedRowIterator operator-(int op) const CADET_NOEXCEPT
	{
		return BandedRowIterator(*this, -op);
	}

	inline friend BandedRowIterator operator-(int op, const BandedRowIterator& it) CADET_NOEXCEPT
	{
		return BandedRowIterator(it, -op);
	}

	/**
	 * @brief Returns the underlying matrix this iterator is pointing into
	 * @return Matrix this iterator is pointing into
	 */
	inline const MatrixType& matrix() const CADET_NOEXCEPT
	{
		return *_matrix;
	}

	/**
	 * @brief Returns the number of (potentially) non-zero elements per row (the total bandwidth)
	 * @return The number of (potentially) non-zero elements per row
	 */
	inline int nonZeroColumnsPerRow() const CADET_NOEXCEPT
	{
		return _matrix->apparentStride();
	}

#ifdef CADET_DEBUG
	/**
	 * @brief Returns the index of the current row
	 * @return Index of the current row
	 */
	inline int row() const CADET_NOEXCEPT
	{
		return _row;
	}
#endif

private:
	MatrixType* _matrix; //!< Underlying matrix
	double* _pos;        //!< Current position, points to the lowest subdiagonal of a row
#ifdef CADET_DEBUG
	int _row; //!< Index of the current row
#endif
};

/**
 * @brief Iterates over rows of a banded matrix like BandMatrix or FactorizableBandMatrix
 * @tparam MatrixType Type of the matrix class, one of BandMatrix or FactorizableBandMatrix
 */
template <class MatrixType> class ConstBandedRowIterator
{
public:
	/**
	 * @brief Creates an empty ConstBandedRowIterator pointing to nothing
	 * @param [in] mat MatrixType of the ConstBandedRowIterator
	 */
	ConstBandedRowIterator() CADET_NOEXCEPT : _matrix(nullptr), _pos(nullptr)
	{
#ifdef CADET_DEBUG
		_row = -1;
#endif
	}

	/**
	 * @brief Creates a ConstBandedRowIterator for the given MatrixType.
	 * @param [in] mat MatrixType of the ConstBandedRowIterator
	 */
	ConstBandedRowIterator(const MatrixType& mat) CADET_NOEXCEPT : _matrix(&mat), _pos(_matrix->_data)
	{
#ifdef CADET_DEBUG
		_row = 0;
#endif
	}

	/**
	 * @brief Creates a ConstBandedRowIterator for the given MatrixType.
	 * @param [in] mat MatrixType of the ConstBandedRowIterator
	 * @param [in] ptr Pointer to the first row of the matrix
	 */
	ConstBandedRowIterator(const MatrixType& mat, double const* const ptr) CADET_NOEXCEPT : _matrix(&mat), _pos(ptr)
	{
#ifdef CADET_DEBUG
		_row = 0;
#endif
	}

	/**
	 * @brief Creates a ConstBandedRowIterator for the given MatrixType.
	 * @param [in] mat MatrixType of the ConstBandedRowIterator
	 * @param [in] row Index of the row of the iterator points so
	 */
	ConstBandedRowIterator(const MatrixType& mat, int row) CADET_NOEXCEPT
		: _matrix(&mat),
		  _pos(_matrix->_data + row * _matrix->stride())
	{
#ifdef CADET_DEBUG
		_row = row;
#endif
	}

	/**
	 * @brief Creates a ConstBandedRowIterator for the given MatrixType.
	 * @param [in] mat MatrixType of the ConstBandedRowIterator
	 * @param [in] row Index of the row of the iterator points so
	 * @param [in] offset Additional  offset
	 */
	ConstBandedRowIterator(const MatrixType& mat, int row, int offset) CADET_NOEXCEPT
		: _matrix(&mat),
		  _pos(_matrix->_data + row * _matrix->stride() + offset)
	{
#ifdef CADET_DEBUG
		_row = row;
#endif
	}

	ConstBandedRowIterator(const ConstBandedRowIterator& cpy) CADET_NOEXCEPT : _matrix(cpy._matrix), _pos(cpy._pos)
	{
#ifdef CADET_DEBUG
		_row = cpy._row;
#endif
	}
	ConstBandedRowIterator(const ConstBandedRowIterator& cpy, int rowChange) CADET_NOEXCEPT
		: _matrix(cpy._matrix),
		  _pos(cpy._pos + rowChange * static_cast<int>(cpy._matrix->stride()))
	{
#ifdef CADET_DEBUG
		_row = cpy._row + rowChange;
#endif
	}
	ConstBandedRowIterator(const ConstBandedRowIterator&& cpy) CADET_NOEXCEPT : _matrix(cpy._matrix), _pos(cpy._pos)
	{
#ifdef CADET_DEBUG
		_row = cpy._row;
#endif
	}

	inline ConstBandedRowIterator& operator=(const ConstBandedRowIterator& cpy) CADET_NOEXCEPT
	{
		cadet_assert(&cpy != this);

		_matrix = cpy._matrix;
		_pos = cpy._pos;
#ifdef CADET_DEBUG
		_row = cpy._row;
#endif
		return *this;
	}

	inline ConstBandedRowIterator& operator=(ConstBandedRowIterator&& cpy) CADET_NOEXCEPT
	{
		_matrix = cpy._matrix;
		_pos = cpy._pos;
#ifdef CADET_DEBUG
		_row = cpy._row;
#endif
		return *this;
	}

	/**
	 * @brief Accesses an element in the current row where the main diagonal is centered (index @c 0)
	 * @details The @p diagonal determines the element in a row. A negative index indicates a lower diagonal,
	 *          while a positive index indicates an upper diagonal. If @p diagonal is @c 0, then the main
	 *          diagonal is retrieved.
	 *
	 * @param [in] diagonal Index of the diagonal (between negative lower bandwidth and upper bandwidth)
	 *
	 * @return Matrix element at the given position
	 */
	inline double centered(int diagonal) const
	{
		return (*this)(diagonal);
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
	inline double native(int col) const
	{
		cadet_assert(col < _matrix->stride());
		cadet_assert(_row < _matrix->rows());
		return _pos[col];
	}

	inline double operator()(int diagonal) const
	{
		cadet_assert(diagonal <= static_cast<int>(_matrix->_upperBand));
		cadet_assert(-diagonal <= static_cast<int>(_matrix->_lowerBand));
		cadet_assert(_row < _matrix->rows());
		return _pos[diagonal + _matrix->_lowerBand];
	}

	inline double operator[](int diagonal) const
	{
		return (*this)(diagonal);
	}

	inline ConstBandedRowIterator& operator++() CADET_NOEXCEPT
	{
		_pos += _matrix->stride();
#ifdef CADET_DEBUG
		++_row;
#endif
		return *this;
	}

	inline ConstBandedRowIterator& operator--() CADET_NOEXCEPT
	{
		_pos -= _matrix->stride();
#ifdef CADET_DEBUG
		--_row;
#endif
		return *this;
	}

	inline ConstBandedRowIterator& operator+=(int idx) CADET_NOEXCEPT
	{
		_pos += idx * _matrix->stride();
#ifdef CADET_DEBUG
		_row += idx;
#endif
		return *this;
	}

	inline ConstBandedRowIterator& operator-=(int idx) CADET_NOEXCEPT
	{
		_pos -= idx * _matrix->stride();
#ifdef CADET_DEBUG
		_row -= idx;
#endif
		return *this;
	}

	inline ConstBandedRowIterator operator+(int op) const CADET_NOEXCEPT
	{
		return ConstBandedRowIterator(*this, op);
	}

	inline friend ConstBandedRowIterator operator+(int op, const ConstBandedRowIterator& it) CADET_NOEXCEPT
	{
		return ConstBandedRowIterator(it, op);
	}

	inline ConstBandedRowIterator operator-(int op) const CADET_NOEXCEPT
	{
		return ConstBandedRowIterator(*this, -op);
	}

	inline friend ConstBandedRowIterator operator-(int op, const ConstBandedRowIterator& it) CADET_NOEXCEPT
	{
		return ConstBandedRowIterator(it, -op);
	}

	/**
	 * @brief Returns the underlying matrix this iterator is pointing into
	 * @return Matrix this iterator is pointing into
	 */
	inline const MatrixType& matrix() const CADET_NOEXCEPT
	{
		return *_matrix;
	}

	/**
	 * @brief Returns the number of (potentially) non-zero elements per row (the total bandwidth)
	 * @return The number of (potentially) non-zero elements per row
	 */
	inline int nonZeroColumnsPerRow() const CADET_NOEXCEPT
	{
		return _matrix->apparentStride();
	}

#ifdef CADET_DEBUG
	/**
	 * @brief Returns the index of the current row
	 * @return Index of the current row
	 */
	inline int row() const CADET_NOEXCEPT
	{
		return _row;
	}
#endif

private:
	MatrixType const* _matrix; //!< Underlying matrix
	double const* _pos;        //!< Current position, points to the lowest subdiagonal of a row
#ifdef CADET_DEBUG
	int _row; //!< Index of the current row
#endif
};

template <class MatrixType> inline std::ostream& operator<<(std::ostream& out, const BandedRowIterator<MatrixType>& bri)
{
	const int stride = bri.matrix().apparentStride();
	if (stride == 0)
		out << "[]";
	else if (stride == 1)
		out << "[" << bri.native(0) << "]";
	else
	{
		out << "[" << bri.native(0);
		for (int i = 1; i < stride; ++i)
			out << ", " << bri.native(i);
		out << "]";
	}
	return out;
}

/**
 * @brief Represents a band matrix with given number of rows and diagonals
 * @details LAPACK uses column-major storage, whereas this class uses row-major.
 *          Thus, what we call a row here is actually a column for LAPACK.
 *          Concluding, we have to use the transposed LAPACK operations for
 *          solution and matrix-vector multiplication. The ordering is irrelevant
 *          for the factorization.
 *
 *          Because of the transposition induced by the differing ordering,
 *          the number of upper and lower diagonals switches (e.g., upper diagonals
 *          are transposed lower diagonals). The ordering of the elements inside
 *          one (original) row is maintained (i.e., the first element in a row becomes
 *          the first element in a column and the last element in a row transposes to
 *          the last element in a column).
 * @todo Refactor and combine code with FactorizableBandMatrix in order to save LOC
 */
class BandMatrix
{
public:
	typedef BandedRowIterator<BandMatrix> RowIterator;
	friend class BandedRowIterator<BandMatrix>;
	typedef ConstBandedRowIterator<BandMatrix> ConstRowIterator;
	friend class ConstBandedRowIterator<BandMatrix>;

	/**
	 * @brief Creates an empty, unitialized band matrix
	 * @details No memory is allocated for the matrix. Users have to call resize() first.
	 */
	BandMatrix() CADET_NOEXCEPT : _data(nullptr), _lowerBand(0), _upperBand(0), _rows(0)
	{
	}
	~BandMatrix() CADET_NOEXCEPT
	{
		delete[] _data;
	}

	BandMatrix(const BandMatrix& cpy)
		: _data(new double[cpy.stride() * cpy._rows]), _lowerBand(cpy._lowerBand), _upperBand(cpy._upperBand),
		  _rows(cpy._rows)
	{
		copyValues(cpy._data);
	}

	BandMatrix(BandMatrix&& cpy) CADET_NOEXCEPT : _data(cpy._data),
												  _lowerBand(cpy._lowerBand),
												  _upperBand(cpy._upperBand),
												  _rows(cpy._rows)
	{
		cpy._data = nullptr;
	}

	inline BandMatrix& operator=(const BandMatrix& cpy)
	{
		cadet_assert(&cpy != this);

		_lowerBand = cpy._lowerBand;
		_upperBand = cpy._upperBand;
		_rows = cpy._rows;

		delete[] _data;
		_data = new double[cpy.stride() * cpy._rows];

		copyValues(cpy._data);
		return *this;
	}

	inline BandMatrix& operator=(BandMatrix&& cpy) CADET_NOEXCEPT
	{
		_lowerBand = cpy._lowerBand;
		_upperBand = cpy._upperBand;
		_rows = cpy._rows;

		delete[] _data;
		_data = cpy._data;
		cpy._data = nullptr;

		return *this;
	}

	/**
	 * @brief Resizes the matrix to the given size
	 * @details All data is lost in this operation. Note that the allocated memory also
	 *          accounts for the main diagonal which is not counted in @p lowerBand or @p upperBand.
	 *
	 * @param [in] rows Number of rows
	 * @param [in] lowerBand Number of lower diagonals (excluding the main diagonal)
	 * @param [in] upperBand Number of upper diagonals (excluding the main diagonal)
	 */
	inline void resize(int rows, int lowerBand, int upperBand)
	{
		delete[] _data;

		_rows = rows;
		_lowerBand = lowerBand;
		_upperBand = upperBand;

		// Do not forget the main diagonal
		_data = new double[stride() * _rows];
		setAll(0.0);
	}

	/**
	 * @brief Repartitions the matrix by changing lower and upper bandwidth
	 * @details The number of nonzero elements per row (i.e., sum of lower and upper bandwidth)
	 *          must not change. This way, all of the memory is reused.
	 *          Note that the data is not reset to @c 0.0 by this operation.
	 *
	 * @param [in] lowerBand Number of lower diagonals (excluding the main diagonal)
	 * @param [in] upperBand Number of upper diagonals (excluding the main diagonal)
	 */
	inline void repartition(int lowerBand, int upperBand)
	{
		cadet_assert(lowerBand + upperBand == _lowerBand + _upperBand);
		_lowerBand = lowerBand;
		_upperBand = upperBand;
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
		return (*this)(row, diagonal);
	}
	inline double centered(int row, int diagonal) const
	{
		return (*this)(row, diagonal);
	}

	/**
	 * @brief Accesses an element in the matrix where the lowest diagonal is indexed by @c 0
	 * @details In contrast to centered() the index of the column starts with the lowest diagonal (@c 0).
	 *          The main diagonal is, thus, retrieved for @c lowerBand and the highest upper diagonal
	 *          is returned for `lowerBand + upperBand`.
	 *
	 * @param [in] row Index of the row
	 * @param [in] col Index of the column (from @c 0 to `lowerBand + upperBand`)
	 *
	 * @return Matrix element at the given position
	 */
	inline double& native(int row, int col)
	{
		cadet_assert(row < _rows);
		cadet_assert(col < stride());
		return _data[row * stride() + col];
	}

	inline double native(int row, int col) const
	{
		cadet_assert(row < _rows);
		cadet_assert(col < stride());
		return _data[row * stride() + col];
	}

	inline double& operator()(int row, int diagonal)
	{
		cadet_assert(row < _rows);
		cadet_assert(diagonal <= static_cast<int>(_upperBand));
		cadet_assert(-diagonal <= static_cast<int>(_lowerBand));
		return _data[static_cast<int>(row * stride() + _lowerBand) + diagonal];
	}

	inline double operator()(int row, int diagonal) const
	{
		cadet_assert(row < _rows);
		cadet_assert(diagonal <= static_cast<int>(_upperBand));
		cadet_assert(-diagonal <= static_cast<int>(_lowerBand));
		return _data[static_cast<int>(row * stride() + _lowerBand) + diagonal];
	}

	/**
	 * @brief Returns the lower bandwidth
	 * @details The lower bandwidth is the number of diagonals below the main diagonal
	 * @return Number of diagonals below the main diagonal
	 */
	inline int lowerBandwidth() const CADET_NOEXCEPT
	{
		return _lowerBand;
	}

	/**
	 * @brief Returns the upper bandwidth
	 * @details The upper bandwidth is the number of diagonals above the main diagonal
	 * @return Number of diagonals above the main diagonal
	 */
	inline int upperBandwidth() const CADET_NOEXCEPT
	{
		return _upperBand;
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
	 * @brief Returns the number of elements in a row
	 * @return Number of elements in a row
	 */
	inline int stride() const CADET_NOEXCEPT
	{
		return _lowerBand + _upperBand + 1;
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

	inline ConstRowIterator row(int idx) const
	{
		cadet_assert(idx < _rows);
		return ConstRowIterator(*this, idx);
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

	inline double const* rowPtr(int idx) const
	{
		cadet_assert(idx < _rows);
		return _data + stride() * idx;
	}

	/**
	 * @brief Multiplies a submatrix @f$ A @f$ with a given vector @f$ x @f$ using LAPACK
	 * @details Computes @f$ y = Ax @f$, where @f$ A @f$ is a submatrix of this matrix and @f$ x @f$ is given.
	 *          The submatrix is given by its first row and diagonal and its number of rows and columns.
	 * @param [in] x Vector the submatrix is multiplied with
	 * @param [in] startRow Index of the first row of the submatrix
	 * @param [in] startDiag Diagonal index of the first element of the submatrix
	 * @param [in] numRows Number of rows of the submatrix
	 * @param [in] numCols Number of columns of the submatrix
	 * @param [out] y Result of the submatrix-vector multiplication
	 */
	inline void submatrixMultiplyVector(const double* const x, int startRow, int startDiag, int numRows, int numCols,
										double* const y) const
	{
		submatrixMultiplyVector(x, startRow, startDiag, numRows, numCols, 1.0, 0.0, y);
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
	 * @brief Multiplies a submatrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another vector using LAPACK
	 * @details Computes @f$ y = \alpha Ax + \beta y @f$, where @f$ A @f$ is a submatrix of this matrix and @f$ x @f$ is
	 * given. The submatrix is given by its first row and diagonal and its number of rows and columns.
	 * @param [in] x Vector the submatrix is multiplied with
	 * @param [in] startRow Index of the first row of the submatrix
	 * @param [in] startDiag Diagonal index of the first element of the submatrix
	 * @param [in] numRows Number of rows of the submatrix
	 * @param [in] numCols Number of columns of the submatrix
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [out] y Result of the submatrix-vector multiplication
	 */
	void submatrixMultiplyVector(const double* const x, int startRow, int startDiag, int numRows, int numCols,
								 double alpha, double beta, double* const y) const;

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another vector using LAPACK
	 * @details Computes @f$ y = \alpha Ax + \beta y@f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	void multiplyVector(const double* const x, double alpha, double beta, double* const y) const;

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

protected:
	double* _data;  //!< Pointer to the array in which the matrix is stored
	int _lowerBand; //!< Lower bandwidth excluding main diagonal
	int _upperBand; //!< Upper bandwidth excluding main diagonal
	int _rows;      //!< Number of rows

	/**
	 * @brief Returns the number of true elements in a row (same as stride())
	 * @return Number of true elements in a row
	 */
	inline int apparentStride() const CADET_NOEXCEPT
	{
		return stride();
	}

	/**
	 * @brief Copies all values from the source to the local array
	 * @param [in] src Data to be copied
	 */
	inline void copyValues(double const* const src)
	{
		std::copy(src, src + stride() * _rows, _data);
	}
};

std::ostream& operator<<(std::ostream& out, const BandMatrix& bm);

/**
 * @brief Represents a factorizable band matrix with given number of rows and diagonals
 * @details LAPACK uses column-major storage, whereas this class uses row-major.
 *          Thus, what we call a row here is actually a column for LAPACK.
 *          Concluding, we have to use the transposed LAPACK operations for
 *          solution and matrix-vector multiplication. The ordering is irrelevant
 *          for the factorization.
 *
 *          Because of the transposition induced by the differing ordering,
 *          the number of upper and lower diagonals switches (e.g., upper diagonals
 *          are transposed lower diagonals). The ordering of the elements inside
 *          one (original) row is maintained (i.e., the first element in a row becomes
 *          the first element in a column and the last element in a row transposes to
 *          the last element in a column).
 *
 *          LAPACK needs additional space to hold intermediate values when calling a
 *          factorization routine. This space, and the required pivoting arrays, are
 *          also stored in this class.
 * @todo Refactor and combine code with BandMatrix in order to save LOC
 */
class FactorizableBandMatrix
{
public:
	typedef BandedRowIterator<FactorizableBandMatrix> RowIterator;
	friend class BandedRowIterator<FactorizableBandMatrix>;
	typedef ConstBandedRowIterator<FactorizableBandMatrix> ConstRowIterator;
	friend class ConstBandedRowIterator<FactorizableBandMatrix>;

	/**
	 * @brief Creates an empty, unitialized band matrix
	 * @details No memory is allocated for the matrix. Users have to call resize() first.
	 */
	FactorizableBandMatrix() CADET_NOEXCEPT : _data(nullptr),
											  _lowerBand(0),
											  _upperBand(0),
											  _rows(0),
											  _capacity(0),
											  _pivot(nullptr)
	{
	}
	~FactorizableBandMatrix() CADET_NOEXCEPT
	{
		delete[] _pivot;
		delete[] _data;
	}

	FactorizableBandMatrix(const FactorizableBandMatrix& cpy)
		: _data(new double[cpy.stride() * cpy._rows]), _lowerBand(cpy._lowerBand), _upperBand(cpy._upperBand),
		  _rows(cpy._rows), _capacity(cpy._capacity), _pivot(new lapackInt_t[cpy._rows])
	{
		copyValues(cpy._data);
		copyPivot(cpy._pivot);
	}

	FactorizableBandMatrix(FactorizableBandMatrix&& cpy) CADET_NOEXCEPT : _data(cpy._data),
																		  _lowerBand(cpy._lowerBand),
																		  _upperBand(cpy._upperBand),
																		  _rows(cpy._rows),
																		  _capacity(cpy._capacity),
																		  _pivot(cpy._pivot)
	{
		cpy._data = nullptr;
		cpy._pivot = nullptr;
	}

	inline FactorizableBandMatrix& operator=(const BandMatrix& cpy)
	{
		from(cpy);
		return *this;
	}

	inline FactorizableBandMatrix& operator=(const FactorizableBandMatrix& cpy)
	{
		cadet_assert(&cpy != this);

		_lowerBand = cpy._lowerBand;
		_upperBand = cpy._upperBand;
		_rows = cpy._rows;
		_capacity = cpy._capacity;

		delete[] _data;
		_data = new double[_capacity];
		copyValues(cpy._data);

		delete[] _pivot;
		_pivot = new lapackInt_t[_rows];
		copyPivot(cpy._pivot);

		return *this;
	}

	inline FactorizableBandMatrix& operator=(FactorizableBandMatrix&& cpy) CADET_NOEXCEPT
	{
		_lowerBand = cpy._lowerBand;
		_upperBand = cpy._upperBand;
		_rows = cpy._rows;
		_capacity = cpy._capacity;

		delete[] _data;
		_data = cpy._data;
		cpy._data = nullptr;

		delete[] _pivot;
		_pivot = cpy._pivot;
		cpy._pivot = nullptr;

		return *this;
	}

	/**
	 * @brief Resizes the matrix to the given size
	 * @details All data is lost in this operation. Note that the allocated memory also
	 *          accounts for the main diagonal which is not counted in @p lowerBand or @p upperBand.
	 *
	 * @param [in] rows Number of rows
	 * @param [in] lowerBand Number of lower diagonals (excluding the main diagonal)
	 * @param [in] upperBand Number of upper diagonals (excluding the main diagonal)
	 */
	inline void resize(int rows, int lowerBand, int upperBand)
	{
		// Allocating memory is the default case
		if (cadet_unlikely(_capacity >= rows * stride(lowerBand, upperBand)))
		{
			// Use existing memory

			// Capacity does not capture memory for pivot indices
			if (_rows < rows)
			{
				delete[] _pivot;
				_pivot = new lapackInt_t[rows];
			}

			_rows = rows;
			_lowerBand = lowerBand;
			_upperBand = upperBand;
			setAll(0.0);
		}
		else
		{
			// Allocate new memory
			_rows = rows;
			_lowerBand = lowerBand;
			_upperBand = upperBand;
			_capacity = stride() * _rows;

			delete[] _data;
			_data = new double[_capacity];
			setAll(0.0);

			delete[] _pivot;
			_pivot = new lapackInt_t[_rows];
		}
	}

	/**
	 * @brief Repartitions the matrix by changing lower and upper bandwidth
	 * @details There has to be enough capacity for the new matrix size. Use resize() if the capacity is too small.
	 *          Note that the data is not reset to @c 0.0 by this operation.
	 *
	 * @param [in] lowerBand Number of lower diagonals (excluding the main diagonal)
	 * @param [in] upperBand Number of upper diagonals (excluding the main diagonal)
	 */
	inline void repartition(int lowerBand, int upperBand)
	{
		cadet_assert(_capacity >= _rows * stride(lowerBand, upperBand));
		_lowerBand = lowerBand;
		_upperBand = upperBand;
	}

	/**
	 * @brief Copies the given BandMatrix into this factorizable matrix
	 * @details All content is lost and overwritten with the data of the given BandMatrix.
	 *          The memory is reallocated in order to hold the given BandMatrix.
	 * @param [in] bdm BandMatrix that is copied
	 */
	inline void from(const BandMatrix& bdm)
	{
		_rows = bdm.rows();
		_lowerBand = bdm.lowerBandwidth();
		_upperBand = bdm.upperBandwidth();
		_capacity = stride() * _rows;

		delete[] _data;
		_data = new double[_capacity];

		// Copy data over and take into account that the local storage is larger
		copyOver(bdm);

		delete[] _pivot;
		_pivot = new lapackInt_t[_rows];
	}

	/**
	 * @brief Copies the given BandMatrix into this factorizable matrix without resizing it
	 * @details All content is lost and overwritten with the data of the given BandMatrix.
	 *          The factorizable matrix has to be as big as the given BandMatrix.
	 * @param [in] bdm BandMatrix that is copied
	 */
	inline void copyOver(const BandMatrix& bdm)
	{
		cadet_assert(_rows == bdm.rows());
		cadet_assert(_lowerBand == bdm.lowerBandwidth());
		cadet_assert(_upperBand >= bdm.upperBandwidth());

		if (cadet_likely(_upperBand == bdm.upperBandwidth()))
		{
			// Copy data over and take into account that the local storage is larger and that
			// LAPACK needs the first _upperBand values in each row as additional storage
			const double* src = bdm.data();
			double* local = _data + _upperBand;
			const int as = apparentStride();
			const int ls = stride();
			for (int i = 0; i < _rows; ++i, local += ls, src += as)
			{
				std::copy_n(src, as, local);
			}
		}
		else
		{
			// Only copy what bdm has and zero out the remaining entries
			const double* src = bdm.data();
			double* local = _data + _upperBand;
			const int srcStride = bdm.stride();
			const int diff = apparentStride() - srcStride;
			const int ls = stride();
			for (int i = 0; i < _rows; ++i, local += ls, src += srcStride)
			{
				std::copy_n(src, srcStride, local);
				std::fill_n(local + srcStride, diff, 0.0);
			}
		}
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
		return (*this)(row, diagonal);
	}
	inline double centered(int row, int diagonal) const
	{
		return (*this)(row, diagonal);
	}

	/**
	 * @brief Accesses an element in the matrix where the lowest diagonal is indexed by @c 0
	 * @details In contrast to centered() the index of the column starts with the lowest diagonal (@c 0).
	 *          The main diagonal is, thus, retrieved for @c lowerBand and the highest upper diagonal
	 *          is returned for `lowerBand + upperBand`.
	 *
	 * @param [in] row Index of the row
	 * @param [in] col Index of the column (from @c 0 to `lowerBand + upperBand`)
	 *
	 * @return Matrix element at the given position
	 */
	inline double& native(int row, int col)
	{
		cadet_assert(row < _rows);
		cadet_assert(col < stride());
		return _data[row * stride() + col];
	}

	inline double native(int row, int col) const
	{
		cadet_assert(row < _rows);
		cadet_assert(col < stride());
		return _data[row * stride() + col];
	}

	inline double& operator()(int row, int diagonal)
	{
		cadet_assert(row < _rows);
		cadet_assert(diagonal <= static_cast<int>(_upperBand));
		cadet_assert(-diagonal <= static_cast<int>(_lowerBand));
		// Note the additional _upperBand which moves us out of LAPACK's additional memory
		return _data[static_cast<int>(row * stride() + _lowerBand + _upperBand) + diagonal];
	}

	inline double operator()(int row, int diagonal) const
	{
		cadet_assert(row < _rows);
		cadet_assert(diagonal <= static_cast<int>(_upperBand));
		cadet_assert(-diagonal <= static_cast<int>(_lowerBand));
		// Note the additional _upperBand which moves us out of LAPACK's additional memory
		return _data[static_cast<int>(row * stride() + _lowerBand + _upperBand) + diagonal];
	}

	/**
	 * @brief Returns the lower bandwidth
	 * @details The lower bandwidth is the number of diagonals below the main diagonal
	 * @return Number of diagonals below the main diagonal
	 */
	inline int lowerBandwidth() const CADET_NOEXCEPT
	{
		return _lowerBand;
	}

	/**
	 * @brief Returns the upper bandwidth
	 * @details The upper bandwidth is the number of diagonals above the main diagonal
	 * @return Number of diagonals above the main diagonal
	 */
	inline int upperBandwidth() const CADET_NOEXCEPT
	{
		return _upperBand;
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

	inline lapackInt_t* pivot() CADET_NOEXCEPT
	{
		return _pivot;
	}
	inline lapackInt_t const* pivot() const CADET_NOEXCEPT
	{
		return _pivot;
	}

	/**
	 * @brief Returns the total number of elements in a row including additional storage for factorization
	 * @return Total number of elements in a row
	 */
	inline int stride() const CADET_NOEXCEPT
	{
		return stride(_lowerBand, _upperBand);
	}

	/**
	 * @brief Creates a RowIterator pointing to the given row
	 * @param [in] idx Index of the row
	 * @return RowIterator pointing to the given row
	 */
	inline RowIterator row(int idx)
	{
		cadet_assert(idx < _rows);
		return RowIterator(*this, idx, _upperBand);
	}

	inline ConstRowIterator row(int idx) const
	{
		cadet_assert(idx < _rows);
		return ConstRowIterator(*this, idx, _upperBand);
	}

	/**
	 * @brief Provides access to the underlying data in the given row
	 * @param [in] idx Index of the row
	 * @return Pointer to first element in the given row
	 */
	inline double* rowPtr(int idx)
	{
		cadet_assert(idx < _rows);
		return _data + stride() * idx + _upperBand;
	}

	inline double const* rowPtr(int idx) const
	{
		cadet_assert(idx < _rows);
		return _data + stride() * idx + _upperBand;
	}

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ using LAPACK
	 * @details Computes @f$ y = Ax @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	void multiplyVector(const double* const x, double* const y) const;

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another vector using LAPACK
	 * @details Computes @f$ y = \alpha Ax + \beta y@f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	void multiplyVector(const double* const x, double alpha, double beta, double* const y) const;

	/**
	 * @brief Factorizes the BandMatrix using LAPACK (performs LU factorization)
	 * @return @c true if the factorization was successful, otherwise @c false
	 */
	bool factorize();

	/**
	 * @brief Uses the factorized matrix to solve the equation @f$ Ax = b @f$ with LAPACK
	 * @details Before the equation can be solved, the matrix has to be factorized first by calling factorize().
	 * @param [in,out] rhs On entry pointer to the right hand side vector @f$ b @f$ of the equation, on exit the
	 * solution @f$ x @f$
	 * @return @c true if the solution process was successful, otherwise @c false
	 */
	bool solve(double* rhs) const;

	/**
	 * @brief Uses the factorized matrix to solve the equation @f$ Ax = b @f$ with LAPACK
	 * @details Before the equation can be solved, the matrix has to be factorized first by calling factorize().
	 *          It is assumed that row scaling has been applied to the matrix before factorization.
	 *          In order to solve the equation system, the right hand side has to be scaled accordingly.
	 *          This is handled automatically by passing the required scaling factors.
	 * @param [in] scalingFactors Vector with scaling factor for each row
	 * @param [in,out] rhs On entry pointer to the right hand side vector @f$ b @f$ of the equation, on exit the
	 * solution @f$ x @f$
	 * @return @c true if the solution process was successful, otherwise @c false
	 */
	bool solve(double const* scalingFactors, double* rhs) const;

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

protected:
	double* _data;       //!< Pointer to the array in which the matrix is stored
	int _lowerBand;      //!< Lower bandwidth excluding main diagonal
	int _upperBand;      //!< Upper bandwidth excluding main diagonal
	int _rows;           //!< Number of rows
	int _capacity;       //!< Allocated memory in sizeof(double)
	lapackInt_t* _pivot; //!< Pointer to an array which is used for pivoting by factorization methods

	/**
	 * @brief Returns the total number of elements in a row including additional storage for factorization
	 * @param [in] lowerBand Number of lower diagonals (excluding the main diagonal)
	 * @param [in] upperBand Number of upper diagonals (excluding the main diagonal)
	 * @return Total number of elements in a row
	 */
	inline int stride(int lowerBand, int upperBand) const CADET_NOEXCEPT
	{
		return lowerBand + 2 * upperBand + 1;
	}

	/**
	 * @brief Returns the number of true elements in a row not counting additional storage for factorization
	 * @return Number of true elements in a row
	 */
	inline int apparentStride() const CADET_NOEXCEPT
	{
		return _lowerBand + _upperBand + 1;
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
		std::copy(src, src + _rows, _pivot);
	}
};

std::ostream& operator<<(std::ostream& out, const FactorizableBandMatrix& fbm);

} // namespace linalg

} // namespace cadet

#endif // LIBCADET_BANDMATRIX_HPP_
