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
 * Defines a rectangular dense matrix comprised of AD elements
 */

#ifndef LIBCADET_ACTIVEDENSEMATRIX_HPP_
#define LIBCADET_ACTIVEDENSEMATRIX_HPP_

#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"
#include "AutoDiff.hpp"

#include <algorithm>

namespace cadet
{

namespace linalg
{

/**
 * @brief Represents a dense matrix base class providing common functionality
 * @details This class uses row-major storage ordering. It does not provide
 *          factorization functionality or other means to solve linear systems.
 */
class ActiveDenseMatrix
{
public:
	/**
	 * @brief Creates an empty, unitialized matrix
	 * @details No memory is allocated for the matrix. Users have to call resize() first.
	 */
	ActiveDenseMatrix() CADET_NOEXCEPT : _data(nullptr), _rows(0), _cols(0)
	{
	}
	~ActiveDenseMatrix() CADET_NOEXCEPT
	{
		delete[] _data;
	}

	ActiveDenseMatrix(const ActiveDenseMatrix& cpy)
		: ActiveDenseMatrix(new active[cpy.stride() * cpy._rows], cpy._rows, cpy._cols)
	{
		copyValues(cpy._data);
	}

	ActiveDenseMatrix(ActiveDenseMatrix&& cpy) CADET_NOEXCEPT : ActiveDenseMatrix(cpy._data, cpy._rows, cpy._cols)
	{
		cpy._data = nullptr;
	}

	inline ActiveDenseMatrix& operator=(const ActiveDenseMatrix& cpy)
	{
		cadet_assert(&cpy != this);

		// Determine if reallocation is necessary
		const bool reAllocMat = (cpy.stride() * cpy._rows) > stride() * _rows;

		// Change size and copy all the stuff
		_cols = cpy._cols;
		_rows = cpy._rows;

		if (reAllocMat)
		{
			delete[] _data;
			_data = new active[cpy.stride() * cpy._rows];
		}
		copyValues(cpy._data);

		return *this;
	}

	inline ActiveDenseMatrix& operator=(ActiveDenseMatrix&& cpy) CADET_NOEXCEPT
	{
		_cols = cpy._cols;
		_rows = cpy._rows;

		delete[] _data;
		_data = cpy._data;
		cpy._data = nullptr;

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
	inline void copyFrom(const ActiveDenseMatrix& cpy)
	{
		cadet_assert(&cpy != this);

		cadet_assert(stride() * _rows >= cpy.stride() * cpy.rows());
		cadet_assert(std::min(_rows, _cols) >= std::min(cpy.rows(), cpy.columns()));

		_rows = cpy.rows();
		_cols = cpy.columns();

		copyValues(cpy.data());
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
		_data = new active[stride() * _rows];

		setAll(0.0);
	}

	/**
	 * @brief Sets all matrix elements to the given value
	 * @param [in] val Value all matrix elements are set to
	 */
	inline void setAll(const active& val)
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
	inline active& diagonalElement(int row, int diagonal)
	{
		return (*this)(row, diagonal);
	}
	inline const active& diagonalElement(int row, int diagonal) const
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
	inline active& native(int row, int col)
	{
		cadet_assert(row < _rows);
		cadet_assert(col < _cols);
		return _data[row * stride() + col];
	}

	inline const active& native(int row, int col) const
	{
		cadet_assert(row < _rows);
		cadet_assert(col < _cols);
		return _data[row * stride() + col];
	}

	inline active& operator()(int row, int diagonal)
	{
		cadet_assert(row < _rows);
		cadet_assert(diagonal < static_cast<int>(_cols - row));
		cadet_assert(-diagonal <= static_cast<int>(row));
		return _data[static_cast<int>(row * stride() + row) + diagonal];
	}

	inline const active& operator()(int row, int diagonal) const
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
	inline active* data() CADET_NOEXCEPT
	{
		return _data;
	}
	inline active const* data() const CADET_NOEXCEPT
	{
		return _data;
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
	inline active* rowPtr(int idx)
	{
		cadet_assert(idx < _rows);
		return _data + stride() * idx;
	}

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$
	 * @details Computes @f$ y = Ax @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	template <typename operand_t, typename result_t>
	inline void multiplyVector(const operand_t* const x, result_t* const y) const
	{
		for (int r = 0; r < _rows; ++r)
		{
			y[r] = 0.0;
			active const* const row = _data + r * stride();
			for (int c = 0; c < _cols; ++c)
				y[r] += static_cast<typename DoubleDemoter<result_t>::type>(row[c]) *
						static_cast<typename DoubleActiveDemoter<result_t, operand_t>::type>(x[c]);
		}
	}

	/**
	 * @brief Multiplies the transpose of the matrix @f$ A @f$ with a given vector @f$ x @f$
	 * @details Computes @f$ y = A^Tx @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	template <typename operand_t, typename result_t>
	inline void transposedMultiplyVector(const operand_t* const x, result_t* const y) const
	{
		for (int r = 0; r < _rows; ++r)
		{
			y[r] = 0.0;
			active const* const col = _data + r;
			for (int c = 0; c < _cols; ++c)
				y[r] += static_cast<typename DoubleDemoter<result_t>::type>(col[c * stride()]) *
						static_cast<typename DoubleActiveDemoter<result_t, operand_t>::type>(x[c]);
		}
	}

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another vector
	 * @details Computes @f$ y = \alpha Ax + \beta y @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	template <typename operand_t, typename result_t>
	void multiplyVector(const operand_t* const x, double alpha, double beta, result_t* const y) const
	{
		for (int r = 0; r < _rows; ++r)
		{
			y[r] *= beta;
			active const* const row = _data + r * stride();
			for (int c = 0; c < _cols; ++c)
				y[r] += alpha * static_cast<typename DoubleDemoter<result_t>::type>(row[c]) *
						static_cast<typename DoubleActiveDemoter<result_t, operand_t>::type>(x[c]);
		}
	}

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another vector
	 * @details Computes @f$ y = \alpha Ax + y @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	template <typename operand_t, typename result_t>
	void multiplyVector(const operand_t* const x, double alpha, result_t* const y) const
	{
		for (int r = 0; r < _rows; ++r)
		{
			active const* const row = _data + r * stride();
			for (int c = 0; c < _cols; ++c)
				y[r] += alpha * static_cast<typename DoubleDemoter<result_t>::type>(row[c]) *
						static_cast<typename DoubleActiveDemoter<result_t, operand_t>::type>(x[c]);
		}
	}

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another vector
	 * @details Computes @f$ y = \alpha Ax + y @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	template <typename operand_t, typename result_t>
	void multiplyVector(const operand_t* const x, const active& alpha, result_t* const y) const
	{
		for (int r = 0; r < _rows; ++r)
		{
			active const* const row = _data + r * stride();
			for (int c = 0; c < _cols; ++c)
				y[r] += static_cast<typename DoubleDemoter<result_t>::type>(alpha) *
						static_cast<typename DoubleDemoter<result_t>::type>(row[c]) *
						static_cast<typename DoubleActiveDemoter<result_t, operand_t>::type>(x[c]);
		}
	}

	/**
	 * @brief Multiplies the transpose of the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another
	 * vector
	 * @details Computes @f$ y = \alpha A^T x + \beta y @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ A^T x @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	template <typename operand_t, typename result_t>
	void transposedMultiplyVector(const operand_t* const x, double alpha, double beta, result_t* const y) const
	{
		for (int r = 0; r < _rows; ++r)
		{
			y[r] *= beta;
			active const* const col = _data + r;
			for (int c = 0; c < _cols; ++c)
				y[r] += alpha * static_cast<typename DoubleDemoter<result_t>::type>(col[c * stride()]) *
						static_cast<typename DoubleActiveDemoter<result_t, operand_t>::type>(x[c]);
		}
	}

	/**
	 * @brief Multiplies the transpose of the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another
	 * vector
	 * @details Computes @f$ y = \alpha A^T x + y @f$, where @f$ A @f$ is this matrix and @f$ x @f$ is given.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ A^T x @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	template <typename operand_t, typename result_t>
	void transposedMultiplyVector(const operand_t* const x, double alpha, result_t* const y) const
	{
		for (int r = 0; r < _rows; ++r)
		{
			active const* const col = _data + r;
			for (int c = 0; c < _cols; ++c)
				y[r] += alpha * static_cast<typename DoubleDemoter<result_t>::type>(col[c * stride()]) *
						static_cast<typename DoubleActiveDemoter<result_t, operand_t>::type>(x[c]);
		}
	}

	/**
	 * @brief Multiplies the matrix @f$ A @f$ with a given vector @f$ x @f$ and adds it to another vector
	 * @details For given vector @f$ x @f$, compute
	 *          @f[ \begin{align}
	 *                  y_1 &= \alpha_1 (Ax)_1 + \beta y_1 \\
	 *                  y_2 &= \alpha_2 (Ax)_2 + \beta y_2,
	 *          \end{align} @f]
	 *          where the resulting vector is partitioned into two parts.
	 * @param [in] x Vector this matrix is multiplied with
	 * @param [in] alpha1 Factor @f$ \alpha_1 @f$ in front of @f$ (Ax)_1 @f$
	 * @param [in] alpha2 Factor @f$ \alpha_2 @f$ in front of @f$ (Ax)_2 @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [in] idxSplit Index of the first element of @f$ y_2 @f$
	 * @param [out] y Result of the matrix-vector multiplication
	 */
	template <typename operand_t, typename result_t, typename numeric_t>
	void multiplyVectorSplit(const operand_t* const x, double alpha1, const numeric_t& alpha2, double beta,
							 int idxSplit, result_t* const y) const
	{
		for (int r = 0; r < idxSplit; ++r)
		{
			y[r] *= beta;
			active const* const row = _data + r * stride();
			for (int c = 0; c < _cols; ++c)
				y[r] += alpha1 * static_cast<typename DoubleDemoter<result_t>::type>(row[c]) *
						static_cast<typename DoubleActiveDemoter<result_t, operand_t>::type>(x[c]);
		}

		for (int r = idxSplit; r < _rows; ++r)
		{
			y[r] *= beta;
			active const* const row = _data + r * stride();
			for (int c = 0; c < _cols; ++c)
				y[r] += alpha2 * static_cast<typename DoubleDemoter<result_t>::type>(row[c]) *
						static_cast<typename DoubleActiveDemoter<result_t, operand_t>::type>(x[c]);
		}
	}

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
	active* _data; //!< Pointer to the array in which the matrix is stored
	int _rows;     //!< Number of rows
	int _cols;     //!< Number of columns

	/**
	 * @brief Initializes the matrix with the given memory of the given size
	 * @param [in] data Pointer to data array of size `rows * cols`
	 * @param [in] rows Number of rows
	 * @param [in] cols Number of columns
	 */
	ActiveDenseMatrix(active* const data, int rows, int cols) CADET_NOEXCEPT : _data(data), _rows(rows), _cols(cols)
	{
	}

	/**
	 * @brief Copies all values from the source to the local array
	 * @param [in] src Data to be copied
	 */
	inline void copyValues(active const* const src)
	{
		std::copy(src, src + stride() * _rows, _data);
	}
};

} // namespace linalg

} // namespace cadet

#endif // LIBCADET_ACTIVEDENSEMATRIX_HPP_
