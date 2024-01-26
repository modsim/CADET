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
 * Defines a sparse matrix
 */

#ifndef LIBCADET_SPARSEMATRIX_HPP_
#define LIBCADET_SPARSEMATRIX_HPP_

#include <vector>
#include <ostream>

#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"

namespace cadet
{

namespace linalg
{

/**
 * @brief Represents a sparse matrix in coordinate list format (i.e., storage is a list of [row, column, value] tuples)
 * @details The elements of the SparseMatrix can be accessed by operator() and addElement(). If operator() is used, a
 *          lookup is performed first. If the element is found, it is returned. Otherwise, a new element at the given
 *          position is added. Contrary to operator(), addElement() will always add a new element and does not check if
 *          it already exists.
 *
 *          This matrix format is meant as intermediate format for constructing a sparse matrix. Users are encouraged to
 *          convert their SparseMatrix to a CompressedSparseMatrix, which requires significantly less storage.
 * @tparam real_t Type of the stored elements
 */
template <class real_t> class SparseMatrix
{
public:
	/**
	 * @brief Creates an empty SparseMatrix with capacity @c 0
	 * @details Users have to call resize() prior to populating the matrix.
	 */
	SparseMatrix() CADET_NOEXCEPT : _curIdx(0)
	{
	}

	/**
	 * @brief Creates an empty SparseMatrix with the given capacity
	 * @param [in] nnz Capacity, that is the maximum number of non-zero elements
	 */
	SparseMatrix(unsigned int nnz) : _curIdx(0)
	{
		resize(nnz);
	}

	~SparseMatrix() CADET_NOEXCEPT
	{
	}

	// Default copy and assignment semantics
	SparseMatrix(const SparseMatrix& cpy) = default;
	SparseMatrix(SparseMatrix&& cpy) CADET_NOEXCEPT = default;

	SparseMatrix& operator=(const SparseMatrix& cpy) = default;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	SparseMatrix& operator=(SparseMatrix&& cpy) CADET_NOEXCEPT = default;
#else
	SparseMatrix& operator=(SparseMatrix&& cpy) = default;
#endif

	/**
	 * @brief Copies a SparseMatrix of different type
	 * @details The cast from @c otherReal_t to @c real_t has to be possible.
	 * @param [in] cpy Source matrix to be copied
	 * @tparam otherReal_t Element type of source matrix
	 */
	template <class otherReal_t>
	SparseMatrix(const SparseMatrix<otherReal_t>& cpy) : _rows(cpy._rows), _cols(cpy._cols), _curIdx(cpy._curIdx)
	{
		_values.reserve(cpy._values.size());
		for (std::size_t i = 0; i < cpy._values.size(); ++i)
			_values.push_back(static_cast<real_t>(cpy._values[i]));
	}

	/**
	 * @brief Copies a SparseMatrix
	 * @param [in] src Source matrix to be copied
	 */
	inline void copyFrom(const SparseMatrix<real_t>& src)
	{
		_rows = src.rows();
		_cols = src.cols();
		_curIdx = src.numNonZero();
		_values = src.values();
	}

	/**
	 * @brief Copies a SparseMatrix of different type
	 * @details The cast from @c otherReal_t to @c real_t has to be possible.
	 * @param [in] src Source matrix to be copied
	 * @tparam otherReal_t Element type of source matrix
	 */
	template <class otherReal_t> inline void copyFrom(const SparseMatrix<otherReal_t>& src)
	{
		_rows = src.rows();
		_cols = src.cols();
		_curIdx = src.numNonZero();

		const std::vector<otherReal_t>& srcVals = src.values();
		_values.clear();
		_values.reserve(srcVals.size());
		for (std::size_t i = 0; i < srcVals.size(); ++i)
			_values.push_back(static_cast<real_t>(srcVals[i]));
	}

	/**
	 * @brief Resets all elements to @c 0
	 * @details The capacity of the SparseMatrix is not changed.
	 */
	inline void clear() CADET_NOEXCEPT
	{
		_curIdx = 0;
	}

	/**
	 * @brief Resets the maximum number of non-zero elements / the capacity
	 * @details The matrix is reset to an empty state. All previous content is lost.
	 *
	 * @param [in] nnz Maximum number of non-zero elements
	 */
	inline void resize(unsigned int nnz)
	{
		_rows.clear();
		_rows.resize(nnz);

		_cols.clear();
		_cols.resize(nnz);

		_values.clear();
		_values.resize(nnz);

		_curIdx = 0;
	}

	/**
	 * @brief Returns the capacity, that is the maximum number of non-zero elements which can be stored in the matrix
	 * @details Note that the capacity is not the current number of non-zero elements.
	 * @return Maximum number of non-zero elements that can be stored in the matrix
	 */
	inline unsigned int capacity() const CADET_NOEXCEPT
	{
		return _rows.size();
	}

	/**
	 * @brief Inserts a new element at the given position to the given value
	 * @details If the element does not exist and the capacity is not exhausted,
	 *          a new element is created. As the capacity is not increased by
	 *          this method, it will fail when the capacity is exhausted and
	 *          a new element would have to be created.
	 *
	 * @param [in] row Row index
	 * @param [in] col Column index
	 * @param [in] val Value of the element at the given position
	 */
	inline void addElement(int row, int col, const real_t& val)
	{
		cadet_assert(_curIdx < _rows.size());

		_rows[_curIdx] = row;
		_cols[_curIdx] = col;
		_values[_curIdx] = val;

		++_curIdx;
	}

	/**
	 * @brief Accesses an element at the given position
	 * @details If the element does not exist and the capacity is not exhausted,
	 *          a new element is created. As the capacity is not increased by
	 *          this method, it will fail when the capacity is exhausted and
	 *          a new element would have to be created.
	 *
	 * @param [in] row Row index
	 * @param [in] col Column index
	 * @return Value of the element at the given position
	 */
	inline real_t& operator()(int row, int col)
	{
		// Try to find the element
		for (unsigned int i = 0; i < _curIdx; ++i)
		{
			if ((_rows[i] == row) && (_cols[i] == col))
				return _values[i];
		}

		// Value not found, so add it
		cadet_assert(_curIdx < _rows.size());

		_rows[_curIdx] = row;
		_cols[_curIdx] = col;
		_values[_curIdx] = 0.0;

		++_curIdx;

		return _values[_curIdx - 1];
	}

	/**
	 * @brief Accesses an element at the given position
	 * @details If the element does not exist, @ç 0.0 is returned.
	 * @param [in] row Row index
	 * @param [in] col Column index
	 * @return Value of the element at the given position
	 */
	inline const real_t operator()(int row, int col) const
	{
		// Try to find the element
		for (unsigned int i = 0; i < _curIdx; ++i)
		{
			if ((_rows[i] == row) && (_cols[i] == col))
				return _values[i];
		}
		return real_t();
	}

	/**
	 * @brief Multiplies this sparse matrix with a vector
	 * @details Computes the matrix vector operation \f$y = Ax. \f$
	 * @param [in] x Vector @f$ x @f$ to multiply with
	 * @param [out] out Vector @f$ y @f$ to write to
	 * @tparam arg_t Type of the vector \f$ x \f$
	 * @tparam result_t Type of the vector \f$ y \f$
	 */
	template <typename arg_t, typename result_t>
	inline void multiplyVector(arg_t const* const x, result_t* const out) const
	{
		for (unsigned int i = 0; i < _curIdx; ++i)
			out[_rows[i]] = _values[i] * x[_cols[i]];
	}

	/**
	 * @brief Multiplies this sparse matrix with a vector
	 * @details Computes the matrix vector operation \f$y = \alpha Ax. \f$
	 * @param [in] x Vector @f$ x @f$ to multiply with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [out] out Vector @f$ y @f$ to write to
	 * @tparam arg_t Type of the vector \f$ x \f$
	 * @tparam result_t Type of the vector \f$ y \f$
	 */
	template <typename arg_t, typename result_t>
	inline void multiplyVector(arg_t const* const x, double alpha, result_t* const out) const
	{
		for (unsigned int i = 0; i < _curIdx; ++i)
			out[_rows[i]] = alpha * _values[i] * x[_cols[i]];
	}

	/**
	 * @brief Multiplies this sparse matrix with a vector and adds another vector to it
	 * @details Computes the matrix vector operation \f$y = \alpha Ax + \beta y. \f$
	 * @param [in] x Vector @f$ x @f$ to multiply with
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
	 * @param [in,out] out Vector @f$ y @f$ to write to
	 * @tparam arg_t Type of the vector \f$ x \f$
	 * @tparam result_t Type of the vector \f$ y \f$
	 */
	template <typename arg_t, typename result_t>
	inline void multiplyVector(arg_t const* const x, double alpha, double beta, result_t* const out) const
	{
		for (unsigned int i = 0; i < _curIdx; ++i)
			out[_rows[i]] = alpha * _values[i] * x[_cols[i]] + beta * out[_rows[i]];
	}

	/**
	 * @brief Multiplies this sparse matrix with a vector and adds the result to another vector
	 * @details Computes the matrix vector operation \f$ b + Ax \f$, where the matrix vector
	 *          product is added to @p out, which is \f$ b \f$.
	 *
	 * @param [in] x Vector to multiply with
	 * @param [in,out] out Vector to add the matrix-vector product to
	 * @tparam arg_t Type of the vector \f$ x \f$
	 * @tparam result_t Type of the vector \f$ y \f$
	 */
	template <typename arg_t, typename result_t>
	inline void multiplyAdd(arg_t const* const x, result_t* const out) const
	{
		for (unsigned int i = 0; i < _curIdx; ++i)
			out[_rows[i]] += _values[i] * x[_cols[i]];
	}

	/**
	 * @brief Multiplies this sparse matrix with a vector and adds the scaled result to another vector
	 * @details Computes the matrix vector operation \f$ b + \alpha Ax \f$, where the matrix vector
	 *          product is added to @p out, which is \f$ b \f$.
	 *
	 * @param [in] x Vector to multiply with
	 * @param [in,out] out Vector to add the matrix-vector product to
	 * @param [in] alpha Scale factor
	 * @tparam arg_t Type of the vector \f$ x \f$
	 * @tparam result_t Type of the vector \f$ y \f$
	 */
	template <typename arg_t, typename result_t>
	inline void multiplyAdd(arg_t const* const x, result_t* const out, double alpha) const
	{
		for (unsigned int i = 0; i < _curIdx; ++i)
			out[_rows[i]] += alpha * _values[i] * x[_cols[i]];
	}

	/**
	 * @brief Multiplies this sparse matrix with a vector and subtracts the result from another vector
	 * @details Computes the matrix vector operation \f$ b - Ax \f$, where the matrix vector
	 *          product is subtracted from @p out, which is \f$ b \f$.
	 *
	 * @param [in] x Vector to multiply with
	 * @param [in,out] out Vector to subtract the matrix-vector product from
	 * @tparam arg_t Type of the vector \f$ x \f$
	 * @tparam result_t Type of the vector \f$ y \f$
	 */
	template <typename arg_t, typename result_t>
	inline void multiplySubtract(arg_t const* const x, result_t* const out) const
	{
		for (unsigned int i = 0; i < _curIdx; ++i)
			out[_rows[i]] -= _values[i] * x[_cols[i]];
	}

	/**
	 * @brief Multiplies this sparse matrix with a vector and subtracts the scaled result from another vector
	 * @details Computes the matrix vector operation \f$ b - \alpha Ax \f$, where the matrix vector
	 *          product is subtracted from @p out, which is \f$ b \f$.
	 *
	 * @param [in] x Vector to multiply with
	 * @param [in,out] out Vector to subtract the matrix-vector product from
	 * @param [in] alpha Scale factor
	 * @tparam arg_t Type of the vector \f$ x \f$
	 * @tparam result_t Type of the vector \f$ y \f$
	 */
	template <typename arg_t, typename result_t>
	inline void multiplySubtract(arg_t const* const x, result_t* const out, double alpha) const
	{
		for (unsigned int i = 0; i < _curIdx; ++i)
			out[_rows[i]] -= alpha * _values[i] * x[_cols[i]];
	}

	/**
	 * @brief Multiplies a row span of this sparse matrix with a vector and subtracts the result from another vector
	 * @details Computes the matrix vector operation \f$ b - Ax \f$ for the row span [ @p startRow, @pendRow ),
	 *          where \f$ b \f$ is given by @p out, which also receives the result.
	 *
	 * @param [in] x Vector to multiply with
	 * @param [in,out] out Vector to subtract the matrix-vector product from
	 * @param [in] startRow Index of the first row
	 * @param [in] endRow Index one past the last row
	 * @tparam arg_t Type of the vector \f$ x \f$
	 * @tparam result_t Type of the vector \f$ y \f$
	 */
	template <typename arg_t, typename result_t>
	inline void multiplySubtract(arg_t const* const x, result_t* const out, unsigned int startRow,
								 unsigned int endRow) const
	{
		for (unsigned int i = 0; i < _curIdx; ++i)
		{
			const unsigned int r = _rows[i];
			if ((r >= startRow) && (r < endRow))
				out[r] -= _values[i] * x[_cols[i]];
		}
	}

	/**
	 * @brief Returns a vector with row indices
	 * @details Not all elements in the vector are actually set. Only the first numNonZero()
	 *          elements are used.
	 * @return Vector with row indices
	 */
	inline const std::vector<int>& rows() const CADET_NOEXCEPT
	{
		return _rows;
	}

	/**
	 * @brief Returns a vector with column indices
	 * @details Not all elements in the vector are actually set. Only the first numNonZero()
	 *          elements are used.
	 * @return Vector with column indices
	 */
	inline const std::vector<int>& cols() const CADET_NOEXCEPT
	{
		return _cols;
	}

	/**
	 * @brief Returns a vector with element values
	 * @details Not all elements in the vector are actually set. Only the first numNonZero()
	 *          elements are used.
	 * @return Vector with element values
	 */
	inline const std::vector<real_t>& values() const CADET_NOEXCEPT
	{
		return _values;
	}

	/**
	 * @brief Returns the number of (structurally) non-zero elements in the matrix
	 * @return Number of (structurally) non-zero elements in the matrix
	 */
	inline unsigned int numNonZero() const CADET_NOEXCEPT
	{
		return _curIdx;
	}

private:
	std::vector<int> _rows;      //!< List with row indices of elements
	std::vector<int> _cols;      //!< List with column indices of elements
	std::vector<real_t> _values; //!< List with values of elements
	unsigned int _curIdx;        //!< Index of the first unused element
};

typedef SparseMatrix<double> DoubleSparseMatrix;

std::ostream& operator<<(std::ostream& out, const DoubleSparseMatrix& sm);

} // namespace linalg

} // namespace cadet

#endif // LIBCADET_SPARSEMATRIX_HPP_
