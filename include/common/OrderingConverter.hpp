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
 * Provides means to convert between column-major and row-major storage ordering of tensors
 */

#ifndef CADET_ORDERINGCONVERTER_HPP_
#define CADET_ORDERINGCONVERTER_HPP_

#include <vector>

#include "common/CompilerSpecific.hpp"

namespace cadet
{

/**
 * @brief Converts tensors between row-major and column-major ordering
 * @details Here, row-major ordering means that the last subscript changes the fastest and
 *          the first subscript the slowest. For a matrix, this means that the rows are
 *          appended to a long array. Column-major, on the contrary, means that the first
 *          subscript changes the fastest and the last the slowest. This would mean that
 *          the columns of a matrix are appended to form a long array.
 *
 *          This class helps in converting between those to orderings. Given the rank and
 *          the dimensions of the tensor, a linear index in a column- or row-major array
 *          is converted to a subscript index (index for each dimension). This subscript
 *          index, in turn, is then converted to the requested linear index in column-
 *          or row-major storage order.
 */
class OrderingConverter
{
public:
	/**
	 * @brief Index type for subscripts and linear indices
	 */
	typedef std::size_t index_t;

	/**
	 * @brief Initializes an OrderingConverter
	 * @param [in] dims Vector with the dimensions of the tensor (length of this vector is the rank)
	 * @tparam T Datatype of the dimensions, has to fit into @c std::size_t
	 */
	template <typename T> OrderingConverter(const std::vector<T> dims) : _subscript(dims.size(), 0)
	{
		prepareConversionToCol(dims.data(), dims.size());
		prepareConversionToRow(dims.data(), dims.size());
	}

	/**
	 * @brief Initializes an OrderingConverter
	 * @param [in] dims Pointer to first element of array with tensor dimensions
	 * @param [in] size Length of the @p dims array, rank of the tensor
	 * @tparam T Datatype of the dimensions, has to fit into @c std::size_t
	 */
	template <typename T> OrderingConverter(T const* dims, unsigned int size) : _subscript(size, 0)
	{
		cadet_assert(size > 0);
		prepareConversionToCol(dims, size);
		prepareConversionToRow(dims, size);
	}

	/**
	 * @brief Converts a linear index in row-major storage to its column-major counterpart
	 * @param [in] idx Linear index into row-major storage
	 * @return Linear index to same element in column-major storage
	 */
	inline index_t rowToCol(index_t idx) const
	{
		linearToSubscriptRowMajor(idx, _subscript);
		return subscriptToLinear(_subscript.data(), _prodToCol);
	}

	/**
	 * @brief Converts an index into a matrix in row-major storage to its column-major counterpart
	 * @details A tensor of rank 2 is assumed (i.e., a matrix).
	 * @param [in] idxRow Index of the row
	 * @param [in] idxCol Index of the column
	 * @return Linear index to same element in column-major storage
	 */
	inline index_t rowToCol(index_t idxRow, index_t idxCol) const
	{
		cadet_assert(_subscript.size() == 2);
		return idxRow * _prodToCol[0] + idxCol * _prodToCol[1];
	}

	/**
	 * @brief Converts an index into a 3d-array in row-major storage to its column-major counterpart
	 * @details A tensor of rank 3 is assumed (i.e., a vector of matrices).
	 * @param [in] idxRow Index of the row
	 * @param [in] idxCol Index of the column
	 * @param [in] idxPage Index of the page
	 * @return Linear index to same element in column-major storage
	 */
	inline index_t rowToCol(index_t idxRow, index_t idxCol, index_t idxPage) const
	{
		cadet_assert(_subscript.size() == 3);
		return idxRow * _prodToCol[0] + idxCol * _prodToCol[1] + idxPage * _prodToCol[2];
	}

	/**
	 * @brief Converts a linear index in column-major storage to its row-major counterpart
	 * @param [in] idx Linear index into column-major storage
	 * @return Linear index to same element in row-major storage
	 */
	inline index_t colToRow(index_t idx) const
	{
		linearToSubscriptColMajor(idx, _subscript);
		return subscriptToLinear(_subscript.data(), _prodToRow);
	}

	/**
	 * @brief Converts an index into a matrix in column-major storage to its row-major counterpart
	 * @details A tensor of rank 2 is assumed (i.e., a matrix).
	 * @param [in] idxRow Index of the row
	 * @param [in] idxCol Index of the column
	 * @return Linear index to same element in row-major storage
	 */
	inline index_t colToRow(index_t idxRow, index_t idxCol) const
	{
		cadet_assert(_subscript.size() == 2);
		return idxRow * _prodToRow[0] + idxCol * _prodToRow[1];
	}

	/**
	 * @brief Converts an index into a 3d-array in column-major storage to its row-major counterpart
	 * @details A tensor of rank 3 is assumed (i.e., a vector of matrices).
	 * @param [in] idxRow Index of the row
	 * @param [in] idxCol Index of the column
	 * @param [in] idxPage Index of the page
	 * @return Linear index to same element in row-major storage
	 */
	inline index_t colToRow(index_t idxRow, index_t idxCol, index_t idxPage) const
	{
		cadet_assert(_subscript.size() == 3);
		return idxRow * _prodToRow[0] + idxCol * _prodToRow[1] + idxPage * _prodToRow[2];
	}

	/**
	 * @brief Converts a linear index of a row-major storage to a subscript index
	 * @details A subscript index is a vector with an index for each dimension of the tensor.
	 * @param [in] index Linear index to row-major storage
	 * @param [out] idx Subscript index
	 */
	inline void linearToSubscriptRowMajor(index_t index, std::vector<index_t>& idx) const
	{
		cadet_assert(idx.size() == _subscript.size());
		for (std::size_t i = 0; i < _prodToRow.size() - 1; ++i)
		{
			idx[i] = index / _prodToRow[i];
			index -= idx[i] * _prodToRow[i];
		}
		idx.back() = index;
	}

	/**
	 * @brief Converts a linear index of a column-major storage to a subscript index
	 * @details A subscript index is a vector with an index for each dimension of the tensor.
	 * @param [in] index Linear index to column-major storage
	 * @param [out] idx Subscript index
	 */
	inline void linearToSubscriptColMajor(index_t index, std::vector<index_t>& idx) const
	{
		cadet_assert(idx.size() == _subscript.size());
		for (std::size_t i = _prodToCol.size() - 1; i >= 1; --i)
		{
			idx[i] = index / _prodToCol[i];
			index -= idx[i] * _prodToCol[i];
		}
		idx[0] = index;
	}

	/**
	 * @brief Converts a subscript index to a linear index to row-major storage
	 * @param [in] idx Subscript index
	 * @return Linear index to row-major storage
	 */
	inline index_t subscriptToRowMajor(const std::vector<index_t>& idx) const
	{
		cadet_assert(idx.size() == _subscript.size());
		return subscriptToLinear(idx.data(), _prodToRow);
	}

	/**
	 * @brief Converts a subscript index to a linear index to row-major storage
	 * @param [in] idx Pointer to first element of subscript index array
	 * @return Linear index to row-major storage
	 * @tparam T Datatype of the subscript indices, has to fit into @c std::size_t
	 */
	template <typename T> inline index_t subscriptToRowMajor(T const* idx) const
	{
		return subscriptToLinear(idx, _prodToRow);
	}

	/**
	 * @brief Converts a subscript index to a linear index to column-major storage
	 * @param [in] idx Subscript index
	 * @return Linear index to column-major storage
	 */
	inline index_t subscriptToColMajor(const std::vector<index_t>& idx) const
	{
		cadet_assert(idx.size() == _subscript.size());
		return subscriptToLinear(idx.data(), _prodToCol);
	}

	/**
	 * @brief Converts a subscript index to a linear index to column-major storage
	 * @param [in] idx Pointer to first element of subscript index array
	 * @return Linear index to column-major storage
	 * @tparam T Datatype of the subscript indices, has to fit into @c std::size_t
	 */
	template <typename T> inline index_t subscriptToColMajor(T const* idx) const
	{
		return subscriptToLinear(idx, _prodToCol);
	}

	/**
	 * @brief Returns the current subscript index as set by the last call to colToRow() or rowToCol()
	 * @return Subscript index produced by the last call to colToRow() or rowToCol()
	 */
	inline const std::vector<index_t> subscriptIndex() const
	{
		return _subscript;
	}

protected:
	std::vector<index_t> _prodToRow;         //!< Helper vector for conversions from / to row-major
	std::vector<index_t> _prodToCol;         //!< Helper vector for conversions from / to column-major
	mutable std::vector<index_t> _subscript; //!< Cache for subscript index

	template <typename T> void prepareConversionToCol(T const* dims, unsigned int size)
	{
		_prodToCol.resize(size);
		unsigned int p = 1;
		for (unsigned int i = 0; i < size; ++i)
		{
			const index_t temp = dims[i];
			_prodToCol[i] = p;
			p *= temp;
		}
	}

	template <typename T> void prepareConversionToRow(T const* dims, unsigned int size)
	{
		_prodToRow.resize(size);
		unsigned int p = 1;
		for (int i = size - 1; i >= 0; --i)
		{
			const index_t temp = dims[i];
			_prodToRow[i] = p;
			p *= temp;
		}
	}

	/**
	 * @brief Synthesizes a linear index from a given subscript index
	 * @details The linear index is computed by taking the dot-product with the helper vector
	 *          corresponding to the target storage order.
	 * @param [in] idx Pointer to first element of subscript index array
	 * @param [in] dimsProd Helper array for the requested target storage order
	 * @return Linear index to the requested target storage
	 */
	template <typename T> inline index_t subscriptToLinear(T const* idx, const std::vector<index_t>& dimsProd) const
	{
		index_t retIndex = 0;
		for (std::size_t i = 0; i < dimsProd.size(); ++i)
			retIndex += idx[i] * dimsProd[i];
		return retIndex;
	}
};

} // namespace cadet

#endif // CADET_ORDERINGCONVERTER_HPP_
