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
 * Defines operations on subsets of vectors
 */

#ifndef LIBCADET_SUBSET_HPP_
#define LIBCADET_SUBSET_HPP_

#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"

#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"

#include <algorithm>

namespace cadet
{

namespace linalg
{

struct ConstIndexArray
{
	int const* indices;
	int len;

	inline int operator[](int idx) const
	{
		return indices[idx];
	}
};

struct IndexArray
{
	int* indices;
	int len;

	inline operator ConstIndexArray() const
	{
		return ConstIndexArray{indices, len};
	}
	inline int operator[](int idx) const
	{
		return indices[idx];
	}
	inline int& operator[](int idx)
	{
		return indices[idx];
	}
};

struct ConstMaskArray
{
	int const* mask;
	int len;

	inline int operator[](int idx) const
	{
		return mask[idx];
	}
};

struct MaskArray
{
	int* mask;
	int len;

	inline operator ConstMaskArray() const
	{
		return ConstMaskArray{mask, len};
	}
	inline int operator[](int idx) const
	{
		return mask[idx];
	}
	inline int& operator[](int idx)
	{
		return mask[idx];
	}
};

/**
 * @brief Converts a mask (boolean) array to an index array
 * @details The index array contains the indices of the @c true items
 *          in the mask array.
 *
 *          The memory of the given mask array is used for constructing the index array.
 *          That is, the MaskArray @p data is no longer available after the function call.
 * @param [in] data Mask array
 * @return Index array
 */
inline IndexArray maskToIndexArray(const MaskArray& data)
{
	int idx = 0;
	for (int i = 0; i < data.len; ++i)
	{
		if (data.mask[i])
		{
			data.mask[idx] = i;
			++idx;
		}
	}
	return IndexArray{data.mask, idx};
}

/**
 * @brief Counts the number of active elements in a mask
 * @param [in] mask Mask
 * @return Number of active elements in mask
 */
inline int numMaskActive(const ConstMaskArray& mask)
{
	return std::count_if(mask.mask, mask.mask + mask.len, [](const int i) -> bool { return i; });
}

/**
 * @brief Counts the number of active elements in a mask up to a given index
 * @param [in] mask Mask
 * @param [in] len Length of the mask subset that is checked
 * @return Number of active elements in mask subset
 */
inline int numMaskActive(const ConstMaskArray& mask, int len)
{
	cadet_assert(len <= mask.len);
	cadet_assert(len >= 0);
	return std::count_if(mask.mask, mask.mask + len, [](const int i) -> bool { return i; });
}

/**
 * @brief Counts the number of active elements in a subset of a mask
 * @param [in] mask Mask
 * @param [in] start Index of the first item in the subset of the mask
 * @param [in] len Length of the mask subset
 * @return Number of active elements in mask subset
 */
inline unsigned int numMaskActive(const ConstMaskArray& mask, int start, int len)
{
	cadet_assert(start + len <= mask.len);
	cadet_assert((start >= 0) && (len >= 0));
	return std::count_if(mask.mask + start, mask.mask + start + len, [](const int i) -> bool { return i; });
}

/**
 * @brief Counts the number of active partitions in a mask
 * @details The mask is divided into partitions that have an irregular number of items.
 * @param [in] mask Mask
 * @param [in] partSize Array with number of items in each partition
 * @param [in] nPart Number of partitions
 * @return Number of active partitions in the mask
 */
inline unsigned int numMaskActivePartitions(const ConstMaskArray& mask, unsigned int const* const partSize,
											unsigned int nPart)
{
	unsigned int partStartIdx = 0;
	unsigned int numActivePart = 0;
	for (unsigned int part = 0; part < nPart; ++part)
	{
		for (unsigned int i = 0; i < partSize[part]; ++i)
		{
			if (mask.mask[partStartIdx + i])
			{
				++numActivePart;
				break;
			}
		}

		partStartIdx += partSize[part];
	}
	return numActivePart;
}

inline void conservedMoietiesFromPartitionedMask(const ConstMaskArray& mask, unsigned int const* const partSize,
												 unsigned int nPart, double const* const src, double* const dest,
												 double factorLiquid, double factorSolid)
{
	unsigned int bndIdx = nPart;
	unsigned int rIdx = 0;
	for (unsigned int comp = 0; comp < nPart; ++comp)
	{
		if (!mask.mask[comp])
		{
			bndIdx += partSize[comp];
			continue;
		}

		dest[rIdx] = factorLiquid * src[comp];

		for (unsigned int bnd = 0; bnd < partSize[comp]; ++bnd, ++bndIdx)
		{
			if (mask.mask[bndIdx])
				dest[rIdx] += factorSolid * src[bndIdx];
		}

		++rIdx;
	}
}

/**
 * @brief Selects a subset of a vector / array
 * @details Copies the elements at the given indices to the beginning of the @p dest array.
 * @param [in] src Source array
 * @param [in] idx Index array
 * @param [out] dest Destination array
 */
inline void selectVectorSubset(double const* const src, const ConstIndexArray& idx, double* const dest)
{
	for (int i = 0; i < idx.len; ++i)
		dest[i] = src[idx.indices[i]];
}

/**
 * @brief Copies a vector / array to a subset of another array
 * @details Copies the @p src array to the elements of the @p dest array at the given indices.
 * @param [in] src Source array
 * @param [in] idx Index array
 * @param [out] dest Destination array
 */
inline void applyVectorSubset(double const* const src, const ConstIndexArray& idx, double* const dest)
{
	for (int i = 0; i < idx.len; ++i)
		dest[idx.indices[i]] = src[i];
}

/**
 * @brief Selects a subset of a vector / array
 * @details Copies the elements selected by the mask to the beginning of the @p dest array.
 * @param [in] src Source array
 * @param [in] mask Mask array
 * @param [out] dest Destination array
 */
inline void selectVectorSubset(double const* const src, const ConstMaskArray& mask, double* const dest)
{
	int k = 0;
	for (int i = 0; i < mask.len; ++i)
	{
		if (mask.mask[i])
		{
			dest[k] = src[i];
			++k;
		}
	}
}

/**
 * @brief Copies a vector / array to a subset of another array
 * @details Copies the @p src array to the elements of the @p dest array selected by the mask.
 * @param [in] src Source array
 * @param [in] mask Mask array
 * @param [out] dest Destination array
 */
inline void applyVectorSubset(double const* const src, const ConstMaskArray& mask, double* const dest)
{
	int k = 0;
	for (int i = 0; i < mask.len; ++i)
	{
		if (mask.mask[i])
		{
			dest[i] = src[k];
			++k;
		}
	}
}

/**
 * @brief Copies a vector / array to a subset of another array
 * @details Copies the @p src array to the elements of the @p dest array selected by the mask.
 * @param [in] src Source array
 * @param [in] mask Mask array
 * @param [out] dest Destination array
 */
template <typename T> inline void applyVectorSubset(double const* const src, const ConstMaskArray& mask, T* const dest)
{
	int k = 0;
	for (int i = 0; i < mask.len; ++i)
	{
		if (mask.mask[i])
		{
			dest[i].setValue(src[k]);
			++k;
		}
	}
}

/**
 * @brief Fills the given subset of an array with the given value
 * @param [in,out] data Array
 * @param [in] idx Index array
 * @param [in] value Value that is used to fill the array
 */
inline void fillVectorSubset(double* const data, const ConstIndexArray& idx, double value)
{
	for (int i = 0; i < idx.len; ++i)
		data[idx.indices[i]] = value;
}

/**
 * @brief Fills the given subset of an array with the given value
 * @param [in,out] data Array
 * @param [in] mask Mask array
 * @param [in] value Value that is used to fill the array
 */
inline void fillVectorSubset(double* const data, const ConstMaskArray& mask, double value)
{
	for (int i = 0; i < mask.len; ++i)
	{
		if (mask.mask[i])
			data[i] = value;
	}
}

/**
 * @brief Computes @f$ y = \beta y + \alpha x_{\text{mask}} @f$
 * @param [in] x Array @f$ x @f$
 * @param [in] idx Index array
 * @param [in] alpha Factor @f$ \alpha @f$
 * @param [in] beta Factor @f$ \beta @f$
 * @param [in,out] y Array @f$ y @f$
 */
inline void vectorSubsetAdd(double const* const x, const ConstIndexArray& idx, double alpha, double beta,
							double* const y)
{
	for (int i = 0; i < idx.len; ++i)
		y[i] = beta * y[i] + alpha * x[idx.indices[i]];
}

/**
 * @brief Computes @f$ y = \beta y + \alpha x_{\text{mask}} @f$
 * @param [in] x Array @f$ x @f$
 * @param [in] mask Mask array
 * @param [in] alpha Factor @f$ \alpha @f$
 * @param [in] beta Factor @f$ \beta @f$
 * @param [in,out] y Array @f$ y @f$
 */
inline void vectorSubsetAdd(double const* const x, const ConstMaskArray& mask, double alpha, double beta,
							double* const y)
{
	int k = 0;
	for (int i = 0; i < mask.len; ++i)
	{
		if (mask.mask[i])
		{
			y[k] = beta * y[k] + alpha * x[i];
			++k;
		}
	}
}

/**
 * @brief Copies a subset of a given dense matrix into this matrix
 * @details Copies a submatrix indentified by row and column masks from a given source in dense
 *          storage into a destination matrix. The submatrix has to fully fit into the destination
 *          matrix and is placed in the top left corner. The rest of the matrix is left unchanged.
 * @param [in] mat Source matrix in dense storage format
 * @param [in] rowIdx Index array for rows
 * @param [in] colIdx Index array for columns
 * @param [in] dest Destination matrix in dense storage format
 */
inline void copyMatrixSubset(const detail::DenseMatrixBase& src, const ConstIndexArray& rowIdx,
							 const ConstIndexArray& colIdx, detail::DenseMatrixBase& dest)
{
	cadet_assert(dest.rows() >= rowIdx.len);
	cadet_assert(dest.columns() >= colIdx.len);
	cadet_assert(src.rows() > rowIdx.indices[rowIdx.len - 1]);
	cadet_assert(src.columns() > colIdx.indices[colIdx.len - 1]);

	double* ptrDest = dest.data();
	for (int r = 0; r < rowIdx.len; ++r, ptrDest += dest.stride())
	{
		double const* ptrSrc = src.data() + src.stride() * rowIdx.indices[r];
		for (int c = 0; c < colIdx.len; ++c)
			ptrDest[c] = ptrSrc[colIdx.indices[c]];
	}
}

/**
 * @brief Copies a subset of a given dense matrix into this matrix
 * @details Copies a submatrix indentified by row and column masks from a given source in dense
 *          storage into a destination matrix. The submatrix has to fully fit into the destination
 *          matrix and is placed in the left corner identified by a row offset. The rest of the
 *          matrix is left unchanged.
 * @param [in] mat Source matrix in dense storage format
 * @param [in] rowIdx Index array for rows
 * @param [in] colIdx Index array for columns
 * @param [in] srcStartRow Offset applied to the row mask in the source matrix
 * @param [in] srcStartCol Offset applied to the column mask in the source matrix
 * @param [in] dest Destination matrix in dense storage format
 * @param [in] destStartRow Index of the first row written in the target matrix
 */
inline void copyMatrixSubset(const detail::DenseMatrixBase& src, const ConstIndexArray& rowIdx,
							 const ConstIndexArray& colIdx, int srcStartRow, int srcStartCol,
							 detail::DenseMatrixBase& dest, int destStartRow)
{
	cadet_assert(destStartRow >= 0);
	cadet_assert(dest.rows() >= rowIdx.len + destStartRow);
	cadet_assert(dest.columns() >= colIdx.len);
	cadet_assert((srcStartRow >= 0) && (srcStartCol >= 0));
	cadet_assert(src.rows() > rowIdx.indices[rowIdx.len - 1] + srcStartRow);
	cadet_assert(src.columns() > colIdx.indices[colIdx.len - 1] + srcStartCol);

	double* ptrDest = dest.data() + dest.stride() * destStartRow;
	for (int r = 0; r < rowIdx.len; ++r, ptrDest += dest.stride())
	{
		double const* ptrSrc = src.data() + src.stride() * (rowIdx.indices[r] + srcStartRow) + srcStartCol;
		for (int c = 0; c < colIdx.len; ++c)
			ptrDest[c] = ptrSrc[colIdx.indices[c]];
	}
}

/**
 * @brief Copies a subset of a given dense matrix into this matrix
 * @details Copies a submatrix indentified by row and column masks from a given source in dense
 *          storage into a destination matrix. The submatrix has to fully fit into the destination
 *          matrix and is placed in the top left corner. The rest of the matrix is left unchanged.
 * @param [in] mat Source matrix in dense storage format
 * @param [in] rowMask Mask array for rows
 * @param [in] colMask Mask array for columns
 * @param [in] dest Destination matrix in dense storage format
 */
inline void copyMatrixSubset(const detail::DenseMatrixBase& src, const ConstMaskArray& rowMask,
							 const ConstMaskArray& colMask, detail::DenseMatrixBase& dest)
{
	cadet_assert(dest.rows() >= numMaskActive(rowMask));
	cadet_assert(dest.columns() >= numMaskActive(colMask));
	cadet_assert(src.rows() >= rowMask.len);
	cadet_assert(src.columns() >= colMask.len);

	double* ptrDest = dest.data();
	for (int r = 0; r < rowMask.len; ++r)
	{
		if (!rowMask.mask[r])
			continue;

		double const* const ptrSrc = src.data() + src.stride() * r;
		int idx = 0;
		for (int c = 0; c < colMask.len; ++c)
		{
			if (!colMask.mask[c])
				continue;

			ptrDest[idx] = ptrSrc[c];
			++idx;
		}

		ptrDest += dest.stride();
	}
}

/**
 * @brief Copies a subset of a given dense matrix into this matrix
 * @details Copies a submatrix indentified by row and column masks from a given source in dense
 *          storage into a destination matrix. The submatrix has to fully fit into the destination
 *          matrix and is placed in the left corner identified by a row offset. The rest of the
 *          matrix is left unchanged.
 * @param [in] mat Source matrix in dense storage format
 * @param [in] rowMask Mask array for rows
 * @param [in] colMask Mask array for columns
 * @param [in] srcStartRow Offset applied to the row mask in the source matrix
 * @param [in] srcStartCol Offset applied to the column mask in the source matrix
 * @param [in] dest Destination matrix in dense storage format
 * @param [in] destStartRow Index of the first row written in the target matrix
 */
inline void copyMatrixSubset(const detail::DenseMatrixBase& src, const ConstMaskArray& rowMask,
							 const ConstMaskArray& colMask, int srcStartRow, int srcStartCol,
							 detail::DenseMatrixBase& dest, int destStartRow)
{
	cadet_assert(destStartRow >= 0);
	cadet_assert(dest.rows() >= numMaskActive(rowMask) + destStartRow);
	cadet_assert(dest.columns() >= numMaskActive(colMask));
	cadet_assert((srcStartRow >= 0) && (srcStartCol >= 0));
	cadet_assert(src.rows() >= rowMask.len + srcStartRow);
	cadet_assert(src.columns() >= colMask.len + srcStartCol);

	double* ptrDest = dest.data() + dest.stride() * destStartRow;
	for (int r = 0; r < rowMask.len; ++r)
	{
		if (!rowMask.mask[r])
			continue;

		double const* const ptrSrc = src.data() + src.stride() * (r + srcStartRow) + srcStartCol;
		int idx = 0;
		for (int c = 0; c < colMask.len; ++c)
		{
			if (!colMask.mask[c])
				continue;

			ptrDest[idx] = ptrSrc[c];
			++idx;
		}

		ptrDest += dest.stride();
	}
}

/**
 * @brief Copies a subset of a given band matrix into this matrix
 * @details Copies a submatrix indentified by row and column masks from a given source in banded
 *          storage into a destination matrix. The submatrix has to fully fit into the destination
 *          matrix and is placed in the top left corner. The rest of the matrix is left unchanged.
 * @param [in] mat Source matrix in banded storage format
 * @param [in] rowMask Mask array for rows
 * @param [in] colMask Mask array for columns
 * @param [in] rowOffset Offset to the first row to copy
 * @param [in] diagOffset Offset to the first diagonal to copy
 * @param [in] dest Destination matrix in dense storage format
 */
inline void copyMatrixSubset(const BandMatrix& src, const ConstMaskArray& rowMask, const ConstMaskArray& colMask,
							 int rowOffset, int diagOffset, detail::DenseMatrixBase& dest)
{
	cadet_assert(dest.rows() >= numMaskActive(rowMask));
	cadet_assert(dest.columns() >= numMaskActive(colMask));
	cadet_assert(src.rows() >= rowMask.len);
	cadet_assert(static_cast<int>(src.upperBandwidth()) + 1 + diagOffset >= rowMask.len);
	cadet_assert(-diagOffset <= static_cast<int>(src.lowerBandwidth()));

	const int srcStride = src.stride();
	double const* ptrSrc = src.data() + srcStride * rowOffset + diagOffset + src.lowerBandwidth();
	double* ptrDest = dest.data();
	for (int r = 0; r < rowMask.len; ++r, ptrSrc += srcStride - 1)
	{
		if (!rowMask.mask[r])
			continue;

		int idx = 0;
		for (int c = 0; c < colMask.len; ++c)
		{
			if (!colMask.mask[c])
				continue;

			ptrDest[idx] = ptrSrc[c];
			++idx;
		}

		ptrDest += dest.stride();
	}
}

} // namespace linalg

} // namespace cadet

#endif // LIBCADET_SUBSET_HPP_
