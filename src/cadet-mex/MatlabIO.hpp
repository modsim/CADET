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
 * Provides Matlab IO support (reading & writing from Matlab structs)
 */

#ifndef CADET_MEX_MATLABIO_HPP_
#define CADET_MEX_MATLABIO_HPP_

#include <mex.h>

#include <vector>
#include <string>

#include "MatlabUndocumentedSupport.hpp"

namespace cadet
{

namespace mex
{

namespace io
{

/**
 * @brief Checks whether the given @p dataSet is of the requested type
 * @param [in] dataSet Matlab dataset
 * @tparam Numeric_t Requested type
 * @return @c true if the dataset is of the requested type, otherwise @c false
 */
template <typename Numeric_t> inline bool isType(mxArray const* dataSet)
{
	return false;
}

template <> inline bool isType<double>(mxArray const* dataSet)
{
	return mxIsDouble(dataSet);
}

template <> inline bool isType<int>(mxArray const* dataSet)
{
	return mxIsInt32(dataSet);
}

/*
template <>
inline bool isType<int32_t>(mxArray const* dataSet) { return mxIsInt32(dataSet); }
*/

template <> inline bool isType<int64_t>(mxArray const* dataSet)
{
	return mxIsInt64(dataSet);
}

template <> inline bool isType<uint32_t>(mxArray const* dataSet)
{
	return mxIsUint32(dataSet);
}

template <> inline bool isType<uint64_t>(mxArray const* dataSet)
{
	return mxIsUint64(dataSet);
}

template <> inline bool isType<std::string>(mxArray const* dataSet)
{
	return mxIsChar(dataSet);
}

/**
 * @brief Checks whether a given Matlab dataset is empty
 * @param [in] dataSet Matlab dataset
 * @return @c true if the dataset is empty, otherwise @c false
 */
inline bool isEmpty(mxArray const* dataSet)
{
	return mxIsEmpty(dataSet);
}

/**
 * @brief Checks whether a given Matlab dataset contains a scalar
 * @param [in] dataSet Matlab dataset
 * @return @c true if the dataset is a scalar, otherwise @c false
 */
inline bool isScalar(mxArray const* dataSet)
{
	return mxGetNumberOfElements(dataSet) == 1;
}

/**
 * @brief Checks whether a given Matlab dataset contains a vector
 * @param [in] dataSet Matlab dataset
 * @return @c true if the dataset is a vector, otherwise @c false
 */
inline bool isVector(mxArray const* dataSet)
{
	// Expect column or row vector
	const mwSize numDims = mxGetNumberOfDimensions(dataSet);
	if (numDims > 2)
	{
		// Error: We've got a tensor
		return false;
	}

	const mwSize* dims = mxGetDimensions(dataSet);
	if ((dims[0] > 1) && (dims[1] > 1))
	{
		// Error: We've got a matrix
		return false;
	}
	return true;
}

/**
 * @brief Checks whether a given Matlab dataset contains a matrix
 * @param [in] dataSet Matlab dataset
 * @return @c true if the dataset is a matrix, otherwise @c false
 */
inline bool isMatrix(mxArray const* dataSet)
{
	// Expect tensor of rank 2 (aka matrix)
	const mwSize numDims = mxGetNumberOfDimensions(dataSet);
	if (numDims != 2)
	{
		// Error: We've got a tensor or vector
		return false;
	}
	return true;
}

/**
 * @brief Checks whether a given Matlab dataset contains a cell array
 * @param [in] dataSet Matlab dataset
 * @return @c true if the dataset is a cell array, otherwise @c false
 */
inline bool isCell(mxArray const* dataSet)
{
	return mxIsCell(dataSet);
}

/**
 * @brief Returns the total number of elements in a Matlab dataset
 * @param [in] dataSet Matlab dataset
 * @return Total number of elements in a dataset
 */
inline std::size_t numElements(mxArray const* dataSet)
{
	return mxGetNumberOfElements(dataSet);
}

/**
 * @brief Returns a pointer to the beginning of the underlying data array of a Matlab dataset
 * @param [in] dataSet Matlab dataset
 * @tparam Numeric_t Requested type
 * @return Pointer to underlying array of the given Matlab @p dataSet
 */
template <typename Numeric_t> inline Numeric_t* data(mxArray* dataSet)
{
	return nullptr;
}

template <typename Numeric_t> inline Numeric_t const* data(mxArray const* dataSet)
{
	return nullptr;
}

template <> inline double* data<double>(mxArray* dataSet)
{
	return mxGetPr(dataSet);
}

template <> inline double const* data<double>(mxArray const* dataSet)
{
	return mxGetPr(dataSet);
}

template <> inline int* data<int>(mxArray* dataSet)
{
	return static_cast<int*>(mxGetData(dataSet));
}

template <> inline int const* data<int>(mxArray const* dataSet)
{
	return static_cast<int const*>(mxGetData(dataSet));
}

/*
template <>
inline int32_t* data<int32_t>(mxArray* dataSet) { return static_cast<int32_t*>(mxGetData(dataSet)); }

template <>
inline int32_t const* data<int32_t>(mxArray const* dataSet) { return static_cast<int32_t const*>(mxGetData(dataSet)); }
*/

template <> inline int64_t* data<int64_t>(mxArray* dataSet)
{
	return static_cast<int64_t*>(mxGetData(dataSet));
}

template <> inline int64_t const* data<int64_t>(mxArray const* dataSet)
{
	return static_cast<int64_t const*>(mxGetData(dataSet));
}

template <> inline uint32_t* data<uint32_t>(mxArray* dataSet)
{
	return static_cast<uint32_t*>(mxGetData(dataSet));
}

template <> inline uint32_t const* data<uint32_t>(mxArray const* dataSet)
{
	return static_cast<uint32_t const*>(mxGetData(dataSet));
}

template <> inline uint64_t* data<uint64_t>(mxArray* dataSet)
{
	return static_cast<uint64_t*>(mxGetData(dataSet));
}

template <> inline uint64_t const* data<uint64_t>(mxArray const* dataSet)
{
	return static_cast<uint64_t const*>(mxGetData(dataSet));
}

template <typename Numeric_t> inline Numeric_t scalar(mxArray const* dataSet)
{
	return *(data<Numeric_t>(dataSet));
}

template <typename Numeric_t> inline Numeric_t& scalar(mxArray* dataSet)
{
	return *(data<Numeric_t>(dataSet));
}

template <typename Numeric_t> inline mxArray* createArray(unsigned int n, unsigned int m)
{
	return nullptr;
}

template <> inline mxArray* createArray<int32_t>(unsigned int m, unsigned int n)
{
	return mxCreateUninitNumericMatrix(m, n, mxINT32_CLASS, mxREAL);
}

template <> inline mxArray* createArray<uint32_t>(unsigned int m, unsigned int n)
{
	return mxCreateUninitNumericMatrix(m, n, mxUINT32_CLASS, mxREAL);
}

template <> inline mxArray* createArray<int64_t>(unsigned int m, unsigned int n)
{
	return mxCreateUninitNumericMatrix(m, n, mxINT64_CLASS, mxREAL);
}

template <> inline mxArray* createArray<uint64_t>(unsigned int m, unsigned int n)
{
	return mxCreateUninitNumericMatrix(m, n, mxUINT64_CLASS, mxREAL);
}

template <> inline mxArray* createArray<double>(unsigned int m, unsigned int n)
{
	return mxCreateUninitNumericMatrix(m, n, mxDOUBLE_CLASS, mxREAL);
}

template <typename Source_t, typename Dest_t> inline mxArray* scalar(Source_t value)
{
	mxArray* const data = createArray<Dest_t>(1, 1);
	*static_cast<Dest_t*>(mxGetData(data)) = value;
	return data;
}

template <> inline mxArray* scalar<double, double>(double value)
{
	mxArray* const data = mxCreateUninitNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
	*mxGetPr(data) = value;
	return data;
}

template <typename Source_t, typename Dest_t> inline mxArray* scalar(Source_t value, Source_t minusOne)
{
	mxArray* const data = createArray<Dest_t>(1, 1);
	if (value == minusOne)
		*static_cast<Dest_t*>(mxGetData(data)) = static_cast<Dest_t>(-1);
	else
		*static_cast<Dest_t*>(mxGetData(data)) = value;
	return data;
}

} // namespace io
} // namespace mex
} // namespace cadet

#endif // CADET_MEX_MATLABIO_HPP_
