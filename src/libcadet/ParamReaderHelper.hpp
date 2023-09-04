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
 * Provides helper functions for handling parameters
 */

#ifndef LIBCADET_PARAMREADERHELPER_HPP_
#define LIBCADET_PARAMREADERHELPER_HPP_

#include "cadet/ParameterId.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/Exceptions.hpp"

#include "common/CompilerSpecific.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <numeric>

namespace cadet
{
/**
 * @brief Reads a scalar, possibly vectorial, parameter into a vector
 * @details This function automatically detects if the parameter is vectorial.
 *          If the user always wants a vectorial result, the single scalar value is
 *          replicated to reach the requested array size.
 *
 * @param [in] dest Destination vector in which the data is saved
 * @param [in] paramProvider Parameter provider from which is read
 * @param [in] dataSet Name of the dataset
 * @param [in] nExpand How often a single scalar parameter is replicated (@c 1 if scalars should remain scalar)
 * @return @c true if a scalar value was read, or @c false if an actual array (more than 1 element) was read
 * @tparam ValType Type of the parameter, such as @c active or @c double
 */
template <typename ValType>
inline bool readScalarParameterOrArray(std::vector<ValType>& dest, IParameterProvider& paramProvider,
									   const std::string& dataSet, unsigned int nExpand)
{
	dest.clear();
	if (paramProvider.isArray(dataSet))
	{
		// Copy all values from provider to destination
		const std::vector<double> vals = paramProvider.getDoubleArray(dataSet);
		dest.resize(vals.size());
		for (std::size_t i = 0; i < vals.size(); ++i)
			dest[i] = vals[i];

		return vals.size() == 1;
	}
	else
	{
		// Only scalar value in provider, expand to given size
		const double val = paramProvider.getDouble(dataSet);
		dest.resize(nExpand);
		for (unsigned int i = 0; i < nExpand; ++i)
			dest[i] = val;

		return true;
	}
}

/**
 * @brief Reads a vector valued parameter, that may also be a matrix
 * @details It is automatically detected if the parameter is matrixoid or vectorial.
 *          If the user always wants a matrixoid result, the vectorial parameter
 *          is replicated to reach the requested size. The ordering is
 *          vec0rep0, vec1rep0, vec2rep0, vec0rep1, vec1rep1, vec2rep1, etc.
 *
 * @param [in] dest Destination vector in which the data is saved
 * @param [in] paramProvider Parameter provider from which is read
 * @param [in] dataSet Name of the dataset
 * @param [in] nExpect Expected number of elements in the vectorial parameter
 * @param [in] nExpand How often the vectorial parameter is replicated (@c 1 if vectorials should remain vectorial)
 * @tparam ValType Type of the parameter, such as @c active or @c double
 */
template <typename ValType>
inline void readParameterMatrix(std::vector<ValType>& dest, IParameterProvider& paramProvider,
								const std::string& dataSet, unsigned int nExpect, unsigned int nExpand)
{
	dest.clear();
	const std::vector<double> vals = paramProvider.getDoubleArray(dataSet);

	if (vals.size() == nExpect)
	{
		// Read expected number of values, so copy them
		dest.resize(nExpect);
		for (unsigned int i = 0; i < nExpect; ++i)
			dest[i] = vals[i];
	}
	else
	{
		// We copy the read values nExpand times
		dest.resize(vals.size() * nExpand);
		for (unsigned int j = 0; j < nExpand; ++j)
		{
			for (std::size_t i = 0; i < vals.size(); ++i)
				dest[i + j * vals.size()] = vals[i];
		}
	}
}

/**
 * @brief Reads a parameter that depends on components and bound states available in a state-major matrix
 * @details The ordering is expected to be state-major, that is, for a 3 component and 2 bound state parameter
 *          the parameters @f$p_{\text{State},\text{Comp}} @f$ are stored like this:
 *          @f[ \begin{pmatrix} p_{11} & p_{12} & p_{13} \\ p_{21} & p_{22} & p_{23} \\ p_{31} & p_{32} & p_{33}
 *              \Leftrightarrow \left( p_{11}, p_{12}, p_{13}, p_{21}, \dots, p_{23} \right). @f]
 *          The rows of the matrix which contain the parameters for all components of one specific bound state are
 *          appended to one another to form a long array.
 *
 * @param [in] dest Destination vector in which the data is saved
 * @param [in] paramProvider Parameter provider from which is read
 * @param [in] dataSet Name of the dataset
 * @param [nComp] Number of required components
 * @param [nStates] Number of required bound states
 * @tparam SliceContainer_t Type of the slice container, such as @c cadet::util::SlicedVector
 * @tparam ValType Type of the parameter, such as @c active or @c double
 */
template <typename SliceContainer_t, typename ValType>
inline void readBoundStateDependentParameter(SliceContainer_t& dest, IParameterProvider& paramProvider,
											 const std::string& dataSet, unsigned int nComp, unsigned int nStates)
{
	dest.clear();
	const std::vector<double> vals = paramProvider.getDoubleArray(dataSet);

	if (vals.size() < nComp * nStates)
	{
		throw InvalidParameterException("Not enough elements in dataset " + dataSet + " (expected " +
										std::to_string(nComp * nStates) + " but got only " +
										std::to_string(vals.size()) + ")");
	}

	for (unsigned int i = 0; i < nStates; ++i)
	{
		dest.pushBackSlice(nComp);
		ValType* const destCur = dest[i];
		for (unsigned int j = 0; j < nComp; ++j)
			destCur[j] = vals[i * nComp + j];
	}
}

/**
 * @brief Reads a parameter that depends on components and bound states available in a component-major matrix
 * @details The ordering is expected to be component-major, that is, for a 3 component and 2 bound state parameter
 *          the parameters @f$p_{\text{State},\text{Comp}} @f$ are stored like this:
 *          @f[ \begin{pmatrix} p_{11} & p_{12} & p_{13} \\ p_{21} & p_{22} & p_{23} \\ p_{31} & p_{32} & p_{33}
 *              \Leftrightarrow \left( p_{11}, p_{21}, p_{12}, p_{22}, \dots, p_{23} \right). @f]
 *          The rows of the matrix which contain the parameters for all components of one specific bound state are
 *          appended to one another to form a long array.
 *
 * @param [in] dest Destination vector in which the data is saved
 * @param [in] paramProvider Parameter provider from which is read
 * @param [in] dataSet Name of the dataset
 * @param [nComp] Number of required components
 * @param [nStates] Array with number of bound states for each component
 * @tparam SliceContainer_t Type of the slice container, such as @c cadet::util::SlicedVector
 * @tparam ValType Type of the parameter, such as @c active or @c double
 */
template <typename SliceContainer_t, typename ValType>
inline void readBoundStateDependentParameter(SliceContainer_t& dest, IParameterProvider& paramProvider,
											 const std::string& dataSet, unsigned int nComp,
											 unsigned int const* const nStates)
{
	dest.clear();
	const std::vector<double> vals = paramProvider.getDoubleArray(dataSet);

	unsigned int curIdx = 0;
	for (unsigned int i = 0; i < nComp; ++i)
	{
		if (vals.size() < curIdx + nStates[i])
			throw InvalidParameterException("Not enough elements in dataset " + dataSet + " (expected at least " +
											std::to_string(curIdx + nStates[i]) + " but got only " +
											std::to_string(vals.size()) + ")");

		dest.pushBackSlice(nStates[i]);
		ValType* const destCur = dest[i];
		for (unsigned int j = 0; j < nStates[i]; ++j, ++curIdx)
			destCur[j] = vals[curIdx];
	}
}

/**
 * @brief Reads a matrix valued parameter that depends on components and bound states available in a component-row-major
 * 3D array
 * @details The ordering is expected to be component-row-major, that is, for a 3 component and 2 bound state matrix
 * valued parameter the parameters @f$p_{\text{Comp},\text{State},\text{State}} @f$ are stored like this:
 *          @f[ p_{i,\cdot,\cdot} = \begin{pmatrix} p_{i,1,1} & p_{i,1,2} \\ p_{i,2,1} & p_{i,2,2} \end{pmatrix}
 *              \Leftrightarrow \left( p_{111}, p_{112}, p_{121}, p_{122}, p_{211}, p_{212}, p_{221}, p_{222}, \dots,
 * p_{322} \right). @f] The parameters are stored in the same ordering as they are read.
 *
 * @param [in] dest Destination vector in which the data is saved
 * @param [in] paramProvider Parameter provider from which is read
 * @param [in] dataSet Name of the dataset
 * @param [nComp] Number of required components
 * @param [nStates] Array with number of bound states for each component
 * @tparam SliceContainer_t Type of the slice container, such as @c cadet::util::SlicedVector
 * @tparam ValType Type of the parameter, such as @c active or @c double
 */
template <typename SliceContainer_t, typename ValType>
inline void readMatrixValuedBoundStateDependentParameter(SliceContainer_t& dest, IParameterProvider& paramProvider,
														 const std::string& dataSet, unsigned int nComp,
														 unsigned int const* const nStates)
{
	dest.clear();
	const std::vector<double> vals = paramProvider.getDoubleArray(dataSet);

	unsigned int curIdx = 0;
	for (unsigned int i = 0; i < nComp; ++i)
	{
		const unsigned int curSize = nStates[i] * nStates[i];
		if (vals.size() < curIdx + curSize)
			throw InvalidParameterException("Not enough elements in dataset " + dataSet + " (expected at least " +
											std::to_string(curIdx + curSize) + " but got only " +
											std::to_string(vals.size()) + ")");

		dest.pushBackSlice(curSize);
		ValType* const destCur = dest[i];
		for (unsigned int j = 0; j < curSize; ++j, ++curIdx)
			destCur[j] = vals[curIdx];
	}
}

/**
 * @brief Selects an array of possible section dependent vectorial parameters from a list
 * @details The parameters may be section dependent, but do not have to. The function is given
 *          a vector with all parameter values for all indices (vectorial parameter) and sections
 *          (if section dependent). The vector contains only @p nElements elements if the
 *          parameter is not section dependent. Otherwise, it should contain as much elements
 *          as @c nSec * nElements. The correct element array is selected according to the index of
 *          the current section and the section dependency of the parameter.
 *
 * @param [in] data Vector with all parameter values
 * @param [in] nElements Number of elements of the vectorial parameter (without taking section dependency into account)
 * @param [in] secIdx Index of the current section
 * @return Pointer to the first element of the correct vectorial parameter array
 */
template <typename T>
inline T const* getSectionDependentSlice(const std::vector<T>& data, unsigned int nElements, unsigned int secIdx)
{
	// Check if the vector contains enough elements
	if (cadet_unlikely(data.size() >= nElements * (secIdx + 1)))
	{
		// Advance data pointer to correct position
		T const* out = data.data();
		return out + nElements * secIdx;
	}
	else
		return data.data();
}

/**
 * @brief Selects a possibly section dependent scalar parameter from a list of parameters
 * @details The parameter may be section dependent, but does not have to. The function is given
 *          a vector with all parameter values. The vector contains only one element if the
 *          parameter is not section dependent. Otherwise, it should contain as much elements
 *          as there are sections. The correct element is selected according to the index of
 *          the current section and the section dependency of the parameter.
 *
 * @param [in] data Vector with all parameter values (for every section, or just one)
 * @param [in] secIdx Index of the current section
 * @return The correct scalar parameter
 */
template <typename T> inline const T& getSectionDependentScalar(const std::vector<T>& data, unsigned int secIdx)
{
	if (cadet_unlikely(data.size() > 1))
	{
		cadet_assert(data.size() > secIdx);
		return data[secIdx];
	}
	else
	{
		cadet_assert(data.size() == 1);
		return data[0];
	}
}

/**
 * @brief Registers a 1D parameter array
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Array with parameters to be registered
 * @param [in] size Number of elements in the array
 * @param [in] pic Callable that returns a ParameterId based on whether there is more than one item in the array and the
 * array index
 */
template <class ParamIdCreator>
inline void registerParam1DArray(std::unordered_map<ParameterId, active*>& map, active* params, unsigned int size,
								 ParamIdCreator&& pic)
{
	const bool multi = size > 1;
	for (unsigned int i = 0; i < size; ++i)
	{
		map[pic(multi, i)] = params + i;
	}
}

/**
 * @brief Registers a 1D parameter array
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Vector with parameters to be registered
 * @param [in] pic Callable that returns a ParameterId based on whether there is more than one item in the array and the
 * array index
 */
template <class ParamIdCreator>
inline void registerParam1DArray(std::unordered_map<ParameterId, active*>& map, std::vector<active>& params,
								 ParamIdCreator&& pic)
{
	registerParam1DArray(map, params.data(), params.size(), std::forward<ParamIdCreator>(pic));
}

/**
 * @brief Registers a sliced 1D parameter array
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Sliced vector with parameters to be registered
 * @param [in] pic Callable that returns a ParameterId based on slice index and item index within slice
 */
template <typename SliceContainer_t, class ParamIdCreator>
inline void registerParam1DNonUniform(std::unordered_map<ParameterId, active*>& map, SliceContainer_t& params,
									  ParamIdCreator pic)
{
	const unsigned int numSlices = params.slices();
	for (unsigned int s = 0; s < numSlices; ++s)
	{
		active* const slice = params[s];
		for (unsigned int i = 0; i < params.sliceSize(s); ++i)
		{
			map[pic(numSlices > 1, s, i)] = slice + i;
		}
	}
}

/**
 * @brief Registers a 2D parameter array
 * @details The linearized array is processed sequentially. For each linear index, the corresponding outer and inner
 * index is computed. For a row-major storage, this corresponds to row and column index, respectively. In this case, the
 * @p innerSize is the number of columns. The callable @p pic is used to construct a ParameterId based on whether there
 * are multiple outer indices (e.g., more than one row), the outer index, and the inner index.
 *
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Linearized array with parameters to be registered
 * @param [in] size Number of elements in the array
 * @param [in] pic Callable that returns a ParameterId based on whether more than one outer index is present, the outer
 * index, and the inner index
 * @param [in] innerSize Size of the inner dimension
 */
template <class ParamIdCreator>
inline void registerParam2DArray(std::unordered_map<ParameterId, active*>& map, active* params, unsigned int size,
								 ParamIdCreator&& pic, unsigned int innerSize)
{
	const bool multiOuter = size > innerSize;
	for (unsigned int i = 0; i < size; ++i)
	{
		const unsigned int idxOuter = i / innerSize;
		const unsigned int idxInner = i % innerSize;
		map[pic(multiOuter, idxOuter, idxInner)] = params + i;
	}
}

/**
 * @brief Registers a 2D parameter array
 * @details The linearized array is processed sequentially. For each linear index, the corresponding outer and inner
 * index is computed. For a row-major storage, this corresponds to row and column index, respectively. In this case, the
 * @p innerSize is the number of columns. The callable @p pic is used to construct a ParameterId based on whether there
 * are multiple outer indices (e.g., more than one row), the outer index, and the inner index.
 *
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Linearized vector with parameters to be registered
 * @param [in] pic Callable that returns a ParameterId based on whether more than one outer index is present, the outer
 * index, and the inner index
 * @param [in] innerSize Size of the inner dimension
 */
template <class ParamIdCreator>
inline void registerParam2DArray(std::unordered_map<ParameterId, active*>& map, std::vector<active>& params,
								 ParamIdCreator&& pic, unsigned int innerSize)
{
	registerParam2DArray(map, params.data(), params.size(), std::forward<ParamIdCreator>(pic), innerSize);
}

/**
 * @brief Registers a sliced 2D parameter array
 * @details The linearized array is processed sequentially. For each linear index, the corresponding outer, slice index,
 *          and item index in slice is computed. For a row-major storage, this corresponds to row, column slice, and
 *          slice item index, respectively. In this case, the sizes of the column slices are given by @p innerSizes.
 *          The callable @p pic is used to construct a ParameterId based on whether there are multiple outer indices
 *          (e.g., more than one row), the outer index, the inner slice index, and the item index in the slice.
 *
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Linearized vector with parameters to be registered
 * @param [in] pic Callable that returns a ParameterId based on whether more than one outer index is present, outer
 * index, inner slice index, item index within slice
 * @param [in] innerSices Array with sizes of the inner slices
 * @param [in] numSizes Number of slices, number of elements in @p innerSlices
 */
template <class ParamIdCreator>
inline void registerParam2DNonUniformArray(std::unordered_map<ParameterId, active*>& map, std::vector<active>& params,
										   ParamIdCreator pic, unsigned int const* innerSizes, unsigned int numSizes)
{
	const bool multiOuter = params.size() > std::accumulate(innerSizes, innerSizes + numSizes, 0u);
	unsigned int idx = 0;
	unsigned int idxOuter = 0;
	while (idx < params.size())
	{
		for (unsigned int i = 0; i < numSizes; ++i)
		{
			for (unsigned j = 0; j < innerSizes[i]; ++j, ++idx)
			{
				map[pic(multiOuter, idxOuter, i, j)] = &params[idx];
			}
		}
		++idxOuter;
	}
}

/**
 * @brief Registers a 3D parameter array
 * @details The linearized array is processed sequentially. For each linear index, the corresponding outer, mid, and
 * inner index is computed. For a page-row-major storage, this corresponds to page, row, and column index, respectively.
 * In this case, the @p innerSize is the number of columns and @p midSize is the number of rows. The callable @p pic is
 * used to construct a ParameterId based on whether there are multiple outer indices (e.g., more than one page), the
 * outer index, the mid index, and the inner index.
 *
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Linearized array with parameters to be registered
 * @param [in] size Number of elements in the array
 * @param [in] pic Callable that returns a ParameterId based on whether more than one outer index is present, outer
 * index, mid index, and inner index
 * @param [in] innerSize Size of the inner dimension
 * @param [in] midSize Size of the mid dimension
 */
template <class ParamIdCreator>
inline void registerParam3DArray(std::unordered_map<ParameterId, active*>& map, active* params, unsigned int size,
								 ParamIdCreator&& pic, unsigned int innerSize, unsigned int midSize)
{
	const bool multiOuter = size > innerSize * midSize;
	for (unsigned int i = 0; i < size; ++i)
	{
		const unsigned int idxOuter = i / (innerSize * midSize);
		const unsigned int idxRem = i % (innerSize * midSize);
		const unsigned int idxMid = idxRem / innerSize;
		const unsigned int idxInner = idxRem % innerSize;
		map[pic(multiOuter, idxOuter, idxMid, idxInner)] = params + i;
	}
}

/**
 * @brief Registers a 3D parameter array
 * @details The linearized array is processed sequentially. For each linear index, the corresponding outer, mid, and
 * inner index is computed. For a page-row-major storage, this corresponds to page, row, and column index, respectively.
 * In this case, the @p innerSize is the number of columns and @p midSize is the number of rows. The callable @p pic is
 * used to construct a ParameterId based on whether there are multiple outer indices (e.g., more than one page), the
 * outer index, the mid index, and the inner index.
 *
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Linearized vector with parameters to be registered
 * @param [in] pic Callable that returns a ParameterId based on whether more than one outer index is present, outer
 * index, mid index, and inner index
 * @param [in] innerSize Size of the inner dimension
 * @param [in] midSize Size of the mid dimension
 */
template <class ParamIdCreator>
inline void registerParam3DArray(std::unordered_map<ParameterId, active*>& map, std::vector<active>& params,
								 ParamIdCreator&& pic, unsigned int innerSize, unsigned int midSize)
{
	registerParam3DArray(map, params.data(), params.size(), std::forward<ParamIdCreator>(pic), innerSize, midSize);
}

/**
 * @brief Registers a sliced 3D parameter array
 * @details The linearized array is processed sequentially. For each linear index, the corresponding outer, mid, slice
 * index, and item index in slice is computed. For a page-row-major storage, this corresponds to page, row, column
 * slice, and slice item index, respectively. In this case, the sizes of the column slices are given by @p innerSizes
 * and
 *          @p midSize is the number of rows. The callable @p pic is used to construct a ParameterId based on whether
 * there are multiple outer indices (e.g., more than one page), the outer index, the mid index, the inner slice index,
 * and the item index in the slice.
 *
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Linearized vector with parameters to be registered
 * @param [in] pic Callable that returns a ParameterId based on whether more than one outer index is present, outer
 * index, mid index, inner slice index, item index within slice
 * @param [in] innerSices Array with sizes of the inner slices
 * @param [in] numSizes Number of slices, number of elements in @p innerSlices
 * @param [in] midSize Size of the mid dimension
 */
template <class ParamIdCreator>
inline void registerParam3DNonUniformArray(std::unordered_map<ParameterId, active*>& map, std::vector<active>& params,
										   ParamIdCreator pic, unsigned int const* innerSizes, unsigned int numSizes,
										   unsigned int midSize)
{
	const bool multiOuter = params.size() > std::accumulate(innerSizes, innerSizes + numSizes, 0u) * midSize;
	unsigned int idx = 0;
	unsigned int idxOuter = 0;
	while (idx < params.size())
	{
		for (unsigned int k = 0; k < midSize; ++k)
		{
			for (unsigned int i = 0; i < numSizes; ++i)
			{
				for (unsigned j = 0; j < innerSizes[i]; ++j, ++idx)
				{
					map[pic(multiOuter, idxOuter, k, i, j)] = &params[idx];
				}
			}
		}
		++idxOuter;
	}
}

/**
 * @brief Registers scalar parameters that may be section dependent
 *
 * @param [in] nameHash Hashed name of the parameter
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Vector with parameters to be registered (only 1 if not section dependent)
 * @param [in] unitOpIdx Index of the unit operation
 */
inline void registerScalarSectionDependentParam(const StringHash nameHash,
												std::unordered_map<ParameterId, active*>& map,
												std::vector<active>& params, UnitOpIdx unitOpIdx,
												ParticleTypeIdx parType)
{
	registerParam1DArray(map, params, [=](bool multi, unsigned int sec) {
		return makeParamId(nameHash, unitOpIdx, CompIndep, parType, BoundStateIndep, ReactionIndep,
						   multi ? sec : SectionIndep);
	});
}

/**
 * @brief Registers scalar parameters that may be bound phase dependent
 *
 * @param [in] nameHash Hashed name of the parameter
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Vector with parameters to be registered (only 1 if not bound state dependent)
 * @param [in] unitOpIdx Index of the unit operation
 */
inline void registerScalarBoundStateDependentParam(const StringHash nameHash,
												   std::unordered_map<ParameterId, active*>& map,
												   std::vector<active>& params, UnitOpIdx unitOpIdx,
												   ParticleTypeIdx parType)
{
	registerParam1DArray(map, params, [=](bool multi, unsigned int bnd) {
		return makeParamId(nameHash, unitOpIdx, CompIndep, parType, bnd, ReactionIndep, SectionIndep);
	});
}

/**
 * @brief Registers vector parameters (component dependent)
 * @param [in] nameHash Hashed name of the parameter
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Vector with parameters to be registered
 * @param [in] unitOpIdx Index of the unit operation
 */
inline void registerComponentDependentParam(const StringHash nameHash, std::unordered_map<ParameterId, active*>& map,
											std::vector<active>& params, UnitOpIdx unitOpIdx, ParticleTypeIdx parType)
{
	registerParam1DArray(map, params, [=](bool multi, unsigned int comp) {
		return makeParamId(nameHash, unitOpIdx, comp, parType, BoundStateIndep, ReactionIndep, SectionIndep);
	});
}

/**
 * @brief Registers vector parameters (component dependent) that may be section dependent
 * @details The ordering is section-major, that is comp0sec0, comp1sec0, comp2sec0, comp0sec1, comp1sec1, comp2sec1, ...
 *          The number of components is given in @p compStride.
 *
 * @param [in] nameHash Hashed name of the parameter
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Vector with parameters to be registered
 * @param [in] unitOpIdx Index of the unit operation
 * @param [in] compStride Distance between two parameters with consecutive sections
 */
inline void registerComponentSectionDependentParam(const StringHash nameHash,
												   std::unordered_map<ParameterId, active*>& map,
												   std::vector<active>& params, UnitOpIdx unitOpIdx,
												   ParticleTypeIdx parType, unsigned int compStride)
{
	registerParam2DArray(
		map, params,
		[=](bool multi, unsigned int sec, unsigned int comp) {
			return makeParamId(nameHash, unitOpIdx, comp, parType, BoundStateIndep, ReactionIndep,
							   multi ? sec : SectionIndep);
		},
		compStride);
}

/**
 * @brief Registers vector parameters (component dependent) that are also dependent on the bound phase
 * @details The ordering is state-major, that is comp0state0, comp1state0, comp2state0, comp0state1, comp1state1,
 * comp2state1, ... The number of components is given in @p compStride.
 *
 * @param [in] nameHash Hashed name of the parameter
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Vector with parameters to be registered
 * @param [in] unitOpIdx Index of the unit operation
 * @param [in] compStride Distance between two parameters with consecutive bound phases
 */
inline void registerComponentBoundStateDependentParam(const StringHash nameHash,
													  std::unordered_map<ParameterId, active*>& map,
													  std::vector<active>& params, UnitOpIdx unitOpIdx,
													  ParticleTypeIdx parType, unsigned int compStride)
{
	registerParam2DArray(
		map, params,
		[=](bool multi, unsigned int bnd, unsigned int comp) {
			return makeParamId(nameHash, unitOpIdx, comp, parType, bnd, ReactionIndep, SectionIndep);
		},
		compStride);
}

/**
 * @brief Registers vector parameters (component dependent) that are also dependent on the bound phase
 * @details The ordering is state-major, that is comp0state0, comp1state0, comp2state0, comp0state1, comp1state1,
 * comp2state1, ... It is assumed that only one bound state is present.
 *
 * @param [in] nameHash Hashed name of the parameter
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Vector with parameters to be registered
 * @param [in] unitOpIdx Index of the unit operation
 */
inline void registerComponentBoundStateDependentParam(const StringHash nameHash,
													  std::unordered_map<ParameterId, active*>& map,
													  std::vector<active>& params, UnitOpIdx unitOpIdx,
													  ParticleTypeIdx parType)
{
	registerComponentBoundStateDependentParam(nameHash, map, params, unitOpIdx, parType, params.size());
}

/**
 * @brief Registers parameters that depend on component and bound state stored in state-major ordering
 * @details The ordering is state-major, that is comp0state0, comp1state0, comp2state0, comp0state1, comp1state1,
 * comp2state1, ...
 *
 * @param [in] nameHash Hashed name of the parameter
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Sliced vector with parameters to be registered (each slice corresponds to one bound state and
 * contains all components)
 * @param [in] unitOpIdx Index of the unit operation
 * @tparam SliceContainer_t Type of the slice container, such as @c cadet::util::SlicedVector
 */
template <typename SliceContainer_t>
inline void registerComponentBoundStateDependentParam(const StringHash nameHash,
													  std::unordered_map<ParameterId, active*>& map,
													  SliceContainer_t& params, UnitOpIdx unitOpIdx,
													  ParticleTypeIdx parType)
{
	registerParam1DNonUniform(map, params, [=](bool multi, unsigned int state, unsigned int comp) {
		return makeParamId(nameHash, unitOpIdx, comp, parType, state, ReactionIndep, SectionIndep);
	});
}

/**
 * @brief Registers parameters that depend on component and bound state stored in component-major ordering
 * @details The ordering is component-major, that is comp0state0, comp0state1, comp1state0, comp1state1, comp2state0,
 * comp2state1, ...
 *
 * @param [in] nameHash Hashed name of the parameter
 * @param [in,out] map Map to which the parameters are added
 * @param [in] params Sliced vector with parameters to be registered (each slice corresponds to one component and
 * contains all bound states)
 * @param [in] unitOpIdx Index of the unit operation
 * @tparam SliceContainer_t Type of the slice container, such as @c cadet::util::SlicedVector
 */
template <typename SliceContainer_t>
inline void registerComponentBoundStateDependentParamCompMajor(const StringHash nameHash,
															   std::unordered_map<ParameterId, active*>& map,
															   SliceContainer_t& params, UnitOpIdx unitOpIdx,
															   ParticleTypeIdx parType)
{
	registerParam1DNonUniform(map, params, [=](bool multi, unsigned int comp, unsigned int state) {
		return makeParamId(nameHash, unitOpIdx, comp, parType, state, ReactionIndep, SectionIndep);
	});
}

} // namespace cadet

#endif // LIBCADET_PARAMREADERHELPER_HPP_
