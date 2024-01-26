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
 * Defines a SlicedVector which replaces vectors of vectors
 */

#ifndef LIBCADET_SLICEDVECTOR_HPP_
#define LIBCADET_SLICEDVECTOR_HPP_

#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"
#include <vector>
#include <algorithm>

namespace cadet
{

namespace util
{

/**
 * @brief A dynamically growing vector of slices (vectors)
 * @details Replaces an append-only vector<vector<T>> data structure.
 *          The data is linearized to a single array and indices of start
 *          and end of each slice are saved in a separate vector.
 * @tparam T Type of the saved data
 */
template <typename T> class SlicedVector
{
public:
	typedef typename std::vector<T>::size_type size_type;

	/**
	 * @brief Creates an empty SlicedVector
	 */
	SlicedVector() : _index(1, 0)
	{
	}
	~SlicedVector() CADET_NOEXCEPT
	{
	}

	// Default copy and move mechanisms
	SlicedVector(const SlicedVector<T>& cpy) = default;
	SlicedVector(SlicedVector<T>&& cpy) CADET_NOEXCEPT = default;

	inline SlicedVector<T>& operator=(const SlicedVector<T>& cpy) = default;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	inline SlicedVector<T>& operator=(SlicedVector<T>&& cpy) CADET_NOEXCEPT = default;
#else
	inline SlicedVector<T>& operator=(SlicedVector<T>&& cpy) = default;
#endif

	/**
	 * @brief Checks whether this SlicedVector is empty
	 * @return @c true if it is empty, otherwise @c false
	 */
	inline bool empty() const
	{
		return _index.size() == 1;
	}

	/**
	 * @brief Clears the SlicedVector and deletes all stored items
	 */
	inline void clear()
	{
		// Always keep the first element (0) in _index
		_index.resize(1);
		_values.clear();
	}

	/**
	 * @brief Appends a single item (slice of size 1)
	 * @param [in] item Item to be appended
	 */
	inline void pushBack(const T& item)
	{
		_index.push_back(_index.back() + 1);
		_values.push_back(item);
	}

	/**
	 * @brief Appends a slice of the given size
	 * @details The new elements are default constructed.
	 * @param [in] size Size of the slice
	 */
	inline void pushBackSlice(size_type size)
	{
		_index.push_back(_index.back() + size);
		_values.resize(_values.size() + size);
	}

	/**
	 * @brief Appends a given slice
	 * @param [in] slice Slice to append
	 */
	inline void pushBackSlice(const std::vector<T>& slice)
	{
		_index.push_back(_index.back() + slice.size());
		_values.insert(_values.end(), slice.begin(), slice.end());
	}

	/**
	 * @brief Appends a given slice
	 * @param [in] data Pointer to the first element of the slice
	 * @param [in] size Size of the slice
	 */
	inline void pushBackSlice(T const* data, size_type size)
	{
		_index.push_back(_index.back() + size);
		_values.insert(_values.end(), data, data + size);
	}

	/**
	 * @brief Append an element to the last slice
	 * @param [in] value Element to append
	 */
	inline void pushBackInLastSlice(const T& value)
	{
		cadet_assert(!empty());

		// Increase end index of last slice
		++_index.back();
		_values.push_back(value);
	}

	/**
	 * @brief Removes the last element in the last slice
	 */
	inline void popBackInLastSlice()
	{
		cadet_assert(!empty());
		cadet_assert(sliceSize(slices() - 1) > 0);

		// Decrease end index of last slice
		--_index.back();
		_values.pop_back();
	}

	/**
	 * @brief Checks if this SlicedVector is identical to a given one
	 * @param [in] rhs SlicedVector to compare with
	 * @return @c true if both vectors are the same, otherwise @c false
	 */
	inline bool operator==(const SlicedVector& rhs) const
	{
		return (_index == rhs._index) && (_values == rhs._values);
	}

	/**
	 * @brief Swaps this SlicedVector with a given SlicedVector
	 * @param [in,out] rhs SlicedVector to be swapped with
	 */
	inline void swap(SlicedVector& rhs)
	{
		_values.swap(rhs._values);
		_index.swap(rhs._index);
	}

	/**
	 * @brief Returns the number of slices
	 * @return Number of slices
	 */
	inline size_type slices() const CADET_NOEXCEPT
	{
		return _index.size() - 1;
	}

	/**
	 * @brief Returns a pointer to the first element of a given slice
	 * @param [in] idxSlice Index of the slice
	 * @return Pointer to the first element of the given slice
	 */
	inline T const* operator[](size_type idxSlice) const
	{
		return _values.data() + _index[idxSlice];
	}

	/**
	 * @brief Returns a pointer to the first element of a given slice
	 * @param [in] idxSlice Index of the slice
	 * @return Pointer to the first element of the given slice
	 */
	inline T* operator[](size_type idxSlice)
	{
		return _values.data() + _index[idxSlice];
	}

	/**
	 * @brief Returns the element at the given position in the given slice
	 * @param [in] idxSlice Index of the slice
	 * @param [in] idxElem Index of the element within the slice
	 * @return Element at the given index of the given slice
	 */
	inline const T& operator()(size_type idxSlice, size_type idxElem) const
	{
		return _values[_index[idxSlice] + idxElem];
	}

	/**
	 * @brief Returns the element at the given position in the given slice
	 * @param [in] idxSlice Index of the slice
	 * @param [in] idxElem Index of the element within the slice
	 * @return Element at the given index of the given slice
	 */
	inline T& operator()(size_type idxSlice, size_type idxElem)
	{
		return _values[_index[idxSlice] + idxElem];
	}

	/**
	 * @brief Returns the element at the given linear index
	 * @param [in] idx Linear index of the requested element
	 * @return Element at the given linear index
	 */
	inline T& native(size_type idx)
	{
		return _values[idx];
	}

	/**
	 * @brief Returns the element at the given linear index
	 * @param [in] idx Linear index of the requested element
	 * @return Element at the given linear index
	 */
	inline const T& native(size_type idx) const
	{
		return _values[idx];
	}

	/**
	 * @brief Returns the element at the given position in the given slice
	 * @param [in] idxSlice Index of the slice
	 * @param [in] idxElem Index of the element within the slice
	 * @return Element at the given index of the given slice
	 */
	inline const T& at(size_type idxSlice, size_type idxElem) const
	{
		return _values[_index[idxSlice] + idxElem];
	}

	/**
	 * @brief Returns the element at the given position in the given slice
	 * @param [in] idxSlice Index of the slice
	 * @param [in] idxElem Index of the element within the slice
	 * @return Element at the given index of the given slice
	 */
	inline T& at(size_type idxSlice, size_type idxElem)
	{
		return _values[_index[idxSlice] + idxElem];
	}

	/**
	 * @brief Returns a pointer to the first element of a given slice
	 * @param [in] idxSlice Index of the slice
	 * @return Pointer to the first element of the given slice
	 */
	inline T const* at(size_type idxSlice) const
	{
		return _values.data() + _index[idxSlice];
	}

	/**
	 * @brief Returns a pointer to the first element of a given slice
	 * @param [in] idxSlice Index of the slice
	 * @return Pointer to the first element of the given slice
	 */
	inline T* at(size_type idxSlice)
	{
		return _values.data() + _index[idxSlice];
	}

	/**
	 * @brief Returns a pointer to the first element of the last slice
	 * @return Pointer to the first element of the last slice
	 */
	inline T const* back() const
	{
		cadet_assert(!empty());
		return &_values[_index[_index.size() - 2]];
	}

	inline T* back()
	{
		cadet_assert(!empty());
		return &_values[_index[_index.size() - 2]];
	}

	/**
	 * @brief Returns the number of elements in a slice
	 * @param [in] idxSlice Index of the slice
	 * @return Number of elmenets in the slice
	 */
	inline size_type sliceSize(size_type idxSlice) const
	{
		return _index[idxSlice + 1] - _index[idxSlice];
	}

	/**
	 * @brief Returns the total number of stored items
	 * @details All items in every slice are counted.
	 * @return Total number of stored items
	 */
	inline size_type size() const CADET_NOEXCEPT
	{
		return _values.size();
	}

	/**
	 * @brief Finds the slice that contains a given item
	 * @param [in] value Item to be found
	 * @return Index of the slice that contains the requested @p value,
	 *         or a value larger than slices() if @p value was not found
	 */
	inline size_type findSlice(const T& value) const
	{
		for (size_type i = 0; i < slices(); ++i)
		{
			for (size_type j = _index[i]; j < _index[i + 1]; ++j)
			{
				if (_values[j] == value)
				{
					return i;
				}
			}
		}
		return slices() + 1;
	}

	/**
	 * @brief Finds the slice that contains a given item
	 * @param [in] value Item to be found
	 * @param [out] len The size of the found slice
	 * @return Pointer to the first item of the slice that contains the requested @p value,
	 *         @c nullptr if @p value was not found
	 */
	inline T const* findSlice(const T& value, size_type& len) const
	{
		for (size_type i = 0; i < slices(); ++i)
		{
			for (size_type j = _index[i]; j < _index[i + 1]; ++j)
			{
				if (_values[j] == value)
				{
					len = _index[i + 1] - _index[i];
					return &_values[_index[i]];
				}
			}
		}
		return nullptr;
	}

	/**
	 * @brief Finds the slice that contains a given item
	 * @param [in] value Item to be found
	 * @param [out] index Index of the item in the slice, if found
	 * @param [out] linearIndex Index of the item in the linearized array, if found
	 * @return Index of the slice that contains the requested @p value,
	 *         or a value larger than slices() if @p value was not found
	 */
	inline size_type findElementAndSlice(const T& value, size_type& index, size_type& linearIndex) const
	{
		for (size_type i = 0; i < slices(); ++i)
		{
			for (size_type j = _index[i]; j < _index[i + 1]; ++j)
			{
				if (_values[j] == value)
				{
					index = j - _index[i];
					linearIndex = j;
					return i;
				}
			}
		}
		return slices() + 1;
	}

	/**
	 * @brief Finds the index of a given item in the linearized storage
	 * @param [in] value Item to be found
	 * @return Index of the requested @p value in the linearized storage,
	 *         or a value larger than size() if @p value was not found
	 */
	inline size_type findLinear(const T& value) const
	{
		for (size_type i = 0; i < _values.size(); ++i)
		{
			if (_values[i] == value)
				return i;
		}
		return _values.size() + 1;
	}

	/**
	 * @brief Returns the offset of the slice with the given index in the linear array
	 * @param [in] idx Index of the slice
	 * @return Offset of the given slice in the linearized storage
	 */
	inline size_type sliceOffset(size_type idx) const
	{
		return _index[idx];
	}

	/**
	 * @brief Reserves memory without initializing it
	 * @details Corresponds to std::vector<T>::reserve().
	 * @param [in] numElems Number of elements to reserve space for
	 */
	inline void reserve(size_type numElems)
	{
		_values.reserve(numElems);
	}

	/**
	 * @brief Reserves memory without initializing it
	 * @details Corresponds to std::vector<T>::reserve().
	 * @param [in] numElems Number of elements to reserve space for
	 * @param [in] numSlices Number of slices to reserve space for
	 */
	inline void reserve(size_type numElems, size_type numSlices)
	{
		_values.reserve(numElems);
		_index.reserve(numSlices + 1);
	}

	/**
	 * @brief Fills all slices with the same value
	 * @param [in] val Value that is copied to all elements of all slices
	 */
	inline void fill(const T& val = T())
	{
		std::fill(_values.begin(), _values.end(), val);
	}

	/**
	 * @brief Returns a pointer to the first element of the underlying linear array
	 * @return Pointer to first element of underlying linear array
	 */
	inline T* data() CADET_NOEXCEPT
	{
		return _values.data();
	}

	inline T const* data() const CADET_NOEXCEPT
	{
		return _values.data();
	}

	/**
	 * @brief Returns a pointer to the first element of the slice index array
	 * @return Pointer to first element of slice index array
	 */
	inline size_type* indices() CADET_NOEXCEPT
	{
		return _index.data();
	}

	inline size_type const* indices() const CADET_NOEXCEPT
	{
		return _index.data();
	}

protected:
	std::vector<T> _values;        //!< Holds all values in a linearized fashion
	std::vector<size_type> _index; //!< Holds starting indices of slices
};

} // namespace util
} // namespace cadet

#endif // LIBCADET_SLICEDVECTOR_HPP_
