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
 * Provides several numerical stencil implementations.
 */

#ifndef LIBCADET_STENCIL_HPP_
#define LIBCADET_STENCIL_HPP_

#include <initializer_list>
#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"

namespace cadet
{
/**
 * @brief A stencil that works on the original state vector
 * @details Supports strides on the state vector for non-consecutive ordering.
 *          Stencils are supposed to be a lightweight view into the state vector providing easy access to
 *          cells of one single component.
 * @tparam T Underlying state element type
 */
template <typename T> class StridedStencil
{
public:
	/**
	 * @brief Creates a StridedStencil on the given array
	 * @param [in] data Pointer to original state vector
	 * @param [in] stride Stride in the state vector (i.e., number of elements between consecutive data items)
	 */
	StridedStencil(T const* const data, const unsigned int stride) : _data(data), _stride(stride)
	{
	}

	inline const T& operator[](const int idx) const
	{
		return _data[idx * static_cast<int>(_stride)];
	}

	/**
	 * @brief Advances the stencil to the next cell
	 */
	inline void advance() CADET_NOEXCEPT
	{
		_data += _stride;
	}

	inline void advance(const T&) CADET_NOEXCEPT
	{
		advance();
	}

	inline StridedStencil& operator++()
	{
		advance();
		return *this;
	}

protected:
	T const* _data;
	const unsigned int _stride;
};

/**
 * @brief A stencil that caches its values into a separate array for spatial locality in memory
 * @details Stencils are supposed to be a lightweight view into the state vector providing easy
 *          access to cells of one single component. The stencil size has to be odd.
 * @tparam T Underlying state element type
 * @tparam MemoryPoolType Memory pool used to allocate the cache of the stencil
 */
template <typename T, class MemoryPoolType> class CachingStencil
{
public:
	/**
	 * @brief Creates a CachingStencil with given size
	 * @details The @p size has to be odd.
	 * @param [in] size Size of the stencil
	 * @param [in] memPool Memory pool used for the stencil (e.g., ArrayPool)
	 */
	CachingStencil(const unsigned int size, MemoryPoolType& memPool)
		: _cache(memPool.template create<T>(size)), _size(size), _offset(0u), _memPool(memPool)
	{
	}
	CachingStencil(const unsigned int size, MemoryPoolType& memPool, unsigned int offset)
		: _cache(memPool.template create<T>(size) + offset), _size(size), _offset(offset), _memPool(memPool)
	{
	}
	CachingStencil(const std::initializer_list<T> vals, MemoryPoolType& memPool) : CachingStencil(vals.size(), memPool)
	{
		initialize(vals);
	}
	CachingStencil(const std::initializer_list<T> vals, MemoryPoolType& memPool, unsigned int offset)
		: CachingStencil(vals.size(), memPool, offset)
	{
		initialize(vals);
	}
	~CachingStencil() CADET_NOEXCEPT
	{
		_memPool.template destroy<T>();
	}

	inline const T& operator[](const int idx) const
	{
		cadet_assert(idx >= -static_cast<int>(_offset));
		cadet_assert(idx < static_cast<int>(_size - _offset));
		return _cache[idx];
	}
	inline T& operator[](const int idx)
	{
		cadet_assert(idx >= -static_cast<int>(_offset));
		cadet_assert(idx < static_cast<int>(_size - _offset));
		return _cache[idx];
	}

	inline const T& native(const unsigned int i) const
	{
		cadet_assert(i < _size);
		return _cache[static_cast<int>(i) - _offset];
	}

	inline void initialize(const std::initializer_list<T> vals)
	{
		cadet_assert(vals.size() <= _size);

		T const* ptr = vals.begin();

		for (int i = 0; i < static_cast<int>(_size); ++i, ++ptr)
			_cache[i - static_cast<int>(_offset)] = *ptr;
	}

	/**
	 * @brief Advances the stencil to the next cell
	 * @param val New volume average added to the stencil
	 */
	inline void advance(const T& val) CADET_NOEXCEPT
	{
		// Rotate cache backwards, that is, assign ..., _cache[0] = _cache[1], _cache[1] = _cache[2]
		for (int i = -static_cast<int>(_offset); i < static_cast<int>(_size) - 1 - static_cast<int>(_offset); ++i)
			_cache[i] = _cache[i + 1];

		// Update cache
		_cache[static_cast<int>(_size) - static_cast<int>(_offset) - 1] = val;
	}

protected:
	T* const _cache;            //!< Cache for stencil elements
	const unsigned int _size;   //!< Size of the stencil
	const unsigned int _offset; //!< Offset of the pointer in the stencil
	MemoryPoolType& _memPool;   //!< Memory pool for the stencil
};

/**
 * @brief Specialization of the CachingStencil for doubles
 * @todo Is this specialization of the CachingStencil necessary?
 */
template <class MemoryPoolType> class CachingStencil<double, MemoryPoolType>
{
public:
	CachingStencil(const unsigned int size, MemoryPoolType& memPool)
		: _cache(memPool.template create<double>(size)), _size(size), _offset(0u), _memPool(memPool)
	{
	}
	CachingStencil(const unsigned int size, MemoryPoolType& memPool, unsigned int offset)
		: _cache(memPool.template create<double>(size) + offset), _size(size), _offset(offset), _memPool(memPool)
	{
	}
	CachingStencil(const std::initializer_list<double> vals, MemoryPoolType& memPool)
		: CachingStencil(vals.size(), memPool)
	{
		initialize(vals);
	}
	CachingStencil(const std::initializer_list<double> vals, MemoryPoolType& memPool, unsigned int offset)
		: CachingStencil(vals.size(), memPool, offset)
	{
		initialize(vals);
	}
	~CachingStencil() CADET_NOEXCEPT
	{
		_memPool.template destroy<double>();
	}

	inline double operator[](const int idx) const
	{
		cadet_assert(idx >= -static_cast<int>(_offset));
		cadet_assert(idx < static_cast<int>(_size - _offset));
		return _cache[idx];
	}
	inline double& operator[](const int idx)
	{
		cadet_assert(idx >= -static_cast<int>(_offset));
		cadet_assert(idx < static_cast<int>(_size - _offset));
		return _cache[idx];
	}

	inline double native(const unsigned int i) const
	{
		cadet_assert(i < _size);
		return _cache[static_cast<int>(i) - static_cast<int>(_offset)];
	}

	inline void advance(const double val) CADET_NOEXCEPT
	{
		// Rotate cache backwards, that is, assign ..., _cache[0] = _cache[1], _cache[1] = _cache[2]
		for (int i = -static_cast<int>(_offset); i < static_cast<int>(_size) - 1 - static_cast<int>(_offset); ++i)
			_cache[i] = _cache[i + 1];

		// Update cache
		_cache[static_cast<int>(_size) - static_cast<int>(_offset) - 1] = val;
	}

	inline void initialize(const std::initializer_list<double> vals)
	{
		cadet_assert(vals.size() <= _size);

		double const* ptr = vals.begin();

		for (int i = 0; i < static_cast<int>(_size); ++i, ++ptr)
			_cache[i - static_cast<int>(_offset)] = *ptr;
	}

protected:
	double* const _cache;
	const unsigned int _size;
	const unsigned int _offset;
	MemoryPoolType& _memPool;
};

} // namespace cadet

#endif // LIBCADET_STENCIL_HPP_
