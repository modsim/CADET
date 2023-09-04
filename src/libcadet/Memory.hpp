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
 * Provides functionality for constructing arrays and objects in raw memory.
 */

#ifndef LIBCADET_MEMORY_HPP_
#define LIBCADET_MEMORY_HPP_

#include "common/CompilerSpecific.hpp"
#include <memory>
#include <new>
#ifdef CADET_DEBUG
#include <functional>
#endif

namespace cadet
{
/**
 * @brief Treats a memory block as array of the given type
 * @details Treats a given memory block as an array of the given type.
 *          It is assumed that the pointer @p mem is correctly aligned for the given type @p T.
 *
 *          The array has to be released after use by calling releaseRawArray().
 *
 *          See
 *             https://stackoverflow.com/questions/15254/can-placement-new-for-arrays-be-used-in-a-portable-way
 *             https://stackoverflow.com/questions/41624685/is-placement-new-legally-required-for-putting-an-int-into-a-char-array
 *             https://stackoverflow.com/questions/13466556/aligned-storage-and-strict-aliasing
 *
 * @param [in] mem Pointer to memory block
 * @param [in] numElements Number of array elements
 * @return Pointer to first array element of the given type
 */
template <typename T> inline T* rawMemoryAsArray(void* mem, unsigned int numElements)
{
	if (cadet_unlikely((mem == nullptr) || (numElements == 0)))
		return nullptr;

	// Call constructor on every element
	T* const ptr = reinterpret_cast<T*>(mem);
	for (T* cur = ptr + 1; cur != ptr + numElements; ++cur)
		new (cur) T;

	return new (ptr) T;

	// This could be done differently in C++17: Do a full loop
	// from 0 to numElements - 1 and use std::launder() on the
	// first element instead of taking the result of placement
	// new operator `new(ptr) T`.
}

/**
 * @brief Releases an array created from rawMemoryAsArray()
 * @details Calls the destructor on each array element.
 *          This is a no-op for integral types (e.g., int, double).
 *
 * @param [in] mem Pointer to first element of array in raw memory block
 * @param [in] numElements Number of array elements
 */
template <typename T> inline void releaseRawArray(T* const mem, unsigned int numElements)
{
	if (cadet_unlikely((mem == nullptr) || (numElements == 0)))
		return;

	// Call destructor on every element
	// This should actually be in reverse order (last items might depend on first items)
	for (T* cur = mem; cur != mem + numElements; ++cur)
		cur->~T();
}

/**
 * @brief Releases a scalar
 * @details Calls the destructor of the item.
 *          This is a no-op for integral types (e.g., int, double).
 *
 * @param [in] mem Pointer to scalar item in raw memory block
 */
template <typename T> inline void releaseRawScalar(T* const mem)
{
	if (cadet_unlikely(mem == nullptr))
		return;

	mem->~T();
}

/**
 * @brief Manages a buffer for arrays of various types
 * @details An ArrayPool manages a block of memory that can be used to create arrays of
 *          various types. It is initialized to allocate a block of memory big enough to
 *          hold the maximum number of elements of the biggest type. The user can then
 *          create arrays of different types and sizes without actually allocating new
 *          memory (saves performance). Note that the elements have to be default constructible.
 */
class ArrayPool
{
public:
	/**
	 * @brief Creates an empty ArrayPool
	 */
	ArrayPool() : _mem(nullptr), _numElements(0)
	{
#ifdef CADET_DEBUG
		_capacity = 0;
#endif
	}

	/**
	 * @brief Creates an ArrayPool with the given size in bytes
	 * @details The size of the pool should be computed by taking @c sizeof(BiggestType) * maxNumElements
	 *
	 * @param [in] maxBytes Size of the pool in bytes
	 * @param [in] maxAlign Most strict alignment of all types that the pool will hold
	 */
	ArrayPool(unsigned int maxBytes, unsigned int maxAlign) : _mem(::operator new(maxBytes + maxAlign)), _numElements(0)
	{
#ifdef CADET_DEBUG
		_capacity = maxBytes;
#endif
	}

	/**
	 * @brief Creates an ArrayPool with the given size in bytes
	 * @details The size of the pool should be computed by taking @c sizeof(BiggestType) * maxNumElements
	 *
	 * @param [in] maxBytes Size of the pool in bytes
	 */
	ArrayPool(unsigned int maxBytes) : ArrayPool(maxBytes, 0)
	{
	}

	~ArrayPool() CADET_NOEXCEPT
	{
		::operator delete(_mem);
	}

	/**
	 * @brief Resizes the memory pool to the given size
	 * @details Already created arrays are invalid as the underlying memory pool is reallocated.
	 *          The user has to make sure that the created array is destroyed prior to calling resize().
	 *
	 * @param [in] maxBytes Size of the pool in bytes
	 */
	inline void resize(unsigned int maxBytes)
	{
		resize(maxBytes, 0);
	}

	/**
	 * @brief Resizes the memory pool to the given size
	 * @details Already created arrays are invalid as the underlying memory pool is reallocated.
	 *          The user has to make sure that the created array is destroyed prior to calling resize().
	 *
	 * @param [in] maxBytes Size of the pool in bytes
	 * @param [in] maxAlign Most strict alignment of all types that the pool will hold
	 */
	void resize(unsigned int maxBytes, unsigned int maxAlign)
	{
		::operator delete(_mem);
		_mem = ::operator new(maxBytes + maxAlign);
		_numElements = 0;

#ifdef CADET_DEBUG
		_capacity = maxBytes;
#endif
	}

	/**
	 * @brief Creates an array in the pool
	 * @details Since the array is always created in the same memory, previous arrays will be overwritten.
	 *
	 * @param [in] n Number of elements
	 * @tparam T Type of elements
	 * @return Pointer to array of the requested size and type
	 */
	template <typename T> inline T* create(unsigned int n)
	{
		cadet_assert(sizeof(T) * n <= _capacity);
		cadet_assert(sizeof(T) % alignof(T) == 0);
		_numElements = n;

		// Align the memory
		void* ptr = _mem;
		std::size_t space = sizeof(T) + alignof(T);
#ifdef CADET_DEBUG
		void* const ptr2 = std::align(alignof(T), sizeof(T), ptr, space);
		cadet_assert(ptr2 != nullptr);
#else
		std::align(alignof(T), sizeof(T), ptr, space);
#endif

		return rawMemoryAsArray<T>(ptr, n);
	}

	/**
	 * @brief Destroys a currently active array
	 * @details The destructor of the array elements is called
	 *
	 * @tparam T Type of elements
	 */
	template <typename T> inline void destroy()
	{
		// Get aligned pointer to first item
		void* ptr = _mem;
		std::size_t space = sizeof(T) + alignof(T);
#ifdef CADET_DEBUG
		void* const ptr2 = std::align(alignof(T), sizeof(T), ptr, space);
		cadet_assert(ptr2);
#else
		std::align(alignof(T), sizeof(T), ptr, space);
#endif

		// Call destructor on every element
		releaseRawArray(reinterpret_cast<T*>(ptr), _numElements);
		_numElements = 0;
	}

	inline unsigned int numElements() const CADET_NOEXCEPT
	{
		return _numElements;
	}

protected:
	void* _mem;                //<! Memory block
	unsigned int _numElements; //<! Current number of created elements
#ifdef CADET_DEBUG
	unsigned int _capacity; //<! Capacity of the pool, only available in debug builds
#endif
};

/**
 * @brief Tool for computing the size of a linear memory buffer
 * @details A linear memory buffer can hold different sets of items. This class estimates
 *          the size of the buffer for different item sets. An item set is defined by
 *          calling add<T>(...) for each item and closing the set with a call to commit().
 *          The required size of the buffer in bytes is computed by bufferSize().
 */
class LinearMemorySizer
{
public:
	LinearMemorySizer() : _maxSize(0), _curSize(0)
	{
	}

	/**
	 * @brief Adds a single item of type T to the item set
	 * @tparam T Type of the array items
	 */
	template <typename T> inline void add()
	{
		_curSize += sizeof(T) + alignof(T);
	}

	/**
	 * @brief Adds an array of type T and size @p arraySize to the item set
	 * @param [in] arraySize Number of elements in the array
	 * @tparam T Type of the array items
	 */
	template <typename T> inline void add(std::size_t arraySize)
	{
		_curSize += arraySize * sizeof(T) + alignof(T);
	}

	/**
	 * @brief Adds a memory block of size @p blockSize to the item set
	 * @param [in] blockSize Size of the block in bytes
	 */
	inline void addBlock(std::size_t blockSize)
	{
		_curSize += blockSize;
	}

	/**
	 * @brief Sizes the item set such that a block of given size fits in
	 * @param [in] block Size of the memory block in bytes
	 */
	inline void fitBlock(std::size_t block)
	{
		_curSize = std::max(_curSize, block);
	}

	/**
	 * @brief Commits the current item set
	 */
	inline void commit()
	{
		_maxSize = std::max(_maxSize, _curSize);
		_curSize = 0;
	}

	/**
	 * @brief Computes the size of the buffer in bytes
	 * @details Computes the size of the buffer required to hold all item sets recorded so far.
	 * @return Size of the buffer in bytes
	 */
	inline std::size_t bufferSize() const
	{
		return std::max(_maxSize, _curSize);
	}

private:
	std::size_t _maxSize; //<! Maximum size of all sets of items
	std::size_t _curSize; //<! Size of the current set of items
};

class LinearBufferAllocator;
template <typename T> class ConstBufferedArray;
template <typename T> class ConstBufferedScalar;

/**
 * @brief Handle to an array constructed in a linear buffer
 * @details Calls the destructor of the elements upon destruction.
 * @tparam T Underlying type of the array
 */
template <typename T> class BufferedArray
{
public:
	friend class LinearBufferAllocator;
	friend class ConstBufferedArray<T>;

	BufferedArray() : _ptr(nullptr)
	{
	}
	BufferedArray(const BufferedArray&) = delete;
	BufferedArray(BufferedArray&& rhs) CADET_NOEXCEPT : _ptr(rhs._ptr), _numElements(rhs._numElements)
	{
		rhs._ptr = nullptr;
		rhs._numElements = 0;
	}

	~BufferedArray()
	{
		releaseRawArray(_ptr, _numElements);
	}

	BufferedArray& operator=(const BufferedArray&) = delete;
	BufferedArray& operator=(BufferedArray&& rhs) CADET_NOEXCEPT
	{
		_ptr = rhs._ptr;
		_numElements = rhs._numElements;

		rhs._ptr = nullptr;
		rhs._numElements = 0;
		return *this;
	}

	inline T& operator[](int idx)
	{
		return _ptr[idx];
	}
	inline const T& operator[](int idx) const
	{
		return _ptr[idx];
	}

	inline T& operator*()
	{
		return *_ptr;
	}
	inline const T& operator*() const
	{
		return *_ptr;
	}

	inline T* operator->()
	{
		return _ptr;
	}
	inline T* operator->() const
	{
		return _ptr;
	}

	explicit operator T*() const
	{
		return _ptr;
	}

private:
	BufferedArray(T* ptr, unsigned int numElements) : _ptr(ptr), _numElements(numElements)
	{
	}

	T* _ptr;                   //<! Pointer to first item of array
	unsigned int _numElements; //<! Number of elements in array
};

/**
 * @brief Handle to a constant array constructed in a linear buffer
 * @details Calls the destructor of the elements upon destruction.
 * @tparam T Underlying type of the array
 */
template <typename T> class ConstBufferedArray
{
public:
	friend class LinearBufferAllocator;

	ConstBufferedArray() : _ptr(nullptr)
	{
	}
	ConstBufferedArray(const ConstBufferedArray&) = delete;
	ConstBufferedArray(ConstBufferedArray&& rhs) CADET_NOEXCEPT : _ptr(rhs._ptr), _numElements(rhs._numElements)
	{
		rhs._ptr = nullptr;
		rhs._numElements = 0;
	}

	~ConstBufferedArray()
	{
		releaseRawArray(_ptr, _numElements);
	}

	ConstBufferedArray& operator=(const ConstBufferedArray&) = delete;
	ConstBufferedArray& operator=(ConstBufferedArray&& rhs) CADET_NOEXCEPT
	{
		_ptr = rhs._ptr;
		_numElements = rhs._numElements;

		rhs._ptr = nullptr;
		rhs._numElements = 0;
		return *this;
	}

	ConstBufferedArray(BufferedArray<T>&& rhs) CADET_NOEXCEPT : _ptr(rhs._ptr)
	{
		rhs._ptr = nullptr;
		rhs._numElements = 0;
	}

	ConstBufferedArray operator=(BufferedArray<T>&& rhs) CADET_NOEXCEPT
	{
		_ptr = rhs._ptr;
		_numElements = rhs._numElements;

		rhs._ptr = nullptr;
		rhs._numElements = 0;
		return *this;
	}

	inline const T& operator[](int idx) const
	{
		return _ptr[idx];
	}
	inline const T& operator*() const
	{
		return *_ptr;
	}
	inline T const* operator->() const
	{
		return _ptr;
	}

	explicit operator T const*() const
	{
		return _ptr;
	}

private:
	ConstBufferedArray(T* ptr, unsigned int numElements) : _ptr(ptr), _numElements(numElements)
	{
	}

	T const* _ptr;             //<! Pointer to first item of array
	unsigned int _numElements; //<! Number of elements in array
};

/**
 * @brief Handle to a scalar value constructed in a linear buffer
 * @details Calls the destructor of the scalar value upon destruction.
 * @tparam T Underlying type of the scalar
 */
template <typename T> class BufferedScalar
{
public:
	friend class LinearBufferAllocator;
	friend class ConstBufferedScalar<T>;

	BufferedScalar() : _ptr(nullptr)
	{
	}
	BufferedScalar(const BufferedScalar&) = delete;
	BufferedScalar(BufferedScalar&& rhs) CADET_NOEXCEPT : _ptr(rhs._ptr)
	{
		rhs._ptr = nullptr;
	}

	~BufferedScalar()
	{
		if (_ptr)
			_ptr->~T();
	}

	BufferedScalar& operator=(const BufferedScalar&) = delete;
	BufferedScalar& operator=(BufferedScalar&& rhs) CADET_NOEXCEPT
	{
		_ptr = rhs._ptr;
		rhs._ptr = nullptr;
		return *this;
	}

	inline T& operator*()
	{
		return *_ptr;
	}
	inline const T& operator*() const
	{
		return *_ptr;
	}

	inline T* operator->()
	{
		return _ptr;
	}
	inline T* operator->() const
	{
		return _ptr;
	}

	explicit operator T*() const
	{
		return _ptr;
	}

private:
	BufferedScalar(T* ptr) : _ptr(ptr)
	{
	}

	T* _ptr; //<! Pointer to item
};

/**
 * @brief Handle to a constant scalar value constructed in a linear buffer
 * @details Calls the destructor of the scalar value upon destruction.
 * @tparam T Underlying type of the scalar
 */
template <typename T> class ConstBufferedScalar
{
public:
	friend class LinearBufferAllocator;

	ConstBufferedScalar() : _ptr(nullptr)
	{
	}
	ConstBufferedScalar(const ConstBufferedScalar&) = delete;
	ConstBufferedScalar(ConstBufferedScalar&& rhs) CADET_NOEXCEPT : _ptr(rhs._ptr)
	{
		rhs._ptr = nullptr;
	}

	~ConstBufferedScalar()
	{
		if (_ptr)
			_ptr->~T();
	}

	ConstBufferedScalar& operator=(const ConstBufferedScalar&) = delete;
	ConstBufferedScalar& operator=(ConstBufferedScalar&& rhs) CADET_NOEXCEPT
	{
		_ptr = rhs._ptr;
		rhs._ptr = nullptr;
		return *this;
	}

	ConstBufferedScalar(BufferedScalar<T>&& rhs) CADET_NOEXCEPT : _ptr(rhs._ptr)
	{
		rhs._ptr = nullptr;
	}

	ConstBufferedScalar& operator=(BufferedScalar<T>&& rhs) CADET_NOEXCEPT
	{
		_ptr = rhs._ptr;
		rhs._ptr = nullptr;
		return *this;
	}

	inline const T& operator*() const
	{
		return *_ptr;
	}
	inline T const* operator->() const
	{
		return _ptr;
	}

	explicit operator T const*() const
	{
		return _ptr;
	}

private:
	ConstBufferedScalar(T* ptr) : _ptr(ptr)
	{
	}

	T const* _ptr; //<! Pointer to item
};

/**
 * @brief Linear heap allocator for scalars and arrays
 * @details Uses a given buffer to allocate small objects and arrays.
 */
class LinearBufferAllocator
{
public:
	LinearBufferAllocator() : _mem(nullptr)
	{
	}
	explicit LinearBufferAllocator(void* mem) : _mem(mem)
	{
	}
#ifdef CADET_DEBUG
	explicit LinearBufferAllocator(void* mem, void* end) : _mem(mem), _end(end)
	{
	}
#else
	explicit LinearBufferAllocator(void* mem, void* end) : _mem(mem)
	{
	}
#endif

	~LinearBufferAllocator() CADET_NOEXCEPT
	{
	}

	LinearBufferAllocator(const LinearBufferAllocator&) CADET_NOEXCEPT = default;
#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	LinearBufferAllocator(LinearBufferAllocator&&) CADET_NOEXCEPT = default;
#else
	LinearBufferAllocator(LinearBufferAllocator&&) = default;
#endif

	LinearBufferAllocator& operator=(const LinearBufferAllocator&) CADET_NOEXCEPT = default;
	LinearBufferAllocator& operator=(LinearBufferAllocator&&) CADET_NOEXCEPT = default;

	/**
	 * @brief Sets the buffer the allocator operates on
	 * @param [in] ptr Pointer to buffer
	 */
	void setBuffer(void* ptr) CADET_NOEXCEPT
	{
		_mem = ptr;
#ifdef CADET_DEBUG
		_end = nullptr;
#endif
	}

#ifdef CADET_DEBUG
	void setBuffer(void* ptr, void* end) CADET_NOEXCEPT
	{
		_mem = ptr;
		_end = end;
	}
#endif
	/**
	 * @brief Advances the buffer by the size of an array without actually allocating it
	 * @param [in] numElements Number of array elements
	 * @tparam T Type of the array
	 */
	template <typename T> void advanceArray(std::size_t numElements)
	{
		cadet_assert(sizeof(T) % alignof(T) == 0);

		// Align _mem as required
		std::size_t space = sizeof(T) + alignof(T);
		void* const ptr2 = std::align(alignof(T), sizeof(T), _mem, space);
		cadet_assert(ptr2 != nullptr);

		_mem = static_cast<char*>(_mem) + sizeof(T) * numElements;
	}

	/**
	 * @brief Allocates an array in the buffer
	 * @details The lifetime of the allocated array is not managed and has to be
	 *          terminated by the caller. This can be done by calling releaseRawArray()
	 *          on the return value with the corresponding number of elements.
	 * @param [in] numElements Number of array elements
	 * @tparam T Type of the array
	 * @return Pointer to first element of array
	 */
	template <typename T> T* unmanagedArray(std::size_t numElements)
	{
		cadet_assert(sizeof(T) % alignof(T) == 0);
#ifdef CADET_DEBUG
		cadet_assert(!_end || std::less<void*>()(_mem, _end));
#endif

		// Align _mem as required
		std::size_t space = sizeof(T) + alignof(T);
#ifdef CADET_DEBUG
		void* const ptr2 = std::align(alignof(T), sizeof(T), _mem, space);
		cadet_assert(ptr2 != nullptr);
#else
		std::align(alignof(T), sizeof(T), _mem, space);
#endif

		// Construct elements in buffer and advance pointer
		T* const ptr = rawMemoryAsArray<T>(_mem, numElements);
		_mem = static_cast<char*>(_mem) + sizeof(T) * numElements;

#ifdef CADET_DEBUG
		cadet_assert(!_end || std::less_equal<void*>()(_mem, _end));
#endif

		return ptr;
	}

	/**
	 * @brief Allocates an array in the buffer
	 * @details The lifetime of the array is managed by wrapping it in a BufferedArray.
	 *          When the BufferedArray handle is destroyed, the array is released automatically.
	 * @param [in] numElements Number of array elements
	 * @tparam T Type of the array
	 * @return Handle to the array
	 */
	template <typename T> BufferedArray<T> array(std::size_t numElements)
	{
		return BufferedArray<T>(unmanagedArray<T>(numElements), numElements);
	}

	/**
	 * @brief Advances the buffer by the size of a scalar without actually allocating it
	 * @tparam T Type of the scalar
	 */
	template <typename T> T* advanceScalar()
	{
		// Align _mem as required
		std::size_t space = sizeof(T) + alignof(T);
#ifdef CADET_DEBUG
		void* const ptr2 = std::align(alignof(T), sizeof(T), _mem, space);
		cadet_assert(ptr2 != nullptr);
#else
		std::align(alignof(T), sizeof(T), _mem, space);
#endif

		_mem = static_cast<char*>(_mem) + sizeof(T);
	}

	/**
	 * @brief Allocates a scalar in the buffer
	 * @details The lifetime of the allocated scalar is not managed and has to be
	 *          terminated by the caller. This can be done by calling releaseRawScalar()
	 *          on the return value with the corresponding number of elements.
	 * @tparam T Type of the scalar
	 * @return Pointer to the scalar
	 */
	template <typename T> T* unmanagedScalar()
	{
#ifdef CADET_DEBUG
		cadet_assert(!_end || std::less<void*>()(_mem, _end));
#endif

		// Align _mem as required
		std::size_t space = sizeof(T) + alignof(T);
#ifdef CADET_DEBUG
		void* const ptr2 = std::align(alignof(T), sizeof(T), _mem, space);
		cadet_assert(ptr2 != nullptr);
#else
		std::align(alignof(T), sizeof(T), _mem, space);
#endif

		// Construct element in buffer and advance pointer
		T* const ptr = new (_mem) T;
		_mem = static_cast<char*>(_mem) + sizeof(T);

#ifdef CADET_DEBUG
		cadet_assert(!_end || std::less_equal<void*>()(_mem, _end));
#endif

		return ptr;
	}

	/**
	 * @brief Allocates a scalar in the buffer
	 * @details The lifetime of the scalar is managed by wrapping it in a BufferedScalar.
	 *          When the BufferedScalar handle is destroyed, the scalar is released automatically.
	 * @tparam T Type of the scalar
	 * @return Handle to the scalar
	 */
	template <typename T> BufferedScalar<T> scalar()
	{
		return BufferedScalar<T>(unmanagedScalar<T>());
	}

	/**
	 * @brief Allocates a block of raw memory
	 * @param [in] blockSize Size of the block in bytes
	 * @return Pointer to block of raw memory
	 */
	void* raw(std::size_t blockSize)
	{
#ifdef CADET_DEBUG
		cadet_assert(!_end || std::less<void*>()(_mem, _end));
#endif

		void* const ptr = _mem;
		_mem = static_cast<char*>(_mem) + blockSize;

#ifdef CADET_DEBUG
		cadet_assert(!_end || std::less_equal<void*>()(_mem, _end));
#endif

		return ptr;
	}

	/**
	 * @brief Returns a pointer to the remaining memory
	 * @details Memory is not reserved and can be used in a subsequent call to array(), scalar(), or raw().
	 * @return Pointer to memory
	 */
	void* remainingMemoryAsRaw()
	{
		return _mem;
	}

	/**
	 * @brief Returns a LinearBufferAllocator that manages the remaining memory
	 * @details This function is used to set a checkpoint on the current LinearBufferAllocator.
	 *          While the remaining memory is used by the returned allocator, the position of
	 *          the current one is left unchanged. A typical use-case consists in partitioning
	 *          a memory block and then reusing the remaining space multiple times in a loop
	 *          without changing the previous items.
	 * @return LinearBufferAllocator operating on the remaining memory
	 */
#ifndef CADET_DEBUG
	LinearBufferAllocator manageRemainingMemory() const CADET_NOEXCEPT
	{
		return LinearBufferAllocator(_mem);
	}
#else
	LinearBufferAllocator manageRemainingMemory() const CADET_NOEXCEPT
	{
		return LinearBufferAllocator(_mem, _end);
	}
#endif
private:
	void* _mem; //<! Current position in the buffer
#ifdef CADET_DEBUG
	void* _end; //<! Pointer to end of buffer
#endif
};

/**
 * @brief Linear heap allocator for scalars and arrays
 * @details Allocates a portion of memory from the heap and uses it to allocate small objects and arrays.
 */
class LinearHeapAllocator
{
public:
	LinearHeapAllocator() : _mem(nullptr), _curPos(nullptr), _capacity(0), _free(0)
	{
	}
	LinearHeapAllocator(std::size_t numBytes)
		: _mem(::operator new(numBytes)), _curPos(_mem), _capacity(numBytes), _free(numBytes)
	{
	}
	~LinearHeapAllocator()
	{
		::operator delete(_mem);
	}

	LinearHeapAllocator(const LinearHeapAllocator&) CADET_NOEXCEPT = delete;
#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	LinearHeapAllocator(LinearHeapAllocator&&) CADET_NOEXCEPT = default;
#else
	LinearHeapAllocator(LinearHeapAllocator&&) = default;
#endif

	LinearHeapAllocator& operator=(const LinearHeapAllocator&) CADET_NOEXCEPT = delete;
	LinearHeapAllocator& operator=(LinearHeapAllocator&&) CADET_NOEXCEPT = default;

	/**
	 * @brief Resizes the buffer
	 * @details All allocated objects have to be destroyed before calling resize().
	 * @param [in] numBytes Size of the buffer in bytes
	 */
	void resize(std::size_t numBytes)
	{
		if (numBytes == 0)
		{
			reset();
			return;
		}

		if (numBytes != _capacity)
		{
			::operator delete(_mem);
			_mem = ::operator new(numBytes);
			_capacity = numBytes;
		}

		reset();
	}

	/**
	 * @brief Resets the buffer
	 * @details All allocated objects have to be destroyed before calling reset().
	 */
	void reset() CADET_NOEXCEPT
	{
		_curPos = _mem;
		_free = _capacity;
	}

	/**
	 * @brief Allocates an array in the buffer
	 * @param [in] numElements Number of array elements
	 * @tparam T Type of the array
	 * @return Handle to the array
	 */
	template <typename T> BufferedArray<T> array(std::size_t numElements)
	{
		cadet_assert(sizeof(T) * numElements <= _free);
		cadet_assert(sizeof(T) % alignof(T) == 0);

		// Align _curPos as required
		void* const ptr2 = std::align(alignof(T), sizeof(T) * numElements, _curPos, _free);
		cadet_assert(ptr2 != nullptr);

		// Decrease available free space
		_free -= sizeof(T) * numElements;

		// Construct elements in buffer and advance pointer
		T* const ptr = rawMemoryAsArray<T>(_curPos, numElements);
		_curPos = static_cast<char*>(_curPos) + sizeof(T) * numElements;
		return BufferedArray<T>(ptr, numElements);
	}

	/**
	 * @brief Allocates a scalar in the buffer
	 * @tparam T Type of the scalar
	 * @return Handle to the scalar
	 */
	template <typename T> BufferedScalar<T> scalar()
	{
		cadet_assert(sizeof(T) <= _free);

		// Align _curPos as required
		void* const ptr2 = std::align(alignof(T), sizeof(T), _curPos, _free);
		cadet_assert(ptr2 != nullptr);

		// Decrease available free space
		_free -= sizeof(T);

		// Construct element in buffer and advance pointer
		T* const ptr = new (_curPos) T;
		_curPos = static_cast<char*>(_curPos) + sizeof(T);
		return BufferedScalar<T>(ptr);
	}

	/**
	 * @brief Allocates a block of raw memory
	 * @param [in] blockSize Size of the block in bytes
	 * @return Pointer to block of raw memory
	 */
	void* raw(std::size_t blockSize)
	{
		cadet_assert(_free <= blockSize);
		void* const ptr = _curPos;

		_free -= blockSize;
		_curPos = static_cast<char*>(_curPos) + blockSize;
		return ptr;
	}

	/**
	 * @brief Returns a pointer to the remaining memory
	 * @details Memory is not reserved and can be used in a subsequent call to array(), scalar(), or raw().
	 * @return Pointer to memory
	 */
	void* remainingMemoryAsRaw()
	{
		return _curPos;
	}

	/**
	 * @brief Returns a LinearBufferAllocator that manages the remaining memory
	 * @details This function is used to set a checkpoint on the current LinearBufferAllocator.
	 *          While the remaining memory is used by the returned allocator, the position of
	 *          the current one is left unchanged. A typical use-case consists in partitioning
	 *          a memory block and then reusing the remaining space multiple times in a loop
	 *          without changing the previous items.
	 *
	 *          Note that the remaining memory is not used up. This can cause problems if the
	 *          memory is overwritten or allocated multiple times.
	 * @return LinearBufferAllocator operating on the remaining memory
	 */
#ifndef CADET_DEBUG
	LinearBufferAllocator manageRemainingMemory() const CADET_NOEXCEPT
	{
		return LinearBufferAllocator(_curPos);
	}
#else
	LinearBufferAllocator manageRemainingMemory() const CADET_NOEXCEPT
	{
		return LinearBufferAllocator(_curPos, static_cast<char*>(_mem) + _capacity);
	}
#endif
private:
	void* _mem;            //<! Memory buffer
	void* _curPos;         //<! Current position in the buffer
	std::size_t _capacity; //<! Total capacity of the buffer in bytes
	std::size_t _free;     //<! Remaining free space in the buffer in bytes
};

} // namespace cadet

#endif // LIBCADET_MEMORY_HPP_
