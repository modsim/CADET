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
 * Helper functions and macros for parallelization.
 */

#ifndef LIBCADET_PARALLEL_SUPPORT_HPP_
#define LIBCADET_PARALLEL_SUPPORT_HPP_

#include "cadet/cadetCompilerInfo.hpp"
#include "common/CompilerSpecific.hpp"
#include "Memory.hpp"

#ifdef CADET_PARALLELIZE
#define CADET_PARFOR_END )
#define CADET_PARNODE_END )
#define CADET_PAR_CONTINUE return

#include "tbb/task_arena.h"
#include <vector>

namespace cadet
{
namespace util
{

/**
 * @brief Provides a distinct portion of memory for each thread (thread local storage)
 * @details For each thread, a memory buffer of certain size is allocated.
 *          The buffers are aligned to cache lines in order to avoid false sharing.
 *          Note that it is not guaranteed that the exact same thread always obtains
 *          the same buffers as the thread index of a thread may change over time.
 */
class ThreadLocalStorage
{
public:
	ThreadLocalStorage() : _data(0)
	{
	}
	~ThreadLocalStorage() = default;

	ThreadLocalStorage(const ThreadLocalStorage&) = delete;
	ThreadLocalStorage(ThreadLocalStorage&& mv) CADET_NOEXCEPT = default;

	ThreadLocalStorage& operator=(const ThreadLocalStorage&) = delete;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	ThreadLocalStorage& operator=(ThreadLocalStorage&&) CADET_NOEXCEPT = default;
#else
	ThreadLocalStorage& operator=(ThreadLocalStorage&&) = default;
#endif

	/**
	 * @brief Allocates storage for each thread with the given number of bytes
	 * @details The number of threads can expand or shrink and the buffers are
	 *          created or released accordingly.
	 * @param [in] numThreads Number of threads
	 * @param [in] storageSize Number of bytes to store
	 */
	inline void resize(unsigned int numThreads, unsigned int storageSize)
	{
		if (numThreads < _data.size())
		{
			// Remove superfluous buffers at the end
			for (std::size_t i = 0; i < numThreads - _data.size(); ++i)
				_data.pop_back();
		}
		else if (numThreads > _data.size())
		{
			// Add some more buffers
			const unsigned int oldSize = _data.size();
			_data.resize(numThreads);
			for (std::size_t i = oldSize; i < _data.size(); ++i)
				_data[i].resize(storageSize);
		}
	}

	/**
	 * @brief Allocates storage for each thread with the given number of bytes
	 * @details The number of threads can expand or shrink and the buffers are
	 *          created or released accordingly.
	 *
	 *          The number of threads is inferred from TBB.
	 * @param [in] storageSize Number of bytes to store
	 */
	inline void resize(unsigned int storageSize)
	{
		resize(tbb::this_task_arena::max_concurrency(), storageSize);
	}

	/**
	 * @brief Resets the memory blocks for another use
	 */
	inline void reset() CADET_NOEXCEPT
	{
		for (LinearHeapAllocator& lha : _data)
			lha.reset();
	}

	/**
	 * @brief Access memory buffer of the current thread
	 * @return Memory buffer of the current thread
	 */
	inline LinearBufferAllocator get() const
	{
#ifdef CADET_DEBUG
		const int threadIdx = tbb::this_task_arena::current_thread_index();
		cadet_assert(threadIdx != tbb::task_arena::not_initialized);
		cadet_assert(threadIdx >= 0);
		cadet_assert(threadIdx < _data.size());
#endif
		return _data[tbb::this_task_arena::current_thread_index()].manageRemainingMemory();
	}

	/**
	 * @brief Access memory buffer of the given thread index
	 * @param [in] idx Thread index
	 * @return Memory buffer of the given thread index
	 */
	inline LinearBufferAllocator get(unsigned int idx) const
	{
#ifdef CADET_DEBUG
		cadet_assert(idx < _data.size());
#endif
		return _data[idx].manageRemainingMemory();
	}

private:
	std::vector<LinearHeapAllocator> _data;
};

/**
 * @brief Returns the maximum number of threads at this point in the code
 * @details The current maximum number of threads may change from point in code
 *          to point in code. It is governed by several TBB settings, most importantly
 *          the first instantiation of tbb::task_scheduler_init.
 * @return Maximum number of threads
 */
inline unsigned int getMaxThreads()
{
	return tbb::this_task_arena::max_concurrency();
}

} // namespace util
} // namespace cadet

#else
#define CADET_PARFOR_END
#define CADET_PARNODE_END
#define CADET_PAR_CONTINUE continue

namespace cadet
{
namespace util
{

/**
 * @brief Provides a distinct portion of memory for each thread (thread local storage)
 * @details As there is only one thread, this class simply wraps a single memory buffer.
 */
class ThreadLocalStorage
{
public:
	ThreadLocalStorage()
	{
	}
	ThreadLocalStorage(const ThreadLocalStorage&) = delete;
	ThreadLocalStorage(ThreadLocalStorage&& mv) CADET_NOEXCEPT = default;
	~ThreadLocalStorage() = default;

	ThreadLocalStorage& operator=(const ThreadLocalStorage&) = delete;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	ThreadLocalStorage& operator=(ThreadLocalStorage&&) CADET_NOEXCEPT = default;
#else
	ThreadLocalStorage& operator=(ThreadLocalStorage&&) = default;
#endif

	/**
	 * @brief Allocates storage for each thread with the given number of bytes
	 * @param [in] numThreads Number of threads
	 * @param [in] storageSize Number of bytes to store
	 */
	inline void resize(unsigned int numThreads, unsigned int storageSize)
	{
		_memory.resize(storageSize);
	}

	/**
	 * @brief Allocates arrays for each thread with the given number of elements
	 * @param [in] numThreads Number of threads
	 * @param [in] storageSize Number of bytes to store
	 */
	inline void resize(unsigned int storageSize)
	{
		resize(1u, storageSize);
	}

	/**
	 * @brief Resets the memory blocks for another use
	 */
	inline void reset() CADET_NOEXCEPT
	{
		_memory.reset();
	}

	/**
	 * @brief Access memory buffer of the current thread
	 * @return Memory buffer of the current thread
	 */
	inline LinearBufferAllocator get() const CADET_NOEXCEPT
	{
		return _memory.manageRemainingMemory();
	}

	/**
	 * @brief Access memory buffer of the given thread index
	 * @param [in] idx Thread index
	 * @return Memory buffer of the given thread index
	 */
	inline LinearBufferAllocator get(unsigned int idx) const CADET_NOEXCEPT
	{
		return _memory.manageRemainingMemory();
	}

private:
	LinearHeapAllocator _memory; //<! Memory
};

/**
 * @brief Returns the maximum number of threads at this point in the code
 * @details This is just one in the single threaded case.
 * @return Maximum number of threads
 */
inline unsigned int getMaxThreads()
{
	return 1;
}

} // namespace util
} // namespace cadet

#endif

#endif // LIBCADET_PARALLEL_SUPPORT_HPP_
