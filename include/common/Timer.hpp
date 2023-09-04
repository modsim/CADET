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
 * Provides an accumulating timer for measuring execution time.
 * If CADET_USE_PLATFORM_TIMER is defined, then a platform-dependent timer implementation
 * is used. Otherwise, an implementation using OpenMP (if available) or the standard C++
 * library is selected (in this order).
 */

#ifndef CADET_TIMER_HPP_
#define CADET_TIMER_HPP_

namespace cadet
{

/**
 * @brief Base class for all timers
 * @details Uses policy design pattern to inject the actual timer implementation. A valid
 *          timer implementation has to provide the following functions:
 *          <pre>
 *              void start()
 *              double stopCore() const
 *          </pre>
 * @tparam timer_t Actual timer implementation
 */
template <class timer_t> class BaseTimer : public timer_t
{
public:
	BaseTimer() : timer_t(), _totalElapsed(0.0)
	{
	}

	/**
	 * @brief Stops the currently running timer and returns the elapsed time
	 * @details Accumulates the total elapsed time over all start() and stop() calls.
	 * @return Elapsed time since the last call to start() in seconds
	 */
	inline double stop()
	{
		const double elapsed = timer_t::stopCore();
		_totalElapsed += elapsed;

		return elapsed;
	}

	/**
	 * @brief Returns the total elapsed time between all start() and stop() calls in seconds
	 * @return Total elapsed time in seconds
	 */
	inline double totalElapsedTime() const
	{
		return _totalElapsed;
	}

	/**
	 * @brief Returns the total elapsed time between all start() and stop() calls in milliseconds
	 * @return Total elapsed time in milliseconds
	 */
	inline double totalElapsedTimeMs() const
	{
		return _totalElapsed * 1000.0;
	}

protected:
	double _totalElapsed;
};

} // namespace cadet

#ifdef CADET_USE_PLATFORM_TIMER

#ifdef _WIN32
  // Windows (x64 and x86)

#define NOMINMAX
#define _WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace cadet
{

/**
 * @brief Windows timer implementation using QueryPerformanceFrequency() and QueryPerformanceCounter()
 */
class WindowsTimer
{
public:
	WindowsTimer()
	{
		// Get frequency of the clock
		QueryPerformanceFrequency(&_frequency);
	}

	inline void start()
	{
		QueryPerformanceCounter(&_startCount);
	}

protected:
	inline double stopCore() const
	{
		LARGE_INTEGER lastCount QueryPerformanceCounter(&lastCount);

		// Convert elapsed time to seconds
		return static_cast<double>(lastCount.QuadPart - _startCount.QuadPart) /
			   static_cast<double>(_frequency.QuadPart);
	}

protected:
	LARGE_INTEGER _frequency;
	LARGE_INTEGER _startCount;
};

typedef BaseTimer<WindowsTimer> Timer;

} // namespace cadet

#elif __unix__ || __linux__
  // Unix and Linux
// Do not forget to link to librt via -lrt flag

#include <time.h>
#include <sys/time.h>

namespace cadet
{

/**
 * @brief Linux timer implementation using clock_gettime() in the monotonic mode
 */
class LinuxTimer
{
public:
	// Constructor
	LinuxTimer()
	{
	}

	inline void start()
	{
		clock_gettime(CLOCK_MONOTONIC, &_startCount);
	}

protected:
	inline double stopCore() const
	{
		timespec lastCount;
		clock_gettime(CLOCK_MONOTONIC, &lastCount);

		// Convert elapsed time to seconds
		if (lastCount.tv_nsec < _startCount.tv_nsec)
		{
			return static_cast<double>(lastCount.tv_sec - _startCount.tv_sec) -
				   static_cast<double>(_startCount.tv_nsec - lastCount.tv_nsec) * 1.0e-9;
		}
		else
		{
			return static_cast<double>(lastCount.tv_sec - _startCount.tv_sec) +
				   static_cast<double>(lastCount.tv_nsec - _startCount.tv_nsec) * 1.0e-9;
		}
	}

protected:
	timespec _startCount;
};

typedef BaseTimer<LinuxTimer> Timer;

} // namespace cadet

#elif __APPLE__
  // Mac OS X

#include <mach/mach.h>
#include <mach/mach_time.h>

namespace cadet
{

/**
 * @brief Mac OS X timer implementation using mach_absolute_time()
 */
class OSXTimer
{
public:
	// Constructor
	OSXTimer()
	{
		mach_timebase_info(&_timebaseInfo);
	}

	inline void start()
	{
		_startCount = mach_absolute_time();
	}

protected:
	inline double stopCore() const
	{
		const uint64_t lastCount = mach_absolute_time();

		// Convert elapsed time to seconds
		// Hopefully this does not overflow
		return static_cast<double>((lastCount - _startCount) * _timebaseInfo.numer / _timebaseInfo.denom) / 1.0e9;
	}

protected:
	mach_timebase_info_data_t _timebaseInfo;
	uint64_t _startCount;
};

typedef BaseTimer<OSXTimer> Timer;

} // namespace cadet

#endif

#else

#ifdef _OPENMP

#include <omp.h>

namespace cadet
{

/**
 * @brief OpenMP timer implementation using omp_get_wtime()
 */
class OpenMPTimer
{
public:
	// Constructor
	OpenMPTimer() : _startTime(0.0)
	{
	}

	inline void start()
	{
		_startTime = omp_get_wtime();
	}

	inline double resolution() const
	{
		return omp_get_wtick();
	}

protected:
	inline double stopCore() const
	{
		return omp_get_wtime() - _startTime;
	}

protected:
	double _startTime;
};

typedef BaseTimer<OpenMPTimer> Timer;

} // namespace cadet

#elif defined(CADET_PARALLELIZE)

#include <tbb/tick_count.h>

namespace cadet
{

/**
 * @brief TBB timer implementation using tbb:tick_count::now()
 */
class TBBTimer
{
public:
	// Constructor
	TBBTimer() : _startTime()
	{
	}

	inline void start()
	{
		_startTime = tbb::tick_count::now();
	}

	inline double resolution() const
	{
		return tbb::tick_count::resolution();
	}

protected:
	inline double stopCore() const
	{
		return (tbb::tick_count::now() - _startTime).seconds();
	}

protected:
	tbb::tick_count _startTime;
};

typedef BaseTimer<TBBTimer> Timer;

} // namespace cadet

#else

#include <chrono>

namespace cadet
{

/**
 * @brief Standard C++ library timer implementation using std::chrono::high_resolution_clock
 */
class StdTimer
{
public:
	// Constructor
	StdTimer() : _startTime()
	{
	}

	inline void start()
	{
		_startTime = std::chrono::high_resolution_clock::now();
	}

protected:
	inline double stopCore() const
	{
		return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - _startTime)
			.count();
	}

protected:
	typename std::chrono::high_resolution_clock::time_point _startTime;
};

typedef BaseTimer<StdTimer> Timer;

} // namespace cadet

#endif

#endif

#endif // CADET_TIMER_HPP_
