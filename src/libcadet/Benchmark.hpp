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
 * Provides benchmark functionality.
 */

#ifndef LIBCADET_BENCHMARK_HPP_
#define LIBCADET_BENCHMARK_HPP_

#ifdef CADET_BENCHMARK_MODE

#include "common/Timer.hpp"

#define BENCH_TIMER(name) mutable ::cadet::Timer name;
#define BENCH_START(name) name.start()
#define BENCH_STOP(name) name.stop()

/**
 * @brief Starts and stops a given timer on construction and desctruction, respectively
 */
class BenchmarkScope
{
public:
	BenchmarkScope(::cadet::Timer& timer) : _timer(timer)
	{
		_timer.start();
	}
	~BenchmarkScope()
	{
		_timer.stop();
	}

private:
	::cadet::Timer& _timer;
};

#define BENCH_SCOPE(name) BenchmarkScope scope##name(name)

#else

#define BENCH_TIMER(name)
#define BENCH_START(name)
#define BENCH_STOP(name)
#define BENCH_SCOPE(name)

#endif

#endif // LIBCADET_BENCHMARK_HPP_
