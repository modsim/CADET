// =============================================================================
//  SFAD - Simple Forward Automatic Differentiation
//
//  Copyright © 2015-2019: Samuel Leweke¹
//
//    ¹ Forschungszentrum Juelich GmbH, IBG-1, Juelich, Germany.
//
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#ifndef _SFAD_COMMON_HPP_
#define _SFAD_COMMON_HPP_

#ifndef SFAD_DEFAULT_DIR
#define SFAD_DEFAULT_DIR 80
#endif

#ifndef SFAD_GLOBAL_GRAD_SIZE
#define SFAD_GLOBAL_GRAD_SIZE std::size_t sfad::detail::globalGradSize = SFAD_DEFAULT_DIR;
#endif

// Improve branch prediction by marking likely and unlikely execution paths
// Example:
//     if (some_unlikely_condition) { ... }
// transforms to
//     if (sfad_unlikely(some_unlikely_condition)) { ... }
#ifdef __GNUC__
#define sfad_likely(x) __builtin_expect(!!(x), 1)
#define sfad_unlikely(x) __builtin_expect(!!(x), 0)
#endif

#ifndef sfad_likely
#define sfad_likely(x) (x)
#define sfad_unlikely(x) (x)
#endif

#if defined(__clang__) && defined(__apple_build_version__)
// Apple Clang
#if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_noexcept)
#define SFAD_COMPILER_CXX_NOEXCEPT 1
#else
#define SFAD_COMPILER_CXX_NOEXCEPT 0
#endif

#elif defined(__clang__)
// Clang
#if ((__clang_major__ * 100) + __clang_minor__) >= 304 && __has_feature(cxx_noexcept)
#define SFAD_COMPILER_CXX_NOEXCEPT 1
#else
#define SFAD_COMPILER_CXX_NOEXCEPT 0
#endif

#elif defined(__INTEL_COMPILER) || defined(__ICC)
// Intel icpc
#if (__INTEL_COMPILER >= 1400)
#define SFAD_COMPILER_CXX_NOEXCEPT 1
#else
#define SFAD_COMPILER_CXX_NOEXCEPT 0
#endif

#elif defined(__GNUC__)
// GNU GCC
#if (__GNUC__ * 100 + __GNUC_MINOR__) >= 406 &&                                                                        \
	(__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#define SFAD_COMPILER_CXX_NOEXCEPT 1
#else
#define SFAD_COMPILER_CXX_NOEXCEPT 0
#endif

#elif defined(_MSC_VER)
// MS Visual C++
#if _MSC_VER >= 1900
#define SFAD_COMPILER_CXX_NOEXCEPT 1
#else
#define SFAD_COMPILER_CXX_NOEXCEPT 0
#endif

#endif

#if SFAD_COMPILER_CXX_NOEXCEPT
#define SFAD_NOEXCEPT noexcept
#define SFAD_NOEXCEPT_EXPR(X) noexcept(X)
#else
#define SFAD_NOEXCEPT
#define SFAD_NOEXCEPT_EXPR(X)
#endif

#include <algorithm>

namespace sfad
{
namespace detail
{
extern std::size_t globalGradSize;
}

inline void setGradientSize(const std::size_t n) SFAD_NOEXCEPT
{
	detail::globalGradSize = n;
}

inline std::size_t getGradientSize() SFAD_NOEXCEPT
{
	return detail::globalGradSize;
}
} // namespace sfad

#endif
