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

// CADET_STRONG_INLINE uses __forceinline on MSVC and Intel ICPC, but doens't use always_inline of GCC.
#if (defined _MSC_VER) || (defined __INTEL_COMPILER)
#define CADET_STRONG_INLINE __forceinline
#else
#define CADET_STRONG_INLINE inline
#endif

// CADET_ALWAYS_INLINE makes the function inline by force, use with care
#ifdef __GNUC__
#define CADET_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define CADET_ALWAYS_INLINE CADET_STRONG_INLINE
#endif

// CADET_DONT_INLINE explicitly prevents inlining
#if (defined __GNUC__)
#define CADET_DONT_INLINE __attribute__((noinline))
#elif (defined _MSC_VER)
#define CADET_DONT_INLINE __declspec(noinline)
#else
#define CADET_DONT_INLINE
#endif

// Assume release build as default
#if defined(NDEBUG) || !defined(DEBUG)
#ifndef CADET_NO_DEBUG
#define CADET_NO_DEBUG
#endif
#ifdef CADET_DEBUG
#undef CADET_DEBUG
#endif
#endif
#if defined(DEBUG) || defined(CADET_DEBUG)
#define CADET_DEBUG
#undef CADET_NO_DEBUG
#include <cassert>
#endif

#ifdef CADET_NO_DEBUG
// Disable assert in release
#define cadet_assert_impl(x)
#else
// Use standard assert
#define cadet_assert_impl(x) assert(x)
#endif

// Allow user supplied cadet_assert
#ifndef cadet_assert
#define cadet_assert(x) cadet_assert_impl(x)
#endif

#ifdef CADET_NO_DEBUG
#define CADET_ONLY_USED_FOR_DEBUG(x) (void)x
#else
#define CADET_ONLY_USED_FOR_DEBUG(x)
#endif

// Improve branch prediction by marking likely and unlikely execution paths
// Example:
//     if (some_unlikely_condition) { ... }
// transforms to
//     if (cadet_unlikely(some_unlikely_condition)) { ... }
#ifdef __GNUC__
#define cadet_likely(x) __builtin_expect(!!(x), 1)
#define cadet_unlikely(x) __builtin_expect(!!(x), 0)
#endif

#ifndef cadet_likely
#define cadet_likely(x) (x)
#define cadet_unlikely(x) (x)
#endif
