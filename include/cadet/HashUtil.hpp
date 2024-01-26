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
 * Provides helper functions for hashing.
 */

#ifndef LIBCADET_HASHUTIL_HPP_
#define LIBCADET_HASHUTIL_HPP_

#include "cadet/cadetCompilerInfo.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>

namespace cadet
{

/**
 * @brief Combines a hash value into a cumulative hash value
 * @details Based on HashLen16() function taken from CityHash (Copyright by Google Inc.,
 *          see https://github.com/google/cityhash) protected by the MIT license.
 *          This is taken from http://stackoverflow.com/a/8980550
 * @todo Check if this works on 32 bit systems
 *
 * @param [in,out] seed Cumulative hash value to which the given element is added
 * @param [in] hash Hash value added to the cumulative hash
 */
inline void hash_combine(uint64_t& seed, uint64_t hash) CADET_NOEXCEPT
{
	const uint64_t kMul = 0x9ddfea08eb382d69ULL;
	uint64_t a = (hash ^ seed) * kMul;
	a ^= (a >> 47);
	uint64_t b = (seed ^ a) * kMul;
	b ^= (b >> 47);
	seed = b * kMul;
}

/**
 * @brief Combines an element into a cumulative hash value
 * @details Based on HashLen16() function taken from CityHash (Copyright by Google Inc.,
 *          see https://github.com/google/cityhash) protected by the MIT license.
 *          The element to be added is hashed using std::hash template.
 *
 * @param [in,out] seed Cumulative hash value to which the given element is added
 * @param [in] v Value added to the cumulative hash
 * @tparam T Type of the element to be added to the hash
 */
template <class T> inline void hash_combine(uint64_t& seed, const T& v) CADET_NOEXCEPT
{
	std::hash<T> hasher;
	hash_combine(seed, hasher(v));
}

#ifdef CADET_COMPILER_CXX_VARIADIC_TEMPLATES

// Hides implementation of the pack_ints template function
namespace impl
{
template <typename T>
CADET_CONSTEXPR inline uint64_t pack_ints_impl(uint64_t val, std::size_t cumSize, T v) CADET_NOEXCEPT
{
	// End of the recursive computation
	return val | (static_cast<uint64_t>(v) << cumSize);
}

template <typename T, typename... Args>
CADET_CONSTEXPR inline uint64_t pack_ints_impl(uint64_t val, std::size_t cumSize, T first, Args... args) CADET_NOEXCEPT
{
	// Take current item, shift it to the corresponding position, and combine it with the result up to this point
	// Increment the position by the size of the item
	// Invoke the process once more for the next item
	return pack_ints_impl(val | (static_cast<uint64_t>(first) << cumSize), cumSize + std::numeric_limits<T>::digits,
						  args...);
}
} // namespace impl

/**
 * @brief Packs given integers in a bigger integer of type uint64_t
 * @details Uses bit shifting to pack some small integers into a bigger one.
 *          The user has to make sure that the small integers fit (i.e., they
 *          do not exceed the size of the destination integer).
 *          The first item occupies the lowest bits and the last item is
 *          shifted to the higher bits (according to the size of all items).
 * @todo Determine return type via template meta programming
 *
 * @param args Integers to pack
 * @return Packed integer
 */
template <typename... Args> CADET_CONSTEXPR inline uint64_t pack_ints(Args... args) CADET_NOEXCEPT
{
	return impl::pack_ints_impl(0u, 0, args...);
}

#else

/**
 * @brief Packs given integers in a bigger integer of type uint64_t
 * @details Uses bit shifting to pack some small integers into a bigger one.
 *          The user has to make sure that the small integers fit (i.e., they
 *          do not exceed the size of the destination integer).
 *          The first item occupies the lowest bits and the last item is
 *          shifted to the higher bits (according to the size of all items).
 * @todo Determine return type via template meta programming
 *
 * @param a1 Integers to pack
 * @return Packed integer
 */
template <typename A1> CADET_CONSTEXPR inline uint64_t pack_ints(A1 a1) CADET_NOEXCEPT
{
	return static_cast<uint64_t>(a1);
}

/**
 * @brief Packs given integers in a bigger integer of type uint64_t
 * @details Uses bit shifting to pack some small integers into a bigger one.
 *          The user has to make sure that the small integers fit (i.e., they
 *          do not exceed the size of the destination integer).
 *          The first item occupies the lowest bits and the last item is
 *          shifted to the higher bits (according to the size of all items).
 * @todo Determine return type via template meta programming
 *
 * @param a1 Integers to pack
 * @param a2 Integers to pack
 * @return Packed integer
 */
template <typename A1, typename A2> CADET_CONSTEXPR inline uint64_t pack_ints(A1 a1, A2 a2) CADET_NOEXCEPT
{
	return (static_cast<uint64_t>(a1) << 0) | (static_cast<uint64_t>(a2) << std::numeric_limits<A1>::digits);
}

/**
 * @brief Packs given integers in a bigger integer of type uint64_t
 * @details Uses bit shifting to pack some small integers into a bigger one.
 *          The user has to make sure that the small integers fit (i.e., they
 *          do not exceed the size of the destination integer).
 *          The first item occupies the lowest bits and the last item is
 *          shifted to the higher bits (according to the size of all items).
 * @todo Determine return type via template meta programming
 *
 * @param a1 Integers to pack
 * @param a2 Integers to pack
 * @param a3 Integers to pack
 * @return Packed integer
 */
template <typename A1, typename A2, typename A3>
CADET_CONSTEXPR inline uint64_t pack_ints(A1 a1, A2 a2, A3 a3) CADET_NOEXCEPT
{
	return (static_cast<uint64_t>(a1) << 0) | (static_cast<uint64_t>(a2) << std::numeric_limits<A1>::digits) |
		   (static_cast<uint64_t>(a3) << (std::numeric_limits<A1>::digits + std::numeric_limits<A2>::digits));
}

/**
 * @brief Packs given integers in a bigger integer of type uint64_t
 * @details Uses bit shifting to pack some small integers into a bigger one.
 *          The user has to make sure that the small integers fit (i.e., they
 *          do not exceed the size of the destination integer).
 *          The first item occupies the lowest bits and the last item is
 *          shifted to the higher bits (according to the size of all items).
 * @todo Determine return type via template meta programming
 *
 * @param a1 Integers to pack
 * @param a2 Integers to pack
 * @param a3 Integers to pack
 * @param a4 Integers to pack
 * @return Packed integer
 */
template <typename A1, typename A2, typename A3, typename A4>
CADET_CONSTEXPR inline uint64_t pack_ints(A1 a1, A2 a2, A3 a3, A4 a4) CADET_NOEXCEPT
{
	return (static_cast<uint64_t>(a1) << 0) | (static_cast<uint64_t>(a2) << std::numeric_limits<A1>::digits) |
		   (static_cast<uint64_t>(a3) << (std::numeric_limits<A1>::digits + std::numeric_limits<A2>::digits)) |
		   (static_cast<uint64_t>(a4) << (std::numeric_limits<A1>::digits + std::numeric_limits<A2>::digits +
										  std::numeric_limits<A3>::digits));
}

/**
 * @brief Packs given integers in a bigger integer of type uint64_t
 * @details Uses bit shifting to pack some small integers into a bigger one.
 *          The user has to make sure that the small integers fit (i.e., they
 *          do not exceed the size of the destination integer).
 *          The first item occupies the lowest bits and the last item is
 *          shifted to the higher bits (according to the size of all items).
 * @todo Determine return type via template meta programming
 *
 * @param a1 Integers to pack
 * @param a2 Integers to pack
 * @param a3 Integers to pack
 * @param a4 Integers to pack
 * @param a5 Integers to pack
 * @return Packed integer
 */
template <typename A1, typename A2, typename A3, typename A4, typename A5>
CADET_CONSTEXPR inline uint64_t pack_ints(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5) CADET_NOEXCEPT
{
	return (static_cast<uint64_t>(a1) << 0) | (static_cast<uint64_t>(a2) << std::numeric_limits<A1>::digits) |
		   (static_cast<uint64_t>(a3) << (std::numeric_limits<A1>::digits + std::numeric_limits<A2>::digits)) |
		   (static_cast<uint64_t>(a4) << (std::numeric_limits<A1>::digits + std::numeric_limits<A2>::digits +
										  std::numeric_limits<A3>::digits)) |
		   (static_cast<uint64_t>(a5) << (std::numeric_limits<A1>::digits + std::numeric_limits<A2>::digits +
										  std::numeric_limits<A3>::digits + std::numeric_limits<A4>::digits));
}

#endif

} // namespace cadet

#endif // LIBCADET_HASHUTIL_HPP_
