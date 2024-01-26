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
 * Provides compile- and runtime string hashing
 */

#ifndef LIBCADET_STRINGUTIL_HPP_
#define LIBCADET_STRINGUTIL_HPP_

#include "cadet/cadetCompilerInfo.hpp"

#include <cstddef>
#include <cstdint>
#include <cctype>
#include <string>

namespace cadet
{

/** @brief Type that holds a hash of a string **/
typedef uint64_t StringHash;

namespace util
{

namespace hash
{

typedef uint64_t sip_word;

CADET_CONSTEXPR static inline sip_word rotl64(const sip_word word, const unsigned int count)
{
	return (word << count) | ((word & 0xffffffffffffffffull) >> (64 - count));
}

#if CADET_COMPILER_CXX_CONSTEXPR

/**
 * @brief Compiletime string with SipHash24 hashing
 */
class ConstString
{
public:
	template <std::size_t N> constexpr ConstString(const char (&s)[N]) : m_Str(s), m_Size(N - 1)
	{
	}
	constexpr ConstString(const char* s, std::size_t len) : m_Str(s), m_Size(len)
	{
	}

	constexpr std::size_t size() const
	{
		return m_Size;
	}

	constexpr unsigned char operator[](std::size_t n) const
	{
		return static_cast<unsigned char>(m_Str[n]);
	}

	/**
	 * @brief Computes the SipHash value of the compiletime string (without NULL terminator)
	 * @param [in] key0 Arbitrary key, defaults to 0
	 * @param [in] key1 Arbitrary key, defaults to 0
	 * @return SipHash of the string
	 */
	constexpr inline sip_word hash(const sip_word key0 = 0, const sip_word key1 = 0) const
	{
		return lastPhase(mainLoop(makeState(key0, key1), *this, 0, m_Size - (m_Size & 7)),
						 tailWord(*this, m_Size - (m_Size & 7), m_Size));
	}

private:
	/**
	 * @brief 256 bit state
	 */
	struct state4
	{
		constexpr state4() : state0(0), state1(0), state2(0), state3(0)
		{
		}
		constexpr state4(const sip_word a, const sip_word b, const sip_word c, const sip_word d)
			: state0(a), state1(b), state2(c), state3(d)
		{
		}
		sip_word state0;
		sip_word state1;
		sip_word state2;
		sip_word state3;
	};

	constexpr inline sip_word getSipWord(const unsigned int offset, const unsigned int idx) const
	{
		return static_cast<sip_word>((*this)[offset + idx]);
	}

	constexpr static inline sip_word read64(const ConstString& cs, const unsigned int offset)
	{
		return ((cs.getSipWord(offset, 0)) | (cs.getSipWord(offset, 1) << 8) | (cs.getSipWord(offset, 2) << 16) |
				(cs.getSipWord(offset, 3) << 24) | (cs.getSipWord(offset, 4) << 32) | (cs.getSipWord(offset, 5) << 40) |
				(cs.getSipWord(offset, 6) << 48) | (cs.getSipWord(offset, 7) << 56));
	}

	constexpr static inline state4 mixFirst(const state4& s)
	{
		return {s.state0 + s.state1, rotl64(s.state1, 13), s.state2 + s.state3, rotl64(s.state3, 16)};
	}

	constexpr static inline state4 mixSecond(const state4& s)
	{
		return {rotl64(s.state0, 32), s.state1 ^ s.state0, s.state2, s.state3 ^ s.state2};
	}

	constexpr static inline state4 mixThird(const state4& s)
	{
		return {s.state0 + s.state3, rotl64(s.state1, 17), s.state2 + s.state1, rotl64(s.state3, 21)};
	}

	constexpr static inline state4 mixFourth(const state4& s)
	{
		return {s.state0, s.state1 ^ s.state2, rotl64(s.state2, 32), s.state3 ^ s.state0};
	}

	constexpr static inline state4 mix(const state4& s)
	{
		return mixFourth(mixThird(mixSecond(mixFirst(s))));
	}

	constexpr static inline state4 mixLoop(const state4& s, const unsigned int rounds)
	{
		return (rounds > 0) ? mixLoop(mix(s), rounds - 1) : s;
	}

	constexpr static inline sip_word tailWordInternal(const ConstString& cs, const sip_word tailWord,
													  const std::size_t offset, const unsigned int tailSize)
	{
		return (tailSize == 0)
				   ? tailWord
				   : tailWordInternal(
						 cs, tailWord | (static_cast<sip_word>(cs[offset + tailSize - 1]) << ((tailSize - 1) * 8)),
						 offset, tailSize - 1);
	}

	constexpr static inline sip_word tailWord(const ConstString& cs, const std::size_t offset, const std::size_t insize)
	{
		return tailWordInternal(cs, static_cast<sip_word>(insize) << 56, offset, insize & 7);
	}

	constexpr static inline state4 mixin(const state4& s, const sip_word tailWord)
	{
		// SipHash24, so take 2 rounds here (mixing of tail word)
		return mixLoop({s.state0, s.state1, s.state2, s.state3 ^ tailWord}, 2);
	}

	constexpr static inline state4 finalMix(const state4& s, const sip_word tailWord)
	{
		// SipHash24, so take 4 rounds here (finalize)
		return mixLoop({s.state0 ^ tailWord, s.state1, s.state2 ^ 0xff, s.state3}, 4);
	}

	constexpr static inline sip_word finalizeHash(const state4& s)
	{
		return (s.state0 ^ s.state1 ^ s.state2 ^ s.state3) & 0xffffffffffffffffull;
	}

	constexpr static inline state4 bottomMainLoop(const state4& s, const sip_word curWord)
	{
		return {s.state0 ^ curWord, s.state1, s.state2, s.state3};
	}

	constexpr static inline state4 execMainLoop(const state4& s, const sip_word curWord)
	{
		// SipHash24, so take 2 rounds here (main loop)
		return bottomMainLoop(mixLoop({s.state0, s.state1, s.state2, s.state3 ^ curWord}, 2), curWord);
	}

	constexpr static inline state4 mainLoop(const state4& s, const ConstString& cs, const unsigned int offset,
											const unsigned int end)
	{
		return (offset < end) ? mainLoop(execMainLoop(s, read64(cs, offset)), cs, offset + 8, end) : s;
	}

	constexpr static inline state4 makeState(const sip_word key0, const sip_word key1)
	{
		return {key0 ^ 0x736f6d6570736575ull, key1 ^ 0x646f72616e646f6dull, key0 ^ 0x6c7967656e657261ull,
				key1 ^ 0x7465646279746573ull};
	}

	constexpr static inline sip_word lastPhase(const state4& s, const sip_word tailWord)
	{
		return finalizeHash(finalMix(mixin(s, tailWord), tailWord));
	}

	const char* m_Str;
	std::size_t m_Size;
};

#endif

/**
 * @brief SipHash reading
 * @details Implementation taken from
 * https://chiselapp.com/user/Justin_be_my_guide/repository/siple/artifact/f7d1b255a09c1d67
 */
static inline sip_word read64(unsigned char const* const p)
{
	return (((sip_word)p[0]) | ((sip_word)p[1] << 8) | ((sip_word)p[2] << 16) | ((sip_word)p[3] << 24) |
			((sip_word)p[4] << 32) | ((sip_word)p[5] << 40) | ((sip_word)p[6] << 48) | ((sip_word)p[7] << 56));
}

/**
 * @brief SipHash mixing
 * @details Implementation taken from
 * https://chiselapp.com/user/Justin_be_my_guide/repository/siple/artifact/f7d1b255a09c1d67
 */
static inline void mix(sip_word& state0, sip_word& state1, sip_word& state2, sip_word& state3)
{
	state0 += state1;
	state2 += state3;
	state1 = rotl64(state1, 13);
	state3 = rotl64(state3, 16);
	state1 ^= state0;
	state3 ^= state2;
	state0 = rotl64(state0, 32);
	state2 += state1;
	state0 += state3;
	state1 = rotl64(state1, 17);
	state3 = rotl64(state3, 21);
	state1 ^= state2;
	state3 ^= state0;
	state2 = rotl64(state2, 32);
}

/**
 * @brief Computes the SipHash value of the given data at runtime
 * @details Implementation taken from
 * https://chiselapp.com/user/Justin_be_my_guide/repository/siple/artifact/f7d1b255a09c1d67
 *
 * @param [in] in Input data
 * @param [in] insize Size of the input data (in byte)
 * @param [in] key0 Arbitrary key (defaults to 0)
 * @param [in] key1 Arbitrary key (defaults to 0)
 * @tparam int rounds Number of mixing rounds
 * @tparam int finalisation_rounds Number of final mixing rounds
 * @return SipHash value of the given data
 */
template <const int rounds, const int finalisation_rounds>
static inline sip_word siphash(const void* in, const std::size_t insize, const sip_word key0 = 0,
							   const sip_word key1 = 0)
{
	sip_word state0 = key0 ^ 0x736f6d6570736575ull;
	sip_word state1 = key1 ^ 0x646f72616e646f6dull;
	sip_word state2 = key0 ^ 0x6c7967656e657261ull;
	sip_word state3 = key1 ^ 0x7465646279746573ull;

	const unsigned char* char_in = (const unsigned char*)in;
	const int tail_size = insize & 7;
	const unsigned char* main_loop_end = char_in + insize - tail_size;

	// The main loop
	for (; char_in != main_loop_end; char_in += 8)
	{
		sip_word current_word = read64(char_in);

		state3 ^= current_word;
		for (int i = 0; i < rounds; ++i)
			mix(state0, state1, state2, state3);
		state0 ^= current_word;
	}

	// We're left with 0..7 bytes.
	sip_word tail_word = ((sip_word)insize) << 56;

	switch (tail_size)
	{
	case 7:
		tail_word |= (sip_word)char_in[6] << 48;
		[[fallthrough]];
	case 6:
		tail_word |= (sip_word)char_in[5] << 40;
		[[fallthrough]];
	case 5:
		tail_word |= (sip_word)char_in[4] << 32;
		[[fallthrough]];
	case 4:
		tail_word |= (sip_word)char_in[3] << 24;
		[[fallthrough]];
	case 3:
		tail_word |= (sip_word)char_in[2] << 16;
		[[fallthrough]];
	case 2:
		tail_word |= (sip_word)char_in[1] << 8;
		[[fallthrough]];
	case 1:
		tail_word |= (sip_word)char_in[0];
		[[fallthrough]];
	case 0:
		break;
	}

	// Mix-in the tail block.
	state3 ^= tail_word;
	for (int i = 0; i < rounds; ++i)
		mix(state0, state1, state2, state3);
	state0 ^= tail_word;

	// The final mix
	state2 ^= 0xff;
	for (int i = 0; i < finalisation_rounds; ++i)
		mix(state0, state1, state2, state3);

	return (state0 ^ state1 ^ state2 ^ state3) & 0xffffffffffffffffull;
}

} // namespace hash

/**
 * @brief Computes the SipHash24 value of the given string at runtime
 * @details Computes a 64 bit hash value of the string at runtime
 *
 * @param [in] str String
 * @param [in] key0 Arbitrary key (defaults to 0)
 * @param [in] key1 Arbitrary key (defaults to 0)
 * @return SipHash24 value of the given string
 */
inline StringHash SipHash24runtime(const std::string& str, const StringHash key0 = 0, const StringHash key1 = 0)
{
	return util::hash::siphash<2, 4>(str.c_str(), str.size(), key0, key1);
}

/**
 * @brief Computes the SipHash24 value of the given string at runtime
 * @details Computes a 64 bit hash value of the string at runtime
 *
 * @param [in] str String
 * @param [in] key0 Arbitrary key (defaults to 0)
 * @param [in] key1 Arbitrary key (defaults to 0)
 * @return SipHash24 value of the given string
 */
inline StringHash SipHash24runtime(char const* const str, const StringHash key0 = 0, const StringHash key1 = 0)
{
	unsigned int len = 0;
	for (; str[len] != 0; ++len)
		;
	return util::hash::siphash<2, 4>(str, len, key0, key1);
}

#if CADET_COMPILER_CXX_CONSTEXPR

/**
 * @brief Computes the SipHash24 value of the given string
 * @details Computes a 64 bit hash value of the string at compiletime
 *
 * @param [in] cs Compiletime string
 * @param [in] key0 Arbitrary key (defaults to 0)
 * @param [in] key1 Arbitrary key (defaults to 0)
 * @return SipHash24 value of the given compiletime string
 */
constexpr inline StringHash SipHash24(const util::hash::ConstString& cs, const StringHash key0 = 0,
									  const StringHash key1 = 0)
{
	return cs.hash(key0, key1);
}

#else

/**
 * @brief Computes the SipHash24 value of the given string at runtime
 * @details Computes a 64 bit hash value of the string at runtime
 *
 * @param [in] str String
 * @param [in] key0 Arbitrary key (defaults to 0)
 * @param [in] key1 Arbitrary key (defaults to 0)
 * @return SipHash24 value of the given string
 */
inline StringHash SipHash24(const std::string& str, const StringHash key0 = 0, const StringHash key1 = 0)
{
	return SipHash24runtime(str, key0, key1);
}

#endif

/**
 * @brief Checks whether two strings are equal regardless of upper and lower case
 * @param [in] a First string
 * @param [in] b Second string
 * @return @c true if the strings are equal, otherwise @c false
 */
inline bool caseInsensitiveEquals(const std::string& a, const std::string& b)
{
	const unsigned int sz = a.size();
	if (b.size() != sz)
		return false;
	for (unsigned int i = 0; i < sz; ++i)
	{
		if (tolower(a[i]) != tolower(b[i]))
			return false;
	}
	return true;
}

} // namespace util

#if CADET_COMPILETIME_HASH

/**
 * @brief Computes the SipHash24 value of the given string
 * @details Computes a 64 bit hash value of the string at compiletime using SipHash24 with both keys 0
 *
 * @param [in] str Compiletime string
 * @param [in] len Length of the compiletime string
 * @return SipHash24 value of the given compiletime string
 */
constexpr StringHash operator"" _hash(const char* str, std::size_t len)
{
	return util::hash::ConstString(str, len).hash();
}

#endif

} // namespace cadet

#endif // LIBCADET_STRINGUTIL_HPP_
