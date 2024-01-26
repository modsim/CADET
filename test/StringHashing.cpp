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

#include <catch.hpp>

#include "cadet/StringUtil.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

////////////////////////////////////////////////////////////////////////////////////
///////////////////////// BEGIN REFERENCE IMPLEMENTATION ///////////////////////////
////////////////////////////////////////////////////////////////////////////////////

/*
   SipHash reference C implementation
   Copyright (c) 2016 Jean-Philippe Aumasson <jeanphilippe.aumasson@gmail.com>
   To the extent possible under law, the author(s) have dedicated all copyright
   and related and neighboring rights to this software to the public domain
   worldwide. This software is distributed without any warranty.
   You should have received a copy of the CC0 Public Domain Dedication along
   with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 */

// Reference implementation taken from <https://github.com/veorq/SipHash>

#include <stdio.h>
#include <string.h>

/* default: SipHash-2-4 */
#define MAXLEN 64
#define KEYLEN 16
#ifndef DOUBLE
#define HASHLEN 8
#else
#define HASHLEN 16
#endif

#define cROUNDS 2
#define dROUNDS 4

#define ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))

#define U32TO8_LE(p, v)                                                                                                \
	(p)[0] = (uint8_t)((v));                                                                                           \
	(p)[1] = (uint8_t)((v) >> 8);                                                                                      \
	(p)[2] = (uint8_t)((v) >> 16);                                                                                     \
	(p)[3] = (uint8_t)((v) >> 24);

#define U64TO8_LE(p, v)                                                                                                \
	U32TO8_LE((p), (uint32_t)((v)));                                                                                   \
	U32TO8_LE((p) + 4, (uint32_t)((v) >> 32));

#define U8TO64_LE(p)                                                                                                   \
	(((uint64_t)((p)[0])) | ((uint64_t)((p)[1]) << 8) | ((uint64_t)((p)[2]) << 16) | ((uint64_t)((p)[3]) << 24) |      \
	 ((uint64_t)((p)[4]) << 32) | ((uint64_t)((p)[5]) << 40) | ((uint64_t)((p)[6]) << 48) |                            \
	 ((uint64_t)((p)[7]) << 56))

#define SIPROUND                                                                                                       \
	do                                                                                                                 \
	{                                                                                                                  \
		v0 += v1;                                                                                                      \
		v1 = ROTL(v1, 13);                                                                                             \
		v1 ^= v0;                                                                                                      \
		v0 = ROTL(v0, 32);                                                                                             \
		v2 += v3;                                                                                                      \
		v3 = ROTL(v3, 16);                                                                                             \
		v3 ^= v2;                                                                                                      \
		v0 += v3;                                                                                                      \
		v3 = ROTL(v3, 21);                                                                                             \
		v3 ^= v0;                                                                                                      \
		v2 += v1;                                                                                                      \
		v1 = ROTL(v1, 17);                                                                                             \
		v1 ^= v2;                                                                                                      \
		v2 = ROTL(v2, 32);                                                                                             \
	} while (0)

#define TRACE

int siphashRefMain(uint8_t* out, const uint8_t* in, uint64_t inlen, const uint8_t* k)
{
	/* "somepseudorandomlygeneratedbytes" */
	uint64_t v0 = 0x736f6d6570736575ULL;
	uint64_t v1 = 0x646f72616e646f6dULL;
	uint64_t v2 = 0x6c7967656e657261ULL;
	uint64_t v3 = 0x7465646279746573ULL;
	uint64_t b;
	uint64_t k0 = U8TO64_LE(k);
	uint64_t k1 = U8TO64_LE(k + 8);
	uint64_t m;
	int i;
	const uint8_t* end = in + inlen - (inlen % sizeof(uint64_t));
	const int left = inlen & 7;
	b = ((uint64_t)inlen) << 56;
	v3 ^= k1;
	v2 ^= k0;
	v1 ^= k1;
	v0 ^= k0;

#ifdef DOUBLE
	v1 ^= 0xee;
#endif

	for (; in != end; in += 8)
	{
		m = U8TO64_LE(in);
		v3 ^= m;

		TRACE;
		for (i = 0; i < cROUNDS; ++i)
			SIPROUND;

		v0 ^= m;
	}

	switch (left)
	{
	case 7:
		b |= ((uint64_t)in[6]) << 48;
		[[fallthrough]];
	case 6:
		b |= ((uint64_t)in[5]) << 40;
		[[fallthrough]];
	case 5:
		b |= ((uint64_t)in[4]) << 32;
		[[fallthrough]];
	case 4:
		b |= ((uint64_t)in[3]) << 24;
		[[fallthrough]];
	case 3:
		b |= ((uint64_t)in[2]) << 16;
		[[fallthrough]];
	case 2:
		b |= ((uint64_t)in[1]) << 8;
		[[fallthrough]];
	case 1:
		b |= ((uint64_t)in[0]);
		break;
	case 0:
		break;
	}

	v3 ^= b;

	TRACE;
	for (i = 0; i < cROUNDS; ++i)
		SIPROUND;

	v0 ^= b;

#ifndef DOUBLE
	v2 ^= 0xff;
#else
	v2 ^= 0xee;
#endif

	TRACE;
	for (i = 0; i < dROUNDS; ++i)
		SIPROUND;

	b = v0 ^ v1 ^ v2 ^ v3;
	U64TO8_LE(out, b);

#ifdef DOUBLE
	v1 ^= 0xdd;

	TRACE;
	for (i = 0; i < dROUNDS; ++i)
		SIPROUND;

	b = v0 ^ v1 ^ v2 ^ v3;
	U64TO8_LE(out + 8, b);
#endif

	return 0;
}

inline uint64_t sipHashRef(char const* data, unsigned int len)
{
	uint8_t in[MAXLEN], out[HASHLEN], k[KEYLEN];

	for (int i = 0; i < KEYLEN; ++i)
		k[i] = 0;

	siphashRefMain(out, reinterpret_cast<const uint8_t*>(data), len, k);

	return *(reinterpret_cast<uint64_t*>(out));
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////// END REFERENCE IMPLEMENTATION ////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

using cadet::util::SipHash24;
using cadet::util::SipHash24runtime;
using cadet::operator"" _hash;

#if CADET_COMPILETIME_HASH

template <typename Enumeration>
inline typename std::underlying_type<Enumeration>::type as_integer(Enumeration const value)
{
	return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

// Force compile time evaluation
enum class Bla : uint64_t
{
	Hallo = "Hallo"_hash,
	DasIstEinLangerTestAuchMitZahlenDrin12 = "DasIstEinLangerTestAuchMitZahlenDrin12"_hash,
	A = "A"_hash,
	AB = "AB"_hash,
	ABC = "ABC"_hash,
	ABCD = "ABCD"_hash,
	ABCDE = "ABCDE"_hash,
	ABCDEF = "ABCDEF"_hash,
	ABCDEFG = "ABCDEFG"_hash,
	ABCDEFGH = "ABCDEFGH"_hash,
	ABCDEFGHI = "ABCDEFGHI"_hash,
	ABCDEFGHIJ = "ABCDEFGHIJ"_hash,
	ABCDEFGHIJK = "ABCDEFGHIJK"_hash,
	ABCDEFGHIJKL = "ABCDEFGHIJKL"_hash,
	ABCDEFGHIJKLM = "ABCDEFGHIJKLM"_hash,
	ABCDEFGHIJKLMN = "ABCDEFGHIJKLMN"_hash,
	ABCDEFGHIJKLMNO = "ABCDEFGHIJKLMNO"_hash,
	ABCDEFGHIJKLMNOP = "ABCDEFGHIJKLMNOP"_hash,
	ABCDEFGHIJKLMNOPQ = "ABCDEFGHIJKLMNOPQ"_hash,
	ABCDEFGHIJKLMNOPQR = "ABCDEFGHIJKLMNOPQR"_hash,
	ABCDEFGHIJKLMNOPQRS = "ABCDEFGHIJKLMNOPQRS"_hash,
	ABCDEFGHIJKLMNOPQRST = "ABCDEFGHIJKLMNOPQRST"_hash,
	ABCDEFGHIJKLMNOPQRSTU = "ABCDEFGHIJKLMNOPQRSTU"_hash,
	ABCDEFGHIJKLMNOPQRSTUV = "ABCDEFGHIJKLMNOPQRSTUV"_hash,
	ABCDEFGHIJKLMNOPQRSTUVW = "ABCDEFGHIJKLMNOPQRSTUVW"_hash,
	ABCDEFGHIJKLMNOPQRSTUVWX = "ABCDEFGHIJKLMNOPQRSTUVWX"_hash,
	ABCDEFGHIJKLMNOPQRSTUVWXY = "ABCDEFGHIJKLMNOPQRSTUVWXY"_hash,
	ABCDEFGHIJKLMNOPQRSTUVWXYZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"_hash,
};

TEST_CASE("String literal compile time hash", "[Hash],[CompileTime]")
{
	REQUIRE(sipHashRef("Hallo", 5) == as_integer(Bla::Hallo));
	REQUIRE(sipHashRef("DasIstEinLangerTestAuchMitZahlenDrin12", 38) ==
			as_integer(Bla::DasIstEinLangerTestAuchMitZahlenDrin12));
	REQUIRE(sipHashRef("A", 1) == as_integer(Bla::A));
	REQUIRE(sipHashRef("AB", 2) == as_integer(Bla::AB));
	REQUIRE(sipHashRef("ABC", 3) == as_integer(Bla::ABC));
	REQUIRE(sipHashRef("ABCD", 4) == as_integer(Bla::ABCD));
	REQUIRE(sipHashRef("ABCDE", 5) == as_integer(Bla::ABCDE));
	REQUIRE(sipHashRef("ABCDEF", 6) == as_integer(Bla::ABCDEF));
	REQUIRE(sipHashRef("ABCDEFG", 7) == as_integer(Bla::ABCDEFG));
	REQUIRE(sipHashRef("ABCDEFGH", 8) == as_integer(Bla::ABCDEFGH));
	REQUIRE(sipHashRef("ABCDEFGHI", 9) == as_integer(Bla::ABCDEFGHI));
	REQUIRE(sipHashRef("ABCDEFGHIJ", 10) == as_integer(Bla::ABCDEFGHIJ));
	REQUIRE(sipHashRef("ABCDEFGHIJK", 11) == as_integer(Bla::ABCDEFGHIJK));
	REQUIRE(sipHashRef("ABCDEFGHIJKL", 12) == as_integer(Bla::ABCDEFGHIJKL));
	REQUIRE(sipHashRef("ABCDEFGHIJKLM", 13) == as_integer(Bla::ABCDEFGHIJKLM));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMN", 14) == as_integer(Bla::ABCDEFGHIJKLMN));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNO", 15) == as_integer(Bla::ABCDEFGHIJKLMNO));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOP", 16) == as_integer(Bla::ABCDEFGHIJKLMNOP));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQ", 17) == as_integer(Bla::ABCDEFGHIJKLMNOPQ));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQR", 18) == as_integer(Bla::ABCDEFGHIJKLMNOPQR));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRS", 19) == as_integer(Bla::ABCDEFGHIJKLMNOPQRS));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRST", 20) == as_integer(Bla::ABCDEFGHIJKLMNOPQRST));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTU", 21) == as_integer(Bla::ABCDEFGHIJKLMNOPQRSTU));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUV", 22) == as_integer(Bla::ABCDEFGHIJKLMNOPQRSTUV));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVW", 23) == as_integer(Bla::ABCDEFGHIJKLMNOPQRSTUVW));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVWX", 24) == as_integer(Bla::ABCDEFGHIJKLMNOPQRSTUVWX));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVWXY", 25) == as_integer(Bla::ABCDEFGHIJKLMNOPQRSTUVWXY));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 26) == as_integer(Bla::ABCDEFGHIJKLMNOPQRSTUVWXYZ));
}

#endif

#if CADET_COMPILER_CXX_CONSTEXPR

TEST_CASE("Compile time hash", "[Hash],[CompileTime]")
{
	REQUIRE(sipHashRef("Hallo", 5) == SipHash24("Hallo"));
	REQUIRE(sipHashRef("DasIstEinLangerTestAuchMitZahlenDrin12", 38) ==
			SipHash24("DasIstEinLangerTestAuchMitZahlenDrin12"));
	REQUIRE(sipHashRef("A", 1) == SipHash24("A"));
	REQUIRE(sipHashRef("AB", 2) == SipHash24("AB"));
	REQUIRE(sipHashRef("ABC", 3) == SipHash24("ABC"));
	REQUIRE(sipHashRef("ABCD", 4) == SipHash24("ABCD"));
	REQUIRE(sipHashRef("ABCDE", 5) == SipHash24("ABCDE"));
	REQUIRE(sipHashRef("ABCDEF", 6) == SipHash24("ABCDEF"));
	REQUIRE(sipHashRef("ABCDEFG", 7) == SipHash24("ABCDEFG"));
	REQUIRE(sipHashRef("ABCDEFGH", 8) == SipHash24("ABCDEFGH"));
	REQUIRE(sipHashRef("ABCDEFGHI", 9) == SipHash24("ABCDEFGHI"));
	REQUIRE(sipHashRef("ABCDEFGHIJ", 10) == SipHash24("ABCDEFGHIJ"));
	REQUIRE(sipHashRef("ABCDEFGHIJK", 11) == SipHash24("ABCDEFGHIJK"));
	REQUIRE(sipHashRef("ABCDEFGHIJKL", 12) == SipHash24("ABCDEFGHIJKL"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLM", 13) == SipHash24("ABCDEFGHIJKLM"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMN", 14) == SipHash24("ABCDEFGHIJKLMN"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNO", 15) == SipHash24("ABCDEFGHIJKLMNO"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOP", 16) == SipHash24("ABCDEFGHIJKLMNOP"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQ", 17) == SipHash24("ABCDEFGHIJKLMNOPQ"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQR", 18) == SipHash24("ABCDEFGHIJKLMNOPQR"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRS", 19) == SipHash24("ABCDEFGHIJKLMNOPQRS"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRST", 20) == SipHash24("ABCDEFGHIJKLMNOPQRST"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTU", 21) == SipHash24("ABCDEFGHIJKLMNOPQRSTU"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUV", 22) == SipHash24("ABCDEFGHIJKLMNOPQRSTUV"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVW", 23) == SipHash24("ABCDEFGHIJKLMNOPQRSTUVW"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVWX", 24) == SipHash24("ABCDEFGHIJKLMNOPQRSTUVWX"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVWXY", 25) == SipHash24("ABCDEFGHIJKLMNOPQRSTUVWXY"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 26) == SipHash24("ABCDEFGHIJKLMNOPQRSTUVWXYZ"));
}

#endif

TEST_CASE("String hash", "[Hash]")
{
	REQUIRE(sipHashRef("Hallo", 5) == SipHash24runtime("Hallo"));
	REQUIRE(sipHashRef("DasIstEinLangerTestAuchMitZahlenDrin12", 38) ==
			SipHash24runtime("DasIstEinLangerTestAuchMitZahlenDrin12"));
	REQUIRE(sipHashRef("A", 1) == SipHash24runtime("A"));
	REQUIRE(sipHashRef("AB", 2) == SipHash24runtime("AB"));
	REQUIRE(sipHashRef("ABC", 3) == SipHash24runtime("ABC"));
	REQUIRE(sipHashRef("ABCD", 4) == SipHash24runtime("ABCD"));
	REQUIRE(sipHashRef("ABCDE", 5) == SipHash24runtime("ABCDE"));
	REQUIRE(sipHashRef("ABCDEF", 6) == SipHash24runtime("ABCDEF"));
	REQUIRE(sipHashRef("ABCDEFG", 7) == SipHash24runtime("ABCDEFG"));
	REQUIRE(sipHashRef("ABCDEFGH", 8) == SipHash24runtime("ABCDEFGH"));
	REQUIRE(sipHashRef("ABCDEFGHI", 9) == SipHash24runtime("ABCDEFGHI"));
	REQUIRE(sipHashRef("ABCDEFGHIJ", 10) == SipHash24runtime("ABCDEFGHIJ"));
	REQUIRE(sipHashRef("ABCDEFGHIJK", 11) == SipHash24runtime("ABCDEFGHIJK"));
	REQUIRE(sipHashRef("ABCDEFGHIJKL", 12) == SipHash24runtime("ABCDEFGHIJKL"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLM", 13) == SipHash24runtime("ABCDEFGHIJKLM"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMN", 14) == SipHash24runtime("ABCDEFGHIJKLMN"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNO", 15) == SipHash24runtime("ABCDEFGHIJKLMNO"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOP", 16) == SipHash24runtime("ABCDEFGHIJKLMNOP"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQ", 17) == SipHash24runtime("ABCDEFGHIJKLMNOPQ"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQR", 18) == SipHash24runtime("ABCDEFGHIJKLMNOPQR"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRS", 19) == SipHash24runtime("ABCDEFGHIJKLMNOPQRS"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRST", 20) == SipHash24runtime("ABCDEFGHIJKLMNOPQRST"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTU", 21) == SipHash24runtime("ABCDEFGHIJKLMNOPQRSTU"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUV", 22) == SipHash24runtime("ABCDEFGHIJKLMNOPQRSTUV"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVW", 23) == SipHash24runtime("ABCDEFGHIJKLMNOPQRSTUVW"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVWX", 24) == SipHash24runtime("ABCDEFGHIJKLMNOPQRSTUVWX"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVWXY", 25) == SipHash24runtime("ABCDEFGHIJKLMNOPQRSTUVWXY"));
	REQUIRE(sipHashRef("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 26) == SipHash24runtime("ABCDEFGHIJKLMNOPQRSTUVWXYZ"));
}
