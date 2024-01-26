// http://sizeofvoid.blogspot.de/2012/07/compile-time-hashes.html

#include <cstdint>
#include <stddef.h>

struct uint128_t
{
	constexpr uint128_t() : h1(0), h2(0)
	{
	}
	constexpr uint128_t(uint64_t _h1, uint64_t _h2) : h1(_h1), h2(_h2)
	{
	}
	constexpr bool operator==(const uint128_t& rhs) const
	{
		return h1 == rhs.h1 && h2 == rhs.h2;
	}
	constexpr bool operator!=(const uint128_t& rhs) const
	{
		return !(*this == rhs);
	}
	uint64_t h1;
	uint64_t h2;
};

class ConstString
{
public:
	template <size_t N> constexpr ConstString(const char (&s)[N]) : m_Str(s), m_Size(N - 1)
	{
	}
	constexpr ConstString(const char* s, size_t len) : m_Str(s), m_Size(len)
	{
	}

	constexpr size_t size() const
	{
		return m_Size;
	}

	constexpr uint8_t operator[](size_t n) const
	{
		return (uint8_t)m_Str[n];
	}

	constexpr uint8_t getU8(size_t n) const
	{
		return (uint8_t)m_Str[n];
	}

	constexpr uint64_t getU64(size_t n) const
	{
		return uint64_t((uint8_t)m_Str[n * 8 + 0]) << 0 | uint64_t((uint8_t)m_Str[n * 8 + 1]) << 8 |
			   uint64_t((uint8_t)m_Str[n * 8 + 2]) << 16 | uint64_t((uint8_t)m_Str[n * 8 + 3]) << 24 |
			   uint64_t((uint8_t)m_Str[n * 8 + 4]) << 32 | uint64_t((uint8_t)m_Str[n * 8 + 5]) << 40 |
			   uint64_t((uint8_t)m_Str[n * 8 + 6]) << 48 | uint64_t((uint8_t)m_Str[n * 8 + 7]) << 56;
	}

	constexpr inline uint128_t hash128(const uint64_t seed = 0x1234567) const
	{
		return _calcfinal(size(), _calcrest(*this, (size() / 16) * 16, size() & 15,
											_calcblocks(*this, size() / 16, 0, uint128_t(seed, seed))));
	}

	constexpr inline uint64_t hash(uint64_t seed = 0x1234567) const
	{
		return hash128(seed).h1;
	}

private:
	// The code here is a bit messy, but is essentially a functional representation of the MurmurHash3_x64_128
	// implementation rebuilt from the ground up.

	constexpr inline static uint64_t _c1()
	{
		return 0x87c37b91114253d5LLU;
	}
	constexpr inline static uint64_t _c2()
	{
		return 0x4cf5ad432745937fLLU;
	}

	constexpr inline static uint64_t rotl64c(uint64_t x, int8_t r)
	{
		return (x << r) | (x >> (64 - r));
	}

	constexpr inline static uint64_t _downshift_and_xor(uint64_t k)
	{
		return k ^ (k >> 33);
	}

	constexpr inline static uint64_t _calcblock_h(const uint128_t value, const uint64_t h1, const uint64_t h2)
	{
		return (h2 + rotl64c(h1 ^ (_c2() * rotl64c(value.h1 * _c1(), 31)), 27)) * 5 + 0x52dce729;
	}

	constexpr inline static uint128_t _calcblock(const uint128_t value, uint64_t h1, uint64_t h2)
	{
		return uint128_t(_calcblock_h(value, h1, h2),
						 (_calcblock_h(value, h1, h2) + rotl64c(h2 ^ (_c1() * rotl64c(value.h2 * _c2(), 33)), 31)) * 5 +
							 0x38495ab5);
	}

	constexpr inline static uint128_t _calcblocks(const ConstString cs, const int nblocks, const int index,
												  const uint128_t accum)
	{
		return nblocks == 0 ? accum
			   : index == nblocks - 1
				   ? _calcblock(uint128_t(cs.getU64(index * 2 + 0), cs.getU64(index * 2 + 1)), accum.h1, accum.h2)
				   : _calcblocks(
						 cs, nblocks, index + 1,
						 _calcblock(uint128_t(cs.getU64(index * 2 + 0), cs.getU64(index * 2 + 1)), accum.h1, accum.h2));
	}

	constexpr static uint128_t _add(const uint128_t value)
	{
		return uint128_t(value.h1 + value.h2, value.h2 * 2 + value.h1);
	}

	constexpr static uint64_t _fmix_64(uint64_t k)
	{
		return _downshift_and_xor(_downshift_and_xor(_downshift_and_xor(k) * 0xff51afd7ed558ccdLLU) *
								  0xc4ceb9fe1a85ec53LLU);
	}

	constexpr static uint128_t _fmix(const uint128_t value)
	{
		return uint128_t(_fmix_64(value.h1), _fmix_64(value.h2));
	}

	constexpr static uint64_t _calcrest_xor(const ConstString cs, const int offset, const int index, const uint64_t k)
	{
		return k ^ (uint64_t(cs[offset + index]) << (index * 8));
	}

	constexpr static uint64_t _calcrest_k(const ConstString cs, const size_t offset, const size_t index,
										  const size_t len, const uint64_t k)
	{
		return index == (len - 1) ? _calcrest_xor(cs, offset, index, k)
								  : _calcrest_xor(cs, offset, index, _calcrest_k(cs, offset, index + 1, len, k));
	}

	constexpr static uint128_t _calcrest(const ConstString cs, const uint64_t offset, const size_t restlen,
										 const uint128_t value)
	{
		return restlen == 0 ? value
			   : restlen > 8
				   ? uint128_t(
						 value.h1 ^
							 (rotl64c(_calcrest_k(cs, offset, 0, restlen > 8 ? 8 : restlen, 0) * _c1(), 31) * _c2()),
						 value.h2 ^ (rotl64c(_calcrest_k(cs, offset + 8, 0, restlen - 8, 0) * _c2(), 33) * _c1()))
				   : uint128_t(
						 value.h1 ^
							 (rotl64c(_calcrest_k(cs, offset, 0, restlen > 8 ? 8 : restlen, 0) * _c1(), 31) * _c2()),
						 value.h2);
	}

	constexpr static uint128_t _calcfinal(const size_t len, const uint128_t value)
	{
		return _add(_fmix(_add(uint128_t(value.h1 ^ len, value.h2 ^ len))));
	}

	const char* m_Str;
	size_t m_Size;
};

//-----------------------------------------------------------------------------

constexpr uint64_t MurmurHash3c_64(const ConstString& cs, uint64_t seed = 0x1234567)
{
	return cs.hash(seed);
}

constexpr uint64_t operator"" _hash(const char* str, size_t len)
{
	return ConstString(str, len).hash();
}

inline void MurmurHash3c_x64_128(const char* key, int len, const uint32_t seed, void* out)
{
	((uint128_t*)out)[0] = ConstString(key, len).hash128(seed);
}

enum ETest
{
	testval1 = MurmurHash3c_64("hello enum"),
	testval2 = "hello enum"_hash,
};

#include <iostream>

int main(int argc, char** argv)
{
	constexpr uint64_t sh = "hello string"_hash;

	std::cout << "Hash: " << sh << std::endl;
	std::cout << "Hash: " << testval1 << std::endl;
	std::cout << "Hash: " << testval2 << std::endl;
	return 0;
}
