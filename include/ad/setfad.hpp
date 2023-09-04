// =============================================================================
//  SETFAD - Simple Expression Template Forward Automatic Differentiation
//  (Part of SFAD library)
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

#ifndef _SETFAD_MAIN_HPP_
#define _SETFAD_MAIN_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "sfad-common.hpp"

namespace sfad
{
namespace detail
{
template <typename real_t> real_t sqr(const real_t& x) SFAD_NOEXCEPT
{
	return x * x;
}
} // namespace detail

// Base class for all expression type trees
template <typename T, typename real_t> class Expr
{
public:
	inline const T& base() const SFAD_NOEXCEPT
	{
		return static_cast<const T&>(*this);
	}
	inline const real_t value() const SFAD_NOEXCEPT
	{
		return base().value();
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return base().gradient(idx);
	}

private:
	// Cannot assign to expression
	Expr& operator=(const Expr&)
	{
		return *this;
	}
};

// Main data type that actually holds the derivative vector
template <typename real_t> class FwdET : public Expr<FwdET<real_t>, real_t>
{
public:
	typedef std::size_t idx_t;

	FwdET(const real_t val) SFAD_NOEXCEPT : _val(0)
	{
		setADValue(real_t(0));
	}
	FwdET(const real_t val) SFAD_NOEXCEPT : _val(val)
	{
		setADValue(real_t(0));
	}
	FwdET(const real_t val, real_t const* const grad) SFAD_NOEXCEPT : _val(val)
	{
		std::copy_n(grad, detail::globalGradSize, _grad);
	}
	FwdET(const FwdET<real_t>& cpy) SFAD_NOEXCEPT = default;
	FwdET(FwdET<real_t>&& other) SFAD_NOEXCEPT = default;

	// Contains the one (and only) loop in expression template paradigm
	template <typename A> FwdET(const Expr<A, real_t>& other) SFAD_NOEXCEPT : _val(other.value())
	{
		for (idx_t i = 0; i < detail::globalGradSize; ++i)
			_grad[i] = other.gradient(i);
	}

	~FwdET() SFAD_NOEXCEPT = default;

	FwdET<real_t>& operator=(FwdET<real_t>&& other) SFAD_NOEXCEPT = default;
	FwdET<real_t>& operator=(const FwdET<real_t>& other) SFAD_NOEXCEPT = default;

	inline const idx_t gradientSize() const SFAD_NOEXCEPT
	{
		return detail::globalGradSize;
	}

	template <typename T> friend void swap(FwdET<T>& x, FwdET<T>& y) SFAD_NOEXCEPT;

	// ADOL-C compatibility

	inline real_t getValue() SFAD_NOEXCEPT
	{
		return _val;
	}
	inline const real_t getValue() const SFAD_NOEXCEPT
	{
		return _val;
	}
	inline void setValue(const real_t v) SFAD_NOEXCEPT
	{
		_val = v;
	}

	inline real_t getADValue(const idx_t idx)
	{
		return _grad[idx];
	}
	inline const real_t getADValue(const idx_t idx) const
	{
		return _grad[idx];
	}
	inline void setADValue(const idx_t idx, const real_t v)
	{
		_grad[idx] = v;
	}
	inline void setADValue(const real_t v)
	{
		fillADValue(v);
	}

	inline void fillADValue(const real_t v)
	{
		fillADValue(0, detail::globalGradSize, v);
	}
	inline void fillADValue(const idx_t start, const real_t v)
	{
		fillADValue(start, detail::globalGradSize, v);
	}
	inline void fillADValue(const idx_t start, const idx_t end, const real_t v)
	{
		std::fill(_grad + start, _grad + end, v);
	}

	// Modern C++ accessor

	inline real_t& operator[](const idx_t idx)
	{
		return _grad[idx];
	}
	inline const real_t operator[](const idx_t idx) const
	{
		return _grad[idx];
	}

	// Explicit cast operator to underlying scalar type
	explicit operator real_t() const SFAD_NOEXCEPT
	{
		return _val;
	}

	// Assignment
	inline FwdET<real_t>& operator=(const real_t v)
	{
		_val = v;
		setADValue(real_t(0));

		return *this;
	}

	// Contains the one (and only) loop in expression template paradigm
	template <class T> inline FwdET<real_t>& operator=(const Expr<T, real_t>& v)
	{
		_val = v.value();
		for (idx_t i = 0; i < detail::globalGradSize; ++i)
			_grad[i] = v.gradient(i);

		return *this;
	}

	// Expr class interface
	const real_t value() const SFAD_NOEXCEPT
	{
		return _val;
	}
	const real_t gradient(std::size_t idx) const
	{
		return _grad[idx];
	}

	// AssignOp Operators, i.e., +=, -=, *=, /=

	// Addition
	inline FwdET<real_t>& operator+=(const real_t v)
	{
		_val += v;
		return *this;
	}

	inline FwdET<real_t>& operator+=(const FwdET<real_t>& a)
	{
		_val += a._val;
		for (idx_t i = 0; i < detail::globalGradSize; ++i)
			_grad[i] += a._grad[i];

		return *this;
	}

	// Substraction
	inline FwdET<real_t>& operator-=(const real_t v)
	{
		_val -= v;
		return *this;
	}

	inline FwdET<real_t>& operator-=(const FwdET<real_t>& a)
	{
		_val -= a._val;
		for (idx_t i = 0; i < detail::globalGradSize; ++i)
			_grad[i] -= a._grad[i];

		return *this;
	}

	// Multiplication
	inline FwdET<real_t>& operator*=(const real_t v)
	{
		_val *= v;
		for (idx_t i = 0; i < detail::globalGradSize; ++i)
			_grad[i] *= v;
		return *this;
	}

	inline FwdET<real_t>& operator*=(const FwdET<real_t>& a)
	{
		for (idx_t i = 0; i < detail::globalGradSize; ++i)
			_grad[i] = a._val * _grad[i] + _val * a._grad[i];

		_val *= a._val;
		return *this;
	}

	// Division
	inline FwdET<real_t>& operator/=(const real_t v)
	{
		_val /= v;
		for (idx_t i = 0; i < detail::globalGradSize; ++i)
			_grad[i] /= v;
		return *this;
	}

	inline FwdET<real_t>& operator/=(const FwdET<real_t>& a)
	{
		for (idx_t i = 0; i < detail::globalGradSize; ++i)
			//				_grad[i] = (_grad[i] - _val / a._val * a._grad[i]) / a._val;
			_grad[i] = (_grad[i] * a._val - _val * a._grad[i]) / (a._val * a._val);

		_val /= a._val;
		return *this;
	}

protected:
	real_t _val;
	real_t _grad[SFAD_DEFAULT_DIR];
};

// Basic arithmetics

template <class A, class B, typename real_t> class Add : public Expr<Add<A, B, real_t>, real_t>
{
public:
	Add(const Expr<A, real_t>& a, const Expr<B, real_t>& b) SFAD_NOEXCEPT : _a(a.base()), _b(b.base())
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return _a.value() + _b.value();
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return _a.gradient(idx) + _b.gradient(idx);
	}

protected:
	const A& _a;
	const B& _b;
};

template <class A, class B, typename real_t>
inline Add<A, B, real_t> operator+(const Expr<A, real_t>& a, const Expr<B, real_t>& b) SFAD_NOEXCEPT
{
	return Add<A, B, real_t>(a.base(), b.base());
}

template <class A, class B, typename real_t> class Subtract : public Expr<Subtract<A, B, real_t>, real_t>
{
public:
	Subtract(const Expr<A, real_t>& a, const Expr<B, real_t>& b) SFAD_NOEXCEPT : _a(a.base()), _b(b.base())
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return _a.value() - _b.value();
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return _a.gradient(idx) - _b.gradient(idx);
	}

protected:
	const A& _a;
	const B& _b;
};

template <class A, class B, typename real_t>
inline Subtract<A, B, real_t> operator-(const Expr<A, real_t>& a, const Expr<B, real_t>& b) SFAD_NOEXCEPT
{
	return Subtract<A, B, real_t>(a.base(), b.base());
}

template <class A, class B, typename real_t> class Multiply : public Expr<Multiply<A, B, real_t>, real_t>
{
public:
	Multiply(const Expr<A, real_t>& a, const Expr<B, real_t>& b) SFAD_NOEXCEPT : _a(a.base()),
																				 _b(b.base()),
																				 _cacheA(_a.value()),
																				 _cacheB(_b.value())
	{
	}

	//	    inline const real_t value() const SFAD_NOEXCEPT { return _a.value() * _b.value(); }
	//	    inline const real_t gradient(std::size_t idx) const { return _a.gradient(idx) * _b.value() + _a.value() *
	//_b.gradient(idx); }
	inline const real_t value() const SFAD_NOEXCEPT
	{
		return _cacheA * _cacheB;
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return _a.gradient(idx) * _cacheB + _cacheA * _b.gradient(idx);
	}

protected:
	const A& _a;
	const B& _b;
	const real_t _cacheA;
	const real_t _cacheB;
};

template <class A, class B, typename real_t>
inline Multiply<A, B, real_t> operator*(const Expr<A, real_t>& a, const Expr<B, real_t>& b) SFAD_NOEXCEPT
{
	return Multiply<A, B, real_t>(a.base(), b.base());
}

template <class A, class B, typename real_t> class Divide : public Expr<Divide<A, B, real_t>, real_t>
{
public:
	Divide(const Expr<A, real_t>& a, const Expr<B, real_t>& b) SFAD_NOEXCEPT : _a(a.base()),
																			   _b(b.base()),
																			   _cacheA(_a.value()),
																			   _cacheB(_b.value())
	{
	}

	//	    inline const real_t value() const SFAD_NOEXCEPT { return _a.value() / _b.value(); }
	//	    inline const real_t gradient(std::size_t idx) const { return (_a.gradient(idx) * _b.value() - _a.value() *
	//_b.gradient(idx)) / (_b.value() * _b.value()); }
	inline const real_t value() const SFAD_NOEXCEPT
	{
		return _cacheA / _cacheB;
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return (_a.gradient(idx) * _cacheB - _cacheA * _b.gradient(idx)) / (_cacheB * _cacheB);
	}

protected:
	const A& _a;
	const B& _b;
	const real_t _cacheA;
	const real_t _cacheB;
};

template <class A, class B, typename real_t>
inline Divide<A, B, real_t> operator/(const Expr<A, real_t>& a, const Expr<B, real_t>& b) SFAD_NOEXCEPT
{
	return Divide<A, B, real_t>(a.base(), b.base());
}

template <class A, typename real_t> class ScalarAdd : public Expr<ScalarAdd<A, real_t>, real_t>
{
public:
	ScalarAdd(const Expr<A, real_t>& a, const real_t b) SFAD_NOEXCEPT : _a(a.base()), _b(b)
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return _a.value() + _b;
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return _a.gradient(idx);
	}

protected:
	const A& _a;
	const real_t _b;
};

template <class A, typename real_t>
inline ScalarAdd<A, real_t> operator+(const Expr<A, real_t>& a, const real_t b) SFAD_NOEXCEPT
{
	return ScalarAdd<A, real_t>(a.base(), b);
}

template <class A, typename real_t>
inline ScalarAdd<A, real_t> operator+(const real_t a, const Expr<A, real_t>& b) SFAD_NOEXCEPT
{
	return ScalarAdd<A, real_t>(b.base(), a);
}

template <class A, typename real_t>
inline ScalarAdd<A, real_t> operator-(const Expr<A, real_t>& a, const real_t b) SFAD_NOEXCEPT
{
	return ScalarAdd<A, real_t>(a.base(), -b);
}

template <class A, typename real_t> class ScalarSubtract : public Expr<ScalarSubtract<A, real_t>, real_t>
{
public:
	ScalarSubtract(const real_t a, const Expr<A, real_t>& b) SFAD_NOEXCEPT : _a(a), _b(b.base())
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return _a - _b.value();
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return -_b.gradient(idx);
	}

protected:
	const real_t _a;
	const A& _b;
};

template <class A, typename real_t>
inline ScalarSubtract<A, real_t> operator-(const real_t a, const Expr<A, real_t>& b) SFAD_NOEXCEPT
{
	return ScalarSubtract<A, real_t>(a, b.base());
}

template <class A, typename real_t> class ScalarMultiply : public Expr<ScalarMultiply<A, real_t>, real_t>
{
public:
	ScalarMultiply(const Expr<A, real_t>& a, const real_t b) SFAD_NOEXCEPT : _a(a.base()), _b(b)
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return _a.value() * _b;
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return _b * _a.gradient(idx);
	}

protected:
	const A& _a;
	const real_t _b;
};

template <class A, typename real_t>
inline ScalarMultiply<A, real_t> operator*(const Expr<A, real_t>& a, const real_t b) SFAD_NOEXCEPT
{
	return ScalarMultiply<A, real_t>(a.base(), b);
}

template <class A, typename real_t>
inline ScalarMultiply<A, real_t> operator*(const real_t a, const Expr<A, real_t>& b) SFAD_NOEXCEPT
{
	return ScalarMultiply<A, real_t>(b.base(), a);
}

template <class A, typename real_t> class DivideScalar : public Expr<DivideScalar<A, real_t>, real_t>
{
public:
	DivideScalar(const Expr<A, real_t>& a, const real_t b) SFAD_NOEXCEPT : _a(a.base()), _b(b)
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return _a.value() / _b;
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return _a.gradient(idx) / _b;
	}

protected:
	const A& _a;
	const real_t _b;
};

template <class A, typename real_t>
inline DivideScalar<A, real_t> operator/(const Expr<A, real_t>& a, const real_t b) SFAD_NOEXCEPT
{
	return DivideScalar<A, real_t>(a.base(), b);
}

template <class A, typename real_t> class ScalarDivide : public Expr<ScalarDivide<A, real_t>, real_t>
{
public:
	ScalarDivide(const real_t a, const Expr<A, real_t>& b) SFAD_NOEXCEPT : _a(a), _b(b.base()), _cache(_b.value())
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return _a / _b.value();
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return -_a * _b.gradient(idx) / (_cache * _cache);
	}

protected:
	const real_t _a;
	const A& _b;
	const real_t _cache;
};

template <class A, typename real_t>
inline ScalarDivide<A, real_t> operator/(const real_t a, const Expr<A, real_t>& b) SFAD_NOEXCEPT
{
	return ScalarDivide<A, real_t>(a, b.base());
}

template <class A, typename real_t> class UnaryMinus : public Expr<UnaryMinus<A, real_t>, real_t>
{
public:
	UnaryMinus(const Expr<A, real_t>& a) SFAD_NOEXCEPT : _a(a.base())
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return -_a.value();
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return -_a.gradient(idx);
	}

protected:
	const A& _a;
};

template <class A, typename real_t> inline UnaryMinus<A, real_t> operator-(const Expr<A, real_t>& a) SFAD_NOEXCEPT
{
	return UnaryMinus<A, real_t>(a.base());
}

template <class A, typename real_t> inline A operator+(const Expr<A, real_t>& a) SFAD_NOEXCEPT
{
	return a.base();
}

// Conditionals

#define SETFAD_CONDITIONAL_ET_FUNC(OPNAME, OPERATOR)                                                                   \
	template <class A, class B, typename real_t>                                                                       \
	inline bool OPNAME(const Expr<A, real_t>& a, const Expr<B, real_t>& b) SFAD_NOEXCEPT                               \
	{                                                                                                                  \
		return a.value() OPERATOR b.value();                                                                           \
	}                                                                                                                  \
                                                                                                                       \
	template <class A, typename real_t> inline bool OPNAME(const Expr<A, real_t>& a, const real_t b) SFAD_NOEXCEPT     \
	{                                                                                                                  \
		return a.value() OPERATOR b;                                                                                   \
	}                                                                                                                  \
                                                                                                                       \
	template <class A, class B, typename real_t> inline bool OPNAME(real_t a, const Expr<B, real_t>& b) SFAD_NOEXCEPT  \
	{                                                                                                                  \
		return a OPERATOR b.value();                                                                                   \
	}

SETFAD_CONDITIONAL_ET_FUNC(operator==, ==)
SETFAD_CONDITIONAL_ET_FUNC(operator!=, !=)
SETFAD_CONDITIONAL_ET_FUNC(operator>, >)
SETFAD_CONDITIONAL_ET_FUNC(operator<, <)
SETFAD_CONDITIONAL_ET_FUNC(operator>=, >=)
SETFAD_CONDITIONAL_ET_FUNC(operator<=, <=)

// Unary math functions

template <class A, typename real_t> class Exp : public Expr<Exp<A, real_t>, real_t>
{
public:
	Exp(const Expr<A, real_t>& a) SFAD_NOEXCEPT : _a(a.base()), _cache(std::exp(_a.value()))
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return _cache;
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return _cache * _a.gradient(idx);
	}

protected:
	const A& _a;
	const real_t _cache;
};

template <class A, typename real_t> inline Exp<A, real_t> exp(const Expr<A, real_t>& a) SFAD_NOEXCEPT
{
	return Exp<A, real_t>(a.base());
}

#define SETFAD_UNARY_FUNC_ET_CLASS(CLASSNAME, VALOP, CACHEOP, FUNCDIFF, FUNCNAME)                                      \
	template <class A, typename real_t> class CLASSNAME : public Expr<CLASSNAME<A, real_t>, real_t>                    \
	{                                                                                                                  \
	public:                                                                                                            \
		CLASSNAME(const Expr<A, real_t>& a) SFAD_NOEXCEPT : _a(a.base()), _cache(CACHEOP)                              \
		{                                                                                                              \
		}                                                                                                              \
                                                                                                                       \
		inline const real_t value() const SFAD_NOEXCEPT                                                                \
		{                                                                                                              \
			return VALOP;                                                                                              \
		}                                                                                                              \
		inline const real_t gradient(std::size_t idx) const                                                            \
		{                                                                                                              \
			return FUNCDIFF;                                                                                           \
		}                                                                                                              \
                                                                                                                       \
	protected:                                                                                                         \
		const A& _a;                                                                                                   \
		const real_t _cache;                                                                                           \
	};                                                                                                                 \
                                                                                                                       \
	template <class A, typename real_t> inline CLASSNAME<A, real_t> FUNCNAME(const Expr<A, real_t>& a) SFAD_NOEXCEPT   \
	{                                                                                                                  \
		return CLASSNAME<A, real_t>(a.base());                                                                         \
	}

SETFAD_UNARY_FUNC_ET_CLASS(Log, std::log(_cache), _a.value(), _a.gradient(idx) / _cache, log)
SETFAD_UNARY_FUNC_ET_CLASS(Square, detail::sqr(_cache), _a.value(), _a.gradient(idx) * real_t(2) * _cache, sqr)
SETFAD_UNARY_FUNC_ET_CLASS(Sqrt, _cache, std::sqrt(_a.value()), _a.gradient(idx) / (real_t(2) * _cache), sqrt)

SETFAD_UNARY_FUNC_ET_CLASS(Sin, std::sin(_a.value()), std::cos(_a.value()), _a.gradient(idx) * _cache, sin)
SETFAD_UNARY_FUNC_ET_CLASS(Cos, std::cos(_a.value()), -std::sin(_a.value()), _a.gradient(idx) * _cache, cos)
SETFAD_UNARY_FUNC_ET_CLASS(Tan, std::tan(_a.value()), detail::sqr(std::cos(_a.value())), _a.gradient(idx) / _cache, tan)

SETFAD_UNARY_FUNC_ET_CLASS(Asin, std::asin(_a.value()), std::sqrt(real_t(1) - detail::sqr(_a.value())),
						   _a.gradient(idx) / _cache, asin)
SETFAD_UNARY_FUNC_ET_CLASS(Acos, std::acos(_a.value()), -std::sqrt(real_t(1) - detail::sqr(_a.value())),
						   _a.gradient(idx) / _cache, acos)
SETFAD_UNARY_FUNC_ET_CLASS(Atan, std::atan(_a.value()), real_t(1) + detail::sqr(_a.value()), _a.gradient(idx) / _cache,
						   atan)

SETFAD_UNARY_FUNC_ET_CLASS(Sinh, std::sinh(_a.value()), std::cosh(_a.value()), _a.gradient(idx) * _cache, sinh)
SETFAD_UNARY_FUNC_ET_CLASS(Cosh, std::cosh(_a.value()), std::sinh(_a.value()), _a.gradient(idx) * _cache, cosh)
SETFAD_UNARY_FUNC_ET_CLASS(Tanh, _cache, std::tanh(_a.value()),
						   _a.gradient(idx) * (real_t(1) - _cache) * (real_t(1) + _cache), tanh)

SETFAD_UNARY_FUNC_ET_CLASS(Asinh, std::asinh(_a.value()), std::sqrt(real_t(1) + detail::sqr(_a.value())),
						   _a.gradient(idx) / _cache, asinh)
SETFAD_UNARY_FUNC_ET_CLASS(Acosh, std::acosh(_a.value()), std::sqrt(detail::sqr(_a.value()) - real_t(1)),
						   _a.gradient(idx) / _cache, acosh)
SETFAD_UNARY_FUNC_ET_CLASS(Atanh, std::atanh(_a.value()), real_t(1) - detail::sqr(_a.value()),
						   _a.gradient(idx) / _cache, atanh)

SETFAD_UNARY_FUNC_ET_CLASS(Abs, std::abs(_a.value()), _a.value() > real_t(0) ? real_t(1) : real_t(-1),
						   _a.gradient(idx) * _cache, abs)

template <class A, typename real_t> inline Abs<A, real_t> fabs(const Expr<A, real_t>& a) SFAD_NOEXCEPT
{
	return Abs<A, real_t>(a.base());
}

// Unary functions that reset gradient to zero

#define SETFAD_UNARY_FUNC_ET_CLASS_RESET(CLASSNAME, FUNCAPPLY, FUNCNAME)                                               \
	template <class A, typename real_t> class CLASSNAME : public Expr<CLASSNAME<A, real_t>, real_t>                    \
	{                                                                                                                  \
	public:                                                                                                            \
		CLASSNAME(const Expr<A, real_t>& a) SFAD_NOEXCEPT : _a(a.base())                                               \
		{                                                                                                              \
		}                                                                                                              \
                                                                                                                       \
		inline const real_t value() const SFAD_NOEXCEPT                                                                \
		{                                                                                                              \
			return FUNCAPPLY(_a.value());                                                                              \
		}                                                                                                              \
		inline const real_t gradient(std::size_t idx) const                                                            \
		{                                                                                                              \
			return real_t(0);                                                                                          \
		}                                                                                                              \
                                                                                                                       \
	protected:                                                                                                         \
		const A& _a;                                                                                                   \
	};                                                                                                                 \
                                                                                                                       \
	template <class A, typename real_t> inline CLASSNAME<A, real_t> FUNCNAME(const Expr<A, real_t>& a) SFAD_NOEXCEPT   \
	{                                                                                                                  \
		return CLASSNAME<A, real_t>(a.base());                                                                         \
	}

SETFAD_UNARY_FUNC_ET_CLASS_RESET(Ceil, std::ceil, ceil)
SETFAD_UNARY_FUNC_ET_CLASS_RESET(Floor, std::floor, floor)

// Binary math functions

template <class A, class B, typename real_t> class Pow : public Expr<Pow<A, B, real_t>, real_t>
{
public:
	Pow(const Expr<A, real_t>& a, const Expr<A, real_t>& b) SFAD_NOEXCEPT
		: _a(a.base()),
		  _b(b.base()),
		  _cacheA(_a.value()),
		  _cacheB(_b.value()),
		  _cacheC(_cacheB* std::pow(_cacheA, _cacheB - real_t(1))),
		  _cacheD(std::pow(_cacheA, _cacheB) * std::log(_cacheA))
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return std::pow(_cacheA, _cacheB);
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return _a.gradient(idx) * _cacheC + _b.gradient(idx) * _cacheD;
	}

protected:
	const A& _a;
	const B& _b;
	const real_t _cacheA;
	const real_t _cacheB;
	const real_t _cacheC;
	const real_t _cacheD;
};

template <class A, class B, typename real_t>
inline Pow<A, B, real_t> pow(const Expr<A, real_t>& a, const Expr<B, real_t>& b) SFAD_NOEXCEPT
{
	return Pow<A, B, real_t>(a.base(), b.base());
}

template <class A, typename real_t> class PowScalar : public Expr<PowScalar<A, real_t>, real_t>
{
public:
	PowScalar(const Expr<A, real_t>& a, const real_t b) SFAD_NOEXCEPT : _a(a.base()),
																		_b(b),
																		_cacheA(_a.value()),
																		_cacheB(_b* std::pow(_cacheA, _b - real_t(1)))
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return std::pow(_cacheA, _b);
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return _a.gradient(idx) * _cacheB;
	}

protected:
	const A& _a;
	const real_t _b;
	const real_t _cacheA;
	const real_t _cacheB;
};

template <class A, typename real_t>
inline PowScalar<A, real_t> pow(const Expr<A, real_t>& a, const real_t b) SFAD_NOEXCEPT
{
	return PowScalar<A, real_t>(a.base(), b);
}

template <class A, typename real_t> class ScalarPow : public Expr<ScalarPow<A, real_t>, real_t>
{
public:
	ScalarPow(const real_t a, const Expr<A, real_t>& b) SFAD_NOEXCEPT : _a(a),
																		_b(b.base()),
																		_cacheA(_b.value()),
																		_cacheB(std::pow(_a, _cacheA) * std::log(_a))
	{
	}

	inline const real_t value() const SFAD_NOEXCEPT
	{
		return std::pow(_a, _b.value());
	}
	inline const real_t gradient(std::size_t idx) const
	{
		return _b.gradient(idx) * _cacheB;
	}

protected:
	const real_t _a;
	const A& _b;
	const real_t _cacheA;
	const real_t _cacheB;
};

template <class A, typename real_t>
inline ScalarPow<A, real_t> pow(const real_t a, const Expr<A, real_t>& b) SFAD_NOEXCEPT
{
	return ScalarPow<A, real_t>(a, b.base());
}

template <typename real_t> void swap(FwdET<real_t>& x, FwdET<real_t>& y) SFAD_NOEXCEPT
{
	using std::swap;
	swap(x._val, y._val);
	swap(x._grad, y._grad);
}

} // namespace sfad

#endif
