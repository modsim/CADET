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
 * Defines a helper class for comparing two doubles in tests.
 */

#ifndef CADETTEST_APPROX_HPP_
#define CADETTEST_APPROX_HPP_

#include <type_traits>
#include <stdexcept>
#include <cmath>
#include <limits>

#include <catch.hpp>

namespace cadet
{
namespace test
{

namespace detail
{
// Performs equivalent check of std::fabs(lhs - rhs) <= margin
// But without the subtraction to allow for INFINITY in comparison
inline bool marginComparison(double lhs, double rhs, double margin)
{
	return (lhs + margin >= rhs) && (rhs + margin >= lhs);
}
} // namespace detail

/**
 * @brief Represents an approximate number to compare against
 * @details This class is based on Catch::Approx and only differs in the order of equality checks.
 *          Whereas CATCH checks absolute error first, this class first checks the relative error.
 */
class RelApprox
{
private:
	bool equalityComparisonImpl(const double other) const
	{
		// First try with relative comparison, then try absolute comparison
		return detail::marginComparison(_value, other, _epsilon * (_scale + std::fabs(_value))) ||
			   detail::marginComparison(_value, other, _margin);
	}

public:
	explicit RelApprox(double value)
		: _epsilon(std::numeric_limits<float>::epsilon() * 100.0), _margin(0.0), _scale(0.0), _value(value)
	{
	}

	static RelApprox custom()
	{
		return RelApprox(0.0);
	}

	static inline double defaultEpsilon()
	{
		return std::numeric_limits<float>::epsilon() * 100.0;
	}
	static inline double defaultMargin()
	{
		return 0.0;
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	RelApprox operator()(T const& value)
	{
		RelApprox approx(static_cast<double>(value));
		approx.epsilon(_epsilon);
		approx.margin(_margin);
		approx.scale(_scale);
		return approx;
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	explicit RelApprox(T const& value) : RelApprox(static_cast<double>(value))
	{
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	friend bool operator==(const T& lhs, RelApprox const& rhs)
	{
		const auto lhs_v = static_cast<double>(lhs);
		return rhs.equalityComparisonImpl(lhs_v);
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	friend bool operator==(RelApprox const& lhs, const T& rhs)
	{
		return operator==(rhs, lhs);
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	friend bool operator!=(T const& lhs, RelApprox const& rhs)
	{
		return !operator==(lhs, rhs);
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	friend bool operator!=(RelApprox const& lhs, T const& rhs)
	{
		return !operator==(rhs, lhs);
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	friend bool operator<=(T const& lhs, RelApprox const& rhs)
	{
		return (static_cast<double>(lhs) < rhs._value) || (lhs == rhs);
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	friend bool operator<=(RelApprox const& lhs, T const& rhs)
	{
		return (lhs._value < static_cast<double>(rhs)) || (lhs == rhs);
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	friend bool operator>=(T const& lhs, RelApprox const& rhs)
	{
		return (static_cast<double>(lhs) > rhs._value) || (lhs == rhs);
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	friend bool operator>=(RelApprox const& lhs, T const& rhs)
	{
		return (lhs._value > static_cast<double>(rhs)) || (lhs == rhs);
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	RelApprox& epsilon(T const& newEpsilon)
	{
		const double epsilonAsDouble = static_cast<double>(newEpsilon);
		if (epsilonAsDouble < 0 || epsilonAsDouble > 1.0)
			throw std::domain_error("Invalid RelApprox::epsilon: " + Catch::Detail::stringify(epsilonAsDouble) +
									", RelApprox::epsilon has to be between 0 and 1");

		_epsilon = epsilonAsDouble;
		return *this;
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	RelApprox& margin(T const& newMargin)
	{
		const double marginAsDouble = static_cast<double>(newMargin);
		if (marginAsDouble < 0)
			throw std::domain_error("Invalid RelApprox::margin: " + Catch::Detail::stringify(marginAsDouble) +
									", RelApprox::Margin has to be non-negative.");

		_margin = marginAsDouble;
		return *this;
	}

	template <typename T, typename = typename std::enable_if<std::is_constructible<double, T>::value>::type>
	RelApprox& scale(T const& newScale)
	{
		_scale = static_cast<double>(newScale);
		return *this;
	}

	std::string toString() const
	{
		Catch::ReusableStringStream rss;
		rss << "RelApprox( " << ::Catch::Detail::stringify(_value) << " )";
		return rss.str();
	}

private:
	double _epsilon;
	double _margin;
	double _scale;
	double _value;
};

/**
 * @brief Creates a RelApprox object with given values
 * @param [in] val Reference value
 * @param [in] relTol Relative tolerance
 * @param [in] absTol Absolute tolerance
 * @return RelApprox object with the given values
 */
inline RelApprox makeApprox(double val, double relTol, double absTol)
{
	return RelApprox(val).epsilon(relTol).margin(absTol);
}

} // namespace test
} // namespace cadet

namespace Catch
{

template <> struct StringMaker<cadet::test::RelApprox>
{
	static std::string convert(cadet::test::RelApprox const& value)
	{
		return value.toString();
	}
};

} // end namespace Catch

// Export to global namespace
using cadet::test::RelApprox;

#endif
