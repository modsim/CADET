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
 * Defines utility functions for tests.
 */

#ifndef CADETTEST_TESTUTILS_HPP_
#define CADETTEST_TESTUTILS_HPP_

#include <functional>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace cadet
{

namespace test
{

namespace util
{

/**
 * @brief Fills the vector with the values of a given function
 * @details The function @p f uses the current index to assign a value.
 * @param [out] y Populated vector
 * @param [in] f Function for computing the content of the vector
 * @param [in] numDofs Size of the vector
 */
inline void populate(double* y, std::function<double(unsigned int)> f, unsigned int numDofs)
{
	for (unsigned int i = 0; i < numDofs; ++i)
		y[i] = f(i);
}

/**
 * @brief Sequentially places a sequence multiple times into an array
 * @details Writes a given sequence multiple times into an array by concatenation.
 *          Writes a total of @c size * times elements.
 * @param [out] dest Target array
 * @param [in] src Source sequence
 * @param [in] size Length of the source sequence
 * @param [in] times Number of repetitions of the source sequence
 */
inline void repeat(double* dest, double const* src, unsigned int size, unsigned int times)
{
	for (unsigned int i = 0; i < times; ++i, dest += size)
	{
		std::copy(src, src + size, dest);
	}
}

/**
 * @brief ParameterProvider scope guard for unit_XYZ group
 * @details Opens and closes the unit_XYZ ParameterProvider scope.
 */
template <typename ParamProvider> class ModelGroupScope
{
public:
	/**
	 * @brief Enters a given unit_XYZ scope
	 * @param [in,out] jpp ParameterProvider
	 * @param [in] idxUnit Index of unit operation
	 */
	ModelGroupScope(ParamProvider& jpp, unsigned int idxUnit) : _jpp(jpp)
	{
		_active = jpp.exists("model");
		if (_active)
		{
			std::ostringstream ss;
			_jpp.pushScope("model");
			ss << "unit_" << std::setfill('0') << std::setw(3) << idxUnit;
			_jpp.pushScope(ss.str());
		}
	}

	/**
	 * @brief Enters unit_000 scope
	 * @param [in,out] jpp ParameterProvider
	 */
	ModelGroupScope(ParamProvider& jpp) : ModelGroupScope(jpp, 0)
	{
	}

	~ModelGroupScope()
	{
		if (_active)
		{
			_jpp.popScope();
			_jpp.popScope();
		}
	}

private:
	bool _active;
	ParamProvider& _jpp;
};

/**
 * @brief Enters a given unit_XYZ and leaves it when leaving the current scope
 * @param [in,out] jpp ParameterProvider
 * @param [in] idxUnit Index of unit operation
 */
template <typename ParamProvider>
inline ModelGroupScope<ParamProvider> makeModelGroupScope(ParamProvider& jpp, unsigned int idxUnit)
{
	return ModelGroupScope<ParamProvider>(jpp, idxUnit);
}

/**
 * @brief Enters a given unit_000 and leaves it when leaving the current scope
 * @param [in,out] jpp ParameterProvider
 */
template <typename ParamProvider> inline ModelGroupScope<ParamProvider> makeModelGroupScope(ParamProvider& jpp)
{
	return ModelGroupScope<ParamProvider>(jpp);
}

/**
 * @brief ParameterProvider scope guard for possibly existent group
 * @details Opens and closes the given ParameterProvider scope if it exists.
 */
template <typename ParamProvider> class OptionalGroupScope
{
public:
	/**
	 * @brief Enters a given scope if it exists
	 * @param [in,out] jpp ParameterProvider
	 * @param [in] grp Name of group
	 */
	OptionalGroupScope(ParamProvider& jpp, const std::string& grp) : _jpp(jpp)
	{
		_active = jpp.exists(grp);
		if (_active)
			_jpp.pushScope(grp);
	}

	~OptionalGroupScope()
	{
		if (_active)
			_jpp.popScope();
	}

private:
	bool _active;
	ParamProvider& _jpp;
};

/**
 * @brief Enters a given group and leaves it when leaving the current scope
 * @details Does nothing if the group does not exist.
 * @param [in,out] jpp ParameterProvider
 * @param [in] grp Name of the group
 */
template <typename ParamProvider>
inline OptionalGroupScope<ParamProvider> makeOptionalGroupScope(ParamProvider& jpp, const std::string& grp)
{
	return OptionalGroupScope<ParamProvider>(jpp, grp);
}

/**
 * @brief ParameterProvider scope guard for existent group
 * @details Opens and closes the given ParameterProvider scope.
 */
template <typename ParamProvider> class GroupScope
{
public:
	/**
	 * @brief Enters a given scope if it exists
	 * @param [in,out] jpp ParameterProvider
	 * @param [in] grp Name of group
	 */
	GroupScope(ParamProvider& jpp, const std::string& grp) : _jpp(jpp)
	{
		_jpp.pushScope(grp);
	}

	~GroupScope()
	{
		_jpp.popScope();
	}

private:
	ParamProvider& _jpp;
};

/**
 * @brief Enters a given group and leaves it when leaving the current scope
 * @details The group has to exist.
 * @param [in,out] jpp ParameterProvider
 * @param [in] grp Name of the group
 */
template <typename ParamProvider>
inline GroupScope<ParamProvider> makeGroupScope(ParamProvider& jpp, const std::string& grp)
{
	return GroupScope<ParamProvider>(jpp, grp);
}

} // namespace util
} // namespace test
} // namespace cadet

#endif // CADETTEST_TESTUTILS_HPP_
