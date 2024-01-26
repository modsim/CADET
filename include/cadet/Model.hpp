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
 * Defines model interfaces.
 */

#ifndef LIBCADET_MODEL_HPP_
#define LIBCADET_MODEL_HPP_

#include <unordered_map>
#ifdef CADET_BENCHMARK_MODE
#include <vector>
#endif

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"
#include "cadet/ParameterId.hpp"

namespace cadet
{

/**
 * @brief Interface common to all unit operation models
 * @details Users of the library are not supposed to implement this interface! The simulator provided
 *          by this library is not capable of simulating models defined by custom user code.
 *          Instead, this interface defines all possible interactions of user code with models implemented
 *          in this library.
 */
class CADET_API IModel
{
public:
	virtual ~IModel() CADET_NOEXCEPT
	{
	}

	/**
	 * @brief Returns the unit operation Id, which is just an index
	 * @details Index of this unit operation model in a system
	 * @return Unit operation index
	 */
	virtual UnitOpIdx unitOperationId() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the name of this unit operation model
	 * @return Name of this unit operation model
	 */
	virtual const char* unitOperationName() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Sets a parameter value
	 * @details The parameter identified by its unique parameter is set to the given value.
	 *
	 * @param [in] pId ParameterId that identifies the parameter uniquely
	 * @param [in] value Value of the parameter
	 *
	 * @return @c true if the parameter has been successfully set to the given value,
	 *         otherwise @c false (e.g., if the parameter is not available in this model)
	 */
	virtual bool setParameter(const ParameterId& pId, int value) = 0;
	virtual bool setParameter(const ParameterId& pId, double value) = 0;
	virtual bool setParameter(const ParameterId& pId, bool value) = 0;

	/**
	 * @brief Checks whether a given parameter exists
	 * @param [in] pId ParameterId that identifies the parameter uniquely
	 * @return @c true if the parameter exists, otherwise @c false
	 */
	virtual bool hasParameter(const ParameterId& pId) const = 0;

	/**
	 * @brief Returns all parameters with their current values that can be made sensitive
	 * @return Map with all parameters that can be made sensitive along with their current value
	 */
	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const = 0;

	/**
	 * @brief Returns the value of a parameter that can be made sensitive
	 * @details Returns @c NaN if the parameter is not present in the model.
	 * @param [in] pId ParameterId that identifies the parameter uniquely
	 * @return Value of the parameter or @c NaN if the parameter was not found
	 */
	virtual double getParameterDouble(const ParameterId& pId) const = 0;

	/**
	 * @brief Determines whether analytical Jacobians are used instead of AD Jacobians
	 *
	 * @param [in] analyticJac @c true if analytic Jacobians should be used (recommended), @c false for AD Jacobians
	 */
	virtual void useAnalyticJacobian(const bool analyticJac) = 0;

#ifdef CADET_BENCHMARK_MODE
	/**
	 * @brief Returns a vector with benchmark timings in seconds
	 * @details The description of the items in the vector are given by benchmarkDescriptions().
	 * @return Benchmark timings in seconds
	 */
	virtual std::vector<double> benchmarkTimings() const = 0;

	/**
	 * @brief Returns an array with descriptions of the benchmark timings returned by benchmarkTimings()
	 * @return Descriptions of benchmark timings
	 */
	virtual char const* const* benchmarkDescriptions() const = 0;
#endif
};

} // namespace cadet

#endif // LIBCADET_MODEL_HPP_
