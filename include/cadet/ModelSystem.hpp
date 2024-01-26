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
 * Defines model system interfaces.
 */

#ifndef LIBCADET_MODELSYSTEM_HPP_
#define LIBCADET_MODELSYSTEM_HPP_

#include <unordered_map>
#include <tuple>
#ifdef CADET_BENCHMARK_MODE
#include <vector>
#endif

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"
#include "cadet/ParameterId.hpp"

namespace cadet
{

class IModel;
class IExternalFunction;

/**
 * @brief Interface for the main system of unit operation models
 * @details Users of the library are not supposed to implement this interface! The simulator provided
 *          by this library is not capable of simulating models defined by custom user code.
 *          Instead, this interface defines all possible interactions of user code with this entity.
 */
class CADET_API IModelSystem
{
public:
	virtual ~IModelSystem() CADET_NOEXCEPT
	{
	}

	/**
	 * @brief Returns the highest unit operation index in the model system
	 * @return Highest unit operation index in the system or UnitOpIndep if there are no unit operations
	 */
	virtual UnitOpIdx maxUnitOperationId() const CADET_NOEXCEPT = 0;

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
	 * @brief Returns the value of a parameter that can be made sensitive
	 * @details Returns @c NaN if the parameter is not present in the model.
	 * @param [in] pId ParameterId that identifies the parameter uniquely
	 * @return Value of the parameter or @c NaN if the parameter was not found
	 */
	virtual double getParameterDouble(const ParameterId& pId) const = 0;

	/**
	 * @brief Returns all parameters with their current values that can be made sensitive
	 * @return Map with all parameters that can be made sensitive along with their current value
	 */
	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const = 0;

	/**
	 * @brief Adds a given unit operation model to the system
	 * @details Ownership of the unit operation model is transferred to the IModelSystem
	 * @param [in] unitOp Pointer to unit operation model to be added to the system
	 */
	virtual void addModel(IModel* unitOp) = 0;

	/**
	 * @brief Returns a unit operation model identified by linear index (registration order) from the system
	 * @param [in] index Index of the queried model
	 * @return Unit operation at the queried index or @c nullptr if the index is invalid
	 */
	virtual IModel* getModel(unsigned int index) = 0;
	virtual IModel const* getModel(unsigned int index) const = 0;

	/**
	 * @brief Returns a unit operation model identified by unit operation index (creation order) from the system
	 * @param [in] unitOpIdx Unit operation index of the queried model
	 * @return Unit operation with the queried unit operation index or @c nullptr if no such unit operation exists
	 */
	virtual IModel* getUnitOperationModel(unsigned int unitOpIdx) = 0;
	virtual IModel const* getUnitOperationModel(unsigned int unitOpIdx) const = 0;

	/**
	 * @brief Returns the number of unit operation models in the system
	 * @return Number of unit operation models in the system
	 */
	virtual unsigned int numModels() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Removes the given unit operation model from the system
	 * @details Ownership is also removed from the system and transferred to the caller
	 * @param [in] unitOp Unit operation to be removed from the system
	 */
	virtual void removeModel(IModel const* unitOp) = 0;

	/**
	 * @brief Removes the given unit operation model from the system if it exists
	 * @details Ownership is also removed from the system and transferred to the caller
	 * @param [in] unitOp Unit operation ID of the model to be removed from the system
	 * @return Pointer to the removed model if it was found, otherwise @c nullptr
	 */
	virtual IModel* removeModel(UnitOpIdx unitOp) = 0;

	/**
	 * @brief Adds an external function and hands over ownership to this object
	 * @details Ownership of the given IExternalFunction object is transferred
	 *          to this IModelSystem object. It can be detached by calling
	 *          removeExternalFunction().
	 * @param [in] extFun Object implementing IExternalFunction interface
	 * @return Index of the external function in this IModelSystem
	 */
	virtual unsigned int addExternalFunction(IExternalFunction& extFun) = 0;

	/**
	 * @brief Returns an external functions from the system
	 * @param [in] index Index of the queried external function
	 * @return External function at the queried index or @c nullptr if the index is invalid
	 */
	virtual IExternalFunction* getExternalFunction(unsigned int index) = 0;
	virtual IExternalFunction const* getExternalFunction(unsigned int index) const = 0;

	/**
	 * @brief Returns the number of external functions in the system
	 * @return Number of external functions in the system
	 */
	virtual unsigned int numExternalFunctions() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Removes the given IExternalFunction from this IModelSystem instance
	 * @details After the call this IModelSystem object does not longer own the
	 *          given IExternalFunction.
	 * @param [in] extFun External function object to be removed
	 */
	virtual void removeExternalFunction(IExternalFunction const* extFun) = 0;

	/**
	 * @brief Returns the start and end indices of a unit operation's local slice in the global state vector
	 * @param [in] unitOp Unit operation ID for which local state position is required
	 * @return Start and end indices of the local slice of model @p unitOp
	 */
	virtual std::tuple<unsigned int, unsigned int> getModelStateOffsets(UnitOpIdx unitOp) const CADET_NOEXCEPT = 0;

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

#endif // LIBCADET_MODELSYSTEM_HPP_
