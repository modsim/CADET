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
 * Defines an IParameterDependence base class.
 */

#ifndef LIBCADET_PARAMETERDEPENDENCEBASE_HPP_
#define LIBCADET_PARAMETERDEPENDENCEBASE_HPP_

#include "model/ParameterDependence.hpp"
#include "ParamIdUtil.hpp"

#include <unordered_map>

namespace cadet
{

namespace model
{

/**
 * @brief Defines a ParameterDependence base class that can be used to implement other parameter dependences
 * @details This base class can be used as a starting point for new parameter dependences.
 *          Some common parameter handling is provided using a hash map (std::unordered_map).
 */
class ParameterDependenceBase : public IParameterDependence
{
public:
	ParameterDependenceBase();

	virtual ~ParameterDependenceBase() CADET_NOEXCEPT;

	virtual bool requiresConfiguration() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool usesParamProviderInDiscretizationConfig() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx,
						   const std::string& name);
	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp,
											  unsigned int const* nBound, unsigned int const* boundOffset);

	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const;
	virtual bool hasParameter(const ParameterId& pId) const;

	virtual bool setParameter(const ParameterId& pId, int value);
	virtual bool setParameter(const ParameterId& pId, double value);
	virtual bool setParameter(const ParameterId& pId, bool value);

	virtual active* getParameter(const ParameterId& pId);

protected:
	int _nComp;                        //!< Number of components
	unsigned int const* _nBoundStates; //!< Array with number of bound states for each component
	unsigned int const* _boundOffset;  //!< Array with offsets to the first bound state of each component
	unsigned int _nTotalBoundStates;

	std::unordered_map<ParameterId, active*> _parameters; //!< Map used to translate ParameterIds to actual variables

	/**
	 * @brief Configures the reaction model
	 * @details This function implements the (re-)configuration of a reaction model. It is called when
	 *          the reaction model is configured or reconfigured. On call the _parameters map will always
	 *          be empty.
	 * @param [in] paramProvider Parameter provider
	 * @param [in] unitOpIdx Unit operation index
	 * @param [in] parTypeIdx Particle type index
	 * @param [in] name Name of the parameter
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx,
							   const std::string& name) = 0;
};

/**
 * @brief Inserts implementations of all parameter() and analyticJacobian() method variants
 * @details An IParameterDependence implementation has to provide liquidParameter(), combinedParameterLiquid(),
 *          combinedParameterSolid(), analyticJacobianLiquidAdd(), analyticJacobianCombinedAddLiquid(), and
 *          analyticJacobianCombinedAddSolid() methods for different variants of state and parameter type.
 *          This macro saves some time by providing those implementations. It assumes that the implementation
 *          provides templatized liquidParameterImpl(), combinedParameterLiquidImpl(),
 *          combinedParameterSolidImpl(), analyticJacobianLiquidAddImpl(), analyticJacobianCombinedAddLiquidImpl(),
 *          and analyticJacobianCombinedAddSolidImpl() functions that realize all required variants.
 *
 *          The implementation is inserted inline in the class declaration.
 */
#define CADET_PARAMETERDEPENDENCE_BOILERPLATE                                                                          \
	virtual active liquidParameter(const ColumnPosition& colPos, const active& param, active const* y, int comp) const \
	{                                                                                                                  \
		return liquidParameterImpl<active, active>(colPos, param, y, comp);                                            \
	}                                                                                                                  \
                                                                                                                       \
	virtual active liquidParameter(const ColumnPosition& colPos, const active& param, double const* y, int comp) const \
	{                                                                                                                  \
		return liquidParameterImpl<double, active>(colPos, param, y, comp);                                            \
	}                                                                                                                  \
                                                                                                                       \
	virtual active liquidParameter(const ColumnPosition& colPos, double param, active const* y, int comp) const        \
	{                                                                                                                  \
		return liquidParameterImpl<active, double>(colPos, param, y, comp);                                            \
	}                                                                                                                  \
                                                                                                                       \
	virtual double liquidParameter(const ColumnPosition& colPos, double param, double const* y, int comp) const        \
	{                                                                                                                  \
		return liquidParameterImpl<double, double>(colPos, param, y, comp);                                            \
	}                                                                                                                  \
                                                                                                                       \
	virtual void analyticJacobianLiquidAdd(const ColumnPosition& colPos, double param, double const* y, int comp,      \
										   double factor, int offset, linalg::BandMatrix::RowIterator jac) const       \
	{                                                                                                                  \
		analyticJacobianLiquidAddImpl(colPos, param, y, comp, factor, offset, jac);                                    \
	}                                                                                                                  \
                                                                                                                       \
	virtual void analyticJacobianLiquidAdd(const ColumnPosition& colPos, double param, double const* y, int comp,      \
										   double factor, int offset, linalg::DenseBandedRowIterator jac) const        \
	{                                                                                                                  \
		analyticJacobianLiquidAddImpl(colPos, param, y, comp, factor, offset, jac);                                    \
	}                                                                                                                  \
                                                                                                                       \
	virtual active combinedParameterLiquid(const ColumnPosition& colPos, const active& param, active const* yLiquid,   \
										   active const* ySolid, int comp) const                                       \
	{                                                                                                                  \
		return combinedParameterLiquidImpl<active, active>(colPos, param, yLiquid, ySolid, comp);                      \
	}                                                                                                                  \
                                                                                                                       \
	virtual active combinedParameterLiquid(const ColumnPosition& colPos, const active& param, double const* yLiquid,   \
										   double const* ySolid, int comp) const                                       \
	{                                                                                                                  \
		return combinedParameterLiquidImpl<double, active>(colPos, param, yLiquid, ySolid, comp);                      \
	}                                                                                                                  \
                                                                                                                       \
	virtual active combinedParameterLiquid(const ColumnPosition& colPos, double param, active const* yLiquid,          \
										   active const* ySolid, int comp) const                                       \
	{                                                                                                                  \
		return combinedParameterLiquidImpl<active, double>(colPos, param, yLiquid, ySolid, comp);                      \
	}                                                                                                                  \
                                                                                                                       \
	virtual double combinedParameterLiquid(const ColumnPosition& colPos, double param, double const* yLiquid,          \
										   double const* ySolid, int comp) const                                       \
	{                                                                                                                  \
		return combinedParameterLiquidImpl<double, double>(colPos, param, yLiquid, ySolid, comp);                      \
	}                                                                                                                  \
                                                                                                                       \
	virtual active combinedParameterSolid(const ColumnPosition& colPos, const active& param, active const* yLiquid,    \
										  active const* ySolid, int bnd) const                                         \
	{                                                                                                                  \
		return combinedParameterSolidImpl<active, active>(colPos, param, yLiquid, ySolid, bnd);                        \
	}                                                                                                                  \
                                                                                                                       \
	virtual active combinedParameterSolid(const ColumnPosition& colPos, const active& param, double const* yLiquid,    \
										  double const* ySolid, int bnd) const                                         \
	{                                                                                                                  \
		return combinedParameterSolidImpl<double, active>(colPos, param, yLiquid, ySolid, bnd);                        \
	}                                                                                                                  \
                                                                                                                       \
	virtual active combinedParameterSolid(const ColumnPosition& colPos, double param, active const* yLiquid,           \
										  active const* ySolid, int bnd) const                                         \
	{                                                                                                                  \
		return combinedParameterSolidImpl<active, double>(colPos, param, yLiquid, ySolid, bnd);                        \
	}                                                                                                                  \
                                                                                                                       \
	virtual double combinedParameterSolid(const ColumnPosition& colPos, double param, double const* yLiquid,           \
										  double const* ySolid, int bnd) const                                         \
	{                                                                                                                  \
		return combinedParameterSolidImpl<double, double>(colPos, param, yLiquid, ySolid, bnd);                        \
	}                                                                                                                  \
                                                                                                                       \
	virtual void analyticJacobianCombinedAddLiquid(const ColumnPosition& colPos, double param, double const* yLiquid,  \
												   double const* ySolid, int comp, double factor, int offset,          \
												   linalg::BandMatrix::RowIterator jac) const                          \
	{                                                                                                                  \
		analyticJacobianCombinedAddLiquidImpl(colPos, param, yLiquid, ySolid, comp, factor, offset, jac);              \
	}                                                                                                                  \
                                                                                                                       \
	virtual void analyticJacobianCombinedAddLiquid(const ColumnPosition& colPos, double param, double const* yLiquid,  \
												   double const* ySolid, int comp, double factor, int offset,          \
												   linalg::DenseBandedRowIterator jac) const                           \
	{                                                                                                                  \
		analyticJacobianCombinedAddLiquidImpl(colPos, param, yLiquid, ySolid, comp, factor, offset, jac);              \
	}                                                                                                                  \
                                                                                                                       \
	virtual void analyticJacobianCombinedAddSolid(const ColumnPosition& colPos, double param, double const* yLiquid,   \
												  double const* ySolid, int bnd, double factor, int offset,            \
												  linalg::BandMatrix::RowIterator jac) const                           \
	{                                                                                                                  \
		analyticJacobianCombinedAddSolidImpl(colPos, param, yLiquid, ySolid, bnd, factor, offset, jac);                \
	}                                                                                                                  \
                                                                                                                       \
	virtual void analyticJacobianCombinedAddSolid(const ColumnPosition& colPos, double param, double const* yLiquid,   \
												  double const* ySolid, int bnd, double factor, int offset,            \
												  linalg::DenseBandedRowIterator jac) const                            \
	{                                                                                                                  \
		analyticJacobianCombinedAddSolidImpl(colPos, param, yLiquid, ySolid, bnd, factor, offset, jac);                \
	}

} // namespace model
} // namespace cadet

#endif // LIBCADET_PARAMETERDEPENDENCEBASE_HPP_
