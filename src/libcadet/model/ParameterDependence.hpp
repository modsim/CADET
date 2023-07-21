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
 * Defines parameter dependence interfaces.
 */

#ifndef LIBCADET_PARAMDEPINTERFACE_HPP_
#define LIBCADET_PARAMDEPINTERFACE_HPP_

#include <unordered_map>
#include <string>

#include "cadet/ParameterProvider.hpp"
#include "cadet/ParameterId.hpp"
#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"
#include "linalg/SparseMatrix.hpp"
#include "AutoDiff.hpp"

namespace cadet
{

class IParameterProvider;
struct ColumnPosition;
class IModel;

namespace model
{

/**
 * @brief Defines how a parameter depends on a state
 * @details Represents the dependence of a parameter on the state (i.e., concentrations
 *          inside a cell). Scalar as well as vector-valued parameters are considered.
 *          Both simple and extended cells are supported. A simple cell contains only
 *          liquid phase components. An extended cell contains liquid and solid phase
 *          components.
 */
class IParameterStateDependence
{
public:

	virtual ~IParameterStateDependence() CADET_NOEXCEPT { }

	/**
	 * @brief Returns the name of the parameter dependence
	 * @details This name is also used to identify and create the dependence in the factory.
	 * @return Name of the parameter dependence
	 */
	virtual const char* name() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the parameter dependence requires additional parameters supplied by configure()
	 * @details After construction of an IParameterStateDependence object, configureModelDiscretization() is called.
	 *          The parameter dependence may require to read additional parameters from the reaction group
	 *          of the parameter provider. This opportunity is given by a call to configure(). However, a
	 *          parameter dependence may not require this. This function communicates to the outside
	 *          whether a call to configure() is necessary.
	 * @return @c true if configure() has to be called, otherwise @c false
	 */
	virtual bool requiresConfiguration() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the parameter dependence uses the IParameterProvider in configureModelDiscretization()
	 * @details If the IParameterProvider is used in configureModelDiscretization(), it has to be in the correct scope.
	 * @return @c true if the IParameterProvider is used in configureModelDiscretization(), otherwise @c false
	 */
	virtual bool usesParamProviderInDiscretizationConfig() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Sets the number of components in the model
	 * @details This function is called prior to configure() by the underlying model.
	 *          It can only be called once. Model parameters are configured by configure().
	 * 
	 * @param [in] paramProvider Parameter provider
	 * @param [in] nComp Number of components
	 * @param [in] nBound Array of size @p nComp which contains the number of bound states for each component
	 * @param [in] boundOffset Array of size @p nComp with offsets to the first bound state of each component beginning from the solid phase
	 */
	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBound, unsigned int const* boundOffset) = 0;

	/**
	 * @brief Configures the model by extracting all non-structural parameters (e.g., model parameters) from the given @p paramProvider
	 * @details The scope of the cadet::IParameterProvider is left unchanged on return.
	 * 
	 *          The structure of the model is left unchanged, that is, the number of degrees of
	 *          freedom stays the same (e.g., number of bound states is left unchanged). Only
	 *          true (non-structural) model parameters are read and changed.
	 *          
	 *          This function may only be called if configureModelDiscretization() has been called
	 *          in the past. Contrary to configureModelDiscretization(), it can be called multiple
	 *          times.
	 * 
	 * @param [in] paramProvider Parameter provider
	 * @param [in] unitOpIdx Index of the unit operation this parameter dependence belongs to
	 * @param [in] parTypeIdx Index of the particle type this parameter dependence belongs to
	 * @param [in] name Name of the parameter
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, const std::string& name) = 0;

	/**
	 * @brief Returns the maximum number of non-zero Jacobian elements per row in the liquid cell 
	 * @return Maximum number of non-zero Jacobian elements per row in liquid cell
	 */
	virtual int jacobianElementsPerRowLiquid() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the maximum number of non-zero Jacobian elements per row in the combined cell 
	 * @return Maximum number of non-zero Jacobian elements per row in combined cell
	 */
	virtual int jacobianElementsPerRowCombined() const CADET_NOEXCEPT = 0;

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
	 * @brief Sets a parameter value
	 * @details The parameter identified by its unique parameter is set to the given value.
	 * 
	 * @param [in] pId ParameterId that identifies the parameter uniquely
	 * @param [in] value Value of the parameter
	 * @return @c true if the parameter has been successfully set to the given value,
	 *         otherwise @c false (e.g., if the parameter is not available in this model)
	 */
	virtual bool setParameter(const ParameterId& pId, int value) = 0;
	virtual bool setParameter(const ParameterId& pId, double value) = 0;
	virtual bool setParameter(const ParameterId& pId, bool value) = 0;

	/**
	 * @brief Returns a pointer to the parameter identified by the given id
	 * @param [in] pId Parameter Id of the sensitive parameter
	 * @return a pointer to the parameter if the parameter dependence contains the parameter, otherwise @c nullptr
	 */
	virtual active* getParameter(const ParameterId& pId) = 0;

	/**
	 * @brief Evaluates the parameter of component @p comp in one liquid phase cell
	 * @details This function is called simultaneously from multiple threads.
	 * 
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] param Value of the unmodified parameter
	 * @param [in] y Pointer to first component in the current cell
	 * @param [in] comp Index of the component the parameter belongs to (or @c -1 if independent of components)
	 * @return Actual parameter value
	 */
	virtual active liquidParameter(const ColumnPosition& colPos, const active& param, active const* y, int comp) const = 0;
	virtual active liquidParameter(const ColumnPosition& colPos, const active& param, double const* y, int comp) const = 0;
	virtual active liquidParameter(const ColumnPosition& colPos, double param, active const* y, int comp) const = 0;
	virtual double liquidParameter(const ColumnPosition& colPos, double param, double const* y, int comp) const = 0;

	/**
	 * @brief Adds the analytical Jacobian of the parameter dependence for one liquid phase cell
	 * @details Adds the Jacobian of the parameter dependence for the given component and liquid
	 *          phase cell to the full Jacobian. The Jacobian is premultiplied with a given
	 *          factor. That is, this function realizes
	 *          \f[ \begin{align} 
	 *              J = J + \gamma J_P
	 *          \end{align} \f]
	 *          where @f$ J_P @f$ is the Jacobian of the parameter dependence.
	 * 
	 *          This function is called simultaneously from multiple threads.
	 *          It can be left out (empty implementation) if AD is used to evaluate the Jacobian.
	 *
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] param Value of the unmodified parameter
	 * @param [in] y Pointer to first component in the current cell
	 * @param [in] comp Index of the component the parameter belongs to (or @c -1 if independent of components)
	 * @param [in] factor Factor @f$ \gamma @f$
	 * @param [in] offset Diaognal offset in the row iterator
	 * @param [in] row Row index
	 * @param [in,out] jac Row iterator pointing to the row of component @p comp in the underlying matrix that stores the Jacobian
	 */
	virtual void analyticJacobianLiquidAdd(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, linalg::BandMatrix::RowIterator jac) const = 0;
	virtual void analyticJacobianLiquidAdd(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, linalg::DenseBandedRowIterator jac) const = 0;

	/**
	 * @brief Adds the analytical Jacobian of the parameter dependence for one liquid phase cell
	 * @details Adds the Jacobian of the parameter dependence for the given component and liquid
	 *          phase cell to the full Jacobian. The Jacobian is premultiplied with a given
	 *          factor. That is, this function realizes
	 *          \f[ \begin{align} 
	 *              J = J + \gamma J_P
	 *          \end{align} \f]
	 *          where @f$ J_P @f$ is the Jacobian of the parameter dependence.
	 * 
	 *          This function is called simultaneously from multiple threads.
	 *          It can be left out (empty implementation) if AD is used to evaluate the Jacobian.
	 *
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] param Value of the unmodified parameter
	 * @param [in] y Pointer to first component in the current cell
	 * @param [in] comp Index of the component the parameter belongs to (or @c -1 if independent of components)
	 * @param [in] factor Factor @f$ \gamma @f$
	 * @param [in] offset Offset in the row pointing to the first liquid phase component
	 * @param [in] row Row index
	 * @param [in,out] jac Jacobian
	 */
	virtual void analyticJacobianLiquidAdd(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const = 0;

	/**
	 * @brief Evaluates the parameter of component @p comp in the liquid phase of one combined phase cell
	 * @details This function is called simultaneously from multiple threads.
	 * 
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] param Value of the unmodified parameter
	 * @param [in] yLiquid Pointer to first component in the current cell's liquid phase
	 * @param [in] ySolid Pointer to first component in the current cell's solid phase
	 * @param [in] comp Index of the component the parameter belongs to (or @c -1 if independent of components)
	 * @return Actual parameter value
	 */
	virtual active combinedParameterLiquid(const ColumnPosition& colPos, const active& param, active const* yLiquid, active const* ySolid, int comp) const = 0;
	virtual active combinedParameterLiquid(const ColumnPosition& colPos, const active& param, double const* yLiquid, double const* ySolid, int comp) const = 0;
	virtual active combinedParameterLiquid(const ColumnPosition& colPos, double param, active const* yLiquid, active const* ySolid, int comp) const = 0;
	virtual double combinedParameterLiquid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp) const = 0;

	/**
	 * @brief Evaluates the parameter of component @p comp in the solid phase of one combined phase cell
	 * @details This function is called simultaneously from multiple threads.
	 * 
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] param Value of the unmodified parameter
	 * @param [in] yLiquid Pointer to first component in the current cell's liquid phase
	 * @param [in] ySolid Pointer to first component in the current cell's solid phase
	 * @param [in] bnd Index of the bound state the parameter belongs to (or @c -1 if independent of bound states)
	 * @return Actual parameter value
	 */
	virtual active combinedParameterSolid(const ColumnPosition& colPos, const active& param, active const* yLiquid, active const* ySolid, int bnd) const = 0;
	virtual active combinedParameterSolid(const ColumnPosition& colPos, const active& param, double const* yLiquid, double const* ySolid, int bnd) const = 0;
	virtual active combinedParameterSolid(const ColumnPosition& colPos, double param, active const* yLiquid, active const* ySolid, int bnd) const = 0;
	virtual double combinedParameterSolid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd) const = 0;

	/**
	 * @brief Adds the analytical Jacobian of the parameter dependence for the liquid phase in one combined phase cell
	 * @details Adds the Jacobian of the parameter dependence for the given component and combined
	 *          phase cell to the full Jacobian. The Jacobian is premultiplied with a given
	 *          factor. That is, this function realizes
	 *          \f[ \begin{align} 
	 *              J = J + \gamma J_P
	 *          \end{align} \f]
	 *          where @f$ J_P @f$ is the Jacobian of the parameter dependence.
	 * 
	 *          This function is called simultaneously from multiple threads.
	 *          It can be left out (empty implementation) if AD is used to evaluate the Jacobian.
	 *
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] param Value of the unmodified parameter
	 * @param [in] yLiquid Pointer to first component in the current cell's liquid phase
	 * @param [in] ySolid Pointer to first component in the current cell's solid phase
	 * @param [in] comp Index of the component the parameter belongs to (or @c -1 if independent of components)
	 * @param [in] factor Factor @f$ \gamma @f$
	 * @param [in] offset Diaognal offset in the row iterator
	 * @param [in,out] jac Row iterator pointing to the row of component @p comp in the underlying matrix that stores the Jacobian
	 */
	virtual void analyticJacobianCombinedAddLiquid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, linalg::BandMatrix::RowIterator jac) const = 0;
	virtual void analyticJacobianCombinedAddLiquid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, linalg::DenseBandedRowIterator jac) const = 0;

	/**
	 * @brief Adds the analytical Jacobian of the parameter dependence for the liquid phase in one combined phase cell
	 * @details Adds the Jacobian of the parameter dependence for the given component and combined
	 *          phase cell to the full Jacobian. The Jacobian is premultiplied with a given
	 *          factor. That is, this function realizes
	 *          \f[ \begin{align} 
	 *              J = J + \gamma J_P
	 *          \end{align} \f]
	 *          where @f$ J_P @f$ is the Jacobian of the parameter dependence.
	 * 
	 *          This function is called simultaneously from multiple threads.
	 *          It can be left out (empty implementation) if AD is used to evaluate the Jacobian.
	 *
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] param Value of the unmodified parameter
	 * @param [in] yLiquid Pointer to first component in the current cell's liquid phase
	 * @param [in] ySolid Pointer to first component in the current cell's solid phase
	 * @param [in] comp Index of the component the parameter belongs to (or @c -1 if independent of components)
	 * @param [in] factor Factor @f$ \gamma @f$
	 * @param [in] offset Offset in the row pointing to the first liquid phase component
	 * @param [in] row Row index
	 * @param [in,out] jac Jacobian
	 */
	virtual void analyticJacobianCombinedAddLiquid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const = 0;

	/**
	 * @brief Adds the analytical Jacobian of the parameter dependence for the solid phase in one combined phase cell
	 * @details Adds the Jacobian of the parameter dependence for the given bound state and combined
	 *          phase cell to the full Jacobian. The Jacobian is premultiplied with a given
	 *          factor. That is, this function realizes
	 *          \f[ \begin{align} 
	 *              J = J + \gamma J_P
	 *          \end{align} \f]
	 *          where @f$ J_P @f$ is the Jacobian of the parameter dependence.
	 * 
	 *          This function is called simultaneously from multiple threads.
	 *          It can be left out (empty implementation) if AD is used to evaluate the Jacobian.
	 *
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] param Value of the unmodified parameter
	 * @param [in] yLiquid Pointer to first component in the current cell's liquid phase
	 * @param [in] ySolid Pointer to first component in the current cell's solid phase
	 * @param [in] bnd Index of the bound state the parameter belongs to (or @c -1 if independent of bound states)
	 * @param [in] factor Factor @f$ \gamma @f$
	 * @param [in] offset Diaognal offset in the row iterator
	 * @param [in,out] jac Row iterator pointing to the row of bound state @p bnd in the underlying matrix that stores the Jacobian
	 */
	virtual void analyticJacobianCombinedAddSolid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, linalg::BandMatrix::RowIterator jac) const = 0;
	virtual void analyticJacobianCombinedAddSolid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, linalg::DenseBandedRowIterator jac) const = 0;

	/**
	 * @brief Adds the analytical Jacobian of the parameter dependence for the solid phase in one combined phase cell
	 * @details Adds the Jacobian of the parameter dependence for the given bound state and combined
	 *          phase cell to the full Jacobian. The Jacobian is premultiplied with a given
	 *          factor. That is, this function realizes
	 *          \f[ \begin{align} 
	 *              J = J + \gamma J_P
	 *          \end{align} \f]
	 *          where @f$ J_P @f$ is the Jacobian of the parameter dependence.
	 * 
	 *          This function is called simultaneously from multiple threads.
	 *          It can be left out (empty implementation) if AD is used to evaluate the Jacobian.
	 *
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] param Value of the unmodified parameter
	 * @param [in] yLiquid Pointer to first component in the current cell's liquid phase
	 * @param [in] ySolid Pointer to first component in the current cell's solid phase
	 * @param [in] bnd Index of the bound state the parameter belongs to (or @c -1 if independent of bound states)
	 * @param [in] factor Factor @f$ \gamma @f$
	 * @param [in] offset Offset in the row pointing to the first liquid phase component
	 * @param [in] row Row index
	 * @param [in,out] jac Jacobian
	 */
	virtual void analyticJacobianCombinedAddSolid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const = 0;

protected:
};

/**
 * @brief Defines how a parameter depends on another parameter
 * @details Represents the dependence of a parameter on another parameter.
 */
class IParameterParameterDependence
{
public:

	virtual ~IParameterParameterDependence() CADET_NOEXCEPT { }

	/**
	 * @brief Returns the name of the parameter dependence
	 * @details This name is also used to identify and create the dependence in the factory.
	 * @return Name of the parameter dependence
	 */
	virtual const char* name() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the parameter dependence requires additional parameters supplied by configure()
	 * @details After construction of an IParameterParameterDependence object, configureModelDiscretization() is called.
	 *          The parameter dependence may require to read additional parameters from the reaction group
	 *          of the parameter provider. This opportunity is given by a call to configure(). However, a
	 *          parameter dependence may not require this. This function communicates to the outside
	 *          whether a call to configure() is necessary.
	 * @return @c true if configure() has to be called, otherwise @c false
	 */
	virtual bool requiresConfiguration() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the parameter dependence uses the IParameterProvider in configureModelDiscretization()
	 * @details If the IParameterProvider is used in configureModelDiscretization(), it has to be in the correct scope.
	 * @return @c true if the IParameterProvider is used in configureModelDiscretization(), otherwise @c false
	 */
	virtual bool usesParamProviderInDiscretizationConfig() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Configures the parameter dependence
	 * @details This function is called prior to configure() by the underlying model.
	 *          It can only be called once. Model parameters are configured by configure().
	 * 
	 * @param [in] paramProvider Parameter provider
	 */
	virtual bool configureModelDiscretization(IParameterProvider& paramProvider) = 0;

	/**
	 * @brief Configures the model by extracting all non-structural parameters (e.g., model parameters) from the given @p paramProvider
	 * @details The scope of the cadet::IParameterProvider is left unchanged on return.
	 * 
	 *          The structure of the model is left unchanged, that is, the number of degrees of
	 *          freedom stays the same (e.g., number of bound states is left unchanged). Only
	 *          true (non-structural) model parameters are read and changed.
	 *          
	 *          This function may only be called if configureModelDiscretization() has been called
	 *          in the past. Contrary to configureModelDiscretization(), it can be called multiple
	 *          times.
	 * 
	 * @param [in] paramProvider Parameter provider
	 * @param [in] unitOpIdx Index of the unit operation this parameter dependence belongs to
	 * @param [in] parTypeIdx Index of the particle type this parameter dependence belongs to
	 * @param [in] bndIdx Index of the bound state this parameter dependence belongs to
	 * @param [in] name Name of the parameter
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, BoundStateIdx bndIdx, const std::string& name) = 0;

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
	 * @brief Sets a parameter value
	 * @details The parameter identified by its unique parameter is set to the given value.
	 * 
	 * @param [in] pId ParameterId that identifies the parameter uniquely
	 * @param [in] value Value of the parameter
	 * @return @c true if the parameter has been successfully set to the given value,
	 *         otherwise @c false (e.g., if the parameter is not available in this model)
	 */
	virtual bool setParameter(const ParameterId& pId, int value) = 0;
	virtual bool setParameter(const ParameterId& pId, double value) = 0;
	virtual bool setParameter(const ParameterId& pId, bool value) = 0;

	/**
	 * @brief Returns a pointer to the parameter identified by the given id
	 * @param [in] pId Parameter Id of the sensitive parameter
	 * @return a pointer to the parameter if the parameter dependence contains the parameter, otherwise @c nullptr
	 */
	virtual active* getParameter(const ParameterId& pId) = 0;

	/**
	 * @brief Evaluates the parameter
	 * @details This function is called simultaneously from multiple threads.
	 * 
	 * @param [in] model Model that owns this parameter dependence
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] comp Index of the component the parameter belongs to (or @c -1 if independent of components)
	 * @param [in] parType Index of the particle type the parameter belongs to (or @c -1 if independent of particle types)
	 * @param [in] bnd Index of the bound state the parameter belongs to (or @c -1 if independent of bound states)
	 * @param [in] depVal parameter-dependent value
	 * @param [in] indepVal parameter depVal depends on
	 * @return Actual parameter value
	 */
	virtual double getValue(const IModel& model, const ColumnPosition& colPos, int comp, int parType, int bnd, double depVal, double indepVal) const = 0;

	/**
	 * @brief Evaluates the parameter
	 * @details This function is called simultaneously from multiple threads.
	 * 
	 * @param [in] model Model that owns this parameter dependence
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] comp Index of the component the parameter belongs to (or @c -1 if independent of components)
	 * @param [in] parType Index of the particle type the parameter belongs to (or @c -1 if independent of particle types)
	 * @param [in] bnd Index of the bound state the parameter belongs to (or @c -1 if independent of bound states)
	 * @param [in] depVal parameter-dependent value
	 * @param [in] indepVal parameter depVal depends on
	 * @return Actual parameter value
	 */
	virtual active getValue(const IModel& model, const ColumnPosition& colPos, int comp, int parType, int bnd, const active& depVal, const active& indepVal) const = 0;

protected:
};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_PARAMDEPINTERFACE_HPP_
