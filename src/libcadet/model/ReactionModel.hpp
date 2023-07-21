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
 * Defines reaction model interfaces.
 */

#ifndef LIBCADET_REACTIONMODELINTERFACE_HPP_
#define LIBCADET_REACTIONMODELINTERFACE_HPP_

#include <unordered_map>

#include "CompileTimeConfig.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/ParameterId.hpp"
#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"
#include "linalg/BandedEigenSparseRowIterator.hpp"
#include "linalg/CompressedSparseMatrix.hpp"
#include "AutoDiff.hpp"
#include "Memory.hpp"

namespace cadet
{

class IParameterProvider;
class IExternalFunction;

struct ColumnPosition;

namespace model
{

/**
 * @brief Defines an internal DynamicReactionModel interface
 * @details Represents dynamic reactions between components in a finite volume cell.
 *          Both simple and extended cells are supported. A simple cell contains only
 *          liquid phase components. An extended cell contains liquid and solid phase
 *          components.
 */
class IDynamicReactionModel
{
public:

	virtual ~IDynamicReactionModel() CADET_NOEXCEPT { }

	/**
	 * @brief Returns the name of the dynamic reaction model
	 * @details This name is also used to identify and create the dynamic reaction model in the factory.
	 * @return Name of the dynamic reaction model
	 */
	virtual const char* name() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the dynamic reaction model requires additional parameters supplied by configure()
	 * @details After construction of an IDynamicReactionModel object, configureModelDiscretization() is called.
	 *          The dynamic reaction model may require to read additional parameters from the reaction group
	 *          of the parameter provider. This opportunity is given by a call to configure(). However, a
	 *          dynamic reaction model may not require this. This function communicates to the outside
	 *          whether a call to configure() is necessary.
	 * @return @c true if configure() has to be called, otherwise @c false
	 */
	virtual bool requiresConfiguration() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the dynamic reaction model uses the IParameterProvider in configureModelDiscretization()
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
	 * @param [in] unitOpIdx Index of the unit operation this dynamic reaction model belongs to
	 * @param [in] parTypeIdx Index of the particle type this dynamic reaction model belongs to
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx) = 0;

	/**
	 * @brief Sets external functions for this dynamic reaction model
	 * @details The external functions are not owned by this IDynamicReactionModel.
	 * 
	 * @param [in] extFuns Pointer to array of IExternalFunction objects of size @p size
	 * @param [in] size Number of elements in the IExternalFunction array @p extFuns
	 */
	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size) = 0;

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
	 * @return a pointer to the parameter if the dynamic reaction model contains the parameter, otherwise @c nullptr
	 */
	virtual active* getParameter(const ParameterId& pId) = 0;

	/**
	 * @brief Returns whether this dynamic reaction model depends on time
	 * @details Dynamic reaction models may depend on time if external functions are used.
	 * @return @c true if the model is time-dependent, otherwise @c false
	 */
	virtual bool dependsOnTime() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether this dynamic reaction model requires workspace
	 * @details The workspace may be required for evaluation of residual and
	 *          Jacobian. A workspace is a memory buffer whose size is given by
	 *          workspaceSize().
	 * @return @c true if the model requires a workspace, otherwise @c false
	 */
	virtual bool requiresWorkspace() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the size of the required workspace in bytes
	 * @details The memory is required for externally dependent dynamic reaction models.
	 * @param [in] nComp Number of components
	 * @param [in] totalNumBoundStates Total number of bound states
	 * @param [in] nBoundStates Array with bound states for each component
	 * @return Size of the workspace in bytes
	 */
	virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nBoundStates) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Evaluates the residual for one liquid phase cell
	 * @details Adds the dynamic reaction terms to the residual vector for the given
	 *          liquid phase cell. The reaction terms are premultiplied with a given
	 *          factor. That is, this function realizes
	 *          \f[ \begin{align} 
	 *              \text{res} = \text{res} + \gamma R
	 *          \end{align} \f]
	 *          where @f$ R @f$ are the reaction terms.
	 *          
	 *          This function is called simultaneously from multiple threads.
	 * 
	 * @param [in] t Current time point
	 * @param [in] secIdx Index of the current section
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] y Pointer to first component in the current cell
	 * @param [out] res Pointer to residual of the first component in the current particle shell's liquid phase
	 * @param [in] factor Factor @f$ \gamma @f$
	 * @param [in,out] workSpace Memory work space
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residualLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* y,
		active* res, const active& factor, LinearBufferAllocator workSpace) const = 0;

	virtual int residualLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* y,
		active* res, double factor, LinearBufferAllocator workSpace) const = 0;

	virtual int residualLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y,
		active* res, double factor, LinearBufferAllocator workSpace) const = 0;

	virtual int residualLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y,
		double* res, double factor, LinearBufferAllocator workSpace) const = 0;

	/**
	 * @brief Adds the analytical Jacobian of the reaction terms for one liquid phase cell
	 * @details Adds the Jacobian of the dynamic reaction terms for the given liquid phase
	 *          cell to the full Jacobian. The reaction terms are premultiplied with a given
	 *          factor. That is, this function realizes
	 *          \f[ \begin{align} 
	 *              J = J + \gamma J_R
	 *          \end{align} \f]
	 *          where @f$ J_R @f$ is the Jacobian of the reaction terms.
	 * 
	 *          This function is called simultaneously from multiple threads.
	 *          It can be left out (empty implementation) if AD is used to evaluate the Jacobian.
	 *
	 * @param [in] t Current time point
	 * @param [in] secIdx Index of the current section
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] y Pointer to first component in the current cell
	 * @param [in] factor Factor @f$ \gamma @f$
	 * @param [in,out] jac Row iterator pointing to the first component row of the underlying matrix in which the Jacobian is stored
	 * @param [in,out] workSpace Memory work space
	 */
	virtual void analyticJacobianLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor, linalg::BandMatrix::RowIterator jac, LinearBufferAllocator workSpace) const = 0;
	virtual void analyticJacobianLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor, linalg::DenseBandedRowIterator jac, LinearBufferAllocator workSpace) const = 0;
	virtual void analyticJacobianLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor, linalg::BandedSparseRowIterator jac, LinearBufferAllocator workSpace) const = 0;
	#ifdef ENABLE_DG
	virtual void analyticJacobianLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor, linalg::BandedEigenSparseRowIterator jac, LinearBufferAllocator workSpace) const = 0;
	#endif

	/**
	 * @brief Evaluates the residual for one combined phase cell
	 * @details Adds the dynamic reaction terms to the residual vector for the given
	 *          combined phase cell. The reaction terms are premultiplied with a given
	 *          factor. That is, this function realizes
	 *          \f[ \begin{align} 
	 *              \text{res} = \text{res} + \gamma R
	 *          \end{align} \f]
	 *          where @f$ R @f$ are the reaction terms.
	 *          
	 *          This function is called simultaneously from multiple threads.
	 * 
	 * @param [in] t Current time point
	 * @param [in] secIdx Index of the current section
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] yLiquid Pointer to first component in the current cell's liquid phase
	 * @param [in] ySolid Pointer to first component in the current cell's solid phase
	 * @param [out] resLiquid Pointer to residual of the first component in the current particle shell's liquid phase
	 * @param [out] resSolid Pointer to residual of the first bound state of the first component in the current particle shell
	 * @param [in] factor Factor @f$ \gamma @f$
	 * @param [in,out] workSpace Memory work space
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residualCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* yLiquid,
		active const* ySolid, active* resLiquid, active* resSolid, double factor, LinearBufferAllocator workSpace) const = 0;

	virtual int residualCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid,
		double const* ySolid, active* resLiquid, active* resSolid, double factor, LinearBufferAllocator workSpace) const = 0;

	virtual int residualCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid,
		double const* ySolid, double* resLiquid, double* resSolid, double factor, LinearBufferAllocator workSpace) const = 0;

	/**
	 * @brief Adds the analytical Jacobian of the reaction terms for one combined phase cell
	 * @details Adds the Jacobian of the dynamic reaction terms for the given combined phase
	 *          cell to the full Jacobian. The reaction terms are premultiplied with a given
	 *          factor. That is, this function realizes
	 *          \f[ \begin{align} 
	 *              J = J + \gamma J_R
	 *          \end{align} \f]
	 *          where @f$ J_R @f$ is the Jacobian of the reaction terms.
	 * 
	 *          This function is called simultaneously from multiple threads.
	 *          It can be left out (empty implementation) if AD is used to evaluate the Jacobian.
	 *
	 * @param [in] t Current time point
	 * @param [in] secIdx Index of the current section
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] yLiquid Pointer to first component in the current cell's liquid phase
	 * @param [in] ySolid Pointer to first component in the current cell's solid phase
	 * @param [in] factor Factor @f$ \gamma @f$
	 * @param [in,out] jacLiquid Row iterator pointing to the first component row of the underlying matrix in which the Jacobian is stored
	 * @param [in,out] jacSolid Row iterator pointing to the first bound state row of the underlying matrix in which the Jacobian is stored
	 * @param [in,out] workSpace Memory work space
	 */
	virtual void analyticJacobianCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid, double factor, linalg::BandMatrix::RowIterator jacLiquid, linalg::BandMatrix::RowIterator jacSolid, LinearBufferAllocator workSpace) const = 0;
	virtual void analyticJacobianCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid, double factor, linalg::DenseBandedRowIterator jacLiquid, linalg::DenseBandedRowIterator jacSolid, LinearBufferAllocator workSpace) const = 0;
	virtual void analyticJacobianCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid, double factor, linalg::BandMatrix::RowIterator jacLiquid, linalg::DenseBandedRowIterator jacSolid, LinearBufferAllocator workSpace) const = 0;
	#ifdef ENABLE_DG
	virtual void analyticJacobianCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid, double factor, linalg::BandedEigenSparseRowIterator jacLiquid, linalg::DenseBandedRowIterator jacSolid, LinearBufferAllocator workSpace) const = 0;
	#endif

	/**
	 * @brief Returns the number of reactions for a liquid phase cell
	 * @return Number of reactions
	 */
	virtual unsigned int numReactionsLiquid() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of reactions for a combined phase cell
	 * @return Number of reactions
	 */
	virtual unsigned int numReactionsCombined() const CADET_NOEXCEPT = 0;

protected:
};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_REACTIONMODELINTERFACE_HPP_
