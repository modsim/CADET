// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

/**
 * @file 
 * Defines the ExchangeModel interface.
 */

#ifndef LIBCADET_EXCHANGEMODELINTERFACE_HPP_ //AB dont know if this is the right path
#define LIBCADET_EXCHANGEMODELINTERFACE_HPP_ //AB dont know if this is the right path

#include <unordered_map>

#include "CompileTimeConfig.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/ParameterId.hpp"
#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"

#ifdef ENABLE_DG
	#include "linalg/BandedEigenSparseRowIterator.hpp"
#endif

#include "AutoDiff.hpp"
#include "SimulationTypes.hpp"
#include "Memory.hpp"

namespace cadet
{

class IParameterProvider;
class IExternalFunction;

struct ColumnPosition;

namespace model
{

/**
 * @brief Defines an inter-channel Exchange Model interface
 * @details The exchange model calculates the fluxes between two channels of the MCT model.
 *			In the MCT model there is no solid phase, so the exchange model models fluxes betwen two liquid phases.
 *
 *          Each exchange can be quasi-stationary, or dynamic.
 *          A dynamic reaction generates the terms
 *          @f[\begin{align}
 *              \frac{\mathrm{d}c_{p,i}}{\mathrm{d}t} + \dots - \frac{1-\varepsilon_p}{\varepsilon_p} v_{i,j}\left( c_p, q \right) &= 0\\
 *              \frac{\mathrm{d}q_{i,j}}{\mathrm{d}t} + \dots + v_{i,j}\left( c_p, q \right) &= 0
 *          \end{align}@f]
 *          where @f$ v_{i,j} @f$ is the flux of the reaction between @f$ c_{p,i} @f$ and @f$ q_{i,j} @f$.
 *          Here, @f$ q_{i,j} @f$ denotes bound state @f$ j @f$ of component @f$ i @f$.
 * 
 *          A quasi-stationary reaction, on the other hand, results in
 *          @f[\begin{align}
 *              v_{i,j}\left( c_p, q \right) &= 0.
 *          \end{align}@f]
 * 
 *          The ordering inside the liquit phase is component-major.
 *			That means, all channel stages are listed one after the other.
 *          For a model having
 *          4 components with two channels, respectively, the ordering is
 *          comp0ex0, comp1ex0, comp2ex0, comp0ex1, comp1ex1, comp2ex1.
 */
class IExchangeModel
{
public:

	virtual ~IExchangeModel() CADET_NOEXCEPT { }

	/**
	 * @brief Returns the name of the exchange model
	 * @details This name is also used to identify and create the exchange model in the factory.
	 * @return Name of the exchange model
	 */
	virtual const char* name() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the exchange model requires additional parameters supplied by configure()
	 * @details After construction of an IExchangeModel object, configureModelDiscretization() is called.
	 *          The exchange model may require to read additional parameters from the adsorption group
	 *          of the parameter provider. This opportunity is given by a call to configure().
	 *          However, a exchange model may not require this. This function communicates to the outside
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
	 * @brief Sets fixed parameters of the exchange model (e.g., the number of components and bound states)
	 * @details This function is called prior to configure() by the underlying model.
	 *          It can only be called once. Whereas non-structural model parameters
	 *          (e.g., rate constants) are configured by configure(), this function
	 *          sets structural parameters (e.g., number of components and bound states).
	 *          In particular, it determines whether each exchange reaction is quasi-stationary.
	 * 
	 *          Before this function is called, usesParamProviderInDiscretizationConfig()
	 *          is queried to determine whether the @p paramProvider is used by this
	 *          function. If it is used, the @p paramProvider is put in the corresponding
	 *          parameter scope. On exit, the @p paramProvider has to point to the same
	 *          scope.
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
	 * @param [in] paramProvider Parameter provider
	 * @param [in] unitOpIdx Index of the unit operation this exchange model belongs to
	 * @param [in] parTypeIdx Index of the particle type this exchange model belongs to
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx) = 0;

	/**
	 * @brief Returns the ParameterId of all channel phase initial conditions (equations)
	 * @details The array has to be filled in the order of the equations.
	 * @param [out] params Array with ParameterId objects to fill
	 * @param [in] unitOpIdx Index of the unit operation this exchange model belongs to
	 * @param [in] parTypeIdx Index of the particle type this exchange model belongs to
	 */
	virtual void fillChannelPhaseInitialParameters(ParameterId* params, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Sets external functions for this exchange model
	 * @details The external functions are not owned by this IExchangeModel.
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
	 * 
	 * @return @c true if the parameter has been successfully set to the given value,
	 *         otherwise @c false (e.g., if the parameter is not available in this model)
	 */
	virtual bool setParameter(const ParameterId& pId, int value) = 0;
	virtual bool setParameter(const ParameterId& pId, double value) = 0;
	virtual bool setParameter(const ParameterId& pId, bool value) = 0;

	/**
	 * @brief Returns a pointer to the parameter identified by the given id
	 * @param [in] pId Parameter Id of the sensitive parameter
	 * @return a pointer to the parameter if the exchange model contains the parameter, otherwise @c nullptr
	 */
	virtual active* getParameter(const ParameterId& pId) = 0;


	/**
	 * @brief Returns whether this exchange model supports multi-state exchange
	 * @return @c true if multi-state exchange is supported, otherwise @c false
	 */
	//AB virtual bool supportsMultistate() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether this exchange model supports non-exchange components
	 * @details Non-exchange components do not have an entry in the solid phase.
	 * @return @c true if non-exchange components are supported, otherwise @c false
	 */
	virtual bool supportsNonExchange() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether this exchange model has quasi-stationary reaction fluxes
	 * @return @c true if quasi-stationary fluxes are present, otherwise @c false
	 */
	virtual bool hasQuasiStationaryReactions() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether this exchange model has dynamic reaction fluxes
	 * @return @c true if dynamic fluxes are present, otherwise @c false
	 */
	virtual bool hasDynamicReactions() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether this exchange model depends on time
	 * @details Binding models may depend on time if external functions are used.
	 * @return @c true if the model is time-dependent, otherwise @c false
	 */
	virtual bool dependsOnTime() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether this exchange model requires workspace
	 * @details The workspace may be required for consistent initialization and / or evaluation
	 *          of residual and Jacobian. A workspace is a memory buffer whose size is given by
	 *          workspaceSize().
	 * @return @c true if the model requires a workspace, otherwise @c false
	 */
	virtual bool requiresWorkspace() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the size of the required workspace in bytes
	 * @details The memory is required for externally dependent exchange models.
	 * @param [in] nComp Number of components
	 * @param [in] totalNumBoundStates Total number of bound states
	 * @param [in] nBoundStates Array with bound states for each component
	 * @return Size of the workspace in bytes
	 */
	virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nBoundStates) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the amount of required AD seed vectors / directions
	 * @details Only internally required AD directions count (e.g., for Jacobian computation).
	 *          Directions used for parameter sensitivities should not be included here.
	 * @return The number of required AD seed vectors / directions
	 */
	virtual unsigned int requiredADdirs() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Evaluates the fluxes
	 * @details The exchange model is responsible for calculating the flux from the mobile to the solid phase.
	 * 
	 *          This function is called simultaneously from multiple threads.
	 *          It needs to overwrite all values of @p res as the result array @p res is not
	 *          zeroed on entry.
	 * @param [in] t Current time point
	 * @param [in] secIdx Index of the current section
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] y Pointer to first bound state of the first component in the current particle shell
	 * @param [in] yCp Pointer to first component of the mobile phase in the current particle shell
	 * @param [out] res Pointer to result array that is filled with the fluxes
	 * @param [in,out] workSpace Memory work space
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* y, active const* yCp, active* res, LinearBufferAllocator workSpace, WithParamSensitivity) const = 0;
	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* y, active const* yCp, active* res, LinearBufferAllocator workSpace, WithoutParamSensitivity) const = 0;
	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp, active* res, LinearBufferAllocator workSpace) const = 0;
	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp, double* res, LinearBufferAllocator workSpace) const = 0;

	/**
	 * @brief Evaluates the Jacobian of the fluxes analytically
	 * @details This function is called simultaneously from multiple threads.
	 *          It can be left out (empty implementation) if AD is used to evaluate the Jacobian.
	 *          
	 *          The Jacobian matrix is assumed to be zeroed out by the caller.
	 * @param [in] t Current time point
	 * @param [in] secIdx Index of the current section
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] y Pointer to first bound state of the first component in the current particle shell
	 * @param [in] offsetCp Offset from the first component of the mobile phase in the current particle shell to @p y
	 * @param [in,out] jac Row iterator pointing to the first bound states row of the underlying matrix in which the Jacobian is stored
	 * @param [in,out] workSpace Memory work space
	 */
	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, linalg::BandMatrix::RowIterator jac, LinearBufferAllocator workSpace) const = 0;
	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, linalg::DenseBandedRowIterator jac, LinearBufferAllocator workSpace) const = 0;
#ifdef ENABLE_DG
	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, linalg::BandedEigenSparseRowIterator jac, LinearBufferAllocator workSpace) const = 0;
#endif
	/**
	 * @brief Calculates the time derivative of the quasi-stationary bound state equations
	 * @details Calculates @f$ \frac{\partial \text{flux}_{\text{qs}}}{\partial t} @f$ for the quasi-stationary equations
	 *          in the flux.
	 * 
	 *          This function is called simultaneously from multiple threads.
	 *          It can be left out (empty implementation) which leads to slightly incorrect initial conditions
	 *          when using externally dependent exchange models.
	 * @param [in] t Current time point
	 * @param [in] secIdx Index of the current section
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in] yCp Pointer to first component in the liquid phase of the current particle shell
	 * @param [in] y Pointer to first bound state of the first component in the current particle shell
	 * @param [out] dResDt Pointer to array that stores the time derivative
	 * @param [in,out] workSpace Memory work space
	 */
	virtual void timeDerivativeQuasiStationaryFluxes(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yCp, double const* y, double* dResDt, LinearBufferAllocator workSpace) const = 0;

	/**
	 * @brief Returns an array that determines whether each exchange reaction is quasi-stationary
	 * @details Each entry represents a truth value (converts to @c bool) that determines whether
	 *          the corresponding exchange reaction is quasi-stationary (@c true) or not (@c false).
	 * @return Array that determines quasi-stationarity of exchange reactions
	 */
	virtual int const* reactionQuasiStationarity() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether a nonlinear solver is required to perform consistent initialization
	 * @details This function is called before a nonlinear solver attempts to solve the algebraic
	 *          equations. If consistent initialization can be performed using analytic / direct
	 *          formulas (i.e., without running a nonlinear solver), this function would be the
	 *          correct place to do it.
	 *          
	 *          This function is called simultaneously from multiple threads.
	 * @param [in] t Current time point
	 * @param [in] secIdx Index of the current section
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in,out] y Pointer to first bound state of the first component in the current particle shell
	 * @param [in] yCp Pointer to first component of the mobile phase in the current particle shell
	 * @param [in,out] workSpace Memory work space
	 * @return @c true if a nonlinear solver is required for consistent initialization, otherwise @c false
	 */
	virtual bool preConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const = 0;

	/**
	 * @brief Called after a nonlinear solver has attempted consistent initialization
	 * @details In consistent initialization, first preConsistentInitialState() is called.
	 *          Depending on the return value, a nonlinear solver is applied to the algebraic
	 *          equations and, afterwards, this function is called to clean up or refine the
	 *          solution.
	 *          
	 *          This function is called simultaneously from multiple threads.
	 * @param [in] t Current time point
	 * @param [in] secIdx Index of the current section
	 * @param [in] colPos Position in normalized coordinates (column inlet = 0, column outlet = 1; outer shell = 1, inner center = 0)
	 * @param [in,out] y Pointer to first bound state of the first component in the current particle shell
	 * @param [in] yCp Pointer to first component of the mobile phase in the current particle shell
	 * @param [in,out] workSpace Memory work space
	 */
	virtual void postConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const = 0;

protected:
};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_EXCHANGEMODELINTERFACE_HPP_
