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
 * Defines an unit operation model interface.
 */

#ifndef LIBCADET_IUNITOPERATION_HPP_
#define LIBCADET_IUNITOPERATION_HPP_

#include "cadet/Model.hpp"
#include "AutoDiff.hpp"

#include <vector>

namespace cadet
{

class ISolutionRecorder;
class IParameterProvider;
class IConfigHelper;
class IExternalFunction;

struct AdJacobianParams;
struct SimulationTime;
struct SimulationState;
struct ConstSimulationState;

namespace util
{
	class ThreadLocalStorage;
}

/**
 * @brief Defines an unit operation model interface
 * @details 
 */
class IUnitOperation : public IModel
{
public:
	/**
	 * @brief Return the number of required DOFs
	 * @details This includes any additional coupling DOFs.
	 * @return The number of required DOFs
	 */
	virtual unsigned int numDofs() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Return the number of pure DOFs excluding all additional coupling DOFs
	 * @details If <tt>numDofs() == numPureDofs()</tt>, then no additional coupling DOFs
	 *          are used. Otherwise, the number of pure DOFs (e.g., sum of DOFs of all
	 *          submodels) is returned.
	 * @return The number of pure DOFs
	 */
	virtual unsigned int numPureDofs() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether AD is used for computing the system Jacobian
	 * @details This is independent of any parameter sensitivity.
	 * @return @c true if AD is required, otherwise @c false
	 */
	virtual bool usesAD() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the amount of required AD seed vectors / directions
	 * @details Only internally required AD directions count (e.g., for Jacobian computation).
	 *          Directions used for parameter sensitivities should not be included here.
	 * @return The number of required AD seed vectors / directions
	 */
	virtual unsigned int requiredADdirs() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Configures the model discretization by extracting all structural parameters from the given @p paramProvider
	 * @details The scope of the cadet::IParameterProvider is left unchanged on return.
	 *          
	 *          Only the structure / discretization (e.g., number of components, adsorption
	 *          model, bound states) is configured. The actual model parameters are configured
	 *          in configure(). 
	 *          
	 *          This function can only be called once. Once the discretization is set, it cannot
	 *          be changed.
	 * 
	 * @param [in] paramProvider Parameter provider
	 * @param [in] helper Used to inject or create required objects
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, const IConfigHelper& helper) = 0;

	/**
	 * @brief (Re-)configures the model by extracting all non-structural parameters (e.g., model parameters) from the given @p paramProvider
	 * @details The scope of the cadet::IParameterProvider is left unchanged on return.
	 *          
	 *          The structure of the model is left unchanged, that is, the number of degrees of
	 *          freedom stays the same. Only true (non-structural) model parameters are read and
	 *          changed. Parameters that concern discretization (e.g., number of cells), model
	 *          structure (e.g., number of components, binding model), and numerical solution
	 *          (e.g., tolerances in GMRES iterations) are left untouched.
	 *          
	 *          This function may only be called if configureModelDiscretization() has been called
	 *          in the past. Contrary to configureModelDiscretization(), it can be called multiple
	 *          times.
	 * 
	 * @param [in] paramProvider Parameter provider
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configure(IParameterProvider& paramProvider) = 0;

	/**
	 * @brief Reports the given solution to the cadet::ISolutionRecorder
	 * @param [in] reporter Where to report the solution to
	 * @param [in] solution Solution that is reported (from index 0 to numDofs())
	 */
	virtual void reportSolution(ISolutionRecorder& reporter, double const* const solution) const = 0;

	/**
	 * @brief Reports the solution structure (ordering of DOFs) to the cadet::ISolutionRecorder
	 * @param [in] reporter Where to report the solution structure to
	 */
	virtual void reportSolutionStructure(ISolutionRecorder& reporter) const = 0;

	/**
	 * @brief Marks a parameter as sensitive (i.e., sensitivities for this parameter are to be computed)
	 * @param [in] pId Parameter Id of the sensitive parameter
	 * @param [in] adDirection AD direction assigned to this parameter
	 * @param [in] adValue Value of the derivative in the given direction
	 * @return @c true if the parameter has been found in the model, otherwise @c false
	 */
	virtual bool setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue) = 0;

	/**
	 * @brief Sets the value of a parameter that can be sensitive
	 * @details This also sets values of parameters that are not marked as sensitive at the moment.
	 *          If the parameter is part of a fused sensitivity, then all fused parameters are set
	 *          to the same value by calling this function for each parameter.
	 *          
	 *          Note that the AD directions are kept invariant (i.e., they are not resetted).
	 * @param [in] id Parameter ID of the parameter to be manipulated
	 * @param [in] value Value of the parameter
	 */
	virtual void setSensitiveParameterValue(const ParameterId& id, double value) = 0;

	/**
	 * @brief Clears all sensitive parameters
	 */
	virtual void clearSensParams() = 0;

	/**
	 * @brief Returns the number of sensitive parameters in this unit operation
	 * @return Number of sensitive parameters
	 */
	virtual unsigned int numSensParams() const = 0;

	/**
	 * @brief Notifies the model that a discontinuous section transition is in progress
	 * @details This function is called after time integration of a section has finished and a new
	 *          section is about to be integrated. This allows the model to update internal state before
	 *          consistent initialization is performed.
	 * 
	 *          This function is also called at the beginning of the time integration, which allows
	 *          the model to perform setup operations.
	 *          
	 *          If AD is used by the model, the function has the opportunity to update the seed vectors.
	 *          The general initialization of the seed vectors is performed by prepareADvectors().
	 *          
	 * @param [in] t Current time point
	 * @param [in] secIdx Index of the new section that is about to be integrated
	 * @param [in] simState Simulation state
	 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
	 */
	virtual void notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac) = 0;

	/**
	 * @brief Applies initial conditions to the state vector and its time derivative
	 * @details The initial conditions do not need to be consistent at this point. On a (discontinuous)
	 *          transition from one section to the next, consistentInitialConditions() is called by
	 *          the time integrator in order to compute consistent initial conditions. Therefore,
	 *          consistentInitialConditions() is also called at the beginning of the simulation, that is,
	 *          the initial conditions set by this function will be corrected for consistency.
	 *          Note that the state vector and its time derivative are pre-initialized with zero by the
	 *          time integrator.
	 * 
	 * @param [in,out] simState State of the simulation (state vector and its time derivatives) to be updated with initial values
	 */
	virtual void applyInitialCondition(const SimulationState& simState) const = 0;

	/**
	 * @brief Reads initial conditions for the state vector and its time derivative from the given parameter provider
	 * @details The scope of the cadet::IParameterProvider is left unchanged on return.
	 * 
	 * @param [in] paramProvider Parameter provider
	 */
	virtual void readInitialCondition(IParameterProvider& paramProvider) = 0;

	/**
	 * @brief Computes the residual
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [out] res Pointer to local residual vector
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem) = 0;

	/**
	 * @brief Computes the residual and updates the Jacobian
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [out] res Pointer to local residual vector
	 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem) = 0;

	/**
	 * @brief Computes the solution of the linear system involving the system Jacobian
	 * @details The system \f[ \left( \frac{\partial F}{\partial y} + \alpha \frac{\partial F}{\partial \dot{y}} \right) x = b \f]
	 *          has to be solved. The right hand side \f$ b \f$ is given by @p rhs, the Jacobians are evaluated at the
	 *          point \f$(y, \dot{y})\f$ given by @p y and @p yDot. The residual @p res at this point, \f$ F(t, y, \dot{y}) \f$,
	 *          may help with this. Error weights (see IDAS guide) are given in @p weight. The solution is returned in @p rhs.
	 *
	 * @param [in] t Current time point
	 * @param [in] alpha Value of \f$ \alpha \f$ (arises from BDF time discretization)
	 * @param [in] tol Error tolerance for the solution of the linear system from outer Newton iteration
	 * @param [in,out] rhs On entry the right hand side of the linear equation system, on exit the solution
	 * @param [in] weight Vector with error weights
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int linearSolve(double t, double alpha, double tol, double* const rhs, double const* const weight,
		const ConstSimulationState& simState) = 0;

	/**
	 * @brief Prepares the AD system vectors by constructing seed vectors
	 * @details Sets the seed vectors used in AD. Since the AD vector slice is fully managed by the model,
	 *          the seeds are unchanged during one time integration (except for possible changes in
	 *          notifyDiscontinuousSectionTransition()). This function is called at the beginning of
	 *          every time integration and should initialize the AD seed vectors. 
	 *          
	 *          If those vectors do not change during one time integration, their initialization should
	 *          be performed in this function. The notifyDiscontinuousSectionTransition() function is
	 *          only used to update seed vectors during time integration.
	 * 
	 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
	 */
	virtual void prepareADvectors(const AdJacobianParams& adJac) const = 0;

	/**
	 * @brief Sets initial sensitivity system state
	 * @details Given the DAE \f[ F(t, y, \dot{y}, p) = 0, \f] the corresponding (linear) forward sensitivity
	 *          system reads \f[ \frac{\partial F}{\partial y}(t, y, \dot{y}) s + \frac{\partial F}{\partial \dot{y}}(t, y, \dot{y}) \dot{s} + \frac{\partial F}{\partial p}(t, y, \dot{y}) = 0. \f]
	 *          The initial value of \f$ s_0 = \frac{\mathrm{d} y_0}{\mathrm{d}p} \f$ is set by this function.
	 *          Note that consistent initialization is performed later, which in particular calculates a matching
	 *          \f$ \dot{s}_0 = \frac{\mathrm{d} \dot{y}_0}{\mathrm{d}p}. \f$
	 *          
	 *          The sensitivity state vectors passed to this function are initialized to @c 0. They only have
	 *          to change in case of sensitivities with respect to initial conditions.
	 * 
	 * @param [out] vecSensY Sensitivity subsystem state vectors
	 */
	virtual void initializeSensitivityStates(const std::vector<double*>& vecSensY) const = 0;

	/**
	 * @brief Sets the section time vector
	 * @details The integration time is partitioned into sections. All parameters and
	 *          equations are assumed continuous inside one section. Thus, sections
	 *          provide means to implement discontinuous behavior (e.g., pulse injection profiles,
	 *          switching of valves). After initialization, the simulator notifies all entities
	 *          such as models or data sources of its section times.
	 *          
	 *          The vector of section times consists of strictly increasing time points
	 *          @f[ t_0 < t_1 < t_2 < \dots t_N @f]
	 *          which mark the beginning and end of a section. The @f$ i@f$-th section is given by
	 *          @f$ \left[ t_i, t_{i+1} \right]. @f$
	 *          If a transition from one section to the next is continuous, the @p secContinuity flag
	 *          for that transition is @c true. In this case, the time integrator will not stop at the
	 *          transition time point and reinitialize consistently (which will be done for discontinuous
	 *          transitions). 
	 * 
	 * @param [in] secTimes Vector with section time points (length is @p nSections + 1)
	 * @param [in] secContinuity Vector of flags that indicate a continuous (@c true) or discontinuous (@c false) 
	 *             transition from the current section to the next one (length is @p nSections - 1). For instance,
	 *             the first element indicates whether the transition from section @c 0 to @c 1 is continuous.
	 * @param [in] nSections Number of sections
	 */
	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections) = 0;

	/**
	 * @brief Expand a (short) error tolerance specification into a detailed / full one
	 * @details The format of the short error specification is model specific. Common variants
	 *          are error tolerances for each component in each phase (independent of the specific
	 *          discretization of the model). The short error spec is expanded into a full one which
	 *          contains an error tolerance for each (pure) DOF of the model.
	 * 
	 * @param [in] errorSpec Pointer to first element of an array containing the short error spec
	 * @param [in] errorSpecSize Size of the short error tolerance spec
	 * @param [out] expandOut Pointer to the first element of an array receiving the full error specification
	 */
	virtual void expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut) = 0;

	/**
	 * @brief Returns the number of components
	 * @details It is assumed that the number of components is also the number of inputs
	 *          and outputs of the unit operation.
	 * @return Number of components
	 */
	virtual unsigned int numComponents() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether this unit operation possesses an inlet
	 * @return @c true if the unit operation can take in a stream, otherwise @c false
	 */
	virtual bool hasInlet() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether this unit operation possesses an outlet
	 * @return @c true if the unit operation can output a stream, otherwise @c false
	 */
	virtual bool hasOutlet() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of inlet ports
	 * @return Number of inlet ports
	 */
	virtual unsigned int numInletPorts() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of outlet ports
	 * @return Number of outlet ports
	 */
	virtual unsigned int numOutletPorts() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the local index of the first outlet component of the given port in the managed state vector slice
	 * @details As each unit operation manages a slice of the global state vector on its own, the
	 *          position of inlet and outlet components as DOF in the slice may differ. This function
	 *          returns the local index in the slice of the first outlet component of the given port.
	 * @param [in] port Index of the port
	 * @return Local index of the first outlet component in the state vector slice
	 */
	virtual unsigned int localOutletComponentIndex(unsigned int port) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the local index stride in the managed state vector slice
	 * @details This function returns the number of elements between two components of a given port in the slice of
	 *          the global state vector.
	 * @param [in] port Index of the port
	 * @return Local stride in the state vector slice
	 */
	virtual unsigned int localOutletComponentStride(unsigned int port) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the local index of the first inlet component of the given port in the managed state vector slice
	 * @details As each unit operation manages a slice of the global state vector on its own, the
	 *          position of inlet and outlet components as DOF in the slice may differ. This function
	 *          returns the local index in the slice of the first inlet component of the given port.
	 * @param [in] port Index of the port
	 * @return Local index of the first inlet component in the state vector slice
	 */
	virtual unsigned int localInletComponentIndex(unsigned int port) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the local index stride in the managed state vector slice
	 * @details This function returns the number of elements between two components of a given port in the slice of
	 *          the global state vector.
	 * @param [in] port Index of the port
	 * @return Local stride in the state vector slice
	 */
	virtual unsigned int localInletComponentStride(unsigned int port) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Evaluates the residuals with AD to compute the sensitivity derivatives
	 * @details Evaluates @f$ \frac{\partial F}{\partial p} @f$, where @f$ p @f$ are the sensitive parameters
	 *          and @f$ F @f$ is the residual function of this model.
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [in,out] adRes Pointer to local residual vector of AD datatypes for computing the sensitivity derivatives
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem) = 0;

	/**
	 * @brief Computes the residual of the forward sensitivity systems using the result of residualSensFwdAdOnly()
	 * @details Assembles and evaluates the residuals of the sensitivity systems
	 *          @f[ \frac{F}{\partial y} s + \frac{F}{\partial \dot{y}} \dot{s} + \frac{\partial F}{\partial p_i} = 0. @f]
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [in] yS Pointers to local sensitivity state vectors
	 * @param [in] ySdot Pointers to local sensitivity time derivative state vectors
	 * @param [out] resS Pointers to local sensitivity residuals
	 * @param [in] adRes Pointer to local residual vector of AD datatypes with the sensitivity derivatives from residualSensFwdAdOnly()
	 * @param [in] tmp1 Temporary storage in the size of local state vector @p y
	 * @param [in] tmp2 Temporary storage in the size of local state vector of @p y
	 * @param [in] tmp3 Temporary storage in the size of local state vector of @p y
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState, 
		const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS, active const* adRes, 
		double* const tmp1, double* const tmp2, double* const tmp3) = 0;

	/**
	 * @brief Evaluates the residuals with AD to compute the parameter sensitivities and at the same time updates the Jacobian
	 * @details Evaluates @f$ \frac{\partial F}{\partial p} @f$, where @f$ p @f$ are the sensitive parameters
	 *          and @f$ F @f$ is the residual function of this model. At the same time updates the Jacobian of the system.
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem) = 0;

	/**
	 * @brief Computes consistent initial values (state variables without their time derivatives)
	 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] the initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ have
	 *          to be consistent. This functions updates the initial state \f$ y_0 \f$ and, therefore, provides the
	 *          first step in consistent initialization.
	 *          
	 *          This function is to be used with consistentInitialTimeDerivative(). Do not mix normal and lean
	 *          consistent initialization!
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in,out] vecStateY State vector with initial values that are to be updated for consistency
	 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
	 * @param [in] errorTol Error tolerance for algebraic equations
	 */
	virtual void consistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem) = 0;

	/**
	 * @brief Computes consistent initial time derivatives
	 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] the initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ have
	 *          to be consistent. This functions overwrites the initial time derivatives \f$ \dot{y}_0 \f$ such that
	 *          the residual is zero. Thus, this function provides the final step in consistent initialization.
	 * 
	 *          This function is to be used with consistentInitialState(). Do not mix normal and lean
	 *          consistent initialization!
	 *          
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] vecStateY Consistently initialized state vector
	 * @param [in,out] vecStateYdot On entry, residual without taking time derivatives into account. On exit, consistent state time derivatives.
	 */
	virtual void consistentInitialTimeDerivative(const SimulationTime& simTime, double const* vecStateY, double* const vecStateYdot, util::ThreadLocalStorage& threadLocalMem) = 0;

	/**
	 * @brief Computes approximately / partially consistent initial values (state variables without their time derivatives)
	 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] the initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ have
	 *          to be consistent. This functions updates the initial state \f$ y_0 \f$ and, therefore, provides the
	 *          first step in consistent initialization.
	 *          
	 *          This function is possibly faster than consistentInitialState(), but updates only a part of the
	 *          state vector. Hence, the result is not guaranteed to be consistent.
	 * 
	 *          This function is to be used with leanConsistentInitialTimeDerivative(). Do not mix normal and lean
	 *          consistent initialization!
	 *
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in,out] vecStateY State vector with initial values that are to be updated for consistency
	 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
	 * @param [in] errorTol Error tolerance for algebraic equations
	 */
	virtual void leanConsistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem) = 0;

	/**
	 * @brief Computes approximately / partially consistent initial time derivatives
	 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] the initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ have
	 *          to be consistent. This functions overwrites the initial time derivatives \f$ \dot{y}_0 \f$ such that
	 *          the residual is zero. Thus, this function provides the final step in consistent initialization.
	 * 
	 *          This function is possibly faster than consistentInitialTimeDerivative(), but updates only a part of the
	 *          time derivative vector. Hence, the result is not guaranteed to be consistent.
	 * 
	 *          This function is to be used with leanConsistentInitialState(). Do not mix normal and lean
	 *          consistent initialization!
	 *          
	 * @param [in] t Current time point
	 * @param [in] vecStateY (Lean) consistent initial state
	 * @param [in,out] vecStateYdot On entry, inconsistent state time derivatives. On exit, partially consistent state time derivatives.
	 * @param [in] res On entry, residual without taking time derivatives into account. The data is overwritten during execution of the function.
	 */
	virtual void leanConsistentInitialTimeDerivative(double t, double const* const vecStateY, double* const vecStateYdot, double* const res, util::ThreadLocalStorage& threadLocalMem) = 0;

	/**
	 * @brief Computes consistent initial conditions for all sensitivity subsystems
	 * @details Given the DAE \f[ F(t, y, \dot{y}, p) = 0, \f] the corresponding (linear) forward sensitivity
	 *          system reads \f[ \frac{\partial F}{\partial y}(t, y, \dot{y}) s + \frac{\partial F}{\partial \dot{y}}(t, y, \dot{y}) \dot{s} + \frac{\partial F}{\partial p}(t, y, \dot{y}) = 0. \f]
	 *          The initial values of \f$ s_0 = \frac{\mathrm{d} y_0}{\mathrm{d}p} \f$ and \f$ \dot{s}_0 = \frac{\mathrm{d} \dot{y}_0}{\mathrm{d}p} \f$
	 *          have to be consistent, that means, they have to satisfy the sensitivity equation. This function computes the correct \f$ s_0 \f$ and \f$ \dot{s}_0 \f$
	 *          given \f$ y_0 \f$ and \f$ s_0 \f$.
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [in,out] vecSensY Sensitivity subsystem state vectors
	 * @param [in,out] vecSensYdot Time derivative state vectors of the sensitivity subsystems to be initialized
	 * @param [in] adRes Pointer to local residual vector of AD datatypes with parameter sensitivities
	 */
	virtual void consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
		std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem) = 0;

	/**
	 * @brief Computes approximately / partially consistent initial conditions for all sensitivity subsystems
	 * @details Given the DAE \f[ F(t, y, \dot{y}, p) = 0, \f] the corresponding (linear) forward sensitivity
	 *          system reads \f[ \frac{\partial F}{\partial y}(t, y, \dot{y}) s + \frac{\partial F}{\partial \dot{y}}(t, y, \dot{y}) \dot{s} + \frac{\partial F}{\partial p}(t, y, \dot{y}) = 0. \f]
	 *          The initial values of \f$ s_0 = \frac{\mathrm{d} y_0}{\mathrm{d}p} \f$ and \f$ \dot{s}_0 = \frac{\mathrm{d} \dot{y}_0}{\mathrm{d}p} \f$
	 *          have to be consistent, that means, they have to satisfy the sensitivity equation. This function computes the correct \f$ s_0 \f$ and \f$ \dot{s}_0 \f$
	 *          given \f$ y_0 \f$ and \f$ s_0 \f$.
	 *          
	 *          This function is possibly faster than consistentInitialSensitivity(), but updates only a part of the
	 *          vectors. Hence, the result is not guaranteed to be consistent. 
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [in,out] vecSensY Sensitivity subsystem state vectors
	 * @param [in,out] vecSensYdot Time derivative state vectors of the sensitivity subsystems to be initialized
	 * @param [in] adRes Pointer to local residual vector of AD datatypes with parameter sensitivities
	 */
	virtual void leanConsistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
		std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem) = 0;

	/**
	 * @brief Sets external functions for this binding model
	 * @details The external functions are not owned by this IBindingModel.
	 * @param [in] extFuns Pointer to array of IExternalFunction objects of size @p size
	 * @param [in] size Number of elements in the IExternalFunction array @p extFuns
	 */
	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size) = 0;

	/**
	 * @brief Sets the flow rates for the current time section
	 * @details The flow rates may change due to valve switches.
	 * @param [in] in Array with total volumetric inlet flow rate for each port
	 * @param [in] out Array with total volumetric outlet flow rate for each port
	 */
	virtual void setFlowRates(active const* in, active const* out) CADET_NOEXCEPT = 0;

	/**
	* @brief Returns whether this unit operation supports non-matching volumetric inlet and outlet flow rates
	* @details If inlet and outlet flow rates do not match, mass is accumulated or lost during time integration.
	* @return @c true if flow rates are allowed to differ, otherwise @c false
	*/
	virtual bool canAccumulate() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Multiplies the given vector with the system Jacobian (i.e., @f$ \frac{\partial F}{\partial y}\left(t, y, \dot{y}\right) @f$)
	 * @details Actually, the operation @f$ z = \alpha \frac{\partial F}{\partial y} x + \beta z @f$ is performed.
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [in] yS Vector @f$ x @f$ that is transformed by the Jacobian @f$ \frac{\partial F}{\partial y} @f$
	 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ \frac{\partial F}{\partial y} @f$
	 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ z @f$
	 * @param [in,out] ret Vector @f$ z @f$ which stores the result of the operation
	 */
	virtual void multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret) = 0;

	/**
	 * @brief Multiplies the time derivative Jacobian @f$ \frac{\partial F}{\partial \dot{y}}\left(t, y, \dot{y}\right) @f$ with a given vector
	 * @details The operation @f$ z = \frac{\partial F}{\partial \dot{y}} x @f$ is performed.
	 *          The matrix-vector multiplication is performed matrix-free (i.e., no matrix is explicitly formed).
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [in] sDot Vector @f$ x @f$ that is transformed by the Jacobian @f$ \frac{\partial F}{\partial \dot{y}} @f$
	 * @param [out] ret Vector @f$ z @f$ which stores the result of the operation
	 */
	virtual void multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret) = 0;

	/**
	 * @brief Returns the amount of thread local memory required by the unit operation
	 * @details The amount of thread local memory is returned in bytes.
	 * @return Required thread local memory size in bytes
	 */
	virtual unsigned int threadLocalMemorySize() const CADET_NOEXCEPT = 0;
};

} // namespace cadet

#endif  // LIBCADET_IUNITOPERATION_HPP_
