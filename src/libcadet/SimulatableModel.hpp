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
 * Defines an internal interface for models that can be simulated.
 */

#ifndef LIBCADET_SIMULATABLEMODEL_HPP_
#define LIBCADET_SIMULATABLEMODEL_HPP_

#include <vector>

#include "cadet/Model.hpp"
#include "AutoDiff.hpp"
#include "cadet/ModelSystem.hpp"

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

/**
 * @brief Defines a model that can be simulated
 */
class ISimulatableModel : public IModelSystem
{
public:

	virtual ~ISimulatableModel() CADET_NOEXCEPT { }

	/**
	 * @brief Return the number of required DOFs
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
	 * @brief (Re-)configures a unit operation model by extracting its non-structural parameters (e.g., model parameters) from the given @p paramProvider
	 * @details The scope of the cadet::IParameterProvider is left unchanged on return.
	 * @param [in] paramProvider Parameter provider
	 * @param [in] unitOpIdx ID of the unit operation model (creation order)
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configureModel(IParameterProvider& paramProvider, unsigned int unitOpIdx) = 0;

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
	 * @param [in] simState State of the simulation
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
	 * @brief Computes consistent initial values and time derivatives
	 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] the initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ have
	 *          to be consistent. This functions updates the initial state \f$ y_0 \f$ and overwrites the time
	 *          derivative \f$ \dot{y}_0 \f$ such that they are consistent.
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in,out] simState State of the simulation (state vector and its time derivatives) with initial values that are to be updated for consistency
	 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
	 * @param [in] errorTol Error tolerance for algebraic equations
	 */
	virtual void consistentInitialConditions(const SimulationTime& simTime, const SimulationState& simState, const AdJacobianParams& adJac, double errorTol) = 0;

	/**
	 * @brief Computes consistent initial conditions for all sensitivity subsystems
	 * @details Given the DAE \f[ F(t, y, \dot{y}, p) = 0, \f] the corresponding (linear) forward sensitivity
	 *          system reads 
	 *          \f[ \frac{\partial F}{\partial y}(t, y, \dot{y}) s + \frac{\partial F}{\partial \dot{y}}(t, y, \dot{y}) \dot{s} + \frac{\partial F}{\partial p}(t, y, \dot{y}) = 0. \f]
	 *          The initial values of \f$ s_0 = \frac{\mathrm{d} y_0}{\mathrm{d}p} \f$ and \f$ \dot{s}_0 = \frac{\mathrm{d} \dot{y}_0}{\mathrm{d}p} \f$
	 *          have to be consistent, that means, they have to satisfy the sensitivity equation. This function computes the correct \f$ s_0 \f$ and \f$ \dot{s}_0 \f$
	 *          given \f$ y_0 \f$ and \f$ s_0 \f$.
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [in,out] vecSensY Sensitivity subsystem state vectors
	 * @param [in,out] vecSensYdot Time derivative state vectors of the sensitivity subsystems to be initialized
	 * @param [in,out] adRes Pointer to global residual vector of AD datatypes for computing the parameter sensitivities
	 * @param [in,out] adY Pointer to global state vector of AD datatypes that can be used for computing the Jacobian (or @c nullptr if AD is disabled)
	 */
	virtual void consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
		std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active* const adRes, active* const adY) = 0;

	/**
	 * @brief Computes approximately / partially consistent initial values and time derivatives
	 * @details Given the DAE \f[ F(t, y, \dot{y}) = 0, \f] the initial values \f$ y_0 \f$ and \f$ \dot{y}_0 \f$ have
	 *          to be consistent. This functions updates parts of the initial state \f$ y_0 \f$ and overwrites parts of
	 *          the time derivative \f$ \dot{y}_0 \f$ such that they are approximately consistent.
	 *          
	 *          This function is possibly faster than consistentInitialConditions(), but updates only a part of the
	 *          state and time derivative vector. Hence, the result is not guaranteed to be consistent. 
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in,out] simState State of the simulation (state vector and its time derivatives) with initial values that are to be updated for consistency
	 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
	 * @param [in] errorTol Error tolerance for algebraic equations
	 */
	virtual void leanConsistentInitialConditions(const SimulationTime& simTime, const SimulationState& simState, const AdJacobianParams& adJac, double errorTol) = 0;

	/**
	 * @brief Computes approximately / partially consistent initial conditions for all sensitivity subsystems
	 * @details Given the DAE \f[ F(t, y, \dot{y}, p) = 0, \f] the corresponding (linear) forward sensitivity
	 *          system reads 
	 *          \f[ \frac{\partial F}{\partial y}(t, y, \dot{y}) s + \frac{\partial F}{\partial \dot{y}}(t, y, \dot{y}) \dot{s} + \frac{\partial F}{\partial p}(t, y, \dot{y}) = 0. \f]
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
	 * @param [in,out] adRes Pointer to global residual vector of AD datatypes for computing the parameter sensitivities
	 * @param [in,out] adY Pointer to global state vector of AD datatypes that can be used for computing the Jacobian (or @c nullptr if AD is disabled)
	 */
	virtual void leanConsistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
		std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active* const adRes, active* const adY) = 0;

	/**
	 * @brief Computes the residual
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [out] res Pointer to global residual vector
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res) = 0;

	/**
	 * @brief Computes the residual and updates the Jacobian
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [out] res Pointer to global residual vector
	 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac) = 0;

	/**
	 * @brief Computes the @f$ \ell^\infty@f$-norm of the residual vector
	 * 
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @return the @f$ \ell^\infty@f$-norm of the residual vector
	 */
	virtual double residualNorm(const SimulationTime& simTime, const ConstSimulationState& simState) = 0;

	/**
	 * @brief Computes the residual of the forward sensitivity systems
	 * 
	 * @param [in] nSens Number of sensitivity subsystems
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [out] res Pointer to global residual vector
	 * @param [in] yS Pointers to global sensitivity state vectors
	 * @param [in] ySdot Pointers to global sensitivity time derivative state vectors
	 * @param [out] resS Pointers to global sensitivity residuals
	 * @param [in,out] adRes Pointer to global residual vector of AD datatypes for computing the sensitivity derivatives
	 * @param [in] tmp1 Temporary storage in the size of global state vector @p y
	 * @param [in] tmp2 Temporary storage in the size of global state vector of @p y
	 * @param [in] tmp3 Temporary storage in the size of global state vector of @p y
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residualSensFwd(unsigned int nSens, const SimulationTime& simTime,
		const ConstSimulationState& simState, double const* const res,
		const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS,
		active* const adRes, double* const tmp1, double* const tmp2, double* const tmp3) = 0;

	/**
	 * @brief Computes the residual of the forward sensitivity systems and evaluates the Jacobian
	 * 
	 * @param [in] nSens Number of sensitivity subsystems
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [out] res Pointer to global residual vector
	 * @param [in] yS Pointers to global sensitivity state vectors
	 * @param [in] ySdot Pointers to global sensitivity time derivative state vectors
	 * @param [out] resS Pointers to global sensitivity residuals
	 * @param [in,out] adJac Jacobian information for AD (AD vectors for residual and state, direction offset)
	 * @param [in] tmp1 Temporary storage in the size of global state vector @p y
	 * @param [in] tmp2 Temporary storage in the size of global state vector of @p y
	 * @param [in] tmp3 Temporary storage in the size of global state vector of @p y
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residualSensFwdWithJacobian(unsigned int nSens, const SimulationTime& simTime,
		const ConstSimulationState& simState, double const* const res,
		const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS,
		const AdJacobianParams& adJac, double* const tmp1, double* const tmp2, double* const tmp3) = 0;

	/**
	 * @brief Computes the @f$ \ell^\infty@f$-norms of the forward sensitivity residual vectors
	 * 
	 * @param [in] nSens Number of sensitivity subsystems
	 * @param [in] simTime Simulation time information (time point, section index, pre-factor of time derivatives)
	 * @param [in] simState State of the simulation (state vector and its time derivative)
	 * @param [in] yS Pointers to global sensitivity state vectors
	 * @param [in] ySdot Pointers to global sensitivity time derivative state vectors
	 * @param [out] norms Pointer to array that holds the residual norms
	 * @param [in,out] adRes Pointer to global residual vector of AD datatypes for computing the sensitivity derivatives
	 * @param [in] tmp Temporary storage in the size of global state vector @p y
	 */
	virtual void residualSensFwdNorm(unsigned int nSens, const SimulationTime& simTime, 
			const ConstSimulationState& simState,
			const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, double* const norms,
			active* const adRes, double* const tmp) = 0;

	/**
	 * @brief Computes the solution of the linear system involving the system Jacobian
	 * @details The system \f[ \left( \frac{\partial F}{\partial y} + \alpha \frac{\partial F}{\partial \dot{y}} \right) x = b \f]
	 *          has to be solved. The right hand side \f$ b \f$ is given by @p rhs, the Jacobians are evaluated at the
	 *          point \f$(y, \dot{y})\f$ given by @p y and @p yDot. The residual @p res at this point, \f$ F(t, y, \dot{y}) \f$,
	 *          may help with this. Error weights (see IDAS guide) are given in @p weight. The solution is returned in @p rhs.
	 *          
	 *          Prior to calling linearSolve() the time integrator calls assembleDAEJacobian() with the same point
	 *          in time and state \f$(t, y, \dot{y})\f$.
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
	 * @details Sets the seed vectors used in AD. Since the AD vector is fully managed by the model,
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
	 * @brief Calculates error tolerances for additional coupling DOFs
	 * @details ModelSystem uses additional DOFs to decouple a system of unit operations for parallelization.
	 *          These additional DOFs don't get an error tolerance from the user because he shouldn't be
	 *          aware of those (implementation detail). This function is responsible for calculating error
	 *          tolerances for these additional coupling DOFs.
	 * 
	 * @param [in] errorTol Pointer to array of error tolerances for system without coupling DOFs
	 * @param [in] errorTolLength Length of @p errorTol array
	 * @return Vector with error tolerances for additional coupling DOFs
	 */
	virtual std::vector<double> calculateErrorTolsForAdditionalDofs(double const* errorTol, unsigned int errorTolLength) = 0;

	/**
	 * @brief Performs setup of parallelization for the given number of threads
	 * @details This function is called upon the beginning of the time integration process.
	 * @param [in] numThreads Number of threads
	 */
	virtual void setupParallelization(unsigned int numThreads) = 0;

protected:
};

} // namespace cadet

#endif  // LIBCADET_SIMULATABLEMODEL_HPP_
