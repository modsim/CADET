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
 * Defines the Simulator interface.
 */

#ifndef LIBCADET_SIMULATOR_HPP_
#define LIBCADET_SIMULATOR_HPP_

#include <string>
#include <vector>
#include <unordered_map>
#include <type_traits>

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"
#include "cadet/ParameterId.hpp"

namespace cadet
{

class IModelSystem;
class ISolutionRecorder;
class IParameterProvider;
class INotificationCallback;

enum class ConsistentInitialization : int
{
	/**
	 * @brief Do not use consistent initialization at all
	 */
	None = 0,
	/**
	 * @brief Perform full consistent initialization at the beginning and each discontinuous section transition
	 */
	Full = 1,
	/**
	 * @brief Perform full consistent initialization once at the beginning
	 */
	FullFirstOnly = 2,
	/**
	 * @brief Perform lean consistent initialization at the beginning and each discontinuous section transition
	 */
	Lean = 3,
	/**
	 * @brief Perform lean consistent initialization once at the beginning
	 */
	LeanFirstOnly = 4,
	/**
	 * @brief Perform full consistent initialization at the beginning and lean initialization on every following
	 * discontinuous section transition
	 */
	FullOnceThenLean = 5,
	/**
	 * @brief Do not initialize at the beginning but perform full initialization on every following discontinuous
	 * section transition
	 */
	NoneOnceThenFull = 6,
	/**
	 * @brief Do not initialize at the beginning but perform lean initialization on every following discontinuous
	 * section transition
	 */
	NoneOnceThenLean = 7,
};

/**
 * @brief Converts a ConsistentInitialization to a string
 * @param [in] ci ConsistentInitialization to be converted
 * @return String representation of the ConsistentInitialization
 */
inline const char* to_string(ConsistentInitialization ci) CADET_NOEXCEPT
{
	switch (ci)
	{
	case ConsistentInitialization::None:
		return "None";
	case ConsistentInitialization::FullFirstOnly:
		return "FullFirstOnly";
	case ConsistentInitialization::LeanFirstOnly:
		return "LeanFirstOnly";
	case ConsistentInitialization::Full:
		return "Full";
	case ConsistentInitialization::Lean:
		return "Lean";
	case ConsistentInitialization::FullOnceThenLean:
		return "FullOnceThenLean";
	case ConsistentInitialization::NoneOnceThenFull:
		return "NoneOnceThenFull";
	case ConsistentInitialization::NoneOnceThenLean:
		return "NoneOnceThenLean";
	}
	return "Unknown";
}

/**
 * @brief Converts a string to a ConsistentInitialization
 * @param [in] ci ConsistentInitialization as string
 * @return ConsistentInitialization corresponding to the given string
 */
inline ConsistentInitialization to_consistentinitialization(const std::string& ci) CADET_NOEXCEPT
{
	if (ci == "None")
		return ConsistentInitialization::None;
	else if (ci == "FullFirstOnly")
		return ConsistentInitialization::FullFirstOnly;
	else if (ci == "LeanFirstOnly")
		return ConsistentInitialization::LeanFirstOnly;
	else if (ci == "Full")
		return ConsistentInitialization::Full;
	else if (ci == "Lean")
		return ConsistentInitialization::Lean;
	else if (ci == "FullOnceThenLean")
		return ConsistentInitialization::FullOnceThenLean;
	else if (ci == "NoneOnceThenFull")
		return ConsistentInitialization::NoneOnceThenFull;
	else if (ci == "NoneOnceThenLean")
		return ConsistentInitialization::NoneOnceThenLean;

	return ConsistentInitialization::Full;
}

/**
 * @brief Converts an integer to the ConsistentInitialization enum
 * @param [in] ci Integer representing an ConsistentInitialization mode
 * @return ConsistentInitialization
 */
inline ConsistentInitialization toConsistentInitialization(int ci) CADET_NOEXCEPT
{
	switch (ci)
	{
	case static_cast<typename std::underlying_type<ConsistentInitialization>::type>(ConsistentInitialization::None):
		return ConsistentInitialization::None;
	case static_cast<typename std::underlying_type<ConsistentInitialization>::type>(ConsistentInitialization::Full):
		return ConsistentInitialization::Full;
	case static_cast<typename std::underlying_type<ConsistentInitialization>::type>(
		ConsistentInitialization::FullFirstOnly):
		return ConsistentInitialization::FullFirstOnly;
	case static_cast<typename std::underlying_type<ConsistentInitialization>::type>(ConsistentInitialization::Lean):
		return ConsistentInitialization::Lean;
	case static_cast<typename std::underlying_type<ConsistentInitialization>::type>(
		ConsistentInitialization::LeanFirstOnly):
		return ConsistentInitialization::LeanFirstOnly;
	case static_cast<typename std::underlying_type<ConsistentInitialization>::type>(
		ConsistentInitialization::FullOnceThenLean):
		return ConsistentInitialization::FullOnceThenLean;
	case static_cast<typename std::underlying_type<ConsistentInitialization>::type>(
		ConsistentInitialization::NoneOnceThenFull):
		return ConsistentInitialization::NoneOnceThenFull;
	case static_cast<typename std::underlying_type<ConsistentInitialization>::type>(
		ConsistentInitialization::NoneOnceThenLean):
		return ConsistentInitialization::NoneOnceThenLean;
	}
	return ConsistentInitialization::Full;
}

/**
 * @brief Checks whether the given argument refers to a valid ConsistentInitialization mode
 * @param [in] ci Candidate to be checked
 * @return @c true if the candidate is a valid ConsistentInitialization mode, otherwise @c false
 */
inline bool isValidConsistentInitialization(int ci) CADET_NOEXCEPT
{
	return (ci >= static_cast<typename std::underlying_type<ConsistentInitialization>::type>(
					  ConsistentInitialization::None)) &&
		   (ci <= static_cast<typename std::underlying_type<ConsistentInitialization>::type>(
					  ConsistentInitialization::NoneOnceThenLean));
}

/**
 * @brief Provides functionality to simulate a model using a time integrator
 */
class CADET_API ISimulator
{
public:
	virtual ~ISimulator() CADET_NOEXCEPT
	{
	}

	//! \brief Sets a parameter sensitive
	//!
	//! The method enables sensitivity computation for the given parameter.
	//! For each parameter marked as sensitive, another system in size of the original DAE system
	//! is to be solved.
	//!
	//! \param  [in]    id Parameter ID of the sensitive parameter
	virtual void setSensitiveParameter(const ParameterId& id) = 0;

	//! \brief Sets a parameter sensitive
	//!
	//! The method enables sensitivity computation for the given parameter.
	//! For each parameter marked as sensitive, another system in size of the original DAE system
	//! is to be solved.
	//!
	//! \param  [in]    id Parameter ID of the sensitive parameter
	//! \param  [in]    absTolS Absolute tolerance used in the sensitivity computation for this parameter (try 1.0e-5)
	virtual void setSensitiveParameter(const ParameterId& id, double absTolS) = 0;

	//! \brief Fuses multiple parameters to one sensitivity
	//!
	//! Multiple parameters are fused into a single parameter sensitivity.
	//! Given two parameters @f$ p_1, p_2 @f$, this is equivalent to setting
	//! the constraint @f$ p_1 = p_2 = p @f$ and computing the parameter sensitivity
	//! of any of the two parameters:
	//! @f[ f(p_1, p_2) = f(p, p) \quad \Rightarrow \quad \frac{df}{dp} = \partial_1 f + \partial_2 f. @f]
	//! One additional system in size of the original DAE system is to be solved regardless
	//! of the number of fused parameters.
	//!
	//! \param  [in]    ids Parameter IDs of the fused sensitive parameters
	//! \param  [in]    numParams Number of fused parameters
	//! \param  [in]    absTolS Absolute tolerance used in the sensitivity computation for this parameter (try 1.0e-5)
	virtual void setSensitiveParameter(ParameterId const* ids, unsigned int numParams, double absTolS) = 0;

	//! \brief Fuses multiple parameters to one sensitivity
	//!
	//! Multiple parameters are fused into a single parameter sensitivity that is a
	//! linear combination of the single parameters. Given two parameters @f$ p_1, p_2 @f$, this means setting
	//! @f[ \begin{pmatrix} p_1 \\ p_2 \end{pmatrix} = g\left(p\right) = \begin{pmatrix} a_1 p \\ a_2 p \end{pmatrix}
	//! @f] and computing the parameter sensitivity of @f$ p@f$:
	//! @f[ \frac{d}{dp} f(g(p)) = a_1 \partial_1 f + a_2 \partial_2 f. @f]
	//! One additional system in size of the original DAE system is to be solved.
	//!
	//! \param  [in]    ids Parameter IDs of the fused sensitive parameters
	//! \param  [in]    diffFactors Factors @f$ a_i @f$ of the linear combination
	//! \param  [in]    numParams Number of fused parameters
	//! \param  [in]    absTolS Absolute tolerance used in the sensitivity computation for this parameter (try 1.0e-5)
	virtual void setSensitiveParameter(ParameterId const* ids, double const* diffFactors, unsigned int numParams,
									   double absTolS) = 0;

	//! \brief Reset the simulator to compute no sensitivity at all
	virtual void clearSensParams() = 0;

	//! \brief Returns the number of sensitive parameters
	//! All parameters that are set sensitive are counted. This includes model parameters as well as inlet parameters.
	//! \return The number of sensitive parameters
	virtual unsigned int numSensParams() const CADET_NOEXCEPT = 0;

	//! \brief Sets the value of a sensitive parameter
	//! If the parameter is part of a fused sensitivity, then all fused parameters are set
	//! to the same value (honouring their linear coefficients).
	//! \param  [in]    id Parameter ID of the parameter to be manipulated
	//! \param  [in]    value Value of the parameter
	virtual void setSensitiveParameterValue(const ParameterId& id, double value) = 0;

	//! \brief Sets the value of a sensitive parameter
	//! If the parameter is part of a fused sensitivity, then all fused parameters are set
	//! to the same value (honouring their linear coefficients).
	//! \param  [in]    idx Index of the sensitive parameter to be manipulated
	//! \param  [in]    value Value of the parameter
	virtual void setSensitiveParameterValue(unsigned int idx, double value) = 0;

	//! \brief Sets the linear factors of a sensitive parameter group
	//! \param  [in]    idx Index of the sensitive parameter group to be manipulated
	//! \param  [in]    factors Array with linear factors of the individual parameters in the group
	virtual void setSensitiveParameterFactors(unsigned int idx, double const* factors) = 0;

	//! \brief Sets the value of a parameter
	//! This function does not respect fused parameters and treats them as individuals.
	//! \param  [in]    id Parameter ID of the parameter to be manipulated
	//! \param  [in]    value Value of the parameter
	virtual void setParameterValue(const ParameterId& id, double value) = 0;

	/**
	 * @brief Checks whether a given parameter exists
	 * @param [in] pId   pId   ParameterId that identifies the parameter uniquely
	 * @return @c true if the parameter exists, otherwise @c false
	 */
	virtual bool hasParameter(const ParameterId& pId) const = 0;

	/**
	 * @brief Returns all parameters with their current values that can be made sensitive
	 * @return Map with all parameters that can be made sensitive along with their current value
	 */
	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const = 0;

	//! \brief Sets the time points at which a solution is written
	//!
	//! The simulator returns solutions by default only at its internal integration timepoints.
	//! This timepoints are usually unevenly spaced and are more dense in regions where
	//! gradients are steep. If the user is interested in solutions at specific timepoints
	//! he can set them using this method. The points may lie in the range \f$[t_{\text{start}}, t_{\text{end}}]\f$.
	//! Note that the timepoints should not exceed the section times.
	//!
	//! \param  [in]    solutionTimes  A vector containing the timepoints at which a solution should be computed
	virtual void setSolutionTimes(const std::vector<double>& solutionTimes) = 0;

	//! \brief Returns the time points at which a solution has been written
	//!
	//! \return Vector containing the timepoints at which a solution has been be computed
	virtual const std::vector<double>& getSolutionTimes() const = 0;

	//! \brief Sets the timepoints of (potential) discontinuities
	//!
	//! Sets the timepoints of (potential) discontinuities where an integrator
	//! restart with adjusted initial conditions might be required.
	//! This applies to all timepoints at which the inlet concentration function is not
	//! continuously differentiable. Hereby the simulation is split into sections which are solved
	//! independently. The \p sectionTimes vector must contain at least \f$t_{\text{start}}\f$
	//! and \f$t_{\text{end}}\f$ and is of length <tt>nsec + 1</tt>.
	//!
	//! \param  [in]    sectionTimes    A vector containing the timepoints of all sections
	virtual void setSectionTimes(const std::vector<double>& sectionTimes) = 0;

	//! \brief Sets the timepoints of (potential) discontinuities
	//!
	//! Sets the timepoints of (potential) discontinuities where an integrator
	//! restart with adjusted initial conditions might be required.
	//! This applies to all timepoints at which the inlet concentration function is not
	//! continuously differentiable. Hereby the simulation is split into sections which are solved
	//! independently. The \p sectionTimes vector must contain at least \f$t_{start}\f$
	//! and \f$t_{end}\f$ and is of length <tt>nsec + 1</tt>.
	//!
	//! Also sets the continuity of section transitions. A continuous section transition from section i
	//! to section i+1 is indicated by setting the ith entry of \p sectionContinuity to <tt>true</tt>.
	//! The integrator will not be reset on continuous section transitions, thus increasing performance.
	//!
	//! \param  [in]    sectionTimes        A vector containing the timepoints of all sections
	//! \param  [in]    sectionContinuity   A vector determining the continuity of section transitions
	virtual void setSectionTimes(const std::vector<double>& sectionTimes,
								 const std::vector<bool>& sectionContinuity) = 0;

	/**
	 * @brief Initializes the time integrator with the given model system
	 * @details Allocates internal memory and initializes and sets up the IDAS package.
	 *
	 * @param [in] model Model system that is to be simulated
	 */
	virtual void initializeModel(IModelSystem& model) = 0;

	/**
	 * @brief Applies the initial conditions that are currently set
	 * @details Initial conditions are set by calling setInitialCondition(IParameterProvider& paramProvider).
	 *          This function has to be called once for applyInitialCondition() to work, which will do nothing
	 *          otherwise.
	 *
	 *          This function applies the initial conditions to the state vectors. Consistent initialization
	 *          will be performed later according to the mode set by setConsistentInitialization().
	 */
	virtual void applyInitialCondition() = 0;

	/**
	 * @brief Reads initial condition of the model from the given parameter provider
	 * @details The scope of the cadet::IParameterProvider is unchanged on return.
	 *          Note that the initial conditions are not applied, which is performed by applyInitialCondition().
	 *
	 * @param [in] paramProvider Parameter provider
	 */
	virtual void setInitialCondition(IParameterProvider& paramProvider) = 0;

	/**
	 * @brief Applies the given initial state to the model
	 * @details The initial state consists of the state @f$ y_0 @f$ and its time derivative @f$\dot{y}_0@f$.
	 *          It has to be consistent, which means that
	 *          @f[ f\left( t_0, y_0, \dot{y}_0 ) = 0 @f]
	 *          has to hold. In other words, the initial condition has to fulfill the DAE at the initial
	 *          time point @f$ t_0@f$.
	 *
	 *          If only @f$ y_0 @f$ is given, the simulator will try to find a matching @f$ \dot{y}_0@f$ later.
	 *          This is done according to the consistent initialization mode set by setConsistentInitialization().
	 *          Because of nonlinearities, it is usually faster to also provide @f$ \dot{y}_0@f$.
	 *
	 * @param [in] initState Initial state @f$ y_0 @f$ of the model
	 */
	virtual void applyInitialCondition(double const* const initState) = 0;

	/**
	 * @brief Applies the given initial state to the model
	 * @details The initial state consists of the state @f$ y_0 @f$ and its time derivative @f$\dot{y}_0@f$.
	 *          It has to be consistent, which means that
	 *          @f[ f\left( t_0, y_0, \dot{y}_0 ) = 0 @f]
	 *          has to hold. In other words, the initial condition has to fulfill the DAE at the initial
	 *          time point @f$ t_0@f$.
	 *
	 *          If only @f$ y_0 @f$ is given, the simulator will try to find a matching @f$ \dot{y}_0@f$ later.
	 *          This is done according to the consistent initialization mode set by setConsistentInitialization().
	 *          Because of nonlinearities, it is usually faster to also provide @f$ \dot{y}_0@f$.
	 *
	 * @param [in] initState Initial state @f$ y_0 @f$ of the model
	 * @param [in] initStateDot Initial time derivative state @f$ \dot{y}_0 @f$ of the model
	 */
	virtual void applyInitialCondition(double const* const initState, double const* const initStateDot) = 0;

	/**
	 * @brief Applies the given initial state to the forward sensitivity systems
	 * @details The initial sensitivities are given by the argument @p initSens and their time derivatives
	 *          by @p initSensDot. The model is not invoked to make changes to the initial values.
	 *
	 *          If one of the parameters (@p initSens or @p initSensDot) is @c nullptr, the state vector
	 *          and its time derivative will be set to zero and consistent values will be computed later.
	 *
	 *          Consistent initialization is performed later according to the mode set by
	 * setConsistentInitializationSens().
	 *
	 * @param [in] initSens Pointer to array with pointers to initial sensitivities or @c nullptr
	 * @param [in] initSensDot Pointer to array with pointers to initial sensitivity time derivatives or @c nullptr
	 */
	virtual void applyInitialConditionFwdSensitivities(double const* const* const initSens,
													   double const* const* const initSensDot) = 0;

	/**
	 * @brief Initializes the forward sensitivity subsystems with zero initial values
	 * @details The initial sensitivities are set to zero and (later) handed over to the model for consistent
	 * initialization.
	 */
	virtual void initializeFwdSensitivities() = 0;

	/**
	 * @brief Skips consistent initialization at the beginning of integrate()
	 * @details By skipping consistent initialization at the beginning of integrate(), a previous
	 *          simulation can be resumed.
	 *
	 *          More detailed options can be applied with setConsistentInitialization().
	 */
	virtual void skipConsistentInitialization() = 0;

	/**
	 * @brief Sets consistent initialization mode
	 * @details Sets consistent initialization mode for the subsequent calls to integrate().
	 *          Use skipConsistentInitialization() to skip the first consistent initialization.
	 *
	 * @param [in] ci Consistent initialization mode
	 */
	virtual void setConsistentInitialization(ConsistentInitialization ci) = 0;

	/**
	 * @brief Sets consistent initialization mode of the sensitivity systems
	 * @details Sets consistent initialization mode for the subsequent calls to integrate().
	 *          Use skipConsistentInitialization() to skip the first consistent initialization.
	 *
	 * @param [in] ci Consistent initialization mode of the sensitivity systems
	 */
	virtual void setConsistentInitializationSens(ConsistentInitialization ci) = 0;

	/**
	 * @brief Initializes the forward sensitivity subsystems with given initial values
	 * @details The initial sensitivities are given by the argument @p initSens and their time derivatives
	 *          by @p initSensDot. The user is responsible for checking whether the initial values are
	 *          correct and consistent with the sensitive parameters. The model is not invoked to make
	 *          changes to the initial values.
	 *
	 * @param [in] initSens Pointer to array with pointers to initial sensitivities
	 * @param [in] initSensDot Pointer to array with pointers to initial sensitivity time derivatives
	 */
	virtual void initializeFwdSensitivities(double const* const* const initSens,
											double const* const* const initSensDot) = 0;

	/**
	 * @brief Sets the solution recorder which receives the solution of each time step
	 * @details The solution recorder is invoked for every time step when a solution and its sensitivities
	 *          are to be saved. Setting the recorder to @c NULL disables recording the solution.
	 *
	 * @param [in] recorder Implementation of the cadet::ISolutionRecorder interface
	 */
	virtual void setSolutionRecorder(ISolutionRecorder* recorder) = 0;

	/**
	 * @brief Starts the solution of the system specified for this simulator object
	 * @details Checks all model parameters to lie inside their possible bounds and then runs the time integration
	 *          from \f$t_{start}\f$ to \f$t_{end}\f$, restarting the time integrator at every section time specified
	 *          by #setSectionTimes. Solutions are stored by default at internal timesteps of the integration routine,
	 *          or if a call to #setSolutionTimes was made, at the specified user defined solution timepoints.
	 */
	virtual void integrate() = 0;

	/**
	 * @brief Returns the bare state vector for the last timepoint
	 * @details The method returns the last solution as it was written to the memory.
	 *          The whole state vetor (including column, particle and flux parts) is returned.
	 *          The ordering is rather unhandy, since the column, the particle, and the flux parts
	 *          are of different size.
	 *
	 * @param [out] len Length of the state vector
	 * @return Pointer to the first element of the state vector
	 */
	virtual double const* getLastSolution(unsigned int& len) const = 0;

	/**
	 * @brief Returns the bare time derivative state vector for the last timepoint
	 * @details See getLastSolution().
	 *
	 * @param [out] len Length of the state vector
	 * @return Pointer to first element of the state vector
	 */
	virtual double const* getLastSolutionDerivative(unsigned int& len) const = 0;

	/**
	 * @brief Returns the bare sensitivity state vectors for the last timepoint
	 * @details The method returns the last solution of the sensitivity systems as it was written to memory.
	 *          The whole state vetor (including column, particle and flux parts) is returned.
	 *          The ordering is rather unhandy, since the column, the particle and the flux parts
	 *          are of different size.
	 *
	 * @param [out] len Length of the sensitivity state vector
	 * @return Array with pointers to sensitivity state vectors
	 */
	virtual std::vector<double const*> getLastSensitivities(unsigned int& len) const = 0;

	/**
	 * @brief Returns the bare time derivative state vectors of the sensitivity subsystems for the last timepoint
	 * @details See getLastSolutionDerivative().
	 *
	 * @param [out] len Length of the sensitivity state vector
	 * @return Array with pointers to sensitivity state vectors
	 */
	virtual std::vector<double const*> getLastSensitivityDerivatives(unsigned int& len) const = 0;

	/**
	 * @brief Returns the simulated model
	 * @return Simulated model or @c NULL
	 */
	virtual IModelSystem* model() CADET_NOEXCEPT = 0;
	virtual IModelSystem const* model() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Return the number of DOFs in the current simulation
	 * @return The number of DOFs
	 */
	virtual unsigned int numDofs() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Reads all configuration values from the given parameter provider
	 * @details The scope of the cadet::IParameterProvider is unchanged on return.
	 *
	 * @param [in] paramProvider Parameter provider
	 */
	virtual void configure(IParameterProvider& paramProvider) = 0;

	/**
	 * @brief Rereads all configuration values from the given parameter provider
	 * @details The scope of the cadet::IParameterProvider is unchanged on return.
	 *
	 * @param [in] paramProvider Parameter provider
	 */
	virtual void reconfigure(IParameterProvider& paramProvider) = 0;

	/**
	 * @brief Rereads all configuration values of the current model from the given parameter provider
	 * @details The scope of the cadet::IParameterProvider is unchanged on return.
	 *          This function only reconfigures the current model without recursing
	 *          into submodels.
	 * @param [in] paramProvider Parameter provider
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool reconfigureModel(IParameterProvider& paramProvider) = 0;

	/**
	 * @brief Rereads all configuration values of the current model from the given parameter provider
	 * @details The scope of the cadet::IParameterProvider is unchanged on return.
	 *          This function only reconfigures the unit operation model identified
	 *          by its unit operation ID without recursing into submodels.
	 * @param [in] paramProvider Parameter provider
	 * @param [in] unitOpIdx Unit operation index of the model to be reconfigured
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool reconfigureModel(IParameterProvider& paramProvider, unsigned int unitOpIdx) = 0;

	/**
	 * @brief Configures the time integrator
	 * @details Alternative to configure() and reconfigure().
	 * @param [in] relTol Relative error tolerance
	 * @param [in] absTol Absolute error tolerance for each DOF
	 * @param [in] initStepSize Initial step size for each section
	 * @param [in] maxSteps Maximum number of time integration steps
	 * @param [in] maxStepSize Maximum step size of the time integrator
	 */
	virtual void configureTimeIntegrator(double relTol, double absTol, double initStepSize, unsigned int maxSteps,
										 double maxStepSize) = 0;
	virtual void configureTimeIntegrator(double relTol, double absTol, const std::vector<double>& initStepSizes,
										 unsigned int maxSteps, double maxStepSize) = 0;

	/**
	 * @brief Sets the error tolerances of the sensitivity systems
	 * @details One absolute error tolerance per sensitivity system is required.
	 * @param [in] relTol Relative error tolerance
	 * @param [in] absTol Vector with absolute error tolerance for each sensitivity system
	 */
	virtual void setSensitivityErrorTolerance(double relTol, double const* absTol) = 0;

	/**
	 * @brief Sets the number of threads for the simulation
	 * @details If the number of threads @p nThreads is @c 0, the maximum number of threads is used.
	 * @param [in] nThreads Number of threads
	 */
	virtual void setNumThreads(unsigned int nThreads) CADET_NOEXCEPT = 0;

	/**
	 * @brief Sets the relative error tolerance of the time integrator
	 * @details This tolerance is used for all elements of the state vector.
	 *          Each element @f$ r_i @f$ in the residual is weighted by
	 *          @f[ \frac{1}{\text{relTol} \left|r_i\right| + \text{absTol}_i}. @f]
	 * @param [in] relTol Relative error tolerance
	 */
	virtual void setRelativeErrorTolerance(double relTol) = 0;

	/**
	 * @brief Sets the absolute error tolerance of the time integrator
	 * @details This tolerance is used for all elements of the state vector.
	 *          Each element @f$ r_i @f$ in the residual is weighted by
	 *          @f[ \frac{1}{\text{relTol} \left|r_i\right| + \text{absTol}_i}, @f]
	 *          which means that @f$ \text{absTol}_i @f$ does not depend on the
	 *          index @f$ i @f$.
	 * @param [in] absTol Absolute error tolerance
	 */
	virtual void setAbsoluteErrorTolerance(double absTol) = 0;

	/**
	 * @brief Sets the absolute error tolerance of the time integrator
	 * @details Each element @f$ r_i @f$ in the residual is weighted by
	 *          @f[ \frac{1}{\text{relTol} \left|r_i\right| + \text{absTol}_i}. @f]
	 * @param [in] absTol Vector with absolute error tolerances for each component of the state vector
	 */
	virtual void setAbsoluteErrorTolerance(const std::vector<double>& absTol) = 0;

	/**
	 * @brief Sets the error tolerance of the nonlinear algebraic equation solvers
	 * @details This tolerance is used for consistent initialization of the nonlinear
	 *          algebraic equations.
	 * @param [in] algTol Algebraic error tolerance
	 */
	virtual void setAlgebraicErrorTolerance(double algTol) CADET_NOEXCEPT = 0;

	/**
	 * @brief Sets the initial step size of the time integrator for each section
	 * @details Uses the same initial time step size for all sections.
	 * @param [in] stepSize Size of the first attempted time step in each section
	 */
	virtual void setInitialStepSize(double stepSize) = 0;

	/**
	 * @brief Sets the initial step size of the time integrator for each section
	 * @details Uses a different initial time step size for each section.
	 * @param [in] stepSize Vector of time step sizes for the first attempted step in each section
	 */
	virtual void setInitialStepSize(const std::vector<double>& stepSize) = 0;

	/**
	 * @brief Sets the maximum number of time steps in each section
	 * @details If an integration process requires more than the specified number of time steps,
	 *          the time integrator aborts with an exception.
	 * @param [in] maxSteps Maximum number of time steps in each section
	 */
	virtual void setMaximumSteps(unsigned int maxSteps) = 0;

	/**
	 * @brief Sets the maximum time step size
	 * @details Set to @c 0.0 to not impose any step size restriction.
	 * @param [in] maxStepSize Maximum size of time steps
	 */
	virtual void setMaximumStepSize(double maxStepSize) = 0;

	/**
	 * @brief Sets the relative error tolerance for sensitivity systems in the time integrator
	 * @details This tolerance is used for all elements of the sensitivity state vectors
	 *          in all sensitivity systems. Each element @f$ r_i @f$ in the residual is
	 *          weighted by @f[ \frac{1}{\text{relTol} \left|r_i\right| + \text{absTol}_i}. @f]
	 * @param [in] relTol Relative error tolerance
	 */
	virtual void setRelativeErrorToleranceSens(double relTol) = 0;

	/**
	 * @brief Controls whether forward sensitivity systems are taken into account for the local error test in time
	 * integration
	 * @details The error in time integration can be estimated and is used for step size
	 *          and order selection. This function controls whether forward sensitivity
	 *          systems are taken into account in the local error test, which they are
	 *          by default (@c true).
	 *
	 * @param [in] enabled Determines whether sensitivities are taken into account in local error test
	 */
	virtual void setSensitivityErrorControl(bool enabled) = 0;

	/**
	 * @brief Sets the maximum number of Newton iterations for a time integration step
	 * @details Each time step requires the solution of a nonlinear equation system which
	 *          is performed by Newton iteration. Due to a good starting point (the last
	 *          time step's solution is supposed to be very close to the current step's
	 *          solution) a maximum of 3 iterations is used by default (see SUNDIALS 2.7.0).
	 *          The iteration is stopped when the specified number of iterations is
	 *          exceeded. A final convergence test is performed.
	 *
	 * @param [in] nIter Maximum number of Newton iterations for a time step
	 */
	virtual void setMaxNewtonIteration(unsigned int nIter) = 0;

	/**
	 * @brief Sets the maximum number of (local truncation) error test failures for a time integration step
	 * @details The error in time integration can be estimated and is used for step size
	 *          and order selection. The maximum number of failed local time step error
	 *          tests defaults to 7 (see SUNDIALS 2.7.0). Time integration fails when
	 *          the number of failures exceeds the specified value.
	 *
	 * @param [in] nFails Maximum number of local error test failures for a time step
	 */
	virtual void setMaxErrorTestFails(unsigned int nFails) = 0;

	/**
	 * @brief Sets the maximum number of convergence test failures in the Newton iteration of a time step
	 * @details Each time step requires the solution of a nonlinear equation system which
	 *          is performed by Newton iteration. If the convergence check of the iteration
	 *          fails, the step size is reduced and a new iteration is started. The default
	 *          for the number of failed convergence tests (i.e., Newton iterations) is set
	 *          to 10 (see SUNDIALS 2.7.0). When the specified number of failures is exceeded,
	 *          time integration fails.
	 *
	 * @param [in] nFails Maximum number of convergence test failures
	 */
	virtual void setMaxConvergenceFails(unsigned int nFails) = 0;

	/**
	 * @brief Sets the maximum number of Newton iterations for each forward sensitivity system in a time integration
	 * step
	 * @details Each time step requires the solution of a nonlinear equation system which
	 *          is performed by Newton iteration. After the original system has been solved,
	 *          each forward sensitivity system is solved using a separate Newton iteration.
	 *          The maximum number of iterations is controlled by this function and defaults
	 *          to 3 (see SUNDIALS 2.7.0). After the iteration a convergence test is performed.
	 *
	 * @param [in] nIter Maximum number of Newton iterations for a time step
	 */
	virtual void setMaxSensNewtonIteration(unsigned int nIter) = 0;

	/**
	 * @brief Returns the elapsed time of the last simulation run in seconds
	 * @return Elapsed time the last call of integrate() took in seconds
	 */
	virtual double lastSimulationDuration() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the accumulated time of all simulation runs in seconds
	 * @return Accumulated time of all calls of integrate() in seconds
	 */
	virtual double totalSimulationDuration() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Sets the receiver for notifications
	 * @param[in] nc Object to receive notifications or @c nullptr to disable notifications
	 */
	virtual void setNotificationCallback(INotificationCallback* nc) CADET_NOEXCEPT = 0;
};

} // namespace cadet

#endif // LIBCADET_SIMULATOR_HPP_
