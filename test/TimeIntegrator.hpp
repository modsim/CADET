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
 * ODE/DAE solver for tests
 */

#ifndef CADETTEST_TIMEINTEGRATOR_SKEL_HPP_
#define CADETTEST_TIMEINTEGRATOR_SKEL_HPP_

#include <vector>
#include <unordered_map>

#include "SundialsVector.hpp"
#include <idas/idas_impl.h>

#include "cadet/cadetCompilerInfo.hpp"

namespace cadet
{

namespace test
{

class IDiffEqModel
{
public:
	virtual ~IDiffEqModel() CADET_NOEXCEPT { }

	/**
	 * @brief Return the number of required DOFs
	 * @return The number of required DOFs
	 */
	virtual int numDofs() const CADET_NOEXCEPT = 0;

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
	 * @param [in,out] vecStateY State of the simulation
	 * @param [in,out] vecStateYdot Time derivative of simulation state
	 */
	virtual void notifyDiscontinuousSectionTransition(double t, int secIdx, double* vecStateY, double* vecStateYdot) = 0;

	/**
	 * @brief Computes the residual
	 * 
	 * @param [in] time Simulation time
	 * @param [in] secIdx Current time section
	 * @param [in] vecStateY State of the simulation
	 * @param [in] vecStateYdot Time derivative of simulation state
	 * @param [out] res Pointer to global residual vector
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residual(double time, int secIdx, double const* vecStateY, double const* vecStateYdot, double* res) = 0;

	/**
	 * @brief Computes the residual and updates the Jacobian
	 * 
	 * @param [in] time Simulation time
	 * @param [in] secIdx Current time section
	 * @param [in] vecStateY State of the simulation
	 * @param [in] vecStateYdot Time derivative of simulation state
	 * @param [out] res Pointer to global residual vector
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int residualWithJacobian(double time, int secIdx, double const* vecStateY, double const* vecStateYdot, double* res) = 0;

	/**
	 * @brief Computes the @f$ \ell^\infty@f$-norm of the residual vector
	 * 
	 * @param [in] time Simulation time
	 * @param [in] secIdx Current time section
	 * @param [in] vecStateY State of the simulation
	 * @param [in] vecStateYdot Time derivative of simulation state
	 * @return the @f$ \ell^\infty@f$-norm of the residual vector
	 */
	virtual double residualNorm(double time, int secIdx, double const* vecStateY, double const* vecStateYdot) = 0;

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
	 * @param [in] vecStateY State of the simulation
	 * @param [in] vecStateYdot Time derivative of simulation state
	 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
	 */
	virtual int linearSolve(double t, double alpha, double tol, double* rhs, double const* weight,
		double const* vecStateY, double const* vecStateYdot) = 0;

	/**
	 * @brief Applies initial conditions to the state vector and its time derivative
	 * @details The initial conditions do not need to be consistent at this point. On a (discontinuous)
	 *          transition from one section to the next, notifyDiscontinuousSectionTransition() is called by
	 *          the time integrator in order to compute consistent initial conditions. Therefore,
	 *          notifyDiscontinuousSectionTransition() is also called at the beginning of the simulation, that is,
	 *          the initial conditions set by this function will be corrected for consistency.
	 *          Note that the state vector and its time derivative are pre-initialized with zero by the
	 *          time integrator.
	 * 
	 * @param [in,out] vecStateY State of the simulation
	 * @param [in,out] vecStateYdot Time derivative of simulation state
	 */
	virtual void applyInitialCondition(double* vecStateY, double* vecStateYdot) const = 0;

	/**
	 * @brief Save the current solution
	 * @param [in] t Current time point
	 * @param [in] vecStateY State of the simulation
	 * @param [in] vecStateYdot Time derivative of simulation state
	 */
    virtual void saveSolution(double t, double const* vecStateY, double const* vecStateYdot) = 0;
};

/**
 * @brief Solves an ODE or DAE
 * @details This class is used to run tests that involve ODE / DAE systems
 *          but not the CADET simulator.
 */
class TimeIntegrator
{
public:

	TimeIntegrator();
	~TimeIntegrator() CADET_NOEXCEPT;

	void setSolutionTimes(const std::vector<double>& solutionTimes);
	const std::vector<double>& getSolutionTimes() const;
	void setSectionTimes(const std::vector<double>& sectionTimes);
	void setSectionTimes(const std::vector<double>& sectionTimes, const std::vector<bool>& sectionContinuity);

	void initializeModel(IDiffEqModel& model);

	void integrate();

	void configureTimeIntegrator(double relTol, double absTol, double initStepSize, unsigned int maxSteps, double maxStepSize);
	void configureTimeIntegrator(double relTol, double absTol, const std::vector<double>& initStepSizes, unsigned int maxSteps, double maxStepSize);

	void setRelativeErrorTolerance(double relTol);
	void setAbsoluteErrorTolerance(double absTol);
	void setAbsoluteErrorTolerance(const std::vector<double>& absTol);
	void setInitialStepSize(double stepSize);
	void setInitialStepSize(const std::vector<double>& stepSize);
	void setMaximumSteps(unsigned int maxSteps);
	void setMaximumStepSize(double maxStepSize);
	void setMaxNewtonIteration(unsigned int nIter);
	void setMaxErrorTestFails(unsigned int nFails);
	void setMaxConvergenceFails(unsigned int nFails);
	void setMaxSensNewtonIteration(unsigned int nIter);

	IDiffEqModel* model() CADET_NOEXCEPT { return _model; }
	IDiffEqModel const* model() const CADET_NOEXCEPT { return _model; }

    int currentSection() const CADET_NOEXCEPT { return _curSec; }

protected:

	/**
	 * @brief Computes the index of the next section from the given time @p t
	 * @details Returns the lowest index @c i with @f$ t_i \geq t @f$, where 
	 *          @f$ t_i @f$ is an element of @c _sectionTimes.
	 * @param [in] t Current time
	 * @param [in] startIdx Index of the first section the search should begin with
	 * @return Index of the next section corresponding to time @p t
	 */
	int getNextSection(double t, int startIdx) const;

    void updateMainErrorTolerances();

	IDiffEqModel* _model; //!< Simulated model, not owned by the Simulator

	void* _idaMemBlock; //!< IDAS internal memory

	/**
	 * @brief Determines whether the transition from section i to section i+1 is continuous.
	 * @details The solver will be reset only at discontinuous transitions. The i-th element 
	 *          corresponds to the transition from _sectionTimes[i+1] to _sectionTimes[i+2]. 
	 *          Therefore size = nsec - 1.
	 */
	std::vector<bool> _sectionContinuity;

	std::vector<double> _solutionTimes; //!< Contains the time transformed user specified times for writing solutions to the output

	N_Vector _vecStateY; //!< IDAS state vector	
	N_Vector _vecStateYdot; //!< IDAS state vector time derivative
	std::vector<double> _sectionTimes; //!< Stores the section times

	std::vector<double> _absTol; //!< Absolute tolerance for the original system in the time integration
	double _relTol; //!< Relative tolerance for the original system in the time integration
	std::vector<double> _initStepSize; //!< Initial step size for the time integrator
	unsigned int _maxSteps; //!< Maximum number of time integration steps
	double _maxStepSize; //!< Maximum time step size

	unsigned int _maxNewtonIter; //!< Maximum number of Newton iterations for original DAE system
	unsigned int _maxErrorTestFail; //!< Maximum number of local time integration error test failures
	unsigned int _maxConvTestFail; //!< Maximum number of Newton iteration failures

	int _curSec; //!< Index of the current section
};

} // namespace test

} // namespace cadet

#endif  // CADETTEST_TIMEINTEGRATOR_SKEL_HPP_
