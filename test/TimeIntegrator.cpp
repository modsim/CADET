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

#include <idas/idas.h>
#include <idas/idas_impl.h>
#include "TimeIntegrator.hpp"

#include <vector>
#include <sstream>
#include <algorithm>
#include <cstdlib>

#include "LoggingUtils.hpp"
#include "Logging.hpp"

namespace
{
	inline bool hasNaN(double const* const y, unsigned int size)
	{
		for (unsigned int i = 0; i < size; ++i)
		{
			if (std::isnan(y[i]))
				return true;
		}
		return false;
	}

	inline bool hasNaN(const N_Vector p)
	{
		return hasNaN(NVEC_DATA(p), NVEC_LENGTH(p));
	}

	inline std::string getIDAReturnFlagName(int solverFlag)
	{
		char const* const retFlagName = IDAGetReturnFlagName(solverFlag);
		const std::string flagName = retFlagName;
		std::free(const_cast<char*>(retFlagName));

		return flagName;
	}

	/**
	 * @brief IDAS error handler function
	 * @details Handles errors reported by the IDAS solver. See section 4.6.2 of the IDAS manual for details.
	 */
	void idasErrorHandler(int error_code, const char* module, const char* function, char* msg, void* eh_data)
	{
		std::ostringstream oss;
		oss << "In function '" << function << "' of module '" << module << "', error code '" << getIDAReturnFlagName(error_code) << "':\n" << msg;

		// @todo Find an error handling system and put it here
		if (error_code < 0)
		{
			// Error
			LOG(Error) << oss.str();
		}
		else
		{
			// Warning
			LOG(Warning) << oss.str();
		}
	}

	/**
	* @brief IDAS wrapper function to call the model's residual() method
	*/
	int residualDaeWrapper(double t, N_Vector y, N_Vector yDot, N_Vector res, void* userData)
	{
		cadet::test::TimeIntegrator* const sim = static_cast<cadet::test::TimeIntegrator*>(userData);
		const int secIdx = sim->currentSection();

		LOG(Trace) << "==> Residual at t = " << t << " sec = " << secIdx;

		return sim->model()->residualWithJacobian(t, secIdx, NVEC_DATA(y), NVEC_DATA(yDot), NVEC_DATA(res));
	}

	/**
	* @brief IDAS wrapper function to call the model's linearSolve() method
	*/
	int linearSolveWrapper(IDAMem IDA_mem, N_Vector rhs, N_Vector weight, N_Vector y, N_Vector yDot, N_Vector res)
	{
		cadet::test::TimeIntegrator* const sim = static_cast<cadet::test::TimeIntegrator*>(IDA_mem->ida_lmem);
		const double t = IDA_mem->ida_tn;
		const double alpha = IDA_mem->ida_cj;
		const double tol = IDA_mem->ida_epsNewt;

		LOG(Trace) << "==> Solve at t = " << t << " alpha = " << alpha << " tol = " << tol;

		return sim->model()->linearSolve(t, alpha, tol, NVEC_DATA(rhs), NVEC_DATA(weight), NVEC_DATA(y), NVEC_DATA(yDot));
	}
}

namespace cadet
{

namespace test
{

	TimeIntegrator::TimeIntegrator() : _model(nullptr), _idaMemBlock(nullptr), _vecStateY(nullptr),
		_vecStateYdot(nullptr), _absTol(1, 1.0e-8), _relTol(1.0e-6), _initStepSize(1, 1.0e-6), 
        _maxSteps(10000), _maxStepSize(0.0), _maxNewtonIter(3), _maxErrorTestFail(7), _maxConvTestFail(10),
		_curSec(0)
	{
	}

	TimeIntegrator::~TimeIntegrator() CADET_NOEXCEPT
	{
		if (_vecStateYdot)
			NVec_Destroy(_vecStateYdot);
		if (_vecStateY)
			NVec_Destroy(_vecStateY);

		if (_idaMemBlock)
			IDAFree(&_idaMemBlock);
	}

	void TimeIntegrator::initializeModel(IDiffEqModel& model)
	{
		_model = &model;

		// Allocate and initialize state vectors
		const unsigned int nDOFs = _model->numDofs();
		_vecStateY = NVec_New(nDOFs);
		_vecStateYdot = NVec_New(nDOFs);

		// Initialize with all zeros, correct initial conditions will be set later
		NVec_Const(0.0, _vecStateY);
		NVec_Const(0.0, _vecStateYdot);

		// Create IDAS internal memory
		_idaMemBlock = IDACreate();

		// IDAS Step 4.1: Specify error handler function
		IDASetErrHandlerFn(_idaMemBlock, &idasErrorHandler, this);

		// IDAS Step 5: Initialize the solver
		_model->applyInitialCondition(NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot));

		// Use 0.0 as beginning of simulation time if we haven't set section times yet
		if (_sectionTimes.size() > 0)
			IDAInit(_idaMemBlock, &residualDaeWrapper, _sectionTimes[0], _vecStateY, _vecStateYdot);
		else
			IDAInit(_idaMemBlock, &residualDaeWrapper, 0.0, _vecStateY, _vecStateYdot);

		// IDAS Step 6: Specify integration tolerances (S: scalar; V: array)
		updateMainErrorTolerances();

		// IDAS Step 7.1: Set optional inputs

		// Set time integrator parameters
		IDASetMaxNumSteps(_idaMemBlock, _maxSteps);
		IDASetMaxStep(_idaMemBlock, _maxStepSize);
		IDASetMaxNonlinIters(_idaMemBlock, _maxNewtonIter);
		IDASetMaxErrTestFails(_idaMemBlock, _maxErrorTestFail);
		IDASetMaxConvFails(_idaMemBlock, _maxConvTestFail);

		// Specify the linear solver.
		IDAMem IDA_mem = static_cast<IDAMem>(_idaMemBlock);

		IDA_mem->ida_lsolve         = &linearSolveWrapper;
		IDA_mem->ida_lmem           = this;
		IDA_mem->ida_linit          = nullptr;
		IDA_mem->ida_lsetup         = nullptr;
		IDA_mem->ida_lperf          = nullptr;
		IDA_mem->ida_lfree          = nullptr;
//		IDA_mem->ida_efun           = &weightWrapper;
//		IDA_mem->ida_user_efun      = 1;
#if CADET_SUNDIALS_IFACE <= 2
		IDA_mem->ida_setupNonNull   = false;
#endif

		// Attach user data structure
		IDASetUserData(_idaMemBlock, this);
	}

	void TimeIntegrator::configureTimeIntegrator(double relTol, double absTol, double initStepSize, unsigned int maxSteps, double maxStepSize)
	{
		_absTol.clear();
		_absTol.push_back(absTol);

		_relTol = relTol;
		_maxSteps = maxSteps;
		_maxStepSize = maxStepSize;

		_initStepSize.clear();
		_initStepSize.push_back(initStepSize);

		updateMainErrorTolerances();
	}

	void TimeIntegrator::configureTimeIntegrator(double relTol, double absTol, const std::vector<double>& initStepSizes, unsigned int maxSteps, double maxStepSize)
	{
		_absTol.clear();
		_absTol.push_back(absTol);

		_relTol = relTol;
		_maxSteps = maxSteps;
		_maxStepSize = maxStepSize;
		_initStepSize = initStepSizes;

		updateMainErrorTolerances();
	}

	void TimeIntegrator::updateMainErrorTolerances()
	{
		if (!_idaMemBlock)
			return;

		if (_absTol.size() > 1)
		{
			if (!_model)
				return;

            const unsigned int nDofs = _model->numDofs();
			N_Vector absTolTemp = NVec_New(nDofs);

			// Check whether user has given us full absolute error for all (pure) DOFs
			if (_absTol.size() >= nDofs)
			{
				// Copy error tolerances for pure data
				std::copy(_absTol.data(), _absTol.data() + nDofs, NVEC_DATA(absTolTemp));
			}

			IDASVtolerances(_idaMemBlock, _relTol, absTolTemp);
			NVec_Destroy(absTolTemp);
		}
		else
			IDASStolerances(_idaMemBlock, _relTol, _absTol[0]);
	}

	const std::vector<double>& TimeIntegrator::getSolutionTimes() const
	{
		return _solutionTimes;
	}

	void TimeIntegrator::setSectionTimes(const std::vector<double>& sectionTimes)
	{
		setSectionTimes(sectionTimes, std::vector<bool>(sectionTimes.size() - 1, false));
	}

	void TimeIntegrator::setSectionTimes(const std::vector<double>& sectionTimes, const std::vector<bool>& sectionContinuity)
	{
		// Ensure that at least one section is defined
		if (sectionTimes.size() < 2)
			throw std::invalid_argument("At least one section has to be specified!");

		// Ensure that all section start times are smaller than their end times
		for (std::size_t i = 0; i < sectionTimes.size() - 1; ++i)
			if (sectionTimes[i] > sectionTimes[i + 1])
			{
				LOG(Error) << "The end time of each section must be greater than its start time (failed for section " << i << ")!";
				return;
			}

        _sectionTimes = sectionTimes;
        _sectionContinuity = sectionContinuity;
	}

	int TimeIntegrator::getNextSection(double t, int startIdx) const
	{
		if (t < _sectionTimes[startIdx])
			return -1;

		for (std::size_t i = startIdx; i < _sectionTimes.size() - 1; ++i)
		{
			if (_sectionTimes[i] >= t)
				return i;
		}

		return -1;
	}

	void TimeIntegrator::integrate()
	{
		// In this function the model is integrated by IDAS from the SUNDIALS package.
		// The authors of IDAS recommend to restart the time integrator when a discontinuity
		// is encountered (see https://computation.llnl.gov/casc/sundials/support/notes.html#disc).
		// The sectionTime (together with the sectionContinuity) array indicates such
		// discontinuitites and the solver is restarted accordingly. This also requires
		// the computation of consistent initial values for each restart.

		std::vector<double>::const_iterator it;
		double tOut = 0.0;

		const bool writeAtUserTimes = _solutionTimes.size() > 0;

		// Decide whether to use user specified solution output times (IDA_NORMAL)
		// or internal integrator steps (IDA_ONE_STEP)
		int idaTask = IDA_ONE_STEP;
		if (writeAtUserTimes)
		{
			idaTask = IDA_NORMAL;
		}

		LOG(Debug) << "Integration span: [" << _sectionTimes[0] << ", " << _sectionTimes.back() << "] sections";

		if (writeAtUserTimes)
		{
			LOG(Debug) << "Solution time span: [" << _solutionTimes[0] << ", " << _solutionTimes.back() << "]";
		}

		double curT = _sectionTimes[0];
		_curSec = 0;
		const double tEnd = writeAtUserTimes ? _solutionTimes.back() : _sectionTimes.back();
		while (curT < tEnd)
		{
			// Get smallest index with t_i >= curT (t_i being a _sectionTimes element)
			// This will return i if curT == _sectionTimes[i], which effectively advances
			// the index if required
			_curSec = getNextSection(curT, _curSec);
			const double startTime = _sectionTimes[_curSec];

			// Determine continuous time slice
			unsigned int skip = 1; // Always finish the current section
			for (std::size_t i = _curSec; i < _sectionTimes.size() - 2; ++i)
			{
				if (!_sectionContinuity[i])
					break;

				// This is a continuous section transition, so we don't need to
				// restart the integrator and just integrate for a longer time
				++skip;
			}

			const double endTime = writeAtUserTimes ? std::min(_sectionTimes[_curSec + skip], tEnd) : _sectionTimes[_curSec + skip];
			curT = startTime;

			LOG(Debug) << " ###### SECTION " << _curSec << " from " << startTime << " to " << endTime;

			// IDAS Step 7.3: Set the initial step size
			const double stepSize = _initStepSize.size() > 1 ? _initStepSize[_curSec] : _initStepSize[0];
			IDASetInitStep(_idaMemBlock, stepSize);

			// IDAS Step 7.4: Set the stop time
			IDASetStopTime(_idaMemBlock, endTime);

			// Update Jacobian and compute consistent initial values
			_model->notifyDiscontinuousSectionTransition(curT, _curSec, NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot));

			// IDAS Step 5.2: Re-initialization of the solver
			IDAReInit(_idaMemBlock, startTime, _vecStateY, _vecStateYdot);

			// Inititalize the IDA solver flag
			int solverFlag = IDA_SUCCESS;

			if (writeAtUserTimes)
			{
				// Write initial conditions only if desired by user
				if (_curSec == 0 && _solutionTimes.front() == curT)
					_model->saveSolution(curT, NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot));

				// Initialize iterator and forward it to the first solution time that lies inside the current section
				it = _solutionTimes.begin();
				while ((*it) <= startTime) ++it;
			}
			else
			{
				// Always write initial conditions if solutions are written at integration times
				if (_curSec == 0)
                    _model->saveSolution(curT, NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot));

				// Here tOut - only during the first call to IDASolve - specifies the direction
				// and rough scale of the independent variable, see IDAS Guide p.33
				tOut = endTime;
			}

			// Main loop which integrates the system until reaching the end time of the current section
			// or until an error occures
			while ((solverFlag == IDA_SUCCESS) || (solverFlag == IDA_ROOT_RETURN))
			{
				// Update tOut if we write solutions at user specified times
				if (writeAtUserTimes)
				{
					// Check if user specified times are sufficiently long.
					// otherwise integrate till IDA_TSTOP_RETURN
					if (it == _solutionTimes.end())
						break;
					else
						tOut = *it;
				}

				// IDA Step 11: Advance solution in time
				solverFlag = IDASolve(_idaMemBlock, tOut, &curT, _vecStateY, _vecStateYdot, idaTask);
				LOG(Debug) << "Solve from " << curT << " to " << tOut << " => "
					<< (solverFlag == IDA_SUCCESS ? "IDA_SUCCESS" : "") << (solverFlag == IDA_TSTOP_RETURN ? "IDA_TSTOP_RETURN" : "");

#ifdef CADET_DEBUG
				{
					long nTimeSteps = 0;
					IDAGetNumSteps(_idaMemBlock, &nTimeSteps);

					double curStepSize = 0.0;
					IDAGetCurrentStep(_idaMemBlock, &curStepSize);

					double lastStepSize = 0.0;
					IDAGetLastStep(_idaMemBlock, &lastStepSize);

					long nResEvals = 0;
					IDAGetNumResEvals(_idaMemBlock, &nResEvals);

					long nErrTestFail = 0;
					IDAGetNumErrTestFails(_idaMemBlock, &nErrTestFail);

					long nNonLin = 0;
					IDAGetNumNonlinSolvIters(_idaMemBlock, &nNonLin);

					long nConvFail = 0;
					IDAGetNumNonlinSolvConvFails(_idaMemBlock, &nConvFail);

					LOG(Debug) << "=== #Steps: " << nTimeSteps << "\n=== #Residual evals: " << nResEvals << "\n=== #Error test fails: " << nErrTestFail
						<< "\n=== #Newton iters: " << nNonLin << "\n=== #Conv test fails: " << nConvFail << "\n=== Last step size: " << lastStepSize
						<< "\n=== Next step size: " << curStepSize;
				}
	#endif
				switch (solverFlag)
				{
				case IDA_SUCCESS:
					// tOut was reached
					_model->saveSolution(curT, NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot));

					if (writeAtUserTimes)
						++it;

					break;
				case IDA_ROOT_RETURN:
					// A root was found
					// Eventually call some routine
					break;
				case IDA_TSTOP_RETURN:
					// Section end time was reached (in previous step)
					if (!writeAtUserTimes && (endTime == _sectionTimes.back()))
					{
						// Write a solution for the ultimate endTime in the last section,
						// when we write at integration times.
    					_model->saveSolution(curT, NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot));
					}

					break;
				default:
					// An error occured
					const std::string errorFlag = getIDAReturnFlagName(solverFlag);
					LOG(Error) << "IDASolve returned " << errorFlag << " at t = " << curT;

                    return;
				} // switch

			} // while

		} // for (_sec ...)

	}

	void TimeIntegrator::setRelativeErrorTolerance(double relTol)
	{
		_relTol = relTol;
		updateMainErrorTolerances();
	}

	void TimeIntegrator::setAbsoluteErrorTolerance(double absTol)
	{
		_absTol.clear();
		_absTol.push_back(absTol);
		updateMainErrorTolerances();
	}

	void TimeIntegrator::setAbsoluteErrorTolerance(const std::vector<double>& absTol)
	{
		_absTol = absTol;
		updateMainErrorTolerances();
	}

	void TimeIntegrator::setInitialStepSize(double stepSize)
	{
		_initStepSize.clear();
		_initStepSize.push_back(stepSize);
	}

	void TimeIntegrator::setInitialStepSize(const std::vector<double>& stepSize)
	{
		_initStepSize = stepSize;
	}

	void TimeIntegrator::setMaximumSteps(unsigned int maxSteps)
	{
		_maxSteps = maxSteps;
		if (_idaMemBlock)
			IDASetMaxNumSteps(_idaMemBlock, _maxSteps);
	}

	void TimeIntegrator::setMaximumStepSize(double maxStepSize)
	{
		_maxStepSize = std::max(0.0, maxStepSize);
		if (_idaMemBlock)
			IDASetMaxStep(_idaMemBlock, _maxStepSize);
	}

	void TimeIntegrator::setMaxNewtonIteration(unsigned int nIter)
	{
		_maxNewtonIter = nIter;
		if (_idaMemBlock)
			IDASetMaxNonlinIters(_idaMemBlock, nIter);
	}

	void TimeIntegrator::setMaxErrorTestFails(unsigned int nFails)
	{
		_maxErrorTestFail = nFails;
		if (_idaMemBlock)
			IDASetMaxErrTestFails(_idaMemBlock, nFails);
	}

	void TimeIntegrator::setMaxConvergenceFails(unsigned int nFails)
	{
		_maxConvTestFail = nFails;
		if (_idaMemBlock)
			IDASetMaxConvFails(_idaMemBlock, nFails);
	}

	void TimeIntegrator::setSolutionTimes(const std::vector<double>& solutionTimes)
	{
		_solutionTimes = solutionTimes;
	}

} // namespace test

} // namespace cadet
