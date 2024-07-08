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

#include "cadet/Exceptions.hpp"
#include "cadet/SolutionRecorder.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/Notification.hpp"
#include "SimulatorImpl.hpp"
#include "SimulatableModel.hpp"
#include "ParamIdUtil.hpp"
#include "SimulationTypes.hpp"

#include <idas/idas.h>
#include <idas/idas_impl.h>
#include "SundialsVector.hpp"

#include <vector>
#include <sstream>
#include <algorithm>
#include <cstdlib>

#include "AutoDiff.hpp"
#include "LoggingUtils.hpp"
#include "Logging.hpp"

#ifdef CADET_PARALLELIZE
	#include <tbb/parallel_for.h>

	#define TBB_PREVIEW_GLOBAL_CONTROL 1
	#include <tbb/global_control.h>

	#ifdef CADET_TBB_GLOBALCTRL
		#include <tbb/task_arena.h>
	#endif
#endif

namespace
{
	template <class T>
	const std::vector<T> convertNVectorToStdVectorPtrs(N_Vector* vec, unsigned int numVec)
	{
		std::vector<T> sensState(numVec, nullptr);
		for (unsigned int i = 0; i < numVec; ++i)
			sensState[i] = NVEC_DATA(vec[i]);

		return sensState;
	}

	const std::vector<double*> convertNVectorToStdVectorPtrs(unsigned int& len, N_Vector* vec, unsigned int numVec)
	{
		if (!vec || (numVec == 0))
		{
			return std::vector<double*>(0, nullptr);
		}

		len = NVEC_LENGTH(vec[0]);
		return convertNVectorToStdVectorPtrs<double*>(vec, numVec);
	}

	const std::vector<const double*> convertNVectorToStdVectorConstPtrs(unsigned int& len, N_Vector* vec, unsigned int numVec)
	{
		if (!vec || (numVec == 0))
		{
			return std::vector<const double*>(0, nullptr);
		}

		len = NVEC_LENGTH(vec[0]);
		return convertNVectorToStdVectorPtrs<const double*>(vec, numVec);
	}

	/**
	 * @brief Checks whether a given parameter @p id corresponds to a SECTION_TIMES parameter
	 * @param [in] id Parameter id to be checked
	 * @param [in] nSectionTimes Number of sections
	 * @return @c true if the given id corresponds to a SECTION_TIMES parameter, otherwise @c false
	 */
	inline bool isSectionTimeParameter(const cadet::ParameterId& id, unsigned int nSectionTimes)
	{
		return ((id.name == cadet::hashString("SECTION_TIMES")) && (id.section < nSectionTimes) && (id.component == cadet::CompIndep) &&
			(id.particleType == cadet::ParTypeIndep) && (id.boundState == cadet::BoundStateIndep) && (id.reaction == cadet::ReactionIndep) &&
			(id.unitOperation == cadet::UnitOpIndep));
	}

	/**
	 * @brief Determines the method of consistent initialization for the transition into the given section
	 * @param [in] mode Consistent initialization mode set by user
	 * @param [in] curSec Index of current section
	 * @return One of @c Full, @c Lean, @c None
	 */
	inline cadet::ConsistentInitialization currentConsistentInitMode(cadet::ConsistentInitialization mode, unsigned int curSec)
	{
		switch (mode)
		{
			case cadet::ConsistentInitialization::None:
				return cadet::ConsistentInitialization::None;

			case cadet::ConsistentInitialization::Full:
				return cadet::ConsistentInitialization::Full;

			case cadet::ConsistentInitialization::FullFirstOnly:
				if (curSec == 0)
					return cadet::ConsistentInitialization::Full;
				else
					return cadet::ConsistentInitialization::None;

			case cadet::ConsistentInitialization::Lean:
				return cadet::ConsistentInitialization::Lean;

			case cadet::ConsistentInitialization::LeanFirstOnly:
				if (curSec == 0)
					return cadet::ConsistentInitialization::Lean;
				else
					return cadet::ConsistentInitialization::None;

			case cadet::ConsistentInitialization::FullOnceThenLean:
				if (curSec == 0)
					return cadet::ConsistentInitialization::Full;
				else
					return cadet::ConsistentInitialization::Lean;

			case cadet::ConsistentInitialization::NoneOnceThenFull:
				if (curSec == 0)
					return cadet::ConsistentInitialization::None;
				else
					return cadet::ConsistentInitialization::Full;

			case cadet::ConsistentInitialization::NoneOnceThenLean:
				if (curSec == 0)
					return cadet::ConsistentInitialization::None;
				else
					return cadet::ConsistentInitialization::Lean;
		}
		return cadet::ConsistentInitialization::None;
	}

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
}

namespace cadet
{
	namespace log
	{
		inline std::ostream& operator<<(std::ostream& os, const N_Vector& nv)
		{
			double const* const ptrY = NVEC_DATA(nv);
			os << "[";
			for (int i = 0; i < NVEC_LENGTH(nv)-1; ++i)
				os << ptrY[i] << ",";
			os << ptrY[NVEC_LENGTH(nv)-1] << "]";
			return os;
		}
	}

	/**
	 * @brief IDAS error handler function
	 * @details Handles errors reported by the IDAS solver. See section 4.6.2 of the IDAS manual for details.
	 */
	void idasErrorHandler(int error_code, const char* module, const char* function, char* msg, void* eh_data)
	{
//		cadet::Simulator* const sim = static_cast<cadet::Simulator*>(eh_data);

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
		cadet::Simulator* const sim = static_cast<cadet::Simulator*>(userData);
		const unsigned int secIdx = sim->getCurrentSection(t);

		LOG(Trace) << "==> Residual at t = " << t << " sec = " << secIdx;

		if (sim->_modifiedNewton)
			return sim->_model->residual(cadet::SimulationTime{t, secIdx}, cadet::ConstSimulationState{NVEC_DATA(y), NVEC_DATA(yDot)}, NVEC_DATA(res));

		return sim->_model->residualWithJacobian(cadet::SimulationTime{t, secIdx}, cadet::ConstSimulationState{NVEC_DATA(y), NVEC_DATA(yDot)}, NVEC_DATA(res),
			cadet::AdJacobianParams{sim->_vecADres, sim->_vecADy, sim->numSensitivityAdDirections()});
	}

	int jacobianUpdateWrapper(IDAMem IDA_mem, N_Vector y, N_Vector yDot, N_Vector res, N_Vector tempv1, N_Vector tempv2, N_Vector tempv3)
	{
		cadet::Simulator* const sim = static_cast<cadet::Simulator*>(IDA_mem->ida_lmem);
		const double t = IDA_mem->ida_tn;
		const unsigned int secIdx = sim->getCurrentSection(t);

		LOG(Trace) << "==> Jacobian at t = " << t;

		return sim->_model->jacobian(cadet::SimulationTime{ t, secIdx }, cadet::ConstSimulationState{ NVEC_DATA(y), NVEC_DATA(yDot) }, NVEC_DATA(tempv1),
			cadet::AdJacobianParams{ sim->_vecADres, sim->_vecADy, sim->numSensitivityAdDirections() });
	}

	/**
	* @brief Change the error weights in the state vector
	* @details This sets the error weight to 0 for the network coupling equations, duplicated inlets
	*          and inlet and outlet state vector entries. Those entries are all solved exactly and
	*          without this the solver takes more steps and smaller steps for some simulations.
	*          The problem is the largest error is usually on the first and last column cells in the GRM
	*          and since the algebraic systems are solved exactly they end up duplicating those errors exactly.
	*          In many cases this leads to a system that would have passed error control failing due to the error
	*          doubleing or more in size.
	* @param [in] y Current state vector
	* @param [out] ewt Weight vector
	* @param [in] user_data User data originally supplied to IDAS
	*/
/*
	int weightWrapper(N_Vector y, N_Vector ewt, void *user_data)
	{
		cadet::Simulator* const sim = static_cast<cadet::Simulator*>(user_data);
		double * localY = NVEC_DATA(y);
		double * localEWT = NVEC_DATA(ewt);

		const double rtol = sim->_relTol;
		const double abstolV = sim->_absTol;
		const double abstol = abstolV[0];
		const unsigned int numDofs = sim->numDofs();

		unsigned int index = 0;

		for (unsigned int i = 0; i < sim->_model->numModels(); ++i)
		{
			IUnitOperation* const m = sim->_model->getUnitOperationModel(i);
			const unsigned int modelDof = m->numDofs();

			//Handle Inlets
			if (m->hasOutlet() && !m->hasInlet())
			{
				for (unsigned int j = 0; j < modelDof; ++j, ++index)
				{
					localEWT[index] = 0;
				}
			}

			//Handle Normal Operations
			if (m->hasOutlet() && m->hasInlet())
			{
				const unsigned int numComps = m->numComponents();
				const unsigned int remaining = modelDof - numComps;
				for (unsigned int j = 0; j < numComps; ++j, ++index)
				{
					localEWT[index] = 0;
				}
				for (unsigned int j = 0; j < remaining; ++j, ++index)
				{
					localEWT[index] = 1 / (rtol * localY[i] + abstol);
				}
			}

			//Handle Outlets
			if (!m->hasOutlet() && m->hasInlet())
			{
				for (unsigned int j = 0; j < modelDof; ++j, ++index)
				{
					localEWT[index] = 0;
				}
			}

		}

		const unsigned int remaining = numDofs - index;

		for (unsigned int i = 0; i < remaining; ++i, ++index)
		{
			localEWT[index] = 0;
		}

		NVEC_DATA(ewt) = localEWT;

		return 0;
	}
*/

	/**
	* @brief IDAS wrapper function to call the model's linearSolve() method
	*/
	int linearSolveWrapper(IDAMem IDA_mem, N_Vector rhs, N_Vector weight, N_Vector y, N_Vector yDot, N_Vector res)
	{
		cadet::Simulator* const sim = static_cast<cadet::Simulator*>(IDA_mem->ida_lmem);
		const double t = IDA_mem->ida_tn;
		const double alpha = IDA_mem->ida_cj;
		const double tol = IDA_mem->ida_epsNewt;

		LOG(Trace) << "==> Solve at t = " << t << " alpha = " << alpha << " tol = " << tol;

		return sim->_model->linearSolve(t, alpha, tol, NVEC_DATA(rhs), NVEC_DATA(weight), cadet::ConstSimulationState{NVEC_DATA(y), NVEC_DATA(yDot)});
	}

	/**
	* @brief IDAS wrapper function to call the model's residualSensFwd() method
	*/
	int residualSensWrapper(int ns, double t, N_Vector y, N_Vector yDot, N_Vector res,
			N_Vector* yS, N_Vector* ySDot, N_Vector* resS,
			void *userData, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
	{
		cadet::Simulator* const sim = static_cast<cadet::Simulator*>(userData);
		const std::vector<const double*> sensY = convertNVectorToStdVectorPtrs<const double*>(yS, ns);
		const std::vector<const double*> sensYdot = convertNVectorToStdVectorPtrs<const double*>(ySDot, ns);
		std::vector<double*> sensRes = convertNVectorToStdVectorPtrs<double*>(resS, ns);
		const unsigned int secIdx = sim->getCurrentSection(t);

		LOG(Trace) << "==> Residual SENS at t = " << t << " sec = " << secIdx;

/*
		reinterpret_cast<cadet::model::ModelSystem*>(sim->_model)->genJacobian(t, secIdx, NVEC_DATA(y), NVEC_DATA(yDot));
		reinterpret_cast<cadet::model::ModelSystem*>(sim->_model)->genJacobian(ns, t, NVEC_DATA(y), NVEC_DATA(yDot), NVEC_DATA(res),
			sensY, sensYdot, sensRes, sim->_vecADres, NVEC_DATA(tmp1), NVEC_DATA(tmp2), NVEC_DATA(tmp3));
*/

		if (sim->_modifiedNewton)
		{
			return sim->_model->residualSensFwdWithJacobian(ns, cadet::SimulationTime{t, secIdx}, cadet::ConstSimulationState{NVEC_DATA(y), NVEC_DATA(yDot)}, NVEC_DATA(res),
				sensY, sensYdot, sensRes, cadet::AdJacobianParams{sim->_vecADres, sim->_vecADy, sim->numSensitivityAdDirections()}, NVEC_DATA(tmp1), NVEC_DATA(tmp2), NVEC_DATA(tmp3));
		}

		return sim->_model->residualSensFwd(ns, cadet::SimulationTime{t, secIdx}, cadet::ConstSimulationState{NVEC_DATA(y), NVEC_DATA(yDot)}, NVEC_DATA(res),
			sensY, sensYdot, sensRes, sim->_vecADres, NVEC_DATA(tmp1), NVEC_DATA(tmp2), NVEC_DATA(tmp3));
	}

	Simulator::Simulator() : _model(nullptr), _solRecorder(nullptr), _idaMemBlock(nullptr), _vecStateY(nullptr),
		_vecStateYdot(nullptr), _vecFwdYs(nullptr), _vecFwdYsDot(nullptr),
		_relTolS(1.0e-9), _absTol(1, 1.0e-12), _relTol(1.0e-9), _initStepSize(1, 1.0e-6), _maxSteps(10000), _maxStepSize(0.0),
		_nThreads(0), _sensErrorTestEnabled(true), _maxNewtonIter(4), _maxErrorTestFail(10), _maxConvTestFail(10),
		_maxNewtonIterSens(4), _curSec(0), _skipConsistencyStateY(false), _skipConsistencySensitivity(false),
		_consistentInitMode(ConsistentInitialization::Full), _consistentInitModeSens(ConsistentInitialization::Full),
		_vecADres(nullptr), _vecADy(nullptr), _lastIntTime(0.0), _notification(nullptr)
	{
#if defined(ACTIVE_SFAD) || defined(ACTIVE_SETFAD)
		LOG(Debug) << "Resetting AD directions from " << ad::getDirections() << " to default " << ad::getMaxDirections();
		ad::setDirections(ad::getMaxDirections());
#endif
	}

	Simulator::~Simulator() CADET_NOEXCEPT
	{
		clearModel();
	}

	void Simulator::clearModel() CADET_NOEXCEPT
	{
		delete[] _vecADy;
		delete[] _vecADres;

		if ((_sensitiveParams.slices() > 0) && _vecFwdYs)
		{
			NVec_DestroyArray(_vecFwdYs, _sensitiveParams.slices());
			NVec_DestroyArray(_vecFwdYsDot, _sensitiveParams.slices());
		}
		_sensitiveParams.clear();

		if (_vecStateYdot)
			NVec_Destroy(_vecStateYdot);
		if (_vecStateY)
			NVec_Destroy(_vecStateY);

		if (_idaMemBlock)
			IDAFree(&_idaMemBlock);
	}

	void Simulator::initializeModel(IModelSystem& model)
	{
		// Clean up
		clearModel();

		// Require ISimulatableModel descendant
		_model = reinterpret_cast<ISimulatableModel*>(&model);

		// Allocate and initialize state vectors
		const unsigned int nDOFs = _model->numDofs();
		_vecStateY = NVec_New(nDOFs);
		_vecStateYdot = NVec_New(nDOFs);

		// Propagate section times if available
		if (_sectionTimes.size() > 0)
		{
			bool* const secCont = new bool[_sectionContinuity.size()];
			std::copy(_sectionContinuity.begin(), _sectionContinuity.end(), secCont);

			// Convert from active to double
			double* const secTimes = new double[_sectionTimes.size()];
			for (std::size_t i = 0; i < _sectionTimes.size(); ++i)
				secTimes[i] = static_cast<double>(_sectionTimes[i]);

			_model->setSectionTimes(secTimes, secCont, _sectionTimes.size() - 1);

			delete[] secTimes;
			delete[] secCont;
		}

		// Initialize with all zeros, correct initial conditions will be set later
		NVec_Const(0.0, _vecStateY);
		NVec_Const(0.0, _vecStateYdot);

		// Create IDAS internal memory
		_idaMemBlock = IDACreate();

		// IDAS Step 4.1: Specify error handler function
		IDASetErrHandlerFn(_idaMemBlock, &idasErrorHandler, this);

		// IDAS Step 5: Initialize the solver
		_model->applyInitialCondition(SimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)});

		// Use 0.0 as beginning of simulation time if we haven't set section times yet
		if (_sectionTimes.size() > 0)
			IDAInit(_idaMemBlock, &residualDaeWrapper, static_cast<double>(_sectionTimes[0]), _vecStateY, _vecStateYdot);
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
		IDASetSensMaxNonlinIters(_idaMemBlock, _maxNewtonIterSens);

		// Specify the linear solver.
		IDAMem IDA_mem = static_cast<IDAMem>(_idaMemBlock);

		IDA_mem->ida_lsolve         = &linearSolveWrapper;
		IDA_mem->ida_lmem           = this;
		IDA_mem->ida_linit          = nullptr;
		IDA_mem->ida_lsetup         = _modifiedNewton ? &jacobianUpdateWrapper : nullptr;
		IDA_mem->ida_lperf          = nullptr;
		IDA_mem->ida_lfree          = nullptr;
//		IDA_mem->ida_efun           = &weightWrapper;
//		IDA_mem->ida_user_efun      = 1;
#if CADET_SUNDIALS_IFACE <= 2
		IDA_mem->ida_setupNonNull   = false;
#endif

		// Attach user data structure
		IDASetUserData(_idaMemBlock, this);

		// Allocate memory for AD if required
		if (_model->usesAD())
		{
			_vecADres = new active[nDOFs];
			_vecADy = new active[nDOFs];
		}
	}

	void Simulator::updateMainErrorTolerances()
	{
		if (!_idaMemBlock)
			return;

		if (_absTol.size() > 1)
		{
			if (!_model)
				return;

			N_Vector absTolTemp = NVec_New(_model->numDofs());
			const unsigned int pureDofs = _model->numPureDofs();

			// Check whether user has given us full absolute error for all (pure) DOFs
			if (_absTol.size() >= pureDofs)
			{
				// Copy error tolerances for pure data
				std::copy(_absTol.data(), _absTol.data() + pureDofs, NVEC_DATA(absTolTemp));

				// Calculate error tolerances for coupling DOFs and append them
				const std::vector<double> addAbsErrTol = _model->calculateErrorTolsForAdditionalDofs(_absTol.data(), _absTol.size());
				std::copy(addAbsErrTol.data(), addAbsErrTol.data() + addAbsErrTol.size(), NVEC_DATA(absTolTemp) + pureDofs);
			}
			else
			{
				// We've received an expandable error specification
				_model->expandErrorTol(_absTol.data(), _absTol.size(), NVEC_DATA(absTolTemp));
			}

			IDASVtolerances(_idaMemBlock, _relTol, absTolTemp);
			NVec_Destroy(absTolTemp);
		}
		else
			IDASStolerances(_idaMemBlock, _relTol, _absTol[0]);
	}

	void Simulator::preFwdSensInit(unsigned int nSens)
	{
		// Turn off solution of sensitivity systems (this will be overridden by a call to IDASensInit below)
		// In fact, this has only an effect, if at first a computation with sensitivities is performed and then
		// (without clearing and reallocating internal memory by cs_free/cs_malloc) another computation without
		// sensitivities is started.
		IDASensToggleOff(_idaMemBlock);

		if (_vecFwdYs)
		{
			NVec_DestroyArray(_vecFwdYs, nSens);
			NVec_DestroyArray(_vecFwdYsDot, nSens);
			_vecFwdYs = nullptr;
			_vecFwdYsDot = nullptr;
		}

		// Allocate sensitivity state vectors
		if (nSens > 0)
		{
			_vecFwdYs     = NVec_CloneArray(nSens, _vecStateY);
			_vecFwdYsDot  = NVec_CloneArray(nSens, _vecStateYdot);

			// Allocate memory for AD if not already done
			if (!_vecADres)
				_vecADres = new active[_model->numDofs()];
		}
	}

	void Simulator::postFwdSensInit(unsigned int nSens)
	{
		// Initialize IDA sensitivity computation
		// TODO: Use IDASensReInit if this is not the first time sensitivities are activated
		IDASensInit(_idaMemBlock, nSens, IDA_STAGGERED, &cadet::residualSensWrapper, _vecFwdYs, _vecFwdYsDot);

		// Set sensitivity integration tolerances
		IDASensSStolerances(_idaMemBlock, _relTolS, _absTolS.data());

		// Activate sensitivity error control
		IDASetSensErrCon(_idaMemBlock, _sensErrorTestEnabled);
	}

	void Simulator::initializeFwdSensitivities()
	{
		const unsigned int nSens = _sensitiveParams.slices();
		preFwdSensInit(nSens);

		if (nSens == 0)
			return;

		for (unsigned int dir = 0; dir < nSens; ++dir)
		{
			// Initialize sensitivity vectors with 0.0
			NVec_Const(0.0, _vecFwdYs[dir]);
			NVec_Const(0.0, _vecFwdYsDot[dir]);
		}

		// Apply initial values due to sensitivites with respect to initial conditions
		_model->initializeSensitivityStates(convertNVectorToStdVectorPtrs<double*>(_vecFwdYs, nSens));

		// Compute consistent initial conditions for sensitivity systems later
		_skipConsistencySensitivity = false;

		postFwdSensInit(nSens);
	}

	void Simulator::initializeFwdSensitivities(double const * const* const initSens, double const * const* const initSensDot)
	{
		const unsigned int nSens = _sensitiveParams.slices();
		preFwdSensInit(nSens);

		if (nSens == 0)
			return;

		if (!initSens)
			throw std::invalid_argument("Pointer to initial sensitivities is NULL");

		if (!initSensDot)
			throw std::invalid_argument("Pointer to initial sensitivity time derivatives is NULL");

		for (unsigned int dir = 0; dir < nSens; ++dir)
		{
			double* const yS = NVEC_DATA(_vecFwdYs[dir]);
			double const* const src = initSens[dir];
			double* const ySdot = NVEC_DATA(_vecFwdYsDot[dir]);
			double const* const srcDot = initSensDot[dir];

			// Initialize sensitivity vectors with given data
			std::copy(src, src + NVEC_LENGTH(_vecFwdYs[dir]), yS);
			std::copy(srcDot, srcDot + NVEC_LENGTH(_vecFwdYsDot[dir]), ySdot);
		}

		// Don't assume that consistent values were given
		_skipConsistencySensitivity = false;

		postFwdSensInit(nSens);
	}

	void Simulator::applyInitialConditionFwdSensitivities(double const * const* const initSens, double const * const* const initSensDot)
	{
		const unsigned int nSens = _sensitiveParams.slices();
		if (nSens == 0)
			return;

		if (initSens && initSensDot)
		{
			for (unsigned int dir = 0; dir < nSens; ++dir)
			{
				double* const yS = NVEC_DATA(_vecFwdYs[dir]);
				double const* const src = initSens[dir];
				double* const ySdot = NVEC_DATA(_vecFwdYsDot[dir]);
				double const* const srcDot = initSensDot[dir];

				// Initialize sensitivity vectors with given data
				std::copy(src, src + NVEC_LENGTH(_vecFwdYs[dir]), yS);
				std::copy(srcDot, srcDot + NVEC_LENGTH(_vecFwdYsDot[dir]), ySdot);
			}
		}
		else
		{
			for (unsigned int dir = 0; dir < nSens; ++dir)
			{
				// Initialize sensitivity vectors with 0.0
				NVec_Const(0.0, _vecFwdYs[dir]);
				NVec_Const(0.0, _vecFwdYsDot[dir]);
			}

			// Apply initial values due to sensitivites with respect to initial conditions
			_model->initializeSensitivityStates(convertNVectorToStdVectorPtrs<double*>(_vecFwdYs, nSens));
		}

		// Don't assume that consistent values were given
		_skipConsistencySensitivity = false;
	}

	void Simulator::applyInitialCondition()
	{
		_model->applyInitialCondition(SimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)});
		IDAReInit(_idaMemBlock, static_cast<double>(_sectionTimes[0]), _vecStateY, _vecStateYdot);

		// Better check for consistency
		_skipConsistencyStateY = false;
	}

	void Simulator::setInitialCondition(IParameterProvider& paramProvider)
	{
		_model->readInitialCondition(paramProvider);
	}

	void Simulator::applyInitialCondition(double const* const initState)
	{
		// Copy initial state
		double* const y = NVEC_DATA(_vecStateY);
		std::copy(initState, initState + NVEC_LENGTH(_vecStateY), y);

		IDAReInit(_idaMemBlock, static_cast<double>(_sectionTimes[0]), _vecStateY, _vecStateYdot);

		// We need to compute matching yDot for consistency
		_skipConsistencyStateY = false;
	}

	void Simulator::applyInitialCondition(double const* const initState, double const* const initStateDot)
	{
		// Copy initial state
		double* const y = NVEC_DATA(_vecStateY);
		std::copy(initState, initState + NVEC_LENGTH(_vecStateY), y);

		// Copy initial time derivative state
		double* const yDot = NVEC_DATA(_vecStateYdot);
		std::copy(initStateDot, initStateDot + NVEC_LENGTH(_vecStateY), yDot);

		IDAReInit(_idaMemBlock, static_cast<double>(_sectionTimes[0]), _vecStateY, _vecStateYdot);

		// Do not assume that the initial state is consistent
		_skipConsistencyStateY = false;
	}

	void Simulator::skipConsistentInitialization()
	{
		_skipConsistencyStateY = true;
		_skipConsistencySensitivity = true;
	}

	void Simulator::setConsistentInitialization(ConsistentInitialization ci)
	{
		_consistentInitMode = ci;
	}

	void Simulator::setConsistentInitializationSens(ConsistentInitialization ci)
	{
		_consistentInitModeSens = ci;
	}

	std::unordered_map<ParameterId, double> Simulator::getAllParameterValues() const
	{
		std::unordered_map<ParameterId, double> data;
		if (_model)
			data = _model->getAllParameterValues();

		// Add section times
		const StringHash secTimesName = hashString("SECTION_TIMES");
		for (std::size_t i = 0; i < _sectionTimes.size(); ++i)
			data[makeParamId(secTimesName, UnitOpIndep, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, i)] = static_cast<double>(_sectionTimes[i]);

		return data;
	}

	bool Simulator::hasParameter(const ParameterId& pId) const
	{
		if (isSectionTimeParameter(pId, _sectionTimes.size()))
			return true;

		if (_model)
			return _model->hasParameter(pId);

		return false;
	}

	void Simulator::setSensitiveParameter(const ParameterId& id)
	{
		setSensitiveParameter(id, 1.0e-5);
	}

	void Simulator::setSensitiveParameter(const ParameterId& id, double absTolS)
	{
		setSensitiveParameter(&id, nullptr, 1, absTolS);
	}

	void Simulator::setSensitiveParameter(ParameterId const* ids, double const* diffFactors, unsigned int numParams, double absTolS)
	{
		// Set AD directions
		const unsigned int adDir = numSensitivityAdDirections();
		for (unsigned int i = 0; i < numParams; ++i)
		{
			double localDiffFactor = 1.0;
			if (diffFactors)
				localDiffFactor = diffFactors[i];

			bool paramFound = setSectionTimesSensitive(ids[i], adDir, localDiffFactor);
			paramFound = _model->setSensitiveParameter(ids[i], adDir, localDiffFactor) || paramFound;

			if (!paramFound)
			{
				LOG(Warning) << "Warning: Unkown parameter " << ids[i] << " in parameter join was ignored";
			}
		}

		_sensitiveParams.pushBackSlice(ids, numParams);
		_absTolS.push_back(absTolS);

		if (diffFactors)
			_sensitiveParamsFactor.insert(_sensitiveParamsFactor.end(), diffFactors, diffFactors + numParams);
		else
			_sensitiveParamsFactor.insert(_sensitiveParamsFactor.end(), numParams, 1.0);
	}

	void Simulator::setSensitiveParameter(ParameterId const* ids, unsigned int numParams, double absTolS)
	{
		setSensitiveParameter(ids, nullptr, numParams, absTolS);
	}

	void Simulator::resetSensParams()
	{
		using std::to_string;

		unsigned int globalIdx = 0;
		for (unsigned int i = 0; i < _sensitiveParams.slices(); ++i)
		{
			ParameterId const* const ids = _sensitiveParams[i];

			for (unsigned int j = 0; j < _sensitiveParams.sliceSize(i); ++j, ++globalIdx)
			{
				bool paramFound = setSectionTimesSensitive(ids[j], i, _sensitiveParamsFactor[globalIdx]);
				paramFound = _model->setSensitiveParameter(ids[j], i, _sensitiveParamsFactor[globalIdx]) || paramFound;

				if (!paramFound)
					throw InvalidParameterException("Sensitive parameter " + to_string(ids[j]) + " disappeared");
			}
		}
	}

	bool Simulator::setSectionTimesSensitive(const ParameterId& id, unsigned int adDirection, double adValue)
	{
		if (isSectionTimeParameter(id, _sectionTimes.size()))
		{
			// Correct adValue
			LOG(Debug) << "Found parameter " << id << " in SECTION_TIMES: Dir " << adDirection << " is set to "
				<< adValue << " [" << static_cast<double>(_sectionTimes[id.section]) << " < " << _sectionTimes.size() << "]";
			_sectionTimes[id.section].setADValue(adDirection, adValue);

			return true;
		}
		return false;
	}

	void Simulator::clearSensParams()
	{
		_sensitiveParams.clear();
		_sensitiveParamsFactor.clear();
		_absTolS.clear();

		_model->clearSensParams();
		for (std::size_t i = 0; i < _sectionTimes.size(); ++i)
			_sectionTimes[i].setADValue(0.0);

		initializeFwdSensitivities();
	}

	unsigned int Simulator::numSensParams() const CADET_NOEXCEPT
	{
		return _sensitiveParams.slices();
	}

	void Simulator::setSensitiveParameterValue(const ParameterId& id, double value)
	{
		if (isSectionTimeParameter(id, _sectionTimes.size()))
		{
			_sectionTimes[id.section].setValue(value);
			// Do not exit here, since model can also contain instances of SECTION_TIMES
		}

		if (!_model)
			return;

		util::SlicedVector<ParameterId>::size_type temp;
		util::SlicedVector<ParameterId>::size_type linearIndex;
		util::SlicedVector<ParameterId>::size_type idxSlice = _sensitiveParams.findElementAndSlice(id, temp, linearIndex);
		if (idxSlice < _sensitiveParams.slices())
		{
			ParameterId const* const fusedIds = _sensitiveParams[idxSlice];
			const util::SlicedVector<ParameterId>::size_type sliceOffset = _sensitiveParams.sliceOffset(idxSlice);

			// Noramlize parameter value
			value /= _sensitiveParamsFactor[linearIndex];

			// Take care of linear factors
			// @todo add constant offset to linear combination
			for (unsigned int i = 0; i < _sensitiveParams.sliceSize(idxSlice); ++i)
				_model->setSensitiveParameterValue(fusedIds[i], _sensitiveParamsFactor[sliceOffset + i] * value);
		}
	}

	void Simulator::setSensitiveParameterValue(unsigned int idx, double value)
	{
		if (idx >= _sensitiveParams.slices())
			return;

		ParameterId const* const paramIds = _sensitiveParams[idx];
		const util::SlicedVector<ParameterId>::size_type sliceOffset = _sensitiveParams.sliceOffset(idx);

		for (unsigned int i = 0; i < _sensitiveParams.sliceSize(idx); ++i)
		{
			const ParameterId& id = paramIds[i];
			if (isSectionTimeParameter(id, _sectionTimes.size()))
			{
				_sectionTimes[id.section].setValue(value);
				// Do not exit here, since model can also contain instances of SECTION_TIMES
			}

			if (!_model)
				continue;

			// Take care of linear factors
			// @todo add constant offset to linear combination
			_model->setSensitiveParameterValue(id, _sensitiveParamsFactor[sliceOffset + i] * value);
		}
	}

	void Simulator::setSensitiveParameterFactors(unsigned int idx, double const* factors)
	{
		if (idx >= _sensitiveParams.slices())
			return;

		ParameterId const* const paramIds = _sensitiveParams[idx];
		const util::SlicedVector<ParameterId>::size_type sliceOffset = _sensitiveParams.sliceOffset(idx);

		for (unsigned int i = 0; i < _sensitiveParams.sliceSize(idx); ++i)
		{
			// Update the linear factor
			_sensitiveParamsFactor[sliceOffset + i] = factors[i];

			const ParameterId& id = paramIds[i];
			setSectionTimesSensitive(id, i, factors[i]);

			if (_model)
				_model->setSensitiveParameter(id, i, factors[i]);
		}
	}

	void Simulator::setParameterValue(const ParameterId& id, double value)
	{
		if (isSectionTimeParameter(id, _sectionTimes.size()))
		{
			_sectionTimes[id.section].setValue(value);
			// Do not exit here, since model can also contain instances of SECTION_TIMES
		}

		if (_model)
			_model->setParameter(id, value);
	}

	void Simulator::setSolutionTimes(const std::vector<double>& solutionTimes)
	{
		_solutionTimes = solutionTimes;
	}

	const std::vector<double>& Simulator::getSolutionTimes() const
	{
		return _solutionTimes;
	}

	void Simulator::setSectionTimes(const std::vector<double>& sectionTimes)
	{
		setSectionTimes(sectionTimes, std::vector<bool>(sectionTimes.size() - 1, false));
	}

	void Simulator::setSectionTimes(const std::vector<double>& sectionTimes, const std::vector<bool>& sectionContinuity)
	{
		// Ensure that at least one section is defined
		if (sectionTimes.size() < 2)
			throw std::invalid_argument("At least one section has to be specified!");

		// Ensure that all section start times are smaller than their end times
		for (std::size_t i = 0; i < sectionTimes.size() - 1; ++i)
			if (sectionTimes[i] > sectionTimes[i + 1])
				throw InvalidParameterException("The end time of each section must be greater than its start time (failed for section " + std::to_string(i) + ")!");

		// Send section times to models
		if (_model)
		{
			bool* const secCont = new bool[sectionContinuity.size()];
			std::copy(sectionContinuity.begin(), sectionContinuity.end(), secCont);
			_model->setSectionTimes(sectionTimes.data(), secCont, sectionTimes.size() - 1);
			delete[] secCont;
		}

		// Copy section times into AD (active) data type
		_sectionTimes.clear();
		_sectionTimes.reserve(sectionTimes.size());
		for (std::size_t i = 0; i < sectionTimes.size(); ++i)
			_sectionTimes.push_back(sectionTimes[i]);

		_sectionContinuity = sectionContinuity;

		// Set AD sensitivities
		unsigned int globalIdx = 0;
		for (unsigned int i = 0; i < _sensitiveParams.slices(); ++i)
		{
			ParameterId const* const ids = _sensitiveParams[i];

			for (unsigned int j = 0; j < _sensitiveParams.sliceSize(i); ++j, ++globalIdx)
			{
				setSectionTimesSensitive(ids[j], i, _sensitiveParamsFactor[globalIdx]);
			}
		}
	}

	void Simulator::setSolutionRecorder(ISolutionRecorder* recorder)
	{
		_solRecorder = recorder;
		if (_solRecorder)
		{
			_solRecorder->prepare(NVEC_LENGTH(_vecStateY), _sensitiveParams.slices(), _solutionTimes.size());
			_model->reportSolutionStructure(*_solRecorder);
		}
	}

	void Simulator::integrate()
	{
		// In this function the model is integrated by IDAS from the SUNDIALS package.
		// The authors of IDAS recommend to restart the time integrator when a discontinuity
		// is encountered (see https://computation.llnl.gov/casc/sundials/support/notes.html#disc).
		// The sectionTime (together with the sectionContinuity) array indicates such
		// discontinuitites and the solver is restarted accordingly. This also requires
		// the computation of consistent initial values for each restart.

		// This sets up the tbb thread limiter
		// TBB can use up to _nThreads but it may use fewer
#ifdef CADET_PARALLELIZE
	#ifdef CADET_TBB_GLOBALCTRL
		tbb::global_control tbbGlobalControl(tbb::global_control::max_allowed_parallelism, (_nThreads > 0) ? _nThreads : tbb::this_task_arena::max_concurrency());
	#else
		tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
		if (_nThreads > 0)
			init.initialize(_nThreads);
		else
			init.initialize(tbb::task_scheduler_init::default_num_threads());
	#endif
		_model->setupParallelization(tbb::this_task_arena::max_concurrency());
#else
		_model->setupParallelization(1);
#endif

		// Set number of threads in SUNDIALS OpenMP-enabled implementation
#ifdef CADET_SUNDIALS_OPENMP
		if (_vecStateY)
			NVec_SetThreads(_vecStateY, _nThreads);
		if (_vecStateYdot)
			NVec_SetThreads(_vecStateYdot, _nThreads);

		for (unsigned int i = 0; i < _sensitiveParams.slices(); ++i)
		{
			NVec_SetThreads(_vecFwdYs[i], _nThreads);
			NVec_SetThreads(_vecFwdYsDot[i], _nThreads);
		}
#endif

		// Set number of AD directions
		// @todo This is problematic if multiple Simulators are run concurrently!
#if defined(ACTIVE_SFAD) || defined(ACTIVE_SETFAD)
		LOG(Debug) << "Setting AD directions from " << ad::getDirections() << " to " << numSensitivityAdDirections() + _model->requiredADdirs();
		if (numSensitivityAdDirections() + _model->requiredADdirs() > ad::getMaxDirections())
			throw InvalidParameterException("Requested " + std::to_string(numSensitivityAdDirections() + _model->requiredADdirs()) + " AD directions, but only "
				+ std::to_string(ad::getMaxDirections()) + " are supported");

		ad::setDirections(numSensitivityAdDirections() + _model->requiredADdirs());
#endif

		if (_notification)
			_notification->timeIntegrationStart();

		_timerIntegration.start();

		// Setup AD vectors by model
		_model->prepareADvectors(AdJacobianParams{_vecADres, _vecADy, numSensitivityAdDirections()});

		std::vector<double>::const_iterator it;
		double tOut = 0.0;

		const bool writeAtUserTimes = _solutionTimes.size() > 0;
		const bool wantSensitivities = _sensitiveParams.slices() > 0;

		LOG(Debug) << "#MaxNewton: " << _maxNewtonIter << ", #MaxErrTestFail: " << _maxErrorTestFail << ", #MaxConvTestFail: " << _maxConvTestFail;
		if (wantSensitivities)
		{
			LOG(Debug) << "Sensitvities in error test: " << _sensErrorTestEnabled << ", #MaxNewtonSens: " << _maxNewtonIterSens;
		}

		if (_solRecorder)
		{
			_solRecorder->notifyIntegrationStart(NVEC_LENGTH(_vecStateY), _sensitiveParams.slices(), _solutionTimes.size());
			_model->reportSolutionStructure(*_solRecorder);
		}

		// Decide whether to use user specified solution output times (IDA_NORMAL)
		// or internal integrator steps (IDA_ONE_STEP)
		int idaTask = IDA_ONE_STEP;
		if (writeAtUserTimes)
		{
			idaTask = IDA_NORMAL;
		}

		LOG(Debug) << "Integration span: [" << static_cast<double>(_sectionTimes[0]) << ", " << static_cast<double>(_sectionTimes.back()) << "] sections";

		if (writeAtUserTimes)
		{
			LOG(Debug) << "Solution time span: [" << _solutionTimes[0] << ", " << _solutionTimes.back() << "]";
		}

		double curT = static_cast<double>(_sectionTimes[0]);
		_curSec = 0;
		const double tEnd = writeAtUserTimes ? _solutionTimes.back() : static_cast<double>(_sectionTimes.back());
		while (curT < tEnd)
		{
			// Get smallest index with t_i >= curT (t_i being a _sectionTimes element)
			// This will return i if curT == _sectionTimes[i], which effectively advances
			// the index if required
			_curSec = getNextSection(curT, _curSec);
			const double startTime = static_cast<double>(_sectionTimes[_curSec]);

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

			const double endTime = writeAtUserTimes ? std::min(static_cast<double>(_sectionTimes[_curSec + skip]), tEnd) : static_cast<double>(_sectionTimes[_curSec + skip]);
			curT = startTime;

			LOG(Debug) << " ###### SECTION " << _curSec << " from " << startTime << " to " << endTime;

			// IDAS Step 7.3: Set the initial step size
			const double stepSize = _initStepSize.size() > 1 ? _initStepSize[_curSec] : _initStepSize[0];
			IDASetInitStep(_idaMemBlock, stepSize);

			// IDAS Step 7.4: Set the stop time
			IDASetStopTime(_idaMemBlock, endTime);

			// Update Jacobian
			_model->notifyDiscontinuousSectionTransition(curT, _curSec, ConstSimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)}, AdJacobianParams{_vecADres, _vecADy, numSensitivityAdDirections()});

			// Compute consistent initial values
			LOG(Debug) << "---====--- CONSISTENCY ---====--- ";
			const double consPrev = _model->residualNorm(SimulationTime{curT, _curSec}, ConstSimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)});
			LOG(Debug) << " ==========> Consistency error prev: " << consPrev;

			if (!_skipConsistencyStateY && (_consistentInitMode != ConsistentInitialization::None))
			{
				const ConsistentInitialization mode = currentConsistentInitMode(_consistentInitMode, _curSec);
				if (mode == ConsistentInitialization::Full)
				{
					_model->consistentInitialConditions(SimulationTime{curT, _curSec}, SimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)},
						AdJacobianParams{_vecADres, _vecADy, numSensitivityAdDirections()}, _algTol);

					const double consPost = _model->residualNorm(SimulationTime{curT, _curSec}, ConstSimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)});
					LOG(Debug) << " ==========> Consistency error post Full: " << consPost;
				}
				else if (mode == ConsistentInitialization::Lean)
				{
					_model->leanConsistentInitialConditions(SimulationTime{curT, _curSec}, SimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)},
						AdJacobianParams{_vecADres, _vecADy, numSensitivityAdDirections()}, _algTol);

					const double consPost = _model->residualNorm(SimulationTime{curT, _curSec}, ConstSimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)});
					LOG(Debug) << " ==========> Consistency error post Lean: " << consPost;
				}
				else
				{
					LOG(Debug) << " ==========> Consistent initialization NOT performed (mode " << to_string(_consistentInitMode) << ")";
				}

				LOG(Debug) << "y = " << log::VectorPtr<double>(NVEC_DATA(_vecStateY), _model->numDofs()) << ";";
				LOG(Debug) << "yDot = " << log::VectorPtr<double>(NVEC_DATA(_vecStateYdot), _model->numDofs()) << ";";
				LOG(Debug) << "Contains NaN: y = " << hasNaN(_vecStateY) << " yDot = " << hasNaN(_vecStateYdot);
			}
			_skipConsistencyStateY = false;

			if (wantSensitivities && !_skipConsistencySensitivity && (_consistentInitModeSens != ConsistentInitialization::None))
			{
#ifdef CADET_DEBUG
				const std::vector<const double*> sensYdbg = convertNVectorToStdVectorPtrs<const double*>(_vecFwdYs, _sensitiveParams.slices());
				const std::vector<const double*> sensYdotDbg = convertNVectorToStdVectorPtrs<const double*>(_vecFwdYsDot, _sensitiveParams.slices());

				std::vector<double> norms(_sensitiveParams.slices(), 0.0);
				std::vector<double> temp(_model->numDofs(), 0.0);
				_model->residualSensFwdNorm(_sensitiveParams.slices(), SimulationTime{curT, _curSec}, ConstSimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)},
					sensYdbg, sensYdotDbg, norms.data(), _vecADres, temp.data());

				LOG(Debug) << " ==========> Sens consistency error prev: " << norms;
#endif

				const ConsistentInitialization mode = currentConsistentInitMode(_consistentInitModeSens, _curSec);
				if (mode == ConsistentInitialization::Full)
				{
					// Compute consistent initial conditions for sensitivity subsystems
					std::vector<double*> sensY = convertNVectorToStdVectorPtrs<double*>(_vecFwdYs, _sensitiveParams.slices());
					std::vector<double*> sensYdot = convertNVectorToStdVectorPtrs<double*>(_vecFwdYsDot, _sensitiveParams.slices());
					_model->consistentInitialSensitivity(SimulationTime{curT, _curSec}, ConstSimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)}, sensY, sensYdot, _vecADres, _vecADy);

#ifdef CADET_DEBUG
					_model->residualSensFwdNorm(_sensitiveParams.slices(), SimulationTime{curT, _curSec}, ConstSimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)},
						sensYdbg, sensYdotDbg, norms.data(), _vecADres, temp.data());

					LOG(Debug) << " ==========> Sens consistency error post Full: " << norms;
#endif
				}
				else if (mode == ConsistentInitialization::Lean)
				{
					// Compute consistent initial conditions for sensitivity subsystems
					std::vector<double*> sensY = convertNVectorToStdVectorPtrs<double*>(_vecFwdYs, _sensitiveParams.slices());
					std::vector<double*> sensYdot = convertNVectorToStdVectorPtrs<double*>(_vecFwdYsDot, _sensitiveParams.slices());
					_model->leanConsistentInitialSensitivity(SimulationTime{curT, _curSec}, ConstSimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)}, sensY, sensYdot, _vecADres, _vecADy);

#ifdef CADET_DEBUG
					_model->residualSensFwdNorm(_sensitiveParams.slices(), SimulationTime{curT, _curSec}, ConstSimulationState{NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot)},
						sensYdbg, sensYdotDbg, norms.data(), _vecADres, temp.data());

					LOG(Debug) << " ==========> Sens consistency error post Lean: " << norms;
#endif
				}
				else
				{
					LOG(Debug) << " ==========> Sens consistent initialization NOT performed (mode " << to_string(_consistentInitModeSens) << ")";
				}

#ifdef CADET_DEBUG
				for (std::size_t j = 0; j < norms.size(); ++j)
				{
					LOG(Debug) << "sensY[" << j << "] = " << log::VectorPtr<double>(NVEC_DATA(_vecFwdYs[j]), _model->numDofs()) << ";";
					LOG(Debug) << "sensYdot[" << j << "] = " << log::VectorPtr<double>(NVEC_DATA(_vecFwdYsDot[j]), _model->numDofs()) << ";";
					LOG(Debug) << "Contains NaN: sensY[" << j << "] = " << hasNaN(_vecFwdYs[j]) << " sensYdot[" << j << "] = " << hasNaN(_vecFwdYsDot[j]);
				}
#endif
			}
			_skipConsistencySensitivity = false;

			// Notify user and check for user abort
			if (_notification)
			{
				const double progress = (curT - static_cast<double>(_sectionTimes[0])) / (tEnd - static_cast<double>(_sectionTimes[0]));
				if (!_notification->timeIntegrationSection(_curSec, curT, NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot), progress))
				{
					_lastIntTime = _timerIntegration.stop();
					return;
				}
			}

			// IDAS Step 5.2: Re-initialization of the solver
			IDAReInit(_idaMemBlock, startTime, _vecStateY, _vecStateYdot);
			if (wantSensitivities)
				IDASensReInit(_idaMemBlock, IDA_STAGGERED, _vecFwdYs, _vecFwdYsDot);

			// Inititalize the IDA solver flag
			int solverFlag = IDA_SUCCESS;

			if (writeAtUserTimes)
			{
				// Write initial conditions only if desired by user
				if (_curSec == 0 && _solutionTimes.front() == curT)
					writeSolution(curT);

				// Initialize iterator and forward it to the first solution time that lies inside the current section
				it = _solutionTimes.begin();
				while ((*it) <= startTime) ++it;
			}
			else
			{
				// Always write initial conditions if solutions are written at integration times
				if (_curSec == 0) writeSolution(curT);

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

					if (wantSensitivities)
					{
						long nSensResEvals = 0;
						IDAGetSensNumResEvals(_idaMemBlock, &nSensResEvals);

						long nSensErrTestFails = 0;
						IDAGetSensNumErrTestFails(_idaMemBlock, &nSensErrTestFails);

						long nSensNonLin = 0;
						IDAGetSensNumNonlinSolvIters(_idaMemBlock, &nSensNonLin);

						long nSensConvFail = 0;
						IDAGetSensNumNonlinSolvConvFails(_idaMemBlock, &nSensConvFail);

						LOG(Debug) << "=== #Sens residual evals: " << nSensResEvals << "\n=== #Sens error test fails: " << nSensErrTestFails
							<< "\n=== #Sens Newton iters: " << nSensNonLin << "\n=== #Sens conv test fails: " << nSensConvFail;
					}
				}
	#endif
				switch (solverFlag)
				{
				case IDA_SUCCESS:
					// tOut was reached

					// Extract sensitivity information from IDA (required for consistent initialization
					// and output of sensitivities)
					if (wantSensitivities)
					{
						IDAGetSens(_idaMemBlock, &curT, _vecFwdYs);
						IDAGetSensDky(_idaMemBlock, curT, 1, _vecFwdYsDot);
					}
					writeSolution(curT);

					if (writeAtUserTimes)
						++it;

					// Notify user and check for user abort
					if (_notification)
					{
						const double progress = (curT - static_cast<double>(_sectionTimes[0])) / (tEnd - static_cast<double>(_sectionTimes[0]));
						if (!_notification->timeIntegrationStep(_curSec, curT, NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot), progress))
						{
							_lastIntTime = _timerIntegration.stop();
							return;
						}
					}
					break;
				case IDA_ROOT_RETURN:
					// A root was found
					// Eventually call some routine
					break;
				case IDA_TSTOP_RETURN:
					// Extract sensitivity information from IDA (required for consistent initialization
					// and output of sensitivities)
					if (wantSensitivities)
					{
						IDAGetSens(_idaMemBlock, &curT, _vecFwdYs);
						IDAGetSensDky(_idaMemBlock, curT, 1, _vecFwdYsDot);
					}

					// Section end time was reached (in previous step)
					if (!writeAtUserTimes && (endTime == static_cast<double>(_sectionTimes.back())))
					{
						// Write a solution for the ultimate endTime in the last section,
						// when we write at integration times.
						writeSolution(curT);
					}

					// Notify user and check for user abort
					if (_notification)
					{
						const double progress = (curT - static_cast<double>(_sectionTimes[0])) / (tEnd - static_cast<double>(_sectionTimes[0]));
						if (!_notification->timeIntegrationStep(_curSec, curT, NVEC_DATA(_vecStateY), NVEC_DATA(_vecStateYdot), progress))
						{
							_lastIntTime = _timerIntegration.stop();
							return;
						}
					}
					break;
				default:
					_lastIntTime = _timerIntegration.stop();

					// An error occured
					const std::string errorFlag = getIDAReturnFlagName(solverFlag);
					LOG(Error) << "IDASolve returned " << errorFlag << " at t = " << curT;

					if (_notification)
					{
						const double progress = (curT - static_cast<double>(_sectionTimes[0])) / (tEnd - static_cast<double>(_sectionTimes[0]));
						_notification->timeIntegrationError(errorFlag.c_str(), _curSec, curT, progress);
					}

					throw IntegrationException(std::string("Error in IDASolve: ") + errorFlag + std::string(" at t = ") + std::to_string(curT)); //todo might not be necessary
					break;
				} // switch

			} // while

		} // for (_sec ...)

		_lastIntTime = _timerIntegration.stop();

		if (_notification)
			_notification->timeIntegrationEnd();
	}

	double const* Simulator::getLastSolution(unsigned int& len) const
	{
		len = NVEC_LENGTH(_vecStateY);
		return NVEC_DATA(_vecStateY);
	}

	double const* Simulator::getLastSolutionDerivative(unsigned int& len) const
	{
		len = NVEC_LENGTH(_vecStateYdot);
		return NVEC_DATA(_vecStateYdot);
	}

	std::vector<double const*> Simulator::getLastSensitivities(unsigned int& len) const
	{
		return convertNVectorToStdVectorConstPtrs(len, _vecFwdYs, _sensitiveParams.slices());
	}

	std::vector<double const*> Simulator::getLastSensitivityDerivatives(unsigned int& len) const
	{
		return convertNVectorToStdVectorConstPtrs(len, _vecFwdYsDot, _sensitiveParams.slices());
	}

	void Simulator::configureTimeIntegrator(double relTol, double absTol, double initStepSize, unsigned int maxSteps, double maxStepSize)
	{
		_absTol.clear();
		_absTol.push_back(absTol);

		_relTol = relTol;
		_maxSteps = maxSteps;
		_maxStepSize = maxStepSize;

		_initStepSize.clear();
		_initStepSize.push_back(initStepSize);
	}

	void Simulator::configureTimeIntegrator(double relTol, double absTol, const std::vector<double>& initStepSizes, unsigned int maxSteps, double maxStepSize)
	{
		_absTol.clear();
		_absTol.push_back(absTol);

		_relTol = relTol;
		_maxSteps = maxSteps;
		_maxStepSize = maxStepSize;
		_initStepSize = initStepSizes;
	}

	void Simulator::configure(IParameterProvider& paramProvider)
	{
		paramProvider.pushScope("time_integrator");

		if (paramProvider.exists("USE_MODIFIED_NEWTON"))
			_modifiedNewton = paramProvider.getBool("USE_MODIFIED_NEWTON");
		else
			_modifiedNewton = false;

		_absTol.clear();
		if (paramProvider.isArray("ABSTOL"))
			_absTol = paramProvider.getDoubleArray("ABSTOL");
		else
			_absTol.push_back(paramProvider.getDouble("ABSTOL"));

		_relTol = paramProvider.getDouble("RELTOL");
		_algTol = paramProvider.getDouble("ALGTOL");
		_maxSteps = paramProvider.getInt("MAX_STEPS");

		if (paramProvider.exists("MAX_STEP_SIZE"))
			_maxStepSize = paramProvider.getDouble("MAX_STEP_SIZE");

		_initStepSize.clear();
		if (paramProvider.isArray("INIT_STEP_SIZE"))
			_initStepSize = paramProvider.getDoubleArray("INIT_STEP_SIZE");
		else
			_initStepSize.push_back(paramProvider.getDouble("INIT_STEP_SIZE"));

		if (paramProvider.exists("RELTOL_SENS"))
			_relTolS = paramProvider.getDouble("RELTOL_SENS");
		else
			_relTolS = _relTol;

		if (paramProvider.exists("ERRORTEST_SENS"))
			_sensErrorTestEnabled = paramProvider.getBool("ERRORTEST_SENS");

		if (paramProvider.exists("MAX_NEWTON_ITER"))
			_maxNewtonIter = paramProvider.getInt("MAX_NEWTON_ITER");

		if (paramProvider.exists("MAX_ERRTEST_FAIL"))
			_maxErrorTestFail = paramProvider.getInt("MAX_ERRTEST_FAIL");

		if (paramProvider.exists("MAX_CONVTEST_FAIL"))
			_maxConvTestFail = paramProvider.getInt("MAX_CONVTEST_FAIL");

		if (paramProvider.exists("MAX_NEWTON_ITER_SENS"))
			_maxNewtonIterSens = paramProvider.getInt("MAX_NEWTON_ITER_SENS");

		paramProvider.popScope();

		if (paramProvider.exists("NTHREADS"))
		{
			// Ensure numThreads >= 0
			const int numThreads = paramProvider.getInt("NTHREADS");
			_nThreads = std::max(numThreads, 0);
		}
		else
			_nThreads = 0;

		_solutionTimes.clear();
		if (paramProvider.exists("USER_SOLUTION_TIMES"))
			_solutionTimes = paramProvider.getDoubleArray("USER_SOLUTION_TIMES");

		if (static_cast<double>(_solutionTimes.back()) > static_cast<double>(_sectionTimes.back()))
		{
			throw InvalidParameterException("USER_SOLUTION_TIMES exceed section times");
		}

		if (paramProvider.exists("CONSISTENT_INIT_MODE"))
			_consistentInitMode = toConsistentInitialization(paramProvider.getInt("CONSISTENT_INIT_MODE"));

		if (paramProvider.exists("CONSISTENT_INIT_MODE_SENS"))
			_consistentInitModeSens = toConsistentInitialization(paramProvider.getInt("CONSISTENT_INIT_MODE_SENS"));

		// @todo: Read more configuration values
	}

	void Simulator::reconfigure(IParameterProvider& paramProvider)
	{
		configure(paramProvider);
	}

	void Simulator::setSensitivityErrorTolerance(double relTol, double const* absTol)
	{
		_relTolS = relTol;
		_absTolS.clear();

		const unsigned int nSens = _sensitiveParams.slices();
		if (_idaMemBlock && (nSens > 0) && _vecFwdYs && absTol)
		{
			_absTolS.insert(_absTolS.end(), absTol, absTol + nSens);

			// Set sensitivity integration tolerances
			IDASensSStolerances(_idaMemBlock, _relTolS, _absTolS.data());
		}
	}

	void Simulator::setRelativeErrorToleranceSens(double relTol)
	{
		_relTolS = relTol;
		if (_idaMemBlock && (_sensitiveParams.slices() > 0) && _vecFwdYs)
		{
			// Set sensitivity integration tolerances
			IDASensSStolerances(_idaMemBlock, _relTolS, _absTolS.data());
		}
	}

	void Simulator::setRelativeErrorTolerance(double relTol)
	{
		_relTol = relTol;
		updateMainErrorTolerances();
	}

	void Simulator::setAbsoluteErrorTolerance(double absTol)
	{
		_absTol.clear();
		_absTol.push_back(absTol);
		updateMainErrorTolerances();
	}

	void Simulator::setAbsoluteErrorTolerance(const std::vector<double>& absTol)
	{
		_absTol = absTol;
		updateMainErrorTolerances();
	}

	void Simulator::setInitialStepSize(double stepSize)
	{
		_initStepSize.clear();
		_initStepSize.push_back(stepSize);
	}

	void Simulator::setInitialStepSize(const std::vector<double>& stepSize)
	{
		_initStepSize = stepSize;
	}

	void Simulator::setMaximumSteps(unsigned int maxSteps)
	{
		_maxSteps = maxSteps;
		if (_idaMemBlock)
			IDASetMaxNumSteps(_idaMemBlock, _maxSteps);
	}

	void Simulator::setMaximumStepSize(double maxStepSize)
	{
		_maxStepSize = std::max(0.0, maxStepSize);
		if (_idaMemBlock)
			IDASetMaxStep(_idaMemBlock, _maxStepSize);
	}

	void Simulator::setSensitivityErrorControl(bool enabled)
	{
		_sensErrorTestEnabled = enabled;
		if (_idaMemBlock && (_sensitiveParams.slices() > 0) && _vecFwdYs)
			IDASetSensErrCon(_idaMemBlock, enabled);
	}

	void Simulator::setMaxNewtonIteration(unsigned int nIter)
	{
		_maxNewtonIter = nIter;
		if (_idaMemBlock)
			IDASetMaxNonlinIters(_idaMemBlock, nIter);
	}

	void Simulator::setMaxErrorTestFails(unsigned int nFails)
	{
		_maxErrorTestFail = nFails;
		if (_idaMemBlock)
			IDASetMaxErrTestFails(_idaMemBlock, nFails);
	}

	void Simulator::setMaxConvergenceFails(unsigned int nFails)
	{
		_maxConvTestFail = nFails;
		if (_idaMemBlock)
			IDASetMaxConvFails(_idaMemBlock, nFails);
	}

	void Simulator::setMaxSensNewtonIteration(unsigned int nIter)
	{
		_maxNewtonIterSens = nIter;
		if (_idaMemBlock && (_sensitiveParams.slices() > 0) && _vecFwdYs)
			IDASetSensMaxNonlinIters(_idaMemBlock, nIter);
	}



	bool Simulator::reconfigureModel(IParameterProvider& paramProvider)
	{
		if (!_model)
			return false;

		// Reconfigure the model
		const bool success = _model->configure(paramProvider);

		// Set all AD directions for parameter sensitivities again
		resetSensParams();

		return success;
	}

	bool Simulator::reconfigureModel(IParameterProvider& paramProvider, unsigned int unitOpIdx)
	{
		if (!_model)
			return false;

		// Reconfigure the model
		const bool success = _model->configureModel(paramProvider, unitOpIdx);

		// Set all AD directions for parameter sensitivities again
		resetSensParams();

		return success;
	}

	void Simulator::writeSolution(double t)
	{
		if (!_solRecorder)
			return;

		_solRecorder->beginTimestep(t);

		_solRecorder->beginSolution();
		_model->reportSolution(*_solRecorder, NVEC_DATA(_vecStateY));
		_solRecorder->endSolution();

		_solRecorder->beginSolutionDerivative();
		_model->reportSolution(*_solRecorder, NVEC_DATA(_vecStateYdot));
		_solRecorder->endSolutionDerivative();

		for (unsigned int i = 0; i < _sensitiveParams.slices(); ++i)
		{
			_solRecorder->beginSensitivity(*_sensitiveParams[i], i);
			_model->reportSolution(*_solRecorder, NVEC_DATA(_vecFwdYs[i]));
			_solRecorder->endSensitivity(*_sensitiveParams[i], i);

			_solRecorder->beginSensitivityDerivative(*_sensitiveParams[i], i);
			_model->reportSolution(*_solRecorder, NVEC_DATA(_vecFwdYsDot[i]));
			_solRecorder->endSensitivityDerivative(*_sensitiveParams[i], i);
		}

		_solRecorder->endTimestep();
	}

	unsigned int Simulator::getNextSection(double t, unsigned int startIdx) const
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

	unsigned int Simulator::getCurrentSection(double t) const
	{
		//TODO: Use binary search

		for (std::size_t i = _curSec; i < _sectionTimes.size() - 1; ++i)
		{
			if ((t >= _sectionTimes[i]) && (t <= _sectionTimes[i+1]))
				return i;
		}

		return -1;
	}

	IModelSystem* Simulator::model() CADET_NOEXCEPT
	{
		return _model;
	}

	IModelSystem const* Simulator::model() const CADET_NOEXCEPT
	{
		return _model;
	}

	unsigned int Simulator::numDofs() const CADET_NOEXCEPT
	{
		if (_model)
			return _model->numDofs();
		return 0;
	}

	void Simulator::setNumThreads(unsigned int nThreads) CADET_NOEXCEPT
	{
		_nThreads = nThreads;
	}

	void Simulator::setNotificationCallback(INotificationCallback* nc) CADET_NOEXCEPT
	{
		_notification = nc;
	}

} // namespace cadet
