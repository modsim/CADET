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

#include "model/ModelSystemImpl.hpp"

#include "SimulationTypes.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include "ParallelSupport.hpp"
#ifdef CADET_PARALLELIZE
#include <tbb/parallel_for.h>
#endif

#include "model/ModelSystemImpl-Helper.hpp"

namespace cadet
{

namespace model
{

int ModelSystem::linearSolve(double t, double alpha, double outerTol, double* const rhs, double const* const weight,
							 const ConstSimulationState& simState)
{
	if (_linearModelOrdering.sliceSize(_curSwitchIndex) == 0)
	{
		// Parallel
		return linearSolveParallel(t, alpha, outerTol, rhs, weight, simState);
	}
	else
	{
		// Linear
		return linearSolveSequential(t, alpha, outerTol, rhs, weight, simState);
	}
}

int ModelSystem::linearSolveSequential(double t, double alpha, double outerTol, double* const rhs,
									   double const* const weight, const ConstSimulationState& simState)
{
	// TODO: Add early out error checks

	BENCH_SCOPE(_timerLinearSolve);

	// Topological sort needs to be iterated backwards (each item depends on all items behind it)
	int const* order = _linearModelOrdering[_curSwitchIndex] + _models.size() - 1;
	for (int i = 0; i < static_cast<int>(_models.size()); ++i, --order)
	{
		const int idxUnit = *order;
		IUnitOperation* const m = _models[idxUnit];
		const unsigned int offset = _dofOffset[idxUnit];

		if (m->hasInlet() > 0)
		{
			// Solve inlet first
			const unsigned int finalOffset = _dofOffset.back();

			// N_{f,x} Outlet (lower) matrices; Bottom macro-row
			// N_{f,x,1} * y_1 + ... + N_{f,x,nModels} * y_{nModels} + y_{coupling} = f
			// y_{coupling} = f - N_{f,x,1} * y_1 - ... - N_{f,x,nModels} * y_{nModels}
			for (std::size_t j = 0; j < _models.size(); ++j)
			{
				const unsigned int offset2 = _dofOffset[j];
				_jacFN[j].multiplySubtract(rhs + offset2, rhs + finalOffset, _conDofOffset[idxUnit],
										   _conDofOffset[idxUnit + 1]);
			}

			// Calculate inlet DOF for unit operation based on the coupling conditions.
			// y_{unit op inlet} - y_{coupling} = 0
			// y_{unit op inlet} = y_{coupling}
			unsigned int idxCoupling = finalOffset + _conDofOffset[idxUnit];
			for (unsigned int port = 0; port < m->numInletPorts(); ++port)
			{
				const unsigned int localIndex = m->localInletComponentIndex(port);
				const unsigned int localStride = m->localInletComponentStride(port);
				for (unsigned int comp = 0; comp < m->numComponents(); ++comp)
				{
					rhs[offset + localIndex + comp * localStride] = rhs[idxCoupling];
					++idxCoupling;
				}
			}
		}

		// Solve unit operation itself
		_errorIndicator[idxUnit] =
			m->linearSolve(t, alpha, outerTol, rhs + offset, weight + offset, applyOffset(simState, offset));
	}

	return totalErrorIndicatorFromLocal(_errorIndicator);
}

int ModelSystem::linearSolveParallel(double t, double alpha, double outerTol, double* const rhs,
									 double const* const weight, const ConstSimulationState& simState)
{
	// TODO: Add early out error checks

	BENCH_SCOPE(_timerLinearSolve);

	const unsigned int finalOffset = _dofOffset[_models.size()];

#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), _models.size(), [=](std::size_t i)
#else
	for (std::size_t i = 0; i < _models.size(); ++i)
#endif
	{
		IUnitOperation* const m = _models[i];
		const unsigned int offset = _dofOffset[i];
		_errorIndicator[i] =
			m->linearSolve(t, alpha, outerTol, rhs + offset, weight + offset, applyOffset(simState, offset));
	} CADET_PARFOR_END;

	// Solve last row of L with backwards substitution: y_f = b_f - \sum_{i=0}^{N_z} J_{f,i} y_i
	// Note that we cannot easily parallelize this loop since the results of the sparse
	// matrix-vector multiplications are added in-place to rhs. We would need one copy of rhs
	// for each thread and later fuse them together (reduction statement).
	for (std::size_t i = 0; i < _models.size(); ++i)
	{
		const unsigned int offset = _dofOffset[i];
		_jacFN[i].multiplySubtract(rhs + offset, rhs + finalOffset);
	}


	// Now, rhs contains the full intermediate solution y = L^{-1} b

	// Initialize temporary storage by copying over the fluxes
	std::fill_n(_tempState, finalOffset, 0.0);
	std::copy_n(rhs + finalOffset, numCouplingDOF(), _tempState + finalOffset);


	// ==== Step 3: Solve Schur-complement to get x_f = S^{-1} y_f
	// Column and particle parts remain unchanged.
	// The only thing to be done is the iterative (and approximate)
	// solution of the Schur complement system:
	//     S * x_f = y_f

	// Note that rhs is updated in-place with the solution of the Schur-complement
	// The temporary storage is only needed to hold the right hand side of the Schur-complement
	const double tolerance = std::sqrt(static_cast<double>(numDofs())) * outerTol * _schurSafety;

	// The network version of the schurCompletmentMatrixVector function need access to more information than the current interface
	// Instead of changing the interface a lambda function is used and closed over the additional variables
	auto schurComplementMatrixVectorPartial = [&, this](void* userData, double const* x, double* z) -> int
	{
		return ModelSystem::schurComplementMatrixVector(x, z, t, alpha, outerTol, weight, simState);
	};

	_gmres.matrixVectorMultiplier(schurComplementMatrixVectorPartial);

	// Reset error indicator as it is used in schurComplementMatrixVector()
	const int curError = totalErrorIndicatorFromLocal(_errorIndicator);
	std::fill(_errorIndicator.begin(), _errorIndicator.end(), 0);

	const int gmresResult = _gmres.solve(tolerance, weight + finalOffset, _tempState + finalOffset, rhs + finalOffset);

	// Set last cumulative error to all elements to restore state (in the end only total error matters)
	std::fill(_errorIndicator.begin(), _errorIndicator.end(), updateErrorIndicator(curError, gmresResult));

	// Reset temporary memory
	std::fill_n(_tempState, finalOffset, 0.0);

	// At this point, rhs contains the intermediate solution [y_0, ..., y_{N_z}, x_f]

	// ==== Step 4: Solve U * x = y by backward substitution
	// The fluxes are already solved and remain unchanged
#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), _models.size(), [=](std::size_t idxModel)
#else
	for (std::size_t idxModel = 0; idxModel < _models.size(); ++idxModel)
#endif
	{
		IUnitOperation* const m = _models[idxModel];
		const unsigned int offset = _dofOffset[idxModel];

		// Compute tempState_i = N_{i,f} * y_f
		_jacNF[idxModel].multiplyVector(rhs + finalOffset, _tempState + offset);

		// Apply N_i^{-1} to tempState_i
		const int linSolve =
			m->linearSolve(t, alpha, outerTol, _tempState + offset, weight + offset, applyOffset(simState, offset));
		_errorIndicator[idxModel] = updateErrorIndicator(_errorIndicator[idxModel], linSolve);

		// Compute rhs_i = y_i - N_i^{-1} * N_{i,f} * y_f = y_i - tempState_i
		const unsigned int offsetNext = _dofOffset[idxModel + 1];
		for (unsigned int i = offset; i < offsetNext; ++i)
		{
			rhs[i] -= _tempState[i];
		}
	} CADET_PARFOR_END;

	return totalErrorIndicatorFromLocal(_errorIndicator);
}

/**
* @brief Performs the matrix-vector product @f$ z = Sx @f$ with the Schur-complement @f$ S @f$ from the Jacobian
* @details The Schur-complement @f$ S @f$ is given by
*          @f[ \begin{align}
S &= J_f - J_{f,0} \, J_0^{-1} \, J_{0,f} - \sum_{p=1}^{N_z}{J_{f,p} \, J_p^{-1} \, J_{p,f}} \\
&= I - \sum_{p=0}^{N_z}{J_{f,p} \, J_p^{-1} \, J_{p,f}},
\end{align} @f]
*          where @f$ J_f = I @f$ is the identity matrix and the off-diagonal blocks @f$ J_{i,f} @f$
*          and @f$ J_{f,i} @f$ for @f$ i = 0, \dots, N_{z} @f$ are sparse.
*
*          The matrix-vector multiplication is executed in parallel as follows:
*              -# Compute @f$ J_{f,i} \, J_i^{-1} \, J_{i,f} @f$ independently (in parallel with respect to index @f$ i
@f$)
*              -# Subtract the result from @f$ z @f$ in a critical section to avoid race conditions
*
* @param [in] x Vector @f$ x @f$ the matrix @f$ S @f$ is multiplied with
* @param [out] z Result of the matrix-vector multiplication
* @return @c 0 if successful, any other value in case of failure
*/
int ModelSystem::schurComplementMatrixVector(double const* x, double* z, double t, double alpha, double outerTol,
											 double const* const weight, const ConstSimulationState& simState) const
{
	BENCH_SCOPE(_timerMatVec);

	// Copy x over to result z, which corresponds to the application of the identity matrix
	std::copy(x, x + numCouplingDOF(), z);

	// Inlets and outlets don't participate in the Schur solver since one of NF or FN for them is always 0
	// As a result we only have to work with items that have both an inlet and an outlet
#ifdef CADET_PARALLELIZE
	tbb::parallel_for(std::size_t(0), _inOutModels.size(), [=](std::size_t i)
#else
	for (std::size_t i = 0; i < _inOutModels.size(); ++i)
#endif
	{
		const unsigned int idxModel = _inOutModels[i];
		IUnitOperation* const m = _models[idxModel];
		const unsigned int offset = _dofOffset[idxModel];

		_jacNF[idxModel].multiplyVector(x, _tempState + offset);

		// Apply N_i^{-1} to tempState_i
		const int linSolve =
			m->linearSolve(t, alpha, outerTol, _tempState + offset, weight + offset, applyOffset(simState, offset));
		_errorIndicator[idxModel] = updateErrorIndicator(_errorIndicator[idxModel], linSolve);

		// Apply J_{f,i} and subtract results from z
		{
#ifdef CADET_PARALLELIZE
			SchurComplementMutex::scoped_lock l(_schurMutex);
#endif
			_jacFN[idxModel].multiplySubtract(_tempState + offset, z);
		}
	} CADET_PARFOR_END;

	return totalErrorIndicatorFromLocal(_errorIndicator);
}

/**
 * @brief Multiplies a vector with the full Jacobian of the entire system (i.e., @f$ \frac{\partial F}{\partial
 * y}\left(t, y, \dot{y}\right) @f$)
 * @details Actually, the operation @f$ z = \alpha \frac{\partial F}{\partial y} x + \beta z @f$ is performed.
 * @param [in] simTime Current simulation time point
 * @param [in] simState Simulation state vectors
 * @param [in] yS Vector @f$ x @f$ that is transformed by the Jacobian @f$ \frac{\partial F}{\partial y} @f$
 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ \frac{\partial F}{\partial y} @f$
 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ z @f$
 * @param [in,out] ret Vector @f$ z @f$ which stores the result of the operation
 */
void ModelSystem::multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState,
									   double const* yS, double alpha, double beta, double* ret)
{
	for (std::size_t idxModel = 0; idxModel < _models.size(); ++idxModel)
	{
		IUnitOperation* const m = _models[idxModel];
		const unsigned int offset = _dofOffset[idxModel];
		m->multiplyWithJacobian(simTime, simState, yS + offset, alpha, beta, ret + offset);
	}
	multiplyWithMacroJacobian(yS, alpha, beta, ret);
}

/**
 * @brief Multiplies a vector with the full time derivative Jacobian of the entire system (i.e., @f$ \frac{\partial
 * F}{\partial \dot{y}}\left(t, y, \dot{y}\right) @f$)
 * @details The operation @f$ z = \frac{\partial F}{\partial \dot{y}} x @f$ is performed.
 * @param [in] simTime Current simulation time point
 * @param [in] simState Simulation state vectors
 * @param [in] yS Vector @f$ x @f$ that is transformed by the Jacobian @f$ \frac{\partial F}{\partial \dot{y}} @f$
 * @param [in,out] ret Vector @f$ z @f$ which stores the result of the operation
 */
void ModelSystem::multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState,
												 double const* yS, double* ret)
{
	for (std::size_t idxModel = 0; idxModel < _models.size(); ++idxModel)
	{
		IUnitOperation* const m = _models[idxModel];
		const unsigned int offset = _dofOffset[idxModel];
		m->multiplyWithDerivativeJacobian(simTime, simState, yS + offset, ret + offset);
	}
	std::fill(ret + _dofOffset.back(), ret + numDofs(), 0.0);
}

#ifdef CADET_DEBUG

/**
 * @brief Generate full system Jacobian FD and multiplyWithJacobian
 * @details During debugging this allows you to generate the full jacobian and verify the jacobian structure
 *          is what it should be. The system uses FD and multiplyWithJacobian to create the full jacobian.
 *          Use this function with a debugger and pull the values out of memory to visualize it.
 *
 * @param [in] simTime Current simulation time point
 * @param [in] simState Simulation state vectors
 */
void ModelSystem::genJacobian(const SimulationTime& simTime, const ConstSimulationState& simState)
{
	// This method is only for debugging. No point in optimizing it
	const unsigned int size = numDofs();

	// Jacobians are saved in column-major ordering (i.e., each column is added to the array sequentially / columns are
	// stacked together)
	std::vector<double> jacobian(size * size, 0.0);
	std::vector<double> jacobianDot(size * size, 0.0);

	std::vector<double> jacobianFD(size * size, 0.0);
	std::vector<double> jacobianFDDot(size * size, 0.0);

	const double h = 1e-5;

	std::vector<double> f(size, 0.0);
	std::vector<double> fdot(size, 0.0);
	std::vector<double> fh(size, 0.0);
	std::vector<double> fhdot(size, 0.0);

	std::vector<double> res(size, 0.0);
	std::vector<double> resh(size, 0.0);

	// create Jacobian
	for (unsigned int i = 0; i < size; ++i)
	{
		// Clear res and resh
		std::fill(res.begin(), res.end(), 0.0);
		std::fill(resh.begin(), resh.end(), 0.0);

		// Copy y and yDot
		std::copy_n(simState.vecStateY, size, &f[0]);
		std::copy_n(simState.vecStateY, size, &fh[0]);

		std::copy_n(simState.vecStateYdot, size, &fdot[0]);
		std::copy_n(simState.vecStateYdot, size, &fhdot[0]);

		// Change ith entry
		double stepSize = h;
		if (f[i] != 0.0)
			stepSize = f[i] * h;

		f[i] -= stepSize / 2;
		fh[i] += stepSize / 2;

		residual(simTime, ConstSimulationState{&f[0], &fdot[0]}, &res[0]);
		residual(simTime, ConstSimulationState{&fh[0], &fhdot[0]}, &resh[0]);

		for (unsigned int j = 0; j < size; ++j)
		{
			jacobianFD[i * size + j] = (resh[j] - res[j]) / stepSize;
		}
	}

	// create JacobianDot
	for (unsigned int i = 0; i < size; ++i)
	{
		// Clear res and resh
		std::fill(res.begin(), res.end(), 0.0);
		std::fill(resh.begin(), resh.end(), 0.0);

		// Copy y and yDot
		std::copy_n(simState.vecStateY, size, &f[0]);
		std::copy_n(simState.vecStateY, size, &fh[0]);

		std::copy_n(simState.vecStateYdot, size, &fdot[0]);
		std::copy_n(simState.vecStateYdot, size, &fhdot[0]);

		// Change ith entry
		double stepSize = h;
		if (fdot[i] != 0.0)
			stepSize = fdot[i] * h;

		fdot[i] -= stepSize / 2;
		fhdot[i] += stepSize / 2;

		residual(simTime, ConstSimulationState{&f[0], &fdot[0]}, &res[0]);
		residual(simTime, ConstSimulationState{&fh[0], &fhdot[0]}, &resh[0]);

		for (unsigned int j = 0; j < size; ++j)
		{
			jacobianFDDot[i * size + j] = (resh[j] - res[j]) / stepSize;
		}
	}

	std::vector<double> unit(size, 0.0);

	for (unsigned int i = 0; i < size; ++i)
	{
		std::fill(res.begin(), res.end(), 0.0);
		// Clear res and resh
		unit[i] = 1.0;

		multiplyWithJacobian(simTime, simState, unit.data(), 1.0, 0.0, res.data());
		std::copy(res.begin(), res.end(), jacobian.begin() + i * size);

		unit[i] = 0.0;
	}

	for (unsigned int i = 0; i < size; ++i)
	{
		std::fill(res.begin(), res.end(), 0.0);
		// Clear res and resh
		unit[i] = 1.0;

		multiplyWithDerivativeJacobian(simTime, simState, unit.data(), res.data());
		std::copy(res.begin(), res.end(), jacobianDot.begin() + i * size);

		unit[i] = 0.0;
	}

	LOG(Debug) << "jacFD = " << log::MatrixPtr<double>(jacobianFD.data(), size, size, true);
	LOG(Debug) << "jacFDDot = " << log::MatrixPtr<double>(jacobianFDDot.data(), size, size, true);
	LOG(Debug) << "jac = " << log::MatrixPtr<double>(jacobian.data(), size, size, true);
	LOG(Debug) << "jacDot = " << log::MatrixPtr<double>(jacobianDot.data(), size, size, true);
}

/**
 * @brief Generate full system Jacobian with Sensitivities using FD and multiplyWithJacobian
 * @details During debugging this allows you to generate the full sensitivity jacobian and verify the jacobian structure
 *          is what it should be. The system uses FD and multiplyWithJacobian to create the full jacobian.
 *          Use this function with a debugger and pull the values out of memory to visualize it.
 *
 * @param [in] t Current time point
 * @param [in] simTime Current simulation time point
 * @param [in] simState Simulation state vectors
 * @param [in] residual vector
 * @param [in] yS Sensitivity State Vector
 * @param [in] ySdot Sensitivity State Vector
 * @param [in] resS Sensitivity residual vector
 * @param [in] adRes
 * @param [in] tmp1
 * @param [in] tmp2
 * @param [in] tmp3
 */
void ModelSystem::genJacobian(unsigned int nSens, const SimulationTime& simTime, const ConstSimulationState& simState,
							  double const* const res, const std::vector<const double*>& yS,
							  const std::vector<const double*>& ySdot, const std::vector<double*>& resS,
							  active* const adRes, double* const tmp1, double* const tmp2, double* const tmp3)
{
	// This method is only for debugging. Don't bother optimizing it
	const unsigned int size = numDofs();

	// Jacobians are saved in column-major ordering (i.e., each column is added to the array sequentially / columns are
	// stacked together)
	std::vector<std::vector<double>> jacobianFD(nSens, std::vector<double>(size * size));
	std::vector<std::vector<double>> jacobianFDDot(nSens, std::vector<double>(size * size));

	const double h = 1e-5;

	// -h/2
	std::vector<double> tmp1mh(size, 0.0);
	std::vector<double> tmp2mh(size, 0.0);
	std::vector<double> tmp3mh(size, 0.0);

	//  h/2
	std::vector<double> tmp1ph(size, 0.0);
	std::vector<double> tmp2ph(size, 0.0);
	std::vector<double> tmp3ph(size, 0.0);

	std::vector<active> adResmh(size, 0.0);
	std::vector<active> adResph(size, 0.0);

	std::vector<double*> ySmh(nSens);
	std::vector<double*> ySdotmh(nSens);
	std::vector<double*> resSmh(nSens);

	std::vector<double*> ySph(nSens);
	std::vector<double*> ySdotph(nSens);
	std::vector<double*> resSph(nSens);

	std::vector<const double*> CySmh(nSens);
	std::vector<const double*> CySdotmh(nSens);

	std::vector<const double*> CySph(nSens);
	std::vector<const double*> CySdotph(nSens);

	// Allocate memory
	for (unsigned int j = 0; j < nSens; ++j)
	{
		ySmh[j] = new double[size];
		ySdotmh[j] = new double[size];
		resSmh[j] = new double[size];

		ySph[j] = new double[size];
		ySdotph[j] = new double[size];
		resSph[j] = new double[size];
	}

	for (unsigned int j = 0; j < nSens; ++j)
	{
		CySmh[j] = ySmh[j];
		CySdotmh[j] = ySdotmh[j];
		CySph[j] = ySph[j];
		CySdotph[j] = ySdotph[j];
	}

	// create Jacobian
	for (unsigned int i = 0; i < size; ++i)
	{
		// need to make copies of yS, ySdot, resS, adRes, tmp1, tmp2, tmp3

		// adRes
		std::copy_n(adRes, size, &adResmh[0]);
		std::copy_n(adRes, size, &adResph[0]);

		// tmp1
		std::copy_n(tmp1, size, &tmp1mh[0]);
		std::copy_n(tmp1, size, &tmp1ph[0]);

		// tmp2
		std::copy_n(tmp2, size, &tmp2mh[0]);
		std::copy_n(tmp2, size, &tmp2ph[0]);

		// tmp3
		std::copy_n(tmp3, size, &tmp3mh[0]);
		std::copy_n(tmp3, size, &tmp3ph[0]);

		// Clear sync up
		for (unsigned int j = 0; j < nSens; ++j)
		{
			std::copy_n(yS[j], size, ySmh[j]);
			std::copy_n(yS[j], size, ySph[j]);

			std::copy_n(ySdot[j], size, ySdotmh[j]);
			std::copy_n(ySdot[j], size, ySdotph[j]);

			std::copy_n(resS[j], size, resSmh[j]);
			std::copy_n(resS[j], size, resSph[j]);
		}

		std::vector<double> stepSize(nSens, false);

		// Change ith entry
		for (unsigned int j = 0; j < nSens; ++j)
		{
			const double val = ySmh[j][i];
			if (val == 0.0)
			{
				ySmh[j][i] -= h / 2;
				ySph[j][i] += h / 2;
				stepSize[j] = h;
			}
			else
			{
				ySmh[j][i] -= val * h / 2;
				ySph[j][i] += val * h / 2;
				stepSize[j] = val * h;
			}
		}

		// clear jacobian

		// -h/2
		residualSensFwd(nSens, simTime, simState, res, CySmh, CySdotmh, resSmh, &adResmh[0], &tmp1mh[0], &tmp2mh[0],
						&tmp3mh[0]);

		// +h/2
		residualSensFwd(nSens, simTime, simState, res, CySph, CySdotph, resSph, &adResph[0], &tmp1ph[0], &tmp2ph[0],
						&tmp3ph[0]);

		for (unsigned int sens = 0; sens < nSens; ++sens)
		{
			for (unsigned int j = 0; j < size; ++j)
			{
				// Residual is negative so it has to be negated to get the correct jacobian
				jacobianFD[sens][i * size + j] = (resSph[sens][j] - resSmh[sens][j]) / stepSize[sens];
			}
		}
	}

	// create Jacobian
	for (unsigned int i = 0; i < size; ++i)
	{
		// need to make copies of yS, ySdot, resS, adRes, tmp1, tmp2, tmp3

		// adRes
		std::copy_n(adRes, size, &adResmh[0]);
		std::copy_n(adRes, size, &adResph[0]);

		// tmp1
		std::copy_n(tmp1, size, &tmp1mh[0]);
		std::copy_n(tmp1, size, &tmp1ph[0]);

		// tmp2
		std::copy_n(tmp2, size, &tmp2mh[0]);
		std::copy_n(tmp2, size, &tmp2ph[0]);

		// tmp3
		std::copy_n(tmp3, size, &tmp3mh[0]);
		std::copy_n(tmp3, size, &tmp3ph[0]);

		// Clear sync up
		for (unsigned int j = 0; j < nSens; ++j)
		{
			std::copy_n(yS[j], size, ySmh[j]);
			std::copy_n(yS[j], size, ySph[j]);

			std::copy_n(ySdot[j], size, ySdotmh[j]);
			std::copy_n(ySdot[j], size, ySdotph[j]);

			std::copy_n(resS[j], size, resSmh[j]);
			std::copy_n(resS[j], size, resSph[j]);
		}

		std::vector<double> stepSize(nSens, false);

		// Change ith entry
		for (unsigned int j = 0; j < nSens; ++j)
		{
			const double val = ySdotmh[j][i];
			if (val == 0.0)
			{
				ySdotmh[j][i] -= h / 2;
				ySdotph[j][i] += h / 2;
				stepSize[j] = h;
			}
			else
			{
				ySdotmh[j][i] -= val * h / 2;
				ySdotph[j][i] += val * h / 2;
				stepSize[j] = val * h;
			}
		}

		// clear jacobian

		// -h/2
		residualSensFwd(nSens, simTime, simState, res, CySmh, CySdotmh, resSmh, &adResmh[0], &tmp1mh[0], &tmp2mh[0],
						&tmp3mh[0]);

		// +h/2
		residualSensFwd(nSens, simTime, simState, res, CySph, CySdotph, resSph, &adResph[0], &tmp1ph[0], &tmp2ph[0],
						&tmp3ph[0]);

		for (unsigned int sens = 0; sens < nSens; ++sens)
		{
			for (unsigned int j = 0; j < size; ++j)
			{
				// Residual is negative so it has to be negated to get the correct jacobian
				jacobianFDDot[sens][i * size + j] = (resSph[sens][j] - resSmh[sens][j]) / stepSize[sens];
			}
		}
	}

	// Free memory

	for (unsigned int j = 0; j < nSens; ++j)
	{
		delete ySmh[j];
		delete ySdotmh[j];
		delete resSmh[j];

		delete ySph[j];
		delete ySdotph[j];
		delete resSph[j];
	}

	for (unsigned int i = 0; i < nSens; ++i)
	{
		LOG(Debug) << "jacSens" << i << " = " << log::MatrixPtr<double>(jacobianFD[i].data(), size, size, true);
		LOG(Debug) << "jacSensDot" << i << " = " << log::MatrixPtr<double>(jacobianFDDot[i].data(), size, size, true);
	}
}

#endif

} // namespace model

} // namespace cadet
