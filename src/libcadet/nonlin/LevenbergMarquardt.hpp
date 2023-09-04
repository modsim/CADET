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
 * Provides Levenberg-Marquardt methods for solving nonlinear equation systems
 */

#ifndef LIBCADET_LEVENBERGMARQUARDT_HPP_
#define LIBCADET_LEVENBERGMARQUARDT_HPP_

#include "common/CompilerSpecific.hpp"
#include "nonlin/Solver.hpp"

#include <cmath>
#include <functional>
#include <vector>
#include <algorithm>
#include <limits>

#include "linalg/DenseMatrix.hpp"
#include "linalg/Norms.hpp"

namespace cadet
{

namespace nonlin
{
namespace detail
{
inline bool checkLevenbergMarquardtConvergence(double trialSumSq, double residualSumSq, double maxGrad, double resTol,
											   double tolOpt, double relFactor)
{
	// Check if residual is small enough
	if (trialSumSq <= resTol * resTol)
		return true;
	// Check if gradient is (comparably) small
	if (maxGrad <= relFactor * tolOpt)
		return true;
	// Check if relative decrease in residual is small
	if (std::abs(trialSumSq - residualSumSq) <= resTol * residualSumSq)
		return true;

	return false;
}
} // namespace detail

/**
 * @brief Iterate output policy that does nothing
 */
struct VoidLMIterateOutputPolicy
{
	inline static void outerIteration(unsigned int idxIter, double residualNorm, double const* const res,
									  double const* const x, double const* const dx, double damping, unsigned int size)
	{
	}
	inline static void innerIteration(unsigned int idxIter, double residualNorm, double const* const res,
									  double const* const x, double const* const dx, double damping, unsigned int size)
	{
	}
};

/**
 * @brief Solves nonlinear equations using a Levenberg-Marquardt method
 * @details This is a port of the implementation provided by MATLAB.
 *
 *          This algorithm is designed to solve the nonlinear least squares problem
 *          @f[\begin{align} \text{min}_x \lVert f(x) \rVert_2^2, \qquad f: \mathbb{R}^n \to \mathbb{R}^n.
 * \end{align}@f] If the nonlinear equation system
 *          @f[\begin{align} f(x) = 0 \end{align}@f]
 *          is solvable, the least squares problem is equivalent to finding such a solution.
 *          A solution is indicated by the error test
 *          @f[\begin{align} \left\lVert f(x) \right\rVert_{\ell^2} \leq \text{tol} \end{align}@f]
 *          with a given tolerance (@p resTol). However, since the system may not possess a solution,
 *          the algorithm also stops on small gradients
 *          @f[ \begin{align} \lVert \nabla \frac{1}{2}\lVert f(x) \rVert_2^2 \rVert_\infty = \lVert
 * \left[J_f(x)\right]^T f(x) \rVert_\infty \leq \text{tol} \end{align} @f] and stagnating merit function decrease
 *          @f[ \begin{align} \left\lvert \left\lVert f\left(x_{n+1} \right) \right\rVert_2^2 - \left\lVert f\left(x_{n}
 * \right) \right\rVert_2^2 \right\rvert \leq \text{tol} \left\lVert f\left(x_{n} \right) \right\rVert_2^2. \end{align}
 * @f]
 *
 *          The residuals @f$ f(x) @f$ are evaluated using the given function @p residual.
 *          The Jacobian @f$ J_f(x) @f$ of @f$ f @f$ at point @f$ x @f$ is evaluated using the
 *          given function @p jacobian.
 *
 *          The algorithm uses a damping factor @f$ \lambda @f$ to interpolate between gradient descent and
 *          Newton iterations. The damping factor @f$ \lambda @f$ is adapted in a line-search-like fashion.
 *          In each iteration, a linear least squares system
 *          @f[\begin{align} \text{min}_{\Delta x} \lVert \begin{pmatrix}J_f(x) \\ \lambda I \end{pmatrix} \Delta x +
 * \begin{pmatrix}f(x) \\ 0 \end{pmatrix} \rVert_2^2 \end{align}@f] is solved, which is equivalent to solving
 *          @f[\begin{align} \left(\left[ J_f(x) \right]^T J_f(x) + \lambda I\right) \Delta x = -\left[ J_f(x) \right]^T
 * f(x) \end{align}.@f]
 *
 *          The initial damping factor @p damping is usually chosen as @c 1e-2.
 * @param [in] residual Function providing the residual @f$ f(x) @f$ at position @f$ x @f$ of the nonlinear equation
 *             system @f$ f(x) = 0 @f$ to be solved.
 *             The signature of the function is
 *             `bool residual(double const* const x, double* const r)`
 *             where the return value communicates whether the evaluation has been successful.
 *             On exit, the residual is returned in-place in @f$ r @f$.
 *             If an error is indicated by returning @c false, the algorithm aborts with failure.
 * @param [in] jacobian Function calculating the system Jacobian @f$ J_f(x) @f$ at position @f$ x@f$.
 *             The signature of the function is
 *             `bool jacobianSolver(double const* const x, linalg::detail::DenseMatrixBase& jac)`
 *             where the return value communicates whether the Jacobian has been computed correctly.
 *             If an error is indicated by returning @c false, the algorithm aborts with failure.
 * @param [in] maxIter Maximum number of iterations
 * @param [in] resTol Termination criterion on the residual @f$\ell^2@f$-norm
 * @param [in] damping Initial damping factor (see details for advice)
 * @param [in,out] point On entry initial guess, on exit solution or last iterate
 * @param [in] workingMemory Additional memory of size @f$ 7n @f$ required for performing the iterations, where @f$ n
 * @f$ is the problem @p size
 * @param [in,out] jacMatrix Dense matrix used for storing and solving the linear systems (e.g., the Jacobian)
 * @param [in] size Size of the problem (i.e., number of equations, length of residual, columns of Jacobian etc.)
 * @tparam IterateOutputPolicy Policy that handles output of intermediate values (useful for debugging), see
 * VoidLMIterateOutputPolicy
 * @return @c true if a solution meeting the residual tolerance was found, @c false otherwise
 * @todo Implement scaling of linear systems and norms
 */
template <typename IterateOutputPolicy = VoidLMIterateOutputPolicy>
bool levenbergMarquardt(std::function<bool(double const* const, double* const)> residual,
						std::function<bool(double const* const, linalg::detail::DenseMatrixBase& jac)> jacobian,
						unsigned int maxIter, double resTol, double damping, double* const point,
						double* const workingMemory, linalg::detail::DenseMatrixBase& jacMatrix, unsigned int size)
{
	const double dampingMultiplier = 10.0;
	const unsigned int maxTrials = 10;

	// Split working memory into parts
	double* const residualMem = workingMemory;
	double* const newResidual = workingMemory + 2 * size;
	double* const dx = workingMemory + 3 * size;
	double* const trialPoint = workingMemory + 4 * size;
	double* const workspace = workingMemory + 5 * size; // 2 * size

	unsigned int nTrials = 0;
	const double tolOpt = resTol * 1e-4;

	// Allocate matrix for linear least squares problems
	// Upper part of matrix is the system Jacobian, lower part is identity matrix for damping
	linalg::DenseMatrix factoredJac;
	factoredJac.resize(2 * size, size);

	// Get residual and Jacobian
	if (!residual(point, residualMem))
		return false;
	if (!jacobian(point, jacMatrix))
		return false;

	// Compute residual sum of squares
	double residualSumSq = linalg::l2NormSquared(residualMem, size);

	// Compute gradient J^T r
	jacMatrix.transposedMultiplyVector(residualMem, newResidual);
	const double relFactor =
		std::max(linalg::linfNorm(newResidual, size), std::sqrt(std::numeric_limits<double>::epsilon()));

	IterateOutputPolicy::outerIteration(0, std::sqrt(residualSumSq), residualMem, point, nullptr, damping, size);

	// Check if initial point is already optimal
	if (detail::checkLevenbergMarquardtConvergence(std::numeric_limits<double>::infinity(), residualSumSq,
												   linalg::linfNorm(newResidual, size), resTol, tolOpt, relFactor))
		return true;

	// Main loop
	for (unsigned int kIter = 0; kIter < maxIter; ++kIter)
	{
		// Copy jacobian matrix into system matrix and reset the rest
		factoredJac.submatrixAssign(jacMatrix, 0, 0, jacMatrix.rows(), jacMatrix.columns());
		factoredJac.submatrixSetAll(0.0, size, 0, jacMatrix.rows(), jacMatrix.rows());

		factoredJac.rowScaleFactors(workspace, size);
		factoredJac.scaleRows(workspace, size);

		// Add scaled identity matrix for damping and prepare right hand side
		const double sqrtDamping = std::sqrt(damping);
		for (unsigned int i = 0; i < size; ++i)
		{
			factoredJac.native(i + size, i) = sqrtDamping;

			// Negate residual
			dx[i] = -residualMem[i] / workspace[i];

			// Set lower part of right hand side to zero
			dx[i + size] = 0.0;
		}

		// Compute step
		if (!factoredJac.leastSquaresSolve(dx, workspace, 2 * size))
			return false;

		// Compute trial step x_trial = x_cur + step
		for (unsigned int i = 0; i < size; ++i)
			trialPoint[i] = point[i] + dx[i];

		// Evaluate residual at new position
		if (!residual(trialPoint, newResidual))
			return false;

		const double trialSumSq = linalg::l2NormSquared(newResidual, size);
		if (trialSumSq < residualSumSq)
		{
			// Copy residual and trial point over
			std::copy(newResidual, newResidual + size, residualMem);
			std::copy(trialPoint, trialPoint + size, point);

			if (nTrials == 0)
			{
				// Previous step was successful, so try to reduce damping
				damping /= dampingMultiplier;
			}
			else
			{
				// Previous step was not successful, so we still need to evaluate the Jacobian
				if (!jacobian(trialPoint, jacMatrix))
					return false;
			}

			// Compute gradient J^T r
			jacMatrix.transposedMultiplyVector(residualMem, newResidual);
			const double maxGrad = linalg::linfNorm(newResidual, size);

			IterateOutputPolicy::outerIteration(kIter + 1, std::sqrt(trialSumSq), residualMem, point, dx, damping,
												size);

			// Check convergence
			if (detail::checkLevenbergMarquardtConvergence(trialSumSq, residualSumSq, maxGrad, resTol, tolOpt,
														   relFactor))
				return true;

			// Update residual sum of squares
			residualSumSq = trialSumSq;

			// This was a successful step
			nTrials = 0;
		}
		else
		{
			IterateOutputPolicy::innerIteration(kIter + 1, std::sqrt(trialSumSq), residualMem, trialPoint, dx, damping,
												size);

			// We have failed, let's increase damping and try again
			damping *= dampingMultiplier;
			++nTrials;

			// Check abort conditions
			if (nTrials >= maxTrials)
				break;
		}
	}

	return false;
}

/**
 * @brief Uses the Levenberg-Marquardt method for solving nonlinear equations
 * @details Wraps levenbergMarquardt() function.
 */
class LevenbergMarquardtSolver : public Solver
{
public:
	LevenbergMarquardtSolver();
	virtual ~LevenbergMarquardtSolver();

	static const char* identifier()
	{
		return "LEVMAR";
	}
	virtual const char* name() const
	{
		return LevenbergMarquardtSolver::identifier();
	}
	virtual bool configure(IParameterProvider& paramProvider);

	virtual unsigned int workspaceSize(unsigned int problemSize) const
	{
		return 7 * problemSize;
	}

	virtual unsigned int numTuningParameters() const
	{
		return 2;
	}

	virtual bool solve(std::function<bool(double const* const, double* const)> residual,
					   std::function<bool(double const* const, linalg::detail::DenseMatrixBase& jac)> jacobian,
					   double tol, double* const point, double* const workingMemory,
					   linalg::detail::DenseMatrixBase& jacMatrix, unsigned int size) const;

protected:
	double _initDamping;   //!< Initial damping factor
	unsigned int _maxIter; //!< Maximum number of iterations
};

} // namespace nonlin

} // namespace cadet

#endif // LIBCADET_LEVENBERGMARQUARDT_HPP_
