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
 * Provides adaptive trust-region Newton methods for solving nonlinear equation systems
 */

#ifndef LIBCADET_ADAPTRUSTNEWTON_HPP_
#define LIBCADET_ADAPTRUSTNEWTON_HPP_

#include "common/CompilerSpecific.hpp"
#include "nonlin/Solver.hpp"

#include <cmath>
#include <functional>
#include <algorithm>
#include <vector>

#include "linalg/Norms.hpp"

namespace cadet
{

namespace nonlin
{

/**
 * @brief Iterate output policy that does nothing
 */
struct VoidNewtonIterateOutputPolicy
{
	inline static void outerIteration(unsigned int idxIter, double residualNorm, double const* const res,
									  double const* const x, double const* const dx, unsigned int size)
	{
	}
	inline static void innerIteration(unsigned int idxIter, double residualNorm, double const* const res,
									  double const* const x, double const* const dx, double damping, double mu,
									  unsigned int size)
	{
	}
};

/**
 * @brief Solves nonlinear equations using a residual oriented descent based global adaptive trust-region Newton method
 * @details This is an implementation of the NLEQ-RES algorithm described in \cite Deuflhard2011 (p. 131).
 *          It is a global adaptive trust-region Newton method based on affine contravariance.
 *          In the framework of affine invariance, the solution of the inner trust region problems is
 *          a point on the ordinary Newton step line segment. Thus, in the end this algorithm is an adaptively
 *          damped Newton method with global convergence.
 *
 *          This algorithm is designed to solve the nonlinear equation system
 *          @f[\begin{align} f(x) = 0, \qquad f: \mathbb{R}^n \to \mathbb{R}^n. \end{align}@f]
 *          A solution is indicated by the error test
 *          @f[\begin{align} \left\lVert f(x) \right\rVert_{\ell^2} \leq \text{tol} \end{align}@f]
 *          with a given tolerance (@p resTol).
 *
 *          The residuals @f$ f(x) @f$ are evaluated using the given function @p residual.
 *          The resulting linear equation systems
 *          @f[\begin{align} J_f(x) \Delta x = r, \end{align}@f]
 *          where @f$ J_f(x) @f$ denotes the Jacobian of @f$ f @f$ at point @f$ x @f$ and @f$ r = f(x) @f$
 *          is a given right hand side, are solved using the given function @p jacobianSolver.
 *
 *          The initial damping factor @p damping and the minimal damping factor @p minDamping
 *          can be chosen based on reference values for different problem difficulties:
 *          | Difficulty          | damping | minDamping |
 *          | ------------------- | ------- | ---------- |
 *          | Mildly nonlinear    | 1.0     | 1e-4       |
 *          | Highly nonlinear    | 1e-2    | 1e-4       |
 *          | Extremely nonlinear | 1e-4    | 1e-8       |
 *          In the case of extremely nonlinear problems, a restricted error test is recommended (not available
 *          through public interface).
 *
 *          Note that, since this method is based on residual monotonicity, this method can fail
 *          in practice even if the conditions guaranteeing global convergence are satisfied.
 *          This is, for example, the case for ill-conditioned Jacobians, since then @f$ x_{k+1} \approx x_k @f$
 *          and the iteration is stalled. See pp. 137 in the Deuflhard book.
 *          A more robust, but also more costly, algorithm is implemented in robustAdaptiveTrustRegionNewtonMethod().
 * @param [in] residual Function providing the residual @f$ f(x) @f$ at position @f$ x @f$ of the nonlinear equation
 *             system @f$ f(x) = 0 @f$ to be solved.
 *             The signature of the function is
 *             `bool residual(double const* const x, double* const r)`
 *             where the return value communicates whether the evaluation has been successful.
 *             On exit, the residual is returned in-place in @f$ r @f$.
 *             If an error is indicated by returning @c false, the algorithm aborts with failure.
 * @param [in] jacobianSolver Function calculating the solution of the linear system @f$ J_f(x) \Delta x = r@f$
 *             with Jacobian matrix @f$ J_f(x) @f$ at position @f$ x @f$ for a given right hand side @f$ r @f$.
 *             The signature of the function is
 *             `bool jacobianSolver(double const* const x, double* const r)`
 *             where the return value communicates whether the system has been solved correctly.
 *             On exit, the solution of the linear system is returned in-place in @f$ r @f$ which on entry
 *             contains the right hand side vector of the linear system.
 *             If an error is indicated by returning @c false, the algorithm aborts with failure.
 * @param [in] maxIter Maximum number of iterations
 * @param [in] resTol Termination criterion on the residual @f$\ell^2@f$-norm
 * @param [in] damping Initial damping factor (see details for advice)
 * @param [in] minDamping Minimal damping factor (see details for advice)
 * @param [in,out] point On entry initial guess, on exit solution or last iterate
 * @param [in] workingMemory Additional memory of size @f$ 4n @f$ required for performing the iterations, where @f$ n
 * @f$ is the problem @p size
 * @param [in] size Size of the problem (i.e., number of equations, length of residual, columns of Jacobian etc.)
 * @tparam IterateOutputPolicy Policy that handles output of intermediate values (useful for debugging), see
 * VoidNewtonIterateOutputPolicy
 * @return @c true if a solution meeting the residual tolerance was found, @c false otherwise
 * @todo Make algorithm more robust by providing means to solver and residual functions to trigger decrease of damping
 * factor (e.g. for negative concentrations)
 * @todo Implement scaling of linear systems and norms
 */
template <typename IterateOutputPolicy = VoidNewtonIterateOutputPolicy>
bool adaptiveTrustRegionNewtonMethod(std::function<bool(double const* const, double* const)> residual,
									 std::function<bool(double const* const, double* const)> jacobianSolver,
									 unsigned int maxIter, double resTol, double damping, double minDamping,
									 double* const point, double* const workingMemory, unsigned int size)
{
	//		const double thetaMax = 0.25; // Determines whether switch to quasi-newton updates is performed (not
	// implemented)
	const bool restricted = false; // Determines if restricted monotonicity test is used
	double mu = 0.0;
	double lastResidualNorm = 0.0;

	// Split working memory into parts
	double* const residualMem = workingMemory;
	double* const dx = workingMemory + size;
	double* const trialPoint = workingMemory + 2 * size;
	double* const lastResidual = workingMemory + 3 * size;

	// Evaluate residual
	if (!residual(point, lastResidual))
		return false;

	// Copy residual
	std::copy(lastResidual, lastResidual + size, dx);

	double residualNorm = linalg::l2Norm(lastResidual, size);

	IterateOutputPolicy::outerIteration(0, residualNorm, lastResidual, point, trialPoint, size);

	// Main loop
	for (unsigned int kIter = 0; kIter < maxIter; ++kIter)
	{
		// Convergence test
		if (residualNorm <= resTol)
			return true;

		// Solve F'(x) * dx = F(x)
		// Since we have omitted the minus sign here, we have to take care of the negation later
		if (!jacobianSolver(point, dx))
			return false;

		if (kIter > 0)
		{
			// Compute prediction of damping factor
			mu *= lastResidualNorm / residualNorm;
			damping = std::min(1.0, mu);
		}

		lastResidualNorm = residualNorm;

		// Line search loop: Use regularity test as abort condition
		while (damping >= minDamping)
		{
			// Compute x_{n+1} = x_n + lambda * dx
			// Note that we are subtracting here because we have omitted the negation in the Jacobian solution
			for (unsigned int i = 0; i < size; ++i)
				trialPoint[i] = point[i] - damping * dx[i];

			// Evaluate residual
			if (!residual(trialPoint, residualMem))
				return false;

			residualNorm = linalg::l2Norm(residualMem, size);

			IterateOutputPolicy::innerIteration(kIter + 1, residualNorm, residualMem, trialPoint, dx, damping, mu,
												size);

			// Calculate monitoring quantities
			const double theta = residualNorm / lastResidualNorm;

			mu = 0.0;
			const double factor = 1.0 - damping;
			for (unsigned int i = 0; i < size; ++i)
			{
				mu += sqr(residualMem[i] - factor * lastResidual[i]);
			}
			mu = 0.5 * lastResidualNorm * damping * damping / std::sqrt(mu);

			if ((!restricted && (theta >= 1.0)) || (restricted && (theta > 1.0 - damping * 0.25)))
			{
				// Shrink damping and try again
				damping = std::min(mu, 0.5 * damping);
				continue;
			}

			const double dampingNew = std::min(1.0, mu);
			if (dampingNew >= 4.0 * damping)
			{
				damping = dampingNew;
				continue;
			}

			// Accept the step
			break;
		}

		// Check if regularity test failed or line search loop was aborted because of other reasons
		if (damping < minDamping)
			return false;

		IterateOutputPolicy::outerIteration(kIter + 1, residualNorm, residualMem, trialPoint, dx, size);

		// Copy accepted point and last residual
		std::copy(trialPoint, trialPoint + size, point);
		std::copy(residualMem, residualMem + size, lastResidual);
		std::copy(residualMem, residualMem + size, dx);
	}
	return false;
}

/**
 * @brief Uses an adaptive trust-region Newton method for solving nonlinear equations
 * @details Wraps adaptiveTrustRegionNewtonMethod() function.
 */
class AdaptiveTrustRegionNewtonSolver : public Solver
{
public:
	AdaptiveTrustRegionNewtonSolver();
	virtual ~AdaptiveTrustRegionNewtonSolver();

	static const char* identifier()
	{
		return "ATRN_RES";
	}
	virtual const char* name() const
	{
		return AdaptiveTrustRegionNewtonSolver::identifier();
	}
	virtual bool configure(IParameterProvider& paramProvider);

	virtual unsigned int workspaceSize(unsigned int problemSize) const
	{
		// Method requires 4 * problemSize but we add an additional problemSize for preconditioning
		return 5 * problemSize;
	}

	virtual unsigned int numTuningParameters() const
	{
		return 3;
	}

	virtual bool solve(std::function<bool(double const* const, double* const)> residual,
					   std::function<bool(double const* const, linalg::detail::DenseMatrixBase& jac)> jacobian,
					   double tol, double* const point, double* const workingMemory,
					   linalg::detail::DenseMatrixBase& jacMatrix, unsigned int size) const;

protected:
	double _initDamping;   //!< Initial damping factor
	double _minDamping;    //!< Minimal damping factor
	unsigned int _maxIter; //!< Maximum number of iterations
};

/**
 * @brief Solves nonlinear equations using an error oriented descent based global adaptive trust-region Newton method
 * @details This is an implementation of the NLEQ-ERR algorithm described in \cite Deuflhard2011 (pp. 148).
 *          It is a global adaptive trust-region Newton method based on affine covariance.
 *          In the framework of affine invariance, the solution of the inner trust region problems is
 *          a point on the ordinary Newton step line segment. Thus, in the end this algorithm is an adaptively
 *          damped Newton method with global convergence.
 *
 *          This algorithm is designed to solve the nonlinear equation system
 *          @f[\begin{align} f(x) = 0, \qquad f: \mathbb{R}^n \to \mathbb{R}^n. \end{align}@f]
 *          A solution is indicated by the error test
 *          @f[\begin{align} \left\lVert \Delta x \right\rVert_{\ell^2} \leq \text{tol} \end{align}@f]
 *          with a given tolerance (@p errTol).
 *
 *          The residuals @f$ f(x) @f$ are evaluated using the given function @p residual.
 *          The resulting linear equation systems
 *          @f[\begin{align} J_f(x) \Delta x = r, \end{align}@f]
 *          where @f$ J_f(x) @f$ denotes the Jacobian of @f$ f @f$ at point @f$ x @f$ and @f$ r = f(x) @f$
 *          is a given right hand side, are solved using the given function @p jacobianSolver.
 *          In this algorithm, the linear system has to be solved several times with varying right hand sides @f$ r @f$.
 *          A repeated solution is requested by the function @p jacobianResolver, which is always preceeded by a
 *          call to @p jacobianSolver. Thus, a dense matrix can be factorized once in @p jacobianSolver and used
 *          in subsequent calls of @p jacobianResolver.
 *
 *          The initial damping factor @p damping and the minimal damping factor @p minDamping
 *          can be chosen based on reference values for different problem difficulties:
 *          | Difficulty          | damping | minDamping |
 *          | ------------------- | ------- | ---------- |
 *          | Mildly nonlinear    | 1.0     | 1e-4       |
 *          | Highly nonlinear    | 1e-2    | 1e-4       |
 *          | Extremely nonlinear | 1e-4    | 1e-8       |
 *          In the case of extremely nonlinear problems, a restricted error test is recommended (not available
 *          through public interface).
 * @param [in] residual Function providing the residual @f$ f(x) @f$ at position @f$ x @f$ of the nonlinear equation
 *             system @f$ f(x) = 0 @f$ to be solved.
 *             The signature of the function is
 *             `bool residual(double const* const x, double* const r)`
 *             where the return value communicates whether the evaluation has been successful.
 *             On exit, the residual is returned in-place in @f$ r @f$.
 *             If an error is indicated by returning @c false, the algorithm aborts with failure.
 * @param [in] jacobianSolver Function calculating the solution of the linear system @f$ J_f(x) \Delta x = r@f$
 *             with Jacobian matrix @f$ J_f(x) @f$ at position @f$ x @f$ for a given right hand side @f$ r @f$.
 *             The signature of the function is
 *             `bool jacobianSolver(double const* const x, double* const r)`
 *             where the return value communicates whether the system has been solved correctly.
 *             On exit, the solution of the linear system is returned in-place in @f$ r @f$ which on entry
 *             contains the right hand side vector of the linear system.
 *             If an error is indicated by returning @c false, the algorithm aborts with failure.
 * @param [in] jacobianResolver Function calculating the solution of the linear system @f$ J_f(x) \Delta x = r@f$
 *             with Jacobian matrix @f$ J_f(x) @f$ at the position @f$ x @f$ of a previous call to @p jacobianSolver
 *             for a given right hand side @f$ r @f$. The signature of the function is
 *             `bool jacobianResolver(double* const r)`
 *             where the return value communicates whether the system has been solved correctly.
 *             On exit, the solution of the linear system is returned in-place in @f$ r @f$ which on entry
 *             contains the right hand side vector of the linear system.
 *             If an error is indicated by returning @c false, the algorithm aborts with failure.
 * @param [in] maxIter Maximum number of iterations
 * @param [in] errTol Termination criterion on the Newton step size @f$\ell^2@f$-norm
 * @param [in] damping Initial damping factor (see details for advice)
 * @param [in] minDamping Minimal damping factor (see details for advice)
 * @param [in,out] point On entry initial guess, on exit solution or last iterate
 * @param [in] workingMemory Additional memory of size @f$ 4n @f$ required for performing the iterations, where @f$ n
 * @f$ is the problem @p size
 * @param [in] size Size of the problem (i.e., number of equations, length of residual, columns of Jacobian etc.)
 * @tparam IterateOutputPolicy Policy that handles output of intermediate values (useful for debugging), see
 * VoidNewtonIterateOutputPolicy
 * @return @c true if a solution meeting the residual tolerance was found, @c false otherwise
 * @todo Make algorithm more robust by providing means to solver and residual functions to trigger decrease of damping
 * factor (e.g. for negative concentrations)
 * @todo Implement scaling of linear systems and norms
 */
template <typename IterateOutputPolicy = VoidNewtonIterateOutputPolicy>
bool robustAdaptiveTrustRegionNewtonMethod(std::function<bool(double const* const, double* const)> residual,
										   std::function<bool(double const* const, double* const)> jacobianSolver,
										   std::function<bool(double* const)> jacobianResolver, unsigned int maxIter,
										   double errTol, double damping, double minDamping, double* const point,
										   double* const workingMemory, unsigned int size)
{
	//		const double thetaMax = 0.25; // Determines whether switch to quasi-newton updates is performed (not
	// implemented)
	const bool restricted = false; // Determines if restricted monotonicity test is used
	double mu = 0.0;

	// Split working memory into parts
	double* const dx = workingMemory;
	double* const trialPoint = workingMemory + size;
	double* const lastDxBar = workingMemory + 2 * size;
	double* const lastResidual = workingMemory + 3 * size;

	// Evaluate residual
	if (!residual(point, dx))
		return false;

	double errNorm = 0.0;
	double lastErrNorm = 0.0;
	double errNormTrial = 0.0;

	// Main loop
	for (unsigned int kIter = 0; kIter < maxIter; ++kIter)
	{
		// Solve F'(x) * dx = F(x)
		// Since we have omitted the minus sign here, we have to take care of the negation later
		if (!jacobianSolver(point, dx))
			return false;

		lastErrNorm = errNorm;
		errNorm = linalg::l2Norm(dx, size);

		IterateOutputPolicy::outerIteration(kIter, errNorm, dx, point, dx, size);

		// Convergence test
		if (errNorm <= errTol)
		{
			// Solution is x + dx = x - (-dx)
			// Note that we have to negate dx here since we didn't do that when solving with the Jacobian above
			for (unsigned int i = 0; i < size; ++i)
				point[i] -= dx[i];
			return true;
		}

		if (kIter > 0)
		{
			// Compute prediction of damping factor
			mu = 0.0;
			for (unsigned int i = 0; i < size; ++i)
			{
				mu += sqr(lastDxBar[i] - dx[i]);
			}
			mu = (lastErrNorm * errNormTrial) / (std::sqrt(mu) * errNorm) * damping;

			damping = std::min(1.0, mu);
		}

		// Line search loop: Use regularity test as abort condition
		while (damping >= minDamping)
		{
			// Compute trial point
			for (unsigned int i = 0; i < size; ++i)
				trialPoint[i] = point[i] - damping * dx[i];

			// Evaluate residual and solve linear system
			if (!residual(trialPoint, lastResidual))
				return false;

			std::copy(lastResidual, lastResidual + size, lastDxBar);

			if (!jacobianResolver(lastDxBar))
				return false;

			errNormTrial = linalg::l2Norm(lastDxBar, size);
			const double theta = errNormTrial / errNorm;

			IterateOutputPolicy::innerIteration(kIter + 1, errNormTrial, lastDxBar, trialPoint, dx, damping, mu, size);

			// Compute new mu value
			const double factor = 1.0 - damping;
			for (unsigned int i = 0; i < size; ++i)
			{
				mu += sqr(lastDxBar[i] - factor * dx[i]);
			}
			mu = 0.5 * errNorm * damping * damping / std::sqrt(mu);

			if ((!restricted && (theta >= 1.0)) || (restricted && (theta > 1.0 - damping * 0.25)))
			{
				// Shrink damping and try again
				damping = std::min(mu, 0.5 * damping);
				continue;
			}

			const double dampingNew = std::min(1.0, mu);

			if ((damping == 1.0) && (dampingNew == 1.0) && (errNormTrial <= errTol))
			{
				// Convergence detected
				// Note that we have to negate dx here since we didn't do that when solving with the Jacobian above
				for (unsigned int i = 0; i < size; ++i)
					point[i] -= lastDxBar[i];
				return true;
			}

			if (dampingNew >= 4.0 * damping)
			{
				damping = dampingNew;
				continue;
			}

			// Accept the step
			break;
		}

		// Check if regularity test failed or line search loop was aborted because of other reasons
		if (damping < minDamping)
			return false;

		// Copy accepted point
		std::copy(trialPoint, trialPoint + size, point);
		std::copy(lastResidual, lastResidual + size, dx);
	}

	return false;
}

/**
 * @brief Uses a more robust adaptive trust-region Newton method for solving nonlinear equations
 * @details Wraps robustAdaptiveTrustRegionNewtonMethod() function.
 */
class RobustAdaptiveTrustRegionNewtonSolver : public Solver
{
public:
	RobustAdaptiveTrustRegionNewtonSolver();
	virtual ~RobustAdaptiveTrustRegionNewtonSolver();

	static const char* identifier()
	{
		return "ATRN_ERR";
	}
	virtual const char* name() const
	{
		return RobustAdaptiveTrustRegionNewtonSolver::identifier();
	}
	virtual bool configure(IParameterProvider& paramProvider);

	virtual unsigned int workspaceSize(unsigned int problemSize) const
	{
		// Method requires 4 * problemSize but we add an additional problemSize for preconditioning
		return 5 * problemSize;
	}

	virtual unsigned int numTuningParameters() const
	{
		return 3;
	}

	virtual bool solve(std::function<bool(double const* const, double* const)> residual,
					   std::function<bool(double const* const, linalg::detail::DenseMatrixBase& jac)> jacobian,
					   double tol, double* const point, double* const workingMemory,
					   linalg::detail::DenseMatrixBase& jacMatrix, unsigned int size) const;

protected:
	double _initDamping;   //!< Initial damping factor
	double _minDamping;    //!< Minimum damping factor
	unsigned int _maxIter; //!< Maximum number of iterations
};

} // namespace nonlin

} // namespace cadet

#endif // LIBCADET_ADAPTRUSTNEWTON_HPP_
