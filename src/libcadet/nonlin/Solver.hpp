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
 * Provides a unified interface for nonlinear equation solvers
 */

#ifndef LIBCADET_NONLINGENERALSOLVER_HPP_
#define LIBCADET_NONLINGENERALSOLVER_HPP_

#include <functional>
#include <string>

namespace cadet
{

class IParameterProvider;

namespace linalg
{
namespace detail
{
class DenseMatrixBase;
}
} // namespace linalg

namespace nonlin
{

/**
 * @brief General interface for all nonlinear equation solvers
 * @details Solves nonlinear equations @f$ F(x) = 0 @f$, where @f$ F\colon \mathds{R}^n \to \mathds{R}^n @f$
 *          and the Jacobian @f$ J_F(x) @f$ is available.
 */
class Solver
{
public:
	Solver()
	{
	}
	virtual ~Solver()
	{
	}

	/**
	 * @brief Returns the name of the nonlinear solver
	 * @details This name is also used to identify and create the nonlinear solver in the factory.
	 * @return Name of the nonlinear equation solver
	 */
	virtual const char* name() const = 0;

	/**
	 * @brief Configures the solver by extracting all parameters from the given @p paramProvider
	 * @details The scope of the cadet::IParameterProvider is left unchanged on return.
	 *          If no parameters are provided, the solver configuration is left unchanged.
	 *
	 * @param [in] paramProvider Parameter provider
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configure(IParameterProvider& paramProvider) = 0;

	/**
	 * @brief Returns the required amount of working memory (doubles) for a given problem size
	 * @param [in] problemSize Number of unknowns of the problem
	 * @return Number of required doubles (working memory)
	 */
	virtual unsigned int workspaceSize(unsigned int problemSize) const = 0;

	/**
	 * @brief Returns the number of additional tuning parameters
	 * @return Number of additional tuning parameters
	 */
	virtual unsigned int numTuningParameters() const = 0;

	/**
	 * @brief Solves the nonlinear equation system
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
	 * @param [in] tol Error tolerance used as termination criterion
	 * @param [in,out] point On entry initial guess, on exit solution or last iterate
	 * @param [in] workingMemory Additional memory, size is given by workspaceSize() function
	 * @param [in,out] jacMatrix Dense matrix used for storing and solving the linear systems (e.g., the Jacobian)
	 * @param [in] size Size of the problem (i.e., number of equations, length of residual, columns of Jacobian etc.)
	 * @return @c true if a solution meeting the residual tolerance was found, @c false otherwise
	 */
	virtual bool solve(std::function<bool(double const* const, double* const)> residual,
					   std::function<bool(double const* const, linalg::detail::DenseMatrixBase& jac)> jacobian,
					   double tol, double* const point, double* const workingMemory,
					   linalg::detail::DenseMatrixBase& jacMatrix, unsigned int size) const = 0;
};

/**
 * @brief Creates solvers with the given @p name
 * @details If a solver with the requested @p name does not exist, a default solver is returned.
 * @param [in] name Name of the solver
 * @return The requested solver or the default solver if a solver with the given name does not exist
 */
Solver* createSolver(const std::string& name);

} // namespace nonlin

} // namespace cadet

#endif // LIBCADET_NONLINGENERALSOLVER_HPP_
