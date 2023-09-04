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
 * Provides a GMRES (Generalized Minimal Residual) algorithm wrapper
 */

#ifndef LIBCADET_GMRES_HPP_
#define LIBCADET_GMRES_HPP_

#include "cadet/cadetCompilerInfo.hpp"
#include "cadet/Exceptions.hpp"

#include <functional>

// Forward declare SUNDIALS types
#if CADET_SUNDIALS_IFACE == 2
typedef struct _SpgmrMemRec SpgmrMemRec;
#elif CADET_SUNDIALS_IFACE == 3
typedef struct _generic_SUNLinearSolver* SUNLinearSolver;
#endif

typedef struct _generic_N_Vector* N_Vector;

namespace cadet
{

namespace linalg
{

/**
 * @brief Supported orthogonalization methods
 */
enum class Orthogonalization : unsigned int
{
	ClassicalGramSchmidt = 0,
	ModifiedGramSchmidt = 1,
};

/**
 * @brief Converts an integer to the Orthogonalization enum
 * @param [in] ortho Integer representing an orthogonalization type
 * @return Orthogonalization
 */
inline Orthogonalization toOrthogonalization(unsigned int ortho)
{
	switch (ortho)
	{
	case static_cast<typename std::underlying_type<Orthogonalization>::type>(Orthogonalization::ClassicalGramSchmidt):
		return Orthogonalization::ClassicalGramSchmidt;
	case static_cast<typename std::underlying_type<Orthogonalization>::type>(Orthogonalization::ModifiedGramSchmidt):
		return Orthogonalization::ModifiedGramSchmidt;
	}
	throw InvalidParameterException("Unknown orthogonalization type");
}

/**
 * @brief Implements the Generalized Minimal Residual (GMRES) method for solving the linear system @f$ Ax = b @f$
 * @details Wraps the implementation provided by SUNDIALS.
 */
class Gmres
{
public:
	/**
	 * @brief Prototype of matrix-vector multiplication function provided to GMRES algorithm
	 * @details Performs a matrix vector multiplication @f$ z = Ax @f$.
	 *
	 * @param [in] userData User data
	 * @param [in] x Vector the matrix is multiplied with
	 * @param [out] z Result of the multiplication (memory is provided by the caller)
	 * @return @c 0 if successful, any other value in case of failure
	 */
	typedef std::function<int(void* userData, double const* x, double* z)> MatrixVectorMultFun;

	Gmres() CADET_NOEXCEPT;
	~Gmres() CADET_NOEXCEPT;

	/**
	 * @brief Initializes the GMRES algorithm
	 * @details Applies settings and allocates memory.
	 * @param [in] matrixSize Size of the square matrix @f$ A @f$
	 * @param [in] maxKrylov Maximum number of stored Krylov vectors (between @c 1 and @p matrixSize)
	 */
	void initialize(unsigned int matrixSize, unsigned int maxKrylov);

	/**
	 * @brief Initializes the GMRES algorithm
	 * @details Applies settings and allocates memory.
	 * @param [in] matrixSize Size of the square matrix @f$ A @f$
	 * @param [in] maxKrylov Maximum number of stored Krylov vectors (between @c 1 and @p matrixSize)
	 * @param [in] om Orthogonalization method used by GMRES
	 * @param [in] maxRestarts Maximum number of restarts
	 */
	void initialize(unsigned int matrixSize, unsigned int maxKrylov, Orthogonalization om, unsigned int maxRestarts);

	/**
	 * @brief Uses the configured GMRES method to solve a linear equation system @f$ Ax = b @f$
	 * @details The GMRES method, begin a Krylov subspace method, only requires matrix-vector products
	 *          with the matrix @f$ A @f$. These products are provided by a user-defined function
	 *          specified in matrixVectorMultiplier().
	 *
	 * @param tolerance Threshold on the weighted l^2 norm of the residual which terminates the iteration
	 * @param weight Weight vector used in the error norm
	 * @param rhs Right hand side vector @f$ b @f$
	 * @param sol On entry the initial guess, on exit the solution if the method has converged
	 * @return @c 0 on success, a positive value on recoverable error, and a negative value on
	 *         critical failure (use getReturnFlagName() to convert the return flag to a string)
	 */
	int solve(double tolerance, double const* weight, double const* rhs, double* sol);

	/**
	 * @brief Returns the orthogonalization method used by GMRES
	 * @return Orthogonalization method used by GMRES
	 */
	inline Orthogonalization orthoMethod() const CADET_NOEXCEPT
	{
		return _ortho;
	}
	/**
	 * @brief Sets the orthogonalization method used by GMRES
	 * @param [in] om Orthogonalization method
	 */
	inline void orthoMethod(Orthogonalization om) CADET_NOEXCEPT
	{
		_ortho = om;
	}

	/**
	 * @brief Returns the maximum number of restarts
	 * @return Maximum number of restarts
	 */
	inline unsigned int maxRestarts() const CADET_NOEXCEPT
	{
		return _maxRestarts;
	}
	/**
	 * @brief Sets the maximum number of restarts
	 * @param [in] mr Maximum number of restarts
	 */
	inline void maxRestarts(unsigned int mr) CADET_NOEXCEPT
	{
		_maxRestarts = mr;
	}

	/**
	 * @brief Returns the size of the square matrix to be solved
	 * @return Number of rows or columns of the square matrix to be solved
	 */
	inline unsigned int matrixSize() const CADET_NOEXCEPT
	{
		return _matrixSize;
	}

	/**
	 * @brief Returns the matrix-vector multiplication function
	 * @return Matrix-vector multiplication function
	 */
	inline MatrixVectorMultFun matrixVectorMultiplier() const CADET_NOEXCEPT
	{
		return _matVecMul;
	}
	/**
	 * @brief Sets the matrix-vector multiplication function
	 * @param [in] mvm Matrix-vector multiplication function
	 */
	inline void matrixVectorMultiplier(MatrixVectorMultFun mvm) CADET_NOEXCEPT
	{
		_matVecMul = mvm;
	}
	/**
	 * @brief Sets the matrix-vector multiplication function
	 * @param [in] mvm Matrix-vector multiplication function
	 * @param [in] ud User data passed to the matrix-vector multiplication function
	 */
	inline void matrixVectorMultiplier(MatrixVectorMultFun mvm, void* ud) CADET_NOEXCEPT
	{
		_matVecMul = mvm;
		_userData = ud;
	}

	/**
	 * @brief Returns the user data passed to the matrix-vector multiplication function
	 * @return User data
	 */
	inline void* userData() const CADET_NOEXCEPT
	{
		return _userData;
	}
	/**
	 * @brief Sets the user data passed to the matrix-vector multiplication function
	 * @param [in] ud User data
	 */
	inline void userData(void* ud) CADET_NOEXCEPT
	{
		_userData = ud;
	}

	/**
	 * @brief Translates the return value of solve() to a human readable SUNDIALS error code
	 * @param [in] flag Return value of solve()
	 * @return Error keyword
	 */
	const char* getReturnFlagName(int flag) const CADET_NOEXCEPT;

#ifdef CADET_BENCHMARK_MODE
	/**
	 * @brief Returns the total number of iterations over all calls of solve()
	 * @return Total number of iterations
	 */
	inline int numIterations() const CADET_NOEXCEPT
	{
		return _numIter;
	}
#endif

protected:
#if CADET_SUNDIALS_IFACE == 2
	SpgmrMemRec* _mem; //!< SUNDIALS memory
#elif CADET_SUNDIALS_IFACE == 3
	SUNLinearSolver _linearSolver; //!< SUNDIALS linear solver object
#endif
	Orthogonalization _ortho;       //!< Orthogonalization method
	unsigned int _maxRestarts;      //!< Maximum number of restarts
	unsigned int _matrixSize;       //!< Size of the square matrix
	MatrixVectorMultFun _matVecMul; //!< Matrix-vector multiplication function required for GMRES algorithm
	void* _userData;                //!< User data for matrix-vector multiplication function

#ifdef CADET_BENCHMARK_MODE
	int _numIter; //!< Accumulated number of iterations
	friend int gmresCallback(void* userData, N_Vector v, N_Vector z);
#endif
};

} // namespace linalg

} // namespace cadet

#endif // LIBCADET_GMRES_HPP_
