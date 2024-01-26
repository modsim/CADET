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
 * Interfaces the compressed sparse matrix with the UMFPACK solver
 */

#include "linalg/CompressedSparseMatrix.hpp"

#if !defined(LIBCADET_UMFPACKSPARSEMATRIX_HPP_) && defined(UMFPACK_FOUND)
#define LIBCADET_UMFPACKSPARSEMATRIX_HPP_

namespace cadet
{

namespace linalg
{

/**
 * @brief Sparse matrix with compressed row storage that can be factorized
 */
class UMFPackSparseMatrix : public CompressedSparseMatrix
{
public:
	/**
	 * @brief Creates an empty UMFPackSparseMatrix with capacity @c 0
	 * @details Users have to call resize() prior to populating the matrix.
	 */
	UMFPackSparseMatrix();

	/**
	 * @brief Creates an empty UMFPackSparseMatrix with the given capacity
	 * @param [in] numRows Matrix size (i.e., number of rows or columns)
	 * @param [in] numNonZeros Maximum number of non-zero elements
	 */
	UMFPackSparseMatrix(unsigned int numRows, unsigned int numNonZeros);

	~UMFPackSparseMatrix() CADET_NOEXCEPT;

	// Default copy and assignment semantics
	UMFPackSparseMatrix(const UMFPackSparseMatrix& cpy) = default;
	UMFPackSparseMatrix(UMFPackSparseMatrix&& cpy) CADET_NOEXCEPT = default;

	UMFPackSparseMatrix& operator=(const UMFPackSparseMatrix& cpy) = default;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	UMFPackSparseMatrix& operator=(UMFPackSparseMatrix&& cpy) CADET_NOEXCEPT = default;
#else
	UMFPackSparseMatrix& operator=(UMFPackSparseMatrix&& cpy) = default;
#endif

	/**
	 * @brief Prepares data structures for factorization
	 * @details Has to be called whenever the sparsity pattern changes.
	 */
	void prepare();

	/**
	 * @brief Factorizes the BandMatrix using LAPACK (performs LU factorization)
	 * @details Assumes that prepare() has been called before.
	 * @return @c true if the factorization was successful, otherwise @c false
	 */
	bool factorize();

	/**
	 * @brief Uses the factorized matrix to solve the equation @f$ Ax = b @f$ with LAPACK
	 * @details Before the equation can be solved, the matrix has to be factorized first by calling factorize().
	 * @param [in,out] rhs On entry pointer to the right hand side vector @f$ b @f$ of the equation, on exit the
	 * solution @f$ x @f$
	 * @return @c true if the solution process was successful, otherwise @c false
	 */
	bool solve(double* rhs) const;

protected:
	void* _symbolic;                                 //!< Symbolic info for UMFPACK (orderings)
	void* _numeric;                                  //!< Factorization from UMFPACK (L, U factors)
	mutable std::vector<double> _info;               //!< UMFPACK statistics
	std::vector<double> _options;                    //!< UMFPACK options
	mutable std::vector<double> _result;             //!< Result cache
	mutable std::vector<sparse_int_t> _workSpaceIdx; //!< UMFPACK workspace for solving linear system
	mutable std::vector<double> _workSpace;          //!< UMFPACK workspace for solving linear system
};

} // namespace linalg

} // namespace cadet

#endif // LIBCADET_UMFPACKSPARSEMATRIX_HPP_
