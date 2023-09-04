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
 * Interfaces the compressed sparse matrix with the SuperLU solver
 */

/*
 * By default, SuperLU allocates memory as necessary during factorization.
 * This will happen each time the matrix is factorized, without exception.
 *
 * It is possible to preallocate memory (or grow a buffer over a short time)
 * for all factorizations with a constant pattern. In order to enable this
 * method, #define LIBCADET_SUPERLU_MANAGE_MEMORY.
 *
 * Note that, as of now, when using preallocated memory, each call to
 * factorize() will leak 64 bytes somewhere in SuperLU according to LLVM ASAN.
 */

#include "linalg/CompressedSparseMatrix.hpp"

#if !defined(LIBCADET_SUPERLUSPARSEMATRIX_HPP_) && defined(SUPERLU_FOUND)
#define LIBCADET_SUPERLUSPARSEMATRIX_HPP_

#ifndef LIBCADET_SUPERLUSPARSEMATRIX_NOFORWARD
struct SuperMatrix;
struct SuperLUStat_t;
struct GlobalLU_t;
#endif

namespace cadet
{

namespace linalg
{

/**
 * @brief Sparse matrix with compressed row storage that can be factorized
 */
class SuperLUSparseMatrix : public CompressedSparseMatrix
{
public:
	/**
	 * @brief Creates an empty SuperLUSparseMatrix with capacity @c 0
	 * @details Users have to call resize() prior to populating the matrix.
	 */
	SuperLUSparseMatrix();

	/**
	 * @brief Creates an empty SuperLUSparseMatrix with the given capacity
	 * @param [in] numRows Matrix size (i.e., number of rows or columns)
	 * @param [in] numNonZeros Maximum number of non-zero elements
	 */
	SuperLUSparseMatrix(unsigned int numRows, unsigned int numNonZeros);

	~SuperLUSparseMatrix() CADET_NOEXCEPT;

	// Default copy and assignment semantics
	SuperLUSparseMatrix(const SuperLUSparseMatrix& cpy) = default;
	SuperLUSparseMatrix(SuperLUSparseMatrix&& cpy) CADET_NOEXCEPT = default;

	SuperLUSparseMatrix& operator=(const SuperLUSparseMatrix& cpy) = default;

#ifdef COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE
	SuperLUSparseMatrix& operator=(SuperLUSparseMatrix&& cpy) CADET_NOEXCEPT = default;
#else
	SuperLUSparseMatrix& operator=(SuperLUSparseMatrix&& cpy) = default;
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
	void allocateMatrixStructs();

	SuperMatrix* _mat;        //!< SuperLU matrix info for main matrix
	SuperMatrix* _rhsMat;     //!< SuperLU matrix info for right hand side
	SuperMatrix* _permMat;    //!< Permuted SuperLU matrix info for main matrix
	bool _firstFactorization; //!< Determines whether this is the first factorization with the sparsity pattern
	SuperLUStat_t* _stats;    //!< SuperLU statistics
	GlobalLU_t* _globalLU;    //!< Holds factorization info (row factors)
	SuperMatrix* _lower;      //!< Lower triangular matrix (no storage, just pointer)
	SuperMatrix* _upper;      //!< Upper triangular matrix (no storage, just pointer)
	sparse_int_t* _permCols;  //!< Column permutations (structurally determined from sparsity pattern)
	sparse_int_t* _permRows;  //!< Row permutations (due to pivoting during factorization)
	sparse_int_t* _eTree;     //!< SuperLU elimination tree
#ifdef LIBCADET_SUPERLU_MANAGE_MEMORY
	std::vector<char> _memory; //!< SuperLU workspace memory
#endif
};

} // namespace linalg

} // namespace cadet

#endif // LIBCADET_SUPERLUSPARSEMATRIX_HPP_
