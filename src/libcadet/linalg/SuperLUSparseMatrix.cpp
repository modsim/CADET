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

#include "slu_ddefs.h"

#define LIBCADET_SUPERLUSPARSEMATRIX_NOFORWARD
#include "linalg/SuperLUSparseMatrix.hpp"

namespace cadet
{

namespace linalg
{

SuperLUSparseMatrix::SuperLUSparseMatrix()
	: CompressedSparseMatrix(), _permMat(nullptr), _firstFactorization(true), _permCols(nullptr), _permRows(nullptr),
	  _eTree(nullptr)
#ifdef LIBCADET_SUPERLU_MANAGE_MEMORY
	  ,
	  _memory(0)
#endif
{
	allocateMatrixStructs();
}

SuperLUSparseMatrix::SuperLUSparseMatrix(unsigned int numRows, unsigned int numNonZeros)
	: CompressedSparseMatrix(numRows, numNonZeros), _permMat(nullptr), _firstFactorization(true)
#ifdef LIBCADET_SUPERLU_MANAGE_MEMORY
	  ,
	  _memory(0)
#endif
{
	allocateMatrixStructs();

	_permCols = new sparse_int_t[numRows];
	_permRows = new sparse_int_t[numRows];
	_eTree = new sparse_int_t[numRows];
}

SuperLUSparseMatrix::~SuperLUSparseMatrix() CADET_NOEXCEPT
{
	StatFree(_stats);

#ifdef LIBCADET_SUPERLU_MANAGE_MEMORY
	Destroy_SuperMatrix_Store(_lower);
	Destroy_SuperMatrix_Store(_upper);
#else
	if (!_firstFactorization)
	{
		Destroy_SuperNode_Matrix(_lower);
		Destroy_CompCol_Matrix(_upper);
	}
#endif

	if (_permMat)
	{
		Destroy_CompCol_Permuted(_permMat);
		delete _permMat;
	}

	delete static_cast<NCformat*>(_mat->Store);
	delete static_cast<DNformat*>(_rhsMat->Store);
	delete _mat;
	delete _rhsMat;
	delete _stats;
	delete _globalLU;
	delete _lower;
	delete _upper;
	delete[] _permCols;
	delete[] _permRows;
	delete[] _eTree;
}

void SuperLUSparseMatrix::allocateMatrixStructs()
{
	_stats = new SuperLUStat_t;
	StatInit(_stats);

	_globalLU = new GlobalLU_t;

	_lower = new SuperMatrix;
	_upper = new SuperMatrix;

	_mat = new SuperMatrix;
	_mat->Store = nullptr;
	_mat->Store = new NCformat;
	_mat->Stype = SLU_NC;
	_mat->Dtype = SLU_D;
	_mat->Mtype = SLU_GE;

	_rhsMat = new SuperMatrix;
	_rhsMat->Store = nullptr;
	_rhsMat->Store = new DNformat;
	_rhsMat->Stype = SLU_DN;
	_rhsMat->Dtype = SLU_D;
	_rhsMat->Mtype = SLU_GE;
}

void SuperLUSparseMatrix::prepare()
{
	// Set fields of SuperLU matrix structs
	NCformat* const storage = static_cast<NCformat*>(_mat->Store);

	_mat->nrow = rows();
	_mat->ncol = rows();
	storage->nnz = numNonZeros();
	storage->nzval = _values.data();
	storage->rowind = _colIdx.data();
	storage->colptr = _rowStart.data();

	// Set fields of SuperLU right hand side (dense)
	_rhsMat->nrow = rows();
	_rhsMat->ncol = 1;
	static_cast<DNformat*>(_rhsMat->Store)->lda = rows();

	// Maybe allocate memory for permutation and elimination tree
	if (!_permCols)
	{
		_permCols = new sparse_int_t[rows()];
		_permRows = new sparse_int_t[rows()];
		_eTree = new sparse_int_t[rows()];
	}

	/*
	 * Get column permutation vector _permCols[], according to permc_spec:
	 *   permc_spec = NATURAL:  natural ordering
	 *   permc_spec = MMD_AT_PLUS_A: minimum degree on structure of A'+A
	 *   permc_spec = MMD_ATA:  minimum degree on structure of A'*A
	 *   permc_spec = COLAMD:   approximate minimum degree column ordering
	 *   permc_spec = MY_PERMC: the ordering already supplied in _permCols[]
	 */
	get_perm_c(COLAMD, _mat, _permCols);

	// Permute the columns
	superlu_options_t options;
	set_default_options(&options);
	options.Fact = DOFACT;
	options.SymmetricMode = NO;

	if (!_permMat)
		_permMat = new SuperMatrix;
	else
		Destroy_CompCol_Permuted(_permMat);

	sp_preorder(&options, _mat, _permCols, _eTree, _permMat);

	_firstFactorization = true;
}

bool SuperLUSparseMatrix::factorize()
{
	// TODO: Compare SamePattern vs SamePattern_SameRowPerm
	const fact_t mode = SamePattern_SameRowPerm;

	// Set options
	superlu_options_t options;
	set_default_options(&options);
	options.Fact = (_firstFactorization ? DOFACT : mode);
	options.SymmetricMode = NO;

	const int panelSize = sp_ienv(1);
	const int superNodeRelaxation = sp_ienv(2);

#ifdef LIBCADET_SUPERLU_MANAGE_MEMORY
	// Allocate memory if necessary
	if (_firstFactorization)
	{
		// Guess memory in bytes
		int memSize = 0;
		dgstrf(&options, _permMat, superNodeRelaxation, panelSize, _eTree, nullptr, -1, _permCols, _permRows, _lower,
			   _upper, _globalLU, _stats, &memSize);

		// SuperLU returns numer of bytes + rows()
		_memory.resize(memSize - rows());
	}
#else
	if (!_firstFactorization && (mode == SamePattern))
	{
		Destroy_SuperNode_Matrix(_lower);
		Destroy_CompCol_Matrix(_upper);
	}
#endif

	// Factorize
	int info = 1;
	while (info != 0)
	{

#ifdef LIBCADET_SUPERLU_MANAGE_MEMORY
		dgstrf(&options, _permMat, superNodeRelaxation, panelSize, _eTree, _memory.data(), _memory.size(), _permCols,
			   _permRows, _lower, _upper, _globalLU, _stats, &info);
#else
		dgstrf(&options, _permMat, superNodeRelaxation, panelSize, _eTree, nullptr, 0, _permCols, _permRows, _lower,
			   _upper, _globalLU, _stats, &info);
#endif

		if (cadet_likely(info == 0))
		{
			// Matrix factorized successfully
			_firstFactorization = false;

			//			mem_usage_t memInfo;
			//			dQuerySpace(_lower, _upper, &memInfo);
			return true;
		}
#ifdef LIBCADET_SUPERLU_MANAGE_MEMORY
		else if (info > rows())
		{
			// Insufficient memory
			const int memSize = info - rows();
			if (_memory.size() < memSize)
			{
				// Allocate the requested memory size
				_memory.resize(memSize);
			}
			else
			{
				// Increase by 10%
				_memory.resize(_memory.size() + _memory.size() / 10);
			}

			// Try again with more memory
		}
#else
		else if (info > rows())
		{
			// Out of memory
			return false;
		}
#endif
		else if (info > 0)
		{
			// Diagonal element (info, info) is exactly 0.0
			//			LOG(Debug) << "Diagonal element (" << (info-1) << ", " << (info-1) << ") is exactly 0.0";
			_firstFactorization = false;
			return false;
		}
		else if (info < 0)
		{
			// Error in argument -info passed to dgstrf()
			return false;
		}
	}

	return false;
}

bool SuperLUSparseMatrix::solve(double* rhs) const
{
	static_cast<DNformat*>(_rhsMat->Store)->nzval = rhs;

	int info = 0;
	dgstrs(TRANS, _lower, _upper, _permCols, _permRows, _rhsMat, _stats, &info);
	return info == 0;
}

} // namespace linalg

} // namespace cadet
