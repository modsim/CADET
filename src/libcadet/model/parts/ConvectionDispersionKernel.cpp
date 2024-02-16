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
 * Implements the sparsity pattern of the convection dispersion transport operator.
 */

#include "model/parts/ConvectionDispersionKernel.hpp"
#include "Weno.hpp"
#include "HighResKoren.hpp"
#include "linalg/CompressedSparseMatrix.hpp"

namespace cadet
{

namespace model
{

namespace parts
{

namespace convdisp
{

namespace impl
{

	class DummyStencil
	{
	public:
		DummyStencil() { }
		inline double operator[](const int idx) const { return 0.0; }
	};

	template <typename ReconstrType>
	void sparsityPattern(linalg::SparsityPatternRowIterator itBegin, unsigned int nComp, unsigned int nCol, int strideCell, double u, ReconstrType& reconstruction)
	{
		impl::DummyStencil stencil;

		if (u >= 0.0)
		{
			for (unsigned int comp = 0; comp < nComp; ++comp)
			{
				linalg::SparsityPatternRowIterator jac = itBegin + comp;

				// Reset reconstructed output
				double vm = 0.0; // reconstructed value

				int recOrder = 0;

				// Iterate over all cells
				for (unsigned int col = 0; col < nCol; ++col)
				{
					// Time derivative
					jac.centered(0);

					// ------------------- Dispersion -------------------

					// Right side, leave out if we're in the last cell (boundary condition)
					if (cadet_likely(col < nCol - 1))
					{
						// jac.centered(0);
						jac.centered(strideCell);
					}

					// Left side, leave out if we're in the first cell (boundary condition)
					if (cadet_likely(col > 0))
					{
						// jac.centered(0);
						jac.centered(-strideCell);
					}

					// ------------------- Convection -------------------

					// Add convection through this cell's left face
					if (cadet_likely(col > 0))
					{
						for (int i = 0; i < 2 * recOrder - 1; ++i)
							// Note that we have an offset of -1 here (compared to the right cell face below), since
							// the reconstructed value depends on the previous stencil (which has now been moved by one cell)
							jac.centered((i - recOrder) * strideCell);
					}

					// Reconstruct concentration on this cell's right face -- we only need the order here
					recOrder = reconstruction.template reconstruct<double, impl::DummyStencil>(col, nCol, stencil, vm);

					// Right side
					for (int i = 0; i < 2 * recOrder - 1; ++i)
						jac.centered((i - recOrder + 1) * strideCell);

					if (cadet_likely(col < nCol - 1))
						jac += strideCell;
				}
			}
		}
		else
		{
			for (unsigned int comp = 0; comp < nComp; ++comp)
			{
				linalg::SparsityPatternRowIterator jac = itBegin + strideCell * (nCol - 1) + comp;

				// Reset reconstructed output
				double vm = 0.0; // reconstructed value

				int recOrder = 0;

				// Iterate over all cells (backwards)
				// Note that col wraps around to unsigned int's maximum value after 0
				for (unsigned int col = nCol - 1; col < nCol; --col)
				{
					// Time derivative
					jac.centered(0);

					// ------------------- Dispersion -------------------

					// Right side, leave out if we're in the first cell (boundary condition)
					if (cadet_likely(col < nCol - 1))
					{
						// jac.centered(0);
						jac.centered(strideCell);
					}

					// Left side, leave out if we're in the last cell (boundary condition)
					if (cadet_likely(col > 0))
					{
						// jac.centered(0);
						jac.centered(-strideCell);
					}

					// ------------------- Convection -------------------

					// Add convection through this cell's right face
					if (cadet_likely(col < nCol - 1))
					{
						for (int i = 0; i < 2 * recOrder - 1; ++i)
							// Note that we have an offset of +1 here (compared to the left cell face below), since
							// the reconstructed value depends on the previous stencil (which has now been moved by one cell)
							jac.centered((recOrder - i) * strideCell);
					}

					// Reconstruct concentration on this cell's left face -- we only need the order here
					recOrder = reconstruction.template reconstruct<double, impl::DummyStencil>(col, nCol, stencil, vm);

					// Left face
					for (int i = 0; i < 2 * recOrder - 1; ++i)
						jac.centered((recOrder - i - 1) * strideCell);

					if (cadet_likely(col > 0))
						jac -= strideCell;
				}
			}
		}
	}
} // namespace impl

void sparsityPattern(linalg::SparsityPatternRowIterator itBegin, unsigned int nComp, unsigned int nCol, int strideCell, double u, Weno& weno)
{
	impl::sparsityPattern(itBegin, nComp, nCol, strideCell, u, weno);
}

void sparsityPattern(linalg::SparsityPatternRowIterator itBegin, unsigned int nComp, unsigned int nCol, int strideCell, double u, HighResolutionKoren& koren)
{
	impl::sparsityPattern(itBegin, nComp, nCol, strideCell, u, koren);
}

} // namespace convdisp
} // namespace parts
} // namespace model
} // namespace cadet
