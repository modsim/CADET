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
 * Implements the kernel of the radial convection dispersion transport operator.
 */

#include "model/parts/RadialConvectionDispersionKernel.hpp"
#include "Weno.hpp"
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

} // namespace impl

void sparsityPatternRadial(linalg::SparsityPatternRowIterator itBegin, unsigned int nComp, unsigned int nCol, int strideCell, double u, Weno& weno)
{
	impl::DummyStencil stencil;

	if (u >= 0.0)
	{
		for (unsigned int comp = 0; comp < nComp; ++comp)
		{
			linalg::SparsityPatternRowIterator jac = itBegin + comp;

			// Time derivative
			jac.centered(0);

			// Reset WENO output
			double vm = 0.0; // reconstructed value

			int wenoOrder = 0;

			// Iterate over all cells
			for (unsigned int col = 0; col < nCol; ++col)
			{
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
					for (int i = 0; i < 2 * wenoOrder - 1; ++i)
						// Note that we have an offset of -1 here (compared to the right cell face below), since
						// the reconstructed value depends on the previous stencil (which has now been moved by one cell)
						jac.centered((i - wenoOrder) * strideCell);
				}

				// Reconstruct concentration on this cell's right face -- we only need the WENO order here
				wenoOrder = weno.reconstruct<double, impl::DummyStencil>(1e-12, col, nCol, stencil, vm);

				// Right side
				for (int i = 0; i < 2 * wenoOrder - 1; ++i)
					jac.centered((i - wenoOrder + 1) * strideCell);

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

			// Time derivative
			jac.centered(0);

			// Reset WENO output
			double vm = 0.0; // reconstructed value

			int wenoOrder = 0;

			// Iterate over all cells (backwards)
			// Note that col wraps around to unsigned int's maximum value after 0
			for (unsigned int col = nCol - 1; col < nCol; --col)
			{
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
					for (int i = 0; i < 2 * wenoOrder - 1; ++i)
						// Note that we have an offset of +1 here (compared to the left cell face below), since
						// the reconstructed value depends on the previous stencil (which has now been moved by one cell)
						jac.centered((wenoOrder - i) * strideCell);
				}

				// Reconstruct concentration on this cell's left face -- we only need the WENO order here
				wenoOrder = weno.reconstruct<double, impl::DummyStencil>(1e-12, col, nCol, stencil, vm);

				// Left face
				for (int i = 0; i < 2 * wenoOrder - 1; ++i)
					jac.centered((wenoOrder - i - 1) * strideCell);

				if (cadet_likely(col > 0))
					jac -= strideCell;
			}
		}
	}
}

} // namespace convdisp
} // namespace parts
} // namespace model
} // namespace cadet
