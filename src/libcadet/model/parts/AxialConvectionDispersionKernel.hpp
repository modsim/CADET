// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

/**
 * @file 
 * Implements the kernel of the axial convection dispersion transport operator.
 */

#ifndef LIBCADET_AXIALCONVECTIONDISPERSIONKERNEL_HPP_
#define LIBCADET_AXIALCONVECTIONDISPERSIONKERNEL_HPP_

#include "AutoDiff.hpp"
#include "Memory.hpp"
#include "Stencil.hpp"
#include "linalg/CompressedSparseMatrix.hpp"
#include "SimulationTypes.hpp"
#include "model/ParameterDependence.hpp"
#include "model/UnitOperation.hpp"

namespace cadet
{

class Weno;
class HighResolutionKoren;

namespace model
{

namespace parts
{

namespace convdisp
{

template <typename T, typename Reconstruction_t>
struct AxialFlowParameters
{
	T u;
	active const* d_ax;
	T h;
	double* reconstructionDerivatives; //!< Holds derivatives of the reconstruction scheme
	Reconstruction_t* reconstruction; //!< The reconstruction scheme implementation
	ArrayPool* stencilMemory; //!< Provides memory for the stencil
	int strideCell;
	unsigned int nComp;
	unsigned int nCol;
	unsigned int offsetToInlet; //!< Offset to the first component of the inlet DOFs in the local state vector
	unsigned int offsetToBulk; //!< Offset to the first component of the first bulk cell in the local state vector
	IParameterParameterDependence* parDep;
	const IModel& model;
};


namespace impl
{
	template <typename StateType, typename ResidualType, typename ParamType, typename ReconstrType, typename RowIteratorType, bool wantJac, bool wantRes = true>
	int residualForwardsAxialFlow(const SimulationTime& simTime, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin, const AxialFlowParameters<ParamType, ReconstrType>& p)
	{
		// h is the size of the cell
		const ParamType h2 = p.h * p.h;

		// The stencil caches parts of the state vector for better spatial coherence
		typedef CachingStencil<StateType, ArrayPool> StencilType;
		StencilType stencil(std::max(p.reconstruction->stencilSize(), 3u), *p.stencilMemory, std::max(p.reconstruction->order() - 1, 1));

		// The RowIterator is always centered on the main diagonal.
		// This means that jac[0] is the main diagonal, jac[-1] is the first lower diagonal,
		// and jac[1] is the first upper diagonal. We can also access the rows from left to
		// right beginning with the last lower diagonal moving towards the main diagonal and
		// continuing to the last upper diagonal by using the native() method.
		RowIteratorType jac;

		ResidualType* const resBulk = wantRes ? res + p.offsetToBulk : nullptr;
		StateType const* const yBulk = y + p.offsetToBulk;

		for (unsigned int comp = 0; comp < p.nComp; ++comp)
		{
			if (wantJac)
				jac = jacBegin + comp;

			ResidualType* const resBulkComp = wantRes ? resBulk + comp : nullptr;
			StateType const* const yBulkComp = yBulk + comp;

			// Add time derivative to each cell
			if (yDot && wantRes)
			{
				double const* const yDotBulkComp = yDot + p.offsetToBulk + comp;
				for (unsigned int col = 0; col < p.nCol; ++col)
					resBulkComp[col * p.strideCell] = yDotBulkComp[col * p.strideCell];
			}
			else if (wantRes)
			{
				for (unsigned int col = 0; col < p.nCol; ++col)
					resBulkComp[col * p.strideCell] = 0.0;
			}

			// Fill stencil (left side with zeros, right side with states)
			for (int i = -std::max(p.reconstruction->order(), 2) + 1; i < 0; ++i)
				stencil[i] = 0.0;
			for (int i = 0; i < std::max(p.reconstruction->order(), 2); ++i)
				stencil[i] = yBulkComp[i * p.strideCell];

			// Reset reconstruction output
			StateType vm(0.0); // reconstructed value
			if (wantJac)
				std::fill(p.reconstructionDerivatives, p.reconstructionDerivatives + p.reconstruction->stencilSize(), 0.0);

			int wenoOrder = 0;
			const ParamType d_ax = static_cast<ParamType>(p.d_ax[comp]);

			// Iterate over all cells
			for (unsigned int col = 0; col < p.nCol; ++col)
			{
				// ------------------- Dispersion -------------------

				// Right side, leave out if we're in the last cell (boundary condition)
				if (cadet_likely(col < p.nCol - 1))
				{
					const double relCoord = static_cast<double>(col + 1) / p.nCol;
					const ParamType d_ax_right = d_ax * p.parDep->getValue(p.model, ColumnPosition{ relCoord, 0.0, 0.0 }, comp, ParTypeIndep, BoundStateIndep, static_cast<ParamType>(p.u));
					if (wantRes)
						resBulkComp[col * p.strideCell] -= d_ax_right / h2 * (stencil[1] - stencil[0]);
					// Jacobian entries
					if (wantJac)
					{
						jac[0] += static_cast<double>(d_ax_right) / static_cast<double>(h2);
						jac[p.strideCell] -= static_cast<double>(d_ax_right) / static_cast<double>(h2);
					}
				}

				// Left side, leave out if we're in the first cell (boundary condition)
				if (cadet_likely(col > 0))
				{
					const double relCoord = static_cast<double>(col) / p.nCol;
					const ParamType d_ax_left = d_ax * p.parDep->getValue(p.model, ColumnPosition{ relCoord, 0.0, 0.0 }, comp, ParTypeIndep, BoundStateIndep, static_cast<ParamType>(p.u));
					if (wantRes)
						resBulkComp[col * p.strideCell] -= d_ax_left / h2 * (stencil[-1] - stencil[0]);
					// Jacobian entries
					if (wantJac)
					{
						jac[0] += static_cast<double>(d_ax_left) / static_cast<double>(h2);
						jac[-p.strideCell] -= static_cast<double>(d_ax_left) / static_cast<double>(h2);
					}
				}

				// ------------------- Convection -------------------

				// Add convection through this cell's left face
				if (cadet_likely(col > 0))
				{
					// Remember that vm still contains the reconstructed value of the previous 
					// cell's *right* face, which is identical to this cell's *left* face!
					if (wantRes)
						resBulkComp[col * p.strideCell] -= p.u / p.h * vm;

					// Jacobian entries
					if (wantJac)
					{
						for (int i = 0; i < 2 * wenoOrder - 1; ++i)
							// Note that we have an offset of -1 here (compared to the right cell face below), since
							// the reconstructed value depends on the previous stencil (which has now been moved by one cell)
							jac[(i - wenoOrder) * p.strideCell] -= static_cast<double>(p.u) / static_cast<double>(p.h) * p.reconstructionDerivatives[i];
					}
				}
				else if (wantRes)
				{
					// In the first cell we need to apply the boundary condition: inflow concentration
					resBulkComp[col * p.strideCell] -= p.u / p.h * y[p.offsetToInlet + comp];
				}

				// Reconstruct concentration on this cell's right face
				if (wantJac)
					wenoOrder = p.reconstruction->template reconstruct<StateType, StencilType>(col, p.nCol, stencil, vm, p.reconstructionDerivatives);
				else
					wenoOrder = p.reconstruction->template reconstruct<StateType, StencilType>(col, p.nCol, stencil, vm);

				// Right side
				if (wantRes)
					resBulkComp[col * p.strideCell] += p.u / p.h * vm;
				// Jacobian entries
				if (wantJac)
				{
					for (int i = 0; i < 2 * wenoOrder - 1; ++i)
						jac[(i - wenoOrder + 1) * p.strideCell] += static_cast<double>(p.u) / static_cast<double>(p.h) * p.reconstructionDerivatives[i];
				}

				// Update stencil
				const unsigned int shift = std::max(p.reconstruction->order(), 2);
				if (cadet_likely(col + shift < p.nCol))
					stencil.advance(yBulkComp[(col + shift) * p.strideCell]);
				else
					stencil.advance(0.0);

				if (wantJac)
				{
					if (cadet_likely(col < p.nCol - 1))
						jac += p.strideCell;
				}
			}
		}

		// Film diffusion with flux into beads is added in residualFlux() function

		return 0;
	}

	template <typename StateType, typename ResidualType, typename ParamType, typename ReconstrType, typename RowIteratorType, bool wantJac, bool wantRes = true>
	int residualBackwardsAxialFlow(const SimulationTime& simTime, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin, const AxialFlowParameters<ParamType, ReconstrType>& p)
	{
		const ParamType h2 = p.h * p.h;

		// The stencil caches parts of the state vector for better spatial coherence
		typedef CachingStencil<StateType, ArrayPool> StencilType;
		StencilType stencil(std::max(p.reconstruction->stencilSize(), 3u), *p.stencilMemory, std::max(p.reconstruction->order() - 1, 1));

		// The RowIterator is always centered on the main diagonal.
		// This means that jac[0] is the main diagonal, jac[-1] is the first lower diagonal,
		// and jac[1] is the first upper diagonal. We can also access the rows from left to
		// right beginning with the last lower diagonal moving towards the main diagonal and
		// continuing to the last upper diagonal by using the native() method.
		RowIteratorType jac;

		ResidualType* const resBulk = wantRes ? res + p.offsetToBulk : nullptr;
		StateType const* const yBulk = y + p.offsetToBulk;

		for (unsigned int comp = 0; comp < p.nComp; ++comp)
		{
			if (wantJac)
				jac = jacBegin + p.strideCell * (p.nCol - 1) + comp;

			ResidualType* const resBulkComp = wantRes ? resBulk + comp : nullptr;
			StateType const* const yBulkComp = yBulk + comp;

			// Add time derivative to each cell
			if (yDot && wantRes)
			{
				double const* const yDotBulkComp = yDot + p.offsetToBulk + comp;
				for (unsigned int col = 0; col < p.nCol; ++col)
					resBulkComp[col * p.strideCell] = yDotBulkComp[col * p.strideCell];
			}
			else if (wantRes)
			{
				for (unsigned int col = 0; col < p.nCol; ++col)
					resBulkComp[col * p.strideCell] = 0.0;
			}

			// Fill stencil (left side with zeros, right side with states)
			for (int i = -std::max(p.reconstruction->order(), 2) + 1; i < 0; ++i)
				stencil[i] = 0.0;
			for (int i = 0; i < std::max(p.reconstruction->order(), 2); ++i)
				stencil[i] = yBulkComp[(p.nCol - static_cast<unsigned int>(i) - 1) * p.strideCell];

			// Reset reconstruction output
			StateType vm(0.0); // reconstructed value
			if (wantJac)
				std::fill(p.reconstructionDerivatives, p.reconstructionDerivatives + p.reconstruction->stencilSize(), 0.0);

			int wenoOrder = 0;
			const ParamType d_ax = static_cast<ParamType>(p.d_ax[comp]);

			// Iterate over all cells (backwards)
			// Note that col wraps around to unsigned int's maximum value after 0
			for (unsigned int col = p.nCol - 1; col < p.nCol; --col)
			{
				// ------------------- Dispersion -------------------

				// Right side, leave out if we're in the first cell (boundary condition)
				if (cadet_likely(col < p.nCol - 1))
				{
					const double relCoord = static_cast<double>(col + 1) / p.nCol;
					const ParamType d_ax_right = d_ax * p.parDep->getValue(p.model, ColumnPosition{ relCoord, 0.0, 0.0 }, comp, ParTypeIndep, BoundStateIndep, static_cast<ParamType>(p.u));
					if (wantRes)
						resBulkComp[col * p.strideCell] -= d_ax_right / h2 * (stencil[-1] - stencil[0]);
					// Jacobian entries
					if (wantJac)
					{
						jac[0] += static_cast<double>(d_ax_right) / static_cast<double>(h2);
						jac[p.strideCell] -= static_cast<double>(d_ax_right) / static_cast<double>(h2);
					}
				}

				// Left side, leave out if we're in the last cell (boundary condition)
				if (cadet_likely(col > 0))
				{
					const double relCoord = static_cast<double>(col) / p.nCol;
					const ParamType d_ax_left = d_ax * p.parDep->getValue(p.model, ColumnPosition{ relCoord, 0.0, 0.0 }, comp, ParTypeIndep, BoundStateIndep, static_cast<ParamType>(p.u));
					if (wantRes)
						resBulkComp[col * p.strideCell] -= d_ax_left / h2 * (stencil[1] - stencil[0]);
					// Jacobian entries
					if (wantJac)
					{
						jac[0] += static_cast<double>(d_ax_left) / static_cast<double>(h2);
						jac[-p.strideCell] -= static_cast<double>(d_ax_left) / static_cast<double>(h2);
					}
				}

				// ------------------- Convection -------------------

				// Add convection through this cell's right face
				if (cadet_likely(col < p.nCol - 1))
				{
					// Remember that vm still contains the reconstructed value of the previous 
					// cell's *left* face, which is identical to this cell's *right* face!
					if (wantRes)
						resBulkComp[col * p.strideCell] += p.u / p.h * vm;

					// Jacobian entries
					if (wantJac)
					{
						for (int i = 0; i < 2 * wenoOrder - 1; ++i)
							// Note that we have an offset of +1 here (compared to the left cell face below), since
							// the reconstructed value depends on the previous stencil (which has now been moved by one cell)
							jac[(wenoOrder - i) * p.strideCell] += static_cast<double>(p.u) / static_cast<double>(p.h) * p.reconstructionDerivatives[i];					
					}
				}
				else if (wantRes)
				{
					// In the last cell (z = L) we need to apply the boundary condition: inflow concentration
					resBulkComp[col * p.strideCell] += p.u / p.h * y[p.offsetToInlet + comp];
				}

				// Reconstruct concentration on this cell's left face
				if (wantJac)
					wenoOrder = p.reconstruction->template reconstruct<StateType, StencilType>(col, p.nCol, stencil, vm, p.reconstructionDerivatives);
				else
					wenoOrder = p.reconstruction->template reconstruct<StateType, StencilType>(col, p.nCol, stencil, vm);

				// Left face
				if (wantRes)
					resBulkComp[col * p.strideCell] -= p.u / p.h * vm;
				// Jacobian entries
				if (wantJac)
				{
					for (int i = 0; i < 2 * wenoOrder - 1; ++i)
						jac[(wenoOrder - i - 1) * p.strideCell] -= static_cast<double>(p.u) / static_cast<double>(p.h) * p.reconstructionDerivatives[i];
				}

				// Update stencil (be careful because of wrap-around, might cause reading memory very far away [although never used])
				const unsigned int shift = std::max(p.reconstruction->order(), 2);
				if (cadet_likely(col - shift < p.nCol))
					stencil.advance(yBulkComp[(col - shift) * p.strideCell]);
				else
					stencil.advance(0.0);

				if (wantJac)
				{
					if (cadet_likely(col > 0))
						jac -= p.strideCell;
				}
			}
		}

		// Film diffusion with flux into beads is added in residualFlux() function

		return 0;
	}

} // namespace impl


template <typename StateType, typename ResidualType, typename ParamType, typename ReconstrType, typename RowIteratorType, bool wantJac, bool wantRes = true>
int residualKernelAxial(const SimulationTime& simTime, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin, const AxialFlowParameters<ParamType, ReconstrType>& p)
{
	if (p.u >= 0.0)
		return impl::residualForwardsAxialFlow<StateType, ResidualType, ParamType, ReconstrType, RowIteratorType, wantJac, wantRes>(simTime, y, yDot, res, jacBegin, p);
	else
		return impl::residualBackwardsAxialFlow<StateType, ResidualType, ParamType, ReconstrType, RowIteratorType, wantJac, wantRes>(simTime, y, yDot, res, jacBegin, p);
}

void sparsityPatternAxial(linalg::SparsityPatternRowIterator itBegin, unsigned int nComp, unsigned int nCol, int strideCell, double u, Weno& weno);
void sparsityPatternAxial(linalg::SparsityPatternRowIterator itBegin, unsigned int nComp, unsigned int nCol, int strideCell, double u, HighResolutionKoren& koren);

} // namespace convdisp
} // namespace parts
} // namespace model
} // namespace cadet

#endif  // LIBCADET_AXIALCONVECTIONDISPERSIONKERNEL_HPP_
