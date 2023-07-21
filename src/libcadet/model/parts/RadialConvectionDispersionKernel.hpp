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

#ifndef LIBCADET_RADIALCONVECTIONDISPERSIONKERNEL_HPP_
#define LIBCADET_RADIALCONVECTIONDISPERSIONKERNEL_HPP_

#include "AutoDiff.hpp"
#include "Memory.hpp"
#include "Weno.hpp"
#include "Stencil.hpp"
#include "linalg/CompressedSparseMatrix.hpp"
#include "SimulationTypes.hpp"
#include "model/ParameterDependence.hpp"
#include "model/UnitOperation.hpp"

namespace cadet
{

class IModel;

namespace model
{

namespace parts
{

namespace convdisp
{

template <typename T>
struct RadialFlowParameters
{
	T u;
	active const* d_rad;
	active const* cellCenters; //!< Midpoints of the cells
	active const* cellSizes; //!< Cell sizes
	active const* cellBounds; //!< Cell boundaries
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
	template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
	int residualForwardsRadialFlow(const SimulationTime& simTime, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin, const RadialFlowParameters<ParamType>& p)
	{
		// The stencil caches parts of the state vector for better spatial coherence
		typedef CachingStencil<StateType, ArrayPool> StencilType;
//		StencilType stencil(std::max(p.weno->stencilSize(), 3u), *p.stencilMemory, std::max(p.weno->order() - 1, 1));
		StencilType stencil(std::max(1u, 3u), *p.stencilMemory, std::max(1 - 1, 1));

		// The RowIterator is always centered on the main diagonal.
		// This means that jac[0] is the main diagonal, jac[-1] is the first lower diagonal,
		// and jac[1] is the first upper diagonal. We can also access the rows from left to
		// right beginning with the last lower diagonal moving towards the main diagonal and
		// continuing to the last upper diagonal by using the native() method.
		RowIteratorType jac;

		ResidualType* const resBulk = res + p.offsetToBulk;
		StateType const* const yBulk = y + p.offsetToBulk;

		for (unsigned int comp = 0; comp < p.nComp; ++comp)
		{
			if (wantJac)
				jac = jacBegin + comp;

			ResidualType* const resBulkComp = resBulk + comp;
			StateType const* const yBulkComp = yBulk + comp;

			// Add time derivative to each cell
			if (yDot)
			{
				double const* const yDotBulkComp = yDot + p.offsetToBulk + comp;
				for (unsigned int col = 0; col < p.nCol; ++col)
					resBulkComp[col * p.strideCell] = yDotBulkComp[col * p.strideCell];
			}
			else
			{
				for (unsigned int col = 0; col < p.nCol; ++col)
					resBulkComp[col * p.strideCell] = 0.0;
			}

			// Fill stencil (left side with zeros, right side with states)
			for (int i = -std::max(1, 2) + 1; i < 0; ++i)
				stencil[i] = 0.0;
			for (int i = 0; i < std::max(1, 2); ++i)
				stencil[i] = yBulkComp[i * p.strideCell];

			// Reset WENO output
			StateType vm(0.0); // reconstructed value
//			if (wantJac)
//				std::fill(p.wenoDerivatives, p.wenoDerivatives + p.weno->stencilSize(), 0.0);

			int wenoOrder = 1;
			const ParamType d_rad = static_cast<ParamType>(p.d_rad[comp]);

			// Iterate over all cells
			for (unsigned int col = 0; col < p.nCol; ++col)
			{
				const ParamType denom = static_cast<ParamType>(p.cellCenters[col]) * static_cast<ParamType>(p.cellSizes[col]);

				// ------------------- Dispersion -------------------

				// Right side, leave out if we're in the last cell (boundary condition)
				if (cadet_likely(col < p.nCol - 1))
				{
					const double relCoord = (static_cast<double>(p.cellBounds[col+1]) - static_cast<double>(p.cellBounds[0])) / (static_cast<double>(p.cellBounds[p.nCol - 1]) - static_cast<double>(p.cellBounds[0]));
					const ParamType d_rad_right = p.parDep->getValue(p.model, ColumnPosition{relCoord, 0.0, 0.0}, comp, ParTypeIndep, BoundStateIndep, d_rad, static_cast<ParamType>(p.u) / static_cast<ParamType>(p.cellBounds[col+1]));
					resBulkComp[col * p.strideCell] -= d_rad_right * static_cast<ParamType>(p.cellBounds[col+1]) / denom * (stencil[1] - stencil[0]) / (static_cast<ParamType>(p.cellCenters[col+1]) - static_cast<ParamType>(p.cellCenters[col]));
					// Jacobian entries
					if (wantJac)
					{
						const double val = static_cast<double>(d_rad_right) * static_cast<double>(p.cellBounds[col+1]) / static_cast<double>(denom) / (static_cast<double>(p.cellCenters[col+1]) - static_cast<double>(p.cellCenters[col]));
						jac[0] += val;
						jac[p.strideCell] -= val;
					}
				}

				// Left side, leave out if we're in the first cell (boundary condition)
				if (cadet_likely(col > 0))
				{
					const double relCoord = (static_cast<double>(p.cellBounds[col]) - static_cast<double>(p.cellBounds[0])) / (static_cast<double>(p.cellBounds[p.nCol - 1]) - static_cast<double>(p.cellBounds[0]));
					const ParamType d_rad_left = p.parDep->getValue(p.model, ColumnPosition{relCoord, 0.0, 0.0}, comp, ParTypeIndep, BoundStateIndep, d_rad, static_cast<ParamType>(p.u) / static_cast<ParamType>(p.cellBounds[col]));
					resBulkComp[col * p.strideCell] -= d_rad_left * static_cast<ParamType>(p.cellBounds[col]) / denom * (stencil[-1] - stencil[0]) / (static_cast<ParamType>(p.cellCenters[col-1]) - static_cast<ParamType>(p.cellCenters[col]));
					// Jacobian entries
					if (wantJac)
					{
						const double val = static_cast<double>(d_rad_left) * static_cast<double>(p.cellBounds[col]) / static_cast<double>(denom) / (static_cast<double>(p.cellCenters[col-1]) - static_cast<double>(p.cellCenters[col]));
						jac[0] += val;
						jac[-p.strideCell] -= val;
					}
				}

				// ------------------- Convection -------------------

				// Add convection through this cell's left face
				if (cadet_likely(col > 0))
				{
					// Remember that vm still contains the reconstructed value of the previous 
					// cell's *right* face, which is identical to this cell's *left* face!
					resBulkComp[col * p.strideCell] -= p.u / denom * vm;

					// Jacobian entries
					if (wantJac)
					{
						for (int i = 0; i < 2 * wenoOrder - 1; ++i)
							// Note that we have an offset of -1 here (compared to the right cell face below), since
							// the reconstructed value depends on the previous stencil (which has now been moved by one cell)
							jac[(i - wenoOrder) * p.strideCell] -= static_cast<double>(p.u) / static_cast<double>(denom);
					}
				}
				else
				{
					// In the first cell we need to apply the boundary condition: inflow concentration
					resBulkComp[col * p.strideCell] -= p.u / denom * y[p.offsetToInlet + comp];
				}

				// Reconstruct concentration on this cell's right face
				if (wantJac)
				{
					wenoOrder = 1;
					vm = stencil[0];
					// wenoOrder = p.weno->template reconstruct<StateType, StencilType>(p.wenoEpsilon, col, p.nCol, stencil, vm, p.wenoDerivatives);
				}
				else
				{
					wenoOrder = 1;
					vm = stencil[0];
					// wenoOrder = p.weno->template reconstruct<StateType, StencilType>(p.wenoEpsilon, col, p.nCol, stencil, vm);
				}

				// Right side
				resBulkComp[col * p.strideCell] += p.u / denom * vm;
				// Jacobian entries
				if (wantJac)
				{
					for (int i = 0; i < 2 * wenoOrder - 1; ++i)
						jac[(i - wenoOrder + 1) * p.strideCell] += static_cast<double>(p.u) / static_cast<double>(denom);
				}

				// Update stencil
				const unsigned int shift = std::max(1, 2);
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

	template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
	int residualBackwardsRadialFlow(const SimulationTime& simTime, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin, const RadialFlowParameters<ParamType>& p)
	{
		// The stencil caches parts of the state vector for better spatial coherence
		typedef CachingStencil<StateType, ArrayPool> StencilType;
//		StencilType stencil(std::max(p.weno->stencilSize(), 3u), *p.stencilMemory, std::max(p.weno->order() - 1, 1));
		StencilType stencil(std::max(1u, 3u), *p.stencilMemory, std::max(1 - 1, 1));

		// The RowIterator is always centered on the main diagonal.
		// This means that jac[0] is the main diagonal, jac[-1] is the first lower diagonal,
		// and jac[1] is the first upper diagonal. We can also access the rows from left to
		// right beginning with the last lower diagonal moving towards the main diagonal and
		// continuing to the last upper diagonal by using the native() method.
		RowIteratorType jac;

		ResidualType* const resBulk = res + p.offsetToBulk;
		StateType const* const yBulk = y + p.offsetToBulk;

		for (unsigned int comp = 0; comp < p.nComp; ++comp)
		{
			if (wantJac)
				jac = jacBegin + p.strideCell * (p.nCol - 1) + comp;

			ResidualType* const resBulkComp = resBulk + comp;
			StateType const* const yBulkComp = yBulk + comp;

			// Add time derivative to each cell
			if (yDot)
			{
				double const* const yDotBulkComp = yDot + p.offsetToBulk + comp;
				for (unsigned int col = 0; col < p.nCol; ++col)
					resBulkComp[col * p.strideCell] = yDotBulkComp[col * p.strideCell];
			}
			else
			{
				for (unsigned int col = 0; col < p.nCol; ++col)
					resBulkComp[col * p.strideCell] = 0.0;
			}

			// Fill stencil (left side with zeros, right side with states)
			for (int i = -std::max(1, 2) + 1; i < 0; ++i)
				stencil[i] = 0.0;
			for (int i = 0; i < std::max(1, 2); ++i)
				stencil[i] = yBulkComp[(p.nCol - static_cast<unsigned int>(i) - 1) * p.strideCell];

			// Reset WENO output
			StateType vm(0.0); // reconstructed value
//			if (wantJac)
//				std::fill(p.wenoDerivatives, p.wenoDerivatives + p.weno->stencilSize(), 0.0);

			int wenoOrder = 1;
			const ParamType d_rad = static_cast<ParamType>(p.d_rad[comp]);

			// Iterate over all cells (backwards)
			// Note that col wraps around to unsigned int's maximum value after 0
			for (unsigned int col = p.nCol - 1; col < p.nCol; --col)
			{
				const ParamType denom = static_cast<ParamType>(p.cellCenters[col]) * static_cast<ParamType>(p.cellSizes[col]);

				// ------------------- Dispersion -------------------

				// Right side, leave out if we're in the first cell (boundary condition)
				if (cadet_likely(col < p.nCol - 1))
				{
					const double relCoord = (static_cast<double>(p.cellBounds[col+1]) - static_cast<double>(p.cellBounds[0])) / (static_cast<double>(p.cellBounds[p.nCol - 1]) - static_cast<double>(p.cellBounds[0]));
					const ParamType d_rad_right = p.parDep->getValue(p.model, ColumnPosition{relCoord, 0.0, 0.0}, comp, ParTypeIndep, BoundStateIndep, d_rad, static_cast<ParamType>(p.u) / static_cast<ParamType>(p.cellBounds[col+1]));
					resBulkComp[col * p.strideCell] -= d_rad_right * static_cast<ParamType>(p.cellBounds[col+1]) / denom * (stencil[-1] - stencil[0]) / (static_cast<ParamType>(p.cellCenters[col+1]) - static_cast<ParamType>(p.cellCenters[col]));
					// Jacobian entries
					if (wantJac)
					{
						const double val = static_cast<double>(d_rad_right) * static_cast<double>(p.cellBounds[col+1]) / static_cast<double>(denom) / (static_cast<double>(p.cellCenters[col+1]) - static_cast<double>(p.cellCenters[col]));
						jac[0] += val;
						jac[p.strideCell] -= val;
					}
				}

				// Left side, leave out if we're in the last cell (boundary condition)
				if (cadet_likely(col > 0))
				{
					const double relCoord = (static_cast<double>(p.cellBounds[col]) - static_cast<double>(p.cellBounds[0])) / (static_cast<double>(p.cellBounds[p.nCol - 1]) - static_cast<double>(p.cellBounds[0]));
					const ParamType d_rad_left = p.parDep->getValue(p.model, ColumnPosition{relCoord, 0.0, 0.0}, comp, ParTypeIndep, BoundStateIndep, d_rad, static_cast<ParamType>(p.u) / static_cast<ParamType>(p.cellBounds[col]));
					resBulkComp[col * p.strideCell] -= d_rad_left * static_cast<ParamType>(p.cellBounds[col]) / denom * (stencil[1] - stencil[0]) / (static_cast<ParamType>(p.cellCenters[col-1]) - static_cast<ParamType>(p.cellCenters[col]));
					// Jacobian entries
					if (wantJac)
					{
						const double val = static_cast<double>(d_rad_left) * static_cast<double>(p.cellBounds[col]) / static_cast<double>(denom) / (static_cast<double>(p.cellCenters[col-1]) - static_cast<double>(p.cellCenters[col]));
						jac[0] += val;
						jac[-p.strideCell] -= val;
					}
				}

				// ------------------- Convection -------------------

				// Add convection through this cell's right face
				if (cadet_likely(col < p.nCol - 1))
				{
					// Remember that vm still contains the reconstructed value of the previous 
					// cell's *left* face, which is identical to this cell's *right* face!
					resBulkComp[col * p.strideCell] += p.u / denom * vm;

					// Jacobian entries
					if (wantJac)
					{
						for (int i = 0; i < 2 * wenoOrder - 1; ++i)
							// Note that we have an offset of +1 here (compared to the left cell face below), since
							// the reconstructed value depends on the previous stencil (which has now been moved by one cell)
							jac[(wenoOrder - i) * p.strideCell] += static_cast<double>(p.u) / static_cast<double>(denom);
					}
				}
				else
				{
					// In the last cell (z = L) we need to apply the boundary condition: inflow concentration
					resBulkComp[col * p.strideCell] += p.u / denom * y[p.offsetToInlet + comp];
				}

				// Reconstruct concentration on this cell's left face
				if (wantJac)
				{
					wenoOrder = 1;
					vm = stencil[0];
//					wenoOrder = p.weno->template reconstruct<StateType, StencilType>(p.wenoEpsilon, col, p.nCol, stencil, vm, p.wenoDerivatives);
				}
				else
				{
					wenoOrder = 1;
					vm = stencil[0];
//					wenoOrder = p.weno->template reconstruct<StateType, StencilType>(p.wenoEpsilon, col, p.nCol, stencil, vm);
				}

				// Left face
				resBulkComp[col * p.strideCell] -= p.u / denom * vm;
				// Jacobian entries
				if (wantJac)
				{
					for (int i = 0; i < 2 * wenoOrder - 1; ++i)
						jac[(wenoOrder - i - 1) * p.strideCell] -= static_cast<double>(p.u) / static_cast<double>(denom);
				}

				// Update stencil (be careful because of wrap-around, might cause reading memory very far away [although never used])
				const unsigned int shift = std::max(1, 2);
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


template <typename StateType, typename ResidualType, typename ParamType, typename RowIteratorType, bool wantJac>
int residualKernelRadial(const SimulationTime& simTime, StateType const* y, double const* yDot, ResidualType* res, RowIteratorType jacBegin, const RadialFlowParameters<ParamType>& p)
{
	if (p.u >= 0.0)
		return impl::residualForwardsRadialFlow<StateType, ResidualType, ParamType, RowIteratorType, wantJac>(simTime, y, yDot, res, jacBegin, p);
	else
		return impl::residualBackwardsRadialFlow<StateType, ResidualType, ParamType, RowIteratorType, wantJac>(simTime, y, yDot, res, jacBegin, p);
}

void sparsityPatternRadial(linalg::SparsityPatternRowIterator itBegin, unsigned int nComp, unsigned int nCol, int strideCell, double u, Weno& weno);

} // namespace convdisp
} // namespace parts
} // namespace model
} // namespace cadet

#endif  // LIBCADET_RADIALCONVECTIONDISPERSIONKERNEL_HPP_
