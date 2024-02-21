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
 * Implements the kernel of a particle cell with mobile and solid phase.
 */

#ifndef LIBCADET_BINDINGCELLKERNEL_HPP_
#define LIBCADET_BINDINGCELLKERNEL_HPP_

#include "AutoDiff.hpp"
#include "model/BindingModel.hpp"
#include "model/ReactionModel.hpp"
#include "SimulationTypes.hpp"

#include "LoggingUtils.hpp"
#include "Logging.hpp"

namespace cadet
{

	namespace model
	{

		namespace parts
		{

			namespace cell
			{

				namespace
				{
					template <typename KernelParamsType>
					void bindingFlux(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* y, active* res, const KernelParamsType& params, LinearBufferAllocator buffer, WithParamSensitivity)
					{
						params.binding->flux(t, secIdx, colPos, y, y - params.nComp, res, buffer, WithParamSensitivity());
					}

					template <typename KernelParamsType>
					void bindingFlux(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* y, active* res, const KernelParamsType& params, LinearBufferAllocator buffer, WithoutParamSensitivity)
					{
						params.binding->flux(t, secIdx, colPos, y, y - params.nComp, res, buffer, WithoutParamSensitivity());
					}

					template <typename KernelParamsType>
					void bindingFlux(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, active* res, const KernelParamsType& params, LinearBufferAllocator buffer, WithParamSensitivity)
					{
						params.binding->flux(t, secIdx, colPos, y, y - params.nComp, res, buffer);
					}

					template <typename KernelParamsType>
					void bindingFlux(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double* res, const KernelParamsType& params, LinearBufferAllocator buffer, WithoutParamSensitivity)
					{
						params.binding->flux(t, secIdx, colPos, y, y - params.nComp, res, buffer);
					}
				}

				struct CellParameters
				{
					unsigned int nComp;
					unsigned int const* nBound;
					unsigned int const* boundOffset;
					unsigned int nTotalBound;
					int const* qsReaction;
					const active& porosity;
					active const* poreAccessFactor;
					IBindingModel* binding;
					IDynamicReactionModel* dynReaction;
				};

				template <typename StateType, typename ResidualType, typename ParamType, typename KernelParamsType, typename RowIteratorType, bool wantJac, bool handleMobilePhaseDerivative>
				void residualKernel(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* y,
					double const* yDot, ResidualType* res, RowIteratorType jacBase, const KernelParamsType& params, LinearBufferAllocator buffer)
				{
					// Mobile phase
					if (handleMobilePhaseDerivative)
						std::fill(res, res + params.nComp, 0.0);

					// Add time derivatives
					if (yDot)
					{
						for (unsigned int comp = 0; comp < params.nComp; ++comp, ++res, ++y)
						{
							const ParamType invBetaP = (1.0 - static_cast<ParamType>(params.porosity)) / (params.poreAccessFactor ? static_cast<ParamType>(params.poreAccessFactor[comp]) * static_cast<ParamType>(params.porosity) : static_cast<ParamType>(params.porosity));

							// Ultimately, we need dc_{p,comp} / dt + 1 / beta_p * [ sum_i  dq_comp^i / dt ]
							// where the bound states in the brackets are the quasi-stationary states only.
							// Compute the sum in the brackets first, then divide by beta_p and add dc_p / dt

							// Sum dq_comp^1 / dt + dq_comp^2 / dt + ... + dq_comp^{N_comp} / dt
							double dqSum = 0.0;
							for (unsigned int i = 0; i < params.nBound[comp]; ++i)
							{
								// Index explanation:
								//   + nComp skip to solid phase
								//   + boundOffset[comp] jump to component (skips all bound states of previous components)
								//   + i go to current bound state
								dqSum += yDot[params.nComp + params.boundOffset[comp] + i];
							}

							// Divide by beta_p and add dcp_i / dt
							if (handleMobilePhaseDerivative)
								*res = (yDot[comp] + invBetaP * dqSum);
							else
								*res += invBetaP * dqSum;
						}
					}
					else
					{
						// Move pointers to solid phase
						res += params.nComp;
						y += params.nComp;
					}

					// Solid phase

					// Binding
					bindingFlux(t, secIdx, colPos, y, res, params, buffer, typename ParamSens<ParamType>::enabled());
					if (wantJac)
					{
						// static_cast should be sufficient here, but this statement is also analyzed when wantJac = false
						params.binding->analyticJacobian(t, secIdx, colPos, reinterpret_cast<double const*>(y), params.nComp, jacBase + params.nComp, buffer);
					}

					if (params.binding->hasDynamicReactions() && yDot)
					{
						unsigned int idx = 0;
						for (unsigned int comp = 0; comp < params.nComp; ++comp)
						{
							for (unsigned int state = 0; state < params.nBound[comp]; ++state, ++idx)
							{
								// Skip quasi-stationary fluxes
								if (params.qsReaction[idx])
									continue;

								// Add time derivative to solid phase
								res[idx] += yDot[params.nComp + idx];
							}
						}
					}

					// Reaction

					if (params.dynReaction)
					{
						BufferedArray<ResidualType> fluxSolid = buffer.template array<ResidualType>(params.nTotalBound);

						std::fill_n(static_cast<ResidualType*>(fluxSolid), params.nTotalBound, 0.0);
						params.dynReaction->residualCombinedAdd(t, secIdx, colPos, y - params.nComp, y, res - params.nComp, static_cast<ResidualType*>(fluxSolid), -1.0, buffer);

						unsigned int idx = 0;
						for (unsigned int comp = 0; comp < params.nComp; ++comp)
						{
							const ParamType invBetaP = (1.0 - static_cast<ParamType>(params.porosity)) / (params.poreAccessFactor ? static_cast<ParamType>(params.poreAccessFactor[comp]) * static_cast<ParamType>(params.porosity) : static_cast<ParamType>(params.porosity));

							for (unsigned int bnd = 0; bnd < params.nBound[comp]; ++bnd, ++idx)
							{
								// Add reaction term to mobile phase
								res[-static_cast<int>(params.nComp) + static_cast<int>(comp)] += static_cast<typename DoubleActiveDemoter<ParamType, ResidualType>::type>(invBetaP) * fluxSolid[idx];

								if (!params.qsReaction[idx])
								{
									// Add reaction term to solid phase
									res[idx] += fluxSolid[idx];
								}
							}
						}

						if (wantJac)
						{
							if (params.nTotalBound > 0)
							{
								BufferedArray<double> fluxSolidJacobian = buffer.template array<double>(params.nTotalBound * (params.nTotalBound + params.nComp));
								linalg::DenseMatrixView dmv(static_cast<double*>(fluxSolidJacobian), nullptr, params.nTotalBound, params.nTotalBound + params.nComp);
								dmv.setAll(0.0);

								// static_cast should be sufficient here, but this statement is also analyzed when wantJac = false
								params.dynReaction->analyticJacobianCombinedAdd(t, secIdx, colPos, reinterpret_cast<double const*>(y - params.nComp), reinterpret_cast<double const*>(y), -1.0, jacBase, dmv.row(0, params.nComp), buffer);

								idx = 0;
								for (unsigned int comp = 0; comp < params.nComp; ++comp)
								{
									const double invBetaP = (1.0 - static_cast<double>(params.porosity)) / (params.poreAccessFactor ? static_cast<double>(params.poreAccessFactor[comp]) * static_cast<double>(params.porosity) : static_cast<double>(params.porosity));

									for (unsigned int bnd = 0; bnd < params.nBound[comp]; ++bnd, ++idx)
									{
										// Add Jacobian row to mobile phase
										(jacBase + comp).addArray(dmv.rowPtr(idx), -static_cast<int>(comp), dmv.columns(), invBetaP);

										if (!params.qsReaction[idx])
										{
											// Add Jacobian row to solid phase
											(jacBase + params.nComp + idx).addArray(dmv.rowPtr(idx), -static_cast<int>(params.nComp + idx), dmv.columns(), 1.0);
										}
									}
								}
							}
							else
							{
								// We do not have bound states, but still need to obtain the Jacobian for the liquid phase.
								// So we pass a row iterator for the solid phase that does not point anywhere and hope that the
								// reaction model does not interact with it.

								// static_cast should be sufficient here, but this statement is also analyzed when wantJac = false
								params.dynReaction->analyticJacobianCombinedAdd(t, secIdx, colPos, reinterpret_cast<double const*>(y - params.nComp), reinterpret_cast<double const*>(y), -1.0, jacBase, linalg::DenseBandedRowIterator(), buffer);
							}
						}
					}
				}


				/**
				 * @brief Executes multiplication of particle shell Jacobian wrt. to state variable
				 * @details [long description]
				 * @param [in] mobileSdot Vector @f$ s @f$ the Jacobian is multiplied with
				 * @param [out] mobileRes Resulting vector @f$ \alpha Js @f$
				 * @param [in] nComp Number of components
				 * @param [in] nBoundPerComp Array with number of bound states for each component
				 * @param [in] boundOffset Array with offset to bound states of each component
				 * @param [in] nTotalBound Total number of bound states (of all components)
				 * @param [in] qsReaction Array that indicates whether a reaction is quasi-stationary
				 * @param [in] factor Factor @f$ \alpha @f$
				 * @param [in] qsFactor Factor of the @f$ \mathrm{d}q / \mathrm{d}t @f$ terms added to the mobile phase (only for quasi-stationary bound states)
				 */
				template <bool handleMobilePhaseDerivative>
				inline void multiplyWithDerivativeJacobianKernel(double const* const mobileSdot, double* const mobileRes, unsigned int nComp,
					unsigned int const* const nBoundPerComp, unsigned int const* const boundOffset, const unsigned int nTotalBound,
					int const* const qsReaction, double factor, double qsFactor)
				{
					// Mobile phase
					for (unsigned int comp = 0; comp < nComp; ++comp)
					{
						// Add derivative with respect to dc_p / dt to Jacobian
						if (handleMobilePhaseDerivative)
							mobileRes[comp] = factor * mobileSdot[comp];

						// Add derivative with respect to quasi-stationary dq / dt to Jacobian
						for (unsigned int i = 0; i < nBoundPerComp[comp]; ++i)
						{
							// Index explanation:
							//   nComp -> skip mobile phase
							//   + boundOffset[comp] skip bound states of all previous components
							//   + i go to current bound state
							mobileRes[comp] += qsFactor * mobileSdot[nComp + boundOffset[comp] + i];
						}
					}

					// Solid phase
					double const* const solidSdot = mobileSdot + nComp;
					double* const solidRet = mobileRes + nComp;

					for (unsigned int bnd = 0; bnd < nTotalBound; ++bnd)
					{
						// Add derivative with respect to dynamic states to Jacobian
						if (qsReaction[bnd])
							solidRet[bnd] = 0.0;
						else
							solidRet[bnd] = factor * solidSdot[bnd];
					}
				}

				/**
				 * @brief Adds Jacobian @f$ \frac{\partial F}{\partial \dot{y}} @f$ to bead rows of system Jacobian
				 * @details Actually adds @f$ \alpha \frac{\partial F}{\partial \dot{y}} @f$, which is useful
				 *          for constructing the linear system in BDF time discretization.
				 * @param [in,out] jac On entry, RowIterator of the particle block pointing to the beginning of a bead shell;
				 *                     on exit, the iterator points to the end of the bead shell
				 * @param [in] alpha Value of \f$ \alpha \f$ (arises from BDF time discretization)
				 * @param [in] porosity Particle porosity
				 * @param [in] nComp Number of components
				 * @param [in] nBoundPerComp Array with number of bound states for each component
				 * @param [in] poreAccessFactor Array with pore access factors
				 * @param [in] nTotalBound Total number of bound states
				 * @param [in] offsetBoundComp Array with offsets to bound states of each component
				 * @param [in] qsReaction Array that indicates whether a reaction is quasi-stationary
				 */
				template <typename rowIter_t, bool handleMobilePhaseDerivative>
				void addTimeDerivativeToJacobianParticleShell(rowIter_t& jac, double alpha, double porosity, int nComp, unsigned int const* const nBoundPerComp,
					active const* const poreAccessFactor, const unsigned int nTotalBound, unsigned int const* const offsetBoundComp, int const* const qsReaction)
				{
					// Mobile phase
					for (int comp = 0; comp < nComp; ++comp, ++jac)
					{
						// Add derivative with respect to dc_p / dt to Jacobian
						if (handleMobilePhaseDerivative)
							jac[0] += alpha;

						const double invBetaP = (1.0 - porosity) / (static_cast<double>(poreAccessFactor[comp]) * porosity);

						// Add derivative with respect to dq / dt to Jacobian
				//		const int nBound = static_cast<int>(nBoundPerComp[comp]);
						for (int i = 0; i < static_cast<int>(nBoundPerComp[comp]); ++i)
						{
							const int idxBoundState = offsetBoundComp[comp] + i;

							// Index explanation:
							//   -comp -> go back to beginning of liquid phase
							//   + nComp skip to solid phase
							//   + idxBoundState go to current bound state
							jac[nComp - comp + idxBoundState] += alpha * invBetaP;
						}
					}

					// Solid phase
					for (unsigned int bnd = 0; bnd < nTotalBound; ++bnd, ++jac)
					{
						// Add derivative with respect to dynamic states to Jacobian
						if (qsReaction[bnd])
							continue;

						// Add derivative with respect to dq / dt to Jacobian
						jac[0] += alpha;
					}
				}

			} // namespace cell
		} // namespace parts
	} // namespace model
} // namespace cadet

#endif  // LIBCADET_BINDINGCELLKERNEL_HPP_