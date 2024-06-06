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

#include "model/LumpedRateModelWithPoresDG2D.hpp"
#include "model/BindingModel.hpp"
#include "model/parts/BindingCellKernel.hpp"
#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"
#include "AdUtils.hpp"

#include <algorithm>
#include <functional>

#include <iostream> // todo delete
#include <iomanip>

#include "LoggingUtils.hpp"
#include "Logging.hpp"

#include "ParallelSupport.hpp"

#ifdef CADET_PARALLELIZE
	#include <tbb/parallel_for.h>
	#include <tbb/flow_graph.h>

	typedef tbb::flow::continue_node< tbb::flow::continue_msg > node_t;
	typedef const tbb::flow::continue_msg & msg_t;
#endif

namespace cadet
{

namespace model
{
/**
 * @brief Computes the solution of the linear system involving the system Jacobian
 * @details The system \f[ \left( \frac{\partial F}{\partial y} + \alpha \frac{\partial F}{\partial \dot{y}} \right) x = b \f]
 *          has to be solved. The right hand side \f$ b \f$ is given by @p rhs, the Jacobians are evaluated at the
 *          point \f$(y, \dot{y})\f$ given by @p y and @p yDot. The residual @p res at this point, \f$ F(t, y, \dot{y}) \f$,
 *          may help with this. Error weights (see IDAS guide) are given in @p weight. The solution is returned in @p rhs.
 *
 *          The full Jacobian @f$ J = \left( \frac{\partial F}{\partial y} + \alpha \frac{\partial F}{\partial \dot{y}} \right) @f$ is given by
 *          @f[ \begin{align}
				J =
				\left[\begin{array}{c|ccc|c}
					 J_0 & & & & J_{0,f} \\
					\hline
					 & J_1 & & & J_{1,f} \\
					 & & \ddots & & \vdots \\
					 & & & J_{N_z} & J_{N_z,f} \\
					\hline
					 J_{f,0} & J_{f,1} & \dots & J_{f,N_z} & J_f
				\end{array}\right].
			\end{align} @f]
 *          By decomposing the Jacobian @f$ J @f$ into @f$ J = LU @f$, we get
 *          @f[ \begin{align}
				L &= \left[\begin{array}{c|ccc|c}
					  J_0     &         &        &           &   \\
					\hline
							  & J_1     &        &           &   \\
							  &         & \ddots &           &   \\
							  &         &        & J_{N_z}   &   \\
					\hline
					  J_{f,0} & J_{f,1} & \dots & J_{f,N_z} & I
				\end{array}\right], \\
				U &= \left[\begin{array}{c|ccc|c}
					  I &   &        &   & J_0^{-1} \, J_{0,f}       \\
					\hline
						& I &        &   & J_1^{-1} \, J_{1,f}       \\
						&   & \ddots &   & \vdots                    \\
						&   &        & I & J_{N_z}^{-1} \, J_{N_z,f} \\
					\hline
						&   &        &   & S
				\end{array}\right].
			\end{align} @f]
 *          Here, the Schur-complement @f$ S @f$ is given by
 *          @f[ \begin{align}
				S = J_f - J_{f,0} \, J_0^{-1} \, J_{0,f} - \sum_{p=1}^{N_z}{J_{f,p} \, J_p^{-1} \, J_{p,f}}.
			\end{align} @f]
 *          Note that @f$ J_f = I @f$ is the identity matrix and that the off-diagonal blocks @f$ J_{i,f} @f$
 *          and @f$ J_{f,i} @f$ for @f$ i = 0, \dots, N_{z} @f$ are sparse.
 *
 *          Exploiting the decomposition, the solution procedure @f$ x = J^{-1}b = \left( LU \right)^{-1}b = U^{-1} L^{-1} b @f$
 *          works as follows:
 *              -# Factorize the diagonal blocks @f$ J_0, \dots, J_{N_z} @f$
 *              -# Solve @f$ y = L^{-1} b @f$ by forward substitution. This is accomplished by first solving the diagonal
 *                 blocks independently, that is,
 *                 @f[ y_i = J_{i}^{-1} b_i. @f]
 *                 Then, calculate the flux-part @f$ y_f @f$ by substituting in the already calculated solutions @f$ y_i @f$:
 *                 @f[ y_f = b_f - \sum_{i=0}^{N_z} J_{f,i} y_i. @f]
 *              -# Solve the Schur-complement @f$ S x_f = y_f @f$ using an iterative method that only requires
 *                 matrix-vector products. The already inverted diagonal blocks @f$ J_i^{-1} @f$ come in handy here.
 *              -# Solve the rest of the @f$ U x = y @f$ system by backward substitution. To be more precise, compute
 *                 @f[ x_i = y_i - J_i^{-1} J_{i,f} y_f. @f]
 *
 *
 * @param [in] t Current time point
 * @param [in] alpha Value of \f$ \alpha \f$ (arises from BDF time discretization)
 * @param [in] outerTol Error tolerance for the solution of the linear system from outer Newton iteration
 * @param [in,out] rhs On entry the right hand side of the linear equation system, on exit the solution
 * @param [in] weight Vector with error weights
 * @param [in] simState State of the simulation (state vector and its time derivatives) at which the Jacobian is evaluated
 * @return @c 0 on success, @c -1 on non-recoverable error, and @c +1 on recoverable error
 */
int LumpedRateModelWithPoresDG2D::linearSolve(double t, double alpha, double outerTol, double* const rhs, double const* const weight,
	const ConstSimulationState& simState)
{
	BENCH_SCOPE(_timerLinearSolve);

	Indexer idxr(_disc);

	// ==== Step 1: Factorize global Jacobian (without inlet DOFs)

	// Factorize partial Jacobians only if required
	if (_factorizeJacobian)
	{
		// Assemble and factorize discretized bulk Jacobian
		assembleDiscretizedGlobalJacobian(alpha, idxr);

		_globalSolver.factorize(_globalJacDisc);

		if (cadet_unlikely(_globalSolver.info() != Eigen::Success)) {
			LOG(Error) << "Factorize() failed";
		}

		// Do not factorize again at next call without changed Jacobians
		_factorizeJacobian = false;
	}

	// ====== Step 1.5: Solve J c_uo = b_uo - A * c_in = b_uo - A*b_in

	//std::cout << std::setprecision(1);
	//std::cout << "_jacInlet\n" << _jacInlet << std::endl; // todo delete
	//std::cout << "_globalJacDisc\n" << _globalJacDisc.toDense() << std::endl;

	// rhs is passed twice but due to the values in jacA the writes happen to a different area of the rhs than the reads.

	// Handle inlet DOFs: // todo backward flow.
	for (int comp = 0; comp < _disc.nComp; comp++)
	{
		Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>, 0, Eigen::InnerStride<Eigen::Dynamic>> rInlet(rhs + comp, _disc.radNPoints, Eigen::InnerStride<Eigen::Dynamic>(idxr.strideColRadialNode()));
		Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>, 0, Eigen::InnerStride<Eigen::Dynamic>> rInletDep(rhs + idxr.offsetC() + comp, _disc.axNPoints * _disc.radNPoints, Eigen::InnerStride<Eigen::Dynamic>(idxr.strideColRadialNode()));

		rInletDep += _jacInlet * rInlet;
	}

	// ==== Step 2: Solve system of pure DOFs
	// The result is stored in rhs (in-place solution)

	Eigen::Map<Eigen::VectorXd> r(rhs, numDofs());

	r.segment(idxr.offsetC(), numPureDofs()) = _globalSolver.solve(r.segment(idxr.offsetC(), numPureDofs()));

	if (cadet_unlikely(_globalSolver.info() != Eigen::Success))
	{
		LOG(Error) << "Solve() failed";
	}

	// The full solution is now stored in rhs
	return 0;
}
/**
 * @brief Assembles bulk Jacobian @f$ J_i @f$ (@f$ i > 0 @f$) of the time-discretized equations
 * @details The system \f[ \left( \frac{\partial F}{\partial y} + \alpha \frac{\partial F}{\partial \dot{y}} \right) x = b \f]
 *          has to be solved. The system Jacobian of the original equations,
 *          \f[ \frac{\partial F}{\partial y}, \f]
 *          is already computed (by AD or manually in residualImpl() with @c wantJac = true). This function is responsible
 *          for adding
 *          \f[ \alpha \frac{\partial F}{\partial \dot{y}} \f]
 *          to the system Jacobian, which yields the Jacobian of the time-discretized equations
 *          \f[ F\left(t, y_0, \sum_{k=0}^N \alpha_k y_k \right) = 0 \f]
 *          when a BDF method is used. The time integrator needs to solve this equation for @f$ y_0 @f$, which requires
 *          the solution of the linear system mentioned above (@f$ \alpha_0 = \alpha @f$ given in @p alpha).
 *
 * @param [in] alpha Value of \f$ \alpha \f$ (arises from BDF time discretization)
 */
void LumpedRateModelWithPoresDG2D::assembleDiscretizedGlobalJacobian(double alpha, Indexer idxr) {

	// set to static (per section) jacobian
	_globalJacDisc = _globalJac;

	// add time derivative to bulk jacobian
	_convDispOp.addTimeDerivativeToJacobian(alpha, _globalJacDisc);

	// Add time derivatives to particle shells
	for (unsigned int parType = 0; parType < _disc.nParType; parType++) {
		linalg::BandedEigenSparseRowIterator jac(_globalJacDisc, idxr.offsetCp(ParticleTypeIndex{ parType }) - idxr.offsetC());
		for (unsigned int j = 0; j < _disc.nBulkPoints; ++j)
		{
			// Mobile and solid phase (advances jac accordingly)
			addTimeDerivativeToJacobianParticleBlock(jac, idxr, alpha, parType);
		}
	}
}
/**
 * @brief Adds Jacobian @f$ \frac{\partial F}{\partial \dot{y}} @f$ to bead rows of system Jacobian
 * @details Actually adds @f$ \alpha \frac{\partial F}{\partial \dot{y}} @f$, which is useful
 *          for constructing the linear system in BDF time discretization.
 * @param [in,out] jac On entry, RowIterator of the particle block pointing to the beginning of a bead shell;
 *                     on exit, the iterator points to the end of the bead shell
 * @param [in] idxr Indexer
 * @param [in] alpha Value of \f$ \alpha \f$ (arises from BDF time discretization)
 * @param [in] parType Index of the particle type
 */
void LumpedRateModelWithPoresDG2D::addTimeDerivativeToJacobianParticleBlock(linalg::BandedEigenSparseRowIterator& jac, const Indexer& idxr, double alpha, unsigned int parType)
{
	// Mobile phase
	for (int comp = 0; comp < static_cast<int>(_disc.nComp); ++comp, ++jac)
	{
		// Add derivative with respect to dc_p / dt to Jacobian
		jac[0] += alpha;

		const double invBetaP = (1.0 - static_cast<double>(_parPorosity[parType])) / (static_cast<double>(_poreAccessFactor[parType * _disc.nComp + comp]) * static_cast<double>(_parPorosity[parType]));

		// Add derivative with respect to dq / dt to Jacobian
		const int nBound = static_cast<int>(_disc.nBound[parType * _disc.nComp + comp]);
		for (int i = 0; i < nBound; ++i)
		{
			// Index explanation:
			//   -comp -> go back to beginning of liquid phase
			//   + strideParLiquid() skip to solid phase
			//   + offsetBoundComp() jump to component (skips all bound states of previous components)
			//   + i go to current bound state
			jac[idxr.strideParLiquid() - comp + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ static_cast<unsigned int>(comp) }) + i] += alpha * invBetaP;
		}
	}

	// Solid phase
	int const* const qsReaction = _binding[parType]->reactionQuasiStationarity();
	for (unsigned int bnd = 0; bnd < _disc.strideBound[parType]; ++bnd, ++jac)
	{
		// Add derivative with respect to dynamic states to Jacobian
		if (qsReaction[bnd])
			continue;

		// Add derivative with respect to dq / dt to Jacobian
		jac[0] += alpha;
	}
}

}  // namespace model
}  // namespace cadet
