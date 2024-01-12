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
 * Defines a discrete calculus toolbox for nodal spectral elements used in Discontinuous Galerkin Spectral Element Methods (DGSEM).
 * @details See @cite Kopriva2009
 */

#ifndef LIBCADET_DGTOOLBOX_HPP_
#define LIBCADET_DGTOOLBOX_HPP_

#include <Eigen/Dense>

namespace cadet
{

namespace model
{

namespace parts
{

namespace dgtoolbox
{
/**
 * @brief computes the node in physical space corresponding to a reference node
 * @param [in] delta element spacing
 * @param [in] elemIdx element index starting at 0
 * @param [in] xi reference node
 */
template <typename ParamType>
ParamType mapRefToPhys(const std::vector<ParamType> delta, const unsigned int elemIdx, const double xi)
{
	//return std::accumulate(delta.begin(), delta.begin() + elemIdx, delta[elemIdx] / 2.0 * (xi + 1.0));

	ParamType map = 0.0;
	for (unsigned int i = 0; i < elemIdx; i++)
		map += delta[i];
	return map + delta[elemIdx] / 2.0 * (xi + 1.0);
}
/**
 * @brief computes the node in referece space corresponding to a physical node
 * @param [in] deltaX element spacing
 * @param [in] elemIdx element index starting at 0
 * @param [in] x physical node
 */
template <typename ParamType>
ParamType mapPhysToRef(const std::vector<ParamType> deltaX, const unsigned int elemIdx, const double x)
{
	//return (x - std::accumulate(deltaX.begin(), deltaX.begin() + elemIdx, 0.0)) * 2.0 / deltaX[elemIdx] - 1.0;

	ParamType map = 0.0;
	for (unsigned int i = 0; i < elemIdx; i++)
		map += deltaX[i];
	return  (x - map) * 2.0 / deltaX[elemIdx] - 1.0;
}
/**
 * @brief computes the Legendre-Gauss-Lobatto nodes and (inverse) quadrature weights
 * @param [in] polyDeg polynomial degree
 * @param [in, out] nodes Legendre Gauss Lobatto nodes
 * @param [in, out] invWeights Legendre Gauss quadrature weights
 * @param [in] invertWeights specifies if weights should be inverted
 */
void lglNodesWeights(const unsigned int polyDeg, Eigen::VectorXd& nodes, Eigen::VectorXd& invWeights, bool invertWeights = true);
/**
 * @brief computes the Legendre polynomial and its derivative
 * @param [in] polyDeg polynomial degree
 * @param [in, out] leg Legendre polynomial
 * @param [in, out] legDer Legendre polynomial derivative
 * @param [in] x evaluation point
 */
void legendrePolynomialAndDerivative(const int polyDeg, double& leg, double& legDer, const double x);
/**
 * @brief calculates the Gauss quadrature mass matrix for LGL interpolation polynomial on LG points
 * @detail exact integration of polynomials up to order 2N - 1
 * @param [in] LGLnodes Legendre Gauss Lobatto nodes
 * @param [in] nLGNodes number of Gauss quadrature nodes
 */
Eigen::MatrixXd gaussQuadratureMMatrix(const Eigen::VectorXd LGLnodes, const int nLGNodes);
/**
 * @brief calculates the barycentric weights for fast polynomial evaluation
 * @param [in] polyDeg polynomial degree
 * @param [in, out] baryWeights vector to store barycentric weights. Must already be initialized with ones!
 */
Eigen::VectorXd barycentricWeights(const unsigned int polyDeg, const Eigen::VectorXd nodes);
/**
 * @brief calculates the nodal (lagrange) polynomial derivative matrix
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
 */
Eigen::MatrixXd derivativeMatrix(const unsigned int polyDeg, const Eigen::VectorXd nodes);
/**
 * @brief calculates the inverse mass matrix via transformation to orthonormal Jacobi (modal) basis
 * @detail the mass matrix used to compute integrals of the form \int_E \ell_i(\xi) \ell_j(\xi) (1 - \xi)^\alpha (1 + \xi)^\beta d\xi
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
 * @param [in] alpha Jacobi polynomial coefficient
 * @param [in] beta Jacobi polynomial coefficient
 */
Eigen::MatrixXd invMMatrix(const unsigned int polyDeg, const Eigen::VectorXd nodes, const double alpha = 0.0, const double beta = 0.0);
/**
 * @brief calculates the mass matrix via transformation to orthonormal Jacobi (modal) basis
 * @detail mass matrix used to compute integrals of the form \int_E \ell_i(\xi) \ell_j(\xi) (1 - \xi)^\alpha (1 + \xi)^\beta d\xi
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
 */
Eigen::MatrixXd mMatrix(const unsigned int polyDeg, const Eigen::VectorXd nodes, const double alpha, const double beta);
/**
 * @brief calculates a specific second order nodal stiffness matrix
 * @detail for integrals including terms of the form (1 - \xi)^\alpha (1 + \xi)^\beta. Computation via transformation to the respective Jacobi polynomial
 * @param [in] polyDeg polynomial degree
 * @param [in] a Jacobi polynomial parameter
 * @param [in] b Jacobi polynomial parameter
 * @param [in] nodes polynomial interpolation nodes
 */
Eigen::MatrixXd secondOrderStiffnessMatrix(const unsigned int polyDeg, const double alpha, const double beta, const Eigen::VectorXd nodes);
/**
 * @brief calculates the stiffness matrix via transformation to orthonormal Jacobi (modal) basis
 * @detail exact integration for integrals of the form \int_E \ell_i(\xi) \ell_j(\xi) (1 - \xi)^\alpha (1 + \xi)^\beta d\xi
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
 * @param [in] alpha Jacobi polynomial coefficient
 * @param [in] beta Jacobi polynomial coefficient
 */
Eigen::MatrixXd stiffnessMatrix(const unsigned int polyDeg, const Eigen::VectorXd nodes, const double alpha = 0.0, const double beta = 0.0);
/**
 * @brief calculates the polynomial interpolation matrix between two sets of nodes
 * @param [in] newNodes set of nodes, the solution is interpolated to
 * @param [in] oldNodes set of nodes, the solution is interpolated from
 * @param [in] baryWeights barycentric weights of the polynomial to be interpolated
 */
Eigen::MatrixXd polynomialInterpolationMatrix(const Eigen::VectorXd newNodes, const Eigen::VectorXd oldNodes, const Eigen::VectorXd baryWeights);
/**
 * @brief returns a quadratic lifting matrix
 * @param [in] size quadratic matrix size
 */
Eigen::MatrixXd liftingMatrixQuadratic(const unsigned int size);
/**
 * @brief returns a (size x 2) lifting matrix
 * @param [in] rows number of matrix rows
 */
Eigen::MatrixXd liftingMatrix(const unsigned int size);
/**
 * @brief evaluates the jth Lagrange basis functions at given nodes
 * @param [in] j index of Lagrange basis function
 * @param [in] baseNodes interpolation nodes of Lagrange basis
 * @param [in] evalNodes evaluation nodes in [-1, 1]
 */
Eigen::VectorXd evalLagrangeBasis(const int j, const Eigen::VectorXd baseNodes, const Eigen::VectorXd evalNodes);
/**
 * @brief evaluates the jth Lagrange basis functions at given nodes
 * @param [in, out] coords DG coordinate array
 * @param [in] nElem number of DG elements
 * @param [in] nNodes number of DG nodes (per element)
 * @param [in] DGnodes array of DG nodes
 * @param [in] length domain length
 * @param [in] leftElemBndries left element boundaries array
 */
void writeDGCoordinates(double* coords, const int nElem, const int nNodes, const double* DGnodes, const double length, const double* leftElemBndries);

} // namespace dgtoolbox
} // namespace parts
} // namespace model
} // namespace cadet

#endif  // LIBCADET_DGTOOLBOX_HPP_