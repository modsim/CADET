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
 * Defines a discrete calculus toolbox for nodal spectral elements.
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
 * @brief computation of barycentric weights for fast polynomial evaluation
 * @param [in] polyDeg polynomial degree
 * @param [in, out] baryWeights vector to store barycentric weights. Must already be initialized with ones!
 */
Eigen::VectorXd barycentricWeights(const unsigned int polyDeg, const Eigen::VectorXd nodes);
/**
 * @brief computation of nodal (lagrange) polynomial derivative matrix
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
 */
Eigen::MatrixXd derivativeMatrix(const unsigned int polyDeg, const Eigen::VectorXd nodes);
/**
 * @brief calculates the inverse mass matrix via transformation to orthonormal Jacobi (modal) basis
 * @detail the mass matrix used to compute integrals of the form \int_E \ell_i(\xi) \ell_j(\xi) (1 - \xi)^\alpha (1 + \xi)^\beta d\xi
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
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

} // namespace dgtoolbox
} // namespace parts
} // namespace model
} // namespace cadet

#endif  // LIBCADET_DGTOOLBOX_HPP_