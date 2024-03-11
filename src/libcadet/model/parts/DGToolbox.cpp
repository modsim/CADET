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
 * Defines the convection dispersion transport operator according to the discontinuous Galerkin discretization.
 */

#include "model/parts/DGToolbox.hpp"
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>
#endif

using namespace Eigen;

namespace cadet
{

namespace model
{

namespace parts
{

namespace dgtoolbox
{

/**
 * @brief computes the Legendre polynomial L_N and q = L_N+1 - L_N-2 and q' at point x
 * @param [in] polyDeg polynomial degree
 * @param [in] x evaluation point
 * @param [in] L <- L(x)
 * @param [in] q <- q(x) = L_N+1 (x) - L_N-2(x)
 * @param [in] qder <- q'(x) = [L_N+1 (x) - L_N-2(x)]'
 */
void qAndL(const unsigned int polyDeg, const double x, double& L, double& q, double& qder) {
	// auxiliary variables (Legendre polynomials)
	double L_2 = 1.0;
	double L_1 = x;
	double Lder_2 = 0.0;
	double Lder_1 = 1.0;
	double Lder = 0.0;
	for (double k = 2; k <= polyDeg; k++) { // note that this function is only called for polyDeg >= 2.
		L = ((2 * k - 1) * x * L_1 - (k - 1) * L_2) / k;
		Lder = Lder_2 + (2 * k - 1) * L_1;
		L_2 = L_1;
		L_1 = L;
		Lder_2 = Lder_1;
		Lder_1 = Lder;
	}
	q = ((2.0 * polyDeg + 1) * x * L - polyDeg * L_2) / (polyDeg + 1.0) - L_2;
	qder = Lder_1 + (2.0 * polyDeg + 1) * L_1 - Lder_2;
}
/**
 * @brief computes the Legendre-Gauss-Lobatto nodes and (inverse) quadrature weights
 * @param [in] polyDeg polynomial degree
 * @param [in, out] nodes Legendre Gauss Lobatto nodes
 * @param [in, out] invWeights Legendre Gauss quadrature weights
 * @param [in] invertWeights specifies if weights should be inverted
 */
void lglNodesWeights(const unsigned int polyDeg, VectorXd& nodes, VectorXd& invWeights, bool invertWeights) {
	// tolerance and max #iterations for Newton iteration
	int nIterations = 10;
	double tolerance = 1e-15;
	// Legendre polynomial and derivative
	double L = 0;
	double q = 0;
	double qder = 0;
	switch (polyDeg) {
	case 0:
		throw std::invalid_argument("Polynomial degree must be at least 1 !");
		break;
	case 1:
		nodes[0] = -1;
		invWeights[0] = 1;
		nodes[1] = 1;
		invWeights[1] = 1;
		break;
	default:
		nodes[0] = -1;
		nodes[polyDeg] = 1;
		invWeights[0] = 2.0 / (polyDeg * (polyDeg + 1.0));
		invWeights[polyDeg] = invWeights[0];
		// use symmetrie, only compute half of points and weights
		for (unsigned int j = 1; j <= floor((polyDeg + 1) / 2) - 1; j++) {
			//  first guess for Newton iteration
			nodes[j] = -cos(M_PI * (j + 0.25) / polyDeg - 3 / (8.0 * polyDeg * M_PI * (j + 0.25)));
			// Newton iteration to find roots of Legendre Polynomial
			for (unsigned int k = 0; k <= nIterations; k++) {
				qAndL(polyDeg, nodes[j], L, q, qder);
				nodes[j] = nodes[j] - q / qder;
				if (abs(q / qder) <= tolerance * abs(nodes[j])) {
					break;
				}
			}
			// calculate weights
			qAndL(polyDeg, nodes[j], L, q, qder);
			invWeights[j] = 2.0 / (polyDeg * (polyDeg + 1.0) * pow(L, 2.0));
			nodes[polyDeg - j] = -nodes[j]; // copy to second half of points and weights
			invWeights[polyDeg - j] = invWeights[j];
		}
	}
	if (polyDeg % 2 == 0) { // for even polyDeg we have an odd number of points which include 0.0
		qAndL(polyDeg, 0.0, L, q, qder);
		nodes[polyDeg / 2] = 0;
		invWeights[polyDeg / 2] = 2.0 / (polyDeg * (polyDeg + 1.0) * pow(L, 2.0));
	}
	// inverse the weights
	invWeights = invWeights.cwiseInverse();
}
/**
 * @brief computes the Legendre polynomial and its derivative
 * @param [in] polyDeg polynomial degree
 * @param [in, out] leg Legendre polynomial
 * @param [in, out] legDer Legendre polynomial derivative
 * @param [in] x evaluation point
 */
void legendrePolynomialAndDerivative(const int polyDeg, double& leg, double& legDer, const double x) {

	switch (polyDeg) {
	case 0:
		leg = 1.0;
		legDer = 0.0;
		break;
	case 1:
		leg = x;
		legDer = 1.0;
		break;
	default:
		double leg_2 = 1.0;
		double leg_1 = x;
		double legDer_2 = 0.0;
		double legDer_1 = 1.0;

		for (int k = 2; k <= polyDeg; k++) {
			leg = (2.0 * k - 1.0) / k * x * leg_1 - (k - 1.0) / k * leg_2;
			legDer = legDer_2 + (2.0 * k - 1.0) * leg_1;
			leg_2 = leg_1;
			leg_1 = leg;
			legDer_2 = legDer_1;
			legDer_1 = legDer;
		}
	}
}
/**
 * @brief computes the Legendre-Gauss nodes and quadrature weights
 * @param [in] polyDeg polynomial degree
 * @param [in, out] nodes Legendre Gauss nodes
 * @param [in, out] weights Legendre Gauss quadrature weights
 * @param [in] invertWeights specifies if weights should be inverted
 */
void lgNodesWeights(const unsigned int polyDeg, VectorXd& nodes, VectorXd& weights, bool invertWeights = true) {
	// tolerance and max #iterations for Newton iteration
	int nIterations = 10;
	double tolerance = 1e-15;

	switch (polyDeg) {
	case 0:
		nodes[0] = 0.0;
		weights[0] = 2.0;
		break;
	case 1:
		nodes[0] = -std::sqrt(1.0 / 3.0);
		weights[0] = 1;
		nodes[1] = -nodes[0];
		weights[1] = weights[0];
		break;
	default:

		double leg = 0.0;
		double legDer = 0.0;
		double delta = 0.0;

		for (int j = 0; j <= std::floor((polyDeg + 1) / 2) - 1; j++)
		{
			nodes[j] = -std::cos((2.0 * j + 1.0) / (2.0 * polyDeg + 2.0) * M_PI);
			for (int k = 0; k <= nIterations; k++)
			{
				legendrePolynomialAndDerivative(polyDeg + 1, leg, legDer, nodes[j]);
				delta = -leg / legDer;
				nodes[j] = nodes[j] + delta;
				if (std::abs(delta) <= tolerance * std::abs(nodes[j]))
					break;
			}
			legendrePolynomialAndDerivative(polyDeg + 1, leg, legDer, nodes[j]);
			nodes[polyDeg - j] = -nodes[j];
			weights[j] = 2.0 / ((1.0 - std::pow(nodes[j], 2.0)) * std::pow(legDer, 2.0));
			weights[polyDeg - j] = weights[j];
		}

		if (polyDeg % 2 == 0)
		{
			legendrePolynomialAndDerivative(polyDeg + 1, leg, legDer, 0.0);
			nodes[polyDeg / 2] = 0.0;
			weights[polyDeg / 2] = 2.0 / std::pow(legDer, 2.0);
		}
	}
}
/**
 * @brief evaluates a Lagrange polynomial built on input nodes at a set of points
 * @detail can be used to establish quadrature rules
 * @param [in] j index of Lagrange basis function
 * @param [in] intNodes interpolation nodes the Lagrange basis is constructed with
 * @param [in] evalNodes nodes the Lagrange basis is evaluated at
 */
VectorXd evalLagrangeBasis(const int j, const VectorXd intNodes, const VectorXd evalNodes) {

	const int nIntNodes = intNodes.size();
	const int nEvalNodes = evalNodes.size();
	VectorXd evalEll = VectorXd::Zero(nEvalNodes);

	double nominator = 1.0;
	double denominator = 1.0;

	for (int i = 0; i < nIntNodes; i++)
		if (i != j)
			denominator *= (intNodes[j] - intNodes[i]);

	for (int k = 0; k < nEvalNodes; k++)
	{
		for (int i = 0; i < nIntNodes; i++)
		{
			if (i != j)
			{
				if (std::abs(evalNodes[k] - intNodes[i]) < std::numeric_limits<double>::epsilon())
				{
					nominator = denominator;
					break;
				}
				else
					nominator *= (evalNodes[k] - intNodes[i]);
			}
		}
		evalEll[k] = nominator / denominator;
		nominator = 1.0;
	}

	return evalEll;
}
/**
 * @brief calculates the Gauss quadrature mass matrix for LGL interpolation polynomial on LG points
 * @detail exact integration of polynomials up to order 2N - 1
 * @param [in] LGLnodes Legendre Gauss Lobatto nodes
 * @param [in] nLGNodes number of Gauss quadrature nodes
 */
MatrixXd gaussQuadratureMMatrix(const VectorXd LGLnodes, const int nLGNodes) {

	const int Ldegree = nLGNodes - 1; // Legendre polynomial degree
	const int nLGLnodes = LGLnodes.size();

	MatrixXd evalEll = MatrixXd::Zero(nLGNodes, nLGNodes);
	MatrixXd massMatrix = MatrixXd::Zero(nLGNodes, nLGNodes);

	VectorXd LGnodes = VectorXd::Zero(nLGNodes);
	VectorXd LGweigths = VectorXd::Zero(nLGNodes);
	lgNodesWeights(Ldegree, LGnodes, LGweigths, false);

	for (int i = 0; i < nLGLnodes; i++)
		evalEll.row(i) = evalLagrangeBasis(i, LGLnodes, LGnodes);

	VectorXd aux = VectorXd::Zero(nLGNodes);

	for (int i = 0; i < nLGLnodes; i++)
	{
		for (int j = 0; j < nLGLnodes; j++)
		{
			aux = evalEll.row(i).array() * evalEll.row(j).array();
			massMatrix(i, j) = (aux.array() * LGweigths.array()).sum();
		}
	}

	return massMatrix;
}
/**
 * @brief computation of barycentric weights for fast polynomial evaluation
 * @param [in] polyDeg polynomial degree
 * @param [in, out] baryWeights vector to store barycentric weights. Must already be initialized with ones!
 */
VectorXd barycentricWeights(const unsigned int polyDeg, const VectorXd nodes) {

	VectorXd baryWeights = VectorXd::Ones(polyDeg + 1u);

	for (unsigned int j = 1; j <= polyDeg; j++) {
		for (unsigned int k = 0; k <= j - 1; k++) {
			baryWeights[k] = baryWeights[k] * (nodes[k] - nodes[j]) * 1.0;
			baryWeights[j] = baryWeights[j] * (nodes[j] - nodes[k]) * 1.0;
		}
	}
	for (unsigned int j = 0; j <= polyDeg; j++) {
		baryWeights[j] = 1 / baryWeights[j];
	}
	
	return baryWeights;
}
/**
 * @brief computation of nodal polynomial derivative matrix
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
 */
MatrixXd derivativeMatrix(const unsigned int polyDeg, const VectorXd nodes) {

	MatrixXd polyDerM = MatrixXd::Zero(polyDeg + 1u, polyDeg + 1u);
	VectorXd baryWeights = barycentricWeights(polyDeg, nodes);

	for (unsigned int i = 0; i <= polyDeg; i++) {
		for (unsigned int j = 0; j <= polyDeg; j++) {
			if (i != j) {
				polyDerM(i, j) = baryWeights[j] / (baryWeights[i] * (nodes[i] - nodes[j]));
				polyDerM(i, i) += -polyDerM(i, j);
			}
		}
	}

	return polyDerM;
}
/**
 * @brief factor to normalize Jacobi polynomials
 */
double orthonFactor(const int polyDeg, double a, double b) {

	double n = static_cast<double> (polyDeg);
	return std::sqrt(((2.0 * n + a + b + 1.0) * std::tgamma(n + 1.0) * std::tgamma(n + a + b + 1.0))
		/ (std::pow(2.0, a + b + 1.0) * std::tgamma(n + a + 1.0) * std::tgamma(n + b + 1.0)));
}
/**
 * @brief calculates the Vandermonde matrix of the normalized Jacobi polynomials
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
 * @param [in] a Jacobi polynomial parameter
 * @param [in] b Jacobi polynomial parameter
 */
MatrixXd jacVandermondeMatrix(const unsigned int polyDeg, const VectorXd nodes, const double a, const double b) {

	const unsigned int nNodes = polyDeg + 1u;
	MatrixXd V(nNodes, nNodes);

	// degree 0
	V.block(0, 0, nNodes, 1) = VectorXd::Ones(nNodes) * orthonFactor(0, a, b);
	// degree 1
	for (int node = 0; node < static_cast<int>(nNodes); node++) {
		V(node, 1) = ((nodes[node] - 1.0) / 2.0 * (a + b + 2.0) + (a + 1.0)) * orthonFactor(1, a, b);
	}

	for (int deg = 2; deg <= static_cast<int>(nNodes - 1); deg++) {

		for (int node = 0; node < static_cast<int>(nNodes); node++) {

			double orthn_1 = orthonFactor(deg, a, b) / orthonFactor(deg - 1, a, b);
			double orthn_2 = orthonFactor(deg, a, b) / orthonFactor(deg - 2, a, b);

			// recurrence relation
			V(node, deg) = orthn_1 * ((2.0 * deg + a + b - 1.0) * ((2.0 * deg + a + b) * (2.0 * deg + a + b - 2.0) * nodes[node] + a * a - b * b) * V(node, deg - 1));
			V(node, deg) -= orthn_2 * (2.0 * (deg + a - 1.0) * (deg + b - 1.0) * (2.0 * deg + a + b) * V(node, deg - 2));
			V(node, deg) /= 2.0 * deg * (deg + a + b) * (2.0 * deg + a + b - 2.0);
		}
	}

	return V;
}
double jacPDerivativePreFactor(const unsigned int pIndex, const unsigned int derOrder, const double a, const double b) {

	double prefac = std::tgamma(a + b + static_cast<double>(pIndex) + 1.0 + static_cast<double>(derOrder)) / (std::pow(2.0, static_cast<double>(derOrder)) * std::tgamma(a + b + static_cast<double>(pIndex) + 1.0));

	return prefac * orthonFactor(pIndex - derOrder, a + static_cast<double>(derOrder), b + static_cast<double>(derOrder)) / orthonFactor(pIndex, a, b);
}
/**
 * @brief calculates the Vandermonde matrix of the normalized Legendre polynomials
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
 */
MatrixXd legVandermondeMatrix(const unsigned int polyDeg, const VectorXd nodes) {
	return jacVandermondeMatrix(polyDeg, nodes, 0.0, 0.0);
}
/**
 * @brief calculates the inverse mass matrix via transformation to orthonormal Jacobi (modal) basis
 * @detail the mass matrix used to compute integrals of the form \int_E \ell_i(\xi) \ell_j(\xi) (1 - \xi)^\alpha (1 + \xi)^\beta d\xi
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
 */
Eigen::MatrixXd invMMatrix(const unsigned int polyDeg, const Eigen::VectorXd nodes, const double alpha, const double beta) {
	return (jacVandermondeMatrix(polyDeg, nodes, alpha, beta) * (jacVandermondeMatrix(polyDeg, nodes, alpha, beta).transpose()));
}
/**
 * @brief calculates the mass matrix via transformation to orthonormal Jacobi (modal) basis
 * @detail mass matrix used to compute integrals of the form \int_E \ell_i(\xi) \ell_j(\xi) (1 - \xi)^\alpha (1 + \xi)^\beta d\xi
 * @param [in] polyDeg polynomial degree
 * @param [in] nodes polynomial interpolation nodes
 */
Eigen::MatrixXd mMatrix(const unsigned int polyDeg, const Eigen::VectorXd nodes, const double alpha, const double beta) {
	return invMMatrix(polyDeg, nodes, alpha, beta).inverse();
}
/**
 * @brief calculates the derivative of the Vandermonde matrix of the normalized Legendre polynomials
 * @param [in] polyDeg polynomial degree
 * @param [in] a Jacobi polynomial parameter
 * @param [in] b Jacobi polynomial parameter
 * @param [in] nodes polynomial interpolation nodes
 */
MatrixXd jacDerVandermondeMatrix(const unsigned int polyDeg, const double a, const double b, const VectorXd nodes) {

	MatrixXd derVan = MatrixXd::Zero(polyDeg + 1, polyDeg + 1);
	derVan.block(0, 1, polyDeg + 1, polyDeg) = jacVandermondeMatrix(polyDeg, nodes, a + 1.0, b + 1.0).block(0, 0, polyDeg + 1, polyDeg);

	for (int hm = 1; hm < polyDeg + 1; hm++)
		derVan.block(0, hm, polyDeg + 1, 1) *= jacPDerivativePreFactor(hm, 1, a, b);
		//derVan.block(0, hm, polyDeg + 1, 1) *= std::sqrt(hm * (hm + a + b + 1)); // todo

	return derVan;
}
/**
 * @brief calculates the second order nodal stiffness matrix (via transformation to normalized Legendre polynomials)
 * @param [in] polyDeg polynomial degree
 * @param [in] a Jacobi polynomial parameter
 * @param [in] b Jacobi polynomial parameter
 * @param [in] nodes polynomial interpolation nodes
 */
MatrixXd secondOrderStiffnessMatrix(const unsigned int polyDeg, const double alpha, const double beta, const VectorXd nodes) {
	return derivativeMatrix(polyDeg, nodes).transpose() * mMatrix(polyDeg, nodes, alpha, beta) * derivativeMatrix(polyDeg, nodes);
}

} // namespace dgtoolbox
} // namespace parts
} // namespace model
} // namespace cadet