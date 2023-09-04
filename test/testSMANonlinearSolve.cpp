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

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>

#include "linalg/DenseMatrix.hpp"
#include "nonlin/AdaptiveTrustRegionNewton.hpp"
#include "nonlin/LevenbergMarquardt.hpp"
#include <tclap/CmdLine.h>
#include "common/TclapUtils.hpp"

// #define USE_QR_FACTORIZATION

void printVector(const char* prefix, double const* const vec, unsigned int size)
{
	std::cout << prefix << " [" << vec[0];
	for (unsigned int i = 1; i < size; ++i)
		std::cout << "; " << vec[i];
	std::cout << "];" << std::endl;
}

void printDiffVector(const char* prefix, double const* const vecA, double const* const vecB, unsigned int size)
{
	std::cout << prefix << " [" << vecA[0] - vecB[0];
	for (unsigned int i = 1; i < size; ++i)
		std::cout << "; " << vecA[i] - vecB[i];
	std::cout << "];" << std::endl;
}

struct StdOutNewtonIterateOutputPolicy
{
	inline static void outerIteration(unsigned int idxIter, double residualNorm, double const* const res,
									  double const* const x, double const* const dx, unsigned int size)
	{
		std::cout << idxIter << " Res " << residualNorm << " dx " << cadet::linalg::l2Norm(dx, size) << std::endl;
	}

	inline static void innerIteration(unsigned int idxIter, double residualNorm, double const* const res,
									  double const* const x, double const* const dx, double damping, double mu,
									  unsigned int size)
	{
		std::cout << " -> lambda " << damping << " mu " << mu << " Res " << residualNorm << " dx "
				  << cadet::linalg::l2Norm(dx, size) << std::endl;
	}
};

struct StdOutLevMarIterateOutputPolicy
{
	inline static void outerIteration(unsigned int idxIter, double residualNorm, double const* const res,
									  double const* const x, double const* const dx, double damping, unsigned int size)
	{
		std::cout << idxIter << " Res " << residualNorm << " lambda " << damping;
		if (dx)
			std::cout << " dx " << cadet::linalg::l2Norm(dx, size);
		std::cout << std::endl;
	}
	inline static void innerIteration(unsigned int idxIter, double residualNorm, double const* const res,
									  double const* const x, double const* const dx, double damping, unsigned int size)
	{
		std::cout << " -> lambda " << damping << " Res " << residualNorm << " dx " << cadet::linalg::l2Norm(dx, size)
				  << std::endl;
	}
};

struct SMAProblem
{
	const double initPoint[4] = {1.0485785488181000e+03, 1.1604726694141368e+01, 1.1469542586742687e+01,
								 9.7852311988018670e+00};
	const double yCp[4] = {5.8377002519964755e+01, 2.9352296732047269e-03, 1.5061023667222263e-02,
						   1.3523701213590386e-01};
	const double _kA[4] = {0.0, 35.5, 1.59, 7.7};
	const double _kD[4] = {0.0, 1000.0, 1000.0, 1000.0};
	const double _lambda = 1.2e3;
	const double _nu[4] = {0.0, 4.7, 5.29, 3.7};
	const double _sigma[4] = {0.0, 11.83, 10.6, 10.0};

	cadet::linalg::DenseMatrix _jacMatrix;
#ifdef USE_QR_FACTORIZATION
	std::vector<double> _workspace;
#endif

	inline const char* name() const
	{
		return "SMAProblem";
	}
	inline int size() const
	{
		return 4;
	}
	void init()
	{
		_jacMatrix.resize(size(), size());
#ifdef USE_QR_FACTORIZATION
		_workspace = std::vector<double>(2 * size(), 0.0);
#endif
	}

	bool residual(double const* const x, double* const res)
	{
		// Salt equation: q_0 - Lambda + Sum[nu_j * q_j, j] == 0
		//           <=>  q_0 == Lambda - Sum[nu_j * q_j, j]
		// Also compute \bar{q}_0 = q_0 - Sum[sigma_j * q_j, j]
		res[0] = x[0] - _lambda;
		double q0_bar = x[0];

		for (int j = 1; j < size(); ++j)
		{
			res[0] += _nu[j] * x[j];
			q0_bar -= _sigma[j] * x[j];
		}

		// Protein equations: dq_i / dt - ( k_{a,i} * c_{p,i} * \bar{q}_0^{nu_i} - k_{d,i} * q_i * c_{p,0}^{nu_i} ) == 0
		//               <=>  dq_i / dt == k_{a,i} * c_{p,i} * \bar{q}_0^{nu_i} - k_{d,i} * q_i * c_{p,0}^{nu_i}
		for (int i = 1; i < size(); ++i)
		{
			const double c0_pow_nu = pow(yCp[0], _nu[i]);
			const double q0_bar_pow_nu = pow(q0_bar, _nu[i]);

			// Residual
			res[i] = _kD[i] * x[i] * c0_pow_nu - _kA[i] * yCp[i] * q0_bar_pow_nu;
		}
		return true;
	}

	bool jacobian(double const* const x, cadet::linalg::detail::DenseMatrixBase& mat)
	{
		cadet::linalg::DenseBandedRowIterator jac = mat.row(0);
		double q0_bar = x[0];

		// Salt equation: q_0 - Lambda + Sum[nu_j * q_j, j] == 0
		jac[0] = 1.0;
		for (int j = 1; j < size(); ++j)
		{
			jac[j] = _nu[j];

			// Calculate \bar{q}_0 = q_0 - Sum[sigma_j * q_j, j]
			q0_bar -= _sigma[j] * x[j];
		}

		// Advance to protein equations
		++jac;

		// Protein equations: dq_i / dt - ( k_{a,i} * c_{p,i} * \bar{q}_0^{nu_i} - k_{d,i} * q_i * c_{p,0}^{nu_i} ) == 0
		// We have already computed \bar{q}_0 in the loop above
		for (int i = 1; i < size(); ++i)
		{
			// Getting to c_{p,0}: -bndIdx takes us to q_0, another -nComp to c_{p,0}. This means jac[-bndIdx - nComp]
			// corresponds to c_{p,0}. Getting to c_{p,i}: -bndIdx takes us to q_0, another -nComp to c_{p,0} and a +i
			// to c_{p,i}.
			//                     This means jac[i - bndIdx - nComp] corresponds to c_{p,i}.

			const double ka = _kA[i];
			const double kd = _kD[i];
			const double nu = _nu[i];

			const double c0_pow_nu = pow(yCp[0], nu);
			const double q0_bar_pow_nu_m1 = pow(q0_bar, nu - 1.0);

			// dres_i / dq_0
			jac[-i] = -ka * yCp[i] * nu * q0_bar_pow_nu_m1;

			// Fill dres_i / dq_j
			for (int j = 1; j < size(); ++j)
			{
				// dres_i / dq_j
				jac[j - i] = -ka * yCp[i] * nu * q0_bar_pow_nu_m1 * (-_sigma[j]);
				// Getting to q_j: -bndIdx takes us to q_0, another +bndIdx2 to q_j. This means jac[bndIdx2 - bndIdx]
				// corresponds to q_j.
			}

			// Add to dres_i / dq_i
			jac[0] += kd * c0_pow_nu;

			// Advance to next equation and Jacobian row
			++jac;
		}
		return true;
	}

	bool jacobianSolve(double const* const x, double* const sol)
	{
		if (!jacobian(x, _jacMatrix))
			return false;

#ifndef USE_QR_FACTORIZATION
		// LU factorization
		if (!_jacMatrix.factorize())
			return false;

		if (!_jacMatrix.solve(sol))
			return false;
#else
		// QR factorization
		if (!_jacMatrix.robustFactorize(_workspace.data()))
			return false;

		if (!_jacMatrix.robustSolve(sol, _workspace.data()))
			return false;
#endif
		return true;
	}

	bool jacobianResolve(double* const sol)
	{
#ifndef USE_QR_FACTORIZATION
		return _jacMatrix.solve(sol);
#else
		return _jacMatrix.robustSolve(sol, _workspace.data());
#endif
	}
};

template <class Prob> void runNewtonRes(double stdDev)
{
	Prob p;
	std::cout << "======= NEWTON RESIDUAL ======\n";
	std::cout << "===== " << p.name() << " =====\n";
	std::cout << "Size: " << p.size() << std::endl;

	p.init();

	const unsigned int maxIter = 100;
	const double resTol = 1e-12;

	// Mild nonlinear: 1.0, 1e-4; Highly nonlinear: 1e-2, 1e-4; Extremely nonlinear: 1e-4, 1e-8, restricted
	const double initDamping = 1e-4;
	const double minDamping = 1e-8;

	std::vector<double> sol(p.initPoint, p.initPoint + p.size());
	std::vector<double> tempMem(4 * p.size(), 0.0);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1);

	// Randomize
	for (std::size_t i = 0; i < sol.size(); ++i)
		sol[i] += stdDev * sol[i] * distribution(generator);

	printVector("Origi", p.initPoint, p.size());
	printVector("Start", sol.data(), p.size());
	printDiffVector("Diffe", sol.data(), p.initPoint, p.size());

	const bool success = cadet::nonlin::adaptiveTrustRegionNewtonMethod<StdOutNewtonIterateOutputPolicy>(
		[&](double const* const x, double* const res) { return p.residual(x, res); },
		[&](double const* const x, double* const res) { return p.jacobianSolve(x, res); }, maxIter, resTol, initDamping,
		minDamping, sol.data(), tempMem.data(), p.size());

	p.residual(sol.data(), tempMem.data());

	std::cout << "Method " << (success ? "SUCCESS" : "FAIL") << std::endl;
	std::cout << "Residual norm: " << cadet::linalg::l2Norm(tempMem.data(), p.size()) << std::endl;
	printVector("Residual", tempMem.data(), p.size());
	printVector("Solution", sol.data(), p.size());
	if (success)
		printDiffVector("Differen", sol.data(), p.initPoint, p.size());
}

template <class Prob> void runNewtonErr(double stdDev)
{
	Prob p;
	std::cout << "======== NEWTON ERROR ========\n";
	std::cout << "===== " << p.name() << " =====\n";
	std::cout << "Size: " << p.size() << std::endl;

	p.init();

	const unsigned int maxIter = 100;
	const double resTol = 1e-12;

	// Mild nonlinear: 1.0, 1e-4; Highly nonlinear: 1e-2, 1e-4; Extremely nonlinear: 1e-4, 1e-8, restricted
	const double initDamping = 1e-4;
	const double minDamping = 1e-8;

	std::vector<double> sol(p.initPoint, p.initPoint + p.size());
	std::vector<double> tempMem(4 * p.size(), 0.0);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1);

	// Randomize
	for (std::size_t i = 0; i < sol.size(); ++i)
		sol[i] += stdDev * sol[i] * distribution(generator);

	printVector("Origi", p.initPoint, p.size());
	printVector("Start", sol.data(), p.size());
	printDiffVector("Diffe", sol.data(), p.initPoint, p.size());

	const bool success = cadet::nonlin::robustAdaptiveTrustRegionNewtonMethod<StdOutNewtonIterateOutputPolicy>(
		[&](double const* const x, double* const res) { return p.residual(x, res); },
		[&](double const* const x, double* const res) { return p.jacobianSolve(x, res); },
		[&](double* const res) { return p.jacobianResolve(res); }, maxIter, resTol, initDamping, minDamping, sol.data(),
		tempMem.data(), p.size());

	p.residual(sol.data(), tempMem.data());

	std::cout << "Method " << (success ? "SUCCESS" : "FAIL") << std::endl;
	std::cout << "Residual norm: " << cadet::linalg::l2Norm(tempMem.data(), p.size()) << std::endl;
	printVector("Residual", tempMem.data(), p.size());
	printVector("Solution", sol.data(), p.size());
	if (success)
		printDiffVector("Differen", sol.data(), p.initPoint, p.size());
}

template <class Prob> void runLevMar(double stdDev)
{
	Prob p;
	std::cout << "===== LEVENBERG-MARQUADRT ====\n";
	std::cout << "===== " << p.name() << " =====\n";
	std::cout << "Size: " << p.size() << std::endl;

	p.init();

	const unsigned int maxIter = 500;
	const double resTol = 1e-12;

	const double initDamping = 1e-2;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1);

	std::vector<double> sol(p.initPoint, p.initPoint + p.size());
	std::vector<double> tempMem(7 * p.size(), 0.0);

	// Randomize
	for (std::size_t i = 0; i < sol.size(); ++i)
		sol[i] += stdDev * sol[i] * distribution(generator);

	printVector("Origi", p.initPoint, p.size());
	printVector("Start", sol.data(), p.size());
	printDiffVector("Diffe", sol.data(), p.initPoint, p.size());
	const bool success = cadet::nonlin::levenbergMarquardt<StdOutLevMarIterateOutputPolicy>(
		[&](double const* const x, double* const res) { return p.residual(x, res); },
		[&](double const* const x, cadet::linalg::detail::DenseMatrixBase& mat) { return p.jacobian(x, mat); }, maxIter,
		resTol, initDamping, sol.data(), tempMem.data(), p._jacMatrix, p.size());

	p.residual(sol.data(), tempMem.data());

	std::cout << "Method " << (success ? "SUCCESS" : "FAIL") << std::endl;
	std::cout << "Residual norm: " << cadet::linalg::l2Norm(tempMem.data(), p.size()) << std::endl;
	printVector("Residual", tempMem.data(), p.size());
	printVector("Solution", sol.data(), p.size());
	if (success)
		printDiffVector("Differen", sol.data(), p.initPoint, p.size());
}

int main(int argc, char** argv)
{
	double stdDev;
	bool useTRNRes;
	bool useTRNErr;
	bool useLevMar;
	try
	{
		TCLAP::CustomOutputWithoutVersion customOut("testSMANonlinearSolve");
		TCLAP::CmdLine cmd("Tests nonlinear solvers with a small SMA example", ' ', "1.0");
		cmd.setOutput(&customOut);

		cmd >> (new TCLAP::SwitchArg("r", "newtonRes", "Residual based adaptive trust region Newton"))
				   ->storeIn(&useTRNRes);
		cmd >>
			(new TCLAP::SwitchArg("e", "newtonErr", "Error based adaptive trust region Newton"))->storeIn(&useTRNErr);
		cmd >> (new TCLAP::SwitchArg("l", "levMar", "Levenberg-Marquardt"))->storeIn(&useLevMar);
		cmd >> (new TCLAP::ValueArg<double>("s", "stddev",
											"Standard deviation of relative random perturbation (default: 0.01)", false,
											0.01, "StdDev"))
				   ->storeIn(&stdDev);

		cmd.parse(argc, argv);
	}
	catch (const TCLAP::ArgException& e)
	{
		std::cerr << "ERROR: " << e.error() << " for argument " << e.argId() << std::endl;
		return 1;
	}

	std::cout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1);

	if (!useTRNErr && !useTRNRes && !useLevMar)
	{
		useTRNErr = true;
		useTRNRes = true;
		useLevMar = true;
	}

	if (useTRNErr)
		runNewtonErr<SMAProblem>(stdDev);
	if (useTRNRes)
		runNewtonRes<SMAProblem>(stdDev);
	if (useLevMar)
		runLevMar<SMAProblem>(stdDev);

	return 0;
}
