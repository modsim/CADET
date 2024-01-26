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
#include <algorithm>

#include "linalg/DenseMatrix.hpp"
#include "nonlin/AdaptiveTrustRegionNewton.hpp"

void printVector(const char* prefix, double const* const vec, unsigned int size)
{
	std::cout << prefix << " [" << vec[0];
	for (unsigned int i = 1; i < size; ++i)
		std::cout << "; " << vec[i];
	std::cout << "];" << std::endl;
}

struct StdOutIterateOutputPolicy
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

struct Problem1
{
	const double initPoint[1] = {10.0};
	const double solution[1] = {2.0};

	double jac;

	inline const char* name() const
	{
		return "Problem1";
	}
	inline unsigned int size() const
	{
		return 1;
	}
	void init()
	{
	}

	bool residual(double const* const x, double* const res)
	{
		res[0] = x[0] * x[0] - 4.0;
		return true;
	}

	bool jacobianSolve(double const* const x, double* const sol)
	{
		jac = x[0];
		sol[0] = sol[0] / (2.0 * x[0]);
		return true;
	}

	bool jacobianResolve(double* const sol)
	{
		sol[0] = sol[0] / (2.0 * jac);
		return true;
	}
};

struct Problem2
{
	const double initPoint[2] = {-5.0, 4.0};
	const double solution[2] = {1.0, 0.0};

	cadet::linalg::DenseMatrix jac;

	inline const char* name() const
	{
		return "Problem2";
	}
	inline unsigned int size() const
	{
		return 2;
	}
	void init()
	{
		jac.resize(size(), size());
	}

	bool residual(double const* const x, double* const res)
	{
		res[0] = (x[0] - 1.0) * (x[0] - 1.0) + x[1] * x[1];
		res[1] = std::exp(x[1]) - 1.0;
		return true;
	}

	bool jacobianSolve(double const* const x, double* const sol)
	{
		// Build Jacobian at x
		jac.native(0, 0) = 2.0 * (x[0] - 1.0);
		jac.native(0, 1) = 2 * x[1];
		jac.native(1, 0) = 0.0;
		jac.native(1, 1) = std::exp(x[1]);

		if (!jac.factorize())
			return false;

		if (!jac.solve(sol))
			return false;

		return true;
	}

	bool jacobianResolve(double* const sol)
	{
		return jac.solve(sol);
	}
};

struct Problem3
{
	const double initPoint[2] = {1.0, 1.0};
	const double solution[2] = {5e1, 10.0};

	cadet::linalg::DenseMatrix jac;

	inline const char* name() const
	{
		return "Problem3";
	}
	inline unsigned int size() const
	{
		return 2;
	}
	void init()
	{
		jac.resize(size(), size());
	}

	bool residual(double const* const x, double* const res)
	{
		res[0] = x[1] - 10.0;
		res[1] = x[0] * x[1] - 5e2;
		return true;
	}

	bool jacobianSolve(double const* const x, double* const sol)
	{
		// Build Jacobian at x
		jac.native(0, 0) = 0.0;
		jac.native(0, 1) = 1.0;
		jac.native(1, 0) = x[1];
		jac.native(1, 1) = x[0];

		if (!jac.factorize())
			return false;

		if (!jac.solve(sol))
			return false;

		return true;
	}

	bool jacobianResolve(double* const sol)
	{
		return jac.solve(sol);
	}
};

template <class Prob> void run()
{
	Prob p;
	std::cout << "===== " << p.name() << " =====\n";
	std::cout << "Size: " << p.size() << std::endl;

	p.init();

	const unsigned int maxIter = 100;
	const double resTol = 1e-10;

	// Mild nonlinear: 1.0, 1e-4; Highly nonlinear: 1e-2, 1e-4; Extremely nonlinear: 1e-4, 1e-8, restricted
	const double initDamping = 0.01;
	const double minDamping = 1e-4;

	std::vector<double> sol(p.initPoint, p.initPoint + p.size());
	std::vector<double> tempMem(4 * p.size(), 0.0);

	/*
		const bool success = cadet::nonlin::adaptiveTrustRegionNewtonMethod<StdOutIterateOutputPolicy>([&](double const*
	   const x, double* const res) { return p.residual(x, res); },
			[&](double const* const x, double* const res) { return p.jacobianSolve(x, res); }, maxIter, resTol,
			initDamping, minDamping, sol.data(), tempMem.data(), p.size());
	*/

	const bool success = cadet::nonlin::robustAdaptiveTrustRegionNewtonMethod<StdOutIterateOutputPolicy>(
		[&](double const* const x, double* const res) { return p.residual(x, res); },
		[&](double const* const x, double* const res) { return p.jacobianSolve(x, res); },
		[&](double* const res) { return p.jacobianResolve(res); }, maxIter, resTol, initDamping, minDamping, sol.data(),
		tempMem.data(), p.size());

	std::cout << "Method " << (success ? "SUCCESS" : "FAIL") << std::endl;
	printVector("Residual", tempMem.data(), p.size());
	printVector("Solution", sol.data(), p.size());
	printVector("Referenc", p.solution, p.size());

	double errorLinf = 0.0;
	for (std::size_t i = 0; i < p.size(); ++i)
	{
		errorLinf = std::max(std::abs(sol[i] - p.solution[i]), errorLinf);
	}
	std::cout << "Linf-Error: " << errorLinf << std::endl;
}

int main(int argc, char** argv)
{
	std::cout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1);
	run<Problem1>();
	run<Problem2>();
	run<Problem3>();

	return 0;
}
