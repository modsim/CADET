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
 * Provides a piecewise cubic polynomial external function.
 */

#include "cadet/ExternalFunction.hpp"
#include "cadet/ParameterProvider.hpp"
#include "common/CompilerSpecific.hpp"

#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>

namespace cadet
{

namespace model
{

/**
 * @brief A piecewise cubic polynomial external function
 * @details This external function takes into account time and axial position in the
 *          column, but ignores radial position in the bead. A quantity of interest
 *          is measured at the column outlet and represented by a piecewise cubic
 *          polynomial (which is more general than a cubic spline due to missing
 *          smoothness and continuity conditions). It is assumed that this quantity
 *          is transported inside the column with a known velocity. Since the restart
 *          of the time integrator cannot be detected as long as the section times
 *          do not align with the time integrator's continuity sections, the profile
 *          should at least be continuous.
 *
 *          Note that the @c SECTION_TIMES parameter is independent of the global
 *          section times. The coefficients of the polynomials in the respective
 *          sections are just appended to one array for each coefficient. Thus,
 *          all constant coefficients are stored in @c CONST_COEFF (from the first
 *          section to the last one).
 */
class PiecewiseCubicPolyExternalFunction : public cadet::IExternalFunction
{
public:
	PiecewiseCubicPolyExternalFunction()
	{
	}

	virtual ~PiecewiseCubicPolyExternalFunction() CADET_NOEXCEPT
	{
	}

	static const char* identifier()
	{
		return "PIECEWISE_CUBIC_POLY";
	}
	virtual const char* name() const CADET_NOEXCEPT
	{
		return PiecewiseCubicPolyExternalFunction::identifier();
	}

	virtual bool configure(IParameterProvider* paramProvider)
	{
		if (!paramProvider)
			return false;

		_sectionTimes = paramProvider->getDoubleArray("SECTION_TIMES");
		_const = paramProvider->getDoubleArray("CONST_COEFF");
		_lin = paramProvider->getDoubleArray("LIN_COEFF");
		_quad = paramProvider->getDoubleArray("QUAD_COEFF");
		_cub = paramProvider->getDoubleArray("CUBE_COEFF");
		_velocity = paramProvider->getDouble("VELOCITY");

		// Check sizes
		return (_sectionTimes.size() >= 2) && (_const.size() == _sectionTimes.size() - 1) &&
			   (_const.size() == _lin.size()) && (_const.size() == _quad.size()) && (_const.size() == _cub.size());
	}

	virtual double externalProfile(double t, double z, double rho, double r, unsigned int sec)
	{
		// Compute transformed time and find index of the section
		const double transT = (1.0 - z) / _velocity + t;

		// If we don't have data, perform constant extrapolation
		if (transT <= _sectionTimes[0])
			return _const[0];
		if (transT >= _sectionTimes.back())
		{
			const int n = _sectionTimes.size();
			const double tShift = _sectionTimes[n - 1] - _sectionTimes[n - 2];
			return _const.back() + tShift * (_lin.back() + tShift * (_quad.back() + tShift * _cub.back()));
		}

		// Find the the interval [_sectionTimes[idx], _sectionTimes[idx+1]] in which transT is located
		const std::vector<double>::iterator it = std::lower_bound(_sectionTimes.begin(), _sectionTimes.end(), transT);
		const std::size_t idx = (it - _sectionTimes.begin()) - (*it > transT ? 1 : 0);

		// This function evaluates a piecewise cubic polynomial given on some intervals
		// called sections. On each section a polynomial of degree 3 is evaluated:
		//
		//   p_i(t) = CUBE_COEFF[i] * (t - t_i)^3 + QUAD_COEFF[i] * (t - t_i)^2 + LIN_COEFF[i] * (t - t_i) +
		//   CONST_COEFF[i],
		//
		// where p_i is the polynomial on section i given by the interval [t_i, t_{i+1}].

		const double tShift = transT - _sectionTimes[idx];

		// Evaluate polynomial using Horner's scheme
		return _const[idx] + tShift * (_lin[idx] + tShift * (_quad[idx] + tShift * _cub[idx]));
	}

	virtual double timeDerivative(double t, double z, double rho, double r, unsigned int sec)
	{
		// Compute transformed time and find index of the section
		const double transT = (1.0 - z) / _velocity + t;

		// If we don't have data, perform constant extrapolation
		if (transT <= _sectionTimes[0])
			return 0.0;
		if (transT >= _sectionTimes.back())
			return 0.0;

		// Find the the interval [_sectionTimes[idx], _sectionTimes[idx+1]] in which transT is located
		const std::vector<double>::iterator it = std::lower_bound(_sectionTimes.begin(), _sectionTimes.end(), transT);
		const std::size_t idx = (it - _sectionTimes.begin()) - (*it > transT ? 1 : 0);

		// This function evaluates a piecewise cubic polynomial given on some intervals
		// called sections. On each section a polynomial of degree 3 is evaluated:
		//
		//   p_i(t) = CUBE_COEFF[i] * (t - t_i)^3 + QUAD_COEFF[i] * (t - t_i)^2 + LIN_COEFF[i] * (t - t_i) +
		//   CONST_COEFF[i], p_i'(t) = 3 * CUBE_COEFF[i] * (t - t_i)^2 + 2 * QUAD_COEFF[i] * (t - t_i) + LIN_COEFF[i],
		//
		// where p_i is the polynomial on section i given by the interval [t_i, t_{i+1}].

		const double tShift = transT - _sectionTimes[idx];

		// Evaluate polynomial using Horner's scheme
		return _lin[idx] + tShift * (2.0 * _quad[idx] + tShift * 3.0 * _cub[idx]);
	}

	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity,
								 unsigned int nSections) CADET_NOEXCEPT
	{
	}

private:
	double _velocity; //!< Velocity of the movement of the external profile in [1/s] (normalized by column length)
	std::vector<double> _sectionTimes; //!< Section times

	std::vector<double> _const; //!< Constant coefficient of each polynomial piece
	std::vector<double> _lin;   //!< Linear coefficient of each polynomial piece
	std::vector<double> _quad;  //!< Quadratic coefficient of each polynomial piece
	std::vector<double> _cub;   //!< Cubic coefficient of each polynomial piece
};

namespace extfun
{
void registerPiecewiseCubicPoly(std::unordered_map<std::string, std::function<IExternalFunction*()>>& extFuns)
{
	extFuns[PiecewiseCubicPolyExternalFunction::identifier()] = []() {
		return new PiecewiseCubicPolyExternalFunction();
	};
}
} // namespace extfun

} // namespace model
} // namespace cadet
