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
 * Provides a linearly interpolated external function from moving data points without radial dependence.
 */

#include "cadet/ExternalFunction.hpp"
#include "cadet/ParameterProvider.hpp"
#include "common/CompilerSpecific.hpp"

#include <vector>
#include <functional>
#include <string>
#include <unordered_map>
#include <algorithm>

namespace cadet
{

namespace model
{

/**
 * @brief An external function that linearly interpolates a given (x,y) point set
 * @details This external function takes into account time and axial position in the
 *          column, but ignores radial position in the bead. A quantity of interest
 *          is measured at the column outlet and recorded in a (time, value) like list.
 *          It is assumed that this quantity is transported inside the column with a
 *          known velocity. Based on the current time and the axial position inside
 *          the column, the correct time interval (subject to the velocity) of the
 *          data points is chosen and the measurements used for linear interpolation.
 */
class LinearInterpolationExternalFunction : public IExternalFunction
{
public:
	LinearInterpolationExternalFunction()
	{
	}
	virtual ~LinearInterpolationExternalFunction()
	{
	}

	static const char* identifier()
	{
		return "LINEAR_INTERP_DATA";
	}
	virtual const char* name() const CADET_NOEXCEPT
	{
		return LinearInterpolationExternalFunction::identifier();
	}

	virtual bool configure(IParameterProvider* paramProvider)
	{
		if (!paramProvider)
			return false;

		// Read the external profile and corresponding deltas
		_dataY = paramProvider->getDoubleArray("DATA");
		_time = paramProvider->getDoubleArray("TIME");

		// Velocity is applied to the profile in flow direction
		_velocity = paramProvider->getDouble("VELOCITY");

		return true;
	}

	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections)
	{
	}

	virtual double externalProfile(double t, double z, double rho, double r, unsigned int sec)
	{
		// z denotes relative axial position in the column, i.e., z is in [0,1].
		//
		// The coordinate system of the external profile begins at the column outlet
		// and points backward to the column inlet.
		//
		//       1 / velocity                                                 0
		//    t <---|---------------------------------------------------------|
		//           _________________________________________________________
		//          |                                                         |
		//   Inlet  |  =>               Column                            =>  | Outlet
		//          |_________________________________________________________|
		//
		//
		//          The column moves along the profile to the left with _velocity.
		//          Thus, after some time, the coordinate system has shifted:
		//
		//       1 / _velocity + t        (1 - z) / _velocity + t             t
		//    t <---|------------------------------|--------------------------|-------------|
		//           _________________________________________________________
		//          |                                                         |
		//   Inlet  |  =>               Column                            =>  | Outlet
		//          |_________________________________________________________|
		//
		//          |---------------------------------------------------------|-----> z
		//          0              local column axial coordinate              1
		//
		// We now have to compute the time point when the column outlet will reach
		// the requested position subject to the velocity _velocity.

		// Compute transformed time
		const double transT = (1.0 - z) / _velocity + t;

		// Use constant extrapolation on both sides of the external profile
		if (transT <= _time[0])
			return _dataY.front();
		else if (transT >= _time.back())
			return _dataY.back();

		// In the middle use linear interpolation

		// Find the the interval [_time[idx], _time[idx+1]] in which transT is located
		const std::vector<double>::iterator it = std::lower_bound(_time.begin(), _time.end(), transT);
		const std::size_t idx = (it - _time.begin()) - (*it > transT ? 1 : 0);

		// Now idx is the index of the left and idx + 1 is the index of the right data point
		// Perform linear interpolation
		return _dataY[idx] + (_dataY[idx + 1] - _dataY[idx]) * (transT - _time[idx]) / (_time[idx + 1] - _time[idx]);
	}

	virtual double timeDerivative(double t, double z, double rho, double r, unsigned int sec)
	{
		// Compute transformed time
		const double transT = (1.0 - z) / _velocity + t;

		// Use constant extrapolation on both sides of the external profile => slope is 0.0
		if (transT <= _time[0])
			return 0.0;
		else if (transT >= _time.back())
			return 0.0;

		// In the middle use linear interpolation

		// Find the the interval [_time[idx], _time[idx+1]] in which transT is located
		const std::vector<double>::iterator it = std::lower_bound(_time.begin(), _time.end(), transT);
		const std::size_t idx = (it - _time.begin()) - (*it > transT ? 1 : 0);

		// Now idx is the index of the left and idx + 1 is the index of the right data point
		// Return slope of linear interpolation
		return (_dataY[idx + 1] - _dataY[idx]) / (_time[idx + 1] - _time[idx]);
	}

private:
	double _velocity; //!< Velocity of the movement of the external profile in [1/s] (normalized by column length)
	std::vector<double> _dataY; //!< External profile data points (function values)
	std::vector<double> _time;  //!< Time point of each measurement in [s]
};

namespace extfun
{
void registerLinearInterpolation(std::unordered_map<std::string, std::function<IExternalFunction*()>>& extFuns)
{
	extFuns[LinearInterpolationExternalFunction::identifier()] = []() {
		return new LinearInterpolationExternalFunction();
	};
}
} // namespace extfun

} // namespace model
} // namespace cadet
