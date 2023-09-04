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
 * Provides an external function that calls a Matlab function handle.
 */

#ifndef MATLAB_MEX_FILE
#define MATLAB_MEX_FILE
#endif

#include <mex.h>

// Everything that includes mex.h should go here

// Take care of namespace pollution / macros
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <functional>

#include "cadet/ExternalFunction.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/ModelBuilder.hpp"

namespace cadet
{

namespace mex
{

/**
 * @brief An external function that calls a Matlab function handle
 */
class MatlabExternalFunction : public cadet::IExternalFunction
{
public:
	MatlabExternalFunction()
	{
	}

	virtual ~MatlabExternalFunction() CADET_NOEXCEPT
	{
	}

	static const char* identifier()
	{
		return "MATLAB";
	}
	virtual const char* name() const CADET_NOEXCEPT
	{
		return MatlabExternalFunction::identifier();
	}

	virtual bool configure(IParameterProvider* paramProvider)
	{
		if (!paramProvider)
			return false;

		// Check sizes
		return true;
	}

	virtual double externalProfile(double t, double z, double rho, double r, unsigned int sec)
	{

		// Again, we're out of data, so perform constant extrapolation
		return 0.0;
	}

	virtual double timeDerivative(double t, double z, double rho, double r, unsigned int sec)
	{
		// Again, we're out of data, so perform constant extrapolation
		return 0.0;
	}

	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity,
								 unsigned int nSections) CADET_NOEXCEPT
	{
	}

private:
};

void registerMatlabExtFun(IModelBuilder& builder)
{
	builder.registerExternalFunctionType(MatlabExternalFunction::identifier(),
										 []() { return new MatlabExternalFunction(); });
}

} // namespace mex
} // namespace cadet
