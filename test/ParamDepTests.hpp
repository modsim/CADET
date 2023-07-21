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
 * Defines tests for parameter dependencies.
 */

#ifndef CADETTEST_PARAMDEPTEST_HPP_
#define CADETTEST_PARAMDEPTEST_HPP_

#include "cadet/cadetCompilerInfo.hpp"

#include <limits>

namespace cadet
{

namespace model
{
	class IParameterStateDependence;
}

namespace test
{

namespace paramdep
{

	class ConfiguredParameterDependence
	{
	public:

		ConfiguredParameterDependence(ConfiguredParameterDependence&& cpy) CADET_NOEXCEPT 
			: _paramDep(cpy._paramDep), _nComp(cpy._nComp), _nBound(cpy._nBound), _boundOffset(cpy._boundOffset)
		{
			cpy._paramDep = nullptr;
			cpy._nBound = nullptr;
			cpy._boundOffset = nullptr;
		}

		~ConfiguredParameterDependence();

		inline ConfiguredParameterDependence& operator=(ConfiguredParameterDependence&& cpy) CADET_NOEXCEPT
		{
			_paramDep = cpy._paramDep;
			_nComp = cpy._nComp;
			_nBound = cpy._nBound;
			_boundOffset = cpy._boundOffset;

			cpy._paramDep = nullptr;
			cpy._nBound = nullptr;
			cpy._boundOffset = nullptr;

			return *this;
		}

		static ConfiguredParameterDependence create(const char* name, unsigned int nComp, unsigned int const* nBound, const char* config);

		inline cadet::model::IParameterStateDependence& model() { return *_paramDep; }
		inline const cadet::model::IParameterStateDependence& model() const { return *_paramDep; }

		inline unsigned int nComp() const { return _nComp; }
		inline unsigned int const* nBound() const { return _nBound; }
		inline unsigned int const* boundOffset() const { return _boundOffset; }

		inline unsigned int numBoundStates() const { return _boundOffset[_nComp - 1] + _nBound[_nComp - 1]; }

	private:

		ConfiguredParameterDependence(cadet::model::IParameterStateDependence* paramDep, unsigned int nComp, unsigned int const* nBound, unsigned int const* boundOffset) 
			: _paramDep(paramDep), _nComp(nComp), _nBound(nBound), _boundOffset(boundOffset)
		{
		}

		cadet::model::IParameterStateDependence* _paramDep;
		unsigned int _nComp;
		unsigned int const* _nBound;
		unsigned int const* _boundOffset;
	};

	/**
	 * @brief Checks the analytic Jacobian of the binding model against AD
	 * @param [in] modelName Name of the binding model
	 * @param [in] nComp Number of components
	 * @param [in] nBound Array with number of bound states for each component
	 * @param [in] config JSON string with binding model parameters
	 * @param [in] point Liquid phase values to check Jacobian at
	 * @param [in] absTol Absolute error tolerance
	 * @param [in] relTol Relative error tolerance
	 */
	void testLiquidJacobianAD(const char* modelName, unsigned int nComp, unsigned int const* nBound, const char* config, double const* point, double absTol = 0.0, double relTol = std::numeric_limits<float>::epsilon() * 100.0);

	/**
	 * @brief Checks the analytic Jacobian of the binding model against AD
	 * @param [in] modelName Name of the binding model
	 * @param [in] nComp Number of components
	 * @param [in] nBound Array with number of bound states for each component
	 * @param [in] config JSON string with binding model parameters
	 * @param [in] point Liquid phase and solid phase values to check Jacobian at
	 * @param [in] absTol Absolute error tolerance
	 * @param [in] relTol Relative error tolerance
	 */
	void testCombinedJacobianAD(const char* modelName, unsigned int nComp, unsigned int const* nBound, const char* config, double const* point, double absTol = 0.0, double relTol = std::numeric_limits<float>::epsilon() * 100.0);

} // namespace binding
} // namespace test
} // namespace cadet

#endif  // CADETTEST_PARAMDEPTEST_HPP_
