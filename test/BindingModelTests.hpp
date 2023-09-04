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
 * Defines tests for binding models.
 */

#ifndef CADETTEST_BINDINGMODELTEST_HPP_
#define CADETTEST_BINDINGMODELTEST_HPP_

#include <limits>
#include "cadet/ExternalFunction.hpp"
#include "Memory.hpp"

namespace cadet
{

namespace model
{
class IBindingModel;
}

namespace test
{

namespace binding
{

class ConstExternalFunction : public cadet::IExternalFunction
{
public:
	virtual bool configure(cadet::IParameterProvider* paramProvider)
	{
		return true;
	}
	virtual const char* name() const CADET_NOEXCEPT
	{
		return "CONSTFUN";
	}
	virtual double externalProfile(double t, double z, double rho, double r, unsigned int sec)
	{
		return 1.0;
	}
	virtual double timeDerivative(double t, double z, double rho, double r, unsigned int sec)
	{
		return 0.0;
	}
	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections)
	{
	}
};

class LinearExternalFunction : public cadet::IExternalFunction
{
public:
	virtual bool configure(cadet::IParameterProvider* paramProvider)
	{
		return true;
	}
	virtual const char* name() const CADET_NOEXCEPT
	{
		return "LINFUN";
	}
	virtual double externalProfile(double t, double z, double rho, double r, unsigned int sec)
	{
		return t;
	}
	virtual double timeDerivative(double t, double z, double rho, double r, unsigned int sec)
	{
		return 1.0;
	}
	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections)
	{
	}
};

class ConfiguredBindingModel
{
public:
	ConfiguredBindingModel(ConfiguredBindingModel&& cpy) CADET_NOEXCEPT : _binding(cpy._binding),
																		  _nComp(cpy._nComp),
																		  _nBound(cpy._nBound),
																		  _boundOffset(cpy._boundOffset),
																		  _buffer(std::move(cpy._buffer)),
																		  _bufferMemory(cpy._bufferMemory),
																		  _extFuns(std::move(cpy._extFuns))
	{
		cpy._binding = nullptr;
		cpy._nBound = nullptr;
		cpy._boundOffset = nullptr;
		cpy._bufferMemory = nullptr;
	}

	~ConfiguredBindingModel();

	inline ConfiguredBindingModel& operator=(ConfiguredBindingModel&& cpy) CADET_NOEXCEPT
	{
		_binding = cpy._binding;
		_nComp = cpy._nComp;
		_nBound = cpy._nBound;
		_boundOffset = cpy._boundOffset;
		_buffer = std::move(cpy._buffer);
		_bufferMemory = cpy._bufferMemory;
		_extFuns = std::move(cpy._extFuns);

		cpy._binding = nullptr;
		cpy._nBound = nullptr;
		cpy._boundOffset = nullptr;
		cpy._bufferMemory = nullptr;

		return *this;
	}

	static ConfiguredBindingModel create(const char* name, unsigned int nComp, unsigned int const* nBound,
										 bool isKinetic, const char* config);
	static ConfiguredBindingModel create(const char* name, unsigned int nComp, unsigned int const* nBound,
										 int const* isKinetic, const char* config);

	void increaseBufferSize(int inc);
	int requiredBufferSize() CADET_NOEXCEPT;

	inline cadet::model::IBindingModel& model()
	{
		return *_binding;
	}
	inline const cadet::model::IBindingModel& model() const
	{
		return *_binding;
	}

	inline cadet::LinearBufferAllocator buffer()
	{
		return _buffer;
	}
	inline unsigned int nComp() const
	{
		return _nComp;
	}
	inline unsigned int const* nBound() const
	{
		return _nBound;
	}
	inline unsigned int const* boundOffset() const
	{
		return _boundOffset;
	}

	inline unsigned int numBoundStates() const
	{
		return _boundOffset[_nComp - 1] + _nBound[_nComp - 1];
	}

private:
	ConfiguredBindingModel(cadet::model::IBindingModel* binding, unsigned int nComp, unsigned int const* nBound,
						   unsigned int const* boundOffset, void* bufferStart, void* bufferEnd,
						   std::vector<cadet::IExternalFunction*>&& extFuns)
		: _binding(binding), _nComp(nComp), _nBound(nBound), _boundOffset(boundOffset), _buffer(bufferStart, bufferEnd),
		  _bufferMemory(bufferStart), _extFuns(std::move(extFuns))
	{
	}

	cadet::model::IBindingModel* _binding;
	unsigned int _nComp;
	unsigned int const* _nBound;
	unsigned int const* _boundOffset;
	cadet::LinearBufferAllocator _buffer;
	void* _bufferMemory;
	std::vector<cadet::IExternalFunction*> _extFuns;
};

/**
 * @brief Checks the analytic Jacobian of the binding model against AD
 * @param [in] modelName Name of the binding model
 * @param [in] nComp Number of components
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] isKinetic Determines whether kinetic or quasi-stationary binding mode is applied
 * @param [in] config JSON string with binding model parameters
 * @param [in] point Liquid phase and solid phase values to check Jacobian at
 * @param [in] skipStructureTest Determines whether the structural test using finite differences is skipped
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testJacobianAD(const char* modelName, unsigned int nComp, unsigned int const* nBound, bool isKinetic,
					const char* config, double const* point, bool skipStructureTest = false, double absTol = 0.0,
					double relTol = std::numeric_limits<float>::epsilon() * 100.0);

/**
 * @brief Checks residual and analytic Jacobian of normal model variant against externally dependent ones
 * @param [in] modelName Name of the binding model
 * @param [in] modelNameExt Name of the externally dependent binding model variant
 * @param [in] nComp Number of components
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] isKinetic Determines whether kinetic or quasi-stationary binding mode is applied
 * @param [in] config JSON string with binding model parameters for both variants
 * @param [in] point Liquid phase and solid phase values to evaluate residual at
 */
void testNormalExternalConsistency(const char* modelName, const char* modelNameExt, unsigned int nComp,
								   unsigned int const* nBound, bool isKinetic, const char* config, double const* point);

/**
 * @brief Checks whether Jacobian columns of non-binding liquid phase components are all zero
 * @param [in] modelName Name of the binding model
 * @param [in] nComp Number of components
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] isKinetic Determines whether kinetic or quasi-stationary binding mode is applied
 * @param [in] config JSON string with binding model parameters
 * @param [in] useAD Determines whether the Jacobian is computed via AD or analytically
 * @param [in] point Liquid phase and solid phase values to evaluate Jacobian at
 */
void testNonBindingConsistency(const char* modelName, unsigned int nComp, unsigned int const* nBound, bool isKinetic,
							   const char* config, bool useAD, double const* point);

/**
 * @brief Checks whether Jacobian and residual of variants with all-binding and some non-binding components match
 * @param [in] modelName Name of the binding model
 * @param [in] nCompBnd Number of components in all binding variant
 * @param [in] nCompNonBnd Number of components in non-binding variant
 * @param [in] nBound Array with number of bound states for each component in all binding variant
 * @param [in] nBoundNonBnd Array with number of bound states for each component in non-binding variant
 * @param [in] isKinetic Determines whether kinetic or quasi-stationary binding mode is applied
 * @param [in] configBnd JSON string with binding model parameters for all binding variant
 * @param [in] configNonBnd JSON string with binding model parameters for non-binding variant
 * @param [in] useAD Determines whether the Jacobian is computed via AD or analytically
 * @param [in] pointBnd Liquid phase and solid phase values to evaluate Jacobian at in all binding variant
 * @param [in] pointNonBnd Liquid phase and solid phase values to evaluate Jacobian at in non-binding variant
 */
void testNonbindingBindingConsistency(const char* modelName, unsigned int nCompBnd, unsigned int nCompNonBnd,
									  unsigned int const* nBound, unsigned int const* nBoundNonBnd, bool isKinetic,
									  const char* configBnd, const char* configNonBnd, bool useAD,
									  double const* pointBnd, double const* pointNonBnd);

} // namespace binding
} // namespace test
} // namespace cadet

#endif // CADETTEST_BINDINGMODELTEST_HPP_
