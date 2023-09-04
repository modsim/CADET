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
 * Defines tests for reaction models.
 */

#ifndef CADETTEST_REACTIONODELTEST_HPP_
#define CADETTEST_REACTIONODELTEST_HPP_

#include "cadet/ParameterId.hpp"
#include <limits>
#include "Memory.hpp"

namespace cadet
{

class JsonParameterProvider;
class IExternalFunction;

namespace model
{
class IDynamicReactionModel;
}

namespace test
{

namespace reaction
{
class ConfiguredDynamicReactionModel
{
public:
	ConfiguredDynamicReactionModel(ConfiguredDynamicReactionModel&& cpy) CADET_NOEXCEPT
		: _reaction(cpy._reaction),
		  _nComp(cpy._nComp),
		  _nBound(cpy._nBound),
		  _boundOffset(cpy._boundOffset),
		  _buffer(std::move(cpy._buffer)),
		  _bufferMemory(cpy._bufferMemory),
		  _extFuns(cpy._extFuns)
	{
		cpy._reaction = nullptr;
		cpy._nBound = nullptr;
		cpy._boundOffset = nullptr;
		cpy._bufferMemory = nullptr;
		cpy._extFuns = nullptr;
	}

	~ConfiguredDynamicReactionModel();

	inline ConfiguredDynamicReactionModel& operator=(ConfiguredDynamicReactionModel&& cpy) CADET_NOEXCEPT
	{
		_reaction = cpy._reaction;
		_nComp = cpy._nComp;
		_nBound = cpy._nBound;
		_boundOffset = cpy._boundOffset;
		_buffer = std::move(cpy._buffer);
		_bufferMemory = cpy._bufferMemory;
		_extFuns = cpy._extFuns;

		cpy._reaction = nullptr;
		cpy._nBound = nullptr;
		cpy._boundOffset = nullptr;
		cpy._bufferMemory = nullptr;
		cpy._extFuns = nullptr;

		return *this;
	}

	static ConfiguredDynamicReactionModel create(const char* name, unsigned int nComp, unsigned int const* nBound,
												 const char* config);

	void increaseBufferSize(int inc);
	int requiredBufferSize() CADET_NOEXCEPT;

	inline cadet::model::IDynamicReactionModel& model()
	{
		return *_reaction;
	}
	inline const cadet::model::IDynamicReactionModel& model() const
	{
		return *_reaction;
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
	ConfiguredDynamicReactionModel(cadet::model::IDynamicReactionModel* reaction, unsigned int nComp,
								   unsigned int const* nBound, unsigned int const* boundOffset, void* bufferStart,
								   void* bufferEnd, cadet::IExternalFunction* extFuns)
		: _reaction(reaction), _nComp(nComp), _nBound(nBound), _boundOffset(boundOffset),
		  _buffer(bufferStart, bufferEnd), _bufferMemory(bufferStart), _extFuns(extFuns)
	{
	}

	cadet::model::IDynamicReactionModel* _reaction;
	unsigned int _nComp;
	unsigned int const* _nBound;
	unsigned int const* _boundOffset;
	cadet::LinearBufferAllocator _buffer;
	void* _bufferMemory;
	cadet::IExternalFunction* _extFuns;
};

/**
 * @brief Checks the analytic Jacobians of the dynamic reaction model against AD
 * @param [in] modelName Name of the reaction model
 * @param [in] nComp Number of components
 * @param [in] nBound Array with number of bound states for each component
 * @param [in] config JSON string with reaction model parameters
 * @param [in] point Liquid phase and solid phase values to check Jacobian at
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testDynamicJacobianAD(const char* modelName, unsigned int nComp, unsigned int const* nBound, const char* config,
						   double const* point, double absTol = 0.0,
						   double relTol = std::numeric_limits<float>::epsilon() * 100.0);

/**
 * @brief Extends a model with dynamic reactions in each phase and particle type
 * @param [in,out] jpp ParameterProvider to extend
 * @param [in] unit Index of unit operation
 * @param [in] bulk Determines whether reactions are added to bulk volume
 * @param [in] particle Determines whether reactions are added to each particle type
 * @param [in] particleModifiers Determines whether reaction rates in particles are modified by the respective other
 * phase
 */
void extendModelWithDynamicReactions(cadet::JsonParameterProvider& jpp, UnitOpIdx unit, bool bulk, bool particle,
									 bool particleModifiers);

/**
 * @brief Checks the full analytic Jacobian of a unit operation model with dynamic reactions against AD
 * @param [in] jpp Unit operation configuration
 * @param [in] bulk Determines whether reactions are added to bulk volume
 * @param [in] particle Determines whether reactions are added to each particle type
 * @param [in] particleModifiers Determines whether reaction rates in particles are modified by the respective other
 * phase
 */
void testUnitJacobianDynamicReactionsAD(cadet::JsonParameterProvider& jpp, bool bulk, bool particle,
										bool particleModifiers);

/**
 * @brief Checks the full analytic Jacobian of a unit operation model with dynamic reactions against AD
 * @details Uses a model with linear binding model and two components.
 * @param [in] uoType Unit operation type
 * @param [in] bulk Determines whether reactions are added to bulk volume
 * @param [in] particle Determines whether reactions are added to each particle type
 * @param [in] particleModifiers Determines whether reaction rates in particles are modified by the respective other
 * phase
 */
void testUnitJacobianDynamicReactionsAD(const std::string& uoType, bool bulk, bool particle, bool particleModifiers);

/**
 * @brief Checks the (analytic) time derivative Jacobian against FD for a model with dynamic reactions
 * @details Uses centered finite differences.
 * @param [in] jpp Unit operation configuration
 * @param [in] bulk Determines whether reactions are added to bulk volume
 * @param [in] particle Determines whether reactions are added to each particle type
 * @param [in] particleModifiers Determines whether reaction rates in particles are modified by the respective other
 * phase
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testTimeDerivativeJacobianDynamicReactionsFD(cadet::JsonParameterProvider& jpp, bool bulk, bool particle,
												  bool particleModifiers, double h, double absTol, double relTol);

/**
 * @brief Checks the (analytic) time derivative Jacobian against FD for a model with dynamic reactions
 * @details Uses centered finite differences. Uses a model with linear binding model and two components.
 * @param [in] uoType Unit operation type
 * @param [in] bulk Determines whether reactions are added to bulk volume
 * @param [in] particle Determines whether reactions are added to each particle type
 * @param [in] particleModifiers Determines whether reaction rates in particles are modified by the respective other
 * phase
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testTimeDerivativeJacobianDynamicReactionsFD(const std::string& uoType, bool bulk, bool particle,
												  bool particleModifiers, double h, double absTol, double relTol);

} // namespace reaction
} // namespace test
} // namespace cadet

#endif // CADETTEST_REACTIONODELTEST_HPP_
