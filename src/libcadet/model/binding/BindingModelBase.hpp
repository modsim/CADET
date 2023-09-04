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
 * Defines a BindingModelBase class.
 */

#ifndef LIBCADET_BINDINGMODELBASE_HPP_
#define LIBCADET_BINDINGMODELBASE_HPP_

#include "model/BindingModel.hpp"
#include "model/binding/BindingModelMacros.hpp"
#include "ParamIdUtil.hpp"

#include <vector>
#include <unordered_map>

namespace cadet
{

namespace model
{

/**
 * @brief Defines a BindingModel base class that can be used to implement other binding models
 * @details This base class can be used as a starting point for new binding models.
 *          Some common parameter handling is provided using a hash map (std::unordered_map).
 *
 *          By default, this base class assumes that the binding model does not use salt and
 *          does not support multiple bound states. Non-binding components are assumed to be
 *          supported, though. Note that these default settings can be overwritten and do not
 *          influence the features provided by this base class.
 */
class BindingModelBase : public IBindingModel
{
public:
	BindingModelBase();

	virtual ~BindingModelBase() CADET_NOEXCEPT;

	virtual bool requiresConfiguration() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool usesParamProviderInDiscretizationConfig() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx);
	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp,
											  unsigned int const* nBound, unsigned int const* boundOffset);
	virtual void fillBoundPhaseInitialParameters(ParameterId* params, UnitOpIdx unitOpIdx,
												 ParticleTypeIdx parTypeIdx) const CADET_NOEXCEPT;

	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const;
	virtual bool hasParameter(const ParameterId& pId) const;

	virtual bool setParameter(const ParameterId& pId, int value);
	virtual bool setParameter(const ParameterId& pId, double value);
	virtual bool setParameter(const ParameterId& pId, bool value);

	virtual active* getParameter(const ParameterId& pId);

	virtual bool hasSalt() const CADET_NOEXCEPT
	{
		return false;
	}
	virtual bool supportsMultistate() const CADET_NOEXCEPT
	{
		return false;
	}
	virtual bool supportsNonBinding() const CADET_NOEXCEPT
	{
		return true;
	}

	virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates,
									   unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		return 0;
	}

	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size)
	{
	}

	virtual void timeDerivativeQuasiStationaryFluxes(double t, unsigned int secIdx, const ColumnPosition& colPos,
													 double const* yCp, double const* y, double* dResDt,
													 LinearBufferAllocator workSpace) const
	{
	}

	virtual int const* reactionQuasiStationarity() const CADET_NOEXCEPT
	{
		return _reactionQuasistationarity.data();
	}
	virtual bool hasQuasiStationaryReactions() const CADET_NOEXCEPT
	{
		return _hasQuasiStationary;
	}
	virtual bool hasDynamicReactions() const CADET_NOEXCEPT
	{
		return _hasDynamic;
	}

	virtual bool preConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y,
										   double const* yCp, LinearBufferAllocator workSpace) const
	{
		return true;
	}
	virtual void postConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y,
											double const* yCp, LinearBufferAllocator workSpace) const
	{
	}

protected:
	int _nComp;                                  //!< Number of components
	unsigned int const* _nBoundStates;           //!< Array with number of bound states for each component
	std::vector<int> _reactionQuasistationarity; //!< Determines whether each bound state is quasi-stationary (@c true)
												 //!< or not (@c false)
	bool _hasQuasiStationary;                    //!< Caches whether the model contains quasi-stationary reaction fluxes
	bool _hasDynamic;                            //!< Caches whether the model contains dynamic reaction fluxes

	std::unordered_map<ParameterId, active*> _parameters; //!< Map used to translate ParameterIds to actual variables

	/**
	 * @brief Configures the binding model
	 * @details This function implements the (re-)configuration of a binding model. It is called when
	 *          the binding model is configured or reconfigured. On call the _parameters map will always
	 *          be empty and _kineticBinding is already configured.
	 * @param [in] paramProvider Parameter provider
	 * @param [in] unitOpIdx Unit operation index
	 * @param [in] parTypeIdx Particle type index
	 * @return @c true if the configuration was successful, otherwise @c false
	 */
	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx) = 0;
};

/**
 * @brief Defines a BindingModel base class that can be used to implement binding models with parameter handlers
 * @details In addition to BindingModelBase, this class supports the ParameterHandler pattern, which allows
 *          to easily add support for externally dependent parameters.
 * @tparam handler_t Type that can add support for external function dependence
 */
template <typename handler_t> class ParamHandlerBindingModelBase : public BindingModelBase
{
public:
	ParamHandlerBindingModelBase()
	{
	}

	virtual const char* name() const CADET_NOEXCEPT
	{
		return handler_t::identifier();
	}
	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size)
	{
		_paramHandler.setExternalFunctions(extFuns, size);
	}
	virtual bool dependsOnTime() const CADET_NOEXCEPT
	{
		return handler_t::dependsOnTime();
	}
	virtual bool requiresWorkspace() const CADET_NOEXCEPT
	{
		return handler_t::requiresWorkspace();
	}

	virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates,
									   unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		return _paramHandler.cacheSize(nComp, totalNumBoundStates, nBoundStates);
	}

protected:
	handler_t _paramHandler; //!< Handles parameters and their dependence on external functions

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx)
	{
		// Read parameters
		_paramHandler.configure(paramProvider, _nComp, _nBoundStates);

		// Register parameters
		_paramHandler.registerParameters(_parameters, unitOpIdx, parTypeIdx, _nComp, _nBoundStates);

		return true;
	}
};

/**
 * @brief Inserts implementations for common functions required by BindingModelBase forwarding them to templatized
 * functions
 * @details An implementation of IBindingModel has to provide some virtual functions.
 *          This macro provides the implementation of those functions by forwarding them to the templatized
 *          functions residualImpl() and jacobianImpl(), which are assumed to be present in the class.
 *
 *          The implementation is inserted inline in the class declaration.
 */
#define CADET_BINDINGMODELBASE_BOILERPLATE                                                                             \
	CADET_BINDINGMODEL_RESIDUAL_BOILERPLATE                                                                            \
	CADET_BINDINGMODEL_JACOBIAN_BOILERPLATE

} // namespace model
} // namespace cadet

#endif // LIBCADET_BINDINGMODELBASE_HPP_
