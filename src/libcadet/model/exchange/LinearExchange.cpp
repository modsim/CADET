// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================



#include "model/BindingModel.hpp"
#include "model/ExternalFunctionSupport.hpp"
#include "ParamIdUtil.hpp"
#include "model/ModelUtils.hpp"
#include "cadet/Exceptions.hpp"
#include "model/Parameters.hpp"
#include "LocalVector.hpp"
#include "SimulationTypes.hpp"
#include "Memory.hpp"

#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <iterator>
#include <tuple>

namespace cadet
{

namespace model
{

/**
 * @brief Handles linear exchange model parameters that do not depend on external functions
 */
class LinearParamHandler : public ConstParamHandlerBase
{
public:

	/**
	 * @brief Holds actual parameter data
	 */
	struct ConstParams
	{
		std::vector<active> kA; //!< Adsorption rate
		std::vector<active> kD; //!< Desorption rate
	};

	typedef ConstParams params_t;
	typedef ConstParams const* ParamsHandle;

	/**
	 * @brief Returns name of the binding model
	 * @return Name of the binding model
	 */
	static const char* identifier() { return "LINEAR_EX"; }

	LinearParamHandler() CADET_NOEXCEPT : _kA(&_localParams.kA), _kD(&_localParams.kD) { }

	/**
	 * @brief Reads parameters and verifies them
	 * @details See IexchangeModel::configure() for details.
	 * @param [in] paramProvider IParameterProvider used for reading parameters
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 * @return @c true if the parameters were read and validated successfully, otherwise @c false
	 */
	inline bool configure(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nChannel)
	{
		_kA.configure("LIN_KA", paramProvider, nComp, nChannel);
		_kD.configure("LIN_KD", paramProvider, nComp, nChannel);
		return validateConfig(nComp, nChannel);
	}

	/**
	 * @brief Registers all local parameters in a map for further use
	 * @param [in,out] parameters Map in which the parameters are stored
	 * @param [in] unitOpIdx Index of the unit operation used for registering the parameters
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 */
	inline void registerParameters(std::unordered_map<ParameterId, active*>& parameters, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp, unsigned int const* nChannel)
	{
		_kA.registerParam("LIN_KA", parameters, unitOpIdx, parTypeIdx, nComp, nChannel);
		_kD.registerParam("LIN_KD", parameters, unitOpIdx, parTypeIdx, nComp, nChannel);
	}

	/**
	 * @brief Reserves space in the storage of the parameters
	 * @param [in] numElem Number of elements (total)
	 * @param [in] numSlices Number of slices / exchange site types
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 */
	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp, unsigned int const* nChannel) \
	{
		_kA.reserve(numElem, numSlices, nComp, nChannel);
		_kD.reserve(numElem, numSlices, nComp, nChannel);
	}

	/**
	 * @brief Updates the parameter cache in order to take the external profile into account
	 * @param [in] t Current time
	 * @param [in] z Axial coordinate in the column
	 * @param [in] r Radial coordinate in the bead
	 * @param [in] secIdx Index of the current section
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 * @param [in,out] workSpace Memory buffer for updated data
	 * @return Externally dependent parameter values
	 */
	inline ParamsHandle update(double t, unsigned int secIdx, const ColumnPosition& colPos, unsigned int nComp, unsigned int const* nChannel, LinearBufferAllocator& workSpace) const
	{
		return &_localParams;
	}

	/**
	 * @brief Calculates time derivative in case of external dependence
	 * @param [in] t Current time
	 * @param [in] z Axial coordinate in the column
	 * @param [in] r Radial coordinate in the bead
	 * @param [in] secIdx Index of the current section
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 * @param [in,out] workSpace Memory buffer for updated data
	 * @return Time derivatives of externally dependent parameters
	 */
	inline std::tuple<ParamsHandle, ParamsHandle> updateTimeDerivative(double t, unsigned int secIdx, const ColumnPosition& colPos, unsigned int nComp, unsigned int const* nChannel, LinearBufferAllocator& workSpace) const
	{
		return std::make_tuple<ParamsHandle, ParamsHandle>(&_localParams, nullptr);
	}

protected:

	/**
	 * @brief Validates recently read parameters
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 * @return @c true if the parameters were validated successfully, otherwise @c false
	 */
	inline bool validateConfig(unsigned int nComp, unsigned int const* nChannel)
	{
		if ((_kA.size() != _kD.size()) || (_kA.size() < nComp))
			throw InvalidParameterException("LIN_KA and LIN_KD have to have the same size");

		return true;
	}

	ConstParams _localParams; //!< Actual parameter data

	// Handlers provide configure(), reserve(), and registerParam() for parameters
	ScalarComponentDependentParameter _kA; //!< Handler for adsorption rate
	ScalarComponentDependentParameter _kD; //!< Handler for desorption rate
};

/**
 * @brief Handles linear exchange model parameters that depend on an external function
 */
class ExtLinearParamHandler : public ExternalParamHandlerBase
{
public:

	/**
	 * @brief Holds actual parameter data
	 * @details The parameter data will be stored in a memory buffer. This requires
	 *          control over the location of the data, which is not provided by
	 *          std::vector and util::SlicedVector. There are "local" equivalents,
	 *          util::LocalVector and util::LocalSlicedVector, for these types that
	 *          allow to control the placement of the payload data. A single value (i.e.,
	 *          a single active or double) can simply be put in the struct.
	 */
	struct VariableParams
	{
		util::LocalVector<active> kA; //!< Adsorption rate
		util::LocalVector<active> kD; //!< Desorption rate
	};

	typedef VariableParams params_t;
	typedef ConstBufferedScalar<params_t> ParamsHandle;

	ExtLinearParamHandler() CADET_NOEXCEPT { }

	/**
	 * @brief Returns name of the exchange model
	 * @return Name of the exchange model
	 */
	static const char* identifier() { return "EXT_LINEAR"; }

	/**
	 * @brief Reads parameters and verifies them
	 * @details See IBindingModel::configure() for details.
	 * @param [in] paramProvider IParameterProvider used for reading parameters
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 * @return @c true if the parameters were read and validated successfully, otherwise @c false
	 */
	inline bool configure(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nChannel)
	{
		_kA.configure("LIN_KA", paramProvider, nComp, nChannel);
		_kD.configure("LIN_KD", paramProvider, nComp, nChannel);
		
		// Number of externally dependent parameters (2) needs to be given to ExternalParamHandlerBase::configure()
		ExternalParamHandlerBase::configure(paramProvider, 2);
		return validateConfig(nComp, nChannel);
	}

	/**
	 * @brief Registers all local parameters in a map for further use
	 * @param [in,out] parameters Map in which the parameters are stored
	 * @param [in] unitOpIdx Index of the unit operation used for registering the parameters
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 */
	inline void registerParameters(std::unordered_map<ParameterId, active*>& parameters, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp, unsigned int const* nChannel)
	{
		_kA.registerParam("LIN_KA", parameters, unitOpIdx, parTypeIdx, nComp, nChannel);
		_kD.registerParam("LIN_KD", parameters, unitOpIdx, parTypeIdx, nComp, nChannel);
	}

	/**
	 * @brief Reserves space in the storage of the parameters
	 * @param [in] numElem Number of elements (total)
	 * @param [in] numSlices Number of slices / exchange site types
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 */
	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp, unsigned int const* nChannel) \
	{
		_kA.reserve(numElem, numSlices, nComp, nChannel);
		_kD.reserve(numElem, numSlices, nComp, nChannel);
	}

	/**
	 * @brief Updates the parameter cache in order to take the external profile into account
	 * @details The data and the returned value are constructed in the given @p workSpace memory buffer.
	 * @param [in] t Current time
	 * @param [in] z Axial coordinate in the column
	 * @param [in] r Radial coordinate in the bead
	 * @param [in] secIdx Index of the current section
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 * @param [in,out] workSpace Memory buffer for updated data
	 * @return Externally dependent parameter values
	 */
	inline ParamsHandle update(double t, unsigned int secIdx, const ColumnPosition& colPos, unsigned int nComp, unsigned int const* nChannel, LinearBufferAllocator& workSpace) const
	{
		// Allocate params_t and buffer for function evaluation
		BufferedScalar<params_t> localParams = workSpace.scalar<params_t>();
		BufferedArray<double> extFunBuffer = workSpace.array<double>(2);

		// Evaluate external functions in buffer
		evaluateExternalFunctions(t, secIdx, colPos, 2, static_cast<double*>(extFunBuffer));

		// Prepare the buffer for the data and update the data
		_kA.prepareCache(localParams->kA, workSpace);
		_kA.update(cadet::util::dataOfLocalVersion(localParams->kA), extFunBuffer[0], nComp, nChannel);

		_kD.prepareCache(localParams->kD, workSpace);
		_kD.update(cadet::util::dataOfLocalVersion(localParams->kD), extFunBuffer[1], nComp, nChannel);

		return localParams;
	}

	/**
	 * @brief Calculates time derivative in case of external dependence
	 * @details The time derivatives are constructed in the given @p workSpace memory buffer.
	 * @param [in] t Current time
	 * @param [in] z Axial coordinate in the column
	 * @param [in] r Radial coordinate in the bead
	 * @param [in] secIdx Index of the current section
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 * @param [in,out] workSpace Memory buffer for updated data
	 * @return Tuple with externally dependent parameter values and their time derivatives
	 */
	inline std::tuple<ParamsHandle, ParamsHandle> updateTimeDerivative(double t, unsigned int secIdx, const ColumnPosition& colPos, unsigned int nComp, unsigned int const* nChannel, LinearBufferAllocator& workSpace) const
	{
		// Allocate params_t for parameters and their time derivatives
		BufferedScalar<params_t> localParams = workSpace.scalar<params_t>();
		BufferedScalar<params_t> p = workSpace.scalar<params_t>();

		// Allocate buffer for external function values and their time derivatives
		BufferedArray<double> extFunBuffer = workSpace.array<double>(2);
		BufferedArray<double> extDerivBuffer = workSpace.array<double>(2);

		// Evaluate external functions and their time derivatives
		evaluateExternalFunctions(t, secIdx, colPos, 2, static_cast<double*>(extFunBuffer));
		evaluateTimeDerivativeExternalFunctions(t, secIdx, colPos, 2, static_cast<double*>(extDerivBuffer));

		// Prepare the buffer for the data and update the data
		_kA.prepareCache(localParams->kA, workSpace);
		_kA.update(cadet::util::dataOfLocalVersion(localParams->kA), extFunBuffer[0], nComp, nChannel);

		_kA.prepareCache(p->kA, workSpace);
		_kA.updateTimeDerivative(cadet::util::dataOfLocalVersion(p->kA), extFunBuffer[0], extDerivBuffer[0], nComp, nChannel);

		_kD.prepareCache(localParams->kD, workSpace);
		_kD.update(cadet::util::dataOfLocalVersion(localParams->kD), extFunBuffer[1], nComp, nChannel);

		_kD.prepareCache(p->kD, workSpace);
		_kD.updateTimeDerivative(cadet::util::dataOfLocalVersion(p->kD), extFunBuffer[1], extDerivBuffer[1], nComp, nChannel);

		return std::make_tuple<ParamsHandle, ParamsHandle>(std::move(localParams), std::move(p));;
	}

	/**
	 * @brief Returns how much memory is required for caching in bytes
	 * @details Memory size in bytes.
	 * @param [in] nComp Number of components
	 * @param [in] totalNumBoundStates Total number of bound states
	 * @param [in] nChannel Array with bound states for each component
	 * @return Memory size in bytes
	 */
	inline std::size_t cacheSize(unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nChannel) const CADET_NOEXCEPT
	{
		// Required buffer memory:
		//  + params_t object
		//  + buffer for external function evaluations (2 parameters)
		//  + buffer for external function time derivative evaluations (2 parameters)
		//  + buffer for actual parameter data (memory for _kA data + memory for _kD data)
		//  + buffer for parameter time derivatives (memory for _kA data + memory for _kD data)
		return 2 * sizeof(params_t) + alignof(params_t) 
			+ 2 * 2 * sizeof(double) + alignof(double) 
			+ 2 * (_kA.additionalDynamicMemory(nComp, totalNumBoundStates, nChannel) + _kD.additionalDynamicMemory(nComp, totalNumBoundStates, nChannel));
	}

protected:

	/**
	 * @brief Validates recently read parameters
	 * @param [in] nComp Number of components
	 * @param [in] nChannel Array with number of bound states for each component
	 * @return @c true if the parameters were validated successfully, otherwise @c false
	 */
	inline bool validateConfig(unsigned int nComp, unsigned int const* nChannel)
	{
		if ((_kA.size() != _kD.size()) || (_kA.size() < nComp))
			throw InvalidParameterException("LIN_KA and LIN_KD have to have the same size");

		return true;
	}

	// Handlers provide configure(), reserve(), and registerParam() for parameters
	ExternalScalarComponentDependentParameter _kA; //!< Handler for adsorption rate
	ExternalScalarComponentDependentParameter _kD; //!< Handler for desorption rate
};


/**
 * @brief Defines the linear exchange model
 * @details Implements the linear adsorption model \f$ \frac{\mathrm{d}q}{\mathrm{d}t} = k_a c_p - k_d q \f$.
 *          Multiple bound states are not supported. Components without bound state (i.e., non-exchange components)
 *          are supported.
 *          
 *          See @cite Guiochon2006
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
class LinearExchangeBase : public IPhaseTransitionModel
{
public:

	LinearExchangeBase() : _nComp(0), _nChannel(0) { }
	virtual ~LinearExchangeBase() CADET_NOEXCEPT { }

	static const char* identifier() { return ParamHandler_t::identifier(); }
	virtual const char* name() const CADET_NOEXCEPT { return ParamHandler_t::identifier(); }
	virtual bool requiresConfiguration() const CADET_NOEXCEPT { return true; }
	virtual bool usesParamProviderInDiscretizationConfig() const CADET_NOEXCEPT { return true; }

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int nChannel, unsigned int const* boundOffset)
	{
		_nComp = nComp;
		_nChannel = nChannel; // nChannel realy int ? -> by not nBoundStates not ?

		return true;
	}

	virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx)
	{
		_parameters.clear();

		// Read parameters (k_a and k_d)
		_paramHandler.configure(paramProvider, _nComp, _nChannel);

		// Register parameters
		_paramHandler.registerParameters(_parameters, unitOpIdx, parTypeIdx, _nComp, _nChannel);

		return true;
	}

	virtual void fillExchangeInitialParameters(ParameterId* params, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx) const CADET_NOEXCEPT
	{
		unsigned int ctr = 0;
		for (int c = 0; c < _nComp; ++c)
		{
			for (unsigned int bp = 0; bp < _nChannel; ++bp, ++ctr)
				params[ctr] = makeParamId(hashString("INIT_C"), unitOpIdx, c, parTypeIdx, bp, ReactionIndep, SectionIndep);
		}
	}
	// ----------------------------------------------------------------------------------------------------------------------------// 
	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const
	{
		std::unordered_map<ParameterId, double> data;
		std::transform(_parameters.begin(), _parameters.end(), std::inserter(data, data.end()),
		               [](const std::pair<const ParameterId, active*>& p) { return std::make_pair(p.first, static_cast<double>(*p.second)); });

		return data;
	}

	virtual bool hasParameter(const ParameterId& pId) const
	{
		return _parameters.find(pId) != _parameters.end();
	}

	virtual bool setParameter(const ParameterId& pId, int value)
	{
		return false;
	}

	virtual bool setParameter(const ParameterId& pId, double value)
	{
		auto paramHandle = _parameters.find(pId);
		if (paramHandle != _parameters.end())
		{
			paramHandle->second->setValue(value);
			return true;
		}

		return false;
	}

	virtual bool setParameter(const ParameterId& pId, bool value)
	{
		return false;
	}

	virtual active* getParameter(const ParameterId& pId){
		auto paramHandle = _parameters.find(pId);
		if (paramHandle != _parameters.end())
		{
			return paramHandle->second;
		}

		return nullptr;
	}

	virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nChannel) const CADET_NOEXCEPT
	{
		return _paramHandler.cacheSize(nComp, totalNumBoundStates, nChannel);
	}

	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size) { _paramHandler.setExternalFunctions(extFuns, size); }

	// The next three flux() function implementations and two analyticJacobian() function
	// implementations are usually hidden behind
	// CADET_BINDINGMODELBASE_BOILERPLATE
	// which just expands to the six implementations below.

	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos,
		active const* y, active const* yCp, active* res, LinearBufferAllocator workSpace, WithParamSensitivity) const
	{
		return fluxImpl<active, active, active>(t, secIdx, colPos, y, yCp, res, workSpace);
	}

	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos,
		active const* y, active const* yCp, active* res, LinearBufferAllocator workSpace, WithoutParamSensitivity) const
	{
		return fluxImpl<active, active, double>(t, secIdx, colPos, y, yCp, res, workSpace);
	}

	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos,
		double const* y, double const* yCp, active* res, LinearBufferAllocator workSpace) const
	{
		return fluxImpl<double, active, active>(t, secIdx, colPos, y, yCp, res, workSpace);
	}

	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos,
		double const* y, double const* yCp, double* res, LinearBufferAllocator workSpace) const
	{
		return fluxImpl<double, double, double>(t, secIdx, colPos, y, yCp, res, workSpace);
	}

	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, linalg::BandMatrix::RowIterator jac, LinearBufferAllocator workSpace) const
	{
		jacobianImpl(t, secIdx, colPos, y, offsetCp, jac, workSpace);
	}

	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, linalg::DenseBandedRowIterator jac, LinearBufferAllocator workSpace) const
	{
		jacobianImpl(t, secIdx, colPos, y, offsetCp, jac, workSpace);
	}

	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, linalg::BandedEigenSparseRowIterator jac, LinearBufferAllocator workSpace) const
	{
		jacobianImpl(t, secIdx, colPos, y, offsetCp, jac, workSpace);
	}

	virtual void timeDerivativeQuasiStationaryFluxes(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yCp, double const* y, double* dResDt, LinearBufferAllocator workSpace) const
	{
		if (!hasQuasiStationaryReactions())
			return;

		if (!ParamHandler_t::dependsOnTime())
			return;

		// Update external function and compute time derivative of parameters
		typename ParamHandler_t::ParamsHandle p;
		typename ParamHandler_t::ParamsHandle dpDt;
		std::tie(p, dpDt) = _paramHandler.updateTimeDerivative(t, secIdx, colPos, _nComp, _nChannel, workSpace);

		unsigned int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nChannel[i] == 0)
				continue;

			dResDt[bndIdx] = -static_cast<double>(dpDt->kA[i]) * yCp[i] + static_cast<double>(dpDt->kD[i]) * y[bndIdx];

			// Next bound component
			++bndIdx;
		}
	}

	virtual bool preConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const
	{
		// Due to mass conservation, we need to run a nonlinear solver (although the model is simple).
		return true;
	}

	virtual void postConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const
	{
		// There's nothing to do here since the algebraic equations have already been solved in preConsistentInitialState()
	}

	virtual bool hasSalt() const CADET_NOEXCEPT { return false; }
	virtual bool supportsMultistate() const CADET_NOEXCEPT { return false; }
	virtual bool supportsNonExchange() const CADET_NOEXCEPT { return true; }

	virtual bool hasQuasiStationaryReactions() const CADET_NOEXCEPT
	{
		return std::any_of(_reactionQuasistationarity.begin(), _reactionQuasistationarity.end(), [](int i) -> bool { return i; });
	}

	virtual bool hasDynamicReactions() const CADET_NOEXCEPT
	{
		return std::any_of(_reactionQuasistationarity.begin(), _reactionQuasistationarity.end(), [](int i) -> bool { return !static_cast<bool>(i); });
	}

	virtual bool dependsOnTime() const CADET_NOEXCEPT { return ParamHandler_t::dependsOnTime(); }
	virtual bool requiresWorkspace() const CADET_NOEXCEPT { return ParamHandler_t::requiresWorkspace(); }
	virtual unsigned int requiredADdirs() const CADET_NOEXCEPT { return 0; }
	virtual int const* reactionQuasiStationarity() const CADET_NOEXCEPT { return _reactionQuasistationarity.data(); }

protected:
	int _nComp; //!< Number of components
	unsigned int const* _nChannel; //!< Array with number of bound states for each component
	std::vector<int> _reactionQuasistationarity; //!< Determines whether each bound state is quasi-stationary (@c true) or not (@c false)

	ParamHandler_t _paramHandler; //!< Parameters

	std::unordered_map<ParameterId, active*> _parameters; //!< Map used to translate ParameterIds to actual variables

	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT { return true; }

	template <typename StateType, typename ResidualType, typename ParamType>

	int fluxImpl(double t, unsigned int secIdx, const ColumnPosition& colPos,
		StateType const* y, StateType const* yCp, ResidualType* res, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nChannel, workSpace);

		// Implement -k_a * c_{p,i} + k_d * q_i

		unsigned int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nChannel[i] == 0)
				continue;

			res[bndIdx] = -static_cast<ParamType>(p->kA[i]) * yCp[i] + static_cast<ParamType>(p->kD[i]) * y[bndIdx];

			// Next bound component
			++bndIdx;
		}

		return 0;
	}

	template <typename RowIterator>
	inline void jacobianImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, RowIterator jac, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p = _paramHandler.update(t, secIdx, colPos, _nComp, _nChannel, workSpace);

		int bndIdx = 0;
		for (int i = 0; i < _nComp; ++i)
		{
			// Skip components without bound states (bound state index bndIdx is not advanced)
			if (_nChannel[i] == 0)
				continue;

			jac[0] = static_cast<double>(p->kD[i]); // dres / dq_i
			jac[i - bndIdx - offsetCp] = -static_cast<double>(p->kA[i]); // dres / dc_{p,i}
			// The distance from liquid phase to solid phase is reduced for each non-exchange component
			// since a bound state is neglected. The number of neglected bound states so far is i - bndIdx.
			// Thus, by going back offsetCp - (i - bndIdx) = -[ i - bndIdx - offsetCp ] we get to the corresponding
			// liquid phase component.

			++bndIdx;
			++jac;
		}
	}

};

typedef LinearExchangeBase<LinearParamHandler> LinearExchange;
typedef LinearExchangeBase<ExtLinearParamHandler> ExternalLinearExchange;

namespace exchange
{
	void registerLinearModel(std::unordered_map<std::string, std::function<model::IPhaseTransitionModel*()>>& exchange)
	{
		exchange[LinearExchange::identifier()] = []() { return new LinearExchange(); };
		exchange[ExternalLinearExchange::identifier()] = []() { return new ExternalLinearExchange(); };
	}
}  // namespace exchange

}  // namespace model

}  // namespace cadet
