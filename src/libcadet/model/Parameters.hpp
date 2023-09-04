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
 * Provides parameter classes.
 */

#ifndef LIBCADET_PARAMETERS_HPP_
#define LIBCADET_PARAMETERS_HPP_

#include "common/CompilerSpecific.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/ParameterId.hpp"
#include "AutoDiff.hpp"
#include "SlicedVector.hpp"
#include "ParamReaderHelper.hpp"
#include "model/binding/RefConcentrationSupport.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <type_traits>
#include <algorithm>

namespace cadet
{

namespace model
{

/**
 * @brief Scalar bool external parameter
 * @details Just a single bool value.
 */
class ScalarBoolParameter
{
public:
	/**
	 * @brief Underlying type
	 */
	typedef bool storage_t;

	ScalarBoolParameter(bool& p) : _p(&p)
	{
	}
	ScalarBoolParameter(bool* p) : _p(p)
	{
	}

	/**
	 * @brief Reads parameters and verifies them
	 * @details See IBindingModel::configure() for details.
	 * @param [in] varName Name of the parameter
	 * @param [in] paramProvider IParameterProvider used for reading parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 * @return @c true if the parameters were read and validated successfully, otherwise @c false
	 */
	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		*_p = paramProvider.getBool(varName);
	}

	/**
	 * @brief Registers the parameters in a map for further use
	 * @param [in] varName Name of the parameter
	 * @param [in,out] parameters Map in which the parameters are stored
	 * @param [in] unitOpIdx Index of the unit operation used for registering the parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
	}

	/**
	 * @brief Reserves space in the storage of the parameters
	 * @param [in] numElem Total number of components in all slices / binding site types
	 * @param [in] numSlices Number of slices / binding site types
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
	}

	inline const bool& get() const CADET_NOEXCEPT
	{
		return *_p;
	}
	inline bool& get() CADET_NOEXCEPT
	{
		return *_p;
	}

	/**
	 * @brief Returns the number of elements in the parameter
	 * @return Number of elements in the parameter
	 */
	inline std::size_t size() const CADET_NOEXCEPT
	{
		return 1;
	}

protected:
	bool* _p;
};

/**
 * @brief Scalar external parameter
 * @details Just a single value.
 */
class ScalarParameter
{
public:
	/**
	 * @brief Underlying type
	 */
	typedef active storage_t;

	ScalarParameter(active& p) : _p(&p)
	{
	}
	ScalarParameter(active* p) : _p(p)
	{
	}

	/**
	 * @brief Reads parameters and verifies them
	 * @details See IBindingModel::configure() for details.
	 * @param [in] varName Name of the parameter
	 * @param [in] paramProvider IParameterProvider used for reading parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 * @return @c true if the parameters were read and validated successfully, otherwise @c false
	 */
	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		*_p = paramProvider.getDouble(varName);
	}

	/**
	 * @brief Reads parameters and verifies them
	 * @details See IBindingModel::configure() for details.
	 * @param [in] varName Name of the parameter
	 * @param [in] paramProvider IParameterProvider used for reading parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 * @param [in] defaultVal Default value if parameter is not available in @p paramProvider
	 * @return @c true if the parameters were read and validated successfully, otherwise @c false
	 */
	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates, double defaultVal)
	{
		if (paramProvider.exists(varName))
			*_p = paramProvider.getDouble(varName);
		else
			*_p = defaultVal;
	}

	/**
	 * @brief Registers the parameters in a map for further use
	 * @param [in] varName Name of the parameter
	 * @param [in,out] parameters Map in which the parameters are stored
	 * @param [in] unitOpIdx Index of the unit operation used for registering the parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		parameters[makeParamId(hashStringRuntime(varName), unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep,
							   ReactionIndep, SectionIndep)] = _p;
	}

	/**
	 * @brief Reserves space in the storage of the parameters
	 * @param [in] numElem Total number of components in all slices / binding site types
	 * @param [in] numSlices Number of slices / binding site types
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
	}

	inline const active& get() const CADET_NOEXCEPT
	{
		return *_p;
	}
	inline active& get() CADET_NOEXCEPT
	{
		return *_p;
	}

	/**
	 * @brief Returns the number of elements in the parameter
	 * @return Number of elements in the parameter
	 */
	inline std::size_t size() const CADET_NOEXCEPT
	{
		return 1;
	}

protected:
	active* _p;
};

/**
 * @brief Component dependent scalar external parameter
 * @details A single value per component.
 */
class ScalarComponentDependentParameter
{
public:
	typedef std::vector<active> storage_t;

	ScalarComponentDependentParameter(std::vector<active>& p) : _p(&p)
	{
	}
	ScalarComponentDependentParameter(std::vector<active>* p) : _p(p)
	{
	}

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		readParameterMatrix(*_p, paramProvider, varName, nComp, 1);
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		registerComponentBoundStateDependentParam(hashStringRuntime(varName), parameters, *_p, unitOpIdx, parTypeIdx);
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		_p->reserve(nComp);
	}

	inline std::size_t size() const CADET_NOEXCEPT
	{
		return _p->size();
	}

	inline const std::vector<active>& get() const CADET_NOEXCEPT
	{
		return *_p;
	}
	inline std::vector<active>& get() CADET_NOEXCEPT
	{
		return *_p;
	}

protected:
	std::vector<active>* _p;
};

/**
 * @brief Bound state dependent scalar external parameter
 * @details A single value per bound state. Can be used for multiple binding site types.
 */
class ScalarBoundStateDependentParameter
{
public:
	typedef std::vector<active> storage_t;

	ScalarBoundStateDependentParameter(std::vector<active>& p) : _p(&p)
	{
	}
	ScalarBoundStateDependentParameter(std::vector<active>* p) : _p(p)
	{
	}

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		readScalarParameterOrArray(*_p, paramProvider, varName, 1);
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		registerScalarBoundStateDependentParam(hashStringRuntime(varName), parameters, *_p, unitOpIdx, parTypeIdx);
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		_p->reserve(numSlices);
	}

	inline std::size_t size() const CADET_NOEXCEPT
	{
		return _p->size();
	}

	inline const std::vector<active>& get() const CADET_NOEXCEPT
	{
		return *_p;
	}
	inline std::vector<active>& get() CADET_NOEXCEPT
	{
		return *_p;
	}

protected:
	std::vector<active>* _p;
};

/**
 * @brief Reaction dependent scalar external parameter
 * @details A single value per reaction.
 */
class ScalarReactionDependentParameter
{
public:
	typedef std::vector<active> storage_t;

	ScalarReactionDependentParameter(std::vector<active>& p) : _p(&p)
	{
	}
	ScalarReactionDependentParameter(std::vector<active>* p) : _p(p)
	{
	}

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		if (paramProvider.exists(varName))
			readParameterMatrix(*_p, paramProvider, varName, 1, 1);
		else
			_p->clear();
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		const StringHash nameHash = hashStringRuntime(varName);
		registerParam1DArray(parameters, *_p, [=](bool multi, unsigned int r) {
			return makeParamId(nameHash, unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep, r, SectionIndep);
		});
	}

	inline void reserve(unsigned int nReactions, unsigned int nComp, unsigned int nBoundStates)
	{
		_p->reserve(nReactions);
	}

	inline std::size_t size() const CADET_NOEXCEPT
	{
		return _p->size();
	}

	inline const std::vector<active>& get() const CADET_NOEXCEPT
	{
		return *_p;
	}
	inline std::vector<active>& get() CADET_NOEXCEPT
	{
		return *_p;
	}

protected:
	std::vector<active>* _p;
};

/**
 * @brief Component and bound state dependent external parameter
 * @details A single value per component and bound state / binding site type.
 * @tparam compMajor Determines whether the values are stored in component-major or bound-state-/binding-site-type-major
 * ordering
 */
template <bool compMajor> class BaseComponentBoundStateDependentParameter
{
public:
	typedef util::SlicedVector<active> storage_t;

	BaseComponentBoundStateDependentParameter(util::SlicedVector<active>& p) : _p(&p)
	{
	}
	BaseComponentBoundStateDependentParameter(util::SlicedVector<active>* p) : _p(p)
	{
	}

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		if (compMajor)
		{
			readBoundStateDependentParameter<util::SlicedVector<active>, active>(*_p, paramProvider, varName, nComp,
																				 nBoundStates);
		}
		else
		{
			const unsigned int numStates = firstNonEmptyBoundStates(nBoundStates, nComp);
			readBoundStateDependentParameter<util::SlicedVector<active>, active>(*_p, paramProvider, varName, nComp,
																				 numStates);
		}
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		if (compMajor)
		{
			registerComponentBoundStateDependentParamCompMajor(hashStringRuntime(varName), parameters, *_p, unitOpIdx,
															   parTypeIdx);
		}
		else
		{
			registerComponentBoundStateDependentParam(hashStringRuntime(varName), parameters, *_p, unitOpIdx,
													  parTypeIdx);
		}
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		_p->reserve(numElem, numSlices);
	}

	inline typename util::SlicedVector<active>::size_type slices() const CADET_NOEXCEPT
	{
		return _p->slices();
	}
	inline typename util::SlicedVector<active>::size_type size() const CADET_NOEXCEPT
	{
		return _p->size();
	}

	inline const util::SlicedVector<active>& get() const CADET_NOEXCEPT
	{
		return *_p;
	}
	inline util::SlicedVector<active>& get() CADET_NOEXCEPT
	{
		return *_p;
	}

protected:
	util::SlicedVector<active>* _p;
};

/**
 * @brief Component and bound state dependent parameter
 * @details Components contain values for each bound state.
 */
typedef BaseComponentBoundStateDependentParameter<true> ComponentMajorBoundStateDependentParameter;

/**
 * @brief Component and bound state dependent parameter
 * @details Binding site types contain values for each component.
 */
typedef BaseComponentBoundStateDependentParameter<false> ComponentBoundStateMajorDependentParameter;

/**
 * @brief Component dependent bound-state matrix valued external parameter
 * @details Holds a square matrix for each component that has the size of the number of corresponding bound states.
 */
class ComponentDependentBoundStateMatrixParameter
{
public:
	typedef util::SlicedVector<active> storage_t;

	ComponentDependentBoundStateMatrixParameter(util::SlicedVector<active>& p) : _p(&p)
	{
	}
	ComponentDependentBoundStateMatrixParameter(util::SlicedVector<active>* p) : _p(p)
	{
	}

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		readMatrixValuedBoundStateDependentParameter<util::SlicedVector<active>, active>(*_p, paramProvider, varName,
																						 nComp, nBoundStates);
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		registerComponentBoundStateDependentParamCompMajor(hashStringRuntime(varName), parameters, *_p, unitOpIdx,
														   parTypeIdx);
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		unsigned int sumSquared = 0;
		for (unsigned int i = 0; i < nComp; ++i)
			sumSquared += nBoundStates[i] * nBoundStates[i];

		_p->reserve(sumSquared, nComp);
	}

	inline typename util::SlicedVector<active>::size_type slices() const CADET_NOEXCEPT
	{
		return _p->slices();
	}
	inline typename util::SlicedVector<active>::size_type size() const CADET_NOEXCEPT
	{
		return _p->size();
	}

	inline const util::SlicedVector<active>& get() const CADET_NOEXCEPT
	{
		return *_p;
	}
	inline util::SlicedVector<active>& get() CADET_NOEXCEPT
	{
		return *_p;
	}

protected:
	util::SlicedVector<active>* _p;
};

/**
 * @brief Component dependent component vector valued parameter
 * @details Holds a vector for each component that has the size of the number of components.
 */
class ComponentDependentComponentVectorParameter
{
public:
	typedef util::SlicedVector<active> storage_t;

	ComponentDependentComponentVectorParameter(util::SlicedVector<active>& p) : _p(&p)
	{
	}
	ComponentDependentComponentVectorParameter(util::SlicedVector<active>* p) : _p(p)
	{
	}

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		readBoundStateDependentParameter<util::SlicedVector<active>, active>(*_p, paramProvider, varName, nComp, nComp);
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		registerComponentBoundStateDependentParam(hashStringRuntime(varName), parameters, *_p, unitOpIdx, parTypeIdx);
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		const unsigned int compSquared = nComp * nComp;
		_p->reserve(compSquared, nComp);
	}

	inline typename util::SlicedVector<active>::size_type slices() const CADET_NOEXCEPT
	{
		return _p->slices();
	}
	inline typename util::SlicedVector<active>::size_type size() const CADET_NOEXCEPT
	{
		return _p->size();
	}

	inline const util::SlicedVector<active>& get() const CADET_NOEXCEPT
	{
		return *_p;
	}
	inline util::SlicedVector<active>& get() CADET_NOEXCEPT
	{
		return *_p;
	}

protected:
	util::SlicedVector<active>* _p;
};

/**
 * @brief Scalar reference conentrations
 * @details Just one liquid and one solid phase reference concentration.
 */
class ReferenceConcentrationParameter
{
public:
	/**
	 * @brief Underlying type
	 */
	typedef active storage_t;

	ReferenceConcentrationParameter(active& refC, active& refQ) : _refC(&refC), _refQ(&refQ)
	{
	}
	ReferenceConcentrationParameter(active* refC, active* refQ) : _refC(refC), _refQ(refQ)
	{
	}

	/**
	 * @brief Reads parameters and verifies them
	 * @details See IBindingModel::configure() for details.
	 * @param [in] varName Name of the parameter
	 * @param [in] paramProvider IParameterProvider used for reading parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 * @return @c true if the parameters were read and validated successfully, otherwise @c false
	 */
	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		readReferenceConcentrations(paramProvider, varName, *_refC, *_refQ);
	}

	/**
	 * @brief Registers the parameters in a map for further use
	 * @param [in] varName Name of the parameter
	 * @param [in,out] parameters Map in which the parameters are stored
	 * @param [in] unitOpIdx Index of the unit operation used for registering the parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		parameters[makeParamId(hashStringRuntime(varName + "REFC0"), unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep,
							   ReactionIndep, SectionIndep)] = _refC;
		parameters[makeParamId(hashStringRuntime(varName + "REFQ"), unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep,
							   ReactionIndep, SectionIndep)] = _refQ;
	}

	/**
	 * @brief Reserves space in the storage of the parameters
	 * @param [in] numElem Total number of components in all slices / binding site types
	 * @param [in] numSlices Number of slices / binding site types
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
	}

	/**
	 * @brief Returns the number of elements in the parameter
	 * @return Number of elements in the parameter
	 */
	inline std::size_t size() const CADET_NOEXCEPT
	{
		return 1;
	}

	inline const active& getC() const CADET_NOEXCEPT
	{
		return *_refC;
	}
	inline active& getC() CADET_NOEXCEPT
	{
		return *_refC;
	}

	inline const active& getQ() const CADET_NOEXCEPT
	{
		return *_refQ;
	}
	inline active& getQ() CADET_NOEXCEPT
	{
		return *_refQ;
	}

protected:
	active* _refC; //!< Reference liquid phase concentration
	active* _refQ; //!< Reference solid phase concentration
};

/**
 * @brief Vectorial reference conentrations
 * @details Liquid and one solid phase reference concentrations per binding site type.
 */
class ReferenceConcentrationBoundStateDependentParameter
{
public:
	/**
	 * @brief Underlying type
	 */
	typedef std::vector<active> storage_t;

	ReferenceConcentrationBoundStateDependentParameter(std::vector<active>& refC, std::vector<active>& refQ)
		: _refC(&refC), _refQ(&refQ)
	{
	}
	ReferenceConcentrationBoundStateDependentParameter(std::vector<active>* refC, std::vector<active>* refQ)
		: _refC(refC), _refQ(refQ)
	{
	}

	/**
	 * @brief Reads parameters and verifies them
	 * @details See IBindingModel::configure() for details.
	 * @param [in] varName Name of the parameter
	 * @param [in] paramProvider IParameterProvider used for reading parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 * @return @c true if the parameters were read and validated successfully, otherwise @c false
	 */
	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		const unsigned int numStates = firstNonEmptyBoundStates(nBoundStates, nComp);
		readReferenceConcentrations(paramProvider, numStates, varName, *_refC, *_refQ);
	}

	/**
	 * @brief Registers the parameters in a map for further use
	 * @param [in] varName Name of the parameter
	 * @param [in,out] parameters Map in which the parameters are stored
	 * @param [in] unitOpIdx Index of the unit operation used for registering the parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		registerScalarBoundStateDependentParam(hashStringRuntime(varName + "REFC0"), parameters, *_refC, unitOpIdx,
											   parTypeIdx);
		registerScalarBoundStateDependentParam(hashStringRuntime(varName + "REFQ"), parameters, *_refQ, unitOpIdx,
											   parTypeIdx);
	}

	/**
	 * @brief Reserves space in the storage of the parameters
	 * @param [in] numElem Total number of components in all slices / binding site types
	 * @param [in] numSlices Number of slices / binding site types
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		_refC->reserve(numSlices);
		_refQ->reserve(numSlices);
	}

	/**
	 * @brief Returns the number of elements in the parameter
	 * @return Number of elements in the parameter
	 */
	inline std::size_t size() const CADET_NOEXCEPT
	{
		return _refC->size();
	}

protected:
	std::vector<active>* _refC; //!< Reference liquid phase concentration
	std::vector<active>* _refQ; //!< Reference solid phase concentration
};

/**
 * @brief Scalar external parameter
 * @details Just a single value.
 */
class ExternalScalarParameter
{
public:
	/**
	 * @brief Underlying type
	 */
	typedef active storage_t;

	/**
	 * @brief Reads parameters and verifies them
	 * @details See IBindingModel::configure() for details.
	 * @param [in] varName Name of the parameter
	 * @param [in] paramProvider IParameterProvider used for reading parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 * @return @c true if the parameters were read and validated successfully, otherwise @c false
	 */
	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		_base = paramProvider.getDouble("EXT_" + varName);
		_linear = paramProvider.getDouble("EXT_" + varName + "_T");
		_quad = paramProvider.getDouble("EXT_" + varName + "_TT");
		_cube = paramProvider.getDouble("EXT_" + varName + "_TTT");
	}

	/**
	 * @brief Registers the parameters in a map for further use
	 * @param [in] varName Name of the parameter
	 * @param [in,out] parameters Map in which the parameters are stored
	 * @param [in] unitOpIdx Index of the unit operation used for registering the parameters
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		parameters[makeParamId(hashStringRuntime("EXT_" + varName), unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep,
							   ReactionIndep, SectionIndep)] = &_base;
		parameters[makeParamId(hashStringRuntime("EXT_" + varName + "_T"), unitOpIdx, CompIndep, parTypeIdx,
							   BoundStateIndep, ReactionIndep, SectionIndep)] = &_linear;
		parameters[makeParamId(hashStringRuntime("EXT_" + varName + "_TT"), unitOpIdx, CompIndep, parTypeIdx,
							   BoundStateIndep, ReactionIndep, SectionIndep)] = &_quad;
		parameters[makeParamId(hashStringRuntime("EXT_" + varName + "_TTT"), unitOpIdx, CompIndep, parTypeIdx,
							   BoundStateIndep, ReactionIndep, SectionIndep)] = &_cube;
	}

	/**
	 * @brief Reserves space in the storage of the parameters
	 * @param [in] numElem Total number of components in all slices / binding site types
	 * @param [in] numSlices Number of slices / binding site types
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
	}

	/**
	 * @brief Calculates a parameter in order to take the external profile into account
	 * @param [out] result Stores the result of the paramter
	 * @param [in] extVal Value of the external function
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void update(active& result, double extVal, unsigned int nComp, unsigned int const* nBoundStates) const
	{
		update(&result, extVal, nComp, nBoundStates);
	}

	/**
	 * @brief Calculates a parameter in order to take the external profile into account
	 * @param [out] result Stores the result of the paramter
	 * @param [in] extVal Value of the external function
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void update(active* result, double extVal, unsigned int nComp, unsigned int const* nBoundStates) const
	{
		*result = _base + extVal * (_linear + extVal * (_quad + extVal * _cube));
	}

	/**
	 * @brief Calculates time derivative of parameter in case of external dependence
	 * @param [out] result Stores the result of the paramter
	 * @param [in] extVal Value of the external function
	 * @param [in] extTimeDiff Time derivative of the external function
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void updateTimeDerivative(active& result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		updateTimeDerivative(&result, extVal, extTimeDiff, nComp, nBoundStates);
	}

	/**
	 * @brief Calculates time derivative of parameter in case of external dependence
	 * @param [out] result Stores the result of the paramter
	 * @param [in] extVal Value of the external function
	 * @param [in] extTimeDiff Time derivative of the external function
	 * @param [in] nComp Number of components
	 * @param [in] nBoundStates Array with number of bound states for each component
	 */
	inline void updateTimeDerivative(active* result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		*result = extTimeDiff * (static_cast<double>(_linear) + extVal * (2.0 * static_cast<double>(_quad) +
																		  3.0 * extVal * static_cast<double>(_cube)));
	}

	/**
	 * @brief Returns the base value that does not depend on an external value
	 * @return Constant base value
	 */
	inline storage_t& base() CADET_NOEXCEPT
	{
		return _base;
	}
	inline const storage_t& base() const CADET_NOEXCEPT
	{
		return _base;
	}

	/**
	 * @brief Returns the amount of additional memory (usually dynamically allocated by containers) for storing the
	 * final parameters
	 * @details In a model, externally dependent parameters are stored in a struct, usually called
	 *          VariableParams. This is sufficient for "static" parameter types that do
	 *          not require additional memory (which is usually allocated dynamically).
	 *          For containers using additional dynamic memory, only the container itself
	 *          is stored in the struct. Memory for the content of the container (i.e., the
	 *          elements) is still required. This function computes this amount of additional
	 *          memory.
	 *
	 * @param [in] nComp Number of components
	 * @param [in] totalNumBoundStates Total number of bound states
	 * @param [in] nBoundStates Array with number of bound states for each component
	 * @return Amount of additional memory in bytes
	 */
	inline std::size_t additionalDynamicMemory(unsigned int nComp, unsigned int totalNumBoundStates,
											   unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		return 0;
	}

	/**
	 * @brief Prepares the cache for the updated values
	 * @details The cache is a local version of storage_t (e.g., LocalVector).
	 * @param [in,out] cache Cache object to be prepared
	 * @param [in] ptr Pointer to cache buffer
	 */
	template <typename T> inline void prepareCache(T& cache, LinearBufferAllocator& buffer) const
	{
	}

	/**
	 * @brief Returns the number of elements in the parameter
	 * @return Number of elements in the parameter
	 */
	inline std::size_t size() const CADET_NOEXCEPT
	{
		return 1;
	}

protected:
	active _base;
	active _linear;
	active _quad;
	active _cube;
};

/**
 * @brief Component dependent scalar external parameter
 * @details A single value per component.
 */
class ExternalScalarComponentDependentParameter
{
public:
	typedef std::vector<active> storage_t;

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		readParameterMatrix(_base, paramProvider, "EXT_" + varName, nComp, 1);
		readParameterMatrix(_linear, paramProvider, "EXT_" + varName + "_T", nComp, 1);
		readParameterMatrix(_quad, paramProvider, "EXT_" + varName + "_TT", nComp, 1);
		readParameterMatrix(_cube, paramProvider, "EXT_" + varName + "_TTT", nComp, 1);
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName), parameters, _base, unitOpIdx,
												  parTypeIdx);
		registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_T"), parameters, _linear,
												  unitOpIdx, parTypeIdx);
		registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_TT"), parameters, _quad,
												  unitOpIdx, parTypeIdx);
		registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_TTT"), parameters, _cube,
												  unitOpIdx, parTypeIdx);
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		_base.reserve(nComp);
		_linear.reserve(nComp);
		_quad.reserve(nComp);
		_cube.reserve(nComp);
	}

	inline void update(std::vector<active>& result, double extVal, unsigned int nComp,
					   unsigned int const* nBoundStates) const
	{
		update(result.data(), extVal, nComp, nBoundStates);
	}

	inline void update(active* result, double extVal, unsigned int nComp, unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] = _base[i] + extVal * (_linear[i] + extVal * (_quad[i] + extVal * _cube[i]));
	}

	inline void updateTimeDerivative(std::vector<active>& result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		updateTimeDerivative(result.data(), extVal, extTimeDiff, nComp, nBoundStates);
	}

	inline void updateTimeDerivative(active* result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] = extTimeDiff *
						(static_cast<double>(_linear[i]) +
						 extVal * (2.0 * static_cast<double>(_quad[i]) + 3.0 * extVal * static_cast<double>(_cube[i])));
	}

	inline std::size_t size() const CADET_NOEXCEPT
	{
		return _base.size();
	}
	inline bool allSameSize() const CADET_NOEXCEPT
	{
		return (_base.size() == _linear.size()) && (_base.size() == _quad.size()) && (_base.size() == _cube.size());
	}

	inline std::size_t additionalDynamicMemory(unsigned int nComp, unsigned int totalNumBoundStates,
											   unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		return nComp * sizeof(active) + alignof(active);
	}

	inline storage_t& base() CADET_NOEXCEPT
	{
		return _base;
	}
	inline const storage_t& base() const CADET_NOEXCEPT
	{
		return _base;
	}

	template <typename T> inline void prepareCache(T& cache, LinearBufferAllocator& buffer) const
	{
		cache.fromTemplate(buffer, _base);
	}

protected:
	std::vector<active> _base;
	std::vector<active> _linear;
	std::vector<active> _quad;
	std::vector<active> _cube;
};

/**
 * @brief Bound state dependent scalar external parameter
 * @details A single value per bound state. Can be used for multiple binding site types.
 */
class ExternalScalarBoundStateDependentParameter
{
public:
	typedef std::vector<active> storage_t;

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		readScalarParameterOrArray(_base, paramProvider, "EXT_" + varName, 1);
		readScalarParameterOrArray(_linear, paramProvider, "EXT_" + varName + "_T", 1);
		readScalarParameterOrArray(_quad, paramProvider, "EXT_" + varName + "_TT", 1);
		readScalarParameterOrArray(_cube, paramProvider, "EXT_" + varName + "_TTT", 1);
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		registerScalarBoundStateDependentParam(hashStringRuntime("EXT_" + varName), parameters, _base, unitOpIdx,
											   parTypeIdx);
		registerScalarBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_T"), parameters, _linear,
											   unitOpIdx, parTypeIdx);
		registerScalarBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_TT"), parameters, _quad,
											   unitOpIdx, parTypeIdx);
		registerScalarBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_TTT"), parameters, _cube,
											   unitOpIdx, parTypeIdx);
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		_base.reserve(numSlices);
		_linear.reserve(numSlices);
		_quad.reserve(numSlices);
		_cube.reserve(numSlices);
	}

	inline void update(std::vector<active>& result, double extVal, unsigned int nComp,
					   unsigned int const* nBoundStates) const
	{
		update(result.data(), extVal, nComp, nBoundStates);
	}

	inline void update(active* result, double extVal, unsigned int nComp, unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] = _base[i] + extVal * (_linear[i] + extVal * (_quad[i] + extVal * _cube[i]));
	}

	inline void updateTimeDerivative(std::vector<active>& result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		updateTimeDerivative(result.data(), extVal, extTimeDiff, nComp, nBoundStates);
	}

	inline void updateTimeDerivative(active* result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] = extTimeDiff *
						(static_cast<double>(_linear[i]) +
						 extVal * (2.0 * static_cast<double>(_quad[i]) + 3.0 * extVal * static_cast<double>(_cube[i])));
	}

	inline std::size_t size() const CADET_NOEXCEPT
	{
		return _base.size();
	}
	inline bool allSameSize() const CADET_NOEXCEPT
	{
		return (_base.size() == _linear.size()) && (_base.size() == _quad.size()) && (_base.size() == _cube.size());
	}

	inline std::size_t additionalDynamicMemory(unsigned int nComp, unsigned int totalNumBoundStates,
											   unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		for (unsigned int i = 0; i < nComp; ++i)
		{
			if (nBoundStates[i] != 0)
				return nBoundStates[i] * sizeof(active) + alignof(active);
		}
		return 0;
	}

	inline storage_t& base() CADET_NOEXCEPT
	{
		return _base;
	}
	inline const storage_t& base() const CADET_NOEXCEPT
	{
		return _base;
	}

	template <typename T> inline void prepareCache(T& cache, LinearBufferAllocator& buffer) const
	{
		cache.fromTemplate(buffer, _base);
	}

protected:
	std::vector<active> _base;
	std::vector<active> _linear;
	std::vector<active> _quad;
	std::vector<active> _cube;
};

/**
 * @brief Reaction dependent scalar external parameter
 * @details A single value per reaction.
 */
class ExternalScalarReactionDependentParameter
{
public:
	typedef std::vector<active> storage_t;

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		readParameterMatrix(_base, paramProvider, "EXT_" + varName, 1, 1);
		readParameterMatrix(_linear, paramProvider, "EXT_" + varName + "_T", 1, 1);
		readParameterMatrix(_quad, paramProvider, "EXT_" + varName + "_TT", 1, 1);
		readParameterMatrix(_cube, paramProvider, "EXT_" + varName + "_TTT", 1, 1);
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		const StringHash hashConst = hashStringRuntime("EXT_" + varName);
		registerParam1DArray(parameters, _base, [=](bool multi, unsigned int r) {
			return makeParamId(hashConst, unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep, r, SectionIndep);
		});

		const StringHash hashLinear = hashStringRuntime("EXT_" + varName + "_T");
		registerParam1DArray(parameters, _linear, [=](bool multi, unsigned int r) {
			return makeParamId(hashLinear, unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep, r, SectionIndep);
		});

		const StringHash hashQuad = hashStringRuntime("EXT_" + varName + "_TT");
		registerParam1DArray(parameters, _quad, [=](bool multi, unsigned int r) {
			return makeParamId(hashQuad, unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep, r, SectionIndep);
		});

		const StringHash hashCube = hashStringRuntime("EXT_" + varName + "_TTT");
		registerParam1DArray(parameters, _cube, [=](bool multi, unsigned int r) {
			return makeParamId(hashCube, unitOpIdx, CompIndep, parTypeIdx, BoundStateIndep, r, SectionIndep);
		});
	}

	inline void reserve(unsigned int nReactions, unsigned int nComp, unsigned int nBoundStates)
	{
		_base.reserve(nReactions);
		_linear.reserve(nReactions);
		_quad.reserve(nReactions);
		_cube.reserve(nReactions);
	}

	inline void update(std::vector<active>& result, double extVal, unsigned int nComp,
					   unsigned int const* nBoundStates) const
	{
		update(result.data(), extVal, nComp, nBoundStates);
	}

	inline void update(active* result, double extVal, unsigned int nComp, unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] = _base[i] + extVal * (_linear[i] + extVal * (_quad[i] + extVal * _cube[i]));
	}

	inline void updateTimeDerivative(std::vector<active>& result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		updateTimeDerivative(result.data(), extVal, extTimeDiff, nComp, nBoundStates);
	}

	inline void updateTimeDerivative(active* result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] = extTimeDiff *
						(static_cast<double>(_linear[i]) +
						 extVal * (2.0 * static_cast<double>(_quad[i]) + 3.0 * extVal * static_cast<double>(_cube[i])));
	}

	inline std::size_t size() const CADET_NOEXCEPT
	{
		return _base.size();
	}
	inline bool allSameSize() const CADET_NOEXCEPT
	{
		return (_base.size() == _linear.size()) && (_base.size() == _quad.size()) && (_base.size() == _cube.size());
	}

	inline std::size_t additionalDynamicMemory(unsigned int nReactions, unsigned int nComp,
											   unsigned int totalNumBoundStates) const CADET_NOEXCEPT
	{
		return nReactions * sizeof(active) + alignof(active);
	}

	inline storage_t& base() CADET_NOEXCEPT
	{
		return _base;
	}
	inline const storage_t& base() const CADET_NOEXCEPT
	{
		return _base;
	}

	template <typename T> inline void prepareCache(T& cache, LinearBufferAllocator& buffer) const
	{
		cache.fromTemplate(buffer, _base);
	}

protected:
	std::vector<active> _base;
	std::vector<active> _linear;
	std::vector<active> _quad;
	std::vector<active> _cube;
};

/**
 * @brief Component and bound state dependent external parameter
 * @details A single value per component and bound state / binding site type.
 * @tparam compMajor Determines whether the values are stored in component-major or bound-state-/binding-site-type-major
 * ordering
 */
template <bool compMajor> class ExternalBaseComponentBoundStateDependentParameter
{
public:
	typedef util::SlicedVector<active> storage_t;

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		if (compMajor)
		{
			readBoundStateDependentParameter<util::SlicedVector<active>, active>(_base, paramProvider, "EXT_" + varName,
																				 nComp, nBoundStates);
			readBoundStateDependentParameter<util::SlicedVector<active>, active>(
				_linear, paramProvider, "EXT_" + varName + "_T", nComp, nBoundStates);
			readBoundStateDependentParameter<util::SlicedVector<active>, active>(
				_quad, paramProvider, "EXT_" + varName + "_TT", nComp, nBoundStates);
			readBoundStateDependentParameter<util::SlicedVector<active>, active>(
				_cube, paramProvider, "EXT_" + varName + "_TTT", nComp, nBoundStates);
		}
		else
		{
			const unsigned int numStates = firstNonEmptyBoundStates(nBoundStates, nComp);
			readBoundStateDependentParameter<util::SlicedVector<active>, active>(_base, paramProvider, "EXT_" + varName,
																				 nComp, numStates);
			readBoundStateDependentParameter<util::SlicedVector<active>, active>(
				_linear, paramProvider, "EXT_" + varName + "_T", nComp, numStates);
			readBoundStateDependentParameter<util::SlicedVector<active>, active>(
				_quad, paramProvider, "EXT_" + varName + "_TT", nComp, numStates);
			readBoundStateDependentParameter<util::SlicedVector<active>, active>(
				_cube, paramProvider, "EXT_" + varName + "_TTT", nComp, numStates);
		}
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		if (compMajor)
		{
			registerComponentBoundStateDependentParamCompMajor(hashStringRuntime("EXT_" + varName), parameters, _base,
															   unitOpIdx, parTypeIdx);
			registerComponentBoundStateDependentParamCompMajor(hashStringRuntime("EXT_" + varName + "_T"), parameters,
															   _linear, unitOpIdx, parTypeIdx);
			registerComponentBoundStateDependentParamCompMajor(hashStringRuntime("EXT_" + varName + "_TT"), parameters,
															   _quad, unitOpIdx, parTypeIdx);
			registerComponentBoundStateDependentParamCompMajor(hashStringRuntime("EXT_" + varName + "_TTT"), parameters,
															   _cube, unitOpIdx, parTypeIdx);
		}
		else
		{
			registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName), parameters, _base, unitOpIdx,
													  parTypeIdx);
			registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_T"), parameters, _linear,
													  unitOpIdx, parTypeIdx);
			registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_TT"), parameters, _quad,
													  unitOpIdx, parTypeIdx);
			registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_TTT"), parameters, _cube,
													  unitOpIdx, parTypeIdx);
		}
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		_base.reserve(numElem, numSlices);
		_linear.reserve(numElem, numSlices);
		_quad.reserve(numElem, numSlices);
		_cube.reserve(numElem, numSlices);
	}

	inline void update(util::SlicedVector<active>& result, double extVal, unsigned int nComp,
					   unsigned int const* nBoundStates) const
	{
		update(result.data(), extVal, nComp, nBoundStates);
	}

	inline void update(active* result, double extVal, unsigned int nComp, unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] =
				_base.native(i) + extVal * (_linear.native(i) + extVal * (_quad.native(i) + extVal * _cube.native(i)));
	}

	inline void updateTimeDerivative(util::SlicedVector<active>& result, double extVal, double extTimeDiff,
									 unsigned int nComp, unsigned int const* nBoundStates) const
	{
		updateTimeDerivative(result.data(), extVal, extTimeDiff, nComp, nBoundStates);
	}

	inline void updateTimeDerivative(active* result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] = extTimeDiff * (static_cast<double>(_linear.native(i)) +
									   extVal * (2.0 * static_cast<double>(_quad.native(i)) +
												 3.0 * extVal * static_cast<double>(_cube.native(i))));
	}

	inline typename util::SlicedVector<active>::size_type slices() const CADET_NOEXCEPT
	{
		return _base.slices();
	}
	inline typename util::SlicedVector<active>::size_type size() const CADET_NOEXCEPT
	{
		return _base.size();
	}
	inline bool allSameSize() const CADET_NOEXCEPT
	{
		return (_base.size() == _linear.size()) && (_base.size() == _quad.size()) && (_base.size() == _cube.size());
	}

	inline std::size_t additionalDynamicMemory(unsigned int nComp, unsigned int totalNumBoundStates,
											   unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		if (compMajor)
			return totalNumBoundStates * sizeof(active) + alignof(active) +
				   (nComp + 1) * sizeof(typename util::SlicedVector<active>::size_type) +
				   alignof(typename util::SlicedVector<active>::size_type);
		else
		{
			const unsigned int numStates = firstNonEmptyBoundStates(nBoundStates, nComp);
			return numStates * nComp * sizeof(active) + alignof(active) +
				   (numStates + 1) * sizeof(typename util::SlicedVector<active>::size_type) +
				   alignof(typename util::SlicedVector<active>::size_type);
		}
	}

	inline storage_t& base() CADET_NOEXCEPT
	{
		return _base;
	}
	inline const storage_t& base() const CADET_NOEXCEPT
	{
		return _base;
	}

	template <typename T> inline void prepareCache(T& cache, LinearBufferAllocator& buffer) const
	{
		cache.fromTemplate(buffer, _base);
	}

protected:
	util::SlicedVector<active> _base;
	util::SlicedVector<active> _linear;
	util::SlicedVector<active> _quad;
	util::SlicedVector<active> _cube;
};

/**
 * @brief Component and bound state dependent external parameter
 * @details Components contain values for each bound state.
 */
typedef ExternalBaseComponentBoundStateDependentParameter<true> ExternalComponentMajorBoundStateDependentParameter;

/**
 * @brief Component and bound state dependent external parameter
 * @details Binding site types contain values for each component.
 */
typedef ExternalBaseComponentBoundStateDependentParameter<false> ExternalComponentBoundStateMajorDependentParameter;

/**
 * @brief Component dependent bound-state matrix valued external parameter
 * @details Holds a square matrix for each component that has the size of the number of corresponding bound states.
 */
class ExternalComponentDependentBoundStateMatrixParameter
{
public:
	typedef util::SlicedVector<active> storage_t;

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		readMatrixValuedBoundStateDependentParameter<util::SlicedVector<active>, active>(
			_base, paramProvider, "EXT_" + varName, nComp, nBoundStates);
		readMatrixValuedBoundStateDependentParameter<util::SlicedVector<active>, active>(
			_linear, paramProvider, "EXT_" + varName + "_T", nComp, nBoundStates);
		readMatrixValuedBoundStateDependentParameter<util::SlicedVector<active>, active>(
			_quad, paramProvider, "EXT_" + varName + "_TT", nComp, nBoundStates);
		readMatrixValuedBoundStateDependentParameter<util::SlicedVector<active>, active>(
			_cube, paramProvider, "EXT_" + varName + "_TTT", nComp, nBoundStates);
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		registerComponentBoundStateDependentParamCompMajor(hashStringRuntime("EXT_" + varName), parameters, _base,
														   unitOpIdx, parTypeIdx);
		registerComponentBoundStateDependentParamCompMajor(hashStringRuntime("EXT_" + varName + "_T"), parameters,
														   _linear, unitOpIdx, parTypeIdx);
		registerComponentBoundStateDependentParamCompMajor(hashStringRuntime("EXT_" + varName + "_TT"), parameters,
														   _quad, unitOpIdx, parTypeIdx);
		registerComponentBoundStateDependentParamCompMajor(hashStringRuntime("EXT_" + varName + "_TTT"), parameters,
														   _cube, unitOpIdx, parTypeIdx);
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		unsigned int sumSquared = 0;
		for (unsigned int i = 0; i < nComp; ++i)
			sumSquared += nBoundStates[i] * nBoundStates[i];

		_base.reserve(sumSquared, nComp);
		_linear.reserve(sumSquared, nComp);
		_quad.reserve(sumSquared, nComp);
		_cube.reserve(sumSquared, nComp);
	}

	inline void update(util::SlicedVector<active>& result, double extVal, unsigned int nComp,
					   unsigned int const* nBoundStates) const
	{
		update(result.data(), extVal, nComp, nBoundStates);
	}

	inline void update(active* result, double extVal, unsigned int nComp, unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] =
				_base.native(i) + extVal * (_linear.native(i) + extVal * (_quad.native(i) + extVal * _cube.native(i)));
	}

	inline void updateTimeDerivative(util::SlicedVector<active>& result, double extVal, double extTimeDiff,
									 unsigned int nComp, unsigned int const* nBoundStates) const
	{
		updateTimeDerivative(result.data(), extVal, extTimeDiff, nComp, nBoundStates);
	}

	inline void updateTimeDerivative(active* result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] = extTimeDiff * (static_cast<double>(_linear.native(i)) +
									   extVal * (2.0 * static_cast<double>(_quad.native(i)) +
												 3.0 * extVal * static_cast<double>(_cube.native(i))));
	}

	inline typename util::SlicedVector<active>::size_type slices() const CADET_NOEXCEPT
	{
		return _base.slices();
	}
	inline typename util::SlicedVector<active>::size_type size() const CADET_NOEXCEPT
	{
		return _base.size();
	}
	inline bool allSameSize() const CADET_NOEXCEPT
	{
		return (_base.size() == _linear.size()) && (_base.size() == _quad.size()) && (_base.size() == _cube.size());
	}

	inline std::size_t additionalDynamicMemory(unsigned int nComp, unsigned int totalNumBoundStates,
											   unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		unsigned int sumSquared = 0;
		for (unsigned int i = 0; i < nComp; ++i)
			sumSquared += nBoundStates[i] * nBoundStates[i];

		return sumSquared * sizeof(active) + alignof(active) +
			   (nComp + 1) * sizeof(typename util::SlicedVector<active>::size_type) +
			   alignof(typename util::SlicedVector<active>::size_type);
	}

	inline storage_t& base() CADET_NOEXCEPT
	{
		return _base;
	}
	inline const storage_t& base() const CADET_NOEXCEPT
	{
		return _base;
	}

	template <typename T> inline void prepareCache(T& cache, LinearBufferAllocator& buffer) const
	{
		cache.fromTemplate(buffer, _base);
	}

protected:
	util::SlicedVector<active> _base;
	util::SlicedVector<active> _linear;
	util::SlicedVector<active> _quad;
	util::SlicedVector<active> _cube;
};

/**
 * @brief Component dependent component vector valued external parameter
 * @details Holds a vector for each component that has the size of the number of components.
 */
class ExternalComponentDependentComponentVectorParameter
{
public:
	typedef util::SlicedVector<active> storage_t;

	inline void configure(const std::string& varName, IParameterProvider& paramProvider, unsigned int nComp,
						  unsigned int const* nBoundStates)
	{
		readBoundStateDependentParameter<util::SlicedVector<active>, active>(_base, paramProvider, "EXT_" + varName,
																			 nComp, nComp);
		readBoundStateDependentParameter<util::SlicedVector<active>, active>(_linear, paramProvider,
																			 "EXT_" + varName + "_T", nComp, nComp);
		readBoundStateDependentParameter<util::SlicedVector<active>, active>(_quad, paramProvider,
																			 "EXT_" + varName + "_TT", nComp, nComp);
		readBoundStateDependentParameter<util::SlicedVector<active>, active>(_cube, paramProvider,
																			 "EXT_" + varName + "_TTT", nComp, nComp);
	}

	inline void registerParam(const std::string& varName, std::unordered_map<ParameterId, active*>& parameters,
							  UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp,
							  unsigned int const* nBoundStates)
	{
		registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName), parameters, _base, unitOpIdx,
												  parTypeIdx);
		registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_T"), parameters, _linear,
												  unitOpIdx, parTypeIdx);
		registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_TT"), parameters, _quad,
												  unitOpIdx, parTypeIdx);
		registerComponentBoundStateDependentParam(hashStringRuntime("EXT_" + varName + "_TTT"), parameters, _cube,
												  unitOpIdx, parTypeIdx);
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp,
						unsigned int const* nBoundStates)
	{
		const unsigned int compSquared = nComp * nComp;

		_base.reserve(compSquared, nComp);
		_linear.reserve(compSquared, nComp);
		_quad.reserve(compSquared, nComp);
		_cube.reserve(compSquared, nComp);
	}

	inline void update(util::SlicedVector<active>& result, double extVal, unsigned int nComp,
					   unsigned int const* nBoundStates) const
	{
		update(result.data(), extVal, nComp, nBoundStates);
	}

	inline void update(active* result, double extVal, unsigned int nComp, unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] =
				_base.native(i) + extVal * (_linear.native(i) + extVal * (_quad.native(i) + extVal * _cube.native(i)));
	}

	inline void updateTimeDerivative(util::SlicedVector<active>& result, double extVal, double extTimeDiff,
									 unsigned int nComp, unsigned int const* nBoundStates) const
	{
		updateTimeDerivative(result.data(), extVal, extTimeDiff, nComp, nBoundStates);
	}

	inline void updateTimeDerivative(active* result, double extVal, double extTimeDiff, unsigned int nComp,
									 unsigned int const* nBoundStates) const
	{
		for (std::size_t i = 0; i < _base.size(); ++i)
			result[i] = extTimeDiff * (static_cast<double>(_linear.native(i)) +
									   extVal * (2.0 * static_cast<double>(_quad.native(i)) +
												 3.0 * extVal * static_cast<double>(_cube.native(i))));
	}

	inline typename util::SlicedVector<active>::size_type slices() const CADET_NOEXCEPT
	{
		return _base.slices();
	}
	inline typename util::SlicedVector<active>::size_type size() const CADET_NOEXCEPT
	{
		return _base.size();
	}
	inline bool allSameSize() const CADET_NOEXCEPT
	{
		return (_base.size() == _linear.size()) && (_base.size() == _quad.size()) && (_base.size() == _cube.size());
	}

	inline std::size_t additionalDynamicMemory(unsigned int nComp, unsigned int totalNumBoundStates,
											   unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		return nComp * nComp * sizeof(active) + alignof(active) +
			   (nComp + 1) * sizeof(typename util::SlicedVector<active>::size_type) +
			   alignof(typename util::SlicedVector<active>::size_type);
	}

	inline storage_t& base() CADET_NOEXCEPT
	{
		return _base;
	}
	inline const storage_t& base() const CADET_NOEXCEPT
	{
		return _base;
	}

	template <typename T> inline void prepareCache(T& cache, LinearBufferAllocator& buffer) const
	{
		cache.fromTemplate(buffer, _base);
	}

protected:
	util::SlicedVector<active> _base;
	util::SlicedVector<active> _linear;
	util::SlicedVector<active> _quad;
	util::SlicedVector<active> _cube;
};

} // namespace model

} // namespace cadet

#endif // LIBCADET_PARAMETERS_HPP_
