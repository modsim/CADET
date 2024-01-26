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
 * Defines the ParameterId and provides compile time and runtime hash functions for parameters.
 */

#ifndef LIBCADET_PARAMETERID_HPP_
#define LIBCADET_PARAMETERID_HPP_

#include <cstdint>
#include <string>

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"
#include "cadet/StringUtil.hpp"
#include "cadet/HashUtil.hpp"

namespace cadet
{

typedef uint16_t UnitOpIdx;
typedef uint8_t ComponentIdx;
typedef uint8_t ParticleTypeIdx;
typedef uint8_t BoundStateIdx;
typedef uint8_t ReactionIdx;
typedef uint16_t SectionIdx;

const UnitOpIdx UnitOpIndep = static_cast<UnitOpIdx>(-1);
const ComponentIdx CompIndep = static_cast<ComponentIdx>(-1);
const ParticleTypeIdx ParTypeIndep = static_cast<ParticleTypeIdx>(-1);
const BoundStateIdx BoundStateIndep = static_cast<BoundStateIdx>(-1);
const ReactionIdx ReactionIndep = static_cast<ReactionIdx>(-1);
const SectionIdx SectionIndep = static_cast<SectionIdx>(-1);

/**
 * @brief Uniquely identifies a parameter
 */
struct ParameterId
{
	/** @brief Hashed name of the parameter **/
	StringHash name;
	/** @brief Unit operation index or -1 if independent of unit operation (e.g., section times) **/
	UnitOpIdx unitOperation;
	/** @brief Component index or -1 if independent of component (e.g., reaction rate) **/
	ComponentIdx component;
	/** @brief Particle type index or -1 if independent of particle type (e.g., column porosity)  **/
	ParticleTypeIdx particleType;
	/** @brief Bound state index or -1 if independent of bound state (e.g., reaction rate) **/
	BoundStateIdx boundState;
	/** @brief Reaction index or -1 if independent of reaction (e.g., transport and binding parameters) **/
	ReactionIdx reaction;
	/** @brief Section index or -1 if independent of section (e.g., porosity) **/
	SectionIdx section;
};

/** @brief Type that holds a hash of ParameterId and identifies a parameter uniquely **/
typedef uint64_t ParamIdHash;

#if CADET_COMPILER_CXX_CONSTEXPR

/**
 * @brief Computes the string hash at compiletime
 * @details Use the hashStringRuntime for runtime hashing.
 *
 * @param [in] str String
 * @return Hash of the string
 */
constexpr inline StringHash hashString(const util::hash::ConstString& cs)
{
	return util::SipHash24(cs);
}

#else

/**
 * @brief Computes the string hash at runtime
 * @details Use the _hash string literal for compile time hashing.
 *
 * @param [in] str String
 * @return Hash of the string
 */
inline StringHash hashString(const std::string& str)
{
	return util::SipHash24runtime(str);
}

/**
 * @brief Computes the string hash at runtime
 * @details Use the _hash string literal for compile time hashing.
 *
 * @param [in] str String
 * @return Hash of the string
 */
inline StringHash hashString(char const* const str)
{
	return util::SipHash24runtime(str);
}

#endif

/**
 * @brief Computes the string hash at runtime
 * @details Use the _hash string literal for compile time hashing.
 *
 * @param [in] str String
 * @return Hash of the string
 */
inline StringHash hashStringRuntime(const std::string& str)
{
	return util::SipHash24runtime(str);
}

/**
 * @brief Computes the string hash at runtime
 * @details Use the _hash string literal for compile time hashing.
 *
 * @param [in] str String
 * @return Hash of the string
 */
inline StringHash hashStringRuntime(char const* const str)
{
	return util::SipHash24runtime(str);
}

/**
 * @brief Computes the unique parameter Id hash for a given parameter
 * @details This hash function can be used at compile- and runtime.
 *
 * @param [in] nameHash Hash of the parameter name computed by hashString()
 * @param [in] unitOperation Index of the unit operation this parameter belongs to
 * @param [in] component Index of the component this parameter belongs to
 * @param [in] boundState Index of the bound state this parameter belongs to
 * @param [in] reaction Index of the reaction this parameter belongs to
 * @param [in] section Index of the section this parameter belongs to
 * @return Unique parameter id hash
 */
inline ParamIdHash hashParameter(const StringHash nameHash, const UnitOpIdx unitOperation, const ComponentIdx component,
								 const ParticleTypeIdx parType, const BoundStateIdx boundState,
								 const ReactionIdx reaction, const SectionIdx section)
{
	ParamIdHash hash = nameHash;

	// Combine indices in one big integer by shifting them
	// Note that ParamIdHash has to be big enough to hold all indices
	const ParamIdHash combinedIdx = pack_ints(unitOperation, component, parType, boundState, reaction, section);
	hash_combine(hash, combinedIdx);

	return hash;
}

/**
 * @brief Computes the unique parameter Id hash for a given parameter
 * @details This hash function can be used at compile- and runtime.
 *
 * @param [in] name Name of the parameter
 * @param [in] unitOperation Index of the unit operation this parameter belongs to
 * @param [in] component Index of the component this parameter belongs to
 * @param [in] boundState Index of the bound state this parameter belongs to
 * @param [in] reaction Index of the reaction this parameter belongs to
 * @param [in] section Index of the section this parameter belongs to
 * @return Unique parameter id hash
 */
inline ParamIdHash hashParameter(const std::string& name, const UnitOpIdx unitOperation, const ComponentIdx component,
								 const ParticleTypeIdx parType, const BoundStateIdx boundState,
								 const ReactionIdx reaction, const SectionIdx section)
{
	return hashParameter(hashStringRuntime(name), unitOperation, component, parType, boundState, reaction, section);
}

/**
 * @brief Computes the unique parameter Id hash for a given parameter
 * @details This hash function is supposed to be used at runtime only.
 *
 * @param [in] param Parameter Id
 * @return Unique parameter id hash
 */
inline ParamIdHash hashParameter(const ParameterId& param)
{
	return hashParameter(param.name, param.unitOperation, param.component, param.particleType, param.boundState,
						 param.reaction, param.section);
}

/**
 * @brief Builds the unique ParameterId for a given parameter
 *
 * @param [in] name Name of the parameter
 * @param [in] unitOperation Index of the unit operation this parameter belongs to
 * @param [in] component Index of the component this parameter belongs to
 * @param [in] boundState Index of the bound state this parameter belongs to
 * @param [in] reaction Index of the reaction this parameter belongs to
 * @param [in] section Index of the section this parameter belongs to
 * @return Unique parameter id
 */
inline ParameterId makeParamId(const std::string& name, const UnitOpIdx unitOperation, const ComponentIdx component,
							   const ParticleTypeIdx parType, const BoundStateIdx boundState,
							   const ReactionIdx reaction, const SectionIdx section)
{
	return ParameterId{hashStringRuntime(name), unitOperation, component, parType, boundState, reaction, section};
}

/**
 * @brief Builds the unique ParameterId for a given parameter
 *
 * @param [in] name Hash of the parameter name
 * @param [in] unitOperation Index of the unit operation this parameter belongs to
 * @param [in] component Index of the component this parameter belongs to
 * @param [in] boundState Index of the bound state this parameter belongs to
 * @param [in] reaction Index of the reaction this parameter belongs to
 * @param [in] section Index of the section this parameter belongs to
 * @return Unique parameter id
 */
inline ParameterId makeParamId(const StringHash name, const UnitOpIdx unitOperation, const ComponentIdx component,
							   const ParticleTypeIdx parType, const BoundStateIdx boundState,
							   const ReactionIdx reaction, const SectionIdx section)
{
	return ParameterId{name, unitOperation, component, parType, boundState, reaction, section};
}

} // namespace cadet

#endif // LIBCADET_PARAMETERID_HPP_
