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
 * Provides helper functions for ParameterId containers.
 */

#ifndef LIBCADET_PARAMIDUTIL_HPP_
#define LIBCADET_PARAMIDUTIL_HPP_

#include <ostream>
#include <sstream>
#include "cadet/ParameterId.hpp"

namespace std
{
template <> struct hash<cadet::ParameterId>
{
	inline std::size_t operator()(const cadet::ParameterId& pId) const CADET_NOEXCEPT
	{
		return cadet::hashParameter(pId);
	}
};
} // namespace std

namespace cadet
{

inline bool operator==(const ParameterId& a, const ParameterId& b) CADET_NOEXCEPT
{
	return (a.name == b.name) && (a.unitOperation == b.unitOperation) && (a.component == b.component) &&
		   (a.particleType == b.particleType) && (a.boundState == b.boundState) && (a.reaction == b.reaction) &&
		   (a.section == b.section);
}

inline bool operator!=(const ParameterId& a, const ParameterId& b) CADET_NOEXCEPT
{
	return !(a == b);
}

inline bool operator<(const ParameterId& a, const ParameterId& b) CADET_NOEXCEPT
{
	return std::tie(a.name, a.unitOperation, a.component, a.particleType, a.boundState, a.reaction, a.section) <
		   std::tie(b.name, b.unitOperation, b.component, b.particleType, b.boundState, b.reaction, b.section);
}

inline std::ostream& operator<<(std::ostream& out, const ParameterId& pId)
{
	out << "{" << hashParameter(pId) << " = " << pId.name << ", Unit " << static_cast<unsigned int>(pId.unitOperation)
		<< " Comp " << static_cast<unsigned int>(pId.component) << " ParticleType "
		<< static_cast<unsigned int>(pId.particleType) << " BoundState " << static_cast<unsigned int>(pId.boundState)
		<< " Reaction " << static_cast<unsigned int>(pId.reaction) << " Section "
		<< static_cast<unsigned int>(pId.section) << "}";
	return out;
}

inline std::string to_string(const ParameterId& pId)
{
	std::stringstream out;
	out << "{" << hashParameter(pId) << " = " << pId.name << ", Unit " << static_cast<unsigned int>(pId.unitOperation)
		<< " Comp " << static_cast<unsigned int>(pId.component) << " ParticleType "
		<< static_cast<unsigned int>(pId.particleType) << " BoundState " << static_cast<unsigned int>(pId.boundState)
		<< " Reaction " << static_cast<unsigned int>(pId.reaction) << " Section "
		<< static_cast<unsigned int>(pId.section) << "}";
	return out.str();
}

} // namespace cadet

#endif // LIBCADET_PARAMIDUTIL_HPP_
