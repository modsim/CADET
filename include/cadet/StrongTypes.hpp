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
 * Defines strong types for interfaces.
 */

#ifndef LIBCADET_STRONGTYPES_HPP_
#define LIBCADET_STRONGTYPES_HPP_

#include "cadet/cadetCompilerInfo.hpp"

namespace cadet
{

namespace model
{

struct ComponentIndex
{
	unsigned int value;
};

struct AxialCellIndex
{
	unsigned int value;
};

struct ParticleTypeIndex
{
	unsigned int value;
};

struct ParticleIndex
{
	unsigned int value;
};

struct ShellIndex
{
	unsigned int value;
};

} // namespace model
} // namespace cadet

#endif // LIBCADET_STRONGTYPES_HPP_
