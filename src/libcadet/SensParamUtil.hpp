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
 * Provides helper functions for entities that can contain sensitive parameters.
 */

#ifndef LIBCADET_SENSPARAMUTIL_HPP_
#define LIBCADET_SENSPARAMUTIL_HPP_

#include <vector>
#include <unordered_set>

namespace cadet
{
template <class Elem_t> inline bool contains(const typename std::vector<Elem_t>& vec, const Elem_t& item)
{
	const typename std::vector<Elem_t>::const_iterator it = std::find(vec.begin(), vec.end(), item);
	return it != vec.end();
}

template <class Elem_t> inline bool contains(const typename std::unordered_set<Elem_t>& set, const Elem_t& item)
{
	return set.find(item) != set.end();
}
} // namespace cadet

#endif // LIBCADET_SENSPARAMUTIL_HPP_
