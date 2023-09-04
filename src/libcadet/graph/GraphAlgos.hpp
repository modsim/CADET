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
 * Provides algorithms for graphs
 */

#ifndef LIBCADET_GRAPHALGOS_HPP_
#define LIBCADET_GRAPHALGOS_HPP_

#include <vector>
#include "SlicedVector.hpp"

namespace cadet
{

namespace graph
{

/**
 * @brief      Converts connection list to adjacency list
 *
 * @param      conList  Connection list (6 items per line)
 * @param[in]  nUnits   Number of unit operations
 * @param[in]  nCon     Number of connections in @p conList
 *
 * @return     Adjacency list for each unit operation
 */
cadet::util::SlicedVector<int> adjacencyListFromConnectionList(int const* conList, int nUnits, int nCon);

/**
 * @brief      Performs a topological sort of the given directed graph
 * @details    Topological sorting finds an ordering of the nodes (unit operations)
 *             such that all dependencies (inputs) of a node are listed before the
 *             node itself is listed.
 *
 *             In addition, cycles in the graph are detected.
 *
 *             Based on Cormen et al., Introduction to Algorithms (3rd ed.), Sec. 22.4
 *
 * @param[in]  adjList    List of adjacent nodes for each node, see adjacencyListFromConnectionList()
 * @param[out] topoOrder  Reverse topological order (last item has to be processed first)
 *
 * @return     @c true if the graph contains cycles, @c false otherwise
 */
bool topologicalSort(const cadet::util::SlicedVector<int>& adjList, std::vector<int>& topoOrder);

} // namespace graph

} // namespace cadet

#endif // LIBCADET_GRAPHALGOS_HPP_
