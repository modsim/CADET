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

#include "graph/GraphAlgos.hpp"

namespace cadet
{

namespace graph
{

cadet::util::SlicedVector<int> adjacencyListFromConnectionList(int const* conList, int nUnits, int nCon)
{
	cadet::util::SlicedVector<int> adj;
	adj.reserve(nCon, nUnits);

	std::vector<int> adjNode;
	adjNode.reserve(nUnits);

	// TODO: Improve runtime

	for (int i = 0; i < nUnits; ++i)
	{
		adjNode.clear();
		for (int j = 0; j < nCon; ++j)
		{
			const int from = conList[6 * j];
			if (from != i)
				continue;

			const int to = conList[6 * j + 1];

			bool found = false;
			for (int k = 0; k < static_cast<int>(adjNode.size()); ++k)
			{
				if (adjNode[k] == to)
				{
					found = true;
					break;
				}
			}

			if (!found)
				adjNode.push_back(to);
		}

		adj.pushBackSlice(adjNode);
	}

	return adj;
}

namespace detail
{

bool topologicalSortHelper(const cadet::util::SlicedVector<int>& adjList, int u, std::vector<char>& colors,
						   std::vector<int>& topoOrder)
{
	// Set color to gray (1)
	colors[u] = 1;

	// Iterate over adjacent nodes
	int const* const adj = adjList[u];
	const int nAdj = adjList.sliceSize(u);
	for (int n = 0; n < nAdj; ++n)
	{
		const int nu = adj[n];
		if (colors[nu] == 0)
		{
			// Depth-first traversal
			// Escalate cycles to caller
			if (topologicalSortHelper(adjList, nu, colors, topoOrder))
				return true;
		}
		else if (colors[nu] == 1)
		{
			// Detected cycle
			return true;
		}
	}

	// Set color to black (2)
	colors[u] = 2;

	// Append node to topological ordering (reverse)
	topoOrder.push_back(u);

	// No cycle so far
	return false;
}

} // namespace detail

bool topologicalSort(const cadet::util::SlicedVector<int>& adjList, std::vector<int>& topoOrder)
{
	const int nUnits = adjList.slices();
	topoOrder.clear();
	topoOrder.reserve(nUnits);

	// Set color of each node to white (0)
	std::vector<char> colors(nUnits, 0);

	for (int u = 0; u < nUnits; ++u)
	{
		// Visit node if color is white (0)
		if (colors[u] != 0)
			continue;

		// Explore adjacent nodes and detect cycles
		if (detail::topologicalSortHelper(adjList, u, colors, topoOrder))
			return true;
	}

	return false;
}

} // namespace graph

} // namespace cadet
