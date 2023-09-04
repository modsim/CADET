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

#include <catch.hpp>

#include "graph/GraphAlgos.hpp"

#include <algorithm>

namespace
{
bool dependsOn(const int nUnits, const std::vector<int>& connections, const int unit, const int unitDep)
{
	// Checks whether unit depends on unitDep
	for (std::size_t i = 0; i < connections.size() / 6; ++i)
	{
		if ((connections[i * 6] == unitDep) && (connections[i * 6 + 1] == unit))
			return true;
	}
	return false;
}

void checkTopoOrdering(const int nUnits, const std::vector<int>& connections, const std::vector<int>& topoOrder)
{
	CHECK(nUnits == topoOrder.size());

	// Check if each unit is in topoOrder
	for (int i = 0; i < nUnits; ++i)
	{
		CHECK(std::find(topoOrder.begin(), topoOrder.end(), i) != topoOrder.end());
	}

	// Check order
	for (int i = topoOrder.size() - 1; i >= 0; --i)
	{
		const int u = topoOrder[i];

		// Unit u must not depend on all units before it in topoOrder
		for (int j = 0; j < i; ++j)
		{
			CHECK(!dependsOn(nUnits, connections, u, topoOrder[j]));
		}
	}
}

bool contains(const int* data, int size, int val)
{
	for (int i = 0; i < size; ++i)
	{
		if (data[i] == val)
			return true;
	}

	return false;
}

void checkAdjacencyList(const int nUnits, const std::vector<int>& connections,
						const cadet::util::SlicedVector<int>& adjList)
{
	REQUIRE(adjList.slices() == nUnits);

	for (int i = 0; i < nUnits; ++i)
	{
		for (int j = 0; j < nUnits; ++j)
		{
			const int s = adjList.sliceSize(i);
			int const* const list = adjList[i];

			// Check connection from i to j
			if (dependsOn(nUnits, connections, j, i))
			{
				// i connects to j, so list for i should contain j
				CHECK(contains(list, s, j));
			}
			else
			{
				// i does not connect to j, so list for i should not contain j
				CHECK(!contains(list, s, j));
			}
		}
	}
}
} // namespace

TEST_CASE("Linear graph all ports all comps no cycles", "[Graph]")
{
	const int nUnits = 4;
	const std::vector<int> connections = {0, 1, -1, -1, -1, -1, 1, 2, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("Linear graph specific ports all comps no cycles", "[Graph]")
{
	const int nUnits = 4;
	const std::vector<int> connections = {0, 1, 0, 0, -1, -1, 1, 2, 0, 0, -1, -1, 2, 3, 0, 0, -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("Linear graph all ports specific comps no cycles", "[Graph]")
{
	const int nUnits = 4;
	const std::vector<int> connections = {0, 1, -1, -1, 0, 0, 0, 1, -1, -1, 1, 1, 0, 1, -1, -1, 2, 2,
										  1, 2, -1, -1, 0, 0, 1, 2, -1, -1, 1, 1, 1, 2, -1, -1, 2, 2,
										  2, 3, -1, -1, 0, 0, 2, 3, -1, -1, 1, 1, 2, 3, -1, -1, 2, 2};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("Linear graph specific ports and comps no cycles", "[Graph]")
{
	const int nUnits = 4;
	const std::vector<int> connections = {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2,
										  1, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2,
										  2, 3, 0, 0, 0, 0, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 2, 2};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("Two ports linear graph specific ports all comps no cycles", "[Graph]")
{
	/*
			 1     1     1
		  /-->--O-->--O-->--\
	2 -->O                   O--> 2
		  \-->--O-->--O-->--/
			 1     1     1
	*/

	const int nUnits = 4;
	const std::vector<int> connections = {0, 1, 0, 0, -1, -1, 0, 1, 0, 1, -1, -1, 1, 2, 0, 1, -1, -1,
										  1, 2, 1, 0, -1, -1, 2, 3, 0, 0, -1, -1, 2, 3, 1, 0, -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("Two ports linear graph specific ports and comps no cycles", "[Graph]")
{
	/*
			 1     1     1
		  /-->--O-->--O-->--\
	2 -->O                   O--> 2
		  \-->--O-->--O-->--/
			 1     1     1
	*/

	const int nUnits = 4;
	const std::vector<int> connections = {
		0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 2, 2,
		1, 2, 0, 1, 0, 0, 1, 2, 0, 1, 1, 1, 1, 2, 0, 1, 2, 2, 1, 2, 1, 0, 0, 0, 1, 2, 1, 0, 1, 1, 1, 2, 1, 0, 2, 2,
		2, 3, 0, 0, 0, 0, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 2, 2, 2, 3, 1, 0, 0, 0, 2, 3, 1, 0, 1, 1, 2, 3, 1, 0, 2, 2};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("N to 1 to N graph specific ports all comps no cycles", "[Graph]")
{
	/*
		O--\     /--O
			--O--
		O--/     \--O
	*/

	const int nUnits = 5;
	const std::vector<int> connections = {0, 2, 0, 0, -1, -1, 1, 2, 0, 0, -1, -1,
										  2, 3, 0, 0, -1, -1, 2, 4, 0, 0, -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("N to 1 to N graph all ports specific comps no cycles", "[Graph]")
{
	/*
		O--\     /--O
			--O--
		O--/     \--O
	*/

	const int nUnits = 5;
	const std::vector<int> connections = {0, 2, 0, 0, -1, -1, 1, 2, 0, 1, -1, -1,
										  2, 3, 1, 0, -1, -1, 2, 4, 0, 0, -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("Linear graph specific ports twisted comps no cycles", "[Graph]")
{
	const int nUnits = 4;
	const std::vector<int> connections = {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2,
										  1, 2, 0, 0, 0, 2, 1, 2, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0,
										  2, 3, 0, 0, 0, 0, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 2, 2};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("Two cycles graph mixed ports all comps", "[Graph]")
{
	/*
		___________________
		|                 |
	0---2--\     /--5---  |
			--4--      |  |
	1---3--/     \--6-----
		|______________|
	*/

	const int nUnits = 7;
	const std::vector<int> connections = {0,  2,  0,  0,  -1, -1, 1,  3,  0,  0,  -1, -1, 2,  4,  -1, -1,
										  -1, -1, 3,  4,  -1, -1, -1, -1, 4,  5,  0,  0,  -1, -1, 4,  6,
										  -1, -1, -1, -1, 5,  3,  0,  1,  -1, -1, 6,  2,  0,  1,  -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	REQUIRE(cycle);
}

TEST_CASE("N to 1 graph specific ports mixed comps no cycles", "[Graph]")
{
	/*
		O--\
			--O--O
		O--/
	*/

	const int nUnits = 4;
	const std::vector<int> connections = {0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 2,  0,
										  0, 0, 2, 1, 2, 0, 0, 0, 3, 2, 3, 0, 0, -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("Self loop graph all ports all comps with cycles", "[Graph]")
{
	/*
		O--\
			--O--O   /-O-\
		O--/         \   /
					  ---
	*/

	const int nUnits = 5;
	const std::vector<int> connections = {0, 3, -1, -1, -1, -1, 1, 3, -1, -1, -1, -1,
										  2, 2, -1, -1, -1, -1, 3, 4, -1, -1, -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	REQUIRE(cycle);
}

TEST_CASE("Two connected components graph all ports all comps no cycles", "[Graph]")
{
	/*
		O--\
			--O--O     O--O--O     O
		O--/
	*/

	const int nUnits = 8;
	const std::vector<int> connections = {0, 2, -1, -1, -1, -1, 1, 2, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1,

										  4, 5, -1, -1, -1, -1, 5, 6, -1, -1, -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("Multi connected components graph all ports all comps no cycles", "[Graph]")
{
	/*
		9 Units
		0 -> 3 -> 4 -> 1 -> 7 -> 2   (5,6,8 not connected)
	*/

	const int nUnits = 9;
	const std::vector<int> connections = {4,  1,  -1, -1, -1, -1, 0,  3,  -1, -1, -1, -1, 1,  7,  -1,
										  -1, -1, -1, 3,  4,  -1, -1, -1, -1, 7,  2,  -1, -1, -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}

TEST_CASE("Multi connected components graph all ports all comps no cycles 2", "[Graph]")
{
	/*
		9 Units
		0 -> 5 -> 6 -> 1 -> 7 -> 8 -> 2   (3,4 not connected)
	*/

	const int nUnits = 9;
	const std::vector<int> connections = {1, 7, -1, -1, -1, -1, 0, 5, -1, -1, -1, -1, 5, 6, -1, -1, -1, -1,
										  6, 1, -1, -1, -1, -1, 8, 2, -1, -1, -1, -1, 7, 8, -1, -1, -1, -1};
	cadet::util::SlicedVector<int> adjList =
		cadet::graph::adjacencyListFromConnectionList(connections.data(), nUnits, connections.size() / 6);
	checkAdjacencyList(nUnits, connections, adjList);

	std::vector<int> topoOrder;
	const bool cycle = cadet::graph::topologicalSort(adjList, topoOrder);
	checkTopoOrdering(nUnits, connections, topoOrder);

	REQUIRE(!cycle);
}
