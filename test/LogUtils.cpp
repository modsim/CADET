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

#include "LoggingUtils.hpp"

#define CADET_LOGGING_DISABLE
#include "Logging.hpp"

#include <sstream>
#include <vector>

TEST_CASE("Log matrix output from linear array", "[Logging]")
{
	std::stringstream ss;

	SECTION("Rectangular matrix")
	{
		SECTION("Row-major")
		{
			const std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
			ss << cadet::log::MatrixPtr<int>(data.data(), 3, 4, false);
			REQUIRE(ss.str() == "[0,1,2,3;4,5,6,7;8,9,10,11]");
		}

		SECTION("Column-major")
		{
			const std::vector<int> data = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
			ss << cadet::log::MatrixPtr<int>(data.data(), 3, 4, true);
			REQUIRE(ss.str() == "[0,1,2,3;4,5,6,7;8,9,10,11]");
		}
	}

	SECTION("Square matrix")
	{
		SECTION("Row-major")
		{
			const std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7, 8};
			ss << cadet::log::MatrixPtr<int>(data.data(), 3, 3, false);
			REQUIRE(ss.str() == "[0,1,2;3,4,5;6,7,8]");
		}

		SECTION("Column-major")
		{
			const std::vector<int> data = {0, 3, 6, 1, 4, 7, 2, 5, 8};
			ss << cadet::log::MatrixPtr<int>(data.data(), 3, 3, true);
			REQUIRE(ss.str() == "[0,1,2;3,4,5;6,7,8]");
		}
	}
}
