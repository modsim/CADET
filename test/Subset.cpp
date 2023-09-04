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

#include <vector>
#include <limits>
#include <algorithm>

#include "linalg/Subset.hpp"
#include "linalg/Norms.hpp"

#include "MatrixHelper.hpp"

TEST_CASE("Extract subset of BandMatrix", "[BandMatrix],[Subset],[LinAlg]")
{
	using cadet::linalg::BandMatrix;
	using cadet::linalg::DenseMatrix;

	std::vector<int> maskData{1, 0, 1, 0};
	cadet::linalg::ConstMaskArray cma{maskData.data(), static_cast<int>(maskData.size())};

	const BandMatrix bm = cadet::test::createBandMatrix<BandMatrix>(10, 2, 3);
	DenseMatrix dm;
	dm.resize(2, 2);

	cadet::linalg::copyMatrixSubset(bm, cma, cma, 0, 0, dm);
	cadet::test::checkMatrixAgainstLinearArray(dm.data(), {1.0, 3.0, 10.0, 12.0});

	cadet::linalg::copyMatrixSubset(bm, cma, cma, 0, 1, dm);
	cadet::test::checkMatrixAgainstLinearArray(dm.data(), {2.0, 4.0, 11.0, 13.0});

	cadet::linalg::copyMatrixSubset(bm, cma, cma, 1, 0, dm);
	cadet::test::checkMatrixAgainstLinearArray(dm.data(), {6.0, 8.0, 16.0, 18.0});

	cadet::linalg::copyMatrixSubset(bm, cma, cma, 1, 1, dm);
	cadet::test::checkMatrixAgainstLinearArray(dm.data(), {7.0, 9.0, 17.0, 19.0});
}
