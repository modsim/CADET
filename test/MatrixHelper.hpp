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
 * Defines helper functions for checking matrices.
 */

#ifndef CADETTEST_MATRIXHELPER_HPP_
#define CADETTEST_MATRIXHELPER_HPP_

#include <vector>
#include "Approx.hpp"

namespace cadet
{

namespace test
{

/**
 * @brief Checks a given matrix against a linear array
 * @details The matrix is expected as dense matrix in row-major format.
 * @param [in] mat Dense matrix to be checked
 * @param [in] matRef Reference values
 */
inline void checkMatrixAgainstLinearArray(double const* mat, const std::vector<double>& matRef)
{
	for (std::size_t i = 0; i < matRef.size(); ++i)
		CHECK(matRef[i] == RelApprox(mat[i]));
}

/**
 * @brief Create a BandMatrix of given type and fill entries with their linear array index (1-based)
 * @param [in] Number of rows
 * @param [in] Lower bandwidth (excluding main diagonal)
 * @param [in] Upper bandwidth (excluding main diagonal)
 * @tparam Matrix_t Type of banded matrix to create
 * @return Banded matrix of given shape
 */
template <typename Matrix_t> Matrix_t createBandMatrix(unsigned int rows, unsigned int lower, unsigned int upper)
{
	Matrix_t bm;
	bm.resize(rows, lower, upper);

	double val = 1.0;
	for (int row = 0; row < bm.rows(); ++row)
	{
		const int lower = std::max(-static_cast<int>(bm.lowerBandwidth()), -static_cast<int>(row));
		const int upper = std::min(static_cast<int>(bm.upperBandwidth()), static_cast<int>(bm.rows() - row) - 1);
		for (int col = lower; col <= upper; ++col)
		{
			bm.centered(row, col) = val;
			val += 1.0;
		}
	}
	return bm;
}

} // namespace test
} // namespace cadet

#endif // CADETTEST_MATRIXHELPER_HPP_
