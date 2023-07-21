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

#include "linalg/BandMatrix.hpp"
#include "linalg/DenseMatrix.hpp"
#include "linalg/SparseMatrix.hpp"
#include "AdUtils.hpp"

#include <Eigen/Sparse>
#include <limits>
#include <algorithm>

namespace cadet
{

namespace ad
{

void prepareAdVectorSeedsForBandMatrix(active* const adVec, int adDirOffset, int rows, 
	int lowerBandwidth, int upperBandwidth, int diagDir)
{
	// Start with diagonal Jacobian element
	int dir = diagDir;
	for (int eq = 0; eq < rows; ++eq)
	{
		// Clear previously set directions
		adVec[eq].fillADValue(adDirOffset, 0.0);
		// Set direction
		adVec[eq].setADValue(adDirOffset + dir, 1.0);

		// Wrap around at end of row and jump to lowest subdiagonal
		if (dir == diagDir + upperBandwidth)
			dir = diagDir - lowerBandwidth;
		else
			++dir;
	}
}

void extractBandedJacobianFromAd(active const* const adVec, int adDirOffset, int diagDir, linalg::BandMatrix& mat)
{
	const int lowerBandwidth = mat.lowerBandwidth();
	const int upperBandwidth = mat.upperBandwidth();
	const int stride = lowerBandwidth + 1 + upperBandwidth;
	for (int eq = 0; eq < mat.rows(); ++eq)
	{
		// Start with lowest subdiagonal and stay in the range of the columns:
		// diagDir might be too big for the matrix and, hence, dir ranges between
		// diagDir - lowerBandwidth <= dir <= diagDir + upperBandwidth
		int dir = diagDir - lowerBandwidth + eq % stride;

		// Loop over diagonals
		for (int diag = 0; diag < stride; ++diag)
		{
			mat.native(eq, diag) = adVec[eq].getADValue(adDirOffset + dir);

			// Wrap around at end of row and jump to lowest subdiagonal
			if (dir == diagDir + upperBandwidth)
				dir = diagDir - lowerBandwidth;
			else
				++dir;
		}
	}
}

void extractBandedEigenJacobianFromAd(active const* const adVec, int adDirOffset, int diagDir, const int lowerBandwidth, const int upperBandwidth, Eigen::SparseMatrix<double, Eigen::RowMajor>& mat)
{
	const int stride = lowerBandwidth + 1 + upperBandwidth;
	for (int eq = 0; eq < mat.rows(); ++eq)
	{
		// Start with lowest subdiagonal and stay in the range of the columns:
		// diagDir might be too big for the matrix and, hence, dir ranges between
		// diagDir - lowerBandwidth <= dir <= diagDir + upperBandwidth
		int dir = diagDir - lowerBandwidth + eq % stride;

		// Loop over diagonals
		for (int diag = 0; diag < stride; ++diag)
		{
			if (eq - lowerBandwidth + diag >= 0 && // left block boundary
				eq - lowerBandwidth + diag < mat.rows() && // right block boundary
				adVec[eq].getADValue(adDirOffset + dir) != 0.0 // keep pattern
				)
				mat.coeffRef(eq, eq - lowerBandwidth + diag) = adVec[eq].getADValue(adDirOffset + dir);

			// Wrap around at end of row and jump to lowest subdiagonal
			if (dir == diagDir + upperBandwidth)
				dir = diagDir - lowerBandwidth;
			else
				++dir;
		}
	}
}

void extractBandedBlockEigenJacobianFromAd(active const* const adVec, int adDirOffset, int diagDir, const int lowerBandwidth, const int upperBandwidth,
	const int blockOffset, const int nCols, Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, const int matrixOffset)
{
	const int stride = lowerBandwidth + 1 + upperBandwidth;
	for (int eq = blockOffset; eq < blockOffset + nCols; ++eq)
	{
		// Start with lowest subdiagonal and stay in the range of the columns:
		// diagDir might be too big for the matrix and, hence, dir ranges between
		// diagDir - lowerBandwidth <= dir <= diagDir + upperBandwidth
		int dir = diagDir - lowerBandwidth + eq % stride;

		// Loop over diagonals
		for (int diag = 0; diag < stride; ++diag)
		{
			if (eq - lowerBandwidth + diag >= blockOffset && // left block boundary
				eq - lowerBandwidth + diag < blockOffset + nCols && // right block boundary
				adVec[eq].getADValue(adDirOffset + dir) != 0.0 // keep pattern
				)
				mat.coeffRef(matrixOffset + eq, matrixOffset + eq - lowerBandwidth + diag) = adVec[eq].getADValue(adDirOffset + dir);

			// Wrap around at end of row and jump to lowest subdiagonal
			if (dir == diagDir + upperBandwidth)
				dir = diagDir - lowerBandwidth;
			else
				++dir;
		}
	}
}

void prepareAdVectorSeedsForDenseMatrix(active* const adVec, int adDirOffset, int cols)
{
	for (int col = 0; col < cols; ++col)
	{
		// Clear previously set directions
		adVec[col].fillADValue(adDirOffset, 0.0);
		// Set direction
		adVec[col].setADValue(adDirOffset + col, 1.0);
	}
}

void extractDenseJacobianFromAd(active const* const adVec, int adDirOffset, linalg::detail::DenseMatrixBase& mat)
{
	for (int eq = 0; eq < mat.rows(); ++eq)
	{
		// Loop over columns
		for (int col = 0; col < mat.columns(); ++col)
		{
			mat.native(eq, col) = adVec[eq].getADValue(adDirOffset + col);
		}
	}
}

void extractDenseJacobianFromBandedAd(active const* const adVec, int row, int adDirOffset, int diagDir, 
	int lowerBandwidth, int upperBandwidth, linalg::detail::DenseMatrixBase& mat)
{
	const int stride = lowerBandwidth + 1 + upperBandwidth;
	for (int eq = 0; eq < mat.rows(); ++eq)
	{
		// Start with lowest subdiagonal and stay in the range of the columns:
		// diagDir might be too big for the matrix and, hence, dir ranges between
		// diagDir - lowerBandwidth <= dir <= diagDir + upperBandwidth
		int dir = diagDir - lowerBandwidth + ((eq + row) % stride);

		// Loop over diagonals
		for (int diag = 0; diag < stride; ++diag)
		{
			// Calculate column index from subdiagonal
			const int diagCol = static_cast<int>(diag) - static_cast<int>(lowerBandwidth) + static_cast<int>(eq);
			if ((diagCol >= 0) && (diagCol < static_cast<int>(mat.columns())))
				mat.native(eq, diagCol) = adVec[row + eq].getADValue(adDirOffset + dir);

			// Wrap around at end of row and jump to lowest subdiagonal
			if (dir == diagDir + upperBandwidth)
				dir = diagDir - lowerBandwidth;
			else
				++dir;
		}
	}
}

double compareBandedJacobianWithAd(active const* const adVec, int adDirOffset, int diagDir, const linalg::BandMatrix& mat)
{
	const int lowerBandwidth = mat.lowerBandwidth();
	const int upperBandwidth = mat.upperBandwidth();
	const int stride = lowerBandwidth + 1 + upperBandwidth;

	double maxDiff = 0.0;
	for (int eq = 0; eq < mat.rows(); ++eq)
	{
		// Start with lowest subdiagonal and stay in the range of the columns:
		// diagDir might be too big for the matrix and, hence, dir ranges between
		// diagDir - lowerBandwidth <= dir <= diagDir + upperBandwidth
		int dir = diagDir - lowerBandwidth + eq % stride;

		// Loop over diagonals
		for (int diag = 0; diag < stride; ++diag)
		{
			double baseVal = adVec[eq].getADValue(adDirOffset + dir);
			if (std::isnan(mat.native(eq, diag)) || std::isnan(baseVal))
				return std::numeric_limits<double>::quiet_NaN();
			const double diff = std::abs(mat.native(eq, diag) - baseVal);

			baseVal = std::abs(baseVal);
			if (baseVal > 0.0)
				maxDiff = std::max(maxDiff, diff / baseVal);
			else
				maxDiff = std::max(maxDiff, diff);

			// Wrap around at end of row and jump to lowest subdiagonal
			if (dir == diagDir + upperBandwidth)
				dir = diagDir - lowerBandwidth;
			else
				++dir;
		}
	}
	return maxDiff;
}

double compareDenseJacobianWithAd(active const* const adVec, int adDirOffset, const linalg::detail::DenseMatrixBase& mat)
{
	double maxDiff = 0.0;
	for (int eq = 0; eq < mat.rows(); ++eq)
	{
		// Loop over columns
		for (int col = 0; col < mat.columns(); ++col)
		{
			double baseVal = adVec[eq].getADValue(adDirOffset + col);
			if (std::isnan(mat.native(eq, col)) || std::isnan(baseVal))
				return std::numeric_limits<double>::quiet_NaN();
			const double diff = std::abs(mat.native(eq, col) - baseVal);

			baseVal = std::abs(baseVal);
			if (baseVal > 0.0)
				maxDiff = std::max(maxDiff, diff / baseVal);
			else
				maxDiff = std::max(maxDiff, diff);
		}
	}
	return maxDiff;
}

double compareDenseJacobianWithBandedAd(active const* const adVec, int row, int adDirOffset, int diagDir, 
	int lowerBandwidth, int upperBandwidth, const linalg::detail::DenseMatrixBase& mat)
{
	double maxDiff = 0.0;
	const int stride = lowerBandwidth + 1 + upperBandwidth;
	for (int eq = 0; eq < mat.rows(); ++eq)
	{
		// Start with lowest subdiagonal and stay in the range of the columns:
		// diagDir might be too big for the matrix and, hence, dir ranges between
		// diagDir - lowerBandwidth <= dir <= diagDir + upperBandwidth
		int dir = diagDir - lowerBandwidth + ((eq + row) % stride);

		// Loop over diagonals
		for (int diag = 0; diag < stride; ++diag)
		{
			// Calculate column index from subdiagonal
			const int diagCol = static_cast<int>(diag) - static_cast<int>(lowerBandwidth) + static_cast<int>(eq);
			if ((diagCol >= 0) && (diagCol < static_cast<int>(mat.columns())))
			{
				double baseVal = adVec[row + eq].getADValue(adDirOffset + dir);
				if (std::isnan(mat.native(eq, diagCol)) || std::isnan(baseVal))
					return std::numeric_limits<double>::quiet_NaN();
				const double diff = std::abs(mat.native(eq, diagCol) - baseVal);
				
				baseVal = std::abs(baseVal);
				if (baseVal > 0.0)
					maxDiff = std::max(maxDiff, diff / baseVal);
				else
					maxDiff = std::max(maxDiff, diff);
			}

			// Wrap around at end of row and jump to lowest subdiagonal
			if (dir == diagDir + upperBandwidth)
				dir = diagDir - lowerBandwidth;
			else
				++dir;
		}
	}
	return maxDiff;
}

void adMatrixVectorMultiply(const linalg::SparseMatrix<active>& mat, double const* x, double* y, double alpha, double beta, int adDir)
{
	const std::vector<int>& rows = mat.rows();
	const std::vector<int>& cols = mat.cols();
	const std::vector<active>& values = mat.values();
	const int numNonZero = mat.numNonZero();

	for (int i = 0; i < numNonZero; ++i)
	{
		y[rows[i]] = alpha * values[i].getADValue(adDir) * x[cols[i]] + beta * y[rows[i]];
	}
}

DenseJacobianExtractor::DenseJacobianExtractor() { }

void DenseJacobianExtractor::extractJacobian(active const* adRes, int row, int adDirOffset, linalg::detail::DenseMatrixBase& mat) const
{
	extractDenseJacobianFromAd(adRes + row, adDirOffset + row, mat);
}

double DenseJacobianExtractor::compareWithJacobian(active const* adRes, int row, int adDirOffset, linalg::detail::DenseMatrixBase& mat) const
{
	return compareDenseJacobianWithAd(adRes + row, adDirOffset + row, mat);
}


BandedJacobianExtractor::BandedJacobianExtractor(int diagDir, int lowerBandwidth, int upperBandwidth)
	: _diagDir(diagDir), _lowerBandwidth(lowerBandwidth), _upperBandwidth(upperBandwidth)
	{ }

void BandedJacobianExtractor::extractJacobian(active const* adRes, int row, int adDirOffset, linalg::detail::DenseMatrixBase& mat) const
{
	extractDenseJacobianFromBandedAd(adRes, row, adDirOffset, _diagDir, _lowerBandwidth, _upperBandwidth, mat);
}

double BandedJacobianExtractor::compareWithJacobian(active const* adRes, int row, int adDirOffset, linalg::detail::DenseMatrixBase& mat) const
{
	return compareDenseJacobianWithBandedAd(adRes, row, adDirOffset, _diagDir, _lowerBandwidth, _upperBandwidth, mat);
}

}  // namespace ad

}  // namespace cadet
