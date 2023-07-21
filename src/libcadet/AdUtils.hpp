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
 * Provides utilities for AD vectors and matrices
 */

#ifndef LIBCADET_ADUTILS_HPP_
#define LIBCADET_ADUTILS_HPP_

#include "AutoDiff.hpp"

#include<Eigen/Sparse>

namespace cadet
{

namespace linalg
{
	class BandMatrix;

	namespace detail
	{
		class DenseMatrixBase;
	}

	template <class real_t> class SparseMatrix;
}

namespace ad
{

/**
 * @brief Sets seed vectors on an AD vector for computing a banded Jacobian
 * @details The band structure of a Jacobian is exploited by band compression.
 *          This is explained in @cite Puttmann2016.
 * @todo Provide more details
 * @param [in,out] adVec Vector of AD datatypes whose seed vectors are to be set
 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
 * @param [in] rows Number of Jacobian rows (length of the AD vector)
 * @param [in] lowerBandwidth Lower bandwidth (number of lower subdiagonals) of the banded Jacobian
 * @param [in] upperBandwidth Upper bandwidth (number of upper superdiagonals) of the banded Jacobian
 * @param [in] diagDir Diagonal direction index
 */
void prepareAdVectorSeedsForBandMatrix(active* const adVec, int adDirOffset, int rows, 
	int lowerBandwidth, int upperBandwidth, int diagDir);

/**
 * @brief Extracts a band matrix from band compressed AD seed vectors
 * @details Uses the results of an AD computation with seed vectors set by prepareAdVectorSeedsForBandMatrix() to
			assemble the Jacobian which is a band matrix.
 * @param [in] adVec Vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
 * @param [in] diagDir Diagonal direction index
 * @param [out] mat BandMatrix to be populated with the Jacobian
 */
void extractBandedJacobianFromAd(active const* const adVec, int adDirOffset, int diagDir, linalg::BandMatrix& mat);

/**
 * @brief Extracts a band (sub)matrix (Eigen lib) from band compressed AD seed vectors
 * @details Uses the results of an AD computation with seed vectors set by prepareAdVectorSeedsForBandMatrix() to
			assemble the Jacobian block which is a band matrix. The block must be on the main diagonal.
 * @param [in] adVec Vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
 * @param [in] diagDir Diagonal direction index
 * @param [in] lowerBandwidth lower band width
 * @param [in] upperBandwidth upper band width
 * @param [in] blockOffset offset to diagonal block
 * @param [in] nCols number of columns of the extracted block
 * @param [out] mat Eigen matrix to be populated with the Jacobian block
 */
void extractBandedBlockEigenJacobianFromAd(active const* const adVec, int adDirOffset, int diagDir, const int lowerBandwidth, const int upperBandwidth,
	const int blockOffset, const int nCols, Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, const int matrixOffset = 0);
/**
 * @brief Extracts a band matrix (Eigen lib) from band compressed AD seed vectors
 * @details Uses the results of an AD computation with seed vectors set by prepareAdVectorSeedsForBandMatrix() to
			assemble the Jacobian block which is a band matrix. The block must be on the main diagonal.
 * @param [in] adVec Vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
 * @param [in] diagDir Diagonal direction index
 * @param [in] lowerBandwidth lower band width
 * @param [in] upperBandwidth upper band width
 * @param [out] mat Eigen matrix to be populated with the Jacobian block
 */
void extractBandedEigenJacobianFromAd(active const* const adVec, int adDirOffset, int diagDir, const int lowerBandwidth, const int upperBandwidth, Eigen::SparseMatrix<double, Eigen::RowMajor>& mat);

/**
 * @brief Sets seed vectors on an AD vector for computing a dense Jacobian
 * @param [in,out] adVec Vector of AD datatypes whose seed vectors are to be set
 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
 * @param [in] cols Nnumber of Jacobian columns (length of the AD vector and number of seed vectors)
 */
void prepareAdVectorSeedsForDenseMatrix(active* const adVec, int adDirOffset, int cols); 

/**
 * @brief Extracts a dense matrix from AD seed vectors
 * @details Uses the results of an AD computation with seed vectors set by prepareAdVectorSeedsForDenseMatrix() to
			assemble the Jacobian which is a dense matrix.
 * @param [in] adVec Vector of AD datatypes with seed vectors
 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
 * @param [out] mat DenseMatrix to be populated with the Jacobian, where the matrix is of the correct size and allocated
 */
void extractDenseJacobianFromAd(active const* const adVec, int adDirOffset, linalg::detail::DenseMatrixBase& mat);

/**
 * @brief Extracts a dense submatrix from band compressed AD seed vectors
 * @details Uses the results of an AD computation with seed vectors set by prepareAdVectorSeedsForBandMatrix() to
			assemble a subset of the banded Jacobian into a dense matrix.
			The subset is taken from the top left element of the band matrix (i.e., the first element on the
			main diagonal).
 * @param [in] adVec Vector of AD datatypes with band compressed seed vectors pointing to the first row of the band matrix
 * @param [in] row Index of the first row to be extracted
 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
 * @param [in] diagDir Diagonal direction index
 * @param [in] lowerBandwidth Lower bandwidth (number of lower subdiagonals) of the banded Jacobian
 * @param [in] upperBandwidth Upper bandwidth (number of upper superdiagonals) of the banded Jacobian
 * @param [out] mat Dense matrix to be populated with the Jacobian submatrix
 */
void extractDenseJacobianFromBandedAd(active const* const adVec, int row, int adDirOffset, int diagDir, 
	int lowerBandwidth, int upperBandwidth, linalg::detail::DenseMatrixBase& mat);

/**
 * @brief Compares a banded Jacobian with an AD version derived by band compressed AD seed vectors
 * @details Uses the results of an AD computation with seed vectors set by prepareAdVectorSeedsForBandMatrix() to
			compare the results with a given banded Jacobian. The AD Jacobian is treated as base and the analytic
			Jacobian is compared against it. The relative difference
			@f[ \Delta_{ij} = \begin{cases} \left\lvert \frac{ J_{\text{ana},ij} - J_{\text{ad},ij} }{ J_{\text{ad},ij} }\right\rvert, & J_{\text{ad},ij} \neq 0 \\ 
							   \left\lvert J_{\text{ana},ij} - J_{\text{ad},ij} \right\rvert, & J_{\text{ad},ij} = 0 \end{cases} @f]
			is computed for each matrix entry. The maximum of all @f$ \Delta_{ij} @f$ is returned.
 * @param [in] adVec Vector of AD datatypes with band compressed seed vectors
 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
 * @param [in] diagDir Diagonal direction index
 * @param [in] mat BandMatrix populated with the analytic Jacobian
 * @return The maximum absolute relative difference between the matrix elements
 */
double compareBandedJacobianWithAd(active const* const adVec, int adDirOffset, int diagDir, const linalg::BandMatrix& mat);

/**
 * @brief Compares a dense Jacobian with an AD version derived by AD seed vectors
 * @details Uses the results of an AD computation with seed vectors set by prepareAdVectorSeedsForDenseMatrix() to
			compare the results with a given dense Jacobian. The AD Jacobian is treated as base and the analytic
			Jacobian is compared against it. The relative difference
			@f[ \Delta_{ij} = \begin{cases} \left\lvert \frac{ J_{\text{ana},ij} - J_{\text{ad},ij} }{ J_{\text{ad},ij} }\right\rvert, & J_{\text{ad},ij} \neq 0 \\ 
							   \left\lvert J_{\text{ana},ij} - J_{\text{ad},ij} \right\rvert, & J_{\text{ad},ij} = 0 \end{cases} @f]
			is computed for each matrix entry. The maximum of all @f$ \Delta_{ij} @f$ is returned.
 * @param [in] adVec Vector of AD datatypes with seed vectors
 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
 * @param [in] mat Dense matrix populated with the analytic Jacobian
 * @return The maximum absolute relative difference between the matrix elements
 */
double compareDenseJacobianWithAd(active const* const adVec, int adDirOffset, const linalg::detail::DenseMatrixBase& mat);

/**
 * @brief Compares a dense submatrix with a band compressed AD version
 * @details Uses the results of an AD computation with seed vectors set by prepareAdVectorSeedsForBandMatrix() to
			compare the results with a given dense submatrix of the Jacobian. The AD Jacobian is treated as base
			and the analytic Jacobian is compared against it. The relative difference
			@f[ \Delta_{ij} = \begin{cases} \left\lvert \frac{ J_{\text{ana},ij} - J_{\text{ad},ij} }{ J_{\text{ad},ij} }\right\rvert, & J_{\text{ad},ij} \neq 0 \\ 
							   \left\lvert J_{\text{ana},ij} - J_{\text{ad},ij} \right\rvert, & J_{\text{ad},ij} = 0 \end{cases} @f]
			is computed for each matrix entry. The maximum of all @f$ \Delta_{ij} @f$ is returned.
			The submatrix is taken from the top left element of the band matrix (i.e., the first element on the
			main diagonal).
 * @param [in] adVec Vector of AD datatypes with band compressed seed vectors pointing to the first row of the band matrix
 * @param [in] row Index of the first row to be extracted
 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
 * @param [in] diagDir Diagonal direction index
 * @param [in] lowerBandwidth Lower bandwidth (number of lower subdiagonals) of the banded Jacobian
 * @param [in] upperBandwidth Upper bandwidth (number of upper superdiagonals) of the banded Jacobian
 * @param [in] mat Dense matrix populated with the dense Jacobian submatrix
 * @return The maximum absolute relative difference between the matrix elements
 */
double compareDenseJacobianWithBandedAd(active const* const adVec, int row, int adDirOffset, int diagDir, 
	int lowerBandwidth, int upperBandwidth, const linalg::detail::DenseMatrixBase& mat);

/**
 * @brief Performs the operation @f$ y = \alpha A x + \beta y @f$ using the derivative matrix
 * @details The provided sparse matrix @p mat actually consists of multiple matrices: A native one
 *          using the actual values stored in @p mat and several derivative matrices given by the
 *          AD vectors of the active type. This function selects one of those derivative matrices
 *          and performs the matrix-vector operation.
 * @param [in] mat Matrix of AD datatype
 * @param [in] x Vector to multiply the matrix with
 * @param [in,out] y Destination vector which is updated with the result
 * @param [in] alpha Factor @f$ \alpha @f$ in front of @f$ Ax @f$
 * @param [in] beta Factor @f$ \beta @f$ in front of @f$ y @f$
 * @param [in] adDir AD direction to use (selects the derivative matrix)
 */
void adMatrixVectorMultiply(const linalg::SparseMatrix<active>& mat, double const* x, double* y, double alpha, double beta, int adDir);

/**
 * @brief Copies the results (0th derivative) of an AD vector to a double vector
 * @param [in] adVec Source vector of AD datatypes
 * @param [out] dest Destination vector
 * @param [in] size Size of the vectors
 * @todo Check if loop unrolling is beneficial
 */
inline void copyFromAd(active const* const adVec, double* const dest, int size)
{
	for (int i = 0; i < size; ++i)
		dest[i] = static_cast<double>(adVec[i]);
}


/**
 * @brief Copies an AD direction of an AD vector to a double vector
 * @param [in] adVec Source vector of AD datatypes
 * @param [out] dest Destination vector
 * @param [in] size Size of the vectors
 * @param [in] adDir AD direction
 * @todo Check if loop unrolling is beneficial
 */
inline void copyFromAdDirection(active const* const adVec, double* const dest, int size, int adDir)
{
	for (int i = 0; i < size; ++i)
		dest[i] = adVec[i].getADValue(adDir);
}

/**
 * @brief Copies the values of a double vector into an AD vector without modifying its derivatives
 * @param [in] src Source vector
 * @param [out] adVec Destination vector of AD datatypes
 * @param [in] size Size of the vectors
 * @todo Check if loop unrolling is beneficial
 */
inline void copyToAd(double const* const src, active* const adVec, int size)
{
	for (int i = 0; i < size; ++i)
		adVec[i].setValue(src[i]);
}

/**
 * @brief Copies the values of a double vector into an AD vector resetting its derivatives to @c 0.0
 * @param [in] src Source vector
 * @param [out] adVec Destination vector of AD datatypes
 * @param [in] size Size of the vectors
 * @todo Check if loop unrolling is beneficial
 */
inline void copyToAdResetSens(double const* const src, active* const adVec, int size)
{
	for (int i = 0; i < size; ++i)
		adVec[i] = src[i];
}

/**
 * @brief Sets the values of an AD vector without modifying its derivatives
 * @param [out] adVec Destination vector of AD datatypes
 * @param [in] size Size of the vectors
 * @param [in] val Value
 * @todo Check if loop unrolling is beneficial
 */
inline void fillAd(active* const adVec, int size, double val)
{
	for (int i = 0; i < size; ++i)
		adVec[i].setValue(val);
}

/**
 * @brief Resets a vector of AD datatypes erasing both its value and its derivatives
 * @param [in,out] adVec Vector of AD datatypes to be reset
 * @param [in] size Length of the vector
 * @todo Check if loop unrolling is beneficial
 */
inline void resetAd(active* const adVec, int size)
{
	for (int i = 0; i < size; ++i)
		adVec[i] = 0.0;
}

/**
 * @brief Interface for extracting Jacobians via AD
 */
class IJacobianExtractor
{
public:
	virtual ~IJacobianExtractor() { }

	/**
	 * @brief Extracts a Jacobian from an AD vector
	 * @details Extracts a Jacobian matrix from an AD vector into a matrix.
	 * @param [in] adRes AD vector used for evaluating a function
	 * @param [in] row Index of the first row to be extracted from the AD data
	 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
	 * @param [out] mat Matrix which stores the Jacobian
	 */
	virtual void extractJacobian(active const* adRes, int row, int adDirOffset, linalg::detail::DenseMatrixBase& mat) const = 0;

	/**
	 * @brief Compares the AD Jacobian with a given Jacobian matrix and returns the maximum absolute difference
	 * @param [in] adRes AD vector used for evaluating a function
	 * @param [in] row Index of the first row to be extracted from the AD data
	 * @param [in] adDirOffset Offset in the AD directions (can be used to move past parameter sensitivity directions)
	 * @param [in] mat Matrix which stores another Jacobian used for comparison
	 * @return Maximum absolute difference between AD and given Jacobian matrix
	 */
	virtual double compareWithJacobian(active const* adRes, int row, int adDirOffset, linalg::detail::DenseMatrixBase& mat) const = 0;
};

/**
 * @brief Extracts Jacobians from AD setup for dense matrices
 * @details The seed vectors in the AD vector are standard unit vectors.
 */
class DenseJacobianExtractor : public IJacobianExtractor
{
public:
	DenseJacobianExtractor();
	virtual void extractJacobian(active const* adRes, int row, int adDirOffset, linalg::detail::DenseMatrixBase& mat) const;
	virtual double compareWithJacobian(active const* adRes, int row, int adDirOffset, linalg::detail::DenseMatrixBase& mat) const;
protected:
};

/**
 * @brief Extracts Jacobians from AD setup for band matrices
 * @details The seed vectors in the AD vector use band compression.
 */
class BandedJacobianExtractor : public IJacobianExtractor
{
public:
	BandedJacobianExtractor(int diagDir, int lowerBandwidth, int upperBandwidth);
	virtual void extractJacobian(active const* adRes, int row, int adDirOffset, linalg::detail::DenseMatrixBase& mat) const;
	virtual double compareWithJacobian(active const* adRes, int row, int adDirOffset, linalg::detail::DenseMatrixBase& mat) const;
protected:
	int _diagDir;
	int _lowerBandwidth;
	int _upperBandwidth;
};

} // namespace ad

} // namespace cadet

#endif  // LIBCADET_ADUTILS_HPP_
