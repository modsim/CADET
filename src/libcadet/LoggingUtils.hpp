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
 * Utilities for logging various data types.
 */

#ifndef LIBCADET_LOGGING_UTILS_HPP_
#define LIBCADET_LOGGING_UTILS_HPP_

#ifndef CADET_LOGGING_DISABLE
//	#include "cadet/SolutionExporter.hpp"
#include "AutoDiff.hpp"

#include <ostream>
#endif

namespace cadet
{
namespace log
{
/**
 * @brief Container for logging arrays given by pointer and number of elements
 * @tparam T Type of the underlying array items
 */
template <class T> struct VectorPtr
{
	const T* data;
	unsigned int nElem;

	VectorPtr(T const* d, unsigned int n) : data(d), nElem(n)
	{
	}
};

/**
 * @brief Container for logging matrices given by pointer to linear array, matrix size, and storage order
 * @tparam T Type of the underlying array items
 */
template <class T> struct MatrixPtr
{
	const T* data;
	unsigned int nRows;
	unsigned int nCols;
	bool colMajor;

	MatrixPtr(T const* d, unsigned int nr, unsigned int nc) : data(d), nRows(nr), nCols(nc), colMajor(false)
	{
	}
	MatrixPtr(T const* d, unsigned int nr, unsigned int nc, bool cm) : data(d), nRows(nr), nCols(nc), colMajor(cm)
	{
	}
};

#ifndef CADET_LOGGING_DISABLE
/*
	inline std::ostream& operator<<(std::ostream& os, const cadet::ISolutionExporter& v)
	{
		os << "bulk = [";
		const unsigned int nColDof = v.numBulkDofs();
		if (nColDof > 0)
		{
			double const* colPtr = v.concentration();
			for (unsigned int i = 0; i < nColDof-1; ++i)
				os << colPtr[i] << ",";
			os << colPtr[nColDof-1];
		}
		os << "]\n";

		const unsigned int parStride = v.numRadialCells() * (v.numComponents() + v.numBoundStates());
		double const* parPtr = v.mobilePhase();
		for (unsigned int i = 0; i < v.numAxialCells(); ++i, parPtr += parStride)
		{
			os << "par" << i << " = [";
			for (unsigned int j = 0; j < parStride-1; ++j)
				os << parPtr[j] << ",";
			os << parPtr[parStride-1] << "]\n";
		}

		os << "flux = [";
		const unsigned int nFluxDof = v.numFluxDofs();
		double const* fluxPtr = v.flux();
		for (unsigned int i = 0; i < nFluxDof-1; ++i)
			os << fluxPtr[i] << ",";
		os << fluxPtr[nFluxDof-1] << "]";
		return os;
	}
*/

inline std::ostream& operator<<(std::ostream& os, const cadet::active& v)
{
	os << static_cast<double>(v) << " [";
	if (cadet::ad::getDirections() > 0)
	{
		for (unsigned int i = 0; i < cadet::ad::getDirections() - 1; ++i)
			os << v.getADValue(i) << ", ";
		os << v.getADValue(cadet::ad::getDirections() - 1);
	}
	os << "]";
	return os;
}

template <class T> inline std::ostream& operator<<(std::ostream& os, const cadet::log::VectorPtr<T>& v)
{
	os << "[";
	if (v.nElem > 0)
	{
		for (unsigned int i = 0; i < v.nElem - 1; ++i)
			os << v.data[i] << ",";
		os << v.data[v.nElem - 1];
	}
	os << "]";
	return os;
}

template <class T> inline std::ostream& operator<<(std::ostream& os, const cadet::log::MatrixPtr<T>& v)
{
	os << "[";
	if (v.nRows * v.nCols > 0)
	{
		if (v.colMajor)
		{
			// All rows but last
			for (unsigned int i = 0; i < v.nRows - 1; ++i)
			{
				// All columns but last
				for (unsigned int j = 0; j < v.nCols - 1; ++j)
				{
					os << v.data[j * v.nRows + i] << ",";
				}
				// Last column
				os << v.data[(v.nCols - 1) * v.nRows + i] << ";";
			}

			// Last row
			for (unsigned int j = 0; j < v.nCols - 1; ++j)
			{
				os << v.data[j * v.nRows + v.nRows - 1] << ",";
			}
			os << v.data[v.nCols * v.nRows - 1];
		}
		else
		{
			// All rows but last
			for (unsigned int i = 0; i < v.nRows - 1; ++i)
			{
				// All columns but last
				for (unsigned int j = 0; j < v.nCols - 1; ++j)
				{
					os << v.data[i * v.nCols + j] << ",";
				}
				// Last column
				os << v.data[(i + 1) * v.nCols - 1] << ";";
			}

			// Last row
			for (unsigned int j = 0; j < v.nCols - 1; ++j)
			{
				os << v.data[(v.nRows - 1) * v.nCols + j] << ",";
			}
			os << v.data[v.nCols * v.nRows - 1];
		}
	}
	os << "]";
	return os;
}

#endif

} // namespace log
} // namespace cadet

#endif // LIBCADET_LOGGING_UTILS_HPP_
