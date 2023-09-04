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
 * Defines some useful macros for IBindingModel implementations.
 */

#ifndef LIBCADET_BINDINGMODELMACROS_HPP_
#define LIBCADET_BINDINGMODELMACROS_HPP_

/**
 * @brief Inserts implementations of all flux() method variants
 * @details An IBindingModel implementation has to provide flux() methods for different variants of state and
 *          parameter type. This macro saves some time by providing those implementations. It assumes that the
 *          implementation provides a templatized fluxImpl() function which realizes all required variants.
 *
 *          The implementation is inserted inline in the class declaration.
 */
#define CADET_BINDINGMODEL_RESIDUAL_BOILERPLATE                                                                        \
	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* y, active const* yCp,  \
					 active* res, LinearBufferAllocator workSpace, WithParamSensitivity) const                         \
	{                                                                                                                  \
		return fluxImpl<active, active, active, active>(t, secIdx, colPos, y, yCp, res, workSpace);                    \
	}                                                                                                                  \
                                                                                                                       \
	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* y, active const* yCp,  \
					 active* res, LinearBufferAllocator workSpace, WithoutParamSensitivity) const                      \
	{                                                                                                                  \
		return fluxImpl<active, active, active, double>(t, secIdx, colPos, y, yCp, res, workSpace);                    \
	}                                                                                                                  \
                                                                                                                       \
	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp,  \
					 active* res, LinearBufferAllocator workSpace) const                                               \
	{                                                                                                                  \
		return fluxImpl<double, double, active, active>(t, secIdx, colPos, y, yCp, res, workSpace);                    \
	}                                                                                                                  \
                                                                                                                       \
	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double const* yCp,  \
					 double* res, LinearBufferAllocator workSpace) const                                               \
	{                                                                                                                  \
		return fluxImpl<double, double, double, double>(t, secIdx, colPos, y, yCp, res, workSpace);                    \
	}

/**
 * @brief Inserts implementations of all analyticJacobian() method variants
 * @details An IBindingModel implementation has to provide analyticJacobian() methods for different variants
 *          of state and parameter type. This macro saves some time by providing those implementations. It
 *          assumes that the implementation provides a templatized jacobianImpl() function that realizes all
 *          required variants.
 *
 *          The implementation is inserted inline in the class declaration.
 */
#define CADET_BINDINGMODEL_JACOBIAN_BOILERPLATE                                                                        \
	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y,        \
								  int offsetCp, linalg::BandMatrix::RowIterator jac, LinearBufferAllocator workSpace)  \
		const                                                                                                          \
	{                                                                                                                  \
		jacobianImpl(t, secIdx, colPos, y, y - offsetCp, offsetCp, jac, workSpace);                                    \
	}                                                                                                                  \
                                                                                                                       \
	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y,        \
								  int offsetCp, linalg::DenseBandedRowIterator jac, LinearBufferAllocator workSpace)   \
		const                                                                                                          \
	{                                                                                                                  \
		jacobianImpl(t, secIdx, colPos, y, y - offsetCp, offsetCp, jac, workSpace);                                    \
	}

#endif // LIBCADET_BINDINGMODELMACROS_HPP_
