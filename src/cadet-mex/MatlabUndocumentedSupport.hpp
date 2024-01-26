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
 * Declares (possibly) undocumented Matlab functions
 */

#ifndef CADET_MEX_MATLAB_UNDOCUMENTED_SUPPORT_HPP_
#define CADET_MEX_MATLAB_UNDOCUMENTED_SUPPORT_HPP_

#include <mex.h>

#ifndef MATLAB_HAVE_CREATEUNINITNUMERICMATRIX
extern "C" mxArray* mxCreateUninitNumericMatrix(mwSize m, mwSize n, mxClassID classid, mxComplexity flag);
#endif

#ifndef MATLAB_HAVE_CREATEUNINITNUMERICARRAY
extern "C" mxArray* mxCreateUninitNumericArray(size_t ndim, size_t* dims, mxClassID classid, mxComplexity ComplexFlag);
#endif

#endif // CADET_MEX_MATLAB_UNDOCUMENTED_SUPPORT_HPP_
