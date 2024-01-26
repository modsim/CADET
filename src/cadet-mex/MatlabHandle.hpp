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
 * Provides Matlab IO support (reading & writing from Matlab structs)
 */

#ifndef CADET_MEX_MATLABHANDLE_HPP_
#define CADET_MEX_MATLABHANDLE_HPP_

#include <mex.h>

#include <cstdint>

namespace cadet
{

namespace mex
{

// Adapt Matlab handle type to different pointer sizes on 32bit and 64bit systems
#ifdef CADET_MEX_32BIT
const mxClassID MatlabPtrHandleTypeId = mxUINT32_CLASS;
typedef uint32_t MatlabPtrHandle;
#else
const mxClassID MatlabPtrHandleTypeId = mxUINT64_CLASS;
typedef uint64_t MatlabPtrHandle;
#endif

/**
 * @brief Converts a pointer to a Matlab handle
 * @details The address the pointer is pointing to is converted to an integer
 *          and returned in a Matlab scalar. This function also enables reference
 *          counting of all objects created by this MEX file. By calling mexLock()
 *          a Matlab internal counter is incremented. The MEX file cannot be
 *          unloaded from memory as long as objects are still present (counter
 *          is greater than 0).
 * @param [in] ptr Pointer to be converted
 * @return Matlab integer handle
 * @tparam Object_t Type of the object the pointer points to
 */
template <class Object_t> inline mxArray* convertPtr2Mat(Object_t* ptr)
{
	// Lock the MEX file for each created object (acts as reference counter)
	mexLock();

	// Convert pointer to Matlab uint64 handle
	mxArray* const out = mxCreateNumericMatrix(1, 1, MatlabPtrHandleTypeId, mxREAL);
	MatlabPtrHandle* const handle = static_cast<MatlabPtrHandle*>(mxGetData(out));
	*handle = reinterpret_cast<MatlabPtrHandle>(ptr);
	return out;
}

/**
 * @brief Convert a Matlab handle to a pointer
 * @details The Matlab scalar integer encodes the address of an object in memory.
 * @param [in] hd Matlab integer handle
 * @return Pointer to an object
 * @tparam Object_t Expected type of the object the handle points to
 */
template <class Object_t> inline Object_t* convertMat2Ptr(const mxArray* hd)
{
	if ((mxGetNumberOfElements(hd) != 1) || (mxGetClassID(hd) != MatlabPtrHandleTypeId) || mxIsComplex(hd))
		mexErrMsgIdAndTxt("CADET:mexError", "Handle must be a real uint64 scalar.\n");

	MatlabPtrHandle* const handle = static_cast<MatlabPtrHandle*>(mxGetData(hd));
	return reinterpret_cast<Object_t*>(*handle);
}

/**
 * @brief Deletes a created object passed as a Matlab handle
 * @details The Matlab scalar integer encodes the address of an object in memory.
 *          This function also decrements the Matlab internal reference counter
 *          of all created objects of this MEX file (see convertPtr2Mat() for its
 *          counterpart). By calling mexUnlock() the counter is decremented and
 *          once 0 is reached, the MEX file can be safely unloaded without memory
 *          leaks.
 * @param [in] hd Matlab integer handle
 * @tparam Object_t Expected type of the object the handle points to
 */
template <class Object_t> inline void destroyObject(const mxArray* hd)
{
	// Destroy object
	delete convertMat2Ptr<Object_t>(hd);

	// Decrement reference counter by unlocking the MEX file
	mexUnlock();
}

} // namespace mex
} // namespace cadet

#endif // CADET_MEX_MATLABHANDLE_HPP_
