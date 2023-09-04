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
 * Provides function for creating and destroying important classes.
 */

#ifndef LIBCADET_FACTORYFUNCS_HPP_
#define LIBCADET_FACTORYFUNCS_HPP_

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"

namespace cadet
{

class IModelBuilder;
class ISimulator;

/**
 * @brief Creates an IModelBuilder object
 * @sa cadetCreateModelBuilder()
 * @return IModelBuilder object or @c NULL if something went wrong
 */
CADET_API IModelBuilder* createModelBuilder();

/**
 * @brief Destroys a given model builder
 * @details Because a different memory space is assigned to dynamically loaded libraries,
 *          memory allocated by the library has to be freed in the library. Thus, users
 *          have to explicitly destroy their IModelBuilder objects here.
 * @sa cadetDestroyModelBuilder()
 * @param [in] builder IModelBuilder to be destroyed
 */
CADET_API void destroyModelBuilder(IModelBuilder* const builder) CADET_NOEXCEPT;

/**
 * @brief Creates an ISimulator object
 * @sa cadetCreateSimulator()
 * @return ISimulator object or @c NULL if something went wrong
 */
CADET_API ISimulator* createSimulator();

/**
 * @brief Destroys a given simulator
 * @details Because a different memory space is assigned to dynamically loaded libraries,
 *          memory allocated by the library has to be freed in the library. Thus, users
 *          have to explicitly destroy their ISimulator objects here.
 * @sa cadetDestroySimulator()
 * @param [in] sim ISimulator to be destroyed
 */
CADET_API void destroySimulator(ISimulator* const sim) CADET_NOEXCEPT;

} // namespace cadet

extern "C"
{
	/**
	 * @brief Creates an IModelBuilder object
	 * @return IModelBuilder object or @c NULL if something went wrong
	 */
	CADET_API cadet::IModelBuilder* cadetCreateModelBuilder();

	/**
	 * @brief Destroys a given model builder
	 * @details Because a different memory space is assigned to dynamically loaded libraries,
	 *          memory allocated by the library has to be freed in the library. Thus, users
	 *          have to explicitly destroy their IModelBuilder objects here.
	 *
	 * @param [in] builder IModelBuilder to be destroyed
	 */
	CADET_API void cadetDestroyModelBuilder(cadet::IModelBuilder* const builder);

	/**
	 * @brief Creates an ISimulator object
	 * @return ISimulator object or @c NULL if something went wrong
	 */
	CADET_API cadet::ISimulator* cadetCreateSimulator();

	/**
	 * @brief Destroys a given simulator
	 * @details Because a different memory space is assigned to dynamically loaded libraries,
	 *          memory allocated by the library has to be freed in the library. Thus, users
	 *          have to explicitly destroy their ISimulator objects here.
	 *
	 * @param [in] sim ISimulator to be destroyed
	 */
	CADET_API void cadetDestroySimulator(cadet::ISimulator* const sim);
}

#endif // LIBCADET_FACTORYFUNCS_HPP_
