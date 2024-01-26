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
 * Provides version info.
 */

#ifndef LIBCADET_LIBVERSIONINFO_HPP_
#define LIBCADET_LIBVERSIONINFO_HPP_

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"

namespace cadet
{

/**
 * @brief Returns the version string of the libcadet library
 * @sa cadetGetLibraryVersion()
 * @return Version string
 */
CADET_API const char* getLibraryVersion() CADET_NOEXCEPT;

/**
 * @brief Returns the git commit hash of the source which was used to build the binaries
 * @sa cadetGetLibraryCommitHash()
 * @return Git commit hash as string
 */
CADET_API const char* getLibraryCommitHash() CADET_NOEXCEPT;

/**
 * @brief Returns the git refspec of the source which was used to build the binaries
 * @sa cadetGetLibraryBranchRefspec()
 * @return Git refspec
 */
CADET_API const char* getLibraryBranchRefspec() CADET_NOEXCEPT;

/**
 * @brief Returns the versions of the dependencies used for building the binaries
 * @details The format is DEPNAME1=VERSION;DEPNAME2=VERSION; where each dependency is
 *          terminated by a semicolon.
 * @sa cadetGetLibraryDependencyVersions()
 * @return Dependency versions string
 */
CADET_API const char* getLibraryDependencyVersions() CADET_NOEXCEPT;

/**
 * @brief Returns the build type (Debug, Release, RelWithDebInfo, RelMinSize)
 * @sa cadetGetLibraryBuildType()
 * @return Build type
 */
CADET_API const char* getLibraryBuildType() CADET_NOEXCEPT;

/**
 * @brief Returns the compiler including its version used for building the library
 * @sa cadetGetLibraryCompiler()
 * @return Compiler and its version
 */
CADET_API const char* getLibraryCompiler() CADET_NOEXCEPT;

/**
 * @brief Returns the compiler flags used for building the library
 * @sa cadetGetLibraryCompilerFlags()
 * @return Compiler flags
 */
CADET_API const char* getLibraryCompilerFlags() CADET_NOEXCEPT;

/**
 * @brief Returns the git refspec of the source which was used to build the binaries
 * @sa cadetGetLibraryBuildHost()
 * @return Git refspec
 */
CADET_API const char* getLibraryBuildHost() CADET_NOEXCEPT;
} // namespace cadet

#endif // LIBCADET_LIBVERSIONINFO_HPP_
