# =============================================================================
#  CADET
#  
#  Copyright © 2008-2022: The CADET Authors
#            Please see the AUTHORS and CONTRIBUTORS file.
#  
#  All rights reserved. This program and the accompanying materials
#  are made available under the terms of the GNU Public License v3.0 (or, at
#  your option, any later version) which accompanies this distribution, and
#  is available at http://www.gnu.org/licenses/gpl.html
# =============================================================================
 
# Require a fairly new cmake version
cmake_minimum_required(VERSION 3.12)

# Prohibit in-source build
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
	message(FATAL_ERROR "In-source build prohibited.")
endif()

# Set module path in order to use custom CMake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

find_package(Git)

# Write the current version number to variable
if (GIT_FOUND)
	if (EXISTS "${CMAKE_SOURCE_DIR}/.git")
		execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=0 HEAD
		                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		                OUTPUT_VARIABLE CADET_VERSION
		                OUTPUT_STRIP_TRAILING_WHITESPACE)

		if (NOT "${CADET_VERSION}" STREQUAL "")
			message(STATUS "Get version from git")

			# Remove first character ("v")
			string(LENGTH "${CADET_VERSION}" CADET_VERSION_STRLEN)
			math(EXPR CADET_VERSION_STRLEN "${CADET_VERSION_STRLEN}-1")
			string(SUBSTRING "${CADET_VERSION}" 1 ${CADET_VERSION_STRLEN}  CADET_VERSION)
		endif()
	endif()
endif()

# In case of missing tags, default to versions.txt file
if ("${CADET_VERSION}" STREQUAL "")
	message(STATUS "Get version from file")
	file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" CADET_VERSION)
endif()

message(STATUS "CADET version: ${CADET_VERSION}")

# Get current commit hash from git
if (GIT_FOUND)
	include(GetGitRevisionDescription)
	get_git_head_revision(GIT_REFSPEC GIT_SHA1)
endif()
if (NOT DEFINED GIT_SHA1)
	set(GIT_SHA1 "NO-COMMIT-HASH")
	set(GIT_REFSPEC "NO-REFSPEC")
endif()
message(STATUS "Current git HEAD: ${GIT_REFSPEC} SHA1 ${GIT_SHA1}")

# Name of the current project
project(CadetFramework
	VERSION ${CADET_VERSION}
	DESCRIPTION "Liquid column chromatography simulator"
	HOMEPAGE_URL "https://github.com/modsim/CADET"
	LANGUAGES CXX C)

# Always use '-fPIC'/'-fPIE' option
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Hide symbols by default
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Enable folders for IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ---------------------------------------------------
#   Other configuration options
# ---------------------------------------------------

include(FeatureSummary)

option(ENABLE_LOGGING "Enables logging" ON)
add_feature_info(ENABLE_LOGGING ENABLE_LOGGING "Enables logging")

option(ENABLE_BENCHMARK "Enables benchmark mode (fine-grained timing)" OFF)
add_feature_info(ENABLE_BENCHMARK ENABLE_BENCHMARK "Enables benchmark mode (fine-grained timing)")

option(ENABLE_PLATFORM_TIMER "Use a platform-dependent timer" OFF)
add_feature_info(ENABLE_PLATFORM_TIMER ENABLE_PLATFORM_TIMER "Use a platform-dependent timer")

option(ENABLE_THREADING "Use multi-threading" ON)
add_feature_info(ENABLE_THREADING ENABLE_THREADING "Use multi-threading")

option(ENABLE_DEBUG_THREADING "Use multi-threading in debug builds" OFF)
add_feature_info(ENABLE_DEBUG_THREADING ENABLE_DEBUG_THREADING "Use multi-threading in debug builds")

option(ENABLE_2D_MODELS "Build 2D models (e.g., 2D general rate model, multichannel transport)" ON)
add_feature_info(ENABLE_2D_MODELS ENABLE_2D_MODELS "Build 2D models (e.g., 2D general rate model, multichannel transport)")

option(ENABLE_SUNDIALS_OPENMP "Prefer OpenMP vector implementation of SUNDIALS if available (for large problems)" OFF)
add_feature_info(ENABLE_SUNDIALS_OPENMP ENABLE_SUNDIALS_OPENMP "Prefer OpenMP vector implementation of SUNDIALS if available (for large problems)")

option(ENABLE_ANALYTIC_JACOBIAN_CHECK "Enable verification of analytical Jacobian by AD" OFF)
add_feature_info(ENABLE_ANALYTIC_JACOBIAN_CHECK ENABLE_ANALYTIC_JACOBIAN_CHECK "Enable verification of analytical Jacobian by AD")

set(ADLIB "sfad" CACHE STRING "Selects the AD library, options are 'sfad', 'setfad'")
string(TOLOWER ${ADLIB} ADLIB)


option(ENABLE_CADET_CLI "Build CADET command line interface" ON)
add_feature_info(ENABLE_CADET_CLI ENABLE_CADET_CLI "Build CADET command line interface")

option(ENABLE_CADET_MEX "Build CADET Matlab MEX interface" ON)
add_feature_info(ENABLE_CADET_MEX ENABLE_CADET_MEX "Build CADET Matlab MEX interface")

option(ENABLE_CADET_TOOLS "Build CADET tools" ON)
add_feature_info(ENABLE_CADET_TOOLS ENABLE_CADET_TOOLS "Build CADET tools")

option(ENABLE_TESTS "Build CADET tests" ON)
add_feature_info(ENABLE_TESTS ENABLE_TESTS "Build CADET tests")

option(ENABLE_PACKAGED_SUNDIALS "Use packaged SUNDIALS code" ON)
add_feature_info(ENABLE_PACKAGED_SUNDIALS ENABLE_PACKAGED_SUNDIALS "Use packaged SUNDIALS code")

option(ENABLE_IPO "Enable interprocedural optimization if compiler supports it" ON)
add_feature_info(ENABLE_IPO ENABLE_IPO "Enable interprocedural optimization if compiler supports it")

option(ENABLE_STATIC_LINK_DEPS "Prefer static over dynamic linking of dependencies" OFF)
add_feature_info(ENABLE_STATIC_LINK_DEPS ENABLE_STATIC_LINK_DEPS "Prefer static over dynamic linking of dependencies")

option(ENABLE_STATIC_LINK_LAPACK "Prefer static over dynamic linking of LAPACK and BLAS" OFF)
add_feature_info(ENABLE_STATIC_LINK_LAPACK ENABLE_STATIC_LINK_LAPACK "Prefer static over dynamic linking of LAPACK and BLAS")

option(ENABLE_STATIC_LINK_CLI "Prefer static over dynamic linking for CADET CLI" OFF)
add_feature_info(ENABLE_STATIC_LINK_CLI ENABLE_STATIC_LINK_CLI "Prefer static over dynamic linking for CADET CLI")

option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath" ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath")

# Hande RPATH on OSX when not installing to a system directory, see
# https://groups.google.com/d/msg/fenics-dev/KSCrob4M_1M/zsJwdN-SCAAJ
# and https://cmake.org/Wiki/CMake_RPATH_handling#Always_full_RPATH
if (UNIX)
	# The RPATH to be used when installing, but only if it's not a system directory
	set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
	list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
	if ("${isSystemDir}" STREQUAL "-1")
		set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
	endif()
endif()

# Option that allows users to build release or debug version
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel" FORCE)
	message(STATUS "Build type: ${CMAKE_BUILD_TYPE} (default)")
endif()

# ---------------------------------------------------
#   Check build environment
# ---------------------------------------------------
include(WriteCompilerDetectionHeader)

set(TBB_TARGET "")
if (ENABLE_THREADING)
	set(TBB_PREFER_STATIC_LIBS ${ENABLE_STATIC_LINK_DEPS})
	set(TBB_USE_DEBUG_BUILD OFF)
	find_package(TBB COMPONENTS tbb OPTIONAL_COMPONENTS tbb_preview)
	set_package_properties(TBB PROPERTIES
		TYPE RECOMMENDED
		PURPOSE "Accelerates computation via multi-threading"
	)

	set(CADET_PARALLEL_FLAG "")
	if (TBB_FOUND)
		# Use tbb_preview instead of tbb if it has been found
		if (TBB_tbb_preview_FOUND)
			set(TBB_TARGET "TBB::TBBpreview")
		else()
			set(TBB_TARGET "TBB::TBB")
		endif()

		if (CMAKE_BUILD_TYPE STREQUAL "Debug")
			if (ENABLE_DEBUG_THREADING)
				set(CADET_PARALLEL_FLAG "CADET_PARALLELIZE")
			endif()
		else()
			set(CADET_PARALLEL_FLAG "CADET_PARALLELIZE")
		endif()

		get_target_property(TBB_IFACE_COMP_DEF ${TBB_TARGET} INTERFACE_COMPILE_DEFINITIONS)
		if (TBB_IFACE_COMP_DEF)
			list(APPEND TBB_IFACE_COMP_DEF ${CADET_PARALLEL_FLAG})
		else()
			set(TBB_IFACE_COMP_DEF ${CADET_PARALLEL_FLAG})
		endif()
		if (TBB_IFACE_COMP_DEF)
			set_target_properties(${TBB_TARGET} PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${TBB_IFACE_COMP_DEF}")
		endif()
		unset(TBB_IFACE_COMP_DEF)

		if (TBB_INTERFACE_VERSION GREATER_EQUAL 11004)
			# Use global_control instead of task_scheduler_init
			get_target_property(TBB_IFACE_COMP_DEF ${TBB_TARGET} INTERFACE_COMPILE_DEFINITIONS)
			set_target_properties(${TBB_TARGET} PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${TBB_IFACE_COMP_DEF}")
			if (TBB_IFACE_COMP_DEF)
				list(APPEND TBB_IFACE_COMP_DEF "CADET_TBB_GLOBALCTRL")
			else()
				set(TBB_IFACE_COMP_DEF "CADET_TBB_GLOBALCTRL")
			endif()
			if (TBB_IFACE_COMP_DEF)
				set_target_properties(${TBB_TARGET} PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${TBB_IFACE_COMP_DEF}")
			endif()
			unset(TBB_IFACE_COMP_DEF)
		endif()
	endif()
endif()

set(BLA_STATIC ${ENABLE_STATIC_LINK_LAPACK})
find_package(LAPACK)
set_package_properties(LAPACK PROPERTIES
	TYPE RECOMMENDED
	PURPOSE "Solution of dense linear systems"
)

if (NOT ENABLE_PACKAGED_SUNDIALS)
	# SUNDIALS_ROOT environment variable can be used to find SUNDIALS package
	set(SUNDIALS_PREFER_STATIC_LIBRARIES ${ENABLE_STATIC_LINK_DEPS})
	find_package(SUNDIALS REQUIRED COMPONENTS sundials_idas sundials_nvecserial OPTIONAL_COMPONENTS sundials_nvecopenmp)
	set_package_properties(SUNDIALS PROPERTIES
		TYPE REQUIRED
		PURPOSE "Time integration"
	)

	# Check whether OpenMP is available in SUNDIAL'S NVECTOR module
	set(SUNDIALS_NVEC_TARGET "SUNDIALS::sundials_nvecserial")
	if (SUNDIALS_sundials_nvecopenmp_LIBRARY AND ENABLE_SUNDIALS_OPENMP)
		# Prefer OpenMP over serial version
		set(SUNDIALS_NVEC_TARGET "SUNDIALS::sundials_nvecopenmp")

		get_target_property(SUNDIALS_IFACE_COMP_DEF SUNDIALS::sundials_nvecopenmp INTERFACE_COMPILE_DEFINITIONS)
		if (SUNDIALS_IFACE_COMP_DEF)
			list(APPEND SUNDIALS_IFACE_COMP_DEF "CADET_SUNDIALS_OPENMP")
		else()
			set(SUNDIALS_IFACE_COMP_DEF "CADET_SUNDIALS_OPENMP")
		endif()
		set_target_properties(SUNDIALS::sundials_nvecopenmp PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${SUNDIALS_IFACE_COMP_DEF}")
		unset(SUNDIALS_IFACE_COMP_DEF)
	endif()

	# Determine SUNDIALS interface version
	if (SUNDIALS_FOUND)
		get_target_property(SUNDIALS_IFACE_COMP_DEF SUNDIALS::sundials_idas INTERFACE_COMPILE_DEFINITIONS)
		if (SUNDIALS_IFACE_COMP_DEF)
			list(APPEND SUNDIALS_IFACE_COMP_DEF "CADET_SUNDIALS_IFACE=${SUNDIALS_VERSION_MAJOR}")
		else()
			set(SUNDIALS_IFACE_COMP_DEF "CADET_SUNDIALS_IFACE=${SUNDIALS_VERSION_MAJOR}")
		endif()
		set_target_properties(SUNDIALS::sundials_idas PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${SUNDIALS_IFACE_COMP_DEF}")
		unset(SUNDIALS_IFACE_COMP_DEF)
	endif()
else()
	set(SUNDIALS_FOUND TRUE)
	set(SUNDIALS_VERSION "3.2.1")
	set(SUNDIALS_NVEC_TARGET "SUNDIALS::sundials_nvecserial")
	add_subdirectory(ThirdParty/sundials)

	add_library(SUNDIALS::sundials_idas INTERFACE IMPORTED)
	target_link_libraries(SUNDIALS::sundials_idas INTERFACE sundials_idas_static)
	target_include_directories(SUNDIALS::sundials_idas INTERFACE ThirdParty/sundials/include ThirdParty/sundials/src "${CMAKE_BINARY_DIR}/ThirdParty/sundials/include")
	set_target_properties(SUNDIALS::sundials_idas PROPERTIES INTERFACE_COMPILE_DEFINITIONS "CADET_SUNDIALS_IFACE=3")

	add_library(SUNDIALS::sundials_nvecserial INTERFACE IMPORTED)
	target_include_directories(SUNDIALS::sundials_nvecserial INTERFACE ThirdParty/sundials/include "${CMAKE_BINARY_DIR}/ThirdParty/sundials/include")
	target_link_libraries(SUNDIALS::sundials_nvecserial INTERFACE sundials_nvecserial_static)
endif()

if (ENABLE_CADET_TOOLS OR ENABLE_CADET_CLI)
	set(HDF5_USE_STATIC_LIBRARIES ${ENABLE_STATIC_LINK_DEPS})
	find_package(HDF5 COMPONENTS C)
	set_package_properties(HDF5 PROPERTIES
		DESCRIPTION "Hierarchical Data Format 5 (HDF5)"
		URL "https://www.hdfgroup.org/HDF5"
		TYPE RECOMMENDED
		PURPOSE "File IO"
	)

	if (HDF5_FOUND)

		# Create custom HDF5 target if CMake's FindHDF5 is too old
		if (NOT TARGET HDF5::HDF5)
			list(LENGTH HDF5_C_LIBRARIES HDF5_C_LEN)
			if (HDF5_C_LEN GREATER 1)
				list(GET HDF5_C_LIBRARIES 0 HDF5_MAIN_LIBRARY)
				set(HDF5_SUPPORT_LIBRARIES ${HDF5_C_LIBRARIES})
				list(REMOVE_AT HDF5_SUPPORT_LIBRARIES 0)
			else()
				set(HDF5_MAIN_LIBRARY ${HDF5_C_LIBRARIES}) 
				set(HDF5_SUPPORT_LIBRARIES)
			endif()

			add_library(HDF5::HDF5 UNKNOWN IMPORTED)
			set_target_properties(HDF5::HDF5 PROPERTIES
				IMPORTED_LOCATION ${HDF5_MAIN_LIBRARY}
				INTERFACE_INCLUDE_DIRECTORIES "${HDF5_C_INCLUDE_DIRS}"
#				INTERFACE_COMPILE_DEFINITIONS ${HDF5_C_DEFINITIONS}
			)

			if (HDF5_SUPPORT_LIBRARIES)
				target_link_libraries(HDF5::HDF5 INTERFACE ${HDF5_SUPPORT_LIBRARIES})
			endif()

			unset(HDF5_SUPPORT_LIBRARIES)
			unset(HDF5_MAIN_LIBRARY)
			unset(HDF5_C_LEN)
		endif()

		# Make sure HDF5_LIBRARY_DIRS is defined
		if ((NOT DEFINED HDF5_LIBRARY_DIRS) OR (NOT HDF5_LIBRARY_DIRS) OR ("${HDF5_LIBRARY_DIRS}" STREQUAL ""))
			list(GET HDF5_LIBRARIES 0 HDF5_LIB_TEMP)
			get_filename_component(HDF5_LIBRARY_DIRS ${HDF5_LIB_TEMP} DIRECTORY)
			unset(HDF5_LIB_TEMP)
		endif()

		# Check if we need additional libraries for linking (i.e., zlib, szip)
		include(${CMAKE_ROOT}/Modules/CheckCXXSourceCompiles.cmake)
		include(${CMAKE_ROOT}/Modules/CMakePushCheckState.cmake)

		cmake_push_check_state(RESET)

		# Set libs and includes
		set(CMAKE_REQUIRED_LIBRARIES ${HDF5_LIBRARIES})
		set(CMAKE_REQUIRED_INCLUDES ${HDF5_INCLUDE_DIRS})

		CHECK_CXX_SOURCE_COMPILES("#include <hdf5.h>\nint main(int argc, char** argv){\n H5Zfilter_avail(H5Z_FILTER_SZIP);\nH5Zfilter_avail(H5Z_FILTER_DEFLATE);\nreturn 0;\n}\n" HDF5_DONT_NEED_ZLIBS)

		# Reset libs and includes
		cmake_pop_check_state()

		# Find szip and zlib libs if we need them
		if (NOT HDF5_DONT_NEED_ZLIBS)

			# Prefer static libs if enabled
			set(_HDF5_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
			if(ENABLE_STATIC_LINK_DEPS)
				if(WIN32)
					set(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
				else()
					set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
				endif()
			endif()

			find_library(HDF5_SZLIB NAMES libszip szip PATHS ${HDF5_LIBRARY_DIRS})
			find_library(HDF5_ZLIB NAMES libzlib zlib PATHS ${HDF5_LIBRARY_DIRS})

			if (HDF5_SZLIB)
				list(APPEND HDF5_LIBRARIES ${HDF5_SZLIB})
				add_library(HDF5::SZLIB UNKNOWN IMPORTED)
				set_target_properties(HDF5::SZLIB PROPERTIES IMPORTED_LOCATION ${HDF5_SZLIB})
				target_link_libraries(HDF5::HDF5 INTERFACE HDF5::SZLIB)
			endif()
			if (HDF5_ZLIB)
				list(APPEND HDF5_LIBRARIES ${HDF5_ZLIB})
				add_library(HDF5::ZLIB UNKNOWN IMPORTED)
				set_target_properties(HDF5::ZLIB PROPERTIES IMPORTED_LOCATION ${HDF5_ZLIB})
				target_link_libraries(HDF5::HDF5 INTERFACE HDF5::ZLIB)
			endif()
			unset(HDF5_SZLIB)
			unset(HDF5_ZLIB)

			set(CMAKE_FIND_LIBRARY_SUFFIXES ${_HDF5_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
			unset(_HDF5_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
		endif()
	endif()
endif()

if (ENABLE_2D_MODELS)
	set(SUPERLU_PREFER_STATIC_LIBS ${ENABLE_STATIC_LINK_DEPS})
	find_package(SuperLU)
	set_package_properties(SuperLU PROPERTIES
		TYPE RECOMMENDED
		PURPOSE "Sparse matrix solver"
	)

	set(UMFPACK_PREFER_STATIC_LIBS ${ENABLE_STATIC_LINK_DEPS})
	find_package(UMFPACK)
	set_package_properties(UMFPACK PROPERTIES
		TYPE RECOMMENDED
		PURPOSE "Sparse matrix solver"
	)
endif()

if (ENABLE_CADET_MEX)

	# Set Matlab_ROOT_DIR from MATLAB_ROOT environment variable
	if (DEFINED ENV{MATLAB_ROOT})
		set(Matlab_ROOT_DIR $ENV{MATLAB_ROOT})
	endif()
	find_package(Matlab COMPONENTS MX_LIBRARY LAPACK_LIBRARY OPTIONAL_COMPONENTS TBB_LIBRARY UMFPACK_LIBRARY)
	set_package_properties(Matlab PROPERTIES
	    URL "https://www.mathworks.com/products/matlab.html"
	    DESCRIPTION "MATLAB Scientific Computing Environment"
		TYPE RECOMMENDED
		PURPOSE "Native MATLAB interface"
	)

	option(MATLAB_USE_SYSTEM_LAPACK "Forcefully enables usage of system's LAPACK instead of MATLAB's one" ON)
	add_feature_info(MATLAB_USE_SYSTEM_LAPACK MATLAB_USE_SYSTEM_LAPACK "Forcefully enables usage of system's LAPACK instead of MATLAB's one")

	option(MATLAB_FORCE_UMFPACK "Forcefully use MATLAB's UMFPACK as sole sparse solver in MEX interface" OFF)
	add_feature_info(MATLAB_FORCE_UMFPACK MATLAB_FORCE_UMFPACK "Forcefully use MATLAB's UMFPACK as sole sparse solver in MEX interface")

	option(MATLAB_ENABLE_THREADING "Enable multi-threading in MATLAB" ON)
	add_feature_info(MATLAB_ENABLE_THREADING MATLAB_ENABLE_THREADING "Enable multi-threading in MATLAB")

	# Check TBB <-> Matlab compatibility
	option(MATLAB_FORCE_ENABLE_TBB "Forcefully enable usage of MATLAB's TBB library in case system TBB is not compatible and not enforced" OFF)
	add_feature_info(MATLAB_FORCE_ENABLE_TBB MATLAB_FORCE_ENABLE_TBB "Forcefully enable usage of MATLAB's TBB library in case system TBB is not compatible and not enforced")

	option(MATLAB_USE_SYSTEM_TBB "Forcefully enables usage of system's TBB instead of MATLAB's one" ON)
	add_feature_info(MATLAB_USE_SYSTEM_TBB MATLAB_USE_SYSTEM_TBB "Forcefully enables usage of system TBB instead of MATLAB's one")

	if (Matlab_FOUND)

		if (Matlab_TBB_LIBRARY_FOUND)
			# Combine system TBB headers with MATLAB TBB binary (dangerous as versions certainly don't match)
			add_library(MATLAB::TBB UNKNOWN IMPORTED)
			set_target_properties(MATLAB::TBB PROPERTIES 
				INTERFACE_COMPILE_DEFINITIONS "__TBB_NO_IMPLICIT_LINKAGE=1;${CADET_PARALLEL_FLAG}"
				IMPORTED_LOCATION             "${Matlab_TBB_LIBRARY}"
				INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIRS}"
			)
		endif()

		set(Matlab_TBB_TARGET "")
		if (MATLAB_ENABLE_THREADING)
			if (TBB_tbb_preview_FOUND OR (TBB_FOUND AND MATLAB_USE_SYSTEM_TBB))
				# Linking to tbb_preview usually works
				set(Matlab_TBB_TARGET ${TBB_TARGET})
			else()
				if (Matlab_TBB_LIBRARY_FOUND AND TBB_FOUND)
					if (TBB_INTERFACE_VERSION LESS_EQUAL Matlab_TBB_INTERFACE_VERSION)
						# System TBB is older than or same age as Matlab's TBB
						set(Matlab_TBB_TARGET "MATLAB::TBB")
					else()
						if (MATLAB_FORCE_ENABLE_TBB)
							set(Matlab_TBB_TARGET "MATLAB::TBB")
						else()
							message(WARNING "Matlab's TBB version is older (${Matlab_TBB_INTERFACE_VERSION}) than the system headers (${TBB_INTERFACE_VERSION}). TBB parallelization is disabled in MEX interface. Set -DMATLAB_FORCE_ENABLE_TBB=ON to enforce TBB support.")
						endif()
					endif()
				endif()
			endif()
		endif()
	else()
		message(STATUS "Matlab not found, disabling ENABLE_CADET_MEX")
		set(ENABLE_CADET_MEX OFF)
	endif()

endif()

set(IPO_AVAILABLE OFF)
if (ENABLE_IPO)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT IPO_RESULT OUTPUT IPO_OUT LANGUAGES CXX)
	if (IPO_RESULT)
		set(IPO_AVAILABLE ON)
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
	else()
		message(WARNING "IPO is not supported: ${IPO_OUT}")
	endif()
	unset(IPO_RESULT)
	unset(IPO_OUT)
endif()


# ---------------------------------------------------
#   Add selected modules to the build system and add the targets to the list of all targets
# ---------------------------------------------------

add_library(CADET::CompileOptions INTERFACE IMPORTED)
target_compile_features(CADET::CompileOptions INTERFACE cxx_std_17)
set(CMAKE_CXX_EXTENSIONS OFF)

if (WIN32)
	target_compile_definitions(CADET::CompileOptions INTERFACE NOMINMAX)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	target_compile_definitions(CADET::CompileOptions INTERFACE CADET_LOGLEVEL_MIN=Trace DEBUG _DEBUG)
else()
	target_compile_definitions(CADET::CompileOptions INTERFACE CADET_LOGLEVEL_MIN=Warning NDEBUG)
endif()

if (ENABLE_BENCHMARK)
	target_compile_definitions(CADET::CompileOptions INTERFACE CADET_BENCHMARK_MODE)
endif()

if (ENABLE_PLATFORM_TIMER)
	target_compile_definitions(CADET::CompileOptions INTERFACE CADET_USE_PLATFORM_TIMER)
	if ((NOT APPLE) AND (NOT WIN32))
		target_link_libraries(CADET::CompileOptions INTERFACE rt)
	endif()
endif()

target_compile_options(CADET::CompileOptions INTERFACE $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
		-Wall -pedantic-errors -Wextra -Wno-unused-parameter -Wno-unused-function> #-Wconversion -Wsign-conversion
	$<$<CXX_COMPILER_ID:MSVC>:
		/W4>
)

add_library(CADET::AD INTERFACE IMPORTED)
if (ADLIB STREQUAL "sfad")
	message(STATUS "AD library: SFAD")
	target_compile_definitions(CADET::AD INTERFACE ACTIVE_SFAD)
	target_include_directories(CADET::AD INTERFACE "${CMAKE_SOURCE_DIR}/include/ad")
elseif (ADLIB STREQUAL "setfad")
	message(STATUS "AD library: SETFAD")
	target_compile_definitions(CADET::AD INTERFACE ACTIVE_SETFAD)
	target_include_directories(CADET::AD INTERFACE "${CMAKE_SOURCE_DIR}/include/ad")
else()
	message(FATAL_ERROR "Unkown AD library ${ADLIB} (options are 'sfad', 'setfad')")
endif()


# Build tools
add_subdirectory(src/build-tools)

# CADET library
add_subdirectory(src/libcadet)

if (ENABLE_CADET_CLI)
	add_subdirectory(src/cadet-cli)
endif()

if (ENABLE_CADET_MEX)
	add_subdirectory(src/cadet-mex)
endif()

if (ENABLE_CADET_TOOLS)
	add_subdirectory(src/tools)
endif()

if (ENABLE_TESTS)
	add_subdirectory(test)

	# Add "make check" target
	include(ProcessorCount)
	ProcessorCount(NPROC)
	add_custom_target(check COMMAND test/testRunner -d yes --tbbthreads ${NPROC})
endif()


# ---------------------------------------------------
#   Set properties, definitions, install target etc.
# ---------------------------------------------------

# Packaging support
include(CPack)
set(CPACK_PACKAGE_VENDOR "CADET")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_STRIP_FILES ON)

# Combine LICENSE.txt and LICENSE-ThirdParty.txt
file(WRITE "${CMAKE_BINARY_DIR}/LICENSE.txt" "####################################################################################\n")
file(APPEND "${CMAKE_BINARY_DIR}/LICENSE.txt" "##                                      CADET                                     ##\n")
file(APPEND "${CMAKE_BINARY_DIR}/LICENSE.txt" "####################################################################################\n")

file(READ "${CMAKE_SOURCE_DIR}/LICENSE.txt" LICENSE_TEXT)
file(APPEND "${CMAKE_BINARY_DIR}/LICENSE.txt" "\n${LICENSE_TEXT}")

file(READ "${CMAKE_SOURCE_DIR}/LICENSE-ThirdParty.txt" LICENSE_THIRDPARTY)
file(APPEND "${CMAKE_BINARY_DIR}/LICENSE.txt" "\n\n${LICENSE_THIRDPARTY}")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/LICENSE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

set(CPACK_SOURCE_IGNORE_FILES
	/.git
	/\\\\.DS_Store
)

message("")
message("--------------------------- Feature Summary ---------------------------")
         
feature_summary(WHAT ALL)

# Summary
message("")
message("------------------------------- Summary -------------------------------")
message("C++ compiler name: ${CMAKE_CXX_COMPILER_ID} at ${CMAKE_CXX_COMPILER}")
message("Build type: ${CMAKE_BUILD_TYPE}")
message("Source dir: ${CMAKE_SOURCE_DIR}")
message("Binary dir: ${CMAKE_BINARY_DIR}")
message("Install dir: ${CMAKE_INSTALL_PREFIX}")
message("C Flags: ${CMAKE_C_FLAGS}")
message("C++ Flags: ${CMAKE_CXX_FLAGS}")
message("IPO enabled: ${IPO_AVAILABLE}")
message("------------------------------- Modules -------------------------------")
message("CADET-CLI: ${ENABLE_CADET_CLI}")
message("CADET-MEX: ${ENABLE_CADET_MEX}")
message("Tools: ${ENABLE_CADET_TOOLS}")
message("Tests: ${ENABLE_TESTS}")
message("------------------------------- Options -------------------------------")
message("Logging: ${ENABLE_LOGGING}")
message("Benchmark mode: ${ENABLE_BENCHMARK}")
message("Platform-dependent timer: ${ENABLE_PLATFORM_TIMER}")
message("AD library: ${ADLIB}")
message("2D Models: ${ENABLE_2D_MODELS}")
message("Check analytic Jacobian: ${ENABLE_ANALYTIC_JACOBIAN_CHECK}")
message("----------------------------- Dependencies ----------------------------")

message("Found BLAS: ${BLAS_FOUND}")
if (BLAS_FOUND)
	message("  Linker flags ${BLAS_LINKER_FLAGS}")
	message("  Libs ${BLAS_LIBRARIES}")
	message("  Underscore suffix ${BLAS_UNDERSCORE_SUFFIX}")
endif()

message("Found LAPACK: ${LAPACK_FOUND}")
if (LAPACK_FOUND)
	message("  Linker flags ${LAPACK_LINKER_FLAGS}")
	message("  Libs ${LAPACK_LIBRARIES}")
endif()

message("Found TBB: ${TBB_FOUND}")
if (TBB_FOUND)
	message("  Version ${TBB_VERSION} (Interface ${TBB_INTERFACE_VERSION})")
	message("  Include ${TBB_INCLUDE_DIRS}")
	message("  Definitions ${TBB_DEFINITIONS}")
	message("  Libs ${TBB_LIBRARIES}")
endif()

if (ENABLE_PACKAGED_SUNDIALS)
	message("Found SUNDIALS: ${SUNDIALS_FOUND}")
	message("  Version ${SUNDIALS_VERSION}")
	message("  Packaged")
else()
	message("Found SUNDIALS: ${SUNDIALS_FOUND}")
	if (SUNDIALS_FOUND)
		message("  Version ${SUNDIALS_VERSION}")
		message("  Includes ${SUNDIALS_INCLUDE_DIRS}")
		message("  Libs ${SUNDIALS_LIBRARIES}")
	endif()
endif()

if (ENABLE_2D_MODELS)
	message("Found SuperLU: ${SUPERLU_FOUND}")
	if (SUPERLU_FOUND)
		message("  Version ${SUPERLU_VERSION}")
		message("  Includes ${SUPERLU_INCLUDE_DIRS}")
		message("  Libs ${SUPERLU_LIBRARIES}")
		message("  Integer type ${SUPERLU_INT_TYPE}")
	endif()
	message("Found UMFPACK: ${UMFPACK_FOUND}")
	if (UMFPACK_FOUND)
		message("  Version ${UMFPACK_VERSION}")
		message("  Includes ${UMFPACK_INCLUDE_DIRS}")
		message("  Libs ${UMFPACK_LIBRARIES}")
	endif()
endif()

message("Found HDF5: ${HDF5_FOUND}")
if (HDF5_FOUND)
	message("  Version ${HDF5_VERSION}")
	message("  Includes ${HDF5_INCLUDE_DIRS}")
	message("  Libs ${HDF5_LIBRARIES}")
	message("  Defs ${HDF5_C_DEFINITIONS}")
endif()

if (ENABLE_CADET_MEX)
	message("Found Matlab: ${Matlab_FOUND}")
	if (Matlab_FOUND)
		message("  Version ${Matlab_VERSION_STRING}")
		message("  In ${Matlab_ROOT_DIR}")
		message("  Includes ${Matlab_INCLUDE_DIRS}")
		message("  Libs ${Matlab_LIBRARIES}")
		message("  MKL ${Matlab_LAPACK_LIBRARY} ${Matlab_BLAS_LIBRARY}")
		message("  LAPACK functions have trailing underscore: ${Matlab_LAPACK_TRAILING_UNDERSCORE}")
		message("  Declares mxCreateUninitNumericMatrix: ${Matlab_DECLARES_CREATEUNINITNUMERICMATRIX}")
		message("  Declares mxCreateUninitNumericArray: ${Matlab_DECLARES_CREATEUNINITNUMERICARRAY}")
		message("  TBB ${Matlab_TBB_LIBRARY} (interface version ${Matlab_TBB_INTERFACE_VERSION})")
		message("  Parallelization (TBB compatibility): ${Matlab_TBB_COMPATIBLE}")
		message("  UMFPACK ${Matlab_UMFPACK_LIBRARY} (enforced ${MATLAB_FORCE_UMFPACK})")
	endif()
endif()
message("-----------------------------------------------------------------------")
message("")
