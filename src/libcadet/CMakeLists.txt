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

# Name of the current project
project(CadetLibrary CXX C)

# Git integration
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/VersionInfo.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/VersionInfo.cpp" ESCAPE_QUOTES @ONLY)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/CompileTimeConfig.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/CompileTimeConfig.hpp" @ONLY)

# Compiler features
write_compiler_detection_header(
	FILE "${CMAKE_BINARY_DIR}/cadet/cadetCompilerInfo.hpp"
	PREFIX CADET
	COMPILERS GNU Clang AppleClang MSVC Intel
	FEATURES cxx_noexcept cxx_user_literals cxx_constexpr cxx_variadic_templates
	PROLOG "\
// =============================================================================\n\
//  CADET\n\
//  \n\
//  Copyright © 2008-2022: The CADET Authors\n\
//            Please see the AUTHORS and CONTRIBUTORS file.\n\
//  \n\
//  All rights reserved. This program and the accompanying materials\n\
//  are made available under the terms of the GNU Public License v3.0 (or, at\n\
//  your option, any later version) which accompanies this distribution, and\n\
//  is available at http://www.gnu.org/licenses/gpl.html\n\
// =============================================================================\n"
	EPILOG "\
#if CADET_COMPILER_CXX_CONSTEXPR\n\
	#define CADET_CONST_OR_CONSTEXPR constexpr\n\
#else\n\
	#define CADET_CONST_OR_CONSTEXPR const\n\
#endif\n\
#if CADET_COMPILER_CXX_USER_LITERALS && CADET_COMPILER_CXX_CONSTEXPR\n\
	#define CADET_COMPILETIME_HASH 1\n\
#else\n\
	#define CADET_COMPILETIME_HASH 0\n\
#endif\n"
)

# Check for some compiler bugs
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

cmake_push_check_state(RESET)

# Enable C++11
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	if (WIN32)
		set(CMAKE_REQUIRED_FLAGS "/Qstd=c++11")
	else()
		set(CMAKE_REQUIRED_FLAGS "-std=c++11")
	endif()
elseif ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
	set(CMAKE_REQUIRED_FLAGS "-std=c++11")
elseif ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_REQUIRED_FLAGS "-std=c++11")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	set(CMAKE_REQUIRED_FLAGS "")
endif ()

set(CMAKE_REQUIRED_INCLUDES "${CMAKE_BINARY_DIR}/cadet/")
CHECK_CXX_SOURCE_COMPILES("#include \"cadetCompilerInfo.hpp\" \n class Test { public: Test() { }\n Test& operator=(Test&& cpy) CADET_NOEXCEPT = default; \n private: \n int a; }; int main(int argc, char** argv) { return 0; }" COMPILER_SUPPORT_NOEXCEPT_DEFAULTED_MOVE)

cmake_pop_check_state()

# LIBCADET_BINDINGMODEL_SOURCES holds all source files of binding models
set(LIBCADET_BINDINGMODEL_SOURCES
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/DummyBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/LinearBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/StericMassActionBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/LangmuirBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/AntiLangmuirBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/BiLangmuirBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/KumarLangmuirBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/MobilePhaseModulatorLangmuirBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/ExtendedMobilePhaseModulatorLangmuirBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/SelfAssociationBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/BiStericMassActionBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/SaskaBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/MultiStateStericMassActionBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/SimplifiedMultiStateStericMassActionBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/MultiComponentSpreadingBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/GeneralizedIonExchangeBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/ColloidalBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/FreundlichLDFBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/LangmuirLDFBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/LangmuirLDFCBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/BiLangmuirLDFBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/HICWaterOnHydrophobicSurfacesBinding.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/HICConstantWaterActivityBinding.cpp
)

# LIBCADET_REACTIONMODEL_SOURCES holds all source files of reaction models
set(LIBCADET_REACTIONMODEL_SOURCES
	${CMAKE_SOURCE_DIR}/src/libcadet/model/reaction/DummyReaction.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/reaction/MassActionLawReaction.cpp
)

# LIBCADET_SOURCES holds all source files for LIBCADET target
set(LIBCADET_SOURCES
	${CMAKE_CURRENT_BINARY_DIR}/VersionInfo.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/Logging.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/api/CAPIv1.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/FactoryFuncs.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/ModelBuilderImpl.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/SimulatorImpl.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/AutoDiff.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/AdUtils.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/Weno.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/BindingModelFactory.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/ReactionModelFactory.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/ParameterDependenceFactory.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/graph/GraphAlgos.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/ModelSystemImpl.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/ModelSystemImpl-Residual.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/ModelSystemImpl-LinearSolver.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/ModelSystemImpl-InitialConditions.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/UnitOperationBase.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/InletModel.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/OutletModel.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/StirredTankModel.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/LumpedRateModelWithoutPores.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/LumpedRateModelWithPores.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/LumpedRateModelWithPores-LinearSolver.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/LumpedRateModelWithPores-InitialConditions.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/GeneralRateModel.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/GeneralRateModel-LinearSolver.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/GeneralRateModel-InitialConditions.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/ParameterMultiplexing.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/parts/ConvectionDispersionOperator.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/parts/ConvectionDispersionKernel.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/reaction/ReactionModelBase.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/BindingModelBase.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/inlet/PiecewiseCubicPoly.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/extfun/LinearInterpolationExternalFunction.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/extfun/PiecewiseCubicPolyExternalFunction.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/paramdep/ParameterDependenceBase.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/paramdep/LiquidSaltSolidParameterDependence.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/model/paramdep/DummyParameterDependence.cpp
)

# LIBCADET_NONLINALG_SOURCES holds all source files for LIBCADET_NONLINALG target
set (LIBCADET_NONLINALG_SOURCES
	${CMAKE_SOURCE_DIR}/src/libcadet/linalg/BandMatrix.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/linalg/DenseMatrix.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/linalg/SparseMatrix.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/linalg/CompressedSparseMatrix.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/linalg/Gmres.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/nonlin/AdaptiveTrustRegionNewton.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/nonlin/LevenbergMarquardt.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/nonlin/CompositeSolver.cpp
	${CMAKE_SOURCE_DIR}/src/libcadet/nonlin/Solver.cpp
)

if (ENABLE_2D_MODELS)
	set(LIBCADET_NONLINALG_SPARSE_SOURCES)
	if (SUPERLU_FOUND)
		list(APPEND LIBCADET_NONLINALG_SPARSE_SOURCES ${CMAKE_SOURCE_DIR}/src/libcadet/linalg/SuperLUSparseMatrix.cpp)
		set(SPARSE_INT_TYPE "${SUPERLU_INT_TYPE}")
	endif()
	if (UMFPACK_FOUND)
		list(APPEND LIBCADET_NONLINALG_SPARSE_SOURCES ${CMAKE_SOURCE_DIR}/src/libcadet/linalg/UMFPackSparseMatrix.cpp)
	endif()
	list (APPEND LIBCADET_SOURCES
		${CMAKE_SOURCE_DIR}/src/libcadet/model/parts/TwoDimensionalConvectionDispersionOperator.cpp
		${CMAKE_SOURCE_DIR}/src/libcadet/model/GeneralRateModel2D.cpp
		${CMAKE_SOURCE_DIR}/src/libcadet/model/GeneralRateModel2D-LinearSolver.cpp
		${CMAKE_SOURCE_DIR}/src/libcadet/model/GeneralRateModel2D-InitialConditions.cpp
		${CMAKE_SOURCE_DIR}/src/libcadet/model/parts/MultiChannelConvectionDispersionOperator.cpp
		${CMAKE_SOURCE_DIR}/src/libcadet/model/MultiChannelTransportModel.cpp
		${CMAKE_SOURCE_DIR}/src/libcadet/model/MultiChannelTransportModel-LinearSolver.cpp
		${CMAKE_SOURCE_DIR}/src/libcadet/model/MultiChannelTransportModel-InitialConditions.cpp
	)

	if (NOT SUPERLU_FOUND)
		set(SPARSE_INT_TYPE "int")
	endif()
else()
	set(SPARSE_INT_TYPE "int")
endif()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/linalg/SparseSolverInterface.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/SparseSolverInterface.hpp" @ONLY)

# Preprocess binding and reaction models
foreach(BM IN LISTS LIBCADET_BINDINGMODEL_SOURCES)
	get_filename_component(BMFILEWE ${BM} NAME_WE)
	get_filename_component(BMFILE ${BM} NAME)
	add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${BMFILE}
		COMMAND templateCodeGen ${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/ExternalFunctionTemplate.cpp ${BM} ${CMAKE_CURRENT_BINARY_DIR}/${BMFILE} 
		MAIN_DEPENDENCY ${BM}
		DEPENDS ${CMAKE_SOURCE_DIR}/src/libcadet/model/binding/ExternalFunctionTemplate.cpp
		COMMENT "Generating code for ${BMFILEWE}"
	)
	list (APPEND LIBCADET_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${BMFILE}")
endforeach()

foreach(RM IN LISTS LIBCADET_REACTIONMODEL_SOURCES)
	get_filename_component(RMFILEWE ${RM} NAME_WE)
	get_filename_component(RMFILE ${RM} NAME)
	add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${RMFILE}
		COMMAND templateCodeGen ${CMAKE_SOURCE_DIR}/src/libcadet/model/reaction/ExternalFunctionTemplate.cpp ${RM} ${CMAKE_CURRENT_BINARY_DIR}/${RMFILE} 
		MAIN_DEPENDENCY ${RM}
		DEPENDS ${CMAKE_SOURCE_DIR}/src/libcadet/model/reaction/ExternalFunctionTemplate.cpp
		COMMENT "Generating code for ${RMFILEWE}"
	)
	list (APPEND LIBCADET_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${RMFILE}")
endforeach()


add_library(CADET::LibOptions INTERFACE IMPORTED)
if (ENABLE_ANALYTIC_JACOBIAN_CHECK)
	target_compile_definitions(CADET::LibOptions INTERFACE CADET_CHECK_ANALYTIC_JACOBIAN)
endif()


set(LIBCADET_TARGETS)
if (LAPACK_FOUND)
	set(LIB_LAPACK_DEFINE "CADET_LAPACK_TRAILING_UNDERSCORE")

	# Add the build target for the static nonlinalg library
	add_library(libcadet_nonlinalg_static STATIC ${LIBCADET_NONLINALG_SOURCES} ${LIBCADET_NONLINALG_SPARSE_SOURCES})
	set_target_properties(libcadet_nonlinalg_static PROPERTIES OUTPUT_NAME cadet_nonlinalg_static)
	target_compile_definitions(libcadet_nonlinalg_static PRIVATE libcadet_nonlinalg_static_EXPORTS ${LIB_LAPACK_DEFINE})
	target_link_libraries(libcadet_nonlinalg_static PUBLIC CADET::CompileOptions PRIVATE SUNDIALS::sundials_idas ${SUNDIALS_NVEC_TARGET} ${LAPACK_LIBRARIES})

	if (ENABLE_2D_MODELS)
		if (SUPERLU_FOUND)
			target_link_libraries(libcadet_nonlinalg_static PRIVATE SuperLU::SuperLU)
		endif()
		if (UMFPACK_FOUND)
			target_link_libraries(libcadet_nonlinalg_static PRIVATE UMFPACK::UMFPACK)
		endif()
	endif()


	# Add the build target for CADET object library
	add_library(libcadet_object OBJECT ${LIBCADET_SOURCES})
	target_compile_definitions(libcadet_object PRIVATE libcadet_EXPORTS ${LIB_LAPACK_DEFINE})
	target_link_libraries(libcadet_object PUBLIC CADET::CompileOptions CADET::LibOptions PRIVATE CADET::AD libcadet_nonlinalg_static SUNDIALS::sundials_idas ${SUNDIALS_NVEC_TARGET} ${TBB_TARGET})

	# ---------------------------------------------------
	#   Build the static library
	# ---------------------------------------------------

	add_library(libcadet_static STATIC $<TARGET_OBJECTS:libcadet_object>)
	set_target_properties(libcadet_static PROPERTIES OUTPUT_NAME cadet_static)
	target_link_libraries(libcadet_static PUBLIC CADET::CompileOptions CADET::LibOptions PRIVATE CADET::AD libcadet_nonlinalg_static SUNDIALS::sundials_idas ${SUNDIALS_NVEC_TARGET} ${TBB_TARGET})

	# ---------------------------------------------------
	#   Build the shared library
	# ---------------------------------------------------

	add_library(libcadet_shared SHARED $<TARGET_OBJECTS:libcadet_object>)
	set_target_properties(libcadet_shared PROPERTIES OUTPUT_NAME cadet)
	target_link_libraries (libcadet_shared PUBLIC CADET::CompileOptions CADET::LibOptions PRIVATE CADET::AD libcadet_nonlinalg_static SUNDIALS::sundials_idas ${SUNDIALS_NVEC_TARGET} ${TBB_TARGET})

	list(APPEND LIBCADET_TARGETS libcadet_nonlinalg_static libcadet_object libcadet_static libcadet_shared)

	unset(LIB_LAPACK_DEFINE)
endif ()

# ---------------------------------------------------
#   Build the Matlab library
# ---------------------------------------------------
if (ENABLE_CADET_MEX)

	# Determine #defines for LAPACK implementation
	set(MATLAB_LAPACK_DEFINE)
	if ((NOT MATLAB_USE_SYSTEM_LAPACK) AND Matlab_LAPACK_LIBRARY_FOUND)
		if (Matlab_LAPACK_TRAILING_UNDERSCORE)
			set(MATLAB_LAPACK_DEFINE "CADET_LAPACK_TRAILING_UNDERSCORE")
		endif()
	else()
		set(MATLAB_LAPACK_DEFINE "CADET_LAPACK_TRAILING_UNDERSCORE")
	endif()

	set(LIBCADET_NONLINALG_MEX_SOURCES ${LIBCADET_NONLINALG_SPARSE_SOURCES})
	if (MATLAB_FORCE_UMFPACK)
		set(LIBCADET_NONLINALG_MEX_SOURCES ${CMAKE_SOURCE_DIR}/src/libcadet/linalg/UMFPackSparseMatrix.cpp)
	endif()

	# Add the build target for the mex nonlinalg library
	add_library(libcadet_nonlinalg_mex STATIC ${LIBCADET_NONLINALG_SOURCES} ${LIBCADET_NONLINALG_MEX_SOURCES})
	set_target_properties(libcadet_nonlinalg_mex PROPERTIES 
		OUTPUT_NAME cadet_nonlinalg_mex
		LINK_SEARCH_START_STATIC ON
		LINK_SEARCH_END_STATIC ON
	)
	target_compile_definitions(libcadet_nonlinalg_mex PRIVATE CADET_MATLABMEX MATLAB_MEX_FILE ${MATLAB_LAPACK_DEFINE} libcadet_mex_EXPORTS)
	target_link_libraries(libcadet_nonlinalg_mex PUBLIC CADET::CompileOptions PRIVATE SUNDIALS::sundials_idas ${SUNDIALS_NVEC_TARGET})
	
	# Link against Matlab's or system's LAPACK
	if ((NOT MATLAB_USE_SYSTEM_LAPACK) AND Matlab_LAPACK_LIBRARY_FOUND)
		target_link_libraries(libcadet_nonlinalg_mex PUBLIC CADET::CompileOptions PRIVATE ${Matlab_LAPACK_LIBRARY} ${Matlab_BLAS_LIBRARY})
	else()
		target_link_libraries(libcadet_nonlinalg_mex PUBLIC CADET::CompileOptions PRIVATE ${LAPACK_LIBRARIES})
	endif()

	if (ENABLE_2D_MODELS AND NOT MATLAB_FORCE_UMFPACK)
		if (UMFPACK_FOUND)
			target_link_libraries(libcadet_nonlinalg_mex PRIVATE UMFPACK::UMFPACK)
		endif()
		if (SUPERLU_FOUND)
			target_link_libraries(libcadet_nonlinalg_mex PRIVATE SuperLU::SuperLU)
		endif()
	elseif (ENABLE_2D_MODELS AND MATLAB_FORCE_UMFPACK)
		target_compile_definitions(libcadet_nonlinalg_mex PUBLIC CADET_FORCE_MATLAB_UMFPACK)
		target_link_libraries(libcadet_nonlinalg_mex PRIVATE ${Matlab_UMFPACK_LIBRARY})
	endif()

	# Add the build target for the mex CADET library
	add_library(libcadet_mex STATIC ${LIBCADET_SOURCES})
	set_target_properties(libcadet_mex PROPERTIES 
		OUTPUT_NAME cadet_mex
		LINK_SEARCH_START_STATIC ON
		LINK_SEARCH_END_STATIC ON
	)
	target_compile_definitions(libcadet_mex PRIVATE CADET_MATLABMEX MATLAB_MEX_FILE ${MATLAB_LAPACK_DEFINE} libcadet_mex_EXPORTS)
	target_link_libraries(libcadet_mex PUBLIC CADET::CompileOptions CADET::LibOptions PRIVATE CADET::AD libcadet_nonlinalg_mex SUNDIALS::sundials_idas ${SUNDIALS_NVEC_TARGET} ${Matlab_TBB_TARGET})

	# Enable usage of Matlab's LAPACK libs
	if ((NOT MATLAB_USE_SYSTEM_LAPACK) AND Matlab_LAPACK_LIBRARY_FOUND)
		target_compile_definitions(libcadet_nonlinalg_mex PRIVATE CADET_MATLAB_LAPACK)
		target_compile_definitions(libcadet_mex PRIVATE CADET_MATLAB_LAPACK)
	endif()

	list(APPEND LIBCADET_TARGETS libcadet_mex libcadet_nonlinalg_mex)

	unset(MATLAB_LAPACK_DEFINE)
endif()

foreach(_TARGET IN LISTS LIBCADET_TARGETS)
	target_include_directories(${_TARGET}
		PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include> 
		PRIVATE ${CMAKE_SOURCE_DIR}/src/libcadet ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}
	)
	set_target_properties(${_TARGET} PROPERTIES DEBUG_POSTFIX _d VERSION ${CADET_VERSION} SOVERSION "0")
	set_target_properties(${_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()

unset(LIBCADET_NONLINALG_SOURCES)
unset(LIBCADET_NONLINALG_SPARSE_SOURCES)
unset(LIBCADET_SOURCES)

# ---------------------------------------------------
#   Set installation related stuff
# ---------------------------------------------------

list(REMOVE_ITEM LIBCADET_TARGETS libcadet_object)
if (ENABLE_CADET_MEX)
	list(REMOVE_ITEM LIBCADET_TARGETS libcadet_mex libcadet_nonlinalg_mex)
endif()

install(CODE "MESSAGE(\"\nInstall LIBCADET\n\")")
install(TARGETS ${LIBCADET_TARGETS} EXPORT libcadet-targets)
install(FILES ${CMAKE_BINARY_DIR}/cadet/cadetCompilerInfo.hpp DESTINATION include/cadet)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/cadet DESTINATION include)
#install(EXPORT libcadet-targets DESTINATION lib NAMESPACE "CADET::" FILE CADET.cmake)

# ---------------------------------------------------
unset(LIBCADET_TARGETS)

# Info message
message(STATUS "Added LIBCADET module")