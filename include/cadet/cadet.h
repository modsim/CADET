// =============================================================================
//  CADET - The Chromatography Analysis and Design Toolkit
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

/**
 * @file 
 * Defines the C API of the library.
 */

#ifndef LIBCADET_CAPI_HPP_
#define LIBCADET_CAPI_HPP_

#include "cadet/LibExportImport.hpp"

#include <stdint.h>

#define CADET_OK(x) (x >= 0)
#define CADET_ERR(x) (x < 0)

extern "C"
{

	/**
	 * @brief Returns the version string of the libcadet library
	 * @return Version string
	 */
	CADET_API const char* cdtGetLibraryVersion();

	/**
	 * @brief Returns the git commit hash of the source which was used to build the binaries
	 * @return Git commit hash as string
	 */
	CADET_API const char* cdtGetLibraryCommitHash();

	/**
	 * @brief Returns the git refspec of the source which was used to build the binaries
	 * @return Git refspec
	 */
	CADET_API const char* cdtGetLibraryBranchRefspec();

	/**
	 * @brief Returns the versions of the dependencies used for building the binaries
	 * @details The format is DEPNAME1=VERSION;DEPNAME2=VERSION; where each dependency is
	 *          terminated by a semicolon.
	 * @return Dependency versions string
	 */
	CADET_API const char* cdtGetLibraryDependencyVersions();

	/**
	 * @brief Returns the build type (Debug, Release, RelWithDebInfo, RelMinSize)
	 * @return Build type
	 */
	CADET_API const char* cdtGetLibraryBuildType();

	/**
	 * @brief Returns the compiler including its version used for building the library
	 * @return Compiler and its version
	 */
	CADET_API const char* cdtGetLibraryCompiler();

	/**
	 * @brief Returns the compiler flags used for building the library
	 * @return Compiler flags
	 */
	CADET_API const char* cdtGetLibraryCompilerFlags();

	/**
	 * @brief Returns the git refspec of the source which was used to build the binaries
	 * @return Git refspec
	 */
	CADET_API const char* cdtGetLibraryBuildHost();




	/**
	 * @brief LogLevel represents the severity of log messages
	 * @details The levels are nested, such that the highest level (Trace) includes all lower levels.
	 */
	enum cdtLogLevel
	{
		/**
		 * @brief Nothing is logged
		 */
		cdtLogLevelNone = 0,
		/**
		 * @brief Non-recoverable (fatal) error (terminates program / simulation)
		 */
		cdtLogLevelFatal = 1,
		/**
		 * @brief Error, which may be recoverable
		 */
		cdtLogLevelError = 2,
		/**
		 * @brief Warning
		 */
		cdtLogLevelWarning = 3,
		/**
		 * @brief Normal output (informative)
		 */
		cdtLogLevelNormal = 4,
		/**
		 * @brief Additional info output
		 */
		cdtLogLevelInfo = 5,
		/**
		 * @brief Debug messages and values of (intermediate) variables or calculations
		 */
		cdtLogLevelDebug = 6,
		/**
		 * @brief Full trace including entering / leaving functions
		 */
		cdtLogLevelTrace = 7,
	};


	/**
	 * @brief Callback for each log message
	 * @param [in] file Name of the file
	 * @param [in] func Name of the function
	 * @param [in] line Line number
	 * @param [in] lvl Log level encoding the severity of the message
	 * @param [in] lvlStr Name of the log level
	 * @param [in] message Actual log message
	 */
	typedef void (*cdtLogHandler)(const char* file, const char* func, const unsigned int line, int lvl, const char* lvlStr, const char* message);

	/**
	 * @brief Sets the log receiver replacing any previously set receiver
	 * @param [in] recv Pointer to handler implementation or @c NULL
	 */
	CADET_API void cdtSetLogReceiver(cdtLogHandler recv);

	/**
	 * @brief Sets the log level
	 * @details All messages on a lower log level (i.e., higher severity or information content)
	 *          are filtered out.
	 * @param [in] lvl New log level
	 */
	CADET_API void cdtSetLogLevel(int lvl);

	/**
	 * @brief Returns the current log level
	 * @return Current log level
	 */
	CADET_API int cdtGetLogLevel();



	/**
	 * @brief Return and error codes
	 */
	enum cdtErrors
	{
		/**
		 * @brief Requested data has not been stored / is not available
		 */
		cdtDataNotStored = -3,
		/**
		 * @brief Input arguments invalid
		 */
		cdtErrorInvalidInputs = -2,
		/**
		 * @brief General error
		 */
		cdtError = -1,
		/**
		 * @brief Success
		 */
		cdtOK = 0
	};

	/**
	 * Result of operation (in general: < 0 indicates failure, >= 0 indicates success)
	 */
	typedef int cdtResult;



	/**
	 * @brief ParameterProvider interface is used for querying parameters and data
	 */
	typedef struct
	{
		/**
		 * @brief User data passed to every callback function
		 */
		void* userData;

		/**
		 * @brief Returns the value of a parameter of type double
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [out] val Value of the parameter
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getDouble)(void* userData, const char* paramName, double* val);

		/**
		 * @brief Returns the value of a parameter of type int
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [out] val Value of the parameter
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getInt)(void* userData, const char* paramName, int* val);

		/**
		 * @brief Returns the value of a parameter of type bool
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [out] val Value of the parameter (@c 0 if false, @c 1 if true)
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getBool)(void* userData, const char* paramName, uint8_t* val);

		/**
		 * @brief Returns the value of a parameter of type string
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [out] val Value of the parameter (the string is copied internally)
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getString)(void* userData, const char* paramName, char const** val);

		/**
		 * @brief Returns a parameter array of type double
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [out] numElements Number of elements in the array
		 * @param [out] vals Pointer to contiguous array of elements (will be copied internally)
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getDoubleArray)(void* userData, const char* paramName, int* numElements, double** vals);

		/**
		 * @brief Returns a parameter array of type int
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [out] numElements Number of elements in the array
		 * @param [out] vals Pointer to contiguous array of elements (will be copied internally)
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getIntArray)(void* userData, const char* paramName, int* numElements, int** vals);

		/**
		 * @brief Returns a parameter array of type bool
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [out] numElements Number of elements in the array
		 * @param [out] vals Pointer to contiguous array of elements (will be copied internally)
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getBoolArray)(void* userData, const char* paramName, int* numElements, uint8_t** vals);

		/**
		 * @brief Returns a parameter array of type string
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [out] numElements Number of elements in the array
		 * @param [out] vals Pointer to contiguous array of elements (will be copied internally)
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getStringArray)(void* userData, const char* paramName, int* numElements, char const*** vals);

		/**
		 * @brief Returns an item of a parameter array of type double
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [in] idx Index of the requested element
		 * @param [out] val Value
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getDoubleArrayItem)(void* userData, const char* paramName, int idx, double* val);

		/**
		 * @brief Returns an item of a parameter array of type int
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [in] idx Index of the requested element
		 * @param [out] val Value
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getIntArrayItem)(void* userData, const char* paramName, int idx, int* val);

		/**
		 * @brief Returns an item of a parameter array of type bool
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [in] idx Index of the requested element
		 * @param [out] val Value
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getBoolArrayItem)(void* userData, const char* paramName, int idx, uint8_t* val);

		/**
		 * @brief Returns an item of a parameter array of type string
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [in] idx Index of the requested element
		 * @param [out] val Value (will be copied internally)
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*getStringArrayItem)(void* userData, const char* paramName, int idx, char const** val);

		/**
		 * @brief Checks whether a given parameter or scope exists
		 * @param [in] userData User supplied data
		 * @param [in] elemName Name of the parameter or scope
		 * @return Non-zero if the parameter exists, otherwise @c 0
		 */
		int (*exists)(void* userData, const char* elemName);

		/**
		 * @brief Checks whether a given parameter is an array
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @param [out] res Non-zero if the parameter is an array, otherwise @c 0
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*isArray)(void* userData, const char* paramName, uint8_t* res);

		/**
		 * @brief Returns the number of elements (of an array) in a given field
		 * @param [in] userData User supplied data
		 * @param [in] paramName Name of the parameter
		 * @return Number of elements in the given field (or non-positive number if field does not exist)
		 */
		int (*numElements)(void* userData, const char* paramName);

		/**
		 * @brief Changes to a given namespace subscope
		 * @param [in] userData User supplied data
		 * @param [in] scope Name of the scope
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*pushScope)(void* userData, const char* scope);

		/**
		 * @brief Changes to the parent of the current namespace scope
		 * @param [in] userData User supplied data
		 * @return @c cdtOK on success, @c cdtError on error
		 */
		cdtResult (*popScope)(void* userData);

	} cdtParameterProvider;




	/**
	 * Driver object
	 */
	typedef struct cdtDriver cdtDriver;

	/**
	 * Holds function pointers to API version 1
	 */
	typedef struct
	{
		/**
		 * @brief Creates a driver that handles simulation on a high level
		 * @return Driver handle or @c NULL if an error occurred
		 */
		cdtDriver* (*createDriver)(void);

		/**
		 * @brief Deletes a driver created by createDriver
		 * @param[in] drv Driver handle
		 */
		void (*deleteDriver)(cdtDriver* drv);

		/**
		 * @brief Runs a full simulation using the given driver and parameter provider
		 * @param[in] drv Driver handle
		 * @param[in] paramProvider Callback parameter provider
		 * @return @c cdtOK on success, a negative value indicating the error otherwise
		 */
		cdtResult (*runSimulation)(cdtDriver* drv, cdtParameterProvider const* paramProvider);

		/**
		 * @brief Returns the number of particle types
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] nParTypes Number of particle types
		 */
		cdtResult (*getNumParTypes)(cdtDriver* drv, int unitOpId, int* nParTypes);

		/**
		 * @brief Returns the solution of the last simulation at unit inlet
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nPort Number of ports
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionInlet)(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime, int* nPort, int* nComp);

		/**
		 * @brief Returns the solution of the last simulation at unit outlet
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nPort Number of ports
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionOutlet)(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime, int* nPort, int* nComp);

		/**
		 * @brief Returns the solution of the last simulation of the column bulk phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of radial cells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionBulk)(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nComp);

		/**
		 * @brief Returns the solution of the last simulation of the column particle liquid phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] parType Particle type index
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParShells Number of particle shells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionParticle)(cdtDriver* drv, int unitOpId, int parType, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParShells, int* nComp);

		/**
		 * @brief Returns the solution of the last simulation of the column particle solid phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] parType Particle type index
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParShells Number of particle shells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionSolid)(cdtDriver* drv, int unitOpId, int parType, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParShells, int* nBound);

		/**
		 * @brief Returns the solution of the last simulation of the column particle flux
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParticleTypes Number of particle types
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionFlux)(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParticleTypes, int* nComp);

		/**
		 * @brief Returns the solution of the last simulation of the unit volume
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 */
		cdtResult (*getSolutionVolume)(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime);

		/**
		 * @brief Returns the solution derivative of the last simulation at unit inlet
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nPort Number of ports
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionDerivativeInlet)(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime, int* nPort, int* nComp);

		/**
		 * @brief Returns the solution derivative of the last simulation at unit outlet
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nPort Number of ports
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionDerivativeOutlet)(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime, int* nPort, int* nComp);

		/**
		 * @brief Returns the solution derivative of the last simulation of the column bulk phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of radial cells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionDerivativeBulk)(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nComp);

		/**
		 * @brief Returns the solution derivative of the last simulation of the column particle liquid phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] parType Particle type index
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParShells Number of particle shells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionDerivativeParticle)(cdtDriver* drv, int unitOpId, int parType, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParShells, int* nComp);

		/**
		 * @brief Returns the solution derivative of the last simulation of the column particle solid phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] parType Particle type index
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParShells Number of particle shells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionDerivativeSolid)(cdtDriver* drv, int unitOpId, int parType, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParShells, int* nBound);

		/**
		 * @brief Returns the solution derivative of the last simulation of the column particle flux
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParticleTypes Number of particle types
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSolutionDerivativeFlux)(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParticleTypes, int* nComp);

		/**
		 * @brief Returns the solution derivative of the last simulation of the unit volume
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 */
		cdtResult (*getSolutionDerivativeVolume)(cdtDriver* drv, int unitOpId, double const** time, double const** data, int* nTime);

		/**
		 * @brief Returns the parameter sensitivity of the last simulation at unit inlet
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nPort Number of ports
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityInlet)(cdtDriver* drv, int unitOpId, int idx, double const** time, double const** data, int* nTime, int* nPort, int* nComp);

		/**
		 * @brief Returns the parameter sensitivity of the last simulation at unit outlet
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nPort Number of ports
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityOutlet)(cdtDriver* drv, int unitOpId, int idx, double const** time, double const** data, int* nTime, int* nPort, int* nComp);

		/**
		 * @brief Returns the parameter sensitivity of the last simulation of the column bulk phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of radial cells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityBulk)(cdtDriver* drv, int unitOpId, int idx, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nComp);

		/**
		 * @brief Returns the parameter sensitivity of the last simulation of the column particle liquid phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] parType Particle type index
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParShells Number of particle shells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityParticle)(cdtDriver* drv, int unitOpId, int idx, int parType, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParShells, int* nComp);

		/**
		 * @brief Returns the parameter sensitivity of the last simulation of the column particle solid phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] parType Particle type index
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParShells Number of particle shells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivitySolid)(cdtDriver* drv, int unitOpId, int idx, int parType, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParShells, int* nBound);

		/**
		 * @brief Returns the parameter sensitivity of the last simulation of the column particle flux
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParticleTypes Number of particle types
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityFlux)(cdtDriver* drv, int unitOpId, int idx, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParticleTypes, int* nComp);

		/**
		 * @brief Returns the parameter sensitivity of the last simulation of the unit volume
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 */
		cdtResult (*getSensitivityVolume)(cdtDriver* drv, int unitOpId, int idx, double const** time, double const** data, int* nTime);

		/**
		 * @brief Returns the parameter sensitivity derivative of the last simulation at unit inlet
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nPort Number of ports
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityDerivativeInlet)(cdtDriver* drv, int unitOpId, int idx, double const** time, double const** data, int* nTime, int* nPort, int* nComp);

		/**
		 * @brief Returns the parameter sensitivity derivative of the last simulation at unit outlet
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nPort Number of ports
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityDerivativeOutlet)(cdtDriver* drv, int unitOpId, int idx, double const** time, double const** data, int* nTime, int* nPort, int* nComp);

		/**
		 * @brief Returns the parameter sensitivity derivative of the last simulation of the column bulk phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of radial cells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityDerivativeBulk)(cdtDriver* drv, int unitOpId, int idx, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nComp);

		/**
		 * @brief Returns the parameter sensitivity derivative of the last simulation of the column particle liquid phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] parType Particle type index
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParShells Number of particle shells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityDerivativeParticle)(cdtDriver* drv, int unitOpId, int idx, int parType, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParShells, int* nComp);

		/**
		 * @brief Returns the parameter sensitivity derivative of the last simulation of the column particle solid phase
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] parType Particle type index
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParShells Number of particle shells
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityDerivativeSolid)(cdtDriver* drv, int unitOpId, int idx, int parType, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParShells, int* nBound);

		/**
		 * @brief Returns the parameter sensitivity derivative of the last simulation of the column particle flux
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 * @param [out] nTime Number of time points
		 * @param [out] nAxialCells Number of axial cells
		 * @param [out] nRadialCells Number of ports
		 * @param [out] nParticleTypes Number of particle types
		 * @param [out] nComp Number of components
		 */
		cdtResult (*getSensitivityDerivativeFlux)(cdtDriver* drv, int unitOpId, int idx, double const** time, double const** data, int* nTime, int* nAxialCells, int* nRadialCells, int* nParticleTypes, int* nComp);

		/**
		 * @brief Returns the parameter sensitivity derivative of the last simulation of the unit volume
		 * @details Before this function is called, a simulation has to be run successfully.
		 *          The array pointers are only valid until a new simulation is started.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [in] idx Sensitivity ID
		 * @param [out] time Time array pointer
		 * @param [out] data Data array pointer
		 */
		cdtResult (*getSensitivityDerivativeVolume)(cdtDriver* drv, int unitOpId, int idx, double const** time, double const** data, int* nTime);

		/**
		 * @brief Returns the last state of the simulation
		 * @details Before this function is called, a simulation has to be run successfully.
		 * @param [in] drv Driver handle
		 * @param [out] state State vector pointer
		 * @param [out] nStates Number of entries in the state vector
		*/
		cdtResult (*getLastState)(cdtDriver* drv, double const** state, int* nStates);

		/**
		 * @brief Returns the last time derivative state of the simulation
		 * @details Before this function is called, a simulation has to be run successfully.
		 * @param [in] drv Driver handle
		 * @param [out] state State vector pointer
		 * @param [out] nStates Number of entries in the state vector
		*/
		cdtResult (*getLastStateTimeDerivative)(cdtDriver* drv, double const** state, int* nStates);

		/**
		 * @brief Returns the last state of the unit operation of the simulation
		 * @details Before this function is called, a simulation has to be run successfully.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] state State vector pointer
		 * @param [out] nStates Number of entries in the state vector
		*/
		cdtResult (*getLastUnitState)(cdtDriver* drv, int unitOpId, double const** state, int* nStates);

		/**
		 * @brief Returns the last time derivative state of the unit operation of the simulation
		 * @details Before this function is called, a simulation has to be run successfully.
		 * @param [in] drv Driver handle
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] state State vector pointer
		 * @param [out] nStates Number of entries in the state vector
		*/
		cdtResult (*getLastUnitStateTimeDerivative)(cdtDriver* drv, int unitOpId, double const** state, int* nStates);

		/**
		 * @brief Returns the last sensitivity state of the simulation
		 * @details Before this function is called, a simulation has to be run successfully.
		 * @param [in] drv Driver handle
		 * @param [in] sensIdx Sensitivity ID
		 * @param [out] state State vector pointer
		 * @param [out] nStates Number of entries in the state vector
		*/
		cdtResult (*getLastSensitivityState)(cdtDriver* drv, int sensIdx, double const** state, int* nStates);

		/**
		 * @brief Returns the last time derivative sensitivity state of the simulation
		 * @details Before this function is called, a simulation has to be run successfully.
		 * @param [in] drv Driver handle
		 * @param [in] sensIdx Sensitivity ID
		 * @param [out] state State vector pointer
		 * @param [out] nStates Number of entries in the state vector
		*/
		cdtResult (*getLastSensitivityStateTimeDerivative)(cdtDriver* drv, int sensIdx, double const** state, int* nStates);

		/**
		 * @brief Returns the last sensitivity state of the unit operation of the simulation
		 * @details Before this function is called, a simulation has to be run successfully.
		 * @param [in] drv Driver handle
		 * @param [in] sensIdx Sensitivity ID
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] state State vector pointer
		 * @param [out] nStates Number of entries in the state vector
		*/
		cdtResult (*getLastSensitivityUnitState)(cdtDriver* drv, int sensIdx, int unitOpId, double const** state, int* nStates);

		/**
		 * @brief Returns the last time derivative sensitivity state of the unit operation of the simulation
		 * @details Before this function is called, a simulation has to be run successfully.
		 * @param [in] drv Driver handle
		 * @param [in] sensIdx Sensitivity ID
		 * @param [in] unitOpId ID of the unit operation whose solution is returned
		 * @param [out] state State vector pointer
		 * @param [out] nStates Number of entries in the state vector
		*/
		cdtResult (*getLastSensitivityUnitStateTimeDerivative)(cdtDriver* drv, int sensIdx, int unitOpId, double const** state, int* nStates);

	} cdtAPIv010000;

	/**
	 * @brief      Queries API version 1.0.0, which is returned as a struct of function pointers
	 * @param[out] ptr      Pointer to a matching function pointer struct that is populated if the API is available
	 * @return     Success (>= 0) if the API is available, otherwise error (< 0)
	 */
	CADET_API cdtResult cdtGetAPIv010000(cdtAPIv010000* ptr);

}

#endif  // LIBCADET_CAPI_HPP_
