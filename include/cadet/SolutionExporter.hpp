// =============================================================================
//  CADET
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
 * Defines interfaces which export the solution to the user space.
 */

#ifndef LIBCADET_SOLUTIONEXPORTER_HPP_
#define LIBCADET_SOLUTIONEXPORTER_HPP_

#include <cstdint>

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"

namespace cadet
{

/**
 * @brief Interface providing functionality for exporting the solution to the user space
 */
class CADET_API ISolutionExporter
{
public:

	/**
	 * @brief Returns whether the associated model has a flux into the particles
	 * @return @c true if flux into the particles is present, otherwise @c false
	 */
	virtual bool hasParticleFlux() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the associated model has a particle mobile phase
	 * @return @c true if particle mobile phase is present, otherwise @c false
	 */
	virtual bool hasParticleMobilePhase() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the associated model has a solid phase
	 * @return @c true if solid phase is present, otherwise @c false
	 */
	virtual bool hasSolidPhase() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the associated model has volume DOFs
	 * @details Models that can accumulate (e.g., CSTR) usually have varying volume.
	 * @return @c true if volume DOFs are present, otherwise @c false
	 */
	virtual bool hasVolume() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the particle is always lumped to a single element
	 * @details If particles are lumped to a single element, the singleton particle shell dimension can be removed.
	 * @return @c true if particles are always represented by a single element, otherwise @c false
	 */
	virtual bool isParticleLumped() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns whether the primary coordinate is always a single element
	 * @details If the primary coordinate is always a single element, the singleton dimension can be removed.
	 * @return @c true if the state in the primary coordinate direction is always represented by a single element, otherwise @c false
	 */
	virtual bool hasPrimaryExtent() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of components
	 * @return Number of components
	 */
	virtual unsigned int numComponents() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of primary coordinates / points
	 * @details For axial flow columns, this is the number of axial points
	 * @return Number of primary coordinates / points
	 */
	virtual unsigned int numPrimaryCoordinates() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of secondary coordinates / points
	 * @details For axial flow columns, this is the number of radial points
	 * @return Number of secondary coordinates / points
	 */
	virtual unsigned int numSecondaryCoordinates() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of inlet ports
	 * @return Number of inlet ports
	 */
	virtual unsigned int numInletPorts() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of outlet ports
	 * @return Number of outlet ports
	 */
	virtual unsigned int numOutletPorts() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of particle types
	 * @return Number of particle types
	 */
	virtual unsigned int numParticleTypes() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of particle shells
	 * @param [in] parType Particle type index
	 * @return Number of particle shells
	 */
	virtual unsigned int numParticleShells(unsigned int parType) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the total number of bound states for all components of a given particle type
	 * @param [in] parType Particle type index
	 * @return Total number of bound states for all components of a particle type
	 */
	virtual unsigned int numBoundStates(unsigned int parType) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the total number of DOFs in the main mobile phase (e.g., bulk volume)
	 * @return Total number of mobile phase DOFs
	 */
	virtual unsigned int numMobilePhaseDofs() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the total number of mobile phase DOFs in the particles of the given type if particles are supported
	 * @param [in] parType Particle type index
	 * @return Total number of particle mobile phase DOFs
	 */
	virtual unsigned int numParticleMobilePhaseDofs(unsigned int parType) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the total number of mobile phase DOFs in the particles of the given type if particles are supported
	 * @details This includes all particle types.
	 * @return Total number of particle mobile phase DOFs
	 */
	virtual unsigned int numParticleMobilePhaseDofs() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the total number of solid phase DOFs for the given particle type
	 * @param [in] parType Particle type index
	 * @return Total number of solid phase DOFs
	 */
	virtual unsigned int numSolidPhaseDofs(unsigned int parType) const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the total number of solid phase DOFs
	 * @details This includes all particle types.
	 * @return Total number of solid phase DOFs
	 */
	virtual unsigned int numSolidPhaseDofs() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of bulk-bead flux DOFs if particle fluxes are supported
	 * @return Total number of particle flux DOFs
	 */
	virtual unsigned int numParticleFluxDofs() const CADET_NOEXCEPT = 0;

	/**
	 * @brief Returns the number of volume DOFs if volume DOFs are supported
	 * @return Number of volume DOFs
	 */
	virtual unsigned int numVolumeDofs() const CADET_NOEXCEPT = 0;


	/**
	 * @brief Writes the solution of the (bulk) mobile phase to the given buffer
	 * @details The data is written in the order primary-secondary-component,
	 *          where the last index changes the fastest.
	 * 
	 *          For a system with 3 primary coordinates, 2 secondary coordinates,
	 *          and 4 components, the data is written as
	 *          p0s0c0, p0s0c1, p0s0c2, p0s0c3,
	 *          p0s1c0, p0s1c1, p0s1c2, p0s1c3, 
	 *          p1s0c0, p1s0c1, p1s0c2, p1s0c3,
	 *          p1s1c0, p1s1c1, p1s1c2, p1s1c3, ...
	 * 
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeMobilePhase(double* buffer) const = 0;

	/**
	 * @brief Writes the solution of the solid phase to the given buffer
	 * @details The data is written in the order particletype-primary-secondary-particle-component-boundstate,
	 *          where the last index changes the fastest.
	 *          The solution is written for all particle types.
	 * 
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeSolidPhase(double* buffer) const = 0;

	/**
	 * @brief Writes the solution of the particle mobile phase to the given buffer
	 * @details The data is written in the order particletype-primary-secondary-particle-component,
	 *          where the last index changes the fastest.
	 *          The solution is written for all particle types.
	 * 
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeParticleMobilePhase(double* buffer) const = 0;

	/**
	 * @brief Writes the solution of the solid phase to the given buffer
	 * @details The data is written in the order primary-secondary-particle-component-boundstate,
	 *          where the last index changes the fastest.
	 *          The solution is written for the given particle type only.
	 * 
	 *          For a system with 2 primary coordinates, 2 secondary coordinates, 2 particle
	 *          coordinates, 2 components, and 2 bound states per component, the data is written as
	 *          p0s0p0c0s0, p0s0p0c0s1, p0s0p0c1s0, p0s0p0c1s1,
	 *          p0s0p1c0s0, p0s0p1c0s1, p0s0p1c1s0, p0s0p1c1s1,
	 *          p0s1p0c0s0, p0s1p0c0s1, p0s1p0c1s0, p0s1p0c1s1,
	 *          p0s1p1c0s0, p0s1p1c0s1, p0s1p1c1s0, p0s1p1c1s1,
	 *          p1s0p0c0s0, p1s0p0c0s1, p1s0p0c1s0, p1s0p0c1s1,
	 *          p1s0p1c0s0, p1s0p1c0s1, p1s0p1c1s0, p1s0p1c1s1,
	 *          p1s1p0c0s0, p1s1p0c0s1, p1s1p0c1s0, p1s1p0c1s1,
	 *          p1s1p1c0s0, p1s1p1c0s1, p1s1p1c1s0, p1s1p1c1s1, ...
	 * 
	 * @param [in] parType Index of the particle type to be written
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeSolidPhase(unsigned int parType, double* buffer) const = 0;

	/**
	 * @brief Writes the solution of the particle mobile phase to the given buffer
	 * @details The data is written in the order primary-secondary-particle-component,
	 *          where the last index changes the fastest.
	 *          The solution is written for the given particle type only.
	 * 
	 *          For a system with 2 primary coordinates, 2 secondary coordinates, 2 particle
	 *          coordinates, 2 components, and 2 bound states per component, the data is written as
	 *          p0s0p0c0s0, p0s0p0c0s1, p0s0p0c1s0, p0s0p0c1s1,
	 *          p0s0p1c0s0, p0s0p1c0s1, p0s0p1c1s0, p0s0p1c1s1,
	 *          p0s1p0c0s0, p0s1p0c0s1, p0s1p0c1s0, p0s1p0c1s1,
	 *          p0s1p1c0s0, p0s1p1c0s1, p0s1p1c1s0, p0s1p1c1s1,
	 *          p1s0p0c0s0, p1s0p0c0s1, p1s0p0c1s0, p1s0p0c1s1,
	 *          p1s0p1c0s0, p1s0p1c0s1, p1s0p1c1s0, p1s0p1c1s1,
	 *          p1s1p0c0s0, p1s1p0c0s1, p1s1p0c1s0, p1s1p0c1s1,
	 *          p1s1p1c0s0, p1s1p1c0s1, p1s1p1c1s0, p1s1p1c1s1, ...
	 * 
	 * @param [in] parType Index of the particle type to be written
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeParticleMobilePhase(unsigned int parType, double* buffer) const = 0;

	/**
	 * @brief Writes the solution of the flux between primary mobile phase and particle mobile phase to the given buffer
	 * @details The data is written in the order particletype-primary-secondary-component,
	 *          where the last index changes the fastest.
	 *          The solution is written for all particle types.
	 * 
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeParticleFlux(double* buffer) const = 0;

	/**
	 * @brief Writes the solution of the flux between primary mobile phase and particle mobile phase of the selected particle type to the given buffer
	 * @details The data is written in the order particletype-primary-secondary-component,
	 *          where the last index changes the fastest.
	 * 
	 * @param [in] parType Index of the particle type to be written
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeParticleFlux(unsigned int parType, double* buffer) const = 0;

	/**
	 * @brief Writes the solution of the volume to the given buffer
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeVolume(double* buffer) const = 0;

	// virtual int writeSolidVolume(double* buffer) const = 0;


	/**
	 * @brief Writes the solution of the inlet at the given port into the provided buffer
	 * @details Writes all components of the selected port to the provided buffer.
	 * 
	 * @param [in] port Index of the port
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeInlet(unsigned int port, double* buffer) const = 0;

	/**
	 * @brief Writes the solution of the inlet at all ports into the provided buffer
	 * @details Writes all components of all ports to the provided buffer in port-component order.
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeInlet(double* buffer) const = 0;

	/**
	 * @brief Writes the solution of the outlet at the given port into the provided buffer
	 * @details Writes all components of the selected port to the provided buffer.
	 * 
	 * @param [in] port Index of the port
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeOutlet(unsigned int port, double* buffer) const = 0;

	/**
	 * @brief Writes the solution of the outlet at all ports into the provided buffer
	 * @details Writes all components of all ports to the provided buffer in port-component order.
	 * @param [out] buffer Pointer to buffer that receives the data
	 * @return Number of written items
	 */
	virtual int writeOutlet(double* buffer) const = 0;


	/**
	 * @brief Returns primary coordinates (e.g., axial for axial flow columns)
	 * @param [out] coords Pointer to array that is filled with primary coordinates
	 * @return Number of written items
	 */
	virtual int writePrimaryCoordinates(double* coords) const = 0;

	/**
	 * @brief Returns secondary coordinates (e.g., radial for axial flow columns)
	 * @param [out] coords Pointer to array that is filled with secondary coordinates
	 * @return Number of written items
	 */
	virtual int writeSecondaryCoordinates(double* coords) const = 0;

	/**
	 * @brief Returns particle coordinates of the selected particle type
	 * @param [in] parType Particle type index
	 * @param [out] coords Pointer to array that is filled with particle coordinates of DOFs
	 * @return Number of written items
	 */
	virtual int writeParticleCoordinates(unsigned int parType, double* coords) const = 0;
};

} // namespace cadet

#endif  // LIBCADET_SOLUTIONEXPORTER_HPP_
