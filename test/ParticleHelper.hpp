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
 * Defines tests for unit operations with particles.
 */

#ifndef CADETTEST_PARTICLEHELPER_HPP_
#define CADETTEST_PARTICLEHELPER_HPP_

#include "cadet/ParameterId.hpp"

#include <vector>

namespace cadet
{

class JsonParameterProvider;

namespace test
{

namespace particle
{

/**
 * @brief Extends a model to multiple particle types by replicating the first type
 * @param [in,out] jpp ParameterProvider to extend
 * @param [in] unit Index of unit operation
 * @param [in] nTypes Total number of particle types
 * @param [in] volFrac Array with volume fractions of particle types
 */
void extendModelToManyParticleTypes(cadet::JsonParameterProvider& jpp, UnitOpIdx unit, unsigned int nTypes,
									double const* const volFrac);

/**
 * @brief Extends a model to multiple particle types by replicating the first type
 * @param [in,out] jpp ParameterProvider to extend
 * @param [in] nTypes Total number of particle types
 * @param [in] volFrac Array with volume fractions of particle types
 */
void extendModelToManyParticleTypes(cadet::JsonParameterProvider& jpp, unsigned int nTypes,
									double const* const volFrac);

/**
 * @brief Extends a model to multiple particle types by replicating the first type
 * @details Modifies the double-valued parameters of the replicated particle types by a factor given in @p paramFactors.
			The source particle type retains a factor of @c 1.0 that is not included in @p paramFactors, which has
 length @c nTypes-1.
 * @param [in,out] jpp ParameterProvider to extend
 * @param [in] unit Index of unit operation
 * @param [in] nTypes Total number of particle types
 * @param [in] paramFactors Array with factors for replicated double-valued parameters
 * @param [in] volFrac Array with volume fractions of particle types
 */
void extendModelToManyParticleTypes(cadet::JsonParameterProvider& jpp, UnitOpIdx unit, unsigned int nTypes,
									double const* const paramFactors, double const* const volFrac);

/**
 * @brief Extends a model to multiple particle types by replicating the first type
 * @details Modifies the double-valued parameters of the replicated particle types by a factor given in @p paramFactors.
			The source particle type retains a factor of @c 1.0 that is not included in @p paramFactors, which has
 length @c nTypes-1.
 * @param [in,out] jpp ParameterProvider to extend
 * @param [in] nTypes Total number of particle types
 * @param [in] paramFactors Array with factors for replicated double-valued parameters
 * @param [in] volFrac Array with volume fractions of particle types
 */
void extendModelToManyParticleTypes(cadet::JsonParameterProvider& jpp, unsigned int nTypes,
									double const* const paramFactors, double const* const volFrac);

/**
 * @brief Sets the volume fractions of the particle types
 * @param [in,out] jpp ParameterProvider to extend
 * @param [in] unit Index of unit operation
 * @param [in] volFrac Array with volume fractions of particle types
 */
void setParticleTypeVolumeFractions(cadet::JsonParameterProvider& jpp, UnitOpIdx unit,
									const std::vector<double>& volFrac);

/**
 * @brief Sets the volume fractions of the particle types
 * @param [in,out] jpp ParameterProvider to extend
 * @param [in] volFrac Array with volume fractions of particle types
 */
void setParticleTypeVolumeFractions(cadet::JsonParameterProvider& jpp, const std::vector<double>& volFrac);

/**
 * @brief Assigns spatially varying particle type volume fractions based on the current values
 * @param [in,out] jpp ParameterProvider to extend
 * @param [in] nParType Number of particle types
 */
void scrambleParticleTypeFractionsSpatially(cadet::JsonParameterProvider& jpp, unsigned int nParType);

/**
 * @brief Checks whether results of simulation with one particle type matches the one with two (identical) particle
 * types
 * @details The given simulation is performed. The existing particle type is replicated such that there are
 *          two identical particle types. Another simulation is run with the two particle types.
 *          The results of these simulations must match.
 *
 * @param [in] jpp Unit operation configuration
 * @param [in] absTols Array with absolute error tolerances
 * @param [in] relTols Array with relative error tolerances
 */
void testOneVsTwoIdenticalParticleTypes(cadet::JsonParameterProvider& jpp, double absTol, double relTol);

/**
 * @brief Checks whether results of simulation with one particle type matches the one with two (identical) particle
 * types
 * @details The LWE example is taken and simulated. The existing particle type is replicated such that there are
 *          two identical particle types. Another simulation is run with the two particle types.
 *          The results of these simulations must match.
 *
 * @param [in] uoType Unit operation type
 * @param [in] absTols Array with absolute error tolerances
 * @param [in] relTols Array with relative error tolerances
 */
void testOneVsTwoIdenticalParticleTypes(const char* uoType, double absTol, double relTol);

/**
 * @brief Checks whether, when using two separate identical particle types, results of one type match the other
 * @details The given model is modified by replicating the existing particle type such that there are
 *          two identical particle types. Two simulations are run each one using only one particle type.
 *          The results of these simulations must match.
 *
 * @param [in] jpp Unit operation configuration
 * @param [in] absTols Array with absolute error tolerances
 * @param [in] relTols Array with relative error tolerances
 */
void testSeparateIdenticalParticleTypes(cadet::JsonParameterProvider& jpp, double absTol, double relTol);

/**
 * @brief Checks whether, when using two separate identical particle types, results of one type match the other
 * @details The LWE example is taken and the existing particle type is replicated such that there are
 *          two identical particle types. Two simulations are run each one using only one particle type.
 *          The results of these simulations must match.
 *
 * @param [in] uoType Unit operation type
 * @param [in] absTols Array with absolute error tolerances
 * @param [in] relTols Array with relative error tolerances
 */
void testSeparateIdenticalParticleTypes(const char* uoType, double absTol, double relTol);

/**
 * @brief Checks whether a linear binding model with multiple identical particle types produces the same as result as a
 * single type model
 * @details The given model is run. Then, two additional identical particle types are added.
 *          The results of the multi type simulation must match the ones of the single type simulation.
 *          This check is conducted with both dynamic and quasi-stationary binding.
 *
 * @param [in] jpp Unit operation configuration
 * @param [in] absTols Array with absolute error tolerances
 * @param [in] relTols Array with relative error tolerances
 */
void testLinearMixedParticleTypes(cadet::JsonParameterProvider& jpp, double absTol, double relTol);

/**
 * @brief Checks whether a linear binding model with multiple identical particle types produces the same as result as a
 * single type model
 * @details The linear benchmark problem is run. Then, two additional identical particle types are added.
 *          The results of the multi type simulation must match the ones of the single type simulation.
 *          This check is conducted with both dynamic and quasi-stationary binding.
 *
 * @param [in] uoType Unit operation type
 * @param [in] absTols Array with absolute error tolerances
 * @param [in] relTols Array with relative error tolerances
 */
void testLinearMixedParticleTypes(const char* uoType, double absTol, double relTol);

/**
 * @brief Checks the full analytic Jacobian against AD for a model with multiple particle types
 * @param [in] jpp Unit operation configuration
 */
void testJacobianMixedParticleTypes(cadet::JsonParameterProvider& jpp);

/**
 * @brief Checks whether a linear binding model with multiple identical particle types produces the same as result as a
 * single type model
 * @details The linear benchmark problem is run. Then, two additional identical particle types are added.
 *          The results of the multi type simulation must match the ones of the single type simulation.
 *          This check is conducted with both dynamic and quasi-stationary binding.
 *
 *          The particle volume fractions are spatially inhomogeneous.
 *
 * @param [in] uoType Unit operation type
 * @param [in] absTols Array with absolute error tolerances
 * @param [in] relTols Array with relative error tolerances
 */
void testLinearSpatiallyMixedParticleTypes(const char* uoType, double absTol, double relTol);

/**
 * @brief Checks the full analytic Jacobian against AD for a model with multiple particle types
 * @param [in] uoType Unit operation type
 */
void testJacobianMixedParticleTypes(const std::string& uoType);

/**
 * @brief Checks the full analytic Jacobian against AD for a model with multiple particle types and spatial dependence
 * of volume fractions
 * @param [in] uoType Unit operation type
 */
void testJacobianSpatiallyMixedParticleTypes(const std::string& uoType);

/**
 * @brief Checks the (analytic) time derivative Jacobian against FD for a model with multiple particle types
 * @details Uses centered finite differences.
 * @param [in] jpp Unit operation configuration
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testTimeDerivativeJacobianMixedParticleTypesFD(cadet::JsonParameterProvider& jpp, double h, double absTol,
													double relTol);

/**
 * @brief Checks the (analytic) time derivative Jacobian against FD for a model with multiple particle types
 * @details Uses centered finite differences.
 * @param [in] uoType Unit operation type
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testTimeDerivativeJacobianMixedParticleTypesFD(const std::string& uoType, double h, double absTol, double relTol);

/**
 * @brief Checks the bottom macro row and right macro column of the Jacobian against FD for a model with multiple
 * particle types
 * @details Uses centered finite differences to check the flux part of the Jacobian.
 * @param [in] uoType Unit operation type
 * @param [in] h Step size of centered finite differences
 * @param [in] absTol Absolute error tolerance
 * @param [in] relTol Relative error tolerance
 */
void testArrowHeadJacobianSpatiallyMixedParticleTypes(const std::string& uoType, double h, double absTol,
													  double relTol);

} // namespace particle
} // namespace test
} // namespace cadet

#endif // CADETTEST_PARTICLEHELPER_HPP_
