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
 * Defines interfaces which control the solution export to the user space.
 */

#ifndef LIBCADET_SOLUTIONRECORDER_HPP_
#define LIBCADET_SOLUTIONRECORDER_HPP_

#include "cadet/LibExportImport.hpp"
#include "cadet/cadetCompilerInfo.hpp"
#include "cadet/ParameterId.hpp"

namespace cadet
{

class ISolutionExporter;
class IModel;

/**
 * @brief Interface providing functionality for recording the solution in the user space
 * @details Library users implement this interface which is then used by the cadet::ISimulator
 *          to signal the user that a timestep has been finished and the solution is to be
 *          recorded.
 *
 *          First, the solution of the original system is reported by all unit operations. This
 *          is signaled by beginSolution(). Then, the solution of the forward sensitivity systems
 *          is reported, which is indicated by beginSensitivity().
 */
class CADET_API ISolutionRecorder
{
public:
	virtual ~ISolutionRecorder() CADET_NOEXCEPT
	{
	}

	/**
	 * @brief Clears existing data from memory
	 * @details Removes all stored results from memory.
	 */
	virtual void clear() = 0;

	/**
	 * @brief Prepares the recorder and allows it to allocate memory in advance
	 * @details This function is called after the recorder is passed to ISimulator.
	 *
	 * @param [in] numDofs Number of DOFs in the model
	 * @param [in] numSens Number of forward sensitivities
	 * @param [in] numTimesteps Number of anticipated timesteps, can be @c 0 if no estimate is available
	 */
	virtual void prepare(unsigned int numDofs, unsigned int numSens, unsigned int numTimesteps) = 0;

	/**
	 * @brief Notifies the recorder of a starting time integration
	 * @details This function is called before ISimulator starts the time integration process.
	 *          The ISolutionRecorder can choose to simply append the results of the time integration
	 *          or clear all existing results from memory.
	 *
	 * @param [in] numDofs Number of DOFs in the model
	 * @param [in] numSens Number of forward sensitivities
	 * @param [in] numTimesteps Number of anticipated timesteps, can be @c 0 if no estimate is available
	 */
	virtual void notifyIntegrationStart(unsigned int numDofs, unsigned int numSens, unsigned int numTimesteps) = 0;

	/**
	 * @brief Provides the internal structure of a unit operation which can be used for memory allocation
	 * @details After prepare() or notifyIntegrationStart() has been called, this function is called by each unit
	 * operation.
	 *
	 * @param [in] idx Index of the unit operation
	 * @param [in] model Unit operation model
	 * @param [in] exporter Solution exporter providing access to the solution structure (no solution data is available)
	 */
	virtual void unitOperationStructure(UnitOpIdx idx, const IModel& model, const ISolutionExporter& exporter) = 0;

	/**
	 * @brief Signals the beginning of a new timestep solution export
	 * @details After a timestep has been finished, the solution is exported to the user space.
	 *          This function is called once before all unit operations report their solutions
	 *          via beginUnitOperation().
	 *
	 * @param [in] t Current timepoint
	 */
	virtual void beginTimestep(double t) = 0;

	/**
	 * @brief Signals the export of the given unit operation
	 * @details The solution of the given unit operation identified by its index @p idx is
	 *          provided by the corresponding @p exporter.
	 *
	 * @param [in] idx Index of the unit operation
	 * @param [in] model Unit operation model
	 * @param [in] exporter Solution exporter providing access to the solution
	 */
	virtual void beginUnitOperation(UnitOpIdx idx, const IModel& model, const ISolutionExporter& exporter) = 0;

	/**
	 * @brief Signals the end of the solution export of the current unit operation
	 * @details This function is called after beginTimestep().
	 */
	virtual void endUnitOperation() = 0;

	/**
	 * @brief Signals the end of the solution export of the current timestep
	 * @details This function is called once after all unit operations have reported their
	 *          solutions. The next timestep is about to be computed.
	 */
	virtual void endTimestep() = 0;

	/**
	 * @brief Signals the export of the solution of the original system
	 * @details This function is called once before all unit operations report their
	 *          solutions via beginUnitOperation().
	 */
	virtual void beginSolution() = 0;

	/**
	 * @brief Signals the end of the solution export of the original system
	 * @details This function is called once after all unit operations have reported
	 *          their solutions to the user.
	 */
	virtual void endSolution() = 0;

	/**
	 * @brief Signals the export of the time derivative of the solution of the original system
	 * @details This function is called once before all unit operations report their
	 *          solutions via beginUnitOperation().
	 */
	virtual void beginSolutionDerivative() = 0;

	/**
	 * @brief Signals the end of the time derivative of the solution export of the original system
	 * @details This function is called once after all unit operations have reported
	 *          their solutions to the user.
	 */
	virtual void endSolutionDerivative() = 0;

	/**
	 * @brief Signals the beginning of the export of the sensitivity of the given parameter
	 * @details This function is called once for every sensitive parameter and is succeeded
	 *          by calls reporting the sensitivity in each unit operation (beginUnitOperation()).
	 *
	 * @param [in] pId Parameter Id of the sensitive parameter
	 * @param [in] sensIdx Index of the sensitive parameter (among all sensitive parameters)
	 */
	virtual void beginSensitivity(const ParameterId& pId, unsigned int sensIdx) = 0;

	/**
	 * @brief Signals the end of the sensitivity export.
	 * @details This function is called once for each sensitive parameter after all unit
	 *          operations have reported their sensitivities.
	 *
	 * @param [in] pId Parameter Id of the sensitive parameter
	 * @param [in] sensIdx Index of the sensitive parameter (among all sensitive parameters)
	 */
	virtual void endSensitivity(const ParameterId& pId, unsigned int sensIdx) = 0;

	/**
	 * @brief Signals the beginning of the export of the parameter sensitivity's time derivative
	 * @details This function is called once for every sensitive parameter and is succeeded
	 *          by calls reporting the sensitivity in each unit operation (beginUnitOperation()).
	 *
	 * @param [in] pId Parameter Id of the sensitive parameter
	 * @param [in] sensIdx Index of the sensitive parameter (among all sensitive parameters)
	 */
	virtual void beginSensitivityDerivative(const ParameterId& pId, unsigned int sensIdx) = 0;

	/**
	 * @brief Signals the end of the sensitivity time derivative export.
	 * @details This function is called once for each sensitive parameter after all unit
	 *          operations have reported their sensitivities.
	 *
	 * @param [in] pId Parameter Id of the sensitive parameter
	 * @param [in] sensIdx Index of the sensitive parameter (among all sensitive parameters)
	 */
	virtual void endSensitivityDerivative(const ParameterId& pId, unsigned int sensIdx) = 0;
};

} // namespace cadet

#endif // LIBCADET_SOLUTIONRECORDER_HPP_
