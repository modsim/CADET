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
 * Defines the BindingModel interface.
 */

#ifndef LIBCADET_PHASETRANSITIONMODELINTERFACE_HPP_ 
#define LIBCADET_PHASETRANSITIONMODELINTERFACE_HPP_

#include <unordered_map>

#include "CompileTimeConfig.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/ParameterId.hpp"
#include "linalg/DenseMatrix.hpp"
#include "linalg/BandMatrix.hpp"

#ifdef ENABLE_DG
	#include "linalg/BandedEigenSparseRowIterator.hpp"
#endif

#include "AutoDiff.hpp"
#include "SimulationTypes.hpp"
#include "Memory.hpp"

namespace cadet
{

class IParameterProvider;
class IExternalFunction;

struct ColumnPosition;

namespace model
{

class IPhaseTransitionModel
{        
public:

	virtual ~IPhaseTransitionModel() CADET_NOEXCEPT { }

	virtual const char* name() const CADET_NOEXCEPT = 0;

	//virtual bool requiresConfiguration() const CADET_NOEXCEPT = 0;

	//virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBound, unsigned int const* boundOffset) = 0;

	virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx) = 0;

	// virtual void fillBoundPhaseInitialParameters(ParameterId* params, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx) const CADET_NOEXCEPT = 0;

	//virtual void fillChannelInitialParameters(ParameterId* params, UnitOpIdx unitOpIdx) const CADET_NOEXCEPT = 0;


	//virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size) = 0; //X 

	//virtual std::unordered_map<ParameterId, double> getAllParameterValues() const = 0;
	
	//static const char* identifier() CADET_NOEXCEPT { return "IPhaseTransitionModel"; }
	
	//virtual bool hasParameter(const ParameterId& pId) const = 0;
	//virtual bool setParameter(const ParameterId& pId, int value) = 0;
	//virtual bool setParameter(const ParameterId& pId, double value) = 0;
	//virtual bool setParameter(const ParameterId& pId, bool value) = 0;

	//virtual active* getParameter(const ParameterId& pId) = 0;

	//virtual bool hasSalt() const CADET_NOEXCEPT = 0;

	//virtual bool supportsMultistate() const CADET_NOEXCEPT = 0;l bool supportsNonExchange() const CADET_NOEXCEPT = 0;

	
	//virtual bool supportsNonBinding() const CADET_NOEXCEPT = 0;

	//virtual bool hasQuasiStationaryReactions() const CADET_NOEXCEPT = 0;

	//virtual bool hasDynamicReactions() const CADET_NOEXCEPT = 0;

	//virtual bool dependsOnTime() const CADET_NOEXCEPT = 0;

	//virtual bool requiresWorkspace() const CADET_NOEXCEPT = 0;

	//virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nBoundStates) const CADET_NOEXCEPT = 0;

	//virtual unsigned int requiredADdirs() const CADET_NOEXCEPT = 0;

	virtual int flux(unsigned int nChannel, unsigned int nComp, unsigned int nCol, std::vector<active> exchangeMatrix, std::vector<active> crossSections, active const* y, active* res) const = 0;
	//virtual int flux(unsigned int nChannel, unsigned int nComp, unsigned int nCol, std::vector<active> exchangeMatrix, std::vector<active> crossSections, active const* y, double* res) const = 0;
	//virtual int flux(unsigned int nChannel, unsigned int nComp, unsigned int nCol, std::vector<active> exchangeMatrix, std::vector<active> crossSections, double const* y, active* res) const = 0;
	//virtual int flux(unsigned int nChannel, unsigned int nComp, unsigned int nCol, std::vector<active> exchangeMatrix, std::vector<active> crossSections, double const* y, double* res) const = 0;

	virtual void analyticJacobian(unsigned int nChannel, unsigned int nComp, unsigned int nCol, std::vector<active> _exchangeMatrix, active const* y, active* res, linalg::CompressedSparseMatrix jac) const = 0;
	//virtual void analyticJacobian(unsigned int nChannel, unsigned int nComp, unsigned int nCol, std::vector<active> _exchangeMatrix, active const* y, double* res, linalg::BandMatrix::RowIterator jac) const = 0;
	//virtual void analyticJacobian(unsigned int nChannel, unsigned int nComp, unsigned int nCol, std::vector<active> _exchangeMatrix, double const* y, active* res, linalg::BandMatrix::RowIterator jac) const = 0;
	// virtual void analyticJacobian(unsigned int nChannel, unsigned int nComp, unsigned int nCol, std::vector<active> _exchangeMatrix, double const* y, double* res, linalg::BandMatrix::RowIterator jac) const = 0;
#ifdef ENABLE_DG
	//virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, linalg::BandedEigenSparseRowIterator jac, LinearBufferAllocator workSpace) const = 0;
#endif
	virtual void timeDerivativeQuasiStationaryFluxes(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yCp, double const* y, double* dResDt, LinearBufferAllocator workSpace) const = 0;

	virtual int const* reactionQuasiStationarity() const CADET_NOEXCEPT = 0;

	virtual bool preConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const = 0;

	virtual void postConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const = 0;

protected:
};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_PHASETRANSITIONMODELINTERFACE_HPP_
