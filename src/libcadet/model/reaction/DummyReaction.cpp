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

#include "model/ReactionModel.hpp"
#include "ParamIdUtil.hpp"
#include "CompileTimeConfig.hpp"

namespace cadet
{

namespace model
{

/**
 * @brief Defines a dummy reaction model
 * @details The dummy reaction model does absolutely nothing.
 */
class DummyDynamicReaction : public IDynamicReactionModel
{
public:

	virtual ~DummyDynamicReaction() CADET_NOEXCEPT { }

	static const char* identifier() { return "NONE"; }
	virtual const char* name() const CADET_NOEXCEPT { return "NONE"; }

	virtual bool requiresConfiguration() const CADET_NOEXCEPT { return false; }
	virtual bool usesParamProviderInDiscretizationConfig() const CADET_NOEXCEPT { return false; }

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBound, unsigned int const* boundOffset)
	{
		return true;
	}

	virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx) { return true; }

	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size) { }

	virtual bool hasParameter(const ParameterId& pId) const { return false; }

	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const { return std::unordered_map<ParameterId, double>(); }

	virtual bool setParameter(const ParameterId& pId, int value) { return false; }
	virtual bool setParameter(const ParameterId& pId, double value) { return false; }
	virtual bool setParameter(const ParameterId& pId, bool value) { return false; }

	virtual active* getParameter(const ParameterId& pId) { return nullptr; }

	virtual bool dependsOnTime() const CADET_NOEXCEPT { return false; }
	virtual bool requiresWorkspace() const CADET_NOEXCEPT { return false; }
	virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		return 0;
	}

	virtual int residualLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* y,
		active* res, const active& factor, LinearBufferAllocator workSpace) const { return 0; }

	virtual int residualLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* y,
		active* res, double factor, LinearBufferAllocator workSpace) const { return 0; }

	virtual int residualLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y,
		active* res, double factor, LinearBufferAllocator workSpace) const { return 0; }

	virtual int residualLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y,
		double* res, double factor, LinearBufferAllocator workSpace) const { return 0; }

	virtual void analyticJacobianLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor, linalg::BandMatrix::RowIterator jac, LinearBufferAllocator workSpace) const { }
	virtual void analyticJacobianLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor, linalg::DenseBandedRowIterator jac, LinearBufferAllocator workSpace) const { }
	virtual void analyticJacobianLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor, linalg::BandedSparseRowIterator jac, LinearBufferAllocator workSpace) const { }
	#ifdef ENABLE_DG
	virtual void analyticJacobianLiquidAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor, linalg::BandedEigenSparseRowIterator jac, LinearBufferAllocator workSpace) const { }
	#endif

	virtual int residualCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, active const* yLiquid, active const* ySolid,
		active* resLiquid, active* resSolid, double factor, LinearBufferAllocator workSpace) const { return 0; }

	virtual int residualCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid,
		active* resLiquid, active* resSolid, double factor, LinearBufferAllocator workSpace) const { return 0; }

	virtual int residualCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid,
		double* resLiquid, double* resSolid, double factor, LinearBufferAllocator workSpace) const { return 0; }

	virtual void analyticJacobianCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid, double factor, linalg::BandMatrix::RowIterator jacLiquid, linalg::BandMatrix::RowIterator jacSolid, LinearBufferAllocator workSpace) const { }
	virtual void analyticJacobianCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid, double factor, linalg::DenseBandedRowIterator jacLiquid, linalg::DenseBandedRowIterator jacSolid, LinearBufferAllocator workSpace) const { }
	virtual void analyticJacobianCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid, double factor, linalg::BandMatrix::RowIterator jacLiquid, linalg::DenseBandedRowIterator jacSolid, LinearBufferAllocator workSpace) const { }
	#ifdef ENABLE_DG
	virtual void analyticJacobianCombinedAdd(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid, double factor, linalg::BandedEigenSparseRowIterator jacLiquid, linalg::DenseBandedRowIterator jacSolid, LinearBufferAllocator workSpace) const { }
	#endif

	virtual unsigned int numReactionsLiquid() const CADET_NOEXCEPT { return 0; }
	virtual unsigned int numReactionsCombined() const CADET_NOEXCEPT { return 0; }

protected:
};


namespace reaction
{
	void registerDummyReaction(std::unordered_map<std::string, std::function<model::IDynamicReactionModel*()>>& reactions)
	{
		reactions[DummyDynamicReaction::identifier()] = []() { return new DummyDynamicReaction(); };
		reactions["DUMMY"] = []() { return new DummyDynamicReaction(); };
	}
}  // namespace reaction

} // namespace model
} // namespace cadet
