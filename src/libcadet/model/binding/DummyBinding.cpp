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

#include "model/BindingModel.hpp"
#include "ParamIdUtil.hpp"
#include "model/ModelUtils.hpp"

#include <unordered_map>
#include <vector>

namespace cadet
{

namespace model
{

/**
 * @brief Defines the dummy binding model
 * @details The dummy binding model does absolutely nothing.
 */
class DummyBinding : public IBindingModel
{
public:

	DummyBinding() : _stateQuasistationarity(0, false) { }
	virtual ~DummyBinding() CADET_NOEXCEPT { }

	static const char* identifier() { return "NONE"; }
	virtual const char* name() const CADET_NOEXCEPT { return "NONE"; }
	virtual bool requiresConfiguration() const CADET_NOEXCEPT { return false; }
	virtual bool usesParamProviderInDiscretizationConfig() const CADET_NOEXCEPT { return false; }

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBound, unsigned int const* boundOffset)
	{
		_nComp = nComp;
		_nBoundStates = nBound;
		_stateQuasistationarity.resize(numBoundStates(nBound, nComp), false);
		return true;
	}

	virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx)
	{
		return true;
	}

	virtual void fillBoundPhaseInitialParameters(ParameterId* params, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx) const CADET_NOEXCEPT
	{
		unsigned int ctr = 0;
		for (int c = 0; c < _nComp; ++c)
		{
			for (unsigned int bp = 0; bp < _nBoundStates[c]; ++bp, ++ctr)
				params[ctr] = makeParamId(hashString("INIT_Q"), unitOpIdx, c, parTypeIdx, bp, ReactionIndep, SectionIndep);
		}
	}

	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const
	{
		return std::unordered_map<ParameterId, double>();
	}

	virtual bool hasParameter(const ParameterId& pId) const
	{
		return false;
	}

	virtual bool setParameter(const ParameterId& pId, int value)
	{
		return false;
	}

	virtual bool setParameter(const ParameterId& pId, double value)
	{
		return false;
	}

	virtual bool setParameter(const ParameterId& pId, bool value)
	{
		return false;
	}

	virtual active* getParameter(const ParameterId& pId)
	{
		return nullptr;
	}

	virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		return 0;
	}

	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos,
		active const* y, active const* yCp, active* res, LinearBufferAllocator workSpace, WithParamSensitivity) const
	{
		return 0;
	}

	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos,
		active const* y, active const* yCp, active* res, LinearBufferAllocator workSpace, WithoutParamSensitivity) const
	{
		return 0;
	}

	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos,
		double const* y, double const* yCp, active* res, LinearBufferAllocator workSpace) const
	{
		return 0;
	}

	virtual int flux(double t, unsigned int secIdx, const ColumnPosition& colPos,
		double const* y, double const* yCp, double* res, LinearBufferAllocator workSpace) const
	{
		return 0;
	}

	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size) { }

	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, linalg::BandMatrix::RowIterator jac, LinearBufferAllocator workSpace) const
	{
	}

	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, linalg::DenseBandedRowIterator jac, LinearBufferAllocator workSpace) const
	{
	}

	virtual void analyticJacobian(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, int offsetCp, linalg::BandedEigenSparseRowIterator jac, LinearBufferAllocator workSpace) const
	{
	}

	virtual void timeDerivativeQuasiStationaryFluxes(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yCp, double const* y, double* dResDt, LinearBufferAllocator workSpace) const { }

	virtual bool hasSalt() const CADET_NOEXCEPT { return false; }
	virtual bool supportsMultistate() const CADET_NOEXCEPT { return true; }
	virtual bool supportsNonBinding() const CADET_NOEXCEPT { return true; }
	virtual bool hasQuasiStationaryReactions() const CADET_NOEXCEPT { return false; }
	virtual bool hasDynamicReactions() const CADET_NOEXCEPT { return true; }
	virtual bool dependsOnTime() const CADET_NOEXCEPT { return false; }
	virtual bool requiresWorkspace() const CADET_NOEXCEPT { return false; }
	virtual bool implementsAnalyticJacobian() const CADET_NOEXCEPT { return true; }
	virtual int const* reactionQuasiStationarity() const CADET_NOEXCEPT { return _stateQuasistationarity.data(); }

	virtual bool preConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const { return false; }
	virtual void postConsistentInitialState(double t, unsigned int secIdx, const ColumnPosition& colPos, double* y, double const* yCp, LinearBufferAllocator workSpace) const { }

protected:
	int _nComp; //!< Number of components
	unsigned int const* _nBoundStates; //!< Array with number of bound states for each component
	std::vector<int> _stateQuasistationarity; //!< Determines whether each bound state is quasi-stationary (@c true) or not (@c false)
};

namespace binding
{
	void registerDummyModel(std::unordered_map<std::string, std::function<model::IBindingModel*()>>& bindings)
	{
		bindings[DummyBinding::identifier()] = []() { return new DummyBinding(); };
		bindings["DUMMY"] = []() { return new DummyBinding(); };
	}
}  // namespace binding

}  // namespace model

}  // namespace cadet
