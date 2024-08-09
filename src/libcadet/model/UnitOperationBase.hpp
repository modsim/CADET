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
 * Provides a base class for unit operation models.
 */

#ifndef LIBCADET_UNITOPERATIONBASE_HPP_
#define LIBCADET_UNITOPERATIONBASE_HPP_

#include "model/UnitOperation.hpp"
#include "AutoDiff.hpp"
#include "ParamIdUtil.hpp"
#include "nonlin/Solver.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cadet
{

namespace model
{

class IBindingModel;
class IPhaseTransitionModel;
class IDynamicReactionModel;

/**
 * @brief Base class for unit operation models
 * @details Provides parameter handling.
 */
class UnitOperationBase : public IUnitOperation
{
public:

	UnitOperationBase(UnitOpIdx unitOpIdx);
	virtual ~UnitOperationBase() CADET_NOEXCEPT;

	virtual UnitOpIdx unitOperationId() const CADET_NOEXCEPT { return _unitOpIdx; }

	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const;
	virtual double getParameterDouble(const ParameterId& pId) const;
	virtual bool hasParameter(const ParameterId& pId) const;

	virtual bool setParameter(const ParameterId& pId, int value);
	virtual bool setParameter(const ParameterId& pId, double value);
	virtual bool setParameter(const ParameterId& pId, bool value);

	virtual bool setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue);
	virtual void setSensitiveParameterValue(const ParameterId& id, double value);

	virtual void clearSensParams();
	virtual unsigned int numSensParams() const;

	virtual int residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState,
		const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS, active const* adRes,
		double* const tmp1, double* const tmp2, double* const tmp3);

protected:

	void clearBindingModels() CADET_NOEXCEPT;
	void clearDynamicReactionModels() CADET_NOEXCEPT;
	void clearExchangeModels() CADET_NOEXCEPT;	
	void configureNonlinearSolver(IParameterProvider& paramProvider);
	void configureNonlinearSolver();

	unsigned int maxBindingAdDirs() const CADET_NOEXCEPT;

	UnitOpIdx _unitOpIdx; //!< Unit operation index
	std::vector<IBindingModel*> _binding; //!< Binding model
	std::vector<IPhaseTransitionModel*> _exchange; //!< Exchange model
	bool _singleBinding; //!< Determines whether only a single binding model is present
	std::vector<IDynamicReactionModel*> _dynReaction; //!< Dynamic reaction model
	bool _singleDynReaction; //!< Determines whether only a single dynamic reaction model is present

	typedef std::unordered_map<ParameterId, active*> paramMap_t;
	paramMap_t _parameters; //!< Provides access to all parameters
	std::unordered_set<active*> _sensParams; //!< Holds all parameters with activated AD directions

	nonlin::Solver* _nonlinearSolver; //!< Solver for nonlinear equations (consistent initialization)
};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_UNITOPERATIONBASE_HPP_
