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

#include "model/paramdep/ParameterDependenceBase.hpp"
#include "ParamReaderHelper.hpp"
#include "cadet/Exceptions.hpp"
#include "SimulationTypes.hpp"
#include "cadet/ParameterId.hpp"

#include <sstream>
#include <vector>
#include <iomanip>

namespace cadet
{

namespace model
{

/**
 * @brief Defines a parameter dependence that outputs constant 0.0
 */
class ConstantZeroParameterStateDependence : public ParameterStateDependenceBase
{
public:

	ConstantZeroParameterStateDependence() { }
	virtual ~ConstantZeroParameterStateDependence() CADET_NOEXCEPT { }

	static const char* identifier() { return "CONSTANT_ZERO"; }
	virtual const char* name() const CADET_NOEXCEPT { return ConstantZeroParameterStateDependence::identifier(); }

	virtual int jacobianElementsPerRowLiquid() const CADET_NOEXCEPT { return 0; }
	virtual int jacobianElementsPerRowCombined() const CADET_NOEXCEPT { return 0; }

	virtual void analyticJacobianLiquidAdd(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }
	virtual void analyticJacobianCombinedAddLiquid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }
	virtual void analyticJacobianCombinedAddSolid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }

	CADET_PARAMETERSTATEDEPENDENCE_BOILERPLATE

protected:

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, const std::string& name)
	{
		return true;
	}

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type liquidParameterImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* y, int comp) const
	{
		return 0.0;
	}

	template <typename RowIterator>
	void analyticJacobianLiquidAddImpl(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, RowIterator jac) const { }

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type combinedParameterLiquidImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* yLiquid, StateType const* ySolid, int comp) const
	{
		return 0.0;
	}

	template <typename RowIterator>
	void analyticJacobianCombinedAddLiquidImpl(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, RowIterator jac) const { }

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type combinedParameterSolidImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* yLiquid, StateType const* ySolid, int bnd) const
	{
		return 0.0;
	}

	template <typename RowIterator>
	void analyticJacobianCombinedAddSolidImpl(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, RowIterator jac) const { }
};


/**
 * @brief Defines a parameter dependence that outputs constant 1.0
 */
class ConstantOneParameterStateDependence : public ParameterStateDependenceBase
{
public:

	ConstantOneParameterStateDependence() { }
	virtual ~ConstantOneParameterStateDependence() CADET_NOEXCEPT { }

	static const char* identifier() { return "CONSTANT_ONE"; }
	virtual const char* name() const CADET_NOEXCEPT { return ConstantOneParameterStateDependence::identifier(); }

	virtual int jacobianElementsPerRowLiquid() const CADET_NOEXCEPT { return 0; }
	virtual int jacobianElementsPerRowCombined() const CADET_NOEXCEPT { return 0; }

	virtual void analyticJacobianLiquidAdd(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }
	virtual void analyticJacobianCombinedAddLiquid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }
	virtual void analyticJacobianCombinedAddSolid(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, int row, linalg::DoubleSparseMatrix& jac) const { }

	CADET_PARAMETERSTATEDEPENDENCE_BOILERPLATE

protected:

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, const std::string& name)
	{
		return true;
	}

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type liquidParameterImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* y, int comp) const
	{
		return 1.0;
	}

	template <typename RowIterator>
	void analyticJacobianLiquidAddImpl(const ColumnPosition& colPos, double param, double const* y, int comp, double factor, int offset, RowIterator jac) const { }

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type combinedParameterLiquidImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* yLiquid, StateType const* ySolid, int comp) const
	{
		return 1.0;
	}

	template <typename RowIterator>
	void analyticJacobianCombinedAddLiquidImpl(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int comp, double factor, int offset, RowIterator jac) const { }

	template <typename StateType, typename ParamType>
	typename DoubleActivePromoter<StateType, ParamType>::type combinedParameterSolidImpl(const ColumnPosition& colPos, const ParamType& param, StateType const* yLiquid, StateType const* ySolid, int bnd) const
	{
		return 1.0;
	}

	template <typename RowIterator>
	void analyticJacobianCombinedAddSolidImpl(const ColumnPosition& colPos, double param, double const* yLiquid, double const* ySolid, int bnd, double factor, int offset, RowIterator jac) const { }
};


/**
 * @brief Defines a parameter dependence that outputs constant 0.0
 */
class ConstantZeroParameterParameterDependence : public ParameterParameterDependenceBase
{
public:

	ConstantZeroParameterParameterDependence() { }
	virtual ~ConstantZeroParameterParameterDependence() CADET_NOEXCEPT { }

	static const char* identifier() { return "CONSTANT_ZERO"; }
	virtual const char* name() const CADET_NOEXCEPT { return ConstantZeroParameterParameterDependence::identifier(); }

	CADET_PARAMETERPARAMETERDEPENDENCE_BOILERPLATE

protected:

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, BoundStateIdx bndIdx, const std::string& name)
	{
		return true;
	}

	template <typename ParamType>
	ParamType getValueImpl(const IModel& model, const ColumnPosition& colPos, int comp, int parType, int bnd) const
	{
		return 0.0;
	}

	template <typename ParamType>
	ParamType getValueImpl(const IModel& model, const ColumnPosition& colPos, int comp, int parType, int bnd, const ParamType& val) const
	{
		return 0.0;
	}

};


/**
 * @brief Defines a parameter dependence that outputs constant 1.0
 */
class ConstantOneParameterParameterDependence : public ParameterParameterDependenceBase
{
public:

	ConstantOneParameterParameterDependence() { }
	virtual ~ConstantOneParameterParameterDependence() CADET_NOEXCEPT { }

	static const char* identifier() { return "CONSTANT_ONE"; }
	virtual const char* name() const CADET_NOEXCEPT { return ConstantOneParameterParameterDependence::identifier(); }

	CADET_PARAMETERPARAMETERDEPENDENCE_BOILERPLATE

protected:

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, BoundStateIdx bndIdx, const std::string& name)
	{
		return true;
	}

	template <typename ParamType>
	ParamType getValueImpl(const IModel& model, const ColumnPosition& colPos, int comp, int parType, int bnd) const
	{
		return 1.0;
	}

	template <typename ParamType>
	ParamType getValueImpl(const IModel& model, const ColumnPosition& colPos, int comp, int parType, int bnd, const ParamType& val) const
	{
		return 1.0;
	}

};


namespace paramdep
{
	void registerDummyParamDependence(std::unordered_map<std::string, std::function<model::IParameterStateDependence*()>>& paramDeps)
	{
		paramDeps[ConstantOneParameterStateDependence::identifier()] = []() { return new ConstantOneParameterStateDependence(); };
		paramDeps[ConstantZeroParameterStateDependence::identifier()] = []() { return new ConstantZeroParameterStateDependence(); };
		paramDeps["NONE"] = []() { return new ConstantOneParameterStateDependence(); };
	}

	void registerDummyParamDependence(std::unordered_map<std::string, std::function<model::IParameterParameterDependence*()>>& paramDeps)
	{
		paramDeps[ConstantOneParameterParameterDependence::identifier()] = []() { return new ConstantOneParameterParameterDependence(); };
		paramDeps[ConstantZeroParameterParameterDependence::identifier()] = []() { return new ConstantZeroParameterParameterDependence(); };
		paramDeps["NONE"] = []() { return new ConstantOneParameterParameterDependence(); };
	}
}  // namespace paramdep

}  // namespace model

}  // namespace cadet
