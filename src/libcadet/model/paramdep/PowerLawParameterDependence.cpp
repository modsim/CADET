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
#include "SimulationTypes.hpp"
#include "cadet/ParameterId.hpp"

#include <cmath>

namespace cadet
{
namespace model
{

/**
 * @brief Defines a power law parameter parameter dependence
 * @details The parameter dependence is defined as
 * @f[\begin{align}
	p_{new}(p) = \alpha p^k,
   \end{align} @f]
   where @f$ \alpha @f$ and @f$ k @f$ are (meta-)parameters.
 */
class PowerLawParameterParameterDependence : public ParameterParameterDependenceBase
{
public:

	PowerLawParameterParameterDependence() { }
	virtual ~PowerLawParameterParameterDependence() CADET_NOEXCEPT { }

	static const char* identifier() { return "POWER_LAW"; }
	virtual const char* name() const CADET_NOEXCEPT { return PowerLawParameterParameterDependence::identifier(); }

	CADET_PARAMETERPARAMETERDEPENDENCE_BOILERPLATE

protected:
	active _base;
	active _exponent;
	bool _useAbs;

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, BoundStateIdx bndIdx, const std::string& name)
	{
		const std::string baseName = name + "_BASE";
		const std::string expName = name + "_EXPONENT";
		const std::string flagName = name + "_ABS";

		if (paramProvider.exists(baseName))
			_base = paramProvider.getDouble(baseName);
		else
			_base = 1.0;

		_exponent = paramProvider.getDouble(expName);

		if (paramProvider.exists(flagName))
			_useAbs = paramProvider.getBool(flagName);
		else
			_useAbs = true;

		_parameters[makeParamId(hashStringRuntime(baseName), unitOpIdx, CompIndep, parTypeIdx, bndIdx, ReactionIndep, SectionIndep)] = &_base;
		_parameters[makeParamId(hashStringRuntime(expName), unitOpIdx, CompIndep, parTypeIdx, bndIdx, ReactionIndep, SectionIndep)] = &_exponent;

		return true;
	}

	template <typename ParamType>
	ParamType getValueImpl(const IModel& model, const ColumnPosition& colPos, int comp, int parType, int bnd) const
	{
		return 0.0;
	}

	template <typename ParamType>
	ParamType getValueImpl(const IModel& model, const ColumnPosition& colPos, int comp, int parType, int bnd, ParamType val) const
	{
		using std::pow;
		using std::abs;

		if (_useAbs)
			return static_cast<ParamType>(_base) * pow(abs(val), static_cast<ParamType>(_exponent));
		else
			return static_cast<ParamType>(_base) * pow(val, static_cast<ParamType>(_exponent));
	}

};


namespace paramdep
{
	void registerPowerLawParamDependence(std::unordered_map<std::string, std::function<model::IParameterParameterDependence*()>>& paramDeps)
	{
		paramDeps[PowerLawParameterParameterDependence::identifier()] = []() { return new PowerLawParameterParameterDependence(); };
	}
}  // namespace paramdep

}  // namespace model

}  // namespace cadet
