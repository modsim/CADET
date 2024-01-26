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
 * Template for external function support in binding models.
 * This file serves as a template for generating parameter handler classes that
 * hold, configure, and update parameters, which may also depend on external
 * functions.
 */

/* <codegentemplate> */
#include "Memory.hpp"

#include <tuple>

namespace cadet
{

namespace model
{

class {{ name }} : public cadet::model::ConstParamHandlerBase
{
public:
	struct ConstParams
	{
{% for p in parameters %}
	typename {{ p/type }}::storage_t {{ p/varName }};
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}

		{% if length(p/varName) > 1 %}

			{% for v in p/varName %}

	typename {{ p/type }}::storage_t {{ v }};

			{% endfor %}

		{% else %}

	typename {{ p/type }}::storage_t {{ p/varName }};

		{% endif %}

	{% endfor %}
{% endif %}
	};

	typedef ConstParams params_t;
	typedef ConstParams const* ParamsHandle;

	static inline const char* identifier() CADET_NOEXCEPT;

	{{ name }}() CADET_NOEXCEPT :
{% for p in parameters %}
		_{{ p/varName }}(&_localParams.{{ p/varName }})
	{% if not is_last %}
		,
	{% endif %}
{% endfor %}
{% if exists("constantParameters") %}
		,
	{% for p in constantParameters %}
		{% if length(p/varName) > 1 %}
			_{{ p/objName }}({% for v in p/varName %} &_localParams.{{ v }} {% if not is_last %},{% endif %} {% endfor %})
		{% else %}
			_{{ p/varName }}(&_localParams.{{ p/varName }})
		{% endif %}
		{% if not is_last %},{% endif %}
	{% endfor %}
{% endif %}
	{ }

	inline bool configure(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBoundStates)
	{
{% for p in parameters %}
	{% if not existsIn(p, "skipConfig") %}
		_{{ p/varName }}.configure("{{ p/confName }}", paramProvider, nComp, nBoundStates);
	{% endif %}
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if not existsIn(p, "skipConfig") %}
			{% if length(p/varName) > 1 %}
				{% if existsIn(p, "default") %}
					_{{ p/objName }}.configure("{{ p/confPrefix }}", paramProvider, nComp, nBoundStates, {{ p/default }});
				{% else %}
					_{{ p/objName }}.configure("{{ p/confPrefix }}", paramProvider, nComp, nBoundStates);
				{% endif %}
			{% else %}
				{% if existsIn(p, "default") %}
					_{{ p/varName }}.configure("{{ p/confName }}", paramProvider, nComp, nBoundStates, {{ p/default }});
				{% else %}
					_{{ p/varName }}.configure("{{ p/confName }}", paramProvider, nComp, nBoundStates);
				{% endif %}
			{% endif %}
		{% endif %}
	{% endfor %}
{% endif %}
		return validateConfig(nComp, nBoundStates);
	}

	inline void registerParameters(std::unordered_map<ParameterId, active*>& parameters, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp, unsigned int const* nBoundStates)
	{
{% for p in parameters %}
		_{{ p/varName }}.registerParam("{{ p/confName }}", parameters, unitOpIdx, parTypeIdx, nComp, nBoundStates);
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if length(p/varName) > 1 %}
			_{{ p/objName }}.registerParam("{{ p/confPrefix }}", parameters, unitOpIdx, parTypeIdx, nComp, nBoundStates);
		{% else %}
			_{{ p/varName }}.registerParam("{{ p/confName }}", parameters, unitOpIdx, parTypeIdx, nComp, nBoundStates);
		{% endif %}
	{% endfor %}
{% endif %}
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp, unsigned int const* nBoundStates)
	{
{% for p in parameters %}
		_{{ p/varName }}.reserve(numElem, numSlices, nComp, nBoundStates);
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if length(p/varName) > 1 %}
			_{{ p/objName }}.reserve(numElem, numSlices, nComp, nBoundStates);
		{% else %}
			_{{ p/varName }}.reserve(numElem, numSlices, nComp, nBoundStates);
		{% endif %}
	{% endfor %}
{% endif %}
	}

	inline ParamsHandle update(double t, unsigned int secIdx, const ColumnPosition& colPos, unsigned int nComp, unsigned int const* nBoundStates, LinearBufferAllocator& workSpace) const
	{
		return &_localParams;
	}

	inline std::tuple<ParamsHandle, ParamsHandle> updateTimeDerivative(double t, unsigned int secIdx, const ColumnPosition& colPos, unsigned int nComp, unsigned int const* nBoundStates, LinearBufferAllocator& workSpace) const
	{
		return std::make_tuple<ParamsHandle, ParamsHandle>(&_localParams, nullptr);
	}

{% for p in parameters %}
	inline const {{ p/type }}& {{ p/varName }}() const CADET_NOEXCEPT { return _{{ p/varName }}; }
	inline {{ p/type }}& {{ p/varName }}() CADET_NOEXCEPT { return _{{ p/varName }}; }
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if length(p/varName) == 1 %}
			inline const {{ p/type }}& {{ p/varName }}() const CADET_NOEXCEPT { return _{{ p/varName }}; }
			inline {{ p/type }}& {{ p/varName }}() CADET_NOEXCEPT { return _{{ p/varName }}; }
		{% else %}
			inline const {{ p/type }}& {{ p/objName }}() const CADET_NOEXCEPT { return _{{ p/objName }}; }
			inline {{ p/type }}& {{ p/objName }}() CADET_NOEXCEPT { return _{{ p/objName }}; }
		{% endif %}
	{% endfor %}
{% endif %}

	inline char const* prefixInConfiguration() const CADET_NOEXCEPT { return ""; }

protected:
	inline bool validateConfig(unsigned int nComp, unsigned int const* nBoundStates);
	
	ConstParams _localParams;

{% for p in parameters %}
		{{ p/type }} _{{ p/varName }};
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if length(p/varName) > 1 %}
			{{ p/type }} _{{ p/objName }};
		{% else %}
			{{ p/type }} _{{ p/varName }};
		{% endif %}
	{% endfor %}
{% endif %}
};

class {{ externalName }} : public cadet::model::ExternalParamHandlerBase
{
public:
	struct ConstParams
	{
{% if exists("constantParameters") %}
	{% for p in constantParameters %}

		{% if length(p/varName) > 1 %}

			{% for v in p/varName %}

	typename {{ p/type }}::storage_t {{ v }};

			{% endfor %}

		{% else %}

	typename {{ p/type }}::storage_t {{ p/varName }};

		{% endif %}

	{% endfor %}
{% endif %}
	};

	struct VariableParams
	{
{% for p in parameters %}
	typename util::localVersionOf<{{ p/type }}::storage_t>::type {{ p/varName }};
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}

		{% if length(p/varName) > 1 %}

			{% for v in p/varName %}

	typename {{ p/type }}::storage_t {{ v }};

			{% endfor %}

		{% else %}

	typename {{ p/type }}::storage_t {{ p/varName }};

		{% endif %}

	{% endfor %}
{% endif %}
	};
	
	typedef VariableParams params_t;
	typedef ConstBufferedScalar<params_t> ParamsHandle;

	static inline const char* identifier() CADET_NOEXCEPT;
	
	{{ externalName }}() CADET_NOEXCEPT
{% if exists("constantParameters") %}
		:
	{% for p in constantParameters %}
		{% if length(p/varName) > 1 %}
			_{{ p/objName }}({% for v in p/varName %} &_constParams.{{ v }} {% if not is_last %},{% endif %} {% endfor %})
		{% else %}
			_{{ p/varName }}(&_constParams.{{ p/varName }})
		{% endif %}
		{% if not is_last %},{% endif %}
	{% endfor %}
{% endif %}
	{ }

	inline bool configure(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBoundStates)
	{
{% for p in parameters %}
	{% if not existsIn(p, "skipConfig") %}
		_{{ p/varName }}.configure("{{ p/confName }}", paramProvider, nComp, nBoundStates);
	{% endif %}
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if not existsIn(p, "skipConfig") %}
			{% if length(p/varName) > 1 %}
				{% if existsIn(p, "default") %}
					_{{ p/objName }}.configure("{{ p/confPrefix }}", paramProvider, nComp, nBoundStates, {{ p/default }});
				{% else %}
					_{{ p/objName }}.configure("{{ p/confPrefix }}", paramProvider, nComp, nBoundStates);
				{% endif %}
			{% else %}
				{% if existsIn(p, "default") %}
					_{{ p/varName }}.configure("{{ p/confName }}", paramProvider, nComp, nBoundStates, {{ p/default }});
				{% else %}
					_{{ p/varName }}.configure("{{ p/confName }}", paramProvider, nComp, nBoundStates);
				{% endif %}
			{% endif %}
		{% endif %}
	{% endfor %}
{% endif %}
		ExternalParamHandlerBase::configure(paramProvider, {{ length(parameters) }});
		return validateConfig(nComp, nBoundStates);
	}

	inline void registerParameters(std::unordered_map<ParameterId, active*>& parameters, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx, unsigned int nComp, unsigned int const* nBoundStates)
	{
{% for p in parameters %}
		_{{ p/varName }}.registerParam("{{ p/confName }}", parameters, unitOpIdx, parTypeIdx, nComp, nBoundStates);
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if length(p/varName) > 1 %}
			_{{ p/objName }}.registerParam("{{ p/confPrefix }}", parameters, unitOpIdx, parTypeIdx, nComp, nBoundStates);
		{% else %}
			_{{ p/varName }}.registerParam("{{ p/confName }}", parameters, unitOpIdx, parTypeIdx, nComp, nBoundStates);
		{% endif %}
	{% endfor %}
{% endif %}
	}

	inline void reserve(unsigned int numElem, unsigned int numSlices, unsigned int nComp, unsigned int const* nBoundStates)
	{
{% for p in parameters %}
		_{{ p/varName }}.reserve(numElem, numSlices, nComp, nBoundStates);
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if length(p/varName) > 1 %}
			_{{ p/objName }}.reserve(numElem, numSlices, nComp, nBoundStates);
		{% else %}
			_{{ p/varName }}.reserve(numElem, numSlices, nComp, nBoundStates);
		{% endif %}
	{% endfor %}
{% endif %}
	}

	inline ParamsHandle update(double t, unsigned int secIdx, const ColumnPosition& colPos, unsigned int nComp, unsigned int const* nBoundStates, LinearBufferAllocator& workSpace) const
	{
		BufferedScalar<params_t> localParams = workSpace.scalar<params_t>();
		BufferedArray<double> extFunBuffer = workSpace.array<double>({{ length(parameters) }});
		evaluateExternalFunctions(t, secIdx, colPos, {{ length(parameters) }}, static_cast<double*>(extFunBuffer));
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if length(p/varName) > 1 %}
			{% for v in p/varName %}
				localParams->{{ v }} = _constParams.{{ v }};
			{% endfor %}
		{% else %}
			localParams->{{ p/varName }} = _constParams.{{ p/varName }};
		{% endif %}
	{% endfor %}
{% endif %}

{% for p in parameters %}
		_{{ p/varName }}.prepareCache(localParams->{{ p/varName }}, workSpace);
		_{{ p/varName }}.update(cadet::util::dataOfLocalVersion(localParams->{{ p/varName }}), extFunBuffer[{{ index }}], nComp, nBoundStates);
{% endfor %}

		return localParams;
	}

	inline std::tuple<ParamsHandle, ParamsHandle> updateTimeDerivative(double t, unsigned int secIdx, const ColumnPosition& colPos, unsigned int nComp, unsigned int const* nBoundStates, LinearBufferAllocator& workSpace) const
	{
		BufferedScalar<params_t> localParams = workSpace.scalar<params_t>();
		BufferedScalar<params_t> p = workSpace.scalar<params_t>();
		BufferedArray<double> extFunBuffer = workSpace.array<double>({{ length(parameters) }});
		BufferedArray<double> extDerivBuffer = workSpace.array<double>({{ length(parameters) }});
		evaluateExternalFunctions(t, secIdx, colPos, {{ length(parameters) }}, static_cast<double*>(extFunBuffer));
		evaluateTimeDerivativeExternalFunctions(t, secIdx, colPos, {{ length(parameters) }}, static_cast<double*>(extDerivBuffer));

{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if length(p/varName) > 1 %}
			{% for v in p/varName %}
				localParams->{{ v }} = _constParams.{{ v }};
				p->{{ v }} = _constParams.{{ v }};
			{% endfor %}
		{% else %}
			localParams->{{ p/varName }} = _constParams.{{ p/varName }};
			p->{{ p/varName }} = _constParams.{{ p/varName }};
		{% endif %}
	{% endfor %}
{% endif %}

{% for p in parameters %}
		_{{ p/varName }}.prepareCache(localParams->{{ p/varName }}, workSpace);
		_{{ p/varName }}.update(cadet::util::dataOfLocalVersion(localParams->{{ p/varName }}), extFunBuffer[{{ index }}], nComp, nBoundStates);

		_{{ p/varName }}.prepareCache(p->{{ p/varName }}, workSpace);
		_{{ p/varName }}.updateTimeDerivative(cadet::util::dataOfLocalVersion(p->{{ p/varName }}), extFunBuffer[{{ index }}], extDerivBuffer[{{ index }}], nComp, nBoundStates);
{% endfor %}

		return std::make_tuple<ParamsHandle, ParamsHandle>(std::move(localParams), std::move(p));
	}

	inline std::size_t cacheSize(unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		return 2 * sizeof(params_t) + alignof(params_t) + 2 * {{ length(parameters) }} * sizeof(double) + alignof(double) + 2 * (
{% for p in parameters %}
		_{{ p/varName }}.additionalDynamicMemory(nComp, totalNumBoundStates, nBoundStates) {% if not is_last %} + {% endif %}
{% endfor %}
		);
	}

{% for p in parameters %}
	inline const External{{ p/type }}& {{ p/varName }}() const CADET_NOEXCEPT { return _{{ p/varName }}; }
	inline External{{ p/type }}& {{ p/varName }}() CADET_NOEXCEPT { return _{{ p/varName }}; }
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if length(p/varName) == 1 %}
			inline const {{ p/type }}& {{ p/varName }}() const CADET_NOEXCEPT { return _{{ p/varName }}; }
			inline {{ p/type }}& {{ p/varName }}() CADET_NOEXCEPT { return _{{ p/varName }}; }
		{% else %}
			inline const {{ p/type }}& {{ p/objName }}() const CADET_NOEXCEPT { return _{{ p/objName }}; }
			inline {{ p/type }}& {{ p/objName }}() CADET_NOEXCEPT { return _{{ p/objName }}; }
		{% endif %}
	{% endfor %}
{% endif %}

	inline char const* prefixInConfiguration() const CADET_NOEXCEPT { return "EXT_"; }

protected:
	inline bool validateConfig(unsigned int nComp, unsigned int const* nBoundStates);
	
	ConstParams _constParams;

{% for p in parameters %}
		External{{ p/type }} _{{ p/varName }};
{% endfor %}
{% if exists("constantParameters") %}
	{% for p in constantParameters %}
		{% if length(p/varName) > 1 %}
			{{ p/type }} _{{ p/objName }};
		{% else %}
			{{ p/type }} _{{ p/varName }};
		{% endif %}
	{% endfor %}
{% endif %}
};

} // namespace model
} // namespace cadet
