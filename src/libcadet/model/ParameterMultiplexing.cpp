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

#include "model/ParameterMultiplexing.hpp"
#include "cadet/ParameterProvider.hpp"
#include "ParamIdUtil.hpp"
#include "SensParamUtil.hpp"
#include "AutoDiff.hpp"
#include "ParamReaderHelper.hpp"

namespace cadet
{

namespace model
{

bool readAndRegisterMultiplexTypeParam(IParameterProvider& paramProvider,
									   std::unordered_map<ParameterId, active*>& parameters,
									   std::vector<active>& values, const std::string& name, unsigned int nParType,
									   UnitOpIdx uoi)
{
	const bool singleValue = readScalarParameterOrArray(values, paramProvider, name, nParType);
	const StringHash nameHash = hashStringRuntime(name);

	if (singleValue)
		parameters[makeParamId(nameHash, uoi, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] =
			&values[0];
	else
		registerParam1DArray(parameters, values, [=](bool multi, unsigned int type) {
			return makeParamId(nameHash, uoi, CompIndep, type, BoundStateIndep, ReactionIndep, SectionIndep);
		});

	return singleValue;
}

bool multiplexTypeParameterValue(const ParameterId& pId, StringHash nameHash, bool mode, std::vector<active>& data,
								 double value, std::unordered_set<active*> const* sensParams)
{
	if (!mode || (pId.name != nameHash))
		return false;

	if ((pId.section != SectionIndep) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) ||
		(pId.reaction != ReactionIndep) || (pId.particleType != ParTypeIndep))
		return false;
	if (sensParams && !contains(*sensParams, &data[0]))
		return false;

	for (std::size_t i = 0; i < data.size(); ++i)
		data[i].setValue(value);

	return true;
}

bool multiplexTypeParameterAD(const ParameterId& pId, StringHash nameHash, bool mode, std::vector<active>& data,
							  unsigned int adDirection, double adValue, std::unordered_set<active*>& sensParams)
{
	if (!mode || (pId.name != nameHash))
		return false;

	if ((pId.section != SectionIndep) || (pId.component != CompIndep) || (pId.boundState != BoundStateIndep) ||
		(pId.reaction != ReactionIndep) || (pId.particleType != ParTypeIndep))
		return false;

	sensParams.insert(&data[0]);
	for (std::size_t i = 0; i < data.size(); ++i)
		data[i].setADValue(adDirection, adValue);

	return true;
}

MultiplexMode readAndRegisterMultiplexCompTypeSecParam(IParameterProvider& paramProvider,
													   std::unordered_map<ParameterId, active*>& parameters,
													   std::vector<active>& values, const std::string& name,
													   unsigned int nParType, unsigned int nComp, UnitOpIdx uoi)
{
	MultiplexMode mode = MultiplexMode::Independent;
	readScalarParameterOrArray(values, paramProvider, name, 1);
	if (paramProvider.exists(name + "_MULTIPLEX"))
	{
		const int modeConfig = paramProvider.getInt(name + "_MULTIPLEX");
		if (modeConfig == 0)
		{
			mode = MultiplexMode::Component;
			if (values.size() != nComp)
				throw InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name +
												"_MULTIPLEX (should be " + std::to_string(nComp) + ")");
		}
		else if (modeConfig == 1)
		{
			mode = MultiplexMode::ComponentSection;
			if ((values.size() % nComp) != 0)
				throw InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name +
												"_MULTIPLEX (should be positive multiple of " + std::to_string(nComp) +
												")");
		}
		else if (modeConfig == 2)
		{
			mode = MultiplexMode::ComponentType;
			if (values.size() != nComp * nParType)
				throw InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name +
												"_MULTIPLEX (should be " + std::to_string(nComp * nParType) + ")");
		}
		else if (modeConfig == 3)
		{
			mode = MultiplexMode::ComponentSectionType;
			if ((values.size() % (nComp * nParType)) != 0)
				throw InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name +
												"_MULTIPLEX (should be positive multiple of " +
												std::to_string(nComp * nParType) + ")");
		}
	}
	else
	{
		if (values.size() == nComp)
			mode = MultiplexMode::Component;
		else if (values.size() == nComp * nParType)
			mode = MultiplexMode::ComponentType;
		else if ((values.size() % (nComp * nParType)) == 0)
			mode = MultiplexMode::ComponentSectionType;
		else if ((values.size() % nComp) == 0)
			mode = MultiplexMode::ComponentSection;
		else
			throw InvalidParameterException("Could not infer multiplex mode of field " + name + ", set " + name +
											"_MULTIPLEX or change number of elements");
	}

	const StringHash nameHash = hashStringRuntime(name);
	switch (mode)
	{
	case MultiplexMode::Component: {
		std::vector<active> p(nComp * nParType);
		for (unsigned int s = 0; s < nParType; ++s)
			std::copy(values.begin(), values.end(), p.begin() + s * nComp);

		values = std::move(p);

		for (unsigned int s = 0; s < nComp; ++s)
			parameters[makeParamId(nameHash, uoi, s, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] =
				&values[s];
	}
	break;
	case MultiplexMode::ComponentSection: {
		std::vector<active> p(values.size() * nParType);
		for (std::size_t s = 0; s < values.size() / nComp; ++s)
		{
			for (unsigned int pt = 0; pt < nParType; ++pt)
			{
				std::copy(values.begin() + s * nComp, values.begin() + (s + 1) * nComp,
						  p.begin() + s * nParType * nComp + pt * nComp);
			}
		}

		values = std::move(p);

		for (std::size_t s = 0; s < values.size() / nComp; ++s)
		{
			for (unsigned int i = 0; i < nComp; ++i)
				parameters[makeParamId(nameHash, uoi, i, ParTypeIndep, BoundStateIndep, ReactionIndep, s)] =
					&values[s * nParType * nComp + i];
		}
	}
	break;
	case MultiplexMode::ComponentType:
	case MultiplexMode::ComponentSectionType:
		registerParam3DArray(
			parameters, values,
			[=](bool multi, unsigned int sec, unsigned int pt, unsigned int comp) {
				return makeParamId(nameHash, uoi, comp, pt, BoundStateIndep, ReactionIndep, multi ? sec : SectionIndep);
			},
			nComp, nParType);
		break;
	case MultiplexMode::RadialSection:
	case MultiplexMode::Independent:
	case MultiplexMode::ComponentRadial:
	case MultiplexMode::ComponentRadialSection:
	case MultiplexMode::Axial:
	case MultiplexMode::Section:
	case MultiplexMode::Type:
	case MultiplexMode::Radial:
	case MultiplexMode::AxialRadial:
		cadet_assert(false);
		break;
	}

	return mode;
}

bool multiplexCompTypeSecParameterValue(const ParameterId& pId, StringHash nameHash, MultiplexMode mode,
										std::vector<active>& data, unsigned int nParType, unsigned int nComp,
										double value, std::unordered_set<active*> const* sensParams)
{
	if (pId.name != nameHash)
		return false;

	switch (mode)
	{
	case MultiplexMode::Component: {
		if ((pId.component == CompIndep) || (pId.particleType != ParTypeIndep) || (pId.boundState != BoundStateIndep) ||
			(pId.reaction != ReactionIndep) || (pId.section != SectionIndep))
			return false;

		if (sensParams && !contains(*sensParams, &data[pId.component]))
			return false;

		for (unsigned int i = 0; i < nParType; ++i)
			data[i * nComp + pId.component].setValue(value);

		return true;
	}
	case MultiplexMode::ComponentSection: {
		if ((pId.component == CompIndep) || (pId.particleType != ParTypeIndep) || (pId.boundState != BoundStateIndep) ||
			(pId.reaction != ReactionIndep) || (pId.section == SectionIndep))
			return false;

		if (sensParams && !contains(*sensParams, &data[pId.section * nComp * nParType + pId.component]))
			return false;

		for (unsigned int i = 0; i < nParType; ++i)
			data[i * nComp + pId.section * nComp * nParType + pId.component].setValue(value);

		return true;
	}
	case MultiplexMode::ComponentType:
	case MultiplexMode::ComponentSectionType:
	case MultiplexMode::RadialSection:
	case MultiplexMode::Independent:
	case MultiplexMode::ComponentRadial:
	case MultiplexMode::ComponentRadialSection:
	case MultiplexMode::Axial:
	case MultiplexMode::Section:
	case MultiplexMode::Type:
	case MultiplexMode::Radial:
	case MultiplexMode::AxialRadial:
		cadet_assert(false);
		break;
	}

	return false;
}

bool multiplexCompTypeSecParameterAD(const ParameterId& pId, StringHash nameHash, MultiplexMode mode,
									 std::vector<active>& data, unsigned int nParType, unsigned int nComp,
									 unsigned int adDirection, double adValue, std::unordered_set<active*>& sensParams)
{
	if (pId.name != nameHash)
		return false;

	switch (mode)
	{
	case MultiplexMode::Component: {
		if ((pId.component == CompIndep) || (pId.particleType != ParTypeIndep) || (pId.boundState != BoundStateIndep) ||
			(pId.reaction != ReactionIndep) || (pId.section != SectionIndep))
			return false;

		sensParams.insert(&data[pId.component]);

		for (unsigned int i = 0; i < nParType; ++i)
			data[i * nComp + pId.component].setADValue(adDirection, adValue);

		return true;
	}
	case MultiplexMode::ComponentSection: {
		if ((pId.component == CompIndep) || (pId.particleType != ParTypeIndep) || (pId.boundState != BoundStateIndep) ||
			(pId.reaction != ReactionIndep) || (pId.section == SectionIndep))
			return false;

		sensParams.insert(&data[pId.section * nComp * nParType + pId.component]);

		for (unsigned int i = 0; i < nParType; ++i)
			data[i * nComp + pId.section * nComp * nParType + pId.component].setADValue(adDirection, adValue);

		return true;
	}
	case MultiplexMode::ComponentType:
	case MultiplexMode::ComponentSectionType:
	case MultiplexMode::RadialSection:
	case MultiplexMode::Independent:
	case MultiplexMode::ComponentRadial:
	case MultiplexMode::ComponentRadialSection:
	case MultiplexMode::Axial:
	case MultiplexMode::Section:
	case MultiplexMode::Type:
	case MultiplexMode::Radial:
	case MultiplexMode::AxialRadial:
		cadet_assert(false);
		break;
	}

	return false;
}

MultiplexMode readAndRegisterMultiplexBndCompTypeSecParam(IParameterProvider& paramProvider,
														  std::unordered_map<ParameterId, active*>& parameters,
														  std::vector<active>& values, const std::string& name,
														  unsigned int nParType, unsigned int nComp,
														  unsigned int const* strideBound, unsigned int const* nBound,
														  UnitOpIdx uoi)
{
	const unsigned int nTotalBound = strideBound[nParType];
	if (nTotalBound == 0)
		return MultiplexMode::Component;

	MultiplexMode mode = MultiplexMode::Independent;
	readScalarParameterOrArray(values, paramProvider, name, 1);
	if (paramProvider.exists(name + "_MULTIPLEX"))
	{
		const int modeConfig = paramProvider.getInt(name + "_MULTIPLEX");
		if (modeConfig == 0)
		{
			mode = MultiplexMode::Component;
			if (values.size() != strideBound[0])
				throw InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name +
												"_MULTIPLEX (should be " + std::to_string(strideBound[0]) + ")");
		}
		else if (modeConfig == 1)
		{
			mode = MultiplexMode::ComponentSection;
			if ((values.size() % strideBound[0]) != 0)
				throw InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name +
												"_MULTIPLEX (should be positive multiple of " +
												std::to_string(strideBound[0]) + ")");
		}
		else if (modeConfig == 2)
		{
			mode = MultiplexMode::ComponentType;
			if (values.size() != nTotalBound)
				throw InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name +
												"_MULTIPLEX (should be " + std::to_string(nTotalBound) + ")");
		}
		else if (modeConfig == 3)
		{
			mode = MultiplexMode::ComponentSectionType;
			if ((values.size() % nTotalBound) != 0)
				throw InvalidParameterException("Number of elements in field " + name + " inconsistent with " + name +
												"_MULTIPLEX (should be positive multiple of " +
												std::to_string(nTotalBound) + ")");
		}
	}
	else
	{
		if (values.size() == strideBound[0])
			mode = MultiplexMode::Component;
		else if (values.size() == nTotalBound)
			mode = MultiplexMode::ComponentType;
		else if ((values.size() % nTotalBound) == 0)
			mode = MultiplexMode::ComponentSectionType;
		else if ((values.size() % strideBound[0]) == 0)
			mode = MultiplexMode::ComponentSection;
		else
			throw InvalidParameterException("Could not infer multiplex mode of field " + name + ", set " + name +
											"_MULTIPLEX or change number of elements");
	}

	const StringHash nameHash = hashStringRuntime(name);
	switch (mode)
	{
	case MultiplexMode::Component: {
		std::vector<active> p(nTotalBound);
		for (unsigned int s = 0; s < nParType; ++s)
			std::copy(values.begin(), values.end(), p.begin() + s * strideBound[0]);

		values = std::move(p);

		unsigned int idx = 0;
		for (unsigned int c = 0; c < nComp; ++c)
		{
			for (unsigned int s = 0; s < nBound[c]; ++s, ++idx)
				parameters[makeParamId(nameHash, uoi, c, ParTypeIndep, s, ReactionIndep, SectionIndep)] = &values[idx];
		}
	}
	break;
	case MultiplexMode::ComponentSection: {
		std::vector<active> p(values.size() / strideBound[0] * nTotalBound);
		for (std::size_t s = 0; s < values.size() / strideBound[0]; ++s)
		{
			for (unsigned int pt = 0; pt < nParType; ++pt)
			{
				std::copy(values.begin() + s * strideBound[0], values.begin() + (s + 1) * strideBound[0],
						  p.begin() + s * nTotalBound + pt * strideBound[0]);
			}
		}

		values = std::move(p);

		for (std::size_t sec = 0; sec < values.size() / strideBound[0]; ++sec)
		{
			unsigned int idx = nTotalBound * sec;
			for (unsigned int c = 0; c < nComp; ++c)
			{
				for (unsigned int s = 0; s < nBound[c]; ++s, ++idx)
					parameters[makeParamId(nameHash, uoi, c, ParTypeIndep, s, ReactionIndep, sec)] = &values[idx];
			}
		}
	}
	break;
	case MultiplexMode::ComponentType:
	case MultiplexMode::ComponentSectionType: {
		unsigned int idx = 0;
		const bool multiSec = (values.size() != nTotalBound);
		for (std::size_t sec = 0; sec < values.size() / nTotalBound; ++sec)
		{
			for (unsigned int type = 0; type < nParType; ++type)
			{
				for (unsigned int c = 0; c < nComp; ++c)
				{
					for (unsigned int s = 0; s < nBound[type * nComp + c]; ++s, ++idx)
						parameters[makeParamId(nameHash, uoi, c, type, s, ReactionIndep,
											   multiSec ? sec : SectionIndep)] = &values[idx];
				}
			}
		}
	}
	break;
	case MultiplexMode::RadialSection:
	case MultiplexMode::Independent:
	case MultiplexMode::ComponentRadial:
	case MultiplexMode::ComponentRadialSection:
	case MultiplexMode::Axial:
	case MultiplexMode::Section:
	case MultiplexMode::Type:
	case MultiplexMode::Radial:
	case MultiplexMode::AxialRadial:
		cadet_assert(false);
		break;
	}

	return mode;
}

bool multiplexBndCompTypeSecParameterValue(const ParameterId& pId, StringHash nameHash, MultiplexMode mode,
										   std::vector<active>& data, unsigned int nParType, unsigned int nComp,
										   unsigned int const* strideBound, unsigned int const* nBound,
										   unsigned int const* boundOffset, double value,
										   std::unordered_set<active*> const* sensParams)
{
	if (pId.name != nameHash)
		return false;

	if (strideBound[nParType] == 0)
		return true;

	switch (mode)
	{
	case MultiplexMode::Component: {
		if ((pId.component == CompIndep) || (pId.particleType != ParTypeIndep) || (pId.boundState == BoundStateIndep) ||
			(pId.reaction != ReactionIndep) || (pId.section != SectionIndep))
			return false;

		if (sensParams && !contains(*sensParams, &data[boundOffset[pId.component] + pId.boundState]))
			return false;

		for (unsigned int i = 0; i < nParType; ++i)
			data[boundOffset[pId.component] + pId.boundState + i * strideBound[0]].setValue(value);

		return true;
	}
	case MultiplexMode::ComponentSection: {
		if ((pId.component == CompIndep) || (pId.particleType != ParTypeIndep) || (pId.boundState == BoundStateIndep) ||
			(pId.reaction != ReactionIndep) || (pId.section == SectionIndep))
			return false;

		if (sensParams &&
			!contains(*sensParams,
					  &data[pId.section * strideBound[nParType] + boundOffset[pId.component] + pId.boundState]))
			return false;

		for (unsigned int i = 0; i < nParType; ++i)
			data[pId.section * strideBound[nParType] + boundOffset[pId.component] + pId.boundState + i * strideBound[0]]
				.setValue(value);

		return true;
	}
	case MultiplexMode::ComponentType:
	case MultiplexMode::ComponentSectionType:
	case MultiplexMode::RadialSection:
	case MultiplexMode::Independent:
	case MultiplexMode::ComponentRadial:
	case MultiplexMode::ComponentRadialSection:
	case MultiplexMode::Axial:
	case MultiplexMode::Section:
	case MultiplexMode::Type:
	case MultiplexMode::Radial:
	case MultiplexMode::AxialRadial:
		cadet_assert(false);
		break;
	}

	return false;
}

bool multiplexBndCompTypeSecParameterAD(const ParameterId& pId, StringHash nameHash, MultiplexMode mode,
										std::vector<active>& data, unsigned int nParType, unsigned int nComp,
										unsigned int const* strideBound, unsigned int const* nBound,
										unsigned int const* boundOffset, unsigned int adDirection, double adValue,
										std::unordered_set<active*>& sensParams)
{
	if (pId.name != nameHash)
		return false;

	if (strideBound[nParType] == 0)
		return true;

	switch (mode)
	{
	case MultiplexMode::Component: {
		if ((pId.component == CompIndep) || (pId.particleType != ParTypeIndep) || (pId.boundState == BoundStateIndep) ||
			(pId.reaction != ReactionIndep) || (pId.section != SectionIndep))
			return false;

		sensParams.insert(&data[boundOffset[pId.component] + pId.boundState]);

		for (unsigned int i = 0; i < nParType; ++i)
			data[boundOffset[pId.component] + pId.boundState + i * strideBound[0]].setADValue(adDirection, adValue);

		return true;
	}
	case MultiplexMode::ComponentSection: {
		if ((pId.component == CompIndep) || (pId.particleType != ParTypeIndep) || (pId.boundState == BoundStateIndep) ||
			(pId.reaction != ReactionIndep) || (pId.section == SectionIndep))
			return false;

		sensParams.insert(&data[pId.section * strideBound[nParType] + boundOffset[pId.component] + pId.boundState]);

		for (unsigned int i = 0; i < nParType; ++i)
			data[pId.section * strideBound[nParType] + boundOffset[pId.component] + pId.boundState + i * strideBound[0]]
				.setADValue(adDirection, adValue);

		return true;
	}
	case MultiplexMode::ComponentType:
	case MultiplexMode::ComponentSectionType:
	case MultiplexMode::RadialSection:
	case MultiplexMode::Independent:
	case MultiplexMode::ComponentRadial:
	case MultiplexMode::ComponentRadialSection:
	case MultiplexMode::Axial:
	case MultiplexMode::Section:
	case MultiplexMode::Type:
	case MultiplexMode::Radial:
	case MultiplexMode::AxialRadial:
		cadet_assert(false);
		break;
	}

	return false;
}

} // namespace model

} // namespace cadet
