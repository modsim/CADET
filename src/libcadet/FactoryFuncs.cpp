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

#include "cadet/FactoryFuncs.hpp"
#include "SimulatorImpl.hpp"
#include "ModelBuilderImpl.hpp"

namespace cadet
{

IModelBuilder* createModelBuilder()
{
	return new ModelBuilder();
}

void destroyModelBuilder(IModelBuilder* const builder) CADET_NOEXCEPT
{
	delete builder;
}

ISimulator* createSimulator()
{
	return new Simulator();
}

void destroySimulator(ISimulator* const sim) CADET_NOEXCEPT
{
	delete sim;
}

} // namespace cadet

extern "C"
{
	cadet::IModelBuilder* cadetCreateModelBuilder()
	{
		return cadet::createModelBuilder();
	}

	void cadetDestroyModelBuilder(cadet::IModelBuilder* const builder)
	{
		cadet::destroyModelBuilder(builder);
	}

	cadet::ISimulator* cadetCreateSimulator()
	{
		return cadet::createSimulator();
	}

	void cadetDestroySimulator(cadet::ISimulator* const sim)
	{
		cadet::destroySimulator(sim);
	}
}
