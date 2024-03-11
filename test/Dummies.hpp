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

#include "cadet/Model.hpp"
#include "cadet/ParameterId.hpp"

#include "ConfigurationHelper.hpp"

namespace
{
	class DummyConfigHelper : public cadet::IConfigHelper
	{
	public:

		DummyConfigHelper() { }

		virtual cadet::IInletProfile* createInletProfile(const std::string& type) const { return nullptr; }
		virtual cadet::model::IBindingModel* createBindingModel(const std::string& name) const { return nullptr; }
		virtual bool isValidBindingModel(const std::string& name) const { return false; }
		virtual cadet::IExternalFunction* createExternalFunction(const std::string& type) const { return nullptr; }
		virtual cadet::model::IDynamicReactionModel* createDynamicReactionModel(const std::string& name) const { return nullptr; }
		virtual bool isValidDynamicReactionModel(const std::string& name) const { return false; }
		virtual cadet::model::IParameterStateDependence* createParameterStateDependence(const std::string& name) const { return nullptr; }
		virtual bool isValidParameterStateDependence(const std::string& name) const { return false; }
		virtual cadet::model::IParameterParameterDependence* createParameterParameterDependence(const std::string& name) const { return nullptr; }
		virtual bool isValidParameterParameterDependence(const std::string& name) const { return false; }
	};

	class DummyModel : public cadet::IModel
	{
	public:
		DummyModel() { }
		virtual ~DummyModel() CADET_NOEXCEPT { }

		virtual cadet::UnitOpIdx unitOperationId() const CADET_NOEXCEPT { return 0; };
		virtual const char* unitOperationName() const CADET_NOEXCEPT { return "DUMMY"; }

		virtual bool setParameter(const cadet::ParameterId& pId, int value) { return false; }
		virtual bool setParameter(const cadet::ParameterId& pId, double value) { return false; }
		virtual bool setParameter(const cadet::ParameterId& pId, bool value) { return false; }

		virtual bool hasParameter(const cadet::ParameterId& pId) const { return false; }

		virtual std::unordered_map<cadet::ParameterId, double> getAllParameterValues() const { return std::unordered_map<cadet::ParameterId, double>(); }

		virtual double getParameterDouble(const cadet::ParameterId& pId) const { return 0.0; }

		virtual void useAnalyticJacobian(const bool analyticJac) { }

#ifdef CADET_BENCHMARK_MODE
		virtual std::vector<double> benchmarkTimings() const { return std::vector<double>(0); }
		virtual char const* const* benchmarkDescriptions() const { return nullptr; }
#endif
	};

}
