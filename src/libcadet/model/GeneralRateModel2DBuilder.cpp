#include "model/GeneralRateModel2D.hpp"
//#include "model/GeneralRateModel2DDG.hpp"
#include "cadet/ParameterProvider.hpp"
#include "LoggingUtils.hpp"
#include "Logging.hpp"


namespace cadet
{
	namespace model
	{

		IUnitOperation* selectDiscretizationGRM2D(UnitOpIdx uoId, IParameterProvider& paramProvider)
		{
			IUnitOperation* model = nullptr;

			paramProvider.pushScope("discretization");

			if (paramProvider.exists("SPATIAL_METHOD")) {

				const std::string discName = paramProvider.getString("SPATIAL_METHOD");

				/*if (discName == "DG")
					model = new GeneralRateModel2DDG(uoId);
				else*/ if (discName == "FV")
					model = new GeneralRateModel2D(uoId);
				else
				{
					LOG(Error) << "Unknown discretization type " << discName << " for unit " << uoId;
				}
			}
			else {
				model = new GeneralRateModel2D(uoId);
			}

			paramProvider.popScope();

			return model;
		}

		void registerGeneralRateModel2D(std::unordered_map<std::string, std::function<IUnitOperation* (UnitOpIdx, IParameterProvider&)>>& models)
		{

			models[GeneralRateModel2D::identifier()] = selectDiscretizationGRM2D;
			models["GRM2D"] = selectDiscretizationGRM2D;

			//models[GeneralRateModel2DDG::identifier()] = selectDiscretizationGRM2D;
			//models["GRM2D_DG"] = selectDiscretizationGRM2D;

		}

	}  // namespace model

}  // namespace cadet