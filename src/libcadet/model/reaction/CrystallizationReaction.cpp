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

#include "model/reaction/ReactionModelBase.hpp"
#include "model/ModelUtils.hpp"
#include "cadet/Exceptions.hpp"
#include "model/Parameters.hpp"
#include "LocalVector.hpp"
#include "SimulationTypes.hpp"
#include "linalg/ActiveDenseMatrix.hpp"
#include "Memory.hpp"

#include <functional>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <iterator>
#include <vector>

namespace cadet
{

	namespace model
	{
		/**
         * @brief Flux reconstruction scheme coefficients. See @cite Zhang2024_PBM_partI
         */
		namespace detail
		{
			struct Weno3Coefficients
			{
				std::vector<active> q_1_right_coeff;
				std::vector<active> q_0_right_coeff;
				std::vector<active> C_right_coeff;
				std::vector<active> IS_0_coeff;
				std::vector<active> IS_1_coeff;

				Weno3Coefficients(const std::vector<active>& binSizes)
					: q_1_right_coeff(binSizes.size() - 1), q_0_right_coeff(binSizes.size() - 1), C_right_coeff(binSizes.size() - 1), IS_0_coeff(binSizes.size() - 1), IS_1_coeff(binSizes.size() - 1)
				{
					// calculate the coefficients and store them, the first element is empty
					for (int i = 1; i + 1 < binSizes.size(); ++i)
					{
						const active delta_sum = binSizes[i] + binSizes[i - 1] + binSizes[i + 1];
						const active delta_left_sum = binSizes[i] + binSizes[i - 1];
						const active delta_right_sum = binSizes[i] + binSizes[i + 1];
						q_0_right_coeff[i] = binSizes[i + 1] / delta_right_sum;
						q_1_right_coeff[i] = 1.0 + binSizes[i] / delta_left_sum;
						C_right_coeff[i] = delta_left_sum / delta_sum;
						IS_0_coeff[i] = (2.0 * binSizes[i] / delta_right_sum) * (2.0 * binSizes[i] / delta_right_sum);
						IS_1_coeff[i] = (2.0 * binSizes[i] / delta_left_sum) * (2.0 * binSizes[i] / delta_left_sum);
					}
				}
			};

			struct HRCoefficients
			{
				std::vector<active> A_coeff;
				std::vector<active> R_coeff;

				HRCoefficients(const std::vector<active>& binSizes)
					: A_coeff(binSizes.size() - 1), R_coeff(binSizes.size() - 1)
				{
					// calculate the coefficients
					for (int i = 1; i + 1 < binSizes.size(); ++i)
					{
						const active delta_xi_xip1 = binSizes[i] + binSizes[i + 1];
						A_coeff[i] = delta_xi_xip1 / (binSizes[i] + binSizes[i - 1]);
						R_coeff[i] = delta_xi_xip1 / binSizes[i];
					}
				}
			};

			struct Weno5Coefficients
			{
				std::vector<active> q_2_coeff_1;
				std::vector<active> q_2_coeff_2;
				std::vector<active> q_1_coeff_1;
				std::vector<active> q_1_coeff_2;
				std::vector<active> q_0_coeff_1;
				std::vector<active> q_0_coeff_2;
				std::vector<active> C_0;
				std::vector<active> C_1;
				std::vector<active> C_2;
				std::vector<active> IS_0_coeff_1;
				std::vector<active> IS_0_coeff_2;
				std::vector<active> IS_0_coeff_3;
				std::vector<active> IS_1_coeff_1;
				std::vector<active> IS_1_coeff_2;
				std::vector<active> IS_1_coeff_3;
				std::vector<active> IS_2_coeff_1;
				std::vector<active> IS_2_coeff_2;
				std::vector<active> IS_2_coeff_3;

				// weno23 parameters, left
				active IS_0_coeff_weno3;
				active IS_1_coeff_weno3;
				active C_coeff1_weno3;
				active C_coeff2_weno3;
				active q0_coeff1_weno3;
				active q0_coeff2_weno3;
				active q1_coeff1_weno3;
				active q1_coeff2_weno3;
				// weno23 parameters, right
				active IS_0_coeff_weno3_r;
				active IS_1_coeff_weno3_r;
				active C_coeff1_weno3_r;
				active C_coeff2_weno3_r;
				active q0_coeff1_weno3_r;
				active q0_coeff2_weno3_r;
				active q1_coeff1_weno3_r;
				active q1_coeff2_weno3_r;

				Weno5Coefficients(const std::vector<active>& binSizes)
					: q_2_coeff_1(binSizes.size() - 2), q_2_coeff_2(binSizes.size() - 2), q_1_coeff_1(binSizes.size() - 2),
					q_1_coeff_2(binSizes.size() - 2), q_0_coeff_1(binSizes.size() - 2), q_0_coeff_2(binSizes.size() - 2),
					C_0(binSizes.size() - 2), C_1(binSizes.size() - 2), C_2(binSizes.size() - 2),
					IS_0_coeff_1(binSizes.size() - 2), IS_0_coeff_2(binSizes.size() - 2), IS_0_coeff_3(binSizes.size() - 2),
					IS_1_coeff_1(binSizes.size() - 2), IS_1_coeff_2(binSizes.size() - 2), IS_1_coeff_3(binSizes.size() - 2),
					IS_2_coeff_1(binSizes.size() - 2), IS_2_coeff_2(binSizes.size() - 2), IS_2_coeff_3(binSizes.size() - 2)
				{
					// calculate the coefficients and store them, the first and second elements are empty
					for (int i = 2; i + 2 < binSizes.size(); ++i)
					{
						const active delta_sum = binSizes[i - 2] + binSizes[i - 1] + binSizes[i] + binSizes[i + 1] + binSizes[i + 2];
						const active delta_sum_I0 = binSizes[i - 2] + binSizes[i - 1] + binSizes[i];
						const active delta_sum_I1 = binSizes[i - 1] + binSizes[i] + binSizes[i + 1];
						const active delta_sum_I2 = binSizes[i] + binSizes[i + 1] + binSizes[i + 2];
						q_0_coeff_1[i] = binSizes[i + 1] * (delta_sum_I2 - binSizes[i]) / (delta_sum_I2 - binSizes[i + 2]) / delta_sum_I2;
						q_0_coeff_2[i] = binSizes[i + 1] * binSizes[i] / delta_sum_I2 / (delta_sum_I2 - binSizes[i]);
						q_1_coeff_1[i] = binSizes[i] * (delta_sum_I1 - binSizes[i + 1]) / delta_sum_I1 / (delta_sum_I1 - binSizes[i - 1]);
						q_1_coeff_2[i] = binSizes[i] * binSizes[i + 1] / delta_sum_I1 / (delta_sum_I1 - binSizes[i + 1]);
						q_2_coeff_1[i] = binSizes[i] * (delta_sum_I0 - binSizes[i - 2]) / delta_sum_I0 / (delta_sum_I0 - binSizes[i]);
						q_2_coeff_2[i] = 1.0 + binSizes[i] / (binSizes[i - 1] + binSizes[i]) + binSizes[i] / delta_sum_I0;
						C_0[i] = delta_sum_I0 * (delta_sum_I0 - binSizes[i - 2]) / delta_sum / (delta_sum - binSizes[i - 2]);
						C_1[i] = delta_sum_I0 / delta_sum * (delta_sum_I2 - binSizes[i]) / (binSizes[i - 1] + delta_sum_I2) * (1.0 + (delta_sum - binSizes[i - 2]) / (delta_sum - binSizes[i + 2]));
						C_2[i] = binSizes[i + 1] * (binSizes[i + 1] + binSizes[i + 2]) / delta_sum / (delta_sum - binSizes[i + 2]);
						const active IS_0_pre = 4.0 * (binSizes[i] / delta_sum_I2) * (binSizes[i] / delta_sum_I2);
						IS_0_coeff_1[i] = IS_0_pre * (10.0 * binSizes[i] * binSizes[i] + binSizes[i + 1] * (binSizes[i] + binSizes[i + 1])) / (binSizes[i + 1] + binSizes[i + 2]) / (binSizes[i + 1] + binSizes[i + 2]);
						IS_0_coeff_2[i] = IS_0_pre * (20.0 * binSizes[i] * binSizes[i] + 2.0 * binSizes[i + 1] * (binSizes[i] + binSizes[i + 1]) + (2.0 * binSizes[i + 1] + binSizes[i]) * delta_sum_I2) / (binSizes[i + 1] + binSizes[i + 2]) / (binSizes[i + 1] + binSizes[i]);
						IS_0_coeff_3[i] = IS_0_pre * (10.0 * binSizes[i] * binSizes[i] + (2.0 * delta_sum_I2 - binSizes[i + 2]) * (delta_sum_I2 + binSizes[i + 1])) / (binSizes[i] + binSizes[i + 1]) / (binSizes[i] + binSizes[i + 1]);
						const active IS_1_pre = 4.0 * (binSizes[i] / delta_sum_I1) * (binSizes[i] / delta_sum_I1);
						IS_1_coeff_1[i] = IS_1_pre * (10.0 * binSizes[i] * binSizes[i] + binSizes[i + 1] * (binSizes[i] + binSizes[i + 1])) / (binSizes[i - 1] + binSizes[i]) / (binSizes[i - 1] + binSizes[i]);
						IS_1_coeff_2[i] = IS_1_pre * (20.0 * binSizes[i] * binSizes[i] - binSizes[i + 1] * binSizes[i - 1] - (binSizes[i] + binSizes[i + 1]) * (binSizes[i] + binSizes[i - 1])) / (binSizes[i] + binSizes[i + 1]) / (binSizes[i] + binSizes[i - 1]);
						IS_1_coeff_3[i] = IS_1_pre * (10.0 * binSizes[i] * binSizes[i] + binSizes[i - 1] * (binSizes[i - 1] + binSizes[i])) / (binSizes[i] + binSizes[i + 1]) / (binSizes[i] + binSizes[i + 1]);
						const active IS_2_pre = 4.0 * (binSizes[i] / delta_sum_I0) * (binSizes[i] / delta_sum_I0);
						IS_2_coeff_1[i] = IS_2_pre * (10.0 * binSizes[i] * binSizes[i] + binSizes[i - 1] * (binSizes[i - 1] + binSizes[i])) / (binSizes[i - 2] + binSizes[i - 1]) / (binSizes[i - 2] + binSizes[i - 1]);
						IS_2_coeff_2[i] = IS_2_pre * (20.0 * binSizes[i] * binSizes[i] + 2.0 * binSizes[i - 1] * (binSizes[i - 1] + binSizes[i]) + delta_sum_I0 * (2.0 * binSizes[i - 1] + binSizes[i])) / (binSizes[i - 1] + binSizes[i]) / (binSizes[i - 2] + binSizes[i - 1]);
						IS_2_coeff_3[i] = IS_2_pre * (10.0 * binSizes[i] * binSizes[i] + (2.0 * delta_sum_I0 - binSizes[i - 2]) * (delta_sum_I0 + binSizes[i - 1])) / (binSizes[i - 1] + binSizes[i]) / (binSizes[i - 1] + binSizes[i]);
					}

					const int nBins = binSizes.size();
					IS_0_coeff_weno3 = 4.0 * binSizes[1] * binSizes[1] / (binSizes[1] + binSizes[2]) / (binSizes[1] + binSizes[2]);
					IS_1_coeff_weno3 = 4.0 * binSizes[1] * binSizes[1] / (binSizes[0] + binSizes[1]) / (binSizes[0] + binSizes[1]);
					C_coeff1_weno3 = (binSizes[0] + binSizes[1]) / (binSizes[0] + binSizes[1] + binSizes[2]);
					C_coeff2_weno3 = 1.0 - (C_coeff1_weno3);
					q0_coeff1_weno3 = binSizes[2] / (binSizes[1] + binSizes[2]);
					q0_coeff2_weno3 = 1.0 - (q0_coeff1_weno3);
					q1_coeff1_weno3 = 1.0 + binSizes[1] / (binSizes[0] + binSizes[1]);
					q1_coeff2_weno3 = 1.0 - (q1_coeff1_weno3);
					IS_0_coeff_weno3_r = 4.0 * binSizes[nBins - 2] * binSizes[nBins - 2] / (binSizes[nBins - 2] + binSizes[nBins - 1]) / (binSizes[nBins - 2] + binSizes[nBins - 1]);
					IS_1_coeff_weno3_r = 4.0 * binSizes[nBins - 2] * binSizes[nBins - 2] / (binSizes[nBins - 2] + binSizes[nBins - 3]) / (binSizes[nBins - 2] + binSizes[nBins - 3]);
					C_coeff1_weno3_r = (binSizes[nBins - 2] + binSizes[nBins - 1]) / (binSizes[nBins - 3] + binSizes[nBins - 2] + binSizes[nBins - 1]);
					C_coeff2_weno3_r = 1.0 - (C_coeff1_weno3_r);
					q0_coeff1_weno3_r = binSizes[nBins - 1] / (binSizes[nBins - 2] + binSizes[nBins - 1]);
					q0_coeff2_weno3_r = 1.0 - (q0_coeff1_weno3_r);
					q1_coeff1_weno3_r = 1.0 + binSizes[nBins - 2] / (binSizes[nBins - 2] + binSizes[nBins - 3]);
					q1_coeff2_weno3_r = 1.0 - (q1_coeff1_weno3_r);
				}
			};
		} // namespace detail

		/**
		 * @brief Defines the crystallization reaction model
		 */
		class CrystallizationReaction : public IDynamicReactionModel
		{
		public:

			CrystallizationReaction() : _nComp(0), _nBins(0), _bins(0), _binCenters(0), _binSizes(0), _HR(nullptr), _weno3(nullptr), _weno5(nullptr) { }
			virtual ~CrystallizationReaction() CADET_NOEXCEPT
			{
				clearSchemeCoefficients();
			}

			static const char* identifier() { return "CRYSTALLIZATION"; }
			virtual const char* name() const CADET_NOEXCEPT { return identifier(); }

			virtual bool requiresConfiguration() const CADET_NOEXCEPT { return true; }
			virtual bool usesParamProviderInDiscretizationConfig() const CADET_NOEXCEPT { return false; }

			virtual bool configure(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx)
			{
				readScalarParameterOrArray(_bins, paramProvider, "CRY_BINS", 1);

				if (_bins.size() != _nBins + 1)
					throw InvalidParameterException("Expected CRY_BINS to have " + std::to_string(_nBins + 1) + " elements (got " + std::to_string(_bins.size()) + ")");

				registerParam1DArray(_parameters, _bins, [=](bool multi, unsigned int idx) { return makeParamId(hashString("CRY_BINS"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, idx, SectionIndep); });

				_binCenters.resize(_nBins);
				_binSizes.resize(_nBins);
				_binCenterDists.resize(_nBins - 1);
				updateBinCoords();

				_nucleiMassDensity = paramProvider.getDouble("CRY_NUCLEI_MASS_DENSITY");
				_parameters[makeParamId(hashString("CRY_NUCLEI_MASS_DENSITY"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_nucleiMassDensity;

				_volShapeFactor = paramProvider.getDouble("CRY_VOL_SHAPE_FACTOR");
				_parameters[makeParamId(hashString("CRY_VOL_SHAPE_FACTOR"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_volShapeFactor;

				_primaryNucleationRate = paramProvider.getDouble("CRY_PRIMARY_NUCLEATION_RATE");
				_parameters[makeParamId(hashString("CRY_PRIMARY_NUCLEATION_RATE"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_primaryNucleationRate;

				_secondaryNucleationRate = paramProvider.getDouble("CRY_SECONDARY_NUCLEATION_RATE");
				_parameters[makeParamId(hashString("CRY_SECONDARY_NUCLEATION_RATE"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_secondaryNucleationRate;

				_growthRateConstant = paramProvider.getDouble("CRY_GROWTH_RATE_CONSTANT");
				_parameters[makeParamId(hashString("CRY_GROWTH_RATE_CONSTANT"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_growthRateConstant;

				_growthConstant = paramProvider.getDouble("CRY_GROWTH_CONSTANT");
				_parameters[makeParamId(hashString("CRY_GROWTH_CONSTANT"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_growthConstant;

				_growthDispersionRate = paramProvider.getDouble("CRY_GROWTH_DISPERSION_RATE");
				_parameters[makeParamId(hashString("CRY_GROWTH_DISPERSION_RATE"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_growthDispersionRate;

				_growthSchemeOrder = paramProvider.getInt("CRY_GROWTH_SCHEME_ORDER");

				if (!(_growthSchemeOrder == 1 || _growthSchemeOrder == 2 || _growthSchemeOrder == 3 || _growthSchemeOrder == 4))
					throw InvalidParameterException("CRY_GROWTH_SCHEME_ORDER needs to be an int between [1, 4]");

				_a = paramProvider.getDouble("CRY_A");
				_parameters[makeParamId(hashString("CRY_A"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_a;

				_b = paramProvider.getDouble("CRY_B");
				_parameters[makeParamId(hashString("CRY_B"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_b;

				_g = paramProvider.getDouble("CRY_G");
				_parameters[makeParamId(hashString("CRY_G"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_g;

				_p = paramProvider.getDouble("CRY_P");
				_parameters[makeParamId(hashString("CRY_P"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_p;

				_k = paramProvider.getDouble("CRY_K");
				_parameters[makeParamId(hashString("CRY_K"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_k;

				_u = paramProvider.getDouble("CRY_U");
				_parameters[makeParamId(hashString("CRY_U"), unitOpIdx, CompIndep, ParTypeIndep, BoundStateIndep, ReactionIndep, SectionIndep)] = &_u;

				clearSchemeCoefficients();

				if (_growthSchemeOrder == 2)
					_HR = new detail::HRCoefficients(_binSizes);
				else if (_growthSchemeOrder == 3)
					_weno3 = new detail::Weno3Coefficients(_binSizes);
				else if (_growthSchemeOrder == 4)
					_weno5 = new detail::Weno5Coefficients(_binSizes);

				return true;
			}

			virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp, unsigned int const* nBound, unsigned int const* boundOffset)
			{
				_nComp = nComp;

				// Comp 0 is substrate, last comp is equilibrium
				_nBins = _nComp - 2;

				if (_nBins < 1)
					throw InvalidParameterException("Expected at least 3 components (got " + std::to_string(_nComp) + ")");

				return true;
			}

			std::unordered_map<ParameterId, double> getAllParameterValues() const
			{
				std::unordered_map<ParameterId, double> data;
				std::transform(_parameters.begin(), _parameters.end(), std::inserter(data, data.end()),
					[](const std::pair<const ParameterId, active*>& p) { return std::make_pair(p.first, static_cast<double>(*p.second)); });
				return data;
			}

			bool hasParameter(const ParameterId& pId) const
			{
				return _parameters.find(pId) != _parameters.end();
			}

			bool setParameter(const ParameterId& pId, int value) { return false; }
			bool setParameter(const ParameterId& pId, bool value) { return false; }

			bool setParameter(const ParameterId& pId, double value)
			{
				auto paramHandle = _parameters.find(pId);
				if (paramHandle != _parameters.end())
				{
					paramHandle->second->setValue(value);

					// TODO: This does not handle parameter sensitivities wrt. to bin size
					if (pId.name == hashString("CRY_BINS"))
						updateBinCoords();

					return true;
				}

				return false;
			}

			active* getParameter(const ParameterId& pId)
			{
				auto paramHandle = _parameters.find(pId);
				if (paramHandle != _parameters.end())
				{
					return paramHandle->second;
				}

				return nullptr;
			}

			virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size) { }
			virtual bool dependsOnTime() const CADET_NOEXCEPT { return false; }
			virtual bool requiresWorkspace() const CADET_NOEXCEPT { return false; }
			virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates, unsigned int const* nBoundStates) const CADET_NOEXCEPT
			{
				return 0;
			}

			virtual unsigned int numReactionsLiquid() const CADET_NOEXCEPT { return 1; }
			virtual unsigned int numReactionsCombined() const CADET_NOEXCEPT { return 1; }

			CADET_DYNAMICREACTIONMODEL_BOILERPLATE

		protected:

			std::unordered_map<ParameterId, active*> _parameters; //!< Map used to translate ParameterIds to actual variables
			int _nComp; //!< Number of components
			int _nBins; //!< Number of crystal size bins

			std::vector<active> _bins;
			std::vector<active> _binCenters;
			std::vector<active> _binSizes;
			std::vector<active> _binCenterDists;
			active _nucleiMassDensity; //!< rho
			active _volShapeFactor; //!< k_v
			active _primaryNucleationRate; //!< k_p
			active _secondaryNucleationRate; //!< k_b
			active _growthRateConstant; //!< k_g
			active _growthConstant; //!< gamma
			active _growthDispersionRate; //!< D_g
			int _growthSchemeOrder; // can be 1, 2, 3 and 4
			active _a; //!< System constant
			active _b; //!< System constant
			active _g; //!< System constant
			active _p; //!< System constant
			active _k; //!< System constant
			active _u; //!< System constant

			detail::HRCoefficients* _HR;
			detail::Weno3Coefficients* _weno3;
			detail::Weno5Coefficients* _weno5;

			void updateBinCoords() CADET_NOEXCEPT
			{
				for (int i = 0; i < _nBins; ++i)
				{
					if (cadet_likely(i + 1 < _nBins))
						_binCenterDists[i] = 0.5 * (_bins[i + 2] - _bins[i]);

					_binCenters[i] = 0.5 * (_bins[i] + _bins[i + 1]);
					_binSizes[i] = _bins[i + 1] - _bins[i];
				}
			}

			void clearSchemeCoefficients() CADET_NOEXCEPT
			{
				if (_HR)
				{
					delete _HR;
					_HR = nullptr;
				}

				if (_weno3)
				{
					delete _weno3;
					_weno3 = nullptr;
				}

				if (_weno5)
				{
					delete _weno5;
					_weno5 = nullptr;
				}
			}

			/**
            * @brief Residual implementations. See @cite Zhang2024_PBM_partI
            */
			template <typename StateType, typename ResidualType, typename ParamType, typename FactorType>
			int residualLiquidImpl(double t, unsigned int secIdx, const ColumnPosition& colPos,
				StateType const* y, ResidualType* res, const FactorType& factor, LinearBufferAllocator workSpace) const
			{
				typedef typename DoubleActivePromoter<StateType, ParamType>::type StateParam;
				// c_0 is component 0
				// c_eq is last component
				// x_c = bins[0] (i.e. left boundary of first bin is critical nuclei size x_c)

				// Pointer to crystal bins
				StateType const* const yCrystal = y + 1;
				ResidualType* const resCrystal = res + 1;

				// s = (c_0 - c_eq) / c_eq = c_0 / c_eq - 1
				// rewrite it to zero if s drops below 0
				const StateType s = (cadet_likely(y[0] / y[_nComp - 1] - 1.0 > 0)) ? y[0] / y[_nComp - 1] - 1.0 : 0.0;
				const ParamType massDensityShapeFactor = static_cast<ParamType>(_nucleiMassDensity) * static_cast<ParamType>(_volShapeFactor);

				// Numerical approximation of integrals via midpoint rule as volume averages are
				// second order accurate approximation of value at midpoint / center of mass of volume

				const StateParam k_g_times_s_g = static_cast<ParamType>(_growthRateConstant) * pow(s, static_cast<ParamType>(_g));

				StateParam M = 0.0;
				StateParam substrateConversion = 0.0;
				for (int i = 0; i < _nBins; ++i)
				{
					M += yCrystal[i] * static_cast<ParamType>(_binCenters[i]) * static_cast<ParamType>(_binCenters[i]) * static_cast<ParamType>(_binCenters[i]) * static_cast<ParamType>(_binSizes[i]);
					substrateConversion += yCrystal[i] * static_cast<ParamType>(_binCenters[i]) * static_cast<ParamType>(_binCenters[i]) * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_binCenters[i]), static_cast<ParamType>(_p))) * static_cast<ParamType>(_binSizes[i]);
				}
				M *= massDensityShapeFactor;
				substrateConversion *= 3.0 * k_g_times_s_g;

				// B_0 = primary + secondary nucleation rate
				const StateParam B_0 = static_cast<ParamType>(_primaryNucleationRate) * pow(s, static_cast<ParamType>(_u)) + static_cast<ParamType>(_secondaryNucleationRate) * pow(s, static_cast<ParamType>(_b)) * pow(M, static_cast<ParamType>(_k));
				const ParamType x_c_3 = static_cast<ParamType>(_bins[0]) * static_cast<ParamType>(_bins[0]) * static_cast<ParamType>(_bins[0]);

				// mass balance
				res[0] -= factor * massDensityShapeFactor * (B_0 * x_c_3 + substrateConversion);

				StateParam v_g = 0.0;

				// growth flux reconstruction
				const int growth_order = static_cast<int>(_growthSchemeOrder);
				switch (growth_order)
				{
					// upwind scheme
				case 1:
				{
					for (int i = 0; i < _nBins; ++i)
					{
						// Flux through left face
						if (cadet_likely((i > 0) && (i + 1 < _nBins)))
						{
							// flux through the left face
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * (v_g * yCrystal[i - 1]) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// flux through the right face
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * (v_g * yCrystal[i]) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						else if (i == 0)
						{
							// Left boundary condition
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * B_0;
							// upwind
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * yCrystal[i] - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						else
						{
							// first order approximation
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * yCrystal[i - 1] - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// no flux
						}
					}
				}
				break;
				// HR scheme
				case 2:
				{
					StateParam r_x_i = 0.0;
					StateParam phi = 0.0;
					StateParam F_i = 0.0;
					for (int i = 0; i < _nBins; ++i)
					{
						if (cadet_likely((i > 1) && (i + 1 < _nBins)))
						{
							// Flux through left face, modified van Leer flux limiter
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * F_i - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// Flux through left face, modified van Leer flux limiter, update r_x_i, R_i, and growth rate
							r_x_i = static_cast<ParamType>(_HR->A_coeff[i]) * (yCrystal[i] - yCrystal[i - 1] + 1e-10) / (yCrystal[i + 1] - yCrystal[i] + 1e-10);
							if (cadet_likely(r_x_i > 0))
							{
								phi = r_x_i / (static_cast<ParamType>(_HR->R_coeff[i]) - 1.0 + r_x_i);
							}
							else
							{
								phi = 0.0;
							}
							F_i = yCrystal[i] + phi * (yCrystal[i + 1] - yCrystal[i]);
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * F_i - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						else if (i == 1)
						{
							// upwind to F_{1/2} and the diffusion
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * yCrystal[i - 1] - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// HR scheme applied to F_{1+1/2}
							r_x_i = static_cast<ParamType>(_HR->A_coeff[i]) * (yCrystal[i] - yCrystal[i - 1] + 1e-10) / (yCrystal[i + 1] - yCrystal[i] + 1e-10);
							if (cadet_likely(r_x_i > 0))
							{
								phi = r_x_i / (static_cast<ParamType>(_HR->R_coeff[i]) - 1.0 + r_x_i);
							}
							else
							{
								phi = 0.0;
							}
							F_i = yCrystal[i] + phi * (yCrystal[i + 1] - yCrystal[i]);
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * F_i - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						else if (i + 1 == _nBins)
						{
							// HR scheme applied to the influx of the last bin 
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * F_i - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// no flux
						}
						else
						{
							// left boundary condition
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * B_0;
							// upwind to F_1
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * yCrystal[i] - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
					}
				}
				break;
				// weno 23
				case 3:
				{
					StateParam IS_0 = 0.0;
					StateParam IS_1 = 0.0;
					StateParam alpha_0 = 0.0;
					StateParam alpha_1 = 0.0;
					StateParam W_0 = 0.0;
					StateParam W_1 = 0.0;
					StateParam q_0 = 0.0;
					StateParam q_1 = 0.0;
					for (int i = 0; i < _nBins; ++i)
					{
						if (cadet_likely((i > 1) && (i + 1 < _nBins)))
						{
							// flux through left face. W_0, q_0, W_1 and q_1 are coming from the right face of bin (i-1).
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// flux through right face, update IS, alpha, W, q and growth rate
							IS_0 = static_cast<ParamType>(_weno3->IS_0_coeff[i]) * (yCrystal[i + 1] - yCrystal[i]) * (yCrystal[i + 1] - yCrystal[i]);
							IS_1 = static_cast<ParamType>(_weno3->IS_1_coeff[i]) * (yCrystal[i] - yCrystal[i - 1]) * (yCrystal[i] - yCrystal[i - 1]);
							alpha_0 = static_cast<ParamType>(_weno3->C_right_coeff[i]) / (static_cast<ParamType>(_binSizes[i]) + IS_0) / (static_cast<ParamType>(_binSizes[i]) + IS_0);
							alpha_1 = (1.0 - static_cast<ParamType>(_weno3->C_right_coeff[i])) / (static_cast<ParamType>(_binSizes[i]) + IS_1) / (static_cast<ParamType>(_binSizes[i]) + IS_1);
							W_0 = alpha_0 / (alpha_0 + alpha_1);
							W_1 = 1.0 - W_0;
							q_0 = static_cast<ParamType>(_weno3->q_0_right_coeff[i]) * yCrystal[i] + (1.0 - static_cast<ParamType>(_weno3->q_0_right_coeff[i])) * yCrystal[i + 1];
							q_1 = static_cast<ParamType>(_weno3->q_1_right_coeff[i]) * yCrystal[i] + (1.0 - static_cast<ParamType>(_weno3->q_1_right_coeff[i])) * yCrystal[i - 1];
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						// boundary condition
						else if (i + 1 == _nBins)
						{
							// weno3 applied to the influx of the last bin
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// no flux, regularity boundary condition
						}
						else if (i == 1)
						{
							// upwind applied to F_{1/2}
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * yCrystal[i - 1] - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// weno23 applied to F_{1+1/2}
							IS_0 = static_cast<ParamType>(_weno3->IS_0_coeff[i]) * (yCrystal[i + 1] - yCrystal[i]) * (yCrystal[i + 1] - yCrystal[i]);
							IS_1 = static_cast<ParamType>(_weno3->IS_1_coeff[i]) * (yCrystal[i] - yCrystal[i - 1]) * (yCrystal[i] - yCrystal[i - 1]);
							alpha_0 = static_cast<ParamType>(_weno3->C_right_coeff[i]) / (static_cast<ParamType>(_binSizes[i]) + IS_0) / (static_cast<ParamType>(_binSizes[i]) + IS_0);
							alpha_1 = (1.0 - static_cast<ParamType>(_weno3->C_right_coeff[i])) / (static_cast<ParamType>(_binSizes[i]) + IS_1) / (static_cast<ParamType>(_binSizes[i]) + IS_1);
							W_0 = alpha_0 / (alpha_0 + alpha_1);
							W_1 = 1.0 - W_0;
							q_0 = static_cast<ParamType>(_weno3->q_0_right_coeff[i]) * yCrystal[i] + (1.0 - static_cast<ParamType>(_weno3->q_0_right_coeff[i])) * yCrystal[i + 1];
							q_1 = static_cast<ParamType>(_weno3->q_1_right_coeff[i]) * yCrystal[i] + (1.0 - static_cast<ParamType>(_weno3->q_1_right_coeff[i])) * yCrystal[i - 1];
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						else
						{
							// nucleation boundary condition
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * B_0;
							// upwind
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * yCrystal[i] - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
					}
				}
				break;
				// WENO35
				case 4:
				{
					StateParam IS_0 = 0.0;
					StateParam IS_1 = 0.0;
					StateParam IS_2 = 0.0;
					StateParam alpha_0 = 0.0;
					StateParam alpha_1 = 0.0;
					StateParam alpha_2 = 0.0;
					StateParam W_0 = 0.0;
					StateParam W_1 = 0.0;
					StateParam W_2 = 0.0;
					StateParam q_0 = 0.0;
					StateParam q_1 = 0.0;
					StateParam q_2 = 0.0;
					for (int i = 0; i < _nBins; ++i)
					{
						if (cadet_likely((i > 2) && (i + 2 < _nBins)))
						{
							// flux through the left face 
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// flux through the right face
							IS_0 = static_cast<ParamType>(_weno5->IS_0_coeff_1[i]) * (yCrystal[i + 2] - yCrystal[i + 1]) * (yCrystal[i + 2] - yCrystal[i + 1]) + static_cast<ParamType>(_weno5->IS_0_coeff_2[i]) * (yCrystal[i + 2] - yCrystal[i + 1]) * (yCrystal[i] - yCrystal[i + 1]) + static_cast<ParamType>(_weno5->IS_0_coeff_3[i]) * (yCrystal[i] - yCrystal[i + 1]) * (yCrystal[i] - yCrystal[i + 1]);
							IS_1 = static_cast<ParamType>(_weno5->IS_1_coeff_1[i]) * (yCrystal[i - 1] - yCrystal[i]) * (yCrystal[i - 1] - yCrystal[i]) + static_cast<ParamType>(_weno5->IS_1_coeff_2[i]) * (yCrystal[i + 1] - yCrystal[i]) * (yCrystal[i - 1] - yCrystal[i]) + static_cast<ParamType>(_weno5->IS_1_coeff_3[i]) * (yCrystal[i + 1] - yCrystal[i]) * (yCrystal[i + 1] - yCrystal[i]);
							IS_2 = static_cast<ParamType>(_weno5->IS_2_coeff_1[i]) * (yCrystal[i - 2] - yCrystal[i - 1]) * (yCrystal[i - 2] - yCrystal[i - 1]) + static_cast<ParamType>(_weno5->IS_2_coeff_2[i]) * (yCrystal[i] - yCrystal[i - 1]) * (yCrystal[i - 2] - yCrystal[i - 1]) + static_cast<ParamType>(_weno5->IS_2_coeff_3[i]) * (yCrystal[i] - yCrystal[i - 1]) * (yCrystal[i] - yCrystal[i - 1]);
							alpha_0 = static_cast<ParamType>(_weno5->C_0[i]) / (static_cast<ParamType>(_binSizes[i]) + IS_0) / (static_cast<ParamType>(_binSizes[i]) + IS_0);
							alpha_1 = static_cast<ParamType>(_weno5->C_1[i]) / (static_cast<ParamType>(_binSizes[i]) + IS_1) / (static_cast<ParamType>(_binSizes[i]) + IS_1);
							alpha_2 = static_cast<ParamType>(_weno5->C_2[i]) / (static_cast<ParamType>(_binSizes[i]) + IS_2) / (static_cast<ParamType>(_binSizes[i]) + IS_2);
							W_0 = alpha_0 / (alpha_0 + alpha_1 + alpha_2);
							W_1 = alpha_1 / (alpha_0 + alpha_1 + alpha_2);
							W_2 = 1.0 - W_0 - W_1;
							q_0 = yCrystal[i + 1] + static_cast<ParamType>(_weno5->q_0_coeff_1[i]) * (yCrystal[i] - yCrystal[i + 1]) + static_cast<ParamType>(_weno5->q_0_coeff_2[i]) * (yCrystal[i + 1] - yCrystal[i + 2]);
							q_1 = yCrystal[i] + static_cast<ParamType>(_weno5->q_1_coeff_1[i]) * (yCrystal[i + 1] - yCrystal[i]) + static_cast<ParamType>(_weno5->q_1_coeff_2[i]) * (yCrystal[i] - yCrystal[i - 1]);
							q_2 = yCrystal[i - 1] + static_cast<ParamType>(_weno5->q_2_coeff_1[i]) * (yCrystal[i - 2] - yCrystal[i - 1]) + static_cast<ParamType>(_weno5->q_2_coeff_2[i]) * (yCrystal[i] - yCrystal[i - 1]);
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						else if (i == 2)
						{
							// weno23 applied to F_{2-1/2}
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// weno35 applied to F_{2+1/2}
							IS_0 = static_cast<ParamType>(_weno5->IS_0_coeff_1[i]) * (yCrystal[i + 2] - yCrystal[i + 1]) * (yCrystal[i + 2] - yCrystal[i + 1]) + static_cast<ParamType>(_weno5->IS_0_coeff_2[i]) * (yCrystal[i + 2] - yCrystal[i + 1]) * (yCrystal[i] - yCrystal[i + 1]) + static_cast<ParamType>(_weno5->IS_0_coeff_3[i]) * (yCrystal[i] - yCrystal[i + 1]) * (yCrystal[i] - yCrystal[i + 1]);
							IS_1 = static_cast<ParamType>(_weno5->IS_1_coeff_1[i]) * (yCrystal[i - 1] - yCrystal[i]) * (yCrystal[i - 1] - yCrystal[i]) + static_cast<ParamType>(_weno5->IS_1_coeff_2[i]) * (yCrystal[i + 1] - yCrystal[i]) * (yCrystal[i - 1] - yCrystal[i]) + static_cast<ParamType>(_weno5->IS_1_coeff_3[i]) * (yCrystal[i + 1] - yCrystal[i]) * (yCrystal[i + 1] - yCrystal[i]);
							IS_2 = static_cast<ParamType>(_weno5->IS_2_coeff_1[i]) * (yCrystal[i - 2] - yCrystal[i - 1]) * (yCrystal[i - 2] - yCrystal[i - 1]) + static_cast<ParamType>(_weno5->IS_2_coeff_2[i]) * (yCrystal[i] - yCrystal[i - 1]) * (yCrystal[i - 2] - yCrystal[i - 1]) + static_cast<ParamType>(_weno5->IS_2_coeff_3[i]) * (yCrystal[i] - yCrystal[i - 1]) * (yCrystal[i] - yCrystal[i - 1]);
							alpha_0 = static_cast<ParamType>(_weno5->C_0[i]) / (static_cast<ParamType>(_binSizes[i]) + IS_0) / (static_cast<ParamType>(_binSizes[i]) + IS_0);
							alpha_1 = static_cast<ParamType>(_weno5->C_1[i]) / (static_cast<ParamType>(_binSizes[i]) + IS_1) / (static_cast<ParamType>(_binSizes[i]) + IS_1);
							alpha_2 = static_cast<ParamType>(_weno5->C_2[i]) / (static_cast<ParamType>(_binSizes[i]) + IS_2) / (static_cast<ParamType>(_binSizes[i]) + IS_2);
							W_0 = alpha_0 / (alpha_0 + alpha_1 + alpha_2);
							W_1 = alpha_1 / (alpha_0 + alpha_1 + alpha_2);
							W_2 = 1.0 - W_0 - W_1;
							q_0 = yCrystal[i + 1] + static_cast<ParamType>(_weno5->q_0_coeff_1[i]) * (yCrystal[i] - yCrystal[i + 1]) + static_cast<ParamType>(_weno5->q_0_coeff_2[i]) * (yCrystal[i + 1] - yCrystal[i + 2]);
							q_1 = yCrystal[i] + static_cast<ParamType>(_weno5->q_1_coeff_1[i]) * (yCrystal[i + 1] - yCrystal[i]) + static_cast<ParamType>(_weno5->q_1_coeff_2[i]) * (yCrystal[i] - yCrystal[i - 1]);
							q_2 = yCrystal[i - 1] + static_cast<ParamType>(_weno5->q_2_coeff_1[i]) * (yCrystal[i - 2] - yCrystal[i - 1]) + static_cast<ParamType>(_weno5->q_2_coeff_2[i]) * (yCrystal[i] - yCrystal[i - 1]);
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						else if (i == 1)
						{
							// upwind applied to F_{1-1/2}
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * yCrystal[i - 1] - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// weno23 applied to F_{1+1/2}
							IS_0 = static_cast<ParamType>(_weno5->IS_0_coeff_weno3) * (yCrystal[i + 1] - yCrystal[i]) * (yCrystal[i + 1] - yCrystal[i]);
							IS_1 = static_cast<ParamType>(_weno5->IS_1_coeff_weno3) * (yCrystal[i] - yCrystal[i - 1]) * (yCrystal[i] - yCrystal[i - 1]);
							alpha_0 = static_cast<ParamType>(_weno5->C_coeff1_weno3) / (static_cast<ParamType>(_binSizes[i]) + IS_0) / (static_cast<ParamType>(_binSizes[i]) + IS_0);
							alpha_1 = static_cast<ParamType>(_weno5->C_coeff2_weno3) / (static_cast<ParamType>(_binSizes[i]) + IS_1) / (static_cast<ParamType>(_binSizes[i]) + IS_1);
							W_0 = alpha_0 / (alpha_0 + alpha_1);
							W_1 = 1.0 - W_0;
							q_0 = static_cast<ParamType>(_weno5->q0_coeff1_weno3) * yCrystal[i] + static_cast<ParamType>(_weno5->q0_coeff2_weno3) * yCrystal[i + 1];
							q_1 = static_cast<ParamType>(_weno5->q1_coeff1_weno3) * yCrystal[i] + static_cast<ParamType>(_weno5->q1_coeff2_weno3) * yCrystal[i - 1];
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						else if (i == 0)
						{
							// boundary condition
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * B_0;
							// upwind to F_{1/2}
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * yCrystal[i] - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						else if (i + 2 == _nBins)
						{
							// weno35 applied to F_{nx-2-1/2}
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// weno23 applied to F_{nx-2+1/2}
							IS_0 = static_cast<ParamType>(_weno5->IS_0_coeff_weno3_r) * (yCrystal[i + 1] - yCrystal[i]) * (yCrystal[i + 1] - yCrystal[i]);
							IS_1 = static_cast<ParamType>(_weno5->IS_1_coeff_weno3_r) * (yCrystal[i] - yCrystal[i - 1]) * (yCrystal[i] - yCrystal[i - 1]);
							alpha_0 = static_cast<ParamType>(_weno5->C_coeff1_weno3_r) / (static_cast<ParamType>(_binSizes[i]) + IS_0) / (static_cast<ParamType>(_binSizes[i]) + IS_0);
							alpha_1 = static_cast<ParamType>(_weno5->C_coeff2_weno3_r) / (static_cast<ParamType>(_binSizes[i]) + IS_1) / (static_cast<ParamType>(_binSizes[i]) + IS_1);
							W_0 = alpha_0 / (alpha_0 + alpha_1);
							W_1 = 1.0 - W_0;
							q_0 = static_cast<ParamType>(_weno5->q0_coeff1_weno3_r) * yCrystal[i] + static_cast<ParamType>(_weno5->q0_coeff2_weno3_r) * yCrystal[i + 1];
							q_1 = static_cast<ParamType>(_weno5->q1_coeff1_weno3_r) * yCrystal[i] + static_cast<ParamType>(_weno5->q1_coeff2_weno3_r) * yCrystal[i - 1];
							v_g = k_g_times_s_g * (static_cast<ParamType>(_a) + static_cast<ParamType>(_growthConstant) * pow(static_cast<ParamType>(_bins[i + 1]), static_cast<ParamType>(_p)));
							resCrystal[i] -= factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i + 1] - yCrystal[i]) / static_cast<ParamType>(_binCenterDists[i]);
						}
						else
						{
							// weno23 to F_{nx-1-1/2}
							resCrystal[i] += factor / static_cast<ParamType>(_binSizes[i]) * v_g * (W_0 * q_0 + W_1 * q_1) - factor * static_cast<ParamType>(_growthDispersionRate) / static_cast<ParamType>(_binSizes[i]) * (yCrystal[i] - yCrystal[i - 1]) / static_cast<ParamType>(_binCenterDists[i - 1]);
							// no flux BC
						}
					}
				}
				break;
				}

				return 0;
			}

			template <typename StateType, typename ResidualType, typename ParamType>
			int residualCombinedImpl(double t, unsigned int secIdx, const ColumnPosition& colPos,
				StateType const* yLiquid, StateType const* ySolid, ResidualType* resLiquid, ResidualType* resSolid, double factor, LinearBufferAllocator workSpace) const
			{
				return residualLiquidImpl<StateType, ResidualType, ParamType, double>(t, secIdx, colPos, yLiquid, resLiquid, factor, workSpace);
			}

			/**
             * @brief Jacobian implementations. See @cite Zhang2024_PBM_partI
             */

			template <typename RowIterator>
			void jacobianLiquidImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor, RowIterator& jac, LinearBufferAllocator workSpace) const
			{
				// Pointer to crystal bins
				double const* const yCrystal = y + 1;

				// supersaturation s, rewrite it to 0.0 if it drops below 0.0
				double const s = (cadet_likely(y[0] / y[_nComp - 1] - 1.0 > 0)) ? y[0] / y[_nComp - 1] - 1.0 : 0.0;

				// compute the summations 
				double M = 0.0;
				double substrateConversion = 0.0;
				for (int i = 0; i < _nBins; ++i)
				{
					M += yCrystal[i] * static_cast<double>(_binCenters[i]) * static_cast<double>(_binCenters[i]) * static_cast<double>(_binCenters[i]) * static_cast<double>(_binSizes[i]);
					substrateConversion += yCrystal[i] * static_cast<double>(_binCenters[i]) * static_cast<double>(_binCenters[i]) * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_binCenters[i]), static_cast<double>(_p))) * static_cast<double>(_binSizes[i]);
				}

				// define variables to aid jacobian implementation
				double const vG_factor = static_cast<double>(_growthRateConstant) * pow(s, static_cast<double>(_g));

				double const rho_kv = static_cast<double>(_nucleiMassDensity) * static_cast<double>(_volShapeFactor);
				double const x_c_3 = static_cast<double>(_bins[0]) * static_cast<double>(_bins[0]) * static_cast<double>(_bins[0]);

				// rewrite the derivatives to 0.0 if s = 0.0
				double const dBp_dc = cadet_unlikely(s > 0.0) ? static_cast<double>(_primaryNucleationRate) * static_cast<double>(_u) * pow(s, static_cast<double>(_u) - 1.0) / y[_nComp - 1] : 0.0;
				double const dBp_dceq = cadet_unlikely(s > 0.0) ? -static_cast<double>(_primaryNucleationRate) * static_cast<double>(_u) * pow(s, static_cast<double>(_u) - 1.0) * y[0] / y[_nComp - 1] / y[_nComp - 1] : 0.0;
				double const dBs_dc = cadet_unlikely(s > 0.0) ? rho_kv * static_cast<double>(_secondaryNucleationRate) * static_cast<double>(_b) * pow(s, static_cast<double>(_b) - 1.0) * M / y[_nComp - 1] : 0.0;
				double const dBs_dceq = cadet_unlikely(s > 0.0) ? -rho_kv * static_cast<double>(_secondaryNucleationRate) * static_cast<double>(_b) * pow(s, static_cast<double>(_b) - 1.0) * M * y[0] / y[_nComp - 1] / y[_nComp - 1] : 0.0;
				double const dvG_dc_factor = cadet_unlikely(s > 0.0) ? static_cast<double>(_growthRateConstant) * static_cast<double>(_g) * pow(s, static_cast<double>(_g) - 1.0) / y[_nComp - 1] : 0.0;
				double const dvG_dceq_factor = cadet_unlikely(s > 0.0) ? -static_cast<double>(_growthRateConstant) * static_cast<double>(_g) * pow(s, static_cast<double>(_g) - 1.0) * y[0] / y[_nComp - 1] / y[_nComp - 1] : 0.0;
				double const dBs_dni_factor = cadet_unlikely(s > 0.0) ? rho_kv * static_cast<double>(_secondaryNucleationRate) * pow(s, static_cast<double>(_b)) : 0.0;

				// the jacobian has a shape: (_nComp) x (_nComp), undefined ones are 0.0.
				// the first loop i iterates over rows, the second loop j iterates over columns. The offset i is used to move the jac index to 0 at the beginning of iterating over j.
				const int GrowthOrder = static_cast<int>(_growthSchemeOrder);
				int binIdx_i = 0;
				int binIdx_j = 0;
				switch (GrowthOrder)
				{
					// upwind
				case 1:
				{
					for (int i = 0; i < _nComp; ++i)
					{
						if (cadet_likely((i > 1) && (i + 2 < _nComp)))
						{
							// intermediate cells
							binIdx_i = i - 1;
							// dQ_i/dc
							jac[0 - i] -= factor * (dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) * yCrystal[binIdx_i] - dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) * yCrystal[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]);

							// dQ_i/dn_{i-1}
							jac[-1] += factor * vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]);

							// dQ_i/dn_i
							jac[0] -= factor * vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binSizes[binIdx_i]) * (1.0 / static_cast<double>(_binCenterDists[binIdx_i]) + 1.0 / static_cast<double>(_binCenterDists[binIdx_i - 1]));

							// dQ_i/dn_{i+1}
							jac[1] += factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]);

							//dQ_i/dc_{eq}
							jac[_nComp - 1 - i] -= factor * (dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) * yCrystal[binIdx_i] - dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) * yCrystal[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]);
						}
						else if (cadet_unlikely(i == 0))
						{
							// Q_c, mass balance, independent on the scheme
							for (int j = 0; j < _nComp; ++j)
							{
								binIdx_j = j - 1;
								if (cadet_likely((j > 0) && (j + 1 < _nComp)))
								{
									// dQ_c/dn_i
									jac[j - i] -= factor * rho_kv * (x_c_3 * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) + 3.0 * vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_binCenters[binIdx_j]), static_cast<double>(_p))) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]));
								}
								else if (cadet_unlikely(j == 0))
								{
									// dQ_c/dc
									jac[j - i] -= factor * rho_kv * ((dBp_dc + dBs_dc) * x_c_3 + 3.0 * dvG_dc_factor * substrateConversion);
								}
								else
								{
									// dQ_c/dc_eq
									jac[j - i] -= factor * rho_kv * ((dBp_dceq + dBs_dceq) * x_c_3 + 3.0 * dvG_dceq_factor * substrateConversion);
								}
							}
						}
						else if (cadet_unlikely(i == 1))
						{
							binIdx_i = i - 1;
							// Q_0, left BC
							for (int j = 0; j < _nComp; ++j)
							{
								binIdx_j = j - 1;
								if (cadet_likely((j > 2) && (j + 1 < _nComp)))
								{
									// dQ_0/dni
									jac[j - i] += factor * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 0))
								{
									// dQ_0/dc
									jac[j - i] -= factor * (yCrystal[0] * dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) - dBp_dc - dBs_dc) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 1))
								{
									// dQ_0/dn0
									jac[j - i] -= factor * (vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_j + 1]), static_cast<double>(_p))) - dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j])) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 2))
								{
									// dQ_0/dn1
									jac[j - i] += factor * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else
								{
									// dQ_0/dceq
									jac[j - i] -= factor * (yCrystal[0] * dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) - dBp_dceq - dBs_dceq) / static_cast<double>(_binSizes[binIdx_i]);
								}
							}
						}
						else if (cadet_unlikely(i == _nComp - 2))
						{
							binIdx_i = i - 1;
							// Q_{N_x-1}, right BC

							// dQ_{N_x-1}/dc
							jac[0 - i] += factor * yCrystal[binIdx_i - 1] * dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) / static_cast<double>(_binSizes[binIdx_i]);

							// dQ_{N_x-1}/dn_{N_x-2}
							jac[_nComp - 3 - i] += factor * vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]);

							// dQ_{N_x-1}/dn_{N_x-1}
							jac[_nComp - 2 - i] -= factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]);

							// dQ_{N_x-1}/dc_eq
							jac[_nComp - 1 - i] += factor * yCrystal[binIdx_i - 1] * dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) / static_cast<double>(_binSizes[binIdx_i]);
						}
						else
						{
							// Q_{ceq}, the last row is zero, which is the default.
							continue;
						}

						// go to the next row
						++jac;
					}
				}
				break;
				// HR
				case 2:
				{
					// HR related coefficients
					const double epsilon = 1e-10;
					double r_x_i = 0.0;
					double phi = 0.0;
					double F_i = 0.0;
					double dvG_dc_right = 0.0;
					double dvG_dceq_right = 0.0;
					double vg_right = 0.0;

					double r_cluster = 0.0;
					double r_square_cluster = 0.0;
					double A_cluster = 0.0;
					double Ar_cluster = 0.0;
					double ni_difference = 0.0;

					double dFi_wrt_nim1 = 0.0;
					double dFi_wrt_ni = 0.0;
					double dFi_wrt_nip1 = 0.0;

					for (int i = 0; i < _nComp; ++i)
					{
						// Q_i, intermediate cells
						if (cadet_likely((i > 2) && (i + 2 < _nComp)))
						{
							binIdx_i = i - 1;
							// jacobian, left face, which is the right face of a previous cell
							// dQ_i/dc, j=0
							jac[-i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * F_i;

							// dQ_i/dn_{i-2}, j=i-2
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_nim1; // the second term in the product rule is zero

							// dQ_i/dn_{i-1}, j=i-1
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_ni; // full four terms in the product rule

							// dQ_i/dn_i, j=i
							jac[0] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_nip1; // the fourth term in the product rule is zero

							//dQ_i/dc_{eq}, j=_nComp - 1
							jac[_nComp - 1 - i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * F_i;

							// update all coefficients
							// HR-related coefficients, right face
							r_x_i = static_cast<double>(_HR->A_coeff[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1] + epsilon) / (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i] + epsilon);
							if (cadet_likely(r_x_i > 0))
							{
								phi = r_x_i / (static_cast<double>(_HR->R_coeff[binIdx_i]) - 1.0 + r_x_i);
								F_i = yCrystal[binIdx_i] + phi * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);

								// Jacobian related coefficients, right face
								ni_difference = 1.0 - epsilon / (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i] + epsilon);
								r_cluster = phi;
								A_cluster = static_cast<double>(_HR->A_coeff[binIdx_i]) / (static_cast<double>(_HR->R_coeff[binIdx_i]) - 1.0 + r_x_i);
								r_square_cluster = phi * phi;
								Ar_cluster = phi * A_cluster;

								dFi_wrt_nim1 = Ar_cluster * ni_difference - ni_difference * A_cluster;
								dFi_wrt_ni = 1.0 - ni_difference * (r_square_cluster + Ar_cluster) + r_cluster * ni_difference - r_cluster + ni_difference * A_cluster;
								dFi_wrt_nip1 = r_square_cluster * ni_difference - r_cluster * ni_difference + r_cluster;
							}
							else
							{
								phi = 0.0;
								F_i = yCrystal[binIdx_i];

								// Jacobian related coefficients, right face
								dFi_wrt_nim1 = 0.0; 
								dFi_wrt_ni = 1.0; // becomes upwind
								dFi_wrt_nip1 = 0.0;
							}
							vg_right = vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							dvG_dc_right = dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							dvG_dceq_right = dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							// jacobian, right face, and diffusion
							// dQ_i/dc, j=0
							jac[-i] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * F_i;

							// dQ_i/dn_{i-1}, j=i-1
							jac[-1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_nim1 - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // the second term in the product rule is zero

							// dQ_i/dn_i, j=i
							jac[0] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_ni + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binSizes[binIdx_i]) * (1.0 / static_cast<double>(_binCenterDists[binIdx_i]) + 1.0 / static_cast<double>(_binCenterDists[binIdx_i - 1])); // full four terms in the product rule

							// dQ_i/dn_{i+1}, j=i+1
							jac[1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_nip1 - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]); // the fourth term in the product rule is zero

							//dQ_i/dc_{eq}, j=_nComp - 1
							jac[_nComp - 1 - i] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * F_i;
						}

						// Q_c, mass balance, independent on the scheme. Do not change.
						else if (cadet_unlikely(i == 0))
						{
							for (int j = 0; j < _nComp; ++j)
							{
								binIdx_j = j - 1;
								if (cadet_likely((j > 0) && (j + 1 < _nComp)))
								{
									// dQ_c/dn_i
									jac[j - i] -= factor * rho_kv * (x_c_3 * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) + 3.0 * vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_binCenters[binIdx_j]), static_cast<double>(_p))) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]));
								}
								else if (cadet_unlikely(j == 0))
								{
									// dQ_c/dc
									jac[j - i] -= factor * rho_kv * ((dBp_dc + dBs_dc) * x_c_3 + 3.0 * dvG_dc_factor * substrateConversion);
								}
								else
								{
									// dQ_c/dc_eq
									jac[j - i] -= factor * rho_kv * ((dBp_dceq + dBs_dceq) * x_c_3 + 3.0 * dvG_dceq_factor * substrateConversion);
								}
							}
						}

						// Q_0, left BC, also independent on the scheme. Do not change.
						else if (cadet_unlikely(i == 1))
						{
							binIdx_i = i - 1;
							// Q_0, left BC
							for (int j = 0; j < _nComp; ++j)
							{
								binIdx_j = j - 1;
								if (cadet_likely((j > 2) && (j + 1 < _nComp)))
								{
									// dQ_0/dni
									jac[j - i] += factor * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 0))
								{
									// dQ_0/dc
									jac[j - i] -= factor * (yCrystal[0] * dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) - dBp_dc - dBs_dc) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 1))
								{
									// dQ_0/dn0
									jac[j - i] -= factor * (vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_j + 1]), static_cast<double>(_p))) - dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j])) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 2))
								{
									// dQ_0/dn1
									jac[j - i] += factor * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else
								{
									// dQ_0/dceq
									jac[j - i] -= factor * (yCrystal[0] * dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) - dBp_dceq - dBs_dceq) / static_cast<double>(_binSizes[binIdx_i]);
								}
							}
						}

						// Q_1, special case for HR.
						else if (cadet_unlikely(i == 2))
						{
							binIdx_i = i - 1;
							// HR-related coefficients, right face
							r_x_i = static_cast<double>(_HR->A_coeff[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1] + epsilon) / (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i] + epsilon);
							if (cadet_likely(r_x_i > 0))
							{
								phi = r_x_i / (static_cast<double>(_HR->R_coeff[binIdx_i]) - 1.0 + r_x_i);
								F_i = yCrystal[binIdx_i] + phi * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);

								// Jacobian related coefficients, right face
								ni_difference = 1.0 - epsilon / (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i] + epsilon);
								r_cluster = phi;
								A_cluster = static_cast<double>(_HR->A_coeff[binIdx_i]) / (static_cast<double>(_HR->R_coeff[binIdx_i]) - 1.0 + r_x_i);
								r_square_cluster = phi * phi;
								Ar_cluster = phi * A_cluster;

								dFi_wrt_nim1 = Ar_cluster * ni_difference - ni_difference * A_cluster;
								dFi_wrt_ni = 1.0 - ni_difference * (r_square_cluster + Ar_cluster) + r_cluster * ni_difference - r_cluster + ni_difference * A_cluster;
								dFi_wrt_nip1 = r_square_cluster * ni_difference - r_cluster * ni_difference + r_cluster;
							}
							else
							{
								phi = 0.0;
								F_i = yCrystal[binIdx_i];

								// Jacobian related coefficients, right face
								dFi_wrt_nim1 = 0.0; 
								dFi_wrt_ni = 1.0; // becomes upwind
								dFi_wrt_nip1 = 0.0;
							}
							vg_right = vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							dvG_dc_right = dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							dvG_dceq_right = dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							// dQ_1/dc, j=0
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) * yCrystal[0]; // left face, upwind scheme
							jac[-2] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * F_i; // right face, HR

							// dQ_1/dn0, j=1
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // coming from the upwind scheme
							jac[-1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_nim1; // the second term in the product rule is zero

							// dQ_1/dn1, j=2
							jac[0] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_ni + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binSizes[binIdx_i]) * (1.0 / static_cast<double>(_binCenterDists[binIdx_i]) + 1.0 / static_cast<double>(_binCenterDists[binIdx_i - 1])); // full four terms in the product rule

							// dQ_1/dn2, j=3
							jac[1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_nip1 - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]); // the fourth term in the product rule is zero

							// dQ_1/dceq, j=_nComp -1
							jac[_nComp - 3] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) * yCrystal[0];
							jac[_nComp - 3] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * F_i;
						}

						// Q_{N_x-1}, right BC
						else if (cadet_unlikely(i == _nComp - 2))
						{
							binIdx_i = i - 1;
							// dQ_{N_x-1}/dc
							jac[-i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * F_i;

							// dQ_{N_x-1}/dn_{N_x-3}
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_nim1; // the second term in the product rule is zero

							// dQ_{N_x-1}/dn_{N_x-2}
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_ni + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // full four terms in the product rule

							// dQ_{N_x-1}/dn_{N_x-1}
							jac[0] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * dFi_wrt_nip1 - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // the fourth term in the product rule is zero

							//dQ_i/dc_{eq}, j=_nComp - 1
							jac[1] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * F_i;
						}

						// Q_{ceq}, the last row is zero, which is the default. Do not change.
						else
						{
							continue;
						}

						// go to the next row
						++jac;
					}
				}
				break;
				// WENO23
				case 3:
				{
					// WENO23 related coefficients
					double IS_0_right = 0.0;
					double IS_1_right = 0.0;
					double alpha_0_right = 0.0;
					double alpha_1_right = 0.0;
					double W_0_right = 0.0;
					double W_1_right = 0.0;
					double q_0_right = 0.0;
					double q_1_right = 0.0;
					double dvG_dc_right = 0.0;
					double dvG_dceq_right = 0.0;
					double vg_right = 0.0;

					// Jacobian related coefficients
					double four_w1_B1_right = 0.0;
					double four_w0_B0_right = 0.0;
					double four_w0_w1_B1_right = 0.0;
					double four_w0_w0_B0_right = 0.0;
					double four_w1_w1_B1_right = 0.0;
					double four_w0_w1_B0_right = 0.0;

					double dw0_right_dni_m1 = 0.0;
					double dw0_right_dni = 0.0;
					double dw0_right_dni_p1 = 0.0;

					double dw1_right_dni_m1 = 0.0;
					double dw1_right_dni = 0.0;
					double dw1_right_dni_p1 = 0.0;

					double dq0_right_dni = 0.0;
					double dq0_right_dni_p1 = 0.0;

					double dq1_right_dni_m1 = 0.0;
					double dq1_right_dni = 0.0;

					// note: binIdx_j is used only when B_s is involved.
					for (int i = 0; i < _nComp; ++i)
					{
						// Q_i, intermediate cells
						if (cadet_likely((i > 2) && (i + 2 < _nComp)))
						{
							binIdx_i = i - 1;

							// jacobian, left face, which is the right face of a previous cell
							// dQ_i/dc, j=0
							jac[-i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0_right * q_0_right + W_1_right * q_1_right);

							// dQ_i/dn_{i-2}, j=i-2
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_m1 * q_0_right + dw1_right_dni_m1 * q_1_right + W_1_right * dq1_right_dni_m1); // the second term in the product rule is zero

							// dQ_i/dn_{i-1}, j=i-1
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni * q_0_right + W_0_right * dq0_right_dni + dw1_right_dni * q_1_right + W_1_right * dq1_right_dni); // full four terms in the product rule

							// dQ_i/dn_i, j=i
							jac[0] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_p1 * q_0_right + W_0_right * dq0_right_dni_p1 + dw1_right_dni_p1 * q_1_right); // the fourth term in the product rule is zero

							//dQ_i/dc_{eq}, j=_nComp - 1
							jac[_nComp - 1 - i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0_right * q_0_right + W_1_right * q_1_right);

							// update all coefficients
							// WENO23-related coefficients, right face
							IS_0_right = static_cast<double>(_weno3->IS_0_coeff[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);
							IS_1_right = static_cast<double>(_weno3->IS_1_coeff[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);
							alpha_0_right = static_cast<double>(_weno3->C_right_coeff[binIdx_i]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_0_right) / (static_cast<double>(_binSizes[binIdx_i]) + IS_0_right);
							alpha_1_right = (1.0 - static_cast<double>(_weno3->C_right_coeff[binIdx_i])) / (static_cast<double>(_binSizes[binIdx_i]) + IS_1_right) / (static_cast<double>(_binSizes[binIdx_i]) + IS_1_right);
							W_0_right = alpha_0_right / (alpha_0_right + alpha_1_right);
							W_1_right = 1.0 - W_0_right;
							q_0_right = static_cast<double>(_weno3->q_0_right_coeff[binIdx_i]) * yCrystal[binIdx_i] + (1.0 - static_cast<double>(_weno3->q_0_right_coeff[binIdx_i])) * yCrystal[binIdx_i + 1];
							q_1_right = static_cast<double>(_weno3->q_1_right_coeff[binIdx_i]) * yCrystal[binIdx_i] + (1.0 - static_cast<double>(_weno3->q_1_right_coeff[binIdx_i])) * yCrystal[binIdx_i - 1];

							dvG_dc_right = dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							dvG_dceq_right = dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							vg_right = vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							// Jacobian related coefficients, right face
							four_w0_B0_right = 4.0 * W_0_right * static_cast<double>(_weno3->IS_0_coeff[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_0_right);
							four_w1_B1_right = 4.0 * W_1_right * static_cast<double>(_weno3->IS_1_coeff[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_1_right);
							four_w0_w0_B0_right = four_w0_B0_right * W_0_right;
							four_w0_w1_B1_right = four_w1_B1_right * W_0_right;
							four_w0_w1_B0_right = four_w0_B0_right * W_1_right;
							four_w1_w1_B1_right = four_w1_B1_right * W_1_right;

							dw0_right_dni_m1 = -four_w0_w1_B1_right;
							dw0_right_dni = four_w0_B0_right - four_w0_w0_B0_right + four_w0_w1_B1_right;
							dw0_right_dni_p1 = four_w0_w0_B0_right - four_w0_B0_right;

							dw1_right_dni_m1 = four_w1_B1_right - four_w1_w1_B1_right;
							dw1_right_dni = -four_w1_B1_right - four_w0_w1_B0_right + four_w1_w1_B1_right;
							dw1_right_dni_p1 = four_w0_w1_B0_right;

							dq0_right_dni = static_cast<double>(_weno3->q_0_right_coeff[binIdx_i]);
							dq0_right_dni_p1 = 1.0 - static_cast<double>(_weno3->q_0_right_coeff[binIdx_i]);

							dq1_right_dni_m1 = 1.0 - static_cast<double>(_weno3->q_1_right_coeff[binIdx_i]);
							dq1_right_dni = static_cast<double>(_weno3->q_1_right_coeff[binIdx_i]);

							// jacobian, right face, and diffusion
							// dQ_i/dc, j=0
							jac[-i] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0_right * q_0_right + W_1_right * q_1_right);

							// dQ_i/dn_{i-1}, j=i-1
							jac[-1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_m1 * q_0_right + dw1_right_dni_m1 * q_1_right + W_1_right * dq1_right_dni_m1) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // the second term in the product rule is zero

							// dQ_i/dn_i, j=i
							jac[0] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni * q_0_right + W_0_right * dq0_right_dni + dw1_right_dni * q_1_right + W_1_right * dq1_right_dni) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binSizes[binIdx_i]) * (1.0 / static_cast<double>(_binCenterDists[binIdx_i]) + 1.0 / static_cast<double>(_binCenterDists[binIdx_i - 1])); // full four terms in the product rule

							// dQ_i/dn_{i+1}, j=i+1
							jac[1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_p1 * q_0_right + W_0_right * dq0_right_dni_p1 + dw1_right_dni_p1 * q_1_right) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]); // the fourth term in the product rule is zero

							//dQ_i/dc_{eq}, j=_nComp - 1
							jac[_nComp - 1 - i] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0_right * q_0_right + W_1_right * q_1_right);
						}

						// Q_c, mass balance, independent on the scheme. Do not change.
						else if (cadet_unlikely(i == 0))
						{
							for (int j = 0; j < _nComp; ++j)
							{
								binIdx_j = j - 1;
								if (cadet_likely((j > 0) && (j + 1 < _nComp)))
								{
									// dQ_c/dn_i
									jac[j - i] -= factor * rho_kv * (x_c_3 * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) + 3.0 * vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_binCenters[binIdx_j]), static_cast<double>(_p))) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]));
								}
								else if (cadet_unlikely(j == 0))
								{
									// dQ_c/dc
									jac[j - i] -= factor * rho_kv * ((dBp_dc + dBs_dc) * x_c_3 + 3.0 * dvG_dc_factor * substrateConversion);
								}
								else
								{
									// dQ_c/dc_eq
									jac[j - i] -= factor * rho_kv * ((dBp_dceq + dBs_dceq) * x_c_3 + 3.0 * dvG_dceq_factor * substrateConversion);
								}
							}
						}

						// Q_0, left BC, also independent on the scheme. Do not change.
						else if (cadet_unlikely(i == 1))
						{
							binIdx_i = i - 1;
							// Q_0, left BC
							for (int j = 0; j < _nComp; ++j)
							{
								binIdx_j = j - 1;
								if (cadet_likely((j > 2) && (j + 1 < _nComp)))
								{
									// dQ_0/dni
									jac[j - i] += factor * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 0))
								{
									// dQ_0/dc
									jac[j - i] -= factor * (yCrystal[0] * dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) - dBp_dc - dBs_dc) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 1))
								{
									// dQ_0/dn0
									jac[j - i] -= factor * (vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_j + 1]), static_cast<double>(_p))) - dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j])) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 2))
								{
									// dQ_0/dn1
									jac[j - i] += factor * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else
								{
									// dQ_0/dceq
									jac[j - i] -= factor * (yCrystal[0] * dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) - dBp_dceq - dBs_dceq) / static_cast<double>(_binSizes[binIdx_i]);
								}
							}
						}

						// Q_1, special case for weno3.
						else if (cadet_unlikely(i == 2))
						{
							binIdx_i = i - 1;
							// WENO23-related coefficients, right face
							IS_0_right = static_cast<double>(_weno3->IS_0_coeff[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);
							IS_1_right = static_cast<double>(_weno3->IS_1_coeff[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);
							alpha_0_right = static_cast<double>(_weno3->C_right_coeff[binIdx_i]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_0_right) / (static_cast<double>(_binSizes[binIdx_i]) + IS_0_right);
							alpha_1_right = (1.0 - static_cast<double>(_weno3->C_right_coeff[binIdx_i])) / (static_cast<double>(_binSizes[binIdx_i]) + IS_1_right) / (static_cast<double>(_binSizes[binIdx_i]) + IS_1_right);
							W_0_right = alpha_0_right / (alpha_0_right + alpha_1_right);
							W_1_right = 1.0 - W_0_right;
							q_0_right = static_cast<double>(_weno3->q_0_right_coeff[binIdx_i]) * yCrystal[binIdx_i] + (1.0 - static_cast<double>(_weno3->q_0_right_coeff[binIdx_i])) * yCrystal[binIdx_i + 1];
							q_1_right = static_cast<double>(_weno3->q_1_right_coeff[binIdx_i]) * yCrystal[binIdx_i] + (1.0 - static_cast<double>(_weno3->q_1_right_coeff[binIdx_i])) * yCrystal[binIdx_i - 1];

							dvG_dc_right = dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							dvG_dceq_right = dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							vg_right = vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							// Jacobian related coefficients, right face
							four_w0_B0_right = 4.0 * W_0_right * static_cast<double>(_weno3->IS_0_coeff[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_0_right);
							four_w1_B1_right = 4.0 * W_1_right * static_cast<double>(_weno3->IS_1_coeff[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_1_right);
							four_w0_w0_B0_right = four_w0_B0_right * W_0_right;
							four_w0_w1_B1_right = four_w1_B1_right * W_0_right;
							four_w0_w1_B0_right = four_w0_B0_right * W_1_right;
							four_w1_w1_B1_right = four_w1_B1_right * W_1_right;

							dw0_right_dni_m1 = -four_w0_w1_B1_right;
							dw0_right_dni = four_w0_B0_right - four_w0_w0_B0_right + four_w0_w1_B1_right;
							dw0_right_dni_p1 = four_w0_w0_B0_right - four_w0_B0_right;

							dw1_right_dni_m1 = four_w1_B1_right - four_w1_w1_B1_right;
							dw1_right_dni = -four_w1_B1_right - four_w0_w1_B0_right + four_w1_w1_B1_right;
							dw1_right_dni_p1 = four_w0_w1_B0_right;

							dq0_right_dni = static_cast<double>(_weno3->q_0_right_coeff[binIdx_i]);
							dq0_right_dni_p1 = 1.0 - static_cast<double>(_weno3->q_0_right_coeff[binIdx_i]);

							dq1_right_dni_m1 = 1.0 - static_cast<double>(_weno3->q_1_right_coeff[binIdx_i]);
							dq1_right_dni = static_cast<double>(_weno3->q_1_right_coeff[binIdx_i]);

							// dQ_1/dc, j=0
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) * yCrystal[0]; // left face, upwind scheme
							jac[-2] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0_right * q_0_right + W_1_right * q_1_right); // right face, WENO3

							// dQ_1/dn0, j=1
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // coming from the upwind scheme
							jac[-1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_m1 * q_0_right + dw1_right_dni_m1 * q_1_right + W_1_right * dq1_right_dni_m1); // the second term in the product rule is zero

							// dQ_1/dn1, j=2
							jac[0] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni * q_0_right + W_0_right * dq0_right_dni + dw1_right_dni * q_1_right + W_1_right * dq1_right_dni) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binSizes[binIdx_i]) * (1.0 / static_cast<double>(_binCenterDists[binIdx_i]) + 1.0 / static_cast<double>(_binCenterDists[binIdx_i - 1])); // full four terms in the product rule

							// dQ_1/dn2, j=3
							jac[1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_p1 * q_0_right + W_0_right * dq0_right_dni_p1 + dw1_right_dni_p1 * q_1_right) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]); // the fourth term in the product rule is zero

							// dQ_1/dceq, j=_nComp -1
							jac[_nComp - 3] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) * yCrystal[0];
							jac[_nComp - 3] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0_right * q_0_right + W_1_right * q_1_right);
						}

						// Q_{N_x-1}, right BC
						else if (cadet_unlikely(i == _nComp - 2))
						{
							binIdx_i = i - 1;
							// dQ_{N_x-1}/dc
							jac[-i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0_right * q_0_right + W_1_right * q_1_right);

							// dQ_{N_x-1}/dn_{N_x-3}
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_m1 * q_0_right + dw1_right_dni_m1 * q_1_right + W_1_right * dq1_right_dni_m1); // the second term in the product rule is zero

							// dQ_{N_x-1}/dn_{N_x-2}
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni * q_0_right + W_0_right * dq0_right_dni + dw1_right_dni * q_1_right + W_1_right * dq1_right_dni) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // full four terms in the product rule

							// dQ_{N_x-1}/dn_{N_x-1}
							jac[0] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_p1 * q_0_right + W_0_right * dq0_right_dni_p1 + dw1_right_dni_p1 * q_1_right) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // the fourth term in the product rule is zero

							//dQ_i/dc_{eq}, j=_nComp - 1
							jac[1] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0_right * q_0_right + W_1_right * q_1_right);
						}

						// Q_{ceq}, the last row is zero, which is the default. Do not change.
						else
						{
							continue;
						}

						// go to the next row
						++jac;
					}
				}
				break;
				// WENO35
				case 4:
				{
					// WENO23 related coefficients, for bin N_x-2
					const double IS_0_weno3_right = static_cast<double>(_weno5->IS_0_coeff_weno3_r) * (yCrystal[_nComp - 3] - yCrystal[_nComp - 4]) * (yCrystal[_nComp - 3] - yCrystal[_nComp - 4]);
					const double IS_1_weno3_right = static_cast<double>(_weno5->IS_1_coeff_weno3_r) * (yCrystal[_nComp - 4] - yCrystal[_nComp - 5]) * (yCrystal[_nComp - 4] - yCrystal[_nComp - 5]);
					const double alpha_0_weno3_right = static_cast<double>(_weno5->C_coeff1_weno3_r) / (static_cast<double>(_binSizes[_nComp - 4]) + IS_0_weno3_right) / (static_cast<double>(_binSizes[_nComp - 4]) + IS_0_weno3_right);
					const double alpha_1_weno3_right = static_cast<double>(_weno5->C_coeff2_weno3_r) / (static_cast<double>(_binSizes[_nComp - 4]) + IS_1_weno3_right) / (static_cast<double>(_binSizes[_nComp - 4]) + IS_1_weno3_right);
					const double W_0_weno3_right = alpha_0_weno3_right / (alpha_0_weno3_right + alpha_1_weno3_right);
					const double W_1_weno3_right = 1.0 - W_0_weno3_right;
					const double q_0_weno3_right = static_cast<double>(_weno5->q0_coeff1_weno3_r) * yCrystal[_nComp - 4] + static_cast<double>(_weno5->q0_coeff2_weno3_r) * yCrystal[_nComp - 3];
					const double q_1_weno3_right = static_cast<double>(_weno5->q1_coeff1_weno3_r) * yCrystal[_nComp - 4] + static_cast<double>(_weno5->q1_coeff2_weno3_r) * yCrystal[_nComp - 5];

					// WENO23 jacobian related coefficients, for bin N_x-2
					const double four_w0_B0_weno3_right = 4.0 * W_0_weno3_right * static_cast<double>(_weno5->IS_0_coeff_weno3_r) * (yCrystal[_nComp - 3] - yCrystal[_nComp - 4]) / (static_cast<double>(_binSizes[_nComp - 4]) + IS_0_weno3_right);
					const double four_w1_B1_weno3_right = 4.0 * W_1_weno3_right * static_cast<double>(_weno5->IS_1_coeff_weno3_r) * (yCrystal[_nComp - 4] - yCrystal[_nComp - 5]) / (static_cast<double>(_binSizes[_nComp - 4]) + IS_1_weno3_right);
					const double four_w0_w0_B0_weno3_right = four_w0_B0_weno3_right * W_0_weno3_right;
					const double four_w0_w1_B1_weno3_right = four_w1_B1_weno3_right * W_0_weno3_right;
					const double four_w0_w1_B0_weno3_right = four_w0_B0_weno3_right * W_1_weno3_right;
					const double four_w1_w1_B1_weno3_right = four_w1_B1_weno3_right * W_1_weno3_right;

					const double dw0_right_dni_m1_weno3_right = -four_w0_w1_B1_weno3_right;
					const double dw0_right_dni_weno3_right = four_w0_B0_weno3_right - four_w0_w0_B0_weno3_right + four_w0_w1_B1_weno3_right;
					const double dw0_right_dni_p1_weno3_right = four_w0_w0_B0_weno3_right - four_w0_B0_weno3_right;

					const double dw1_right_dni_m1_weno3_right = four_w1_B1_weno3_right - four_w1_w1_B1_weno3_right;
					const double dw1_right_dni_weno3_right = -four_w1_B1_weno3_right - four_w0_w1_B0_weno3_right + four_w1_w1_B1_weno3_right;
					const double dw1_right_dni_p1_weno3_right = four_w0_w1_B0_weno3_right;

					const double dq0_right_dni_weno3_right = static_cast<double>(_weno5->q0_coeff1_weno3_r);
					const double dq0_right_dni_p1_weno3_right = static_cast<double>(_weno5->q0_coeff2_weno3_r);
					const double dq1_right_dni_m1_weno3_right = static_cast<double>(_weno5->q1_coeff2_weno3_r);
					const double dq1_right_dni_weno3_right = static_cast<double>(_weno5->q1_coeff1_weno3_r);

					// WENO23 related coefficients, for bin 2
					const double IS_0_weno3_left = static_cast<double>(_weno5->IS_0_coeff_weno3) * (yCrystal[2] - yCrystal[1]) * (yCrystal[2] - yCrystal[1]);
					const double IS_1_weno3_left = static_cast<double>(_weno5->IS_1_coeff_weno3) * (yCrystal[1] - yCrystal[0]) * (yCrystal[1] - yCrystal[0]);
					const double alpha_0_weno3_left = static_cast<double>(_weno5->C_coeff1_weno3) / (static_cast<double>(_binSizes[1]) + IS_0_weno3_left) / (static_cast<double>(_binSizes[1]) + IS_0_weno3_left);
					const double alpha_1_weno3_left = static_cast<double>(_weno5->C_coeff2_weno3) / (static_cast<double>(_binSizes[1]) + IS_1_weno3_left) / (static_cast<double>(_binSizes[1]) + IS_1_weno3_left);
					const double W_0_weno3_left = alpha_0_weno3_left / (alpha_0_weno3_left + alpha_1_weno3_left);
					const double W_1_weno3_left = 1.0 - W_0_weno3_left;
					const double q_0_weno3_left = static_cast<double>(_weno5->q0_coeff1_weno3) * yCrystal[1] + static_cast<double>(_weno5->q0_coeff2_weno3) * yCrystal[2];
					const double q_1_weno3_left = static_cast<double>(_weno5->q1_coeff1_weno3) * yCrystal[1] + static_cast<double>(_weno5->q1_coeff2_weno3) * yCrystal[0];

					// WENO23 Jacobian related coefficients, for bin 2
					const double four_w0_B0_weno3_left = 4.0 * W_0_weno3_left * static_cast<double>(_weno5->IS_0_coeff_weno3) * (yCrystal[2] - yCrystal[1]) / (static_cast<double>(_binSizes[1]) + IS_0_weno3_left);
					const double four_w1_B1_weno3_left = 4.0 * W_1_weno3_left * static_cast<double>(_weno5->IS_1_coeff_weno3) * (yCrystal[1] - yCrystal[0]) / (static_cast<double>(_binSizes[1]) + IS_1_weno3_left);
					const double four_w0_w0_B0_weno3_left = four_w0_B0_weno3_left * W_0_weno3_left;
					const double four_w0_w1_B1_weno3_left = four_w1_B1_weno3_left * W_0_weno3_left;
					const double four_w0_w1_B0_weno3_left = four_w0_B0_weno3_left * W_1_weno3_left;
					const double four_w1_w1_B1_weno3_left = four_w1_B1_weno3_left * W_1_weno3_left;

					const double dw0_right_dni_m1_weno3_left = -four_w0_w1_B1_weno3_left;
					const double dw0_right_dni_weno3_left = four_w0_B0_weno3_left - four_w0_w0_B0_weno3_left + four_w0_w1_B1_weno3_left;
					const double dw0_right_dni_p1_weno3_left = four_w0_w0_B0_weno3_left - four_w0_B0_weno3_left;

					const double dw1_right_dni_m1_weno3_left = four_w1_B1_weno3_left - four_w1_w1_B1_weno3_left;
					const double dw1_right_dni_weno3_left = -four_w1_B1_weno3_left - four_w0_w1_B0_weno3_left + four_w1_w1_B1_weno3_left;
					const double dw1_right_dni_p1_weno3_left = four_w0_w1_B0_weno3_left;

					const double dq0_right_dni_weno3_left = static_cast<double>(_weno5->q0_coeff1_weno3);
					const double dq0_right_dni_p1_weno3_left = static_cast<double>(_weno5->q0_coeff2_weno3);
					const double dq1_right_dni_m1_weno3_left = static_cast<double>(_weno5->q1_coeff2_weno3);
					const double dq1_right_dni_weno3_left = static_cast<double>(_weno5->q1_coeff1_weno3);

					// WENO35 related coefficients
					double IS_0 = 0.0;
					double IS_1 = 0.0;
					double IS_2 = 0.0;
					double alpha_0 = 0.0;
					double alpha_1 = 0.0;
					double alpha_2 = 0.0;
					double W_0 = 0.0;
					double W_1 = 0.0;
					double W_2 = 0.0;
					double q_0 = 0.0;
					double q_1 = 0.0;
					double q_2 = 0.0;

					double dvG_dc_right = 0.0;
					double dvG_dceq_right = 0.0;
					double vg_right = 0.0;

					// WENO35 Jacobian related coefficients
					double dIS0_wrt_dni_p2 = 0.0;
					double dIS0_wrt_dni_p1 = 0.0;
					double dIS0_wrt_dni = 0.0;
					double dIS1_wrt_dni_p1 = 0.0;
					double dIS1_wrt_dni = 0.0;
					double dIS1_wrt_dni_m1 = 0.0;
					double dIS2_wrt_dni = 0.0;
					double dIS2_wrt_dni_m1 = 0.0;
					double dIS2_wrt_dni_m2 = 0.0;
					double dq0_wrt_dni_p2 = 0.0;
					double dq0_wrt_dni_p1 = 0.0;
					double dq0_wrt_dni = 0.0;
					double dq1_wrt_dni_p1 = 0.0;
					double dq1_wrt_dni = 0.0;
					double dq1_wrt_dni_m1 = 0.0;
					double dq2_wrt_dni = 0.0;
					double dq2_wrt_dni_m1 = 0.0;
					double dq2_wrt_dni_m2 = 0.0;
					double w0_w2_is2 = 0.0;
					double w0_w1_is1 = 0.0;
					double w0_w0_is0 = 0.0;
					double w0_is0 = 0.0;
					double dw0_wrt_dni_m2 = 0.0;
					double dw0_wrt_dni_m1 = 0.0;
					double dw0_wrt_dni = 0.0;
					double dw0_wrt_dni_p1 = 0.0;
					double dw0_wrt_dni_p2 = 0.0;
					double w1_w2_is2 = 0.0;
					double w1_w1_is1 = 0.0;
					double w1_w0_is0 = 0.0;
					double w1_is1 = 0.0;
					double dw1_wrt_dni_m2 = 0.0;
					double dw1_wrt_dni_m1 = 0.0;
					double dw1_wrt_dni = 0.0;
					double dw1_wrt_dni_p1 = 0.0;
					double dw1_wrt_dni_p2 = 0.0;
					double w2_is2 = 0.0;
					double w2_w2_is2 = 0.0;
					double w2_w1_is1 = 0.0;
					double w2_w0_is0 = 0.0;
					double dw2_wrt_dni_m2 = 0.0;
					double dw2_wrt_dni_m1 = 0.0;
					double dw2_wrt_dni = 0.0;
					double dw2_wrt_dni_p1 = 0.0;
					double dw2_wrt_dni_p2 = 0.0;

					// note: binIdx_j is used only when B_s is involved.
					for (int i = 0; i < _nComp; ++i)
					{
						// Q_i, intermediate cells
						if (cadet_likely((i > 3) && (i + 3 < _nComp)))
						{
							binIdx_i = i - 1;
							// jacobian, left face, which is the right face of a previous cell
						    // dQ_i/dc
							jac[-i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2);

							// dQ_i/dn_{i-2}
							jac[-3] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_m2 * q_0 + dw1_wrt_dni_m2 * q_1 + dw2_wrt_dni_m2 * q_2 + W_2 * dq2_wrt_dni_m2); // the second and fourth terms are zero

							// dQ_i/dn_{i-1}
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_m1 * q_0 + dw1_wrt_dni_m1 * q_1 + W_1 * dq1_wrt_dni_m1 + dw2_wrt_dni_m1 * q_2 + W_2 * dq2_wrt_dni_m1); // the second term is zero

							// dQ_i/dn_{i}
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni * q_0 + W_0 * dq0_wrt_dni + dw1_wrt_dni * q_1 + W_1 * dq1_wrt_dni + dw2_wrt_dni * q_2 + W_2 * dq2_wrt_dni); // full six terms of the product

							// dQ_i/dn_{i+1}
							jac[0] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_p1 * q_0 + W_0 * dq0_wrt_dni_p1 + dw1_wrt_dni_p1 * q_1 + W_1 * dq1_wrt_dni_p1 + dw2_wrt_dni_p1 * q_2); // the last term is zero

							// dQ_i/dn_{i+2}
							jac[1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_p2 * q_0 + W_0 * dq0_wrt_dni_p2 + dw1_wrt_dni_p2 * q_1 + dw2_wrt_dni_p2 * q_2); // the fourth and sixth terms are zero

							// dQ_2/dc_eq
							jac[_nComp - 1 - i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2);

							// update all coefficients
							// WENO5, right face
							IS_0 = static_cast<double>(_weno5->IS_0_coeff_1[binIdx_i]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) + static_cast<double>(_weno5->IS_0_coeff_2[binIdx_i]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]) + static_cast<double>(_weno5->IS_0_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]);
							IS_1 = static_cast<double>(_weno5->IS_1_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) + static_cast<double>(_weno5->IS_1_coeff_2[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) + static_cast<double>(_weno5->IS_1_coeff_3[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);
							IS_2 = static_cast<double>(_weno5->IS_2_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + static_cast<double>(_weno5->IS_2_coeff_2[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + static_cast<double>(_weno5->IS_2_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);
							alpha_0 = static_cast<double>(_weno5->C_0[binIdx_i]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_0) / (static_cast<double>(_binSizes[binIdx_i]) + IS_0);
							alpha_1 = static_cast<double>(_weno5->C_1[binIdx_i]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_1) / (static_cast<double>(_binSizes[binIdx_i]) + IS_1);
							alpha_2 = static_cast<double>(_weno5->C_2[binIdx_i]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_2) / (static_cast<double>(_binSizes[binIdx_i]) + IS_2);
							W_0 = alpha_0 / (alpha_0 + alpha_1 + alpha_2);
							W_1 = alpha_1 / (alpha_0 + alpha_1 + alpha_2);
							W_2 = 1.0 - W_0 - W_1;
							q_0 = yCrystal[binIdx_i + 1] + static_cast<double>(_weno5->q_0_coeff_1[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]) + static_cast<double>(_weno5->q_0_coeff_2[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i + 2]);
							q_1 = yCrystal[binIdx_i] + static_cast<double>(_weno5->q_1_coeff_1[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]) + static_cast<double>(_weno5->q_1_coeff_2[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);
							q_2 = yCrystal[binIdx_i - 1] + static_cast<double>(_weno5->q_2_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + static_cast<double>(_weno5->q_2_coeff_2[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);

							dvG_dc_right = dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							dvG_dceq_right = dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							vg_right = vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							// jacobian
							dIS0_wrt_dni_p2 = 2.0 * static_cast<double>(_weno5->IS_0_coeff_1[binIdx_i]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) + static_cast<double>(_weno5->IS_0_coeff_2[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]);
							dIS0_wrt_dni_p1 = -2.0 * static_cast<double>(_weno5->IS_0_coeff_1[binIdx_i]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) + static_cast<double>(_weno5->IS_0_coeff_2[binIdx_i]) * (2.0 * yCrystal[binIdx_i + 1] - yCrystal[binIdx_i + 2] - yCrystal[binIdx_i]) - 2.0 * static_cast<double>(_weno5->IS_0_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]);
							dIS0_wrt_dni = static_cast<double>(_weno5->IS_0_coeff_2[binIdx_i]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) + 2.0 * static_cast<double>(_weno5->IS_0_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]);

							dIS1_wrt_dni_p1 = static_cast<double>(_weno5->IS_1_coeff_2[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) + 2.0 * static_cast<double>(_weno5->IS_1_coeff_3[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);
							dIS1_wrt_dni = -2.0 * static_cast<double>(_weno5->IS_1_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) + static_cast<double>(_weno5->IS_1_coeff_2[binIdx_i]) * (2.0 * yCrystal[binIdx_i] - yCrystal[binIdx_i + 1] - yCrystal[binIdx_i - 1]) - 2.0 * static_cast<double>(_weno5->IS_1_coeff_3[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);
							dIS1_wrt_dni_m1 = 2.0 * static_cast<double>(_weno5->IS_1_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) + static_cast<double>(_weno5->IS_1_coeff_2[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);

							dIS2_wrt_dni = static_cast<double>(_weno5->IS_2_coeff_2[binIdx_i]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + 2.0 * static_cast<double>(_weno5->IS_2_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);
							dIS2_wrt_dni_m1 = -2.0 * static_cast<double>(_weno5->IS_2_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + static_cast<double>(_weno5->IS_2_coeff_2[binIdx_i]) * (2.0 * yCrystal[binIdx_i - 1] - yCrystal[binIdx_i] - yCrystal[binIdx_i - 2]) - 2.0 * static_cast<double>(_weno5->IS_2_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);
							dIS2_wrt_dni_m2 = 2.0 * static_cast<double>(_weno5->IS_2_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + static_cast<double>(_weno5->IS_2_coeff_2[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);

							dq0_wrt_dni_p2 = -static_cast<double>(_weno5->q_0_coeff_2[binIdx_i]);
							dq0_wrt_dni_p1 = static_cast<double>(_weno5->q_0_coeff_2[binIdx_i]) - static_cast<double>(_weno5->q_0_coeff_1[binIdx_i]) + 1.0;
							dq0_wrt_dni = static_cast<double>(_weno5->q_0_coeff_1[binIdx_i]);

							dq1_wrt_dni_p1 = static_cast<double>(_weno5->q_1_coeff_1[binIdx_i]);
							dq1_wrt_dni = -static_cast<double>(_weno5->q_1_coeff_1[binIdx_i]) + static_cast<double>(_weno5->q_1_coeff_2[binIdx_i]) + 1.0;
							dq1_wrt_dni_m1 = -static_cast<double>(_weno5->q_1_coeff_2[binIdx_i]);

							dq2_wrt_dni = static_cast<double>(_weno5->q_2_coeff_2[binIdx_i]);
							dq2_wrt_dni_m1 = -static_cast<double>(_weno5->q_2_coeff_1[binIdx_i]) - static_cast<double>(_weno5->q_2_coeff_2[binIdx_i]) + 1.0;
							dq2_wrt_dni_m2 = static_cast<double>(_weno5->q_2_coeff_1[binIdx_i]);

							// dw_0/dx
							w0_w2_is2 = 2.0 * W_0 * W_2 / (static_cast<double>(_binSizes[binIdx_i]) + IS_2);
							w0_w1_is1 = 2.0 * W_0 * W_1 / (static_cast<double>(_binSizes[binIdx_i]) + IS_1);
							w0_w0_is0 = 2.0 * W_0 * W_0 / (static_cast<double>(_binSizes[binIdx_i]) + IS_0);
							w0_is0 = -2.0 * W_0 / (static_cast<double>(_binSizes[binIdx_i]) + IS_0);

							dw0_wrt_dni_m2 = w0_w2_is2 * dIS2_wrt_dni_m2;
							dw0_wrt_dni_m1 = w0_w1_is1 * dIS1_wrt_dni_m1 + w0_w2_is2 * dIS2_wrt_dni_m1;
							dw0_wrt_dni = w0_is0 * dIS0_wrt_dni + w0_w0_is0 * dIS0_wrt_dni + w0_w1_is1 * dIS1_wrt_dni + w0_w2_is2 * dIS2_wrt_dni;
							dw0_wrt_dni_p1 = w0_is0 * dIS0_wrt_dni_p1 + w0_w0_is0 * dIS0_wrt_dni_p1 + w0_w1_is1 * dIS1_wrt_dni_p1;
							dw0_wrt_dni_p2 = w0_is0 * dIS0_wrt_dni_p2 + w0_w0_is0 * dIS0_wrt_dni_p2;

							// dw_1/dx
							w1_w2_is2 = 2.0 * W_1 * W_2 / (static_cast<double>(_binSizes[binIdx_i]) + IS_2);
							w1_w1_is1 = 2.0 * W_1 * W_1 / (static_cast<double>(_binSizes[binIdx_i]) + IS_1);
							w1_w0_is0 = 2.0 * W_1 * W_0 / (static_cast<double>(_binSizes[binIdx_i]) + IS_0);
							w1_is1 = -2.0 * W_1 / (static_cast<double>(_binSizes[binIdx_i]) + IS_1);

							dw1_wrt_dni_m2 = w1_w2_is2 * dIS2_wrt_dni_m2;
							dw1_wrt_dni_m1 = w1_is1 * dIS1_wrt_dni_m1 + w1_w1_is1 * dIS1_wrt_dni_m1 + w1_w2_is2 * dIS2_wrt_dni_m1;
							dw1_wrt_dni = w1_is1 * dIS1_wrt_dni + w1_w0_is0 * dIS0_wrt_dni + w1_w1_is1 * dIS1_wrt_dni + w1_w2_is2 * dIS2_wrt_dni;
							dw1_wrt_dni_p1 = w1_is1 * dIS1_wrt_dni_p1 + w1_w0_is0 * dIS0_wrt_dni_p1 + w1_w1_is1 * dIS1_wrt_dni_p1;
							dw1_wrt_dni_p2 = w1_w0_is0 * dIS0_wrt_dni_p2;

							// dw2_dx
							w2_is2 = -2.0 * W_2 / (static_cast<double>(_binSizes[binIdx_i]) + IS_2);
							w2_w2_is2 = 2.0 * W_2 * W_2 / (static_cast<double>(_binSizes[binIdx_i]) + IS_2);
							w2_w1_is1 = 2.0 * W_2 * W_1 / (static_cast<double>(_binSizes[binIdx_i]) + IS_1);
							w2_w0_is0 = 2.0 * W_2 * W_0 / (static_cast<double>(_binSizes[binIdx_i]) + IS_0);

							dw2_wrt_dni_m2 = w2_is2 * dIS2_wrt_dni_m2 + w2_w2_is2 * dIS2_wrt_dni_m2;
							dw2_wrt_dni_m1 = w2_is2 * dIS2_wrt_dni_m1 + w2_w1_is1 * dIS1_wrt_dni_m1 + w2_w2_is2 * dIS2_wrt_dni_m1;
							dw2_wrt_dni = w2_is2 * dIS2_wrt_dni + w2_w0_is0 * dIS0_wrt_dni + w2_w1_is1 * dIS1_wrt_dni + w2_w2_is2 * dIS2_wrt_dni;
							dw2_wrt_dni_p1 = w2_w0_is0 * dIS0_wrt_dni_p1 + w2_w1_is1 * dIS1_wrt_dni_p1;
							dw2_wrt_dni_p2 = w2_w0_is0 * dIS0_wrt_dni_p2;

							// jacobian, right face, and diffusion
							// dQ_i/dc
							jac[-i] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2);

							// dQ_i/dn_{i-2}
							jac[-2] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_m2 * q_0 + dw1_wrt_dni_m2 * q_1 + dw2_wrt_dni_m2 * q_2 + W_2 * dq2_wrt_dni_m2); // the second and fourth terms are zero

							// dQ_i/dn_{i-1}
							jac[-1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_m1 * q_0 + dw1_wrt_dni_m1 * q_1 + W_1 * dq1_wrt_dni_m1 + dw2_wrt_dni_m1 * q_2 + W_2 * dq2_wrt_dni_m1) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // the second term is zero

							// dQ_i/dn_{i}
							jac[0] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni * q_0 + W_0 * dq0_wrt_dni + dw1_wrt_dni * q_1 + W_1 * dq1_wrt_dni + dw2_wrt_dni * q_2 + W_2 * dq2_wrt_dni) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binSizes[binIdx_i]) * (1.0 / static_cast<double>(_binCenterDists[binIdx_i]) + 1.0 / static_cast<double>(_binCenterDists[binIdx_i - 1])); // full six terms of the product

							// dQ_i/dn_{i+1}
							jac[1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_p1 * q_0 + W_0 * dq0_wrt_dni_p1 + dw1_wrt_dni_p1 * q_1 + W_1 * dq1_wrt_dni_p1 + dw2_wrt_dni_p1 * q_2) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]); // the last term is zero

							// dQ_i/dn_{i+2}
							jac[2] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_p2 * q_0 + W_0 * dq0_wrt_dni_p2 + dw1_wrt_dni_p2 * q_1 + dw2_wrt_dni_p2 * q_2); // the fourth and sixth terms are zero

							// dQ_2/dc_eq
							jac[_nComp - 1 - i] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2);

						}

						// Q_c, mass balance, independent on the scheme. Do not change.
						else if (cadet_unlikely(i == 0))
						{
							for (int j = 0; j < _nComp; ++j)
							{
								binIdx_j = j - 1;
								if (cadet_likely((j > 0) && (j + 1 < _nComp)))
								{
									// dQ_c/dn_i
									jac[j - i] -= factor * rho_kv * (x_c_3 * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) + 3.0 * vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_binCenters[binIdx_j]), static_cast<double>(_p))) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]));
								}
								else if (cadet_unlikely(j == 0))
								{
									// dQ_c/dc
									jac[j - i] -= factor * rho_kv * ((dBp_dc + dBs_dc) * x_c_3 + 3.0 * dvG_dc_factor * substrateConversion);
								}
								else
								{
									// dQ_c/dc_eq
									jac[j - i] -= factor * rho_kv * ((dBp_dceq + dBs_dceq) * x_c_3 + 3.0 * dvG_dceq_factor * substrateConversion);
								}
							}
						}

						// Q_0, left BC, also independent on the scheme. Do not change.
						else if (cadet_unlikely(i == 1))
						{
							binIdx_i = i - 1;
							// Q_0, left BC
							for (int j = 0; j < _nComp; ++j)
							{
								binIdx_j = j - 1;
								if (cadet_likely((j > 2) && (j + 1 < _nComp)))
								{
									// dQ_0/dni
									jac[j - i] += factor * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 0))
								{
									// dQ_0/dc
									jac[j - i] -= factor * (yCrystal[0] * dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) - dBp_dc - dBs_dc) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 1))
								{
									// dQ_0/dn0
									jac[j - i] -= factor * (vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_j + 1]), static_cast<double>(_p))) - dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j])) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else if (cadet_unlikely(j == 2))
								{
									// dQ_0/dn1
									jac[j - i] += factor * dBs_dni_factor * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binCenters[binIdx_j]) * static_cast<double>(_binSizes[binIdx_j]) / static_cast<double>(_binSizes[binIdx_i]) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]);
								}
								else
								{
									// dQ_0/dceq
									jac[j - i] -= factor * (yCrystal[0] * dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p))) - dBp_dceq - dBs_dceq) / static_cast<double>(_binSizes[binIdx_i]);
								}
							}
						}

						// Q_1, special case for weno35.
						else if (cadet_unlikely(i == 2))
						{
							binIdx_i = i - 1;
							dvG_dc_right = dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							dvG_dceq_right = dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							vg_right = vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							// dQ_1/dc, j=0
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) * yCrystal[0]; // left face, upwind scheme
							jac[-2] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0_weno3_left * q_0_weno3_left + W_1_weno3_left * q_1_weno3_left); // right face, WENO3

							// dQ_1/dn0, j=1
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // coming from the upwind scheme
							jac[-1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_m1_weno3_left * q_0_weno3_left + dw1_right_dni_m1_weno3_left * q_1_weno3_left + W_1_weno3_left * dq1_right_dni_m1_weno3_left); // the second term in the product rule is zero

							// dQ_1/dn1, j=2
							jac[0] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_weno3_left * q_0_weno3_left + W_0_weno3_left * dq0_right_dni_weno3_left + dw1_right_dni_weno3_left * q_1_weno3_left + W_1_weno3_left * dq1_right_dni_weno3_left) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binSizes[binIdx_i]) * (1.0 / static_cast<double>(_binCenterDists[binIdx_i]) + 1.0 / static_cast<double>(_binCenterDists[binIdx_i - 1])); // full four terms in the product rule

							// dQ_1/dn2, j=3
							jac[1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_p1_weno3_left * q_0_weno3_left + W_0_weno3_left * dq0_right_dni_p1_weno3_left + dw1_right_dni_p1_weno3_left * q_1_weno3_left) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]); // the fourth term in the product rule is zero

							// dQ_1/dceq, j=_nComp -1
							jac[_nComp - 3] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i]), static_cast<double>(_p))) * yCrystal[0];
							jac[_nComp - 3] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0_weno3_left * q_0_weno3_left + W_1_weno3_left * q_1_weno3_left);
						}

						// Q_2, special case for weno35.
						else if (cadet_unlikely(i == 3))
						{
							binIdx_i = i - 1;
							// left face uses WENO3
							// dQ_2/dc
							jac[-3] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0_weno3_left * q_0_weno3_left + W_1_weno3_left * q_1_weno3_left);

							// dQ_2/dn0
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_m1_weno3_left * q_0_weno3_left + dw1_right_dni_m1_weno3_left * q_1_weno3_left + W_1_weno3_left * dq1_right_dni_m1_weno3_left); // the second term in the product rule is zero
							
							// dQ_2/dn1
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_weno3_left * q_0_weno3_left + W_0_weno3_left * dq0_right_dni_weno3_left + dw1_right_dni_weno3_left * q_1_weno3_left + W_1_weno3_left * dq1_right_dni_weno3_left); // full four terms in the product rule
							
							// dQ_2/dn2
							jac[0] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_p1_weno3_left * q_0_weno3_left + W_0_weno3_left * dq0_right_dni_p1_weno3_left + dw1_right_dni_p1_weno3_left * q_1_weno3_left); // the fourth term in the product rule is zero

							// dQ_2/dc_eq
							jac[_nComp - 4] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0_weno3_left * q_0_weno3_left + W_1_weno3_left * q_1_weno3_left);

							// WENO5, right face
							IS_0 = static_cast<double>(_weno5->IS_0_coeff_1[binIdx_i]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) + static_cast<double>(_weno5->IS_0_coeff_2[binIdx_i]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]) + static_cast<double>(_weno5->IS_0_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]);
							IS_1 = static_cast<double>(_weno5->IS_1_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) + static_cast<double>(_weno5->IS_1_coeff_2[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) + static_cast<double>(_weno5->IS_1_coeff_3[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);
							IS_2 = static_cast<double>(_weno5->IS_2_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + static_cast<double>(_weno5->IS_2_coeff_2[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + static_cast<double>(_weno5->IS_2_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);
							alpha_0 = static_cast<double>(_weno5->C_0[binIdx_i]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_0) / (static_cast<double>(_binSizes[binIdx_i]) + IS_0);
							alpha_1 = static_cast<double>(_weno5->C_1[binIdx_i]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_1) / (static_cast<double>(_binSizes[binIdx_i]) + IS_1);
							alpha_2 = static_cast<double>(_weno5->C_2[binIdx_i]) / (static_cast<double>(_binSizes[binIdx_i]) + IS_2) / (static_cast<double>(_binSizes[binIdx_i]) + IS_2);
							W_0 = alpha_0 / (alpha_0 + alpha_1 + alpha_2);
							W_1 = alpha_1 / (alpha_0 + alpha_1 + alpha_2);
							W_2 = 1.0 - W_0 - W_1;
							q_0 = yCrystal[binIdx_i + 1] + static_cast<double>(_weno5->q_0_coeff_1[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]) + static_cast<double>(_weno5->q_0_coeff_2[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i + 2]);
							q_1 = yCrystal[binIdx_i] + static_cast<double>(_weno5->q_1_coeff_1[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]) + static_cast<double>(_weno5->q_1_coeff_2[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);
							q_2 = yCrystal[binIdx_i - 1] + static_cast<double>(_weno5->q_2_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + static_cast<double>(_weno5->q_2_coeff_2[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);

							dvG_dc_right = dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							dvG_dceq_right = dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							vg_right = vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

							// jacobian
							dIS0_wrt_dni_p2 = 2.0 * static_cast<double>(_weno5->IS_0_coeff_1[binIdx_i]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) + static_cast<double>(_weno5->IS_0_coeff_2[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]);
							dIS0_wrt_dni_p1 = -2.0 * static_cast<double>(_weno5->IS_0_coeff_1[binIdx_i]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) + static_cast<double>(_weno5->IS_0_coeff_2[binIdx_i]) * (2.0 * yCrystal[binIdx_i + 1] - yCrystal[binIdx_i + 2] - yCrystal[binIdx_i]) - 2.0 * static_cast<double>(_weno5->IS_0_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]);
							dIS0_wrt_dni = static_cast<double>(_weno5->IS_0_coeff_2[binIdx_i]) * (yCrystal[binIdx_i + 2] - yCrystal[binIdx_i + 1]) + 2.0 * static_cast<double>(_weno5->IS_0_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i + 1]);

							dIS1_wrt_dni_p1 = static_cast<double>(_weno5->IS_1_coeff_2[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) + 2.0 * static_cast<double>(_weno5->IS_1_coeff_3[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);
							dIS1_wrt_dni = -2.0 * static_cast<double>(_weno5->IS_1_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) + static_cast<double>(_weno5->IS_1_coeff_2[binIdx_i]) * (2.0 * yCrystal[binIdx_i] - yCrystal[binIdx_i + 1] - yCrystal[binIdx_i - 1]) - 2.0 * static_cast<double>(_weno5->IS_1_coeff_3[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);
							dIS1_wrt_dni_m1 = 2.0 * static_cast<double>(_weno5->IS_1_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 1] - yCrystal[binIdx_i]) + static_cast<double>(_weno5->IS_1_coeff_2[binIdx_i]) * (yCrystal[binIdx_i + 1] - yCrystal[binIdx_i]);

							dIS2_wrt_dni = static_cast<double>(_weno5->IS_2_coeff_2[binIdx_i]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + 2.0 * static_cast<double>(_weno5->IS_2_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);
							dIS2_wrt_dni_m1 = -2.0 * static_cast<double>(_weno5->IS_2_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + static_cast<double>(_weno5->IS_2_coeff_2[binIdx_i]) * (2.0 * yCrystal[binIdx_i - 1] - yCrystal[binIdx_i] - yCrystal[binIdx_i - 2]) - 2.0 * static_cast<double>(_weno5->IS_2_coeff_3[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);
							dIS2_wrt_dni_m2 = 2.0 * static_cast<double>(_weno5->IS_2_coeff_1[binIdx_i]) * (yCrystal[binIdx_i - 2] - yCrystal[binIdx_i - 1]) + static_cast<double>(_weno5->IS_2_coeff_2[binIdx_i]) * (yCrystal[binIdx_i] - yCrystal[binIdx_i - 1]);

							dq0_wrt_dni_p2 = -static_cast<double>(_weno5->q_0_coeff_2[binIdx_i]);
							dq0_wrt_dni_p1 = static_cast<double>(_weno5->q_0_coeff_2[binIdx_i]) - static_cast<double>(_weno5->q_0_coeff_1[binIdx_i]) + 1.0;
							dq0_wrt_dni = static_cast<double>(_weno5->q_0_coeff_1[binIdx_i]);

							dq1_wrt_dni_p1 = static_cast<double>(_weno5->q_1_coeff_1[binIdx_i]);
							dq1_wrt_dni = -static_cast<double>(_weno5->q_1_coeff_1[binIdx_i]) + static_cast<double>(_weno5->q_1_coeff_2[binIdx_i]) + 1.0;
							dq1_wrt_dni_m1 = -static_cast<double>(_weno5->q_1_coeff_2[binIdx_i]);

							dq2_wrt_dni = static_cast<double>(_weno5->q_2_coeff_2[binIdx_i]);
							dq2_wrt_dni_m1 = -static_cast<double>(_weno5->q_2_coeff_1[binIdx_i]) - static_cast<double>(_weno5->q_2_coeff_2[binIdx_i]) + 1.0;
							dq2_wrt_dni_m2 = static_cast<double>(_weno5->q_2_coeff_1[binIdx_i]);

							// dw_0/dx
							w0_w2_is2 = 2.0 * W_0 * W_2 / (static_cast<double>(_binSizes[binIdx_i]) + IS_2);
							w0_w1_is1 = 2.0 * W_0 * W_1 / (static_cast<double>(_binSizes[binIdx_i]) + IS_1);
							w0_w0_is0 = 2.0 * W_0 * W_0 / (static_cast<double>(_binSizes[binIdx_i]) + IS_0);
							w0_is0 = -2.0 * W_0 / (static_cast<double>(_binSizes[binIdx_i]) + IS_0);

							dw0_wrt_dni_m2 = w0_w2_is2 * dIS2_wrt_dni_m2;
							dw0_wrt_dni_m1 = w0_w1_is1 * dIS1_wrt_dni_m1 + w0_w2_is2 * dIS2_wrt_dni_m1;
							dw0_wrt_dni = w0_is0 * dIS0_wrt_dni + w0_w0_is0 * dIS0_wrt_dni + w0_w1_is1 * dIS1_wrt_dni + w0_w2_is2 * dIS2_wrt_dni;
							dw0_wrt_dni_p1 = w0_is0 * dIS0_wrt_dni_p1 + w0_w0_is0 * dIS0_wrt_dni_p1 + w0_w1_is1 * dIS1_wrt_dni_p1;
							dw0_wrt_dni_p2 = w0_is0 * dIS0_wrt_dni_p2 + w0_w0_is0 * dIS0_wrt_dni_p2;

							// dw_1/dx
							w1_w2_is2 = 2.0 * W_1 * W_2 / (static_cast<double>(_binSizes[binIdx_i]) + IS_2);
							w1_w1_is1 = 2.0 * W_1 * W_1 / (static_cast<double>(_binSizes[binIdx_i]) + IS_1);
							w1_w0_is0 = 2.0 * W_1 * W_0 / (static_cast<double>(_binSizes[binIdx_i]) + IS_0);
							w1_is1 = -2.0 * W_1 / (static_cast<double>(_binSizes[binIdx_i]) + IS_1);

							dw1_wrt_dni_m2 = w1_w2_is2 * dIS2_wrt_dni_m2;
							dw1_wrt_dni_m1 = w1_is1 * dIS1_wrt_dni_m1 + w1_w1_is1 * dIS1_wrt_dni_m1 + w1_w2_is2 * dIS2_wrt_dni_m1;
							dw1_wrt_dni = w1_is1 * dIS1_wrt_dni + w1_w0_is0 * dIS0_wrt_dni + w1_w1_is1 * dIS1_wrt_dni + w1_w2_is2 * dIS2_wrt_dni;
							dw1_wrt_dni_p1 = w1_is1 * dIS1_wrt_dni_p1 + w1_w0_is0 * dIS0_wrt_dni_p1 + w1_w1_is1 * dIS1_wrt_dni_p1;
							dw1_wrt_dni_p2 = w1_w0_is0 * dIS0_wrt_dni_p2;

							// dw2_dx
							w2_is2 = -2.0 * W_2 / (static_cast<double>(_binSizes[binIdx_i]) + IS_2);
							w2_w2_is2 = 2.0 * W_2 * W_2 / (static_cast<double>(_binSizes[binIdx_i]) + IS_2);
							w2_w1_is1 = 2.0 * W_2 * W_1 / (static_cast<double>(_binSizes[binIdx_i]) + IS_1);
							w2_w0_is0 = 2.0 * W_2 * W_0 / (static_cast<double>(_binSizes[binIdx_i]) + IS_0);

							dw2_wrt_dni_m2 = w2_is2 * dIS2_wrt_dni_m2 + w2_w2_is2 * dIS2_wrt_dni_m2;
							dw2_wrt_dni_m1 = w2_is2 * dIS2_wrt_dni_m1 + w2_w1_is1 * dIS1_wrt_dni_m1 + w2_w2_is2 * dIS2_wrt_dni_m1;
							dw2_wrt_dni = w2_is2 * dIS2_wrt_dni + w2_w0_is0 * dIS0_wrt_dni + w2_w1_is1 * dIS1_wrt_dni + w2_w2_is2 * dIS2_wrt_dni;
							dw2_wrt_dni_p1 = w2_w0_is0 * dIS0_wrt_dni_p1 + w2_w1_is1 * dIS1_wrt_dni_p1;
							dw2_wrt_dni_p2 = w2_w0_is0 * dIS0_wrt_dni_p2;

							// dQ_2/dc
							jac[-3] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2);

							// dQ_2/dn0
							jac[-2] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_m2 * q_0 + dw1_wrt_dni_m2 * q_1 + dw2_wrt_dni_m2 * q_2 + W_2 * dq2_wrt_dni_m2); // the second and fourth terms are zero

							// dQ_2/dn1
							jac[-1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_m1 * q_0 + dw1_wrt_dni_m1 * q_1 + W_1 * dq1_wrt_dni_m1 + dw2_wrt_dni_m1 * q_2 + W_2 * dq2_wrt_dni_m1) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // the second term is zero

							// dQ_2/dn2
							jac[0] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni * q_0 + W_0 * dq0_wrt_dni + dw1_wrt_dni * q_1 + W_1 * dq1_wrt_dni + dw2_wrt_dni * q_2 + W_2 * dq2_wrt_dni) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binSizes[binIdx_i]) * (1.0 / static_cast<double>(_binCenterDists[binIdx_i]) + 1.0 / static_cast<double>(_binCenterDists[binIdx_i - 1])); // full six terms of the product

							// dQ_2/dn3
							jac[1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_p1 * q_0 + W_0 * dq0_wrt_dni_p1 + dw1_wrt_dni_p1 * q_1 + W_1 * dq1_wrt_dni_p1 + dw2_wrt_dni_p1 * q_2) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]); // the last term is zero

							// dQ_2/dn4
							jac[2] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_p2 * q_0 + W_0 * dq0_wrt_dni_p2 + dw1_wrt_dni_p2 * q_1 + dw2_wrt_dni_p2 * q_2); // the fourth and sixth terms are zero

							// dQ_2/dc_eq
							jac[_nComp - 4] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2);

						}

						// Q_{N_x-2}, special case for WENO35
						else if (cadet_unlikely(i == _nComp - 3))
						{
							binIdx_i = i - 1;
							// left face is weno5
							// dQ_{N_x-2}/dc
							jac[-i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2);

							// dQ_{N_x-1}/dn_{N_x-4}
							jac[-3] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_m2 * q_0 + dw1_wrt_dni_m2 * q_1 + dw2_wrt_dni_m2 * q_2 + W_2 * dq2_wrt_dni_m2); // the second and fourth terms are zero

							// dQ_{N_x-1}/dn_{N_x-3}
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_m1 * q_0 + dw1_wrt_dni_m1 * q_1 + W_1 * dq1_wrt_dni_m1 + dw2_wrt_dni_m1 * q_2 + W_2 * dq2_wrt_dni_m1); // the second term is zero

							// dQ_{N_x-1}/dn_{N_x-2}
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni * q_0 + W_0 * dq0_wrt_dni + dw1_wrt_dni * q_1 + W_1 * dq1_wrt_dni + dw2_wrt_dni * q_2 + W_2 * dq2_wrt_dni); // full six terms of the product

							// dQ_{N_x-1}/dn_{N_x-1}
							jac[0] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_p1 * q_0 + W_0 * dq0_wrt_dni_p1 + dw1_wrt_dni_p1 * q_1 + W_1 * dq1_wrt_dni_p1 + dw2_wrt_dni_p1 * q_2); // the last term is zero

							// dQ_{N_x-1}/dn_{N_x}
							jac[1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_wrt_dni_p2 * q_0 + W_0 * dq0_wrt_dni_p2 + dw1_wrt_dni_p2 * q_1 + dw2_wrt_dni_p2 * q_2); // the fourth and sixth terms are zero

							// dQ_2/dc_eq
							jac[_nComp - 1 - i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0 * q_0 + W_1 * q_1 + W_2 * q_2);

							// right face is weno3
							dvG_dc_right = dvG_dc_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							dvG_dceq_right = dvG_dceq_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));
							vg_right = vG_factor * (static_cast<double>(_a) + static_cast<double>(_growthConstant) * pow(static_cast<double>(_bins[binIdx_i + 1]), static_cast<double>(_p)));

                            // dQ_{N_x-2}/dc
							jac[-i] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0_weno3_right * q_0_weno3_right + W_1_weno3_right * q_1_weno3_right);

							// dQ_{N_x-2}/dn_{N_x-3}
							jac[-1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_m1_weno3_right * q_0_weno3_right + dw1_right_dni_m1_weno3_right * q_1_weno3_right + W_1_weno3_right * dq1_right_dni_m1_weno3_right) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // the second term in the product rule is zero

							// dQ_{N_x-2}/dn_{N_x-2}
							jac[0] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_weno3_right * q_0_weno3_right + W_0_weno3_right * dq0_right_dni_weno3_right + dw1_right_dni_weno3_right * q_1_weno3_right + W_1_weno3_right * dq1_right_dni_weno3_right) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binSizes[binIdx_i]) * (1.0 / static_cast<double>(_binCenterDists[binIdx_i]) + 1.0 / static_cast<double>(_binCenterDists[binIdx_i - 1])); // full four terms in the product rule

							// dQ_{N_x-2}/dn_{N_x-1}
							jac[1] -= factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_p1_weno3_right * q_0_weno3_right + W_0_weno3_right * dq0_right_dni_p1_weno3_right + dw1_right_dni_p1_weno3_right * q_1_weno3_right) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i]) / static_cast<double>(_binSizes[binIdx_i]); // the fourth term in the product rule is zero

							// dQ_{N_x-2}/dc_eq
							jac[_nComp - 1 - i] -= factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0_weno3_right * q_0_weno3_right + W_1_weno3_right * q_1_weno3_right);
						}

						// Q_{N_x-1}, right BC
						else if (cadet_unlikely(i == _nComp - 2))
						{
							binIdx_i = i - 1;
							// left face is weno3
							// dQ_{N_x-1}/dc
							jac[-i] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dc_right * (W_0_weno3_right * q_0_weno3_right + W_1_weno3_right * q_1_weno3_right);

							// dQ_{N_x-1}/dn_{N_x-3}
							jac[-2] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_m1_weno3_right * q_0_weno3_right + dw1_right_dni_m1_weno3_right * q_1_weno3_right + W_1_weno3_right * dq1_right_dni_m1_weno3_right); // the second term in the product rule is zero

							// dQ_{N_x-1}/dn_{N_x-2}
							jac[-1] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_weno3_right * q_0_weno3_right + W_0_weno3_right * dq0_right_dni_weno3_right + dw1_right_dni_weno3_right * q_1_weno3_right + W_1_weno3_right * dq1_right_dni_weno3_right) + factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // full four terms in the product rule

							// dQ_{N_x-1}/dn_{N_x-1}
							jac[0] += factor / static_cast<double>(_binSizes[binIdx_i]) * vg_right * (dw0_right_dni_p1_weno3_right * q_0_weno3_right + W_0_weno3_right * dq0_right_dni_p1_weno3_right + dw1_right_dni_p1_weno3_right * q_1_weno3_right) - factor * static_cast<double>(_growthDispersionRate) / static_cast<double>(_binCenterDists[binIdx_i - 1]) / static_cast<double>(_binSizes[binIdx_i]); // the fourth term in the product rule is zero

							// dQ_i/dc_{eq}
							jac[1] += factor / static_cast<double>(_binSizes[binIdx_i]) * dvG_dceq_right * (W_0_weno3_right * q_0_weno3_right + W_1_weno3_right * q_1_weno3_right);
						}

						// Q_{ceq}, the last row is zero, which is the default. Do not change.
						else
						{
							continue;
						}

						// go to the next row
						++jac;
					}
				}
				break;
				}
			}

			template <typename RowIteratorLiquid, typename RowIteratorSolid>
			void jacobianCombinedImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid, double const* ySolid, double factor, RowIteratorLiquid& jacLiquid, const RowIteratorSolid& jacSolid, LinearBufferAllocator workSpace) const
			{
				jacobianLiquidImpl<RowIteratorLiquid>(t, secIdx, colPos, yLiquid, factor, jacLiquid, workSpace);
			}
		};

		namespace reaction
		{
			void registerCrystallizationReaction(std::unordered_map<std::string, std::function<model::IDynamicReactionModel* ()>>& reactions)
			{
				reactions[CrystallizationReaction::identifier()] = []() { return new CrystallizationReaction(); };
			}
		}  // namespace reaction

	}  // namespace model

}  // namespace cadet
