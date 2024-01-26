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
#include "model/ExternalFunctionSupport.hpp"
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
#include <vector>

/*<codegen>
{
	"name": "MassActionLawParamHandler",
	"externalName": "ExtMassActionLawParamHandler",
	"parameters":
		[
			{ "type": "ScalarReactionDependentParameter", "varName": "kFwdBulk", "confName": "MAL_KFWD_BULK"},
			{ "type": "ScalarReactionDependentParameter", "varName": "kBwdBulk", "confName": "MAL_KBWD_BULK"},
			{ "type": "ScalarReactionDependentParameter", "varName": "kFwdLiquid", "confName": "MAL_KFWD_LIQUID"},
			{ "type": "ScalarReactionDependentParameter", "varName": "kBwdLiquid", "confName": "MAL_KBWD_LIQUID"},
			{ "type": "ScalarReactionDependentParameter", "varName": "kFwdSolid", "confName": "MAL_KFWD_SOLID"},
			{ "type": "ScalarReactionDependentParameter", "varName": "kBwdSolid", "confName": "MAL_KBWD_SOLID"}
		]
}
</codegen>*/

/* Parameter description
 ------------------------
 kFwdBulk = Forward rate for reactions in bulk volume
 kBwdBulk = Backward rate for reactions in bulk volume
 kFwdLiquid = Forward rate for reactions in particle liquid phase
 kBwdLiquid = Backward rate for reactions in particle liquid phase
 kFwdSolid = Forward rate for reactions in particle solid phase
 kBwdSolid = Backward rate for reactions in particle solid phase
*/

namespace cadet
{

namespace model
{

inline const char* MassActionLawParamHandler::identifier() CADET_NOEXCEPT
{
	return "MASS_ACTION_LAW";
}

inline bool MassActionLawParamHandler::validateConfig(unsigned int nReactions, unsigned int nComp,
													  unsigned int const* nBoundStates)
{
	return true;
}

inline const char* ExtMassActionLawParamHandler::identifier() CADET_NOEXCEPT
{
	return "EXT_MASS_ACTION_LAW";
}

inline bool ExtMassActionLawParamHandler::validateConfig(unsigned int nReactions, unsigned int nComp,
														 unsigned int const* nBoundStates)
{
	return true;
}

namespace
{
/**
 * @brief Registers a matrix-valued parameter (row-major storage) with bound states as rows
 * @details The matrix-valued parameter has as many rows as there are bound states in the system.
 * @param [in,out] parameters Parameter map
 * @param [in] unitOpIdx Unit operation id
 * @param [in] parTypeIdx Particle type index
 * @param [in] paramName Name of the parameter
 * @param [in] mat Matrix to register
 * @param [in] nComp Number of components
 * @param [in] boundOffset Array with offsets to bound states of a specific component
 */
inline void registerBoundStateRowMatrix(std::unordered_map<ParameterId, active*>& parameters, UnitOpIdx unitOpIdx,
										ParticleTypeIdx parTypeIdx, const std::string& paramName,
										cadet::linalg::ActiveDenseMatrix& mat, unsigned int nComp,
										unsigned int const* boundOffset)
{
	const cadet::StringHash hashName = cadet::hashStringRuntime(paramName);
	cadet::registerParam2DArray(
		parameters, mat.data(), mat.elements(),
		[=](bool multi, unsigned int row, unsigned int col) {
			const unsigned int comp = std::lower_bound(boundOffset, boundOffset + nComp, row) - boundOffset;
			const unsigned int bnd = row - boundOffset[comp];
			return cadet::makeParamId(hashName, unitOpIdx, comp, parTypeIdx, bnd, col, cadet::SectionIndep);
		},
		mat.columns());
}

/**
 * @brief Registers a matrix-valued parameter (row-major storage) with components as rows
 * @details The matrix-valued parameter has as many rows as there are components in the system.
 * @param [in,out] parameters Parameter map
 * @param [in] unitOpIdx Unit operation id
 * @param [in] parTypeIdx Particle type index
 * @param [in] paramName Name of the parameter
 * @param [in] mat Matrix to register
 */
inline void registerCompRowMatrix(std::unordered_map<ParameterId, active*>& parameters, UnitOpIdx unitOpIdx,
								  ParticleTypeIdx parTypeIdx, const std::string& paramName,
								  cadet::linalg::ActiveDenseMatrix& mat)
{
	const cadet::StringHash hashName = cadet::hashStringRuntime(paramName);
	cadet::registerParam2DArray(
		parameters, mat.data(), mat.elements(),
		[=](bool multi, unsigned int row, unsigned int col) {
			return cadet::makeParamId(hashName, unitOpIdx, row, parTypeIdx, cadet::BoundStateIndep, col,
									  cadet::SectionIndep);
		},
		mat.columns());
}

/**
 * @brief Reads reaction rate exponents and registers them
 * @details Reads a matrix-valued parameter into the given pre-allocated matrix and registers the
 *          parameters in the map. If @p boundOffset is @c nullptr, the matrix has as many columns
 *          as there are components. Otherwise, the matrix has as many columns as there are bound states.
 * @param [in] paramProvider Parameter provider
 * @param [in,out] parameters Parameter map
 * @param [in] unitOpIdx Unit operation id
 * @param [in] parTypeIdx Particle type index
 * @param [in] paramName Name of the parameter
 * @param [out] mat Matrix that holds the values (pre-allocated)
 * @param [in] nComp Number of components
 * @param [in] boundOffset Array with offsets to bound states of a specific component, or @c nullptr
 */
inline void readAndRegisterExponents(cadet::IParameterProvider& paramProvider,
									 std::unordered_map<ParameterId, active*>& parameters, UnitOpIdx unitOpIdx,
									 ParticleTypeIdx parTypeIdx, const std::string& paramName,
									 cadet::linalg::ActiveDenseMatrix& mat, unsigned int nComp,
									 unsigned int const* boundOffset)
{
	if (paramProvider.exists(paramName))
	{
		const std::vector<double> s = paramProvider.getDoubleArray(paramName);
		if (static_cast<int>(s.size()) != mat.elements())
			throw InvalidParameterException("Expected " + paramName + " to be of size " +
											std::to_string(mat.elements()) + "");

		std::copy(s.begin(), s.end(), mat.data());
	}
	else
		mat.setAll(0.0);

	if (boundOffset)
		registerBoundStateRowMatrix(parameters, unitOpIdx, parTypeIdx, paramName, mat, nComp, boundOffset);
	else
		registerCompRowMatrix(parameters, unitOpIdx, parTypeIdx, paramName, mat);
}

/**
 * @brief Calculate gradient of reaction rate (flux) in liquid phase
 * @param [out] fluxGrad Array that holds the gradient of the reaction rate
 * @param [in] r Index of the reaction
 * @param [in] nComp Number of components
 * @param [in] rate Rate constant
 * @param [in] exponents Matrix with exponents in the rate law
 * @param [in] y Array of concentrations
 */
inline void fluxGradLiquid(double* fluxGrad, unsigned int r, unsigned int nComp, double rate,
						   const cadet::linalg::ActiveDenseMatrix& exponents, double const* y)
{
	for (unsigned int c = 0; c < nComp; ++c)
	{
		if (cadet_unlikely(exponents.native(c, r) != 0.0))
			fluxGrad[c] = rate;
		else
			fluxGrad[c] = 0.0;
	}

	// Calculate gradient
	for (unsigned int c = 0; c < nComp; ++c)
	{
		if (cadet_unlikely(exponents.native(c, r) != 0.0))
		{
			const double exponentValue = static_cast<double>(exponents.native(c, r));
			const double v = pow(y[c], exponentValue);
			for (unsigned int j = 0; j < c; ++j)
				fluxGrad[j] *= v;

			fluxGrad[c] *= exponentValue * pow(y[c], exponentValue - 1.0);

			for (unsigned int j = c + 1; j < nComp; ++j)
				fluxGrad[j] *= v;
		}
	}
}

/**
 * @brief Calculate gradient of reaction rate (flux) in liquid-solid phase cell
 * @param [out] fluxGrad Array that holds the gradient of the reaction rate
 * @param [in] r Index of the reaction
 * @param [in] nComp Number of components
 * @param [in] nTotalBoundStates Total number of bound states
 * @param [in] rate Rate constant
 * @param [in] expLiquid Matrix with exponents of the liquid phase concentrations in the rate law
 * @param [in] expSolid Matrix with exponents of the solid phase concentrations in the rate law
 * @param [in] yLiquid Array of liquid phase concentrations (points to first component of liquid phase concentrations)
 * @param [in] ySolid Array of solid phase concentrations (points to first component of solid phase concentrations)
 */
inline void fluxGradCombined(double* fluxGrad, unsigned int r, unsigned int nComp, unsigned int nTotalBoundStates,
							 double rate, const cadet::linalg::ActiveDenseMatrix& expLiquid,
							 const cadet::linalg::ActiveDenseMatrix& expSolid, double const* yLiquid,
							 double const* ySolid)
{
	for (unsigned int c = 0; c < nComp; ++c)
	{
		if (cadet_unlikely(expLiquid.native(c, r) != 0.0))
			fluxGrad[c] = rate;
		else
			fluxGrad[c] = 0.0;
	}

	for (unsigned int c = 0; c < nTotalBoundStates; ++c)
	{
		if (cadet_unlikely(expSolid.native(c, r) != 0.0))
			fluxGrad[nComp + c] = rate;
		else
			fluxGrad[nComp + c] = 0.0;
	}

	// Calculate gradient
	for (unsigned int c = 0; c < nComp; ++c)
	{
		if (cadet_unlikely(expLiquid.native(c, r) != 0.0))
		{
			const double exponentValue = static_cast<double>(expLiquid.native(c, r));
			const double v = pow(yLiquid[c], exponentValue);
			for (unsigned int j = 0; j < c; ++j)
				fluxGrad[j] *= v;

			fluxGrad[c] *= exponentValue * pow(yLiquid[c], exponentValue - 1.0);

			for (unsigned int j = c + 1; j < nComp + nTotalBoundStates; ++j)
				fluxGrad[j] *= v;
		}
	}

	for (unsigned int c = 0; c < nTotalBoundStates; ++c)
	{
		if (cadet_unlikely(expSolid.native(c, r) != 0.0))
		{
			const double exponentValue = static_cast<double>(expSolid.native(c, r));
			const double v = pow(ySolid[c], exponentValue);
			for (unsigned int j = 0; j < nComp + c; ++j)
				fluxGrad[j] *= v;

			fluxGrad[nComp + c] *= exponentValue * pow(ySolid[c], exponentValue - 1.0);

			for (unsigned int j = c + 1; j < nTotalBoundStates; ++j)
				fluxGrad[nComp + j] *= v;
		}
	}
}

template <typename num_t>
inline num_t rateConstantOrZero(const num_t& rate, unsigned int idxReaction,
								const cadet::linalg::ActiveDenseMatrix& exponents, unsigned int nComp)
{
	for (unsigned int c = 0; c < nComp; ++c)
	{
		if (cadet_unlikely(exponents.native(c, idxReaction) != 0.0))
			return rate;
	}
	return 0.0;
}

template <typename num_t>
inline num_t rateConstantOrZero(const num_t& rate, unsigned int idxReaction,
								const cadet::linalg::ActiveDenseMatrix& expLiquid,
								const cadet::linalg::ActiveDenseMatrix& expSolid, unsigned int nComp,
								unsigned int nTotalBoundStates)
{
	for (unsigned int c = 0; c < nComp; ++c)
	{
		if (cadet_unlikely(expLiquid.native(c, idxReaction) != 0.0))
			return rate;
	}
	for (unsigned int c = 0; c < nTotalBoundStates; ++c)
	{
		if (cadet_unlikely(expSolid.native(c, idxReaction) != 0.0))
			return rate;
	}
	return 0.0;
}
} // namespace

/**
 * @brief Defines the multi component Langmuir binding model
 * @details Implements the Langmuir adsorption model: \f[ \begin{align}
 *              \frac{\mathrm{d}q_i}{\mathrm{d}t} &= k_{a,i} c_{p,i} q_{\text{max},i} \left( 1 - \sum_j
 * \frac{q_j}{q_{\text{max},j}} \right) - k_{d,i} q_i \end{align} \f] Multiple bound states are not supported.
 *          Components without bound state (i.e., non-binding components) are supported.
 *
 *          See @cite Langmuir1916.
 * @tparam ParamHandler_t Type that can add support for external function dependence
 */
template <class ParamHandler_t> class MassActionLawReactionBase : public DynamicReactionModelBase
{
public:
	MassActionLawReactionBase()
	{
	}
	virtual ~MassActionLawReactionBase() CADET_NOEXCEPT
	{
	}

	static const char* identifier()
	{
		return ParamHandler_t::identifier();
	}
	virtual const char* name() const CADET_NOEXCEPT
	{
		return ParamHandler_t::identifier();
	}

	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size)
	{
		_paramHandler.setExternalFunctions(extFuns, size);
	}
	virtual bool dependsOnTime() const CADET_NOEXCEPT
	{
		return ParamHandler_t::dependsOnTime();
	}
	virtual bool requiresWorkspace() const CADET_NOEXCEPT
	{
		return true;
	}
	virtual unsigned int workspaceSize(unsigned int nComp, unsigned int totalNumBoundStates,
									   unsigned int const* nBoundStates) const CADET_NOEXCEPT
	{
		return _paramHandler.cacheSize(maxNumReactions(), nComp, totalNumBoundStates) +
			   std::max(maxNumReactions() * sizeof(active), 2 * (_nComp + totalNumBoundStates) * sizeof(double));
	}

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, unsigned int nComp,
											  unsigned int const* nBound, unsigned int const* boundOffset)
	{
		DynamicReactionModelBase::configureModelDiscretization(paramProvider, nComp, nBound, boundOffset);

		if (paramProvider.exists("MAL_STOICHIOMETRY_BULK"))
		{
			const std::size_t numElements = paramProvider.numElements("MAL_STOICHIOMETRY_BULK");
			if (numElements % nComp != 0)
				throw InvalidParameterException(
					"Size of field MAL_STOICHIOMETRY_BULK must be a positive multiple of NCOMP (" +
					std::to_string(nComp) + ")");

			const unsigned int nReactions = numElements / nComp;

			_stoichiometryBulk.resize(nComp, nReactions);
			_expBulkFwd.resize(nComp, nReactions);
			_expBulkBwd.resize(nComp, nReactions);
		}

		if (!nBound || !boundOffset)
			return true;

		if (paramProvider.exists("MAL_STOICHIOMETRY_LIQUID"))
		{
			const std::size_t numElements = paramProvider.numElements("MAL_STOICHIOMETRY_LIQUID");
			if (numElements % nComp != 0)
				throw InvalidParameterException(
					"Size of field MAL_STOICHIOMETRY_LIQUID must be a positive multiple of NCOMP (" +
					std::to_string(nComp) + ")");

			const unsigned int nReactions = numElements / nComp;

			_stoichiometryLiquid.resize(nComp, nReactions);
			_expLiquidFwd.resize(nComp, nReactions);
			_expLiquidBwd.resize(nComp, nReactions);

			_expLiquidFwdSolid.resize(_nTotalBoundStates, nReactions);
			_expLiquidBwdSolid.resize(_nTotalBoundStates, nReactions);
		}

		if (paramProvider.exists("MAL_STOICHIOMETRY_SOLID") && (_nTotalBoundStates > 0))
		{
			const std::size_t numElements = paramProvider.numElements("MAL_STOICHIOMETRY_SOLID");
			if (numElements % _nTotalBoundStates != 0)
				throw InvalidParameterException(
					"Size of field MAL_STOICHIOMETRY_SOLID must be a positive multiple of NTOTALBND (" +
					std::to_string(_nTotalBoundStates) + ")");

			const unsigned int nReactions = numElements / _nTotalBoundStates;

			_stoichiometrySolid.resize(_nTotalBoundStates, nReactions);
			_expSolidFwd.resize(_nTotalBoundStates, nReactions);
			_expSolidBwd.resize(_nTotalBoundStates, nReactions);

			_expSolidFwdLiquid.resize(nComp, nReactions);
			_expSolidBwdLiquid.resize(nComp, nReactions);
		}

		return true;
	}

	virtual unsigned int numReactionsLiquid() const CADET_NOEXCEPT
	{
		return _stoichiometryBulk.columns();
	}
	virtual unsigned int numReactionsCombined() const CADET_NOEXCEPT
	{
		return _stoichiometryLiquid.columns() + _stoichiometrySolid.columns();
	}

	CADET_DYNAMICREACTIONMODEL_BOILERPLATE

protected:
	ParamHandler_t _paramHandler; //!< Handles parameters and their dependence on external functions

	linalg::ActiveDenseMatrix _stoichiometryBulk;
	linalg::ActiveDenseMatrix _expBulkFwd;
	linalg::ActiveDenseMatrix _expBulkBwd;

	linalg::ActiveDenseMatrix _stoichiometryLiquid;
	linalg::ActiveDenseMatrix _expLiquidFwd;
	linalg::ActiveDenseMatrix _expLiquidBwd;
	linalg::ActiveDenseMatrix _expLiquidFwdSolid;
	linalg::ActiveDenseMatrix _expLiquidBwdSolid;

	linalg::ActiveDenseMatrix _stoichiometrySolid;
	linalg::ActiveDenseMatrix _expSolidFwd;
	linalg::ActiveDenseMatrix _expSolidBwd;
	linalg::ActiveDenseMatrix _expSolidFwdLiquid;
	linalg::ActiveDenseMatrix _expSolidBwdLiquid;

	inline int maxNumReactions() const CADET_NOEXCEPT
	{
		return std::max(std::max(_stoichiometryBulk.columns(), _stoichiometryLiquid.columns()),
						_stoichiometrySolid.columns());
	}

	virtual bool configureImpl(IParameterProvider& paramProvider, UnitOpIdx unitOpIdx, ParticleTypeIdx parTypeIdx)
	{
		_paramHandler.configure(paramProvider, maxNumReactions(), _nComp, _nBoundStates);
		_paramHandler.registerParameters(_parameters, unitOpIdx, parTypeIdx, _nComp, _nBoundStates);

		if ((_stoichiometryBulk.columns() > 0) &&
			((static_cast<int>(_paramHandler.kFwdBulk().size()) < _stoichiometryBulk.columns()) ||
			 (static_cast<int>(_paramHandler.kBwdBulk().size()) < _stoichiometryBulk.columns())))
			throw InvalidParameterException(
				"MAL_KFWD_BULK and MAL_KBWD_BULK have to have the same size (number of reactions)");

		if ((_stoichiometryLiquid.columns() > 0) &&
			((static_cast<int>(_paramHandler.kFwdLiquid().size()) < _stoichiometryLiquid.columns()) ||
			 (static_cast<int>(_paramHandler.kBwdLiquid().size()) < _stoichiometryLiquid.columns())))
			throw InvalidParameterException(
				"MAL_KFWD_LIQUID and MAL_KBWD_LIQUID have to have the same size (number of reactions)");

		if ((_stoichiometrySolid.columns() > 0) &&
			((static_cast<int>(_paramHandler.kFwdSolid().size()) < _stoichiometrySolid.columns()) ||
			 (static_cast<int>(_paramHandler.kBwdSolid().size()) < _stoichiometrySolid.columns())))
			throw InvalidParameterException(
				"MAL_KFWD_SOLID and MAL_KBWD_SOLID have to have the same size (number of reactions)");

		if (paramProvider.exists("MAL_STOICHIOMETRY_BULK"))
		{
			const std::vector<double> s = paramProvider.getDoubleArray("MAL_STOICHIOMETRY_BULK");

			if (static_cast<int>(s.size()) != _stoichiometryBulk.elements())
				throw InvalidParameterException(
					"MAL_STOICHIOMETRY_BULK has changed size (number of reactions changed)");

			std::copy(s.begin(), s.end(), _stoichiometryBulk.data());
		}
		registerCompRowMatrix(_parameters, unitOpIdx, parTypeIdx, "MAL_STOICHIOMETRY_BULK", _stoichiometryBulk);

		if (paramProvider.exists("MAL_EXPONENTS_BULK_FWD"))
		{
			const std::vector<double> s = paramProvider.getDoubleArray("MAL_EXPONENTS_BULK_FWD");
			if (static_cast<int>(s.size()) != _stoichiometryBulk.elements())
				throw InvalidParameterException(
					"Expected MAL_EXPONENTS_BULK_FWD and MAL_STOICHIOMETRY_BULK to be of the same size (" +
					std::to_string(_stoichiometryBulk.elements()) + ")");

			std::copy(s.begin(), s.end(), _expBulkFwd.data());
		}
		else
		{
			// Obtain default values from stoichiometry
			std::transform(_stoichiometryBulk.data(), _stoichiometryBulk.data() + _stoichiometryBulk.elements(),
						   _expBulkFwd.data(), [](const active& v) { return std::max(0.0, -static_cast<double>(v)); });
		}
		registerCompRowMatrix(_parameters, unitOpIdx, parTypeIdx, "MAL_EXPONENTS_BULK_FWD", _expBulkFwd);

		if (paramProvider.exists("MAL_EXPONENTS_BULK_BWD"))
		{
			const std::vector<double> s = paramProvider.getDoubleArray("MAL_EXPONENTS_BULK_BWD");
			if (static_cast<int>(s.size()) != _stoichiometryBulk.elements())
				throw InvalidParameterException(
					"Expected MAL_EXPONENTS_BULK_BWD and MAL_STOICHIOMETRY_BULK to be of the same size (" +
					std::to_string(_stoichiometryBulk.elements()) + ")");

			std::copy(s.begin(), s.end(), _expBulkBwd.data());
		}
		else
		{
			// Obtain default values from stoichiometry
			std::transform(_stoichiometryBulk.data(), _stoichiometryBulk.data() + _stoichiometryBulk.elements(),
						   _expBulkBwd.data(), [](const active& v) { return std::max(0.0, static_cast<double>(v)); });
		}
		registerCompRowMatrix(_parameters, unitOpIdx, parTypeIdx, "MAL_EXPONENTS_BULK_BWD", _expBulkBwd);

		if (paramProvider.exists("MAL_STOICHIOMETRY_LIQUID"))
		{
			const std::vector<double> s = paramProvider.getDoubleArray("MAL_STOICHIOMETRY_LIQUID");

			if (static_cast<int>(s.size()) != _stoichiometryLiquid.elements())
				throw InvalidParameterException(
					"MAL_STOICHIOMETRY_LIQUID has changed size (number of reactions changed)");

			std::copy(s.begin(), s.end(), _stoichiometryLiquid.data());
		}
		registerCompRowMatrix(_parameters, unitOpIdx, parTypeIdx, "MAL_STOICHIOMETRY_LIQUID", _stoichiometryLiquid);

		if (paramProvider.exists("MAL_EXPONENTS_LIQUID_FWD"))
		{
			const std::vector<double> s = paramProvider.getDoubleArray("MAL_EXPONENTS_LIQUID_FWD");
			if (static_cast<int>(s.size()) != _stoichiometryLiquid.elements())
				throw InvalidParameterException(
					"Expected MAL_EXPONENTS_LIQUID_FWD and MAL_STOICHIOMETRY_LIQUID to be of the same size (" +
					std::to_string(_stoichiometryLiquid.elements()) + ")");

			std::copy(s.begin(), s.end(), _expLiquidFwd.data());
		}
		else
		{
			// Obtain default values from stoichiometry
			std::transform(_stoichiometryLiquid.data(), _stoichiometryLiquid.data() + _stoichiometryLiquid.elements(),
						   _expLiquidFwd.data(),
						   [](const active& v) { return std::max(0.0, -static_cast<double>(v)); });
		}
		registerCompRowMatrix(_parameters, unitOpIdx, parTypeIdx, "MAL_EXPONENTS_LIQUID_FWD", _expLiquidFwd);

		if (paramProvider.exists("MAL_EXPONENTS_LIQUID_BWD"))
		{
			const std::vector<double> s = paramProvider.getDoubleArray("MAL_EXPONENTS_LIQUID_BWD");
			if (static_cast<int>(s.size()) != _stoichiometryLiquid.elements())
				throw InvalidParameterException(
					"Expected MAL_EXPONENTS_LIQUID_BWD and MAL_STOICHIOMETRY_LIQUID to be of the same size (" +
					std::to_string(_stoichiometryLiquid.elements()) + ")");

			std::copy(s.begin(), s.end(), _expLiquidBwd.data());
		}
		else
		{
			// Obtain default values from stoichiometry
			std::transform(_stoichiometryLiquid.data(), _stoichiometryLiquid.data() + _stoichiometryLiquid.elements(),
						   _expLiquidBwd.data(), [](const active& v) { return std::max(0.0, static_cast<double>(v)); });
		}
		registerCompRowMatrix(_parameters, unitOpIdx, parTypeIdx, "MAL_EXPONENTS_LIQUID_BWD", _expLiquidBwd);

		if (!_nBoundStates || !_boundOffset)
			return true;

		readAndRegisterExponents(paramProvider, _parameters, unitOpIdx, parTypeIdx, "MAL_EXPONENTS_LIQUID_FWD_MODSOLID",
								 _expLiquidFwdSolid, _nComp, _boundOffset);
		readAndRegisterExponents(paramProvider, _parameters, unitOpIdx, parTypeIdx, "MAL_EXPONENTS_LIQUID_BWD_MODSOLID",
								 _expLiquidBwdSolid, _nComp, _boundOffset);

		if (paramProvider.exists("MAL_STOICHIOMETRY_SOLID"))
		{
			const std::vector<double> s = paramProvider.getDoubleArray("MAL_STOICHIOMETRY_SOLID");

			if (static_cast<int>(s.size()) != _stoichiometrySolid.elements())
				throw InvalidParameterException(
					"MAL_STOICHIOMETRY_SOLID has changed size (number of reactions changed)");

			std::copy(s.begin(), s.end(), _stoichiometrySolid.data());
		}
		registerBoundStateRowMatrix(_parameters, unitOpIdx, parTypeIdx, "MAL_STOICHIOMETRY_SOLID", _stoichiometrySolid,
									_nComp, _boundOffset);

		if (paramProvider.exists("MAL_EXPONENTS_SOLID_FWD"))
		{
			const std::vector<double> s = paramProvider.getDoubleArray("MAL_EXPONENTS_SOLID_FWD");
			if (static_cast<int>(s.size()) != _stoichiometrySolid.elements())
				throw InvalidParameterException(
					"Expected MAL_EXPONENTS_SOLID_FWD and MAL_STOICHIOMETRY_SOLID to be of the same size (" +
					std::to_string(_stoichiometrySolid.elements()) + ")");

			std::copy(s.begin(), s.end(), _expSolidFwd.data());
		}
		else
		{
			// Obtain default values from stoichiometry
			std::transform(_stoichiometrySolid.data(), _stoichiometrySolid.data() + _stoichiometrySolid.elements(),
						   _expSolidFwd.data(), [](const active& v) { return std::max(0.0, -static_cast<double>(v)); });
		}
		registerBoundStateRowMatrix(_parameters, unitOpIdx, parTypeIdx, "MAL_EXPONENTS_SOLID_FWD", _expSolidFwd, _nComp,
									_boundOffset);

		if (paramProvider.exists("MAL_EXPONENTS_SOLID_BWD"))
		{
			const std::vector<double> s = paramProvider.getDoubleArray("MAL_EXPONENTS_SOLID_BWD");
			if (static_cast<int>(s.size()) != _stoichiometrySolid.elements())
				throw InvalidParameterException(
					"Expected MAL_EXPONENTS_SOLID_BWD and MAL_STOICHIOMETRY_SOLID to be of the same size (" +
					std::to_string(_stoichiometrySolid.elements()) + ")");

			std::copy(s.begin(), s.end(), _expSolidBwd.data());
		}
		else
		{
			// Obtain default values from stoichiometry
			std::transform(_stoichiometrySolid.data(), _stoichiometrySolid.data() + _stoichiometrySolid.elements(),
						   _expSolidBwd.data(), [](const active& v) { return std::max(0.0, static_cast<double>(v)); });
		}
		registerBoundStateRowMatrix(_parameters, unitOpIdx, parTypeIdx, "MAL_EXPONENTS_SOLID_BWD", _expSolidBwd, _nComp,
									_boundOffset);

		readAndRegisterExponents(paramProvider, _parameters, unitOpIdx, parTypeIdx, "MAL_EXPONENTS_SOLID_FWD_MODLIQUID",
								 _expSolidFwdLiquid, _nComp, nullptr);
		readAndRegisterExponents(paramProvider, _parameters, unitOpIdx, parTypeIdx, "MAL_EXPONENTS_SOLID_BWD_MODLIQUID",
								 _expSolidBwdLiquid, _nComp, nullptr);

		return true;
	}

	template <typename StateType, typename ResidualType, typename ParamType, typename FactorType>
	int residualLiquidImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* y,
						   ResidualType* res, const FactorType& factor, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Calculate fluxes
		typedef typename DoubleActivePromoter<StateType, ParamType>::type flux_t;
		BufferedArray<flux_t> fluxes = workSpace.array<flux_t>(maxNumReactions());
		for (int r = 0; r < _stoichiometryBulk.columns(); ++r)
		{
			flux_t fwd =
				rateConstantOrZero(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(p->kFwdBulk[r]), r,
								   _expBulkFwd, _nComp);
			for (int c = 0; c < _nComp; ++c)
			{
				if (_expBulkFwd.native(c, r) != 0.0)
				{
					if (static_cast<double>(y[c]) > 0.0)
						fwd *= pow(
							static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(y[c]),
							static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(_expBulkFwd.native(c, r)));
					else
					{
						fwd *= 0.0;
						break;
					}
				}
			}

			flux_t bwd =
				rateConstantOrZero(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(p->kBwdBulk[r]), r,
								   _expBulkBwd, _nComp);
			for (int c = 0; c < _nComp; ++c)
			{
				if (_expBulkBwd.native(c, r) != 0.0)
				{
					if (static_cast<double>(y[c]) > 0.0)
						bwd *= pow(
							static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(y[c]),
							static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(_expBulkBwd.native(c, r)));
					else
					{
						bwd *= 0.0;
						break;
					}
				}
			}

			fluxes[r] = fwd - bwd;
		}

		// Add reaction terms to residual
		_stoichiometryBulk.multiplyVector(static_cast<flux_t*>(fluxes), factor, res);

		return 0;
	}

	template <typename StateType, typename ResidualType, typename ParamType>
	int residualCombinedImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, StateType const* yLiquid,
							 StateType const* ySolid, ResidualType* resLiquid, ResidualType* resSolid, double factor,
							 LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		// Calculate fluxes in liquid phase
		typedef typename DoubleActivePromoter<StateType, ParamType>::type flux_t;
		BufferedArray<flux_t> fluxes = workSpace.array<flux_t>(maxNumReactions());
		for (int r = 0; r < _stoichiometryLiquid.columns(); ++r)
		{
			flux_t fwd =
				rateConstantOrZero(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(p->kFwdLiquid[r]), r,
								   _expLiquidFwd, _expLiquidFwdSolid, _nComp, _nTotalBoundStates);
			for (int c = 0; c < _nComp; ++c)
			{
				if (_expLiquidFwd.native(c, r) != 0.0)
				{
					if (static_cast<double>(yLiquid[c]) > 0.0)
						fwd *= pow(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(yLiquid[c]),
								   static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(
									   _expLiquidFwd.native(c, r)));
					else
					{
						fwd *= 0.0;
						break;
					}
				}
			}
			for (int c = 0; c < _nTotalBoundStates; ++c)
			{
				if (_expLiquidFwdSolid.native(c, r) != 0.0)
				{
					if (static_cast<double>(ySolid[c]) > 0.0)
						fwd *= pow(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(ySolid[c]),
								   static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(
									   _expLiquidFwdSolid.native(c, r)));
					else
					{
						fwd *= 0.0;
						break;
					}
				}
			}

			flux_t bwd =
				rateConstantOrZero(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(p->kBwdLiquid[r]), r,
								   _expLiquidBwd, _expLiquidBwdSolid, _nComp, _nTotalBoundStates);
			for (int c = 0; c < _nComp; ++c)
			{
				if (_expLiquidBwd.native(c, r) != 0.0)
				{
					if (static_cast<double>(yLiquid[c]) > 0.0)
						bwd *= pow(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(yLiquid[c]),
								   static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(
									   _expLiquidBwd.native(c, r)));
					else
					{
						bwd *= 0.0;
						break;
					}
				}
			}
			for (int c = 0; c < _nTotalBoundStates; ++c)
			{
				if (_expLiquidBwdSolid.native(c, r) != 0.0)
				{
					if (static_cast<double>(ySolid[c]) > 0.0)
						bwd *= pow(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(ySolid[c]),
								   static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(
									   _expLiquidBwdSolid.native(c, r)));
					else
					{
						bwd *= 0.0;
						break;
					}
				}
			}

			fluxes[r] = fwd - bwd;
		}

		// Add reaction terms to liquid phase residual
		_stoichiometryLiquid.multiplyVector(static_cast<flux_t*>(fluxes), factor, resLiquid);

		if (_nTotalBoundStates == 0)
			return 0;

		// Calculate fluxes in solid phase
		for (int r = 0; r < _stoichiometrySolid.columns(); ++r)
		{
			flux_t fwd =
				rateConstantOrZero(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(p->kFwdSolid[r]), r,
								   _expSolidFwdLiquid, _expSolidFwd, _nComp, _nTotalBoundStates);
			for (int c = 0; c < _nComp; ++c)
			{
				if (_expSolidFwdLiquid.native(c, r) != 0.0)
				{
					if (static_cast<double>(yLiquid[c]) > 0.0)
						fwd *= pow(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(yLiquid[c]),
								   static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(
									   _expSolidFwdLiquid.native(c, r)));
					else
					{
						fwd *= 0.0;
						break;
					}
				}
			}
			for (int c = 0; c < _nTotalBoundStates; ++c)
			{
				if (_expSolidFwd.native(c, r) != 0.0)
				{
					if (static_cast<double>(ySolid[c]) > 0.0)
						fwd *= pow(
							static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(ySolid[c]),
							static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(_expSolidFwd.native(c, r)));
					else
					{
						fwd *= 0.0;
						break;
					}
				}
			}

			flux_t bwd =
				rateConstantOrZero(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(p->kBwdSolid[r]), r,
								   _expSolidBwdLiquid, _expSolidBwd, _nComp, _nTotalBoundStates);
			for (int c = 0; c < _nComp; ++c)
			{
				if (_expSolidBwdLiquid.native(c, r) != 0.0)
				{
					if (static_cast<double>(yLiquid[c]) > 0.0)
						bwd *= pow(static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(yLiquid[c]),
								   static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(
									   _expSolidBwdLiquid.native(c, r)));
					else
					{
						bwd *= 0.0;
						break;
					}
				}
			}
			for (int c = 0; c < _nTotalBoundStates; ++c)
			{
				if (_expSolidBwd.native(c, r) != 0.0)
				{
					if (static_cast<double>(ySolid[c]) > 0.0)
						bwd *= pow(
							static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(ySolid[c]),
							static_cast<typename DoubleActiveDemoter<flux_t, active>::type>(_expSolidBwd.native(c, r)));
					else
					{
						bwd *= 0.0;
						break;
					}
				}
			}

			fluxes[r] = fwd - bwd;
		}

		// Add reaction terms to solid phase residual
		_stoichiometrySolid.multiplyVector(static_cast<flux_t*>(fluxes), factor, resSolid);

		return 0;
	}

	template <typename RowIterator>
	void jacobianLiquidImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* y, double factor,
							const RowIterator& jac, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		BufferedArray<double> fluxes = workSpace.array<double>(2 * _nComp);
		double* const fluxGradFwd = static_cast<double*>(fluxes);
		double* const fluxGradBwd = fluxGradFwd + _nComp;
		for (int r = 0; r < _stoichiometryBulk.columns(); ++r)
		{
			// Calculate gradients of forward and backward fluxes
			fluxGradLiquid(fluxGradFwd, r, _nComp, static_cast<double>(p->kFwdBulk[r]), _expBulkFwd, y);
			fluxGradLiquid(fluxGradBwd, r, _nComp, static_cast<double>(p->kBwdBulk[r]), _expBulkBwd, y);

			// Add gradients to Jacobian
			RowIterator curJac = jac;
			for (int row = 0; row < _nComp; ++row, ++curJac)
			{
				const double colFactor = static_cast<double>(_stoichiometryBulk.native(row, r)) * factor;
				for (int col = 0; col < _nComp; ++col)
					curJac[col - static_cast<int>(row)] += colFactor * (fluxGradFwd[col] - fluxGradBwd[col]);
			}
		}
	}

	template <typename RowIteratorLiquid, typename RowIteratorSolid>
	void jacobianCombinedImpl(double t, unsigned int secIdx, const ColumnPosition& colPos, double const* yLiquid,
							  double const* ySolid, double factor, const RowIteratorLiquid& jacLiquid,
							  const RowIteratorSolid& jacSolid, LinearBufferAllocator workSpace) const
	{
		typename ParamHandler_t::ParamsHandle const p =
			_paramHandler.update(t, secIdx, colPos, _nComp, _nBoundStates, workSpace);

		BufferedArray<double> fluxes = workSpace.array<double>(2 * (_nComp + _nTotalBoundStates));
		double* const fluxGradFwd = static_cast<double*>(fluxes);
		double* const fluxGradBwd = fluxGradFwd + _nComp + _nTotalBoundStates;

		for (int r = 0; r < _stoichiometryLiquid.columns(); ++r)
		{
			// Calculate gradients of forward and backward fluxes
			fluxGradCombined(fluxGradFwd, r, _nComp, _nTotalBoundStates, static_cast<double>(p->kFwdLiquid[r]),
							 _expLiquidFwd, _expLiquidFwdSolid, yLiquid, ySolid);
			fluxGradCombined(fluxGradBwd, r, _nComp, _nTotalBoundStates, static_cast<double>(p->kBwdLiquid[r]),
							 _expLiquidBwd, _expLiquidBwdSolid, yLiquid, ySolid);

			// Add gradients to Jacobian
			RowIteratorLiquid curJac = jacLiquid;
			for (int row = 0; row < _nComp; ++row, ++curJac)
			{
				const double colFactor = static_cast<double>(_stoichiometryLiquid.native(row, r)) * factor;
				for (int col = 0; col < _nComp + _nTotalBoundStates; ++col)
					curJac[col - static_cast<int>(row)] += colFactor * (fluxGradFwd[col] - fluxGradBwd[col]);
			}
		}

		if (_nTotalBoundStates == 0)
			return;

		for (int r = 0; r < _stoichiometrySolid.columns(); ++r)
		{
			// Calculate gradients of forward and backward fluxes
			fluxGradCombined(fluxGradFwd, r, _nComp, _nTotalBoundStates, static_cast<double>(p->kFwdSolid[r]),
							 _expSolidFwdLiquid, _expSolidFwd, yLiquid, ySolid);
			fluxGradCombined(fluxGradBwd, r, _nComp, _nTotalBoundStates, static_cast<double>(p->kBwdSolid[r]),
							 _expSolidBwdLiquid, _expSolidBwd, yLiquid, ySolid);

			// Add gradients to Jacobian
			RowIteratorSolid curJac = jacSolid;
			for (int row = 0; row < _nTotalBoundStates; ++row, ++curJac)
			{
				const double colFactor = static_cast<double>(_stoichiometrySolid.native(row, r)) * factor;
				for (int col = 0; col < _nComp + _nTotalBoundStates; ++col)
					curJac[col - static_cast<int>(_nComp) - static_cast<int>(row)] +=
						colFactor * (fluxGradFwd[col] - fluxGradBwd[col]);
			}
		}
	}
};

typedef MassActionLawReactionBase<MassActionLawParamHandler> MassActionLawReaction;
typedef MassActionLawReactionBase<ExtMassActionLawParamHandler> ExternalMassActionLawReaction;

namespace reaction
{
void registerMassActionLawReaction(
	std::unordered_map<std::string, std::function<model::IDynamicReactionModel*()>>& reactions)
{
	reactions[MassActionLawReaction::identifier()] = []() { return new MassActionLawReaction(); };
	reactions[ExternalMassActionLawReaction::identifier()] = []() { return new ExternalMassActionLawReaction(); };
}
} // namespace reaction

} // namespace model

} // namespace cadet
