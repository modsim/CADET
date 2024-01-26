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
 * @mainpage CADET
 * The library provides a fast and accurate forward simulator for the general rate model of column liquid chromatography
 *
 * @authors    Eric von Lieres, Joel Andersson, Andreas Puettmann, Sebastian Schnittert, Samuel Leweke, William Heymann
 * @version    4.4.0
 * @date       2008-2023
 * @copyright  GNU General Public License v3.0 (or, at your option, any later version).
 */

/**
 * @file
 * Main include file for the public interface to CADET.
 * @todo Check if all headers are included
 */

#include "cadet/cadetCompilerInfo.hpp"
#include "cadet/LibExportImport.hpp"
#include "cadet/LibVersionInfo.hpp"
#include "cadet/StrongTypes.hpp"
#include "cadet/Exceptions.hpp"
#include "cadet/StringUtil.hpp"
#include "cadet/HashUtil.hpp"
#include "cadet/Logging.hpp"
#include "cadet/ParameterProvider.hpp"
#include "cadet/ParameterId.hpp"
#include "cadet/ExternalFunction.hpp"
#include "cadet/InletProfile.hpp"
#include "cadet/Model.hpp"
#include "cadet/ModelSystem.hpp"
#include "cadet/ModelBuilder.hpp"
#include "cadet/SolutionExporter.hpp"
#include "cadet/SolutionRecorder.hpp"
#include "cadet/Simulator.hpp"
#include "cadet/FactoryFuncs.hpp"
#include "cadet/Notification.hpp"
