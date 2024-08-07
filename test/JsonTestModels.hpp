// =============================================================================
//  CADET
//  
//  Copyright © 2008-2024: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

/**
 * @file 
 * Defines a ParameterProvider that uses JSON.
 */

#ifndef CADETTEST_JSONTESTMODELS_HPP_
#define CADETTEST_JSONTESTMODELS_HPP_

#include "common/JsonParameterProvider.hpp"

cadet::JsonParameterProvider createColumnWithSMA(const std::string& uoType, const std::string& spatialScheme);
cadet::JsonParameterProvider createColumnWithTwoCompLinearBinding(const std::string& uoType, const std::string& spatialScheme);
cadet::JsonParameterProvider createColumnLinearBenchmark(bool dynamicBinding, bool nonBinding, const std::string& uoType, const std::string& spatialScheme);
nlohmann::json createLWEJson(const std::string& uoType, const std::string& spatialMethod);
cadet::JsonParameterProvider createLWE(const std::string& uoType, const std::string& spatialScheme);
cadet::JsonParameterProvider createPulseInjectionColumn(const std::string& uoType, const std::string& spatialScheme, bool dynamicBinding);
cadet::JsonParameterProvider createLinearBenchmark(bool dynamicBinding, bool nonBinding, const std::string& uoType, const std::string& spatialScheme);
cadet::JsonParameterProvider createCSTR(unsigned int nComp);
cadet::JsonParameterProvider createCSTRBenchmark(unsigned int nSec, double endTime, double interval);
cadet::JsonParameterProvider createCSTRVarPorBenchmark(unsigned int nSec, double endTime, double interval);
cadet::JsonParameterProvider createCSTRVarPor(unsigned int nComp);

#endif  // CADETTEST_JSONTESTMODELS_HPP_
