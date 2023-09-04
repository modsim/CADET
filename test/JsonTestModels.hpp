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
 * Defines a ParameterProvider that uses JSON.
 */

#ifndef CADETTEST_JSONTESTMODELS_HPP_
#define CADETTEST_JSONTESTMODELS_HPP_

#include "common/JsonParameterProvider.hpp"

cadet::JsonParameterProvider createColumnWithSMA(const std::string& uoType);
cadet::JsonParameterProvider createColumnWithTwoCompLinearBinding(const std::string& uoType);
cadet::JsonParameterProvider createLWE(const std::string& uoType);
cadet::JsonParameterProvider createPulseInjectionColumn(const std::string& uoType, bool dynamicBinding);
cadet::JsonParameterProvider createLinearBenchmark(bool dynamicBinding, bool nonBinding, const std::string& uoType);
cadet::JsonParameterProvider createCSTR(unsigned int nComp);
cadet::JsonParameterProvider createCSTRBenchmark(unsigned int nSec, double endTime, double interval);

#endif // CADETTEST_JSONTESTMODELS_HPP_
