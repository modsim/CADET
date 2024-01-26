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

#ifndef CADETTOOLS_FORMATCONVERTER_HPP_
#define CADETTOOLS_FORMATCONVERTER_HPP_

#include <string>

namespace cadet
{
namespace io
{

class IFileReader;
class IFileWriter;

} // namespace io
} // namespace cadet

/**
 * @brief Copies a group and all of its content (including subgroups) from reader to writer
 * @details Performs a recursive copy operation transferring the content from a reader to a writer object.
 * @param [in] rd Reader
 * @param [in] wr Writer
 * @param [in] path Path to group that is copied
 */
void copyGroup(cadet::io::IFileReader& rd, cadet::io::IFileWriter& wr, const std::string& path);

#endif /* CADETTOOLS_FORMATCONVERTER_HPP_ */
