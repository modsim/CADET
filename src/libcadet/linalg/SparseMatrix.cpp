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

#include "linalg/SparseMatrix.hpp"

#include <sstream>
#include <ostream>

namespace cadet
{

namespace linalg
{

std::ostream& operator<<(std::ostream& out, const DoubleSparseMatrix& sm)
{
	std::ostringstream cols;
	std::ostringstream rows;
	std::ostringstream elems;

	cols.copyfmt(out);
	rows.copyfmt(out);
	elems.copyfmt(out);

	cols << "cols = [";
	rows << "rows = [";
	elems << "elems = [";

	const std::vector<int>& spRows = sm.rows();
	const std::vector<int>& spCols = sm.cols();
	const std::vector<double>& spVals = sm.values();

	for (unsigned int i = 0; i < sm.numNonZero(); ++i)
	{
		if (i > 0)
		{
			cols << ", ";
			rows << ", ";
			elems << ", ";
		}

		cols << spCols[i] + 1;
		rows << spRows[i] + 1;
		elems << spVals[i];
	}

	cols << "];";
	rows << "];";
	elems << "];";

	out << cols.str() << "\n";
	out << rows.str() << "\n";
	out << elems.str();
	return out;
}

} // namespace linalg

} // namespace cadet
