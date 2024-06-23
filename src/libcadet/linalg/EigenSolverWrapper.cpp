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

#include "EigenSolverWrapper.hpp"

#include <string>

namespace cadet
{

namespace linalg
{

    cadet::linalg::EigenSolverBase* setLinearSolver(const std::string solverName)
    {
        if (solverName.find("SparseLU", 0) == 0)
        {
            if (solverName.find("COLAMDOrdering", 0) != std::string::npos)
                return new cadet::linalg::SparseLU<Eigen::COLAMDOrdering<int>>();

            else if (solverName.find("AMD", 0) != std::string::npos) // requires symmetrical matrix
                return new cadet::linalg::SparseLU<Eigen::AMDOrdering<int>>();

            else if (solverName.find("NaturalOrdering", 0) != std::string::npos)
                return new cadet::linalg::SparseLU<Eigen::NaturalOrdering<int>>();

            else
                return new cadet::linalg::SparseLU<Eigen::COLAMDOrdering<int>>();
        }
        else if (solverName.find("SparseQR", 0) == 0)
        {
            if (solverName.find("COLAMDOrdering", 0) != std::string::npos)
                return new cadet::linalg::SparseQR<Eigen::COLAMDOrdering<int>>();

            else if (solverName.find("AMD", 0) != std::string::npos) // requires symmetrical matrix
                return new cadet::linalg::SparseQR<Eigen::COLAMDOrdering<int>>();

            else if (solverName.find("NaturalOrdering", 0) != std::string::npos)
                return new cadet::linalg::SparseQR<Eigen::NaturalOrdering<int>>();

            else
                return new cadet::linalg::SparseQR<Eigen::COLAMDOrdering<int>>();
        }
        else if (solverName.find("BiCGSTAB", 0) == 0)
        {
            if (solverName.find("IdentityPreconditioner", 0) != std::string::npos)
                return new cadet::linalg::BiCGSTAB<Eigen::IdentityPreconditioner>();

            else if (solverName.find("DiagonalPreconditioner", 0) != std::string::npos)
                return new cadet::linalg::BiCGSTAB<Eigen::DiagonalPreconditioner<double>>();

            else if (solverName.find("IncompleteLUT", 0) != std::string::npos)
                return new cadet::linalg::BiCGSTAB<Eigen::IncompleteLUT<double>>();

            else
                return new cadet::linalg::BiCGSTAB<Eigen::DiagonalPreconditioner<double>>();
        }
        else if (solverName.find("LeastSquaresConjugateGradient", 0) == 0)
        {
            if (solverName.find("IdentityPreconditioner", 0) != std::string::npos)
                return new cadet::linalg::LeastSquaresConjugateGradient<Eigen::IdentityPreconditioner>();

            else if (solverName.find("LeastSquareDiagonalPreconditioner", 0) != std::string::npos)
                return new cadet::linalg::LeastSquaresConjugateGradient<Eigen::LeastSquareDiagonalPreconditioner<double>>();

            else
                return new cadet::linalg::LeastSquaresConjugateGradient<Eigen::LeastSquareDiagonalPreconditioner<double>>();
        }
        else {
            throw std::invalid_argument("Unknown linear solver name: " + solverName);
        }
    }

}
}