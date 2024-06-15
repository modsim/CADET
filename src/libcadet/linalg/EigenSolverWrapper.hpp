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

#ifndef LIBCADET_EigenSolverWrapper_HPP_
#define LIBCADET_EigenSolverWrapper_HPP_

#include <Eigen/Sparse>
#include <memory>

namespace cadet
{

namespace linalg
{

class EigenSolverBase {
public:
    virtual ~EigenSolverBase() = default;

    virtual void analyzePattern(const Eigen::SparseMatrix<double>& mat) = 0;
    virtual void factorize(const Eigen::SparseMatrix<double>& mat) = 0;
    virtual Eigen::VectorXd solve(const Eigen::VectorXd& b) = 0;
    virtual Eigen::ComputationInfo info() const = 0;
};

template <typename OrderingType>
class SparseLU : public EigenSolverBase {
public:
    void analyzePattern(const Eigen::SparseMatrix<double>& mat) override {
        solver.analyzePattern(mat);
    }

    void factorize(const Eigen::SparseMatrix<double>& mat) override {
        solver.factorize(mat);
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& b) override {
        return solver.solve(b);
    }

    Eigen::ComputationInfo info() const override {
        return solver.info();
    }

private:
    Eigen::SparseLU<Eigen::SparseMatrix<double>, OrderingType> solver;
};

template <typename OrderingType>
class SparseQR : public EigenSolverBase {
public:
    void analyzePattern(const Eigen::SparseMatrix<double>& mat) override {
        //solver.analyzePattern(mat);
    }

    void factorize(const Eigen::SparseMatrix<double>& mat) override {
        solver.factorize(mat);
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& b) override {
        return solver.solve(b);
    }

    Eigen::ComputationInfo info() const override {
        return solver.info();
    }

private:
    Eigen::SparseQR<Eigen::SparseMatrix<double, Eigen::RowMajor>, OrderingType> solver;
};

template <typename PreConditioner>
class BiCGSTAB : public EigenSolverBase {
public:
    void analyzePattern(const Eigen::SparseMatrix<double>& mat) override {
        solver.analyzePattern(mat);
    }

    void factorize(const Eigen::SparseMatrix<double>& mat) override {

        solver.compute(mat);

        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("BiCGSTAB decomposition failed");
        }
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& b) override {
        return solver.solve(b);
    }

    Eigen::ComputationInfo info() const override {
        return solver.info();
    }

private:
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, PreConditioner> solver;
};

template <typename PreConditioner>
class LeastSquaresConjugateGradient : public EigenSolverBase {
public:
    void analyzePattern(const Eigen::SparseMatrix<double>& mat) override {
        //solver.analyzePattern(mat);
    }

    void factorize(const Eigen::SparseMatrix<double>& mat) override {

        solver.compute(mat);

        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("LeastSquaresConjugateGradient decomposition failed");
        }
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& b) override {
        return solver.solve(b);
    }

    Eigen::ComputationInfo info() const override {
        return solver.info();
    }

private:
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double, Eigen::RowMajor>, PreConditioner> solver;
};

cadet::linalg::EigenSolverBase* setLinearSolver(const std::string solverName);

} // namespace linalg

} // namespace cadet

#endif  // LIBCADET_EigenSolverWrapper_HPP_