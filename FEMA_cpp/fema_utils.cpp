// FEMA utility functions: linear algebra helpers, statistical primitives
#include "fema.h"
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace fema {

// ─── RFX name lookup ─────────────────────────────────────────────────────────
RFX rfx_from_name(const std::string& s) {
    static const std::unordered_map<std::string, RFX> m = {
        {"F", RFX::F}, {"S", RFX::S}, {"E", RFX::E},
        {"A", RFX::A}, {"D", RFX::D}, {"M", RFX::M},
        {"P", RFX::P}, {"H", RFX::H}, {"T", RFX::T}
    };
    auto it = m.find(s);
    if (it == m.end()) throw std::invalid_argument("Unknown random effect: " + s);
    return it->second;
}

// ─── ClusterInfo::V accessor ─────────────────────────────────────────────────
const Mat& ClusterInfo::V(RFX r) const {
    switch (r) {
        case RFX::F: return V_F; case RFX::S: return V_S; case RFX::E: return V_E;
        case RFX::A: return V_A; case RFX::D: return V_D; case RFX::M: return V_M;
        case RFX::P: return V_P; case RFX::H: return V_H; case RFX::T: return V_T;
        default: throw std::invalid_argument("Invalid RFX");
    }
}

// ─── Normal CDF (Abramowitz & Stegun approximation) ──────────────────────────
double normcdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// ─── Normal log-CDF (high precision for tails) ──────────────────────────────
double normlogcdf(double x) {
    if (x > -6.0) {
        double p = normcdf(x);
        return (p > 0.0) ? std::log(p) : -std::numeric_limits<double>::infinity();
    }
    // Mill's ratio expansion for extreme left tail
    double x2 = x * x;
    double log_phi = -0.5 * x2 - 0.5 * std::log(2.0 * M_PI);
    // Phi(x) ≈ phi(x)/|x| * (1 - 1/x^2 + 3/x^4 - ...)
    double abs_x = std::fabs(x);
    double series = 1.0 - 1.0/x2 + 3.0/(x2*x2);
    return log_phi - std::log(abs_x) + std::log(series);
}

// ─── Non-negative least squares (column-wise, efficient) ─────────────────────
// Solves min||Ax - b||_2  s.t. x >= 0 for each column of B
// Uses the active-set method similar to Lawson & Hanson
Mat lsqnonneg(const Mat& A, const Mat& B) {
    const int m = A.rows();
    const int n = A.cols();
    const int ncols = B.cols();
    Mat result = Mat::Zero(n, ncols);

    // Precompute A^T A and use it for all columns
    Mat AtA = A.transpose() * A;

    // For each column of B, solve independently
    #pragma omp parallel for schedule(dynamic)
    for (int col = 0; col < ncols; ++col) {
        Vec b = B.col(col);
        Vec AtB = A.transpose() * b;

        // Start: unconstrained solution
        Vec x = AtA.ldlt().solve(AtB);

        // If all non-negative, done
        bool all_pos = true;
        for (int i = 0; i < n; ++i) {
            if (x(i) < 0.0) { all_pos = false; break; }
        }
        if (all_pos) {
            result.col(col) = x;
            continue;
        }

        // Active set: try combinations with zero-clamping
        // Efficient approach: iteratively clamp negative components
        std::vector<bool> passive(n, true);
        x = Vec::Zero(n);
        Vec w = AtB;  // gradient

        for (int iter = 0; iter < 3 * n; ++iter) {
            // Find the most violating variable in the active set
            int best_idx = -1;
            double best_w = 0.0;
            for (int i = 0; i < n; ++i) {
                if (!passive[i] && w(i) > best_w) {
                    best_w = w(i);
                    best_idx = i;
                }
            }
            if (best_idx < 0) break; // KKT satisfied

            passive[best_idx] = true;

            // Inner loop: solve restricted problem
            for (int inner = 0; inner < 3 * n; ++inner) {
                // Collect passive indices
                std::vector<int> pidx;
                for (int i = 0; i < n; ++i) if (passive[i]) pidx.push_back(i);

                // Solve sub-problem
                int np = pidx.size();
                Mat Ap(m, np);
                for (int j = 0; j < np; ++j) Ap.col(j) = A.col(pidx[j]);
                Vec sp = (Ap.transpose() * Ap).ldlt().solve(Ap.transpose() * b);

                // Check for feasibility
                bool feasible = true;
                for (int j = 0; j < np; ++j) {
                    if (sp(j) <= 0.0) { feasible = false; break; }
                }
                if (feasible) {
                    for (int j = 0; j < np; ++j) x(pidx[j]) = sp(j);
                    break;
                }

                // Find alpha
                double alpha = std::numeric_limits<double>::max();
                int q = -1;
                for (int j = 0; j < np; ++j) {
                    if (sp(j) <= 0.0) {
                        double a = x(pidx[j]) / (x(pidx[j]) - sp(j));
                        if (a < alpha) { alpha = a; q = j; }
                    }
                }

                // Update x
                for (int j = 0; j < np; ++j) {
                    x(pidx[j]) += alpha * (sp(j) - x(pidx[j]));
                }

                // Move binding variables to active set
                for (int j = 0; j < np; ++j) {
                    if (std::fabs(x(pidx[j])) < 1e-15) {
                        passive[pidx[j]] = false;
                        x(pidx[j]) = 0.0;
                    }
                }
            }

            w = AtB - AtA * x;
        }

        result.col(col) = x;
    }
    return result;
}

// ─── Nearest symmetric positive-definite matrix (Higham 1988) ────────────────
Mat nearest_spd(const Mat& A) {
    int n = A.rows();
    if (n != A.cols()) throw std::invalid_argument("nearest_spd: matrix must be square");

    // Symmetrize
    Mat B = 0.5 * (A + A.transpose());

    // Symmetric polar factor H
    Eigen::JacobiSVD<Mat> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat H = svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();

    Mat Ahat = 0.5 * (B + H);
    Ahat = 0.5 * (Ahat + Ahat.transpose()); // ensure symmetry

    // Test positive-definiteness; if not, nudge eigenvalues
    Eigen::LLT<Mat> llt(Ahat);
    if (llt.info() == Eigen::Success) return Ahat;

    Eigen::SelfAdjointEigenSolver<Mat> eig(Ahat);
    Vec D = eig.eigenvalues();
    double min_eig = D.minCoeff();
    if (min_eig < 0.0) {
        double k = 0.0;
        // Add small multiple of identity
        for (int i = 0; i < n; ++i) {
            if (D(i) < 0.0) D(i) = 0.0;
        }
        double eps_val = std::numeric_limits<double>::epsilon() * D.maxCoeff();
        for (int i = 0; i < n; ++i) {
            if (D(i) < eps_val) D(i) = eps_val;
        }
        Ahat = eig.eigenvectors() * D.asDiagonal() * eig.eigenvectors().transpose();
        Ahat = 0.5 * (Ahat + Ahat.transpose());
    }
    return Ahat;
}

// ─── Multivariate normal log-pdf ─────────────────────────────────────────────
double mvnpdf_ln(const Vec& x, const Vec& mu, const Mat& Sigma) {
    int k = x.size();
    Vec diff = x - mu;

    Eigen::LLT<Mat> llt(Sigma);
    if (llt.info() != Eigen::Success) {
        // Fallback to LDLT
        Eigen::LDLT<Mat> ldlt(Sigma);
        if (ldlt.info() != Eigen::Success) return -std::numeric_limits<double>::infinity();
        Vec alpha = ldlt.solve(diff);
        double logdet = 0.0;
        auto D = ldlt.vectorD();
        for (int i = 0; i < k; ++i) {
            if (D(i) <= 0) return -std::numeric_limits<double>::infinity();
            logdet += std::log(D(i));
        }
        return -0.5 * (k * std::log(2.0 * M_PI) + logdet + diff.dot(alpha));
    }

    // Solve L * z = diff
    Mat L = llt.matrixL();
    Vec z = L.triangularView<Eigen::Lower>().solve(diff);

    // log|Sigma| = 2 * sum(log(diag(L)))
    double logdet = 0.0;
    for (int i = 0; i < k; ++i) logdet += std::log(L(i, i));
    logdet *= 2.0;

    return -0.5 * (k * std::log(2.0 * M_PI) + logdet + z.squaredNorm());
}

// ─── Log-likelihood ──────────────────────────────────────────────────────────
double log_likelihood(
    const Vec&                      sig2vec,
    const Mat&                      X,
    const Vec&                      yvec_res,
    const std::vector<ClusterInfo>& clusterinfo,
    const std::vector<SpMat>&       Ss)
{
    int nrfx = sig2vec.size();
    if (nrfx != static_cast<int>(Ss.size()))
        throw std::invalid_argument("sig2vec length must match Ss length");

    for (int i = 0; i < nrfx; ++i)
        if (sig2vec(i) < 0.0) return std::numeric_limits<double>::quiet_NaN();

    // Build combined covariance: Sigma = sum_r sig2vec(r) * Ss[r]
    // We evaluate per-family for efficiency
    double loglike = 0.0;
    for (const auto& ci : clusterinfo) {
        int nj = ci.jvec_fam.size();
        Mat Sigma_fam = Mat::Zero(nj, nj);

        for (int r = 0; r < nrfx; ++r) {
            // Extract sub-block from sparse Ss[r]
            for (int ii = 0; ii < nj; ++ii)
                for (int jj = 0; jj < nj; ++jj)
                    Sigma_fam(ii, jj) += sig2vec(r) * Ss[r].coeff(ci.jvec_fam[ii], ci.jvec_fam[jj]);
        }

        Vec y_fam(nj);
        for (int ii = 0; ii < nj; ++ii) y_fam(ii) = yvec_res(ci.jvec_fam[ii]);

        double ll = mvnpdf_ln(y_fam, Vec::Zero(nj), Sigma_fam);
        if (!std::isfinite(ll)) return std::numeric_limits<double>::quiet_NaN();
        loglike += ll;
    }
    return loglike;
}

// ─── Wald test ───────────────────────────────────────────────────────────────
WaldResult wald_test(
    const std::vector<Mat>&  L,
    const Mat&               beta_hat,
    const std::vector<Mat>&  coeffCovar,
    const Vec&               hypValue,
    bool                     doF,
    int                      numObs)
{
    int numCells = L.size();
    int numX = beta_hat.rows();
    int numY = beta_hat.cols();

    WaldResult res;
    res.W    = Mat::Zero(numCells, numY);
    res.p    = Mat::Zero(numCells, numY);
    res.logp = Mat::Zero(numCells, numY);
    res.df   = VecI::Zero(numCells);
    res.LB_hat.resize(numCells);
    res.LB_SE.resize(numCells);

    Vec hyp = hypValue;
    if (hyp.size() == 0) hyp = Vec::Zero(numCells);

    for (int cc = 0; cc < numCells; ++cc) {
        // Pad contrast if needed
        Mat Lc = L[cc];
        if (Lc.cols() < numX) {
            Mat padded = Mat::Zero(Lc.rows(), numX);
            padded.leftCols(Lc.cols()) = Lc;
            Lc = padded;
        }

        // Numerator dof = rank of L
        Eigen::FullPivLU<Mat> lu(Lc);
        res.df(cc) = lu.rank();

        res.LB_hat[cc] = Mat::Zero(Lc.rows(), numY);
        res.LB_SE[cc]  = Mat::Zero(Lc.rows(), numY);

        for (int yy = 0; yy < numY; ++yy) {
            Vec LB = Lc * beta_hat.col(yy) - Vec::Constant(Lc.rows(), hyp(cc));
            res.LB_hat[cc].col(yy) = LB;

            Mat innerTerm = Lc * coeffCovar[yy] * Lc.transpose();
            res.LB_SE[cc].col(yy) = innerTerm.diagonal().cwiseSqrt();

            // W = LB' * inv(L * Cov * L') * LB
            Eigen::FullPivLU<Mat> lu_inner(innerTerm);
            double w;
            if (lu_inner.rank() < innerTerm.cols()) {
                w = LB.transpose() * innerTerm.completeOrthogonalDecomposition().pseudoInverse() * LB;
            } else {
                w = LB.transpose() * innerTerm.ldlt().solve(LB);
            }
            res.W(cc, yy) = w;
        }

        // Compute p-values
        for (int yy = 0; yy < numY; ++yy) {
            double w = res.W(cc, yy);
            double pval;
            if (doF) {
                w /= res.df(cc);
                res.W(cc, yy) = w;
                // F distribution CDF upper tail: use incomplete beta
                // p = 1 - F_cdf(w, df1, df2)
                // For simplicity, use chi2 approximation: F*df1 ~ chi2(df1) when df2 large
                int df2 = numObs - numX;
                double chi2_approx = w * res.df(cc);
                // Gamma-based chi2 survival: use normal approx for large df
                double z = std::sqrt(2.0 * chi2_approx) - std::sqrt(2.0 * res.df(cc) - 1.0);
                pval = 1.0 - normcdf(z);
            } else {
                // Chi-squared upper tail p-value via normal approximation
                // Wilson-Hilferty approximation
                double k = res.df(cc);
                double z = std::cbrt(w / k) * (1.0 - 2.0 / (9.0 * k))
                         - (1.0 - 2.0 / (9.0 * k));
                z /= std::sqrt(2.0 / (9.0 * k));
                pval = 1.0 - normcdf(z);
            }
            res.p(cc, yy) = pval;
        }
    }

    // -log10(p)
    double realmin = std::numeric_limits<double>::min();
    for (int cc = 0; cc < numCells; ++cc)
        for (int yy = 0; yy < numY; ++yy)
            res.logp(cc, yy) = -std::log10(std::max(realmin, res.p(cc, yy)));

    return res;
}

} // namespace fema
