// FEMA_fit - Core fitting engine for Fast and Efficient Mixed-effects Algorithm
//
// Implements OLS initial fit, Method-of-Moments variance component estimation,
// binning across the random-effects grid, GLS solution, and optional
// wild-bootstrap permutation testing.
#include "fema.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>

namespace fema {

// ─── Internal helpers ────────────────────────────────────────────────────────

// Compute V = sum_r sig2vec(r) * V_r for a given family
static Mat compute_V(const ClusterInfo& ci, const Vec& sig2vec,
                     const std::vector<RFX>& rfx) {
    int nj = ci.jvec_fam.size();
    Mat V = Mat::Zero(nj, nj);
    for (int r = 0; r < static_cast<int>(rfx.size()); ++r) {
        const Mat& Vr = ci.V(rfx[r]);
        if (Vr.rows() > 0) V.noalias() += sig2vec(r) * Vr;
    }
    return V;
}

// Safe matrix inverse with fallback to pseudoinverse
static Mat safe_inverse(const Mat& A) {
    Eigen::LLT<Mat> llt(A);
    if (llt.info() == Eigen::Success) return llt.solve(Mat::Identity(A.rows(), A.cols()));

    Eigen::LDLT<Mat> ldlt(A);
    if (ldlt.info() == Eigen::Success && ldlt.isPositive())
        return ldlt.solve(Mat::Identity(A.rows(), A.cols()));

    // Pseudoinverse fallback
    return A.completeOrthogonalDecomposition().pseudoInverse();
}

// ─── Main fit function ───────────────────────────────────────────────────────
FitResult fit(
    const Mat&                      X,
    const std::vector<std::string>& iid,
    const std::vector<std::string>& eid,
    const std::vector<int>&         fid,
    const Vec&                      agevec,
    const Mat&                      ymat,
    const Mat&                      contrasts,
    const Mat&                      GRM,
    const FitConfig&                cfg,
    FamilyStruct*                   precomputed_fs)
{
    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    // ── Validate inputs ──────────────────────────────────────────────────
    int num_obs = X.rows();
    int num_X   = X.cols();
    int num_y   = ymat.cols();

    // NaN/Inf check
    if (!X.array().isFinite().all() || !ymat.array().isFinite().all())
        throw std::runtime_error("X and/or ymat contain NaN or Inf");
    assert(ymat.rows() == num_obs);

    // Ensure E is last in random effects
    std::vector<RFX> random_effects = cfg.random_effects;
    {
        auto it = std::find(random_effects.begin(), random_effects.end(), RFX::E);
        if (it == random_effects.end()) {
            random_effects.push_back(RFX::E);
        } else if (it != random_effects.end() - 1) {
            random_effects.erase(it);
            random_effects.push_back(RFX::E);
        }
    }
    int num_RFX = random_effects.size();

    // GroupByFamType only supported for {F, S, E}
    bool group_by_fam = cfg.group_by_fam;
    bool unstructuredCov = (cfg.cov_type == CovType::Unstructured);
    {
        std::unordered_set<int> fse_set = {
            static_cast<int>(RFX::F), static_cast<int>(RFX::S), static_cast<int>(RFX::E)
        };
        for (auto r : random_effects) {
            if (fse_set.find(static_cast<int>(r)) == fse_set.end()) {
                group_by_fam = false;
                break;
            }
        }
    }
    if (unstructuredCov) group_by_fam = false;

    bool OLSflag = (cfg.fixed_est_type == FixedEstType::OLS);
    int nperms = cfg.nperms;
    if (unstructuredCov && nperms > 0) {
        std::cerr << "Warning: Permutations not implemented for unstructured covariance\n";
        nperms = 0;
    }
    int nbins = cfg.nbins;
    int niter = cfg.niter;

    // Check rank
    Eigen::FullPivLU<Mat> lu_X(X);
    bool lowRank = lu_X.rank() < num_X;

    // Model singularity index
    {
        Mat XtX = X.transpose() * X;
        Mat dXtX = XtX.diagonal().asDiagonal();
        Eigen::JacobiSVD<Mat> svd1(XtX, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::JacobiSVD<Mat> svd2(dXtX, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double c1 = svd1.singularValues()(0) / svd1.singularValues()(svd1.singularValues().size()-1);
        double c2 = svd2.singularValues()(0) / svd2.singularValues()(svd2.singularValues().size()-1);
        std::cout << "ModelSingularityIndex = " << c1/c2 << std::endl;
    }

    // Zero-pad contrasts
    Mat C = contrasts;
    if (C.rows() > 0 && C.cols() < num_X) {
        Mat Cpad = Mat::Zero(C.rows(), num_X);
        Cpad.leftCols(C.cols()) = C;
        C = Cpad;
    }
    int num_C = C.rows();

    // ── Parse family structure ───────────────────────────────────────────
    FamilyStruct fs;
    if (precomputed_fs) {
        fs = *precomputed_fs;
    } else {
        fs = parse_family(iid, eid, fid, agevec, GRM, random_effects, nbins);
    }

    auto& clusterinfo = fs.clusterinfo;
    int nfam = fs.nfam;
    int nfamtypes = fs.nfamtypes;
    auto& famtypevec = fs.famtypevec;
    auto& sig2grid  = fs.sig2grid;
    auto& sig2gridl = fs.sig2gridl;
    auto& sig2gridu = fs.sig2gridu;
    int nsig2bins   = fs.nsig2bins;

    // Compute Mi = pinv(M) for MoM estimator
    Mat Mi = fs.M.completeOrthogonalDecomposition().pseudoInverse();

    // ── Initialize outputs ───────────────────────────────────────────────
    FitResult res;
    Mat beta_hat    = Mat::Zero(num_X, num_y);
    Mat beta_se     = Mat::Zero(num_X, num_y);
    Mat ymat_hat    = Mat::Zero(num_obs, num_y);
    Mat ymat_res    = Mat::Zero(num_obs, num_y);
    Mat betacon_hat = Mat::Zero(num_C, num_y);
    Mat betacon_se  = Mat::Zero(num_C, num_y);
    VecI binvec     = VecI::Constant(num_y, -1);
    Mat sig2mat;
    Vec sig2tvec;
    Vec logLikvec;
    std::vector<Mat> coeffCovar(num_y);

    // Permutation storage
    std::vector<Mat> beta_hat_perm, beta_se_perm, zmat_perm, sig2mat_perm;
    std::vector<Vec> sig2tvec_perm;

    // RNG for permutations
    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    // Backups for permutation
    Mat sig2mat_bak, ymat_bak, ymat_res_bak, ymat_hat_bak;
    Vec sig2tvec_bak;
    VecI binvec_bak;

    // ── Main loop: perm 0 = original, perm 1..nperms = permuted ─────────
    for (int permi = 0; permi <= nperms; ++permi) {
        auto t_perm = Clock::now();

        // ── Initialize permutation storage on first perm ─────────────
        if (permi == 1) {
            sig2mat_bak  = sig2mat;
            sig2tvec_bak = sig2tvec;
            binvec_bak   = binvec;
            ymat_bak     = ymat;
            ymat_res_bak = ymat_res;

            if (cfg.perm_type == PermType::WildBootstrap) {
                ymat_hat_bak = Mat::Zero(ymat.rows(), ymat.cols());
            } else {
                ymat_hat_bak = ymat_hat;
            }

            beta_hat_perm.resize(nperms + 1);
            beta_se_perm.resize(nperms + 1);
            zmat_perm.resize(nperms + 1);
            sig2mat_perm.resize(nperms + 1);
            sig2tvec_perm.resize(nperms + 1);
        }

        // ── Perform wild bootstrap resampling ────────────────────────
        // Work on a mutable copy for permi > 0
        Mat ymat_work;
        if (permi == 0) {
            ymat_work = ymat;
        } else {
            ymat_work = ymat_hat_bak; // start from fitted or zero
            for (int fi = 0; fi < nfam; ++fi) {
                double w = norm_dist(rng); // family-level wild weight
                for (int j : clusterinfo[fi].jvec_fam) {
                    ymat_work.row(j) += w * ymat_res_bak.row(j);
                }
            }
        }

        // ── OLS initial fit ──────────────────────────────────────────
        Mat XtX = X.transpose() * X;
        Mat iXtX;
        if (lowRank) {
            iXtX = XtX.completeOrthogonalDecomposition().pseudoInverse();
        } else {
            iXtX = XtX.ldlt().solve(Mat::Identity(num_X, num_X));
        }
        beta_hat = iXtX * (X.transpose() * ymat_work);
        ymat_hat = X * beta_hat;
        ymat_res = ymat_work - ymat_hat;

        sig2tvec = (ymat_res.array().square().colwise().sum() / (num_obs - num_X)).transpose();
        beta_se  = (iXtX.diagonal() * sig2tvec.transpose()).cwiseSqrt();

        Mat Cov_beta = iXtX;

        // Coefficient covariance (OLS)
        if (permi == 0) {
            for (int v = 0; v < num_y; ++v)
                coeffCovar[v] = Cov_beta * sig2tvec(v);
        }

        // Contrasts
        for (int ci2 = 0; ci2 < num_C; ++ci2) {
            betacon_hat.row(ci2) = C.row(ci2) * beta_hat;
            double cvc = (C.row(ci2) * Cov_beta * C.row(ci2).transpose())(0, 0);
            betacon_se.row(ci2) = (sig2tvec.transpose() * cvc).cwiseSqrt().transpose();
        }

        // ── Iterative refinement: MoM → binning → GLS ───────────────
        for (int iter = 1; iter <= std::max(1, niter); ++iter) {

            // ── Method of Moments ────────────────────────────────────
            sig2tvec = (ymat_res.array().square().colwise().sum() / (num_obs - num_X)).transpose();

            // LHS = res(subvec1) .* res(subvec2) ./ sig2tvec
            int nind = fs.subvec1.size();
            Mat LHS(nind, num_y);

            #pragma omp parallel for schedule(static)
            for (int idx = 0; idx < nind; ++idx) {
                int i1 = fs.subvec1(idx);
                int i2 = fs.subvec2(idx);
                for (int v = 0; v < num_y; ++v) {
                    LHS(idx, v) = ymat_res(i1, v) * ymat_res(i2, v) / sig2tvec(v);
                }
            }

            // Solve for variance components
            if (cfg.nonneg_flag) {
                sig2mat = lsqnonneg(fs.M, LHS);
                sig2mat = sig2mat.cwiseMax(0.0);
            } else {
                sig2mat = Mi * LHS;
                sig2mat = sig2mat.cwiseMax(0.0);
            }

            // Normalize: each column sums to 1
            for (int v = 0; v < num_y; ++v) {
                double s = std::max(std::numeric_limits<double>::epsilon(), sig2mat.col(v).sum());
                sig2mat.col(v) /= s;
            }

            // ── Binning ──────────────────────────────────────────────
            Vec nvec_bins, tvec_bins;
            if (unstructuredCov || nbins == 0) {
                binvec = VecI::LinSpaced(num_y, 0, num_y - 1);
                nvec_bins = Vec::Zero(num_y);
                tvec_bins = Vec::Zero(num_y);
            } else {
                nvec_bins = Vec::Zero(nsig2bins);
                tvec_bins = Vec::Zero(nsig2bins);
                binvec    = VecI::Constant(num_y, -1);

                for (int bi = 0; bi < nsig2bins; ++bi) {
                    for (int v = 0; v < num_y; ++v) {
                        bool in_bin = true;
                        for (int ri = 0; ri < num_RFX - 1; ++ri) {
                            if (sig2mat(ri, v) < sig2gridl(bi, ri) ||
                                sig2mat(ri, v) >= sig2gridu(bi, ri)) {
                                in_bin = false;
                                break;
                            }
                        }
                        if (in_bin) {
                            binvec(v) = bi;
                            nvec_bins(bi) += 1;
                        }
                    }
                }

                // Coerce unassigned bins to nearest
                for (int v = 0; v < num_y; ++v) {
                    if (binvec(v) < 0) {
                        double min_diff = std::numeric_limits<double>::max();
                        int best_bin = 0;
                        for (int bi = 0; bi < nsig2bins; ++bi) {
                            double diff = 0.0;
                            for (int ri = 0; ri < num_RFX - 1; ++ri)
                                diff += std::fabs(sig2mat(ri, v) - sig2grid(bi, ri));
                            if (diff < min_diff) { min_diff = diff; best_bin = bi; }
                        }
                        binvec(v) = best_bin;
                    }
                }
            }

            // Log-likelihood (optional)
            logLikvec = Vec();
            if (cfg.log_lik_flag) {
                logLikvec.resize(num_y);
                #pragma omp parallel for schedule(dynamic)
                for (int v = 0; v < num_y; ++v) {
                    Vec sv = sig2tvec(v) * sig2mat.col(v);
                    logLikvec(v) = log_likelihood(sv, X, ymat_res.col(v),
                                                  clusterinfo, fs.Ss);
                }
            }

            // Save current estimates for permutation reference
            Mat sig2mat_save  = sig2mat;
            Vec sig2tvec_save = sig2tvec;

            if (permi == 0) {
                res.binvec_save = binvec;
                res.nvec_bins = nvec_bins;
                res.tvec_bins = tvec_bins;
            }

            // For permutations > 0, use original variance estimates
            if (permi > 0) {
                sig2tvec = sig2tvec_bak;
                sig2mat  = sig2mat_bak;
                binvec   = binvec_bak;
            }

            if (iter > niter) break;

            // ── GLS solution ─────────────────────────────────────────
            betacon_hat = Mat::Zero(num_C, num_y);
            betacon_se  = Mat::Zero(num_C, num_y);
            beta_hat    = Mat::Zero(num_X, num_y);
            beta_se     = Mat::Zero(num_X, num_y);
            if (permi == 0) {
                for (int v = 0; v < num_y; ++v)
                    coeffCovar[v] = Mat::Zero(num_X, num_X);
            }

            // Collect unique bins
            std::vector<int> unique_bins;
            {
                std::unordered_set<int> seen;
                for (int v = 0; v < num_y; ++v) {
                    if (binvec(v) >= 0 && seen.insert(binvec(v)).second)
                        unique_bins.push_back(binvec(v));
                }
            }

            for (int bi : unique_bins) {
                auto t_bin = Clock::now();

                // Indices in this bin
                std::vector<int> ivec_bin;
                for (int v = 0; v < num_y; ++v)
                    if (binvec(v) == bi) ivec_bin.push_back(v);

                if (ivec_bin.empty()) continue;

                // Average sig2vec for this bin
                Vec sig2vec = Vec::Zero(num_RFX);
                for (int v : ivec_bin) sig2vec += sig2mat.col(v);
                sig2vec /= ivec_bin.size();

                if (OLSflag) {
                    // OLS path
                    Mat XtX2 = X.transpose() * X;
                    Mat iXtX2;
                    if (lowRank) iXtX2 = XtX2.completeOrthogonalDecomposition().pseudoInverse();
                    else iXtX2 = XtX2.ldlt().solve(Mat::Identity(num_X, num_X));

                    for (int v : ivec_bin) {
                        beta_hat.col(v) = iXtX2 * (X.transpose() * ymat_work.col(v));
                        beta_se.col(v)  = (iXtX2.diagonal() * sig2tvec(v)).cwiseSqrt();
                    }
                    Cov_beta = iXtX2;
                } else {
                    // GLS path: compute W = V^{-1} per family/famtype
                    // XtW = X' * W, B = XtW * X, Bi = B^{-1}
                    Mat XtW = Mat::Zero(num_X, num_obs);

                    if (group_by_fam) {
                        // Pre-compute one W per family type
                        std::vector<Mat> Ws_famtype(nfamtypes);
                        for (int ft = 0; ft < nfamtypes; ++ft) {
                            // Find first family of this type
                            int rep = -1;
                            for (int fi = 0; fi < nfam; ++fi) {
                                if (famtypevec[fi] == ft) { rep = fi; break; }
                            }
                            if (rep < 0) continue;

                            Mat V = compute_V(clusterinfo[rep], sig2vec, random_effects);
                            Ws_famtype[ft] = safe_inverse(V);
                        }

                        for (int fi = 0; fi < nfam; ++fi) {
                            auto& ci2 = clusterinfo[fi];
                            int nj = ci2.jvec_fam.size();
                            // X_fam' * W_famtype
                            Mat X_fam(nj, num_X);
                            for (int r = 0; r < nj; ++r)
                                X_fam.row(r) = X.row(ci2.jvec_fam[r]);

                            Mat prod = X_fam.transpose() * Ws_famtype[famtypevec[fi]];
                            for (int r = 0; r < num_X; ++r)
                                for (int c = 0; c < nj; ++c)
                                    XtW(r, ci2.jvec_fam[c]) = prod(r, c);
                        }
                    } else {
                        // Compute W per family
                        for (int fi = 0; fi < nfam; ++fi) {
                            auto& ci2 = clusterinfo[fi];
                            int nj = ci2.jvec_fam.size();

                            Mat V = compute_V(ci2, sig2vec, random_effects);
                            Mat Wi = safe_inverse(V);

                            Mat X_fam(nj, num_X);
                            for (int r = 0; r < nj; ++r)
                                X_fam.row(r) = X.row(ci2.jvec_fam[r]);

                            Mat prod = X_fam.transpose() * Wi;
                            for (int r = 0; r < num_X; ++r)
                                for (int c = 0; c < nj; ++c)
                                    XtW(r, ci2.jvec_fam[c]) = prod(r, c);
                        }
                    }

                    // B = XtW * X; Bi = inv(B)
                    Mat B = XtW * X;
                    Mat Bi;
                    Eigen::FullPivLU<Mat> lu_B(B);
                    if (lu_B.rank() < num_X) {
                        Bi = B.completeOrthogonalDecomposition().pseudoInverse();
                    } else {
                        Bi = B.ldlt().solve(Mat::Identity(num_X, num_X));
                    }

                    Cov_beta = nearest_spd(Bi);

                    // Compute beta for all voxels in this bin at once
                    // beta_hat(:, ivec_bin) = Bi * XtW * ymat(:, ivec_bin)
                    Mat y_bin(num_obs, ivec_bin.size());
                    for (size_t j = 0; j < ivec_bin.size(); ++j)
                        y_bin.col(j) = ymat_work.col(ivec_bin[j]);

                    Mat bhat = Cov_beta * (XtW * y_bin);
                    for (size_t j = 0; j < ivec_bin.size(); ++j) {
                        beta_hat.col(ivec_bin[j]) = bhat.col(j);
                        beta_se.col(ivec_bin[j])  = (Cov_beta.diagonal() * sig2tvec(ivec_bin[j])).cwiseSqrt();
                    }
                }

                // Save coefficient covariance
                if (permi == 0) {
                    for (int v : ivec_bin)
                        coeffCovar[v] = Cov_beta * sig2tvec(v);
                }

                // Evaluate contrasts
                for (int ci2 = 0; ci2 < num_C; ++ci2) {
                    double cvc = (C.row(ci2) * Cov_beta * C.row(ci2).transpose())(0, 0);
                    for (int v : ivec_bin) {
                        betacon_hat(ci2, v) = (C.row(ci2) * beta_hat.col(v))(0);
                        betacon_se(ci2, v)  = std::sqrt(cvc * sig2tvec(v));
                    }
                }

                auto t_bin_end = Clock::now();
                double bin_sec = std::chrono::duration<double>(t_bin_end - t_bin).count();
                if (permi == 0 && !unstructuredCov && nbins > 0)
                    tvec_bins(bi) = bin_sec;

            } // end bin loop

            // Update residuals
            ymat_hat = X * beta_hat;
            ymat_res = ymat_work - ymat_hat;

            // Restore saved variance components after GLS
            sig2mat  = sig2mat_save;
            sig2tvec = sig2tvec_save;

        } // end iter loop

        // ── Combine contrasts with beta_hat ──────────────────────────
        Mat full_beta, full_se;
        if (num_C > 0) {
            full_beta = Mat(num_C + num_X, num_y);
            full_beta.topRows(num_C) = betacon_hat;
            full_beta.bottomRows(num_X) = beta_hat;
            full_se = Mat(num_C + num_X, num_y);
            full_se.topRows(num_C) = betacon_se;
            full_se.bottomRows(num_X) = beta_se;
        } else {
            full_beta = beta_hat;
            full_se   = beta_se;
        }

        Mat zmat = full_beta.array() / full_se.array();

        // ── Store permutation results ────────────────────────────────
        if (nperms > 0) {
            beta_hat_perm[permi] = full_beta;
            beta_se_perm[permi]  = full_se;
            zmat_perm[permi]     = zmat;
            sig2mat_perm[permi]  = sig2mat;
            sig2tvec_perm[permi] = sig2tvec;

            if (permi > 0) {
                auto t_perm_end = Clock::now();
                double perm_sec = std::chrono::duration<double>(t_perm_end - t_perm).count();
                double elapsed = std::chrono::duration<double>(t_perm_end - t_start).count();
                double remaining = elapsed / permi * (nperms - permi);
                std::cout << "permi=" << permi << "/" << nperms
                          << " (" << perm_sec << "s - remaining " << remaining << "s)" << std::endl;
            }
        }

        // Save original fit results
        if (permi == 0 || nperms == 0) {
            res.beta_hat  = full_beta;
            res.beta_se   = full_se;
            res.sig2tvec  = sig2tvec;
            res.sig2mat   = sig2mat;
            res.logLikvec = logLikvec;
        }
    } // end perm loop

    // ── Extract final results ────────────────────────────────────────────
    if (nperms > 0) {
        res.beta_hat = beta_hat_perm[0];
        res.beta_se  = beta_se_perm[0];
        res.sig2mat  = sig2mat_perm[0];
        res.sig2tvec = sig2tvec_perm[0];
        res.beta_hat_perm = std::move(beta_hat_perm);
        res.beta_se_perm  = std::move(beta_se_perm);
        res.zmat_perm     = std::move(zmat_perm);
        res.sig2mat_perm  = std::move(sig2mat_perm);
        res.sig2tvec_perm = std::move(sig2tvec_perm);
    }

    // Compute z-scores and log-p
    res.zmat    = res.beta_hat.array() / res.beta_se.array();
    res.logpmat = Mat(res.zmat.rows(), res.zmat.cols());
    for (int r = 0; r < res.zmat.rows(); ++r) {
        for (int c = 0; c < res.zmat.cols(); ++c) {
            double z = res.zmat(r, c);
            // logp = -sign(z) * log10(2 * normcdf(-|z|))
            double log_p2;
            double az = std::fabs(z);
            if (az < 37.0) {
                log_p2 = std::log10(2.0 * normcdf(-az));
            } else {
                // Use log-space for extreme tails
                log_p2 = (normlogcdf(-az) + std::log(2.0)) / std::log(10.0);
            }
            res.logpmat(r, c) = (z >= 0 ? -1.0 : 1.0) * log_p2;
        }
    }

    res.coeffCovar    = std::move(coeffCovar);
    res.family_struct = std::move(fs);

    auto t_end = Clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "***Done*** (" << elapsed << " seconds)" << std::endl;

    return res;
}

// ─── Synthesize data for testing ─────────────────────────────────────────────
SynthResult synthesize(
    const Mat& X,
    const std::vector<std::string>& iid,
    const std::vector<std::string>& eid,
    const std::vector<int>&         fid,
    const Vec&                      agevec,
    int num_y,
    const Mat&                      GRM,
    const std::vector<RFX>&         random_effects,
    int nbins,
    double sig2tval)
{
    // Parse family structure
    FamilyStruct fs = parse_family(iid, eid, fid, agevec, GRM, random_effects, nbins);
    int nfam    = fs.nfam;
    int num_RFX = random_effects.size();

    // Assign true variance components across bins
    int nsig2bins = fs.nsig2bins;
    if (nsig2bins == 0) throw std::runtime_error("synthesize: no valid variance bins");

    VecI binvec_true(num_y);
    for (int v = 0; v < num_y; ++v)
        binvec_true(v) = std::min(nsig2bins - 1,
                                  static_cast<int>(nsig2bins * static_cast<double>(v) / num_y));

    Mat sig2mat_true(num_RFX, num_y);
    for (int v = 0; v < num_y; ++v) {
        for (int ri = 0; ri < num_RFX - 1; ++ri)
            sig2mat_true(ri, v) = fs.sig2grid(binvec_true(v), ri);
        sig2mat_true(num_RFX - 1, v) = std::max(0.0,
            1.0 - sig2mat_true.col(v).head(num_RFX - 1).sum());
    }

    Vec sig2tvec_true = Vec::Constant(num_y, sig2tval);

    // Generate data
    Mat ymat = Mat::Zero(X.rows(), num_y);
    std::mt19937 rng(123);

    for (int fi = 0; fi < nfam; ++fi) {
        auto& ci = fs.clusterinfo[fi];
        int nj = ci.jvec_fam.size();

        for (int ri = 0; ri < num_RFX; ++ri) {
            const Mat& Vr = ci.V(random_effects[ri]);
            if (Vr.rows() == 0) continue;

            // Cholesky of Vr for sampling
            Mat Vrd = Vr.cast<double>();
            Eigen::LLT<Mat> llt(Vrd);
            Mat L;
            if (llt.info() == Eigen::Success) {
                L = llt.matrixL();
            } else {
                // Fallback: eigendecomposition
                Eigen::SelfAdjointEigenSolver<Mat> eig(Vrd);
                Vec D = eig.eigenvalues().cwiseMax(0.0).cwiseSqrt();
                L = eig.eigenvectors() * D.asDiagonal();
            }

            // Sample from MVN(0, Vr) for each voxel
            for (int v = 0; v < num_y; ++v) {
                Vec z(nj);
                for (int j = 0; j < nj; ++j) z(j) = std::normal_distribution<double>(0, 1)(rng);
                Vec sample = L * z;
                double scale = std::sqrt(sig2tvec_true(v) * sig2mat_true(ri, v));
                for (int j = 0; j < nj; ++j)
                    ymat(ci.jvec_fam[j], v) += scale * sample(j);
            }
        }
    }

    return {ymat, sig2tvec_true, sig2mat_true};
}

} // namespace fema
