// FEMA C++ Example: synthesize data and run FEMA_fit
//
// This mirrors the MATLAB FEMA_run_on_synthetic_data.m demo.
// It creates synthetic imaging data with known variance components,
// fits the mixed-effects model, and verifies recovery of true parameters.

#include "fema.h"
#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main() {
    using namespace fema;
    using Clock = std::chrono::steady_clock;

    std::cout << "=== FEMA C++ Example: Synthetic Data ===" << std::endl;

    // ── Simulation parameters ────────────────────────────────────────────
    const int n_subjects  = 500;
    const int n_families  = 200;
    const int n_visits    = 2;
    const int n_obs       = n_subjects * n_visits;
    const int n_voxels    = 100;
    const int n_pred      = 3;  // intercept + 2 covariates

    std::mt19937 rng(42);
    std::normal_distribution<double> norm(0, 1);
    std::uniform_real_distribution<double> unif(0, 1);
    std::uniform_int_distribution<int> fam_dist(0, n_families - 1);

    // ── Build subject/family structure ───────────────────────────────────
    std::vector<std::string> iid(n_obs), eid(n_obs);
    std::vector<int> fid(n_obs);
    Vec agevec(n_obs);

    // Assign subjects to families (2-3 members per family)
    std::vector<int> subj_to_fam(n_subjects);
    {
        int fi = 0;
        for (int s = 0; s < n_subjects; ++s) {
            subj_to_fam[s] = fi;
            // ~60% families have 2 members, ~40% have 3
            if (s > 0 && (fi < n_families - 1)) {
                int prev_count = 0;
                for (int k = 0; k < s; ++k)
                    if (subj_to_fam[k] == fi) prev_count++;
                int fam_size = (fi % 5 < 3) ? 2 : 3;
                if (prev_count >= fam_size) fi++;
            }
        }
    }

    for (int v = 0; v < n_visits; ++v) {
        for (int s = 0; s < n_subjects; ++s) {
            int idx = v * n_subjects + s;
            iid[idx] = "S" + std::to_string(s);
            eid[idx] = "visit" + std::to_string(v + 1);
            fid[idx] = subj_to_fam[s];
            agevec(idx) = 10.0 + v * 2.0 + norm(rng) * 0.5;
        }
    }

    // ── Design matrix: intercept + age + random covariate ────────────────
    Mat X(n_obs, n_pred);
    X.col(0).setOnes();
    X.col(1) = agevec;
    for (int i = 0; i < n_obs; ++i)
        X(i, 2) = norm(rng);

    // ── Synthesize imaging data ──────────────────────────────────────────
    std::cout << "\nSynthesizing data (n=" << n_obs
              << ", v=" << n_voxels << ")..." << std::endl;

    auto t0 = Clock::now();
    SynthResult synth = synthesize(
        X, iid, eid, fid, agevec,
        n_voxels, Mat(),  // no GRM
        {RFX::F, RFX::S, RFX::E},
        20, 1.0
    );
    auto t1 = Clock::now();
    std::cout << "  Synthesis: "
              << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;

    // Add fixed effects signal: ymat = X * beta_true (replicated across voxels)
    Vec beta_true(n_pred);
    beta_true << 0.0, 0.5, -0.3; // intercept=0, age=0.5, covariate=-0.3
    Vec signal = X * beta_true;   // [n_obs x 1]
    synth.ymat.colwise() += signal;

    // ── Run FEMA_fit ─────────────────────────────────────────────────────
    std::cout << "\nRunning FEMA_fit..." << std::endl;

    FitConfig cfg;
    cfg.random_effects = {RFX::F, RFX::S, RFX::E};
    cfg.fixed_est_type = FixedEstType::GLS;
    cfg.rand_est_type  = RandEstType::MoM;
    cfg.nonneg_flag    = true;
    cfg.nbins          = 20;
    cfg.niter          = 1;
    cfg.nperms         = 0;

    auto t2 = Clock::now();
    FitResult result = fit(X, iid, eid, fid, agevec, synth.ymat,
                           Mat(), Mat(), cfg);
    auto t3 = Clock::now();
    std::cout << "  FEMA_fit: "
              << std::chrono::duration<double>(t3 - t2).count() << "s" << std::endl;

    // ── Print results ────────────────────────────────────────────────────
    std::cout << "\n=== Results ===" << std::endl;

    // Beta coefficients (mean across voxels)
    std::cout << "\nEstimated beta (mean across voxels):" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    int n_rows_print = std::min(static_cast<int>(result.beta_hat.rows()), n_pred);
    for (int p = 0; p < n_rows_print; ++p) {
        double mean_beta = result.beta_hat.row(p).mean();
        double mean_se   = result.beta_se.row(p).mean();
        double mean_z    = result.zmat.row(p).mean();
        std::cout << "  beta[" << p << "] = " << mean_beta
                  << " (SE=" << mean_se
                  << ", z=" << mean_z
                  << ", true=" << beta_true(p) << ")" << std::endl;
    }

    // Variance components (mean across voxels)
    std::cout << "\nEstimated variance components (mean across voxels):" << std::endl;
    std::vector<std::string> rfx_names = {"F", "S", "E"};
    for (int r = 0; r < std::min(static_cast<int>(result.sig2mat.rows()), 3); ++r) {
        double mean_sig2  = result.sig2mat.row(r).mean();
        double true_sig2  = synth.sig2mat_true.row(r).mean();
        std::cout << "  sig2[" << rfx_names[r] << "] = " << mean_sig2
                  << " (true=" << true_sig2 << ")" << std::endl;
    }
    std::cout << "  sig2t (mean) = " << result.sig2tvec.mean()
              << " (true=" << synth.sig2tvec_true.mean() << ")" << std::endl;

    // ── Wald test example ────────────────────────────────────────────────
    std::cout << "\n=== Wald Test (age effect) ===" << std::endl;
    std::vector<Mat> L = {Mat::Zero(1, n_pred)};
    L[0](0, 1) = 1.0; // test beta[1] (age)

    WaldResult wald = wald_test(L, result.beta_hat, result.coeffCovar);
    double mean_W    = wald.W.mean();
    double mean_logp = wald.logp.mean();
    std::cout << "  Mean W = " << mean_W << std::endl;
    std::cout << "  Mean -log10(p) = " << mean_logp << std::endl;

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
