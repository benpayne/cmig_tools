// FEMA C++ Benchmark: measure OpenMP parallelization speedup
//
// Runs FEMA_fit with realistic synthetic data at different voxel counts,
// comparing OMP_NUM_THREADS=1 vs all available cores.

#include "fema.h"
#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace fema;
using Clock = std::chrono::steady_clock;

struct BenchData {
    Mat X;
    std::vector<std::string> iid, eid;
    std::vector<int> fid;
    Vec agevec;
    Mat ymat;
};

// Build synthetic data with realistic family structure and varied variance components
BenchData make_data(int n_subjects, int n_families, int n_visits, int n_voxels) {
    const int n_obs  = n_subjects * n_visits;
    const int n_pred = 3;

    std::mt19937 rng(42);
    std::normal_distribution<double> norm(0, 1);

    // Assign subjects to families (2-3 members each)
    std::vector<int> subj_to_fam(n_subjects);
    {
        int fi = 0;
        for (int s = 0; s < n_subjects; ++s) {
            subj_to_fam[s] = fi;
            if (s > 0 && fi < n_families - 1) {
                int prev_count = 0;
                for (int k = 0; k < s; ++k)
                    if (subj_to_fam[k] == fi) prev_count++;
                int fam_size = (fi % 5 < 3) ? 2 : 3;
                if (prev_count >= fam_size) fi++;
            }
        }
    }

    std::vector<std::string> iid(n_obs), eid(n_obs);
    std::vector<int> fid(n_obs);
    Vec agevec(n_obs);

    for (int v = 0; v < n_visits; ++v) {
        for (int s = 0; s < n_subjects; ++s) {
            int idx = v * n_subjects + s;
            iid[idx] = "S" + std::to_string(s);
            eid[idx] = "visit" + std::to_string(v + 1);
            fid[idx] = subj_to_fam[s];
            agevec(idx) = 10.0 + v * 2.0 + norm(rng) * 0.5;
        }
    }

    Mat X(n_obs, n_pred);
    X.col(0).setOnes();
    X.col(1) = agevec;
    for (int i = 0; i < n_obs; ++i)
        X(i, 2) = norm(rng);

    // Synthesize with varied variance components across voxels (uses binning grid)
    SynthResult synth = synthesize(
        X, iid, eid, fid, agevec,
        n_voxels, Mat(),
        {RFX::F, RFX::S, RFX::E},
        20, 1.0
    );

    // Add fixed effects signal
    Vec beta_true(n_pred);
    beta_true << 0.0, 0.5, -0.3;
    synth.ymat.colwise() += X * beta_true;

    return {X, iid, eid, fid, agevec, synth.ymat};
}

double run_fit(const BenchData& d, int nthreads) {
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#else
    (void)nthreads;
#endif

    FitConfig cfg;
    cfg.random_effects = {RFX::F, RFX::S, RFX::E};
    cfg.fixed_est_type = FixedEstType::GLS;
    cfg.rand_est_type  = RandEstType::MoM;
    cfg.nonneg_flag    = true;
    cfg.nbins          = 20;
    cfg.niter          = 1;
    cfg.nperms         = 0;

    auto t0 = Clock::now();
    FitResult result = fit(d.X, d.iid, d.eid, d.fid, d.agevec,
                           d.ymat, Mat(), Mat(), cfg);
    auto t1 = Clock::now();

    // Use result to prevent dead-code elimination
    volatile double sink = result.beta_hat(0, 0);
    (void)sink;

    return std::chrono::duration<double>(t1 - t0).count();
}

int main() {
    std::cout << "=== FEMA C++ OpenMP Benchmark ===" << std::endl;

#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    std::cout << "OpenMP enabled, max threads = " << max_threads << std::endl;
#else
    int max_threads = 1;
    std::cout << "OpenMP NOT available (single-threaded only)" << std::endl;
#endif

    const int n_subjects = 500;
    const int n_families = 200;
    const int n_visits   = 2;

    // Test at different voxel counts
    std::vector<int> voxel_counts = {100, 1000, 5000, 10000, 50000};

    std::cout << "\n" << std::left
              << std::setw(10) << "Voxels"
              << std::setw(14) << "1-thread(s)"
              << std::setw(14) << std::to_string(max_threads) + "-thread(s)"
              << std::setw(10) << "Speedup"
              << std::endl;
    std::cout << std::string(48, '-') << std::endl;

    for (int nv : voxel_counts) {
        std::cout << std::flush;

        // Generate data (not timed)
        BenchData data = make_data(n_subjects, n_families, n_visits, nv);

        // Warmup run
        run_fit(data, max_threads);

        // Single-threaded
        double t1 = run_fit(data, 1);

        // Multi-threaded
        double tN = run_fit(data, max_threads);

        double speedup = t1 / tN;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << nv
                  << std::setw(14) << t1
                  << std::setw(14) << tN
                  << std::setw(10) << std::setprecision(2) << speedup << "x"
                  << std::endl;
    }

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
