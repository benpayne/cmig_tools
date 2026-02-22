#pragma once
// FEMA - Fast and Efficient Mixed-effects Algorithm (C++ Implementation)
//
// C++ port of the MATLAB FEMA toolkit for fitting linear mixed effects models
// to large-sample whole-brain imaging data.
//
// Reference:
//   Parekh et al., (2024) - Fast and efficient mixed-effects algorithm for
//   large sample whole-brain imaging data, Human Brain Mapping,
//   https://doi.org/10.1002/hbm.26579

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace fema {

// ─── Type aliases ────────────────────────────────────────────────────────────
using Mat    = Eigen::MatrixXd;
using Vec    = Eigen::VectorXd;
using VecI   = Eigen::VectorXi;
using SpMat  = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

// ─── Enumerations ────────────────────────────────────────────────────────────
enum class CovType      { Analytic, Unstructured };
enum class FixedEstType { GLS, OLS };
enum class RandEstType  { MoM, ML };
enum class PermType     { WildBootstrap, WildBootstrapNN };

// Random effect identifiers (matches MATLAB codes)
enum class RFX : uint8_t {
    F = 0, // Family
    S,     // Subject
    E,     // Error (always last)
    A,     // Additive genetic
    D,     // Dominant genetic
    M,     // Maternal
    P,     // Paternal
    H,     // Home
    T,     // Twin
    COUNT
};

inline const char* rfx_name(RFX r) {
    static const char* names[] = {"F","S","E","A","D","M","P","H","T"};
    return names[static_cast<int>(r)];
}

RFX rfx_from_name(const std::string& s);

// ─── Per-family cluster info ─────────────────────────────────────────────────
struct ClusterInfo {
    std::vector<int> jvec_fam; // observation indices for this family
    std::vector<int> ivec_fam; // element indices into sparse matrix
    int              famtype;

    // Random-effect relatedness matrices (small, per-family)
    // V_X(i,j) captures the relatedness of obs i and j under random effect X
    Mat V_E, V_S, V_F, V_A, V_D, V_M, V_P, V_H, V_T;

    const Mat& V(RFX r) const;
};

// ─── Family structure (output of parse_family) ──────────────────────────────
struct FamilyStruct {
    std::vector<ClusterInfo>  clusterinfo;
    Mat                       M;         // design matrix for MoM
    std::vector<int>          famtypevec;
    int                       nfamtypes;
    std::vector<std::string>  iid;
    std::vector<int>          fid;
    std::vector<std::string>  iid_list;
    std::vector<int>          fid_list;
    int                       nfam;
    Mat  sig2grid, sig2gridl, sig2gridu, sig2gridi;
    VecI sig2gridind;
    int  nsig2bins;
    VecI subvec1, subvec2;
    std::vector<SpMat>        Ss; // sparse relatedness matrices [nRFX]
};

// ─── Configuration for FEMA_fit ──────────────────────────────────────────────
struct FitConfig {
    CovType      cov_type       = CovType::Analytic;
    FixedEstType fixed_est_type = FixedEstType::GLS;
    RandEstType  rand_est_type  = RandEstType::MoM;
    PermType     perm_type      = PermType::WildBootstrap;
    bool         group_by_fam   = true;
    bool         nonneg_flag    = true;
    bool         log_lik_flag   = false;
    bool         hess_flag      = false;
    int          nperms         = 0;
    int          niter          = 1;
    int          nbins          = 20;
    int          num_threads    = 1;
    std::vector<RFX> random_effects = {RFX::F, RFX::S, RFX::E};
};

// ─── Results from FEMA_fit ───────────────────────────────────────────────────
struct FitResult {
    Mat beta_hat;    // [p x v] or [(c+p) x v] with contrasts prepended
    Mat beta_se;     // [p x v]
    Mat zmat;        // [p x v]
    Mat logpmat;     // [p x v] signed -log10(p)
    Vec sig2tvec;    // [1 x v] total variance
    Mat sig2mat;     // [r x v] normalised variance components
    Vec logLikvec;   // [1 x v] log-likelihood

    // Permutation outputs
    std::vector<Mat> beta_hat_perm;  // [nperms+1]
    std::vector<Mat> beta_se_perm;
    std::vector<Mat> zmat_perm;
    std::vector<Vec> sig2tvec_perm;
    std::vector<Mat> sig2mat_perm;

    VecI binvec_save;
    Vec  nvec_bins;
    Vec  tvec_bins;

    // coeffCovar[v] is p x p for each voxel/vertex
    std::vector<Mat> coeffCovar;

    FamilyStruct family_struct;
};

// ─── Wald test results ───────────────────────────────────────────────────────
struct WaldResult {
    Mat W;           // [k x v] Wald statistic
    Mat p;           // [k x v] p-values
    Mat logp;        // [k x v] -log10(p)
    std::vector<Mat> LB_hat; // [k] each [l x v]
    std::vector<Mat> LB_SE;  // [k] each [l x v]
    VecI df;         // [k] numerator dof
};

// ─── Core API ────────────────────────────────────────────────────────────────

// Parse family relatedness structure
FamilyStruct parse_family(
    const std::vector<std::string>& iid,
    const std::vector<std::string>& eid,
    const std::vector<int>&         fid,
    const Vec&                      agevec,
    const Mat&                      GRM,
    const std::vector<RFX>&         random_effects,
    int                             nbins,
    const std::vector<int>&         FatherID = {},
    const std::vector<int>&         MotherID = {},
    const std::vector<int>&         PregID   = {},
    const std::vector<std::string>& HomeID   = {}
);

// Main fitting function
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
    FamilyStruct*                   precomputed_fs = nullptr
);

// Wald test
WaldResult wald_test(
    const std::vector<Mat>&         L,
    const Mat&                      beta_hat,
    const std::vector<Mat>&         coeffCovar,
    const Vec&                      hypValue = Vec(),
    bool                            doF      = false,
    int                             numObs   = 0
);

// Log-likelihood
double log_likelihood(
    const Vec&                     sig2vec,
    const Mat&                     X,
    const Vec&                     yvec_res,
    const std::vector<ClusterInfo>&clusterinfo,
    const std::vector<SpMat>&      Ss
);

// ─── Linear algebra utilities ────────────────────────────────────────────────

// Non-negative least squares (column-wise, fast)
Mat lsqnonneg(const Mat& A, const Mat& B);

// Nearest symmetric positive-definite matrix (Higham 1988)
Mat nearest_spd(const Mat& A);

// Multivariate normal log-pdf
double mvnpdf_ln(const Vec& x, const Vec& mu, const Mat& Sigma);

// Normal CDF (standard)
double normcdf(double x);

// Normal log-CDF (high precision for tails)
double normlogcdf(double x);

// ─── Synthesize data for testing ─────────────────────────────────────────────
struct SynthResult { Mat ymat; Vec sig2tvec_true; Mat sig2mat_true; };

SynthResult synthesize(
    const Mat& X,
    const std::vector<std::string>& iid,
    const std::vector<std::string>& eid,
    const std::vector<int>&         fid,
    const Vec&                      agevec,
    int num_y,
    const Mat&                      GRM,
    const std::vector<RFX>&         random_effects,
    int nbins = 20,
    double sig2tval = 1.0
);

} // namespace fema
