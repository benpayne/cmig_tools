// FEMA_parse_family - Parse family structure for mixed-effects estimation
//
// Creates per-family covariance structure matrices (V_F, V_S, V_E, etc.)
// and assembles global sparse relatedness matrices (Ss).
#include "fema.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <unordered_map>

namespace fema {

// Helper: unique-stable (preserves first-occurrence order, like MATLAB 'stable')
template <typename T>
static void unique_stable(const std::vector<T>& input,
                          std::vector<T>& unique_vals,
                          std::vector<int>& first_idx, // IA
                          std::vector<int>& mapping)   // IC (0-based)
{
    std::unordered_map<T, int> seen;
    unique_vals.clear();
    first_idx.clear();
    mapping.resize(input.size());
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        auto it = seen.find(input[i]);
        if (it == seen.end()) {
            int idx = unique_vals.size();
            seen[input[i]] = idx;
            unique_vals.push_back(input[i]);
            first_idx.push_back(i);
            mapping[i] = idx;
        } else {
            mapping[i] = it->second;
        }
    }
}

// Helper: N-dimensional grid generation (like ndgrid_amd)
// Produces matrix [N_total x ndims] containing all combinations
static Mat ndgrid(const std::vector<Vec>& axes) {
    int ndims = axes.size();
    int total = 1;
    for (auto& a : axes) total *= a.size();

    Mat grid(total, ndims);
    int rep_inner = 1;
    for (int d = 0; d < ndims; ++d) {
        int n = axes[d].size();
        int rep_outer = total / (n * rep_inner);
        int row = 0;
        for (int o = 0; o < rep_outer; ++o)
            for (int i = 0; i < n; ++i)
                for (int r = 0; r < rep_inner; ++r)
                    grid(row++, d) = axes[d](i);
        rep_inner *= n;
    }
    return grid;
}

// Helper: sub2ind for multi-dimensional indexing
static VecI sub2ind(const VecI& dims, const Mat& subs) {
    int nrow = subs.rows();
    int ndim = subs.cols();
    VecI result(nrow);
    for (int i = 0; i < nrow; ++i) {
        int idx = 0;
        int multiplier = 1;
        for (int d = 0; d < ndim; ++d) {
            idx += static_cast<int>(subs(i, d)) * multiplier;
            multiplier *= dims(d);
        }
        result(i) = idx;
    }
    return result;
}

FamilyStruct parse_family(
    const std::vector<std::string>& iid,
    const std::vector<std::string>& eid,
    const std::vector<int>&         fid,
    const Vec&                      agevec,
    const Mat&                      GRM,
    const std::vector<RFX>&         random_effects,
    int                             nbins,
    const std::vector<int>&         FatherID,
    const std::vector<int>&         MotherID,
    const std::vector<int>&         PregID,
    const std::vector<std::string>& HomeID)
{
    int nobs = iid.size();
    int num_RFX = random_effects.size();
    bool has_GRM = GRM.rows() > 0;

    // Unique subjects (stable order)
    std::vector<std::string> iid_list;
    std::vector<int> IA_subj, IC_subj;
    unique_stable(iid, iid_list, IA_subj, IC_subj);
    int nsubj = iid_list.size();

    // Unique families (stable order)
    std::vector<int> fid_list;
    std::vector<int> IA_fam, IC_fam;
    unique_stable(fid, fid_list, IA_fam, IC_fam);
    int nfam = fid_list.size();

    // Per-subject observation indices
    std::vector<std::vector<int>> jvecs_subj(nsubj);
    for (int i = 0; i < nobs; ++i) {
        jvecs_subj[IC_subj[i]].push_back(i);
    }

    // Parse family structure
    std::vector<ClusterInfo> clusterinfo(nfam);
    std::vector<std::string> famtype_strs;
    std::vector<int> famtypevec(nfam, -1);

    for (int fi = 0; fi < nfam; ++fi) {
        // All observations for this family
        std::vector<int> jvec_fam;
        for (int i = 0; i < nobs; ++i) {
            if (IC_fam[i] == fi) jvec_fam.push_back(i);
        }

        // Subjects in this family
        std::vector<int> subj_fam; // IC_subj values
        for (int j : jvec_fam) subj_fam.push_back(IC_subj[j]);

        // Unique subjects and their frequencies
        std::vector<int> subj_unique;
        {
            std::unordered_map<int, int> seen;
            for (int s : subj_fam) {
                if (seen.find(s) == seen.end()) {
                    seen[s] = 0;
                    subj_unique.push_back(s);
                }
            }
        }

        std::vector<int> freq_unique(subj_unique.size());
        for (size_t j = 0; j < subj_unique.size(); ++j) {
            freq_unique[j] = 0;
            for (int s : subj_fam) if (s == subj_unique[j]) freq_unique[j]++;
        }

        // Sort subjects by frequency (ascending)
        std::vector<int> si(subj_unique.size());
        std::iota(si.begin(), si.end(), 0);
        std::sort(si.begin(), si.end(), [&](int a, int b) {
            return freq_unique[a] < freq_unique[b];
        });
        {
            std::vector<int> tmp_su(subj_unique.size()), tmp_fu(subj_unique.size());
            for (size_t i = 0; i < si.size(); ++i) {
                tmp_su[i] = subj_unique[si[i]];
                tmp_fu[i] = freq_unique[si[i]];
            }
            subj_unique = tmp_su;
            freq_unique = tmp_fu;
        }

        // Build family-type string
        std::string str_freq;
        for (size_t i = 0; i < freq_unique.size(); ++i) {
            if (i > 0) str_freq += " ";
            str_freq += std::to_string(freq_unique[i]);
        }

        // Map subjects to canonical index within family
        std::vector<int> subji_jvec(subj_fam.size());
        for (size_t j = 0; j < subj_fam.size(); ++j) {
            for (size_t s = 0; s < subj_unique.size(); ++s) {
                if (subj_fam[j] == subj_unique[s]) {
                    subji_jvec[j] = s;
                    break;
                }
            }
        }

        // Sort by [subject_index, age] for canonical ordering
        std::vector<int> sort_idx(jvec_fam.size());
        std::iota(sort_idx.begin(), sort_idx.end(), 0);
        std::sort(sort_idx.begin(), sort_idx.end(), [&](int a, int b) {
            if (subji_jvec[a] != subji_jvec[b]) return subji_jvec[a] < subji_jvec[b];
            return agevec(jvec_fam[a]) < agevec(jvec_fam[b]);
        });
        {
            std::vector<int> tmp(jvec_fam.size());
            for (size_t i = 0; i < sort_idx.size(); ++i) tmp[i] = jvec_fam[sort_idx[i]];
            jvec_fam = tmp;

            std::vector<int> tmp_sf(subj_fam.size());
            for (size_t i = 0; i < sort_idx.size(); ++i) tmp_sf[i] = subj_fam[sort_idx[i]];
            subj_fam = tmp_sf;
        }

        // Assign family type
        auto it = std::find(famtype_strs.begin(), famtype_strs.end(), str_freq);
        if (it == famtype_strs.end()) {
            famtype_strs.push_back(str_freq);
            famtypevec[fi] = famtype_strs.size() - 1;
        } else {
            famtypevec[fi] = std::distance(famtype_strs.begin(), it);
        }

        // Build relatedness matrices
        int nj = jvec_fam.size();

        // V_E: identity
        Mat V_E = Mat::Identity(nj, nj);

        // V_F: all ones (same family)
        Mat V_F = Mat::Ones(nj, nj);

        // V_S: same subject (longitudinal)
        Mat V_S = Mat::Zero(nj, nj);
        for (int i = 0; i < nj; ++i) {
            for (int j = 0; j < nj; ++j) {
                if (IC_subj[jvec_fam[i]] == IC_subj[jvec_fam[j]])
                    V_S(i, j) = 1.0;
            }
        }

        // V_A: genetic relatedness
        Mat V_A, V_D;
        if (has_GRM) {
            V_A.resize(nj, nj);
            for (int i = 0; i < nj; ++i)
                for (int j = 0; j < nj; ++j) {
                    double val = GRM(IC_subj[jvec_fam[i]], IC_subj[jvec_fam[j]]);
                    V_A(i, j) = std::isfinite(val) ? val : 0.5; // impute missing
                }
            V_D = V_A.array().square().matrix();
        }

        // V_P: paternal
        Mat V_P;
        if (!FatherID.empty()) {
            V_P = Mat::Zero(nj, nj);
            for (int i = 0; i < nj; ++i)
                for (int j = 0; j < nj; ++j)
                    if (FatherID[IC_subj[jvec_fam[i]]] == FatherID[IC_subj[jvec_fam[j]]])
                        V_P(i, j) = 1.0;
        }

        // V_M: maternal
        Mat V_M;
        if (!MotherID.empty()) {
            V_M = Mat::Zero(nj, nj);
            for (int i = 0; i < nj; ++i)
                for (int j = 0; j < nj; ++j)
                    if (MotherID[IC_subj[jvec_fam[i]]] == MotherID[IC_subj[jvec_fam[j]]])
                        V_M(i, j) = 1.0;
        }

        // V_H: home effect
        Mat V_H;
        if (!HomeID.empty()) {
            V_H = Mat::Zero(nj, nj);
            for (int i = 0; i < nj; ++i)
                for (int j = 0; j < nj; ++j)
                    if (HomeID[IC_subj[jvec_fam[i]]] == HomeID[IC_subj[jvec_fam[j]]])
                        V_H(i, j) = 1.0;
        }

        // V_T: twin effect
        Mat V_T;
        if (!PregID.empty()) {
            V_T = Mat::Zero(nj, nj);
            for (int i = 0; i < nj; ++i)
                for (int j = 0; j < nj; ++j)
                    if (PregID[IC_subj[jvec_fam[i]]] == PregID[IC_subj[jvec_fam[j]]])
                        V_T(i, j) = 1.0;
        }

        clusterinfo[fi].jvec_fam = jvec_fam;
        clusterinfo[fi].famtype  = famtypevec[fi];
        clusterinfo[fi].V_E = V_E;
        clusterinfo[fi].V_S = V_S;
        clusterinfo[fi].V_F = V_F;
        clusterinfo[fi].V_A = V_A;
        clusterinfo[fi].V_D = V_D;
        clusterinfo[fi].V_P = V_P;
        clusterinfo[fi].V_M = V_M;
        clusterinfo[fi].V_H = V_H;
        clusterinfo[fi].V_T = V_T;
    }

    // Build global sparse relatedness matrices Ss
    // Count total non-zeros
    int64_t nnz_max = 0;
    for (const auto& ci : clusterinfo) {
        int nj = ci.jvec_fam.size();
        nnz_max += static_cast<int64_t>(nj) * nj;
    }

    std::vector<SpMat> Ss(num_RFX);
    {
        std::vector<std::vector<Triplet>> triplets(num_RFX);
        for (auto& t : triplets) t.reserve(nnz_max);

        for (const auto& ci : clusterinfo) {
            int nj = ci.jvec_fam.size();
            for (int ri = 0; ri < num_RFX; ++ri) {
                const Mat& Vr = ci.V(random_effects[ri]);
                if (Vr.rows() == 0) continue;
                for (int ii = 0; ii < nj; ++ii)
                    for (int jj = 0; jj < nj; ++jj)
                        if (Vr(ii, jj) != 0.0)
                            triplets[ri].emplace_back(ci.jvec_fam[ii], ci.jvec_fam[jj], Vr(ii, jj));
            }
        }

        for (int ri = 0; ri < num_RFX; ++ri) {
            Ss[ri].resize(nobs, nobs);
            Ss[ri].setFromTriplets(triplets[ri].begin(), triplets[ri].end());
            Ss[ri].makeCompressed();
        }
    }

    // Build S_sum and compute subvec1/subvec2 (non-zero locations)
    SpMat S_sum = Ss[0];
    for (int i = 1; i < num_RFX; ++i) S_sum += Ss[i];

    std::vector<int> sv1, sv2;
    for (int k = 0; k < S_sum.outerSize(); ++k)
        for (SpMat::InnerIterator it(S_sum, k); it; ++it) {
            sv1.push_back(it.row());
            sv2.push_back(it.col());
        }

    VecI subvec1 = Eigen::Map<VecI>(sv1.data(), sv1.size());
    VecI subvec2 = Eigen::Map<VecI>(sv2.data(), sv2.size());

    // Build M matrix: [len(indvec) x num_RFX]
    int nind = sv1.size();
    Mat M_mat(nind, num_RFX);
    for (int ri = 0; ri < num_RFX; ++ri) {
        for (int idx = 0; idx < nind; ++idx) {
            M_mat(idx, ri) = Ss[ri].coeff(sv1[idx], sv2[idx]);
        }
    }

    // Build family number map for ivec_fam computation
    // F_num(i,j) = family index for that block
    std::unordered_map<int64_t, int> fnummap;
    for (int fi = 0; fi < nfam; ++fi) {
        for (int ii : clusterinfo[fi].jvec_fam)
            for (int jj : clusterinfo[fi].jvec_fam)
                fnummap[static_cast<int64_t>(ii) * nobs + jj] = fi;
    }

    // Compute ivec_fam for each family
    std::vector<int> fnumvec(nind);
    for (int idx = 0; idx < nind; ++idx) {
        int64_t key = static_cast<int64_t>(sv1[idx]) * nobs + sv2[idx];
        auto it2 = fnummap.find(key);
        fnumvec[idx] = (it2 != fnummap.end()) ? it2->second : -1;
    }

    for (int fi = 0; fi < nfam; ++fi) {
        auto& ci = clusterinfo[fi];
        int nj = ci.jvec_fam.size();

        // Collect indices where fnumvec == fi
        std::vector<int> ivec_raw;
        for (int idx = 0; idx < nind; ++idx) {
            if (fnumvec[idx] == fi) ivec_raw.push_back(idx);
        }

        // Sort by canonical ordering (matching MATLAB sort logic)
        std::vector<int> jv_sorted = ci.jvec_fam;
        std::vector<int> si2(nj);
        std::iota(si2.begin(), si2.end(), 0);
        // jvec_fam is already sorted, compute sorting permutation
        // This mirrors the MATLAB: [sv, si] = sort(jvec_tmp); then I_tmp reindexing
        std::sort(si2.begin(), si2.end(), [&](int a, int b) {
            return ci.jvec_fam[a] < ci.jvec_fam[b];
        });

        // Build reordering matrix I_tmp
        std::vector<int> I_tmp(nj * nj);
        for (int r = 0; r < nj; ++r)
            for (int c = 0; c < nj; ++c)
                I_tmp[r * nj + c] = r * nj + c;

        // Apply si2 permutation
        std::vector<int> I_reorder(nj * nj);
        for (int r = 0; r < nj; ++r)
            for (int c = 0; c < nj; ++c)
                I_reorder[r * nj + c] = I_tmp[si2[r] * nj + si2[c]];

        // Assign ivec_fam
        ci.ivec_fam.resize(ivec_raw.size());
        if (ivec_raw.size() == static_cast<size_t>(nj * nj)) {
            for (size_t k = 0; k < ivec_raw.size(); ++k)
                ci.ivec_fam[I_reorder[k]] = ivec_raw[k];
        } else {
            ci.ivec_fam = ivec_raw;
        }
    }

    // Build variance component grid
    Vec edges = Vec::LinSpaced(nbins + 1, 0.0, 1.0);
    edges(nbins) += 0.0001; // slightly extend last bin

    int num_rfx_minus1 = num_RFX - 1;
    Mat sig2grid, sig2gridl, sig2gridu, sig2gridi_mat;

    if (num_rfx_minus1 == 1) {
        sig2gridi_mat.resize(nbins, 1);
        sig2gridl.resize(nbins, 1);
        sig2gridu.resize(nbins, 1);
        for (int i = 0; i < nbins; ++i) {
            sig2gridi_mat(i, 0) = i;
            sig2gridl(i, 0) = edges(i);
            sig2gridu(i, 0) = edges(i + 1);
        }
    } else {
        // N-dimensional grid
        Vec bin_indices = Vec::LinSpaced(nbins, 0, nbins - 1);
        Vec bin_lower(nbins), bin_upper(nbins);
        for (int i = 0; i < nbins; ++i) {
            bin_lower(i) = edges(i);
            bin_upper(i) = edges(i + 1);
        }

        std::vector<Vec> axes_i(num_rfx_minus1, bin_indices);
        std::vector<Vec> axes_l(num_rfx_minus1, bin_lower);
        std::vector<Vec> axes_u(num_rfx_minus1, bin_upper);

        sig2gridi_mat = ndgrid(axes_i);
        sig2gridl     = ndgrid(axes_l);
        sig2gridu     = ndgrid(axes_u);
    }

    sig2grid = 0.5 * (sig2gridl + sig2gridu);

    // Filter out "impossible" bins (sum of lower bounds > 1)
    std::vector<int> valid;
    for (int i = 0; i < sig2gridl.rows(); ++i) {
        if (sig2gridl.row(i).sum() <= 1.0) valid.push_back(i);
    }

    int nsig2bins = valid.size();
    Mat sg(nsig2bins, num_rfx_minus1), sgl(nsig2bins, num_rfx_minus1);
    Mat sgu(nsig2bins, num_rfx_minus1), sgi(nsig2bins, num_rfx_minus1);
    for (int i = 0; i < nsig2bins; ++i) {
        sg.row(i)  = sig2grid.row(valid[i]);
        sgl.row(i) = sig2gridl.row(valid[i]);
        sgu.row(i) = sig2gridu.row(valid[i]);
        sgi.row(i) = sig2gridi_mat.row(valid[i]);
    }

    // Build sub2ind
    VecI dims = VecI::Constant(num_rfx_minus1, nbins);
    VecI sig2gridind = sub2ind(dims, sgi);

    // Assemble output
    FamilyStruct fs;
    fs.clusterinfo = std::move(clusterinfo);
    fs.M           = M_mat;
    fs.famtypevec  = famtypevec;
    fs.nfamtypes   = famtype_strs.size();
    fs.iid         = iid;
    fs.fid         = fid;
    fs.iid_list    = iid_list;
    fs.fid_list    = fid_list;
    fs.nfam        = nfam;
    fs.sig2grid    = sg;
    fs.sig2gridl   = sgl;
    fs.sig2gridu   = sgu;
    fs.sig2gridi   = sgi;
    fs.sig2gridind = sig2gridind;
    fs.nsig2bins   = nsig2bins;
    fs.subvec1     = subvec1;
    fs.subvec2     = subvec2;
    fs.Ss          = std::move(Ss);

    return fs;
}

} // namespace fema
