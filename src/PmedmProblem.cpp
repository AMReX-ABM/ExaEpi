/*! @file PmedmProblem.cpp
    \brief One PUMA's P-MEDM problem and its perturbation; see PmedmProblem.H.
*/
#include "PmedmProblem.H"

#include <algorithm>
#include <cmath>
#include <string>

#include "KeyedRNG.H"
#include "PopGenStages.H"

namespace PopGen {

// ---------------------------------------------------------------------------------------------
// Problem assembly and perturbation (problem.py, perturb.py)
// ---------------------------------------------------------------------------------------------

namespace {
constexpr double CG0_SE_FACTOR = 0.1;
constexpr double TWO_PI = 6.283185307179586;
} // namespace

int pumaCount (const PopulationBundle& b) {
    return static_cast<int>(b.get<std::int64_t>("solve.bg_offset").size()) - 1;
}

PmedmProblem::PmedmProblem (const PopulationBundle& b, int p) {
    puma = b.strings("solve.puma").at(p);
    fips = std::stoll(puma);
    const auto bgo = b.get<std::int64_t>("solve.bg_offset");
    const auto tro = b.get<std::int64_t>("solve.tract_offset");
    const auto dno = b.get<std::int64_t>("solve.donor_offset");
    const std::int64_t g0 = bgo[p], g1 = bgo[p + 1], t0 = tro[p], t1 = tro[p + 1], d0 = dno[p], d1 = dno[p + 1];
    const auto est_bg_all = b.get<double>("solve.est_bg");
    K = static_cast<int>(est_bg_all.shape.at(1));
    G = static_cast<int>(g1 - g0);
    T = static_cast<int>(t1 - t0);
    D = static_cast<int>(d1 - d0);

    const auto geo = b.get<std::int64_t>("solve.bg_geoid");
    const auto tract = b.get<std::int32_t>("solve.bg_tract");
    bg_geoid.assign(geo.begin() + g0, geo.begin() + g1);
    bg_tract.assign(tract.begin() + g0, tract.begin() + g1);
    const auto di = b.get<std::int32_t>("solve.donor_index");
    const auto hs = b.get<std::int16_t>("solve.donor_hh_size");
    donor_index.assign(di.begin() + d0, di.begin() + d1);
    donor_hh_size.assign(hs.begin() + d0, hs.begin() + d1);
    const auto pw = b.get<double>("solve.prior_weight");
    wt.assign(pw.begin() + d0, pw.begin() + d1);

    const auto se_bg_all = b.get<double>("solve.se_bg");
    const auto est_tr_all = b.get<double>("solve.est_tract");
    const auto se_tr_all = b.get<double>("solve.se_tract");
    est_bg.assign(est_bg_all.begin() + g0 * K, est_bg_all.begin() + g1 * K);
    se_bg.assign(se_bg_all.begin() + g0 * K, se_bg_all.begin() + g1 * K);
    est_tract.assign(est_tr_all.begin() + t0 * K, est_tr_all.begin() + t1 * K);
    se_tract.assign(se_tr_all.begin() + t0 * K, se_tr_all.begin() + t1 * K);

    // Donor constraint matrix from the global CSR rows d0 .. d1.
    const auto ip = b.get<std::int64_t>("solve.c_indptr");
    const auto ix = b.get<std::uint8_t>("solve.c_indices");
    const auto cv = b.get<double>("solve.c_values");
    C.assign(static_cast<std::size_t>(D) * K, 0.0);
    for (int d = 0; d < D; ++d) {
        for (std::int64_t j = ip[d0 + d]; j < ip[d0 + d + 1]; ++j) {
            C[static_cast<std::size_t>(d) * K + ix[j]] = cv[j];
        }
    }

    N = 0;
    for (double w : wt) {
        N += w;
    }
    const double scale = static_cast<double>(D) / (N * N);
    V.assign(nDual(), 0.0);
    for (int k = 0; k < K; ++k) {
        double s = 0;
        for (int g = 0; g < G; ++g) {
            s += se_bg[g * K + k] * se_bg[g * K + k];
        }
        const double sg0 = CG0_SE_FACTOR * std::sqrt(s);
        V[k] = sg0 * sg0 * scale;
        for (int t = 0; t < T; ++t) {
            V[K + k * T + t] = se_tract[t * K + k] * se_tract[t * K + k] * scale;
        }
        for (int g = 0; g < G; ++g) {
            V[K + K * T + k * G + g] = se_bg[g * K + k] * se_bg[g * K + k] * scale;
        }
    }
}

std::vector<double> pmedmTargets (const PmedmProblem& prob, const std::vector<double>& est_bg,
                                  const std::vector<double>& est_tract) {
    const int K = prob.K, T = prob.T, G = prob.G;
    std::vector<double> Y(prob.nDual(), 0.0);
    for (int k = 0; k < K; ++k) {
        double s = 0;
        for (int g = 0; g < G; ++g) {
            s += est_bg[g * K + k];
        }
        Y[k] = s / prob.N;
        for (int t = 0; t < T; ++t) {
            Y[K + k * T + t] = est_tract[t * K + k] / prob.N;
        }
        for (int g = 0; g < G; ++g) {
            Y[K + K * T + k * G + g] = est_bg[g * K + k] / prob.N;
        }
    }
    return Y;
}

std::vector<double> perturbedTargets (const PmedmProblem& prob, std::int64_t seed, std::int64_t rep,
                                      std::vector<double>* est_bg_out) {
    const int K = prob.K, T = prob.T, G = prob.G;
    const KR64 base = KR64(seed, rep, Stage::PERTURB_TARGET).with(prob.fips);
    std::vector<double> y2(static_cast<std::size_t>(G) * K);
    for (int g = 0; g < G; ++g) {
        for (int k = 0; k < K; ++k) {
            const std::size_t i = static_cast<std::size_t>(g) * K + k;
            const double est = prob.est_bg[i], se = prob.se_bg[i];
            if (!(est > 0)) {
                y2[i] = 0.0; // a zero estimate stays zero
                continue;
            }
            // Box-Muller on draws 0 and 1, then a log-normal with mean est and sd se.
            const KR64 key = base.with(k).with(prob.bg_geoid[g]);
            const double z = std::sqrt(-2.0 * std::log1p(-key.u01(0))) * std::cos(TWO_PI * key.u01(1));
            const double r = se / est;
            const double s2 = std::log1p(r * r);
            y2[i] = std::exp(std::log(est) - 0.5 * s2 + std::sqrt(s2) * z);
        }
    }
    // Tracts rebuilt as published tract + (perturbed - published) summed over their block groups.
    std::vector<double> y1(prob.est_tract);
    std::vector<double> delta(static_cast<std::size_t>(T) * K, 0.0);
    for (int g = 0; g < G; ++g) {
        for (int k = 0; k < K; ++k) {
            delta[prob.bg_tract[g] * K + k] += y2[g * K + k] - prob.est_bg[g * K + k];
        }
    }
    for (std::size_t i = 0; i < y1.size(); ++i) {
        y1[i] = std::max(y1[i] + delta[i], 0.0);
    }
    if (est_bg_out) { *est_bg_out = y2; }
    return pmedmTargets(prob, y2, y1);
}

std::vector<double> perturbedLogPrior (const PmedmProblem& prob, std::int64_t seed, std::int64_t rep) {
    const KR64 base = KR64(seed, rep, Stage::PERTURB_PRIOR).with(prob.fips);
    std::vector<double> w(prob.D);
    double total = 0;
    for (int d = 0; d < prob.D; ++d) {
        w[d] = prob.wt[d] * -std::log1p(-base.with(d).u01(0)); // weight x Exp(1)
        total += w[d];
    }
    for (auto& x : w) {
        x = std::log(x / total / prob.G);
    }
    return w;
}

} // namespace PopGen
