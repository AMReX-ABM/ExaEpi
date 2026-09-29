/*! @file PopulationGenerator.cpp
    \brief Placement and population stages; see PopulationGenerator.H. Each section names the
    Python function it ports.
*/
#include "PopulationGenerator.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "KeyedRNG.H"
#include "PopGenStages.H"
#include "Sha256.H"

namespace PopGen {

namespace {

//! Index of a name in a list; throws if missing.
int indexOf (const std::vector<std::string>& names, const std::string& name) {
    const auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) { throw std::runtime_error("population bundle has no constraint " + name); }
    return static_cast<int>(it - names.begin());
}

//! Sequential sum, as numpy.cumsum(x)[-1].
double seqSum (const double* x, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += x[i];
    }
    return s;
}

//! Sequential running sum into cum, as numpy.cumsum.
void runningSum (const double* w, std::size_t n, std::vector<double>& cum) {
    cum.resize(n);
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += w[i];
        cum[i] = s;
    }
}

//! Threads for a parallel region here: all of them, unless already inside one.
int threads () {
#ifdef _OPENMP
    return omp_in_parallel() ? 1 : omp_get_max_threads();
#else
    return 1;
#endif
}

/*! Sort v by less, which must order any two distinct elements (a strict total order up to
    identical values), so the sorted result is unique: whatever the algorithm and thread count, it
    is what std::stable_sort would give. On OpenMP builds the range is cut into one chunk per
    thread, the chunks sorted concurrently, then merged pairwise. */
template <class T, class Less>
void totalSort (std::vector<T>& v, Less less) {
    const std::size_t n = v.size();
    const int nt = threads();
    if (nt == 1 || n < (std::size_t(1) << 16)) {
        std::sort(v.begin(), v.end(), less);
        return;
    }
    const std::size_t nc = static_cast<std::size_t>(nt);
    std::vector<std::size_t> cut(nc + 1);
    for (std::size_t c = 0; c <= nc; ++c) {
        cut[c] = n * c / nc;
    }
    const auto at = [&] (std::size_t k) {
        return v.begin() + static_cast<std::ptrdiff_t>(cut[std::min(k, nc)]);
    };
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
    for (int c = 0; c < nt; ++c) {
        std::sort(at(static_cast<std::size_t>(c)), at(static_cast<std::size_t>(c) + 1), less);
    }
    for (std::size_t w = 1; w < nc; w *= 2) {
        const auto pairs = static_cast<std::int64_t>((nc + 2 * w - 1) / (2 * w));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (std::int64_t q = 0; q < pairs; ++q) {
            const std::size_t lo = 2 * w * static_cast<std::size_t>(q);
            std::inplace_merge(at(lo), at(lo + w), at(lo + 2 * w), less);
        }
    }
}

//! Start of each run of equal values in v, plus v.size() at the end.
template <class T>
std::vector<std::int64_t> runStarts (const std::vector<T>& v) {
    std::vector<std::int64_t> s;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i == 0 || !(v[i] == v[i - 1])) { s.push_back(static_cast<std::int64_t>(i)); }
    }
    s.push_back(static_cast<std::int64_t>(v.size()));
    return s;
}

//! Stable index permutation sorting by a key comparator, as numpy.lexsort (stable): ties go to
//! the lower index, which makes the order total, so totalSort can do it in parallel.
template <class Less>
std::vector<std::int64_t> stableOrder (std::size_t n, Less less) {
    std::vector<std::int64_t> o(n);
    std::iota(o.begin(), o.end(), 0);
    totalSort(o, [&] (std::int64_t x, std::int64_t y) {
        return less(x, y) || (!less(y, x) && x < y);
    });
    return o;
}

} // namespace

void Placements::append (const Placements& o) {
    bg.insert(bg.end(), o.bg.begin(), o.bg.end());
    donor.insert(donor.end(), o.donor.begin(), o.donor.end());
    count.insert(count.end(), o.count.begin(), o.count.end());
}

// ---------------------------------------------------------------------------------------------
// Placement (placement.py: place_puma, categories, true_sizes)
// ---------------------------------------------------------------------------------------------

Placements placePuma (const PopulationBundle& b, const PmedmProblem& prob, const std::vector<double>& al,
                      const std::vector<std::string>& cols, std::int64_t seed, std::int64_t rep) {
    const int D = prob.D, G = prob.G, K = prob.K;
    if (al.size() != static_cast<std::size_t>(D) * G) { throw std::invalid_argument("placePuma: allocation size"); }

    // Household categories: type x size, group quarters, vacant (neither occupied nor GQ).
    std::vector<int> hht;
    for (int j = 0; j < static_cast<int>(cols.size()); ++j) {
        if (cols[j].rfind("hht", 0) == 0 && cols[j].find("hhsize") != std::string::npos) { hht.push_back(j); }
    }
    if (hht.empty()) { throw std::runtime_error("constraints must include household type by household size"); }
    const int gq = indexOf(cols, "group_quarters_pop"), occ = indexOf(cols, "occhu");
    const int popc = indexOf(cols, "population");
    const int V = static_cast<int>(hht.size()) + 2;
    auto Cdk = [&] (int d, int k) {
        return prob.C[static_cast<std::size_t>(d) * K + k];
    };
    std::vector<double> R(static_cast<std::size_t>(D) * V); // D x V
    for (int d = 0; d < D; ++d) {
        for (int v = 0; v < V - 2; ++v) {
            R[d * V + v] = Cdk(d, hht[v]);
        }
        R[d * V + V - 2] = Cdk(d, gq);
        R[d * V + V - 1] = (Cdk(d, occ) == 0.0 && Cdk(d, gq) == 0.0) ? 1.0 : 0.0;
    }

    // 1. Reweight: almat_adj = almat * est_ind.population / household size (vacant: 1).
    std::vector<double> A(static_cast<std::size_t>(D) * G);
    for (int d = 0; d < D; ++d) {
        const double nm = static_cast<double>(prob.donor_hh_size[d]);
        const double adj = nm > 0 ? Cdk(d, popc) / nm : 1.0;
        for (int g = 0; g < G; ++g) {
            A[d * G + g] = al[d * G + g] * adj;
        }
    }

    // 2. Fractional counts per (category, block group), summed sequentially over donors.
    std::vector<double> tosamp(static_cast<std::size_t>(V) * G, 0.0);
    for (int v = 0; v < V; ++v) {
        for (int g = 0; g < G; ++g) {
            double s = 0.0;
            for (int d = 0; d < D; ++d) {
                s += R[d * V + v] * A[d * G + g];
            }
            tosamp[v * G + g] = s;
        }
    }

    // 3. One TRS over the PUMA's flattened (category, block group) matrix, without replacement.
    const std::size_t VG = tosamp.size();
    std::vector<std::int64_t> whole(VG);
    std::vector<double> frac(VG);
    for (std::size_t i = 0; i < VG; ++i) {
        const double fl = std::floor(tosamp[i]);
        whole[i] = static_cast<std::int64_t>(fl);
        frac[i] = tosamp[i] - fl;
    }
    const auto extra = static_cast<std::int64_t>(std::nearbyint(seqSum(frac.data(), VG)));
    std::vector<double> cum;
    const KR64 trs = KR64(seed, rep, Stage::HH_TYPE_TRS).with(prob.fips);
    for (std::int64_t k = 0; k < extra; ++k) {
        runningSum(frac.data(), VG, cum);
        if (cum.back() <= 0) { break; }
        const auto i = floatCdf(cum.data(), static_cast<std::int64_t>(VG), trs.with(k).u64(0));
        whole[i] += 1;
        frac[i] = 0.0;
    }

    // 4. Donors within each (block group, category), with replacement; keep occupied, matched rows.
    const auto off = b.get<std::int64_t>("donors.hh_offset");
    Placements out;
    std::vector<std::int64_t> n_d(D);
    std::vector<double> w(D);
    for (int g = 0; g < G; ++g) {
        std::fill(n_d.begin(), n_d.end(), 0);
        const std::int64_t bg = prob.bg_geoid[g];
        for (int v = 0; v < V; ++v) {
            const std::int64_t n = whole[v * G + g];
            if (n == 0) { continue; }
            for (int d = 0; d < D; ++d) {
                w[d] = A[d * G + g] * R[d * V + v];
            }
            runningSum(w.data(), D, cum);
            if (cum.back() <= 0) { continue; }
            const KR64 kv = KR64(seed, rep, Stage::HH_DRAW).with(bg).with(v);
            for (std::int64_t j = 0; j < n; ++j) {
                ++n_d[floatCdf(cum.data(), D, kv.with(j).u64(0))];
            }
        }
        for (int d = 0; d < D; ++d) {
            const int di = prob.donor_index[d];
            if (n_d[d] > 0 && di >= 0 && off[di + 1] - off[di] > 0) {
                out.bg.push_back(bg);
                out.donor.push_back(di);
                out.count.push_back(n_d[d]);
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// Expansion and S0-S2 (placement.expand; persons.build, childcare, split)
// ---------------------------------------------------------------------------------------------

namespace {
constexpr std::int16_t CHILDCARE_GRADE = 3;
constexpr std::int16_t PRESCHOOL_GRADE = 4;
constexpr std::int16_t KINDERGARTEN_GRADE = 5;
constexpr std::int16_t GRADE_SHIFT = 3;
// Share of all children of each age (0-4) in center-based care: NCES Digest table 202.30, 2019.
// A child in a household where every adult works is twice as likely to be picked.
constexpr double CENTER_CARE_RATE[5] = {0.141, 0.265, 0.265, 0.625, 0.625};

//! Probabilities {p1, p2} for children of weight 1 and 2, p_w = min(1, k w), with k such that
//! n1 p1 + n2 p2 = wanted, or everyone (persons._care_probs).
std::pair<double, double> careProbs (double wanted, std::int64_t n1, std::int64_t n2) {
    if (wanted <= 0) { return {0.0, 0.0}; }
    if (wanted >= static_cast<double>(n1 + n2)) { return {1.0, 1.0}; }
    if (2.0 * wanted <= static_cast<double>(n1 + 2 * n2)) {
        const double k = wanted / static_cast<double>(n1 + 2 * n2);
        return {k, 2.0 * k};
    }
    return {(wanted - static_cast<double>(n2)) / static_cast<double>(n1), 1.0};
}
} // namespace

Persons buildPersons (const PopulationBundle& b, const Placements& pl, std::int64_t seed, std::int64_t rep) {
    const auto off = b.get<std::int64_t>("donors.hh_offset");
    // Households numbered densely within each block group in placement order; then block groups in
    // ascending geoid (a stable sort, so each block group keeps its (h, p) order).
    struct Hh {
        std::int64_t bg, h, donor;
    };
    std::vector<Hh> hh;
    for (std::size_t r = 0; r < pl.size(); ++r) {
        std::int64_t h = (!hh.empty() && hh.back().bg == pl.bg[r]) ? hh.back().h + 1 : 0;
        for (std::int64_t c = 0; c < pl.count[r]; ++c) {
            hh.push_back({pl.bg[r], h++, pl.donor[r]});
        }
    }
    std::stable_sort(hh.begin(), hh.end(), [] (const Hh& x, const Hh& y) {
        return x.bg < y.bg;
    });

    const auto age = b.get<std::int8_t>("donors.age");
    const auto sex = b.get<std::int8_t>("donors.sex");
    const auto race = b.get<std::int8_t>("donors.race");
    const auto naics = b.get<std::int16_t>("donors.naics");
    const auto travel = b.get<std::int8_t>("donors.travel");
    const auto jwmnp = b.get<std::int16_t>("donors.jwmnp");
    const auto veh = b.get<std::int8_t>("donors.veh_occ");
    const auto grade = b.get<std::int8_t>("donors.grade");
    Persons P;
    for (const auto& x : hh) {
        for (std::int64_t s = off[x.donor]; s < off[x.donor + 1]; ++s) {
            P.bg.push_back(x.bg);
            P.h.push_back(x.h);
            P.p.push_back(s - off[x.donor]);
            P.src.push_back(s);
            P.age.push_back(age[s]);
            P.sex.push_back(sex[s]);
            P.race.push_back(race[s]);
            P.naics.push_back(naics[s]);
            P.travel.push_back(travel[s]);
            P.jwmnp.push_back(jwmnp[s]);
            P.veh_occ.push_back(veh[s]);
            P.grade.push_back(static_cast<std::int16_t>(grade[s] >= 0 ? grade[s] + GRADE_SHIFT : -1));
        }
    }
    const std::size_t n = P.size();

    // S1 childcare: under-5s not in school, one keyed Bernoulli each, at the probability that
    // brings each age's share in care (preschoolers included) to CENTER_CARE_RATE. Households are
    // runs of equal (bg, h); weight 2 where no adult lacks an industry.
    std::vector<std::uint8_t> all_work(n, 1);
    for (std::size_t lo = 0, hi; lo < n; lo = hi) {
        bool idle = false;
        for (hi = lo; hi < n && P.bg[hi] == P.bg[lo] && P.h[hi] == P.h[lo]; ++hi) {
            idle = idle || (P.age[hi] >= 18 && P.naics[hi] == -1);
        }
        if (idle) { std::fill(all_work.begin() + lo, all_work.begin() + hi, 0); }
    }
    constexpr int NA = 5;
    std::array<std::int64_t, NA> at_age{}, preschool{}, n1{}, n2{};
    for (std::size_t i = 0; i < n; ++i) {
        const int a = P.age[i];
        if (a < 0 || a >= NA) { continue; }
        if (P.grade[i] < KINDERGARTEN_GRADE) { ++at_age[a]; }
        if (P.grade[i] == PRESCHOOL_GRADE) { ++preschool[a]; }
        if (P.grade[i] == -1) { ++(all_work[i] ? n2 : n1)[a]; }
    }
    std::array<std::pair<double, double>, NA> prob;
    for (int a = 0; a < NA; ++a) {
        prob[a] =
                careProbs(CENTER_CARE_RATE[a] * static_cast<double>(at_age[a]) - static_cast<double>(preschool[a]), n1[a], n2[a]);
    }
    const KR64 kc(seed, rep, Stage::CHILDCARE);
    for (std::size_t i = 0; i < n; ++i) {
        const int a = P.age[i];
        if (P.grade[i] == -1 && a >= 0 && a < NA) {
            const double p = all_work[i] ? prob[a].second : prob[a].first;
            if (kc.with(P.bg[i]).with(P.h[i]).with(P.p[i]).u01(0) < p) { P.grade[i] = CHILDCARE_GRADE; }
        }
    }
    // S2 split: employed = has an industry and (not in school or older than 26); students lose
    // their industry; workers leave school.
    P.employed.resize(n);
    P.student.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const bool in_school = P.grade[i] != -1;
        const bool employed = P.naics[i] != -1 && (!in_school || P.age[i] > 26);
        const bool student = in_school && !employed;
        P.employed[i] = employed;
        P.student[i] = student;
        if (student) { P.naics[i] = -1; }
        if (employed) { P.grade[i] = -1; }
    }
    return P;
}

// ---------------------------------------------------------------------------------------------
// CBP tables (cbp.py: band_sizes, SizeTables)
// ---------------------------------------------------------------------------------------------

namespace {

//! The 256 sizes for one band, or empty when the band has no establishments or employees.
std::vector<std::int64_t> bandSizes (std::int64_t lo, std::int64_t hi, std::int64_t n, std::int64_t e) {
    constexpr int PTS = SizeTables::POINTS;
    if (n <= 0 || e <= 0) { return {}; }
    const std::int64_t hi_eff = hi >= 0 ? hi : std::max(lo + 1, (4 * e) / n);
    double r = static_cast<double>(hi_eff + 1) / static_cast<double>(lo);
    for (int i = 0; i < 8; ++i) {
        r = std::sqrt(r);
    }
    std::vector<double> q(PTS);
    q[0] = static_cast<double>(lo) * std::sqrt(r);
    for (int i = 1; i < PTS; ++i) {
        q[i] = q[i - 1] * r;
    }
    const double mean_q = seqSum(q.data(), PTS) / PTS;
    const double f = (static_cast<double>(e) / static_cast<double>(n)) / mean_q;
    std::vector<std::int64_t> out(PTS);
    for (int i = 0; i < PTS; ++i) {
        out[i] = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::nearbyint(q[i] * f)));
    }
    return out;
}

} // namespace

SizeTables::SizeTables (const PopulationBundle& b) {
    m_default = b.get<std::int32_t>("cbp.default_workgroup_target")[0];
    const auto ws = b.get<std::int16_t>("cbp.wg_state");
    const auto wn = b.get<std::int16_t>("cbp.wg_naics");
    const auto wz = b.get<std::int32_t>("cbp.wg_size");
    for (std::size_t i = 0; i < ws.size(); ++i) {
        m_targets[{ws[i], wn[i]}] = wz[i];
    }
    const auto lo = b.get<std::int32_t>("cbp.est_band_lo");
    const auto hi = b.get<std::int32_t>("cbp.est_band_hi");
    const auto es = b.get<std::int16_t>("cbp.est_state");
    const auto en = b.get<std::int16_t>("cbp.est_naics");
    const auto bands = b.get<std::int32_t>("cbp.est_bands");
    const std::size_t nb = lo.size();
    for (std::size_t r = 0; r < es.size(); ++r) {
        Est est;
        for (std::size_t k = 0; k < nb; ++k) {
            const auto t = bandSizes(lo[k], hi[k], bands(r, 2 * k), bands(r, 2 * k + 1));
            if (t.empty()) { continue; }
            est.counts.push_back(bands(r, 2 * k));
            est.sizes.insert(est.sizes.end(), t.begin(), t.end());
        }
        if (est.counts.empty()) { continue; }
        std::int64_t s = 0;
        for (auto c : est.counts) {
            est.cum.push_back(s += c);
        }
        m_est[{es[r], en[r]}] = std::move(est);
    }
}

std::int64_t SizeTables::target (std::int64_t state, std::int64_t naics) const {
    const auto it = m_targets.find({state, naics});
    return it == m_targets.end() ? m_default : it->second;
}

bool SizeTables::hasEst (std::int64_t state, std::int64_t naics) const {
    return m_est.count({state, naics}) != 0;
}

std::int64_t SizeTables::sample (std::int64_t state, std::int64_t naics, const KR64& k) const {
    const Est& e = m_est.at({state, naics});
    const auto band = intCdf(e.cum.data(), static_cast<std::int64_t>(e.cum.size()), k.u64(0));
    const auto i = index(k.u64(1), POINTS);
    return e.sizes[band * POINTS + i];
}

double SizeTables::mean (std::int64_t state, std::int64_t naics) const {
    const Est& e = m_est.at({state, naics});
    std::int64_t num = 0, den = 0;
    for (std::size_t b = 0; b < e.counts.size(); ++b) {
        std::int64_t s = 0;
        for (int i = 0; i < POINTS; ++i) {
            s += e.sizes[b * POINTS + i];
        }
        num += e.counts[b] * s;
        den += e.counts[b];
    }
    return static_cast<double>(num) / (POINTS * static_cast<double>(den));
}

// ---------------------------------------------------------------------------------------------
// S3 workers (workers.py: allocate, _fill_one; commute.py: background, prior)
// ---------------------------------------------------------------------------------------------

namespace {

constexpr int SOFT_ITERS = 200;
constexpr double IPF_TOL = 1e-9;
constexpr int REPAIR_SWEEPS = 12;
constexpr int TRAVEL_WFH = 7;
constexpr double W_SCALE = 65536.0; // commute.W_SCALE: quantised prior weight per job

//! Index i with edges[i] <= x < edges[i + 1], as numpy.searchsorted(edges, x, "right") - 1.
std::int64_t binOf (const ArrayView<double>& edges, double x) {
    return static_cast<std::int64_t>(std::upper_bound(edges.begin(), edges.end(), x) - edges.begin()) - 1;
}

//! Squared chord distance (km^2) between two points of geo.xyz, summed as commute.chord2.
double chord2 (const double* a, const double* b) {
    const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return (dx * dx + dy * dy) + dz * dz;
}

/*! The corrected prior between worker home block groups (commute.prior), as CSR in both
    directions: quantised weights, and each pair's kernel-distance bin. */
struct Flows {
    std::vector<std::int64_t> homes, dests;                  // sorted geoids
    std::vector<std::int64_t> hd_ptr, hd_col, hd_val, hd_kb; // home rows, destination indices ascending
    std::vector<std::int64_t> dh_ptr, dh_col, dh_val;        // destination rows, home indices ascending
};

//! Index of x in a sorted vector known to contain it.
std::int64_t position (const std::vector<std::int64_t>& v, std::int64_t x) {
    return std::lower_bound(v.begin(), v.end(), x) - v.begin();
}

/*! F.hd_* and F.dh_* from the LODES pairs (home index, destination index, jobs), sorted by (home,
    destination) and unique: the background of commute.background joined to them, both weighted
    by r(distance), rescaled to the pairs' total, quantised by W_SCALE with pairs rounding to 0
    dropped (commute.prior). */
void correctedPrior (const PopulationBundle& b, const std::vector<std::array<std::int64_t, 3>>& lodes, Flows& F) {
    const auto H = static_cast<std::int64_t>(F.homes.size()), Dn = static_cast<std::int64_t>(F.dests.size());
    const auto bgg = b.get<std::int64_t>("geo.geoid");
    const auto xyz = b.get<double>("geo.xyz");
    auto xyzOf = [&] (std::int64_t g) {
        const auto it = std::lower_bound(bgg.begin(), bgg.end(), g);
        if (it == bgg.end() || *it != g) { throw std::runtime_error("block group " + std::to_string(g) + " has no geo.xyz"); }
        return xyz.data() + 3 * (it - bgg.begin());
    };
    std::vector<const double*> xh(H), xd(Dn);
    for (std::int64_t h = 0; h < H; ++h) {
        xh[h] = xyzOf(F.homes[h]);
    }
    for (std::int64_t d = 0; d < Dn; ++d) {
        xd[d] = xyzOf(F.dests[d]);
    }
    const auto params = b.get<double>("commute.params");
    const double alpha = params[0], radius = params[1], floor_km = params[3];
    const auto kmax = static_cast<std::size_t>(params[2]);
    const double r2 = radius * radius, f2 = floor_km * floor_km;
    const auto kern_edges = b.get<double>("commute.kern_edges");
    const auto decay = b.get<double>("commute.decay");
    const auto dist_edges = b.get<double>("commute.dist_edges");
    const auto rel = b.get<double>("commute.r");

    // Jobs at each destination and from each home, summed in pair order (whole numbers).
    std::vector<double> J(Dn, 0.0), Hs(H, 0.0);
    std::vector<std::int64_t> lptr(H + 1, 0);
    for (const auto& e : lodes) {
        J[e[1]] += static_cast<double>(e[2]);
        Hs[e[0]] += static_cast<double>(e[2]);
        ++lptr[e[0] + 1];
    }
    for (std::int64_t h = 0; h < H; ++h) {
        lptr[h + 1] += lptr[h];
    }

    // Background per home: the k nearest destinations within the radius, by (distance, index),
    // weight (alpha H_h) g_d / S with g_d = J_d decay(bin), S summed in distance order.
    std::vector<std::vector<std::pair<std::int64_t, double>>> bgp(H);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads())
#endif
    for (std::int64_t h = 0; h < H; ++h) {
        if (Hs[h] == 0.0) { continue; }
        std::vector<std::pair<double, std::int64_t>> cand;
        for (std::int64_t d = 0; d < Dn; ++d) {
            const double d2 = chord2(xh[h], xd[d]);
            if (d2 <= r2) { cand.emplace_back(d2, d); }
        }
        std::sort(cand.begin(), cand.end());
        if (cand.size() > kmax) { cand.resize(kmax); }
        if (cand.empty()) { continue; }
        std::vector<double> g(cand.size());
        double s = 0.0;
        for (std::size_t i = 0; i < cand.size(); ++i) {
            g[i] = J[cand[i].second] * decay[binOf(kern_edges, std::sqrt(cand[i].first + f2))];
            s += g[i];
        }
        if (s <= 0.0) { continue; }
        auto& out = bgp[h];
        for (std::size_t i = 0; i < cand.size(); ++i) {
            out.emplace_back(cand[i].second, ((alpha * Hs[h]) * g[i]) / s);
        }
        std::sort(out.begin(), out.end());
    }

    // Union with LODES per home in destination order; weight c = jobs (+ background) times
    // r(distance); f rescales the total to sum(c); quantised.
    struct Pair {
        std::int64_t h, d;
        double c, w, d2;
    };
    std::vector<Pair> u;
    u.reserve(lodes.size());
    for (std::int64_t h = 0; h < H; ++h) {
        std::int64_t a = lptr[h];
        const std::int64_t z = lptr[h + 1];
        std::size_t k = 0;
        const auto& bh = bgp[h];
        while (a < z || k < bh.size()) {
            const std::int64_t dl = a < z ? lodes[a][1] : Dn, db = k < bh.size() ? bh[k].first : Dn;
            const std::int64_t d = std::min(dl, db);
            double c = dl == d ? static_cast<double>(lodes[a][2]) : 0.0;
            if (db == d) { c += bh[k].second; }
            if (dl == d) { ++a; }
            if (db == d) { ++k; }
            u.push_back({h, d, c, 0.0, chord2(xh[h], xd[d])});
        }
        bgp[h] = {};
    }
    double sc = 0.0, sw = 0.0;
    for (auto& p : u) {
        p.w = p.c * rel[binOf(dist_edges, std::sqrt(p.d2))];
        sc += p.c;
        sw += p.w;
    }
    const double f = sc / sw;
    F.hd_ptr.assign(H + 1, 0);
    std::vector<std::tuple<std::int64_t, std::int64_t, std::int64_t>> ent;
    for (const auto& p : u) {
        const auto q = static_cast<std::int64_t>(std::floor((p.w * f) * W_SCALE + 0.5));
        if (q <= 0) { continue; }
        ++F.hd_ptr[p.h + 1];
        F.hd_col.push_back(p.d);
        F.hd_val.push_back(q);
        F.hd_kb.push_back(binOf(kern_edges, std::sqrt(p.d2 + f2)));
        ent.emplace_back(p.h, p.d, q);
    }
    u = {};
    for (std::int64_t h = 0; h < H; ++h) {
        F.hd_ptr[h + 1] += F.hd_ptr[h];
    }
    totalSort(ent, [] (const auto& x, const auto& y) { // (h, d) pairs are unique
        return std::get<1>(x) != std::get<1>(y) ? std::get<1>(x) < std::get<1>(y) : std::get<0>(x) < std::get<0>(y);
    });
    F.dh_ptr.assign(Dn + 1, 0);
    for (const auto& [h, d, c] : ent) {
        ++F.dh_ptr[d + 1];
        F.dh_col.push_back(h);
        F.dh_val.push_back(c);
    }
    for (std::int64_t d = 0; d < Dn; ++d) {
        F.dh_ptr[d + 1] += F.dh_ptr[d];
    }
}

struct Fill {
    std::vector<std::int64_t> row, col_d, cnt; // row (index into the rows given), destination index, workers
    std::int64_t unplaced = 0;
};

/*! Soft IPF + column TRS + row repair for industry n over rows (home row_h, time band row_b)
    with the given supply; demand is per destination index.

    As workers._fill_one, whose entries are (row, demanded destination of the row's home) in
    row-major order. All of a home's rows share its destination list, so the entries are not
    stored: entry j of row r is pair j of its home's list, its prior q = weight x kern(band,
    distance bin) and its IPF value v = (q a_r) b_c are recomputed where needed -- the same
    doubles workers.py stores -- and only the counts and two index arrays are kept per entry.
    Every column sum runs over its entries in row order and every row sum in entry order, so the
    loops over rows or columns can run on threads (when called outside a parallel region)
    without changing a bit. */
Fill fillOne (const Flows& F, const ArrayView<double>& kern, const std::vector<std::int64_t>& row_h,
              const std::vector<std::int64_t>& row_b, const std::vector<std::int64_t>& supply,
              const std::vector<std::int64_t>& demand, std::int64_t nrb, int n, std::int64_t seed, std::int64_t rep,
              WorkerStats& st) {
    Fill out;
    const int nt = threads();
    const std::int64_t Dn = static_cast<std::int64_t>(F.dests.size());
    const std::size_t nk = kern.shape[1];
    std::vector<std::int64_t> col_pos(Dn, -1); // demanded destination -> position in ci
    std::vector<std::int64_t> ci;
    for (std::int64_t d = 0; d < Dn; ++d) {
        if (demand[d] > 0) {
            col_pos[d] = static_cast<std::int64_t>(ci.size());
            ci.push_back(d);
        }
    }
    // Each home's demanded destinations, once: position in ci, weight, kernel bin. Rows of a
    // home are adjacent (rows come sorted by (home, band)).
    std::vector<std::int32_t> pc, pkb;
    std::vector<double> pq;
    std::vector<std::int64_t> hp; // list start per distinct home, plus the end
    std::vector<std::int64_t> row_list(row_h.size());
    for (std::size_t r = 0; r < row_h.size(); ++r) {
        if (r > 0 && row_h[r] == row_h[r - 1]) {
            row_list[r] = row_list[r - 1];
            continue;
        }
        row_list[r] = static_cast<std::int64_t>(hp.size());
        hp.push_back(static_cast<std::int64_t>(pc.size()));
        const auto h = row_h[r];
        for (std::int64_t q = F.hd_ptr[h]; q < F.hd_ptr[h + 1]; ++q) {
            const auto c = col_pos[F.hd_col[q]];
            if (c < 0) { continue; }
            pc.push_back(static_cast<std::int32_t>(c));
            pq.push_back(static_cast<double>(F.hd_val[q]));
            pkb.push_back(static_cast<std::int32_t>(F.hd_kb[q]));
        }
    }
    hp.push_back(static_cast<std::int64_t>(pc.size()));
    // Rows with at least one demanded destination; columns reached by some such row.
    std::vector<std::int64_t> ri;
    for (std::size_t r = 0; r < row_h.size(); ++r) {
        if (hp[row_list[r] + 1] > hp[row_list[r]]) {
            ri.push_back(static_cast<std::int64_t>(r));
        } else {
            out.unplaced += supply[r];
        }
    }
    std::vector<std::uint8_t> col_used(ci.size(), 0);
    for (auto c : pc) { // every list belongs to some row, and a non-empty list to a kept one
        col_used[c] = 1;
    }
    std::vector<std::int32_t> col_final(ci.size(), -1);
    std::vector<std::int64_t> ci2;
    for (std::size_t c = 0; c < ci.size(); ++c) {
        if (col_used[c]) {
            col_final[c] = static_cast<std::int32_t>(ci2.size());
            ci2.push_back(ci[c]);
        }
    }
    for (auto& c : pc) {
        c = col_final[c];
    }
    const std::size_t NR = ri.size(), NC = ci2.size();
    // Fill row r: list start ls[r], length through rptr, kernel row kr[r].
    std::vector<std::int64_t> rptr(NR + 1, 0), ls(NR);
    std::vector<const double*> kr(NR);
    for (std::size_t r = 0; r < NR; ++r) {
        const auto l = row_list[ri[r]];
        ls[r] = hp[l];
        rptr[r + 1] = rptr[r] + (hp[l + 1] - hp[l]);
        kr[r] = kern.data() + static_cast<std::size_t>(row_b[ri[r]]) * nk;
    }
    const std::int64_t NZ = rptr[NR];
    if (NZ == 0) {
        for (auto r : ri) {
            out.unplaced += supply[r];
        }
        return out;
    }
    if (NZ >= std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("industry " + std::to_string(n) + ": too many fill entries");
    }
    auto qOf = [&] (std::size_t r, std::int64_t p) { // entry prior, as workers.py's v0
        return pq[p] * kr[r][pkb[p]];
    };
    std::vector<double> rt(NR), ct(NC);
    std::int64_t ssum = 0, dsum = 0;
    for (std::size_t r = 0; r < NR; ++r) {
        rt[r] = static_cast<double>(supply[ri[r]]);
        ssum += supply[ri[r]];
    }
    for (std::size_t c = 0; c < NC; ++c) {
        dsum += demand[ci2[c]];
    }
    const double scale = static_cast<double>(ssum) / static_cast<double>(dsum);
    for (std::size_t c = 0; c < NC; ++c) {
        ct[c] = static_cast<double>(demand[ci2[c]]) * scale;
    }
    // Entries by column, row order within a column (a stable counting sort), and each entry's row.
    std::vector<std::int64_t> cp(NC + 1, 0);
    for (std::size_t r = 0; r < NR; ++r) {
        for (std::int64_t e = rptr[r]; e < rptr[r + 1]; ++e) {
            ++cp[pc[ls[r] + (e - rptr[r])] + 1];
        }
    }
    for (std::size_t c = 0; c < NC; ++c) {
        cp[c + 1] += cp[c];
    }
    std::vector<std::int32_t> oce(NZ), erow(NZ);
    {
        std::vector<std::int64_t> at(cp.begin(), cp.end() - 1);
        for (std::size_t r = 0; r < NR; ++r) {
            for (std::int64_t e = rptr[r]; e < rptr[r + 1]; ++e) {
                oce[at[pc[ls[r] + (e - rptr[r])]]++] = static_cast<std::int32_t>(e);
                erow[e] = static_cast<std::int32_t>(r);
            }
        }
    }
    auto pairOf = [&] (std::int64_t e, std::size_t r) {
        return ls[r] + (e - rptr[r]);
    };

    // Soft IPF, as workers.py: v = q a_i b_j; b_j = sqrt(ct_j / S_j) with S_j = sum_i q_ij a_i,
    // then a_i = rt_i / sum_j q_ij b_j; stop once the column sums move less than tol.
    const double tol = IPF_TOL * std::max(1.0, seqSum(rt.data(), NR));
    std::vector<double> a(NR, 1.0), S(NC), bc(NC), cs(NC), prev(NC);
    // Column sums over each column's entries in row order -- row by row on one thread, column by
    // column on several: S[c] = sum q a_r (the next iteration's prior sums) and, when bcp is
    // given, cs[c] = sum (q a_r) b_c in the same pass, both from the current a.
    auto colSums = [&] (const double* bcp) {
        if (nt == 1) {
            std::fill(S.begin(), S.end(), 0.0);
            if (bcp) { std::fill(cs.begin(), cs.end(), 0.0); }
            for (std::size_t r = 0; r < NR; ++r) {
                const double ar = a[r];
                for (std::int64_t p = ls[r]; p < ls[r] + (rptr[r + 1] - rptr[r]); ++p) {
                    const double t = qOf(r, p) * ar;
                    S[pc[p]] += t;
                    if (bcp) { cs[pc[p]] += t * bcp[pc[p]]; }
                }
            }
            return;
        }
        const auto nc = static_cast<std::int64_t>(NC);
#ifdef _OPENMP
#pragma omp parallel for if (nt > 1) schedule(dynamic, 256) num_threads(nt)
#endif
        for (std::int64_t c = 0; c < nc; ++c) {
            double s = 0.0, v = 0.0;
            for (std::int64_t k = cp[c]; k < cp[c + 1]; ++k) {
                const std::int64_t e = oce[k];
                const auto r = static_cast<std::size_t>(erow[e]);
                const double t = qOf(r, pairOf(e, r)) * a[r];
                s += t;
                if (bcp) { v += t * bcp[c]; }
            }
            S[c] = s;
            if (bcp) { cs[c] = v; }
        }
    };
    bool have_prev = false;
    const auto nr64 = static_cast<std::int64_t>(NR);
    colSums(nullptr);
    for (int it = 0; it < SOFT_ITERS; ++it) {
        for (std::size_t c = 0; c < NC; ++c) {
            bc[c] = std::sqrt(S[c] > 0 ? ct[c] / S[c] : 1.0);
        }
#ifdef _OPENMP
#pragma omp parallel for if (nt > 1) schedule(dynamic, 1024) num_threads(nt)
#endif
        for (std::int64_t r = 0; r < nr64; ++r) {
            double t = 0.0;
            for (std::int64_t p = ls[r]; p < ls[r] + (rptr[r + 1] - rptr[r]); ++p) {
                t += qOf(r, p) * bc[pc[p]];
            }
            a[r] = t > 0 ? rt[r] / t : 0.0;
        }
        colSums(bc.data()); // cs for this iteration, S for the next
        if (have_prev) {
            double moved = 0.0;
            for (std::size_t c = 0; c < NC; ++c) {
                moved += std::abs(cs[c] - prev[c]);
            }
            if (moved < tol) { break; }
        }
        prev = cs;
        have_prev = true;
    }
    auto vOf = [&] (std::int64_t e) { // the final IPF value of entry e
        const auto r = static_cast<std::size_t>(erow[e]);
        const auto p = pairOf(e, r);
        return (qOf(r, p) * a[r]) * bc[pc[p]];
    };

    // Integer column targets: the soft column sums, largest remainder (ties by destination
    // geoid) up to the supply total.
    std::vector<std::int64_t> ct_i(NC);
    std::int64_t have_ct = 0;
    for (std::size_t c = 0; c < NC; ++c) {
        ct_i[c] = static_cast<std::int64_t>(std::floor(cs[c]));
        have_ct += ct_i[c];
    }
    if (ssum > have_ct) {
        const auto o = stableOrder(NC, [&] (std::int64_t x, std::int64_t y) {
            const double fx = -(cs[x] - std::floor(cs[x])), fy = -(cs[y] - std::floor(cs[y]));
            return fx != fy ? fx < fy : F.dests[ci2[x]] < F.dests[ci2[y]];
        });
        for (std::int64_t i = 0; i < ssum - have_ct; ++i) {
            ++ct_i[o[i]];
        }
    }

    std::vector<std::int32_t> cnt(NZ);
    const auto nz64 = static_cast<std::int64_t>(NZ);
#ifdef _OPENMP
#pragma omp parallel for if (nt > 1) schedule(static) num_threads(nt)
#endif
    for (std::int64_t e = 0; e < nz64; ++e) {
        cnt[e] = static_cast<std::int32_t>(std::floor(vOf(e)));
    }
    auto homeOf = [&] (std::size_t r) {
        return F.homes[row_h[ri[r]]];
    };
    auto bandOf = [&] (std::size_t r) {
        return row_b[ri[r]];
    };

    // Column-wise TRS, each column's entries in row order; columns touch only their own entries.
    const auto nc64 = static_cast<std::int64_t>(NC);
#ifdef _OPENMP
#pragma omp parallel for if (nt > 1) schedule(dynamic, 64) num_threads(nt)
#endif
    for (std::int64_t cj = 0; cj < nc64; ++cj) {
        const std::int64_t k0 = cp[cj], m = cp[cj + 1] - cp[cj];
        const std::int64_t dg = F.dests[ci2[cj]];
        std::int64_t have = 0;
        for (std::int64_t k = k0; k < k0 + m; ++k) {
            have += cnt[oce[k]];
        }
        const std::int64_t need = ct_i[cj] - have;
        if (need > 0) {
            std::vector<double> frac(m), cum;
            for (std::int64_t i = 0; i < m; ++i) {
                const double v = vOf(oce[k0 + i]);
                frac[i] = v - std::floor(v);
            }
            if (seqSum(frac.data(), frac.size()) > 0) {
                runningSum(frac.data(), frac.size(), cum);
                const KR64 k = KR64(seed, rep, Stage::IPF_TRS_ADD).with(n).with(dg);
                for (std::int64_t j = 0; j < need; ++j) {
                    ++cnt[oce[k0 + floatCdf(cum.data(), m, k.with(j).u64(0))]];
                }
            }
        } else if (need < 0) {
            std::vector<std::int64_t> nz;
            for (std::int64_t k = k0; k < k0 + m; ++k) {
                if (cnt[oce[k]] > 0) { nz.push_back(oce[k]); }
            }
            std::vector<std::int64_t> id(nz.size());
            std::vector<std::uint64_t> dr(nz.size());
            const KR64 k = KR64(seed, rep, Stage::IPF_TRS_TRIM).with(n).with(dg);
            for (std::size_t i = 0; i < nz.size(); ++i) {
                const auto r = static_cast<std::size_t>(erow[nz[i]]);
                id[i] = homeOf(r) * nrb + bandOf(r);
                dr[i] = k.with(homeOf(r)).with(bandOf(r)).u64(0);
            }
            std::vector<std::int64_t> o(nz.size());
            std::iota(o.begin(), o.end(), 0);
            std::sort(o.begin(), o.end(), [&] (std::int64_t x, std::int64_t y) { // (draw, identity): total
                return dr[x] != dr[y] ? dr[x] < dr[y] : id[x] < id[y];
            });
            const std::size_t take = std::min<std::size_t>(static_cast<std::size_t>(-need), nz.size());
            for (std::size_t i = 0; i < take; ++i) {
                --cnt[nz[o[i]]];
            }
        }
    }

    // Row repair: moves stay inside a row, whose cells are in destination order -- entry order.
    // Additions follow the IPF values; removals the counts. Rows touch only their own entries.
    auto rowDelta = [&] (std::size_t r) {
        std::int64_t got = 0;
        for (std::int64_t e = rptr[r]; e < rptr[r + 1]; ++e) {
            got += cnt[e];
        }
        return supply[ri[r]] - got;
    };
    std::vector<std::int64_t> delta(NR);
    for (int sweep = 0; sweep < REPAIR_SWEEPS; ++sweep) {
        std::int64_t bad = 0;
#ifdef _OPENMP
#pragma omp parallel for if (nt > 1) schedule(dynamic, 1024) num_threads(nt) reduction(+ : bad)
#endif
        for (std::int64_t r = 0; r < nr64; ++r) {
            delta[r] = rowDelta(r);
            bad += delta[r] != 0 ? 1 : 0;
        }
        if (bad == 0) { break; }
#ifdef _OPENMP
#pragma omp parallel for if (nt > 1) schedule(dynamic, 256) num_threads(nt)
#endif
        for (std::int64_t r = 0; r < nr64; ++r) {
            const std::int64_t d = delta[r];
            if (d == 0) { continue; }
            const std::int64_t c0 = rptr[r], m = rptr[r + 1] - rptr[r];
            const KR64 k = KR64(seed, rep, Stage::IPF_REPAIR).with(n).with(homeOf(r)).with(bandOf(r)).with(sweep);
            if (d > 0) {
                std::vector<double> vr(m), cum;
                for (std::int64_t i = 0; i < m; ++i) {
                    vr[i] = vOf(c0 + i);
                }
                runningSum(vr.data(), static_cast<std::size_t>(m), cum);
                std::vector<std::int64_t> add(m, 0);
                for (std::int64_t j = 0; j < d; ++j) {
                    ++add[floatCdf(cum.data(), m, k.with(j).u64(0))];
                }
                for (std::int64_t i = 0; i < m; ++i) {
                    cnt[c0 + i] += static_cast<std::int32_t>(add[i]);
                }
            } else {
                std::vector<std::int64_t> wc(m);
                std::int64_t s = 0;
                for (std::int64_t i = 0; i < m; ++i) {
                    wc[i] = (s += cnt[c0 + i]);
                }
                if (s == 0) { continue; }
                const std::int64_t take = std::min(-d, s);
                for (std::int64_t j = 0; j < take; ++j) {
                    const auto i = intCdf(wc.data(), m, k.with(j).u64(0));
                    if (cnt[c0 + i] > 0) { --cnt[c0 + i]; }
                }
            }
        }
    }
    // Deterministic final pass: settle what repair left on the row's largest cells.
    for (std::size_t r = 0; r < NR; ++r) {
        std::int64_t d = rowDelta(r);
        if (d == 0) { continue; }
        ++st.repair_final;
        const std::int64_t c0 = rptr[r], m = rptr[r + 1] - rptr[r];
        while (d != 0) {
            std::int64_t j = c0;
            for (std::int64_t i = 1; i < m; ++i) {
                if (cnt[c0 + i] > cnt[j]) { j = c0 + i; }
            }
            const std::int64_t step = d > 0 ? 1 : -1;
            if (step < 0 && cnt[j] == 0) { break; }
            cnt[j] += static_cast<std::int32_t>(step);
            d -= step;
        }
        if (rowDelta(r) != 0) { throw std::runtime_error("industry " + std::to_string(n) + ": row sums not exact after repair"); }
    }
    for (std::size_t r = 0; r < NR; ++r) {
        for (std::int64_t e = rptr[r]; e < rptr[r + 1]; ++e) {
            if (cnt[e] <= 0) { continue; }
            if (cnt[e] == 1) { ++st.one_worker_cells; }
            out.row.push_back(ri[r]);
            out.col_d.push_back(ci2[pc[pairOf(e, r)]]);
            out.cnt.push_back(cnt[e]);
        }
    }
    return out;
}

} // namespace

std::vector<std::int64_t> allocateWorkers (const PopulationBundle& b, const Persons& P, const SizeTables& tables,
                                           std::int64_t seed, std::int64_t rep, WorkerStats* stats_out, const Partition* part) {
    WorkerStats st;
    const std::size_t n_persons = P.size();
    std::vector<std::int64_t> work(P.bg);
    // Commuters only: a worker who works from home keeps the home block group.
    std::vector<std::int64_t> W;
    for (std::size_t i = 0; i < n_persons; ++i) {
        if (P.employed[i] && P.travel[i] != TRAVEL_WFH) { W.push_back(static_cast<std::int64_t>(i)); }
    }
    const std::size_t NW = W.size();
    const int n_naics = static_cast<int>(b.get<std::int64_t>("naics.codes.offsets").size()) - 1;

    Flows F;
    for (auto i : W) {
        F.homes.push_back(P.bg[i]);
    }
    std::sort(F.homes.begin(), F.homes.end());
    F.homes.erase(std::unique(F.homes.begin(), F.homes.end()), F.homes.end());
    std::vector<std::int64_t> hidx_w(NW), naics_w(NW);
    for (std::size_t w = 0; w < NW; ++w) {
        hidx_w[w] = position(F.homes, P.bg[W[w]]);
        naics_w[w] = P.naics[W[w]];
    }
    const std::int64_t H = static_cast<std::int64_t>(F.homes.size());

    // 1. The corrected prior: LODES pairs with both ends at worker homes (summed if repeated, as
    //    scipy's CSR does), plus the background, weighted and quantised (commute.prior).
    {
        const auto lh = b.get<std::int64_t>("lodes.home_geoid");
        const auto ld = b.get<std::int64_t>("lodes.dest_geoid");
        const auto ip = b.get<std::int64_t>("lodes.indptr");
        const auto ix = b.get<std::int32_t>("lodes.indices");
        const auto dv = b.get<std::int32_t>("lodes.data");
        auto isHome = [&] (std::int64_t g) {
            return std::binary_search(F.homes.begin(), F.homes.end(), g);
        };
        // (home idx, dest geoid, count) entries, sorted, then repeated pairs summed: the pairs come
        // out in (home, dest) order with the same totals a map accumulating them would hold.
        std::vector<std::array<std::int64_t, 3>> raw;
        for (std::size_t r = 0; r < lh.size(); ++r) {
            if (!isHome(lh[r])) { continue; }
            const auto h = position(F.homes, lh[r]);
            for (std::int64_t q = ip[r]; q < ip[r + 1]; ++q) {
                const std::int64_t dg = ld[ix[q]];
                if (isHome(dg)) { raw.push_back({h, dg, static_cast<std::int64_t>(dv[q])}); }
            }
        }
        totalSort(raw, std::less<std::array<std::int64_t, 3>>());
        std::vector<std::array<std::int64_t, 3>> pairs;
        for (const auto& e : raw) {
            if (!pairs.empty() && pairs.back()[0] == e[0] && pairs.back()[1] == e[1]) {
                pairs.back()[2] += e[2];
            } else {
                pairs.push_back(e);
            }
        }
        raw = {};
        for (const auto& e : pairs) {
            F.dests.push_back(e[1]);
        }
        std::sort(F.dests.begin(), F.dests.end());
        F.dests.erase(std::unique(F.dests.begin(), F.dests.end()), F.dests.end());
        for (auto& e : pairs) { // dest geoid -> index; still in (home, dest) order
            e[1] = position(F.dests, e[1]);
        }
        correctedPrior(b, pairs, F);
    }
    const std::int64_t Dn = static_cast<std::int64_t>(F.dests.size());
    std::vector<std::int64_t> dest_total(Dn, 0);
    for (std::int64_t d = 0; d < Dn; ++d) {
        for (std::int64_t q = F.dh_ptr[d]; q < F.dh_ptr[d + 1]; ++q) {
            dest_total[d] += F.dh_val[q];
        }
    }

    // 2. Commute-reachable industry supply: local = flow_dh @ home_naics, exactly.
    std::vector<std::int64_t> home_naics(static_cast<std::size_t>(H) * n_naics, 0);
    for (std::size_t w = 0; w < NW; ++w) {
        ++home_naics[hidx_w[w] * n_naics + naics_w[w]];
    }
    std::vector<std::int64_t> local(static_cast<std::size_t>(Dn) * n_naics, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
#endif
    for (std::int64_t d = 0; d < Dn; ++d) {
        for (std::int64_t q = F.dh_ptr[d]; q < F.dh_ptr[d + 1]; ++q) {
            const std::int64_t* hn = &home_naics[F.dh_col[q] * n_naics];
            std::int64_t* ln = &local[d * n_naics];
            for (int n = 0; n < n_naics; ++n) {
                ln[n] += F.dh_val[q] * hn[n];
            }
        }
    }

    // 3. Establishment slots per destination: industries from its commute shed, CBP sizes.
    std::map<std::int64_t, std::vector<std::int64_t>> sn_count; // state -> workers per NAICS
    for (std::size_t w = 0; w < NW; ++w) {
        auto& c = sn_count[P.bg[W[w]] / 10000000000LL];
        if (c.empty()) { c.assign(n_naics, 0); }
        ++c[naics_w[w]];
    }
    std::int64_t tsum = 0;
    for (const auto& [stt, c] : sn_count) {
        for (int n = 0; n < n_naics; ++n) {
            if (c[n] > 0) { tsum += c[n] * tables.target(stt, n); }
        }
    }
    const double avg = static_cast<double>(tsum) / static_cast<double>(NW);
    std::vector<std::int64_t> implied(static_cast<std::size_t>(Dn) * n_naics, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
#endif
    for (std::int64_t d = 0; d < Dn; ++d) {
        std::vector<std::int64_t> cum(n_naics);
        const double jobs = static_cast<double>(dest_total[d]) / W_SCALE;
        const std::int64_t ns =
                dest_total[d] > 0 ? std::max<std::int64_t>(1, static_cast<std::int64_t>(std::nearbyint(jobs / avg))) : 0;
        std::int64_t s = 0;
        for (int n = 0; n < n_naics; ++n) {
            cum[n] = (s += local[d * n_naics + n]);
        }
        if (ns == 0 || s == 0) { continue; }
        const std::int64_t dg = F.dests[d], stt = dg / 10000000000LL;
        const KR64 kn = KR64(seed, rep, Stage::SLOT_NAICS).with(dg);
        const KR64 ks = KR64(seed, rep, Stage::SLOT_SIZE).with(dg);
        for (std::int64_t slot = 0; slot < ns; ++slot) {
            const auto n = intCdf(cum.data(), n_naics, kn.with(slot).u64(0));
            implied[d * n_naics + n] += tables.hasEst(stt, n) ? tables.sample(stt, n, ks.with(slot)) : tables.target(stt, n);
        }
    }

    // 4. Rescale each industry to its true worker count, exactly (largest remainder, ties by geoid).
    std::vector<std::int64_t> true_total(n_naics, 0);
    for (auto n : naics_w) {
        ++true_total[n];
    }
    std::vector<std::int64_t> demand(static_cast<std::size_t>(Dn) * n_naics, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads())
#endif
    for (int n = 0; n < n_naics; ++n) {
        const std::int64_t T = true_total[n];
        std::int64_t S = 0;
        for (std::int64_t d = 0; d < Dn; ++d) {
            S += implied[d * n_naics + n];
        }
        if (T == 0 || S == 0) { continue; }
        std::vector<std::int64_t> col(Dn), rem(Dn);
        std::int64_t have = 0;
        for (std::int64_t d = 0; d < Dn; ++d) {
            const std::int64_t q = implied[d * n_naics + n] * T;
            col[d] = q / S;
            rem[d] = q % S;
            have += col[d];
        }
        const std::int64_t shortfall = T - have;
        if (shortfall > 0) {
            const auto o = stableOrder(Dn, [&] (std::int64_t x, std::int64_t y) {
                return rem[x] != rem[y] ? rem[x] > rem[y] : F.dests[x] < F.dests[y];
            });
            for (std::int64_t i = 0; i < shortfall; ++i) {
                ++col[o[i]];
            }
        }
        for (std::int64_t d = 0; d < Dn; ++d) {
            demand[d * n_naics + n] = col[d];
        }
    }

    // 5. Soft IPF fill per industry over (home, time band) rows; each row's workers, in keyed
    //    order, dealt to its cells in destination order.
    const auto kern = b.get<double>("commute.kern");
    const auto band_tab = b.get<std::int8_t>("commute.band");
    const auto nrb = static_cast<std::int64_t>(kern.shape[0]);
    const auto n_modes = static_cast<std::int64_t>(band_tab.shape[0]), n_min = static_cast<std::int64_t>(band_tab.shape[1]);
    std::vector<std::int64_t> rowid_w(NW);
    for (std::size_t w = 0; w < NW; ++w) {
        const std::int64_t t = P.travel[W[w]];
        const std::int64_t band =
                t < 0 ? nrb - 1
                      : band_tab(static_cast<std::size_t>(std::min(t, n_modes - 1)),
                                 static_cast<std::size_t>(std::clamp<std::int64_t>(P.jwmnp[W[w]], 0, n_min - 1)));
        rowid_w[w] = hidx_w[w] * nrb + band;
    }
    std::vector<std::uint8_t> assigned(NW, 0);
    std::vector<std::uint64_t> okey(NW);
    const KR64 ka(seed, rep, Stage::WORK_ASSIGN);
    const auto nw64 = static_cast<std::int64_t>(NW);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads())
#endif
    for (std::int64_t w = 0; w < nw64; ++w) {
        const auto i = W[w];
        okey[w] = ka.with(P.bg[i]).with(P.h[i]).with(P.p[i]).u64(0);
    }
    const auto wsort = stableOrder(NW, [&] (std::int64_t x, std::int64_t y) {
        if (naics_w[x] != naics_w[y]) { return naics_w[x] < naics_w[y]; }
        if (rowid_w[x] != rowid_w[y]) { return rowid_w[x] < rowid_w[y]; }
        if (okey[x] != okey[y]) { return okey[x] < okey[y]; }
        const auto i = W[x], j = W[y];
        return P.h[i] != P.h[j] ? P.h[i] < P.h[j] : P.p[i] < P.p[j];
    });
    std::vector<std::int64_t> nb(n_naics + 1, 0); // industry boundaries in wsort
    for (auto w : wsort) {
        ++nb[naics_w[w] + 1];
    }
    for (int n = 0; n < n_naics; ++n) {
        nb[n + 1] += nb[n];
    }
    // Industries are independent -- each fills and deals out only its own workers -- so they run
    // concurrently: over MPI ranks when the caller gives a Partition (largest first, by the
    // entries each fill can see), and over threads within a rank. An industry's result is the
    // destination of each of its workers in wsort order (-1: not placed), which every rank then
    // applies in industry order; the stats are integer counts. None of it depends on the split.
    std::vector<std::int64_t> cost(n_naics, 0);
    for (std::int64_t h = 0; h < H; ++h) {
        const std::int64_t row = F.hd_ptr[h + 1] - F.hd_ptr[h];
        for (int n = 0; n < n_naics; ++n) {
            if (home_naics[h * n_naics + n] > 0) { cost[n] += row; }
        }
    }
    std::vector<int> by_size(n_naics);
    std::iota(by_size.begin(), by_size.end(), 0);
    std::stable_sort(by_size.begin(), by_size.end(), [&] (int x, int y) {
        return cost[x] > cost[y];
    });
    const int nranks = (part && part->nranks > 1) ? part->nranks : 1, me = nranks > 1 ? part->rank : 0;
    std::vector<int> owner(n_naics, 0);
    {
        std::vector<std::int64_t> load(nranks, 0);
        for (int n : by_size) {
            const int r = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
            owner[n] = r;
            load[r] += cost[n];
        }
    }
    std::vector<int> mine;
    for (int n : by_size) {
        if (owner[n] == me) { mine.push_back(n); }
    }
    std::vector<std::vector<std::int64_t>> dest(n_naics);
    std::vector<WorkerStats> ist(n_naics);
    std::vector<std::string> errors(n_naics);
    auto fillIndustry = [&] (int n) {
        const std::int64_t* wn = wsort.data() + nb[n];
        const std::int64_t nwn = nb[n + 1] - nb[n];
        auto& dn = dest[n];
        dn.assign(static_cast<std::size_t>(nwn), -1);
        std::vector<std::int64_t> dem(Dn);
        std::int64_t dsum = 0;
        for (std::int64_t d = 0; d < Dn; ++d) {
            dsum += (dem[d] = demand[d * n_naics + n]);
        }
        if (true_total[n] == 0 || dsum == 0) { return; }
        // rows: the industry's distinct (home, band), ascending, with their worker counts; each
        // row's workers are contiguous in wn, starting at row_first
        std::vector<std::int64_t> row_h, row_b, sup, row_first;
        for (std::int64_t k = 0; k < nwn; ++k) {
            const auto id = rowid_w[wn[k]];
            if (k == 0 || id != rowid_w[wn[k - 1]]) {
                row_h.push_back(id / nrb);
                row_b.push_back(id % nrb);
                sup.push_back(0);
                row_first.push_back(k);
            }
            ++sup.back();
        }
        WorkerStats& sn = ist[n];
        Fill f;
        try {
            f = fillOne(F, kern, row_h, row_b, sup, dem, nrb, n, seed, rep, sn);
        } catch (const std::exception& e) { // must not leave the parallel region
            errors[n] = e.what();
            return;
        }
        sn.unplaceable += f.unplaced;
        if (f.cnt.empty()) { return; }
        sn.cells += static_cast<std::int64_t>(f.cnt.size());
        // cells come by (row, destination) already: rows ascending, destinations ascending
        // within a row, and F.dests is sorted
        const std::size_t nc = f.cnt.size();
        for (std::size_t a = 0; a < nc;) {
            std::size_t z = a;
            while (z < nc && f.row[z] == f.row[a]) {
                ++z;
            }
            std::int64_t k = row_first[f.row[a]];
            for (std::size_t c = a; c < z; ++c) {
                for (std::int64_t r = 0; r < f.cnt[c]; ++r, ++k) {
                    dn[k] = f.col_d[c];
                }
            }
            a = z;
        }
    };
    // An industry bigger than an even share of this rank's work runs alone on all threads (its
    // fill parallelises over rows and columns); the rest run concurrently, one per thread.
    const int nt = threads();
    std::int64_t mine_cost = 0;
    for (int n : mine) {
        mine_cost += cost[n];
    }
    std::vector<int> big, small;
    for (int n : mine) {
        (nt > 1 && cost[n] * nt >= mine_cost ? big : small).push_back(n);
    }
    for (int n : big) {
        fillIndustry(n);
    }
    const auto n_small = static_cast<int>(small.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
#endif
    for (int q = 0; q < n_small; ++q) {
        fillIndustry(small[q]);
    }
    // Every rank's results, in rank order: a header [failed industries, the four stats], then
    // its industries' destinations in ascending industry order.
    std::vector<std::int64_t> buf(5, 0);
    for (int n : mine) {
        buf[0] += errors[n].empty() ? 0 : 1;
        buf[1] += ist[n].unplaceable;
        buf[2] += ist[n].cells;
        buf[3] += ist[n].repair_final;
        buf[4] += ist[n].one_worker_cells;
    }
    for (int n = 0; n < n_naics; ++n) {
        if (owner[n] == me) { buf.insert(buf.end(), dest[n].begin(), dest[n].end()); }
    }
    dest = {};
    const std::vector<std::int64_t> all = nranks > 1 ? part->allgather(buf) : std::move(buf);
    std::int64_t failed = 0;
    std::size_t pos = 0;
    for (int r = 0; r < nranks; ++r) {
        failed += all[pos];
        st.unplaceable += all[pos + 1];
        st.cells += all[pos + 2];
        st.repair_final += all[pos + 3];
        st.one_worker_cells += all[pos + 4];
        pos += 5;
        for (int n = 0; n < n_naics; ++n) {
            if (owner[n] != r) { continue; }
            for (std::int64_t k = nb[n]; k < nb[n + 1]; ++k, ++pos) {
                const std::int64_t d = all[pos];
                if (d < 0) { continue; }
                work[W[wsort[k]]] = F.dests[d];
                assigned[wsort[k]] = 1;
            }
        }
    }
    if (failed > 0) {
        for (int n : mine) {
            if (!errors[n].empty()) { throw std::runtime_error(errors[n]); }
        }
        throw std::runtime_error("worker allocation failed for " + std::to_string(failed) + " industries on another rank");
    }

    // Fallback: a draw over the home's own prior row; no row, work at home.
    const KR64 kf(seed, rep, Stage::WORK_FALLBACK);
    std::vector<std::int64_t> rc;
    for (std::size_t w = 0; w < NW; ++w) {
        if (assigned[w]) { continue; }
        ++st.fallback;
        const auto i = W[w];
        const auto h = hidx_w[w];
        const std::int64_t s = F.hd_ptr[h], e = F.hd_ptr[h + 1];
        if (e > s) {
            rc.resize(e - s);
            std::int64_t c = 0;
            for (std::int64_t q = s; q < e; ++q) {
                rc[q - s] = (c += F.hd_val[q]);
            }
            const auto j = intCdf(rc.data(), e - s, kf.with(P.bg[i]).with(P.h[i]).with(P.p[i]).u64(0));
            work[i] = F.dests[F.hd_col[s + j]];
        }
    }
    st.workers = static_cast<std::int64_t>(NW);
    if (stats_out) { *stats_out = st; }
    return work;
}

// ---------------------------------------------------------------------------------------------
// S4 students (students.py: allocate, _fill)
// ---------------------------------------------------------------------------------------------

namespace {

const char* const LEVELS[6] = {"P", "E", "M", "H", "U", "C"};
constexpr int LEVEL_LO[6] = {4, 5, 11, 14, 18, 3};
constexpr int LEVEL_HI[6] = {4, 10, 13, 17, 19, 3};
constexpr int LEVEL_C = 5;
const std::vector<int> SCALES = {12, 11, 10, 7, 5};
const std::vector<int> CHILDCARE_SCALES = {12, 11, 10};
// Columns of schools.level_places (P, E, M, H); the level index keying the pass that places
// unplaced preschoolers in childcare.
constexpr int N_PLACE_LEVELS = 4;
constexpr int PRESCHOOL_IN_CHILDCARE = 6;

std::int64_t prefix (std::int64_t geoid, int s) {
    static const std::int64_t pow10[13] = {1,        10,        100,        1000,        10000,        100000,       1000000,
                                           10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000};
    return geoid / pow10[12 - s];
}

struct Schools {
    std::vector<std::int64_t> geoid, ord, students, teachers;
    std::vector<std::int64_t> places; // places per level P, E, M, H, N_PLACE_LEVELS per school
    std::vector<std::string> level;   // level name per school
    std::vector<std::int64_t> county, adj_ptr, adj_ix;

    explicit Schools (const PopulationBundle& b) {
        const auto g = b.get<std::int64_t>("schools.geoid");
        const auto o = b.get<std::int16_t>("schools.ord");
        const auto s = b.get<std::int32_t>("schools.students");
        const auto t = b.get<std::int32_t>("schools.teachers");
        const auto lp = b.get<std::int32_t>("schools.level_places");
        const auto lv = b.get<std::int8_t>("schools.level");
        const auto names = b.strings("schools.level_names");
        if (lp.size() != g.size() * N_PLACE_LEVELS) {
            throw std::runtime_error("schools.level_places does not hold " + std::to_string(N_PLACE_LEVELS) +
                                     " levels per school");
        }
        places.assign(lp.begin(), lp.end());
        for (std::size_t i = 0; i < g.size(); ++i) {
            geoid.push_back(g[i]);
            ord.push_back(o[i]);
            students.push_back(s[i]);
            teachers.push_back(t[i]);
            level.push_back(names.at(lv[i]));
        }
        const auto c = b.get<std::int32_t>("adjacency.county");
        const auto ip = b.get<std::int64_t>("adjacency.indptr");
        const auto ix = b.get<std::int32_t>("adjacency.indices");
        county.assign(c.begin(), c.end());
        adj_ptr.assign(ip.begin(), ip.end());
        adj_ix.assign(ix.begin(), ix.end());
    }

    //! A county and its neighbours, sorted.
    std::vector<std::int64_t> neighbourhood (std::int64_t cty) const {
        std::vector<std::int64_t> nb = {cty};
        const auto k = std::lower_bound(county.begin(), county.end(), cty) - county.begin();
        if (k < static_cast<std::int64_t>(county.size()) && county[k] == cty) {
            for (std::int64_t j = adj_ptr[k]; j < adj_ptr[k + 1]; ++j) {
                nb.push_back(county[adj_ix[j]]);
            }
        }
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        return nb;
    }
};

//! Schools of candidate list cand (indices into rows) in keyed fill order: (draw, geoid, ord).
std::vector<std::int64_t> keyedSchoolOrder (const std::vector<std::int64_t>& cand, const std::vector<std::int64_t>& rows,
                                            const Schools& S, const KR64& base) {
    std::vector<std::uint64_t> dr(cand.size());
    for (std::size_t i = 0; i < cand.size(); ++i) {
        const auto r = rows[cand[i]];
        dr[i] = base.with(S.geoid[r]).with(S.ord[r]).u64(0);
    }
    const auto o = stableOrder(cand.size(), [&] (std::int64_t x, std::int64_t y) {
        if (dr[x] != dr[y]) { return dr[x] < dr[y]; }
        const auto rx = rows[cand[x]], ry = rows[cand[y]];
        return S.geoid[rx] != S.geoid[ry] ? S.geoid[rx] < S.geoid[ry] : S.ord[rx] < S.ord[ry];
    });
    std::vector<std::int64_t> out(cand.size());
    for (std::size_t i = 0; i < cand.size(); ++i) {
        out[i] = cand[o[i]];
    }
    return out;
}

//! One region (students.py, _fill): who (canonical order) into candidate schools; returns
//! (positions in who that were placed, row-local school index for each).
void fillRegion (const std::vector<std::int64_t>& who, const std::vector<std::int64_t>& cand,
                 const std::vector<std::int64_t>& remaining, const std::vector<std::int64_t>& rows, const Schools& S, int li,
                 int scale, std::int64_t region, bool alloc_all, std::int64_t seed, std::int64_t rep,
                 std::vector<std::int64_t>& w_i, std::vector<std::int64_t>& c_i) {
    const std::int64_t need = static_cast<std::int64_t>(who.size());
    auto order = keyedSchoolOrder(cand, rows, S, KR64(seed, rep, Stage::STU_SCHOOL_PERM).with(li).with(scale).with(region));
    const std::size_t m = order.size();
    std::vector<std::int64_t> caps(m), cum(m);
    std::int64_t s = 0;
    for (std::size_t i = 0; i < m; ++i) {
        caps[i] = remaining[order[i]];
        cum[i] = (s += caps[i]);
    }
    const std::int64_t total = m ? cum[m - 1] : 0;
    std::vector<std::int64_t> counts;
    if (need <= total) {
        const auto first = std::lower_bound(cum.begin(), cum.end(), need) - cum.begin();
        const std::size_t n_used = std::min<std::size_t>(static_cast<std::size_t>(first) + 1, m);
        counts.assign(caps.begin(), caps.begin() + static_cast<std::ptrdiff_t>(n_used));
        counts.back() -= cum[n_used - 1] - need;
        order.resize(n_used);
    } else {
        counts = caps;
        if (alloc_all) {
            const std::int64_t shortfall = need - total;
            std::int64_t csum = 0;
            for (auto c : caps) {
                csum += c;
            }
            // Spill weights in school-identity order, not the fill order.
            const auto ident = stableOrder(m, [&] (std::int64_t x, std::int64_t y) {
                const auto rx = rows[order[x]], ry = rows[order[y]];
                return S.geoid[rx] != S.geoid[ry] ? S.geoid[rx] < S.geoid[ry] : S.ord[rx] < S.ord[ry];
            });
            std::vector<std::int64_t> wc(m);
            std::int64_t ws = 0;
            for (std::size_t i = 0; i < m; ++i) {
                wc[i] = (ws += csum > 0 ? caps[ident[i]] : 1);
            }
            const KR64 k = KR64(seed, rep, Stage::STU_SPILL).with(li).with(scale).with(region);
            for (std::int64_t j = 0; j < shortfall; ++j) {
                ++counts[ident[intCdf(wc.data(), static_cast<std::int64_t>(m), k.with(j).u64(0))]];
            }
        }
    }
    w_i.clear();
    c_i.clear();
    std::int64_t pos = 0;
    for (std::size_t i = 0; i < order.size() && pos < need; ++i) {
        for (std::int64_t c = 0; c < counts[i] && pos < need; ++c, ++pos) {
            w_i.push_back(who[pos]);
            c_i.push_back(order[i]);
        }
    }
}

//! students.py, _place: students stud (canonical order) into schools rows, whose remaining places
//! are updated in place; sets school[] and returns which of stud were placed. spill: overfill at
//! the last scale; neighbours: then a county-neighbourhood pass, overfilling if neighbour_overflow.
std::vector<std::uint8_t> placeStudents (const Persons& P, const Schools& S, const std::vector<std::int64_t>& stud,
                                         const std::vector<std::int64_t>& rows, std::vector<std::int64_t>& remaining, int li,
                                         const std::vector<int>& scales, bool spill, bool neighbours, bool neighbour_overflow,
                                         std::int64_t seed, std::int64_t rep, std::vector<std::int64_t>& school) {
    const std::size_t NR = rows.size();
    std::vector<std::uint8_t> placed(stud.size(), 0);
    std::vector<std::int64_t> w_i, c_i;
    auto assign = [&] (std::vector<std::int64_t>& taken) {
        for (std::size_t j = 0; j < w_i.size(); ++j) {
            school[stud[w_i[j]]] = rows[c_i[j]];
            placed[w_i[j]] = 1;
            ++taken[c_i[j]];
        }
    };
    // A school squeezed past capacity has nothing left, not negative places.
    auto take = [&] (const std::vector<std::int64_t>& taken) {
        for (std::size_t r = 0; r < NR; ++r) {
            remaining[r] = std::max<std::int64_t>(remaining[r] - taken[r], 0);
        }
    };
    for (int scale : scales) {
        const bool alloc_all = spill && scale == scales.back();
        std::vector<std::int64_t> todo;
        for (std::size_t j = 0; j < stud.size(); ++j) {
            if (!placed[j]) { todo.push_back(static_cast<std::int64_t>(j)); }
        }
        if (todo.empty()) { break; }
        // Students by region (ascending), each region's in canonical order.
        std::map<std::int64_t, std::vector<std::int64_t>> who_by;
        for (auto j : todo) {
            who_by[prefix(P.bg[stud[j]], scale)].push_back(j);
        }
        std::map<std::int64_t, std::vector<std::int64_t>> cand_by;
        for (std::size_t r = 0; r < NR; ++r) {
            cand_by[prefix(S.geoid[rows[r]], scale)].push_back(static_cast<std::int64_t>(r));
        }
        std::vector<std::int64_t> taken(NR, 0);
        for (const auto& [region, who] : who_by) {
            const auto it = cand_by.find(region);
            if (it == cand_by.end()) { continue; }
            fillRegion(who, it->second, remaining, rows, S, li, scale, region, alloc_all, seed, rep, w_i, c_i);
            assign(taken);
        }
        take(taken);
    }
    if (neighbours) {
        // Each home county plus its neighbours, counties in ascending FIPS.
        std::map<std::int64_t, std::vector<std::int64_t>> who_by;
        for (std::size_t j = 0; j < stud.size(); ++j) {
            if (!placed[j]) { who_by[prefix(P.bg[stud[j]], 5)].push_back(static_cast<std::int64_t>(j)); }
        }
        for (const auto& [cty, who] : who_by) {
            const auto nb = S.neighbourhood(cty);
            std::vector<std::int64_t> cand;
            for (std::size_t r = 0; r < NR; ++r) {
                if (std::binary_search(nb.begin(), nb.end(), prefix(S.geoid[rows[r]], 5))) {
                    cand.push_back(static_cast<std::int64_t>(r));
                }
            }
            if (cand.empty()) { continue; }
            fillRegion(who, cand, remaining, rows, S, li, 0, cty, neighbour_overflow, seed, rep, w_i, c_i);
            std::vector<std::int64_t> taken(NR, 0);
            assign(taken);
            take(taken);
        }
    }
    return placed;
}

} // namespace

std::vector<std::int64_t> allocateStudents (const PopulationBundle& b, Persons& P, std::vector<std::int64_t>& work,
                                            std::int64_t seed, std::int64_t rep,
                                            std::map<std::string, std::pair<std::int64_t, std::int64_t>>* stats) {
    const Schools S(b);
    const std::size_t n = P.size(), NS = S.geoid.size();
    std::vector<std::int64_t> school(n, -1);
    std::vector<std::int64_t> ubg(P.bg);
    ubg.erase(std::unique(ubg.begin(), ubg.end()), ubg.end()); // P is sorted by bg
    std::vector<std::uint8_t> in_pop(NS);
    for (std::size_t i = 0; i < NS; ++i) {
        in_pop[i] = std::binary_search(ubg.begin(), ubg.end(), S.geoid[i]);
    }

    struct Level {
        bool done = false;
        std::vector<std::int64_t> stud, rows, remaining;
        std::vector<std::uint8_t> placed;
    };
    Level lv[6];
    for (int li = 0; li < 6; ++li) {
        const std::string L = LEVELS[li];
        Level& D = lv[li];
        for (std::size_t i = 0; i < n; ++i) {
            if (P.student[i] && P.grade[i] >= LEVEL_LO[li] && P.grade[i] <= LEVEL_HI[li]) {
                D.stud.push_back(static_cast<std::int64_t>(i));
            }
        }
        if (D.stud.empty()) { continue; }
        // A level's places at each school: P/E/M/H their own column, university and childcare the
        // school's students.
        for (std::size_t i = 0; i < NS; ++i) {
            const std::int64_t cap = li < N_PLACE_LEVELS ? S.places[i * N_PLACE_LEVELS + li] : S.students[i];
            if (S.level[i].find(L) != std::string::npos && in_pop[i] && cap > 0) {
                D.rows.push_back(static_cast<std::int64_t>(i));
                D.remaining.push_back(cap);
            }
        }
        // Only childcare overfills at its last scale, and university in its neighbour pass;
        // preschool and K-12 students with no place in reach stay home.
        const bool neighbours = L == "E" || L == "M" || L == "H" || L == "U";
        D.placed = placeStudents(P, S, D.stud, D.rows, D.remaining, li, L == "C" ? CHILDCARE_SCALES : SCALES, L == "C",
                                 neighbours, L == "U", seed, rep, school);
        D.done = true;
    }
    // Preschoolers without a school place take the childcare places left over, and become
    // childcare children there.
    if (lv[0].done && lv[LEVEL_C].done) {
        Level &Pre = lv[0], &C = lv[LEVEL_C];
        std::vector<std::int64_t> todo, who;
        for (std::size_t j = 0; j < Pre.stud.size(); ++j) {
            if (!Pre.placed[j]) {
                todo.push_back(static_cast<std::int64_t>(j));
                who.push_back(Pre.stud[j]);
            }
        }
        if (!todo.empty()) {
            const auto got = placeStudents(P, S, who, C.rows, C.remaining, PRESCHOOL_IN_CHILDCARE, CHILDCARE_SCALES, false, false,
                                           false, seed, rep, school);
            for (std::size_t j = 0; j < todo.size(); ++j) {
                if (got[j]) {
                    P.grade[who[j]] = static_cast<std::int16_t>(LEVEL_LO[LEVEL_C]);
                    Pre.placed[todo[j]] = 1;
                }
            }
        }
    }
    for (int li = 0; li < 6; ++li) {
        const Level& D = lv[li];
        if (!D.done) { continue; }
        std::int64_t unplaced = 0;
        for (std::size_t j = 0; j < D.stud.size(); ++j) {
            if (!D.placed[j]) {
                P.grade[D.stud[j]] = -1;
                ++unplaced;
            }
        }
        if (stats) { (*stats)[LEVELS[li]] = {static_cast<std::int64_t>(D.stud.size()), unplaced}; }
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (school[i] >= 0) { work[i] = S.geoid[school[i]]; }
    }
    return school;
}

// ---------------------------------------------------------------------------------------------
// S5 teachers (teachers.py: allocate, _grades)
// ---------------------------------------------------------------------------------------------

namespace {

struct TeacherType {
    const char* name;
    const char* code;
    std::vector<std::string> levels;
    int glo, ghi;
};

const std::vector<TeacherType>& teacherTypes () {
    static const std::vector<TeacherType> t = {
            {"childcare", "6244", {"C"}, 0, 3},
            {"secondary", "6111", {"E", "EM", "EMH", "M", "MH", "H", "P", "PE", "PEM", "PEMH"}, 4, 17},
            {"university", "611", {"U"}, 18, 19},
    };
    return t;
}

std::pair<int, int> levelRange (const std::string& lv) {
    static const std::map<std::string, std::pair<int, int>> r = {{"C", {3, 3}},   {"P", {4, 4}},    {"E", {5, 10}},
                                                                 {"M", {11, 13}}, {"H", {14, 17}},  {"U", {18, 19}},
                                                                 {"PE", {4, 10}}, {"PEM", {4, 13}}, {"PEMH", {4, 17}},
                                                                 {"EM", {5, 13}}, {"EMH", {5, 17}}, {"MH", {11, 17}}};
    return r.at(lv);
}

} // namespace

void allocateTeachers (const PopulationBundle& b, Persons& P, std::vector<std::int64_t>& work, std::vector<std::int64_t>& school,
                       std::int64_t seed, std::int64_t rep, std::map<std::string, std::pair<std::int64_t, std::int64_t>>* stats) {
    const Schools S(b);
    const auto codes = b.strings("naics.codes");
    const std::size_t n = P.size(), NS = S.geoid.size();
    std::vector<std::int64_t> worker_homes;
    for (std::size_t i = 0; i < n; ++i) {
        if (P.employed[i]) { worker_homes.push_back(P.bg[i]); }
    }
    worker_homes.erase(std::unique(worker_homes.begin(), worker_homes.end()), worker_homes.end());

    const auto& types = teacherTypes();
    for (int ti = 0; ti < static_cast<int>(types.size()); ++ti) {
        const auto& ty = types[ti];
        const auto n_code = static_cast<std::int64_t>(std::find(codes.begin(), codes.end(), ty.code) - codes.begin());
        std::vector<std::int64_t> placed_at(NS, 0);
        for (std::size_t i = 0; i < n; ++i) {
            if (school[i] >= 0 && P.student[i] && P.grade[i] >= ty.glo && P.grade[i] <= ty.ghi) { ++placed_at[school[i]]; }
        }
        std::vector<std::int64_t> rows, need;
        for (std::size_t i = 0; i < NS; ++i) {
            if (std::find(ty.levels.begin(), ty.levels.end(), S.level[i]) == ty.levels.end()) { continue; }
            if (!std::binary_search(worker_homes.begin(), worker_homes.end(), S.geoid[i])) { continue; }
            if (placed_at[i] <= 0) { continue; }
            const std::int64_t nom = S.students[i];
            rows.push_back(static_cast<std::int64_t>(i));
            need.push_back(nom > 0 ? (S.teachers[i] * placed_at[i] + nom - 1) / std::max<std::int64_t>(nom, 1) : 0);
        }
        std::int64_t required = 0;
        for (auto x : need) {
            required += x;
        }
        std::vector<std::int64_t> teach;
        for (std::size_t i = 0; i < n; ++i) {
            if (P.employed[i] && P.naics[i] == n_code && school[i] < 0) { teach.push_back(static_cast<std::int64_t>(i)); }
        }
        std::vector<std::uint8_t> freev(teach.size(), 1);
        const std::size_t NR = rows.size();

        std::vector<int> scales = SCALES;
        scales.push_back(0);
        for (int scale : scales) {
            const int s = scale ? scale : 5;
            std::vector<std::int64_t> t_reg(teach.size()), s_reg(NR);
            for (std::size_t t = 0; t < teach.size(); ++t) {
                t_reg[t] = prefix(work[teach[t]], s);
            }
            for (std::size_t r = 0; r < NR; ++r) {
                s_reg[r] = prefix(S.geoid[rows[r]], s);
            }
            std::vector<std::int64_t> regions;
            for (std::size_t r = 0; r < NR; ++r) {
                if (need[r] > 0) { regions.push_back(s_reg[r]); }
            }
            std::sort(regions.begin(), regions.end());
            regions.erase(std::unique(regions.begin(), regions.end()), regions.end());
            // Teachers and schools of each region, ascending. A region's candidates and pool are
            // these filtered by need and freev when the region comes up, which is what scanning
            // every school and teacher then would find.
            std::map<std::int64_t, std::vector<std::int64_t>> teach_in, rows_in;
            for (std::size_t t = 0; t < teach.size(); ++t) {
                teach_in[t_reg[t]].push_back(static_cast<std::int64_t>(t));
            }
            for (std::size_t r = 0; r < NR; ++r) {
                rows_in[s_reg[r]].push_back(static_cast<std::int64_t>(r));
            }
            for (const auto region : regions) {
                std::vector<std::int64_t> cand, pool;
                for (auto r : rows_in[region]) {
                    if (need[r] > 0) { cand.push_back(r); }
                }
                auto addFree = [&] (std::int64_t reg) {
                    const auto it = teach_in.find(reg);
                    if (it == teach_in.end()) { return; }
                    for (auto t : it->second) {
                        if (freev[t]) { pool.push_back(t); }
                    }
                };
                if (scale) {
                    addFree(region);
                } else {
                    for (auto c : S.neighbourhood(region)) { // sorted, no repeats
                        addFree(c);
                    }
                    std::sort(pool.begin(), pool.end());
                }
                if (pool.empty() || cand.empty()) { continue; }
                std::int64_t nsum = 0;
                for (auto c : cand) {
                    nsum += need[c];
                }
                const std::int64_t num = std::min<std::int64_t>(nsum, static_cast<std::int64_t>(pool.size()));
                const auto order = keyedSchoolOrder(cand, rows, S,
                                                    KR64(seed, rep, Stage::TCH_SCHOOL_PERM).with(ti).with(scale).with(region));
                std::vector<std::int64_t> cum(order.size());
                std::int64_t cs = 0;
                for (std::size_t i = 0; i < order.size(); ++i) {
                    cum[i] = (cs += need[order[i]]);
                }
                const auto first = std::lower_bound(cum.begin(), cum.end(), num) - cum.begin();
                const std::size_t n_used = std::min<std::size_t>(static_cast<std::size_t>(first) + 1, order.size());
                std::vector<std::int64_t> slots;
                for (std::size_t i = 0; i < n_used; ++i) {
                    std::int64_t c = need[order[i]];
                    if (i + 1 == n_used) { c -= cum[n_used - 1] - num; }
                    for (std::int64_t k = 0; k < c; ++k) {
                        slots.push_back(order[i]);
                    }
                }
                // Teachers: the first num of a keyed shuffle of the region's free pool.
                const KR64 tk = KR64(seed, rep, Stage::TCH_PICK).with(ti).with(scale).with(region);
                std::vector<std::uint64_t> dr(pool.size());
                for (std::size_t q = 0; q < pool.size(); ++q) {
                    const auto i = teach[pool[q]];
                    dr[q] = tk.with(P.bg[i]).with(P.h[i]).with(P.p[i]).u64(0);
                }
                const auto o = stableOrder(pool.size(), [&] (std::int64_t x, std::int64_t y) {
                    if (dr[x] != dr[y]) { return dr[x] < dr[y]; }
                    const auto i = teach[pool[x]], j = teach[pool[y]];
                    if (P.bg[i] != P.bg[j]) { return P.bg[i] < P.bg[j]; }
                    return P.h[i] != P.h[j] ? P.h[i] < P.h[j] : P.p[i] < P.p[j];
                });
                const KR64 kg(seed, rep, Stage::TCH_GRADE);
                for (std::int64_t q = 0; q < num; ++q) {
                    const auto pick = pool[o[q]];
                    const auto who = teach[pick];
                    const auto r = rows[slots[q]];
                    school[who] = r;
                    work[who] = S.geoid[r];
                    const auto [lo, hi] = levelRange(S.level[r]);
                    const KR64 k = kg.with(P.bg[who]).with(P.h[who]).with(P.p[who]);
                    std::int64_t g = lo + index(k.u64(0), hi - lo + 1);
                    if (g == 19 && k.u01(1) < 0.5) { g = 18; }
                    P.grade[who] = static_cast<std::int16_t>(g);
                    freev[pick] = 0;
                    --need[slots[q]];
                }
            }
        }
        if (stats) {
            std::int64_t got = 0;
            for (auto f : freev) {
                got += f ? 0 : 1;
            }
            (*stats)[ty.name] = {required, got};
        }
    }

    // Each preschool/K-12 school's teachers dealt out over its grades by enrollment
    // (teachers._share_grades): T teachers, n_g of N students in grade g get floor(T n_g / N), the
    // rest one each to the largest remainders, lower grade first on ties; grades ascending go to
    // the school's teachers in keyed order.
    const int glo = levelRange("P").first, ghi = levelRange("H").second, NG = ghi - glo + 1;
    const auto k12 = [&] (std::size_t i) {
        return school[i] >= 0 && P.grade[i] >= glo && P.grade[i] <= ghi;
    };
    std::vector<std::int64_t> cnt(NS * NG, 0), T(NS, 0), tch;
    for (std::size_t i = 0; i < n; ++i) {
        if (!k12(i)) { continue; }
        if (P.student[i]) { ++cnt[school[i] * NG + (P.grade[i] - glo)]; }
        if (P.employed[i]) {
            ++T[school[i]];
            tch.push_back(static_cast<std::int64_t>(i));
        }
    }
    {
        const KR64 kd(seed, rep, Stage::TCH_GRADE_SHARE);
        std::vector<std::uint64_t> dr(tch.size());
        for (std::size_t t = 0; t < tch.size(); ++t) {
            const auto i = tch[t];
            dr[t] = kd.with(P.bg[i]).with(P.h[i]).with(P.p[i]).u64(0);
        }
        const auto o = stableOrder(tch.size(), [&] (std::int64_t x, std::int64_t y) {
            const auto i = tch[x], j = tch[y];
            if (school[i] != school[j]) { return school[i] < school[j]; }
            if (dr[x] != dr[y]) { return dr[x] < dr[y]; }
            if (P.bg[i] != P.bg[j]) { return P.bg[i] < P.bg[j]; }
            return P.h[i] != P.h[j] ? P.h[i] < P.h[j] : P.p[i] < P.p[j];
        });
        std::vector<std::int64_t> sorted(tch.size());
        for (std::size_t t = 0; t < tch.size(); ++t) {
            sorted[t] = tch[o[t]];
        }
        tch.swap(sorted);
    }
    std::int64_t dealt = 0;
    std::vector<std::int64_t> share(NG), rem(NG);
    for (std::size_t lo = 0, hi; lo < tch.size(); lo = hi) {
        const std::int64_t s = school[tch[lo]];
        for (hi = lo; hi < tch.size() && school[tch[hi]] == s; ++hi) {}
        const std::int64_t* c = &cnt[s * NG];
        std::int64_t N = 0, have = 0;
        for (int g = 0; g < NG; ++g) {
            N += c[g];
        }
        if (N == 0) { continue; }
        for (int g = 0; g < NG; ++g) {
            share[g] = T[s] * c[g] / N;
            rem[g] = T[s] * c[g] % N;
            have += share[g];
        }
        const auto o = stableOrder(NG, [&] (std::int64_t x, std::int64_t y) {
            return rem[x] > rem[y];
        });
        for (std::int64_t k = 0; k < T[s] - have; ++k) {
            ++share[o[k]];
        }
        std::size_t q = lo;
        for (int g = 0; g < NG; ++g) {
            for (std::int64_t k = 0; k < share[g]; ++k) {
                P.grade[tch[q++]] = static_cast<std::int16_t>(glo + g);
            }
        }
        dealt += static_cast<std::int64_t>(hi - lo);
    }
    if (stats) { (*stats)["by grade"] = {static_cast<std::int64_t>(tch.size()), dealt}; }
}

// ---------------------------------------------------------------------------------------------
// S6-S10 groups (groups.py)
// ---------------------------------------------------------------------------------------------

namespace {

constexpr std::int64_t NBORHOOD_SIZE = 500;
constexpr std::int64_t WORKGROUP_SIZE = 20;
constexpr std::int64_t CLASS_SIZE = 20, CLASS_MIN = 5, CLASS_MAX = 50;
constexpr std::int64_t COLLEGE_CLASS_SIZE = 30;

//! ceil(a / b) for a >= 0, b > 0, as Python's -(-a // b).
std::int64_t ceilDiv (std::int64_t a, std::int64_t b) {
    return (a + b - 1) / b;
}

//! Split total among w.size() >= 1 groups in proportion to integer weights, each >= 1 (_apportion).
std::vector<std::int64_t> apportion (const std::vector<std::int64_t>& w, std::int64_t total) {
    const std::int64_t k = static_cast<std::int64_t>(w.size());
    const std::int64_t spare = total - k;
    std::int64_t W = 0;
    for (auto x : w) {
        W += x;
    }
    std::vector<std::int64_t> base(k), rem(k);
    std::int64_t have = 0;
    for (std::int64_t i = 0; i < k; ++i) {
        const std::int64_t q = w[i] * spare;
        base[i] = q / W + 1;
        rem[i] = q % W;
        have += base[i];
    }
    const std::int64_t left = total - have;
    if (left > 0) {
        const auto o = stableOrder(k, [&] (std::int64_t x, std::int64_t y) {
            return rem[x] != rem[y] ? rem[x] > rem[y] : x < y;
        });
        for (std::int64_t i = 0; i < left; ++i) {
            ++base[o[i]];
        }
    }
    return base;
}

} // namespace

Groups assignGroups (const PopulationBundle& b, const Persons& P, const std::vector<std::int64_t>& work,
                     const std::vector<std::int64_t>& school, const SizeTables& tables, std::int64_t seed, std::int64_t rep,
                     std::map<std::string, std::string>* digests) {
    const std::int64_t n = static_cast<std::int64_t>(P.size());
    const auto sg = b.get<std::int64_t>("schools.geoid");
    const auto so = b.get<std::int16_t>("schools.ord");
    Groups R;

    // S6: school id = 1 + rank of (school geoid, ord) among the schools in use at that geoid.
    R.school_id.assign(n, 0);
    {
        std::vector<std::pair<std::int64_t, std::int64_t>> used;
        for (std::int64_t i = 0; i < n; ++i) {
            if (school[i] >= 0 && P.grade[i] != -1) { used.emplace_back(sg[school[i]], so[school[i]]); }
        }
        totalSort(used, std::less<std::pair<std::int64_t, std::int64_t>>());
        used.erase(std::unique(used.begin(), used.end()), used.end());
        std::vector<std::int64_t> local(used.size());
        for (std::size_t j = 0; j < used.size(); ++j) {
            const bool first = j == 0 || used[j].first != used[j - 1].first;
            local[j] = first ? 1 : local[j - 1] + 1;
        }
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads())
#endif
        for (std::int64_t i = 0; i < n; ++i) {
            if (school[i] >= 0 && P.grade[i] != -1) {
                const std::pair<std::int64_t, std::int64_t> key(sg[school[i]], so[school[i]]);
                R.school_id[i] = local[std::lower_bound(used.begin(), used.end(), key) - used.begin()];
            }
        }
    }
    if (digests) { (*digests)["S6 school ids"] = Digest().add(R.school_id).hex16(); }

    // S7: neighbourhood per household (keyed), household cluster = h mod ceil(households / 4).
    R.nborhood.assign(n, 0);
    R.hh_cluster.assign(n, 0);
    {
        const KR64 k(seed, rep, Stage::HOME_NB);
        const auto bgs = runStarts(P.bg); // persons are in block-group order
        const auto n_bgs = static_cast<std::int64_t>(bgs.size()) - 1;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
#endif
        for (std::int64_t g = 0; g < n_bgs; ++g) {
            const std::int64_t a = bgs[g], z = bgs[g + 1];
            std::int64_t n_hh = 0;
            for (std::int64_t i = a; i < z; ++i) {
                n_hh = std::max(n_hh, P.h[i] + 1);
            }
            const std::int64_t pop = z - a;
            const std::int64_t max_nb = std::max<std::int64_t>(1, (2 * pop + NBORHOOD_SIZE) / (2 * NBORHOOD_SIZE));
            const std::int64_t clusters = std::max<std::int64_t>(1, (n_hh + 3) / 4);
            for (std::int64_t i = a; i < z; ++i) {
                R.nborhood[i] = index(k.with(P.bg[i]).with(P.h[i]).u64(0), max_nb);
                R.hh_cluster[i] = P.h[i] % clusters;
            }
        }
    }
    if (digests) { (*digests)["S7 home groups"] = Digest().add(R.nborhood).add(R.hh_cluster).hex16(); }

    // S8: work groups per (work block group, NAICS) for those who go to a workplace.
    R.workgroup.assign(n, 0);
    R.work_group.assign(n, -1);
    {
        std::vector<std::int64_t> el;
        for (std::int64_t i = 0; i < n; ++i) {
            if (P.naics[i] != -1 && R.school_id[i] == 0 && P.travel[i] != TRAVEL_WFH) { el.push_back(i); }
        }
        const KR64 ko(seed, rep, Stage::WG_ORDER);
        std::vector<std::uint64_t> ok(el.size());
        const auto n_el = static_cast<std::int64_t>(el.size());
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads())
#endif
        for (std::int64_t j = 0; j < n_el; ++j) {
            const auto i = el[j];
            ok[j] = ko.with(work[i]).with(P.naics[i]).with(P.bg[i]).with(P.h[i]).with(P.p[i]).u64(0);
        }
        const auto o = stableOrder(el.size(), [&] (std::int64_t x, std::int64_t y) {
            const auto i = el[x], j = el[y];
            if (work[i] != work[j]) { return work[i] < work[j]; }
            if (P.naics[i] != P.naics[j]) { return P.naics[i] < P.naics[j]; }
            if (ok[x] != ok[y]) { return ok[x] < ok[y]; }
            if (P.bg[i] != P.bg[j]) { return P.bg[i] < P.bg[j]; }
            return P.h[i] != P.h[j] ? P.h[i] < P.h[j] : P.p[i] < P.p[j];
        });
        std::vector<std::int64_t> s(el.size());
        for (std::size_t j = 0; j < el.size(); ++j) {
            s[j] = el[o[j]];
        }
        // groups: runs of equal (work geoid, NAICS) in s; each independent, so they run in parallel
        std::vector<std::int64_t> starts;
        for (std::size_t j = 0; j < s.size(); ++j) {
            if (j == 0 || work[s[j]] != work[s[j - 1]] || P.naics[s[j]] != P.naics[s[j - 1]]) {
                starts.push_back(static_cast<std::int64_t>(j));
            }
        }
        const auto n_groups = static_cast<std::int64_t>(starts.size());
        starts.push_back(static_cast<std::int64_t>(s.size()));
        std::vector<std::int64_t> team_count(n_groups);
        const KR64 ke(seed, rep, Stage::WG_EST_SIZE);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads())
#endif
        for (std::int64_t gi = 0; gi < n_groups; ++gi) {
            const auto lo = static_cast<std::size_t>(starts[gi]), hi = static_cast<std::size_t>(starts[gi + 1]);
            const std::int64_t pop = static_cast<std::int64_t>(hi - lo);
            const std::int64_t geo = work[s[lo]], nai = P.naics[s[lo]], state = geo / 10000000000LL;
            const std::int64_t target = std::max<std::int64_t>(1, tables.target(state, nai));
            std::vector<std::int64_t> sizes;
            if (tables.hasEst(state, nai)) {
                const double m = tables.mean(state, nai);
                const std::int64_t k = std::min(std::max<std::int64_t>(1, static_cast<std::int64_t>(std::floor(
                                                                                  static_cast<double>(pop) / m + 0.5))),
                                                pop);
                const KR64 kg = ke.with(geo).with(nai);
                for (std::int64_t j = 0; j < k; ++j) {
                    sizes.push_back(tables.sample(state, nai, kg.with(j)));
                }
            } else {
                const std::int64_t k =
                        std::min(std::max<std::int64_t>(1,
                                                        static_cast<std::int64_t>(std::floor(
                                                                static_cast<double>(pop) / static_cast<double>(target) + 0.5))),
                                 pop);
                sizes.assign(k, target);
            }
            const auto est = apportion(sizes, pop);
            const std::int64_t k = static_cast<std::int64_t>(est.size());
            std::vector<std::int64_t> n_teams(k), team_base(k);
            std::int64_t tb = 0;
            for (std::int64_t e = 0; e < k; ++e) {
                n_teams[e] = std::max<std::int64_t>(1, (2 * est[e] + target) / (2 * target));
                team_base[e] = tb;
                tb += n_teams[e];
            }
            std::int64_t m = 0;
            for (std::int64_t e = 0; e < k; ++e) {
                for (std::int64_t pos = 0; pos < est[e]; ++pos, ++m) {
                    R.workgroup[s[lo + m]] = team_base[e] + pos % n_teams[e] + 1;
                }
            }
            team_count[gi] = tb;
        }
        // dense ids: exclusive scan of team counts over groups in (geoid, NAICS) order
        std::vector<std::int64_t> base(n_groups + 1, 0);
        for (std::int64_t gi = 0; gi < n_groups; ++gi) {
            base[gi + 1] = base[gi] + team_count[gi];
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads())
#endif
        for (std::int64_t gi = 0; gi < n_groups; ++gi) {
            for (std::int64_t j = starts[gi]; j < starts[gi + 1]; ++j) {
                R.work_group[s[j]] = base[gi] + R.workgroup[s[j]] - 1;
            }
        }
    }
    if (digests) { (*digests)["S8 work groups"] = Digest().add(R.workgroup).add(R.work_group).hex16(); }

    // S9: classes per (work block group, school, grade), and admin groups for spare teachers.
    R.school_class.assign(n, 0);
    R.school_class_group.assign(n, -1);
    {
        std::vector<std::int64_t> en;
        for (std::int64_t i = 0; i < n; ++i) {
            if (R.school_id[i] > 0) { en.push_back(i); }
        }
        totalSort(en, [&] (std::int64_t i, std::int64_t j) {
            if (work[i] != work[j]) { return work[i] < work[j]; }
            if (R.school_id[i] != R.school_id[j]) { return R.school_id[i] < R.school_id[j]; }
            if (P.grade[i] != P.grade[j]) { return P.grade[i] < P.grade[j]; }
            return i < j;
        });
        // groups: runs of equal (work geoid, school, grade) in en; independent, so in parallel
        std::vector<std::int64_t> starts;
        for (std::size_t j = 0; j < en.size(); ++j) {
            if (j == 0 || work[en[j]] != work[en[j - 1]] || R.school_id[en[j]] != R.school_id[en[j - 1]] ||
                P.grade[en[j]] != P.grade[en[j - 1]]) {
                starts.push_back(static_cast<std::int64_t>(j));
            }
        }
        const auto n_groups = static_cast<std::int64_t>(starts.size());
        starts.push_back(static_cast<std::int64_t>(en.size()));
        const KR64 ks(seed, rep, Stage::CLASS_SMEAR);
        std::vector<std::int64_t> local(en.size(), 0), n_grp(n_groups); // local: by position in en
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads())
#endif
        for (std::int64_t gi = 0; gi < n_groups; ++gi) {
            const auto lo = static_cast<std::size_t>(starts[gi]), hi = static_cast<std::size_t>(starts[gi + 1]);
            std::int64_t n_st = 0;
            for (std::size_t j = lo; j < hi; ++j) {
                n_st += P.naics[en[j]] == -1 ? 1 : 0;
            }
            const std::int64_t n_te = static_cast<std::int64_t>(hi - lo) - n_st;
            const std::int64_t m0 = en[lo];
            std::int64_t n_classes = 0;
            if (n_st > 0) {
                // college classes by size, since a college's staff is its total employment
                const std::int64_t raw = P.grade[m0] > 17 ? std::max<std::int64_t>(1, ceilDiv(n_st, COLLEGE_CLASS_SIZE))
                                         : n_te > 0       ? n_te
                                                          : std::max<std::int64_t>(1, ceilDiv(n_st, CLASS_SIZE));
                n_classes = std::max(ceilDiv(n_st, CLASS_MAX), std::min(raw, std::max<std::int64_t>(1, n_st / CLASS_MIN)));
            }
            const std::int64_t excess = n_te - n_classes;
            const std::int64_t n_admin = excess > 0 ? ceilDiv(excess, WORKGROUP_SIZE) : 0;
            std::int64_t t_rank = 0, s_rank = 0;
            const KR64 kc = ks.with(work[m0]).with(sg[school[m0]]).with(so[school[m0]]).with(P.grade[m0]);
            for (std::size_t j = lo; j < hi; ++j) {
                const std::int64_t i = en[j];
                std::int64_t c;
                if (P.naics[i] != -1) { // teacher
                    const std::int64_t t = t_rank++;
                    c = t < n_classes ? t : (n_admin > 0 ? -2 - (t - n_classes) % n_admin : 0);
                } else {
                    const std::int64_t r = s_rank++;
                    c = r < CLASS_MIN * n_classes ? r % n_classes : index(kc.with(r).u64(0), n_classes);
                }
                R.school_class[i] = c;
                local[j] = c >= 0 ? c : n_classes + (-2 - c);
            }
            n_grp[gi] = n_classes + n_admin;
        }
        // dense ids: exclusive scan of class and admin group counts in (geoid, school, grade) order
        std::vector<std::int64_t> base(n_groups + 1, 0);
        for (std::int64_t gi = 0; gi < n_groups; ++gi) {
            base[gi + 1] = base[gi] + n_grp[gi];
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads())
#endif
        for (std::int64_t gi = 0; gi < n_groups; ++gi) {
            for (std::int64_t j = starts[gi]; j < starts[gi + 1]; ++j) {
                R.school_class_group[en[j]] = base[gi] + local[j];
            }
        }
    }
    if (digests) { (*digests)["S9 school groups"] = Digest().add(R.school_class).add(R.school_class_group).hex16(); }

    // S10: day neighbourhoods -- atoms packed per daytime block group on an integer midpoint grid.
    R.work_nborhood.assign(n, 0);
    {
        std::vector<std::int64_t> day(n), kind(n), a(n, 0), bb(n, 0), c(n, 0);
        // (day geoid * 100000 + school id) -> members; only ever looked up, so a hash map
        std::unordered_map<std::int64_t, std::int64_t> school_size;
        for (std::int64_t i = 0; i < n; ++i) {
            const bool at_school = R.school_id[i] != 0, at_work = R.workgroup[i] > 0;
            day[i] = (!at_school && !at_work) ? P.bg[i] : work[i];
            if (at_school) { ++school_size[day[i] * 100000 + R.school_id[i]]; }
        }
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads())
#endif
        for (std::int64_t i = 0; i < n; ++i) {
            const bool at_school = R.school_id[i] != 0, at_work = R.workgroup[i] > 0;
            if (at_work) {
                kind[i] = 0;
                a[i] = P.naics[i];
                bb[i] = R.workgroup[i];
            } else if (at_school) {
                if (school_size.at(day[i] * 100000 + R.school_id[i]) > NBORHOOD_SIZE) { // a school too big: by class
                    kind[i] = 2;
                    a[i] = so[school[i]];
                    bb[i] = P.grade[i];
                    c[i] = R.school_class[i];
                } else {
                    kind[i] = 1;
                    a[i] = sg[school[i]];
                    bb[i] = so[school[i]];
                }
            } else {
                kind[i] = 3;
                a[i] = P.h[i];
            }
        }
        // atoms = distinct (day, kind, a, b, c), sorted, and each person's atom. Persons in the
        // order of (atom, index): the day geoid leads, so they are bucketed by day first (a stable
        // counting sort), then each day's few thousand sorted by the rest on their own, in cache
        // and in parallel -- the same order one sort of everyone would give.
        using Atom = std::array<std::int64_t, 5>;
        std::unordered_map<std::int64_t, std::int64_t> day_ix;
        std::vector<std::int64_t> days;
        for (std::int64_t i = 0; i < n; ++i) {
            if (day_ix.emplace(day[i], 0).second) { days.push_back(day[i]); }
        }
        std::sort(days.begin(), days.end());
        const auto n_day = static_cast<std::int64_t>(days.size());
        for (std::int64_t d = 0; d < n_day; ++d) {
            day_ix[days[d]] = d;
        }
        std::vector<std::int64_t> dstart(n_day + 1, 0), by_atom(n);
        {
            std::vector<std::int32_t> dr(n);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads())
#endif
            for (std::int64_t i = 0; i < n; ++i) {
                dr[i] = static_cast<std::int32_t>(day_ix.at(day[i]));
            }
            for (std::int64_t i = 0; i < n; ++i) {
                ++dstart[dr[i] + 1];
            }
            for (std::int64_t d = 0; d < n_day; ++d) {
                dstart[d + 1] += dstart[d];
            }
            std::vector<std::int64_t> fill(dstart.begin(), dstart.end() - 1);
            for (std::int64_t i = 0; i < n; ++i) {
                by_atom[fill[dr[i]]++] = i;
            }
        }
        struct Member {
            std::int64_t kind, a, b, c, i;
            bool operator<(const Member& o) const { return std::tie(kind, a, b, c, i) < std::tie(o.kind, o.a, o.b, o.c, o.i); }
        };
        std::vector<std::int64_t> day_atoms(n_day, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads())
#endif
        for (std::int64_t d = 0; d < n_day; ++d) {
            std::vector<Member> m;
            m.reserve(static_cast<std::size_t>(dstart[d + 1] - dstart[d]));
            for (std::int64_t j = dstart[d]; j < dstart[d + 1]; ++j) {
                const auto i = by_atom[j];
                m.push_back({kind[i], a[i], bb[i], c[i], i});
            }
            std::sort(m.begin(), m.end());
            std::int64_t na = 0;
            for (std::size_t k = 0; k < m.size(); ++k) {
                by_atom[dstart[d] + static_cast<std::int64_t>(k)] = m[k].i;
                const auto& x = m[k];
                if (k == 0 || std::tie(x.kind, x.a, x.b, x.c) != std::tie(m[k - 1].kind, m[k - 1].a, m[k - 1].b, m[k - 1].c)) {
                    ++na;
                }
            }
            day_atoms[d] = na;
        }
        std::vector<std::int64_t> atom_base(n_day + 1, 0);
        for (std::int64_t d = 0; d < n_day; ++d) {
            atom_base[d + 1] = atom_base[d] + day_atoms[d];
        }
        std::vector<Atom> atoms(static_cast<std::size_t>(atom_base[n_day]));
        std::vector<std::int64_t> asize(atoms.size(), 0), atom_of(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads())
#endif
        for (std::int64_t d = 0; d < n_day; ++d) {
            std::int64_t id = atom_base[d] - 1;
            for (std::int64_t j = dstart[d]; j < dstart[d + 1]; ++j) {
                const auto i = by_atom[j];
                const Atom t{day[i], kind[i], a[i], bb[i], c[i]};
                if (j == dstart[d] || atoms[static_cast<std::size_t>(id)] != t) { atoms[static_cast<std::size_t>(++id)] = t; }
                ++asize[static_cast<std::size_t>(id)];
                atom_of[i] = id;
            }
        }
        by_atom = {};
        const std::size_t NA = atoms.size();
        const KR64 kd(seed, rep, Stage::DAY_NB);
        std::vector<std::uint64_t> dk(NA);
        std::vector<std::uint8_t> over(NA);
        const auto na64 = static_cast<std::int64_t>(NA);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads())
#endif
        for (std::int64_t j = 0; j < na64; ++j) {
            const auto& t = atoms[j];
            dk[j] = kd.with(t[0]).with(t[1]).with(t[2]).with(t[3]).with(t[4]).u64(0);
            over[j] = asize[j] > NBORHOOD_SIZE;
        }
        const auto order = stableOrder(NA, [&] (std::int64_t x, std::int64_t y) {
            const auto &X = atoms[x], &Y = atoms[y];
            if (X[0] != Y[0]) { return X[0] < Y[0]; }
            if (over[x] != over[y]) { return over[x] > over[y]; } // oversized first
            if (dk[x] != dk[y]) { return dk[x] < dk[y]; }
            return std::lexicographical_compare(X.begin() + 1, X.end(), Y.begin() + 1, Y.end());
        });
        std::vector<std::int64_t> atom_bin(NA);
        std::vector<std::int64_t> day_starts; // runs of one day geoid in order; independent
        for (std::size_t j = 0; j < NA; ++j) {
            if (j == 0 || atoms[order[j]][0] != atoms[order[j - 1]][0]) { day_starts.push_back(static_cast<std::int64_t>(j)); }
        }
        const auto n_days = static_cast<std::int64_t>(day_starts.size());
        day_starts.push_back(na64);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads())
#endif
        for (std::int64_t gi = 0; gi < n_days; ++gi) {
            const auto lo = static_cast<std::size_t>(day_starts[gi]), hi = static_cast<std::size_t>(day_starts[gi + 1]);
            std::int64_t n_over = 0, total = 0;
            for (std::size_t j = lo; j < hi; ++j) {
                if (over[order[j]]) {
                    ++n_over;
                } else {
                    total += asize[order[j]];
                }
            }
            const std::int64_t n_bins = std::max<std::int64_t>(1, (2 * total + NBORHOOD_SIZE) / (2 * NBORHOOD_SIZE));
            const std::int64_t T = std::max<std::int64_t>(total, 1);
            std::int64_t ex = 0, dense = -1, prev = -1;
            for (std::size_t j = lo; j < hi; ++j) {
                const auto at = order[j];
                const std::int64_t packed = over[at] ? 0 : asize[at];
                const std::int64_t grid = std::min(((2 * ex + packed) * n_bins) / (2 * T), n_bins - 1);
                const std::int64_t bin = over[at] ? static_cast<std::int64_t>(j - lo) : n_over + grid;
                if (j == lo || bin != prev) { ++dense; }
                prev = bin;
                atom_bin[at] = dense;
                ex += packed;
            }
        }
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads())
#endif
        for (std::int64_t i = 0; i < n; ++i) {
            R.work_nborhood[i] = atom_bin[atom_of[i]];
        }
    }
    if (digests) { (*digests)["S10 day neighbourhoods"] = Digest().add(R.work_nborhood).hex16(); }
    return R;
}

std::string placementDigest (const Placements& pl) {
    std::vector<std::size_t> o(pl.size());
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&] (std::size_t x, std::size_t y) {
        return pl.bg[x] != pl.bg[y] ? pl.bg[x] < pl.bg[y] : pl.donor[x] < pl.donor[y];
    });
    std::vector<std::int64_t> bg(o.size()), dn(o.size()), ct(o.size());
    for (std::size_t i = 0; i < o.size(); ++i) {
        bg[i] = pl.bg[o[i]];
        dn[i] = pl.donor[o[i]];
        ct[i] = pl.count[o[i]];
    }
    return Digest().add(bg).add(dn).add(ct).hex16();
}

std::string personsDigest (const Persons& P) {
    return Digest()
            .add(P.bg)
            .add(P.h)
            .add(P.p)
            .add(P.age)
            .add(P.sex)
            .add(P.race)
            .add(P.naics)
            .add(P.travel)
            .add(P.veh_occ)
            .add(P.grade)
            .add(P.employed)
            .add(P.student)
            .hex16();
}

} // namespace PopGen
