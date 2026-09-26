/*! @file PopulationGenerator.cpp
    \brief Placement and population stages; see PopulationGenerator.H. Each section names the
    Python function it ports.
*/
#include "PopulationGenerator.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>

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
constexpr std::int16_t GRADE_SHIFT = 3;
constexpr double CHILDCARE_PROB[5] = {0.32, 0.47, 0.47, 0.83, 0.83};
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
            P.veh_occ.push_back(veh[s]);
            P.grade.push_back(static_cast<std::int16_t>(grade[s] >= 0 ? grade[s] + GRADE_SHIFT : -1));
        }
    }
    const std::size_t n = P.size();

    // S1 childcare: under-5s not in school, one keyed Bernoulli each.
    const KR64 kc(seed, rep, Stage::CHILDCARE);
    for (std::size_t i = 0; i < n; ++i) {
        const int a = P.age[i];
        if (P.grade[i] == -1 && a >= 0 && a < 5) {
            if (kc.with(P.bg[i]).with(P.h[i]).with(P.p[i]).u01(0) < CHILDCARE_PROB[a]) { P.grade[i] = CHILDCARE_GRADE; }
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
// S3 workers (workers.py: allocate, _fill_one)
// ---------------------------------------------------------------------------------------------

namespace {

constexpr int IPF_ITERS = 60;
constexpr double IPF_TOL = 1e-9;
constexpr int REPAIR_SWEEPS = 12;

//! Home x destination LODES flows between worker home block groups, as CSR in both directions.
struct Flows {
    std::vector<std::int64_t> homes, dests;           // sorted geoids
    std::vector<std::int64_t> hd_ptr, hd_col, hd_val; // home rows, destination indices ascending
    std::vector<std::int64_t> dh_ptr, dh_col, dh_val; // destination rows, home indices ascending
};

//! Index of x in a sorted vector known to contain it.
std::int64_t position (const std::vector<std::int64_t>& v, std::int64_t x) {
    return std::lower_bound(v.begin(), v.end(), x) - v.begin();
}

//! Stable index permutation sorting by a key comparator, as numpy.lexsort (stable).
template <class Less>
std::vector<std::int64_t> stableOrder (std::size_t n, Less less) {
    std::vector<std::int64_t> o(n);
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), less);
    return o;
}

struct Fill {
    std::vector<std::int64_t> row_h, col_d, cnt; // home index, destination index, workers
    std::int64_t unplaced = 0;
};

//! IPF + column TRS + row repair for industry n.
Fill fillOne (const Flows& F, const std::vector<std::int64_t>& supply, const std::vector<std::int64_t>& demand, int n,
              std::int64_t seed, std::int64_t rep, WorkerStats& st) {
    Fill out;
    const std::int64_t H = static_cast<std::int64_t>(F.homes.size()), Dn = static_cast<std::int64_t>(F.dests.size());
    std::vector<std::int64_t> col_pos(Dn, -1); // demanded destination -> position in ci
    std::vector<std::int64_t> ci;
    for (std::int64_t d = 0; d < Dn; ++d) {
        if (demand[d] > 0) {
            col_pos[d] = static_cast<std::int64_t>(ci.size());
            ci.push_back(d);
        }
    }
    // Rows with supply and at least one demanded destination; columns reached by some such row.
    std::vector<std::int64_t> ri;
    std::vector<std::uint8_t> col_used(ci.size(), 0);
    for (std::int64_t h = 0; h < H; ++h) {
        if (supply[h] <= 0) { continue; }
        bool any = false;
        for (std::int64_t q = F.hd_ptr[h]; q < F.hd_ptr[h + 1]; ++q) {
            const auto c = col_pos[F.hd_col[q]];
            if (c >= 0) {
                any = true;
                col_used[c] = 1;
            }
        }
        if (any) {
            ri.push_back(h);
        } else {
            out.unplaced += supply[h];
        }
    }
    std::vector<std::int64_t> col_final(ci.size(), -1), ci2;
    for (std::size_t c = 0; c < ci.size(); ++c) {
        if (col_used[c]) {
            col_final[c] = static_cast<std::int64_t>(ci2.size());
            ci2.push_back(ci[c]);
        }
    }
    // Entries in CSR order: rows in ri order, columns ascending.
    std::vector<std::int64_t> rows, cols;
    std::vector<double> v;
    for (std::size_t r = 0; r < ri.size(); ++r) {
        const auto h = ri[r];
        for (std::int64_t q = F.hd_ptr[h]; q < F.hd_ptr[h + 1]; ++q) {
            const auto c = col_pos[F.hd_col[q]];
            if (c < 0) { continue; }
            rows.push_back(static_cast<std::int64_t>(r));
            cols.push_back(col_final[c]);
            v.push_back(static_cast<double>(F.hd_val[q]));
        }
    }
    const std::size_t NR = ri.size(), NC = ci2.size(), NZ = v.size();
    if (NZ == 0) {
        for (auto h : ri) {
            out.unplaced += supply[h];
        }
        return out;
    }
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

    const double tol = IPF_TOL * std::max(1.0, seqSum(rt.data(), NR));
    std::vector<double> rs(NR), cs(NC);
    auto rowSums = [&] () {
        std::fill(rs.begin(), rs.end(), 0.0);
        for (std::size_t e = 0; e < NZ; ++e) {
            rs[rows[e]] += v[e];
        }
    };
    for (int it = 0; it < IPF_ITERS; ++it) {
        rowSums();
        for (std::size_t e = 0; e < NZ; ++e) {
            v[e] = v[e] * (rs[rows[e]] > 0 ? rt[rows[e]] / rs[rows[e]] : 0.0);
        }
        std::fill(cs.begin(), cs.end(), 0.0);
        for (std::size_t e = 0; e < NZ; ++e) {
            cs[cols[e]] += v[e];
        }
        for (std::size_t e = 0; e < NZ; ++e) {
            v[e] = v[e] * (cs[cols[e]] > 0 ? ct[cols[e]] / cs[cols[e]] : 0.0);
        }
        rowSums();
        double err = 0.0;
        for (std::size_t r = 0; r < NR; ++r) {
            err += std::abs(rs[r] - rt[r]);
        }
        if (err < tol) { break; }
    }

    // Column-wise TRS, each column's entries in row (home) order.
    std::vector<std::int64_t> cnt(NZ);
    for (std::size_t e = 0; e < NZ; ++e) {
        cnt[e] = static_cast<std::int64_t>(std::floor(v[e]));
    }
    const auto oc = stableOrder(NZ, [&] (std::int64_t x, std::int64_t y) {
        return cols[x] != cols[y] ? cols[x] < cols[y] : rows[x] < rows[y];
    });
    std::vector<double> frac, cum;
    std::vector<std::int64_t> idx;
    for (std::size_t a = 0; a < NZ;) {
        std::size_t z = a;
        while (z < NZ && cols[oc[z]] == cols[oc[a]]) {
            ++z;
        }
        idx.assign(oc.begin() + a, oc.begin() + z);
        const auto cj = cols[idx[0]];
        const std::int64_t dg = F.dests[ci2[cj]];
        std::int64_t have = 0;
        for (auto e : idx) {
            have += cnt[e];
        }
        const std::int64_t shortfall = static_cast<std::int64_t>(std::nearbyint(ct[cj])) - have;
        if (shortfall > 0) {
            frac.resize(idx.size());
            for (std::size_t i = 0; i < idx.size(); ++i) {
                frac[i] = v[idx[i]] - std::floor(v[idx[i]]);
            }
            if (seqSum(frac.data(), frac.size()) > 0) {
                runningSum(frac.data(), frac.size(), cum);
                const KR64 k = KR64(seed, rep, Stage::IPF_TRS_ADD).with(n).with(dg);
                for (std::int64_t j = 0; j < shortfall; ++j) {
                    ++cnt[idx[floatCdf(cum.data(), static_cast<std::int64_t>(cum.size()), k.with(j).u64(0))]];
                }
            }
        } else if (shortfall < 0) {
            std::vector<std::int64_t> nz;
            for (auto e : idx) {
                if (cnt[e] > 0) { nz.push_back(e); }
            }
            std::vector<std::int64_t> hg(nz.size());
            std::vector<std::uint64_t> dr(nz.size());
            const KR64 k = KR64(seed, rep, Stage::IPF_TRS_TRIM).with(n).with(dg);
            for (std::size_t i = 0; i < nz.size(); ++i) {
                hg[i] = F.homes[ri[rows[nz[i]]]];
                dr[i] = k.with(hg[i]).u64(0);
            }
            const auto o = stableOrder(nz.size(), [&] (std::int64_t x, std::int64_t y) {
                return dr[x] != dr[y] ? dr[x] < dr[y] : hg[x] < hg[y];
            });
            const std::size_t take = std::min<std::size_t>(static_cast<std::size_t>(-shortfall), nz.size());
            for (std::size_t i = 0; i < take; ++i) {
                --cnt[nz[o[i]]];
            }
        }
        a = z;
    }

    // Row repair: moves stay inside a row, whose cells are in destination order.
    const auto orr = stableOrder(NZ, [&] (std::int64_t x, std::int64_t y) {
        return rows[x] != rows[y] ? rows[x] < rows[y] : cols[x] < cols[y];
    });
    std::vector<std::int64_t> rptr(NR + 1, 0);
    for (std::size_t e = 0; e < NZ; ++e) {
        ++rptr[rows[e] + 1];
    }
    for (std::size_t r = 0; r < NR; ++r) {
        rptr[r + 1] += rptr[r];
    }
    auto rowDelta = [&] () {
        std::vector<std::int64_t> got(NR, 0);
        for (std::size_t e = 0; e < NZ; ++e) {
            got[rows[e]] += cnt[e];
        }
        std::vector<std::int64_t> delta(NR);
        for (std::size_t r = 0; r < NR; ++r) {
            delta[r] = supply[ri[r]] - got[r];
        }
        return delta;
    };
    std::vector<std::int64_t> w, wc;
    for (int sweep = 0; sweep < REPAIR_SWEEPS; ++sweep) {
        const auto delta = rowDelta();
        bool bad = false;
        for (std::size_t r = 0; r < NR; ++r) {
            const std::int64_t d = delta[r];
            if (d == 0) { continue; }
            bad = true;
            const std::int64_t* cells = orr.data() + rptr[r];
            const std::int64_t m = rptr[r + 1] - rptr[r];
            const std::int64_t hg = F.homes[ri[r]];
            const KR64 k = KR64(seed, rep, Stage::IPF_REPAIR).with(n).with(hg).with(sweep);
            if (d > 0) {
                wc.resize(m);
                std::int64_t s = 0;
                for (std::int64_t i = 0; i < m; ++i) {
                    wc[i] = (s += 2 * cnt[cells[i]] + 1);
                }
                std::vector<std::int64_t> add(m, 0);
                for (std::int64_t j = 0; j < d; ++j) {
                    ++add[intCdf(wc.data(), m, k.with(j).u64(0))];
                }
                for (std::int64_t i = 0; i < m; ++i) {
                    cnt[cells[i]] += add[i];
                }
            } else {
                wc.resize(m);
                std::int64_t s = 0;
                for (std::int64_t i = 0; i < m; ++i) {
                    wc[i] = (s += cnt[cells[i]]);
                }
                if (s == 0) { continue; }
                const std::int64_t take = std::min(-d, s);
                for (std::int64_t j = 0; j < take; ++j) {
                    const auto i = intCdf(wc.data(), m, k.with(j).u64(0));
                    if (cnt[cells[i]] > 0) { --cnt[cells[i]]; }
                }
            }
        }
        if (!bad) { break; }
    }
    // Deterministic final pass: settle what repair left on the row's largest cells.
    {
        const auto delta = rowDelta();
        for (std::size_t r = 0; r < NR; ++r) {
            std::int64_t d = delta[r];
            if (d == 0) { continue; }
            ++st.repair_final;
            const std::int64_t* cells = orr.data() + rptr[r];
            const std::int64_t m = rptr[r + 1] - rptr[r];
            while (d != 0) {
                std::int64_t j = cells[0];
                for (std::int64_t i = 1; i < m; ++i) {
                    if (cnt[cells[i]] > cnt[j]) { j = cells[i]; }
                }
                const std::int64_t step = d > 0 ? 1 : -1;
                if (step < 0 && cnt[j] == 0) { break; }
                cnt[j] += step;
                d -= step;
            }
        }
        for (auto d : rowDelta()) {
            if (d != 0) { throw std::runtime_error("industry " + std::to_string(n) + ": row sums not exact after repair"); }
        }
    }
    for (std::size_t e = 0; e < NZ; ++e) {
        if (cnt[e] <= 0) { continue; }
        if (cnt[e] == 1) { ++st.one_worker_cells; }
        out.row_h.push_back(ri[rows[e]]);
        out.col_d.push_back(ci2[cols[e]]);
        out.cnt.push_back(cnt[e]);
    }
    return out;
}

} // namespace

std::vector<std::int64_t> allocateWorkers (const PopulationBundle& b, const Persons& P, const SizeTables& tables,
                                           std::int64_t seed, std::int64_t rep, WorkerStats* stats_out) {
    WorkerStats st;
    const std::size_t n_persons = P.size();
    std::vector<std::int64_t> work(P.bg);
    std::vector<std::int64_t> W;
    for (std::size_t i = 0; i < n_persons; ++i) {
        if (P.employed[i]) { W.push_back(static_cast<std::int64_t>(i)); }
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

    // 1. LODES pairs with both ends at worker homes; summed if repeated, as scipy's CSR does.
    {
        const auto lh = b.get<std::int64_t>("lodes.home_geoid");
        const auto ld = b.get<std::int64_t>("lodes.dest_geoid");
        const auto ip = b.get<std::int64_t>("lodes.indptr");
        const auto ix = b.get<std::int32_t>("lodes.indices");
        const auto dv = b.get<std::int32_t>("lodes.data");
        auto isHome = [&] (std::int64_t g) {
            return std::binary_search(F.homes.begin(), F.homes.end(), g);
        };
        std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> pairs; // (home idx, dest geoid)
        for (std::size_t r = 0; r < lh.size(); ++r) {
            if (!isHome(lh[r])) { continue; }
            const auto h = position(F.homes, lh[r]);
            for (std::int64_t q = ip[r]; q < ip[r + 1]; ++q) {
                const std::int64_t dg = ld[ix[q]];
                if (isHome(dg)) { pairs[{h, dg}] += dv[q]; }
            }
        }
        for (const auto& kv : pairs) {
            F.dests.push_back(kv.first.second);
        }
        std::sort(F.dests.begin(), F.dests.end());
        F.dests.erase(std::unique(F.dests.begin(), F.dests.end()), F.dests.end());
        const std::int64_t Dn = static_cast<std::int64_t>(F.dests.size());
        F.hd_ptr.assign(H + 1, 0);
        std::vector<std::tuple<std::int64_t, std::int64_t, std::int64_t>> ent; // (h, d, count)
        for (const auto& kv : pairs) {
            ent.emplace_back(kv.first.first, position(F.dests, kv.first.second), kv.second);
        }
        std::sort(ent.begin(), ent.end());
        for (const auto& [h, d, c] : ent) {
            ++F.hd_ptr[h + 1];
            F.hd_col.push_back(d);
            F.hd_val.push_back(c);
        }
        for (std::int64_t h = 0; h < H; ++h) {
            F.hd_ptr[h + 1] += F.hd_ptr[h];
        }
        std::sort(ent.begin(), ent.end(), [] (const auto& x, const auto& y) {
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
    std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> sn_count;
    for (std::size_t w = 0; w < NW; ++w) {
        ++sn_count[{P.bg[W[w]] / 10000000000LL, naics_w[w]}];
    }
    std::int64_t tsum = 0;
    for (const auto& kv : sn_count) {
        tsum += kv.second * tables.target(kv.first.first, kv.first.second);
    }
    const double avg = static_cast<double>(tsum) / static_cast<double>(NW);
    std::vector<std::int64_t> implied(static_cast<std::size_t>(Dn) * n_naics, 0), cum(n_naics);
    for (std::int64_t d = 0; d < Dn; ++d) {
        const std::int64_t ns = dest_total[d] > 0 ? std::max<std::int64_t>(1, static_cast<std::int64_t>(std::nearbyint(
                                                                                      static_cast<double>(dest_total[d]) / avg)))
                                                  : 0;
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

    // 5. IPF fill per industry; each home's workers, in keyed order, dealt to its cells in
    //    destination order.
    std::vector<std::uint8_t> assigned(NW, 0);
    std::vector<std::uint64_t> okey(NW);
    const KR64 ka(seed, rep, Stage::WORK_ASSIGN);
    for (std::size_t w = 0; w < NW; ++w) {
        const auto i = W[w];
        okey[w] = ka.with(P.bg[i]).with(P.h[i]).with(P.p[i]).u64(0);
    }
    const auto wsort = stableOrder(NW, [&] (std::int64_t x, std::int64_t y) {
        if (naics_w[x] != naics_w[y]) { return naics_w[x] < naics_w[y]; }
        if (hidx_w[x] != hidx_w[y]) { return hidx_w[x] < hidx_w[y]; }
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
    std::vector<std::int64_t> sup(H), dem(Dn);
    for (int n = 0; n < n_naics; ++n) {
        std::int64_t dsum = 0;
        for (std::int64_t d = 0; d < Dn; ++d) {
            dsum += (dem[d] = demand[d * n_naics + n]);
        }
        if (true_total[n] == 0 || dsum == 0) { continue; }
        for (std::int64_t h = 0; h < H; ++h) {
            sup[h] = home_naics[h * n_naics + n];
        }
        const Fill f = fillOne(F, sup, dem, n, seed, rep, st);
        st.unplaceable += f.unplaced;
        if (f.cnt.empty()) { continue; }
        st.cells += static_cast<std::int64_t>(f.cnt.size());
        // cells sorted by (home, destination geoid)
        const auto co = stableOrder(f.cnt.size(), [&] (std::int64_t x, std::int64_t y) {
            return f.row_h[x] != f.row_h[y] ? f.row_h[x] < f.row_h[y] : F.dests[f.col_d[x]] < F.dests[f.col_d[y]];
        });
        const std::int64_t* wn = wsort.data() + nb[n];
        const std::int64_t nwn = nb[n + 1] - nb[n];
        for (std::size_t a = 0; a < co.size();) {
            std::size_t z = a;
            while (z < co.size() && f.row_h[co[z]] == f.row_h[co[a]]) {
                ++z;
            }
            const std::int64_t h = f.row_h[co[a]];
            const std::int64_t lo = std::lower_bound(wn, wn + nwn, h,
                                                     [&] (std::int64_t w, std::int64_t hh) {
                                                         return hidx_w[w] < hh;
                                                     }) -
                                    wn;
            std::int64_t k = lo;
            for (std::size_t c = a; c < z; ++c) {
                for (std::int64_t r = 0; r < f.cnt[co[c]]; ++r, ++k) {
                    work[W[wn[k]]] = F.dests[f.col_d[co[c]]];
                    assigned[wn[k]] = 1;
                }
            }
            a = z;
        }
    }

    // Fallback: a draw over the home's own LODES row; no row, work at home.
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
const std::vector<int> SCALES = {12, 11, 10, 7, 5};
const std::vector<int> CHILDCARE_SCALES = {12, 11, 10};

std::int64_t prefix (std::int64_t geoid, int s) {
    static const std::int64_t pow10[13] = {1,        10,        100,        1000,        10000,        100000,       1000000,
                                           10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000};
    return geoid / pow10[12 - s];
}

struct Schools {
    std::vector<std::int64_t> geoid, ord, students, teachers;
    std::vector<std::string> level; // level name per school
    std::vector<std::int64_t> county, adj_ptr, adj_ix;

    explicit Schools (const PopulationBundle& b) {
        const auto g = b.get<std::int64_t>("schools.geoid");
        const auto o = b.get<std::int16_t>("schools.ord");
        const auto s = b.get<std::int32_t>("schools.students");
        const auto t = b.get<std::int32_t>("schools.teachers");
        const auto lv = b.get<std::int8_t>("schools.level");
        const auto names = b.strings("schools.level_names");
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

    std::vector<std::int64_t> w_i, c_i;
    for (int li = 0; li < 6; ++li) {
        const std::string L = LEVELS[li];
        std::vector<std::int64_t> stud;
        for (std::size_t i = 0; i < n; ++i) {
            if (P.student[i] && P.grade[i] >= LEVEL_LO[li] && P.grade[i] <= LEVEL_HI[li]) {
                stud.push_back(static_cast<std::int64_t>(i));
            }
        }
        if (stud.empty()) { continue; }
        std::vector<std::int64_t> rows, remaining;
        for (std::size_t i = 0; i < NS; ++i) {
            if (S.level[i].find(L) != std::string::npos && in_pop[i]) {
                const auto nlev = static_cast<std::int64_t>(S.level[i].size());
                rows.push_back(static_cast<std::int64_t>(i));
                remaining.push_back((S.students[i] + nlev - 1) / nlev);
            }
        }
        const std::size_t NR = rows.size();
        const auto& scales = L == "C" ? CHILDCARE_SCALES : SCALES;
        std::vector<std::uint8_t> placed(stud.size(), 0);
        auto assign = [&] (std::vector<std::int64_t>& taken) {
            for (std::size_t j = 0; j < w_i.size(); ++j) {
                school[stud[w_i[j]]] = rows[c_i[j]];
                placed[w_i[j]] = 1;
                ++taken[c_i[j]];
            }
        };
        for (int scale : scales) {
            const bool alloc_all = L != "U" && scale == scales.back();
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
            for (std::size_t r = 0; r < NR; ++r) {
                remaining[r] = std::max<std::int64_t>(remaining[r] - taken[r], 1);
            }
        }
        if (L == "U") {
            // University: each home county plus its neighbours, counties in ascending FIPS.
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
                fillRegion(who, cand, remaining, rows, S, li, 0, cty, true, seed, rep, w_i, c_i);
                std::vector<std::int64_t> taken(NR, 0);
                assign(taken);
                for (std::size_t r = 0; r < NR; ++r) {
                    remaining[r] = std::max<std::int64_t>(remaining[r] - taken[r], 1);
                }
            }
        }
        std::int64_t unplaced = 0;
        for (std::size_t j = 0; j < stud.size(); ++j) {
            if (!placed[j]) {
                P.grade[stud[j]] = -1;
                ++unplaced;
            }
        }
        if (stats) { (*stats)[L] = {static_cast<std::int64_t>(stud.size()), unplaced}; }
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
            for (const auto region : regions) {
                std::vector<std::int64_t> cand, pool;
                for (std::size_t r = 0; r < NR; ++r) {
                    if (s_reg[r] == region && need[r] > 0) { cand.push_back(static_cast<std::int64_t>(r)); }
                }
                if (scale) {
                    for (std::size_t t = 0; t < teach.size(); ++t) {
                        if (freev[t] && t_reg[t] == region) { pool.push_back(static_cast<std::int64_t>(t)); }
                    }
                } else {
                    const auto nb = S.neighbourhood(region);
                    for (std::size_t t = 0; t < teach.size(); ++t) {
                        if (freev[t] && std::binary_search(nb.begin(), nb.end(), t_reg[t])) {
                            pool.push_back(static_cast<std::int64_t>(t));
                        }
                    }
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
}

// ---------------------------------------------------------------------------------------------
// S6-S10 groups (groups.py)
// ---------------------------------------------------------------------------------------------

namespace {

constexpr std::int64_t NBORHOOD_SIZE = 500;
constexpr std::int64_t WORKGROUP_SIZE = 20;
constexpr std::int64_t CLASS_SIZE = 20, CLASS_MIN = 5, CLASS_MAX = 50;
constexpr double COLLEGE_INSTRUCTIONAL_FRACTION = 0.1;
constexpr int TRAVEL_WFH = 7;

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
        std::sort(used.begin(), used.end());
        used.erase(std::unique(used.begin(), used.end()), used.end());
        std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> local;
        for (std::size_t j = 0; j < used.size(); ++j) {
            const bool first = j == 0 || used[j].first != used[j - 1].first;
            local[used[j]] = first ? 1 : local[used[j - 1]] + 1;
        }
        for (std::int64_t i = 0; i < n; ++i) {
            if (school[i] >= 0 && P.grade[i] != -1) { R.school_id[i] = local.at({sg[school[i]], so[school[i]]}); }
        }
    }
    if (digests) { (*digests)["S6 school ids"] = Digest().add(R.school_id).hex16(); }

    // S7: neighbourhood per household (keyed), household cluster = h mod ceil(households / 4).
    R.nborhood.assign(n, 0);
    R.hh_cluster.assign(n, 0);
    {
        const KR64 k(seed, rep, Stage::HOME_NB);
        for (std::int64_t a = 0; a < n;) {
            std::int64_t z = a, n_hh = 0;
            while (z < n && P.bg[z] == P.bg[a]) {
                n_hh = std::max(n_hh, P.h[z] + 1);
                ++z;
            }
            const std::int64_t pop = z - a;
            const std::int64_t max_nb = std::max<std::int64_t>(1, (2 * pop + NBORHOOD_SIZE) / (2 * NBORHOOD_SIZE));
            const std::int64_t clusters = std::max<std::int64_t>(1, (n_hh + 3) / 4);
            for (std::int64_t i = a; i < z; ++i) {
                R.nborhood[i] = index(k.with(P.bg[i]).with(P.h[i]).u64(0), max_nb);
                R.hh_cluster[i] = P.h[i] % clusters;
            }
            a = z;
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
        for (std::size_t j = 0; j < el.size(); ++j) {
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
        std::vector<std::int64_t> starts, team_count;
        const KR64 ke(seed, rep, Stage::WG_EST_SIZE);
        for (std::size_t lo = 0; lo < s.size();) {
            std::size_t hi = lo;
            while (hi < s.size() && work[s[hi]] == work[s[lo]] && P.naics[s[hi]] == P.naics[s[lo]]) {
                ++hi;
            }
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
            starts.push_back(static_cast<std::int64_t>(lo));
            team_count.push_back(tb);
            lo = hi;
        }
        // dense ids: exclusive scan of team counts over groups in (geoid, NAICS) order
        std::int64_t base = 0;
        for (std::size_t gi = 0; gi < starts.size(); ++gi) {
            const std::int64_t lo = starts[gi];
            const std::int64_t hi = gi + 1 < starts.size() ? starts[gi + 1] : static_cast<std::int64_t>(s.size());
            for (std::int64_t j = lo; j < hi; ++j) {
                R.work_group[s[j]] = base + R.workgroup[s[j]] - 1;
            }
            base += team_count[gi];
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
        std::stable_sort(en.begin(), en.end(), [&] (std::int64_t i, std::int64_t j) {
            if (work[i] != work[j]) { return work[i] < work[j]; }
            if (R.school_id[i] != R.school_id[j]) { return R.school_id[i] < R.school_id[j]; }
            if (P.grade[i] != P.grade[j]) { return P.grade[i] < P.grade[j]; }
            return i < j;
        });
        const KR64 ks(seed, rep, Stage::CLASS_SMEAR);
        std::vector<std::int64_t> local(n, 0), cls;
        std::int64_t base = 0;
        for (std::size_t lo = 0; lo < en.size();) {
            std::size_t hi = lo;
            while (hi < en.size() && work[en[hi]] == work[en[lo]] && R.school_id[en[hi]] == R.school_id[en[lo]] &&
                   P.grade[en[hi]] == P.grade[en[lo]]) {
                ++hi;
            }
            std::int64_t n_st = 0;
            for (std::size_t j = lo; j < hi; ++j) {
                n_st += P.naics[en[j]] == -1 ? 1 : 0;
            }
            const std::int64_t n_te = static_cast<std::int64_t>(hi - lo) - n_st;
            const std::int64_t m0 = en[lo];
            std::int64_t n_classes = 0;
            if (n_st > 0) {
                const bool college = P.grade[m0] > 17;
                const double eff =
                        college ? static_cast<double>(n_te) * COLLEGE_INSTRUCTIONAL_FRACTION : static_cast<double>(n_te);
                const std::int64_t raw = eff > 0.0 ? std::max<std::int64_t>(1, static_cast<std::int64_t>(eff))
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
                local[i] = c >= 0 ? c : n_classes + (-2 - c);
            }
            for (std::size_t j = lo; j < hi; ++j) {
                R.school_class_group[en[j]] = base + local[en[j]];
            }
            base += n_classes + n_admin;
            lo = hi;
        }
    }
    if (digests) { (*digests)["S9 school groups"] = Digest().add(R.school_class).add(R.school_class_group).hex16(); }

    // S10: day neighbourhoods -- atoms packed per daytime block group on an integer midpoint grid.
    R.work_nborhood.assign(n, 0);
    {
        std::vector<std::int64_t> day(n), kind(n), a(n, 0), bb(n, 0), c(n, 0);
        std::map<std::int64_t, std::int64_t> school_size; // (day geoid * 100000 + school id) -> members
        for (std::int64_t i = 0; i < n; ++i) {
            const bool at_school = R.school_id[i] != 0, at_work = R.workgroup[i] > 0;
            day[i] = (!at_school && !at_work) ? P.bg[i] : work[i];
            if (at_school) { ++school_size[day[i] * 100000 + R.school_id[i]]; }
        }
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
        // atoms = distinct (day, kind, a, b, c), sorted
        using Atom = std::array<std::int64_t, 5>;
        std::map<Atom, std::int64_t> atom_size;
        for (std::int64_t i = 0; i < n; ++i) {
            ++atom_size[{day[i], kind[i], a[i], bb[i], c[i]}];
        }
        const std::size_t NA = atom_size.size();
        std::vector<Atom> atoms;
        std::vector<std::int64_t> asize;
        atoms.reserve(NA);
        for (const auto& [at, sz] : atom_size) {
            atoms.push_back(at);
            asize.push_back(sz);
        }
        const KR64 kd(seed, rep, Stage::DAY_NB);
        std::vector<std::uint64_t> dk(NA);
        std::vector<std::uint8_t> over(NA);
        for (std::size_t j = 0; j < NA; ++j) {
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
        for (std::size_t lo = 0; lo < NA;) {
            std::size_t hi = lo;
            while (hi < NA && atoms[order[hi]][0] == atoms[order[lo]][0]) {
                ++hi;
            }
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
            lo = hi;
        }
        for (std::int64_t i = 0; i < n; ++i) {
            const Atom key = {day[i], kind[i], a[i], bb[i], c[i]};
            const auto j = std::lower_bound(atoms.begin(), atoms.end(), key) - atoms.begin();
            R.work_nborhood[i] = atom_bin[j];
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
