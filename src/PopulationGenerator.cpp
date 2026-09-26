/*! @file PopulationGenerator.cpp
    \brief Placement and population stages; see PopulationGenerator.H. Each section names the
    Python function it ports.
*/
#include "PopulationGenerator.H"

#include <algorithm>
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
