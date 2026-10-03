/*! @file RandomTravel.cpp
    \brief Irregular (long-distance) travel, after Epicast 2.0 -- see RandomTravel.H
*/

#include <algorithm>
#include <cmath>
#include <numeric>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "AgentContainer.H"
#include "RandomTravel.H"
#include "UrbanPopData.H"

using namespace amrex;

void RandomTravel::readInputs () {
    ParmParse pp("agent");

    std::vector<Real> p;
    if (pp.queryarr("random_travel_prob", p)) {
        if (p.size() == 1) {
            for (int a = 0; a < AgeGroups::total; a++) {
                prob_per_step[a] = p[0];
            }
        } else if (p.size() == AgeGroups::total) {
            for (int a = 0; a < AgeGroups::total; a++) {
                prob_per_step[a] = p[a];
            }
        } else {
            Abort("agent.random_travel_prob needs 1 value or one per age group (" + std::to_string(AgeGroups::total) + ")");
        }
    }
    pp.query("random_travel_scale", scale);
    pp.query("random_travel_steps_per_day", steps_per_day);
    pp.query("random_travel_in_state_frac", in_state_frac);
    std::string out_of_state = "remove";
    pp.query("random_travel_out_of_state", out_of_state);
    if (out_of_state == "remove") {
        remove_out_of_state = true;
    } else if (out_of_state == "skip") {
        remove_out_of_state = false;
    } else {
        Abort("agent.random_travel_out_of_state must be remove or skip, not " + out_of_state);
    }
    std::vector<Real> cdf;
    if (pp.queryarr("random_travel_duration_cdf", cdf)) { duration_cdf = cdf; }
    if (duration_cdf.empty() || std::abs(duration_cdf.back() - 1.0_rt) > 1e-6_rt ||
        !std::is_sorted(duration_cdf.begin(), duration_cdf.end())) {
        Abort("agent.random_travel_duration_cdf must be nondecreasing and end at 1");
    }
    if (steps_per_day < 1 || scale < 0.0_rt || in_state_frac < 0.0_rt || in_state_frac > 1.0_rt) {
        Abort("agent.random_travel_steps_per_day must be >= 1, random_travel_scale >= 0 and "
              "random_travel_in_state_frac in [0, 1]");
    }
    for (int a = 0; a < AgeGroups::total; a++) {
        Real p_step = std::min(1.0_rt, scale * prob_per_step[a]);
        prob_per_day[a] = 1.0_rt - std::pow(1.0_rt - p_step, steps_per_day);
    }
}

void AgentContainer::initRandomTravel (const UrbanPopData& urbanpopData) {
    BL_PROFILE("AgentContainer::initRandomTravel");
    auto& rt = m_random_travel;
    rt.readInputs();

    const auto& bgs = urbanpopData.block_groups;
    const int nbg = (int)bgs.size();
    rt.nx = Geom(0).Domain().length(0);

    // Destinations: block groups with residents, grouped by tract (the geoid without its last digit)
    std::vector<int> order;
    for (int b = 0; b < nbg; b++) {
        if (bgs[b].home_population > 0) { order.push_back(b); }
    }
    std::sort(order.begin(), order.end(), [&] (int a, int b) {
        return bgs[a].geoid < bgs[b].geoid;
    });
    Vector<int> tract_offset, bg_by_tract;
    int64_t prev_tract = -1;
    for (int b : order) {
        int64_t tract = bgs[b].geoid / 10;
        if (tract != prev_tract) {
            tract_offset.push_back((int)bg_by_tract.size());
            prev_tract = tract;
        }
        bg_by_tract.push_back(b);
    }
    tract_offset.push_back((int)bg_by_tract.size());
    rt.num_tracts = (int)tract_offset.size() - 1;
    AMREX_ALWAYS_ASSERT(rt.num_tracts > 0);

    rt.tract_offset_d.resize(tract_offset.size());
    Gpu::copyAsync(Gpu::hostToDevice, tract_offset.begin(), tract_offset.end(), rt.tract_offset_d.begin());
    rt.bg_by_tract_d.resize(bg_by_tract.size());
    Gpu::copyAsync(Gpu::hostToDevice, bg_by_tract.begin(), bg_by_tract.end(), rt.bg_by_tract_d.begin());
    rt.duration_cdf_d.resize(rt.duration_cdf.size());
    Gpu::copyAsync(Gpu::hostToDevice, rt.duration_cdf.begin(), rt.duration_cdf.end(), rt.duration_cdf_d.begin());

    // Per block group: the number of nighttime neighborhoods and household clusters among its
    // residents, and of daytime neighborhoods among those who spend the day there (ids are 0-based)
    rt.n_nborhood_d.resize(nbg);
    rt.n_work_nborhood_d.resize(nbg);
    rt.n_hh_cluster_d.resize(nbg);
    auto n_nb = rt.n_nborhood_d.data();
    auto n_wnb = rt.n_work_nborhood_d.data();
    auto n_hhc = rt.n_hh_cluster_d.data();
    ParallelFor(nbg, [=] AMREX_GPU_DEVICE (int b) noexcept {
        n_nb[b] = 0;
        n_wnb[b] = 0;
        n_hhc[b] = 0;
    });
    const int nx = rt.nx;
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = ParticlesAt(lev, mfi);
            const auto np = ptile.numParticles();
            if (np == 0) { continue; }
            auto& soa = ptile.GetStructOfArrays();
            auto home_i = soa.GetIntData(IntIdx::home_i).data();
            auto home_j = soa.GetIntData(IntIdx::home_j).data();
            auto work_i = soa.GetIntData(IntIdx::work_i).data();
            auto work_j = soa.GetIntData(IntIdx::work_j).data();
            auto nborhood = soa.GetIntData(IntIdx::nborhood).data();
            auto work_nborhood = soa.GetIntData(IntIdx::work_nborhood).data();
            auto hh_cluster = soa.GetIntData(IntIdx::hh_cluster).data();
            ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                int hb = home_j[i] * nx + home_i[i];
                int wb = work_j[i] * nx + work_i[i];
                if (hb >= 0 && hb < nbg) {
                    Gpu::Atomic::Max(&n_nb[hb], nborhood[i] + 1);
                    Gpu::Atomic::Max(&n_hhc[hb], hh_cluster[i] + 1);
                }
                if (wb >= 0 && wb < nbg) { Gpu::Atomic::Max(&n_wnb[wb], work_nborhood[i] + 1); }
            });
        }
    }
    Gpu::streamSynchronize();
    for (auto* dv : {&rt.n_nborhood_d, &rt.n_work_nborhood_d, &rt.n_hh_cluster_d}) {
        Vector<int> h(nbg);
        Gpu::copy(Gpu::deviceToHost, dv->begin(), dv->end(), h.begin());
        ParallelDescriptor::ReduceIntMax(h.data(), nbg);
        Gpu::copy(Gpu::hostToDevice, h.begin(), h.end(), dv->begin());
    }

    m_random_travel_on = true;

    Real mean_days = 0.0_rt;
    for (size_t d = 0; d < rt.duration_cdf.size(); d++) {
        mean_days += (Real)(d + 1) * (rt.duration_cdf[d] - (d > 0 ? rt.duration_cdf[d - 1] : 0.0_rt));
    }
    Print() << "Irregular travel: " << rt.num_tracts << " destination tracts (" << bg_by_tract.size()
            << " block groups); daily start probability by age group";
    for (int a = 0; a < AgeGroups::total; a++) {
        Print() << " " << rt.prob_per_day[a];
    }
    Print() << "; mean trip " << mean_days << " days; " << rt.in_state_frac << " of trips in the region, the rest "
            << (rt.remove_out_of_state ? "taking the traveller out of the simulation" : "not taken") << "\n";
}

void AgentContainer::updateRandomTravel (bool start_trips) {
    BL_PROFILE("AgentContainer::updateRandomTravel");
    auto& rt = m_random_travel;
    const auto prob_per_day = rt.prob_per_day;
    const auto cdf = rt.duration_cdf_d.data();
    const int ncdf = (int)rt.duration_cdf_d.size();
    const auto tract_offset = rt.tract_offset_d.data();
    const auto bg_by_tract = rt.bg_by_tract_d.data();
    const int num_tracts = rt.num_tracts;
    const auto n_nb = rt.n_nborhood_d.data();
    const auto n_wnb = rt.n_work_nborhood_d.data();
    const auto n_hhc = rt.n_hh_cluster_d.data();
    const int nx = rt.nx;
    const Real in_state_frac = rt.in_state_frac;
    const bool remove_out_of_state = rt.remove_out_of_state;

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        const auto dx = Geom(lev).CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            auto& aos = ptile.GetArrayOfStructs();
            const auto np = aos.numParticles();
            if (np == 0) { continue; }
            ParticleType* pstruct = &(aos[0]);
            auto& soa = ptile.GetStructOfArrays();
            auto random_travel = soa.GetIntData(IntIdx::random_travel).data();
            auto air_travel = soa.GetIntData(IntIdx::air_travel).data();
            auto withdrawn = soa.GetIntData(IntIdx::withdrawn).data();
            auto age_group = soa.GetIntData(IntIdx::age_group).data();
            auto home_i = soa.GetIntData(IntIdx::home_i).data();
            auto home_j = soa.GetIntData(IntIdx::home_j).data();
            auto trav_i = soa.GetIntData(IntIdx::trav_i).data();
            auto trav_j = soa.GetIntData(IntIdx::trav_j).data();
            auto trav_nborhood = soa.GetIntData(IntIdx::trav_nborhood).data();
            auto trav_work_nborhood = soa.GetIntData(IntIdx::trav_work_nborhood).data();
            auto trav_hh_cluster = soa.GetIntData(IntIdx::trav_hh_cluster).data();
            auto weather = soa.GetIntData(IntIdx::weatherLookup).data();

            ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                ParticleType& p = pstruct[i];
                const bool dead = ptd.m_runtime_idata[i0(0) + IntIdxDisease::status][i] == Status::dead;
                int days_left = random_travel[i];
                if (days_left >= 0) {
                    // a trip of D days was set to D on its first day: it ends on the morning of day D + 1
                    days_left -= 1;
                    if (days_left <= 0 || inHospital(i, ptd) || dead) {
                        // the traveller is already home (placeTravellers(true) runs every evening)
                        if (trav_i[i] >= 0) { weather[i] = -weather[i] - 2; }
                        random_travel[i] = -1;
                    } else {
                        random_travel[i] = days_left;
                        if (trav_i[i] >= 0) {
                            p.pos(0) = static_cast<ParticleReal>((trav_i[i] + 0.5_rt) * dx[0]);
                            p.pos(1) = static_cast<ParticleReal>((trav_j[i] + 0.5_rt) * dx[1]);
                        }
                    }
                    return;
                }
                if (!start_trips || air_travel[i] >= 0 || inHospital(i, ptd) || withdrawn[i] || dead) { return; }
                if (Random(engine) >= prob_per_day[age_group[i]]) { return; }

                // duration: inverse transform sampling of the eCDF (entry d is P(D <= d + 1 days))
                const Real u = Random(engine);
                int d = 0;
                while (d < ncdf - 1 && cdf[d] < u) {
                    ++d;
                }
                const int days = d + 1;

                if (Random(engine) < in_state_frac) {
                    // a uniformly random tract, then a uniformly random block group in it
                    const int t = amrex::min((int)(Random(engine) * num_tracts), num_tracts - 1);
                    const int first = tract_offset[t];
                    const int n = tract_offset[t + 1] - first;
                    const int b = bg_by_tract[first + amrex::min((int)(Random(engine) * n), n - 1)];
                    auto pick = [&] (int count) {
                        return count > 0 ? amrex::min((int)(Random(engine) * count), count - 1) : 0;
                    };
                    trav_i[i] = b % nx;
                    trav_j[i] = b / nx;
                    trav_nborhood[i] = pick(n_nb[b]);
                    trav_work_nborhood[i] = pick(n_wnb[b]);
                    trav_hh_cluster[i] = pick(n_hhc[b]);
                    p.pos(0) = static_cast<ParticleReal>((trav_i[i] + 0.5_rt) * dx[0]);
                    p.pos(1) = static_cast<ParticleReal>((trav_j[i] + 0.5_rt) * dx[1]);
                    weather[i] = -weather[i] - 2;
                    random_travel[i] = days;
                } else if (remove_out_of_state) {
                    trav_i[i] = -1;
                    trav_j[i] = -1;
                    random_travel[i] = days;
                }
            });
        }
    }
    // no Redistribute here: morningCommute()'s follows immediately and moves travellers to their tiles
}

void AgentContainer::placeTravellers (bool at_home) {
    BL_PROFILE("AgentContainer::placeTravellers");
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        const auto dx = Geom(lev).CellSizeArray();
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            auto& aos = ptile.GetArrayOfStructs();
            const auto np = aos.numParticles();
            if (np == 0) { continue; }
            ParticleType* pstruct = &(aos[0]);
            auto& soa = ptile.GetStructOfArrays();
            auto home_i = soa.GetIntData(IntIdx::home_i).data();
            auto home_j = soa.GetIntData(IntIdx::home_j).data();
            auto trav_i = soa.GetIntData(IntIdx::trav_i).data();
            auto trav_j = soa.GetIntData(IntIdx::trav_j).data();
            ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (onTripInRegion(i, ptd)) {
                    ParticleType& p = pstruct[i];
                    p.pos(0) = static_cast<ParticleReal>(((at_home ? home_i[i] : trav_i[i]) + 0.5_rt) * dx[0]);
                    p.pos(1) = static_cast<ParticleReal>(((at_home ? home_j[i] : trav_j[i]) + 0.5_rt) * dx[1]);
                }
            });
        }
    }
    Redistribute();
    AMREX_ASSERT(OK());
}

void AgentContainer::accumulateRandomTravelDiagnostics () {
    BL_PROFILE("AgentContainer::accumulateRandomTravelDiagnostics");
    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
    auto r = ParticleReduce<ReduceData<Real, Real, Real, Real>>(
            *this,
            [=] AMREX_GPU_DEVICE (const ParticleTileType::ConstParticleTileDataType& ptd,
                                 const int i) noexcept -> GpuTuple<Real, Real, Real, Real> {
                const bool in_region = onTripInRegion(i, ptd);
                const bool away = awayFromRegion(i, ptd);
                Real exp_inf = 0.0_rt;
                auto status = ptd.m_runtime_idata[i0(0) + IntIdxDisease::status][i];
                if (status == Status::never || status == Status::susceptible) {
                    exp_inf = amrex::max(0.0_rt, 1.0_rt - (Real)ptd.m_runtime_rdata[r0(0) + RealIdxDisease::prob][i]);
                }
                return {in_region ? 1.0_rt : 0.0_rt, away ? 1.0_rt : 0.0_rt, in_region ? exp_inf : 0.0_rt, exp_inf};
            },
            reduce_ops);
    Real v[4] = {get<0>(r), get<1>(r), get<2>(r), get<3>(r)};
    ParallelDescriptor::ReduceRealSum(v, 4);
    auto& rt = m_random_travel;
    rt.agent_days_in_state += v[0];
    rt.agent_days_away += v[1];
    rt.infections_on_trip += v[2];
    rt.infections_all += v[3];
    rt.days++;
}

void AgentContainer::printRandomTravelSummary () const {
    const auto& rt = m_random_travel;
    if (rt.days == 0) { return; }
    const Real pop = (Real)TotalNumberOfParticles();
    Print() << "Irregular travel over " << rt.days << " days: on average " << rt.agent_days_in_state / rt.days
            << " agents a day on trips in the region (" << 100.0 * rt.agent_days_in_state / rt.days / pop << "%), "
            << rt.agent_days_away / rt.days << " away out of it (" << 100.0 * rt.agent_days_away / rt.days / pop
            << "%); expected infections on trips " << rt.infections_on_trip << " of " << rt.infections_all << " ("
            << (rt.infections_all > 0 ? 100.0 * rt.infections_on_trip / rt.infections_all : 0.0) << "%)\n";
}
