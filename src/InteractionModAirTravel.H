/*! @file InteractionModAirTravel.H
 * \brief Contains the class describing agent interactions for air travel
 */

#ifndef _INTERACTION_MOD_AIRTRAVEL_H_
#define _INTERACTION_MOD_AIRTRAVEL_H_

#include <math.h>

#include "AgentDefinitions.H"
#include "DiseaseParm.H"
#include "InteractionModel.H"

using namespace amrex;

/*! \brief Class describing agent interactions for air travel
 *
 * Two symmetric sub-interactions are applied:
 *  1. Infectious air-travel agents (in a_on_air_travel) infect susceptible local agents.
 *  2. Infectious local agents infect susceptible air-travel agents.
 *
 * Both use xmit_comm[age_group] for community-level exposure and xmit_hood[age_group]
 * for same-neighborhood exposure.  The school_closed flag on the infectious agent selects
 * the _SC variants.
 */
template <typename PCType, typename PTDType, typename PType>
class InteractionModAirTravel {
  public:
    void interactAgents(PCType& a_agents, MultiFab& a_mask, PCType& a_on_air_travel);

    /*! \brief Count infectious agents from src, apply exposure to susceptibles in dst.
     *
     * Both src and dst must share the same MFIter structure (same BoxArray/DistributionMapping).
     * Agents in src with air_travel_flag=true are never excluded — the caller decides which
     * container is the "traveler" side.
     *
     * \param src_is_travel  true when src=a_on_air_travel (skip air_travel<0 filter on src)
     * \param dst_is_travel  true when dst=a_on_air_travel (skip air_travel<0 filter on dst)
     */
    void interactOneWay(PCType& src, PCType& dst, int d, bool src_is_travel, bool dst_is_travel,
                        Vector<Gpu::DeviceVector<int>>& n_comm_d, Vector<Gpu::DeviceVector<int>>& n_comm_sc_d,
                        Vector<Gpu::DeviceVector<int>>& n_comm_asymp_d, Vector<Gpu::DeviceVector<int>>& n_comm_sc_asymp_d,
                        Vector<Gpu::DeviceVector<int>>& n_hood_d, Vector<Gpu::DeviceVector<int>>& n_hood_sc_d,
                        Vector<Gpu::DeviceVector<int>>& n_hood_asymp_d, Vector<Gpu::DeviceVector<int>>& n_hood_sc_asymp_d);
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModAirTravel<PCType, PTDType, PType>::interactAgents (PCType& a_agents, MultiFab& a_mask,
                                                                      PCType& a_on_air_travel) {
    BL_PROFILE("InteractionModAirTravel::interactAgents");
    int n_disease = a_agents.numDiseases();

    // Allocate count arrays once; interactOneWay resizes them per tile as needed.
    Vector<Gpu::DeviceVector<int>> n_comm_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> n_comm_sc_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> n_comm_asymp_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> n_comm_sc_asymp_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> n_hood_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> n_hood_sc_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> n_hood_asymp_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> n_hood_sc_asymp_d(OMP_MAX_THREADS);

    for (int d = 0; d < n_disease; d++) {
        // Air-travel agents infect local susceptibles
        interactOneWay(a_on_air_travel, a_agents, d,
                       /*src_is_travel=*/true, /*dst_is_travel=*/false, n_comm_d, n_comm_sc_d, n_comm_asymp_d, n_comm_sc_asymp_d,
                       n_hood_d, n_hood_sc_d, n_hood_asymp_d, n_hood_sc_asymp_d);

        // Local agents infect air-travel susceptibles
        interactOneWay(a_agents, a_on_air_travel, d,
                       /*src_is_travel=*/false, /*dst_is_travel=*/true, n_comm_d, n_comm_sc_d, n_comm_asymp_d, n_comm_sc_asymp_d,
                       n_hood_d, n_hood_sc_d, n_hood_asymp_d, n_hood_sc_asymp_d);
    }
}

template <typename PCType, typename PTDType, typename PType>
void InteractionModAirTravel<PCType, PTDType, PType>::interactOneWay (
        PCType& src, PCType& dst, int d, bool src_is_travel, bool dst_is_travel, Vector<Gpu::DeviceVector<int>>& n_comm_d,
        Vector<Gpu::DeviceVector<int>>& n_comm_sc_d, Vector<Gpu::DeviceVector<int>>& n_comm_asymp_d,
        Vector<Gpu::DeviceVector<int>>& n_comm_sc_asymp_d, Vector<Gpu::DeviceVector<int>>& n_hood_d,
        Vector<Gpu::DeviceVector<int>>& n_hood_sc_d, Vector<Gpu::DeviceVector<int>>& n_hood_asymp_d,
        Vector<Gpu::DeviceVector<int>>& n_hood_sc_asymp_d) {

    auto lparm_h = src.getDiseaseParameters_h(d);
    Real infect = 1.0_rt - lparm_h->vac_eff;
    Real asymp_inf = lparm_h->asymp_relative_inf;

    for (int lev = 0; lev < src.numLevels(); ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = src.MakeMFIter(lev); mfi.isValid(); ++mfi) {

            // --- Source tile (infectious agents to count) ---
            auto& src_ptile = src.ParticlesAt(lev, mfi);
            const auto& src_ptd = src_ptile.getParticleTileData();
            const auto src_np = src_ptile.GetArrayOfStructs().numParticles();

            // --- Destination tile (susceptible agents to expose) ---
            auto& dst_ptile = dst.ParticlesAt(lev, mfi);
            const auto& dst_ptd = dst_ptile.getParticleTileData();
            const auto dst_np = dst_ptile.GetArrayOfStructs().numParticles();

            if (src_np == 0 && dst_np == 0) { continue; }

            auto& src_soa = src_ptile.GetStructOfArrays();
            auto& dst_soa = dst_ptile.GetStructOfArrays();

            // Community index structure — built from src geometry (same BoxArray for both)
            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(src.Geom(lev), mfi.tilebox(), src.comm_mf[mfi].array());
            auto plo = getCommunityIndex.plo;
            auto dxi = getCommunityIndex.dxi;
            auto domain = getCommunityIndex.domain;
            auto valid_box = getCommunityIndex.valid_box;
            auto bin_size = getCommunityIndex.bin_size;
            int max_communities = getCommunityIndex.max();
            auto d_ptr = getCommunityIndex.comm_to_local_index_d.data();

            int max_nborhood = src.getMaxGroup(IntIdx::nborhood) + 1;

            int comm_sz = max_communities;
            int hood_sz = max_communities * max_nborhood;

            n_comm_d[OMP_THREAD_NUM].resize(comm_sz);
            n_comm_sc_d[OMP_THREAD_NUM].resize(comm_sz);
            n_comm_asymp_d[OMP_THREAD_NUM].resize(comm_sz);
            n_comm_sc_asymp_d[OMP_THREAD_NUM].resize(comm_sz);
            n_hood_d[OMP_THREAD_NUM].resize(hood_sz);
            n_hood_sc_d[OMP_THREAD_NUM].resize(hood_sz);
            n_hood_asymp_d[OMP_THREAD_NUM].resize(hood_sz);
            n_hood_sc_asymp_d[OMP_THREAD_NUM].resize(hood_sz);

            auto n_comm_ptr = n_comm_d[OMP_THREAD_NUM].data();
            auto n_comm_sc_ptr = n_comm_sc_d[OMP_THREAD_NUM].data();
            auto n_comm_asymp_ptr = n_comm_asymp_d[OMP_THREAD_NUM].data();
            auto n_comm_sc_asymp_ptr = n_comm_sc_asymp_d[OMP_THREAD_NUM].data();
            auto n_hood_ptr = n_hood_d[OMP_THREAD_NUM].data();
            auto n_hood_sc_ptr = n_hood_sc_d[OMP_THREAD_NUM].data();
            auto n_hood_asymp_ptr = n_hood_asymp_d[OMP_THREAD_NUM].data();
            auto n_hood_sc_asymp_ptr = n_hood_sc_asymp_d[OMP_THREAD_NUM].data();

            devMemset(n_comm_ptr, 0, comm_sz * sizeof(int));
            devMemset(n_comm_sc_ptr, 0, comm_sz * sizeof(int));
            devMemset(n_comm_asymp_ptr, 0, comm_sz * sizeof(int));
            devMemset(n_comm_sc_asymp_ptr, 0, comm_sz * sizeof(int));
            devMemset(n_hood_ptr, 0, hood_sz * sizeof(int));
            devMemset(n_hood_sc_ptr, 0, hood_sz * sizeof(int));
            devMemset(n_hood_asymp_ptr, 0, hood_sz * sizeof(int));
            devMemset(n_hood_sc_asymp_ptr, 0, hood_sz * sizeof(int));

            auto lparm = src.getDiseaseParameters_d(d);
            auto src_nborhood_ptr = src_soa.GetIntData(IntIdx::nborhood).data();
            auto src_school_cl_ptr = src_soa.GetIntData(IntIdx::school_closed).data();

            // --- Count pass: infectious agents in src tile ---
            bool is_src_travel = src_is_travel;
            ParallelFor(src_np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (!isInfectious(i, src_ptd, d)) { return; }
                if (src_ptd.m_idata[IntIdx::withdrawn][i]) { return; }
                if (inHospital(i, src_ptd)) { return; }
                // Only local agents (not relocated travelers) count from the local container
                if (!is_src_travel && src_ptd.m_idata[IntIdx::air_travel][i] >= 0) { return; }
                if (src_ptd.m_idata[IntIdx::random_travel][i] >= 0) { return; }

                Box tbx;
                auto iv = getParticleCell(src_ptd, i, plo, dxi, domain);
                auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                int community = d_ptr[tidx];
                int nborhood_i = community * max_nborhood + src_nborhood_ptr[i];
                bool sc = (src_school_cl_ptr[i] == 1);
                bool asymp = isAsymptomatic(i, src_ptd, d) || isPresymptomatic(i, src_ptd, d);

                if (asymp) {
                    if (sc) {
                        Gpu::Atomic::AddNoRet(&n_comm_sc_asymp_ptr[community], 1);
                        Gpu::Atomic::AddNoRet(&n_hood_sc_asymp_ptr[nborhood_i], 1);
                    } else {
                        Gpu::Atomic::AddNoRet(&n_comm_asymp_ptr[community], 1);
                        Gpu::Atomic::AddNoRet(&n_hood_asymp_ptr[nborhood_i], 1);
                    }
                } else {
                    if (sc) {
                        Gpu::Atomic::AddNoRet(&n_comm_sc_ptr[community], 1);
                        Gpu::Atomic::AddNoRet(&n_hood_sc_ptr[nborhood_i], 1);
                    } else {
                        Gpu::Atomic::AddNoRet(&n_comm_ptr[community], 1);
                        Gpu::Atomic::AddNoRet(&n_hood_ptr[nborhood_i], 1);
                    }
                }
            });
            Gpu::synchronize();

            auto dst_nborhood_ptr = dst_soa.GetIntData(IntIdx::nborhood).data();
            auto prob_ptr = dst_soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
            bool is_dst_travel = dst_is_travel;

            // --- Apply pass: susceptible agents in dst tile ---
            ParallelFor(dst_np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (!isSusceptible(i, dst_ptd, d)) { return; }
                if (dst_ptd.m_idata[IntIdx::withdrawn][i]) { return; }
                if (inHospital(i, dst_ptd)) { return; }
                if (!is_dst_travel && dst_ptd.m_idata[IntIdx::air_travel][i] >= 0) { return; }
                if (dst_ptd.m_idata[IntIdx::random_travel][i] >= 0) { return; }

                Box tbx;
                auto iv = getParticleCell(dst_ptd, i, plo, dxi, domain);
                auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                int community = d_ptr[tidx];
                int nborhood_i = community * max_nborhood + dst_nborhood_ptr[i];
                int age = dst_ptd.m_idata[IntIdx::age_group][i];

                Real p_comm = 1.0_rt - infect * lparm->xmit_comm[age];
                Real p_comm_sc = 1.0_rt - infect * lparm->xmit_comm_SC[age];
                Real p_hood = 1.0_rt - infect * lparm->xmit_hood[age];
                Real p_hood_sc = 1.0_rt - infect * lparm->xmit_hood_SC[age];
                Real pa_comm = 1.0_rt - asymp_inf * infect * lparm->xmit_comm[age];
                Real pa_comm_sc = 1.0_rt - asymp_inf * infect * lparm->xmit_comm_SC[age];
                Real pa_hood = 1.0_rt - asymp_inf * infect * lparm->xmit_hood[age];
                Real pa_hood_sc = 1.0_rt - asymp_inf * infect * lparm->xmit_hood_SC[age];

                prob_ptr[i] *= static_cast<ParticleReal>(
                        std::pow(p_comm, n_comm_ptr[community]) * std::pow(p_comm_sc, n_comm_sc_ptr[community]) *
                        std::pow(pa_comm, n_comm_asymp_ptr[community]) * std::pow(pa_comm_sc, n_comm_sc_asymp_ptr[community]) *
                        std::pow(p_hood, n_hood_ptr[nborhood_i]) * std::pow(p_hood_sc, n_hood_sc_ptr[nborhood_i]) *
                        std::pow(pa_hood, n_hood_asymp_ptr[nborhood_i]) * std::pow(pa_hood_sc, n_hood_sc_asymp_ptr[nborhood_i]));
            });
            Gpu::synchronize();
        }
    }
}

#endif
