/*! @file InteractionModComm.H
 * \brief Community transmission only — xmit_comm
 *
 * Counts infectious agents at the community level and applies transmission to
 * susceptible agents in the same community.  Agents already counted by
 * InteractionModNborhood (same-neighborhood agents) are subtracted from the
 * community count to avoid double-counting.
 *
 * Parameterized for day vs. night — same convention as InteractionModNborhood:
 *   - Day (m_is_day=true):  uses IntIdx::work_nborhood + WorkNborhoodCandidate
 *   - Night (m_is_day=false): uses IntIdx::nborhood + HomeNborhoodCandidate
 */

#ifndef _INTERACTION_MOD_COMM_H_
#define _INTERACTION_MOD_COMM_H_

#include <math.h>

#include "AgentDefinitions.H"
#include "InteractionModNborhood.H" // for WorkNborhoodCandidate / HomeNborhoodCandidate
#include "InteractionModel.H"

using namespace amrex;

/*! \brief Community contact model — applies xmit_comm only */
template <typename PCType, typename PTDType, typename PType>
class InteractionModComm {
  public:
    /*! \param _is_day true for daytime (work_nborhood), false for nighttime (nborhood) */
    InteractionModComm (bool _is_day) : m_is_day(_is_day) {}

    void interactAgents (PCType& agents, MultiFab&) { fastInteractComm(agents); }

    void fastInteractComm(PCType& agents);

  private:
    bool m_is_day;
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModComm<PCType, PTDType, PType>::fastInteractComm (PCType& agents) {
    BL_PROFILE(__func__);
    int n_disease = agents.numDiseases();

    WorkNborhoodCandidate<PTDType> isWorkCandidate;
    HomeNborhoodCandidate<PTDType> isHomeCandidate;
    bool is_day = m_is_day;

    int nborhood_attr = is_day ? IntIdx::work_nborhood : IntIdx::nborhood;

    // nborhood counts are needed to subtract neighborhood agents from the
    // community count (they are handled by InteractionModNborhood).
    Vector<Gpu::DeviceVector<int>> infected_community_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_community_asymp_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_nborhood_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_nborhood_asymp_d(OMP_MAX_THREADS);

    for (int lev = 0; lev < agents.numLevels(); ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            if (np == 0) { continue; }

            auto& soa = ptile.GetStructOfArrays();
            auto nborhood_ptr = soa.GetIntData(nborhood_attr).data();

            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(agents.Geom(lev), mfi.tilebox(), agents.comm_mf[mfi].array());
            auto plo = getCommunityIndex.plo;
            auto dxi = getCommunityIndex.dxi;
            auto domain = getCommunityIndex.domain;
            auto valid_box = getCommunityIndex.valid_box;
            auto bin_size = getCommunityIndex.bin_size;

            int max_communities = getCommunityIndex.max();
            int max_nborhood = agents.getMaxGroup(nborhood_attr) + 1;

            infected_community_d[OMP_THREAD_NUM].resize(max_communities);
            infected_community_asymp_d[OMP_THREAD_NUM].resize(max_communities);
            infected_nborhood_d[OMP_THREAD_NUM].resize(max_communities * max_nborhood);
            infected_nborhood_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_nborhood);

            auto infected_community_d_ptr = infected_community_d[OMP_THREAD_NUM].data();
            auto infected_community_asymp_d_ptr = infected_community_asymp_d[OMP_THREAD_NUM].data();
            auto infected_nborhood_d_ptr = infected_nborhood_d[OMP_THREAD_NUM].data();
            auto infected_nborhood_asymp_d_ptr = infected_nborhood_asymp_d[OMP_THREAD_NUM].data();

            for (int d = 0; d < n_disease; d++) {
                devMemset(infected_community_d_ptr, 0, infected_community_d[OMP_THREAD_NUM].size() * sizeof(int));
                devMemset(infected_community_asymp_d_ptr, 0, infected_community_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));
                devMemset(infected_nborhood_d_ptr, 0, infected_nborhood_d[OMP_THREAD_NUM].size() * sizeof(int));
                devMemset(infected_nborhood_asymp_d_ptr, 0, infected_nborhood_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));

                auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                auto lparm = agents.getDiseaseParameters_d(d);
                auto lparm_h = agents.getDiseaseParameters_h(d);
                Real scale = 1.0_rt;
                Real infect = 1.0_rt - lparm_h->vac_eff;
                auto d_ptr = getCommunityIndex.comm_to_local_index_d.data();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    bool candidate = is_day ? isWorkCandidate(i, ptd) : isHomeCandidate(i, ptd);
                    if (isInfectious(i, ptd, d) && candidate) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];
                        auto nborhood_i = community * max_nborhood + nborhood_ptr[i];
                        if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                            Gpu::Atomic::AddNoRet(&infected_community_asymp_d_ptr[community], 1);
                            Gpu::Atomic::AddNoRet(&infected_nborhood_asymp_d_ptr[nborhood_i], 1);
                        } else {
                            Gpu::Atomic::AddNoRet(&infected_community_d_ptr[community], 1);
                            Gpu::Atomic::AddNoRet(&infected_nborhood_d_ptr[nborhood_i], 1);
                        }
                    }
                });
                Gpu::synchronize();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    bool candidate = is_day ? isWorkCandidate(i, ptd) : isHomeCandidate(i, ptd);
                    if (isSusceptible(i, ptd, d) && candidate) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];
                        auto nborhood_i = community * max_nborhood + nborhood_ptr[i];

                        int num_infected_nborhood = infected_nborhood_d_ptr[nborhood_i];
                        int num_infected_nborhood_asymp = infected_nborhood_asymp_d_ptr[nborhood_i];

                        // Subtract neighborhood agents — they are handled by InteractionModNborhood
                        int num_infected_community = infected_community_d_ptr[community] - num_infected_nborhood;
                        int num_infected_community_asymp =
                                infected_community_asymp_d_ptr[community] - num_infected_nborhood_asymp;

                        AMREX_ALWAYS_ASSERT(num_infected_community >= 0);
                        Real comm_prob = 1.0_prt - infect * lparm->xmit_comm[ptd.m_idata[IntIdx::age_group][i]] * scale;
                        Real comm_prob_asymp = 1.0_prt - lparm->asymp_relative_inf * infect *
                                                                 lparm->xmit_comm[ptd.m_idata[IntIdx::age_group][i]] * scale;
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(comm_prob, num_infected_community));
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(comm_prob_asymp, num_infected_community_asymp));
                    }
                });
                Gpu::synchronize();
            }
        }
    }
}

#endif
