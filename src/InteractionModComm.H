/*! @file InteractionModComm.H
 * \brief Community transmission only — xmit_comm
 *
 * Counts infectious agents at the community level and applies transmission to
 * susceptible agents in the same community. Same-neighborhood agents are NOT
 * excluded from this count: they are counted here AND separately by
 * InteractionModNborhood, matching Epicast's venue model, where venues are not
 * mutually exclusive -- two agents sharing a neighborhood interact via both the
 * neighborhood venue and the community venue concurrently, not one or the other.
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

    Vector<Gpu::DeviceVector<int>> infected_community_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_community_asymp_d(OMP_MAX_THREADS);

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

            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(agents.Geom(lev), mfi.tilebox(), agents.comm_mf[mfi].array());
            auto density_scale_arr = (is_day ? agents.comm_density_scale_work : agents.comm_density_scale)[mfi].const_array();
            auto plo = getCommunityIndex.plo;
            auto dxi = getCommunityIndex.dxi;
            auto domain = getCommunityIndex.domain;
            auto valid_box = getCommunityIndex.valid_box;
            auto bin_size = getCommunityIndex.bin_size;

            int max_communities = getCommunityIndex.max();

            infected_community_d[OMP_THREAD_NUM].resize(max_communities);
            infected_community_asymp_d[OMP_THREAD_NUM].resize(max_communities);

            auto infected_community_d_ptr = infected_community_d[OMP_THREAD_NUM].data();
            auto infected_community_asymp_d_ptr = infected_community_asymp_d[OMP_THREAD_NUM].data();

            for (int d = 0; d < n_disease; d++) {
                devMemset(infected_community_d_ptr, 0, infected_community_d[OMP_THREAD_NUM].size() * sizeof(int));
                devMemset(infected_community_asymp_d_ptr, 0, infected_community_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));

                auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                auto lparm = agents.getDiseaseParameters_d(d);
                auto lparm_h = agents.getDiseaseParameters_h(d);
                Real infect = 1.0_rt - lparm_h->vac_eff;
                auto d_ptr = getCommunityIndex.comm_to_local_index_d.data();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    bool candidate = is_day ? isWorkCandidate(i, ptd) : isHomeCandidate(i, ptd);
                    if (isInfectious(i, ptd, d) && candidate) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];
                        if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                            Gpu::Atomic::AddNoRet(&infected_community_asymp_d_ptr[community], 1);
                        } else {
                            Gpu::Atomic::AddNoRet(&infected_community_d_ptr[community], 1);
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
                        Real scale = density_scale_arr(iv[0], iv[1], 0);

                        int num_infected_community = infected_community_d_ptr[community];
                        int num_infected_community_asymp = infected_community_asymp_d_ptr[community];

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
