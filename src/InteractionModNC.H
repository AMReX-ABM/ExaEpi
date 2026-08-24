/*! @file InteractionModNC.H
 * \brief Neighborhood-cluster (NC) transmission only — xmit_nc
 *
 * Counts infectious non-withdrawn agents in each NC cluster and applies
 * transmission to non-withdrawn susceptible agents. Same-family members are
 * NOT excluded from the NC count: Epicast's venues are explicitly not
 * mutually exclusive (paper: "if two agents share a household, they will
 * necessarily interact within the household, household-cluster, neighborhood,
 * and community venues"), so a family member is exposed via both xmit_hh
 * (InteractionModHH.H) and xmit_nc concurrently, matching InteractionModComm.H's
 * treatment of the community/neighborhood pair.
 */

#ifndef _INTERACTION_MOD_NC_H_
#define _INTERACTION_MOD_NC_H_

#include "AgentDefinitions.H"
#include "DiseaseParm.H"
#include "InteractionModHH.H" // for HomeCandidate
#include "InteractionModel.H"

using namespace amrex;

/*! \brief NC cluster contact model — applies xmit_nc only */
template <typename PCType, typename PTDType, typename PType>
class InteractionModNC {
  public:
    void interactAgents (PCType& agents, MultiFab&) { fastInteractNC(agents); }

    void fastInteractNC(PCType& agents);
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModNC<PCType, PTDType, PType>::fastInteractNC (PCType& agents) {
    BL_PROFILE(__func__);
    int n_disease = agents.numDiseases();

    HomeCandidate<PTDType> isHomeCandidate;

    Vector<Gpu::DeviceVector<int>> infected_nc_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_nc_asymp_d(OMP_MAX_THREADS);

    for (int lev = 0; lev < agents.numLevels(); ++lev) {
        // getMaxGroup performs an MPI collective on first use per attribute -- must be called
        // once, single-threaded, here, never from inside the omp-parallel region below.
        int max_hcs = agents.getMaxGroup(IntIdx::hh_cluster) + 1;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            if (np == 0) { continue; }

            auto& soa = ptile.GetStructOfArrays();
            auto hh_cluster_ptr = soa.GetIntData(IntIdx::hh_cluster).data();

            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(agents.Geom(lev), mfi.tilebox(), agents.comm_mf[mfi].array());
            auto plo = getCommunityIndex.plo;
            auto dxi = getCommunityIndex.dxi;
            auto domain = getCommunityIndex.domain;
            auto valid_box = getCommunityIndex.valid_box;
            auto bin_size = getCommunityIndex.bin_size;

            int max_communities = getCommunityIndex.max();

            infected_nc_d[OMP_THREAD_NUM].resize(max_communities * max_hcs);
            infected_nc_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_hcs);

            auto infected_nc_d_ptr = infected_nc_d[OMP_THREAD_NUM].data();
            auto infected_nc_asymp_d_ptr = infected_nc_asymp_d[OMP_THREAD_NUM].data();

            for (int d = 0; d < n_disease; d++) {
                for (auto adults : {true, false}) {
                    {
                        BL_PROFILE("fastInteractNC::fill_device_vectors");
                        devMemset(infected_nc_d_ptr, 0, infected_nc_d[OMP_THREAD_NUM].size() * sizeof(int));
                        devMemset(infected_nc_asymp_d_ptr, 0, infected_nc_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));
                    }
                    auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                    auto lparm = agents.getDiseaseParameters_d(d);
                    auto lparm_h = agents.getDiseaseParameters_h(d);
                    Real scale = 1.0_rt;
                    Real infect = (1.0_rt - lparm_h->vac_eff);
                    auto d_ptr = getCommunityIndex.comm_to_local_index_d.data();

                    // Count non-withdrawn infectious agents in each NC cluster bucket.
                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isInfectious(i, ptd, d) && isHomeCandidate(i, ptd) && (isAnAdult(i, ptd) == adults) &&
                            !ptd.m_idata[IntIdx::withdrawn][i]) {
                            Box tbx;
                            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                            auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                            auto community = d_ptr[tidx];
                            int cluster_i = community * max_hcs + hh_cluster_ptr[i];
                            if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                                Gpu::Atomic::AddNoRet(&infected_nc_asymp_d_ptr[cluster_i], 1);
                            } else {
                                Gpu::Atomic::AddNoRet(&infected_nc_d_ptr[cluster_i], 1);
                            }
                        }
                    });
                    Gpu::synchronize();

                    // Apply NC transmission to non-withdrawn susceptible home candidates.
                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isSusceptible(i, ptd, d) && isHomeCandidate(i, ptd) && !ptd.m_idata[IntIdx::withdrawn][i]) {
                            Real xmit_nc_prob = adults ? lparm->xmit_nc_adult[ptd.m_idata[IntIdx::age_group][i]]
                                                       : lparm->xmit_nc_child[ptd.m_idata[IntIdx::age_group][i]];
                            Box tbx;
                            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                            auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                            auto community = d_ptr[tidx];
                            int cluster_i = community * max_hcs + hh_cluster_ptr[i];

                            int num_infected_nc = infected_nc_d_ptr[cluster_i];
                            Real nc_prob = 1.0_rt - infect * xmit_nc_prob * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(nc_prob, num_infected_nc));

                            int num_infected_nc_asymp = infected_nc_asymp_d_ptr[cluster_i];
                            Real nc_prob_asymp = 1.0_rt - lparm->asymp_relative_inf * infect * xmit_nc_prob * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(nc_prob_asymp, num_infected_nc_asymp));
                        }
                    });
                    Gpu::synchronize();
                }
            }
        }
    }
}

#endif
