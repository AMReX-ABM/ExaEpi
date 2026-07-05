/*! @file InteractionModNC.H
 * \brief Neighborhood-cluster (NC) transmission only — xmit_nc
 *
 * Counts infectious non-withdrawn agents in each NC cluster and applies
 * transmission to non-withdrawn susceptible agents.  Same-family members who
 * are not withdrawn appear in both the family bucket and the NC cluster bucket;
 * they are subtracted from the NC count to avoid double-counting with
 * InteractionModHH.
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

    // Counts for the family-not-withdrawn are needed to subtract same-family
    // agents from the NC cluster count (they are handled by InteractionModHH).
    Vector<Gpu::DeviceVector<int>> infected_family_not_withdrawn_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_family_not_withdrawn_asymp_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_nc_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_nc_asymp_d(OMP_MAX_THREADS);

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
            auto family_ptr = soa.GetIntData(IntIdx::family).data();
            auto hh_cluster_ptr = soa.GetIntData(IntIdx::hh_cluster).data();

            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(agents.Geom(lev), mfi.tilebox(), agents.comm_mf[mfi].array());
            auto plo = getCommunityIndex.plo;
            auto dxi = getCommunityIndex.dxi;
            auto domain = getCommunityIndex.domain;
            auto valid_box = getCommunityIndex.valid_box;
            auto bin_size = getCommunityIndex.bin_size;

            int max_communities = getCommunityIndex.max();
            int max_family = agents.getMaxGroup(IntIdx::family) + 1;
            int max_hcs = agents.getMaxGroup(IntIdx::hh_cluster) + 1;

            infected_family_not_withdrawn_d[OMP_THREAD_NUM].resize(max_communities * max_family);
            infected_family_not_withdrawn_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_family);
            infected_nc_d[OMP_THREAD_NUM].resize(max_communities * max_hcs);
            infected_nc_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_hcs);

            auto infected_family_not_withdrawn_d_ptr = infected_family_not_withdrawn_d[OMP_THREAD_NUM].data();
            auto infected_family_not_withdrawn_asymp_d_ptr = infected_family_not_withdrawn_asymp_d[OMP_THREAD_NUM].data();
            auto infected_nc_d_ptr = infected_nc_d[OMP_THREAD_NUM].data();
            auto infected_nc_asymp_d_ptr = infected_nc_asymp_d[OMP_THREAD_NUM].data();

            for (int d = 0; d < n_disease; d++) {
                for (auto adults : {true, false}) {
                    {
                        BL_PROFILE("fastInteractNC::fill_device_vectors");
                        devMemset(infected_family_not_withdrawn_d_ptr, 0,
                                  infected_family_not_withdrawn_d[OMP_THREAD_NUM].size() * sizeof(int));
                        devMemset(infected_family_not_withdrawn_asymp_d_ptr, 0,
                                  infected_family_not_withdrawn_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));
                        devMemset(infected_nc_d_ptr, 0, infected_nc_d[OMP_THREAD_NUM].size() * sizeof(int));
                        devMemset(infected_nc_asymp_d_ptr, 0, infected_nc_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));
                    }
                    auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                    auto lparm = agents.getDiseaseParameters_d(d);
                    auto lparm_h = agents.getDiseaseParameters_h(d);
                    Real scale = 1.0_rt;
                    Real infect = (1.0_rt - lparm_h->vac_eff);
                    auto d_ptr = getCommunityIndex.comm_to_local_index_d.data();

                    // Count non-withdrawn infectious agents in family and NC cluster buckets.
                    // family_not_withdrawn is subtracted from nc to avoid double-counting.
                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isInfectious(i, ptd, d) && isHomeCandidate(i, ptd) && (isAnAdult(i, ptd) == adults) &&
                            !ptd.m_idata[IntIdx::withdrawn][i]) {
                            Box tbx;
                            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                            auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                            auto community = d_ptr[tidx];
                            int family_i = community * max_family + family_ptr[i];
                            int cluster_i = community * max_hcs + hh_cluster_ptr[i];
                            if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                                Gpu::Atomic::AddNoRet(&infected_family_not_withdrawn_asymp_d_ptr[family_i], 1);
                                Gpu::Atomic::AddNoRet(&infected_nc_asymp_d_ptr[cluster_i], 1);
                            } else {
                                Gpu::Atomic::AddNoRet(&infected_family_not_withdrawn_d_ptr[family_i], 1);
                                Gpu::Atomic::AddNoRet(&infected_nc_d_ptr[cluster_i], 1);
                            }
                        }
                    });
                    Gpu::synchronize();

                    // Apply NC transmission to non-withdrawn susceptible home candidates.
                    // Subtract same-family non-withdrawn agents from NC count to avoid double-counting with HH.
                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isSusceptible(i, ptd, d) && isHomeCandidate(i, ptd) && !ptd.m_idata[IntIdx::withdrawn][i]) {
                            Real xmit_nc_prob = adults ? lparm->xmit_nc_adult[ptd.m_idata[IntIdx::age_group][i]]
                                                       : lparm->xmit_nc_child[ptd.m_idata[IntIdx::age_group][i]];
                            Box tbx;
                            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                            auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                            auto community = d_ptr[tidx];
                            int family_i = community * max_family + family_ptr[i];
                            int cluster_i = community * max_hcs + hh_cluster_ptr[i];

                            int num_infected_family_not_withdrawn = infected_family_not_withdrawn_d_ptr[family_i];
                            int num_infected_nc = infected_nc_d_ptr[cluster_i] - num_infected_family_not_withdrawn;
                            AMREX_ALWAYS_ASSERT(num_infected_nc >= 0);
                            Real nc_prob = 1.0_rt - infect * xmit_nc_prob * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(nc_prob, num_infected_nc));

                            int num_infected_family_not_withdrawn_asymp = infected_family_not_withdrawn_asymp_d_ptr[family_i];
                            int num_infected_nc_asymp =
                                    infected_nc_asymp_d_ptr[cluster_i] - num_infected_family_not_withdrawn_asymp;
                            AMREX_ALWAYS_ASSERT(num_infected_nc_asymp >= 0);
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
