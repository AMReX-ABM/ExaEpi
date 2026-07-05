/*! @file InteractionModWork.H
 * \brief Contains the class describing agent interactions at work
 */

#ifndef _INTERACTION_MOD_WORK_H_
#define _INTERACTION_MOD_WORK_H_

#include "AgentDefinitions.H"
#include "InteractionModel.H"

using namespace amrex;

template <typename PTDType>
struct WorkCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator()(const int idx, const PTDType& ptd) const noexcept {
        return !inHospital(idx, ptd) && ptd.m_idata[IntIdx::work_i][idx] >= 0 && ptd.m_idata[IntIdx::workgroup][idx] > 0 &&
               !ptd.m_idata[IntIdx::withdrawn][idx] && ptd.m_idata[IntIdx::air_travel][idx] < 0 &&
               ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};

/*! \brief Class describing agent interactions at work */
template <typename PCType, typename PTDType, typename PType>
class InteractionModWork {
  public:
    /*! \brief Simulate agent interaction at work */
    void interactAgents (PCType& agents, MultiFab&) { fastInteractWork(agents); }

    void fastInteractWork(PCType& agents);
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModWork<PCType, PTDType, PType>::fastInteractWork (PCType& agents) {
    BL_PROFILE(__func__);
    int n_disease = agents.numDiseases();

    WorkCandidate<PTDType> isCandidate;

    Vector<Gpu::DeviceVector<int>> infected_workgroup_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_workgroup_asymp_d(OMP_MAX_THREADS);

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
            auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
            auto naics_ptr = soa.GetIntData(IntIdx::naics).data();

            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(agents.Geom(lev), mfi.tilebox(), agents.comm_mf[mfi].array());
            auto plo = getCommunityIndex.plo;
            auto dxi = getCommunityIndex.dxi;
            auto domain = getCommunityIndex.domain;
            auto valid_box = getCommunityIndex.valid_box;
            auto bin_size = getCommunityIndex.bin_size;
            auto comm_to_local_index_d = getCommunityIndex.comm_to_local_index_d;

            int max_communities = getCommunityIndex.max();
            int max_workgroup = agents.getMaxGroup(IntIdx::workgroup) + 1;
            int max_naics = agents.getMaxGroup(IntIdx::naics) + 1;

            infected_workgroup_d[OMP_THREAD_NUM].resize(max_communities * max_workgroup * max_naics);
            auto infected_workgroup_d_ptr = infected_workgroup_d[OMP_THREAD_NUM].data();

            infected_workgroup_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_workgroup * max_naics);
            auto infected_workgroup_asymp_d_ptr = infected_workgroup_asymp_d[OMP_THREAD_NUM].data();

            // Print() << mfi.tilebox() << " np " << np << " max communities " << max_communities
            //         << " max workgroup " << max_workgroup << " max naics " << max_naics
            //         << " workgroup vector size " << max_communities * max_workgroup * max_naics
            //         << "\n";
            for (int d = 0; d < n_disease; d++) {
                devMemset(infected_workgroup_d_ptr, 0, infected_workgroup_d[OMP_THREAD_NUM].size() * sizeof(int));
                devMemset(infected_workgroup_asymp_d_ptr, 0, infected_workgroup_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));

                auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                auto lparm = agents.getDiseaseParameters_d(d);
                auto lparm_h = agents.getDiseaseParameters_h(d);
                Real scale = 1.0_rt; // TODO this should vary based on cell
                Real infect = 1.0_rt - lparm_h->vac_eff;
                auto d_ptr = getCommunityIndex.comm_to_local_index_d.data();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isInfectious(i, ptd, d) && isCandidate(i, ptd)) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];
                        int wgroup_i = (community * max_workgroup + workgroup_ptr[i]) * max_naics + naics_ptr[i];
                        if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                            Gpu::Atomic::AddNoRet(&infected_workgroup_asymp_d_ptr[wgroup_i], 1);
                        } else {
                            Gpu::Atomic::AddNoRet(&infected_workgroup_d_ptr[wgroup_i], 1);
                        }
                    }
                });
                Gpu::synchronize();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];
                        int wgroup_i = (community * max_workgroup + workgroup_ptr[i]) * max_naics + naics_ptr[i];
                        int num_infected_workgroup = infected_workgroup_d_ptr[wgroup_i];
                        Real workgroup_prob = 1.0_prt - infect * lparm->xmit_work * scale;
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(workgroup_prob, num_infected_workgroup));
                        int num_infected_workgroup_asymp = infected_workgroup_asymp_d_ptr[wgroup_i];
                        Real workgroup_prob_asymp = 1.0_prt - lparm->asymp_relative_inf * infect * lparm->xmit_work * scale;
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(workgroup_prob_asymp, num_infected_workgroup_asymp));
                    }
                });
                Gpu::synchronize();
            }
        }
    }
}

#endif
