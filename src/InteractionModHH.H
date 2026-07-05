/*! @file InteractionModHH.H
 * \brief Household (family) transmission only — xmit_hh
 *
 * Counts infectious agents within each family and applies transmission to
 * susceptible agents in the same family.  Withdrawn infectious agents still
 * transmit within the household; withdrawal does not protect susceptible
 * household members.
 */

#ifndef _INTERACTION_MOD_HH_H_
#define _INTERACTION_MOD_HH_H_

#include "AgentDefinitions.H"
#include "DiseaseParm.H"
#include "InteractionModel.H"

using namespace amrex;

/*! \brief Predicate: agent is eligible for home interactions */
template <typename PTDType>
struct HomeCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator()(const int idx, const PTDType& ptd) const noexcept {
        // FIXME: like in Epicast, hospitalized can transmit at home —
        // assumes family visits the sick person
        return ptd.m_idata[IntIdx::random_travel][idx] < 0 && ptd.m_idata[IntIdx::air_travel][idx] < 0;
    }
};

/*! \brief Household (family) contact model — applies xmit_hh only */
template <typename PCType, typename PTDType, typename PType>
class InteractionModHH : public InteractionModel<PCType, PTDType, PType> {
  public:
    InteractionModHH (bool _fast_bin) : InteractionModel<PCType, PTDType, PType>(_fast_bin) {}
    virtual ~InteractionModHH() = default;

    virtual void interactAgents (PCType& agents, MultiFab&) override { fastInteractHH(agents); }

    void fastInteractHH(PCType& agents);
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModHH<PCType, PTDType, PType>::fastInteractHH (PCType& agents) {
    BL_PROFILE(__func__);
    int n_disease = agents.numDiseases();

    HomeCandidate<PTDType> isHomeCandidate;

    Vector<Gpu::DeviceVector<int>> infected_family_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_family_asymp_d(OMP_MAX_THREADS);

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

            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(agents.Geom(lev), mfi.tilebox(), agents.comm_mf[mfi].array());
            auto plo = getCommunityIndex.plo;
            auto dxi = getCommunityIndex.dxi;
            auto domain = getCommunityIndex.domain;
            auto valid_box = getCommunityIndex.valid_box;
            auto bin_size = getCommunityIndex.bin_size;

            int max_communities = getCommunityIndex.max();
            int max_family = agents.getMaxGroup(IntIdx::family) + 1;

            infected_family_d[OMP_THREAD_NUM].resize(max_communities * max_family);
            infected_family_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_family);

            auto infected_family_d_ptr = infected_family_d[OMP_THREAD_NUM].data();
            auto infected_family_asymp_d_ptr = infected_family_asymp_d[OMP_THREAD_NUM].data();

            for (int d = 0; d < n_disease; d++) {
                for (auto adults : {true, false}) {
                    {
                        BL_PROFILE("fastInteractHH::fill_device_vectors");
                        devMemset(infected_family_d_ptr, 0, infected_family_d[OMP_THREAD_NUM].size() * sizeof(int));
                        devMemset(infected_family_asymp_d_ptr, 0, infected_family_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));
                    }
                    auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                    auto lparm = agents.getDiseaseParameters_d(d);
                    auto lparm_h = agents.getDiseaseParameters_h(d);
                    Real scale = 1.0_rt;
                    Real infect = (1.0_rt - lparm_h->vac_eff);
                    auto d_ptr = getCommunityIndex.comm_to_local_index_d.data();

                    // Count infectious family members (withdrawn members still transmit within HH)
                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isInfectious(i, ptd, d) && isHomeCandidate(i, ptd) && (isAnAdult(i, ptd) == adults)) {
                            Box tbx;
                            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                            auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                            auto community = d_ptr[tidx];
                            int family_i = community * max_family + family_ptr[i];
                            if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                                Gpu::Atomic::AddNoRet(&infected_family_asymp_d_ptr[family_i], 1);
                            } else {
                                Gpu::Atomic::AddNoRet(&infected_family_d_ptr[family_i], 1);
                            }
                        }
                    });
                    Gpu::synchronize();

                    // Apply household transmission to susceptible home candidates
                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isSusceptible(i, ptd, d) && isHomeCandidate(i, ptd)) {
                            Real xmit_family_prob = adults ? lparm->xmit_hh_adult[ptd.m_idata[IntIdx::age_group][i]]
                                                           : lparm->xmit_hh_child[ptd.m_idata[IntIdx::age_group][i]];
                            Box tbx;
                            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                            auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                            auto community = d_ptr[tidx];
                            int family_i = community * max_family + family_ptr[i];
                            int num_infected_family = infected_family_d_ptr[family_i];
                            int num_infected_family_asymp = infected_family_asymp_d_ptr[family_i];
                            Real family_prob = 1.0_rt - infect * xmit_family_prob * scale;
                            Real family_prob_asymp = 1.0_rt - lparm->asymp_relative_inf * infect * xmit_family_prob * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(family_prob, num_infected_family));
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(family_prob_asymp, num_infected_family_asymp));
                        }
                    });
                    Gpu::synchronize();
                }
            }
        }
    }
}

#endif
