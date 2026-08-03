/*! @file InteractionModNborhood.H
 * \brief Neighborhood transmission only — xmit_hood
 *
 * Counts infectious agents in each neighborhood and applies transmission to
 * susceptible agents in the same neighborhood.
 *
 * Parameterized for day vs. night:
 *   - Day (m_is_day=true):  uses IntIdx::work_nborhood; requires random_travel < 0
 *   - Night (m_is_day=false): uses IntIdx::nborhood; allows random travelers
 */

#ifndef _INTERACTION_MOD_NBORHOOD_H_
#define _INTERACTION_MOD_NBORHOOD_H_

#include <math.h>

#include "AgentDefinitions.H"
#include "InteractionModel.H"

using namespace amrex;

/*! \brief Candidate predicate for daytime neighborhood interactions */
template <typename PTDType>
struct WorkNborhoodCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator()(const int idx, const PTDType& ptd) const noexcept {
        return !inHospital(idx, ptd) && !ptd.m_idata[IntIdx::withdrawn][idx] && ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};

/*! \brief Candidate predicate for nighttime neighborhood interactions */
template <typename PTDType>
struct HomeNborhoodCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator()(const int idx, const PTDType& ptd) const noexcept {
        // random travelers are allowed to interact at night (they are sleeping somewhere)
        return !inHospital(idx, ptd) && !ptd.m_idata[IntIdx::withdrawn][idx];
    }
};

/*! \brief Neighborhood contact model — applies xmit_hood only */
template <typename PCType, typename PTDType, typename PType>
class InteractionModNborhood {
  public:
    /*! \param _is_day true for daytime (uses work_nborhood + WorkNborhoodCandidate),
     *                 false for nighttime (uses nborhood + HomeNborhoodCandidate) */
    InteractionModNborhood (bool _is_day) : m_is_day(_is_day) {}

    void interactAgents (PCType& agents, MultiFab&) { fastInteractNborhood(agents); }

    void fastInteractNborhood(PCType& agents);

  private:
    bool m_is_day;
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModNborhood<PCType, PTDType, PType>::fastInteractNborhood (PCType& agents) {
    BL_PROFILE(__func__);
    int n_disease = agents.numDiseases();

    WorkNborhoodCandidate<PTDType> isWorkCandidate;
    HomeNborhoodCandidate<PTDType> isHomeCandidate;
    bool is_day = m_is_day;

    // nborhood attribute index depends on day vs. night
    int nborhood_attr = is_day ? IntIdx::work_nborhood : IntIdx::nborhood;

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
            auto density_scale_arr = (is_day ? agents.comm_density_scale_work : agents.comm_density_scale)[mfi].const_array();
            auto plo = getCommunityIndex.plo;
            auto dxi = getCommunityIndex.dxi;
            auto domain = getCommunityIndex.domain;
            auto valid_box = getCommunityIndex.valid_box;
            auto bin_size = getCommunityIndex.bin_size;

            int max_communities = getCommunityIndex.max();
            int max_nborhood = agents.getMaxGroup(nborhood_attr) + 1;

            infected_nborhood_d[OMP_THREAD_NUM].resize(max_communities * max_nborhood);
            infected_nborhood_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_nborhood);

            auto infected_nborhood_d_ptr = infected_nborhood_d[OMP_THREAD_NUM].data();
            auto infected_nborhood_asymp_d_ptr = infected_nborhood_asymp_d[OMP_THREAD_NUM].data();

            for (int d = 0; d < n_disease; d++) {
                devMemset(infected_nborhood_d_ptr, 0, infected_nborhood_d[OMP_THREAD_NUM].size() * sizeof(int));
                devMemset(infected_nborhood_asymp_d_ptr, 0, infected_nborhood_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));

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
                        auto nborhood_i = community * max_nborhood + nborhood_ptr[i];
                        if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                            Gpu::Atomic::AddNoRet(&infected_nborhood_asymp_d_ptr[nborhood_i], 1);
                        } else {
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
                        Real scale = density_scale_arr(iv[0], iv[1], 0);

                        int num_infected_nborhood = infected_nborhood_d_ptr[nborhood_i];
                        int num_infected_nborhood_asymp = infected_nborhood_asymp_d_ptr[nborhood_i];

                        Real nborhood_prob = 1.0_prt - infect * lparm->xmit_hood[ptd.m_idata[IntIdx::age_group][i]] * scale;
                        Real nborhood_prob_asymp = 1.0_prt - lparm->asymp_relative_inf * infect *
                                                                     lparm->xmit_hood[ptd.m_idata[IntIdx::age_group][i]] * scale;
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(nborhood_prob, num_infected_nborhood));
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(nborhood_prob_asymp, num_infected_nborhood_asymp));
                    }
                });
                Gpu::synchronize();
            }
        }
    }
}

#endif
