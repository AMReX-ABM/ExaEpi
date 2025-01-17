/*! @file InteractionModWork.H
 * \brief Contains the class describing agent interactions at work
 */

#ifndef _INTERACTION_MOD_WORK_H_
#define _INTERACTION_MOD_WORK_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;


#ifndef FAST_INTERACTIONS
template <typename ParticleType>
struct GetWorkerBin
{
    GpuArray<Real,AMREX_SPACEDIM> plo;
    GpuArray<Real,AMREX_SPACEDIM> dxi;
    Box domain;
    IntVect bin_size;
    Box box;
    int max_wg;

    AMREX_GPU_HOST_DEVICE
    unsigned int operator() (const ParticleType& p) const noexcept
    {
        Box tbx;
        auto iv = getParticleCell(p, plo, dxi, domain);
        auto tid = getTileIndex(iv, box, true, bin_size, tbx);
        auto wg = p.idata(0);
        return static_cast<unsigned int>(tid * max_wg + wg);
    }
};

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at work. */
template <typename PTDType>
struct BinaryInteractionWork {
    AMREX_GPU_HOST_DEVICE
    ParticleReal operator() (const int infectious_i,
                             const int susceptible_i,
                             const PTDType& a_ptd,
                             const DiseaseParm* const a_lparm,
                             const Real a_work_scale) const noexcept {
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::workgroup][infectious_i] == a_ptd.m_idata[IntIdx::workgroup][susceptible_i]);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::work_i][infectious_i] == a_ptd.m_idata[IntIdx::work_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::work_j][infectious_i] == a_ptd.m_idata[IntIdx::work_j][susceptible_i]);
        //infect *= i_mask;
        //infect *= j_mask;

        // It's already been assured that (a_ptd.m_idata[IntIdx::workgroup][infectious_i] == a_ptd.m_idata[IntIdx::workgroup][susceptible_i]);
        auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
        auto worksubgroup_ptr = a_ptd.m_idata[IntIdx::worksubgroup];

        if (worksubgroup_ptr[infectious_i] == worksubgroup_ptr[susceptible_i]) {
            // Same subgroup, transmission depends on age group
            if (age_group[infectious_i] == AgeGroups::a18to29) {
                return a_lparm->xmit_work_a18to29[susceptible_i] * a_work_scale;
            } else if (age_group[infectious_i] == AgeGroups::a30to49) {
                return a_lparm->xmit_work_a30to49[susceptible_i] * a_work_scale;
            } else if (age_group[infectious_i] == AgeGroups::a50to64) {
                return a_lparm->xmit_work_a50to64[susceptible_i] * a_work_scale;
            }
        } else { // Different team within the same workgroup/company
            if (age_group[infectious_i] == AgeGroups::a18to29) {
                return a_lparm->xmit_workdiff_a18to29[susceptible_i] * a_work_scale;
            } else if (age_group[infectious_i] == AgeGroups::a30to49) {
                return a_lparm->xmit_workdiff_a30to49[susceptible_i] * a_work_scale;
            } else if (age_group[infectious_i] == AgeGroups::a50to64) {
                return a_lparm->xmit_workdiff_a50to64[susceptible_i] * a_work_scale;
            }
        }

        return 0.;

    }
};
#endif

template <typename PTDType>
struct WorkCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !inHospital(idx, ptd) &&
                ptd.m_idata[IntIdx::work_i][idx] >= 0 &&
                ptd.m_idata[IntIdx::workgroup][idx] > 0 &&
                !ptd.m_idata[IntIdx::withdrawn][idx] &&
                ptd.m_idata[IntIdx::air_travel][idx] < 0 &&
                ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};


/*! \brief Class describing agent interactions at work */
template <typename PCType, typename PTDType, typename PType>
class InteractionModWork : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModWork (bool _fast_bin) : InteractionModel<PCType, PTDType, PType>(_fast_bin) {}

        /*! \brief default destructor */
        virtual ~InteractionModWork () = default;

        /*! \brief Simulate agent interaction at work */
        virtual void interactAgents (PCType& agents, MultiFab&) override {
#ifdef FAST_INTERACTIONS
            fastInteractWork(agents);
#else
            interactAgentsImpl<InteractionModWork<PCType, PTDType, PType>, PCType, PTDType,
                               WorkCandidate<PTDType>,
                               BinaryInteractionWork<PTDType>>(*this, agents, IntIdx::workgroup);
#endif
        }

        void fastInteractWork (PCType &agents);

};

template <typename PCType, typename PTDType, typename PType>
void InteractionModWork<PCType, PTDType, PType>::fastInteractWork (PCType& agents) {
    BL_PROFILE(__func__);
    int n_disease = agents.numDiseases();

    WorkCandidate<PTDType> isCandidate;

    Vector<Gpu::DeviceVector<int>> infected_workgroup_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_worksubgroup_d(OMP_MAX_THREADS);

    for (int lev = 0; lev < agents.numLevels(); ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            if (np == 0) continue;

            auto& soa = ptile.GetStructOfArrays();
            auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
            auto worksubgroup_ptr = soa.GetIntData(IntIdx::worksubgroup).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto naics_ptr = soa.GetIntData(IntIdx::naics).data();

            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(agents.Geom(lev), mfi.tilebox(), agents.comm_mf[mfi].array());

            int max_communities = getCommunityIndex.max();
            int max_workgroup = agents.getMaxGroup(IntIdx::workgroup) + 1;
            int max_subgroup = agents.getMaxGroup(IntIdx::worksubgroup) + 1; // Ensure this matches logic
            int max_naics = agents.getMaxGroup(IntIdx::naics) + 1;

            infected_workgroup_d[OMP_THREAD_NUM].resize(max_communities * max_workgroup * max_naics);
            auto infected_workgroup_d_ptr = infected_workgroup_d[OMP_THREAD_NUM].data();
            infected_worksubgroup_d[OMP_THREAD_NUM].resize(max_communities * max_workgroup * max_naics * max_subgroup);
            auto infected_worksubgroup_d_ptr = infected_worksubgroup_d[OMP_THREAD_NUM].data();

            //Print() << mfi.tilebox() << " np " << np << " max communities " << max_communities
            //        << " max workgroup " << max_workgroup << " max naics " << max_naics
            //        << " workgroup vector size " << max_communities * max_workgroup * max_naics
            //        << "\n";

            for (int d = 0; d < n_disease; d++) {
                // calculate separately for different infected age_group, since they have different transmission rates
                std::vector<int> age_groups = {AgeGroups::a18to29, AgeGroups::a30to49, AgeGroups::a50to64};
                for (auto age_group_type : age_groups) {
                    {BL_PROFILE("fill_modwork_vectors");
                    dev_memset(infected_workgroup_d_ptr, 0, infected_workgroup_d[OMP_THREAD_NUM].size() * sizeof(int));
                    dev_memset(infected_worksubgroup_d_ptr, 0, infected_worksubgroup_d[OMP_THREAD_NUM].size() * sizeof(int));
                    }
                    auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                    auto lparm = agents.getDiseaseParameters_d(d);
                    auto lparm_h = agents.getDiseaseParameters_h(d);
                    Real scale = 1.0_rt;  // TODO: Adjust based on cell
                    Real infect = 1.0_rt - lparm_h->vac_eff;
                    // loop to count infectious agents in each group
                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isInfectious(i, ptd, d) && isCandidate(i, ptd) && (age_group_type == age_group_ptr[i])) {
                            auto community = getCommunityIndex(ptd, i);
                            AMREX_ALWAYS_ASSERT(community <= max_communities);
                            int wgroup_i = (community * max_workgroup + workgroup_ptr[i]) * max_naics + naics_ptr[i];
                            Gpu::Atomic::AddNoRet(&infected_workgroup_d_ptr[wgroup_i], 1);
                            int wsubgroup_i = ((community * max_workgroup + workgroup_ptr[i]) * max_naics + naics_ptr[i]) * max_subgroup + worksubgroup_ptr[i];
                            Gpu::Atomic::AddNoRet(&infected_worksubgroup_d_ptr[wsubgroup_i], 1);
                        }
                    });
                    Gpu::synchronize();
                    // Loop to compute infection probability for each susceptible agent.
                    // For each agent, find count of infectious agents in each group and use that as the exponent to compute the
                    // infection probability. In cases where there is an overlap (e.g. infectious agents in same workgroup
                    // and worksubgroup, adjust the infected counts to avoid double-counting the overlap.
                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                            Real xmit_work_samegroup_prob = 0;
                            Real xmit_work_diffgroup_prob = 0;
                            int age_group = ptd.m_idata[IntIdx::age_group][i];
                            AMREX_ALWAYS_ASSERT(age_group >= AgeGroups::a18to29 && age_group <= AgeGroups::a50to64);
                            if (age_group_type == AgeGroups::a18to29) {
                                xmit_work_samegroup_prob = lparm->xmit_work_a18to29[age_group];
                                xmit_work_diffgroup_prob = lparm->xmit_workdiff_a18to29[age_group];
                            } else if (age_group_type == AgeGroups::a30to49) {
                                xmit_work_samegroup_prob = lparm->xmit_work_a30to49[age_group];
                                xmit_work_diffgroup_prob = lparm->xmit_workdiff_a30to49[age_group];
                            } else if (age_group_type == AgeGroups::a50to64) {
                                xmit_work_samegroup_prob = lparm->xmit_work_a50to64[age_group];
                                xmit_work_diffgroup_prob = lparm->xmit_workdiff_a50to64[age_group];
                            }
                            auto community = getCommunityIndex(ptd, i);
                            AMREX_ALWAYS_ASSERT(community <= max_communities);
                            int wsubgroup_i = ((community * max_workgroup + workgroup_ptr[i]) * max_naics + naics_ptr[i]) * max_subgroup + worksubgroup_ptr[i];
                            int num_infected_worksubgroup = infected_worksubgroup_d_ptr[wsubgroup_i];
                            Real worksamegroup_prob = 1.0_prt - infect * xmit_work_samegroup_prob * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(worksamegroup_prob, num_infected_worksubgroup));

                            int wgroup_i = (community * max_workgroup + workgroup_ptr[i]) * max_naics + naics_ptr[i];
                            int num_infected_workgroup = infected_workgroup_d_ptr[wgroup_i];
                            Real workdiffgroup_prob = 1.0_prt - infect * xmit_work_diffgroup_prob * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(workdiffgroup_prob, num_infected_workgroup - num_infected_worksubgroup));

                            AMREX_ALWAYS_ASSERT(num_infected_workgroup >= num_infected_worksubgroup);
                        }
                    });
                    Gpu::synchronize();
                }
            }
        }
    }
}

#endif
