/*! @file InteractionModWork.H
 * \brief Contains the class describing agent interactions at work
 */

#ifndef _INTERACTION_MOD_WORK_H_
#define _INTERACTION_MOD_WORK_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;


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
        return a_lparm->xmit_work * a_work_scale;
    }
};

template <typename PTDType>
struct WorkCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !isHospitalized(idx, ptd) &&
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

    for (int lev = 0; lev < agents.numLevels(); ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();

            GetCommunityIndex<PTDType> getCommunityIndex(agents.Geom(lev), mfi.validbox());

            int max_communities = getCommunityIndex.max();
            int max_workgroup = agents.getMaxGroup(IntIdx::workgroup) + 1;
            AMREX_ALWAYS_ASSERT(max_workgroup <= np);


            for (int d = 0; d < n_disease; d++) {
                Gpu::DeviceVector<int> infected_workgroup_d(max_communities * max_workgroup, 0);
                auto infected_workgroup_d_ptr = infected_workgroup_d.data();

                auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                auto lparm = agents.getDiseaseParameters_d(d);
                auto lparm_h = agents.getDiseaseParameters_h(d);
                Real scale = 1.0_prt;  // TODO this should vary based on cell
                Real infect = lparm_h->infect * (1.0_rt - lparm_h->vac_eff);

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isInfectious(i, ptd, d) && isCandidate(i, ptd)) {
                        auto community = getCommunityIndex(ptd, i);
                        Gpu::Atomic::Add(&infected_workgroup_d_ptr[community * max_workgroup + workgroup_ptr[i]], 1);
                    }
                });
                Gpu::synchronize();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                        auto community = getCommunityIndex(ptd, i);
                        int num_infected_workgroup = infected_workgroup_d_ptr[community * max_workgroup + workgroup_ptr[i]];
                        ParticleReal workgroup_prob = 1.0_prt - infect * lparm->xmit_work * scale;
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(workgroup_prob, num_infected_workgroup));
                    }
                });
                Gpu::synchronize();
            }
        }
    }
}



#endif
