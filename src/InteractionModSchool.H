/*! @file InteractionModSchool.H
 * \brief Contains the class describing agent interactions at school
 */

#ifndef _INTERACTION_MOD_SCHOOL_H_
#define _INTERACTION_MOD_SCHOOL_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;


#ifndef FAST_INTERACTIONS
/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at school. */
template <typename PTDType>
struct BinaryInteractionSchool {
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    ParticleReal operator() (const int infectious_i, /*!< Index of infectious agent */
                             const int susceptible_i, /*!< Index of susceptible agent */
                             const PTDType& a_ptd, /*!< Particle tile data */
                             const DiseaseParm* const a_lparm, /*!< disease paramters */
                             const Real a_social_scale  /*!< Social scale */) const noexcept {
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::school_id][infectious_i] == a_ptd.m_idata[IntIdx::school_id][susceptible_i]);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::school_grade][infectious_i] == a_ptd.m_idata[IntIdx::school_grade][susceptible_i]);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::work_i][infectious_i] == a_ptd.m_idata[IntIdx::work_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::work_j][infectious_i] == a_ptd.m_idata[IntIdx::work_j][susceptible_i]);

        auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
        auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
        auto school_id_ptr = a_ptd.m_idata[IntIdx::school_id];

        // binned so that infectious and susceptible are in the same school
        int school_id = school_id_ptr[susceptible_i];
        //infect *= i_mask;
        //infect *= j_mask;
        // Elementary/middle/high school in common
        if (school_id > 0 && school_id < 5) {
            if (age_group_ptr[infectious_i] <= 1) {  // Transmitter i is a child
                if (age_group_ptr[susceptible_i] <= 1) {  // Receiver j is a child
                    return a_lparm->xmit_school[school_id] * a_social_scale;
                } else {  // Child student -> adult teacher/staff transmission
                    return a_lparm->xmit_sch_c2a[school_id] * a_social_scale;// * 0.5; // hack for testing
                }
            } else {   // transmitter is an adult
                if (age_group_ptr[susceptible_i] <= 1) {  // Adult teacher/staff -> child student
                    return a_lparm->xmit_sch_a2c[school_id] * a_social_scale;// * 0.25; // hack for testing
                } else {  // adult to adult
                    return a_lparm->xmit_school[school_id] * a_social_scale;
                }
            }

        } else {
            if (nborhood_ptr[infectious_i] == nborhood_ptr[susceptible_i]) {
                if (school_id > 5) {             // Playgroup - currently not implemented in the initialization
                    return a_lparm->xmit_school[6] * a_social_scale;
                } else if (school_id == 5) {     // Day care
                    return a_lparm->xmit_school[5] * a_social_scale;
                }
            }
        }
        return 0.0_prt;
    }
};
#endif

template <typename PTDType>
struct SchoolCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !inHospital(idx, ptd) &&
                ptd.m_idata[IntIdx::school_id][idx] > 0 &&
                ptd.m_idata[IntIdx::school_closed][idx] == 0 &&
                !ptd.m_idata[IntIdx::withdrawn][idx] &&
                ptd.m_idata[IntIdx::air_travel][idx] < 0 &&
                ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};

/*! \brief Class describing agent interactions at school */
template <typename PCType, typename PTDType, typename PType>
class InteractionModSchool : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModSchool (bool _fast_bin) : InteractionModel<PCType, PTDType, PType>(_fast_bin) {}

        /*! \brief default destructor */
        virtual ~InteractionModSchool () = default;

        /*! \brief Simulate agent interaction at school */
        virtual void interactAgents (PCType& agents, MultiFab&) override {
#ifdef FAST_INTERACTIONS
            fastInteractSchool(agents);
#else
            interactAgentsImpl<InteractionModSchool<PCType, PTDType, PType>, PCType, PTDType,
                               SchoolCandidate<PTDType>,
                               BinaryInteractionSchool<PTDType>>(*this, agents, IntIdx::school_id);
#endif
        }

        void fastInteractSchool (PCType &agents);
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModSchool<PCType, PTDType, PType>::fastInteractSchool (PCType& agents) {
    BL_PROFILE(__func__);
    int n_disease = agents.numDiseases();

    SchoolCandidate<PTDType> isCandidate;

    Vector<Gpu::DeviceVector<int>> infected_school_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_daycare_d(OMP_MAX_THREADS);

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
            auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
            auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();

            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(agents.Geom(lev), mfi.tilebox(), agents.comm_mf[mfi].array());

            int max_communities = getCommunityIndex.max();
            int max_school_grade = agents.getMaxGroup(IntIdx::school_grade) + 1;
            int max_school_id = agents.getMaxGroup(IntIdx::school_id) + 1;

            infected_school_d[OMP_THREAD_NUM].resize(max_communities * max_school_id * max_school_grade);
            infected_daycare_d[OMP_THREAD_NUM].resize(max_communities * max_school_id * max_school_grade);
            auto infected_school_d_ptr = infected_school_d[OMP_THREAD_NUM].data();
            auto infected_daycare_d_ptr = infected_daycare_d[OMP_THREAD_NUM].data();

            /*
            Print() << mfi.tilebox() << " np " << np << " max communities " << max_communities
                    << " max school id " << max_school_id << " max school grade " << max_school_grade
                    << " school vector size " << max_communities * max_school_id * max_school_grade
                    << "\n";
            */
            for (int d = 0; d < n_disease; d++) {
                for (auto adults : {true, false}) {
                    dev_memset(infected_school_d_ptr, 0, infected_school_d[OMP_THREAD_NUM].size() * sizeof(int));
                    dev_memset(infected_daycare_d_ptr, 0, infected_daycare_d[OMP_THREAD_NUM].size() * sizeof(int));

                    auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                    auto lparm = agents.getDiseaseParameters_d(d);
                    auto lparm_h = agents.getDiseaseParameters_h(d);
                    Real scale = 1.0_rt;  // TODO this should vary based on cell
                    Real infect = (1.0_rt - lparm_h->vac_eff);

                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isInfectious(i, ptd, d) && isCandidate(i, ptd) && isAnAdult(i, ptd) == adults) {
                            auto community = getCommunityIndex(ptd, i);
                            int pos = (community * max_school_id + school_id_ptr[i]) * max_school_grade + school_grade_ptr[i];
                            if (getSchoolType(school_grade_ptr[i]) == SchoolType::daycare) {
                                Gpu::Atomic::AddNoRet(&infected_daycare_d_ptr[pos], 1);
                            } else {
                                Gpu::Atomic::AddNoRet(&infected_school_d_ptr[pos], 1);
                            }
                        }
                    });
                    Gpu::synchronize();

                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                            auto community = getCommunityIndex(ptd, i);
                            int pos = (community * max_school_id + school_id_ptr[i]) * max_school_grade + school_grade_ptr[i];
                            if (getSchoolType(school_grade_ptr[i]) == SchoolType::daycare) {
                                int num_infected_daycare = infected_daycare_d_ptr[pos];
                                Real daycare_prob = 1.0_rt - infect * lparm->xmit_school[SchoolType::daycare] * scale;
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(daycare_prob, num_infected_daycare));
                            } else {
                                int num_infected_school = infected_school_d_ptr[pos];
                                Real xmit = 0.0_rt;
                                if (adults) {   // transmitters are adults
                                    if (age_group_ptr[i] <= AgeGroups::a5to17) {  // Adult teacher/staff -> child student
                                        xmit = lparm->xmit_school_a2c[getSchoolType(school_grade_ptr[i])];
                                    } else {  // adult to adult - teachers also have grades (the grade they teach)
                                        xmit = lparm->xmit_school[getSchoolType(school_grade_ptr[i])];
                                    }
                                } else { // transmitters are children
                                    if (age_group_ptr[i] <= AgeGroups::a5to17) {  // Receiver j is a child
                                        xmit = lparm->xmit_school[getSchoolType(school_grade_ptr[i])];
                                    } else {  // Child student -> adult teacher/staff transmission
                                        xmit = lparm->xmit_school_c2a[getSchoolType(school_grade_ptr[i])];
                                    }
                                }
                                Real school_prob = 1.0_rt - infect * xmit * scale;
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(school_prob, num_infected_school));
                            }
                        }
                    });
                    Gpu::synchronize();
                }
            }
        }
    }
}


#endif
