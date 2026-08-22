/*! @file InteractionModSchool.H
 * \brief Contains the class describing agent interactions at school
 *
 * Transmission happens within a school_class_group -- a compact ID for an agent's actual
 * (community, school_id, grade, class) mixing bucket, assigned once by
 * AgentContainer::assignSchoolClasses (see IntIdx::school_class_group / TestParams::school_class_size)
 * -- rather than across an entire grade at a school. A school_class_group may be a real
 * classroom (IntIdx::school_class >= 0) or one of a raw group's non-classroom "admin" groups
 * (IntIdx::school_class <= -2, i.e. -2, -3, -4, ..., holding excess teachers beyond
 * one-per-class, sized like a regular workgroup) -- teacher-teacher contact within an admin
 * group is modeled by reusing xmit_work (the same calibrated, age-independent workplace
 * intensity InteractionModWork.H uses for every other industry), not a school-specific
 * constant.
 */

#ifndef _INTERACTION_MOD_SCHOOL_H_
#define _INTERACTION_MOD_SCHOOL_H_

#include "AgentDefinitions.H"
#include "InteractionModel.H"

using namespace amrex;

template <typename PTDType>
struct SchoolCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator()(const int idx, const PTDType& ptd) const noexcept {
        return !inHospital(idx, ptd) && ptd.m_idata[IntIdx::school_id][idx] > 0 && ptd.m_idata[IntIdx::school_closed][idx] == 0 &&
               !ptd.m_idata[IntIdx::withdrawn][idx] && ptd.m_idata[IntIdx::air_travel][idx] < 0 &&
               ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};

/*! \brief Class describing agent interactions at school */
template <typename PCType, typename PTDType, typename PType>
class InteractionModSchool {
  public:
    /*! \brief Simulate agent interaction at school */
    void interactAgents (PCType& agents, MultiFab&) { fastInteractSchool(agents); }

    void fastInteractSchool(PCType& agents);
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModSchool<PCType, PTDType, PType>::fastInteractSchool (PCType& agents) {
    BL_PROFILE(__func__);
    int n_disease = agents.numDiseases();

    SchoolCandidate<PTDType> isCandidate;

    Vector<Gpu::DeviceVector<int>> infected_school_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_daycare_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_school_asymp_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_daycare_asymp_d(OMP_MAX_THREADS);

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
            auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
            auto school_class_group_ptr = soa.GetIntData(IntIdx::school_class_group).data();
            auto school_class_ptr = soa.GetIntData(IntIdx::school_class).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();

            int max_class_group = agents.getMaxGroup(IntIdx::school_class_group) + 1;

            infected_school_d[OMP_THREAD_NUM].resize(max_class_group);
            infected_daycare_d[OMP_THREAD_NUM].resize(max_class_group);
            infected_school_asymp_d[OMP_THREAD_NUM].resize(max_class_group);
            infected_daycare_asymp_d[OMP_THREAD_NUM].resize(max_class_group);
            auto infected_school_d_ptr = infected_school_d[OMP_THREAD_NUM].data();
            auto infected_daycare_d_ptr = infected_daycare_d[OMP_THREAD_NUM].data();
            auto infected_school_asymp_d_ptr = infected_school_asymp_d[OMP_THREAD_NUM].data();
            auto infected_daycare_asymp_d_ptr = infected_daycare_asymp_d[OMP_THREAD_NUM].data();

            for (int d = 0; d < n_disease; d++) {
                for (auto adults : {true, false}) {
                    devMemset(infected_school_d_ptr, 0, infected_school_d[OMP_THREAD_NUM].size() * sizeof(int));
                    devMemset(infected_daycare_d_ptr, 0, infected_daycare_d[OMP_THREAD_NUM].size() * sizeof(int));

                    devMemset(infected_school_asymp_d_ptr, 0, infected_school_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));
                    devMemset(infected_daycare_asymp_d_ptr, 0, infected_daycare_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));

                    auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                    auto lparm = agents.getDiseaseParameters_d(d);
                    auto lparm_h = agents.getDiseaseParameters_h(d);
                    Real infect = (1.0_rt - lparm_h->vac_eff);

                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isInfectious(i, ptd, d) && isCandidate(i, ptd) && isAnAdult(i, ptd) == adults) {
                            int pos = school_class_group_ptr[i];
                            if (getSchoolType(school_grade_ptr[i]) == SchoolType::daycare) {
                                if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                                    Gpu::Atomic::AddNoRet(&infected_daycare_asymp_d_ptr[pos], 1);
                                } else {
                                    Gpu::Atomic::AddNoRet(&infected_daycare_d_ptr[pos], 1);
                                }
                            } else {
                                if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                                    Gpu::Atomic::AddNoRet(&infected_school_asymp_d_ptr[pos], 1);
                                } else {
                                    Gpu::Atomic::AddNoRet(&infected_school_d_ptr[pos], 1);
                                }
                            }
                        }
                    });
                    Gpu::synchronize();

                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                            int pos = school_class_group_ptr[i];
                            if (getSchoolType(school_grade_ptr[i]) == SchoolType::daycare) {
                                if (adults && age_group_ptr[i] > AgeGroups::a5to17) { // adults-adults interaction accounted for
                                                                                      // in InteractionModWork.H
                                    // do nothing!
                                } else {
                                    int num_infected_daycare = infected_daycare_d_ptr[pos];
                                    Real daycare_prob = 1.0_rt - infect * lparm->xmit_school[SchoolType::daycare];
                                    prob_ptr[i] *= static_cast<ParticleReal>(std::pow(daycare_prob, num_infected_daycare));
                                    int num_infected_daycare_asymp = infected_daycare_asymp_d_ptr[pos];
                                    Real daycare_prob_asymp =
                                            1.0_rt - lparm->asymp_relative_inf * infect * lparm->xmit_school[SchoolType::daycare];
                                    prob_ptr[i] *=
                                            static_cast<ParticleReal>(std::pow(daycare_prob_asymp, num_infected_daycare_asymp));
                                }
                            } else {
                                int num_infected_school = infected_school_d_ptr[pos];
                                int num_infected_school_asymp = infected_school_asymp_d_ptr[pos];
                                Real xmit = 0.0_rt;
                                if (adults) {                                    // transmitters are adults
                                    if (age_group_ptr[i] <= AgeGroups::a5to17) { // Adult teacher/staff -> child student
                                        xmit = lparm->xmit_school_a2c[getSchoolType(school_grade_ptr[i])];
                                    } else if (school_class_ptr[i] <= -2) {
                                        // receiver is in one of the raw group's admin (non-classroom)
                                        // groups (school_class == -2, -3, -4, ...) -- teacher-teacher
                                        // contact. Reuse xmit_work directly (not a school-specific constant,
                                        // and not xmit_school_scale'd -- this models the same "co-workers
                                        // sharing a room" contact InteractionModWork.H already represents for
                                        // every other industry, just physically located at a school instead
                                        // of a regular workplace; see AgentContainer::assignSchoolClasses).
                                        // Different admin groups are still isolated from each other by
                                        // school_class_group (the actual transmission bucket, checked via
                                        // `pos` above) -- this check only decides which xmit value to use.
                                        xmit = lparm->xmit_work;
                                    } else {
                                        // adult-adult in a genuine real class (e.g. two adult college
                                        // classmates) -- not modeled
                                        // do nothing!
                                    }
                                } else {                                         // transmitters are children
                                    if (age_group_ptr[i] <= AgeGroups::a5to17) { // Receiver j is a child
                                        xmit = lparm->xmit_school[getSchoolType(school_grade_ptr[i])];
                                    } else { // Child student -> adult teacher/staff transmission
                                        xmit = lparm->xmit_school_c2a[getSchoolType(school_grade_ptr[i])];
                                    }
                                }
                                Real school_prob = 1.0_rt - infect * xmit;
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(school_prob, num_infected_school));
                                Real school_prob_asymp = 1.0_rt - lparm->asymp_relative_inf * infect * xmit;
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(school_prob_asymp, num_infected_school_asymp));
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
